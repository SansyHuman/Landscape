import csv
import os.path
from common.utils import *
import math
import json
from common.sci_parser import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

field_content_index, a_index, c_index, sci_index = -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "SCI":
        sci_index = i

print(f'Field content: {field_content_index}, A: {a_index}, C: {c_index}, SCI: {sci_index}')

field_contents_index = dict()
field_contents = []
a_charges = []
c_charges = []
scis = []

for i in range(1, len(data)):
    field_content = data[i][field_content_index]
    a, c = float(data[i][a_index]), float(data[i][c_index])
    sci = SuperConformalIndex(data[i][sci_index].strip())

    if field_content not in field_contents_index:
        field_contents_index[field_content] = len(field_contents_index)
    field_contents.append(field_contents_index[field_content])
    a_charges.append(a)
    c_charges.append(c)
    scis.append(sci)

print(f"Field contents: {field_contents_index}")

os.makedirs('./data', exist_ok=True)


class ACRatioExpectModel(FullyConnectedNetwork):
    def __init__(self, input_dim: int, output_dim: int, *args: int):
        # args: dimension of hidden layers
        super().__init__(
            input_dim,
            output_dim,
            *([(args[i], nn.ELU()) for i in range(len(args) - 1)] + [(args[-1], nn.ReLU())])
        )


def save_data(errors, test_name: str) -> None:
    json_data = dict()
    sorted_errors = np.sort(errors, axis=None)
    json_data['min_error'] = sorted_errors[0]
    json_data['max_error'] = sorted_errors[-1]
    json_data['avg_error'] = np.mean(sorted_errors)
    json_data['median_error'] = median_sorted(sorted_errors)
    json_data['stdev_error'] = np.std(sorted_errors)

    with open(f'./data/{filename}_{test_name}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

input_spectrum_num = int(input('Enter the number of input spectrum: '))
epoch_step = int(input('Enter the number of steps of epochs. For one step, a number of epochs proceed and check the error: '))
epoch_step_num = int(input('Enter the number of epoches of each steps: '))
inverse_train_ratio = int(input('Enter the inverse ratio of training set, the ratio will be 1/n where n is the input number: '))

data_num = len(a_charges)
input_data = np.zeros((data_num, input_spectrum_num))
output_data = np.zeros((data_num, 1))
errors_epoch = [0.0] * epoch_step
errors = None

for i in range(data_num):
    sci = scis[i]
    for j in range(input_spectrum_num):
        if j >= len(sci.dims):
            input_data[i, j] = 0
        else:
            input_data[i, j] = sci.dims[j]

    output_data[i, 0] = a_charges[i] / c_charges[i]

print(f'Training with ratio {1 / inverse_train_ratio}...')

train_index = list(range(0, data_num, inverse_train_ratio))
test_index = [0] * (data_num - len(train_index))
tmp = 0
for j in range(data_num):
    if j % inverse_train_ratio == 0:
        continue
    else:
        test_index[tmp] = j
        tmp += 1

input_train = input_data[train_index, :]
output_train = output_data[train_index, :]
input_test = input_data[test_index, :]
output_test = output_data[test_index, :]

dataset_train = GenericDataset(input_train, output_train)
print('Train dataset length:', len(dataset_train))
dataset_test = GenericDataset(input_test, output_test)
print('Test dataset length:', len(dataset_test))
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)
for index, (x, y) in enumerate(dataloader_train):
    print(f'{index}/{len(dataloader_train)}', end=' ')
    print('x shape: ', x.shape, end=' ')
    print('y shape: ', y.shape)


model = ACRatioExpectModel(input_spectrum_num, 1,
                            input_spectrum_num * 10,
                            input_spectrum_num * 20,
                            input_spectrum_num * 30,
                            input_spectrum_num * 30,
                            input_spectrum_num * 15,
                            input_spectrum_num * 15,
                            input_spectrum_num * 5,
                            input_spectrum_num * 2
                            ).to(device)
print('A/C ratio expect model shape: ', model(torch.randn(32, input_spectrum_num).to(device)).shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_loss = 1e10

checkpoint = None
checkpoint_file_name = f'./checkpoint_ac_ratio_expect_{input_spectrum_num}_{inverse_train_ratio}.tar'
if os.path.isfile(checkpoint_file_name):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(epoch_step * epoch_step_num):
    model.train()
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0.0
    error = 0.0
    test_cnt = 0

    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            test_loss += loss.item()

            outputs = outputs.cpu().numpy()
            y = y.cpu().numpy()
            err = np.concatenate(np.abs((outputs - y) / y))
            error += np.sum(err)
            test_cnt += len(err)

    print(f'epoch {epoch + 1} test loss: {test_loss / len(dataloader_test)} error: {error * 100 / test_cnt} %')
    if test_loss < best_loss:
        best_loss = test_loss
        print('New best loss obtained. Saving model...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_file_name)

    if (epoch + 1) % epoch_step_num == 0:
        print(f'Calculating error train set ratio 1/{inverse_train_ratio} at epoch {epoch + 1}...')

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_file_name + '.tmp'
        )

        checkpoint = torch.load(checkpoint_file_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with torch.no_grad():
            test_x = torch.tensor(input_data).to(device).float()
            y_real = output_data
            y_expect = model(test_x)
            y_expect = y_expect.cpu().numpy()

            error = np.abs((y_expect - y_real) / y_real * 100).flatten()
            error_avg = np.mean(error)
            errors_epoch[(epoch + 1) // epoch_step_num - 1] = error_avg

            if epoch == epoch_step * epoch_step_num - 1:
                errors = error
                save_data(error, f'ac_ratio_expect_{input_spectrum_num}_{inverse_train_ratio}')

        checkpoint_tmp = torch.load(checkpoint_file_name + '.tmp', map_location=device)
        model.load_state_dict(checkpoint_tmp['model_state_dict'])
        optimizer.load_state_dict(checkpoint_tmp['optimizer_state_dict'])

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(1, 2, squeeze=True)
fig.suptitle(f'A/C ratio expect from {input_spectrum_num} lowest dims')

ax[0].set_title('Epoch - Error')
ax[0].plot([epoch_step_num * (i + 1) for i in range(epoch_step)], errors_epoch)
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Error (%)')

ax[1].set_title('Errors')
ax[1].hist(errors, bins=math.ceil(np.max(errors)))
ax[1].set_yscale('log')
ax[1].set_xlabel('Error (%)')
ax[1].set_ylabel('Number of theories')

plt.savefig(f'./data/{filename}_ac_ratio_expect_{input_spectrum_num}_{inverse_train_ratio}.png')
plt.show()