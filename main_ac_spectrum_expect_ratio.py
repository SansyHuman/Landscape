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


class SpectrumExpectModel(FullyConnectedNetwork):
    def __init__(self, input_dim: int, output_dim: int, *args: int):
        # args: dimension of hidden layers
        super().__init__(
            input_dim,
            output_dim,
            *([(args[i], nn.ELU()) for i in range(len(args) - 1)] + [(args[-1], nn.ReLU())])
        )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

input_spectrum_num = int(input('Enter the number of input spectrum: '))
epoch_num = int(input('Enter the number of steps of epochs. For one step, 100 epochs proceed and check the error: '))
ratio_num = int(input('Enter the number of ratio of training set, starting from 1/2 multiplied repeatedly by 1/2: '))

data_num = len(a_charges)
input_data = np.zeros((data_num, input_spectrum_num))
output_data = np.zeros((data_num, 1))
errors = np.zeros((ratio_num, epoch_num))

for i in range(data_num):
    sci = scis[i]
    for j in range(input_spectrum_num):
        if j >= len(sci.dims):
            input_data[i, j] = 0
        else:
            input_data[i, j] = sci.dims[j]

    output_data[i, 0] = a_charges[i] / c_charges[i]

"""
checkpoint_file_names = [None] * ratio_num
models = [None] * ratio_num
optimizers = [None] * ratio_num
"""

inverse_train_ratio = 1

for i in range(ratio_num):
    inverse_train_ratio *= 2
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

    model = SpectrumExpectModel(input_spectrum_num, 1,
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
    checkpoint_file_name = f'./checkpoint_charge_ratio_expect_ratio_{input_spectrum_num}_{inverse_train_ratio}.tar'
    if os.path.isfile(checkpoint_file_name):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_file_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(epoch_num * 100):
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

        if (epoch + 1) % 100 == 0:
            print(f'Calculating error train set ratio 1/{inverse_train_ratio} at epoch {epoch + 1}...')

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
                errors[i][(epoch + 1) // 100 - 1] = error_avg

    """
    checkpoint_file_names[i] = checkpoint_file_name
    models[i] = model
    optimizers[i] = optimizer
    """

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots()
fig.suptitle('Train set ratio - error')

x = [100 * (i + 1) for i in range(epoch_num)]
for i in range(ratio_num):
    ax.plot(x, errors[i], label=f'1/{2**(i + 1)}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error (%)')
ax.legend()

plt.savefig(f'./data/{filename}_charge_expectation_ratio_{input_spectrum_num}_{ratio_num}.png')
plt.show()

"""
print('Calculating errors of each train set ratio...')

errors = [0.0] * ratio_num
inverse_train_ratio = 1

test_x = torch.tensor(input_data).to(device).float()
y_real = output_data

for i in range(ratio_num):
    inverse_train_ratio *= 2
    checkpoint = torch.load(checkpoint_file_names[i], map_location=device)
    models[i].load_state_dict(checkpoint['model_state_dict'])
    optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        y_expect = models[i](test_x)
        y_expect = y_expect.cpu().numpy()

        error = np.abs((y_expect - y_real) / y_real * 100).flatten()
        error_avg = np.mean(error)
        errors[i] = error_avg

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots()
fig.suptitle('Train set ratio - error')

p = ax.bar([f'1/{2**(i +1)}' for i in range(ratio_num)], errors, width=0.5, align='center')
ax.bar_label(p, fmt='%.2f')
ax.set_xlabel('Train set ratio')
ax.set_ylabel('Error (%)')

plt.savefig(f'./data/{filename}_charge_expectation_ratio_{input_spectrum_num}_{ratio_num}.png')
plt.show()
"""