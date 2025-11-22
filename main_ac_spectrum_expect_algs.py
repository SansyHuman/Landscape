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

algnum = int(input('Enter the number of the files: '))

filenames = []
for _ in range(algnum):
    filenames.append(input('Enter filename: '))

data = [None] * algnum
for i in range(algnum):
    with open(filenames[i]) as csvfile:
        reader = csv.reader(csvfile)
        data[i] = list(reader)

field_content_index, a_index, c_index, sci_index = -1, -1, -1, -1
for i in range(len(data[0][0])):
    if data[0][0][i] == "Name":
        field_content_index = i
    elif data[0][0][i] == "CentralChargeA":
        a_index = i
    elif data[0][0][i] == "CentralChargeC":
        c_index = i
    elif data[0][0][i] == "SCI":
        sci_index = i

print(f'Field content: {field_content_index}, A: {a_index}, C: {c_index}, SCI: {sci_index}')

a_charges = [[] for _ in range(algnum)]
c_charges = [[] for _ in range(algnum)]
scis = [[] for _ in range(algnum)]

for i in range(algnum):
    for j in range(1, len(data[i])):
        a, c = float(data[i][j][a_index]), float(data[i][j][c_index])
        sci = SuperConformalIndex(data[i][j][sci_index].strip())

        a_charges[i].append(a)
        c_charges[i].append(c)
        scis[i].append(sci)

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
epoch_num = int(input('Enter the number of epochs: '))

checkpoint_file_names = [None] * algnum
models = [None] * algnum
optimizers = [None] * algnum
test_x = [None] * algnum
y_real = [None] * algnum

for i in range(algnum):
    print(f'Training {filenames[i]}...')

    input_data = np.zeros((len(a_charges[i]), input_spectrum_num))
    output_data = np.zeros((len(a_charges[i]), 1))

    for j in range(len(a_charges[i])):
        sci = scis[i][j]
        for k in range(input_spectrum_num):
            if k >= len(sci.dims):
                input_data[j, k] = 0
            else:
                input_data[j, k] = sci.dims[k]

        output_data[j, 0] = a_charges[i][j] / c_charges[i][j]

    test_x[i] = torch.tensor(input_data).to(device).float()
    y_real[i] = output_data

    input_train = input_data[0::2, :]
    output_train = output_data[0::2, :]
    input_test = input_data[1::2, :]
    output_test = output_data[1::2, :]

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
    checkpoint_file_name = f'./checkpoint_charge_ratio_expect_alg_{input_spectrum_num}_{filenames[i]}.tar'
    if os.path.isfile(checkpoint_file_name):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_file_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(epoch_num):
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

    checkpoint_file_names[i] = checkpoint_file_name
    models[i] = model
    optimizers[i] = optimizer

print('Applying charge ratio expect to other algebras...')

errors = [[0.0 for _ in range(algnum)] for _ in range(algnum)]

for i in range(algnum):
    checkpoint = torch.load(checkpoint_file_names[i], map_location=device)
    models[i].load_state_dict(checkpoint['model_state_dict'])
    optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        for j in range(algnum):
            y_expect = models[i](test_x[j])
            y_expect = y_expect.cpu().numpy()

            error = np.abs((y_expect - y_real[j]) / y_real[j] * 100).flatten()
            error_avg = np.mean(error)
            errors[i][j] = error_avg

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

ncols = algnum // 2
fig, ax = plt.subplots(nrows=2, ncols=ncols, sharey=True)

fig.suptitle('Charge ratio expect to other algebras')

for i in range(algnum):
    row = i // ncols
    col = i % ncols

    p = ax[row][col].bar(filenames, errors[i], width=0.5, align='center')
    ax[row][col].bar_label(p, fmt='%.2f')

    ax[row][col].set_title(f'Model trained by {filenames[i]}')

plt.savefig(f'./data/charge_expectation_algs_{input_spectrum_num}.png')
plt.show()