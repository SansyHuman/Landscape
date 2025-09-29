import csv
import os.path
from common.utils import prime_numbers
import math
from common.sci_parser import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

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


class SpectrumExpectDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.x_data = torch.tensor(input_data)
        self.y_data = torch.tensor(output_data)

    def __getitem__(self, index):
        return self.x_data[index].float(), self.y_data[index].float()

    def __len__(self):
        return self.x_data.shape[0]


class SpectrumExpectModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *args: int):
        # args: dimension of hidden layers
        super().__init__()
        dims = [input_dim] + list(args) + [output_dim]
        self.layers = nn.Sequential()
        for i in range(len(dims) - 3):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(dims[-3], dims[-2]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, x):
        x = self.layers(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()


def expect_higher_spectrum(a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex], input_spectrum_num: int, output_spectrum_num: int, epoch_num: int) -> None:
    input_dim = 2 + input_spectrum_num
    input_data = np.zeros((len(a_charges), input_dim))
    output_data = np.zeros((len(a_charges), output_spectrum_num))

    for i in range(len(a_charges)):
        input_data[i, 0] = a_charges[i]
        input_data[i, 1] = c_charges[i]
        sci = scis[i]
        for j in range(input_spectrum_num):
            if j >= len(sci.dims):
                input_data[i, 2 + j] = -1 # puts -1 to avoid problem of div by zero
            else:
                input_data[i, 2 + j] = sci.dims[j]

        for j in range(output_spectrum_num):
            index = j + input_spectrum_num
            if index >= len(sci.dims):
                output_data[i, j] = -1
            else:
                output_data[i, j] = sci.dims[index]

    input_train = input_data[0::2,:]
    output_train = output_data[0::2,:]
    input_test = input_data[1::2,:]
    output_test = output_data[1::2,:]

    dataset_train = SpectrumExpectDataset(input_train, output_train)
    print('Train dataset length:', len(dataset_train))
    dataset_test = SpectrumExpectDataset(input_test, output_test)
    print('Test dataset length:', len(dataset_test))

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

    for index, (x, y) in enumerate(dataloader_train):
        print(f'{index}/{len(dataloader_train)}', end=' ')
        print('x shape: ', x.shape, end=' ')
        print('y shape: ', y.shape)


    model = SpectrumExpectModel(input_dim, output_spectrum_num, input_dim * 3, input_dim * 20, input_dim * 5).to(device)
    print('Higher spectrum expect model shape: ', model(torch.randn(32, input_dim).to(device)).shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = 1e10

    checkpoint = None
    checkpoint_file_name = f'./checkpoint_spectrum_expect_{input_spectrum_num}_{output_spectrum_num}.tar'
    if os.path.isfile(checkpoint_file_name):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_file_name)
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

    test_x = torch.tensor(input_data).to(device).float()

    checkpoint = torch.load(checkpoint_file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

    with torch.no_grad():
        y_expect = model(test_x)
        y_expect = y_expect.cpu().numpy()
        y_real = output_data

        error = np.abs((y_expect - y_real) / y_real * 100).flatten()
        error_max = np.max(error)
        print(f'Maximum error: {error_max}')

        plt.hist(error, bins=math.ceil(error_max))
        plt.yscale('log')
        plt.title(f'Spectrum expectation from {input_spectrum_num} to {output_spectrum_num} errors')
        plt.savefig(f'./data/{filename}_spectrum_expectation_{input_spectrum_num}_{output_spectrum_num}.png')
        plt.show()


expect_higher_spectrum(a_charges, c_charges, scis, 2, 4, 10)