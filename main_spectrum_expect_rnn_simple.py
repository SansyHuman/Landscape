import csv
import os.path

from numpy.f2py.auxfuncs import throw_error

from common.utils import *
import math
import json
from common.sci_parser import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpectrumRNN(nn.Module):
    def __init__(self, input_size: int, auxiliary_size: int, hidden_size: int, num_layers: int, output_length: int):
        super(SpectrumRNN, self).__init__()
        self.input_size = input_size
        self.auxiliary_size = auxiliary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = FullyConnectedNetwork(hidden_size + auxiliary_size, input_size)

    def forward(self, input_list: torch.Tensor, auxiliary_list: torch.Tensor, target_tensor: Union[torch.Tensor, None]=None) -> torch.Tensor:
        if target_tensor is not None and target_tensor.size(1) != self.output_length:
            raise ValueError(f'Target tensor must have {self.output_length} elements')

        outputs = []

        rnn_output, rnn_hidden = self.rnn(input_list)
        rnn_output = torch.cat((auxiliary_list, rnn_output[:, -1]), dim=1)

        rnn_input = self.linear(rnn_output).unsqueeze(1)
        outputs.append(rnn_input)

        for i in range(1, self.output_length):
            if target_tensor is not None:
                rnn_input = target_tensor[:, i - 1].unsqueeze(1)

            rnn_output, rnn_hidden = self.rnn(rnn_input, rnn_hidden)
            rnn_output = torch.cat((auxiliary_list, rnn_output.squeeze(1)), dim=1)

            rnn_input = self.linear(rnn_output).unsqueeze(1)
            outputs.append(rnn_input)

        outputs = torch.cat(outputs, dim=1)
        return outputs

input_num = int(input("Number of input spectrum: "))
output_num = int(input("Number of output spectrum: "))
epoch_num = int(input("Number of epochs: "))

input_data = np.zeros((len(a_charges), input_num, 1)) # [dimension_ij]
auxiliary_data = np.zeros((len(a_charges), 2)) # [a_i, c_i]
output_data = np.zeros((len(a_charges), output_num, 1)) # [dimension_ij]

for i in range(len(a_charges)):
    a_charge = a_charges[i]
    c_charge = c_charges[i]
    sci = scis[i]
    for j in range(input_num):
        auxiliary_data[i, 0] = a_charges[j]
        auxiliary_data[i, 1] = c_charges[j]
        if j >= len(sci.dims):
            input_data[i, j, 0] = -1 # -1 represents no such operator
        else:
            input_data[i, j, 0] = sci.dims[j]

    for j in range(output_num):
        index = j + input_num
        if index >= len(sci.dims):
            output_data[i, j, 0] = -1
        else:
            output_data[i, j, 0] = sci.dims[index]

input_train = torch.tensor(input_data[0::2,:,:]).float()
auxiliary_train = torch.tensor(auxiliary_data[0::2,:]).float()
output_train = torch.tensor(output_data[0::2,:,:]).float()
input_test = torch.tensor(input_data[1::2,:,:]).float()
auxiliary_test = torch.tensor(auxiliary_data[1::2,:]).float()
output_test = torch.tensor(output_data[1::2,:,:]).float()

dataset_train = TensorDataset(input_train, auxiliary_train, output_train)
dataset_test = TensorDataset(input_test, auxiliary_test, output_test)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

for index, (x, a, y) in enumerate(dataloader_train):
    print(f'{index}/{len(dataloader_train)}', end=' ')
    print('x shape: ', x.shape, end=' ')
    print('a shape: ', a.shape, end=' ')
    print('y shape: ', y.shape)

hidden_size = 6
num_layers = 4
model = SpectrumRNN(1, 2, hidden_size, num_layers, output_num).to(device)

outputs = model(torch.randn(32, input_num, 1).to(device), torch.randn(32, 2).to(device))
print('Higher spectrum expect model shape: ', outputs.shape)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_loss = 1e10

checkpoint = None
checkpoint_file_name = f'./checkpoint_spectrum_expect_rnn_simple_{input_num}_{output_num}.tar'
if os.path.isfile(checkpoint_file_name):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(epoch_num):
    model.train()

    for x, a, y in dataloader_train:
        x, a, y = x.to(device), a.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x, a, y)
        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

    model.eval()

    test_loss = 0.0
    error = 0.0
    test_cnt = 0
    with torch.no_grad():
        for x, a, y in dataloader_test:
            x, a, y = x.to(device), a.to(device), y.to(device)

            outputs = model(x, a)

            loss = criterion(outputs, y)
            test_loss += loss.item()

            outputs = outputs.cpu().numpy()
            y = y.cpu().numpy()

            err = np.abs((outputs - y) / y).flatten()
            error += np.sum(err)
            test_cnt += len(err)

        print(f'epoch {epoch + 1} test loss: {test_loss / len(dataloader_test)} error: {error / test_cnt * 100} %')
        if test_loss < best_loss:
            best_loss = test_loss
            print('New best loss obtained. Saving model...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_file_name)

test_input = torch.tensor(input_data).to(device).float()
test_auxiliary = torch.tensor(auxiliary_data).to(device).float()

checkpoint = torch.load(checkpoint_file_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

with torch.no_grad():
    outputs = model(test_input, test_auxiliary)
    output_expect = outputs.cpu().numpy()
    output_real = output_data

    error = np.abs((output_expect - output_real) / output_real * 100).flatten()
    error_max = np.max(error)
    print(f'Maximum error: {error_max}')

    json_data = dict()
    sorted_errors = np.sort(error, axis=None)
    json_data['min_error'] = sorted_errors[0]
    json_data['max_error'] = sorted_errors[-1]
    json_data['avg_error'] = np.mean(sorted_errors)
    json_data['median_error'] = median_sorted(sorted_errors)
    json_data['stdev_error'] = np.std(sorted_errors)

    with open(f'./data/{filename}_spectrum_expect_rnn_simple_{input_num}_{output_num}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    error = np.nan_to_num(error, posinf=0.0)
    error_max = np.max(error)

    plt.hist(error, bins=math.ceil(error_max))
    plt.yscale('log')
    plt.title(f'Spectrum expectation using simple RNN from {input_num} to {output_num}')
    plt.xlabel('Error (%)')
    plt.ylabel('Number of theories')
    plt.savefig(f'./data/{filename}_spectrum_expect_rnn_simple_{input_num}_{output_num}.png')
    plt.show()