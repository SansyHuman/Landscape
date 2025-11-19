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

field_content_index, w_index, a_index, c_index, sci_index = -1, -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "Superpotentials":
        w_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "SCI":
        sci_index = i

print(f'Field content: {field_content_index}, A: {a_index}, C: {c_index}, SCI: {sci_index}')

field_contents_index = dict()
field_contents = []
w_lengths = []
a_charges = []
c_charges = []
scis = []

for i in range(1, len(data)):
    field_content = data[i][field_content_index]
    w = data[i][w_index][1:-1]
    w_length = len(w.split(',')) if len(w) > 0 else 0
    a, c = float(data[i][a_index]), float(data[i][c_index])
    sci = SuperConformalIndex(data[i][sci_index].strip())

    if field_content not in field_contents_index:
        field_contents_index[field_content] = len(field_contents_index)
    field_contents.append(field_contents_index[field_content])
    w_lengths.append(w_length)
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


w_median = round(np.median(w_lengths))
print(f'Median superpotential length: {w_median}')

def expect_ac_ratio_short_long(w_lengths: list[int], a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex], input_spectrum_num: int, epoch_num: int) -> None:
    w_train = []
    w_test = []
    for i in range(len(w_lengths)):
        if w_lengths[i] < w_median:
            w_train.append(i)
        else:
            w_test.append(i)

    input_data = np.array(
        [
            [scis[i].dims[j] if j < len(scis[i].dims) else 0 for j in range(input_spectrum_num)]
            for i in range(len(a_charges))
        ]
    )
    output_data = np.array(a_charges) / np.array(c_charges)
    output_data = np.expand_dims(output_data, 1)

    input_train = input_data[w_train, :]
    input_test = input_data[w_test, :]
    output_train = output_data[w_train, :]
    output_test = output_data[w_test, :]

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
                                input_spectrum_num * 2).to(device)
    print('A/C ratio expect model shape: ', model(torch.randn(32, input_spectrum_num).to(device)).shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = 1e10

    checkpoint = None
    checkpoint_file_name = f'./checkpoint_charge_ratio_expect_short_long_{input_spectrum_num}.tar'
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

    test_x = torch.tensor(input_data).to(device).float()

    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        y_expect = model(test_x)
        y_expect = y_expect.cpu().numpy()
        y_real = output_data

        error = np.abs((y_expect - y_real) / y_real * 100).flatten()
        error_max = np.max(error)
        print(f'Maximum error: {error_max}')
        save_data(error, f'{expect_ac_ratio_short_long.__name__}_{input_spectrum_num}')

        plt.hist(error, bins=math.ceil(error_max))
        plt.yscale('log')
        plt.title(f'Charge ratio expectation from {input_spectrum_num} errors')
        plt.xlabel('Error (%)')
        plt.ylabel('Number of theories')
        plt.savefig(f'./data/{filename}_charge_ratio_expectation_short_long_{input_spectrum_num}.png')
        plt.show()

epoch_num = int(input('Input the number of epochs: '))

expect_ac_ratio_short_long(w_lengths, a_charges, c_charges, scis, 6, epoch_num)