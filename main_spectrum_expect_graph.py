import csv
import json
import os.path
import random

from common.inconsistents_parser import serialize_theory_name
from common.sci_parser import SuperConformalIndex
from common.superpotential_parser import Superpotential
from common.utils import prime_numbers, PairData, median_sorted, FullyConnectedNetwork
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader


os.makedirs('./data', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")
input_spectrum_num = int(input("Enter input spectrum number: "))
output_spectrum_num = int(input("Enter output spectrum number: "))
epoch_num = int(input("Enter number of epochs: "))

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

field_content_index, w_index, sci_index = -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "Superpotentials":
        w_index = i
    elif data[0][i] == "SCI":
        sci_index = i

dataset = []
w_obj = Superpotential()

prev_theory = None
for i in range(1, len(data)):
    theory = serialize_theory_name(data[i][field_content_index])
    if theory[0] == 0:
        continue

    if prev_theory != theory:
        prev_theory = theory
        w_obj.set_theory(theory)
    w_obj.set_superpotential(data[i][w_index])

    dynkin_diagram = w_obj.get_theory_data()
    superpotential_graph = w_obj.get_superpotential_data()

    sci = SuperConformalIndex(data[i][sci_index].strip())
    spectrum_num = len(sci.dims)
    input_spectrum = [[-1 if j >= spectrum_num else sci.dims[j] for j in range(input_spectrum_num)]]
    output_spectrum = [[-1 if j + input_spectrum_num >= spectrum_num else sci.dims[j + input_spectrum_num] for j in range(output_spectrum_num)]]

    w_data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                    edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                    x_spectrum = torch.tensor(input_spectrum),
                    y=torch.tensor(output_spectrum))
    dataset.append(w_data)

random.shuffle(dataset)
num_data = len(dataset)
print(f'Number of data: {num_data}')

train_ratio = float(input("Enter the ratio of training data: "))
if train_ratio > 1:
    train_ratio = 1
elif train_ratio < 0:
    train_ratio = 0

train_num = round(num_data * train_ratio)
train_dataset = dataset[:train_num]
test_dataset = dataset[train_num:]

print(f'Number of training data: {len(train_dataset)}')
print(f'Number of testing data: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, follow_batch=['x_1', 'x_2'])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, follow_batch=['x_1', 'x_2'])

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('========')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


class GraphSpectrumExpectModel(nn.Module):
    def __init__(self, dynkin_features: int, w_features: int, dynkin_hidden_channels: list[int], w_hidden_channels: list[int],
                 input_spectrum_num: int, output_spectrum_num: int, spectrum_expect_linear: list[int]):
        super(GraphSpectrumExpectModel, self).__init__()

        assert len(dynkin_hidden_channels) > 0 and len(w_hidden_channels) > 0

        self.conv_dynkin = nn.ModuleList()
        self.conv_w = nn.ModuleList()
        self.norm_dynkin = nn.ModuleList()
        self.norm_w = nn.ModuleList()

        self.conv_dynkin.append(pyg_nn.GraphConv(dynkin_features, dynkin_hidden_channels[0]))
        for i in range(len(dynkin_hidden_channels) - 1):
            self.conv_dynkin.append(pyg_nn.GraphConv(dynkin_hidden_channels[i], dynkin_hidden_channels[i + 1]))
            self.norm_dynkin.append(pyg_nn.norm.GraphNorm(dynkin_hidden_channels[i]))

        self.conv_w.append(pyg_nn.GraphConv(w_features, w_hidden_channels[0]))
        for i in range(len(w_hidden_channels) - 1):
            self.conv_w.append(pyg_nn.GraphConv(w_hidden_channels[i], w_hidden_channels[i + 1]))
            self.norm_w.append(pyg_nn.norm.GraphNorm(w_hidden_channels[i]))

        self.lin = FullyConnectedNetwork(
            dynkin_hidden_channels[-1] + w_hidden_channels[-1] + input_spectrum_num, output_spectrum_num,
            *list(zip(spectrum_expect_linear, [nn.GELU() for _ in range(len(spectrum_expect_linear))]))
        )

    def forward(self, x_dynkin, x_w, x_spectrum, edge_index_dynkin, edge_index_w, batch_dynkin, batch_w):
        for i in range(len(self.conv_dynkin)):
            x_dynkin = self.conv_dynkin[i](x_dynkin, edge_index_dynkin)
            if i != len(self.conv_dynkin) - 1:
                x_dynkin = self.norm_dynkin[i](x_dynkin, batch_dynkin)
                x_dynkin = F.gelu(x_dynkin)
        x_dynkin = pyg_nn.global_mean_pool(x_dynkin, batch_dynkin)

        for i in range(len(self.conv_w)):
            x_w = self.conv_w[i](x_w, edge_index_w)
            if i != len(self.conv_w) - 1:
                x_w = self.norm_w[i](x_w, batch_w)
                x_w = F.gelu(x_w)
        x_w = pyg_nn.global_mean_pool(x_w, batch_w)

        x_total = torch.cat((x_dynkin, x_w, x_spectrum), dim=1)
        x_total = self.lin(x_total)

        return x_total

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Avaliable device: {device}')
criterion = nn.MSELoss()

dynkin_features=dataset[0].x_1.shape[1]
w_features=dataset[0].x_2.shape[1]

model = GraphSpectrumExpectModel(dynkin_features, w_features,
                                [dynkin_features * 2, dynkin_features * 2, dynkin_features * 2],
                                [w_features * 2, w_features * 3, w_features * 3, w_features * 2],
                                 input_spectrum_num, output_spectrum_num,
                                 [
                                     input_spectrum_num * 4,
                                     input_spectrum_num * 8,
                                     input_spectrum_num * 16,
                                     input_spectrum_num * 16,
                                     output_spectrum_num * 4,
                                     output_spectrum_num * 2]
                                 ).to(device)

print(model)
batch = next(iter(test_loader))
print('Charge calculation model shape: ', model(
    batch.x_1.float().to(device), batch.x_2.float().to(device), batch.x_spectrum.float().to(device),
    batch.edge_index_1.to(device), batch.edge_index_2.to(device),
    batch.x_1_batch.to(device), batch.x_2_batch.to(device)
).shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_loss = 1e10

checkpoint = None
checkpoint_file_name = f'./checkpoint_spectrum_expect_{input_spectrum_num}_{output_spectrum_num}_graph.tar'
if os.path.isfile(checkpoint_file_name):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(epoch_num):
    model.train()
    for _, data in enumerate(train_loader):
        x_dynkin = data.x_1.float().to(device)
        x_w = data.x_2.float().to(device)
        x_spectrum = data.x_spectrum.float().to(device)
        edge_index_dynkin = data.edge_index_1.to(device)
        edge_index_w = data.edge_index_2.to(device)
        batch_dynkin = data.x_1_batch.to(device)
        batch_w = data.x_2_batch.to(device)
        y = data.y.float().to(device)

        outputs = model(
            x_dynkin, x_w, x_spectrum,
            edge_index_dynkin, edge_index_w,
            batch_dynkin, batch_w
        )
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0.0
    error = 0.0
    test_cnt = 0

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_dynkin = data.x_1.float().to(device)
            x_w = data.x_2.float().to(device)
            x_spectrum = data.x_spectrum.float().to(device)
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)
            y = data.y.float().to(device)

            outputs = model(
                x_dynkin, x_w, x_spectrum,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )
            loss = criterion(outputs, y)

            test_loss += loss.item()

            outputs = outputs.cpu().numpy()
            ac = y.cpu().numpy()
            err = np.concatenate(np.abs((outputs - ac) / ac))
            error += np.sum(err)
            test_cnt += len(err)

    print(f'epoch {epoch + 1} test loss: {test_loss / len(test_loader)} error: {error * 100 / test_cnt} %')
    if test_loss < best_loss:
        best_loss = test_loss
        print('New best loss obtained. Saving model...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_file_name)

final_loader = DataLoader(dataset, batch_size=256, shuffle=False, follow_batch=['x_1', 'x_2'])

checkpoint = torch.load(checkpoint_file_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

with torch.no_grad():
    error = np.array([])
    error_spectrum = None

    for _, data in enumerate(final_loader):
        x_dynkin = data.x_1.float().to(device)
        x_w = data.x_2.float().to(device)
        x_spectrum = data.x_spectrum.float().to(device)
        edge_index_dynkin = data.edge_index_1.to(device)
        edge_index_w = data.edge_index_2.to(device)
        batch_dynkin = data.x_1_batch.to(device)
        batch_w = data.x_2_batch.to(device)
        y_real = data.y.cpu().numpy()

        y_expect = model(
            x_dynkin, x_w, x_spectrum,
            edge_index_dynkin, edge_index_w,
            batch_dynkin, batch_w
        ).cpu().numpy()

        error_raw = np.abs((y_expect - y_real) / y_real) * 100
        error = np.append(error, error_raw.flatten())
        error_spectrum = error_raw.transpose() if error_spectrum is None else np.hstack((error_spectrum, error_raw.transpose()))

    error_max = np.max(error)
    print(f'Maximum error: {error_max}')

    json_data = dict()
    for i in range(error_spectrum.shape[0]):
        spectrum_data = dict()
        sorted_errors = np.sort(error_spectrum[i], axis=None)
        spectrum_data['min_error'] = float(sorted_errors[0])
        spectrum_data['max_error'] = float(sorted_errors[-1])
        spectrum_data['avg_error'] = float(np.mean(sorted_errors))
        spectrum_data['median_error'] = float(median_sorted(sorted_errors))
        spectrum_data['stdev_error'] = float(np.std(sorted_errors))
        json_data[f'spectrum_{i + 1}'] = spectrum_data

    total_data = dict()
    sorted_errors = np.sort(error, axis=None)
    total_data['min_error'] = float(sorted_errors[0])
    total_data['max_error'] = float(sorted_errors[-1])
    total_data['avg_error'] = float(np.mean(sorted_errors))
    total_data['median_error'] = float(median_sorted(sorted_errors))
    total_data['stdev_error'] = float(np.std(sorted_errors))
    json_data['total'] = total_data

    with open(f'./data/{filename}_spectrum_expect_{input_spectrum_num}_{output_spectrum_num}_graph.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)

    fig.suptitle(f'Spectrum expectation from {input_spectrum_num} to {output_spectrum_num}')

    ax[0].hist(error, bins=math.ceil(error_max))
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Error (%)')
    ax[0].set_ylabel('Number of errors')
    ax[0].set_title('Graph spectrum expectation errors')

    ax[1].plot(
        [i + 1 for i in range(error_spectrum.shape[0])],
        [json_data[f'spectrum_{i + 1}']['avg_error'] for i in range(error_spectrum.shape[0])]
    )
    ax[1].set_xlabel('Spectrum')
    ax[1].set_ylabel('Average error (%)')
    ax[1].set_title('Average errors by spectrum')

    plt.savefig(f'./data/{filename}_spectrum_expect_{input_spectrum_num}_{output_spectrum_num}_graph.png')
    plt.show()
