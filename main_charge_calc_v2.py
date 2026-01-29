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
from sklearn.cluster import DBSCAN


os.makedirs('./data', exist_ok=True)
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

w_set = []
ac_set = []
sci_set = []
theory_index = []
theory_name_index = dict()
theory_index_name = dict()

for i in range(1, len(data)):
    theory_name = data[i][field_content_index]
    theory = serialize_theory_name(theory_name)
    if theory[0] == 0:
        continue

    if theory_name not in theory_name_index:
        theory_name_index[theory_name] = len(theory_name_index)
    theory_index.append(theory_name_index[theory_name])

    w_set.append(data[i][w_index])

    a = float(data[i][a_index])
    c = float(data[i][c_index])
    ac_set.append([a, c])

    sci = SuperConformalIndex(data[i][sci_index].strip())
    sci_set.append(sci)

for name, index in theory_name_index.items():
    theory_index_name[index] = name

num_data = len(w_set)
print(f'Number of data: {num_data}')

class GraphCentralChargeModel(nn.Module):
    def __init__(self, dynkin_features: int, w_features: int, dynkin_hidden_channels: list[int],
                 w_hidden_channels: list[int],
                 w_linear: list[int], total_linear: list[int]):
        super(GraphCentralChargeModel, self).__init__()

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

        self.lin_w = FullyConnectedNetwork(
            w_hidden_channels[-1], w_linear[-1],
            *list(zip(w_linear[:-1], [nn.GELU() for _ in range(len(w_linear) - 1)]))
        )

        self.lin_total = FullyConnectedNetwork(
            dynkin_hidden_channels[-1] + w_linear[-1], total_linear[-1],
            *list(zip(total_linear[:-1], [nn.GELU() for _ in range(len(total_linear) - 1)]))
        )

        self.lin_final = nn.Sequential(
            nn.GELU(),
            nn.Linear(total_linear[-1], 2)
        )

    def forward(self, x_dynkin, x_w, edge_index_dynkin, edge_index_w, batch_dynkin, batch_w):
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

        x_w = self.lin_w(x_w)

        x_hidden_unprocessed = torch.cat((x_dynkin, x_w), dim=1)
        x_hidden_processed = self.lin_total(x_hidden_unprocessed)

        y = self.lin_final(x_hidden_processed)

        return y, x_hidden_unprocessed, x_hidden_processed


class GraphSpectrumExpectModel(nn.Module):
    def __init__(self, dynkin_features: int, w_features: int, dynkin_hidden_channels: list[int],
                 w_hidden_channels: list[int], input_spectrum_num: int, output_spectrum_num: int,
                 w_linear: list[int], total_linear: list[int]):
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

        self.lin_w = FullyConnectedNetwork(
            w_hidden_channels[-1] + input_spectrum_num, w_linear[-1],
            *list(zip(w_linear[:-1], [nn.GELU() for _ in range(len(w_linear) - 1)]))
        )

        self.lin_total = FullyConnectedNetwork(
            dynkin_hidden_channels[-1] + w_linear[-1], total_linear[-1],
            *list(zip(total_linear[:-1], [nn.GELU() for _ in range(len(total_linear) - 1)]))
        )

        self.lin_final = nn.Sequential(
            nn.GELU(),
            nn.Linear(total_linear[-1], output_spectrum_num)
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

        x_w = torch.cat((x_w, x_spectrum), dim=1)
        x_w = self.lin_w(x_w)

        x_total = torch.cat((x_dynkin, x_w), dim=1)
        x_total = self.lin_total(x_total)

        y = self.lin_final(x_total)
        return y


tmp_w_obj = Superpotential(theory_index_name[theory_index[0]], w_set[0])
dynkin_features = tmp_w_obj.get_theory_data().x.shape[1]
w_features = tmp_w_obj.get_superpotential_data().x.shape[1]
total_features = dynkin_features + w_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Avaliable device: {device}')

central_charge_model = GraphCentralChargeModel(
    dynkin_features, w_features,
    [dynkin_features * 2, dynkin_features * 2, dynkin_features * 2],
    [w_features * 2, w_features * 3, w_features * 3, w_features * 2],
    [
        w_features * 4,
        w_features * 4,
        w_features * 8,
        w_features * 8,
        w_features * 8,
        w_features * 8,
        w_features * 32,
        w_features * 32,
        w_features * 32,
        w_features * 32,
        w_features * 16,
        w_features * 16,
        w_features * 4,
        w_features * 2
    ],
    [
        total_features * 2,
        total_features,
    ]
).to(device)
print(f'Charge calculation model: {central_charge_model}')


def calculate_central_charge():
    epoch_num = int(input('Enter the number of epochs: '))

    dataset = []
    w_obj = Superpotential()

    prev_theory = None
    for i in range(len(w_set)):
        theory = serialize_theory_name(theory_index_name[theory_index[i]])
        if prev_theory != theory:
            prev_theory = theory
            w_obj.set_theory(theory)
        w_obj.set_superpotential(w_set[i])

        dynkin_diagram = w_obj.get_theory_data()
        superpotential_graph = w_obj.get_superpotential_data()

        w_data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                          edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                          y=torch.tensor([ac_set[i]]))
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(central_charge_model.parameters(), lr=1e-3)
    best_loss = 1e10

    checkpoint = None
    checkpoint_file_name = f'./checkpoint_charge_calc_v2.tar'
    if os.path.isfile(checkpoint_file_name):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_file_name, map_location=device)
        central_charge_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(epoch_num):
        central_charge_model.train()
        for _, data in enumerate(train_loader):
            x_dynkin = data.x_1.float().to(device)
            x_w = data.x_2.float().to(device)
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)
            y= data.y.float().to(device)

            outputs, _, _ = central_charge_model(
                x_dynkin, x_w,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        central_charge_model.eval()
        test_loss = 0.0
        error = 0.0
        test_cnt = 0

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x_dynkin = data.x_1.float().to(device)
                x_w = data.x_2.float().to(device)
                edge_index_dynkin = data.edge_index_1.to(device)
                edge_index_w = data.edge_index_2.to(device)
                batch_dynkin = data.x_1_batch.to(device)
                batch_w = data.x_2_batch.to(device)
                y = data.y.float().to(device)

                outputs, _, _ = central_charge_model(
                    x_dynkin, x_w,
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
                'model_state_dict': central_charge_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_file_name)

    final_loader = DataLoader(dataset, batch_size=256, shuffle=False, follow_batch=['x_1', 'x_2'])

    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    central_charge_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        error = np.array([])
        error_charge = None

        for _, data in enumerate(final_loader):
            x_dynkin = data.x_1.float().to(device)
            x_w = data.x_2.float().to(device)
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)
            y_real = data.y.cpu().numpy()

            y_expect = central_charge_model(
                x_dynkin, x_w,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )[0].cpu().numpy()

            error_raw = np.abs((y_expect - y_real) / y_real) * 100
            error = np.append(error, error_raw.flatten())
            error_charge = error_raw.transpose() if error_charge is None else np.hstack(
                (error_charge, error_raw.transpose()))

        error_max = np.max(error)
        print(f'Maximum error: {error_max}')

        json_data = dict()

        a_data = dict()
        sorted_errors = np.sort(error_charge[0], axis=None)
        a_data['min_error'] = float(sorted_errors[0])
        a_data['max_error'] = float(sorted_errors[-1])
        a_data['avg_error'] = float(np.mean(sorted_errors))
        a_data['median_error'] = float(median_sorted(sorted_errors))
        a_data['stdev_error'] = float(np.std(sorted_errors))
        json_data['a'] = a_data

        c_data = dict()
        sorted_errors = np.sort(error_charge[1], axis=None)
        c_data['min_error'] = float(sorted_errors[0])
        c_data['max_error'] = float(sorted_errors[-1])
        c_data['avg_error'] = float(np.mean(sorted_errors))
        c_data['median_error'] = float(median_sorted(sorted_errors))
        c_data['stdev_error'] = float(np.std(sorted_errors))
        json_data['c'] = c_data

        total_data = dict()
        sorted_errors = np.sort(error, axis=None)
        total_data['min_error'] = sorted_errors[0]
        total_data['max_error'] = sorted_errors[-1]
        total_data['avg_error'] = np.mean(sorted_errors)
        total_data['median_error'] = median_sorted(sorted_errors)
        total_data['stdev_error'] = np.std(sorted_errors)
        json_data['total'] = total_data

        with open(f'./data/{filename}_charge_calc_v2.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 15

        fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)

        fig.suptitle('Graph charge calculation errors')

        ax[0].hist(error, bins=math.ceil(error_max))
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Error (%)')
        ax[0].set_ylabel('Number of errors')
        ax[0].set_title('Graph charge calculation error distribution')

        ax[1].bar(['a', 'c'], [a_data['avg_error'], c_data['avg_error']])
        ax[1].set_xlabel('Error (%)')
        ax[1].set_ylabel('Central charge')
        ax[1].set_title('Average error or a and c charges')

        plt.savefig(f'./data/{filename}_charge_calc_v2.png')
        plt.show()


def expect_spectrum():
    input_spectrum_num = int(input('Enter input spectrum number: '))
    output_spectrum_num = int(input('Enter output spectrum number: '))
    epoch_num = int(input('Enter the number of epochs: '))

    dataset = []
    w_obj = Superpotential()

    prev_theory = None
    for i in range(len(w_set)):
        theory = serialize_theory_name(theory_index_name[theory_index[i]])
        if prev_theory != theory:
            prev_theory = theory
            w_obj.set_theory(theory)
        w_obj.set_superpotential(w_set[i])

        dynkin_diagram = w_obj.get_theory_data()
        superpotential_graph = w_obj.get_superpotential_data()

        sci = sci_set[i]
        spectrum_num = len(sci.dims)
        input_spectrum = [[-1 if j >= spectrum_num else sci.dims[j] for j in range(input_spectrum_num)]]
        output_spectrum = [[-1 if j + input_spectrum_num >= spectrum_num else sci.dims[j + input_spectrum_num] for j in range(output_spectrum_num)]]

        w_data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                          edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                          x_spectrum=torch.tensor(input_spectrum),
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

    w_total_num = w_features * 2 + input_spectrum_num
    spectrum_expect_model = GraphSpectrumExpectModel(
        dynkin_features, w_features,
        [dynkin_features * 2, dynkin_features * 2, dynkin_features * 2],
        [w_features * 2, w_features * 3, w_features * 3, w_features * 2],
        input_spectrum_num, output_spectrum_num,
        [
            w_total_num * 2,
            w_total_num * 2,
            w_total_num * 4,
            w_total_num * 4,
            w_total_num * 16,
            w_total_num * 16,
            w_total_num * 32,
            w_total_num * 32,
            w_total_num * 8,
            w_total_num * 8,
            w_total_num * 2,
            w_features * 2
        ],
        [
            total_features * 4,
            total_features * 4,
            total_features * 8,
            total_features * 8,
            total_features * 16,
            total_features * 16,
            total_features * 16,
            total_features * 16,
            total_features * 8,
            total_features * 8,
            total_features * 4,
            total_features * 4,
            total_features * 2,
            total_features * 2
        ]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(spectrum_expect_model.parameters(), lr=1e-3)
    best_loss = 1e10

    checkpoint = None
    checkpoint_file_name = f'./checkpoint_spectrum_expect_v2_{input_spectrum_num}_{output_spectrum_num}.tar'
    if os.path.isfile(checkpoint_file_name):
        print('Checkpoint available. Loads checkpoint...')
        checkpoint = torch.load(checkpoint_file_name, map_location=device)
        spectrum_expect_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']

    for epoch in range(epoch_num):
        spectrum_expect_model.train()
        for _, data in enumerate(train_loader):
            x_dynkin = data.x_1.float().to(device)
            x_w = data.x_2.float().to(device)
            x_spectrum = data.x_spectrum.float().to(device)
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)
            y = data.y.float().to(device)

            outputs = spectrum_expect_model(
                x_dynkin, x_w, x_spectrum,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        spectrum_expect_model.eval()
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

                outputs = spectrum_expect_model(
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
                'model_state_dict': spectrum_expect_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_file_name)

    final_loader = DataLoader(dataset, batch_size=256, shuffle=False, follow_batch=['x_1', 'x_2'])

    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    spectrum_expect_model.load_state_dict(checkpoint['model_state_dict'])
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

            y_expect = spectrum_expect_model(
                x_dynkin, x_w, x_spectrum,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            ).cpu().numpy()

            error_raw = np.abs((y_expect - y_real) / y_real) * 100
            error = np.append(error, error_raw.flatten())
            error_spectrum = error_raw.transpose() if error_spectrum is None else np.hstack(
                (error_spectrum, error_raw.transpose()))

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

        with open(f'./data/{filename}_spectrum_expect_v2_{input_spectrum_num}_{output_spectrum_num}.json',
                  'w') as json_file:
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

        plt.savefig(f'./data/{filename}_spectrum_expect_v2_{input_spectrum_num}_{output_spectrum_num}_graph.png')
        plt.show()


def clustering():
    checkpoint_file_name = f'./checkpoint_charge_calc_v2.tar'
    if not os.path.isfile(checkpoint_file_name):
        print('The checkpoint file of charge calculation model does not exist.')

    print('Choose the data to use.')
    print('1. Unprocessed hidden layer')
    print('2. Processed hidden layer')
    feature_index = int(input('>>'))
    assert feature_index == 1 or feature_index == 2

    dataset = []

    w_obj = Superpotential()
    prev_theory = None
    for i in range(len(w_set)):
        theory = serialize_theory_name(theory_index_name[theory_index[i]])
        if prev_theory != theory:
            prev_theory = theory
            w_obj.set_theory(theory)
        w_obj.set_superpotential(w_set[i])

        dynkin_diagram = w_obj.get_theory_data()
        superpotential_graph = w_obj.get_superpotential_data()

        w_data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                          edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                          y=torch.tensor([ac_set[i]]))
        dataset.append(w_data)

    optimizer = torch.optim.Adam(central_charge_model.parameters(), lr=1e-3)

    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    central_charge_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, follow_batch=['x_1', 'x_2'])

    features: np.ndarray = None
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            x_dynkin = data.x_1.float().to(device)
            x_w = data.x_2.float().to(device)
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)

            feature = central_charge_model(
                x_dynkin, x_w,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )[feature_index]
            feature = feature.cpu().numpy()
            features = feature if features is None else np.vstack([features, feature])

    dbscan = DBSCAN(leaf_size=30)
    dbscan.fit(features)
    n_cluster = np.max(dbscan.labels_) + 1
    print(f'Number of clusters: {n_cluster}')

    n_feature = dbscan.components_.shape[1]

    clustered_data = [[[] for _ in range(n_feature + 2)] for _ in
                      range(n_cluster)]  # first two are a and c charge and rests are hidden layer values
    theories_per_cluster = [[0 for _ in range(len(theory_name_index))] for _ in range(n_cluster)]
    noise_theories = [0 for _ in range(len(theory_name_index))]

    feature_name = ['Data', 'a', 'c'] + [f'hidden {i}' for i in range(n_feature)]
    n_noise = 0

    for i in range(num_data):
        cluster = dbscan.labels_[i]
        if cluster < 0:
            n_noise += 1
            noise_theories[theory_index[i]] += 1
            continue

        clustered_data[cluster][0].append(ac_set[i][0])
        clustered_data[cluster][1].append(ac_set[i][1])
        for j in range(n_feature):
            clustered_data[cluster][2 + j].append(features[i, j])

        theories_per_cluster[cluster][theory_index[i]] += 1

    clustered_data_stats = [[] for _ in range(n_cluster)]
    for cluster in range(n_cluster):
        for j in range(n_feature):
            clustered_data[cluster][j].sort()

        cluster_stat = {'Data': 'min'}
        cluster_stat.update(
            {feature_name[j + 1]: f'{clustered_data[cluster][j][0]}' if len(clustered_data[cluster][j]) > 0 else 0 for j
             in range(n_feature + 2)})
        clustered_data_stats[cluster].append(cluster_stat)

        cluster_stat = {'Data': 'max'}
        cluster_stat.update(
            {feature_name[j + 1]: f'{clustered_data[cluster][j][-1]}' if len(clustered_data[cluster][j]) > 0 else 0 for
             j in range(n_feature + 2)})
        clustered_data_stats[cluster].append(cluster_stat)

        cluster_stat = {'Data': 'average'}
        cluster_stat.update(
            {feature_name[j + 1]: f'{np.mean(clustered_data[cluster][j])}' if len(clustered_data[cluster][j]) > 0 else 0
             for j in range(n_feature + 2)})
        clustered_data_stats[cluster].append(cluster_stat)

        cluster_stat = {'Data': 'median'}
        cluster_stat.update({feature_name[j + 1]: f'{median_sorted(clustered_data[cluster][j])}' if len(
            clustered_data[cluster][j]) > 0 else 0 for j in range(n_feature + 2)})
        clustered_data_stats[cluster].append(cluster_stat)

    with open(f'./data/{filename}_ac_clustering_graph_v2_{'unprocessed' if feature_index == 1 else 'processed'}.csv', 'w', newline='') as csv_file:
        csv_file.write('Statistics per cluster\n')
        for cluster in range(n_cluster):
            csv_file.write(f'Cluster {cluster + 1}\n')
            writer = csv.DictWriter(csv_file, fieldnames=feature_name)

            writer.writeheader()
            writer.writerows(clustered_data_stats[cluster])
            csv_file.write('\n')

        csv_file.write('Theories per cluster\n')
        writer = csv.writer(csv_file)
        writer.writerow(['Cluster'] + [theory_index_name[i] for i in range(len(theory_name_index))])
        for cluster in range(n_cluster):
            writer.writerow([f'{cluster + 1}'] + theories_per_cluster[cluster])
        writer.writerow(['Noise'] + noise_theories)
        csv_file.write('\n')

        csv_file.write('Total data, Noises\n')
        csv_file.write(f'{num_data}, {n_noise}\n')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    ac_array = np.array(ac_set)
    fig, ax = plt.subplots()
    ax.scatter(ac_array[:, 0], ac_array[:, 1], s=1, c=dbscan.labels_)
    ax.set_xlabel('a')
    ax.set_ylabel('c')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', rotation='auto')
    fig.suptitle('KMeans cluster by hidden layer of charge ratio model')
    plt.savefig(f'./data/{filename}_ac_clustering_graph_v2_{'unprocessed' if feature_index == 1 else 'processed'}.png')

    plt.show()


def theory_distance():
    checkpoint_file_name = f'./checkpoint_charge_calc_v2.tar'
    if not os.path.isfile(checkpoint_file_name):
        print('The checkpoint file of charge calculation model does not exist.')

    print('Choose the data to use.')
    print('1. Unprocessed hidden layer')
    print('2. Processed hidden layer')
    feature_index = int(input('>>'))
    assert feature_index == 1 or feature_index == 2

    dataset = []

    w_obj = Superpotential()
    prev_theory = None
    for i in range(len(w_set)):
        theory = serialize_theory_name(theory_index_name[theory_index[i]])
        if prev_theory != theory:
            prev_theory = theory
            w_obj.set_theory(theory)
        w_obj.set_superpotential(w_set[i])

        dynkin_diagram = w_obj.get_theory_data()
        superpotential_graph = w_obj.get_superpotential_data()

        w_data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                          edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                          y=torch.tensor([ac_set[i]]))
        dataset.append(w_data)

    num_data = len(dataset)
    print(f'Number of data: {num_data}')

    num_sample = int(input("Enter the number of samples to calculate distance: "))

    optimizer = torch.optim.Adam(central_charge_model.parameters(), lr=1e-3)

    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    central_charge_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, follow_batch=['x_1', 'x_2'])

    features: np.ndarray = None
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            x_dynkin = data.x_1.float().to(device)
            x_w = data.x_2.float().to(device)
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)

            feature = central_charge_model(
                x_dynkin, x_w,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )[feature_index]
            feature = feature.cpu().numpy()
            features = feature if features is None else np.vstack([features, feature])

    sample_index = np.random.choice(num_data, num_sample, replace=False)
    ac_distance = np.array([])
    feature_distance = np.array([])

    ac_array = np.array(ac_set)

    for i in range(num_sample):
        sample_ac = ac_array[sample_index[i]]
        sample_feature = features[sample_index[i]]

        ac_dist = np.linalg.norm(ac_array - sample_ac, axis=1)
        feature_dist = np.linalg.norm(features - sample_feature, axis=1)

        ac_distance = np.append(ac_distance, ac_dist)
        feature_distance = np.append(feature_distance, feature_dist)

    z = np.polyfit(ac_distance, feature_distance, 1)
    p = np.poly1d(z)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    ac_dist_max = np.max(ac_distance)

    fig, ax = plt.subplots()
    ax.scatter(ac_distance, feature_distance, s=0.2)
    ax.plot([0, ac_dist_max], p([0, ac_dist_max]), "r--")
    ax.set_xlabel("Distance in ac space")
    ax.set_ylabel("Distance in hidden layer feature space")
    ax.tick_params(axis='both', rotation='auto')
    fig.suptitle('Comparing the distance in AC space and Feature space')
    plt.savefig(f'./data/{filename}_theory_distance_graph_v2_{'unprocessed' if feature_index == 1 else 'processed'}.png')

    plt.show()


while True:
    print('Program list...')
    print('1. Central charge calculation')
    print('2. Spectrum expectation')
    print('3. Clustering data with hidden layer values')
    print('4. Calculating theory distance with hidden layer values')
    print('-1. Exit')
    program = int(input('>>'))

    if program < 0:
        break
    elif program == 1:
        calculate_central_charge()
    elif program == 2:
        expect_spectrum()
    elif program == 3:
        clustering()
    elif program == 4:
        theory_distance()
