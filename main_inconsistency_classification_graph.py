import os
from common.inconsistents_parser import inconsistents_graph_parser
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import numpy as np

os.makedirs('./data', exist_ok=True)

filename = input("Enter file name to load: ")
inc_path = input("Enter path of inconsistents log files: ")
epoch_num = int(input("Enter number of epochs: "))

dataset = inconsistents_graph_parser(os.path.abspath(filename), os.path.abspath(inc_path))
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


class GraphInconsistencyClassifier(nn.Module):
    def __init__(self, dynkin_features: int, w_features: int, dynkin_hidden_channels: list[int], w_hidden_channels: list[int], dropout=0.5):
        """
        Create a graph inconsistency classification model.
        :param dynkin_features: The number of features of dynkin diagram graph.
        :param w_features: The number of features of superpotential graph.
        :param dynkin_hidden_channels: Dimensions of GCN hidden channels for dynkin diagram graph.
        :param w_hidden_channels: Dimensions of GCN hidden channels for superpotential graph.
        :param dropout: The dropout rate of the model.
        """
        super(GraphInconsistencyClassifier, self).__init__()

        assert len(dynkin_hidden_channels) > 0 and len(w_hidden_channels) > 0

        self.dropout = dropout
        self.conv_dynkin: list[pyg_nn.GCNConv] = []
        self.conv_w: list[pyg_nn.GCNConv] = []

        self.conv_dynkin.append(pyg_nn.GCNConv(dynkin_features, dynkin_hidden_channels[0]))
        for i in range(len(dynkin_hidden_channels) - 1):
            self.conv_dynkin.append(pyg_nn.GCNConv(dynkin_hidden_channels[i], dynkin_hidden_channels[i + 1]))

        self.conv_w.append(pyg_nn.GCNConv(w_features, w_hidden_channels[0]))
        for i in range(len(w_hidden_channels) - 1):
            self.conv_w.append(pyg_nn.GCNConv(w_hidden_channels[i], w_hidden_channels[i + 1]))

        self.lin = nn.Linear(dynkin_hidden_channels[-1] + w_hidden_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_dynkin, x_w, edge_index_dynkin, edge_index_w, batch_dynkin, batch_w):
        for i in range(len(self.conv_dynkin)):
            x_dynkin = self.conv_dynkin[i](x_dynkin)
            if i != len(self.conv_dynkin) - 1:
                x_dynkin = F.elu(x_dynkin)
                x_dynkin = F.dropout(x_dynkin, p=self.dropout, training=self.training)
        x_dynkin = pyg_nn.global_mean_pool(x_dynkin, batch_dynkin)

        for i in range(len(self.conv_w)):
            x_w = self.conv_w[i](x_w)
            if i != len(self.conv_w) - 1:
                x_w = F.elu(x_w)
                x_w = F.dropout(x_w, p=self.dropout, training=self.training)
        x_w = pyg_nn.global_mean_pool(x_w, batch_w)

        x_total = torch.cat((x_dynkin, x_w), dim=1)

        x_total = F.dropout(x_total, p=self.dropout, training=self.training)
        x_total = self.lin(x_total)

        return self.sigmoid(x_total)
