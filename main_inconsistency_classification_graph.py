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

        self.lin = nn.Linear(dynkin_hidden_channels[-1] + w_hidden_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_dynkin, x_w, edge_index_dynkin, edge_index_w, batch_dynkin, batch_w):
        for i in range(len(self.conv_dynkin)):
            x_dynkin = self.conv_dynkin[i](x_dynkin, edge_index_dynkin)
            if i != len(self.conv_dynkin) - 1:
                x_dynkin = self.norm_dynkin[i](x_dynkin, batch_dynkin)
                x_dynkin = F.elu(x_dynkin)
                x_dynkin = F.dropout(x_dynkin, p=self.dropout, training=self.training)
        x_dynkin = pyg_nn.global_mean_pool(x_dynkin, batch_dynkin)

        for i in range(len(self.conv_w)):
            x_w = self.conv_w[i](x_w, edge_index_w)
            if i != len(self.conv_w) - 1:
                x_w = self.norm_w[i](x_w, batch_w)
                x_w = F.elu(x_w)
                x_w = F.dropout(x_w, p=self.dropout, training=self.training)
        x_w = pyg_nn.global_mean_pool(x_w, batch_w)

        x_total = torch.cat((x_dynkin, x_w), dim=1)

        x_total = F.dropout(x_total, p=self.dropout, training=self.training)
        x_total = self.lin(x_total)

        return self.sigmoid(x_total)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Avaliable device: {device}')
criterion = nn.BCELoss()

dynkin_features=dataset[0].x_1.shape[1]
w_features=dataset[0].x_2.shape[1]

model = GraphInconsistencyClassifier(dynkin_features, w_features,
                                     [dynkin_features * 2, dynkin_features * 2, dynkin_features * 2],
                                     [w_features * 2, w_features * 3],
                                     dropout=1.0 / 3.0).to(device)

print(model)
batch = next(iter(test_loader))
print('Inconsistency classification model shape: ', model(
    batch.x_1.float().to(device), batch.x_2.float().to(device),
    batch.edge_index_1.to(device), batch.edge_index_2.to(device),
    batch.x_1_batch.to(device), batch.x_2_batch.to(device)
).shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_loss = 1e10

checkpoint = None
checkpoint_file_name = f'./checkpoint_inconsistency_classification_graph.tar'
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
        edge_index_dynkin = data.edge_index_1.to(device)
        edge_index_w = data.edge_index_2.to(device)
        batch_dynkin = data.x_1_batch.to(device)
        batch_w = data.x_2_batch.to(device)
        y = data.y.float().to(device)

        outputs = model(
            x_dynkin, x_w,
            edge_index_dynkin, edge_index_w,
            batch_dynkin, batch_w
        )
        outputs = torch.squeeze(outputs)
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
            edge_index_dynkin = data.edge_index_1.to(device)
            edge_index_w = data.edge_index_2.to(device)
            batch_dynkin = data.x_1_batch.to(device)
            batch_w = data.x_2_batch.to(device)
            y = data.y.float().to(device)

            outputs = model(
                x_dynkin, x_w,
                edge_index_dynkin, edge_index_w,
                batch_dynkin, batch_w
            )
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, y)

            test_loss += loss.item()

    print(f'epoch {epoch + 1} test loss: {test_loss / len(test_loader)}')
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
    cons_correct = 0
    cons_wrong = 0
    incons_correct = 0
    incons_wrong = 0

    for _, data in enumerate(final_loader):
        x_dynkin = data.x_1.float().to(device)
        x_w = data.x_2.float().to(device)
        edge_index_dynkin = data.edge_index_1.to(device)
        edge_index_w = data.edge_index_2.to(device)
        batch_dynkin = data.x_1_batch.to(device)
        batch_w = data.x_2_batch.to(device)
        y_real = data.y

        y_expect = model(
            x_dynkin, x_w,
            edge_index_dynkin, edge_index_w,
            batch_dynkin, batch_w
        )
        y_expect = torch.squeeze(y_expect)

        for i in range(len(y_expect)):
            if y_real[i] == 0:
                if y_expect[i] < 0.5:
                    cons_correct += 1
                else:
                    cons_wrong += 1
            elif y_real[i] == 1:
                if y_expect[i] >= 0.5:
                    incons_correct += 1
                else:
                    incons_wrong += 1

    cons_error = cons_wrong / (cons_correct + cons_wrong) * 100
    incons_error = incons_wrong / (incons_correct + incons_wrong) * 100
    total_error = (cons_wrong + incons_wrong) / (cons_correct + incons_correct + cons_wrong + incons_wrong) * 100

    with open(f'./data/{filename}_inconsistency_classification_graph.csv', 'w') as csv_file:
        csv_file.write(', Correct, Incorrect, Error (%)\n')
        csv_file.write(f'Consistent, {cons_correct}, {cons_wrong}, {cons_error}\n')
        csv_file.write(f'Inconsistent, {incons_correct}, {incons_wrong}, {incons_error}\n')
        csv_file.write(f'Total, {cons_correct + incons_correct}, {cons_wrong + incons_wrong}, {total_error}')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots()

    fig.suptitle('Inconsistency classification errors')

    p = ax.bar(['Consistent', 'Inconsistent', 'Total'], [cons_error, incons_error, total_error])
    ax.bar_label(p, fmt='%.2f')
    ax.set_ylabel('Error (%)')

    plt.savefig(f'./data/{filename}_inconsistency_classification_graph.png')
    plt.show()
