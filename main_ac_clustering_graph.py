import csv
import sys
import os.path
import math
import json
import pathlib

from matplotlib.widgets import Slider

from common.inconsistents_parser import serialize_theory_name
from common.sci_parser import *

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib as mpl

from torch_geometric.loader import DataLoader
from common.superpotential_parser import Superpotential

os.makedirs('./data', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

field_content_index, w_index, a_index, c_index = -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "Superpotentials":
        w_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i

dataset = []
ac_set = []
theory_index = []
theory_name_index = dict()

w_obj = Superpotential()

prev_theory = None
for i in range(1, len(data)):
    theory_name = data[i][field_content_index]
    theory = serialize_theory_name(theory_name)
    if theory[0] == 0:
        continue

    if prev_theory != theory:
        prev_theory = theory
        w_obj.set_theory(theory)
    w_obj.set_superpotential(data[i][w_index])

    dynkin_diagram = w_obj.get_theory_data()
    superpotential_graph = w_obj.get_superpotential_data()
    a = float(data[i][a_index])
    c = float(data[i][c_index])

    w_data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                    edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                    y=torch.tensor([[a / c]]))
    dataset.append(w_data)
    ac_set.append([a, c])
    if theory_name not in theory_name_index:
        theory_name_index[theory_name] = len(theory_name_index)
    theory_index.append(theory_name_index[theory_name])
ac_set = np.array(ac_set)

num_data = len(dataset)
print(f'Number of data: {num_data}')

checkpoint_file_name = input("Enter checkpoint file name of the graph charge ratio expectation model: ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Avaliable device: {device}')
criterion = nn.MSELoss()

dynkin_features=dataset[0].x_1.shape[1]
w_features=dataset[0].x_2.shape[1]
total_features = dynkin_features + w_features

from main_charge_ratio_calc_graph import GraphCentralChargeModel
model = GraphCentralChargeModel(dynkin_features, w_features,
                                [dynkin_features * 2, dynkin_features * 2, dynkin_features * 2],
                                [w_features * 2, w_features * 3, w_features * 3, w_features * 2],
                                [
                                    total_features * 2,
                                    total_features * 8,
                                    total_features * 16,
                                    total_features * 16,
                                    total_features * 4,
                                    total_features * 4
                                ]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

checkpoint = torch.load(checkpoint_file_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
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

        _, feature = model(
            x_dynkin, x_w,
            edge_index_dynkin, edge_index_w,
            batch_dynkin, batch_w
        )
        feature = feature.cpu().numpy()
        features = feature if features is None else np.vstack([features, feature])
    # features = np.hstack((features, np.array(ac_set)))

dbscan = DBSCAN(leaf_size=30)
dbscan.fit(features)
n_cluster = np.max(dbscan.labels_) + 1
print(f'Number of clusters: {n_cluster}')

n_feature = dbscan.components_.shape[1]

clustered_data = [[[] for _ in range(n_feature + 2)] for _ in range(n_cluster)] # first two are a and c charge and rests are hidden layer values
theories_per_cluster = [[0 for _ in range(len(theory_name_index))] for _ in range(n_cluster)]
noise_theories = [0 for _ in range(len(theory_name_index))]

theory_index_name = dict()
for theory_name, index in theory_name_index.items():
    theory_index_name[index] = theory_name
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
    cluster_stat.update({feature_name[j + 1]: f'{clustered_data[cluster][j][0]}' if len(clustered_data[cluster][j]) > 0 else 0 for j in range(n_feature + 2)})
    clustered_data_stats[cluster].append(cluster_stat)

    cluster_stat = {'Data': 'max'}
    cluster_stat.update({feature_name[j + 1]: f'{clustered_data[cluster][j][-1]}' if len(clustered_data[cluster][j]) > 0 else 0 for j in range(n_feature + 2)})
    clustered_data_stats[cluster].append(cluster_stat)

    cluster_stat = {'Data': 'average'}
    cluster_stat.update({feature_name[j + 1]: f'{np.mean(clustered_data[cluster][j])}' if len(clustered_data[cluster][j]) > 0 else 0 for j in range(n_feature + 2)})
    clustered_data_stats[cluster].append(cluster_stat)

    cluster_stat = {'Data': 'median'}
    cluster_stat.update({feature_name[j + 1]: f'{median_sorted(clustered_data[cluster][j])}' if len(clustered_data[cluster][j]) > 0 else 0 for j in range(n_feature + 2)})
    clustered_data_stats[cluster].append(cluster_stat)

with open(f'./data/{filename}_ac_clustering_graph.csv', 'w', newline='') as csv_file:
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

fig, ax = plt.subplots()
ax.scatter(ac_set[:, 0], ac_set[:, 1], s=1, c=dbscan.labels_)
ax.set_xlabel('a')
ax.set_ylabel('c')
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', rotation='auto')
fig.suptitle('KMeans cluster by hidden layer of charge ratio model')
plt.savefig(f'./data/{filename}_ac_clustering_graph.png')

plt.show()
