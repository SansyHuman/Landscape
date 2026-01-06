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
ac_set = np.array(ac_set)

num_data = len(dataset)
print(f'Number of data: {num_data}')

num_sample = int(input("Enter the number of samples to calculate distance: "))
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

sample_index = np.random.choice(num_data, num_sample, replace=False)
ac_distance = np.array([])
feature_distance = np.array([])

for i in range(num_sample):
    sample_ac = ac_set[sample_index[i]]
    sample_feature = features[sample_index[i]]

    ac_dist = np.linalg.norm(ac_set - sample_ac, axis=1)
    feature_dist = np.linalg.norm(features - sample_feature, axis=1)

    ac_distance = np.append(ac_distance, ac_dist)
    feature_distance = np.append(feature_distance, feature_dist)

z = np.polyfit(ac_distance, feature_distance, 1)
p = np.poly1d(z)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots()
ax.scatter(ac_distance, feature_distance, s=0.2)
ax.plot([0, 300], p([0, 300]), "r--")
ax.set_xlabel("Distance in ac space")
ax.set_ylabel("Distance in hidden layer feature space")
ax.tick_params(axis='both', rotation='auto')
fig.suptitle('Comparing the distance in AC space and Feature space')
plt.savefig(f'./data/{filename}_theory_distance_graph.png')

plt.show()
