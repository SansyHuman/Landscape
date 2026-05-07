from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from common.balanced_sample_tool import TheorySampler
import csv
import sys
import os.path
import math
import json
import pathlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from common.sci_parser import SuperConformalIndex

os.makedirs('./data/clustering', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")

theory_sampler = TheorySampler(filename)

min_a = float(input("Enter minimal value of a central charge: "))
max_a = float(input("Enter maximal value of a central charge: "))
min_c = float(input("Enter minimal value of c central charge: "))
max_c = float(input("Enter maximal value of c central charge: "))
n_samples = int(input("Enter number of samples per theory: "))
n_exponents = int(input("Enter number of exponents to use from SCI: "))
n_reduced = int(input("Enter the reduced dimension of exponents data: "))

sampled = theory_sampler.get_balanced_sample((min_a, max_a), (min_c, max_c), n_samples)
sample_stat = sampled.get_theory_stats()

n_theory = sampled.get_theory_num()
theories = sample_stat["Name"].to_list()
print("The number of theories in the sample: ", n_theory)
print("Theories in the sample: ", theories)

theories_dict = dict()
for i in range(len(theories)):
    theories_dict[theories[i]] = i

data_num = sampled.df.height
theory_data = []
a_data = []
c_data = []
sci_exp_data = []

for i in range(data_num):
    theory_data.append(theories_dict[sampled.df["Name"][i]])
    a_data.append(float(sampled.df["CentralChargeA"][i]))
    c_data.append(float(sampled.df["CentralChargeC"][i]))
    sci = SuperConformalIndex(sampled.df["SCI"][i])
    exp_data = [sci.dims[j] if j < len(sci.dims) else 0 for j in range(n_exponents)]
    sci_exp_data.append(exp_data)

a_data = np.array(a_data)
c_data = np.array(c_data)
sci_exp_data = np.array(sci_exp_data)

reduction_model = TSNE(n_components=n_reduced)
sci_exp_embedded = reduction_model.fit_transform(sci_exp_data)

kmeans = KMeans(n_clusters=n_theory)
kmeans.fit(sci_exp_embedded)

print(kmeans.labels_)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)

fig.suptitle(f'K-Means clustering with t-SNE')

ax[0].set_title(f'Real data exponent number: {n_exponents}')
real_group_data = [[[], []] for _ in range(n_theory)]
for i in range(data_num):
    real_group_data[theory_data[i]][0].append(a_data[i])
    real_group_data[theory_data[i]][1].append(c_data[i])

cmap = plt.cm.get_cmap('jet', n_theory)

for i in range(n_theory):
    ax[0].scatter(real_group_data[i][0], real_group_data[i][1], color=cmap(i), label=theories[i])

ax[0].legend()
ax[0].set_xlabel('a charge')
ax[0].set_ylabel('c charge')

ax[1].set_title(f'Clustered data with t-SNE reduced dimension: {n_reduced}')
cluster_group_data = [[[], []] for _ in range(n_theory)]
for i in range(data_num):
    cluster_group_data[kmeans.labels_[i]][0].append(a_data[i])
    cluster_group_data[kmeans.labels_[i]][1].append(c_data[i])

for i in range(n_theory):
    ax[1].scatter(cluster_group_data[i][0], cluster_group_data[i][1], color=cmap(i), label=f'Cluster {i}')

ax[1].legend()
ax[1].set_xlabel('a charge')
ax[1].set_ylabel('c charge')

plt.show()
