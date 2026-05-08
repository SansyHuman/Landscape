from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import polars as pl

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

os.makedirs('../data/clustering', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")

theory_sampler = TheorySampler(filename)
for row in theory_sampler.get_theory_stats().iter_rows():
    print(row)


def cluster_data(sampled: TheorySampler, n_exponents: int, n_reduced: int, savefile_suffix: str):
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
    cluster_group_stats = [[0 for _ in range(n_theory)] for _ in range(n_theory)]
    for i in range(data_num):
        cluster_group_data[kmeans.labels_[i]][0].append(a_data[i])
        cluster_group_data[kmeans.labels_[i]][1].append(c_data[i])
        cluster_group_stats[kmeans.labels_[i]][theory_data[i]] += 1

    with open(f'../data/clustering/sci_exp_clustering_{n_exponents}_{n_reduced}_{savefile_suffix}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Cluster'] + theories)
        for i in range(n_theory):
            writer.writerow([i] + cluster_group_stats[i])

    most_theory = [0 for _ in range(n_theory)]
    for i in range(n_theory):
        max_value = -1
        max_index = -1
        for j in range(n_theory):
            if cluster_group_stats[i][j] > max_value:
                max_value = cluster_group_stats[i][j]
                max_index = j
        most_theory[i] = max_index

    for i in range(n_theory):
        ax[1].scatter(cluster_group_data[i][0], cluster_group_data[i][1], color=cmap(most_theory[i]),
                      label=theories[most_theory[i]])

    ax[1].legend()
    ax[1].set_xlabel('a charge')
    ax[1].set_ylabel('c charge')

    plt.savefig(
        f'../data/clustering/sci_exp_clustering_{n_exponents}_{n_reduced}_{savefile_suffix}.png')
    plt.show()


def clustering_charge_range():
    min_a = float(input("Enter minimal value of a central charge: "))
    max_a = float(input("Enter maximal value of a central charge: "))
    min_c = float(input("Enter minimal value of c central charge: "))
    max_c = float(input("Enter maximal value of c central charge: "))
    n_samples = int(input("Enter number of samples per theory: "))
    n_exponents = int(input("Enter number of exponents to use from SCI: "))
    n_reduced = int(input("Enter the reduced dimension of exponents data: "))

    sampled = theory_sampler.get_balanced_sample((min_a, max_a), (min_c, max_c), n_samples)
    cluster_data(sampled, n_exponents, n_reduced, f'({min_a}_{max_a})({min_c}_{max_c})_{n_samples}')


def clustering_manual_selection():
    selected = []
    print('Write all theories you want to choose separated with comma.')
    theories = input('>>>').split(',')
    for theory in theories:
        selected.append(theory.strip())
    selected = sorted(selected)

    n_samples = int(input("Enter number of samples per theory: "))
    n_exponents = int(input("Enter number of exponents to use from SCI: "))
    n_reduced = int(input("Enter the reduced dimension of exponents data: "))

    sampled = theory_sampler.get_manual_sample(selected, n_samples)
    cluster_data(sampled, n_exponents, n_reduced, f'{selected}_{n_samples}')


while True:
    print('Select program...')
    print('1. Select theories within the central charge range')
    print('2. Select theories manually')
    print('-1. Exit')
    program = int(input('>>>'))

    if program == 1:
        clustering_charge_range()
    elif program == 2:
        clustering_manual_selection()
    else:
        exit()
