import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import polars as pl
from sklearn.preprocessing import StandardScaler

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

GRID_LO = float(input("Enter lower bound of feature grid: "))
GRID_HI = float(input("Enter upper bound of feature grid: "))
GRID_STEP = float(input("Enter step size of feature grid: "))

GRID = np.arange(GRID_LO, GRID_HI + GRID_STEP, GRID_STEP)
KDE_BANDWIDTH = float(input("Enter bandwidth of feature grid: "))


def cluster_data(sampled: TheorySampler, savefile_suffix: str, show_graph: bool=False, save_dir=None):
    if save_dir is None:
        save_dir = "../data/clustering"
    os.makedirs(save_dir, exist_ok=True)

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
    sci_data: list[SuperConformalIndex] = []

    for i in range(data_num):
        theory_data.append(theories_dict[sampled.df["Name"][i]])
        sci_data.append(SuperConformalIndex(sampled.df["SCI"][i]))

    X = np.stack([sci_data[i].featurize_dimensions(GRID, KDE_BANDWIDTH) for i in range(data_num)])
    y_true = np.asarray(theory_data)

    Xs = StandardScaler().fit_transform(X)

    reduction_model = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42
    )
    X_tsne = reduction_model.fit_transform(Xs)

    pca_model = PCA(n_components=2, random_state=42)
    X_pca = pca_model.fit_transform(Xs)

    kmeans = KMeans(n_clusters=n_theory, n_init=10, random_state=42)
    kmeans.fit(X_tsne)
    y_pred = kmeans.labels_

    cluster_group_stats = [[0 for _ in range(n_theory)] for _ in range(n_theory)]
    for i in range(data_num):
        cluster_group_stats[y_pred[i]][y_true[i]] += 1

    cluster_names = dict()
    cluster_names_used = set()
    for i in range(n_theory):
        stat_rank = np.argsort(cluster_group_stats[i])[::-1]
        for j in range(n_theory):
            if stat_rank[j] in cluster_names_used:
                continue
            cluster_names[i] = stat_rank[j]
            cluster_names_used.add(stat_rank[j])
            break

    accuracy_stats = [0.0 for _ in range(n_theory)]
    total_correct = 0
    total_incorrect = 0
    for i in range(n_theory):
        cluster_theory = cluster_names[i]

        correct = cluster_group_stats[i][cluster_theory]
        incorrect = sum(cluster_group_stats[i]) - correct

        accuracy_stats[i] = correct / (correct + incorrect)
        total_correct += correct
        total_incorrect += incorrect
    total_accuracy = total_correct / (total_correct + total_incorrect)

    with open(f'{save_dir}/sci_exp_clustering_{savefile_suffix}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Cluster'] + theories + ['Accuracy'])
        for i in range(n_theory):
            writer.writerow([f'{i} ({theories[cluster_names[i]]})'] + cluster_group_stats[i] + [accuracy_stats[i]])

    cmap = plt.cm.get_cmap('jet', n_theory)

    def _scatter(ax, coords, labels, names, title):
        for cls in range(n_theory):
            m = labels == cls
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                color=cmap(names[cls]),
                label=theories[names[cls]],
                alpha=0.8,
                edgecolor="white",
                s=60,
            )
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
        ax.grid(True, alpha=0.3)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['font.size'] = 12

    plt.close('all')

    fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True)

    fig.suptitle(f'K-Means clustering with t-SNE')

    tmp = [i for i in range(n_theory)]
    real_names = dict(zip(tmp, tmp))
    _scatter(
        ax[0],
        X_pca,
        y_true,
        real_names,
        'Real data with PCA',
    )
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')

    _scatter(
        ax[1],
        X_tsne,
        y_true,
        real_names,
        "Real data with t-SNE",
    )
    ax[1].set_xlabel("t-SNE 1")
    ax[1].set_ylabel("t-SNE 2")

    _scatter(
        ax[2],
        X_tsne,
        y_pred,
        cluster_names,
        f"KMeans clusters with t-SNE (acc={total_accuracy:.3})",
    )
    ax[2].scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=180,
        c="black",
        label="centroids",
        zorder=5,
    )
    ax[2].legend(frameon=False, loc="best")
    ax[2].set_xlabel("t-SNE 1")
    ax[2].set_ylabel("t-SNE 2")

    plt.savefig(
        f'{save_dir}/sci_exp_clustering_{savefile_suffix}.png')
    if show_graph:
        plt.show()

    return total_accuracy


def save_accuracy(save_dir: str, savefile_suffix: str, accuracies: list[float], total_accuracy: float):
    with open(f'{save_dir}/sci_exp_clustering_{savefile_suffix}_total.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Iteration'] + [i + 1 for i in range(len(accuracies))] + ['Total'])
        writer.writerow(['Accuracy'] + accuracies + [total_accuracy])


def clustering_charge_range():
    min_a = float(input("Enter minimal value of a central charge: "))
    max_a = float(input("Enter maximal value of a central charge: "))
    min_c = float(input("Enter minimal value of c central charge: "))
    max_c = float(input("Enter maximal value of c central charge: "))
    n_samples = int(input("Enter number of samples per theory: "))
    n_iter = int(input("Enter number of iterations: "))

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    save_suffix = f'({min_a}_{max_a})({min_c}_{max_c})_{n_samples}'

    accuracies = []
    for i in range(n_iter):
        sampled = theory_sampler.get_balanced_sample((min_a, max_a), (min_c, max_c), n_samples)
        accuracy = cluster_data(sampled, save_suffix + f'_{i + 1}', save_dir=save_dir, show_graph=i == n_iter - 1)
        print(f'Accuracy of iteration {i + 1}: {accuracy:.3f}')
        accuracies.append(accuracy)

    total_accuracy = sum(accuracies) / len(accuracies)
    print(f'Total accuracy: {total_accuracy:.3f}')

    save_accuracy(save_dir, save_suffix, accuracies, total_accuracy)


def clustering_manual_selection():
    selected = []
    print('Write all theories you want to choose separated with comma.')
    theories = input('>>>').split(',')
    for theory in theories:
        selected.append(theory.strip())
    selected = sorted(selected)

    n_samples = int(input("Enter number of samples per theory: "))
    n_iter = int(input("Enter number of iterations: "))

    save_dir = f"../data/clustering/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    save_suffix = f'{selected}_{n_samples}'

    accuracies = []
    for i in range(n_iter):
        sampled = theory_sampler.get_manual_sample(selected, n_samples)
        accuracy = cluster_data(sampled, save_suffix + f'_{i + 1}', save_dir=save_dir, show_graph=i == n_iter - 1)
        print(f'Accuracy of iteration {i + 1}: {accuracy:.3f}')
        accuracies.append(accuracy)

    total_accuracy = sum(accuracies) / len(accuracies)
    print(f'Total accuracy: {total_accuracy:.3f}')

    save_accuracy(save_dir, save_suffix, accuracies, total_accuracy)


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
