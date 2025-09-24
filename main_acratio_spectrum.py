import csv
import os.path
import math
import json
from common.sci_parser import *

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

filename = input("Enter file name to load: ")

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

field_content_index, a_index, c_index, sci_index = -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "SCI":
        sci_index = i

print(f'Field content: {field_content_index}, A: {a_index}, C: {c_index}, SCI: {sci_index}')

field_contents_index = dict()
field_contents = []
a_charges = []
c_charges = []
scis = []

for i in range(1, len(data)):
    field_content = data[i][field_content_index]
    a, c = float(data[i][a_index]), float(data[i][c_index])
    sci = SuperConformalIndex(data[i][sci_index].strip())

    if field_content not in field_contents_index:
        field_contents_index[field_content] = len(field_contents_index)
    field_contents.append(field_contents_index[field_content])
    a_charges.append(a)
    c_charges.append(c)
    scis.append(sci)

print(f"Field contents: {field_contents_index}")

os.makedirs('./data', exist_ok=True)


def lightest_ac_ratio(field_contents: list[int], a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex]) -> None:
    # simple plot of a/c and smallest dimension
    ac_ratio = np.array(a_charges)/np.array(c_charges)
    smallest_dim = np.array(list(map(lambda sci: sci.smallest_dim, scis)))

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    plt.scatter(ac_ratio, smallest_dim, s=1, c=field_contents, cmap='Spectral')
    plt.title('Charge ratio - smallest dimension')
    plt.xlabel('a/c')
    plt.ylabel('dimension')
    plt.colorbar()

    plt.savefig(f'data/{filename}_acratio_spectrum.png')
    plt.show()

def __median_sorted(data):
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2 - 1] + data[n // 2]) / 2
    else:
        return data[(n - 1) // 2]


def save_data(data, data_name: list[str], cluster_obj: KMeans, test_name: str) -> None:
    # save data of clustering
    n_clusters = cluster_obj.n_clusters
    n_data = len(data_name)
    clustered_data = [[[] for _ in range(n_data)] for _ in range(n_clusters)]

    for i in range(len(data)):
        cluster = cluster_obj.labels_[i]
        for j in range(n_data):
            clustered_data[cluster][j].append(data[i][j])

    json_data = dict()
    json_data['data_name'] = data_name
    json_data['clusters'] = [dict() for _ in range(n_clusters)]

    for i in range(n_clusters):
        json_data['clusters'][i]['num_data'] = len(clustered_data[i][0])
        json_data['clusters'][i]['center'] = list(cluster_obj.cluster_centers_[i])
        for j in range(n_data):
            clustered_data[i][j].sort()
        json_data['clusters'][i]['min'] = [clustered_data[i][k][0] for k in range(n_data)]
        json_data['clusters'][i]['max'] = [clustered_data[i][k][-1] for k in range(n_data)]
        json_data['clusters'][i]['average'] = [np.mean(clustered_data[i][k]) for k in range(n_data)]
        json_data['clusters'][i]['median'] = [__median_sorted(clustered_data[i][k]) for k in range(n_data)]

        with open(f'./data/{filename}_{test_name}.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


def kmeans_second_lightest(a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex], clusters: int) -> None:
    # simple kmeans with smallest and second smallest dimension
    two_dims = np.array([[sci.smallest_dim, (sci.relevant_dims[1] if len(sci.relevant_dims) > 1 else 0)] for sci in scis])

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(two_dims)
    print(f'Iteration number: {kmeans.n_iter_}')
    print(f'Cluster centers: {kmeans.cluster_centers_}')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(1, 2, squeeze=True)
    ax[0].scatter(a_charges, c_charges, s=1, c=kmeans.labels_)
    ax[0].set_xlabel('a')
    ax[0].set_ylabel('c')
    ax[0].tick_params(axis='both', rotation='auto')
    ax[0].set_title('a-c space')

    ax[1].scatter(two_dims[:,0], two_dims[:,1], s=1, c=kmeans.labels_)
    ax[1].scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               c='b', marker='x', linewidths=2)
    ax[1].set_xlabel('lightest dim')
    ax[1].set_ylabel('second lightest dim')
    ax[1].tick_params(axis='both', rotation='auto')
    ax[1].set_title('dimension space')

    fig.suptitle('KMeans cluster by first two smallest dimensions')

    save_data(two_dims,['smallest_dim', 'second_smallest_dim'], kmeans, test_name=kmeans_second_lightest.__name__)
    plt.savefig(f'./data/{filename}_{kmeans_second_lightest.__name__}.png')

    plt.show()

def kmeans_ac_second_lightest(a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex], clusters: int) -> None:
    # simple kmeans with a, c central charges and smallest and second smallest dimension
    two_dims = np.array([[a_charges[i], c_charges[i], scis[i].smallest_dim, (scis[i].relevant_dims[1] if len(scis[i].relevant_dims) > 1 else 0)] for i in range(len(scis))])

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(two_dims)
    print(f'Iteration number: {kmeans.n_iter_}')
    print(f'Cluster centers: {kmeans.cluster_centers_}')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(1, 2, squeeze=True)
    ax[0].scatter(a_charges, c_charges, s=1, c=kmeans.labels_)
    ax[0].scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               c='b', marker='x', linewidths=2)
    ax[0].set_xlabel('a')
    ax[0].set_ylabel('c')
    ax[0].tick_params(axis='both', rotation='auto')
    ax[0].set_title('a-c space')

    ax[1].scatter(two_dims[:, 2], two_dims[:, 3], s=1, c=kmeans.labels_)
    ax[1].scatter(kmeans.cluster_centers_[:, 2],
                  kmeans.cluster_centers_[:, 3],
                  c='b', marker='x', linewidths=2)
    ax[1].set_xlabel('lightest dim')
    ax[1].set_ylabel('second lightest dim')
    ax[1].tick_params(axis='both', rotation='auto')
    ax[1].set_title('dimension space')

    fig.suptitle('KMeans cluster by ac charge and first two smallest dimensions')

    save_data(two_dims, ['a_charge', 'c_charge', 'smallest_dim', 'second_smallest_dim'], kmeans, test_name=kmeans_ac_second_lightest.__name__)
    plt.savefig(f'./data/{filename}_{kmeans_ac_second_lightest.__name__}.png')

    plt.show()


def kmeans_ac_lightest_num(a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex], clusters: int) -> None:
    # simple kmeans with a, c central charges and dimension and number of lighetst operators
    two_dims = np.array([[a_charges[i], c_charges[i], scis[i].smallest_dim, scis[i].relevant_spectrum[scis[i].smallest_dim]] for i in range(len(scis))])

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(two_dims)
    print(f'Iteration number: {kmeans.n_iter_}')
    print(f'Cluster centers: {kmeans.cluster_centers_}')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(1, 2, squeeze=True)
    ax[0].scatter(a_charges, c_charges, s=1, c=kmeans.labels_)
    ax[0].scatter(kmeans.cluster_centers_[:, 0],
                  kmeans.cluster_centers_[:, 1],
                  c='b', marker='x', linewidths=2)
    ax[0].set_xlabel('a')
    ax[0].set_ylabel('c')
    ax[0].tick_params(axis='both', rotation='auto')
    ax[0].set_title('a-c space')

    ax[1].scatter(two_dims[:, 2], two_dims[:, 3], s=1, c=kmeans.labels_)
    ax[1].scatter(kmeans.cluster_centers_[:, 2],
                  kmeans.cluster_centers_[:, 3],
                  c='b', marker='x', linewidths=2)
    ax[1].set_xlabel('lightest dim')
    ax[1].set_ylabel('num of lightest dim')
    ax[1].tick_params(axis='both', rotation='auto')
    ax[1].set_title('dimension space')

    fig.suptitle('KMeans cluster by ac charge and dimension and the number of lightest operator')

    save_data(two_dims, ['a_charge', 'c_charge', 'smallest_dim', 'num_smallest_dim'], kmeans,
              test_name=kmeans_ac_lightest_num.__name__)
    plt.savefig(f'./data/{filename}_{kmeans_ac_lightest_num.__name__}.png')

    plt.show()


while True:
    print("Choose the program.")
    print("1. simple plot of a/c and smallest dimension")
    print("2. simple kmeans with smallest and second smallest dimension")
    print("3. simple kmeans with a, c central charges and smallest and second smallest dimension")
    print("4. simple kmeans with a, c central charges and dimension and number of lighetst operators")
    print('-1. exit')

    program = int(input(">>"))
    n_clusters = 0
    if program > 1:
        print("Input the number of clusters.")
        n_clusters = int(input(">>"))

    if program == 1:
        lightest_ac_ratio(field_contents, a_charges, c_charges, scis)
    elif program == 2:
        kmeans_second_lightest(a_charges, c_charges, scis, n_clusters)
    elif program == 3:
        kmeans_ac_second_lightest(a_charges, c_charges, scis, n_clusters)
    elif program == 4:
        kmeans_ac_lightest_num(a_charges, c_charges, scis, n_clusters)
    elif program == -1:
        break
