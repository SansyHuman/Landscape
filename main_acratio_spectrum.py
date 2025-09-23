import csv
import os.path
import math
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


def lightest_ac_ratio(field_contents: list[int], a_charges: list[float], c_charges: list[float], scis: list[SuperConformalIndex]) -> None:
    # simple plot of a/c and smallest dimension
    ac_ratio = np.array(a_charges)/np.array(c_charges)
    smallest_dim = np.array(list(map(lambda sci: sci.smallest_dim, scis)))

    plt.scatter(ac_ratio, smallest_dim, s=1, c=field_contents, cmap='Spectral')
    plt.title('Charge ratio - smallest dimension')
    plt.xlabel('a/c')
    plt.ylabel('dimension')
    plt.colorbar()
    plt.show()


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
