import csv
import os.path
import math
from common.sci_parser import *

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = None
with open("landscape_SU2adj1nf2.csv") as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

a_index, c_index, sci_index = -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "SCI":
        sci_index = i

print(f'A: {a_index}, C: {c_index}, SCI: {sci_index}')

a_charges = []
c_charges = []
scis = []

for i in range(1, len(data)):
    a, c = float(data[i][a_index]), float(data[i][c_index])
    sci = SuperConformalIndex(data[i][sci_index].strip())
    a_charges.append(a)
    c_charges.append(c)
    scis.append(sci)

# simple plot of a/c and smallest dimension
"""
ac_ratio = np.array(a_charges)/np.array(c_charges)
smallest_dim = np.array(list(map(lambda sci: sci.smallest_dim, scis)))

plt.scatter(ac_ratio, smallest_dim, s=0.1)
plt.title('Charge ratio - smallest dimension')
plt.xlabel('a/c')
plt.ylabel('dimension')
plt.show()
"""

# simple kmeans with smallest and second smallest dimension
two_dims = np.array(list(map(lambda sci: [sci.smallest_dim, sci.relevant_dims[1]], scis)))

kmeans = KMeans(n_clusters=2)
kmeans.fit(two_dims)
print(f'Iteration number: {kmeans.n_iter_}')
print(f'Cluster centers: {kmeans.cluster_centers_}')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots()
ax.scatter(a_charges, c_charges, s=0.15, c=kmeans.labels_)
ax.set_xlabel('a')
ax.set_ylabel('c')
ax.tick_params(axis='both', rotation='auto')
ax.set_title('KMeans cluster by first two smallest dimensions')

plt.show()