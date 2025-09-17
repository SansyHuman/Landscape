import csv
import os.path
from common.utils import prime_numbers
import math
from common.sci_parser import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

ac_ratio = np.array(a_charges)/np.array(c_charges)
smallest_dim = np.array(list(map(lambda sci: sci.smallest_dim, scis)))

plt.scatter(ac_ratio, smallest_dim, s=0.1)
plt.title('Charge ratio - smallest dimension')
plt.xlabel('a/c')
plt.ylabel('dimension')
plt.show()