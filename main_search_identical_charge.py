import csv
import json
import os.path
import random
import math

import numpy as np


os.makedirs('./data', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")
eps = float(input("Enter the maximal difference of charges to be identified: "))

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

id_index, field_content_index, w_index, a_index, c_index = -1, -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "id":
        id_index = i
    elif data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "Superpotentials":
        w_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i


class TheoryData:
    def __init__(self, id: int, theory_name: str, w: str, a: float, c: float):
        self.id = id
        self.theory_name = theory_name
        self.w = w
        self.a = a
        self.c = c


theory_set: list[TheoryData] = []

for i in range(1, len(data)):
    id = int(data[i][id_index])
    theory_name = data[i][field_content_index]
    w = data[i][w_index]
    a = float(data[i][a_index])
    c = float(data[i][c_index])

    theory_set.append(TheoryData(id, theory_name, w, a, c))

theory_set.sort(key=lambda theory: theory.a)

current_a = theory_set[0].a
current_a_start = 0
current_a_end = 0
identical_theories = []


def search_identical_charges(theory_set: list[TheoryData], start: int, end: int) -> list[tuple[TheoryData, TheoryData]]:
    print(f'Searching for identical charges..., {start}, {end}')
    subset = theory_set[start:end+1]
    identicals: list[tuple[TheoryData, TheoryData]] = []
    subset.sort(key=lambda theory: theory.c)
    for i in range(len(subset) - 1):
        if abs(subset[i].c - subset[i + 1].c) <= eps:
            identicals.append((subset[i], subset[i + 1]))

    return identicals


for i in range(len(theory_set) - 1):
    if abs(current_a - theory_set[i + 1].a) <= eps:
        current_a_end = i + 1
    else:
        if current_a_end - current_a_start > 0:
            identical_theories += search_identical_charges(theory_set, current_a_start, current_a_end)
        current_a = theory_set[i + 1].a
        current_a_start = i + 1
        current_a_end = i + 1
if current_a_end - current_a_start > 0:
    identical_theories += search_identical_charges(theory_set, current_a_start, current_a_end)

print(f'The number of theory pairs with identical charges: {len(identical_theories)}')

with open(f'./data/identical_charges_eps_{eps}.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id1', 'id2', 'theory1', 'theory2', 'w1', 'w2', 'delta a', 'delta c'])
    for i in range(len(identical_theories)):
        csvwriter.writerow(
            [
                identical_theories[i][0].id, identical_theories[i][1].id,
                identical_theories[i][0].theory_name, identical_theories[i][1].theory_name,
                identical_theories[i][0].w, identical_theories[i][1].w,
                abs(identical_theories[i][0].a - identical_theories[i][1].a),
                abs(identical_theories[i][0].c - identical_theories[i][1].c)
            ]
        )
