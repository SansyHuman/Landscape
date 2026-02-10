import csv
import json
import os.path
import random
import math

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, expand

from common.superpotential_parser import is_same_gauge_group, is_equivalent_matter_contents

os.makedirs('./data', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")
eps = float(input("Enter the maximal difference of charges to be identified: "))
different_theory_only = input('Only find for different theories? y/n: ').lower()
different_theory_only = True if different_theory_only == 'y' else False
different_gauge_only = False
if different_theory_only:
    tmp = input('Only find for different gauge groups? y/n: ').lower()
    different_gauge_only = True if tmp == 'y' else False

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

id_index, field_content_index, w_index, a_index, c_index, sci_index = -1, -1, -1, -1, -1, -1
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
    elif data[0][i] == "SCI":
        sci_index = i


class TheoryData:
    def __init__(self, id: int, theory_name: str, w: str, a: float, c: float, sci: str):
        self.id = id
        self.theory_name = theory_name
        self.w = w
        self.a = a
        self.c = c
        self.sci = sci


theory_set: list[TheoryData] = []

for i in range(1, len(data)):
    id = int(data[i][id_index])
    theory_name = data[i][field_content_index]
    w = data[i][w_index]
    a = float(data[i][a_index])
    c = float(data[i][c_index])
    sci = data[i][sci_index].replace('^', '**')

    theory_set.append(TheoryData(id, theory_name, w, a, c, sci))

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

    current_c = subset[0].c
    current_c_start = 0
    current_c_end = 0

    def add_identical_theories():
        for j in range(current_c_start, current_c_end):
            for k in range(j + 1, current_c_end + 1):
                add_theory = False
                if different_theory_only:
                    if subset[j].theory_name != subset[k].theory_name:
                        add_theory = True
                        if different_gauge_only:
                            add_theory = not is_same_gauge_group(subset[j].theory_name, subset[k].theory_name)
                else:
                    add_theory = True

                if add_theory:
                    identicals.append((subset[j], subset[k]))

    for i in range(len(subset) - 1):
        if abs(current_c - subset[i + 1].c) <= eps:
            current_c_end = i + 1
        else:
            add_identical_theories()

            current_c = subset[i + 1].c
            current_c_start = i + 1
            current_c_end = i + 1

    add_identical_theories()

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

with open(f'./data/identical_charges_eps_{eps}_{'different_gauges' if different_gauge_only else 'different_theories' if different_theory_only else 'all'}.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id1', 'id2', 'theory1', 'theory2', 'equivalent matter contents', 'w1', 'w2', 'delta a', 'delta c', 'sci1', 'sci2', 'delta sci'])
    for i in range(len(identical_theories)):
        print(i)
        sci1 = parse_expr(identical_theories[i][0].sci)
        sci2 = parse_expr(identical_theories[i][1].sci)
        csvwriter.writerow(
            [
                identical_theories[i][0].id, identical_theories[i][1].id,
                identical_theories[i][0].theory_name, identical_theories[i][1].theory_name,
                is_equivalent_matter_contents(identical_theories[i][0].theory_name, identical_theories[i][1].theory_name),
                identical_theories[i][0].w, identical_theories[i][1].w,
                abs(identical_theories[i][0].a - identical_theories[i][1].a),
                abs(identical_theories[i][0].c - identical_theories[i][1].c),
                str(sci1), str(sci2),
                str(sci1 - sci2)
            ]
        )
