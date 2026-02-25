import csv
import sys
import os.path
import math
import json
import pathlib

from common.superpotential_parser import serialize_w_term_list
from common.superpotential_tree_parser import build_superpotential_tree, SuperpotentialTreeNode

import matplotlib.pyplot as plt
import numpy as np


os.makedirs('./data', exist_ok=True)

filename = input('Enter file name that contains data of landscape: ')

theory_name, superpotential_tree = build_superpotential_tree(filename)


def get_children_stats(tree: SuperpotentialTreeNode, stats: list[int]):
    children_num = len(tree.children)
    stat_index = children_num // 10
    if stat_index >= len(stats):
        for _ in range(stat_index - len(stats) + 1):
            stats.append(0)
    stats[stat_index] += 1

    for child in tree.children:
        get_children_stats(child, stats)


stats = []
get_children_stats(superpotential_tree, stats)

print('Children number statistics')
print('============================')
for i in range(len(stats)):
    print(f'{i * 10} - {i * 10 + 9}: {stats[i]}')

min_children_num = int(input('Enter minimum number of children to fit 3-Delta - delta a plot: '))


def fit_nodes(tree: SuperpotentialTreeNode, theory_data, fit_data, min_children_num):
    # Theory data: [theory id, number of children, a, c]
    # Fit data: [A, n] where delta a = A*(3 - Delta)^n
    if len(tree.children) >= min_children_num:
        dw_dim_data = []
        da_data = []

        r_charges = tree.theory_data.r

        for child in tree.children:
            dw = serialize_w_term_list([child.added_term])[0]
            dw_dim = 0

            additional_a = 0.0
            additional_c = 0.0

            add_data = True
            for op, index, exp in dw:
                if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                    dw_dim += 2.0 / 3.0 * exp
                    additional_a += 1.0 / 48.0
                    additional_c += 1.0 / 24.0
                elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                    dw_dim += 2.0 / 3.0 * exp
                    additional_a += 1.0 / 48.0
                    additional_c += 1.0 / 24.0
                elif op not in r_charges or index - 1 >= len(r_charges[op]):
                    add_data = False
                    break
                else:
                    dw_dim += r_charges[op][index - 1] * exp

            if add_data and dw_dim >= 2:
                add_data = False

            da = child.theory_data.a - tree.theory_data.a - additional_a
            if da > 0:
                print('Positive delta a!')
                print(f'Child a: {child.theory_data.a}')
                print(f'Parent a: {tree.theory_data.a}')
                print(f'Added delta a: {additional_a}')
                print(f'Added term: {child.added_term}')
                add_data = False

            if add_data:
                dw_dim *= 1.5

                dw_dim_data.append(dw_dim)
                da_data.append(da)

        three_minus_dim = np.full_like(len(dw_dim_data), 3.0) - np.array(dw_dim_data)
        da = -np.array(da_data)

        log_dim = np.log(three_minus_dim)
        log_da = np.log(da)

        za = np.polyfit(log_dim, log_da, 1)

        theory_data.append([tree.theory_data.id, len(tree.children), tree.theory_data.a, tree.theory_data.c])
        fit_data.append([float(-np.exp(za[1])), float(za[0])])

    for child in tree.children:
        fit_nodes(child, theory_data, fit_data, min_children_num)


theory_data = []
fit_data = []
fit_nodes(superpotential_tree, theory_data, fit_data, min_children_num)

with open(f'./data/{theory_name}_3-Delta_delta_a_fittings.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Number of children', 'a', 'c', 'A', 'n'])

    for i in range(len(theory_data)):
        writer.writerow(theory_data[i] + fit_data[i])
