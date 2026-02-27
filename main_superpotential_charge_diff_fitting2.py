import csv
import pdb
import sys
import os.path
import math
import json
import pathlib

from common.superpotential_parser import serialize_w_term_list
from common.superpotential_tree_parser import build_superpotential_tree, SuperpotentialTreeNode
from sklearn.metrics import r2_score

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
min_points_num = int(input('Enter the minimum number of normal or flip fixed points to plot: '))


def fit_nodes(tree: SuperpotentialTreeNode, theory_data, fit_data_normal, fit_data_flip, min_children_num, min_points_num):
    # Theory data: [theory id, number of normal children, number of flip children, a, c]
    # Fit data: [A, n] where delta a = A*(3 - Delta)^n
    if len(tree.children) >= min_children_num:
        dw_dim_normal = []
        dw_dim_flip = []
        da_normal = []
        da_flip = []

        r_charges = tree.theory_data.r

        for child in tree.children:
            dw = serialize_w_term_list([child.added_term])[0]
            dw_dim = 0

            additional_a = 0.0
            additional_c = 0.0

            add_data = True
            flip = False
            for op, index, exp in dw:
                if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                    dw_dim += 2.0 / 3.0 * exp
                    additional_a += 1.0 / 48.0
                    additional_c += 1.0 / 24.0
                    flip = True
                elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                    dw_dim += 2.0 / 3.0 * exp
                    additional_a += 1.0 / 48.0
                    additional_c += 1.0 / 24.0
                    flip = True
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

                if flip:
                    dw_dim_flip.append(dw_dim)
                    da_flip.append(da)
                else:
                    dw_dim_normal.append(dw_dim)
                    da_normal.append(da)

        theory_data.append([tree.theory_data.id, len(dw_dim_normal), len(dw_dim_flip), tree.theory_data.a, tree.theory_data.c])

        if len(dw_dim_normal) >= min_points_num:
            three_minus_dim = np.full_like(len(dw_dim_normal), 3.0) - np.array(dw_dim_normal)
            da = -np.array(da_normal)

            log_dim = np.log(three_minus_dim)
            log_da = np.log(da)

            za = np.polyfit(log_dim, log_da, 1)

            fit_data_normal.append([float(-np.exp(za[1])), float(za[0])])
        else:
            fit_data_normal.append([None, None])

        if len(dw_dim_flip) >= min_points_num:
            three_minus_dim = np.full_like(len(dw_dim_flip), 3.0) - np.array(dw_dim_flip)
            da = -np.array(da_flip)

            log_dim = np.log(three_minus_dim)
            log_da = np.log(da)

            za = np.polyfit(log_dim, log_da, 1)

            fit_data_flip.append([float(-np.exp(za[1])), float(za[0])])
        else:
            fit_data_flip.append([None, None])

    for child in tree.children:
        fit_nodes(child, theory_data, fit_data_normal, fit_data_flip, min_children_num, min_points_num)


theory_data = []
fit_data_normal = []
fit_data_flip = []
fit_nodes(superpotential_tree, theory_data, fit_data_normal, fit_data_flip, min_children_num, min_points_num)

with open(f'./data/{theory_name}_3-Delta_delta_a_fittings.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Number of normal children', 'Number of flip children', 'a', 'c', 'A_nomral', 'n_normal', 'A_flip', 'n_flip'])

    for i in range(len(theory_data)):
        writer.writerow(theory_data[i] + fit_data_normal[i] + fit_data_flip[i])

normal_a = []
normal_A = []
normal_n = []
flip_a = []
flip_A = []
flip_n = []

for i in range(len(theory_data)):
    if fit_data_normal[i][0] is not None:
        normal_a.append(theory_data[i][3])
        normal_A.append(fit_data_normal[i][0])
        normal_n.append(fit_data_normal[i][1])
    if fit_data_flip[i][0] is not None:
        flip_a.append(theory_data[i][3])
        flip_A.append(fit_data_flip[i][0])
        flip_n.append(fit_data_flip[i][1])

zA_normal = np.polyfit(normal_a, normal_A, 1) if len(normal_a) > 0 else None
zn_normal = np.polyfit(normal_a, normal_n, 1) if len(normal_a) > 0 else None
zA_flip = np.polyfit(flip_a, flip_A, 1) if len(flip_a) > 0 else None
zn_flip = np.polyfit(flip_a, flip_n, 1) if len(flip_a) > 0 else None

pA_normal = np.poly1d(zA_normal) if zA_normal is not None else None
pn_normal = np.poly1d(zn_normal) if zn_normal is not None else None
pA_flip = np.poly1d(zA_flip) if zA_flip is not None else None
pn_flip = np.poly1d(zn_flip) if zn_flip is not None else None

A_normal_r2 = r2_score(normal_A, pA_normal(normal_a)) if pA_normal is not None else 0
n_normal_r2 = r2_score(normal_n, pn_normal(normal_a)) if pn_normal is not None else 0
A_flip_r2 = r2_score(flip_A, pA_flip(flip_a)) if pA_flip is not None else 0
n_flip_r2 = r2_score(flip_n, pn_flip(flip_a)) if pn_flip is not None else 0

min_a = min(min(normal_a), min(flip_a))
max_a = max(max(normal_a), max(flip_a))
a_fitting = [min_a, max_a]

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(1, 2, squeeze=True)
fig.suptitle('Plots of coefficients of delta a = A(3-Delta)^n and a')

if pA_normal is not None:
    ax[0].scatter(normal_a, normal_A, c='#FF3333', label='Normal')
    ax[0].plot(a_fitting, pA_normal(a_fitting), 'r--')
    ax[0].text(min_a * 0.75 + max_a * 0.25, pA_normal(min_a * 0.75 + max_a * 0.25), f'A_normal = {zA_normal[0]:.3f}a + {zA_normal[1]:.3f}\nR2 = {A_normal_r2:.3f}')
if pA_flip is not None:
    ax[0].scatter(flip_a, flip_A, c='#3333FF', label='Flip')
    ax[0].plot(a_fitting, pA_flip(a_fitting), 'b--')
    ax[0].text((min_a + max_a) / 2, pA_flip((min_a + max_a) / 2), f'A_flip = {zA_flip[0]:.3f}a + {zA_flip[1]:.3f}\nR2 = {A_flip_r2:.3f}')
ax[0].set_title('a - A')
ax[0].set_xlabel('a')
ax[0].set_ylabel('A')
ax[0].legend()

if pn_normal is not None:
    ax[1].scatter(normal_a, normal_n, c='#FF3333', label='Normal')
    ax[1].plot(a_fitting, pn_normal(a_fitting), 'r--')
    ax[1].text(min_a * 0.75 + max_a * 0.25, pn_normal(min_a * 0.75 + max_a * 0.25), f'n_normal = {zn_normal[0]:.3f}a + {zn_normal[1]:.3f}\nR2 = {n_normal_r2:.3f}')
if pn_flip is not None:
    ax[1].scatter(flip_a, flip_n, c='#3333FF', label='Flip')
    ax[1].plot(a_fitting, pn_flip(a_fitting), 'b--')
    ax[1].text((min_a + max_a) / 2, pn_flip((min_a + max_a) / 2), f'n_flip = {zn_flip[0]:.3f}a + {zn_flip[1]:.3f}\nR2 = {n_flip_r2:.3f}')
ax[1].set_title('a - n')
ax[1].set_xlabel('a')
ax[1].set_ylabel('n')
ax[1].legend()

plt.savefig(f'./data/{theory_name}_3-Delta_delta_a_fittings.png')

plt.show()
