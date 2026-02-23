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


def build_dataset(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float], data_color):
    r_charges = tree.theory_data.r
    for child in tree.children:
        dw = serialize_w_term_list([child.added_term])[0]
        dw_dim = 0

        additional_a = 0.0
        additional_c = 0.0
        color = (1.0, 0.3, 0.3)

        add_data = True
        for op, index, exp in dw:
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = (0.3, 0.3, 1.0)
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = (0.3, 0.3, 1.0)
            elif op not in r_charges or index - 1 >= len(r_charges[op]):
                add_data = False
                break
            else:
                dw_dim += r_charges[op][index - 1] * exp

        if add_data and dw_dim >= 2:
            add_data = False

        da = child.theory_data.a - tree.theory_data.a - additional_a
        dc = child.theory_data.c - tree.theory_data.c - additional_c

        if add_data and da >= 0:
            print('Positive delta a!')
            print(f'Child ID: {child.theory_data.id}')
            print(f'Parent ID: {tree.theory_data.id}')
            print(f'Child a: {child.theory_data.a}')
            print(f'Parent a: {tree.theory_data.a}')
            print(f'Added delta a: {additional_a}')
            print(f'Added term: {child.added_term}')
            add_data = False

        if add_data:
            dw_dim *= 1.5

            dw_dim_data.append(dw_dim)
            da_data.append(da)
            dc_data.append(dc)
            data_color.append(color)

        build_dataset(child, dw_dim_data, da_data, dc_data, data_color)


def get_most_children_node(tree: SuperpotentialTreeNode) -> SuperpotentialTreeNode:
    most_children_node = None
    most_children_num = -1

    def most_children_internal(parent: SuperpotentialTreeNode):
        nonlocal most_children_node
        nonlocal most_children_num

        children_num = len(parent.children)
        if children_num > most_children_num:
            most_children_num = children_num
            most_children_node = parent
        for child in parent.children:
            most_children_internal(child)

    most_children_internal(tree)

    return most_children_node


def build_dataset_most_children(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float], data_color):
    most_children_node = get_most_children_node(tree)

    print('Most children node:')
    print(f'ID: {most_children_node.theory_data.id}')
    print(f"W = {most_children_node.theory_data.w}")

    r_charges = most_children_node.theory_data.r

    for child in most_children_node.children:
        dw = serialize_w_term_list([child.added_term])[0]
        dw_dim = 0

        additional_a = 0.0
        additional_c = 0.0
        color = (1.0, 0.3, 0.3)

        add_data = True
        for op, index, exp in dw:
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = (0.3, 0.3, 1.0)
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = (0.3, 0.3, 1.0)
            elif op not in r_charges or index - 1 >= len(r_charges[op]):
                add_data = False
                break
            else:
                dw_dim += r_charges[op][index - 1] * exp

        if add_data and dw_dim >= 2:
            add_data = False

        da = child.theory_data.a - most_children_node.theory_data.a - additional_a
        dc = child.theory_data.c - most_children_node.theory_data.c - additional_c
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
            dc_data.append(dc)
            data_color.append(color)

    return most_children_node


dw_dim_data = []
da_data = []
dc_data = []
data_color = []

print('Choose program...')
print('1. All data')
print('2. Theory with most children')
program_num = int(input('>>'))

most_children_node = None
if program_num == 1:
    build_dataset(superpotential_tree, dw_dim_data, da_data, dc_data, data_color)
elif program_num == 2:
    most_children_node = build_dataset_most_children(superpotential_tree, dw_dim_data, da_data, dc_data, data_color)
else:
    exit()

print(f"Data size: {len(dw_dim_data)}")

three_minus_dim = np.full_like(len(dw_dim_data), 3.0) - np.array(dw_dim_data)
da = -np.array(da_data)
dc = np.abs(-np.array(dc_data))

log_dim = np.log(three_minus_dim)
log_da = np.log(da)
log_dc = np.log(dc)

za = np.polyfit(log_dim, log_da, 1)
zc = np.polyfit(log_dim, log_dc, 1)
pa = np.poly1d(za)
pc = np.poly1d(zc)

min_dim = np.min(three_minus_dim)
max_dim = np.max(three_minus_dim)
dim_plot = np.linspace(min_dim, max_dim, 100)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(1, 2, sharey=True, squeeze=True)
fig.suptitle(f'Plot of 3-Delta and delta a and delta c of {theory_name} theory'
             + f'{f'\nfrom parent theory ID {most_children_node.theory_data.id}' if program_num == 2 else ''}'
             )

ax[0].scatter(three_minus_dim, da_data, s=4, c=data_color)
ax[0].plot(dim_plot, -np.exp(pa(np.log(dim_plot))), 'g--')
ax[0].text(0.01, 0, f'delta a = {-np.exp(za[1]):.3f}*epsilon^{za[0]:.3f}')
ax[1].scatter(three_minus_dim, dc_data, s=4, c=data_color)
ax[1].plot(dim_plot, -np.exp(pc(np.log(dim_plot))), 'g--')
ax[1].text(0.01, 0, f'delta a = {-np.exp(zc[1]):.3f}*epsilon^{zc[0]:.3f}')
ax[0].set_xlabel('3-Delta')
ax[0].set_ylabel('delta a')
ax[1].set_xlabel('3-Delta')
ax[1].set_ylabel('delta c')

plt.savefig(f'./data/{theory_name}_3-Delta_delta_ac_plot{'_most_children' if program_num == 2 else ''}.png')

plt.show()
