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


filename = input('Enter file name that contains data of landscape from one gauge theory: ')
theory_name, superpotential_tree = build_superpotential_tree(filename)

def build_dataset(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float]):
    r_charges = tree.theory_data.r
    for child in tree.children:
        dw = serialize_w_term_list([child.added_term])[0]
        dw_dim = 0

        add_data = True
        for op, index, exp in dw:
            '''
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
            el'''
            # TODO: add contribution by free fields M and X to central charges

            if op not in r_charges or index - 1 >= len(r_charges[op]):
                add_data = False
                break
            else:
                dw_dim += r_charges[op][index - 1] * exp

        if add_data and dw_dim >= 2:
            add_data = False

        da = child.theory_data.a - tree.theory_data.a
        dc = child.theory_data.c - tree.theory_data.c

        if add_data and da >= 0 or dc >= 0:
            add_data = False

        if add_data:
            dw_dim *= 1.5

            dw_dim_data.append(dw_dim)
            da_data.append(da)
            dc_data.append(dc)

        build_dataset(child, dw_dim_data, da_data, dc_data)


def get_most_children_node(tree: SuperpotentialTreeNode) -> SuperpotentialTreeNode:
    most_children_node = None
    most_children_num = -1

    def most_children_internal(parent: SuperpotentialTreeNode):
        nonlocal most_children_node
        nonlocal most_children_num

        children_num = len(parent.children)
        if children_num > most_children_num:
            print(children_num)
            most_children_num = children_num
            most_children_node = parent
        for child in parent.children:
            most_children_internal(child)

    most_children_internal(tree)

    print(most_children_num)
    return most_children_node


def build_dataset_most_children(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float]):
    most_children_node = get_most_children_node(tree)
    r_charges = most_children_node.theory_data.r

    for child in most_children_node.children:
        dw = serialize_w_term_list([child.added_term])[0]
        dw_dim = 0

        add_data = True
        for op, index, exp in dw:
            '''
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
            el'''
            # TODO: add contribution by free fields M and X to central charges

            if op not in r_charges or index - 1 >= len(r_charges[op]):
                add_data = False
                break
            else:
                dw_dim += r_charges[op][index - 1] * exp

        if add_data and dw_dim >= 2:
            add_data = False

        da = child.theory_data.a - tree.theory_data.a
        dc = child.theory_data.c - tree.theory_data.c

        '''
        if add_data and da >= 0 or dc >= 0:
            add_data = False
        '''

        if add_data:
            dw_dim *= 1.5

            dw_dim_data.append(dw_dim)
            da_data.append(da)
            dc_data.append(dc)


dw_dim_data = []
da_data = []
dc_data = []

print('Choose program...')
print('1. All data')
print('2. Theory with most children')
program_num = int(input('>>'))

if program_num == 1:
    build_dataset(superpotential_tree, dw_dim_data, da_data, dc_data)
elif program_num == 2:
    build_dataset_most_children(superpotential_tree, dw_dim_data, da_data, dc_data)
else:
    exit()

print(f"Data size: {len(dw_dim_data)}")

three_minus_dim = np.full_like(len(dw_dim_data), 3.0) - np.array(dw_dim_data)
da = -np.array(da_data)
dc = -np.array(dc_data)

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
fig.suptitle(f'Plot of 3-Delta and delta a and delta c of {theory_name} theory')

ax[0].scatter(three_minus_dim, da_data, s=1)
ax[0].plot(dim_plot, -np.exp(pa(np.log(dim_plot))), 'r--')
ax[0].text(0.1, 0, f'delta a = {-np.exp(za[1]):.3f}*epsilon^{za[0]:.3f}')
ax[1].scatter(three_minus_dim, dc_data, s=1)
ax[1].plot(dim_plot, -np.exp(pc(np.log(dim_plot))), 'r--')
ax[1].text(0.1, 0, f'delta a = {-np.exp(zc[1]):.3f}*epsilon^{zc[0]:.3f}')
ax[0].set_xlabel('3-Delta')
ax[0].set_ylabel('delta a')
ax[1].set_xlabel('3-Delta')
ax[1].set_ylabel('delta c')

plt.savefig(f'./data/{theory_name}_3-Delta_delta_ac_plot{'_most_children' if program_num == 2 else ''}.png')

plt.show()
