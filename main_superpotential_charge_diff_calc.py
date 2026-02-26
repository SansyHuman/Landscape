import csv
import sys
import os.path
import math
import json
import pathlib
import pdb

from common.superpotential_parser import serialize_w_term_list
from common.superpotential_tree_parser import build_superpotential_tree, SuperpotentialTreeNode

import matplotlib.pyplot as plt
import numpy as np


os.makedirs('./data', exist_ok=True)

filename = input('Enter file name that contains data of landscape: ')

theory_name, superpotential_tree = build_superpotential_tree(filename)

normal_color = (1.0, 0.3, 0.3)
flip_color = (0.3, 0.3, 1.0)

def build_dataset(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float], data_color):
    r_charges = tree.theory_data.r
    for child in tree.children:
        dw = serialize_w_term_list([child.added_term])[0]
        dw_dim = 0

        additional_a = 0.0
        additional_c = 0.0
        color = normal_color

        add_data = True
        for op, index, exp in dw:
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
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


def build_dataset_normalized(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float], data_color):
    r_charges = tree.theory_data.r
    for child in tree.children:
        dw = serialize_w_term_list([child.added_term])[0]
        dw_dim = 0

        additional_a = 0.0
        additional_c = 0.0
        color = normal_color

        add_data = True
        for op, index, exp in dw:
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
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
            da_data.append(da / (tree.theory_data.a + additional_a))
            dc_data.append(dc / (tree.theory_data.c + additional_c))
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
        color = normal_color

        add_data = True
        for op, index, exp in dw:
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
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


def build_dataset_most_children_normalized(tree: SuperpotentialTreeNode, dw_dim_data: list[float], da_data: list[float], dc_data: list[float], data_color):
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
        color = normal_color

        add_data = True
        for op, index, exp in dw:
            if op == 'M' and ('M' not in r_charges or index - 1 >= len(r_charges['M'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
            elif op == 'X' and ('X' not in r_charges or index - 1 >= len(r_charges['X'])):
                dw_dim += 2.0 / 3.0 * exp
                additional_a += 1.0 / 48.0
                additional_c += 1.0 / 24.0
                color = flip_color
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
            da_data.append(da / (most_children_node.theory_data.a + additional_a))
            dc_data.append(dc / (most_children_node.theory_data.c + additional_c))
            data_color.append(color)

    return most_children_node

while True:
    dw_dim_data = []
    da_data = []
    dc_data = []
    data_color = []

    print('Choose program...')
    print('1. All data')
    print('2. All data with da/a and dc/c')
    print('3. Theory with most children')
    print('4. Theory with most children with da/a and dc/c')
    program_num = int(input('>>'))

    fig_name = ''
    save_file_name = ''
    a_axis_name = ''
    c_axis_name = ''
    most_children_node = None

    if program_num == 1:
        build_dataset(superpotential_tree, dw_dim_data, da_data, dc_data, data_color)
        fig_name = f'Plot of 3-Delta and delta a and delta c of {theory_name} theory'
        save_file_name = f'./data/{theory_name}_3-Delta_delta_ac_plot.png'
        a_axis_name = 'delta a'
        c_axis_name = 'delta c'
    elif program_num == 2:
        build_dataset_normalized(superpotential_tree, dw_dim_data, da_data, dc_data, data_color)
        fig_name = f'Plot of 3-Delta and delta a / a and delta c / c of {theory_name} theory'
        save_file_name = f'./data/{theory_name}_3-Delta_delta_ac_plot_normalized.png'
        a_axis_name = 'delta a / a'
        c_axis_name = 'delta c / c'
    elif program_num == 3:
        most_children_node = build_dataset_most_children(superpotential_tree, dw_dim_data, da_data, dc_data, data_color)
        fig_name = f'Plot of 3-Delta and delta a and delta c of {theory_name} theory\nfrom parent theory ID {most_children_node.theory_data.id}'
        save_file_name = f'./data/{theory_name}_3-Delta_delta_ac_plot_most_children.png'
        a_axis_name = 'delta a'
        c_axis_name = 'delta c'
    elif program_num == 4:
        most_children_node = build_dataset_most_children_normalized(superpotential_tree, dw_dim_data, da_data, dc_data, data_color)
        fig_name = f'Plot of 3-Delta and delta a / a and delta c / c of {theory_name} theory\nfrom parent theory ID {most_children_node.theory_data.id}'
        save_file_name = f'./data/{theory_name}_3-Delta_delta_ac_plot_most_children_normalized.png'
        a_axis_name = 'delta a / a'
        c_axis_name = 'delta c / c'
    else:
        exit()

    print(f"Data size: {len(dw_dim_data)}")

    dw_dim_data = np.array(dw_dim_data)
    da_data = np.array(da_data)
    dc_data = np.array(dc_data)

    three_minus_dim = np.full_like(len(dw_dim_data), 3.0) - dw_dim_data
    da = -da_data
    dc = np.abs(-dc_data)

    log_dim = np.log(three_minus_dim)
    log_da = np.log(da)
    log_dc = np.log(dc)

    normal_index = []
    flip_index = []
    for i in range(len(data_color)):
        if data_color[i] == normal_color:
            normal_index.append(i)
        else:
            flip_index.append(i)

    log_dim_normal = log_dim[normal_index]
    log_dim_flip = log_dim[flip_index]
    log_da_normal = log_da[normal_index]
    log_da_flip = log_da[flip_index]
    log_dc_normal = log_dc[normal_index]
    log_dc_flip = log_dc[flip_index]

    za_normal = np.polyfit(log_dim_normal, log_da_normal, 1) if len(log_dim_normal) > 0 else None
    za_flip = np.polyfit(log_dim_flip, log_da_flip, 1) if len(log_dim_flip) > 0 else None
    zc_normal = np.polyfit(log_dim_normal, log_dc_normal, 1) if len(log_dim_normal) > 0 else None
    zc_flip = np.polyfit(log_dim_flip, log_dc_flip, 1) if len(log_dim_flip) > 0 else None


    def rsquared(x, y, fit):
        yavg = np.average(y)
        f = -np.exp(fit(np.log(x)))
        ss_res = np.sum((y - f)**2)
        ss_tot = np.sum((y - yavg)**2)

        return 1 - ss_res / ss_tot


    pa_normal = np.poly1d(za_normal) if za_normal is not None else None
    pa_flip = np.poly1d(za_flip) if za_flip is not None else None
    pc_normal = np.poly1d(zc_normal) if zc_normal is not None else None
    pc_flip = np.poly1d(zc_flip) if zc_flip is not None else None

    dim_normal_data = three_minus_dim[normal_index]
    dim_flip_data = three_minus_dim[flip_index]
    da_normal_data = da_data[normal_index]
    da_flip_data = da_data[flip_index]
    dc_normal_data = dc_data[normal_index]
    dc_flip_data = dc_data[flip_index]

    a_normal_r2 = rsquared(dim_normal_data, da_normal_data, pa_normal) if pa_normal is not None else 0
    a_flip_r2 = rsquared(dim_flip_data, da_flip_data, pa_flip) if pa_flip is not None else 0
    c_normal_r2 = rsquared(dim_normal_data, dc_normal_data, pc_normal) if pc_normal is not None else 0
    c_flip_r2 = rsquared(dim_flip_data, dc_flip_data, pc_flip) if pc_flip is not None else 0

    min_dim = np.min(three_minus_dim)
    max_dim = np.max(three_minus_dim)
    dim_plot = np.linspace(min_dim, max_dim, 100)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(1, 2, sharey=True, squeeze=True)
    fig.suptitle(fig_name + f'\nnormal: {len(log_dim_normal)}, flip: {len(log_dim_flip)}')

    ax[0].scatter(three_minus_dim, da_data, s=4, c=data_color)
    if pa_normal is not None:
        ax[0].plot(dim_plot, -np.exp(pa_normal(np.log(dim_plot))), 'r--')
        ax[0].text(min_dim, -np.exp(pa_normal(np.log(dim_plot[-10]))), f'{a_axis_name}_normal = {-np.exp(za_normal[1]):.3f}*epsilon^{za_normal[0]:.3f}\nR2 = {a_normal_r2:.3f}')
    if pa_flip is not None:
        ax[0].plot(dim_plot, -np.exp(pa_flip(np.log(dim_plot))), 'b--')
        ax[0].text(min_dim, 0, f'{a_axis_name}_flip = {-np.exp(za_flip[1]):.3f}*epsilon^{za_flip[0]:.3f}\nR2 = {a_flip_r2:.3f}')
    ax[1].scatter(three_minus_dim, dc_data, s=4, c=data_color)
    if pc_normal is not None:
        ax[1].plot(dim_plot, -np.exp(pc_normal(np.log(dim_plot))), 'r--')
        ax[1].text(min_dim, -np.exp(pc_normal(np.log(dim_plot[-10]))), f'{c_axis_name}_normal = {-np.exp(zc_normal[1]):.3f}*epsilon^{zc_normal[0]:.3f}\nR2 = {c_normal_r2:.3f}')
    if pc_flip is not None:
        ax[1].plot(dim_plot, -np.exp(pc_flip(np.log(dim_plot))), 'b--')
        ax[1].text(min_dim, 0, f'{c_axis_name}_flip = {-np.exp(zc_flip[1]):.3f}*epsilon^{zc_flip[0]:.3f}\nR2 = {c_flip_r2:.3f}')
    ax[0].set_xlabel('3-Delta')
    ax[0].set_ylabel(a_axis_name)
    ax[1].set_xlabel('3-Delta')
    ax[1].set_ylabel(c_axis_name)

    plt.savefig(save_file_name)

    plt.show()
