import csv
import sys
import os.path
import math
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max

import pdb

from matplotlib import cm


filename = input('Enter file name that contains data of landscape: ')

csv.field_size_limit(np.iinfo(np.int32).max)

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

a_index, c_index = -1, -1
for i in range(len(data[0])):
    if data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i

ac_data = []
min_a = np.finfo(np.float32).max
max_a = np.finfo(np.float32).min
min_c = np.finfo(np.float32).max
max_c = np.finfo(np.float32).min

for i in range(1, len(data)):
    a = float(data[i][a_index])
    c = float(data[i][c_index])

    if a < min_a:
        min_a = a
    if a > max_a:
        max_a = a
    if c < min_c:
        min_c = c
    if c > max_c:
        max_c = c

    ac_data.append([a, c])

print("Minimum a: ", min_a)
print("Maximum a: ", max_a)
print("Minimum c: ", min_c)
print("Maximum c: ", max_c)

ac_data = np.array(ac_data)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

def density_unnormalized():
    grid_size = float(input("Enter grid size: "))

    grid_min_a = float(int(min_a))
    grid_max_a = float(int(max_a) + 1)
    grid_min_c = float(int(min_c))
    grid_max_c = float(int(max_c) + 1)

    na = np.array(np.arange(grid_min_a, grid_max_a, grid_size))
    nc = np.array(np.arange(grid_min_c, grid_max_c, grid_size))

    A, C = np.meshgrid(na, nc)
    Density = np.zeros_like(A)
    max_density = 0.0

    for ac in ac_data:
        a_index, c_index = int((ac[0] - grid_min_a) / grid_size), int((ac[1] - grid_min_c) / grid_size)
        Density[c_index, a_index] += 1.0
        if Density[c_index, a_index] > max_density:
            max_density = Density[c_index, a_index]

    Density /= (grid_size * grid_size)
    max_density /= (grid_size * grid_size)

    print(f'Maximum density: {max_density}')

    min_maxima_distance = int(input("Enter minimum grid distance to identify local maxima of density: "))
    local_maxima_index = peak_local_max(Density, min_distance=min_maxima_distance)

    local_max_a = []
    local_max_c = []
    local_max_density = []
    for i in range(len(local_maxima_index)):
        if local_maxima_index[i, 0] >= A.shape[0] or local_maxima_index[i, 1] >= A.shape[1]:
            continue
        a = A[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        c = C[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        density = Density[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        local_max_a.append(a)
        local_max_c.append(c)
        local_max_density.append(density)

    with open(f'./data/landscape_ac_density_grid_{grid_size}_maxima_distance_{min_maxima_distance}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['a', 'c', 'density'])
        for i in range(len(local_max_a)):
            csvwriter.writerow([local_max_a[i], local_max_c[i], local_max_density[i]])

    log_density = np.log(Density + 1)

    fig = plt.figure()
    fig.suptitle('Density in ac space')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title(f'Grid size: {grid_size}')
    ax.plot_surface(A, C, log_density, cmap=cm.coolwarm)
    ax.contour(A, C, log_density, zdir='z', offset=-10, cmap='coolwarm')
    ax.set_xlabel('a')
    ax.set_ylabel('c')
    ax.set_zlabel('log (Density + 1)')
    ax.set_zlim(-10, np.log(max_density + 1))

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title(f'Minimal peak distance: {min_maxima_distance}')
    pos = ax.imshow(log_density, extent=[na[0], na[-1], nc[0], nc[-1]], cmap=cm.coolwarm, origin='lower')
    fig.colorbar(pos, ax=ax)
    ax.plot(local_max_a, local_max_c, 'gx')
    ax.set_xlabel('a')
    ax.set_ylabel('c')

    plt.savefig(f'./data/landscape_ac_density_grid_{grid_size}_maxima_distance_{min_maxima_distance}.png')

    plt.show()


def density_inverse():
    min_inverse_a = 1.0 / max_a
    max_inverse_a = 1.0 / min_a
    min_inverse_c = 1.0 / max_c
    max_inverse_c = 1.0 / min_c

    grid_size = float(input("Enter grid size: "))

    grid_min_a = float(int(min_inverse_a))
    grid_max_a = float(int(max_inverse_a) + 1)
    grid_min_c = float(int(min_inverse_c))
    grid_max_c = float(int(max_inverse_c) + 1)

    na = np.array(np.arange(grid_min_a, grid_max_a, grid_size))
    nc = np.array(np.arange(grid_min_c, grid_max_c, grid_size))

    A, C = np.meshgrid(na, nc)
    Density = np.zeros_like(A)
    max_density = 0.0

    for ac in ac_data:
        a_index, c_index = int((1.0 / ac[0] - grid_min_a) / grid_size), int((1.0 / ac[1] - grid_min_c) / grid_size)
        Density[c_index, a_index] += 1.0
        if Density[c_index, a_index] > max_density:
            max_density = Density[c_index, a_index]

    Density /= (grid_size * grid_size)
    max_density /= (grid_size * grid_size)

    print(f'Maximum density: {max_density}')

    min_maxima_distance = int(input("Enter minimum grid distance to identify local maxima of density: "))
    local_maxima_index = peak_local_max(Density, min_distance=min_maxima_distance)

    local_max_a = []
    local_max_c = []
    local_max_density = []
    for i in range(len(local_maxima_index)):
        if local_maxima_index[i, 0] >= A.shape[0] or local_maxima_index[i, 1] >= A.shape[1]:
            continue
        a = A[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        c = C[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        density = Density[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        local_max_a.append(a)
        local_max_c.append(c)
        local_max_density.append(density)

    with open(f'./data/landscape_inverse_ac_density_grid_{grid_size}_maxima_distance_{min_maxima_distance}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['1/a', '1/c', 'min a', 'max a', 'min c', 'max c', 'density', 'real density'])
        for i in range(len(local_max_a)):
            real_a_range = (1 / (local_max_a[i] + grid_size), 1 / local_max_a[i])
            real_c_range = (1 / (local_max_c[i] + grid_size), 1 / local_max_c[i])
            real_da = real_a_range[1] - real_a_range[0]
            real_dc = real_c_range[1] - real_c_range[0]
            real_density = local_max_density[i] * (grid_size * grid_size) / (real_da * real_dc)

            csvwriter.writerow([local_max_a[i], local_max_c[i],
                                real_a_range[0], real_a_range[1], real_c_range[0], real_c_range[1],
                                local_max_density[i], real_density])

    log_density = np.log(Density + 1)

    fig = plt.figure()
    fig.suptitle('Density in inverse ac space')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title(f'Grid size: {grid_size}')
    ax.plot_surface(A, C, log_density, cmap=cm.coolwarm)
    ax.contour(A, C, log_density, zdir='z', offset=-10, cmap='coolwarm')
    ax.set_xlabel('1/a')
    ax.set_ylabel('1/c')
    ax.set_zlabel('log (Density + 1)')
    ax.set_zlim(-10, np.log(max_density + 1))

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title(f'Minimal peak distance: {min_maxima_distance}')
    pos = ax.imshow(log_density, extent=[na[0], na[-1], nc[0], nc[-1]], cmap=cm.coolwarm, origin='lower')
    fig.colorbar(pos, ax=ax)
    ax.plot(local_max_a, local_max_c, 'gx')
    ax.set_xlabel('1/a')
    ax.set_ylabel('1/c')

    plt.savefig(f'./data/landscape_inverse_ac_density_grid_{grid_size}_maxima_distance_{min_maxima_distance}.png')

    plt.show()


def density_log():
    min_log_48a = np.log(48 * min_a)
    max_log_48a = np.log(48 * max_a)
    min_log_24c = np.log(24 * min_c)
    max_log_24c = np.log(24 * max_c)

    grid_size = float(input("Enter grid size: "))

    grid_min_a = float(int(min_log_48a))
    grid_max_a = float(int(np.log(48 * (max_a * np.e))))
    grid_min_c = float(int(min_log_24c))
    grid_max_c = float(int(np.log(24 * (max_c * np.e))))

    na = np.array(np.arange(grid_min_a, grid_max_a, grid_size))
    nc = np.array(np.arange(grid_min_c, grid_max_c, grid_size))

    A, C = np.meshgrid(na, nc)
    Density = np.zeros_like(A)
    max_density = 0.0

    for ac in ac_data:
        a_index, c_index = int((np.log(48 * ac[0]) - grid_min_a) / grid_size), int((np.log(24 * ac[1]) - grid_min_c) / grid_size)
        Density[c_index, a_index] += 1.0
        if Density[c_index, a_index] > max_density:
            max_density = Density[c_index, a_index]

    Density /= (grid_size * grid_size)
    max_density /= (grid_size * grid_size)

    print(f'Maximum density: {max_density}')

    min_maxima_distance = int(input("Enter minimum grid distance to identify local maxima of density: "))
    local_maxima_index = peak_local_max(Density, min_distance=min_maxima_distance)

    local_max_a = []
    local_max_c = []
    local_max_density = []

    for i in range(len(local_maxima_index)):
        if local_maxima_index[i, 0] >= A.shape[0] or local_maxima_index[i, 1] >= A.shape[1]:
            continue
        a = A[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        c = C[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        density = Density[local_maxima_index[i, 0], local_maxima_index[i, 1]]
        local_max_a.append(a)
        local_max_c.append(c)
        local_max_density.append(density)

    with open(f'./data/landscape_log_ac_density_grid_{grid_size}_maxima_distance_{min_maxima_distance}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['log 48a', 'log 24c', 'min a', 'max a', 'min c', 'max c', 'density', 'real density'])
        for i in range(len(local_max_a)):
            real_a_range = (np.exp(local_max_a[i]) / 48, np.exp(local_max_a[i] + grid_size) / 48)
            real_c_range = (np.exp(local_max_c[i]) / 24, np.exp(local_max_c[i] + grid_size) / 24)
            real_da = real_a_range[1] - real_a_range[0]
            real_dc = real_c_range[1] - real_c_range[0]
            real_density = local_max_density[i] * (grid_size * grid_size) / (real_da * real_dc)

            csvwriter.writerow([local_max_a[i], local_max_c[i],
                                real_a_range[0], real_a_range[1], real_c_range[0], real_c_range[1],
                                local_max_density[i], real_density])

    log_density = np.log(Density + 1)

    fig = plt.figure()
    fig.suptitle('Density in log ac space')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title(f'Grid size: {grid_size}')
    ax.plot_surface(A, C, log_density, cmap=cm.coolwarm)
    ax.contour(A, C, log_density, zdir='z', offset=-10, cmap='coolwarm')
    ax.set_xlabel('log 48a')
    ax.set_ylabel('log 24c')
    ax.set_zlabel('log (Density + 1)')
    ax.set_zlim(-10, np.log(max_density + 1))

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title(f'Minimal peak distance: {min_maxima_distance}')
    pos = ax.imshow(log_density, extent=[na[0], na[-1], nc[0], nc[-1]], cmap=cm.coolwarm, origin='lower')
    fig.colorbar(pos, ax=ax)
    ax.plot(local_max_a, local_max_c, 'gx')
    ax.set_xlabel('log 48a')
    ax.set_ylabel('log 24c')

    plt.savefig(f'./data/landscape_log_ac_density_grid_{grid_size}_maxima_distance_{min_maxima_distance}.png')

    plt.show()


while True:
    print("Select program to run:")
    print("1. Unnormalized density")
    print("2. Inverse density")
    print("3. Log density")
    program_num = int(input('>>'))

    if program_num == 1:
        density_unnormalized()
    elif program_num == 2:
        density_inverse()
    elif program_num == 3:
        density_log()
    else:
        exit()
