import polars as pl
import datetime
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from common.balanced_sample_tool import TheorySampler
import csv
import sys
import os.path
import math
import json
import pathlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from common.sci_parser import SuperConformalIndex


os.makedirs('../data/regression', exist_ok=True)
csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")

theory_sampler = TheorySampler(filename)
for row in theory_sampler.get_theory_stats().iter_rows():
    print(row)

GRID_LO = float(input("Enter lower bound of feature grid: "))
GRID_HI = float(input("Enter upper bound of feature grid: "))
GRID_STEP = float(input("Enter step size of feature grid: "))

GRID = np.arange(GRID_LO, GRID_HI + GRID_STEP, GRID_STEP)
KDE_BANDWIDTH = float(input("Enter bandwidth of feature grid: "))


def fit_data(sampled: TheorySampler, savefile_suffix: str, show_graph: bool=False, save_dir=None):
    if save_dir is None:
        save_dir = "../data/regression"
    os.makedirs(save_dir, exist_ok=True)

    sample_stat = sampled.get_theory_stats()

    n_theory = sampled.get_theory_num()
    theories = sample_stat["Name"].to_list()
    print("The number of theories in the sample: ", n_theory)
    print("Theories in the sample: ", theories)

    theories_dict = dict()
    for i in range(len(theories)):
        theories_dict[theories[i]] = i

    data_num = sampled.df.height
    X = []
    a = []
    c = []
    theory_data = []

    for i in range(data_num):
        theory_data.append(theories_dict[sampled.df["Name"][i]])
        a.append(float(sampled.df["CentralChargeA"][i]))
        c.append(float(sampled.df["CentralChargeC"][i]))
        sci = SuperConformalIndex(sampled.df["SCI"][i])
        X.append(sci.featurize_dimensions(GRID, KDE_BANDWIDTH))
    X = np.asarray(X)
    a = np.asarray(a)
    c = np.asarray(c)
    theory_data = np.asarray(theory_data)

    krr_a = KernelRidge(kernel='rbf')
    krr_c = KernelRidge(kernel='rbf')
    krr_grid = {
        "alpha": np.logspace(-4, 0, 9),
        "gamma": np.logspace(-5, -1, 9),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    krr_search_a = GridSearchCV(
        krr_a,
        krr_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    krr_search_c = GridSearchCV(
        krr_c,
        krr_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )

    krr_search_a.fit(X, a)
    krr_search_c.fit(X, c)

    print('Regression of central charge a')
    print('=================================')
    print(f"  best params : {krr_search_a.best_params_}")
    print(f"  best CV R²  : {krr_search_a.best_score_:.4f}")
    print()

    print('Regression of central charge c')
    print('=================================')
    print(f"  best params : {krr_search_c.best_params_}")
    print(f"  best CV R²  : {krr_search_c.best_score_:.4f}")
    print()

    a_pred = cross_val_predict(
        krr_search_a.best_estimator_,
        X,
        a,
        cv=cv,
        n_jobs=-1
    )
    c_pred = cross_val_predict(
        krr_search_c.best_estimator_,
        X,
        c,
        cv=cv,
        n_jobs=-1
    )

    a_per_theory = [[] for _ in range(n_theory)]
    a_pred_per_theory = [[] for _ in range(n_theory)]

    c_per_theory = [[] for _ in range(n_theory)]
    c_pred_per_theory = [[] for _ in range(n_theory)]

    for i in range(data_num):
        theory_index = theory_data[i]
        a_per_theory[theory_index].append(a[i])
        c_per_theory[theory_index].append(c[i])
        a_pred_per_theory[theory_index].append(a_pred[i])
        c_pred_per_theory[theory_index].append(c_pred[i])

    total_metrics = [] # r2_a, mae_a, rmse_a, r2_c, mae_c, rmse_c
    metrics_per_theory = [[] for _ in range(n_theory)]

    def _calculate_metric(y_true, y_pred):
        return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), root_mean_squared_error(y_true, y_pred)

    total_metrics.extend(_calculate_metric(a, a_pred))
    total_metrics.extend(_calculate_metric(c, c_pred))
    for i in range(n_theory):
        metrics_per_theory[i].extend(_calculate_metric(a_per_theory[i], a_pred_per_theory[i]))
        metrics_per_theory[i].extend(_calculate_metric(c_per_theory[i], c_pred_per_theory[i]))

    with open(f'{save_dir}/sci_exp_regression_{savefile_suffix}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Theory', "R2 of a", "MAE of a", "RMSE of a", "R2 of c", "MAE of c", "RMSE of c"])
        writer.writerow(['Total'] + total_metrics)
        for i in range(n_theory):
            writer.writerow([theories[i]] + metrics_per_theory[i])

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    plt.close('all')

    cmap = plt.cm.get_cmap('jet', n_theory)

    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)

    fig.suptitle(f'Regression of a/c central charge with SCI exponents')

    ax[0].set_title(f'a regression R2={total_metrics[0]:.3f}')
    for i in range(n_theory):
        ax[0].scatter(a_per_theory[i], a_pred_per_theory[i], color=cmap(i), label=theories[i])
    a_range = [np.min(a), np.max(a)]
    ax[0].plot(a_range, a_range, linestyle='--', color='red', label='Exact')
    ax[0].set_xlabel('Real a')
    ax[0].set_ylabel('Predicted a')
    ax[0].legend()

    ax[1].set_title(f'c regression R2={total_metrics[3]:.3f}')
    for i in range(n_theory):
        ax[1].scatter(c_per_theory[i], c_pred_per_theory[i], color=cmap(i), label=theories[i])
    c_range = [np.min(c), np.max(c)]
    ax[1].plot(c_range, c_range, linestyle='--', color='red', label='Exact')
    ax[1].set_xlabel('Real c')
    ax[1].set_ylabel('Predicted c')
    ax[1].legend()

    plt.savefig(f'{save_dir}/sci_exp_regression_{savefile_suffix}.png')

    if show_graph:
        plt.show()


def regression_charge_range():
    min_a = float(input("Enter minimal value of a central charge: "))
    max_a = float(input("Enter maximal value of a central charge: "))
    min_c = float(input("Enter minimal value of c central charge: "))
    max_c = float(input("Enter maximal value of c central charge: "))
    n_samples = int(input("Enter number of samples per theory: "))
    n_iter = int(input("Enter number of iterations: "))

    save_dir = f"../data/regression/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    save_suffix = f'({min_a}_{max_a})({min_c}_{max_c})_{n_samples}'

    for i in range(n_iter):
        sampled = theory_sampler.get_balanced_sample((min_a, max_a), (min_c, max_c), n_samples)
        fit_data(sampled, save_suffix + f'_{i + 1}', save_dir=save_dir, show_graph=i == n_iter - 1)


def regression_manual_selection():
    selected = []
    print('Write all theories you want to choose separated with comma.')
    theories = input('>>>').split(',')
    for theory in theories:
        selected.append(theory.strip())
    selected = sorted(selected)

    n_samples = int(input("Enter number of samples per theory: "))
    n_iter = int(input("Enter number of iterations: "))

    save_dir = f"../data/regression/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"
    save_suffix = f'{selected}_{n_samples}'

    for i in range(n_iter):
        sampled = theory_sampler.get_manual_sample(selected, n_samples)
        fit_data(sampled, save_suffix + f'_{i + 1}', save_dir=save_dir, show_graph=i == n_iter - 1)


while True:
    print('Select program...')
    print('1. Select theories within the central charge range')
    print('2. Select theories manually')
    print('-1. Exit')
    program = int(input('>>>'))

    if program == 1:
        regression_charge_range()
    elif program == 2:
        regression_manual_selection()
    else:
        exit()
