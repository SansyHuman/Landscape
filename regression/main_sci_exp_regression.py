import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

min_a = float(input("Enter minimal value of a central charge: "))
max_a = float(input("Enter maximal value of a central charge: "))
min_c = float(input("Enter minimal value of c central charge: "))
max_c = float(input("Enter maximal value of c central charge: "))
n_samples = int(input("Enter number of samples per theory: "))
n_exponents = int(input("Enter number of exponents to use from SCI: "))
poly_deg = int(input("Enter regression polynomial degree: "))

sampled_train = theory_sampler.get_balanced_sample((min_a, max_a), (min_c, max_c), n_samples)
sampled_test = theory_sampler.get_balanced_sample((min_a, max_a), (min_c, max_c), n_samples)

poly_feature_a = PolynomialFeatures(degree=poly_deg)
poly_feature_c = PolynomialFeatures(degree=poly_deg)
model_a = make_pipeline(poly_feature_a, LinearRegression())
model_c = make_pipeline(poly_feature_c, LinearRegression())

sample_stat = sampled_train.get_theory_stats()

n_theory = sampled_train.get_theory_num()
theories = sample_stat["Name"].to_list()
print("The number of theories in the sample: ", n_theory)
print("Theories in the sample: ", theories)

theories_dict = dict()
for i in range(len(theories)):
    theories_dict[theories[i]] = i

train_num = sampled_train.df.height
train_input = []
train_a = []
train_c = []

for i in range(train_num):
    train_a.append(float(sampled_train.df["CentralChargeA"][i]))
    train_c.append(float(sampled_train.df["CentralChargeC"][i]))
    sci = SuperConformalIndex(sampled_train.df["SCI"][i])
    exp_data = [sci.dims[j] if j < len(sci.dims) else 0 for j in range(n_exponents)]
    train_input.append(exp_data)

test_num = sampled_test.df.height
test_theory = []
test_input = []
test_a = []
test_c = []

for i in range(test_num):
    test_theory.append(theories_dict[sampled_test.df["Name"][i]])
    test_a.append(float(sampled_test.df["CentralChargeA"][i]))
    test_c.append(float(sampled_test.df["CentralChargeC"][i]))
    sci = SuperConformalIndex(sampled_test.df["SCI"][i])
    exp_data = [sci.dims[j] if j < len(sci.dims) else 0 for j in range(n_exponents)]
    test_input.append(exp_data)

model_a.fit(train_input, train_a)
model_c.fit(train_input, train_c)

test_a_pred = model_a.predict(test_input)
test_c_pred = model_c.predict(test_input)
r2_a = r2_score(test_a, test_a_pred)
r2_c = r2_score(test_c, test_c_pred)

r2_per_theory = [[0.0, 0.0] for _ in range(n_theory)]
test_a_per_theory = [[] for _ in range(n_theory)]
test_c_per_theory = [[] for _ in range(n_theory)]
test_a_pred_per_theory = [[] for _ in range(n_theory)]
test_c_pred_per_theory = [[] for _ in range(n_theory)]

for i in range(test_num):
    theory_index = test_theory[i]
    test_a_per_theory[theory_index].append(test_a[i])
    test_c_per_theory[theory_index].append(test_c[i])
    test_a_pred_per_theory[theory_index].append(test_a_pred[i])
    test_c_pred_per_theory[theory_index].append(test_c_pred[i])

for i in range(n_theory):
    r2_per_theory[i][0] = r2_score(test_a_per_theory[i], test_a_pred_per_theory[i])
    r2_per_theory[i][1] = r2_score(test_c_per_theory[i], test_c_pred_per_theory[i])

with open(f'../data/regression/sci_exp_regression_{n_exponents}_({min_a}_{max_a})({min_c}_{max_c})_{n_samples}_{poly_deg}.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Theory', "R2 of a", "R2 of c"])
    for i in range(n_theory):
        writer.writerow([theories[i]] + r2_per_theory[i])

plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True)

fig.suptitle(f'Regression of a/c central charge with SCI exponents\nwith {n_exponents} fit upto {poly_deg} degrees terms')

ax[0].set_title(f'a regression R2={r2_a:.3f}')
cmap = plt.cm.get_cmap('jet', n_theory)
for i in range(n_theory):
    ax[0].scatter(test_a_per_theory[i], test_a_pred_per_theory[i], color=cmap(i), label=theories[i])
a_range = [min_a, max_a]
ax[0].plot(a_range, a_range, linestyle='--', color='red', label='Exact')
ax[0].set_xlabel('Real a')
ax[0].set_ylabel('Predicted a')
ax[0].legend()

ax[1].set_title(f'c regression R2={r2_c:.3f}')
for i in range(n_theory):
    ax[1].scatter(test_c_per_theory[i], test_c_pred_per_theory[i], color=cmap(i), label=theories[i])
c_range = [min_c, max_c]
ax[1].plot(c_range, c_range, linestyle='--', color='red', label='Exact')
ax[1].set_xlabel('Real c')
ax[1].set_ylabel('Predicted c')
ax[1].legend()

plt.savefig(f'../data/regression/sci_exp_regression_{n_exponents}_({min_a}_{max_a})({min_c}_{max_c})_{n_samples}_{poly_deg}.png')
plt.show()
