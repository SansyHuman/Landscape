import polars as pl
from sklearn.linear_model import LinearRegression
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
    train_a.append(float(sampled_test.df["CentralChargeA"][i]))
    train_c.append(float(sampled_test.df["CentralChargeC"][i]))
    sci = SuperConformalIndex(sampled_test.df["SCI"][i])
    exp_data = [sci.dims[j] if j < len(sci.dims) else 0 for j in range(n_exponents)]
    train_input.append(exp_data)
