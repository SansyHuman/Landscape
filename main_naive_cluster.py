from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from common.utils import prime_numbers
import numpy as np
import csv

data = None
with open("landscape_SU2adj1nf2.csv") as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

w_index, a_index, c_index, rational_index = -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Superpotentials":
        w_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "Rational":
        rational_index = i

print(f'Superpotentials: {w_index}, A: {a_index}, C: {c_index}, Rational: {rational_index}')

superpotentials = []
ops = set()

for i in range(1, len(data)):
    w = data[i][w_index][1:-1].split(',')
    if w[0] == '':
        superpotentials.append([[0, 0, []]])
        continue

    w_data = []
    for term in w:
        term_data = [0, 0, []] # M index, X index, [operator, exponent, ...]
        operators = term.strip().split('*')
        for operator in operators:
            op_exp = operator.split('^')
            if len(op_exp) == 1:
                op_exp.append(1)
            else:
                op_exp[1] = int(op_exp[1])
            if op_exp[0][0] == 'M':
                m_index = int(op_exp[0][1:])
                if term_data[0] == 0:
                    term_data[0] = m_index
                    op_exp[1] -= 1
            elif op_exp[0][0] == 'X':
                x_index = int(op_exp[0][1:])
                if term_data[1] == 0:
                    term_data[1] = x_index
                    op_exp[1] -= 1

            if op_exp[1] != 0:
                term_data[2].append(op_exp[0])
                term_data[2].append(op_exp[1])
                ops.add(op_exp[0])

        w_data.append(term_data)
    superpotentials.append(w_data)

ops = sorted(list(ops))
ops_index = prime_numbers(len(ops))
ops_dict = dict(zip(ops, ops_index))
print(f"Operators: {ops_dict}")

terms = set()

for i in range(len(superpotentials)):
    for j in range(len(superpotentials[i])):
        opcode = 1
        op_index = 0
        for k in range(len(superpotentials[i][j][2])):
            if k % 2 == 0:
                op_index = ops_dict[superpotentials[i][j][2][k]]
            else:
                opcode *= (op_index ** superpotentials[i][j][2][k])
        superpotentials[i][j][2] = opcode
        terms.add(opcode)

terms = sorted(list(terms))
terms_dict = dict(zip(terms, range(len(terms))))
print(f"Terms: {terms_dict} total {len(terms)} terms")

superpotential_refined = [[0 for _ in range(len(terms) * 3 + 2)] for _ in range(len(superpotentials))]
for i in range(len(superpotentials)):
    for j in range(len(superpotentials[i])):
        term_index = terms_dict[superpotentials[i][j][2]]
        superpotential_refined[i][term_index * 3] = superpotentials[i][j][0]
        superpotential_refined[i][term_index * 3 + 1] = superpotentials[i][j][1]
        superpotential_refined[i][term_index * 3 + 2] = 1
    superpotential_refined[i][-2] = float(data[i + 1][a_index])
    superpotential_refined[i][-1] = float(data[i + 1][c_index])

depth = 4
superpotential_refined = [superpotential_refined]
rational = [[data[i + 1][rational_index] != '' for i in range(len(superpotential_refined[0]))]]
cluster_centers = []
kmeans = KMeans(n_clusters=2, max_iter=1000, tol=1e-15)

for i in range(depth):
    for n in range(2**i):
        n *= 2
        kmeans.fit(superpotential_refined[n])
        print(f'Number of iterations: {kmeans.n_iter_}')

        cluster_1 = []
        cluster_1_rat = []
        cluster_2 = []
        cluster_2_rat = []
        for j in range(len(kmeans.labels_)):
            if kmeans.labels_[j] == 0:
                cluster_1.append(superpotential_refined[n][j])
                cluster_1_rat.append(rational[n][j])
            else:
                cluster_2.append(superpotential_refined[n][j])
                cluster_2_rat.append(rational[n][j])

        superpotential_refined[n] = cluster_1
        rational[n] = cluster_1_rat
        superpotential_refined.insert(n + 1, cluster_2)
        rational.insert(n + 1, cluster_2_rat)

        if i == depth - 1:
            cluster_centers.append(kmeans.cluster_centers_[0])
            cluster_centers.append(kmeans.cluster_centers_[1])

nr = [0 for _ in range(len(superpotential_refined))]
r = [0 for _ in range(len(superpotential_refined))]

for n in range(len(superpotential_refined)):
    for i in range(len(superpotential_refined[n])):
        if rational[n][i]:
            r[n] += 1
        else:
            nr[n] += 1

for n in range(len(superpotential_refined)):
    print(f'Ratio of rational central charges in cluster {n + 1}: {float(r[n]) * 100 / float(nr[n] + r[n])} %')

x = np.arange(len(nr))
cluster_index = [i + 1 for i in range(len(nr))]
nr = np.array(nr)
r = np.array(r)
ratio = r / (nr + r)

plt.rcParams["figure.figsize"] = (16, 12)
plt.bar(x, ratio)
plt.xticks(x, cluster_index)

plt.title('Ratio of rational central charges in cluster')
plt.xlabel('Cluster')
plt.ylabel('Ratio of rational central charges')

plt.show()