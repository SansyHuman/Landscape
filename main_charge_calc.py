import csv
import os.path
from common.utils import prime_numbers
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
ops_dict = dict(zip(ops, prime_numbers(len(ops))))
print(f"Operators: {ops_dict}")

codes = set()
for i in range(len(superpotentials)):
    for j in range(len(superpotentials[i])):
        code = 1
        opcode = 0
        for k in range(len(superpotentials[i][j][2])):
            if k % 2 == 0:
                opcode = ops_dict[superpotentials[i][j][2][k]]
            else:
                code *= opcode ** superpotentials[i][j][2][k]

        superpotentials[i][j][2] = code
        codes.add(code)

codes = sorted(list(codes))
codes_dict = dict(zip(codes, range(len(codes))))
print(f"Codes: {codes_dict}")

superpotentials_refined = [[0 for _ in range(len(codes) * 3)] for _ in range(len(superpotentials))]
for i in range(len(superpotentials)):
    for term in superpotentials[i]:
        code_index = codes_dict[term[2]]
        superpotentials_refined[i][code_index * 3] = term[0]
        superpotentials_refined[i][code_index * 3 + 1] = term[1]
        superpotentials_refined[i][code_index * 3 + 2] = 1

central_charges = [[0.0, 0.0] for _ in range(len(superpotentials))]
for i in range(len(superpotentials)):
    central_charges[i][0] = float(data[i + 1][a_index])
    central_charges[i][1] = float(data[i + 1][c_index])

class LandscapeDataset(Dataset):
    def __init__(self, superpotentials, central_charges):
        self.x_data = torch.tensor(superpotentials)
        self.y_data = torch.tensor(central_charges)

    def __getitem__(self, index):
        return self.x_data[index].float(), self.y_data[index].float()

    def __len__(self):
        return self.x_data.shape[0]

sup_train = [superpotentials_refined[i] for i in range(0, len(superpotentials_refined), 2)]
c_train = [central_charges[i] for i in range(0, len(central_charges), 2)]
sup_test = [superpotentials_refined[i] for i in range(1, len(superpotentials_refined), 2)]
c_test = [central_charges[i] for i in range(1, len(central_charges), 2)]

dataset_train = LandscapeDataset(sup_train, c_train)
print('Train dataset length:', len(dataset_train))

dataset_test = LandscapeDataset(sup_test, c_test)
print('Test dataset length:', len(dataset_test))

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

for index, (w, ac) in enumerate(dataloader_train):
    print(f'{index}/{len(dataloader_train)}', end=' ')
    print('x shape: ', w.shape, end=' ')
    print('y shape: ', ac.shape)

class CentralChargeModel(nn.Module):
    def __init__(self, w_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(w_size, w_size // 3),
            nn.ELU(),
            nn.Linear(w_size // 3, w_size // 9),
            nn.ReLU(),
            nn.Linear(w_size // 9, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

w_size = len(superpotentials_refined[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cc_model = CentralChargeModel(w_size).to(device)
print('Central Charge model shape:', cc_model(torch.randn(32, w_size).to(device)).shape)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cc_model.parameters(), lr=1e-3)
best_loss = 1e10

if os.path.isfile('./checkpoint.tar'):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load('./checkpoint.tar')
    cc_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(1):
    cc_model.train()
    for w, ac in dataloader_train:
        w = w.to(device)
        ac = ac.to(device)

        outputs = cc_model(w)
        loss = criterion(outputs, ac)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cc_model.eval()
    test_loss = 0.0
    error = 0.0
    test_cnt = 0

    with torch.no_grad():
        for w, ac in dataloader_test:
            w = w.to(device)
            ac = ac.to(device)

            outputs = cc_model(w)
            loss = criterion(outputs, ac)

            test_loss += loss.item()

            outputs = outputs.cpu().numpy()
            ac = ac.cpu().numpy()
            err = np.concatenate(np.abs((outputs - ac) / ac))
            error += np.sum(err)
            test_cnt += len(err)

    print(f'epoch {epoch + 1} test loss: {test_loss / len(dataloader_test)} error: {error * 100 / test_cnt} %')
    if test_loss < best_loss:
        best_loss = test_loss
        print('New best loss obtained. Saving model...')
        torch.save({
            'model_state_dict': cc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, './checkpoint.tar')

test_w = torch.tensor(superpotentials_refined).to(device).float()

checkpoint = torch.load('./checkpoint.tar')
cc_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
best_loss = checkpoint['best_loss']

with torch.no_grad():
    ac_expect = cc_model(test_w)
    ac_expect = ac_expect.cpu().numpy()
    ac_real = np.array(central_charges)

    error = (np.abs((ac_expect - ac_real) / ac_real) * 100).flatten()
    error_max = np.max(error)
    print(f'Maximum error: {error_max}')

    plt.hist(error, bins=math.ceil(error_max))
    plt.yscale('log')
    plt.title('Errors')
    plt.show()