import csv
import math
import os.path
from unittest import case

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from common.utils import prime_numbers

'''
Split the operator to operator letter index and operator index
Index: q, qb, phi, S, Sb, A, Ab, M, X from 0 to 8
'''
def op_index(op: str) -> (int, int):
    for i in range(len(op)):
        if op[i].isdigit():
            return op_letter_index(op[:i]), int(op[i:])

def op_letter_index(letter: str) -> int:
    match letter:
        case "q":
            return 0
        case "qb":
            return 1
        case "phi":
            return 2
        case "S":
            return 3
        case "Sb":
            return 4
        case "A":
            return 5
        case "Ab":
            return 6
        case "M":
            return 7
        case "X":
            return 8
        case _:
            return -1

def op_index_letter(index: int) -> str:
    return ["q", "qb", "phi", "S", "Sb", "A", "Ab", "M", "X"][index]

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
max_w_length = -1
max_op_index = -1

for i in range(1, len(data)):
    w = data[i][w_index][1:-1].split(',')
    if w[0] == '':
        superpotentials.append([])
        continue

    w_data = []
    for term in w:
        term_data = [] # [operator, index, exponent, ...]
        operators = term.strip().split('*')
        for operator in operators:
            op_exp = operator.split('^')
            if len(op_exp) == 1:
                op_exp.append(1)
            else:
                op_exp[1] = int(op_exp[1])

            term_data += op_index(op_exp[0])
            term_data.append(op_exp[1])
            if term_data[-2] > max_op_index:
                max_op_index = term_data[-2]

        w_data.append(term_data)

    superpotentials.append(w_data)
    if len(w_data) > max_w_length:
        max_w_length = len(w_data)

print(f'Maximum superpotential length: {max_w_length}')

'''
input: max_w_length X 9 matrix
each row is each term of the superpotential(if less than max_w_length, fills 0)
each row data is {q, qb, phi, S, Sb, A, Ab, M, X} with each is a code of matter index and power
code: *(i-th prime number where i is the matter index ^ power)
'''

prime_code = prime_numbers(max_op_index)

superpotentials_refined = np.zeros((len(superpotentials), 1, max_w_length, 9))
for i in range(len(superpotentials)):
    w = superpotentials[i]
    for j in range(len(w)):
        op = w[j]
        for k in range(0, len(op), 3):
            opcode = prime_code[op[k + 1] - 1] ** op[k + 2]
            if superpotentials_refined[i][0][j][op[k]] == 0:
                superpotentials_refined[i][0][j][op[k]] = opcode
            else:
                superpotentials_refined[i][0][j][op[k]] *= opcode

central_charges = np.zeros((len(superpotentials), 2))
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

class CentralChargeConvModel(nn.Module):
    def __init__(self, w_size, op_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.ELU(),
            nn.Conv2d(8, 16, 3, 1),
            nn.ELU(),
        )

        final_mat_size = 16 * (w_size - 4) * (op_size - 4)

        self.flat_layers = nn.Sequential(
            nn.Linear(final_mat_size, final_mat_size // 8),
            nn.ELU(),
            nn.Linear(final_mat_size // 8, final_mat_size // 32),
            nn.ReLU(),
            nn.Linear(final_mat_size // 32, 2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.flat_layers(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cc_model = CentralChargeConvModel(max_w_length, 9).to(device)
print('Central Charge model shape:', cc_model(torch.randn(32, 1, max_w_length, 9).to(device)).shape)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cc_model.parameters(), lr=1e-3)
best_loss = 1e10

if os.path.isfile('./checkpoint_conv.tar'):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load('./checkpoint_conv.tar')
    cc_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(100):
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
        }, './checkpoint_conv.tar')

test_w = torch.tensor(superpotentials_refined).to(device).float()

checkpoint = torch.load('./checkpoint_conv.tar')
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