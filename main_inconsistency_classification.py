from common.inconsistents_parser import *
from common.utils import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class InconsistencyClassificationModel(FullyConnectedNetwork):
    def __init__(self, input_dim: int, *args: int):
        # args: dimension of hidden layers
        super().__init__(
            input_dim,
            2,
            *([(args[i], nn.ELU()) for i in range(len(args) - 1)] + [(args[-1], nn.ReLU())])
        )


os.makedirs('./data', exist_ok=True)

filename = input("Enter file name to load: ")
inc_path = input("Enter path of inconsistents log files: ")
epoch_num = int(input("Enter number of epochs: "))

input_data, output_data = inconsistents_parser(os.path.abspath(filename), os.path.abspath(inc_path))
input_dim = input_data.shape[1]

input_train = input_data[0::2,:]
output_train = output_data[0::2,:]
input_test = input_data[1::2,:]
output_test = output_data[1::2,:]

dataset_train = GenericDataset(input_train, output_train)
print('Train dataset length:', len(dataset_train))
dataset_test = GenericDataset(input_test, output_test)
print('Test dataset length:', len(dataset_test))

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

for index, (x, y) in enumerate(dataloader_train):
    print(f'{index}/{len(dataloader_train)}', end=' ')
    print('x shape: ', x.shape, end=' ')
    print('y shape: ', y.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

model = InconsistencyClassificationModel(input_dim,
                                         input_dim * 3,
                                         input_dim * 27,
                                         input_dim * 27,
                                         input_dim * 9,
                                         input_dim * 9,
                                         input_dim * 3,
                                         input_dim * 2).to(device)
print('Inconsistency classification model shape: ', model(torch.randn(32, input_dim).to(device)).shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_loss = 1e10

checkpoint = None
checkpoint_file_name = f'./checkpoint_inconsistency_classification.tar'
if os.path.isfile(checkpoint_file_name):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(epoch_num):
    model.train()
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0.0
    error = 0.0
    test_cnt = 0

    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            test_loss += loss.item()

    print(f'epoch {epoch + 1} test loss: {test_loss / len(dataloader_test)}')
    if test_loss < best_loss:
        best_loss = test_loss
        print('New best loss obtained. Saving model...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_file_name)

test_x = torch.tensor(input_data).to(device).float()

checkpoint = torch.load(checkpoint_file_name, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

with torch.no_grad():
    y_expect = model(test_x)
    y_expect = y_expect.cpu().numpy()
    y_real = output_data

    pred_expect = np.argmax(y_expect, axis=1)
    pred_real = np.argmax(y_real, axis=1)

    cons_correct = 0
    cons_wrong = 0
    incons_correct = 0
    incons_wrong = 0

    for i in range(len(pred_real)):
        if pred_real[i] == 0:
            if pred_real[i] == pred_expect[i]:
                cons_correct += 1
            else:
                cons_wrong += 1
        elif pred_real[i] == 1:
            if pred_real[i] == pred_expect[i]:
                incons_correct += 1
            else:
                incons_wrong += 1
        else:
            raise "Invalide prediction"

    cons_error = cons_wrong / (cons_correct + cons_wrong) * 100
    incons_error = incons_wrong / (incons_correct + incons_wrong) * 100
    total_error = (cons_wrong + incons_wrong) / (cons_correct + incons_correct + cons_wrong + incons_wrong) * 100

    with open(f'./data/{filename}_inconsistency_classification.csv', 'w') as csv_file:
        csv_file.write(', Correct, Incorrect, Error (%)\n')
        csv_file.write(f'Consistent, {cons_correct}, {cons_wrong}, {cons_error}\n')
        csv_file.write(f'Inconsistent, {incons_correct}, {incons_wrong}, {incons_error}\n')
        csv_file.write(f'Total, {cons_correct + incons_correct}, {cons_wrong + incons_wrong}, {total_error}')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots()

    fig.suptitle('Inconsistency classification errors')

    p = ax.bar(['Consistent', 'Inconsistent', 'Total'], [cons_error, incons_error, total_error])
    ax.bar_label(p, fmt='%.2f')
    ax.set_ylabel('Error (%)')

    plt.savefig(f'./data/{filename}_inconsistency_classification.png')
    plt.show()