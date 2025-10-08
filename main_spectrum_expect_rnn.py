import csv
import os.path
from common.utils import *
import math
import json
from common.sci_parser import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


filename = input("Enter file name to load: ")

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

field_content_index, a_index, c_index, sci_index = -1, -1, -1, -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i
    elif data[0][i] == "CentralChargeA":
        a_index = i
    elif data[0][i] == "CentralChargeC":
        c_index = i
    elif data[0][i] == "SCI":
        sci_index = i

print(f'Field content: {field_content_index}, A: {a_index}, C: {c_index}, SCI: {sci_index}')

field_contents_index = dict()
field_contents = []
a_charges = []
c_charges = []
scis = []

for i in range(1, len(data)):
    field_content = data[i][field_content_index]
    a, c = float(data[i][a_index]), float(data[i][c_index])
    sci = SuperConformalIndex(data[i][sci_index].strip())

    if field_content not in field_contents_index:
        field_contents_index[field_content] = len(field_contents_index)
    field_contents.append(field_contents_index[field_content])
    a_charges.append(a)
    c_charges.append(c)
    scis.append(sci)

print(f"Field contents: {field_contents_index}")

os.makedirs('./data', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_token = 0


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, nonlinearity: str='tanh'):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlinearity, batch_first=True)

    def forward(self, input):
        output, hidden = self.rnn(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, output_size: int, length: int, nonlinearity: str='tanh'):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.length = length

        self.rnn = nn.RNN(output_size, hidden_size, num_layers, nonlinearity=nonlinearity, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, self.output_size, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing 포함: 목표를 다음 입력으로 전달
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                decoder_input = decoder_output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)
        return output, hidden


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion) -> None:
    encoder.train()
    decoder.train()

    for input, target in dataloader:
        input, target = input.to(device), target.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target)

        loss = criterion(decoder_outputs, target)

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


def test_epoch(dataloader, encoder, decoder, criterion) -> tuple[float, float]:
    """

    :param dataloader:
    :param encoder:
    :param decoder:
    :param criterion:
    :return: test loss and test error
    """
    encoder.eval()
    decoder.eval()

    test_loss = 0.0
    error = 0.0
    test_cnt = 0
    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)

            encoder_outputs, encoder_hidden = encoder(input)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            loss = criterion(decoder_outputs, target)

            test_loss += loss.item()

            decoder_outputs = decoder_outputs.cpu().numpy()
            target = target.cpu().numpy()
            err = np.abs((decoder_outputs - target) / target).flatten()
            error += np.sum(err)
            test_cnt += len(err)

    return test_loss / len(dataloader), error / test_cnt


input_num = int(input("Number of input spectrum: "))
output_num = int(input("Number of output spectrum: "))
epoch_num = int(input("Number of epochs: "))

input_data = np.zeros((len(a_charges), input_num, 3)) # [a_i, c_i, dimension_ij]
output_data = np.zeros((len(a_charges), output_num, 1)) # [dimension_ij]

for i in range(len(a_charges)):
    a_charge = a_charges[i]
    c_charge = c_charges[i]
    sci = scis[i]
    for j in range(input_num):
        input_data[i, j, 0] = a_charges[j]
        input_data[i, j, 1] = c_charges[j]
        if j >= len(sci.dims):
            input_data[i, j, 2] = -1 # -1 represents no such operator
        else:
            input_data[i, j, 2] = sci.dims[j]

    for j in range(output_num):
        index = j + input_num
        if index >= len(sci.dims):
            output_data[i, j, 0] = -1
        else:
            output_data[i, j, 0] = sci.dims[index]

input_train = input_data[0::2,:,:]
output_train = output_data[0::2,:,:]
input_test = input_data[1::2,:,:]
output_test = output_data[1::2,:,:]

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

hidden_size = 3 * 5
num_layers = 3
encoder = EncoderRNN(3, hidden_size, num_layers, 'relu').to(device)
decoder = DecoderRNN(hidden_size, num_layers, 1, output_num, 'relu').to(device)

encoder_outputs, encoder_hidden = encoder(torch.randn(32, input_num, 3).to(device))
decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
print('Higher spectrum expect model shape: ', decoder_outputs.shape)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_loss = 1e10

checkpoint = None
checkpoint_file_name = f'./checkpoint_spectrum_expect_rnn_{input_num}_{output_num}.tar'
if os.path.isfile(checkpoint_file_name):
    print('Checkpoint available. Loads checkpoint...')
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    best_loss = checkpoint['best_loss']

for epoch in range(epoch_num):
    train_epoch(dataloader_train, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    loss, error = test_epoch(dataloader_test, encoder, decoder, criterion)

    print(f'epoch {epoch + 1} test loss: {loss} error: {error * 100} %')
    if loss < best_loss:
        best_loss = loss
        print('New best loss obtained. Saving model...')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_file_name)