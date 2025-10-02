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


"""
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
"""

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
