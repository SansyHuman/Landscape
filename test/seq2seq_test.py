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


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, 1)
        self.Wb = nn.Linear(hidden_size, hidden_size)
        self.Wc = nn.Linear(hidden_size, hidden_size)

    def forward(self, encoder_hidden, decoder_hidden):
        score = self.Wa(F.tanh(torch.add(self.Wb(decoder_hidden), self.Wc(encoder_hidden))))
        score = score.squeeze(2)

        weight = F.softmax(score, dim=-1)
        weight = weight.unsqueeze(1)

        context = torch.bmm(weight, encoder_hidden)
        return context

encoder_hidden = torch.zeros(32, 4, 9)
decoder_hidden = torch.zeros(32, 1, 9)
attention = BahdanauAttention(9)

print(attention(encoder_hidden, decoder_hidden).shape)