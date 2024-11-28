import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import os, sys
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.join(base_path),os.path.join(base_path, "../"),base_path.rsplit('/')[0]])
from utils import p_log


class EBSNN_LSTM(nn.Module):
    def __init__(self, num_class, embedding_dim, device,
                 bidirectional=True, segment_len=8, dropout_rate=0.5):
        super(EBSNN_LSTM, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda:{}".format(device))
        self.segment_len = segment_len
        self.rnn_dim = 100
        # if bi-direction
        self.rnn_directions = 2 if bidirectional else 1
        self.padding_idx = 256

        # 256 is 'gg', will be set [0,0..0]
        # TODO: ???
        self.byte_embed = nn.Embedding(257, self.embedding_dim, padding_idx=self.padding_idx)
        self.byte_embed.requires_grad = True

        self.rnn1 = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.rnn_dim,
                            batch_first=True, dropout=dropout_rate,
                            bidirectional=(self.rnn_directions == 2))
        self.rnn2 = nn.LSTM(input_size=self.rnn_directions * self.rnn_dim, 
                            hidden_size=self.rnn_dim,
                            batch_first=True, dropout=dropout_rate,
                            bidirectional=(self.rnn_directions == 2))

        self.hc1 = torch.randn(self.rnn_dim, 1)
        self.hc2 = torch.randn(self.rnn_dim, 1)

        self.fc1 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc3 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.num_class)

        self.dropout = nn.Dropout(0.5)

    '''
    x: b * l * 8
    lengths: (b,), every batch's length
    '''

    def forward(self, x):
        # x.shape = (B, L, 8) where L is different for every packet
        # TODO: pad and pack
        seq_lengths = torch.tensor([seq.size(0) for seq in x]).int()
        x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
        # out.shape = (B, l=max(L), 8)
        batch_size = x.size(0)

        x = self.byte_embed(x)  
        # out.shape = (B, l, 8, 64)
        x = rnn_utils.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        ### RNN1
        out1, _ = self.rnn1(x) 
        # out1, _ = self.rnn1(x.view(-1, self.segment_len, self.embedding_dim)) 
        # out.shape = (b*l, 8, f=100)

        ## Attention layer
        # FC
        h = torch.tanh(self.fc1(out1.contiguous().view(
            -1, self.rnn_directions * self.rnn_dim))).view(
                out1.size(0), out1.size(1), -1)
        h = h.cuda(self.device)
        # FC * q
        self.hc1 = self.hc1.cuda(self.device)
        weights = (torch.matmul(h, self.hc1)).view(-1, self.segment_len)
        # softmax(FC * q)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(-1, 1, self.segment_len)
        # H * softmax(FC * q)
        out2 = torch.matmul(weights, out1).view(
            batch_size, -1, self.rnn_dim * self.rnn_directions)
        # out.shape = (b*l, f=100)

        ### RNN2
        out3, (h1_n, h2_n) = self.rnn2(out2)  
        # out.shape = (b, l, f=100)

        ## Attention layer
        # FC
        h2 = torch.tanh(self.fc2(out3.contiguous().view(
            -1, self.rnn_dim * self.rnn_directions))).view(
                out3.size(0), out3.size(1), -1)
        h2 = h2.cuda(self.device)
        # softmax(FC * q)
        self.hc2 = self.hc2.cuda(self.device)
        weights2 = F.softmax((torch.matmul(h2, self.hc2)).view(
            batch_size, -1), dim=1).view(batch_size, 1, -1)
        # H * softmax(FC * q)
        out4 = torch.matmul(weights2, out3).view(
            batch_size, self.rnn_dim * self.rnn_directions)
        # out.shape = (b*l, f=100)

        ### d --> logits
        out = self.dropout(self.fc3(out4))
        # out.shape = (b*l, C)
        return out
