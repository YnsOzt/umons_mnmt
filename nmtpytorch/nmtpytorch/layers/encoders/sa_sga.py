# -*- coding: utf-8 -*-
import torch.nn as nn
from .. import SelfAttention
from .. import SGuidedAttention

class SASGA(nn.Module):
    def __init__(self, hidden_size, n_head=8, ff_size=2048, num_layers=6, dropout=0.1):
        super().__init__()

        self.enc_list = nn.ModuleList([SelfAttention(hidden_size, n_head, ff_size, dropout) for _ in range(num_layers)])
        self.dec_list = nn.ModuleList([SGuidedAttention(hidden_size, n_head, ff_size, dropout) for _ in range(num_layers)])

    def forward(self, y, x, y_mask=None, x_mask=None):
        y = y.transpose(0, 1)
        x = x.transpose(0, 1)
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        y = y.transpose(0, 1)
        x = x.transpose(0, 1)

        return y, x