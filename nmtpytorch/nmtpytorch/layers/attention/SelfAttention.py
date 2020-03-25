# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import torch.nn as nn
from .MultiheadAttention import MultiheadAttention
from .. import LayerNorm
from .. import FFN


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_head, ff_size, dropout=0.1):
        super(SelfAttention, self).__init__()

        self.mhatt = MultiheadAttention(hidden_size, n_head, dropout)
        self.ffn = FFN(hidden_size, ff_size, hidden_size, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, y, y_mask):
        #print(self.dropout1(self.mhatt(y, y, y, y_mask)).shape)
        #print(y.shape)
        """
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))
        
        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))
        """
        y = y + self.mhatt(y, y, y, y_mask)
        y = y + self.ffn(y)
        return y
