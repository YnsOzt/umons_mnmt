# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
from . import MultiheadAttention
import torch.nn as nn
from .. import FFN
from .. import LayerNorm


class SGuidedAttention(nn.Module):
    def __init__(self, hidden_size, n_head, ff_size, dropout=0.1):
        super(SGuidedAttention, self).__init__()

        self.mhatt1 = MultiheadAttention(hidden_size, n_head, dropout)
        self.mhatt2 = MultiheadAttention(hidden_size, n_head, dropout)
        self.ffn = FFN(hidden_size, ff_size, hidden_size, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
