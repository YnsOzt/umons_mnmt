# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, n_head, dropout):
        super(MultiheadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.n_head = n_head

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.linear_merge = nn.Linear(hidden_size, hidden_size)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.n_head,
            int(self.hidden_size / self.n_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.n_head,
            int(self.hidden_size / self.n_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.n_head,
            int(self.hidden_size / self.n_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)
        
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
