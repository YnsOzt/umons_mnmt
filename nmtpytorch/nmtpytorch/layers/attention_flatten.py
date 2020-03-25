import torch.nn as nn
import torch.nn.functional as F
import torch
from . import FFN

class AttentionFlatten(nn.Module):
    def __init__(self, hidden_size, mid_size, flat_glimpses, flat_out_size, dropout=0.1):
        super(AttentionFlatten, self).__init__()

        self.flat_glimpses = flat_glimpses

        self.mlp = FFN(hidden_size, mid_size, flat_glimpses, dropout)

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        
        if x_mask is not None:
          att = att.masked_fill(
              x_mask.squeeze(1).squeeze(1).unsqueeze(2),
              -1e9
          )

        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
            
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted
