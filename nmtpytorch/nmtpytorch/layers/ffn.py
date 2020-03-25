import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout, apply_relu=True):
        super(FFN, self).__init__()

        self.apply_relu = apply_relu
        self.dropout = dropout

        self.linear1 = nn.Linear(in_size, mid_size)

        if apply_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout > 0:
            self.do = nn.Dropout(dropout)

        self.linear2 = nn.Linear(mid_size, out_size)

    def forward(self, x):
        x = self.linear1(x)
        if self.apply_relu:
            x = self.relu(x)
        if self.dropout > 0:
            x = self.do(x)

        x = self.linear2(x)
        return x
