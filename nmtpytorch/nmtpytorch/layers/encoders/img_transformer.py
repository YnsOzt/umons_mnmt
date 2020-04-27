import torch.nn as nn
from . import TextEncoder
from .. import SelfAttention
from .. import SGuidedAttention
from .. import AttentionFlatten

class ImgTransformer(nn.Module):
    def __init__(self, use_sa, use_lstm, use_attflat, dropout_rnn, lstm_num_layers, bidirectional, n_head, dropout_sa,
                 trans_num_layers, ff_dim, flat_mlp_size, n_channels, ctx_size):

        super().__init__(**kwargs)
        self.use_sa = use_sa
        self.use_attflat = use_attflat
        self.use_lstm = use_lstm

        # LSTM
        self.use_lstm = use_lstm
        self.dropout_rnn = dropout_rnn
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional

        # Transfo
        self.n_head = n_head
        self.dropout_sa = dropout_sa
        self.trans_num_layers = trans_num_layers
        self.ff_dim = ff_dim
        # self.flat_mlp_size = flat_mlp_size

        # Viz feat
        self.n_channels = n_channels
        self.adapter = None
        self.ctx_size = ctx_size  # Dec size

        if use_lstm:
            self.enc = nn.LSTM(self.n_channels, self.ctx_size,
                           self.lstm_num_layers, bias=True, batch_first=False,
                           dropout=self.dropout_rnn,
                           bidirectional=self.bidirectional)
            self.n_channels = ctx_size

        if not use_lstm and use_sa:
            self.adapter = nn.Linear(self.n_channels, self.ctx_size)
            self.n_channels = ctx_size

        if use_sa:
            self.enc_list = nn.ModuleList(
                [SelfAttention(self.n_channels, self.n_head, self.ff_dim, self.dropout_sa) for _ in
                 range(self.trans_num_layers)])

        if use_attflat:
            self.attflat = AttentionFlatten(self.n_channels, self.n_channels, 1, self.n_channels)

    def forward(self, x, x_mask):
        # Encode
        if self.use_lstm:
            x, _ = self.enc(x)

        # (t, BS, f) => (BS, t, f)
        x = x.transpose(1, 0)

        if self.adapter is not None:
            x = self.adapter(x)

        if self.use_sa:
            for sa in self.enc_list:
                x = sa(x, x_mask)

        if self.use_attflat:
            x = self.attflat(x, x_mask).unsqueeze(1)

        # (BS, t, f) => (t, BS, f)
        x = x.transpose(1, 0)

        return x, x_mask