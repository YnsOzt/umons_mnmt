import torch.nn as nn
from . import TextEncoder
from .. import SelfAttention
from .. import SGuidedAttention
from .. import AttentionFlatten

class TextTransformer(TextEncoder):
    def __init__(self, use_sa_x, use_sa_y, use_sga, use_attflat, n_head, dropout_sa, trans_num_layers, ff_dim, flat_mlp_size, **kwargs):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.dropout_sa = dropout_sa
        self.trans_num_layers = trans_num_layers
        self.ff_dim = ff_dim
        # self.flat_mlp_size = flat_mlp_size
        self.use_sa_x = use_sa_x
        self.use_sa_y = use_sa_y
        self.use_sga = use_sga
        self.use_attflat = use_attflat
        self.adapter = None

        if use_sa_x:
            self.enc_list_x = nn.ModuleList(
                [SelfAttention(self.ctx_size, self.n_head, self.ff_dim, self.dropout_sa) for _ in
                 range(self.trans_num_layers)])

        #  Adapter
        if use_sga or use_sa_y:
            self.adapter = nn.Linear(self.n_channels, self.ctx_size)
            self.n_channels = self.ctx_size

        if use_sa_y:
            self.enc_list_y = nn.ModuleList(
                [SelfAttention(self.n_channels, self.n_head, self.ff_dim, self.dropout_sa) for _ in
                 range(self.trans_num_layers)])

        if use_sga:
            self.dec_list = nn.ModuleList(
                [SGuidedAttention(self.n_channels, self.n_head, self.ff_dim, self.dropout_sa) for _ in
                 range(self.trans_num_layers)])

        if use_attflat:
            self.attflat = AttentionFlatten(self.n_channels, self.n_channels, 1, self.n_channels)

    def forward_same_len_batches(self, x, y, y_mask):
        # Fetch embeddings
        embs = self.emb(x)

        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Encode
        x, _ = self.enc(embs)

        if self.dropout_ctx > 0:
            x = self.do_ctx(x)

        # (t, BS, f) => (BS, t, f)
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)

        if self.use_sa_x:
            for sa in self.enc_list_x:
                x = sa(x, None)

        if self.adapter is not None:
            y = self.adapter(y)

        if self.use_sa_y:
            for sa in self.enc_list_y:
                y = sa(y, y_mask)

        if self.use_sga:
            for dec in self.dec_list:
                y = dec(y, x, y_mask, None)

        if self.use_attflat:
            y = self.attflat(y, y_mask).unsqueeze(1)

        # (BS, t, f) => (t, BS, f)
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)

        return x, None, y, None