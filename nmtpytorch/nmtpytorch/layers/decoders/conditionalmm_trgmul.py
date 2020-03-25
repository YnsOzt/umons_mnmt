# -*- coding: utf-8 -*-
import torch.nn.functional as F

import torch

from ...utils.nn import get_rnn_hidden_state
from ..attention import Attention, HierarchicalAttention
from . import ConditionalDecoder
from .. import FF


class ConditionalMMDecoderTRGMUL(ConditionalDecoder):
    """A conditional multimodal decoder with multimodal attention implementing the TRGMUL technique to merge Visual feat to textual one."""
    def __init__(self, fusion_type='concat', aux_ctx_name='image', **kwargs):
        super().__init__(**kwargs)
        self.aux_ctx_name = aux_ctx_name


        self.pool_transform = FF(self.ctx_size_dict['image'], self.input_size, bias=False, activ="tanh")

        # Rename textual attention layer
        self.txt_att = self.att
        del self.att

        # Visual attention over convolutional feature maps
        self.img_att = Attention(
            self.ctx_size_dict[self.aux_ctx_name], self.hidden_size,
            transform_ctx=self.transform_ctx, mlp_bias=self.mlp_bias,
            att_type=self.att_type,
            att_activ=self.att_activ,
            att_bottleneck=self.att_bottleneck)

    def f_next(self, ctx_dict, y, h):
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        self.txt_alpha_t, txt_z_t = self.txt_att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])


        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(txt_z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        logit = torch.mul(logit, self.pool_transform(ctx_dict['image'][0])).squeeze(0) #apply mult

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h2_c2)
