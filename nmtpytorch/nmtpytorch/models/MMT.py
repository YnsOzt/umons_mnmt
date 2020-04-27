# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from ..layers import TextEncoder, TextTransformer, ConditionalMMDecoder, ConditionalMMDecoderTRGMUL
from ..datasets import MultimodalDataset
from .nmt import NMT
from ..layers import AttentionFlatten


logger = logging.getLogger('nmtpytorch')


class MMT(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features + normalization of the feats.
    """

    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'enc_type': 'LSTM',
            'fusion_type': 'concat',  # Multimodal context fusion (sum|mul|concat)
            'n_channels': 2048,  # depends on the features used
            'img_sequence': False,  # if true img is sequence of img features,
            # otherwise it's a conv map
            'l2_norm': False,  # L2 normalize features
            'l2_norm_dim': -1,  # Which dimension to L2 normalize
            'bidirectional': False,
            'dropout_sa': 0.0,
            'lstm_num_layers': 2,
            'trans_num_layers': 2,
            'n_head': 4,
            'flat_mlp_size': 520,
            'ff_dim': 640,
            'use_sa_x': False,
            'use_sa_y': False,
            'use_sga': False,
            'use_attflat': False,
            'decoder_type': 'normal'  # specify the decoder that you will use (normal | trgmul)
        })
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad \
                    and 'bias' not in name \
                    and 'norm' not in name \
                    and "enc_list" not in name \
                    and "dec_list" not in name \
                    and "attflat_img" not in name:
                #print(name)
                nn.init.kaiming_normal(param.data)

    def __init__(self, opts):
        super().__init__(opts)

        # Decoder type check
        if self.opts.model['decoder_type'] not in ('normal', 'trgmul'):
          raise Exception("You've specified a wrong decoder_type, it should be one of (normal, trgmul)")

        # Viz dim and decoder dim check
        if self.opts.model['img_sequence'] \
                and not self.opts.model['use_attflat']\
                and self.opts.model['decoder_type'] in ('trgmul'):
            raise Exception("Need use_attflat for dec 1D using viz features > 1D")

        if self.opts.model['use_attflat'] and self.opts.model['decoder_type'] in ('normal'):
            raise Exception("Can use attflat with dec > 1D")

        # Viz dim and encoder dim check
        if not self.opts.model['img_sequence'] \
                and (self.opts.model['use_sa_y'] or self.opts.model['use_sga'] or self.opts.model['use_attflat']):
            raise Exception("Can use SA, SGA, attflat on 1D viz features")


    def setup(self, is_train=True):

        # Encoder definition
        encoder_class = TextEncoder
        if self.opts.model['use_sa_x'] or self.opts.model['use_sa_y'] or self.opts.model['use_sga']:
            encoder_class = TextTransformer

        # Decoder definition
        decoder_class = ConditionalMMDecoder
        if self.opts.model['decoder_type'] in ('trgmul'):
            decoder_class = ConditionalMMDecoderTRGMUL

        # Encoder
        self.text_enc = encoder_class(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],  # 0 to not use
            dropout_ctx=self.opts.model['dropout_ctx'],  # 0 to not use
            dropout_rnn=self.opts.model['dropout_enc'],  # 0 to not use
            lstm_num_layers=self.opts.model['lstm_num_layers'],  # 1
            bidirectional=self.opts.model['bidirectional'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            n_channels=self.opts.model['n_channels'],
            use_sa_x=self.opts.model['use_sa_x'],
            use_sa_y=self.opts.model['use_sa_y'],
            use_sga=self.opts.model['use_sga'],
            use_attflat=self.opts.model['use_attflat'],
            n_head=self.opts.model['n_head'],
            dropout_sa=self.opts.model['dropout_sa'],
            ff_dim=self.opts.model['ff_dim'],
            trans_num_layers=self.opts.model['trans_num_layers'],
            flat_mlp_size=self.opts.model['flat_mlp_size'],
            )

        # Manage sizes for dec
        self.ctx_sizes = {str(self.sl): self.text_enc.ctx_size, 'image': self.text_enc.n_channels}

        # Decoder
        self.dec = decoder_class(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        x = batch[self.sl]
        y = batch['image']
        # (t, f, b) to (t, b, f)
        y = y.view((*y.shape[:2], -1)).permute(2, 0, 1)

        # Img Masking
        y_mask = None
        if self.opts.model['img_sequence']:
            y_mask = y.ne(0).float().sum(2).ne(0).float()

        # L2 norm
        if self.opts.model['l2_norm']:
            y = F.normalize(y, dim=self.opts.model['l2_norm_dim']).detach()

        # For now, no masking for text because we use only forward_same_len_batches
        # For now, no masking for image because always full of features
        x, x_mask, y, y_mask = self.text_enc(x, y, None)

        return {
            str(self.sl): (x, x_mask),
            'image': (y, y_mask),
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)
        return result


