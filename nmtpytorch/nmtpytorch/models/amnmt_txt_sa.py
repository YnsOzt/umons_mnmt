# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import SATextEncoder, ConditionalMMDecoder
from ..datasets import MultimodalDataset
from .nmt import NMT



logger = logging.getLogger('nmtpytorch')


class AttentiveMNMTFeaturesTXTSA(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    visual features
    """

    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',  # Multimodal context fusion (sum|mul|concat)
            'n_channels': 2048,  # depends on the features used
            'alpha_c': 0.0,  # doubly stoch. attention
            'img_sequence': False,  # if true img is sequence of img features,
            # otherwise it's a conv map

            'l2_norm': True,  # L2 normalize features
            'l2_norm_dim': -1,  # Which dimension to L2 normalize

            'enc_bidirectional': True,

            'ff_dim': 640,
            'dropout_sa': 0.0,
            'num_sa_layers': 6,
            'n_head': 4,
        })

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name and 'norm' not in name and "enc_list" not in name and "attflat_img" not in name:
                #print(name)
                nn.init.kaiming_normal(param.data)

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

    def setup(self, is_train=True):
        if self.opts.model['enc_bidirectional']:
            enc_dim = self.opts.model['enc_dim'] * 2
        else:
            enc_dim = self.opts.model['enc_dim']

        self.ctx_sizes = {str(self.sl): enc_dim, 'image': self.opts.model['n_channels']}

        ########################
        # Create Textual Encoder
        ########################
        self.enc = SATextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            bidirectional=self.opts.model['enc_bidirectional'],
            ff_size=self.opts.model['ff_dim'],
            n_head=self.opts.model['n_head'],
            num_sa_layers=self.opts.model['num_sa_layers'],
            dropout_sa=self.opts.model['dropout_sa']
        )
        # Create Decoder
        self.dec = ConditionalMMDecoder(
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
        # Let's start with a None mask by assuming that
        # we have a fixed-length feature collection
        feats_mask = None

        # Be it Numpy or NumpySequence, they return
        # (n_samples, feat_dim, t) by default
        # Convert it to (t, n_samples, feat_dim)
        feats = batch['image'].view(
            (*batch['image'].shape[:2], -1)).permute(2, 0, 1)

        if self.opts.model['img_sequence']:
            # Let's create mask in this case
            feats_mask = feats.ne(0).float().sum(2).ne(0).float()

        # L2 norm
        if self.opts.model['l2_norm']:
            feats = F.normalize(
                feats, dim=self.opts.model['l2_norm_dim']).detach()
        
        return {
            'image': (feats, feats_mask),
            str(self.sl): self.enc(batch[self.sl])
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (1 - torch.cat(self.dec.alphas).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
