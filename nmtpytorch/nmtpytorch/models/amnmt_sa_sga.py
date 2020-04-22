# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from ..layers import TextEncoder, ConditionalMMDecoder, SASGA
from ..datasets import MultimodalDataset
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class AttentiveMNMTFeaturesSASGA(NMT):
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
            'alpha_c': 0.0,  # doubly stoch. attention
            'img_sequence': False,  # if true img is sequence of img features,
            # otherwise it's a conv map
            'l2_norm': True,  # L2 normalize features
            'l2_norm_dim': -1,  # Which dimension to L2 normalize

            'enc_bidirectional': True,

            'ff_dim': 640,
            'dropout_sa': 0.0,
            'num_sa_layers': 2,
            'n_head': 4,
        })

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name and 'norm' not in name and "enc_list" not in name and "dec_list" not in name:
                # print(name)
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
        self.ctx_sizes = {str(self.sl): enc_dim, 'image': enc_dim}  # initialized here ?

        self.imgfeat2hidden = nn.Linear(self.opts.model['n_channels'], enc_dim)

        ########################
        # Create Textual Encoder
        ########################
        self.text_enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],  # 0 to not use
            dropout_ctx=self.opts.model['dropout_ctx'],  # 0 to not use
            dropout_rnn=self.opts.model['dropout_enc'],  # 0 to not use
            num_layers=self.opts.model['n_encoders'],  # 1
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            bidirectional=self.opts.model['enc_bidirectional'])  # False to not use

        ########################
        # Create ENC-DEC
        ########################
        self.sa_sga = SASGA(hidden_size=enc_dim,
                            n_head=self.opts.model['n_head'],
                            ff_size=self.opts.model['ff_dim'],
                            num_layers=self.opts.model['num_sa_layers'],
                            dropout=self.opts.model['dropout_sa'])

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
        img_feats_mask = None

        # Be it Numpy or NumpySequence, they return
        # (n_samples, feat_dim, t) by default
        # Convert it to (t, n_samples, feat_dim)
        img_feats = batch['image'].view(
            (*batch['image'].shape[:2], -1)).permute(2, 0, 1)

        if self.opts.model['img_sequence']:
            # Let's create mask in this case
            img_feats_mask = img_feats.ne(0).float().sum(2).ne(0).float()

        # L2 norm
        if self.opts.model['l2_norm']:
            img_feats = F.normalize(
                img_feats, dim=self.opts.model['l2_norm_dim']).detach()

        img_feats = self.imgfeat2hidden(img_feats)  # transform images to the hidden size
        txt_enc_res = self.text_enc(batch[self.sl])

        lang_feats = txt_enc_res[0]
        lang_feats_mask = txt_enc_res[1]

        # enc-dec
        lang_feats, img_feats = self.sa_sga(lang_feats, img_feats, lang_feats_mask, img_feats_mask)

        return {
            'image': (img_feats, img_feats_mask),
            str(self.sl): (lang_feats, lang_feats_mask),
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (1 - torch.cat(self.dec.alphas).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result


