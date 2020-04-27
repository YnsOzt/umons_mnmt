# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn.functional as F
from ..layers import XuDecoder

from ..datasets import MultimodalDataset

from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class VIC(NMT):
    r"""An Implementation of 'Show, attend and tell' image captioning paper.

    Paper: http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf
    Reference implementation: https://github.com/kelvinxu/arctic-captions
    """
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'dec_dim': 256,             # Decoder hidden size
            'n_channels': 1024,         # Number of channels
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx)
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': True,       # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout': 0,               # Simple dropout
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'selector': True,           # Selector gate
            'alpha_c': 0.0,             # Attention regularization
            'prev2out': True,           # Add prev embedding to output
            'ctx2out': True,            # Add context to output
            'cnn_type': 'resnet50',     # A variant of VGG or ResNet
            'cnn_layer': 'res5c_relu',  # From where to extract features
            'cnn_pretrained': True,     # Should we use pretrained imagenet weights
            'cnn_finetune': None,       # Should we finetune part or all of CNN
            'pool': None,               # ('Avg|Max', kernel_size, stride_size)
            'l2_norm': False,           # L2 normalize features
            'l2_norm_dim': -1,          # Which dimension to L2 normalize
            'resize': 256,              # resize width, height for images
            'crop': 224,                # center crop size after resize
            'replicate': 1,             # number of captions/image
            'direction': None,          # Network directionality, i.e. en->de
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
            'trans_num_layers': 2,
            'n_head': 4,
            'flat_mlp_size': 520,
            'ff_dim': 640,
            'use_sa': False,
            'use_lstm': False,
            'use_attflat': False,
            'img_sequence': False,
            'decoder_type': 'xu',
        }

    def __init__(self, opts):
        super().__init__(opts)
        self.encoder_class = None
        self.decoder_class = None
        self.ctx_sizes = {'image': self.opts.model['n_channels']}

        # Viz dim and decoder dim check
        if self.opts.model['use_attflat'] and self.opts.model['decoder_type'] in ('normal'):
            raise Exception("Can use attflat with dec > 1D")


    def setup(self, is_train=True):

        # Encoder definition
        if self.opts.model['use_sa'] or self.opts.model['use_lstm'] or self.opts.model['use_attflat']:
            self.encoder_class = ImgTransformer

        # Decoder definition
        if self.opts.model['decoder_type'] in ('xu'):
            self.decoder_class = XuDecoder

        # Optional encoder
        if self.encoder_class is not None:
            self.text_enc = encoder_class(
                use_sa=self.opts.model['use_sa'],
                use_lstm=self.opts.model['use_lstm'],
                use_attflat=self.opts.model['use_attflat'],
                dropout_rnn=self.opts.model['dropout_enc'],  # 0 to not use
                lstm_num_layers=self.opts.model['lstm_num_layers'],  # 1
                bidirectional=self.opts.model['bidirectional'],
                n_head=self.opts.model['n_head'],
                dropout_sa=self.opts.model['dropout_sa'],
                ff_dim=self.opts.model['ff_dim'],
                trans_num_layers=self.opts.model['trans_num_layers'],
                flat_mlp_size=self.opts.model['flat_mlp_size'],
                n_channels=self.opts.model['n_channels'],
                ctx_size=self.opts.model['dec_dim']
            )
            self.ctx_sizes = {'image':  self.text_enc.n_channels}

        # Create Decoder
        self.dec = self.decoder_class(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='image',
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout=self.opts.model['dropout'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            selector=self.opts.model['selector'],
            prev2out=self.opts.model['prev2out'],
            ctx2out=self.opts.model['ctx2out'])

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            )
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Get features into (n,c,w*h) and then (w*h,n,c)
        x = batch['image']
        x = x.view((*x.shape[:2], -1)).permute(2, 0, 1)

        if self.opts.model['l2_norm']:
            x = F.normalize(
                x, dim=self.opts.model['l2_norm_dim']).detach()

        # Img Masking
        x_mask = None
        if self.opts.model['img_sequence']:
            x_mask = x.ne(0).float().sum(2).ne(0).float()
        # For now, no masking for text because we use only forward_same_len_batches
        # For now, no masking for image because always full of features
        x, x_mask = self.text_enc(x, None)
        return {'image': (x, None)}

    def forward(self, batch, **kwargs):
        result = super().forward(batch)
        return result
