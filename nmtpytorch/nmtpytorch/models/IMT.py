# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from ..layers import TextEncoder, SATextEncoder, ConditionalMMDecoder, ConditionalMMDecoderTRGMUL, SASGA
from ..datasets import MultimodalDataset
from .nmt import NMT
from ..layers import AttentionFlatten


logger = logging.getLogger('nmtpytorch')


class IMT(NMT):
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
            'l2_norm': False,  # L2 normalize features
            'l2_norm_dim': -1,  # Which dimension to L2 normalize

            'enc_bidirectional': True,

            'ff_dim': 640,
            'dropout_sa': 0.0,
            'num_layers': 2,
            'n_head': 4,
            'flat_mlp_size': 520,
            
            'encoder_type':'simple', # specify the encoder that you will use (simple | sa | sasga)
            'decoder_type':'simple'  # specify the decoder that you will use (simple | trgmul)
        })
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name and 'norm' not in name and "enc_list" not in name and "dec_list" not in name and "attflat_img" not in name:
                #print(name)
                nn.init.kaiming_normal(param.data)

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0
        
        if self.opts.model['encoder_type'] not in ('simple', 'sa', 'sasga'):
          raise Exception("You've specified a wrong encoder_type, it should be one of (simple, sa, sasga)")

        if self.opts.model['decoder_type'] not in ('simple', 'trgmul'):
          raise Exception("You've specified a wrong decoder_type, it should be one of (simple, trgmul)")

    def setup(self, is_train=True):
        if self.opts.model['enc_bidirectional']:
            enc_dim = self.opts.model['enc_dim'] * 2
        else:
            enc_dim = self.opts.model['enc_dim']

        self.ctx_sizes = {str(self.sl): enc_dim, 'image': self.opts.model['n_channels'] } 


        if self.opts.model['encoder_type'] in ("simple", "sasga"):
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

          # if it's the sasga encoder, we will also add this layer.
          if self.opts.model['encoder_type'] == "sasga":
            # layers which will reduce the dimensionality from n_channel to the enc_dim
            # so that the multihead attention for guided attention works correctly
            self.imgfeat2hidden = nn.Linear(self.opts.model['n_channels'], enc_dim)

            # as we are changin the dimension of the auxiliary feats, we should also update the ctx_sizes dict
            self.ctx_sizes['image'] = enc_dim

            ########################
            # Create sa_sga encoder
            ########################
            self.sa_sga = SASGA(hidden_size=enc_dim,
                                  n_head=self.opts.model['n_head'],
                                  ff_size=self.opts.model['ff_dim'],
                                  num_layers=self.opts.model['num_layers'],
                                  dropout=self.opts.model['dropout_sa'])
                                  
        else:
          self.text_enc = SATextEncoder(
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
            num_sa_layers=self.opts.model['num_layers'],
            dropout_sa=self.opts.model['dropout_sa']
          )

        if self.opts.model['decoder_type'] == 'simple':
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

        else: # trgmul decoder will be used
          # attention flatten so that we can feed our image into a trgmul decoder
          self.attflat_img = AttentionFlatten(self.ctx_sizes['image'], self.opts.model['flat_mlp_size'], 1, self.ctx_sizes['image'])  

          # Create Decoder
          self.dec = ConditionalMMDecoderTRGMUL(
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

        txt_enc_res = self.text_enc(batch[self.sl])
        lang_feats = txt_enc_res[0]
        lang_feats_mask = txt_enc_res[1]       
        
        if self.opts.model['encoder_type'] == 'sasga':
          feats = self.imgfeat2hidden(feats)
          # SASGA
          lang_feats, feats = self.sa_sga(lang_feats, feats, lang_feats_mask, feats_mask)


        if self.opts.model['decoder_type'] == 'trgmul':
          # attention flatten if we use TRGMUL as the decoder
          feats = self.attflat_img(feats.transpose(0,1), feats_mask).unsqueeze(0)
        
        return {
            'image': (feats, feats_mask),
            str(self.sl): (lang_feats, lang_feats_mask),
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (1 - torch.cat(self.dec.alphas).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result


