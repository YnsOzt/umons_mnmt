# -*- coding: utf-8 -*-
import logging

import torch
import torch.nn as nn

from ..layers import BiLSTMp, ConditionalDecoder, FF
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


# ASR with ESPNet style BiLSTMp encoder


class ASR(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'feat_dim': 43,                 # Speech features dimensionality
            'emb_dim': 300,                 # Decoder embedding dim
            'enc_dim': 320,                 # Encoder hidden size
            'enc_layers': '1_1_2_2_1_1',    # layer configuration
            'dec_dim': 320,                 # Decoder hidden size
            'proj_dim': 300,                # Intra-LSTM projection layer
            'proj_activ': 'tanh',           # Intra-LSTM projection activation
            'dec_type': 'gru',              # Decoder type (gru|lstm)
            'dec_init': 'mean_ctx',         # How to initialize decoder
                                            # (zero/mean_ctx/feats)
            'dec_init_size': None,          # feature vector dimensionality for
                                            # dec_init == 'feats'
            'dec_init_activ': 'tanh',       # Decoder initialization activation func
            'att_type': 'mlp',              # Attention type (mlp|dot)
            'att_temp': 1.,                 # Attention temperature
            'att_activ': 'tanh',            # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,          # Enables bias in attention mechanism
            'att_bottleneck': 'hid',        # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,      # Transform annotations before attention
            'dropout': 0,                   # Generic dropout overall the architecture
            'tied_dec_embs': False,         # Share decoder embeddings
            'max_len': None,                # Reject samples if len('bucket_by') > max_len
            'bucket_by': None,              # A key like 'en' to define w.r.t
                                            # which dataset batches will be sorted
            'bucket_order': None,           # Can be 'ascending' or 'descending'
                                            # for curriculum learning
            'direction': None,              # Network directionality, i.e. en->de
            'lstm_forget_bias': False,      # Initialize forget gate bias to 1 for LSTM
            'lstm_bias_zero': False,        # Use zero biases for LSTM
            'adaptation': False,            # Enable/disable AM adaptation
            'adaptation_type': 'early',     # Early: shift feats, late: shift encodings
            'adaptation_dim': None,         # Input dim for auxiliary feat vectors
            'adaptation_activ': None,       # Non-linearity for adaptation FF
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        # Inherently non multi-lingual aware
        self.src = self.topology.first_src

        self.tl = self.topology.first_trg
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)

        # Context size is enc_dim because of proj layers
        self.ctx_sizes = {str(self.src): self.opts.model['enc_dim']}

        # Need to be set for early-stop evaluation
        # NOTE: This should come from config or elsewhere
        self.val_refs = self.opts.data['val_set'][self.tl]

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        # Use kaiming_normal for everything as it is a sane default
        # Do not touch biases for now
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

        if self.opts.model['lstm_bias_zero'] or \
                self.opts.model['lstm_forget_bias']:
            for name, param in self.speech_enc.named_parameters():
                if 'bias_hh' in name or 'bias_ih' in name:
                    # Reset bias to 0
                    param.data.fill_(0.0)
                    if self.opts.model['lstm_forget_bias']:
                        # Reset forget gate bias of LSTMs to 1
                        # the tensor organized as: inp,forg,cell,out
                        n = param.numel()
                        param[n // 4: n // 2].data.fill_(1.0)

    def setup(self, is_train=True):
        self.speech_enc = BiLSTMp(
            input_size=self.opts.model['feat_dim'],
            hidden_size=self.opts.model['enc_dim'],
            proj_size=self.opts.model['proj_dim'],
            proj_activ=self.opts.model['proj_activ'],
            dropout=self.opts.model['dropout'],
            layers=self.opts.model['enc_layers'])

        ################
        # Create Decoder
        ################
        self.dec = ConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.src),
            tied_emb=self.opts.model['tied_dec_embs'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout'])

        if self.opts.model['adaptation']:
            if self.opts.model['adaptation_type'] == 'late':
                out_dim = self.opts.model['enc_dim']
            else:
                out_dim = self.opts.model['feat_dim']
            self.vis_proj = FF(self.opts.model['adaptation_dim'],
                               out_dim,
                               activ=self.opts.model['adaptation_activ'],
                               bias=False)

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'])
        logger.info(dataset)
        return dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def encode(self, batch, **kwargs):
        if self.opts.model['adaptation']:
            if self.opts.model['adaptation_type'] == 'self':
                dynamic_shift = self.vis_proj(batch[self.src].mean(0))
                speech_enc = self.speech_enc(batch[self.src] + dynamic_shift)
            else:
                dynamic_shift = self.vis_proj(batch['feats'])
                if self.opts.model['adaptation_type'] == 'early':
                    speech_enc = self.speech_enc(batch[self.src] + dynamic_shift)
                elif self.opts.model['adaptation_type'] == 'late':
                    ctx, mask = self.speech_enc(batch[self.src])
                    ctx.add_(dynamic_shift)
                    speech_enc = (ctx, mask)
        else:
            speech_enc = self.speech_enc(batch[self.src])

        d = {str(self.src): speech_enc}
        if self.opts.model['dec_init'] == 'feats':
            d['feats'] = (batch['feats'], None)
        return d

    def forward(self, batch, **kwargs):
        # Get loss dict
        result = self.dec(self.encode(batch), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in data_loader:
            batch.to_gpu(volatile=True)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    def get_decoder(self, task_id=None):
        """Compatibility function for multi-tasking architectures."""
        return self.dec
