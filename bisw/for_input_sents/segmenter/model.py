#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTMEncoder
from fairseq.modules import FairseqDropout


DEFAULT_MAX_SOURCE_POSITIONS = 1e5


@register_model('bilstm_segmenter')
class BiLSTMSegmenter(FairseqEncoderModel):
    def __init__(self, args, encoder, out_proj):
        super().__init__(encoder)
        self.out_proj = out_proj
        self.dropout_out = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
    @staticmethod
    def add_args(parser):
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=True,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )
        out_proj = Embedding(
            len(task.target_dictionary), args.encoder_hidden_size * 2, task.target_dictionary.pad()
        )

        return cls(args, encoder, out_proj)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Run the forward pass for a BiLSTM segmenter model.

        Feeds a batch of tokens through the encoder to generate boundary tags.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, num_tags)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths, **kwargs)
        logits = self.logits(encoder_out)
        return {
            "encoder_out": logits,
        }

    def logits(self, encoder_out):
        features = encoder_out[0].transpose(0, 1)
        logits = self.dropout_out(features)
        logits = F.linear(logits, self.out_proj.weight)
        return logits


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture("bilstm_segmenter", "bilstm_segmenter")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
