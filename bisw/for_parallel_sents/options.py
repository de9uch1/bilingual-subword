#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def add_common_args(parser):
    group = parser.add_argument_group("Common")
    group.add_argument("--source-lang", "-s", required=True,
                       help="source language")
    group.add_argument("--target-lang", "-t", required=True,
                       help="target language")
    group.add_argument("--trainpref", required=True,
                       help="train file prefix (parallel corpus)")
    group.add_argument("--spm-dir", required=True,
                       help="learned sentencepiece model directory")
    group.add_argument("--train-spm", action="store_true",
                       help="Train sentencepiece models")
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--num-workers", type=int, default=8,
                       help="number of parallel workers")
    group.add_argument("--batch-size", type=int, default=100000,
                       help="number of examples in batch")

    return group


def add_sentencepiece_trainer_args(parser):
    group = parser.add_argument_group("SentencePiece Trainer")
    group.add_argument("--src-vocab-size", type=int, default=8000,
                       help="number of vocabulary size")
    group.add_argument("--tgt-vocab-size", type=int, default=8000,
                       help="number of vocabulary size")
    group.add_argument("--src-character-coverage", type=float, default=0.9995,
                       help="amount of characters covered by the model, good defaults are: "
                       "0.9995 for languages with rich character set like Japanse or Chinese "
                       "and 1.0 for other languages with small character set.")
    group.add_argument("--tgt-character-coverage", type=float, default=0.9995,
                       help="amount of characters covered by the model, good defaults are: "
                       "0.9995 for languages with rich character set like Japanse or Chinese "
                       "and 1.0 for other languages with small character set.")
    group.add_argument("--src-byte-fallback", action="store_true",
                       help="byte code fallback")
    group.add_argument("--tgt-byte-fallback", action="store_true",
                       help="byte code fallback")

    return group


def add_bisw_args(parser):
    group = parser.add_argument_group("Bilingual Subword Segmentation")
    group.add_argument("--use-src-best", action="store_true",
                       help="if set, always use the unigram LM's best for the source-side "
                       "to avoid using a bilingual subword segmenter")
    group.add_argument("--nbest-size", "-n", type=int, default=5,
                       help="N best size for bilingual subword searching")


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_sentencepiece_trainer_args(parser)
    add_bisw_args(parser)

    args = parser.parse_args()
    return args
