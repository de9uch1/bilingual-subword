#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Bilingual Subword Segmantation for Training Data
"""

import os
import sys

from sentencepiece import SentencePieceProcessor

from bisw.for_parallel_sents import (
    options,
    segmenter,
    sentencepiece_trainer,
)


HLINE_LENGTH = 70


def print_stderr(msg):
    print(msg, file=sys.stderr, flush=True)


def print_hline():
    print_stderr("| " + ("-" * HLINE_LENGTH))


def abort():
    print("| Abort.", file=sys.stderr, flush=True)
    exit(1)


def main(args):

    def spm_path(lang=""):
        return os.path.join(args.spm_dir, lang + ".model")

    def train_file(lang=""):
        return "{}.{}".format(args.trainpref, lang)

    print_stderr(args)

    src_lang, tgt_lang = args.source_lang, args.target_lang

    src_spm_path = spm_path(src_lang)
    tgt_spm_path = spm_path(tgt_lang)

    src_spm_exists = os.path.exists(src_spm_path)
    tgt_spm_exists = os.path.exists(tgt_spm_path)

    if args.train_spm:
        print_hline()
        print_stderr("| Start training SentencePiece model...")
        if args.joined_dictionary:
            sentencepiece_trainer.train(
                args,
                [train_file(src_lang), train_file(tgt_lang)],
                src_lang,
            )
            while True:
                try:
                    os.symlink(os.path.basename(src_spm_path), tgt_spm_path)
                    break
                except FileExistsError:
                    os.remove(tgt_spm_path)
        else:
            sentencepiece_trainer.train(args, [train_file(src_lang)], src_lang)
            sentencepiece_trainer.train(args, [train_file(tgt_lang)], tgt_lang, tgt=True)
        print_stderr("\n| Done.")
    else:
        if not src_spm_exists:
            print_stderr("| SentencePiece model not found: '{}'".format(src_spm_path))
        if not tgt_spm_exists:
            print_stderr("| SentencePiece model not found: '{}'".format(tgt_spm_path))
        if not all([src_spm_exists, tgt_spm_exists]):
            print_stderr("| If you set '--train-spm', train the sentencepiece models.")
            abort()

    print_hline()
    print_stderr("| ")
    src_spm = SentencePieceProcessor(src_spm_path)
    print_stderr("| Loaded SentencePiece model: '{}'".format(src_spm_path))
    if args.joined_dictionary:
        tgt_spm = src_spm
    else:
        tgt_spm = SentencePieceProcessor(tgt_spm_path)
    print_stderr("| Loaded SentencePiece model: '{}'".format(tgt_spm_path))

    with \
      open(train_file(args.source_lang), "r") as src_lines, \
      open(train_file(args.target_lang), "r") as tgt_lines \
    :
        segmenter.tokenize_lines(args, src_lines, tgt_lines, src_spm, tgt_spm)


def cli_main():
    args = options.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
