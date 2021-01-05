#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from sentencepiece import SentencePieceTrainer


def train(args, inputs, lang, tgt=False):

    spm_dir = args.spm_dir
    if not os.path.exists(spm_dir):
        os.makedirs(spm_dir)

    train_config = {
        k: getattr(args, ("tgt" if tgt else "src") + "_" + k)
        for k in [
            "vocab_size",
            "character_coverage",
            "byte_fallback",
        ]
     }

    SentencePieceTrainer.train(
        input=inputs,
        model_prefix=os.path.join(args.spm_dir, lang),
        **train_config,
    )
