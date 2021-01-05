#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch


class SequenceReranker(object):
    """Scores the target and return the best for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        symbols_to_strip_from_output=None,
        **kwargs,
    ):
        self.eos = tgt_dict.eos()

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        # compute scores for each model in the ensemble
        avg_lprobs = None
        for model in models:
            model.eval()
            net_output = model(**net_input)

            lprobs = model.get_normalized_probs(
                net_output,
                log_probs=True,
            ).float().cpu()

            if avg_lprobs is None:
                avg_lprobs = lprobs
            else:
                avg_lprobs.add_(lprobs)
        if len(models) > 1:
            avg_lprobs.div_(len(models))

        hypos = []
        bsz = len(sample["candidates"])
        for i in range(bsz):
            # remove padding from ref
            cand_i = torch.stack(sample["candidates"][i]).long()
            cand_size, seq_len = cand_i.size()
            token_indices = torch.arange(seq_len)

            avg_lprobs_i = avg_lprobs[i]
            pos_scores_i = avg_lprobs_i[token_indices, cand_i]
            cand_scores_i = pos_scores_i.sum(dim=-1) / seq_len

            max_score_i, max_cand_i = torch.max(cand_scores_i, dim=-1)

            hypos.append(
                [
                    {
                        "tokens": cand_i[max_cand_i],
                        "score": max_score_i,
                        "positional_scores": pos_scores_i[max_cand_i],
                        "alignment": None,
                        "attention": None,
                    }
                ]
            )
        return hypos
