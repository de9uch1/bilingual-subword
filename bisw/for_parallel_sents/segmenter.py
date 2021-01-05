#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from multiprocessing.pool import Pool

import numpy as np


def argmin_diff_ntokens(nbest, ntokens):
    return np.argmin(
        np.abs(
            np.array([len(seq) for seq in nbest]) - ntokens
        )
    )


def search_bilingual_segmentation_pair(src_nbest, tgt_nbest, use_src_best=False):
    src_best_length = len(src_nbest[0])
    tgt_best_length = len(tgt_nbest[0])

    if use_src_best or src_best_length > tgt_best_length:
        src_idx = 0
        tgt_idx = argmin_diff_ntokens(tgt_nbest, src_best_length)
    else:
        src_idx = argmin_diff_ntokens(src_nbest, tgt_best_length)
        tgt_idx = 0

    return (src_nbest[src_idx], tgt_nbest[tgt_idx])


def bilingual_segmentation(
    batch,
    src_spm, tgt_spm,
    nbest_size,
    use_src_best,
):

    def sample_nbest(line, spm):
        return spm.nbest_encode_as_pieces(line.strip(), nbest_size)

    return [
        search_bilingual_segmentation_pair(
            sample_nbest(src_line, src_spm),
            sample_nbest(tgt_line, tgt_spm),
            use_src_best=use_src_best,
        )
        for (src_line, tgt_line) in batch
    ]


def merge_result(workers, print=print):
    for worker_result in workers:
        for src_line, tgt_line in worker_result.get():
             print("{}\t{}".format(" ".join(src_line), " ".join(tgt_line)))


def parallel_exec(
    func, func_args,
    examples,
    batch_size=100,
    num_workers=8,
):
    workers = []
    batch = []

    def exec_one(proc, batch):
        return proc.apply_async(func, args=(batch, *func_args))

    with Pool(processes=num_workers) as pool:
        for i, example in enumerate(examples, start=1):
            batch.append(example)
            if i % 100000 == 0:
                print("{}...".format(i), end="", file=sys.stderr, flush=True)
            if i % batch_size == 0:
                workers.append(exec_one(pool, batch))
                batch = []
            if len(workers) >= num_workers:
                merge_result(workers)
                workers = []
        workers.append(exec_one(pool, batch))
        merge_result(workers)
        print("Done.", file=sys.stderr, flush=True)


def tokenize_lines(args, src_lines, tgt_lines, src_spm, tgt_spm):
    return parallel_exec(
        func=bilingual_segmentation,
        func_args=(
            src_spm, tgt_spm,
            args.nbest_size,
            args.use_src_best,
        ),
        examples=zip(src_lines, tgt_lines),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
