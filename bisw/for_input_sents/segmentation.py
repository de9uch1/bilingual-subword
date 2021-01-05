#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Segmentation raw text with a trained model. Batches data on-the-fly.
"""

import fileinput
import logging
import math
import os
import sys
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils

from bisw.for_input_sents.segmenter import segmenter_utils


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("bisw.for_input_sents.segmentation")


Batch = namedtuple("Batch", "ids src_tokens src_lengths src_chars")
Sementation = namedtuple("Segmentation", "src_str hypos pos_scores")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    src_chars = []
    tokens = []
    for src_str in lines:
        src_char = encode_fn(src_str)
        src_chars.append(src_char)
        tokens.append(
            task.source_dictionary.encode_line(
                src_char, add_if_not_exist=False, append_eos=False,
            ).long()
        )

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, src_chars=src_chars,
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        src_chars = batch["src_chars"]

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            src_chars=src_chars,
        )


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    assert (
        not args.batch_size or args.batch_size <= args.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Initialize generator
    generator = task.build_generator(models, args)

    def encode_fn(x):
        x = task.spm.encode(x, out_type=str)
        x = segmenter_utils.to_character_sequence("".join(x))
        return x

    def decode_fn(chars, tags):
        x = ""
        for c, t in zip(chars.replace(" ", ""), tags.replace(" ", "")):
            if t == "B":
                x += " "
            x += c
        return x[1:]

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        logger.info("Sentence buffer size: %s", args.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
                "src_chars": batch.src_chars,
            }
            segmentations = task.inference_step(
                generator, models, sample
            )
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), segmentations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        batch.src_chars[i],
                        hypos,
                    )
                )

        # sort output to match input order
        for id_, src_tokens, src_str, hypos in sorted(results, key=lambda x: x[0]):
            print("S-{}\t{}".format(id_, src_str))

            # Process top predictions
            for hypo in hypos[: min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=None,
                    align_dict=None,
                    tgt_dict=tgt_dict,
                )
                detok_hypo_str = decode_fn(src_str, hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print("H-{}\t{}\t{}".format(id_, score, hypo_str))
                # detokenized hypothesis
                print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
                print(
                    "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                )

        # update running id_ counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
