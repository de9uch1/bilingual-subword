#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os

import numpy as np
import torch
from fairseq import metrics, options, utils
from fairseq.data import (
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.tasks import register_task, FairseqTask
from fairseq.tasks.translation import TranslationTask

from bisw.for_input_sents.segmenter import segmenter_utils
from bisw.for_input_sents.segmenter.dataset import SegmentationDataset


logger = logging.getLogger(__name__)


def load_segmentation_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    dataset_impl,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    src_dataset = src_datasets[0]
    tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return SegmentationDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("bisw_segmentation")
class BiSWSegmentationTask(TranslationTask):
    """
    Bilingual subword segmentation for input (source only) sentences.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The bisw_segmentation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

        spm = None
        if args.spm_model:
            from sentencepiece import SentencePieceProcessor
            spm = SentencePieceProcessor(model_file=args.spm_model)

        self.spm = spm
        self.spm_nbest = args.spm_nbest

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("data", help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories")
        parser.add_argument("--left-pad-source", default="True", type=str, metavar="BOOL",
                            help="pad the source on the left")
        parser.add_argument("--left-pad-target", default="False", type=str, metavar="BOOL",
                            help="pad the target on the left")
        parser.add_argument("--max-source-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the source sequence")
        parser.add_argument("--max-target-positions", default=1024, type=int, metavar="N",
                            help="max number of tokens in the target sequence")

        parser.add_argument("--spm-model", type=str,
                            help="sentencepiece model file")
        parser.add_argument("--spm-nbest", type=int, default=1,
                            help="number of sentencepiece N best candidates")
        # fmt: on

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # set language pair
        args.source_lang = "char"
        args.target_lang = "label"

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_segmentation_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            dataset_impl=self.args.dataset_impl,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        from bisw.for_input_sents.segmenter.sequence_reranker import SequenceReranker
        return SequenceReranker(self.target_dictionary)

    def valid_step(self, sample, model, criterion):
        return FairseqTask.valid_step(self, sample, model, criterion)

    def build_dataset_for_inference(self, src_tokens, src_lengths, src_chars=None):
        return SegmentationDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            src_chars=src_chars,
            tgt_dict=self.target_dictionary,
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if self.spm is None:
            raise Exception("Please specify the '--spm-model' option.")

        sample["candidates"] = [
            [
                self.target_dictionary.encode_line(
                    segmenter_utils.to_boundary_tag(" ".join(cand)),
                    append_eos=False,
                )
                for cand in self.spm.nbest_encode_as_pieces(
                    self.spm.decode(s.split()),
                    self.spm_nbest
                )
            ]
            for s in sample["src_chars"]
        ]

        with torch.no_grad():
             return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def reduce_metrics(self, logging_outputs, criterion):
        return FairseqTask.reduce_metrics(self, logging_outputs, criterion)
