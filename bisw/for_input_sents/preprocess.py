#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq import options
from fairseq_cli import preprocess

from bisw.for_input_sents.segmenter import segmenter_utils


def main(args):

    os.makedirs(args.destdir, exist_ok=True)

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, suffix):
        return os.path.join(args.destdir, file_name(prefix, suffix))

    def dest_prefix(prefix):
        return os.path.join(args.destdir, prefix)

    def prepare_label(input_prefix, output_prefix):
        with open(file_name(input_prefix, args.source_lang), "r", encoding="utf-8") as src_file:
            with open(dest_path(output_prefix, "char"), "w", encoding="utf-8") as char_file:
                with open(dest_path(output_prefix, "label"), "w", encoding="utf-8") as label_file:
                    for line in src_file:
                        char_file.write(segmenter_utils.to_character_sequence(line))
                        label_file.write(segmenter_utils.to_boundary_tag(line) + "\n")

    def prepare_dataset():
        if args.trainpref:
            prepare_label(args.trainpref, "train")
        if args.validpref:
            prepare_label(args.validpref, "valid")
        if args.testpref:
            prepare_label(args.testpref, "test")

    def override_args():
        args.source_lang = "char"
        args.target_lang = "label"
        args.only_source = False
        args.joined_dictionary = False

        if args.trainpref:
            args.trainpref = dest_prefix("train")
        if args.validpref:
            args.validpref = dest_prefix("valid")
        if args.testpref:
            args.testpref = dest_prefix("test")

    prepare_dataset()
    override_args()
    preprocess.main(args)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
