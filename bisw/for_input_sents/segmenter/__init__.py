#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .dataset import SegmentationDataset
from bisw.for_input_sents.segmenter import (
    model,
    task,
)
