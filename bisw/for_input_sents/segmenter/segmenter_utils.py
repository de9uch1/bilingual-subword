#!/usr/bin/env python3
# Copyright (c) Hiroyuki Deguchi <deguchi@ai.cs.ehime-u.ac.jp>
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def to_boundary_tag(line: str):
    label_line = ""
    for token in line.split():
        label_line += "B " + ("I " * (len(token) - 1))
    return label_line


def to_character_sequence(line: str):
    return " ".join(c for c in line if c != " ")
