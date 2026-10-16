# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The combined compiler/FIFO experiment changes only declared FIFO depths."""
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import geglu_fifo as gf


def test_combination_retains_entire_dedicated_program_except_fifo_depths():
    original=gf.BASE_PREFIX(); candidate=gf.prefix()
    normalize=lambda text:re.sub(r'(aie.objectfifo @\w+\([^\n]*),[12] : i32\)',r'\1,DEPTH : i32)',text)
    assert normalize(candidate)==normalize(original)
    for family,count in [('qF',16),('qL2',4),('kvF',4),('kvL2',4)]:
        assert len(re.findall(r'aie.objectfifo @'+family+r'_\d+\([^\n]*,2 : i32\)',candidate))==count
