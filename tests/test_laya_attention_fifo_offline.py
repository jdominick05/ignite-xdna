# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attention FIFO variants retain the entire live arithmetic/control body."""
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import attention_fifo as af


@pytest.mark.parametrize('q,kv,l2',[(1,1,1),(2,1,2),(1,2,1),(2,2,2)])
def test_only_fifo_depth_declarations_change(q,kv,l2):
    original=af.ba.prefix(); candidate=af.prefix(q,kv,l2)
    strip=lambda text:[line for line in text.splitlines() if not re.search(r'aie.objectfifo @(?:qF|qL2|kvF|kvL2)_\d+\(',line)]
    assert strip(candidate)==strip(original)
    for family,depth,count in [('qF',q,16),('kvF',kv,4),('kvL2',kv,4),('qL2',l2,4)]:
        assert len(re.findall(r'aie.objectfifo @'+family+r'_\d+\([^\n]*,'+str(depth)+r' : i32\)',candidate))==count
    if (q,kv,l2)==(1,1,1): assert candidate==original


def test_reject_invalid_depths_before_compilation():
    for depths in ((0,1,1),(1,3,1),(1,1,-1),(True,1,1),(1,1.,1)):
        with pytest.raises(ValueError): af.prefix(*depths)
