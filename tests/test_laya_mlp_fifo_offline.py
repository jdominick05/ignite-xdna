# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Buffer candidates preserve all live arithmetic calls and explicit FIFO counts."""
from collections import Counter
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import mlp_fifo as mf


@pytest.mark.parametrize('q,kv,l2',[(1,1,1),(2,1,2),(1,2,1),(2,2,2)])
def test_pruned_prefix_preserves_live_calls_and_sets_requested_depths(q,kv,l2):
    original=mf.n.vector.prefix(); candidate=mf.prefix(q,kv,l2)
    calls=lambda text:Counter(line.strip() for line in text.splitlines() if 'func.call' in line and '@attention_shared(' not in line)
    assert calls(candidate)==calls(original)
    assert candidate.count('func.call @attention_shared(')==0
    assert not re.search(r'%(?:scratch|mask)\d+',candidate)
    assert original.count('sym_name = "attn_scratch_')==16
    for family,depth,count in [('qF',q,16),('kvF',kv,4),('kvL2',kv,4),('qL2',l2,4)]:
        assert len(re.findall(r'aie.objectfifo @'+family+r'_\d+\([^\n]*,'+str(depth)+r' : i32\)',candidate))==count


def test_invalid_depth_does_not_generate_a_candidate():
    for q,kv,l2 in ((0,1,1),(1,3,1),(1,1,-1)):
        with pytest.raises(ValueError): mf.prefix(q,kv,l2)
