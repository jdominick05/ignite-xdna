# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dedicated MLP dispatch retains its live operations and rounding fence."""
from collections import Counter
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import geglu_tune as gt


def test_dedicated_mlp_retains_live_arithmetic_calls():
    original=gt.bm.b.mlp_prefix(); candidate=gt.prefix()
    calls=lambda text:Counter(line.strip() for line in text.splitlines() if 'func.call' in line and '@attention_shared(' not in line)
    assert calls(original)==calls(candidate)
    for name in ('chain_geglu','chain_gemm_step','command_norm','chain_finish'):
        assert candidate.count('func.call @'+name+'(')==16
    assert candidate.count('%is_attention = arith.constant false')==16
    assert candidate.count('stack_size = 5376 : i32')==16
    assert not re.search(r'%(?:scratch|mask)\d+',candidate)
    assert not any('func.call @'+name in candidate for name in ('stem_','vector_'))


def test_dedicated_mlp_keeps_original_fifo_depths():
    original=gt.bm.b.mlp_prefix(); candidate=gt.prefix()
    fifo=lambda text:[line for line in text.splitlines() if 'aie.objectfifo ' in line]
    assert fifo(candidate)==fifo(original)
