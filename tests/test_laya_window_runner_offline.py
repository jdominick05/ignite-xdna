# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Primitive expectation must cover payload, full packet poison, and outside guards."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/name) for name in ('tools','kernels/laya_transformer')]
import laya_window_primitive as runner


@pytest.mark.parametrize('window,slot',[(False,0),(False,1),(True,0)])
def test_complete_output_packet_and_unwritten_regions(window,slot):
    primitive=runner.primitive;plan=primitive.plan();poison=0xdeadbeef
    _,golden=primitive.fixture(11)
    expected,designated=runner.expected_output(window,slot,golden,poison)
    assert expected.shape==designated.shape==(plan['words'],)
    assert np.all(expected[~designated]==poison)
    expected_slots=[0,1] if window else [slot]
    assert designated.sum()==len(expected_slots)*4*2080
    for index in expected_slots:
        begin=plan['output']+index*4*2080
        packets=expected[begin:begin+4*2080].reshape(4,2080)
        assert np.array_equal(packets[:,16:2064].copy().view(np.float32),golden[index])
        sentinel=primitive.candidate.window.bm.b.r.mlp.POISON
        assert np.all(packets[:,:16]==sentinel)
        assert np.all(packets[:,2064:]==sentinel)
