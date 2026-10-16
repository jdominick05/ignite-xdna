# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase splitting must preserve every programmed transfer and existing wait."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import mlp_phases as mp


def traffic(text):
    return [line.strip() for line in text.splitlines() if 'aiex.npu.dma_' in line]


@pytest.mark.parametrize('m',[128,256,384,512])
def test_concatenated_phases_preserve_entire_original_transfer_and_wait_order(m):
    plan=mp.n.c.l.plan(28); original=mp.bm.runtime(plan,m)
    split=[mp.runtime(plan,m,phase) for phase in mp.PHASES]
    assert [line for text in split for line in traffic(text)]==traffic(original)
    assert mp.bm.b.MlpSequence is mp.BaseSequence
    # Each standalone phase must initialize all controls, not inherit another stream's state.
    for text in split:
        assert text.count('index = 2 : ui32')==16
        assert text.count('index = 3 : ui32')==16


def test_invalid_phase_cannot_modify_generator():
    with pytest.raises(ValueError): mp.runtime(mp.n.c.l.plan(28),128,'invalid')
    assert mp.bm.b.MlpSequence is mp.BaseSequence
