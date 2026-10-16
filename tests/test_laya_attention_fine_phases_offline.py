# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fine boundaries retain all packets and independently initialized controls."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import attention_fine_phases as fine


def traffic(text):
    return [line.strip() for line in text.splitlines() if 'aiex.npu.dma_' in line]


@pytest.mark.parametrize('m',[128,256,384,512])
def test_fine_phase_transfers_match_original_and_coarse_prepare(m):
    plan=fine.coarse.n.c.l.plan(28)
    split=[fine.runtime(plan,m,phase) for phase in fine.PHASES]
    assert [line for text in split for line in traffic(text)]==traffic(fine.coarse.ba.runtime(plan,m))
    assert [line for text in split[:3] for line in traffic(text)]==traffic(fine.coarse.runtime(plan,m,'prepare'))
    for text in split:
        before=text[:text.index('aiex.npu.dma_memcpy_nd')]
        for index in range(7):
            assert before.count(f'index = {index} : ui32')==16
        assert traffic(text)[-1].startswith('aiex.npu.dma_wait')
    assert fine.coarse.ba.r.Sequence is fine.coarse.BaseSequence
    assert fine.coarse.PHASES==('prepare','attend','finish')


def test_invalid_fine_phase_does_not_patch_generator():
    with pytest.raises(ValueError): fine.runtime(fine.coarse.n.c.l.plan(28),128,'missing')
    assert fine.coarse.ba.r.Sequence is fine.coarse.BaseSequence
