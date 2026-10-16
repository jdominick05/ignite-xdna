# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase diagnostics preserve transfer packets and initialize controls independently."""
from pathlib import Path
import json
import struct
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import attention_phases as ap


def traffic(text):
    return [line.strip() for line in text.splitlines() if 'aiex.npu.dma_' in line]


@pytest.mark.parametrize('m',[128,256,384,512])
def test_concatenated_phase_transfers_and_waits_are_unchanged(m):
    plan=ap.n.c.l.plan(28)
    original=ap.ba.runtime(plan,m)
    split=[ap.runtime(plan,m,phase) for phase in ap.PHASES]
    assert [line for text in split for line in traffic(text)]==traffic(original)
    assert ap.ba.r.Sequence is ap.BaseSequence
    for text in split:
        assert text.count('index = 2 : ui32')==16
        assert text.count('index = 3 : ui32')==16
        first_dma=text.index('aiex.npu.dma_memcpy_nd')
        for index in range(7):
            assert f'index = {index} : ui32' in text[:first_dma]
        assert traffic(text)[-1].startswith('aiex.npu.dma_wait')


def test_invalid_phase_does_not_replace_original_generator():
    with pytest.raises(ValueError):
        ap.runtime(ap.n.c.l.plan(28),128,'missing')
    assert ap.ba.r.Sequence is ap.BaseSequence


@pytest.mark.parametrize('mutation',['descriptor','control'])
def test_compiled_signature_detects_transfer_and_control_corruption(mutation):
    candidate=Path.cwd()/'scratch/laya_transformer/attention_fifo_v1_double_m128'
    if not (candidate/'build.json').exists():
        pytest.skip('qualified image unavailable; offline artifact check only')
    audits=json.loads((candidate/'build.json').read_text())['audits']
    data=(candidate/'insts.bin').read_bytes()
    original=ap.execution_signature(data,audits)
    changed=bytearray(data)
    if mutation=='descriptor':
        op=next(op for op in ap.n.c.txn.parse(data) if op['op']=='DDR_PATCH')
        # The operand relocation is part of the transport identity.
        offset=op['offset']+op['size']-4
    else:
        addresses=ap.n.c.txn.registers(audits)
        op=next(op for op in ap.n.c.txn.parse(data) if addresses.get(op.get('addr'),(None,None))[1]=='valid')
        offset=op['offset']+16
    struct.pack_into('<I',changed,offset,struct.unpack_from('<I',changed,offset)[0]^1)
    assert ap.execution_signature(bytes(changed),audits)!=original
