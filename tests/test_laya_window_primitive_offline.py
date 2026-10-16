# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared operands and independent outputs must survive primitive lowering."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import window_lookup_primitive as primitive


def test_shared_query_and_k_major_weights_preserve_both_outputs():
    new=primitive.events(True);old=[primitive.events(False,slot) for slot in range(2)]
    query=primitive.walk.addresses(new[0]).reshape(2,4,1040)
    weights=primitive.walk.addresses(new[2]).reshape(2,2,2064)
    outputs=primitive.walk.addresses(new[1]).reshape(2,4*2080)
    for slot in range(2):
        original_q=primitive.walk.addresses(old[slot][0]).reshape(2,4,2064)
        assert np.array_equal(query,original_q[:,:,:1040])
        assert np.array_equal(weights[:,slot],primitive.walk.addresses(old[slot][2]).reshape(2,2064))
        assert np.array_equal(outputs[slot],primitive.walk.addresses(old[slot][1]))
    assert not np.intersect1d(outputs[0],outputs[1]).size
    for event in new:
        words=primitive.walk.addresses(event)
        assert words.min()>=16 and words.max()<primitive.plan()['words']


def test_decode_packed_fixture_and_recompute_independent_integer_products():
    payload,golden=primitive.fixture(11)
    qwords=2*4*2064
    qbits=payload[:qwords].reshape(2,4,2064)[:,:,16:1040].copy().view(np.uint16)
    qbits=qbits.reshape(1,2,4,8,8,4,8).transpose(0,1,2,3,5,4,6).reshape(1,2,4,32,64)
    qbits=qbits.transpose(0,2,3,1,4).reshape(128,128)
    wbits=payload[qwords:].reshape(4,2,2,2064)[:,:,:,:2048].copy().view(np.uint16)
    wbits=wbits.reshape(4,2,2,8,16,8,4).transpose(0,1,2,3,5,4,6).reshape(4,2,2,64,64)
    wbits=wbits.transpose(2,3,1,0,4).reshape(128,512)
    a=(qbits.astype(np.uint32)<<16).view(np.float32)
    b=(wbits.astype(np.uint32)<<16).view(np.float32)
    expected=np.stack([(a@b[:,first:first+64]).reshape(4,2048) for first in (0,256)])
    assert np.array_equal(golden,expected)


@pytest.mark.parametrize('mutation',['width','length','repeat'])
def test_compiled_primitive_gate_rejects_corrupted_controls_or_packets(mutation):
    root=Path.cwd()/'scratch/laya_transformer/window_lookup_primitive_v1'
    if not (root/'build.json').exists():pytest.skip('offline compiled primitive unavailable')
    meta=json.loads((root/'build.json').read_text());audits=meta['layouts']['window']['audits']
    raw=(root/'window/insts.bin').read_bytes()
    primitive.verify_stream(raw,True,0,audits)
    changed=bytearray(raw);ops=primitive.n.c.txn.parse(raw)
    if mutation=='width':
        audit=next(row for row in audits if row['core']==[0,2])
        start=next(start for start,end,name in audit['spans'] if name=='chain_control_0')
        op=next(op for op in ops if op['op']=='WRITE' and op['addr']==(2<<20)+start+24)
        offset=op['offset']+16;bit=1
    elif mutation=='length':
        op=next(op for op in ops if op['op']=='BLOCKWRITE' and op['addr']==0x1d060)
        offset=op['offset']+16;bit=1
    else:
        op=next(op for op in ops if op['op']=='WRITE' and op['addr']==0x1d214)
        offset=op['offset']+16;bit=1<<16
    struct.pack_into('<I',changed,offset,struct.unpack_from('<I',changed,offset)[0]^bit)
    with pytest.raises(ValueError):primitive.verify_stream(bytes(changed),True,0,audits)


def test_only_active_column_has_a_completion_wait():
    for window,slot in [(True,0),(False,0),(False,1)]:
        text=primitive.runtime(window,slot)
        assert text.count('dma_wait')==1
        assert 'symbol = @outL2_0' in text
