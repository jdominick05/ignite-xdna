# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Head transfer ownership and typed address patching, without a device."""
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import head_pre as h
import vector as v
import vector_probe as vp
p=v.p


def test_head_prelude_writes_every_owned_word_and_preserves_encoder_residual():
    base=v.b.plan(); pl=h.plan(base)
    assert base==v.b.plan()  # the encoder plan is not mutated
    names=('norm_input','mlp_norm','packed_norm','projection')
    writes={name:np.zeros(pl['sizes'][name],np.uint8) for name in names}
    entries=re.findall(r'dma_memcpy_nd\(%([ao])\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]\) \{id = (\d+).*metadata = @(q|kv|out)L2_(\d+)',h.runtime(pl))
    assert entries
    for var,start,dims,strides,bid,kind,col in entries:
        start=int(start); dims=list(map(int,dims.split(','))); strides=list(map(int,strides.split(',')))
        last=start+sum((count-1)*stride for count,stride in zip(dims,strides))
        assert 0<=start<=last<pl['words']
        if kind!='out': continue
        assert var=='o' and int(bid)==2
        name=next(n for n in names if pl['offsets'][n]<=start<pl['offsets'][n]+pl['sizes'][n])
        dest=writes[name]; relative=start-pl['offsets'][name]
        for index in np.ndindex(*dims[:3]):
            positions=relative+sum(i*s for i,s in zip(index,strides))+np.arange(dims[3])*strides[3]
            assert positions.max()<dest.size
            dest[positions]+=1
    assert all(np.all(writes[n]==(2 if n=='projection' else 1)) for n in names)


def stream(reads):
    body=b''.join(struct.pack('<12I',129,48,0,0,0,0,0x1d004,0,arg,0,offset,0) for arg,offset in reads)
    return struct.pack('<4I',0x06030000,260,len(reads),len(body)+16)+body


@pytest.mark.parametrize('qtype',[0,1,2])
def test_type_patch_changes_only_four_constant_addresses(qtype):
    pl=h.plan(v.b.plan()); start=pl['offsets']['head_type']*4
    raw=stream([(0,start)]*4+[(1,start),(0,start+4),(0,64)])
    updated=h.prepare(raw,pl,qtype); ops=v.s.txn.parse(updated)
    assert [op['arg_offset'] for op in ops]==[start+qtype*p.KVWORDS*4]*4+[start,start+4,64]
    allowed={j for op in v.s.txn.parse(raw)[:4] for j in range(op['offset']+40,op['offset']+44)}
    assert all(a==b or i in allowed for i,(a,b) in enumerate(zip(raw,updated)))


@pytest.mark.parametrize('qtype',[-1,3,1.5])
def test_type_patch_rejects_invalid_type(qtype):
    with pytest.raises(ValueError): h.prepare(stream([]),h.plan(v.b.plan()),qtype)


def test_type_patch_rejects_missing_transfer():
    pl=h.plan(v.b.plan()); start=pl['offsets']['head_type']*4
    with pytest.raises(ValueError): h.prepare(stream([(0,start)]*3),pl,1)


def test_vector_dispatch_disables_attention_and_suppresses_extra_outputs():
    text=v.prefix()
    assert text.count('%is_attention = arith.constant false')==16
    assert text.count('%not_geglu = arith.cmpi ult, %command,%three : i32')==16
    assert text.count('%non_attention_count = arith.select %is_norm,%c1,%non_norm_count : index')==16
    assert text.count('stack_size = 5376 : i32')==16
    with pytest.raises(ValueError): v.Sequence(100).control(0)
    with pytest.raises(ValueError): v.Sequence(100).control(4)


def test_probe_pack_matches_independent_scalar_four_by_eight_layout():
    values=np.arange(2048,dtype=np.float32).reshape(32,64)/1024
    actual=vp.packed(values).view(np.uint16)
    for row in range(32):
        for col in range(64):
            bits=int(values[row,col].view(np.uint32)); expected=((bits+0x7fff+((bits>>16)&1))>>16)&0xffff
            position=(row//4)*256+(col//8)*32+(row%4)*8+col%8
            assert actual[position]==expected
