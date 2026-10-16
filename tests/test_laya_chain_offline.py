# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repeated-layer ownership, tensor routing and constant-only relocation."""
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import layers as l
b=l.b; p=b.p


def test_all_28_constant_banks_disjoint_and_device_residual_alias():
    plan=l.plan(28); ranges=[]
    for index,bank in enumerate(plan['banks']):
        for name,(start,size) in bank.items(): ranges.append((start,start+size,f'{index}:{name}'))
        view=l.view(plan,index,'local' if index%3 else 'global')
        assert view['offsets']['input']+256*p.QWORDS==bank['qkv'][0]
        assert view['offsets']['norm_input']==plan['offsets']['mlp_norm' if index else 'norm_input']
        assert view['offsets']['norm']==plan['offsets']['norm']
    for name,(start,size) in plan['coefficients'].items(): ranges.append((start,start+size,name))
    ranges.sort()
    assert all(0<=a<z<=plan['words'] for a,z,_ in ranges)
    assert all(a[1]<z[0] for a,z in zip(ranges,ranges[1:]))


def test_repeat_output_every_word_preserves_residual_and_constants():
    plan=l.plan(2); view=l.view(plan,1,'local'); counts=np.zeros(plan['words'],np.uint8)
    text=b.r.runtime(view,False,512,packed_input=True)
    entries=re.findall(r'%o\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]',text)
    for offset,dimensions,strides in entries:
        dims=list(map(int,dimensions.split(','))); stride=list(map(int,strides.split(',')))
        inner=np.arange(dims[3])*stride[3]
        for outer in np.ndindex(*dims[:3]): counts[int(offset)+sum(i*s for i,s in zip(outer,stride))+inner]+=1
    expected=np.zeros_like(counts); off=plan['offsets']; sizes=plan['sizes']
    for field in ('projection','queries','keys','kv','attention','packed_attention','norm'):
        expected[off[field]:off[field]+sizes[field]]=1
    expected[off['packed_norm']:off['packed_norm']+sizes['packed_norm']]=2
    expected[off['mlp_norm']:off['mlp_norm']+sizes['mlp_norm']].reshape(512,p.OUT)[:,:1040]=1
    assert np.array_equal(counts,expected)


def test_repeat_qkv_reads_each_packed_feature_and_core_group():
    plan=l.plan(2); view=l.view(plan,1,'local'); text=b.r.runtime(view,False,512,packed_input=True)
    reads=re.findall(r'%a\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @qL2_(\d+)',text)
    for group in range(12):
        for mg in range(4):
            for col in range(4):
                offset,dims,strides,column=reads[64+(group*4+mg)*4+col]
                assert int(column)==col and list(map(int,dims.split(',')))==[1,16,4,p.QWORDS]
                stride=list(map(int,strides.split(',')))
                for feature in range(16):
                    for core_row in range(4):
                        got=int(offset)+feature*stride[1]+core_row*stride[2]
                        assert got==plan['offsets']['packed_norm']+((feature*4+mg)*4+core_row)*p.OUT


def test_constant_relocation_preserves_activation_output_and_operation_bytes():
    payload=[]
    for arg,word in [(0,110),(0,205),(0,999),(1,110)]:
        words=[129,48,0,0,0,0,0x1d004,0,arg,0,word*4,0]; payload.append(struct.pack('<12I',*words))
    data=struct.pack('<4I',0x06030000,260,4,16+192)+b''.join(payload)
    output,changes,used=l.relocate(data,dict(w=(100,20),gamma=(200,10)),dict(w=(500,20),gamma=(800,10)))
    assert used=={'w','gamma'} and len(changes)==2
    parsed=l.txn.parse(output)
    assert [op['arg_offset']//4 for op in parsed]==[510,805,999,110]
    mutable={byte for change in changes for byte in range(change['value_offset'],change['value_offset']+4)}
    assert all(a==z or i in mutable for i,(a,z) in enumerate(zip(data,output)))
    with pytest.raises(ValueError): l.relocate(data,dict(w=(100,20)),dict(w=(500,21)))
