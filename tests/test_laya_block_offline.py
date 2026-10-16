# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-image arena ownership and MLP operand routing, without a device."""
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]/'kernels/laya_transformer'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('laya_block',HERE/'block.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
p=b.p


def test_mlp_writes_only_owned_outputs_and_preserves_attention_residual():
    pl=b.plan(); counts=np.zeros(pl['words'],np.uint8)
    entries=re.findall(r'%o\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]',b.mlp_runtime(pl))
    for offset,dimensions,strides in entries:
        dims=list(map(int,dimensions.split(','))); strides=list(map(int,strides.split(',')))
        inner=np.arange(dims[3])*strides[3]
        for outer in np.ndindex(*dims[:3]):
            start=int(offset)+sum(i*s for i,s in zip(outer,strides))
            counts[start+inner]+=1
    expected=np.zeros(pl['words'],np.uint8)
    for field in ('mlp_wi','mlp_geglu','mlp_norm'):
        start=pl['offsets'][field]; expected[start:start+pl['sizes'][field]]=1
    start=pl['offsets']['norm']
    expected[start:start+pl['sizes']['norm']].reshape(512,p.OUT)[:,:1040]=1
    assert np.array_equal(counts,expected)


def test_mlp_input_and_geglu_every_feature_tile_address():
    pl=b.plan(); entries=re.findall(r'%a\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @qL2_(\d+)',b.mlp_runtime(pl))
    for group in range(21):
        for mg in range(4):
            for col in range(4):
                offset,dims,strides,found=entries.pop(0)
                assert int(offset)==pl['offsets']['packed_norm']+mg*4*p.OUT and int(found)==col
                assert list(map(int,dims.split(',')))==[1,16,4,p.QWORDS]
                assert list(map(int,strides.split(',')))==[0,16*p.OUT,p.OUT,1]
    for feature in range(41):
        offset,_,_,found=entries.pop(0)
        assert int(offset)==pl['offsets']['mlp_wi']+((feature%4)*21+feature//4)*16*p.OUT
        assert int(found)==feature%4
    for group in range(4):
        for mg in range(4):
            for col in range(4):
                offset,dims,strides,found=entries.pop(0)
                assert int(offset)==pl['offsets']['mlp_geglu']+mg*4*p.OUT and int(found)==col
                assert list(map(int,dims.split(',')))==[1,41,4,p.QWORDS]
                assert list(map(int,strides.split(',')))==[0,16*p.OUT,p.OUT,1]
    assert len(entries)==4


@pytest.mark.parametrize('blocks',[16,41])
def test_operand_inverse_every_token_feature_and_core(blocks):
    bits=(np.arange(512*blocks*64,dtype=np.uint32)%2048+0x3c00).astype(np.uint16).reshape(512,blocks*64)
    raw=np.zeros((blocks,4,4,p.OUT),np.uint32)
    for feature in range(blocks):
        for mg in range(4):
            for row in range(4):
                tile=bits[mg*128+row*32:mg*128+row*32+32,feature*64:feature*64+64]
                raw[feature,mg,row,16:1040].view(np.uint16)[:]=p.attention.tiled(tile,4,8)
    assert np.array_equal(b.unpack_operand(raw.reshape(-1,p.OUT),blocks),(bits.astype(np.uint32)<<16).view(np.float32))


def test_mlp_image_dispatches_only_its_operator_family():
    text=b.mlp_prefix()
    assert '@chain_geglu' in text and '@chain_finish' in text
    assert '@chain_elementwise' not in text and '%legacy = arith.constant false' in text
    seq=b.MlpSequence(1024); seq.control(3)
    with pytest.raises(ValueError): seq.control(4)
