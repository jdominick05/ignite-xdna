# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent byte and DMA-address checks for the composed MLP."""
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]/'kernels/laya_transformer'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('laya_mlp',HERE/'mlp.py')
mlp=importlib.util.module_from_spec(spec); spec.loader.exec_module(mlp)
p=mlp.p


def test_arena_regions_disjoint_aligned_and_complete():
    pl=mlp.plan(); spans=[]
    for name,off in pl['offsets'].items():
        assert off%16==0
        spans.append((off,off+pl['sizes'][name]))
    assert all(a[1]+16==b[0] for a,b in zip(spans,spans[1:]))
    assert spans[-1][1]+16==pl['words']
    with pytest.raises(ValueError): mlp.plan(h=65)


def test_full_model_geglu_pair_addresses_cross_all_columns_and_groups():
    pl=mlp.plan(); runtime=mlp.runtime(pl)
    entries=re.findall(r'%a\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @(q|kv)L2_(\d+)',runtime)
    wi_groups=pl['wi_ng']*4*2
    gates=entries[wi_groups:wi_groups+pl['hk']*2]
    for f in range(pl['hk']):
        for entry,feature,kind in ((gates[2*f],f,'q'),(gates[2*f+1],f+pl['hk'],'kv')):
            offset,dims,strides,found,col=entry
            assert found==kind and int(col)==f%4
            # Feature f belongs to physical column f%4, group f//4.
            expected=pl['offsets']['wi']+((feature%4)*pl['wi_ng']+feature//4)*pl['mg']*4*p.OUT
            assert int(offset)==expected
            assert list(map(int,dims.split(',')))==[1,4,4,p.QWORDS]
            assert list(map(int,strides.split(',')))==[0,4*p.OUT,p.OUT,1]


def test_device_packed_activation_inverse_and_second_gemm_dma_every_word():
    pl=mlp.plan(m=257,k=64,h=192,n=256); m,k,h,n=pl['shape']
    # Distinct bf16 bits at every logical coordinate expose transpositions.
    bits=(np.arange(pl['mg']*128*h,dtype=np.uint32)%1024+0x3c00).astype(np.uint16).reshape(pl['mg']*128,h)
    bits[m:]=0
    arena=np.full(pl['words'],mlp.POISON,np.uint32)
    packets=arena[pl['offsets']['geglu']:pl['offsets']['geglu']+pl['sizes']['geglu']].reshape(pl['hk'],pl['mg'],4,p.OUT)
    for f in range(pl['hk']):
        for group in range(pl['mg']):
            for row in range(4):
                tile=bits[group*128+row*32:group*128+row*32+32,f*64:f*64+64]
                packets[f,group,row,16:1040].view(np.uint16)[:]=p.attention.tiled(tile,4,8)
    expected=(bits[:m].astype(np.uint32)<<16).view(np.float32)
    assert np.array_equal(mlp.gated(arena,pl),expected)
    entries=re.findall(r'%a\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @qL2_\d+',mlp.runtime(pl))
    seen=0
    for group in range(pl['mg']):
        offset,dims,strides=entries[-pl['mg']*4+group*4]
        dims=list(map(int,dims.split(','))); strides=list(map(int,strides.split(',')))
        assert dims==[1,pl['hk'],4,p.QWORDS]
        for f in range(dims[1]):
            for row in range(dims[2]):
                start=int(offset)+f*strides[1]+row*strides[2]
                payload=arena[start+16:start+1040].view(np.uint16)
                tile=bits[group*128+row*32:group*128+row*32+32,f*64:f*64+64]
                assert np.array_equal(payload,p.attention.tiled(tile,4,8))
                seen+=payload.size
    assert seen==pl['mg']*128*h
