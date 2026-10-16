# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent physical-layout and complete arena-write checks."""
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]/'kernels/laya_transformer'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('laya_rotary',HERE/'rotary.py')
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
p=r.p


def test_all_output_dma_words_written_once_with_disjoint_arena_guards():
    pl=r.plan(); counts=np.zeros(pl['words'],np.uint8)
    entries=re.findall(r'%o\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]',r.runtime(pl,True,193))
    for offset,dimensions,strides in entries:
        dims=list(map(int,dimensions.split(','))); strides=list(map(int,strides.split(',')))
        inner=np.arange(dims[3])*strides[3]
        for outer in np.ndindex(*dims[:3]):
            start=int(offset)+sum(i*s for i,s in zip(outer,strides))
            counts[start+inner]+=1
    expected=np.zeros(pl['words'],np.uint8)
    for name in ('projection','queries','keys','kv','attention','packed_attention','norm','packed_norm'):
        start=pl['offsets'][name]; expected[start:start+pl['sizes'][name]]=1
    start=pl['offsets']['norm_input']
    expected[start:start+pl['sizes']['norm_input']].reshape(512,p.OUT)[:,:1040]=1
    assert np.array_equal(counts,expected)


def test_query_packet_inverse_every_head_token_feature():
    bits=(np.arange(16*512*64,dtype=np.uint32)%1024+0x3c00).astype(np.uint16).reshape(16,512,64)
    raw=np.zeros((512,p.OUT),np.uint32)
    for head in range(16):
        for block in range(32):
            raw[head*32+block,16:528].view(np.uint16)[:]=p.attention.tiled(bits[head,block*16:block*16+16],4,8)
    expected=(bits.astype(np.uint32)<<16).view(np.float32)
    assert np.array_equal(r.unpack_query(raw),expected)


def test_key_value_inverse_every_transferred_data_word():
    rng=np.random.default_rng(71)
    key=rng.integers(0x3c00,0x4000,(16,512,64),dtype=np.uint16)
    value=rng.integers(0x3c00,0x4000,(16,512,64),dtype=np.uint16)
    raw=np.zeros((256,p.OUT),np.uint32)
    for head in range(16):
        for block in range(16):
            dst=raw[head*16+block,16:2064].view(np.uint16)
            dst[:2048]=p.attention.tiled(key[head,block*32:block*32+32].T,8,4)
            dst[2048:]=p.attention.tiled(value[head,block*32:block*32+32],8,4)
    gotk,gotv=r.unpack_kv(raw)
    assert np.array_equal(gotk,(key.astype(np.uint32)<<16).view(np.float32))
    assert np.array_equal(gotv,(value.astype(np.uint32)<<16).view(np.float32))


def test_rotary_matrix_stream_addresses_all_qkv_source_tiles():
    pl=r.plan(); entries=re.findall(r'%a\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @qL2_(\d+)',r.runtime(pl,False,512))
    entries=entries[48:]
    for mode in (4,5,6,7):
        for which in range(4):
            for mg in range(4):
                for col in range(4):
                    offset,dims,strides,found=entries.pop(0); head=col*4+which
                    if mode==7: expected=pl['offsets']['keys']+(head*16+mg*4)*p.OUT
                    else:
                        feature=head+(16 if mode==6 else 0)
                        expected=pl['offsets']['projection']+((feature%4)*12+feature//4)*16*p.OUT+mg*4*p.OUT
                    assert int(found)==col and int(offset)==expected
                    assert list(map(int,dims.split(',')))==[1,4,129,16]
                    assert list(map(int,strides.split(',')))==[0,p.OUT,16,1]
    assert len(entries)==4+64+64+4+64


def test_attention_pair_and_output_projection_stream_addresses():
    pl=r.plan(); entries=re.findall(r'%a\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @(q|kv)L2_(\d+)',r.runtime(pl,False,512))
    q=[entry for entry in entries if entry[3]=='q'][48+256+4:]
    kv=[entry for entry in entries if entry[3]=='kv'][48+256+16:]
    for which in range(4):
        for mg in range(4):
            for col in range(4):
                head=col*4+which; source=pl['offsets']['attention']+(head*32+mg*8)*p.OUT
                for entry,expected in ((q.pop(0),source),(kv.pop(0),source+p.OUT)):
                    offset,dims,strides,_,found=entry
                    assert int(offset)==expected and int(found)==col
                    assert list(map(int,dims.split(',')))==[1,4,129,16]
                    assert list(map(int,strides.split(',')))==[0,2*p.OUT,16,1]
    for group in range(4):
        for mg in range(4):
            for col in range(4):
                offset,dims,strides,_,found=q.pop(0)
                assert int(offset)==pl['offsets']['packed_attention']+mg*4*p.OUT and int(found)==col
                assert list(map(int,dims.split(',')))==[1,16,4,p.QWORDS]
                assert list(map(int,strides.split(',')))==[0,16*p.OUT,p.OUT,1]
                offset,_,_,_,found=kv.pop(0)
                assert int(offset)==pl['offsets']['projection_weight']+(col*4+group)*16*p.KVWORDS and int(found)==col
    assert len(q)==len(kv)==4+64
    for col in range(4):
        offset,_,_,_,found=q.pop(0)
        assert int(offset)==pl['offsets']['norm_input']+col*128*p.OUT and int(found)==col
        offset,_,_,_,found=kv.pop(0)
        assert int(offset)==pl['offsets']['affine'] and int(found)==col
    for group in range(4):
        for mg in range(4):
            for col in range(4):
                for entries in (q,kv):
                    offset,_,_,_,found=entries.pop(0)
                    assert int(offset)==pl['offsets']['norm']+mg*128*p.OUT and int(found)==col
    assert not q and not kv


def test_output_projection_scatter_preserves_original_residual_words():
    pl=r.plan(); entries=re.findall(r'%o\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\].*metadata = @outL2_(\d+)',r.runtime(pl,False,512))
    entries=entries[48+256+4+64:]
    for group in range(4):
        for mg in range(4):
            for col in range(4):
                offset,dims,strides,found=entries.pop(0)
                assert int(offset)==pl['offsets']['norm_input']+mg*128*p.OUT+(group*4+col)*65
                assert list(map(int,dims.split(',')))==[1,4,32,65]
                assert list(map(int,strides.split(',')))==[0,32*p.OUT,p.OUT,1]
                assert int(found)==col
                for core in range(4):
                    for row in range(32):
                        start=int(offset)+core*32*p.OUT+row*p.OUT
                        relative=(start-pl['offsets']['norm_input'])%p.OUT
                        assert relative+65<=1040
    assert len(entries)==4+64


def test_attention_to_gemm_packet_inverse_all_rows_heads_and_features():
    bits=(np.arange(512*1024,dtype=np.uint32)%2048+0x3c00).astype(np.uint16).reshape(512,1024)
    raw=np.zeros((16,4,4,p.OUT),np.uint32)
    for head in range(16):
        for group in range(4):
            for row in range(4):
                tile=bits[group*128+row*32:group*128+row*32+32,head*64:head*64+64]
                raw[head,group,row,16:1040].view(np.uint16)[:]=p.attention.tiled(tile,4,8)
    got=r.unpack_activation(raw.reshape(256,p.OUT))
    assert np.array_equal(got,(bits.astype(np.uint32)<<16).view(np.float32))


def test_rtp_writes_only_changed_values_and_rejects_unavailable_operation():
    seq=r.Sequence(1024); seq.control(4,position=0,valid=193,local=1)
    initial=len(re.findall('"aiex.npu.rtp_write"',seq.finish()))
    assert initial==112
    seq.control(4,position=128,valid=193,local=1)
    assert len(re.findall('"aiex.npu.rtp_write"',seq.finish()))==initial+16
    with pytest.raises(ValueError): seq.control(3)
