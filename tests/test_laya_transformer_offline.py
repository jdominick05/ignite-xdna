# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent transformer packet routing; device-free independent checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('laya_transformer_program',Path(__file__).resolve().parents[1]/'kernels/laya_transformer/program.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_gemm_stream_every_operand_tile_and_k_boundary():
    a,b=p.gemm_case(11,'random',128,256)
    data=p.gemm_packet(a,b,0x7fc12345); kb=2; ng=1
    q=data[:16*kb*p.QWORDS].reshape(4,kb,4,p.QWORDS)
    weights=data[16*kb*p.QWORDS:].reshape(4,ng,kb,p.KVWORDS)
    for mg in range(4):
        for kk in range(kb):
            for row in range(4):
                packet=q[mg,kk,row]
                assert list(packet[[2,4,6,7]])==[0x7fc12345,2,int(kk==0),int(kk==kb-1)]
                tile=a[mg*128+row*32:mg*128+row*32+32,kk*64:kk*64+64]
                assert np.array_equal(packet[16:1040].view(np.uint16),p.attention.tiled(p.attention.bits(tile),4,8))
                assert np.all(packet[1040:]==0)
    for col in range(4):
        for group in range(ng):
            for kk in range(kb):
                start=(group*4+col)*64
                tile=b[kk*64:kk*64+64,start:start+64]
                packet=weights[col,group,kk]
                assert np.array_equal(packet[:2048].view(np.uint16),p.attention.tiled(p.attention.bits(tile),8,4))
                assert np.all(packet[2048:]==0)


def test_gemm_output_all_rows_columns_and_tiles():
    ref=np.arange(512*256,dtype=np.float32).reshape(512,256)
    raw=np.empty((4,1,4,4,2048),np.float32)
    for col in range(4):
        for group in range(1):
            for mg in range(4):
                for row in range(4):
                    raw[col,group,mg,row]=ref[mg*128+row*32:mg*128+row*32+32,(group*4+col)*64:(group*4+col+1)*64].reshape(-1)
    assert np.array_equal(p.gemm_output(raw.reshape(-1,2048),256),ref)


def test_geometry_rejects_unreachable_dimensions():
    for k,n in ((0,256),(128,0),(-1,256)):
        with pytest.raises(ValueError): p.geometry('gemm',k,n)
    assert p.geometry('gemm',1024,512)==(384*p.QWORDS,128*p.OUT)


def test_small_head_shapes_pad_dimensions_with_zero_operands():
    a,b=p.gemm_case(11,'basis',1028,2,1)
    packet=p.gemm_packet(a,b,0x7fc12345)
    assert len(packet)==(4+4)*17*p.QWORDS
    raw=np.zeros((4,1,1,4,2048),np.float32)
    assert p.gemm_output(raw,2,1).shape==(1,2)
    raw[0,0,0,0,2]=1
    with pytest.raises(ValueError): p.gemm_output(raw,2,1)


def test_command_layouts_keep_the_same_persistent_program():
    prefix=[p.harness(mode).split('    aie.runtime_sequence')[0] for mode in ('attention','norm','gemm')]
    assert prefix[0]==prefix[1]==prefix[2]


@pytest.mark.parametrize('m,k,n',[(1,1028,2),(4,65,259),(129,64,257)])
def test_unaligned_operands_have_exact_zero_padding(m,k,n):
    a,b=p.gemm_case(29,'random',k,n,m)
    before_a,before_b=a.copy(),b.copy()
    mp,kp,npad=p.padded(m,128),p.padded(k,64),p.padded(n,256)
    expected_a=np.zeros((mp,kp),np.uint16); expected_a[:m,:k]=p.attention.bits(a)
    expected_b=np.zeros((kp,npad),np.uint16); expected_b[:k,:n]=p.attention.bits(b)
    packet=p.gemm_packet(a,b,0x7fc12345); kb=kp//64; mg=mp//128; ng=npad//256
    qa=packet[:mg*4*kb*p.QWORDS].reshape(mg,kb,4,p.QWORDS)
    wb=packet[mg*4*kb*p.QWORDS:].reshape(4,ng,kb,p.KVWORDS)
    for group in range(mg):
        for kk in range(kb):
            for row in range(4):
                raw=qa[group,kk,row,16:1040].view(np.uint16)
                tile=raw.reshape(8,8,4,8).transpose(0,2,1,3).reshape(32,64)
                start=group*128+row*32
                assert np.array_equal(tile,expected_a[start:start+32,kk*64:kk*64+64])
    for col in range(4):
        for group in range(ng):
            for kk in range(kb):
                raw=wb[col,group,kk,:2048].view(np.uint16)
                tile=raw.reshape(8,16,8,4).transpose(0,2,1,3).reshape(64,64)
                start=(group*4+col)*64
                assert np.array_equal(tile,expected_b[kk*64:kk*64+64,start:start+64])
    assert np.array_equal(a,before_a) and np.array_equal(b,before_b)
