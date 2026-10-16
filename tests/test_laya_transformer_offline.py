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
    data=p.gemm_packet(a,b,0x7fc12345); kb=2; ng=2
    q=data[:32*kb*p.QWORDS].reshape(8,kb,4,p.QWORDS)
    weights=data[32*kb*p.QWORDS:].reshape(4,ng,kb,p.KVWORDS)
    for mg in range(8):
        for kk in range(kb):
            for row in range(4):
                packet=q[mg,kk,row]
                assert list(packet[[2,4,6,7]])==[0x7fc12345,2,int(kk==0),int(kk==kb-1)]
                tile=a[mg*64+row*16:mg*64+row*16+16,kk*64:kk*64+64]
                assert np.array_equal(packet[16:528].view(np.uint16),p.attention.tiled(p.attention.bits(tile),4,8))
                assert np.all(packet[528:]==0)
    for col in range(4):
        for group in range(ng):
            for kk in range(kb):
                start=(group*4+col)*32
                tile=b[kk*64:kk*64+64,start:start+32]
                packet=weights[col,group,kk]
                assert np.array_equal(packet[:1024].view(np.uint16),p.attention.tiled(p.attention.bits(tile),8,4))
                assert np.all(packet[1024:]==0)


def test_gemm_output_all_rows_columns_and_tiles():
    ref=np.arange(512*256,dtype=np.float32).reshape(512,256)
    raw=np.empty((4,2,8,4,512),np.float32)
    for col in range(4):
        for group in range(2):
            for mg in range(8):
                for row in range(4):
                    raw[col,group,mg,row]=ref[mg*64+row*16:mg*64+row*16+16,(group*4+col)*32:(group*4+col+1)*32].reshape(-1)
    assert np.array_equal(p.gemm_output(raw.reshape(-1,512),256),ref)


def test_geometry_rejects_unreachable_dimensions():
    for k,n in ((63,256),(128,255),(0,256)):
        with pytest.raises(ValueError): p.geometry('gemm',k,n)
    assert p.geometry('gemm',1024,512)==(p.INPUT_WORDS,512*p.OUT)


def test_command_layouts_keep_the_same_persistent_program():
    prefix=[p.harness(mode).split('    aie.runtime_sequence')[0] for mode in ('attention','norm','gemm')]
    assert prefix[0]==prefix[1]==prefix[2]
