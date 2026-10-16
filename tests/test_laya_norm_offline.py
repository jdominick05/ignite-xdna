# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent packet routing and reference semantics, without an NPU context."""
import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('laya_norm_probe',Path(__file__).resolve().parents[1]/'kernels/laya_norm/probe.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_every_row_routes_through_memtile_split_and_join():
    x=np.broadcast_to(np.arange(512,dtype=np.float32)[:,None],(512,1024)).copy()
    r=x+1024
    g=np.arange(1024,dtype=np.float32); b=-g
    packet=p.pack(x,r,g,b,0x7fc12345)
    assert np.array_equal(packet[:1024].view(np.float32),g)
    assert np.array_equal(packet[1024:2048].view(np.float32),b)
    stream=packet[2048:].reshape(4,32,4,p.IN)
    visited=[]
    for col in range(4):
        for step in range(32):
            for core in range(4):
                row=col*128+step*4+core
                words=stream[col,step,core]
                assert words[0]==0x7fc12345
                assert np.all(words[1:16]==0)
                assert np.array_equal(words[16:1040].view(np.float32),x[row])
                assert np.array_equal(words[1040:].view(np.float32),r[row])
                visited.append(row)
    assert sorted(visited)==list(range(512))


def test_oracle_rounds_residual_before_computing_statistics():
    x,r,g,b=p.fixture(11,'rounding_edges')
    z,y=p.oracle(x,r,g,b)
    assert np.array_equal(z.view(np.uint32),np.add(x,r,dtype=np.float32).view(np.uint32))
    assert y.dtype==np.float64 and np.isfinite(y).all()
    assert z[0,0]==np.float32(1)
    assert z[0,1]==np.nextafter(np.nextafter(np.float32(1),np.float32(2)),np.float32(2))


def test_constant_and_cancelled_rows_return_affine_bias():
    for kind in ('constant','cancel'):
        vals=p.fixture(47,kind)
        _,y=p.oracle(*vals)
        assert np.array_equal(y,np.broadcast_to(vals[3],y.shape))
