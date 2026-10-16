# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attention packet ABI and semantic checks; no device or external model required."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "laya_attention_probe", Path(__file__).resolve().parents[1] / "kernels/laya_attention/probe.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def untile(x, rows, cols, r, c):
    return x.reshape(rows//r,cols//c,r,c).transpose(0,2,1,3).reshape(rows,cols)


def test_bf16_rounds_ties_to_even():
    values = np.array([0x3f808000,0x3f818000,0xbf808000],np.uint32).view(np.float32)
    assert np.array_equal(p.bits(p.rounded(values)),[0x3f80,0x3f82,0xbf80])


def test_vector_gate_detects_linked_and_unresolved_soft_float():
    symbols="00001234 T __mulsf3\n U __fixunssfsi\n00002000 T attention_block\n00003000 T __muldi3\n"
    assert p.software_float_helpers(symbols)==["__fixunssfsi","__mulsf3"]
    assert p.software_float_helpers("00002000 T attention_block\n00003000 T __muldi3\n")==[]


def test_packet_layout_and_query_boundaries():
    q,k,v = p.case(1,11,"random")
    mask = (np.abs(np.arange(512)[:,None]-np.arange(512)[None,:])<=64).astype(np.uint8)
    packets = p.pack(q,k,v,mask,0x7fc12345)
    count, outputs = p.packet_geometry(mask,1)
    assert packets.shape == (count,p.PACKET)
    assert packets[:,0].sum() == outputs == 32
    assert packets[:,1].sum() == outputs
    raw = packets[0,16:].view(np.uint16)
    assert np.array_equal(untile(raw[:1024],16,64,4,8),p.bits(q[0,:16]))
    assert np.array_equal(untile(raw[1024:3072],64,32,8,4),p.bits(k[0,:32].T))
    assert np.array_equal(untile(raw[3072:5120],32,64,8,4),p.bits(v[0,:32]))
    assert np.array_equal(raw[5120:].reshape(16,32),p.bits(mask[:16,:32].astype(np.float32)))
    assert count < 512  # wholly excluded K/V blocks are absent from the stream


def test_fully_masked_rows_match_finite_source_mask():
    q,k,v = p.case(1,29,"random")
    mask = np.full((512,512),2,np.uint8)
    ref = p.oracle(q,k,v,mask)
    assert np.allclose(ref,v.astype(np.float64).mean(axis=1,keepdims=True),atol=1e-14)
    raw = p.pack(q,k,v,mask,0x7fc12345)[0,16:].view(np.uint16)
    assert np.all(raw[:1024] == 0)


def test_local_inclusive_boundary_and_padding_oracle():
    q,k,v = p.case(1,47,"uniform")
    mask = (np.abs(np.arange(512)[:,None]-np.arange(512)[None,:])<=64).astype(np.uint8)
    mask[:,193:] = 0
    mask[~mask.any(axis=1)] = 2
    ref = p.oracle(q,k,v,mask)
    assert np.allclose(ref[0,64],v[0,:129].astype(np.float64).mean(axis=0))
    assert np.allclose(ref[0,192],v[0,128:193].astype(np.float64).mean(axis=0))
    assert np.allclose(ref[0,511],v[0].astype(np.float64).mean(axis=0))


def test_numeric_rejects_partial_writes_and_large_error():
    ref=np.ones((2,64),np.float64)
    assert p.numeric(ref.copy(),ref)["ok"]
    got=ref.copy(); got[-1,-1]=np.nan
    assert not p.numeric(got,ref)["ok"]
    got=ref.copy(); got[-1,-1]=2
    assert not p.numeric(got,ref)["ok"]


def test_oracle_refuses_unresolved_empty_mask():
    q,k,v=p.case(1,11,"random")
    with pytest.raises(ValueError,match="fully masked"):
        p.oracle(q,k,v,np.zeros((512,512),np.uint8))


@pytest.mark.parametrize("local",[False,True])
@pytest.mark.parametrize("valid",[512,193,1,0])
def test_fast_pack_preserves_reference_bytes(local,valid):
    q,k,v=p.case(1,29,"random")
    mask=np.ones((512,512),np.uint8)
    if local:
        mask=(np.abs(np.arange(512)[:,None]-np.arange(512)[None,:])<=64).astype(np.uint8)
    mask[:,valid:]=0; mask[~mask.any(axis=1)]=2
    assert np.array_equal(p.pack(q,k,v,mask,0x7fc12345),p.pack_reference(q,k,v,mask,0x7fc12345))


@pytest.mark.parametrize("local",[False,True])
@pytest.mark.parametrize("valid",[512,193,1,0])
def test_compact_buffer_keeps_head_query_key_and_mask_identity(monkeypatch,local,valid):
    monkeypatch.setitem(sys.modules,"probe",p)
    spec=importlib.util.spec_from_file_location("attention_shared",Path(p.__file__).with_name("shared.py"))
    shared=importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
    q,k,v=p.case(16,47,"random")
    mask=np.ones((512,512),np.uint8)
    if local:
        mask=(np.abs(np.arange(512)[:,None]-np.arange(512)[None,:])<=64).astype(np.uint8)
    mask[:,valid:]=0; mask[~mask.any(axis=1)]=2
    packed=shared.pack(q,k,v,mask,0x7fc12345)
    assert packed.nbytes==3194880
    qs=packed[:16*32*shared.QWORDS].reshape(16,32,shared.QWORDS)
    assert np.all(qs[:,:,1]==valid)
    assert np.all(qs[:,:,3]==(local and valid!=0))
    h,i,j=13,29,11
    qt=untile(qs[h,i,16:].view(np.uint16),16,64,4,8)
    expect=p.bits(q[h,i*16:i*16+16]).copy()
    expect[np.all(mask[i*16:i*16+16]==2,axis=1)]=0
    assert np.array_equal(qt,expect)
    kv=packed[16*32*shared.QWORDS:].reshape(16,16,shared.KVWORDS)[h,j].view(np.uint16)
    assert np.array_equal(untile(kv[:2048],64,32,8,4),p.bits(k[h,j*32:j*32+32].T))
    assert np.array_equal(untile(kv[2048:4096],32,64,8,4),p.bits(v[h,j*32:j*32+32]))
    assert np.array_equal(kv[4096:],np.arange(32))


@pytest.mark.parametrize("valid",[512,193,1,0])
def test_cpu_baseline_matches_finite_mask_semantics(valid):
    pytest.importorskip("onnxruntime")
    q,k,v=p.case(1,29,"random")
    mask=(np.abs(np.arange(512)[:,None]-np.arange(512)[None,:])<=64).astype(np.uint8)
    mask[:,valid:]=0
    mask[~mask.any(axis=1)]=2
    ref=p.oracle(q,k,v,mask)
    got=p.cpu_session(1,mask).run(None,dict(q=q,k=k,v=v))[0]
    assert np.allclose(got,ref,atol=2e-6,rtol=2e-5)
