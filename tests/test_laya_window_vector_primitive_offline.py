# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired queries must preserve live words, core selection, and vector ring wrap."""
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import window_vector_primitive as primitive
import laya_window_vector_primitive as runner


def artifact():
    path=Path.cwd()/'scratch/laya_transformer/window_vector_primitive_v2'
    if not (path/'build.json').exists():pytest.skip('compiled vector primitive unavailable')
    return path


def test_paired_query_complete_objects_preserve_all_live_original_words():
    for old,new in zip(primitive.events(False),primitive.events(True)):
        a=primitive.gemm.walk.addresses(old);b=primitive.gemm.walk.addresses(new)
        if old['kind']=='q':
            a=a.reshape(4,2064);b=b.reshape(2,4,1040)
            assert np.array_equal(a[:,:1040],b[0])
            assert np.array_equal(a[:,1040:],b[1,:,:1024])
            assert np.array_equal(b[1,:,1024:],a[:,0,None]+np.arange(2064,2080))
        else:assert np.array_equal(a,b)
        assert b.min()>=16 and b.max()<primitive.plan()['words']


def test_fixture_exposes_half_and_core_order_mistakes():
    data=primitive.fixture(11);pl=primitive.plan();off=pl['offsets']
    q=data[off['norm_query']:off['norm_query']+4*2080].reshape(4,2080)
    x=q[:,:1040].reshape(4,16,65)[:,:,:64].copy().view(np.float32)
    residual=q[:,1040:2064].copy().view(np.float32)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(residual))
    assert not np.array_equal(x.reshape(4,1024),residual)
    q=data[off['geglu_query']:off['geglu_query']+4*2080].reshape(4,2080)
    gate=q[:,16:2064].copy().view(np.float32).reshape(4,32,64)
    assert gate[0,:,0].min()<-16 and gate[0,:,0].max()>16
    assert not np.array_equal(gate[0],gate[1])
    assert not np.array_equal(gate[:,:16],gate[:,16:])


@pytest.mark.parametrize('window',[False,True])
def test_mixed_vector_core_ledger(window):
    folder=artifact()/('window' if window else 'original')
    result=primitive.control.verify_vector((folder/'prj/input_with_addresses.mlir').read_text(),window)
    assert result['repeated_norm_then_geglu_cycles']==3
    for row in result['cores']:
        assert row['tokens']==dict(q=12 if window else 6,kv=15,out=6)


@pytest.mark.parametrize('mutation',['release','second_query','selected_value'])
def test_corrupted_paired_core_protocol_rejected(mutation):
    text=primitive.control.clean((artifact()/'window/prj/input_with_addresses.mlir').read_text())
    if mutation=='release':
        old='aie.use_lock(%qF_0_cons_prod_lock_0, Release, %c2_i32)';new=old.replace('%c2_i32','%c1_i32')
    elif mutation=='second_query':
        old='@command_norm_split(%54, %57,';new='@command_norm_split(%54, %54,'
    else:
        # Make every core select broadcast value zero. A one-buffer KV ring
        # hides this corruption from pointer-only checks; token ordinal must catch it.
        old='arith.cmpi eq, %78, %59';new='arith.cmpi eq, %78, %c0'
    assert old in text
    with pytest.raises(ValueError):primitive.control.verify_vector(text.replace(old,new),True)


@pytest.mark.parametrize('mutation',['query_length','mode','queue_repeat'])
def test_compiled_vector_transfer_mutation_rejected(mutation):
    import json
    root=artifact();meta=json.loads((root/'build.json').read_text());raw=(root/'window/insts.bin').read_bytes()
    audits=meta['layouts']['window']['audits'];primitive.verify_stream(raw,True,audits)
    ops=primitive.n.c.txn.parse(raw);changed=bytearray(raw)
    if mutation=='query_length':
        op=next(o for o in ops if o['op']=='BLOCKWRITE' and o['addr']==0x1d000);bit=1
    elif mutation=='queue_repeat':
        op=next(o for o in ops if o['op']=='WRITE' and o['addr']==0x1d214);bit=1<<16
    else:
        audit=next(a for a in audits if a['core']==[0,2]);base=next(a for a,b,name in audit['spans'] if name=='chain_control_0')
        op=next(o for o in ops if o['op']=='WRITE' and o['addr']==(2<<20)+base);bit=1
    offset=op['offset']+16
    struct.pack_into('<I',changed,offset,struct.unpack_from('<I',changed,offset)[0]^bit)
    with pytest.raises(ValueError):primitive.verify_stream(bytes(changed),True,audits)


@pytest.mark.parametrize('mutation',['norm_payload','norm_padding','geglu_payload','geglu_padding'])
def test_runner_rejects_nonfinite_payload_and_packet_corruption(mutation):
    outputs=np.full((2,4,2080),0x7fc12345,np.uint32)
    outputs[0,:,:1024]=0;outputs[0,:,1040:2064]=0;outputs[1,:,16:1040]=0
    assert all(runner.sanity(outputs).values())
    operator,index,bits={'norm_payload':(0,0,0x7fc00000),'norm_padding':(0,1024,0),
                         'geglu_payload':(1,16,0x7fc07fc0),'geglu_padding':(1,0,0)}[mutation]
    outputs[operator,0,index]=bits
    assert not all(runner.sanity(outputs).values())
