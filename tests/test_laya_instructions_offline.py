# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit RTP slots, instruction framing and compiled-control equivalence."""
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import instructions as t


def fixture(valid=512,local=False,extra=False):
    audits=[dict(core=[col,row],spans=[(0x1000+(col*4+row-2)*64,0x1040+(col*4+row-2)*64,f'chain_control_{col*4+row-2}')]) for col in range(4) for row in range(2,6)]
    mapping=t.registers(audits); reverse={value:addr for addr,value in mapping.items()}; ops=[]
    def write(core,field,value): ops.append(struct.pack('<6I',0,0,reverse[core,field],0,value,24))
    for core in range(16):
        for field,value in [('mode',2),('valid',512),('local',0)]: write(core,field,value)
    for core in range(16): write(core,'mode',4)
    if extra:
        for core in range(16):
            write(core,'valid',valid); write(core,'local',int(local))
    words=[129,48,0,0,0,0,0x1d004,0,0,0,0,0]; ops.append(struct.pack('<12I',*words))
    for core in range(16): write(core,'mode',0)
    if extra:
        for core in range(16): write(core,'valid',512); write(core,'local',0)
    payload=b''.join(ops)
    return struct.pack('<4I',0x06030000,260,len(ops),len(payload)+16)+payload,audits


@pytest.mark.parametrize('valid,local',[(0,False),(1,True),(193,True),(512,False)])
def test_patch_matches_independent_compiled_control_batches(valid,local):
    data,audits=fixture(); reference,_=fixture(valid,local,True)
    output,changes=t.patch(data,audits,valid,local)
    t.compare_reference(output,reference,audits)
    allowed={byte for item in changes for byte in range(item['value_offset'],item['value_offset']+4)}
    assert len(changes)==32 and len(allowed)==128 and len(output)==len(data)
    assert all(a==b or i in allowed for i,(a,b) in enumerate(zip(data,output)))
    assert {item['address']&0xfffff for item in changes}!={0xe014,0xe018}


def test_live_control_mismatch_is_rejected_despite_identical_other_operations():
    data,audits=fixture(); candidate,_=t.patch(data,audits,193,True); wrong,_=fixture(192,True,True)
    with pytest.raises(ValueError,match='semantics differ'): t.compare_reference(candidate,wrong,audits)


def test_patch_rejects_corrupt_header_missing_core_and_already_patched_base():
    data,audits=fixture()
    with pytest.raises(ValueError): t.patch(data[:-4],audits,1,False)
    with pytest.raises(ValueError): t.patch(data,audits[:-1],1,False)
    patched,_=t.patch(data,audits,193,True)
    with pytest.raises(ValueError): t.patch(patched,audits,192,True)


@pytest.mark.parametrize('valid,local',[(-1,False),(513,False),(1.5,False),(1,1),(True,False)])
def test_patch_rejects_invalid_controls(valid,local):
    data,audits=fixture()
    with pytest.raises(ValueError): t.patch(data,audits,valid,local)
