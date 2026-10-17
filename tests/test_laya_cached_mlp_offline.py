# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject unsafe cached/normal mode transitions before silicon."""
import json
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import cached_mlp_checks as checks
c=checks.c


@pytest.fixture(scope='module')
def artifacts():
    root=Path.cwd()/'scratch/laya_transformer';parent=root/'small_lookup_v3_m128';folder=root/'cached_mlp_v3_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled full cache unavailable')
    old=c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    new=c.checks.state(folder/'prj/cdo_main/main_aie_cdo_init.bin')
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    return parent,folder,old,new,text


def test_actual_all_column_init_packets_and_runtime(artifacts):
    parent,folder,*_=artifacts;proof=checks.verify(parent,folder)
    assert proof['state']['new_cache_descriptors']==32
    assert proof['words']['query_word_addresses_checked']==8927360
    assert proof['cold_runtime']['queues']==439


@pytest.mark.parametrize('address,bit',[(0x1a0200,1),(0x1a0204,4),(0x1a021c,1<<8),
    (0x1a0644,1<<16),(0x1c0200,1),(0x1c0020,1),(0x1a0304,1),(0x61a04a4,4),(0x240000,1)])
def test_reject_bad_cache_normal_other_column_or_core_state(artifacts,address,bit):
    _,_,old,new,text=artifacts;changed=dict(new);changed[address]=changed.get(address,0)^bit
    with pytest.raises(ValueError):checks.state_proof(old,changed,text)


@pytest.mark.parametrize('position',[0,1,127,511,512,650])
def test_reject_runtime_normal_descriptor_or_phase_task_mutation(artifacts,position):
    _,folder,old,_,text=artifacts;raw=bytearray((folder/'insts.bin').read_bytes())
    writes=[op for op in checks.decoder.parse(raw) if op['op']=='WRITE']
    offset=writes[position]['offset']+16
    value=struct.unpack_from('<I',raw,offset)[0];struct.pack_into('<I',raw,offset,value^1)
    meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError):checks.runtime_proof(bytes(raw),text,old,c.q.sq.bm.b.plan(),meta['audits'],True)


def test_finite_normal_counts_use_actual_uneven_geglu_columns():
    plan=c.q.sq.bm.b.plan()
    assert c.repetitions(plan,'geglu')==[(11,11),(10,10),(10,10),(10,10)]
    assert c.repetitions(plan,'wo')==[(82,82)]*4
    assert c.repetitions(plan,'norm')==[(8,8)]*4


def test_disjoint_allocations_include_normal_and_cached_queries(artifacts):
    *_,text=artifacts
    for col in range(4):
        spans=c.allocation_proof(text,col)
        assert len(spans)==9
        assert all(a['end']<=b['start'] for a,b in zip(spans,spans[1:]))
        assert spans[-1]['end']<524288
