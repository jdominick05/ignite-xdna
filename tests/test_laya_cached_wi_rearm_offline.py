# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finite restart delta rejects hidden changes before any device use."""
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import cached_wi_rearm as rearm
import laya_cached_wi_rearm_primitive as primitive


def streams():
    root=Path.cwd()/'scratch/laya_transformer'
    cold=root/'cached_wi_v5_col0/insts.bin';warm=root/'cached_wi_rearm_v1_col0/insts.bin'
    if not warm.exists():pytest.skip('compiled restart instructions unavailable')
    return cold.read_bytes(),warm.read_bytes()


def test_actual_provenance_and_exact_instruction_delta():
    root=Path.cwd()/'scratch/laya_transformer';streams()
    proof=rearm.verify(root/'small_lookup_v3_m128',root/'cached_wi_v5_col0',root/'cached_wi_rearm_v1_col0')
    assert proof['restart_writes']==5
    assert proof['unchanged_cold_operations']==235


@pytest.mark.parametrize('index,field',[(0,8),(0,16),(1,16),(4,16),(5,16),(80,8)])
def test_reject_changed_write_address_value_or_cold_operation(index,field):
    cold,warm=streams();data=bytearray(warm);op=rearm.decoder.parse(warm)[index]
    offset=op['offset']+field
    value=struct.unpack_from('<I',data,offset)[0];struct.pack_into('<I',data,offset,value^1)
    with pytest.raises(ValueError):rearm.instruction_proof(cold,bytes(data))


def test_fixture_outputs_change_with_every_seed_and_integer_sums_are_exact():
    values=[]
    for seed in (11,29,47):
        values.append(primitive.expected(seed,0,0))
        for row in range(4):
            for group in range(21):
                products=[a*b for a,b in (primitive.operands(seed,row,group,k) for k in range(16))]
                assert 64*sum(abs(x) for x in products)<2**24
    assert len(set(values))==3


@pytest.mark.parametrize('relative',[
    'small_lookup_v3_m128/prj/cdo_main/main_aie_cdo_init.bin',
    'cached_wi_v5_col0/prj/frozen_core_0_2.elf',
    'cached_wi_v5_col0/prj/ldScripts_main_core_0_2.ld.script'])
def test_reject_changed_parent_or_resident_provenance(monkeypatch,relative):
    streams();root=Path.cwd()/'scratch/laya_transformer';target=(root/relative).absolute()
    original=rearm.p.attention.sha
    def changed(path):return '0'*64 if Path(path).absolute()==target else original(path)
    monkeypatch.setattr(rearm.p.attention,'sha',changed)
    with pytest.raises(ValueError):rearm.provenance(root/'small_lookup_v3_m128',root/'cached_wi_v5_col0')


def test_real_layer_output_spans_are_disjoint_and_cover_one_column():
    import laya_cached_wi_layer as layer
    plan=rearm.cache.q.sq.bm.b.plan();spans=list(layer.output_slices(plan))
    assert len(spans)==21
    assert sum(item.stop-item.start for item in spans)==174720
    assert all(left.stop<=right.start for left,right in zip(spans,spans[1:]))
    assert spans[-1].stop<plan['words']
