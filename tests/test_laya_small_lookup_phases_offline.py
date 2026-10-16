# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A diagnostic cut must preserve every qualified event and completed boundary."""
import copy
import json
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import small_lookup_phases as phases


def actual():return phases.parent.events(phases.parent.sq.bm.b.plan())


def test_actual_phase_partition_preserves_all_events_and_packet_addresses():
    events=actual();groups=phases.partition(events)
    assert [event for group in groups.values() for event in group]==events
    assert sum(event['kind']!='wait' for event in events)==519
    assert sum(event['kind']=='wait' for event in events)==173
    for phase,mode in phases.PHASES.items():
        assert {event['mode'] for event in groups[phase] if event['kind']!='wait'}=={mode}
        assert groups[phase][-1]['kind']=='wait'


@pytest.mark.parametrize('mutation',['missing_wait','duplicate_queue','phase_early','mode_unknown','missing_phase'])
def test_reject_unsafe_or_incomplete_phase_boundary(mutation):
    events=copy.deepcopy(actual())
    if mutation=='missing_wait':events.pop(next(i for i,event in enumerate(events) if event['kind']=='wait'))
    elif mutation=='duplicate_queue':events.insert(1,copy.deepcopy(events[0]))
    elif mutation=='phase_early':events[1]['mode']=3
    elif mutation=='mode_unknown':events[0]['mode']=7
    else:
        events=[event for group in list(phases.partition(events).values())[:-1] for event in group]
    with pytest.raises(ValueError):phases.partition(events)


def compiled():
    folder=Path.cwd()/'scratch/laya_transformer/small_lookup_phases_v1_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled diagnostic phases unavailable')
    return folder,json.loads((folder/'build.json').read_text())


def test_all_compiled_phase_images_retain_parent_and_cover_qualified_schedule():
    folder,meta=compiled();parent=json.loads((folder.parent/'small_lookup_v3_m128/build.json').read_text())
    plan=phases.parent.sq.bm.b.plan();proof=phases.validate(folder,meta,plan,parent)
    assert sum(row['queues'] for row in proof.values())==519
    assert sum(row['completion_waits'] for row in proof.values())==173
    assert sum(row['word_addresses_checked'] for row in proof.values())==14899264


@pytest.mark.parametrize('phase',list(phases.PHASES))
@pytest.mark.parametrize('mutation',['query_length','rtp_mode','wait_column'])
def test_corrupt_actual_phase_transaction_is_rejected(phase,mutation):
    folder,meta=compiled();raw=(folder/phase/'insts.bin').read_bytes();changed=bytearray(raw)
    ops=phases.n.c.txn.parse(raw)
    if mutation=='query_length':
        op=next(op for op in ops if op['op']=='BLOCKWRITE');offset=op['offset']+16;bit=1
    elif mutation=='wait_column':
        op=next(op for op in ops if op['op']=='TCT');offset=op['offset']+8;bit=1<<16
    else:
        op=next(op for op in ops if op['op']=='WRITE' and op['addr']==(2<<20)+15872)
        offset=op['offset']+16;bit=1
    struct.pack_into('<I',changed,offset,struct.unpack_from('<I',changed,offset)[0]^bit)
    plan=phases.parent.sq.bm.b.plan();wanted=phases.partition(phases.parent.events(plan))[phase]
    with pytest.raises(ValueError):phases.parent.verify(bytes(changed),wanted,meta['layouts'][phase]['audits'],plan['words'])
