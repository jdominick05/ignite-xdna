# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cache preparation must preserve per-core query order and existing allocations."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import cached_wi as cache


def placed():
    path=Path.cwd()/'scratch/laya_transformer/small_lookup_v3_m128/prj/input_with_addresses.mlir'
    if not path.exists():pytest.skip('qualified parent placement unavailable')
    return path.read_text()


def test_segmented_cache_fits_real_gaps_without_relocating_parent_buffers():
    segments=cache.placement_proof(placed())
    assert len(segments)==8
    assert sum(row['end']-row['start'] for row in segments)==266240


def test_cache_preload_changes_only_intercore_delivery_order():
    plan=cache.q.sq.bm.b.plan();events=cache.events(plan)
    queries=[event for event in events if event['kind']=='q']
    assert len(queries)==1
    actual=cache.q.walk.addresses(queries[0]).reshape(4,16,1040)
    original=cache.inventory.phases.partition(cache.q.events(plan))['wi']
    repeated=[cache.q.walk.addresses(event).reshape(16,4,1040) for event in original if event['kind']=='q' and event['col']==0]
    for row in range(4):
        assert all(cache.q.np.array_equal(actual[row],group[:,row,:]) for group in repeated)
    assert sum(event['kind']!='wait' for event in events)==43
    assert sum(event['kind']=='wait' for event in events)==21


def test_lowered_core_programs_and_core_dma_are_unchanged():
    old=placed();new,_=cache.harness(old,cache.q.sq.bm.b.plan())
    for col in range(4):
        for row in range(2,6):
            for marker in (f'aie.core(%tile_{col}_{row})',f'aie.mem(%tile_{col}_{row})'):
                start=old.index(marker);end=cache.region(old,start)
                assert old[start:end] in new
    assert 'aie.use_lock(%wi_cache_dummy, Acquire, %cache_zero)' in new
    assert 'aie.use_lock(%wi_cache_dummy, AcquireGreaterEqual' not in new


def test_reject_cache_overlapping_live_parent_data(monkeypatch):
    original=cache.placements();original[0][0]=(0,15)
    monkeypatch.setattr(cache,'placements',lambda:original)
    with pytest.raises(ValueError):cache.placement_proof(placed())
