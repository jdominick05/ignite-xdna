# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject reuse when a query changes or is overwritten during input projection."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_wi_query_inventory as inventory


def events():return inventory.phases.partition(inventory.q.events(inventory.q.sq.bm.b.plan()))['wi']


def test_actual_reuse_has_identical_columns_and_live_sources():
    result=inventory.reuse(events())
    assert result['all_columns_read_identical_queries']
    assert result['query_source_disjoint_from_all_wi_writes']
    assert result['scheduled_shim_bytes']['q']==22364160
    assert result['one_preload_per_column_bytes']==1064960


@pytest.mark.parametrize('mutation',['changed_query','overwritten_source'])
def test_reject_unqualified_reuse(mutation):
    rows=copy.deepcopy(events());query=next(row for row in rows if row['kind']=='q')
    if mutation=='changed_query':query['offset']+=1
    else:
        output=next(row for row in rows if row['kind']=='out');output['offset']=query['offset']
    with pytest.raises(ValueError):inventory.reuse(rows)


def test_actual_memtile_capacity_is_not_contiguous_cache_space():
    path=Path.cwd()/'scratch/laya_transformer/small_lookup_v3_m128/prj/input_with_addresses.mlir'
    if not path.exists():pytest.skip('placed qualified image unavailable')
    result=inventory.layout(path.read_text(),0)
    assert result['unused_bytes']==441216
    assert result['largest_free_interval_bytes']==253888
    assert 253888<266240<=441216


def test_reject_overlapping_memtile_allocation():
    text='\n'.join(f'%b{i} = aie.buffer(%mem_tile_0_1) {{address = 0 : i32}} : memref<4xi32>' for i in range(5))
    with pytest.raises(ValueError):inventory.layout(text,0)
