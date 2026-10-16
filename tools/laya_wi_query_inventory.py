#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prove query reuse and inspect actual MemTile gaps; no device or speed claim."""
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import small_lookup_phases as phases
q=phases.parent
np=q.np
CAPACITY=0x80000


def reuse(events):
    queries={col:[] for col in range(4)};writes=[];totals=dict(q=0,kv=0,out=0)
    for event in events:
        if event['kind']=='wait':continue
        if event['mode']!=2:raise ValueError('inventory requires only input projection')
        addresses=q.walk.addresses(event);totals[event['kind']]+=addresses.size*4
        if event['output']:writes.append(addresses)
        if event['kind']=='q':queries[event['col']].append(addresses)
    written=np.unique(np.concatenate(writes));rows=[]
    for col,groups in queries.items():
        if len(groups)!=21 or any(not np.array_equal(group,groups[0]) for group in groups):
            raise ValueError('input query does not repeat identically across all output groups')
        first=groups[0]
        if first.size!=16*4*1040 or np.unique(first).size!=first.size:raise ValueError('query packet extent/alias')
        if np.intersect1d(first,written).size:raise ValueError('input projection overwrites reusable query data')
        rows.append(dict(col=col,repetitions=len(groups),one_query_stream_bytes=int(first.size*4),
            address_sequence_sha256=hashlib.sha256(first.astype('<i8').tobytes()).hexdigest()))
    if any(not np.array_equal(queries[0][0],queries[col][0]) for col in range(1,4)):
        raise ValueError('query streams differ across columns')
    return dict(columns=rows,all_columns_read_identical_queries=True,query_source_disjoint_from_all_wi_writes=True,
        scheduled_shim_bytes=totals,one_preload_per_column_bytes=sum(row['one_query_stream_bytes'] for row in rows),
        scope='DERIVED scheduled words, including repeated transfers; not physical DDR traffic or measured bandwidth')


def layout(text,col):
    spans=[]
    pattern=rf'%([\w]+) = aie.buffer\(%mem_tile_{col}_1\) \{{([^}}]+)\}} : memref<(\d+)xi32>'
    for name,attrs,count in re.findall(pattern,text):
        address=re.search(r'\baddress = (\d+)',attrs)
        if address is None:raise ValueError('unplaced MemTile buffer')
        start=int(address[1]);spans.append(dict(name=name,start=start,end=start+4*int(count)))
    if len(spans)!=5:raise ValueError('unexpected qualified MemTile buffer census')
    spans.sort(key=lambda row:row['start']);cursor=0;gaps=[]
    for row in spans:
        if not cursor<=row['start']<row['end']<=CAPACITY:raise ValueError('MemTile overlap or out-of-bounds buffer')
        if cursor<row['start']:gaps.append([cursor,row['start']])
        cursor=row['end']
    if cursor<CAPACITY:gaps.append([cursor,CAPACITY])
    return dict(col=col,spans=spans,free_intervals=gaps,
        unused_bytes=sum(end-start for start,end in gaps),largest_free_interval_bytes=max(end-start for start,end in gaps))


def inventory(folder):
    meta=json.loads((folder/'build.json').read_text());plan=q.sq.bm.b.plan()
    if meta['sources']!=q.sources():raise ValueError('qualified smaller-query source changed')
    if q.p.attention.sha(folder/'insts.bin')!=meta['instructions']:raise ValueError('qualified transaction changed')
    all_events=q.events(plan)
    proof=q.verify((folder/'insts.bin').read_bytes(),all_events,meta['audits'],plan['words'])
    result=reuse(phases.partition(all_events)['wi'])
    ir=folder/'prj/input_with_addresses.mlir';text=ir.read_text()
    layouts=[layout(text,col) for col in range(4)]
    required=result['columns'][0]['one_query_stream_bytes']
    return dict(**result,compiled_transport=proof,memtiles=layouts,
        capacity=dict(bytes=CAPACITY,tag='SPEC',source='docs/SILICON.md: MemTile memory, getMemTileSize() = 0x80000'),
        contiguous_cache_fits_without_moving_existing_buffers=all(row['largest_free_interval_bytes']>=required for row in layouts),
        total_unused_capacity_sufficient=all(row['unused_bytes']>=required for row in layouts),
        evidence=dict(build_sha256=q.p.attention.sha(folder/'build.json'),instructions_sha256=meta['instructions'],
            placed_ir_sha256=q.p.attention.sha(ir),tool_sha256=q.p.attention.sha(Path(__file__)),sources=phases.sources()),
        limits='Address/liveness and allocation proof only; cache descriptors, lock lifecycle, packet delivery and speed are unqualified')


if __name__=='__main__':
    print('WI_QUERY_INVENTORY '+json.dumps(inventory(Path.cwd()/'scratch/laya_transformer/small_lookup_v3_m128')),flush=True)
