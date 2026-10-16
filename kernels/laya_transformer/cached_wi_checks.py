# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline CDO field and per-word replay proof for the one-shot cache prototype."""
from pathlib import Path
import hashlib
import json
import re

import cached_wi as cache
import window_lock_check as locks
q=cache.q
np=q.np


def state(path):
    result={}
    for address,mask,value in locks.packing.cdo_writes(path.read_bytes()):
        result[address]=(result.get(address,0)&(~mask&0xffffffff))|(value&mask)
    return result


def verify_states(old,new,plan):
    wanted=dict(old);base=1<<20
    for row in range(4):wanted[base+0xc0000+16*(2+2*row)]=21
    wanted[base+0xc0000+16*63]=0
    descriptors=[]
    for row in range(4):
        for part in range(2):
            bid=row*2+part
            descriptors.append((bid,row,part,(2+2*row,21) if part==0 else (63,0),
                (63,0) if part==0 else (3+2*row,21),bid+1 if bid<7 else None))
    heads=(10,30,13,33)
    for row,(first,last) in enumerate(((10,11),(30,31),(13,14),(33,34))):
        descriptors.append((first,row,0,(3+2*row,1),(63,0),last))
        descriptors.append((last,row,1,(63,0),(2+2*row,1),None))
    for bid,row,part,acquire,release,next_id in descriptors:
        address=base+0xa0000+32*bid;start,kblocks=cache.placements()[row][part]
        # xaiemlgbl_params.h MEM_TILE_MODULE_DMA_BD0_{0,1,7};
        # the internal MemTile data-memory aperture begins at byte 0x80000.
        wanted[address]=(old[address]&~0x1ffff)|kblocks*1040
        wanted[address+4]=(old[address+4]&~0x03ffffff)|((start+0x80000)//4)
        if next_id is not None:wanted[address+4]|=(next_id<<20)|(1<<19)
        aid,av=acquire;rid,rv=release
        wanted[address+28]=(aid+64)|(((-av)&127)<<8)|(1<<15)|((rid+64)<<16)|(rv<<24)|(1<<31)
    for channel,head in zip(range(2,6),heads):
        wanted[base+0xa0634+8*channel]=head|(20<<16)
    if new!=wanted:
        difference=[hex(address) for address in sorted(new.keys()|wanted.keys()) if new.get(address)!=wanted.get(address)]
        raise ValueError('unexpected compiled cache initialization: '+','.join(difference[:8]))
    # Decode actual compiled addresses, then follow the preload and each finite
    # consumer chain. Compare DDR source identity for every delivered word.
    def payload(bid):
        address=base+0xa0000+32*bid
        start=((new[address+4]&0x7ffff)*4-0x80000)//4
        count=new[address]&0x1ffff
        return np.arange(start,start+count,dtype=np.int64)
    destinations=np.concatenate([payload(bid) for bid in range(8)])
    preload=next(event for event in cache.events(plan) if event['kind']=='q')
    source=q.walk.addresses(preload)
    if len(destinations)!=len(source) or np.unique(destinations).size!=len(source):raise ValueError('cache preload coverage/alias')
    mapping=dict(zip(destinations.tolist(),source.tolist()))
    qualified=cache.inventory.phases.partition(q.events(plan))['wi']
    groups=[q.walk.addresses(event).reshape(16,4,1040) for event in qualified if event['kind']=='q' and event['col']==0]
    checked=0
    for row,(first,last) in enumerate(((10,11),(30,31),(13,14),(33,34))):
        addresses=np.concatenate([payload(first),payload(last)])
        try:once=np.array([mapping[int(address)] for address in addresses],dtype=np.int64)
        except KeyError as error:raise ValueError('cache read precedes a mapped preload word') from error
        repeats=((new[base+0xa0634+8*(row+2)]>>16)&255)+1
        observed=np.tile(once,repeats);expected=np.concatenate([group[:,row,:].reshape(-1) for group in groups])
        if not np.array_equal(observed,expected):raise ValueError('compiled cache replay changes per-core packet words')
        checked+=observed.size
    return dict(compiled_memtile_descriptors_checked=16,consumer_replays=21,delivered_query_word_addresses_checked=checked,
        initialization_changed_only_as_declared=True,all_per_core_query_packets_preserved=True,
        token_ledger='each cold per-core cache starts done=21/ready=0; fill takes21 and publishes21; each complete replay takes1 and returns1; dummy remains0',
        limits='one cache fill per initialized context; runtime rearming and full serving integration are not implemented; assumes delivery/drain, not a global deadlock proof')


def verify(parent,folder):
    parent=Path(parent).absolute();folder=Path(folder).absolute()
    meta=json.loads((folder/'build.json').read_text())
    if meta['parent_build_sha256']!=q.p.attention.sha(parent/'build.json'):raise ValueError('cache parent changed')
    if any(q.p.attention.sha(q.p.ROOT/name)!=sha for name,sha in meta['sources'].items()):raise ValueError('cache sources changed')
    for name,key in [('kernel.o','kernel_object_sha256'),('insts.bin','instructions'),('final.xclbin','xclbin')]:
        if q.p.attention.sha(folder/name)!=meta[key]:raise ValueError('cache artifact changed')
    cdo=folder/'prj/cdo_main';previous=parent/'prj/cdo_main'
    for name in ('main_aie_cdo_elfs.bin','main_aie_cdo_enable.bin'):
        if (cdo/name).read_bytes()!=(previous/name).read_bytes():raise ValueError('resident code/data or enable CDO changed')
    plan=q.sq.bm.b.plan();proof=verify_states(state(previous/'main_aie_cdo_init.bin'),state(cdo/'main_aie_cdo_init.bin'),plan)
    q.verify((folder/'insts.bin').read_bytes(),cache.events(plan),meta['audits'],plan['words'])
    return dict(**proof,code_data_and_enable_cdos_bit_identical=True,
        build_sha256=q.p.attention.sha(folder/'build.json'),cdo_sha256=q.p.attention.sha(cdo/'main_aie_cdo_init.bin'),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    root=Path.cwd()/'scratch/laya_transformer'
    print('CACHED_WI_COMPILED_PROOF '+json.dumps(verify(root/'small_lookup_v3_m128',root/'cached_wi_v5_col0')),flush=True)
