# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Actual image and word-address proofs for independently credited cached MLP."""
import argparse
import json
from pathlib import Path

import cached_mlp as candidate
import instructions as decoder

c=candidate
q=c.q
p=c.p
np=q.np


def expected_states(old,parent_text):
    wanted=dict(old)
    for col in range(4):
        base=(col<<25)|(1<<20)
        before=c.cache.inventory.layout(parent_text,col)['spans']
        after={item['name']:item for item in c.allocation_proof(parent_text,col)}
        # Compiler DCE removes the normal query BD blocks, which have no cold
        # startup task. They are installed once by the cold runtime instead.
        for bid in c.NORMAL_BDS:
            for index in range(8):del wanted[base+0xa0000+32*bid+4*index]
        for bid in range(48):
            address=base+0xa0000+32*bid+4
            if address not in wanted:continue
            value=old[address];start=(value&0x7ffff)*4-0x80000
            owner=[item for item in before if item['start']<=start<item['end']]
            if len(owner)!=1:raise ValueError('compiled parent BD buffer owner')
            delta=after[owner[0]['name']]['start']-owner[0]['start']
            wanted[address]=(value&~0x7ffff)|((start+delta+0x80000)//4)
        for row in range(4):
            done=32+2*row;ready=done+1
            wanted[base+0xc0000+16*done]=21;wanted[base+0xc0000+16*ready]=0
            for bid,acquire,release,next_id in ((16+row,(done,21),(ready,21),17+row if row<3 else None),
                                               (c.CACHE_HEADS[row],(ready,1),(done,1),None)):
                address=base+0xa0000+32*bid
                for index in range(8):wanted[address+4*index]=old[base+0xa0000+4*index]
                wanted[address]=16640
                wanted[address+4]=(c.CACHE_BASE+row*16640*4+0x80000)//4
                if next_id is not None:wanted[address+4]|=(next_id<<20)|(1<<19)
                aid,av=acquire;rid,rv=release
                wanted[address+28]=(aid+64)|(((-av)&127)<<8)|(1<<15)|((rid+64)<<16)|(rv<<24)|(1<<31)
    for col,address,value in c.queue_writes(q.sq.bm.b.plan(),'wi'):
        wanted[(col<<25)|(1<<20)|address]=value
    return wanted


def state_proof(old,new,parent_text):
    wanted=expected_states(old,parent_text)
    if new!=wanted:
        difference=[hex(address) for address in sorted(new.keys()|wanted.keys()) if new.get(address)!=wanted.get(address)]
        raise ValueError('unexpected cached MLP CDO: '+','.join(difference[:12]))
    return dict(entire_init_state_exact=True,new_cache_descriptors=32,normal_descriptors_initialized_by_cold_runtime=64,
        separate_cache_credits=True,normal_query_credits_unchanged=True)


def payload(state,base,bid):
    address=base+0xa0000+32*bid
    first=((state[address+4]&0x7ffff)*4-0x80000)//4
    return np.arange(first,first+(state[address]&0x1ffff),dtype=np.int64)


def word_proof(state,old,parent_text,plan):
    initialized=dict(state)
    for col,address,value in c.normal_registers(old,parent_text):initialized[(col<<25)|(1<<20)|address]=value
    checked=0
    for col in range(4):
        base=(col<<25)|(1<<20)
        preload=next(event for event in c.events(plan,'wi') if event['kind']=='q' and event['col']==col)
        source=q.walk.addresses(preload).reshape(4,16*1040)
        reference=[q.walk.addresses(event).reshape(16,4,1040) for event in c.phases.partition(q.events(plan))['wi']
                   if event['kind']=='q' and event['col']==col]
        for row,head in enumerate(c.CACHE_HEADS):
            if not np.array_equal(payload(initialized,base,16+row),payload(initialized,base,head)):
                raise ValueError('cache physical source/reader span')
            expected=np.concatenate([group[:,row,:].reshape(-1) for group in reference])
            observed=np.tile(source[row],21)
            if not np.array_equal(expected,observed):raise ValueError('cache per-core DDR packet identity')
            checked+=observed.size
        # Normal S2MM fills buffer0 row0..3, then buffer1 row0..3.
        destinations=np.concatenate([payload(initialized,base,bid) for bid in range(8)]).reshape(2,4,1040)
        for row,head in enumerate(c.NORMAL_HEADS):
            actual=np.concatenate([payload(initialized,base,head),payload(initialized,base,head+1)])
            if not np.array_equal(actual,destinations[:,row,:].reshape(-1)):
                raise ValueError('normal ping/pong producer and per-core consumer disagree')
        for phase in ('geglu','wo','norm'):
            first,last=c.repetitions(plan,phase)[col]
            data=np.concatenate([q.walk.addresses(event) for event in c.events(plan,phase) if event['kind']=='q' and event['col']==col])
            if data.size!=first*2*4*1040 or first!=last:raise ValueError('normal finite producer/consumer balance')
            checked+=data.size
    return dict(query_word_addresses_checked=checked,all_core_packet_words_and_order_preserved=True,
        normal_tasks_return_two_credits=True,cache_tasks_return_twentyone_credits=True,
        limits='conditional delivery/drain and FIFO ordering proof; not a global deadlock proof')


def runtime_proof(raw,parent_text,old,plan,audits,cold):
    expected=[];offset=0
    if cold:expected.extend((0,col,address,value) for col,address,value in c.normal_registers(old,parent_text))
    wanted=[]
    for phase in c.phases.PHASES:
        if not(cold and phase=='wi'):
            expected.extend((offset,col,address,value) for col,address,value in c.queue_writes(plan,phase))
        events=c.events(plan,phase);wanted.extend(events);offset+=len(events)
    actual=[];position=0;blocks=patches=masks=0
    for op in decoder.parse(raw):
        address=op.get('addr',0);row=(address>>20)&31;col=address>>25;reg=address&0xfffff
        if op['op']=='WRITE' and row==1:
            if op['size']!=24 or op['val_low']!=0:raise ValueError('query task write encoding')
            actual.append((position,col,reg,op['val']))
        elif op['op']=='WRITE' and row==0:
            if reg not in (0x1d214,0x1d204,0x1d21c):raise ValueError('unknown Shim register write')
            position+=1
        elif op['op']=='TCT':position+=1
        elif op['op']=='BLOCKWRITE':blocks+=1
        elif op['op']=='DDR_PATCH':patches+=1
        elif op['op']=='MASKWRITE':
            if (row,reg,op['mask'],op['val'])!=(0,0x1d200,0xf00,0x1f00):raise ValueError('unknown Shim output control')
            if position>=len(wanted) or (wanted[position]['kind'],wanted[position]['col'])!=('out',col):
                raise ValueError('output control queue boundary')
            masks+=1
        elif op['op']!='WRITE':raise ValueError('unknown cached MLP runtime operation')
    if actual!=expected:raise ValueError('finite task configuration or completed phase boundary changed')
    transport=q.verify(raw,wanted,audits,plan['words'])
    if blocks!=transport['queues'] or patches!=blocks or masks!=transport['completion_waits']:
        raise ValueError('extra/missing Shim descriptors, controls or bindings')
    return dict(**transport,finite_task_writes=len(actual),completed_phase_queue_boundaries_exact=True,
        cold_initializes_normal_descriptors=cold)


def verify(parent,folder):
    meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    original=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    if meta['parent_build_sha256']!=p.attention.sha(parent/'build.json') or original['sources']!=q.sources():
        raise ValueError('cached MLP parent identity')
    if meta['source_sha256']!=p.attention.sha(Path(c.__file__)):raise ValueError('cached MLP builder changed')
    for filename in ('insts.bin','final.xclbin'):
        if p.attention.sha(folder/filename)!=meta[{'insts.bin':'instructions','final.xclbin':'xclbin'}[filename]]:
            raise ValueError('cached MLP artifact changed')
    for filename,sha in original['cdo'].items():
        if p.attention.sha(parent/'prj/cdo_main'/filename)!=sha:raise ValueError('qualified parent CDO changed')
        if 'init' not in filename and (parent/'prj/cdo_main'/filename).read_bytes()!=(folder/'prj/cdo_main'/filename).read_bytes():
            raise ValueError('cached MLP resident code/data/enable changed')
    for audit in original['audits']:
        col,row=audit['core']
        if p.attention.sha(folder/'prj'/f'frozen_core_{col}_{row}.elf')!=audit['elf_sha256']:
            raise ValueError('cached MLP resident ELF changed')
        name=f'ldScripts_main_core_{col}_{row}.ld.script'
        if (folder/'prj'/name).read_bytes()!=(parent/'prj'/name).read_bytes():raise ValueError('cached MLP inherited layout changed')
    if meta['audits']!=original['audits']:raise ValueError('cached MLP core audit changed')
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    old=c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    new=c.checks.state(folder/'prj/cdo_main/main_aie_cdo_init.bin');plan=q.sq.bm.b.plan()
    return dict(state=state_proof(old,new,text),words=word_proof(new,old,text,plan),
        cold_runtime=runtime_proof((folder/'insts.bin').read_bytes(),text,old,plan,meta['audits'],True))


def warm(parent,folder,out):
    proof=verify(parent,folder);plan=q.sq.bm.b.plan()
    text=(folder/'harness.mlir').read_text(encoding='utf-8');start=text.index('    aie.runtime_sequence(');end=c.cache.region(text,start)
    out.mkdir(parents=True,exist_ok=False)
    (out/'harness.mlir').write_text(text[:start]+c.runtime(plan,False)+text[end:],encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    old=c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    warm_proof=runtime_proof((out/'insts.bin').read_bytes(),text,old,plan,meta['audits'],False)
    result=dict(cold_proof=proof,warm_proof=warm_proof,cache_build_sha256=p.attention.sha(folder/'build.json'),
        warm_instructions_sha256=p.attention.sha(out/'insts.bin'),checker_sha256=p.attention.sha(Path(__file__)))
    p.attention.dump(out/'build.json',result);print('CACHED_MLP_COMPILED_PROOF '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','folder','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();warm(args.parent.absolute(),args.folder.absolute(),args.out.absolute())
