# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finite normal queries and independently credited Wi cache in the same image."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
from unittest.mock import patch

import cached_wi as cache
import cached_wi_checks as checks

q=cache.q
p=cache.p
phases=cache.inventory.phases
QWORDS=1040
CACHE_BASE=65536
NORMAL_BUFFERS={'qL2':(344064,409600),'kvL2':(442368,458752)}
NORMAL_HEADS=(10,30,13,33)
CACHE_HEADS=(20,36,21,37)
NORMAL_BDS=tuple(range(8))+(10,11,30,31,13,14,33,34)


def allocation_proof(text,col):
    old=cache.inventory.layout(text,col)
    wanted=[]
    for item in old['spans']:
        item=dict(item);name=item['name'];start=item['start']
        for prefix,addresses in NORMAL_BUFFERS.items():
            match=re.fullmatch(rf'{prefix}_{col}_cons_buff_([01])',name)
            if match:start=addresses[int(match[1])]
        item['end']=start+item['end']-item['start'];item['start']=start;wanted.append(item)
    for row in range(4):
        start=CACHE_BASE+row*16*QWORDS*4
        wanted.append(dict(name=f'wi_cache_{col}_{row}',start=start,end=start+16*QWORDS*4))
    wanted.sort(key=lambda row:row['start'])
    if any(a['end']>b['start'] for a,b in zip(wanted,wanted[1:])):raise ValueError('MLP cache buffer overlap')
    if wanted[-1]['end']>0x80000 or any(item['start']%32 for item in wanted):raise ValueError('MLP cache extent/alignment')
    return wanted


def events(plan,phase):
    original=phases.partition(q.events(plan))[phase]
    if phase!='wi':return original
    result=[];loaded=set()
    for event in original:
        if event['kind']=='q':
            col=event['col']
            if col in loaded:continue
            loaded.add(col);event=copy.deepcopy(event)
            event['dims']=[4,16,65,16];event['strides']=[p.OUT,16*p.OUT,16,1]
        result.append(event)
    return result


def repetitions(plan,phase):
    if phase=='wi':return [(1,21)]*4
    groups=events(plan,phase);counts=[]
    for col in range(4):
        words=sum(q.walk.addresses(event).size for event in groups if event['kind']=='q' and event['col']==col)
        if words%(4*2*QWORDS):raise ValueError('normal query pairs do not drain')
        counts.append(words//(4*2*QWORDS))
    if any(not 1<=count<=256 for count in counts):raise ValueError('normal query repeat width')
    return [(count,count) for count in counts]


def queue_writes(plan,phase):
    heads=CACHE_HEADS if phase=='wi' else NORMAL_HEADS
    for col in range(4):
        producer,replay=repetitions(plan,phase)[col]
        yield col,0xa0604,(16 if phase=='wi' else 0)|((producer-1)<<16)
        for channel,head in zip(range(2,6),heads):yield col,0xa0634+8*channel,head|((replay-1)<<16)


def normal_registers(parent_state,parent_text):
    result=[]
    for col in range(4):
        base=(col<<25)|(1<<20)
        old={item['name']:item for item in cache.inventory.layout(parent_text,col)['spans']}
        new={item['name']:item for item in allocation_proof(parent_text,col)}
        for bid in NORMAL_BDS:
            address=base+0xa0000+32*bid
            for index in range(8):
                value=parent_state[address+4*index]
                if index==1:
                    start=(value&0x7ffff)*4-0x80000
                    owner=[name for name,item in old.items() if item['start']<=start<item['end']]
                    if len(owner)!=1:raise ValueError('normal BD owner')
                    delta=new[owner[0]]['start']-old[owner[0]]['start']
                    value=(value&~0x7ffff)|((start+delta+0x80000)//4)
                    if bid in (7,11,31,14,34):value&=~0x03f80000
                result.append((col,0xa0000+32*bid+4*index,value))
    return result


def runtime(plan,cold=False,normal=()):
    seq=q.sq.bm.b.MlpSequence(plan['words'])
    # The canonicalizer removes unqueued normal BD blocks from initialization.
    # Install those independently addressed BDs once on the first cold request.
    for col,address,value in normal if cold else ():
        a=seq.constant(address);v=seq.constant(value)
        seq.lines.append(f'      aiex.npu.write32({a}, {v}) {{column = {col} : i32, row = 1 : i32}} : i32, i32')
    for phase,mode in phases.PHASES.items():
        if not (cold and phase=='wi'):
            for col,address,value in queue_writes(plan,phase):
                a=seq.constant(address);v=seq.constant(value)
                seq.lines.append(f'      aiex.npu.write32({a}, {v}) {{column = {col} : i32, row = 1 : i32}} : i32, i32')
        first=next(event for event in events(plan,phase) if event['kind']!='wait')
        seq.control(mode,first['kblocks'])
        for event in events(plan,phase):
            if event['kind']=='wait':seq.wait_columns([event['col']])
            else:seq.dma(event['col'],event['kind'],event['offset'],event['dims'],event['strides'],event['output'])
    text='\n'.join(seq.finish().splitlines()[:-2])
    return re.sub(r'@(q|kv|out)L2_([0-3])\b',lambda match:match[0]+'_shim_alloc',text)


def harness(parent_text,plan,cold=False,parent_state=None):
    text=parent_text;allocations={}
    for col in range(4):
        allocations[col]=allocation_proof(parent_text,col)
        for prefix,addresses in NORMAL_BUFFERS.items():
            for index,address in enumerate(addresses):
                pattern=rf'(%{prefix}_{col}_cons_buff_{index} = aie.buffer[^\n]*?address = )\d+( : i32)'
                text,count=re.subn(pattern,rf'\g<1>{address}\2',text)
                if count!=1:raise ValueError('normal L2 allocation changed')
        start=text.index(f'    %memtile_dma_{col}_1 = aie.memtile_dma');end=cache.region(text,start)
        body=text[start:end]
        # Skip normal startup queues; keep every BD defined for runtime use.
        for direction,channel,first,after in [('S2MM',0,1,9),('MM2S',2,21,23),('MM2S',3,26,28),('MM2S',4,31,33),('MM2S',5,36,38)]:
            pattern=rf'%\d+ = aie.dma_start\({direction}, {channel}, \^bb{first}, \^bb{after}\)[^\n]*'
            body,count=re.subn(pattern,f'aie.next_bd ^bb{after}',body)
            if count!=1:raise ValueError('normal query startup census')
        # Two-packet normal task: all phase query counts are even per core.
        for label,bid in ((8,7),(22,11),(27,31),(32,14),(37,34)):
            first=re.search(rf'(?m)^    \^bb{label}:',body);last=re.search(rf'(?m)^    \^bb{label+1}:',body)
            block=body[first.start():last.start()]
            block,count=re.subn(rf', next_bd_id = \d+ : i32', '',block)
            if count!=1:raise ValueError('normal finite BD link census')
            block,count=re.subn(r'aie.next_bd \^bb\d+[^\n]*','aie.end',block)
            if count!=1:raise ValueError('normal finite chain census')
            body=body[:first.start()]+block+body[last.start():]
        body=body.replace('      %c1_i32 = arith.constant 1 : i32','      %cache_twentyone = arith.constant 21 : i32\n      %c1_i32 = arith.constant 1 : i32',1)
        # The old final block is the entry to five additional finite cache tasks.
        final=re.search(r'(?m)^    \^bb40:[^\n]*\n\s*aie.end[^\n]*',body)
        if final is None:raise ValueError('normal DMA final block changed')
        added='    ^bb40:\n      aie.dma_start(S2MM, 0, ^bb100, ^bb104)\n'
        for row in range(4):
            bid=16+row
            added+=f'    ^bb{100+row}:\n      aie.use_lock(%wi_done_{col}_{row}, AcquireGreaterEqual, %cache_twentyone)\n'
            attrs=f'bd_id = {bid} : i32'+(f', next_bd_id = {bid+1} : i32' if row<3 else '')
            added+=f'      aie.dma_bd(%wi_cache_{col}_{row} : memref<16640xi32> offset = 0 len = 16640) {{{attrs}}}\n'
            added+=f'      aie.use_lock(%wi_ready_{col}_{row}, Release, %cache_twentyone)\n'
            added+=f'      aie.next_bd ^bb{101+row}\n' if row<3 else '      aie.end\n'
        for row,head in enumerate(CACHE_HEADS):
            label=104+2*row
            added+=f'    ^bb{label}:\n      aie.dma_start(MM2S, {row+2}, ^bb{label+1}, ^bb{label+2}, repeat_count = 20)\n'
            added+=f'    ^bb{label+1}:\n      aie.use_lock(%wi_ready_{col}_{row}, AcquireGreaterEqual, %c1_i32)\n'
            added+=f'      aie.dma_bd(%wi_cache_{col}_{row} : memref<16640xi32> offset = 0 len = 16640) {{bd_id = {head} : i32}}\n'
            added+=f'      aie.use_lock(%wi_done_{col}_{row}, Release, %c1_i32)\n      aie.end\n'
        added+='    ^bb112:\n      aie.end'
        body=body[:final.start()]+added+body[final.end():]
        declarations=''
        for row in range(4):
            for name,lock,init in [('done',32+2*row,21),('ready',33+2*row,0)]:
                if re.search(rf'aie.lock\(%mem_tile_{col}_1, {lock}\)',text):raise ValueError('cache lock collision')
                declarations+=f'    %wi_{name}_{col}_{row} = aie.lock(%mem_tile_{col}_1, {lock}) {{init = {init} : i32, sym_name = "wi_{name}_{col}_{row}"}}\n'
            address=CACHE_BASE+row*16*QWORDS*4
            declarations+=f'    %wi_cache_{col}_{row} = aie.buffer(%mem_tile_{col}_1) {{address = {address} : i32, sym_name = "wi_cache_{col}_{row}"}} : memref<16640xi32>\n'
        text=text[:start]+declarations+body+text[end:]
    start=text.index('    aie.runtime_sequence(');end=cache.region(text,start)
    normal=normal_registers(parent_state,parent_text) if cold else ()
    return text[:start]+runtime(plan,cold,normal)+text[end:],allocations


def build(parent,out):
    parent=parent.absolute();out=out.absolute();meta=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    if meta['sources']!=q.sources():raise ValueError('qualified arithmetic source changed')
    for name,wanted in meta['cdo'].items():
        if p.attention.sha(parent/'prj/cdo_main'/name)!=wanted:raise ValueError('qualified CDO changed')
    source=parent/'prj/input_with_addresses.mlir';plan=q.sq.bm.b.plan()
    text,allocations=harness(source.read_text(encoding='utf-8'),plan,cold=True,
        parent_state=checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin'))
    out.mkdir(parents=True,exist_ok=False);(out/'prj').mkdir();shutil.copyfile(parent/'kernel.o',out/'kernel.o')
    resident={}
    for audit in meta['audits']:
        col,row=audit['core'];files=[path for path in (parent/'prj').rglob('*.elf') if re.search(rf'core_{col}_{row}\.elf$',path.name)]
        if len(files)!=1 or p.attention.sha(files[0])!=audit['elf_sha256']:raise ValueError('resident parent ELF identity')
        target=out/'prj'/f'frozen_core_{col}_{row}.elf';shutil.copyfile(files[0],target);resident[col,row]=target
        name=f'ldScripts_main_core_{col}_{row}.ld.script';shutil.copyfile(parent/'prj'/name,out/'prj'/name)
    text=cache.external_cores(text,resident)
    (out/'harness.mlir').write_text(text,encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',q.STACK):audits=json.loads(json.dumps(p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
    if audits!=meta['audits']:raise ValueError('resident arithmetic or inherited core layout changed')
    for name in ('main_aie_cdo_elfs.bin','main_aie_cdo_enable.bin'):
        if (out/'prj/cdo_main'/name).read_bytes()!=(parent/'prj/cdo_main'/name).read_bytes():raise ValueError('core/enable CDO changed')
    result=dict(parent_build_sha256=p.attention.sha(parent/'build.json'),source_sha256=p.attention.sha(Path(__file__)),
        allocations=allocations,audits=audits,instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
        normal_repetitions={phase:repetitions(plan,phase) for phase in phases.PHASES},
        scope='compiled only; complete CDO/lock/word and runtime proofs required before hardware')
    p.attention.dump(out/'build.json',result)
    print('CACHED_MLP_BUILD '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args.parent,args.out)
