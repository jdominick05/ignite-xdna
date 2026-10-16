#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline one-column Wi prototype: preload per-core queries and replay them in L2."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path.cwd()/'tools'))
import laya_wi_query_inventory as inventory
q=inventory.q
p=q.p


def placements():
    # Every segment fits one bank and an existing free interval. No parent
    # activation, weight, output, query buffer or core allocation is relocated.
    return [[(327680,15),(147712,1)],[(393216,15),(151872,1)],
        [(458752,15),(156032,1)],[(33280,7),(82176,9)]]


def placement_proof(text):
    old=inventory.layout(text,0);new=[]
    for row,segments in enumerate(placements()):
        if sum(count for _,count in segments)!=16:raise ValueError('cache K-block coverage')
        for part,(start,count) in enumerate(segments):
            end=start+count*1040*4
            if start%32 or start//65536!=(end-1)//65536:raise ValueError('cache alignment/bank span')
            if not any(a<=start<end<=b for a,b in old['free_intervals']):raise ValueError('cache overlaps parent allocation')
            new.append(dict(name=f'wi_cache_{row}_{part}',start=start,end=end,words=count*1040))
    ordered=sorted(new,key=lambda row:row['start'])
    if any(a['end']>b['start'] for a,b in zip(ordered,ordered[1:])):raise ValueError('cache segments overlap')
    return new


def events(plan):
    qualified=inventory.phases.partition(q.events(plan))['wi'];result=[];loaded=False
    for event in qualified:
        if event['col']!=0:continue
        if event['kind']=='q':
            if loaded:continue
            loaded=True;changed=copy.deepcopy(event)
            changed['dims']=[4,16,65,16];changed['strides']=[p.OUT,16*p.OUT,16,1]
            before=q.walk.addresses(event).reshape(16,4,1040).transpose(1,0,2).reshape(-1)
            if not q.np.array_equal(before,q.walk.addresses(changed)):raise ValueError('per-core preload gather differs')
            result.append(changed)
        else:result.append(event)
    return result


def runtime(plan):
    seq=q.sq.bm.b.MlpSequence(plan['words']);seq.control(2,16)
    for event in events(plan):
        if event['kind']=='wait':seq.wait_columns([0])
        else:seq.dma(0,event['kind'],event['offset'],event['dims'],event['strides'],event['output'])
    text='\n'.join(seq.finish().splitlines()[:-2])
    return re.sub(r'@(q|kv|out)L2_0\b',lambda match:match[0]+'_shim_alloc',text)


def region(text,start):
    left=text.index('{',start);depth=1;right=left+1
    while depth:
        if text[right]=='{':depth+=1
        elif text[right]=='}':depth-=1
        right+=1
    return right


def replace_blocks(text,first,after,replacement):
    a=re.search(rf'(?m)^    \^bb{first}:',text)
    b=re.search(rf'(?m)^    \^bb{after}:',text)
    if a is None or b is None or a.start()>=b.start():raise ValueError('qualified DMA block labels changed')
    return text[:a.start()]+replacement+text[b.start():]


def bd(label,bid,row,part,acquire,release,next_label=None,next_bid=None):
    count=placements()[row][part][1]*1040
    result=f'    ^bb{label}:\n'
    if acquire:
        name='prod' if acquire==21 else 'cons'
        value='%cache_repetitions' if acquire==21 else '%c1_i32'
        result+=f'      aie.use_lock(%qL2_0_cons_{name}_lock_{row}, AcquireGreaterEqual, {value})\n'
    else:
        # GE(0) is erased by the upstream canonicalizer. Equal(0) on this
        # dedicated always-zero lock survives and never changes its value.
        result+='      aie.use_lock(%wi_cache_dummy, Acquire, %cache_zero)\n'
    attrs=f'bd_id = {bid} : i32'+(f', next_bd_id = {next_bid} : i32' if next_bid is not None else '')
    result+=f'      aie.dma_bd(%wi_cache_{row}_{part} : memref<{count}xi32> offset = 0 len = {count}) {{{attrs}}}\n'
    if release:
        name='cons' if release==21 else 'prod'
        value='%cache_repetitions' if release==21 else '%c1_i32'
        result+=f'      aie.use_lock(%qL2_0_cons_{name}_lock_{row}, Release, {value})\n'
    else:
        result+='      aie.use_lock(%wi_cache_dummy, Release, %cache_zero)\n'
    result+=f'      aie.next_bd ^bb{next_label}\n' if next_label is not None else '      aie.end\n'
    return result


def harness(parent_text,plan):
    allocations=placement_proof(parent_text)
    text=parent_text
    # The cache object is a complete per-core 16-packet query sequence. A ready
    # token authorizes one replay; 21 returned tokens authorize the next fill.
    text,count=re.subn(r'(%qL2_0_cons_prod_lock_[0-3] = aie.lock[^\n]+\{init = )2( : i32)',r'\g<1>21\2',text)
    if count!=4:raise ValueError('cache producer-lock census changed')
    start=text.index('    %memtile_dma_0_1 = aie.memtile_dma');end=region(text,start)
    body=text[start:end]
    body=body.replace('      %c1_i32 = arith.constant 1 : i32','      %cache_zero = arith.constant 0 : i32\n      %cache_repetitions = arith.constant 21 : i32\n      %c1_i32 = arith.constant 1 : i32',1)
    producer=''
    for row in range(4):
        for part in range(2):
            bid=row*2+part;last=bid==7
            producer+=bd(bid+1,bid,row,part,21 if part==0 else 0,21 if part==1 else 0,
                None if last else bid+2,None if last else bid+1)
    body=replace_blocks(body,1,9,producer)
    for row,(channel,first,after,first_bid,last_bid) in enumerate(((2,21,23,10,11),(3,26,28,30,31),(4,31,33,13,14),(5,36,38,33,34))):
        old=f'aie.dma_start(MM2S, {channel}, ^bb{first}, ^bb{after})'
        if body.count(old)!=1:raise ValueError('qualified query channel changed')
        body=body.replace(old,old[:-1]+', repeat_count = 20)',1)
        consumer=bd(first,first_bid,row,0,1,0,first+1,last_bid)+bd(first+1,last_bid,row,1,0,1)
        body=replace_blocks(body,first,after,consumer)
    buffers=''
    if re.search(r'aie.lock\(%mem_tile_0_1, 63\)',text):raise ValueError('cache dummy lock is already allocated')
    buffers+='    %wi_cache_dummy = aie.lock(%mem_tile_0_1, 63) {init = 0 : i32, sym_name = "wi_cache_dummy"}\n'
    for item in allocations:
        buffers+=f'    %{item["name"]} = aie.buffer(%mem_tile_0_1) {{address = {item["start"]} : i32, mem_bank = {item["start"]//65536} : i32, sym_name = "{item["name"]}"}} : memref<{item["words"]}xi32>\n'
    text=text[:start]+buffers+body+text[end:]
    start=text.index('    aie.runtime_sequence(');end=region(text,start)
    return text[:start]+runtime(plan)+text[end:],allocations


def external_cores(text,files):
    """Use CoreOp's external-ELF contract to avoid recompiling the FIFO wrapper."""
    for (col,row),filename in files.items():
        start=text.index(f'aie.core(%tile_{col}_{row})');left=text.index('{',start)
        end=region(text,start);attrs=text.index('{',end);after=region(text,attrs)
        replacement='{\n      aie.end\n    } '+f'{{stack_size = {q.STACK} : i32, elf_file = "{filename.as_posix()}"}}'
        text=text[:left]+replacement+text[after:]
    return text


def build(parent,out):
    parent=parent.absolute();out=out.absolute();meta=json.loads((parent/'build.json').read_text())
    if meta['sources']!=q.sources() or p.attention.sha(parent/'kernel.o')!=meta['kernel_object_sha256']:
        raise ValueError('qualified arithmetic parent changed')
    proof=inventory.inventory(parent);plan=q.sq.bm.b.plan()
    source=parent/'prj/input_with_addresses.mlir';text,allocations=harness(source.read_text(),plan)
    frozen=dict(inventory.phases.sources(),**{'tools/laya_wi_query_inventory.py':p.attention.sha(Path(inventory.__file__)),
        'kernels/laya_transformer/cached_wi.py':p.attention.sha(Path(__file__))})
    out.mkdir(parents=True,exist_ok=False);shutil.copyfile(parent/'kernel.o',out/'kernel.o')
    (out/'prj').mkdir();resident={};expected={tuple(row['core']):row for row in meta['audits']}
    for filename in (parent/'prj').rglob('*.elf'):
        match=re.search(r'core_(\d+)_(\d+)\.elf$',filename.name)
        if match is None:raise ValueError('unrecognized parent ELF')
        core=tuple(map(int,match.groups()))
        if core in resident or p.attention.sha(filename)!=expected[core]['elf_sha256']:raise ValueError('parent resident ELF identity')
        target=out/'prj'/f'frozen_core_{core[0]}_{core[1]}.elf';shutil.copyfile(filename,target);resident[core]=target
        # External ELFs are not relinked. Retain their qualified linker layout
        # as provenance for audit(), rather than claiming a new link occurred.
        layout=f'ldScripts_main_core_{core[0]}_{core[1]}.ld.script'
        shutil.copyfile(parent/'prj'/layout,out/'prj'/layout)
    if len(resident)!=16:raise ValueError('incomplete resident core set')
    text=external_cores(text,resident)
    (out/'harness.mlir').write_text(text,encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',q.STACK):all_audits=json.loads(json.dumps(p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
    actual={}
    for audit in all_audits:
        core=tuple(audit['core'])
        if audit!=expected[core]:raise ValueError('cache prototype changes qualified resident core ELF/allocation')
        actual[core]=audit
    audits=[actual[core] for core in sorted(actual)]
    if audits!=meta['audits']:raise ValueError('incomplete qualified resident core audit')
    transport=q.verify((out/'insts.bin').read_bytes(),events(plan),audits,plan['words'])
    if any(p.attention.sha(p.ROOT/name)!=sha for name,sha in frozen.items()):raise ValueError('cache source changed during build')
    result=dict(sources=frozen,parent_build_sha256=p.attention.sha(parent/'build.json'),parent_placed_ir_sha256=p.attention.sha(source),
        allocations=allocations,audits=audits,transport=transport,inventory=proof,
        resident_elfs={filename.name:p.attention.sha(filename) for filename in resident.values()},
        inherited_linker_layouts={filename.name:p.attention.sha(filename) for filename in (out/'prj').glob('ldScripts_main_core_*.ld.script')},
        kernel_object_sha256=p.attention.sha(out/'kernel.o'),instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
        policy=dict(active_columns=[0],operation='Wi only',kblocks=16,output_groups=21,cache_ready_tokens=21,consumer_repeat_count=20),
        scope='OFFLINE prototype only; compiled MemTile BD/lock/repeat proof and bounded device gates pending')
    p.attention.dump(out/'build.json',result)
    print('CACHED_WI_BUILD '+json.dumps(dict(build_sha256=p.attention.sha(out/'build.json'),transport=transport,
        core_elfs_identical=True,allocations=allocations,scope=result['scope'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args.parent,args.out)
