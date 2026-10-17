# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-contained cached MLP query setup, independent of context-resume policy."""
import argparse
import json
from pathlib import Path
import re
import shutil
from unittest.mock import patch

import cached_mlp_checks as previous
c=previous.c
p=c.p
q=c.q
QUERY_BDS=c.NORMAL_BDS+tuple(range(16,22))+(36,37)


def registers(old,text):
    ready=previous.expected_states(old,text)
    for col,address,value in c.normal_registers(old,text):ready[(col<<25)|(1<<20)|address]=value
    result=[]
    for col in range(4):
        base=(col<<25)|(1<<20)
        for bid in QUERY_BDS:
            for index in range(8):
                address=0xa0000+32*bid+4*index
                result.append((col,address,ready[base+address]))
        # The measured parent's MemTile control words are zero; START_QUEUE
        # starts these channels. Preserve all channel controls exactly.
        if any(old[base+address]!=0 for address in (0xa0600,0xa0640,0xa0648,0xa0650,0xa0658)):
            raise ValueError('qualified query channel control differs')
    return result


def runtime(plan,old,text):
    seq=q.sq.bm.b.MlpSequence(plan['words'])
    for col,address,value in registers(old,text):
        a=seq.constant(address);v=seq.constant(value)
        seq.lines.append(f'      aiex.npu.write32({a}, {v}) {{column = {col} : i32, row = 1 : i32}} : i32, i32')
    # Reuse the proved phase generator; its constants have a private namespace.
    tail=c.runtime(plan,False)
    tail=re.sub(r'%([A-Za-z_][\w]*)',lambda match:match[0] if match[1] in ('a','o') else '%ph_'+match[1],tail)
    left=tail.index('{')+1;body=tail[left:]
    return '\n'.join(seq.lines)+body


def harness(parent_text,plan,old):
    text,allocations=c.harness(parent_text,plan,cold=False)
    for col in range(4):
        start=text.index(f'    %memtile_dma_{col}_1 = aie.memtile_dma');end=c.cache.region(text,start)
        body=text[start:end]
        pattern=r'(?:%\d+ = )?aie.dma_start\((S2MM|MM2S), (\d+), \^bb(\d+), \^bb(\d+)(?:, repeat_count = 20)?\)'
        changed=0
        def skip(match):
            nonlocal changed
            direction,channel,_,after=match.groups()
            if (direction=='S2MM' and channel=='0') or (direction=='MM2S' and channel in ('2','3','4','5')):
                changed+=1;return 'aie.next_bd ^bb'+after
            return match[0]
        body=re.sub(pattern,skip,body)
        if changed!=5:raise ValueError('query startup must be fully removed')
        text=text[:start]+body+text[end:]
    start=text.index('    aie.runtime_sequence(');end=c.cache.region(text,start)
    return text[:start]+runtime(plan,old,parent_text)+text[end:],allocations


def expected_init(old,text):
    result=previous.expected_states(old,text)
    for col in range(4):
        base=(col<<25)|(1<<20)
        for bid in QUERY_BDS:
            for index in range(8):result.pop(base+0xa0000+32*bid+4*index,None)
        for address in (0xa0604,0xa0644,0xa064c,0xa0654,0xa065c):result.pop(base+address,None)
        for address in (0xa0600,0xa0640,0xa0648,0xa0650,0xa0658):result.pop(base+address,None)
    return result


def runtime_proof(raw,old,text,plan,audits):
    prefix=registers(old,text);ops=previous.decoder.parse(raw)
    for op,(col,address,value) in zip(ops,prefix):
        if (op['op'],op.get('addr'),op.get('val'),op.get('val_low'),op['size'])!=('WRITE',(col<<25)|(1<<20)|address,value,0,24):
            raise ValueError('self-contained query descriptor/enable prefix differs')
    # Feed the remainder to the existing exact phase-boundary checker, retaining
    # the compiled transaction target and rebuilding only count/size framing.
    import struct
    body=b''.join(raw[op['offset']:op['offset']+op['size']] for op in ops[len(prefix):])
    size=16+len(body);padding=(-size)%64
    tail=raw[:8]+struct.pack('<2I',len(ops)-len(prefix),size+padding)+body+bytes(padding)
    proof=previous.runtime_proof(tail,text,old,plan,audits,False)
    return dict(**proof,query_setup_writes=len(prefix),context_resume_policy_independent=True,
        prerequisite='preceding finite query tasks completed; default and completed lock credits agree; no core or MemTile lock reset')


def verify(parent,folder):
    meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    original=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    if meta['parent_build_sha256']!=p.attention.sha(parent/'build.json') or original['sources']!=q.sources():raise ValueError('dispatch parent changed')
    for name,sha in meta['sources'].items():
        if p.attention.sha(p.ROOT/name)!=sha:raise ValueError('dispatch source changed')
    for filename,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
        if p.attention.sha(folder/filename)!=meta[key]:raise ValueError('dispatch artifact changed')
    for filename,sha in original['cdo'].items():
        if p.attention.sha(parent/'prj/cdo_main'/filename)!=sha:raise ValueError('parent CDO changed')
        if 'init' not in filename and (parent/'prj/cdo_main'/filename).read_bytes()!=(folder/'prj/cdo_main'/filename).read_bytes():
            raise ValueError('resident code/data/enable changed')
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    old=c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    new=c.checks.state(folder/'prj/cdo_main/main_aie_cdo_init.bin');wanted=expected_init(old,text)
    if new!=wanted:
        changed=[hex(address) for address in sorted(new.keys()|wanted.keys()) if new.get(address)!=wanted.get(address)]
        raise ValueError('unexpected self-contained CDO: '+','.join(changed[:12]))
    active=dict(new)
    for col,address,value in registers(old,text):active[(col<<25)|(1<<20)|address]=value
    plan=q.sq.bm.b.plan()
    return dict(entire_init_state_exact=True,words=previous.word_proof(active,old,text,plan),
        runtime=runtime_proof((folder/'insts.bin').read_bytes(),old,text,plan,original['audits']))


def build(parent,out):
    parent=parent.absolute();out=out.absolute()
    original=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    old=c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    text,allocations=harness(text,q.sq.bm.b.plan(),old)
    out.mkdir(parents=True,exist_ok=False);(out/'prj').mkdir();resident={}
    shutil.copyfile(parent/'kernel.o',out/'kernel.o')
    for audit in original['audits']:
        col,row=audit['core'];files=[path for path in (parent/'prj').rglob('*.elf') if re.search(rf'core_{col}_{row}\.elf$',path.name)]
        if len(files)!=1 or p.attention.sha(files[0])!=audit['elf_sha256']:raise ValueError('dispatch resident ELF parent changed')
        target=out/'prj'/f'frozen_core_{col}_{row}.elf';shutil.copyfile(files[0],target);resident[col,row]=target
        name=f'ldScripts_main_core_{col}_{row}.ld.script';shutil.copyfile(parent/'prj'/name,out/'prj'/name)
    text=c.cache.external_cores(text,resident);(out/'harness.mlir').write_text(text,encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',q.STACK):audits=json.loads(json.dumps(p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
    if audits!=original['audits']:raise ValueError('dispatch resident core/layout changed')
    sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__))
        for module in (c,previous,__import__(__name__))}
    meta=dict(parent_build_sha256=p.attention.sha(parent/'build.json'),sources=sources,allocations=allocations,audits=audits,
        instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'))
    p.attention.dump(out/'build.json',meta);proof=verify(parent,out)
    print('CACHED_MLP_DISPATCH_BUILD '+json.dumps(dict(build_sha256=p.attention.sha(out/'build.json'),proof=proof)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args.parent,args.out)
