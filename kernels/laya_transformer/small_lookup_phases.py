#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split the qualified smaller-query MLP at its existing completed phase waits."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import small_query_lookup as parent
p=parent.p
n=parent.n
PHASES=dict(wi=2,geglu=3,wo=9,norm=1)


def partition(events):
    result={name:[] for name in PHASES};names={mode:name for name,mode in PHASES.items()}
    current=None;seen=[];pending={}
    for event in events:
        if event['kind']=='wait':
            col=event['col']
            if current is None or pending.pop(col,set())!={'q','kv','out'}:
                raise ValueError('phase wait does not drain a complete queue group')
        else:
            mode=event['mode']
            if mode not in names:raise ValueError('unexpected phase mode')
            if current!=names[mode]:
                if pending:raise ValueError('phase boundary precedes completion')
                current=names[mode];seen.append(current)
            kinds=pending.setdefault(event['col'],set())
            if event['kind'] not in ('q','kv','out') or event['kind'] in kinds:
                raise ValueError('repeated or unknown queue before completion')
            kinds.add(event['kind'])
        result[current].append(event)
    if pending or seen!=list(PHASES):raise ValueError('incomplete or reordered phase stream')
    if [event for group in result.values() for event in group]!=events:
        raise ValueError('phase concatenation differs from qualified schedule')
    return result


def runtime(plan,phase):
    if phase not in PHASES:raise ValueError('unknown small lookup phase')
    events=partition(parent.events(plan))[phase]
    seq=parent.sq.bm.b.MlpSequence(plan['words'])
    first=next(event for event in events if event['kind']!='wait')
    seq.control(first['mode'],first['kblocks'])
    for event in events:
        if event['kind']=='wait':seq.wait_columns([event['col']])
        else:
            if (event['mode'],event['kblocks'])!=(first['mode'],first['kblocks']):
                raise ValueError('phase control changes inside selected stream')
            seq.dma(event['col'],event['kind'],event['offset'],event['dims'],event['strides'],event['output'])
    return seq.finish()


def sources():
    return dict(parent.sources(),**{'kernels/laya_transformer/small_lookup_phases.py':p.attention.sha(Path(__file__))})


def validate(folder,meta,plan,reference):
    if meta['sources']!=sources() or set(meta['layouts'])!=set(PHASES):raise ValueError('stale/incomplete phase build')
    groups=partition(parent.events(plan));proofs={}
    for phase,layout in meta['layouts'].items():
        if layout['audits']!=reference['audits'] or layout['cdo']!=reference['cdo']:
            raise ValueError('phase resident core/init changed')
        for file,key in [('kernel.o','kernel_object_sha256'),('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(folder/phase/file)!=layout[key]:raise ValueError('phase artifact changed')
        if layout['kernel_object_sha256']!=reference['kernel_object_sha256']:raise ValueError('phase arithmetic object changed')
        proof=parent.verify((folder/phase/'insts.bin').read_bytes(),groups[phase],layout['audits'],plan['words'])
        if proof!=layout['transport']:raise ValueError('phase transport proof changed')
        proofs[phase]=proof
    return proofs


def build(base,model,reference_folder,out):
    frozen=sources();bundle=n.Bundle(base,model);reference_folder=reference_folder.absolute();out=out.absolute()
    reference=json.loads((reference_folder/'build.json').read_text())
    if reference['sources']!=parent.sources() or reference['base_identity']!=bundle.identity:
        raise ValueError('qualified smaller-query parent changed')
    for file,key in [('kernel.o','kernel_object_sha256'),('insts.bin','instructions'),('final.xclbin','xclbin')]:
        if p.attention.sha(reference_folder/file)!=reference[key]:raise ValueError('qualified parent artifact changed')
    groups=partition(parent.events(bundle.plan))
    parent.verify((reference_folder/'insts.bin').read_bytes(),parent.events(bundle.plan),reference['audits'],bundle.plan['words'])
    out.mkdir(parents=True,exist_ok=False);layouts={}
    for phase in PHASES:
        folder=out/phase;folder.mkdir();shutil.copyfile(reference_folder/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(parent.prefix()+runtime(bundle.plan,phase),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
            '--dump-intermediates',f'--tmpdir={folder/"prj"}'],folder)
        with patch.object(p,'STACK',parent.STACK):
            audits=json.loads(json.dumps(p.audit(folder,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
        cdo={file.name:p.attention.sha(file) for file in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=reference['audits'] or cdo!=reference['cdo']:raise ValueError('phase compile changes qualified core/init')
        proof=parent.verify((folder/'insts.bin').read_bytes(),groups[phase],audits,bundle.plan['words'])
        layouts[phase]=dict(audits=audits,cdo=cdo,transport=proof,
            kernel_object_sha256=p.attention.sha(folder/'kernel.o'),
            instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('SMALL_PHASE_LAYOUT '+json.dumps(dict(phase=phase,transport=proof,core_and_init_identical=True)),flush=True)
    if sources()!=frozen:raise ValueError('phase source changed during build')
    meta=dict(sources=frozen,base_identity=bundle.identity,parent_build_sha256=p.attention.sha(reference_folder/'build.json'),
        m=128,layouts=layouts,scope='offline diagnostic phase streams; device layer/full parity and timing pending')
    validate(out,meta,bundle.plan,reference);p.attention.dump(out/'build.json',meta)
    print('SMALL_PHASE_BUILD '+json.dumps(dict(build_sha256=p.attention.sha(out/'build.json'),
        all_compiled_addresses_controls_and_packets_exact=True,scope=meta['scope'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('base','model','reference-folder','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args.base,args.model,args.reference_folder,args.out)
