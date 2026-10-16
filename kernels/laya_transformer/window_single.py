#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-output schedule on the unchanged two-state, single-KV window core."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import window_lookup as parent
import window_mlp_transport as transport
p=parent.p


def runtime(plan,m):
    with patch.object(parent,'WIDTH',1):return parent.runtime(plan,m)


def verify(raw,plan,audits):
    with patch.object(parent,'WIDTH',1):return transport.verify(raw,plan,audits)


def sources():
    return dict(parent.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file)
        for file in (Path(__file__),Path(transport.__file__))})


def validate(folder,bundle):
    meta=json.loads((folder/'build.json').read_text())
    expected=dict(m=128,width=1,qwords=1040,kvwords=2064,q_depth=2,kv_depth=1,l2_kv_depth=2,l2_q_depth=2,stack_bytes=3328)
    if meta['sources']!=sources() or meta['base_identity']!=bundle.identity or meta['policy']!=expected:
        raise ValueError('stale single-output window image')
    for name,sha in meta['generated_sources'].items():
        if p.attention.sha(folder/(name+'.cc'))!=sha:raise ValueError('single-output generated source changed')
    for file,key in [('kernel.o','kernel_object_sha256'),('insts.bin','instructions'),('final.xclbin','xclbin')]:
        if p.attention.sha(folder/file)!=meta[key]:raise ValueError('single-output artifact changed')
    if verify((folder/'insts.bin').read_bytes(),bundle.plan,meta['audits'])!=meta['transport']:
        raise ValueError('single-output compiled transport changed')
    return meta


def build(base,model,source,out):
    frozen=sources();bundle=parent.n.Bundle(base,model);old=json.loads((source/'build.json').read_text())
    if old['sources']!=parent.sources() or old['base_identity']!=bundle.identity:
        raise ValueError('window parent changed')
    if p.attention.sha(source/'kernel.o')!=old['kernel_object_sha256']:raise ValueError('window parent core changed')
    out=out.absolute();out.mkdir(parents=True,exist_ok=False);shutil.copyfile(source/'kernel.o',out/'kernel.o')
    for name,sha in old['generated_sources'].items():
        file=source/(name+'.cc')
        if p.attention.sha(file)!=sha:raise ValueError('window parent generated source changed')
        shutil.copyfile(file,out/file.name)
    (out/'harness.mlir').write_text(parent.prefix()+runtime(bundle.plan,128),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',3328):audits=json.loads(json.dumps(p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
    cdo={file.name:p.attention.sha(file) for file in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if audits!=old['audits'] or cdo!=old['cdo']:raise ValueError('single-output changes window core/init')
    proof=verify((out/'insts.bin').read_bytes(),bundle.plan,audits)
    if frozen!=sources():raise ValueError('single-output source changed during build')
    meta=dict(old,sources=frozen,policy=dict(old['policy'],width=1),audits=audits,cdo=cdo,transport=proof,
        parent_sha256=p.attention.sha(source/'build.json'),instructions=p.attention.sha(out/'insts.bin'),
        xclbin=p.attention.sha(out/'final.xclbin'),scope='same window core/init, one output at a time; numeric and speed gates pending')
    p.attention.dump(out/'build.json',meta)
    print('WINDOW_SINGLE_BUILD '+json.dumps(dict(policy=meta['policy'],transport=proof,parent_core_init_identical=True,
        sources=frozen,build_sha256=p.attention.sha(out/'build.json'),scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','source','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.base,args.model,args.source,args.out)
