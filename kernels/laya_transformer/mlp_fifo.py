#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline MLP FIFO candidates: remove dead attention buffers, vary buffering."""
import argparse
import json
from pathlib import Path
import re
import shutil

import bucket_mlp as bm
n=bm.native; p=bm.p


def sources():
    return dict(bm.sources(),**{'kernels/laya_transformer/mlp_fifo.py':p.attention.sha(Path(__file__))})


def prefix(q_depth,kv_depth,l2_q_depth):
    if any(value not in (1,2) for value in (q_depth,kv_depth,l2_q_depth)): raise ValueError('FIFO depth')
    text=n.vector.prefix()
    # The vector prefix makes is_attention a constant false. Only that dead branch
    # references scratch/mask; keep all live GEMM/nonlinear/norm code unchanged.
    if text.count('%is_attention = arith.constant false')!=16: raise ValueError('vector attention branch changed')
    text,count=re.subn(r'^.*func.call @attention_shared\([^\n]*\n','',text,flags=re.M)
    if count!=16: raise ValueError('dead attention call coverage')
    text,count=re.subn(r'^.*%(?:scratch|mask)\d+ = aie.buffer[^\n]*\n','',text,flags=re.M)
    if count!=32: raise ValueError('dead attention buffer coverage')
    if re.search(r'%(?:scratch|mask)\d+',text): raise ValueError('remaining attention buffer use')
    for family,depth,expected in [('qF',q_depth,16),('qL2',l2_q_depth,4),('kvF',kv_depth,4),('kvL2',kv_depth,4)]:
        text,count=re.subn(r'(aie.objectfifo @'+family+r'_\d+\([^\n]*),1 : i32\)',lambda match:match[1]+f',{depth} : i32)',text)
        if count!=expected: raise ValueError('FIFO declaration coverage: '+family)
    return text


def build(base,model,out,m,q_depth,kv_depth,l2_q_depth):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False)
    # Reuse the exact original vector kernel object. Only the MLIR FIFO plumbing changes.
    shutil.copyfile(bundle.folders['vector']/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(prefix(q_depth,kv_depth,l2_q_depth)+bm.runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3: raise ValueError('missing FIFO candidate initialization')
    if sources()!=frozen: raise ValueError('FIFO source changed during build')
    policy=dict(m=m,q_depth=q_depth,kv_depth=kv_depth,l2_q_depth=l2_q_depth,dead_attention_buffers_removed=True)
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,policy=policy,audits=audits,cdo=cdo,
      instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
      kernel_object_sha256=p.attention.sha(out/'kernel.o'),scope='offline buffering candidate; numerical parity and performance unverified'))
    print('MLP_FIFO_BUILD '+json.dumps(dict(policy=policy,code_bytes=sorted({a['code_bytes'] for a in audits}),
      stack_bound=max(sum(a['frames'])+256 for a in audits),scope='compile/audit only; no speed or numerical claim')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    for name in ('q-depth','kv-depth','l2-q-depth'): ap.add_argument('--'+name,type=int,choices=(1,2),default=2)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m,args.q_depth,args.kv_depth,args.l2_q_depth)
