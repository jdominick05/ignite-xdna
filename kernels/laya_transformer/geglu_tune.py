#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dedicated encoder MLP image; unchanged GeGLU source, isolated compiler flags."""
import argparse
import json
from pathlib import Path
import re
import shutil

import bucket_mlp as bm
n=bm.native; p=bm.p


def sources():
    return dict(bm.sources(),**{'kernels/laya_transformer/geglu_tune.py':p.attention.sha(Path(__file__))})


def prefix():
    text=bm.b.mlp_prefix()
    old='%is_attention = arith.cmpi eq, %command, %zero : i32'
    if text.count(old)!=16: raise ValueError('MLP attention branch changed')
    text=text.replace(old,'%is_attention = arith.constant false')
    text,count=re.subn(r'^.*func.call @attention_shared\([^\n]*\n','',text,flags=re.M)
    if count!=16: raise ValueError('dead attention call coverage')
    text,count=re.subn(r'^.*%(?:scratch|mask)\d+ = aie.buffer[^\n]*\n','',text,flags=re.M)
    if count!=32 or re.search(r'%(?:scratch|mask)\d+',text): raise ValueError('dead attention buffer coverage')
    if text.count('stack_size = 4608 : i32')!=16: raise ValueError('MLP stack changed')
    return text.replace('stack_size = 4608 : i32','stack_size = 5376 : i32')


def build(base,model,out,m,opt):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False)
    flags=[{'Oz':'-Oz','O3':'-O3','O3_no_unroll':'-O3'}[opt]]
    if opt=='O3_no_unroll': flags.append('-fno-unroll-loops')
    flags+=['-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    clang=p.attention.PEANO/'bin/clang++.exe'
    p.attention.command([clang,*flags,'-c',Path(__file__).with_name('mlp.cc'),'-o',out/'mlp.o'],out)
    objects=[out/'mlp.o']; unchanged={}
    for name in ('block','gemm','dispatch','norm'):
        obj=out/(name+'.o'); shutil.copyfile(bundle.folders['vector']/(name+'.o'),obj)
        objects.append(obj); unchanged[name]=p.attention.sha(obj)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved MLP helper')
    (out/'harness.mlir').write_text(prefix()+bm.runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen: raise ValueError('missing initialization or changed source')
    meta=dict(sources=frozen,base_identity=bundle.identity,policy=dict(m=m,opt=opt),flags=flags,unchanged_objects=unchanged,
      audits=audits,cdo=cdo,instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
      kernel_object_sha256=p.attention.sha(out/'kernel.o'),compiler_sha256=p.attention.sha(clang),
      scope='unchanged MLP source; dedicated dispatch and compiler candidate; parity and speed unverified')
    p.attention.dump(out/'build.json',meta)
    print('GEGLU_TUNE_BUILD '+json.dumps(dict(policy=meta['policy'],code_bytes=sorted({a['code_bytes'] for a in audits}),
      stack_bound=max(sum(a['frames'])+256 for a in audits),unchanged_objects=unchanged,scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    ap.add_argument('--opt',choices=('Oz','O3','O3_no_unroll'),default='O3')
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m,args.opt)
