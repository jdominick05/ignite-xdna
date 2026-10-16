#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recompile only vector-image GEMM; require the original runtime/data ABI."""
import argparse
import json
from pathlib import Path
import shutil

import native as n
p=n.p


def sources():
    return dict(n.vector.sources(),**{'kernels/laya_transformer/gemm_tune.py':p.attention.sha(Path(__file__))})


def build(base,model,out,opt):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model); vm=bundle.meta['vector']
    out.mkdir(parents=True,exist_ok=False)
    choices={'O2':['-O2'],'O3':['-O3'],'O3_no_unroll':['-O3','-fno-unroll-loops']}
    flags=[*choices[opt],'-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    clang=p.attention.PEANO/'bin/clang++.exe'
    p.attention.command([clang,*flags,'-c',n.c.b.HERE/'gemm.cc','-o',out/'gemm.o'],out)
    objects=[]; unchanged={}
    for name in ('vector','stem','block','gemm','dispatch','norm'):
        if name!='gemm':
            shutil.copyfile(bundle.folders['vector']/(name+'.o'),out/(name+'.o'))
            unchanged[name]=p.attention.sha(out/(name+'.o'))
        objects.append(out/(name+'.o'))
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved tuned vector helper')
    (out/'harness.mlir').write_text(n.vector.prefix()+n.stem.runtime(vm['tokens'],vm['output_words'],vm['target']),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=5376; audits=json.loads(json.dumps(p.audit(out,{})))
    finally: p.STACK=before
    if [a['spans'] for a in audits]!=[a['spans'] for a in vm['audits']]: raise ValueError('tuned vector physical buffers changed')
    if n.c.txn.registers(audits)!=n.c.txn.registers(vm['audits']): raise ValueError('tuned vector control allocations changed')
    if p.attention.sha(out/'insts.bin')!=vm['instructions']: raise ValueError('tuned vector runtime instructions differ')
    if sources()!=frozen: raise ValueError('tuning source changed during build')
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3: raise ValueError('missing tuned initialization')
    meta=dict(sources=frozen,base_identity=bundle.identity,opt=opt,flags=flags,unchanged_objects=unchanged,audits=audits,cdo=cdo,
      instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),gemm_object_sha256=p.attention.sha(out/'gemm.o'),
      compiler_sha256=p.attention.sha(clang),scope='offline candidate build only; numerical parity and speed unverified')
    p.attention.dump(out/'build.json',meta)
    print('GEMM_TUNE_BUILD '+json.dumps(dict(opt=opt,code_bytes=sorted({a['code_bytes'] for a in audits}),
      stack_bound=max(sum(a['frames'])+256 for a in audits),physical_buffers_identical=True,runtime_instructions_identical=True,
      sources=frozen,scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--opt',choices=('O2','O3','O3_no_unroll'),default='O3')
    args=ap.parse_args(); build(args.base,args.model,args.out,args.opt)
