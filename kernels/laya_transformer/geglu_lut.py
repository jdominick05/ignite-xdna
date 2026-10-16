#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compile a separate interpolated GeGLU approximation; no accuracy/speed claim."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import geglu_fifo as gf
n=gf.gt.n; p=gf.p


def sources():
    return dict(gf.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file)
      for file in (Path(__file__),Path(__file__).with_suffix('.cc'))})


def samples():
    return np.asarray([.5*math.erfc(index/(32*math.sqrt(2))) for index in range(512)],np.float32)


def physical_table():
    # Same four-entry duplication used by the qualified FP32 exp lookup.
    return np.repeat(samples().reshape(-1,4),2,axis=0).reshape(-1)


def approximation(values):
    """Offline FP32 formula including fixed-point fraction quantization."""
    values=np.asarray(values,np.float32); magnitude=np.abs(values)
    bounded=np.minimum(magnitude,np.float32(15.9375))
    fixed=np.floor(bounded.astype(np.float64)*65536).astype(np.int32)
    index=fixed>>11; fraction=((fixed&2047).astype(np.float32)*np.float32(1/2048))
    table=samples(); low=table[index]; high=table[index+1]
    tail=low+fraction*(high-low)
    tail=np.where(magnitude>=np.float32(15.9375),np.float32(0),tail)
    cdf=np.where(values<0,tail,np.float32(1)-tail)
    return values*cdf


def build(base,model,out,m):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False)
    values=','.join(float(value).hex()+'f' for value in physical_table())
    table=out/'gelu_tables.cc'
    table.write_text('extern "C" {\nalignas(32) float gelu_tail_ab[1024]={'+values+'};\nalignas(32) float gelu_tail_cd[1024]={'+values+'};\n}\n',encoding='utf-8')
    clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-O3','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    objects=[]
    for name,source in [('geglu_lut',Path(__file__).with_suffix('.cc')),('tables',table)]:
        obj=out/(name+'.o'); objects.append(obj)
        p.attention.command([clang,*flags,'-c',source,'-o',obj],out)
    unchanged={}
    for name in ('block','gemm','dispatch','norm'):
        obj=out/(name+'.o'); shutil.copyfile(bundle.folders['vector']/(name+'.o'),obj)
        objects.append(obj); unchanged[name]=p.attention.sha(obj)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved GeGLU LUT helper')
    (out/'harness.mlir').write_text(gf.prefix()+gf.gt.bm.runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen: raise ValueError('missing init or changed source')
    points=np.linspace(-16,16,200001,dtype=np.float32)
    reference=np.fromiter((float(x)*.5*math.erfc(-float(x)/math.sqrt(2)) for x in points),np.float64)
    error=np.abs(approximation(points).astype(np.float64)-reference)
    meta=dict(sources=frozen,base_identity=bundle.identity,policy=dict(m=m,q_depth=2,kv_depth=2,l2_q_depth=2,
      approximation='512 FP32 Gaussian tail-CDF entries at step1/32; Q16 fractional indexing; linear interpolation; asymptote at abs15.9375'),
      flags=flags,unchanged_objects=unchanged,audits=audits,cdo=cdo,
      instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
      kernel_object_sha256=p.attention.sha(out/'kernel.o'),table_source_sha256=p.attention.sha(table),compiler_sha256=p.attention.sha(clang),
      offline_formula=dict(points=len(points),max_abs_gelu_error=float(error.max()),input_at_max=float(points[error.argmax()])),
      scope='compile and offline formula only; changed approximation, hardware numerical accuracy and performance unverified')
    p.attention.dump(out/'build.json',meta)
    print('GEGLU_LUT_BUILD '+json.dumps(dict(policy=meta['policy'],offline_formula=meta['offline_formula'],
      code_bytes=sorted({row['code_bytes'] for row in audits}),stack_bound=max(sum(row['frames'])+256 for row in audits),scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m)
