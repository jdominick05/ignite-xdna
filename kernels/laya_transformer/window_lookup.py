#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline two-output query reuse with the qualified GELU lookup arithmetic."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import window_mlp as window
import geglu_lut as lookup
sq=window.sq;n=window.n;p=window.p
WIDTH=2


def sources():
    return dict(window.sources(),**lookup.sources(),**{'kernels/laya_transformer/window_lookup.py':p.attention.sha(Path(__file__))})


def prefix():
    text=window.prefix()
    if text.count('memref<6144xi32>')==0:raise ValueError('window state declaration missing')
    return text.replace('memref<6144xi32>','memref<4096xi32>')


def runtime(plan,m):
    with patch.object(window,'WIDTH',WIDTH):return window.runtime(plan,m)


def lookup_source():
    text=Path(lookup.__file__).with_suffix('.cc').read_text()
    text=sq.replace(text,'void chain_geglu(const int32_t *q,const int32_t *v,int32_t *out,int poison)',
        'void chain_geglu_split(const int32_t *q,const int32_t *second,const int32_t *v,int32_t *out,int poison)')
    return sq.replace(text,'aie::load_v<16>(gate+row*64+col)',
        'aie::load_v<16>(row<16 ? gate+row*64+col : reinterpret_cast<const float *>(second)+(row-16)*64+col)')


def build(base,model,parent,out,m):
    out=out.absolute();frozen=sources();bundle=n.Bundle(base,model)
    meta=json.loads((parent/'build.json').read_text())
    if meta['sources']!=lookup.sources() or meta['base_identity']!=bundle.identity:
        raise ValueError('qualified lookup parent changed')
    if p.attention.sha(parent/'gelu_tables.cc')!=meta['table_source_sha256']:
        raise ValueError('qualified lookup tables changed')
    out.mkdir(parents=True,exist_ok=False)
    clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-O3','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
        '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}',
        f'-I{p.ROOT/"kernels/laya_transformer"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    generated={};objects=[];unchanged={}
    split=sq.kernel_sources()
    texts=dict(norm=split['norm'],dispatch=split['dispatch'],lookup=lookup_source(),
        window=Path(window.__file__).with_suffix('.cc').read_text(),tables=(parent/'gelu_tables.cc').read_text())
    for name,text in texts.items():
        source=out/(name+'.cc');source.write_text(text,encoding='utf-8');generated[name]=p.attention.sha(source)
        obj=out/(name+'.o');objects.append(obj)
        p.attention.command([clang,*flags,'-c',source,'-o',obj],out)
    for name in ('block','gemm'):
        obj=out/(name+'.o');shutil.copyfile(bundle.folders['vector']/(name+'.o'),obj)
        objects.append(obj);unchanged[name]=p.attention.sha(obj)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip():
        raise ValueError('unresolved window lookup helper')
    (out/'harness.mlir').write_text(prefix()+runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',window.STACK):audits=p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})
    cdo={file.name:p.attention.sha(file) for file in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen:raise ValueError('missing init or changed source')
    result=dict(sources=frozen,base_identity=bundle.identity,lookup_parent_sha256=p.attention.sha(parent/'build.json'),
        policy=dict(m=m,width=WIDTH,qwords=1040,kvwords=2064,q_depth=2,kv_depth=2,l2_q_depth=2,stack_bytes=window.STACK),
        generated_sources=generated,unchanged_objects=unchanged,audits=audits,cdo=cdo,
        table_source_sha256=meta['table_source_sha256'],instructions=p.attention.sha(out/'insts.bin'),
        xclbin=p.attention.sha(out/'final.xclbin'),kernel_object_sha256=p.attention.sha(out/'kernel.o'),
        scope='offline two-output MLP with inherited lookup; packet/lock primitive, device exactness and speed unqualified')
    p.attention.dump(out/'build.json',result)
    print('WINDOW_LOOKUP_BUILD '+json.dumps(dict(policy=result['policy'],
        code_bytes=sorted({a['code_bytes'] for a in audits}),stack_bound=max(sum(a['frames'])+256 for a in audits),
        scope=result['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','parent','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args();build(args.base,args.model,args.parent,args.out,args.m)
