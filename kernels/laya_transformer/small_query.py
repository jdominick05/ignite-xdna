#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolated MLP image with 1040-word queries and unchanged full weight packets."""
import argparse
import json
from pathlib import Path
import re
import shutil
from unittest.mock import patch

import geglu_fifo as gf
bm=gf.gt.bm; n=gf.gt.n; p=gf.p
QWORDS=1040


def replace(text,old,new,count=1):
    if text.count(old)!=count: raise ValueError('small-query source contract: '+old[:80])
    return text.replace(old,new)


def sources():
    return dict(gf.sources(),**{'kernels/laya_transformer/small_query.py':p.attention.sha(Path(__file__))})


def kernel_sources():
    """Change input addressing only; retain reduction, polynomial and rounding order."""
    norm=(p.ROOT/'kernels/laya_norm/kernel.cc').read_text()
    norm=replace(norm,'void residual_norm(const int32_t *in,const float *affine,int32_t *out)',
      'void residual_norm_split(const int32_t *in,const int32_t *second,const float *affine,int32_t *out)')
    norm=replace(norm,'*r=x+1040;','*r=reinterpret_cast<const float *>(second);')
    dispatch=(p.ROOT/'kernels/laya_transformer/dispatch.cc').read_text()
    dispatch=replace(dispatch,'void residual_norm(const int32_t *,const float *,int32_t *);',
      'void residual_norm_split(const int32_t *,const int32_t *,const float *,int32_t *);')
    dispatch=replace(dispatch,'void command_norm(const int32_t *q,const int32_t *affine,int32_t *out)',
      'void command_norm_split(const int32_t *q,const int32_t *second,const int32_t *affine,int32_t *out)')
    dispatch=replace(dispatch,'residual_norm(q,reinterpret_cast<const float *>(affine),out);',
      'residual_norm_split(q,second,reinterpret_cast<const float *>(affine),out);')
    mlp=(p.ROOT/'kernels/laya_transformer/mlp.cc').read_text()
    mlp=replace(mlp,'void chain_geglu(const int32_t *q,const int32_t *v,int32_t *out,int poison)',
      'void chain_geglu_split(const int32_t *q,const int32_t *second,const int32_t *v,int32_t *out,int poison)')
    mlp=replace(mlp,'aie::load_v<16>(gate+row*64+col)',
      'aie::load_v<16>(row<16 ? gate+row*64+col : reinterpret_cast<const float *>(second)+(row-16)*64+col)')
    return dict(norm=norm,dispatch=dispatch,mlp=mlp)


def prefix():
    text=gf.prefix()
    # Change only query declarations/uses; KV retains all 2064 words.
    lines=text.splitlines()
    for index,line in enumerate(lines):
        if 'aie.objectfifo @qL2_' in line:
            lines[index]=line.replace('8256xi32','4160xi32')
        elif 'aie.objectfifo @qF_' in line:
            lines[index]=line.replace(',2 : i32)',',4 : i32)').replace('2064xi32','1040xi32')
        elif 'aie.objectfifo.link [@qL2_' in line:
            lines[index]=replace(line,'[0, 2064, 4128, 6192]','[0, 1040, 2080, 3120]')
    text='\n'.join(lines)+'\n'
    for name in ('chain_gemm_step','gemm_step'):
        text=replace(text,'@'+name+'(memref<2064xi32>,','@'+name+'(memref<1040xi32>,')
    for old,new,tail in [('command_norm','command_norm_split','memref<2080xi32>'),
      ('chain_geglu','chain_geglu_split','memref<2080xi32>,i32')]:
        text=replace(text,f'@{old}(memref<2064xi32>,memref<2064xi32>,{tail})',
          f'@{new}(memref<1040xi32>,memref<1040xi32>,memref<2064xi32>,{tail})')
    for i in range(16):
        start=text.index(f'    %core{i} = aie.core'); end=text.index(' {link_with',start)
        core=text[start:end]
        first=core.index('        %sq ='); last=core.index(f'        aie.objectfifo.release @qF_{i}(Consume,1)')
        end_release=last+len(f'        aie.objectfifo.release @qF_{i}(Consume,1)')
        body=core[first:last]
        body=replace(body,f'        %override = memref.load %control{i}[%c0] : memref<16xi32>\n','')
        body=replace(body,f'        %sq = aie.objectfifo.acquire @qF_{i}(Consume,1) : !aie.objectfifosubview<memref<2064xi32>>\n','')
        body=replace(body,'        %q = aie.objectfifo.subview.access %sq[0] : !aie.objectfifosubview<memref<2064xi32>> -> memref<2064xi32>\n','')
        # Only lines referring to %q have their first argument/type changed.
        body='\n'.join(line.replace('memref<2064xi32>','memref<1040xi32>',1) if '%q' in line else line
          for line in body.splitlines())+'\n'
        body=replace(body,'@command_norm(%q,%kv,%o) : (memref<1040xi32>,',
          '@command_norm_split(%q,%q1,%kv,%o) : (memref<1040xi32>,memref<1040xi32>,')
        body=replace(body,'@chain_geglu(%q,%kv,%go,%fill) : (memref<1040xi32>,',
          '@chain_geglu_split(%q,%q1,%kv,%go,%fill) : (memref<1040xi32>,memref<1040xi32>,')
        branches=[]
        for count in (1,2):
            acquire=f'        %sq = aie.objectfifo.acquire @qF_{i}(Consume,2) : !aie.objectfifosubview<memref<1040xi32>>\n' if count==2 else ''
            view='%sq' if count==2 else '%sq_initial'
            acquire+=f'        %q = aie.objectfifo.subview.access {view}[0] : !aie.objectfifosubview<memref<1040xi32>> -> memref<1040xi32>\n'
            # In the GEMM branch the unused second pointer aliases the first.
            acquire+=f'        %q1 = aie.objectfifo.subview.access {view}[{count-1}] : !aie.objectfifosubview<memref<1040xi32>> -> memref<1040xi32>\n'
            branches.append(acquire+body+f'        aie.objectfifo.release @qF_{i}(Consume,{count})\n')
        selection=f'''        %sq_initial = aie.objectfifo.acquire @qF_{i}(Consume,1) : !aie.objectfifosubview<memref<1040xi32>>
        %override = memref.load %control{i}[%c0] : memref<16xi32>
        %query_gemm = arith.cmpi eq, %override,%two : i32
        %query_tiled = arith.cmpi eq, %override,%nine : i32
        %query_single = arith.ori %query_gemm,%query_tiled : i1
        scf.if %query_single {{
{branches[0]}        }} else {{
{branches[1]}        }}'''
        core=core[:first]+selection+core[end_release:]
        text=text[:start]+core+text[end:]
    return text


def runtime(plan,m):
    original=bm.b.MlpSequence
    class SmallSequence(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs); self.pending=[]
        def control(self,mode,*args,**kwargs):
            if self.pending: raise ValueError('unflushed query batch')
            self.query_mode=mode
            return super().control(mode,*args,**kwargs)
        def dma(self,col,kind,offset,dims,strides,*args,**kwargs):
            if self.query_mode in (1,3):
                self.pending.append((col,kind,offset,list(dims),list(strides),args,kwargs)); return
            if kind=='q':
                if dims[-1]!=p.QWORDS: raise ValueError('unexpected query extent')
                dims=list(dims); dims[-1]=QWORDS
            return super().dma(col,kind,offset,dims,strides,*args,**kwargs)
        def flush(self):
            by_col={}
            for entry in self.pending: by_col.setdefault(entry[0],{})[entry[1]]=entry
            self.pending=[]
            groups=None
            for rows in by_col.values():
                if set(rows)!={'q','kv','out'}: raise ValueError('incomplete split query batch')
                query=rows['q']; count=1
                for size in query[3][:-1]: count*=size
                if count%4 or query[3][-1]!=2064: raise ValueError('invalid split query group')
                if groups is not None and groups!=count//4: raise ValueError('unequal split groups')
                groups=count//4
            # A four-dimensional Shim descriptor cannot encode groups, halves,
            # cores and the factored 1040-word span together. Complete each group
            # before reusing its descriptor, preserving full FIFO-object boundaries.
            for group in range(groups or 0):
                for col,rows in by_col.items():
                    q=rows['q']; out=rows['out']; kv=rows['kv']
                    super().dma(col,'q',q[2]+group*4*p.OUT,[2,4,65,16],[QWORDS,p.OUT,16,1])
                    super().dma(col,'out',out[2]+group*4*p.OUT,[1,1,4*p.OUT//16,16],[0,0,16,1],*out[5],**out[6])
                    if self.query_mode==1:
                        super().dma(col,'kv',kv[2],[1,1,1,p.KVWORDS],[0,0,0,1])
                    else:
                        super().dma(col,'kv',kv[2]+group*4*p.OUT,[1,1,4,p.KVWORDS],[0,0,p.OUT,1])
                super().wait_columns(list(by_col))
            return bool(by_col)
        def wait(self):
            if not self.flush(): return super().wait()
        def wait_columns(self,columns):
            if not self.flush(): return super().wait_columns(columns)
    with patch.object(bm.b,'MlpSequence',SmallSequence): return bm.runtime(plan,m)


def build(base,model,out,m):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False)
    clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-O3','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    objects=[]; generated={}; unchanged={}
    for name,content in kernel_sources().items():
        source=out/(name+'.cc'); source.write_text(content,encoding='utf-8'); generated[name]=p.attention.sha(source)
        obj=out/(name+'.o'); objects.append(obj)
        p.attention.command([clang,*flags,'-c',source,'-o',obj],out)
    for name in ('block','gemm'):
        obj=out/(name+'.o'); shutil.copyfile(bundle.folders['vector']/(name+'.o'),obj)
        objects.append(obj); unchanged[name]=p.attention.sha(obj)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved small-query helper')
    (out/'harness.mlir').write_text(prefix()+runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',5376): audits=p.audit(out,{})
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen: raise ValueError('missing initialization or changed source')
    meta=dict(sources=frozen,base_identity=bundle.identity,policy=dict(m=m,qwords=QWORDS,kvwords=2064,q_depth=4,kv_depth=2,l2_q_depth=2),
      flags=flags,generated_sources=generated,unchanged_objects=unchanged,audits=audits,cdo=cdo,
      instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),kernel_object_sha256=p.attention.sha(out/'kernel.o'),
      scope='offline small-query MLP; unchanged mathematical expressions, hardware parity and speed unverified')
    p.attention.dump(out/'build.json',meta)
    print('SMALL_QUERY_BUILD '+json.dumps(dict(policy=meta['policy'],code_bytes=sorted({row['code_bytes'] for row in audits}),
      stack_bound=max(sum(row['frames'])+256 for row in audits),scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m)
