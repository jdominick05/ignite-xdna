#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline MLP window experiment: reuse each small Q across up to three N tiles."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import small_query as sq
bm=sq.bm; n=sq.n; p=sq.p
WIDTH=3
STACK=3328


def sources():
    return dict(sq.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file)
      for file in (Path(__file__),Path(__file__).with_suffix('.cc'))})


def prefix():
    text=sq.prefix().replace('memref<2048xi32>','memref<6144xi32>')
    text=sq.replace(text,'stack_size = 5376 : i32',f'stack_size = {STACK} : i32',16)
    for i in range(16):
        old=f'aie.objectfifo @qF_{i}(%m{i//4},{{%t{i}}},4 : i32)'
        text=sq.replace(text,old,old.replace(',4 : i32)',',2 : i32)'))
    text=sq.replace(text,'func.func private @chain_gemm_step(memref<1040xi32>,memref<2064xi32>,memref<6144xi32>,i32)',
      'func.func private @window_gemm_step(memref<1040xi32>,memref<2064xi32>,memref<6144xi32>,i32,i32)')
    text=sq.replace(text,'func.func private @chain_finish(memref<6144xi32>,memref<2080xi32>,i32)',
      'func.func private @window_finish(memref<6144xi32>,memref<2080xi32>,i32,i32)')
    for i in range(16):
        start=text.index(f'    %core{i} = aie.core'); end=text.index(' {link_with',start)
        core=text[start:end]
        core=sq.replace(core,'%c7 = arith.constant 7 : index','%c6 = arith.constant 6 : index\n      %c7 = arith.constant 7 : index')
        core=sq.replace(core,'%non_attention_count = arith.select %is_geglu,%c4,%c1 : index',
          f'''%ordinary_count = arith.select %is_geglu,%c4,%c1 : index
        %width_i = memref.load %control{i}[%c6] : memref<16xi32>
        %width = arith.index_cast %width_i : i32 to index
        %non_attention_count = arith.select %is_gemm,%width,%ordinary_count : index''',2)
        core=sq.replace(core,'scf.for %k = %c0 to %count step %c1 {',
          'scf.for %k = %c0 to %count step %c1 {\n          %slot = arith.index_cast %k : index to i32',2)
        core=sq.replace(core,f'@chain_gemm_step(%q,%kv,%state{i},%reset) : (memref<1040xi32>,memref<2064xi32>,memref<6144xi32>,i32)',
          f'@window_gemm_step(%q,%kv,%state{i},%reset,%slot) : (memref<1040xi32>,memref<2064xi32>,memref<6144xi32>,i32,i32)',2)
        core=sq.replace(core,f'@chain_finish(%state{i},%o,%tiled) : (memref<6144xi32>,memref<2080xi32>,i32)',
          f'@window_finish(%state{i},%o,%tiled,%slot) : (memref<6144xi32>,memref<2080xi32>,i32,i32)',2)
        core=sq.replace(core,'        scf.if %emit {',
          '        scf.if %emit {\n          scf.for %output_slot = %c0 to %width step %c1 {\n            %slot = arith.index_cast %output_slot : index to i32',2)
        core=sq.replace(core,f'          aie.objectfifo.release @outF_{i}(Produce,1)\n        }}\n        aie.objectfifo.release @qF_{i}',
          f'          aie.objectfifo.release @outF_{i}(Produce,1)\n          }}\n        }}\n        aie.objectfifo.release @qF_{i}',2)
        text=text[:start]+core+text[end:]
    return text


def runtime(plan,m):
    if m not in (128,256,384,512): raise ValueError('window MLP row bucket')
    original=bm.b.MlpSequence
    class WindowSequence(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs); self.window_mode=None; self.window_pending=[]
        def width(self,value):
            if not 1<=value<=WIDTH: raise ValueError('window width')
            if self.current.get(6)==value: return
            self.current[6]=value
            name=self.constant(value)
            for i in range(16):
                self.lines.append(f'      "aiex.npu.rtp_write"({name}) {{buffer = @chain_control_{i}, index = 6 : ui32}} : (i32) -> ()')
        def control(self,mode,*args,**kwargs):
            self.flush_window()
            self.window_mode=mode
            super().control(mode,*args,**kwargs)
            if mode not in (2,9): self.width(1)
        def dma(self,*args,**kwargs):
            if self.window_mode in (2,9): self.window_pending.append((args,kwargs)); return
            return super().dma(*args,**kwargs)
        def wait(self):
            if self.window_mode not in (2,9): return super().wait()
        def flush_window(self):
            if not self.window_pending: return
            mode=self.window_mode; groups=21 if mode==2 else 4; kblocks=16 if mode==2 else 41
            expected=groups*(m//128)*4*3
            if len(self.window_pending)!=expected: raise ValueError('unexpected source GEMM schedule')
            self.window_pending=[]; off=plan['offsets']; pitch=16*p.OUT
            for first in range(0,groups,WIDTH):
                width=min(WIDTH,groups-first); self.width(width)
                for mg in range(m//128):
                    for col in range(4):
                        qbase=off['packed_norm' if mode==2 else 'mlp_geglu']+mg*4*p.OUT
                        super().dma(col,'q',qbase,[1,kblocks,4,sq.QWORDS],[0,pitch,p.OUT,1])
                        if mode==2:
                            obase=off['mlp_wi']+((col*21+first)*4+mg)*4*p.OUT
                            super().dma(col,'out',obase,[width,4,p.OUT//16,16],[16*p.OUT,p.OUT,16,1],True)
                            wbase=off['mlp_wi_weight']+(col*21+first)*16*p.KVWORDS
                        else:
                            obase=off['norm']+mg*128*p.OUT+(first*4+col)*65
                            super().dma(col,'out',obase,[width,4,32,65],[4*65,32*p.OUT,p.OUT,1],True)
                            wbase=off['mlp_wo_weight']+(col*4+first)*41*p.KVWORDS
                        # K-major, slot-minor: full weight packets stay intact.
                        super().dma(col,'kv',wbase,[kblocks,width,p.KVWORDS//16,16],[p.KVWORDS,kblocks*p.KVWORDS,16,1])
                    super().wait()
        def finish(self):
            self.flush_window(); return super().finish()
    with patch.object(bm.b,'MlpSequence',WindowSequence): return sq.runtime(plan,m)


def build(base,model,out,m):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False)
    clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-O3','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    objects=[]; generated={}; unchanged={}
    source_text=dict(sq.kernel_sources(),window=Path(__file__).with_suffix('.cc').read_text())
    for name,content in source_text.items():
        source=out/(name+'.cc'); source.write_text(content,encoding='utf-8'); generated[name]=p.attention.sha(source)
        obj=out/(name+'.o'); objects.append(obj)
        p.attention.command([clang,*flags,'-c',source,'-o',obj],out)
    for name in ('block','gemm'):
        obj=out/(name+'.o'); shutil.copyfile(bundle.folders['vector']/(name+'.o'),obj)
        objects.append(obj); unchanged[name]=p.attention.sha(obj)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved window helper')
    (out/'harness.mlir').write_text(prefix()+runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',STACK): audits=p.audit(out,{})
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen: raise ValueError('missing init or changed window source')
    meta=dict(sources=frozen,base_identity=bundle.identity,policy=dict(m=m,width=WIDTH,qwords=sq.QWORDS,kvwords=2064,q_depth=2,kv_depth=2,l2_q_depth=2,stack_bytes=STACK),
      generated_sources=generated,unchanged_objects=unchanged,flags=flags,audits=audits,cdo=cdo,
      instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),kernel_object_sha256=p.attention.sha(out/'kernel.o'),
      scope='offline N-window MLP reuse; independent original K-order state tiles; device parity and latency unverified')
    p.attention.dump(out/'build.json',meta)
    print('WINDOW_MLP_BUILD '+json.dumps(dict(policy=meta['policy'],code_bytes=sorted({row['code_bytes'] for row in audits}),
      stack_bound=max(sum(row['frames'])+256 for row in audits),scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m)
