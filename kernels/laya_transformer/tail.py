#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tail image with mask/softmax/entropy/TopK and action input formatting."""
import argparse
import json
from pathlib import Path

import scorer
v=scorer.v; b=scorer.b; p=scorer.p


def sources():
    return dict(scorer.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in (Path(__file__),Path(__file__).with_suffix('.cc'))})


def prefix():
    text=v.prefix().replace('%is_norm = arith.ori %old_norm,%head_norm : i1','%is_norm = arith.constant false')
    text=text.replace('%pack = arith.cmpi eq, %command,%ten : i32','%pack = arith.constant false')
    text=text.replace('%geglu = arith.cmpi eq, %command,%three : i32','''%geglu = arith.constant false
        %nineteen = arith.constant 19 : i32
        %tail_collecting = arith.cmpi eq, %command,%nineteen : i32''')
    decl=f'''    func.func private @tail_collect(memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>,i32)
    func.func private @tail_finish(memref<{p.QWORDS}xi32>,memref<{p.STATE}xi32>,memref<{p.OUT}xi32>,i32)
    func.func private @tail_vector(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32,i32,i32,i32)
'''
    text=text.replace('    func.func private @vector_bias(',decl+'    func.func private @vector_bias(')
    for i in range(16):
        start=text.index(f'    %core{i} = aie.core'); end=text.index(' {link_with',start); body=text[start:end]
        begin=body.index('                scf.if %pack {'); finish=body.index('              } else {\n                scf.if %legacy {',begin)
        elementwise=f'''                scf.if %tail_collecting {{
                  %packet = arith.index_cast %k : index to i32
                  func.call @tail_collect(%kv,%state{i},%packet) : (memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>,i32) -> ()
                  %collected = arith.cmpi eq, %k,%c3 : index
                  scf.if %collected {{
                    %st = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{p.OUT}xi32>>
                    %to = aie.objectfifo.subview.access %st[0] : !aie.objectfifosubview<memref<{p.OUT}xi32>> -> memref<{p.OUT}xi32>
                    func.call @tail_finish(%q,%state{i},%to,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.STATE}xi32>,memref<{p.OUT}xi32>,i32) -> ()
                    aie.objectfifo.release @outF_{i}(Produce,1)
                  }}
                }} else {{
                  %mine = arith.cmpi eq, %k,%row : index
                  scf.if %mine {{
                    %st = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{p.OUT}xi32>>
                    %to = aie.objectfifo.subview.access %st[0] : !aie.objectfifosubview<memref<{p.OUT}xi32>> -> memref<{p.OUT}xi32>
                    func.call @tail_vector(%q,%kv,%to,%fill,%command,%pack_base,%rowi) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32,i32,i32,i32) -> ()
                    aie.objectfifo.release @outF_{i}(Produce,1)
                  }}
                }}
'''
        body=body[:begin]+elementwise+body[finish:]; text=text[:start]+body+text[end:]
    return text


class Sequence(v.Sequence):
    def control(self,mode,kblocks=1,position=0,**kwargs):
        if mode not in (2,14,15,19,20,21): raise ValueError('tail command')
        # The inherited emitter only validates the original allowed set.
        # Emit this image's independently bounded command family explicitly.
        for index,value in enumerate((mode,kblocks,b.r.mlp.POISON,None,position)):
            if index in self.current and self.current[index]==value: continue
            self.current[index]=value
            for i in range(16):
                actual=i%4 if value is None else value[i//4] if isinstance(value,tuple) else value
                name=self.constant(actual)
                self.lines.append(f'      "aiex.npu.rtp_write"({name}) {{buffer = @chain_control_{i}, index = {index} : ui32}} : (i32) -> ()')


def runtime(words=128000):
    seq=Sequence(words); seq.control(19)
    for col in range(4):
        seq.dma(col,'q',16,[1,1,4,p.QWORDS],[0,0,p.QWORDS,1])
        seq.dma(col,'out',80000+col*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
        seq.dma(col,'kv',40000,[1,1,4,p.KVWORDS],[0,0,p.KVWORDS,1])
    seq.wait(); return seq.finish()


def build(out):
    out.mkdir(parents=True,exist_ok=False)
    clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes']
    objects=[]
    for name,file in [('tail',Path(__file__).with_suffix('.cc')),('block',b.HERE/'block.cc'),('gemm',b.HERE/'gemm.cc')]:
        obj=out/f'{name}.o'; objects.append(obj); p.attention.command([clang,*flags,'-c',file,'-o',obj],out)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved tail helper')
    (out/'harness.mlir').write_text(prefix()+runtime(),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    p.attention.dump(out/'build.json',dict(sources=sources(),audits=audits,words=128000,
      cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')},xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('TAIL_BUILD '+json.dumps(dict(code_bytes=sorted({a['code_bytes'] for a in audits}),stack_bound=max(sum(a['frames'])+256 for a in audits),scope='compile/audit only; device qualification required')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); build(args.out.resolve())
