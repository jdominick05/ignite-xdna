#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MLP/vector program for the remaining Laya head, with attention in image A."""
import argparse
import json
from pathlib import Path
import re
import shutil

import stem as s
b=s.b; p=s.p


def sources():
    return dict(s.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in (Path(__file__),Path(__file__).with_suffix('.cc'))})


def prefix():
    text=s.prefix()
    text=text.replace('%is_attention = arith.cmpi eq, %command, %zero : i32','%is_attention = arith.constant false')
    declarations=f'''    func.func private @vector_bias(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32,i32)
    func.func private @vector_seed(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32)
    func.func private @vector_sum(memref<{p.QWORDS}xi32>,memref<{p.OUT}xi32>,i32)
    func.func private @vector_carry(memref<{p.QWORDS}xi32>,memref<{p.OUT}xi32>,i32)
    func.func private @norm_pack_step(memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>,i32,i32,i32)
    func.func private @norm_pack_finish(memref<{p.STATE}xi32>,memref<{p.OUT}xi32>,i32)
'''
    text=text.replace('    func.func private @stem_copy(',declarations+'    func.func private @stem_copy(')
    text=text.replace('        %is_geglu = arith.cmpi eq, %command,%three : i32','''        %ten = arith.constant 10 : i32
        %thirteen = arith.constant 13 : i32
        %sixteen = arith.constant 16 : i32
        %seventeen = arith.constant 17 : i32
        %eighteen = arith.constant 18 : i32
        %twentytwo = arith.constant 22 : i32
        %carry = arith.cmpi eq, %command,%twentytwo : i32
        %head_seed = arith.cmpi eq, %command,%seventeen : i32
        %head_sum = arith.cmpi eq, %command,%eighteen : i32
        %pack = arith.cmpi eq, %command,%ten : i32
        %geglu = arith.cmpi eq, %command,%three : i32
        %is_geglu = arith.cmpi uge, %command,%three : i32''')
    text=text.replace('%is_norm = arith.ori %plain_norm,%stem_or_copy : i1','''%head_seed_or_sum = arith.ori %head_seed,%head_sum : i1
        %head_norm = arith.ori %head_seed_or_sum,%carry : i1
        %old_norm = arith.ori %plain_norm,%stem_or_copy : i1
        %is_norm = arith.ori %old_norm,%head_norm : i1''')
    text=text.replace('        %not_geglu = arith.cmpi ne, %command,%three : i32',
      '        %not_geglu = arith.cmpi ult, %command,%three : i32')
    old=f'              func.call @command_norm(%q,%kv,%o) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>) -> ()'
    text=text.replace(old,f'''                  scf.if %head_seed {{
                    func.call @vector_seed(%q,%kv,%o,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()
                  }} else {{
                    scf.if %head_sum {{
                      func.call @vector_sum(%q,%o,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()
                    }} else {{
{old}
                    }}
                  }}''')
    text=text.replace(old,f'''                      scf.if %carry {{
                        func.call @vector_carry(%q,%o,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()
                      }} else {{
{old}
                      }}''')
    old=f'                  func.call @chain_geglu(%q,%kv,%go,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()'
    text=text.replace(old,f'''                  scf.if %geglu {{
{old}
                  }} else {{
                    func.call @vector_bias(%q,%kv,%go,%fill,%command) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32,i32) -> ()
                  }}''')
    for i in range(16):
        start=text.index(f'    %core{i} = aie.core'); end=text.index(' {link_with',start); body=text[start:end]
        body=body.replace('        %non_attention_count = arith.select %is_geglu,%c4,%c1 : index',f'''        %c128 = arith.constant 128 : index
        %c127 = arith.constant 127 : index
        %pack_base = memref.load %control{i}[%c4] : memref<16xi32>
        %elementwise_count = arith.select %pack,%c128,%c4 : index
        %non_norm_count = arith.select %is_geglu,%elementwise_count,%c1 : index
        %non_attention_count = arith.select %is_norm,%c1,%non_norm_count : index''')
        body=body.replace('                %mine = arith.cmpi eq, %k,%row : index',f'''                scf.if %pack {{
                  %packet = arith.index_cast %k : index to i32
                  func.call @norm_pack_step(%kv,%state{i},%packet,%rowi,%pack_base) : (memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>,i32,i32,i32) -> ()
                  %collected = arith.cmpi eq, %k,%c127 : index
                  scf.if %collected {{
                    %sp = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{p.OUT}xi32>>
                    %po = aie.objectfifo.subview.access %sp[0] : !aie.objectfifosubview<memref<{p.OUT}xi32>> -> memref<{p.OUT}xi32>
                    func.call @norm_pack_finish(%state{i},%po,%fill) : (memref<{p.STATE}xi32>,memref<{p.OUT}xi32>,i32) -> ()
                    aie.objectfifo.release @outF_{i}(Produce,1)
                  }}
                }} else {{
                %mine = arith.cmpi eq, %k,%row : index''')
        old=f'                  aie.objectfifo.release @outF_{i}(Produce,1)\n                }}\n              }} else {{'
        if body.count(old)!=1: raise ValueError('elementwise output structure')
        body=body.replace(old,f'                  aie.objectfifo.release @outF_{i}(Produce,1)\n                }}\n                }}\n              }} else {{')
        text=text[:start]+body+text[end:]
    return text.replace('stack_size = 4608 : i32','stack_size = 5376 : i32')


class Sequence(b.r.Sequence):
    def control(self,mode,kblocks=1,position=0,**kwargs):
        if mode not in (1,2,3,9,10,11,12,13,14,15,16,17,18,22): raise ValueError('vector operator')
        for index,value in enumerate((mode,kblocks,b.r.mlp.POISON,None,position)):
            if index in self.current and self.current[index]==value: continue
            self.current[index]=value
            for i in range(16):
                actual=i%4 if value is None else value[i//4] if isinstance(value,tuple) else value
                name=self.constant(actual)
                self.lines.append(f'      "aiex.npu.rtp_write"({name}) {{buffer = @chain_control_{i}, index = {index} : ui32}} : (i32) -> ()')


def validate_compatibility(folder,meta,old_mlp):
    """Accept a fresh vector image, never substitute it inside an old context.

    Runtime instructions must have been independently compiled against both
    layouts. Core/init identity is required between the two NEW vector builds;
    the larger stack deliberately moves output buffers from their old address.
    """
    if meta['sources']!=sources() or meta.get('kind')!='vector-without-attention': raise ValueError('stale vector build')
    compat=meta['compatibility']
    if compat['encoder_instructions']!=old_mlp['instructions']: raise ValueError('encoder instruction ABI differs')
    if not compat['own_core_and_init_identical_across_layouts'] or not compat['rtp_allocations_identical'] or compat['stack_bytes']!=5376: raise ValueError('vector ABI qualification missing')
    if s.txn.registers(meta['audits'])!=s.txn.registers(old_mlp['audits']): raise ValueError('vector RTP allocations differ')
    if p.attention.sha(folder/'encoder/insts.bin')!=old_mlp['instructions']: raise ValueError('changed vector encoder instructions')
    for relative in ('prj','encoder/prj'):
        actual={f.name:p.attention.sha(f) for f in (folder/relative).rglob('*_aie_cdo_*.bin')}
        if actual!=meta['cdo']: raise ValueError('vector core/init changed')


def build(source,out,mlp_base):
    source=source.resolve(); out=out.resolve(); frozen=sources()
    base=json.loads((source/'build.json').read_text())
    if base['sources']!=s.sources(): raise ValueError('stale stem base')
    out.mkdir(parents=True,exist_ok=False)
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    clang=p.attention.PEANO/'bin/clang++.exe'
    objects=[]
    for name,file in [('vector',Path(__file__).with_suffix('.cc')),('stem',Path(s.__file__).with_suffix('.cc')),
      ('block',b.HERE/'block.cc'),('gemm',b.HERE/'gemm.cc'),('dispatch',b.HERE/'dispatch.cc'),('norm',b.HERE.parent/'laya_norm/kernel.cc')]:
        obj=out/f'{name}.o'; objects.append(obj)
        p.attention.command([clang,*flags,'-c',file,'-o',obj],out)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('undefined vector helper')
    (out/'harness.mlir').write_text(prefix()+s.runtime(base['tokens'],base['output_words'],base['target']),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if s.txn.registers(audits)!=s.txn.registers(base['audits']): raise ValueError('vector RTP allocations changed')
    if p.attention.sha(out/'insts.bin')!=base['instructions']: raise ValueError('stem runtime ABI changed')
    mlp_base=mlp_base.resolve(); old_mlp=json.loads((mlp_base/'build.json').read_text())
    if old_mlp['sources']!=b.sources(): raise ValueError('stale MLP reference')
    expected=old_mlp['families']['mlp']['global_512']['instructions']
    folder=out/'encoder'; folder.mkdir(); shutil.copyfile(out/'kernel.o',folder/'kernel.o')
    (folder/'harness.mlir').write_text(prefix()+b.mlp_runtime(b.plan()),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
    before=p.STACK
    try:
        p.STACK=5376; second=p.audit(folder,{})
    finally: p.STACK=before
    second_cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
    if second!=audits or second_cdo!=cdo: raise ValueError('vector core/init differs across instruction layouts')
    if p.attention.sha(folder/'insts.bin')!=expected: raise ValueError('encoder MLP runtime ABI changed')
    if sources()!=frozen: raise ValueError('source changed during build')
    result=dict(base,sources=frozen,audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin'),
      base_stem_sha256=p.attention.sha(source/'build.json'),kind='vector-without-attention')
    result['compatibility']=dict(stem_instructions=base['instructions'],encoder_instructions=expected,rtp_allocations_identical=True,
      own_core_and_init_identical_across_layouts=True,base_mlp_build_sha256=p.attention.sha(mlp_base/'build.json'),stack_bytes=5376)
    p.attention.dump(out/'build.json',result)
    print('VECTOR_BUILD '+json.dumps(dict(code_bytes=sorted({a['code_bytes'] for a in audits}),stack_bound=max(sum(a['frames'])+256 for a in audits),
      runtime_instructions_identical=True,own_core_and_init_identical=True,scope='compile/audit only; new operations require device qualification')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build',)); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mlp-base',type=Path,required=True)
    args=ap.parse_args(); build(args.source,args.out,args.mlp_base)
