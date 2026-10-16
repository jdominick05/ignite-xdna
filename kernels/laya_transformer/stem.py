#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host token-indexed Shim DMA, NPU embedding copy and FP32 normalization.

Host code schedules integer addresses. It never gathers embedding values into
an activation buffer; the NPU copies weights directly from the resident table.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
import block as b
import instructions as txn
p=b.p
VOCAB=50368
TABLE=VOCAB*1024
AFFINE=TABLE+1056
INPUT_WORDS=AFFINE+p.KVWORDS+16


def sources():
    files=[Path(__file__),Path(__file__).with_suffix('.cc'),Path(txn.__file__),p.ROOT/'src/ignite_xdna/compiler/scheduler.py']
    return dict(b.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in files})


def prefix():
    text=b.mlp_prefix()
    text=text.replace('    func.func private @command_norm(',f'    func.func private @stem_copy(memref<{p.QWORDS}xi32>,memref<{p.OUT}xi32>,i32)\n    func.func private @stem_norm(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>)\n    func.func private @command_norm(')
    text=text.replace('        %is_norm = arith.cmpi eq, %command, %one : i32',
      '        %eleven = arith.constant 11 : i32\n        %twelve = arith.constant 12 : i32\n        %copy = arith.cmpi eq, %command,%eleven : i32\n        %stem = arith.cmpi eq, %command,%twelve : i32\n        %plain_norm = arith.cmpi eq, %command,%one : i32\n        %stem_or_copy = arith.ori %stem,%copy : i1\n        %is_norm = arith.ori %plain_norm,%stem_or_copy : i1')
    old=f'              func.call @command_norm(%q,%kv,%o) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>) -> ()'
    new=f'''              scf.if %copy {{
                func.call @stem_copy(%q,%o,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()
              }} else {{
                scf.if %stem {{
                  func.call @stem_norm(%q,%kv,%o) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>) -> ()
                }} else {{
{old}
                }}
              }}'''
    if text.count(old)!=16: raise ValueError('norm dispatch structure changed')
    text=text.replace('        %not_norm = arith.cmpi ne, %command, %one : i32',
      '        %not_norm_false = arith.constant false\n        %not_norm = arith.cmpi eq, %is_norm,%not_norm_false : i1')
    return text.replace(old,new)


class Sequence(b.MlpSequence):
    def __init__(self,words):
        super().__init__(words)
        self.lines[0]=f'    aie.runtime_sequence(%a: memref<{INPUT_WORDS}xi32>,%o: memref<{words}xi32>) {{'
    def control(self,mode,**kwargs):
        if mode not in (11,12): return super().control(mode,**kwargs)
        # Emit the existing norm command's four fields, then select copy.
        super().control(1,**kwargs)
        self.current[0]=mode
        for i in range(16):
            name=self.constant(mode)
            self.lines.append(f'      "aiex.npu.rtp_write"({name}) {{buffer = @chain_control_{i}, index = 0 : ui32}} : (i32) -> ()')
    def dma(self,col,kind,offset,dims,strides,output=False,bid=None,arena_read=False):
        before=len(self.lines)
        super().dma(col,kind,offset,dims,strides,output,bid)
        var='%o' if output or arena_read else '%a'; words=self.words if var=='%o' else INPUT_WORDS
        self.lines[before]=re.sub(r'%[ao]\[',var+'[',self.lines[before]).replace(f'memref<{self.words}xi32>',f'memref<{words}xi32>')


def validate_tokens(tokens,count=None):
    value=np.asarray(tokens)
    if value.ndim!=1 or value.dtype.kind not in 'iu' or (count is not None and value.size!=count): raise ValueError('token vector shape/type')
    if value.size%16 or not 16<=value.size<=512 or np.any(value<0) or np.any(value>=VOCAB): raise ValueError('token count/range')
    return value.astype(np.int64,copy=False)


def runtime(tokens,words,target=16):
    tokens=validate_tokens(tokens); percol=len(tokens)//4
    if target<0 or target+512*p.OUT>words: raise ValueError('stem output bounds')
    seq=Sequence(words); seq.control(11)
    for group in range(percol//4):
        for col in range(4):
            start=target+(col*128+group*4)*p.OUT
            seq.dma(col,'out',start,[1,1,4*p.OUT//16,16],[0,0,16,1],True)
            for row,bid in enumerate((0,1,4,5)):
                token=int(tokens[col*percol+group*4+row])
                seq.dma(col,'q',token*1024,[1,1,p.QWORDS//16,16],[0,0,16,1],bid=bid)
            seq.dma(col,'kv',AFFINE,[1,1,1,p.KVWORDS],[0,0,p.KVWORDS,1])
        seq.wait()
    seq.control(12)
    for col in range(4):
        start=target+col*128*p.OUT
        seq.dma(col,'q',start,[1,1,percol,p.QWORDS],[0,0,p.OUT,1],arena_read=True)
        seq.dma(col,'out',start,[1,1,percol*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',AFFINE,[percol//4,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait()
    return seq.finish()


def prepare(raw,original,tokens):
    import struct
    original=validate_tokens(original); tokens=validate_tokens(tokens,len(original)); result=bytearray(raw); offsets=[]
    percol=len(original)//4
    emission=[col*percol+group*4+row for group in range(percol//4) for col in range(4) for row in range(4)]
    for op in txn.parse(raw):
        if op['op']=='DDR_PATCH' and op['arg_idx']==0 and op['arg_offset']<TABLE*4:
            index=len(offsets)
            if index>=len(original) or op['arg_offset']!=int(original[emission[index]])*4096: raise ValueError('unexpected embedding read order')
            offsets.append(op['offset']+40); struct.pack_into('<I',result,offsets[-1],int(tokens[emission[index]])*4096)
    if len(offsets)!=len(original): raise ValueError('missing embedding reads')
    return bytes(result),offsets


def build(source,out,tokens_file,key):
    source=source.resolve(); out=out.resolve(); frozen=sources()
    meta=json.loads((source/'build.json').read_text())
    if meta['sources']!=b.sources(): raise ValueError('stale base program')
    tokens=validate_tokens(np.load(tokens_file)[key]); words=512*p.OUT+32
    out.mkdir(parents=True,exist_ok=False)
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes']
    clang=p.attention.PEANO/'bin/clang++.exe'
    p.attention.command([clang,*flags,'-c',Path(__file__).with_suffix('.cc'),'-o',out/'stem.o'],out)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',source/'mlp/kernel.o',out/'stem.o','-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('undefined helper')
    (out/'harness.mlir').write_text(prefix()+runtime(tokens,words),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=b.r.mlp.STACK; audits=p.audit(out)
    finally: p.STACK=before
    old=meta['families']['mlp']['global_512']; cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if [a['spans'] for a in json.loads(json.dumps(audits))]!=[a['spans'] for a in old['audits']]: raise ValueError('stem changes physical FIFO/control allocations')
    if any(cdo[key]!=old['cdo'][key] for key in ('main_aie_cdo_init.bin','main_aie_cdo_enable.bin')): raise ValueError('stem changes initialization/connectivity')
    if sources()!=frozen: raise ValueError('source changed during build')
    result=dict(sources=frozen,tokens=tokens.tolist(),tokens_file_sha256=p.attention.sha(tokens_file),input_words=INPUT_WORDS,output_words=words,target=16,
      audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin'),
      base_build_sha256=p.attention.sha(source/'build.json'),host_work='token-indexed DMA control addresses; no embedding activation gather')
    p.attention.dump(out/'build.json',result)
    print('STEM_BUILD '+json.dumps(dict(tokens=len(tokens),code_bytes=sorted({a['code_bytes'] for a in audits}),
      max_stack_bound=max(sum(a['frames'])+256 for a in audits),same_physical_layout_and_init=True,host_work=result['host_work'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path); ap.add_argument('--tokens',type=Path,required=True); ap.add_argument('--key',default='probe')
    ap.add_argument('--model',type=Path); args=ap.parse_args()
    if args.command=='build': build(args.source,args.out,args.tokens,args.key)
    else:
        from stem_run import run
        run(args.out,args.model,args.tokens)
