#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Screen fixed 32x64 GEMM code generation without changing registered images."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generated(original, fold_reset):
    """Retain the generic fallback and original MAC order; specialize one shape."""
    if '#ifdef ATTENTION_FP32_PROBS' not in original:
        raise ValueError('missing original product variants')
    start=original.index('  const int kblocks=8,qstride=256;')
    end=original.index('\n#ifdef ATTENTION_FP32_PROBS',start)
    body=original[start:end]
    if body.count('#endif')!=1 or not body.endswith('\n}'):
        raise ValueError('unexpected product body')
    body=body.replace('#endif\n','')
    body=body.replace('const int kblocks=8,qstride=256;',
                      'const int rows=8,cols=16,kblocks=8,qstride=256;')
    if fold_reset:
        clear='  if(reset) for(int i=0;i<rows*cols*16;i+=16) aie::store_v(state+i,aie::zeros<float,16>());\n'
        if body.count(clear)!=1:
            raise ValueError('reset loop changed')
        body=body.replace(clear,'')
        body,count=re.subn(r'aie::load_v<16>\((state\+[^;\n]*?)\)(?=\))',
            r'(reset ? aie::zeros<float,16>() : aie::load_v<16>(\1))',body)
        if count!=8:
            raise ValueError('accumulator initialization coverage')
    generic=original.replace('transformer_matmul','transformer_generic')
    # Keep helpers calling the ABI entry so fixed and fallback paths are exercised.
    cut=generic.index('extern "C" void gemm_step(')
    helpers=original[original.index('extern "C" void gemm_step('):]
    return generic[:cut]+'\nstatic __attribute__((noinline)) void transformer_fixed(const bfloat16 *q,const bfloat16 *b,float *state,int reset) {\n'+body+'\n'+'''
extern "C" void transformer_matmul(const bfloat16 *q,const bfloat16 *b,float *state,int reset,int rows,int cols) {
  if(rows==8 && cols==16) transformer_fixed(q,b,state,reset);
  else transformer_generic(q,b,state,reset,rows,cols);
}
'''+helpers


def screen(root,out,clang,include):
    root=root.absolute();out=out.absolute();out.mkdir(parents=True,exist_ok=False)
    source=root/'kernels/laya_transformer/gemm.cc';original=source.read_text()
    frozen={str(file.relative_to(root)).replace('\\','/'):sha(file)
        for file in (source,Path(__file__).absolute())}
    flags=['-O3','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG',
        '-fstack-size-section','-D__AIE_API_AIE_ADF_HPP__',f'-I{include}','-Wno-attributes']
    def run(args):
        result=subprocess.run([str(x) for x in args],capture_output=True,text=True,check=True)
        return result.stdout
    results=[]
    for name,text in [('generic',original),('fixed',generated(original,False)),
                      ('fixed_reset',generated(original,True))]:
        src=out/(name+'.cc');src.write_text(text,encoding='utf-8');obj=out/(name+'.o')
        run([clang,*flags,'-c',src,'-o',obj])
        sizes=run([clang.parent/'llvm-size.exe','-A',obj])
        frames=run([clang.parent/'llvm-readobj.exe','--stack-sizes',obj])
        asm=run([clang.parent/'llvm-objdump.exe','-dr','--demangle',obj])
        (out/(name+'.asm')).write_text(asm,encoding='utf-8')
        row=dict(variant=name,source_sha256=sha(src),object_sha256=sha(obj),
            disassembly_sha256=sha(out/(name+'.asm')),
            text_bytes=sum(int(x) for x in re.findall(r'^\.text\S*\s+(\d+)\s+',sizes,re.M)),
            frames=[int(x,16) for x in re.findall(r'Size: (0x[0-9a-fA-F]+)',frames)])
        results.append(row);print('GEMM_SPECIALIZE_OBJECT '+json.dumps(row),flush=True)
    if any(sha(root/file)!=digest for file,digest in frozen.items()):
        raise ValueError('source changed during screen')
    report=dict(sources=frozen,compiler_sha256=sha(clang),
        flags=[f for f in flags if not f.startswith('-I')],objects=results,
        scope='offline compilation only; no numerical parity, cycle, utilization or latency claim')
    (out/'screen.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('GEMM_SPECIALIZE_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('root','out','clang','include'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();screen(args.root,args.out,args.clang,args.include)
