# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resident queries, four-core KV multicast, compact host buffer and DMA replay."""
import argparse
import json
from pathlib import Path
import re
import ctypes
from functools import lru_cache
import shutil

import probe as p

QWORDS=528
KVWORDS=2064
INPUT_WORDS=16*32*QWORDS+16*16*KVWORDS


@lru_cache(None)
def native_library(path):
    dll=ctypes.CDLL(str(path))
    dll.pack_attention.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_int,ctypes.c_uint32]
    dll.pack_attention.restype=None
    return dll


def pack_native(q,k,v,mask,poison,path,out=None,mask_params=None):
    arrays=[p.np.asarray(x) for x in (q,k,v)]
    if any(x.shape!=(16,512,64) or x.dtype!=p.np.float32 or not x.flags.c_contiguous for x in arrays):
        raise ValueError('native pack needs contiguous fp32 [16,512,64] bf16 values')
    if mask_params is None:
        valid=int((mask==1).any(axis=0).sum())
        local=bool(valid and not p.np.all(mask[:,:valid]==1))
    else:
        valid,local=mask_params
        if valid not in (0,1,193,512) or type(local) is not bool:
            raise ValueError('invalid prevalidated mask parameters')
    if out is None: out=p.np.empty(INPUT_WORDS,p.np.uint32)
    if out.shape!=(INPUT_WORDS,) or out.dtype!=p.np.uint32 or not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError('invalid native output buffer')
    native_library(str(path)).pack_attention(*(x.ctypes.data for x in arrays),out.ctypes.data,valid,local,poison)
    return out


def build_native(out,fast=False):
    vswhere=Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    vs=p.command([vswhere,'-products','*','-latest','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                  '-property','installationPath'],out).strip()
    vcvars=Path(vs)/'VC/Auxiliary/Build/vcvars64.bat'
    define=' /DATTENTION_PACK_FAST=1' if fast else ''
    command=f'call "{vcvars}" >nul && cl /nologo /LD /O2 /arch:AVX2 /TC{define} "{p.HERE / "pack.c"}" /Fe:attention_pack.dll'
    (out/'build_pack.cmd').write_text('@echo off\n'+command+'\n',encoding='utf-8')
    p.command(['cmd.exe','/d','/c',out/'build_pack.cmd'],out)
    return p.sha(out/'attention_pack.dll')


def pack(q,k,v,mask,poison):
    if len(q)!=16: raise ValueError('shared design requires sixteen heads')
    out=p.np.zeros(INPUT_WORDS,p.np.uint32)
    qs=out[:16*32*QWORDS].reshape(16,32,QWORDS)
    qs[:,:,0]=p.np.arange(32)*16
    # Extract actual key padding independently of local window truncation.
    used=(mask==1).any(axis=0)
    valid=int(used.sum())
    local=bool(valid and not p.np.all(mask[:,:valid]==1))
    qs[:,:,1]=valid; qs[:,:,2]=poison; qs[:,:,3]=int(local)
    qt=p.bits(q.copy()).reshape(16,32,16,64)
    qt[:,p.np.all(mask==2,axis=1).reshape(32,16)]=0
    qt=qt.reshape(16,32,4,4,8,8).transpose(0,1,2,4,3,5).reshape(16,32,1024)
    qs[:,:,16:].view(p.np.uint16)[:]=qt
    kvs=out[16*32*QWORDS:].reshape(16,16,KVWORDS)
    kt=p.bits(k).reshape(16,16,32,64).transpose(0,1,3,2).reshape(16,16,8,8,8,4)
    raw=kvs.view(p.np.uint16)
    raw[:,:,:2048]=kt.transpose(0,1,2,4,3,5).reshape(16,16,2048)
    raw[:,:,2048:4096]=p.bits(v).reshape(16,16,4,8,16,4).transpose(0,1,2,4,3,5).reshape(16,16,2048)
    raw[:,:,4096:]=p.np.arange(32,dtype=p.np.uint16)
    return out


def harness(scratch_bank3=False):
    parts=['module {','  aie.device(npu1) {',
      f'    func.func private @attention_shared(memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{p.STATE}xi32>,memref<{p.SCRATCH}xi32>,memref<256xi32>,i32)',
      f'    func.func private @attention_finish(memref<{p.STATE}xi32>,memref<{p.OUT}xi32>)']
    cores=[]
    for col in range(4):
        parts += [f'    %s{col} = aie.tile({col},0)',f'    %m{col} = aie.tile({col},1)']
        for r in range(4): parts.append(f'    %t{col*4+r} = aie.tile({col},{r+2})')
        consumers=', '.join(f'%t{col*4+r}' for r in range(4))
        parts += [
          f'    aie.objectfifo @qL2_{col}(%s{col},{{%m{col}}},1 : i32) : !aie.objectfifo<memref<{4*QWORDS}xi32>>',
          f'    aie.objectfifo @kvL2_{col}(%s{col},{{%m{col}}},2 : i32) : !aie.objectfifo<memref<{KVWORDS}xi32>>',
          f'    aie.objectfifo @kvF_{col}(%m{col},{{{consumers}}},2 : i32) : !aie.objectfifo<memref<{KVWORDS}xi32>>',
          f'    aie.objectfifo.link [@kvL2_{col}] -> [@kvF_{col}]([] [0])',
          f'    aie.objectfifo @outL2_{col}(%m{col},{{%s{col}}},1 : i32) : !aie.objectfifo<memref<{4*p.OUT}xi32>>',
        ]
        for r in range(4):
            i=col*4+r
            parts += [
              f'    %state{i} = aie.buffer(%t{i}) {{sym_name = "attn_state_{i}"}} : memref<{p.STATE}xi32>',
              f'    %scratch{i} = aie.buffer(%t{i}) {{sym_name = "attn_scratch_{i}"'+(', address = 54400 : i32' if scratch_bank3 else '')+f'}} : memref<{p.SCRATCH}xi32>',
              f'    %mask{i} = aie.buffer(%t{i}) {{sym_name = "attn_mask_{i}"}} : memref<256xi32>',
              f'    aie.objectfifo @qF_{i}(%m{col},{{%t{i}}},1 : i32) : !aie.objectfifo<memref<{QWORDS}xi32>>',
              f'    aie.objectfifo @outF_{i}(%t{i},{{%m{col}}},1 : i32) : !aie.objectfifo<memref<{p.OUT}xi32>>',
            ]
            cores.append(f'''    %core{i} = aie.core(%t{i}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %outsize = arith.constant {p.OUT} : index
      %forever = arith.constant 4294967295 : index
      scf.for %it = %c0 to %forever step %c1 {{
        %sq = aie.objectfifo.acquire @qF_{i}(Consume,1) : !aie.objectfifosubview<memref<{QWORDS}xi32>>
        %q = aie.objectfifo.subview.access %sq[0] : !aie.objectfifosubview<memref<{QWORDS}xi32>> -> memref<{QWORDS}xi32>
        scf.for %k = %c0 to %c16 step %c1 {{
          %skv = aie.objectfifo.acquire @kvF_{col}(Consume,1) : !aie.objectfifosubview<memref<{KVWORDS}xi32>>
          %kv = aie.objectfifo.subview.access %skv[0] : !aie.objectfifosubview<memref<{KVWORDS}xi32>> -> memref<{KVWORDS}xi32>
          %ki = arith.index_cast %k : index to i32
          func.call @attention_shared(%q,%kv,%state{i},%scratch{i},%mask{i},%ki) : (memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{p.STATE}xi32>,memref<{p.SCRATCH}xi32>,memref<256xi32>,i32) -> ()
          aie.objectfifo.release @kvF_{col}(Consume,1)
        }}
        %so = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{p.OUT}xi32>>
        %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<{p.OUT}xi32>> -> memref<{p.OUT}xi32>
        %poison = memref.load %q[%c2] : memref<{QWORDS}xi32>
        scf.for %j = %c0 to %outsize step %c1 {{
          memref.store %poison,%o[%j] : memref<{p.OUT}xi32>
        }}
        func.call @attention_finish(%state{i},%o) : (memref<{p.STATE}xi32>,memref<{p.OUT}xi32>) -> ()
        aie.objectfifo.release @outF_{i}(Produce,1)
        aie.objectfifo.release @qF_{i}(Consume,1)
      }}
      aie.end
    }} {{link_with = "kernel.o",stack_size = {p.STACK} : i32}}''')
        parts += [
          f'    aie.objectfifo.link [@qL2_{col}] -> ['+', '.join(f'@qF_{col*4+r}' for r in range(4))+']([] ['+', '.join(str(r*QWORDS) for r in range(4))+'])',
          '    aie.objectfifo.link ['+', '.join(f'@outF_{col*4+r}' for r in range(4))+f'] -> [@outL2_{col}](['+', '.join(str(r*p.OUT) for r in range(4))+'] [])',
        ]
    parts.extend(cores)
    no=512*p.OUT
    parts.append(f'    aie.runtime_sequence(%a: memref<{INPUT_WORDS}xi32>,%o: memref<{no}xi32>) {{')
    for col in range(4):
        nq=4*32*QWORDS; nkv=4*16*KVWORDS; nout=128*p.OUT
        parts += [
          f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,{col*nq}][1,1,{nq//16},16][0,0,16,1]) {{id = 0 : i64, metadata = @qL2_{col}}} : memref<{INPUT_WORDS}xi32>',
          f'      aiex.npu.dma_memcpy_nd(%o[0,0,0,{col*nout}][1,1,{nout//16},16][0,0,16,1]) {{id = 2 : i64, metadata = @outL2_{col},issue_token = true}} : memref<{no}xi32>',
        ]
        for head in range(4):
            parts.append(f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,{16*32*QWORDS+col*nkv+head*16*KVWORDS}][8,1,16,{KVWORDS}][0,0,{KVWORDS},1]) {{id = {head+3} : i64, metadata = @kvL2_{col}}} : memref<{INPUT_WORDS}xi32>')
    for col in range(4): parts.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')
    return '\n'.join(parts+['    }','  }','}'])+'\n'


def build(out,masks,mode,valid,variant='base'):
    out.mkdir(parents=True,exist_ok=True)
    clang=p.PEANO/'bin/clang++.exe'
    flags=['-O2','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
           '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.INSTALL/"include"}',f'-I{p.UPSTREAM/"aie_runtime_lib/AIE2"}',
           '-Wno-attributes','-Wno-parentheses']
    if variant in ('mask','bank'): flags.append('-DATTENTION_BYPASS_MASK=1')
    if variant in ('unroll','combined'): flags.append('-DATTENTION_UNROLL_2=1')
    p.command([clang,*flags,'-c',p.HERE/'kernel_shared.cc','-o',out/'kernel_core.o'],out)
    p.command([clang,*flags,'-c',p.UPSTREAM/'aie_runtime_lib/AIE2/lut_based_ops.cpp','-o',out/'lut.o'],out)
    p.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',out/'kernel_core.o',out/'lut.o','-o',out/'kernel.o'],out)
    (out/'harness.mlir').write_text(harness(variant=='bank'),encoding='utf-8')
    p.command([p.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.PEANO}','--get-npu-insts',
               '--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    audits=[]
    for elf in sorted((out/'prj').rglob('*.elf')):
        col,row=map(int,re.search(r'core_(\d+)_(\d+)\.elf$',elf.name).groups())
        symbols=p.command([p.PEANO/'bin/llvm-nm.exe',elf],out)
        if p.software_float_helpers(symbols): raise ValueError('software floating point')
        ld=(out/f'prj/ldScripts_main_core_{col}_{row}.ld.script').read_text()
        spans=[(int(a,16)-0x70000,int(a,16)-0x70000+int(n,16),name) for a,name,n in re.findall(
          r'\. = (0x[0-9A-Fa-f]+);\s*\n\s*(\w+) = \.;\s*\n\. \+= (0x[0-9A-Fa-f]+);',ld)]
        for name in ('exp_ilut_ab','exp_ilut_cd','exp_flut_ab','exp_flut_cd'):
            m=re.search(r'^([0-9a-fA-F]+) [A-Za-z] '+name+r'$',symbols,re.M)
            if not m: raise ValueError('missing LUT')
            a=int(m[1],16)-0x70000; spans.append((a,a+1024,name))
        spans.sort()
        if len(spans)<10 or any(a < -0x30000 or b>65536 or a//65536!=(b-1)//65536 for a,b,_ in spans) or any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError(f'invalid memory {col},{row}: {spans}')
        if (0,p.STACK,'_sp_start_value_DM_stack') not in spans: raise ValueError('missing stack')
        sections=p.command([p.PEANO/'bin/llvm-size.exe','-A',elf],out)
        code=sum(int(m[1]) for m in re.findall(r'^(\.(?:text|text\.[^\s]+))\s+(\d+)\s+',sections,re.M))
        if not 0<code<=16384: raise ValueError(f'program size {code}')
        stack=p.command([p.PEANO/'bin/llvm-readobj.exe','--stack-sizes',elf],out)
        frames=[int(x,16) for x in re.findall(r'Size: (0x[0-9a-fA-F]+)',stack)]
        if not frames or sum(frames)+256>p.STACK: raise ValueError('stack overflow')
        audits.append(dict(core=[col,row],placement=spans,code_bytes=code,stack_sizes=frames))
    if len(audits)!=16: raise ValueError('missing cores')
    fast_packing=variant in ('host','combined')
    dll_sha=build_native(out,fast_packing)
    meta=dict(heads=16,mode=mode,valid=valid,variant=variant,resident='none',implementation='shared',transport='shared',cores=16,
              packets=8192,outputs=512,input_words=INPUT_WORDS,core_audits=audits,fast_packing=fast_packing,
              pack_dll_sha256=dll_sha,dependencies={name:p.sha(p.HERE/name) for name in ('kernel_lut.cc','shared.py','pack.c','probe.py')},kernel_source='kernel_shared.cc',kernel_sha256=p.sha(p.HERE/'kernel_shared.cc'),
              upstream_sha256={name:p.sha(p.UPSTREAM/name) for name in ('aie_runtime_lib/AIE2/lut_based_ops.h','aie_runtime_lib/AIE2/lut_based_ops.cpp')},
              masks_sha256=p.sha(masks),xclbin_sha256=p.sha(out/'final.xclbin'),instructions_sha256=p.sha(out/'insts.bin'))
    p.dump(out/'build.json',meta)
    print(json.dumps(meta),flush=True)


def build_suite(out,masks,controls,variant='base',prefix=''):
    """One binary handles both masks and all valid lengths through Q headers."""
    first=out/f'{prefix}global_h16_v512'
    build(first,masks,'global',512,variant)
    base=json.loads((first/'build.json').read_text())
    for mode in ('global','local'):
        for valid in (512,193,1,0):
            dest=out/f'{prefix}{mode}_h16_v{valid}'
            if dest==first: continue
            dest.mkdir(parents=True,exist_ok=True)
            for name in ('final.xclbin','insts.bin','attention_pack.dll'):
                shutil.copyfile(first/name,dest/name)
            p.dump(dest/'build.json',dict(base,mode=mode,valid=valid))
    if controls:
        for mode in ('global','local'):
            source=controls/f'{mode}_h16_v512'
            dest=out/f'vector_{mode}_h16_v512'; dest.mkdir(parents=True,exist_ok=True)
            for name in ('final.xclbin','insts.bin','build.json'):
                shutil.copyfile(source/name,dest/name)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--masks',type=Path,required=True)
    ap.add_argument('--mode',choices=('global','local'),default='global')
    ap.add_argument('--valid',type=int,choices=(512,193,1,0),default=512)
    ap.add_argument('--suite',action='store_true')
    ap.add_argument('--controls',type=Path)
    ap.add_argument('--variant',choices=('base','mask','bank','unroll','host','combined'),default='base')
    ap.add_argument('--race',action='store_true')
    ap.add_argument('--speed-race',action='store_true')
    a=ap.parse_args()
    if a.speed_race:
        for variant in ('base','unroll','host','combined'): build_suite(a.out.resolve(),a.masks,None,variant,variant+'_')
    elif a.race:
        for variant in ('base','mask','bank'): build_suite(a.out.resolve(),a.masks,None,variant,variant+'_')
    elif a.suite: build_suite(a.out.resolve(),a.masks,a.controls,a.variant)
    else: build(a.out.resolve(),a.masks,a.mode,a.valid,a.variant)
