#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One persistent core program for attention and residual/LayerNorm.

This is an integration scaffold, not complete Laya. No host neural operation
supplies a device result. CPU references are used only for qualification.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'laya_attention'))
import probe as attention
import shared

spec=importlib.util.spec_from_file_location('laya_norm_reference',HERE.parent/'laya_norm/probe.py')
norm=importlib.util.module_from_spec(spec)
spec.loader.exec_module(norm)

QWORDS,KVWORDS,OUT=2064,2064,2080
QCOUNT,KVCOUNT=512,256
INPUT_WORDS=QCOUNT*QWORDS+KVCOUNT*KVWORDS
STACK=4096
STATE=2048
MODEL_GEMMS=((512,1024,3072),(512,1024,1024),(512,1024,5248),(512,2624,1024),
             (512,1024,4096),(512,4096,1024),(4,1024,1024),(4,1024,1),(1,1028,256),(1,256,2))


def sources():
    paths=[HERE/'program.py',HERE/'dispatch.cc',HERE/'gemm.cc',HERE.parent/'laya_norm/kernel.cc',HERE.parent/'laya_norm/probe.py',
           *[HERE.parent/'laya_attention'/n for n in ('kernel_lut.cc','kernel_shared.cc','probe.py','shared.py')],
           ROOT/'tools/laya_hybrid.py']
    return {str(p.relative_to(ROOT)).replace('\\','/'):attention.sha(p) for p in paths}


def padded(value,alignment):
    if value<=0: raise ValueError('dimensions must be positive')
    return -(-value//alignment)*alignment


def geometry(mode,k=1024,n=512,m=512):
    if mode!='gemm': return INPUT_WORDS,QCOUNT*OUT
    mp,kp,npad=padded(m,128),padded(k,64),padded(n,256)
    return ((mp//32)*(kp//64)+(npad//64)*(kp//64))*QWORDS,(mp//32)*(npad//64)*OUT


def harness(mode,k=1024,n=512,m=512):
    # Share the qualified FIFO topology, but use fixed larger packets for all
    # commands. Keep modifications local so imported reference packers retain
    # their original ABI.
    before=(shared.QWORDS,shared.INPUT_WORDS,attention.OUT,attention.STATE)
    try:
        shared.QWORDS=QWORDS; shared.INPUT_WORDS=INPUT_WORDS; attention.OUT=OUT; attention.STATE=STATE
        text=shared.harness()
    finally:
        shared.QWORDS,shared.INPUT_WORDS,attention.OUT,attention.STATE=before
    text,count=re.subn(r'(@kv(?:L2|F)_\d+\([^\n]*),2 : i32\)',r'\1,1 : i32)',text)
    if count!=8: raise ValueError('unexpected KV FIFO declarations')
    text=text.replace('    func.func private @attention_finish(',
      f'    func.func private @gemm_step(memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{STATE}xi32>)\n'
      f'    func.func private @gemm_finish(memref<{STATE}xi32>,memref<{OUT}xi32>)\n'
      f'    func.func private @command_norm(memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{OUT}xi32>)\n    func.func private @attention_finish(')
    for col in range(4):
        for r in range(4):
            i=col*4+r
            start=text.index(f'    %core{i} = aie.core')
            end=text.index(' {{link_with',start) if ' {{link_with' in text[start:] else text.index(' {link_with',start)
            # Replace the body while preserving declarations and FIFO links.
            worker=f'''    %core{i} = aie.core(%t{i}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c16 = arith.constant 16 : index
      %outsize = arith.constant {OUT} : index
      %forever = arith.constant 4294967295 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      scf.for %it = %c0 to %forever step %c1 {{
        %sq = aie.objectfifo.acquire @qF_{i}(Consume,1) : !aie.objectfifosubview<memref<{QWORDS}xi32>>
        %q = aie.objectfifo.subview.access %sq[0] : !aie.objectfifosubview<memref<{QWORDS}xi32>> -> memref<{QWORDS}xi32>
        %command = memref.load %q[%c4] : memref<{QWORDS}xi32>
        %is_attention = arith.cmpi eq, %command, %zero : i32
        %is_norm = arith.cmpi eq, %command, %one : i32
        %count = arith.select %is_attention,%c16,%c1 : index
        scf.for %k = %c0 to %count step %c1 {{
          %skv = aie.objectfifo.acquire @kvF_{col}(Consume,1) : !aie.objectfifosubview<memref<{KVWORDS}xi32>>
          %kv = aie.objectfifo.subview.access %skv[0] : !aie.objectfifosubview<memref<{KVWORDS}xi32>> -> memref<{KVWORDS}xi32>
          scf.if %is_attention {{
            %ki = arith.index_cast %k : index to i32
            func.call @attention_shared(%q,%kv,%state{i},%scratch{i},%mask{i},%ki) : (memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{STATE}xi32>,memref<{attention.SCRATCH}xi32>,memref<256xi32>,i32) -> ()
          }} else {{
            scf.if %is_norm {{
              %so = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{OUT}xi32>>
              %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<{OUT}xi32>> -> memref<{OUT}xi32>
              func.call @command_norm(%q,%kv,%o) : (memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{OUT}xi32>) -> ()
              aie.objectfifo.release @outF_{i}(Produce,1)
            }} else {{
              func.call @gemm_step(%q,%kv,%state{i}) : (memref<{QWORDS}xi32>,memref<{KVWORDS}xi32>,memref<{STATE}xi32>) -> ()
            }}
          }}
          aie.objectfifo.release @kvF_{col}(Consume,1)
        }}
        %last = memref.load %q[%c7] : memref<{QWORDS}xi32>
        %is_last = arith.cmpi ne, %last, %zero : i32
        %not_norm = arith.cmpi ne, %command, %one : i32
        %finish = arith.ori %is_attention,%is_last : i1
        %emit = arith.andi %not_norm,%finish : i1
        scf.if %emit {{
          %so = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{OUT}xi32>>
          %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<{OUT}xi32>> -> memref<{OUT}xi32>
          %poison = memref.load %q[%c2] : memref<{QWORDS}xi32>
          scf.for %j = %c0 to %outsize step %c1 {{
            memref.store %poison,%o[%j] : memref<{OUT}xi32>
          }}
          scf.if %is_attention {{
            func.call @attention_finish(%state{i},%o) : (memref<{STATE}xi32>,memref<{OUT}xi32>) -> ()
          }} else {{
            func.call @gemm_finish(%state{i},%o) : (memref<{STATE}xi32>,memref<{OUT}xi32>) -> ()
          }}
          aie.objectfifo.release @outF_{i}(Produce,1)
        }}
        aie.objectfifo.release @qF_{i}(Consume,1)
      }}
      aie.end
    }}'''
            text=text[:start]+worker+text[end:]
    if mode=='attention': return text
    text=text[:text.index('    aie.runtime_sequence')]
    ni,no=geometry(mode,k,n,m)
    lines=[f'    aie.runtime_sequence(%a: memref<{ni}xi32>,%o: memref<{no}xi32>) {{']
    if mode=='gemm':
        kb=padded(k,64)//64; ng=padded(n,256)//256; mg=padded(m,128)//128
        for group in range(ng):
            for col in range(4):
                outsize=mg*4*OUT
                offset=mg*4*kb*QWORDS+(col*ng+group)*kb*KVWORDS
                lines += [
                  f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,0][1,{mg},{kb},{4*QWORDS}][0,{kb*4*QWORDS},{4*QWORDS},1]) {{id = 0 : i64, metadata = @qL2_{col}}} : memref<{ni}xi32>',
                  f'      aiex.npu.dma_memcpy_nd(%o[0,0,0,{(col*ng+group)*outsize}][1,1,{outsize//16},16][0,0,16,1]) {{id = 2 : i64, metadata = @outL2_{col},issue_token = true}} : memref<{no}xi32>',
                  f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,{offset}][{mg},1,{kb},{KVWORDS}][0,0,{KVWORDS},1]) {{id = 3 : i64, metadata = @kvL2_{col}}} : memref<{ni}xi32>']
            # Reuse the finite Shim BD slots only after their consumers finish.
            for col in range(4): lines.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')
        return text+'\n'.join(lines+['    }','  }','}'])+'\n'
    for col in range(4):
        if mode=='norm':
            qdims=f'[1,1,{128*QWORDS//16},16][0,0,16,1]'; qoffset=col*128*QWORDS
            kvdims=f'[1,1,32,{KVWORDS}][0,0,0,1]'; kvoffset=QCOUNT*QWORDS+col*64*KVWORDS
            outsize=128*OUT
        else: raise ValueError('unknown mode')
        lines += [
          f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,{qoffset}]{qdims}) {{id = 0 : i64, metadata = @qL2_{col}}} : memref<{ni}xi32>',
          f'      aiex.npu.dma_memcpy_nd(%o[0,0,0,{col*outsize}][1,1,{outsize//16},16][0,0,16,1]) {{id = 2 : i64, metadata = @outL2_{col},issue_token = true}} : memref<{no}xi32>',
          f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,{kvoffset}]{kvdims}) {{id = 3 : i64, metadata = @kvL2_{col}}} : memref<{ni}xi32>']
    for col in range(4): lines.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')
    return text+'\n'.join(lines+['    }','  }','}'])+'\n'


def audit(out):
    audits=[]
    for elf in sorted((out/'prj').rglob('*.elf')):
        col,row=map(int,re.search(r'core_(\d+)_(\d+)\.elf$',elf.name).groups())
        symbols=attention.command([attention.PEANO/'bin/llvm-nm.exe',elf],out)
        if attention.software_float_helpers(symbols): raise ValueError('software FP helper')
        ld=(out/f'prj/ldScripts_main_core_{col}_{row}.ld.script').read_text()
        spans=[(int(a,16)-0x70000,int(a,16)-0x70000+int(n,16),name) for a,name,n in re.findall(
          r'\. = (0x[0-9A-Fa-f]+);\s*\n\s*(\w+) = \.;\s*\n\. \+= (0x[0-9A-Fa-f]+);',ld)]
        for name in ('exp_ilut_ab','exp_ilut_cd','exp_flut_ab','exp_flut_cd'):
            m=re.search(r'^([0-9a-fA-F]+) [A-Za-z] '+name+r'$',symbols,re.M)
            if not m: raise ValueError('missing LUT')
            a=int(m[1],16)-0x70000; spans.append((a,a+1024,name))
        spans.sort()
        if len(spans)<10 or any(a < -0x30000 or b>65536 or a//65536!=(b-1)//65536 for a,b,_ in spans): raise ValueError('memory bounds')
        if any(a[1]>b[0] for a,b in zip(spans,spans[1:])): raise ValueError('memory overlap')
        sizes=attention.command([attention.PEANO/'bin/llvm-size.exe','-A',elf],out)
        code=sum(int(s) for s in re.findall(r'^\.text\S*\s+(\d+)\s+',sizes,re.M))
        report=attention.command([attention.PEANO/'bin/llvm-readobj.exe','--stack-sizes',elf],out)
        frames=[int(s,16) for s in re.findall(r'Size: (0x[0-9a-fA-F]+)',report)]
        if not 0<code<=16384 or not frames or sum(frames)+256>STACK: raise ValueError('program/stack bounds')
        if f'. += 0x{STACK:X}; /* stack */'.lower() not in ld.lower(): raise ValueError('stack not reserved')
        audits.append(dict(core=[col,row],code_bytes=code,frames=frames,spans=spans,elf_sha256=attention.sha(elf)))
    if len(audits)!=16: raise ValueError('missing cores')
    return audits


def build(out,masks,gemm_k=1024,gemm_n=512,gemm_m=512,all_gemm=False):
    out.mkdir(parents=True,exist_ok=True)
    clang=attention.PEANO/'bin/clang++.exe'
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{attention.INSTALL/"include"}',f'-I{attention.UPSTREAM/"aie_runtime_lib/AIE2"}',
      '-Wno-attributes','-Wno-parentheses','-DATTENTION_SHARED_MATMUL=1']
    objects=[]
    for name,source in [('gemm',HERE/'gemm.cc'),('dispatch',HERE/'dispatch.cc'),('attention',HERE.parent/'laya_attention/kernel_shared.cc'),
                        ('norm',HERE.parent/'laya_norm/kernel.cc'),
                        ('lut',attention.UPSTREAM/'aie_runtime_lib/AIE2/lut_based_ops.cpp')]:
        obj=out/f'{name}.o'; objects.append(obj)
        attention.command([clang,*flags,'-c',source,'-o',obj],out)
    dispatch_code=attention.command([attention.PEANO/'bin/llvm-objdump.exe','-d',out/'dispatch.o'],out)
    if not re.search(r'mov\s+crRnd,\s+#0x0\b',dispatch_code): raise ValueError('compiler removed the rounding-state fence')
    attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    undefined=attention.command([attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out)
    if undefined.strip(): raise ValueError('unresolved program helper: '+undefined)
    layouts={}
    requested=[('attention','attention',None),('norm','norm',None),('gemm','gemm',(gemm_m,gemm_k,gemm_n))]
    if all_gemm: requested.extend((f'gemm_{m}_{k}_{n}','gemm',(m,k,n)) for m,k,n in MODEL_GEMMS)
    for name,mode,shape in requested:
        m,k,n=shape if shape else (512,1024,512)
        folder=out/name; folder.mkdir(exist_ok=True)
        import shutil
        shutil.copyfile(out/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(harness(mode,k,n,m),encoding='utf-8')
        attention.command([attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        ni,no=geometry(mode,k,n,m)
        cdo={p.name:attention.sha(p) for p in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if len(cdo)!=3: raise ValueError('missing initialization payloads')
        layouts[name]=dict(kind=mode,shape=shape,audits=audit(folder),cdo=cdo,xclbin=attention.sha(folder/'final.xclbin'),instructions=attention.sha(folder/'insts.bin'),input_words=ni,output_words=no)
    if any(layouts['attention']['audits']!=value['audits'] for value in layouts.values()): raise ValueError('different persistent core programs')
    if any(layouts['attention']['cdo']!=value['cdo'] for value in layouts.values()): raise ValueError('different persistent initialization')
    meta=dict(sources=sources(),layouts=layouts,masks_sha256=attention.sha(masks),compiler_flags=[f for f in flags if not f.startswith('-I')],rounding_fence_machine_verified=True,
      gates=dict(attention=[attention.ATOL,attention.RTOL,attention.L2],norm=[norm.ATOL,norm.RTOL,norm.L2],residual='FP32 RNE bits exact'),
      gemm_shape=[gemm_m,gemm_k,gemm_n],upstream={n:attention.sha(attention.UPSTREAM/n) for n in ('aie_runtime_lib/AIE2/lut_based_ops.cpp','aie_runtime_lib/AIE2/lut_based_ops.h')})
    attention.dump(out/'build.json',meta)
    print('TRANSFORMER_BUILD '+json.dumps(meta),flush=True)


def attention_packet(q,k,v,mask,poison):
    compact=shared.pack(q,k,v,mask,poison)
    data=np.zeros(INPUT_WORDS,np.uint32)
    data[:QCOUNT*QWORDS].reshape(QCOUNT,QWORDS)[:,:528]=compact[:QCOUNT*528].reshape(QCOUNT,528)
    data[QCOUNT*QWORDS:]=compact[QCOUNT*528:]
    return data


def norm_packet(values,poison,copy_control=False):
    small=norm.pack(*values,poison)
    data=np.zeros(INPUT_WORDS,np.uint32)
    q=data[:QCOUNT*QWORDS].reshape(QCOUNT,QWORDS)
    q[:]=small[norm.PARAM:].reshape(QCOUNT,norm.IN)
    q[:,2]=poison; q[:,4]=1; q[:,5]=int(copy_control)
    kv=data[QCOUNT*QWORDS:].reshape(4,64,KVWORDS)
    kv[:,0,:norm.PARAM]=small[:norm.PARAM]
    return data


def gemm_packet(a,b,poison):
    k,n=b.shape
    m=a.shape[0]
    if a.shape!=(m,k): raise ValueError('activation/weight dimensions disagree')
    mp,kp,npad=padded(m,128),padded(k,64),padded(n,256)
    ap=np.zeros((mp,kp),np.float32); bp=np.zeros((kp,npad),np.float32)
    ap[:m,:k]=a; bp[:k,:n]=b
    ni,_=geometry('gemm',k,n,m); kb=kp//64; ng=npad//256; mg=mp//128
    data=np.zeros(ni,np.uint32)
    q=data[:mg*4*kb*QWORDS].reshape(mg,kb,4,QWORDS)
    q[:,:,:,2]=poison; q[:,:,:,4]=2; q[:,0,:,6]=1; q[:,-1,:,7]=1
    abits=attention.bits(ap).reshape(mg,4,32,kb,64).transpose(0,3,1,2,4)
    tiles=abits.reshape(mg,kb,4,8,4,8,8).transpose(0,1,2,3,5,4,6).copy().reshape(mg,kb,4,2048)
    q[:,:,:,16:1040]=tiles.view(np.uint32)
    weights=data[mg*4*kb*QWORDS:].reshape(4,ng,kb,KVWORDS)
    bbits=attention.bits(bp).reshape(kb,64,ng,4,64).transpose(3,2,0,1,4)
    tiles=bbits.reshape(4,ng,kb,8,8,16,4).transpose(0,1,2,3,5,4,6).copy().reshape(4,ng,kb,4096)
    weights[:,:,:,0:2048]=tiles.view(np.uint32)
    return data


def gemm_output(raw,n,m=512):
    mp,npad=padded(m,128),padded(n,256)
    full=raw.reshape(4,npad//256,mp//128,4,32,64).transpose(2,3,4,1,0,5).reshape(mp,npad)
    if np.any(full[m:]) or np.any(full[:m,n:]): raise ValueError('GEMM padding is not zero')
    return full[:m,:n]


def gemm_case(seed,kind,k,n,m=512):
    rng=np.random.default_rng(seed)
    a=attention.rounded(rng.normal(0,.5,(m,k)))
    b=attention.rounded(rng.normal(0,.5,(k,n)))
    if kind=='zero': a.fill(0)
    if kind=='basis':
        a.fill(0); a[np.arange(m),(np.arange(m)*37+seed)%k]=1
    return a,b


def run(out,masks,model,smoke=False,local_first=False,copy_control=False):
    meta=json.loads((out/'build.json').read_text())
    if meta['sources']!=sources() or meta['masks_sha256']!=attention.sha(masks): raise ValueError('stale sources or masks')
    for mode,layout in meta['layouts'].items():
        if layout['xclbin']!=attention.sha(out/mode/'final.xclbin') or layout['instructions']!=attention.sha(out/mode/'insts.bin'):
            raise ValueError('changed artifact')
    from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign NPU context')
    params=norm.model_parameters(model) if model and not smoke else []
    masks_data=np.load(masks)
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python')
    import pyxrt as xrt
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'attention/final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    S=xrt.xclBOSyncDirection
    instructions={}
    for mode in meta['layouts']:
        words=np.frombuffer((out/mode/'insts.bin').read_bytes(),np.uint32)
        ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(words.tobytes(),0); ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        instructions[mode]=(ib,words.nbytes)
    ib=None
    max_input=max(layout['input_words'] for layout in meta['layouts'].values())
    max_output=max(layout['output_words'] for layout in meta['layouts'].values())
    a=xrt.bo(dev,max_input*4,xrt.bo.host_only,0); o=xrt.bo(dev,max_output*4,xrt.bo.host_only,0)
    av=np.frombuffer(a.map(),np.uint32); ovflat=np.frombuffer(o.map(),np.uint32)
    count=0; order=[]; samples={}
    def execute(mode,payload,poison):
        av.fill(poison); np.copyto(av[:len(payload)],payload); ovflat.fill(poison)
        ov=ovflat[:meta['layouts'][mode]['output_words']].reshape(-1,OUT)
        a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE); o.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        ib,nb=instructions[mode]
        start=time.perf_counter_ns(); h=kernel(3,ib,nb,a,o)
        if h.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('dispatch failed; no retry')
        ms=(time.perf_counter_ns()-start)/1e6
        o.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
        used={'attention':1024,'norm':2048,'gemm':2048}[meta['layouts'][mode]['kind']]
        if not np.array_equal(av[:len(payload)],payload) or not np.all(av[len(payload):]==poison): raise ValueError('input changed')
        if not np.all(ov[:,:16]==poison) or not np.all(ov[:,16+used:]==poison): raise ValueError('guard changed')
        if np.any(ov[:,16:16+used]==poison): raise ValueError('unwritten output')
        if not np.all(ovflat[ov.size:]==poison): raise ValueError('output overrun')
        order.append(mode)
        return ov[:,16:16+used].view(np.float32).copy(),ms
    try:
        print('TRANSFORMER_RUN '+json.dumps(dict(build=meta,smoke=smoke,copy_control=copy_control,contexts=1,model_sha256=attention.sha(model) if model else None,
          scope='standalone commands in one loaded program; not full Laya')),flush=True)
        attention_cases=[(s,kind,mode,valid) for s in ((11,) if smoke else (11,29,47))
          for kind in (('random',) if smoke else ('random','uniform','zero','peaked'))
          for mode in (('local','global') if local_first else ('global','local')) for valid in ((512,) if smoke else (512,193,1,0))]
        norm_cases=[(s,kind,norm.fixture(s,kind)) for s in ((11,) if smoke else (11,29,47))
          for kind in (('random',) if smoke else norm.KINDS)]
        real=norm.fixture(71,'random'); norm_cases.extend((71,name,(*real[:2],g,b)) for name,g,b in params)
        gemm_cases=[(key,layout['shape'],seed,kind) for key,layout in meta['layouts'].items() if layout['kind']=='gemm'
          for seed in ((11,) if smoke else (11,29,47))
          for kind in (('random',) if smoke else ('random','zero','basis'))]
        # Alternate command families to expose retained FIFO/lock state.
        for i in range(max(len(attention_cases),len(norm_cases),len(gemm_cases))):
            if i<len(attention_cases):
                seed,kind,mode,valid=attention_cases[i]
                q,k,v=attention.case(16,seed,kind); mask=masks_data[f'{mode}_{valid}']; ref=attention.oracle(q,k,v,mask)
                previous=None
                for poison in (0x7fc12345,0x7fc54321):
                    payload=attention_packet(q,k,v,mask,poison); raw,ms=execute('attention',payload,poison)
                    got=raw.reshape(16,512,64); verdict=attention.numeric(got,ref)
                    if previous is not None and not np.array_equal(got,previous): raise ValueError('poison dependence')
                    print('TRANSFORMER_CHECK '+json.dumps(dict(command='attention',seed=seed,kind=kind,mode=mode,valid=valid,poison=poison,**verdict)),flush=True)
                    if not verdict['ok']: raise ValueError('attention numerical gate')
                    previous=got; count+=1
                samples.setdefault(f'attention/{mode}/{valid}',[]).append(ms)
            if i<len(gemm_cases):
                key,shape,seed,kind=gemm_cases[i]; m,k,n=shape
                ga,gb=gemm_case(seed,kind,k,n,m); ref=ga.astype(np.float64)@gb.astype(np.float64); previous=None
                for poison in (0x7fc12345,0x7fc54321):
                    raw,ms=execute(key,gemm_packet(ga,gb,poison),poison)
                    got=gemm_output(raw,n,m); verdict=attention.numeric(got,ref)
                    if previous is not None and not np.array_equal(got,previous): raise ValueError('GEMM poison dependence')
                    print('TRANSFORMER_CHECK '+json.dumps(dict(command='gemm',seed=seed,kind=kind,shape=shape,poison=poison,**verdict)),flush=True)
                    if not verdict['ok']: raise ValueError('GEMM numerical gate')
                    previous=got; count+=1
                samples.setdefault(key,[]).append(ms)
            if i<len(norm_cases):
                seed,kind,values=norm_cases[i]; refz,ref=norm.oracle(*values); previous=None
                for poison in (0x7fc12345,0x7fc54321):
                    payload=norm_packet(values,poison,copy_control); raw,ms=execute('norm',payload,poison)
                    gotz,got=raw[:,:1024],raw[:,1024:]
                    diff=np.abs(got-ref); relative_l2=float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-30))
                    ok=np.array_equal(gotz.copy().view(np.uint32),refz.view(np.uint32)) and np.isfinite(got).all() and np.all(diff<=norm.ATOL+norm.RTOL*np.abs(ref)) and relative_l2<=norm.L2
                    if copy_control:
                        ok=np.array_equal(gotz.copy().view(np.uint32),values[0].view(np.uint32)) and np.array_equal(got.copy().view(np.uint32),values[1].view(np.uint32))
                    if previous is not None: ok=ok and np.array_equal(got,previous)
                    print('TRANSFORMER_CHECK '+json.dumps(dict(command='copy' if copy_control else 'norm',seed=seed,kind=kind,poison=poison,ok=bool(ok),maxabs=float(diff.max()),relative_l2=relative_l2)),flush=True)
                    if not ok: raise ValueError('norm numerical gate')
                    previous=got; count+=1
                samples.setdefault('norm',[]).append(ms)
        # Validate switching back to attention after the last norm, in the same
        # context. No output from either family is supplied by the CPU oracle.
        seed,kind,mode,valid=attention_cases[0]
        q,k,v=attention.case(16,seed,kind); mask=masks_data[f'{mode}_{valid}']; ref=attention.oracle(q,k,v,mask)
        got,_=execute('attention',attention_packet(q,k,v,mask,0x7fc12345),0x7fc12345)
        if not attention.numeric(got.reshape(16,512,64),ref)['ok']: raise ValueError('return-to-attention failure')
        print('TRANSFORMER_RESULT '+json.dumps(dict(checks=count,return_to_attention=True,
          command_switches=sum(a!=b for a,b in zip(order,order[1:])),contexts_created=1,correctness_dispatch_ms=samples,
          scope='qualification timings only, no same-sitting speed comparison')),flush=True)
    finally:
        instructions.clear(); ib=None
        del av,ovflat,a,o,kernel,ctx,xb,dev
        gc.collect()
        print('TRANSFORMER_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--masks',type=Path,required=True); ap.add_argument('--model',type=Path); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--local-first',action='store_true'); ap.add_argument('--copy-control',action='store_true')
    ap.add_argument('--gemm-k',type=int,default=1024); ap.add_argument('--gemm-n',type=int,default=512)
    ap.add_argument('--gemm-m',type=int,default=512); ap.add_argument('--all-gemm',action='store_true'); args=ap.parse_args()
    sys.path.insert(0,str(ROOT/'tools'))
    if args.command=='build': build(args.out.resolve(),args.masks,args.gemm_k,args.gemm_n,args.gemm_m,args.all_gemm)
    else: run(args.out.resolve(),args.masks,args.model,args.smoke,args.local_first,args.copy_control)
