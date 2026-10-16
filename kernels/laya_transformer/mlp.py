#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Composed MLP research harness: one dispatch, one arena, device-only intermediates.

CPU oracles inspect outputs only after the entire chain has completed. This is
not the full model. Existing primitive qualification remains independent.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import program as p

HERE=Path(__file__).resolve().parent
POISON=0x7fc12345
STACK=4608


def sources():
    return dict(p.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f)
      for f in (HERE/'mlp.py',HERE/'mlp.cc')})


def plan(m=512,k=1024,h=2624,n=1024):
    if h%64 or min(m,k,h,n)<=0: raise ValueError('positive shapes and 64-aligned hidden dimension required')
    mg=p.padded(m,128)//128; kb=p.padded(k,64)//64
    wi_ng=p.padded(2*h,256)//256; wo_ng=p.padded(n,256)//256; hk=h//64
    sizes={'input':p.geometry('gemm',k,2*h,m)[0], 'wi':4*wi_ng*mg*4*p.OUT,
      'geglu':hk*mg*4*p.OUT,'weight_out':4*wo_ng*hk*p.KVWORDS,'output':4*wo_ng*mg*4*p.OUT}
    offsets={}; cursor=16
    for name,size in sizes.items():
        offsets[name]=cursor; cursor+=size+16
    return dict(shape=[m,k,h,n],mg=mg,kb=kb,wi_ng=wi_ng,wo_ng=wo_ng,hk=hk,sizes=sizes,offsets=offsets,words=cursor)


def core_prefix():
    text=p.harness('gemm').split('    aie.runtime_sequence')[0]
    text=text.replace('    func.func private @gemm_step(',
      f'    func.func private @chain_gemm_step(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>,i32)\n'
      f'    func.func private @chain_geglu(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32)\n    func.func private @gemm_step(')
    for col in range(4):
        for row in range(4):
            i=col*4+row
            start=text.index(f'    %core{i} = aie.core')
            end=text.index(' {link_with',start)
            body=text[start:end]
            body=body.replace('      %c0 =',f'      %c3 = arith.constant 3 : index\n      %c15 = arith.constant 15 : index\n      %three = arith.constant 3 : i32\n      %two = arith.constant 2 : i32\n      %c0 =')
            body=body.replace('      scf.for %it',f'      memref.store %zero,%control{i}[%c15] : memref<16xi32>\n      scf.for %it')
            body=body.replace(f'        %command = memref.load %q[%c4] : memref<{p.QWORDS}xi32>',f'''        %override = memref.load %control{i}[%c0] : memref<16xi32>
        %rowi = memref.load %control{i}[%c3] : memref<16xi32>
        %row = arith.index_cast %rowi : i32 to index
        %legacy = arith.cmpi eq, %override,%zero : i32
        %inband = memref.load %q[%c4] : memref<{p.QWORDS}xi32>
        %command = arith.select %legacy,%inband,%override : i32
        %is_geglu = arith.cmpi eq, %command,%three : i32
        %is_gemm = arith.cmpi eq, %command,%two : i32
        %pos = memref.load %control{i}[%c15] : memref<16xi32>
        %kblocks = memref.load %control{i}[%c1] : memref<16xi32>
        %next = arith.addi %pos,%one : i32
        %final_k = arith.cmpi eq, %next,%kblocks : i32
        %reset_bool = arith.cmpi eq, %pos,%zero : i32
        %reset = arith.extui %reset_bool : i1 to i32
        %next_pos = arith.select %final_k,%zero,%next : i32
        %counter = arith.select %is_gemm,%next_pos,%zero : i32
        memref.store %counter,%control{i}[%c15] : memref<16xi32>
        %rtp_poison = memref.load %control{i}[%c2] : memref<16xi32>
        %inband_poison = memref.load %q[%c2] : memref<{p.QWORDS}xi32>
        %fill = arith.select %legacy,%inband_poison,%rtp_poison : i32''')
            body=body.replace('        %count = arith.select %is_attention,%c16,%c1 : index',
              '        %non_attention_count = arith.select %is_geglu,%c4,%c1 : index\n        %count = arith.select %is_attention,%c16,%non_attention_count : index')
            old=f'              func.call @gemm_step(%q,%kv,%state{i}) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>) -> ()'
            new=f'''              scf.if %is_geglu {{
                %mine = arith.cmpi eq, %k,%row : index
                scf.if %mine {{
                  %sg = aie.objectfifo.acquire @outF_{i}(Produce,1) : !aie.objectfifosubview<memref<{p.OUT}xi32>>
                  %go = aie.objectfifo.subview.access %sg[0] : !aie.objectfifosubview<memref<{p.OUT}xi32>> -> memref<{p.OUT}xi32>
                  func.call @chain_geglu(%q,%kv,%go,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()
                  aie.objectfifo.release @outF_{i}(Produce,1)
                }}
              }} else {{
                scf.if %legacy {{
{old}
                }} else {{
                  func.call @chain_gemm_step(%q,%kv,%state{i},%reset) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.STATE}xi32>,i32) -> ()
                }}
              }}'''
            if old not in body: raise ValueError('core source changed')
            body=body.replace(old,new)
            body=body.replace('        %is_last = arith.cmpi ne, %last, %zero : i32',
              '        %header_last = arith.cmpi ne, %last, %zero : i32\n        %is_last = arith.select %legacy,%header_last,%final_k : i1')
            body=body.replace('        %emit = arith.andi %not_norm,%finish : i1',
              '        %not_geglu = arith.cmpi ne, %command,%three : i32\n        %normal_finish = arith.andi %not_norm,%finish : i1\n        %emit = arith.andi %not_geglu,%normal_finish : i1')
            body=body.replace(f'          %poison = memref.load %q[%c2] : memref<{p.QWORDS}xi32>', '          %poison = arith.addi %fill,%zero : i32')
            decl=f'    %control{i} = aie.buffer(%t{i}) {{sym_name = "chain_control_{i}"}} : memref<16xi32>\n'
            text=text[:start]+decl+body+text[end:]
    return text.replace('stack_size = 4096 : i32',f'stack_size = {STACK} : i32')


def runtime(pl,poison=POISON):
    mg,kb,wi_ng,wo_ng,hk=(pl[x] for x in ('mg','kb','wi_ng','wo_ng','hk'))
    off=pl['offsets']; words=pl['words']
    lines=[f'    aie.runtime_sequence(%a: memref<{words}xi32>,%o: memref<{words}xi32>) {{']
    constants={}
    def constant(v):
        if v not in constants:
            name=f'%value{len(constants)}'; constants[v]=name
            # i32 literal is signed in MLIR.
            signed=v if v<0x80000000 else v-0x100000000
            lines.append(f'      {name} = arith.constant {signed} : i32')
        return constants[v]
    def control(op,kcount):
        for index,val in enumerate((op,kcount,poison)):
            value=constant(val)
            for i in range(16): lines.append(f'      "aiex.npu.rtp_write"({value}) {{buffer = @chain_control_{i}, index = {index} : ui32}} : (i32) -> ()')
        for i in range(16):
            value=constant(i%4)
            lines.append(f'      "aiex.npu.rtp_write"({value}) {{buffer = @chain_control_{i}, index = 3 : ui32}} : (i32) -> ()')
    def dma(col,which,offset,dims,strides,output=False):
        var='%o' if output else '%a'; bid={'q':0,'kv':3,'out':2}[which]
        attr=',issue_token = true' if output else ''
        lines.append(f'      aiex.npu.dma_memcpy_nd({var}[0,0,0,{offset}][{",".join(map(str,dims))}][{",".join(map(str,strides))}]) {{id = {bid} : i64, metadata = @{which}L2_{col}{attr}}} : memref<{words}xi32>')
    def wait(cols):
        for col in cols: lines.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')
    control(2,kb)
    for group in range(wi_ng):
        for col in range(4):
            dma(col,'q',off['input'],[1,mg,kb,4*p.QWORDS],[0,kb*4*p.QWORDS,4*p.QWORDS,1])
            size=mg*4*p.OUT
            dma(col,'out',off['wi']+(col*wi_ng+group)*size,[1,1,size//16,16],[0,0,16,1],True)
            dma(col,'kv',off['input']+mg*4*kb*p.QWORDS+(col*wi_ng+group)*kb*p.KVWORDS,[mg,1,kb,p.KVWORDS],[0,0,p.KVWORDS,1])
        wait(range(4))
    control(3,1)
    for first in range(0,hk,4):
        cols=[]
        for f in range(first,min(first+4,hk)):
            col=f%4; cols.append(col); v=f+hk
            gate=off['wi']+((f%4)*wi_ng+f//4)*mg*4*p.OUT
            value=off['wi']+((v%4)*wi_ng+v//4)*mg*4*p.OUT
            dma(col,'q',gate,[1,mg,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            size=mg*4*p.OUT
            dma(col,'out',off['geglu']+f*size,[1,1,size//16,16],[0,0,16,1],True)
            dma(col,'kv',value,[1,mg,4,p.KVWORDS],[0,4*p.OUT,p.OUT,1])
        wait(cols)
    control(2,hk)
    for group in range(wo_ng):
        # The packet gap plus K and row strides already use the Shim's
        # dimensions after splitting the long inner span. Keep M groups
        # separate, and wait before reusing each finite BD slot.
        for mgroup in range(mg):
            for col in range(4):
                dma(col,'q',off['geglu']+mgroup*4*p.OUT,[1,hk,4,p.QWORDS],[0,mg*4*p.OUT,p.OUT,1])
                size=4*p.OUT
                dma(col,'out',off['output']+((col*wo_ng+group)*mg+mgroup)*size,[1,1,size//16,16],[0,0,16,1],True)
                dma(col,'kv',off['weight_out']+(col*wo_ng+group)*hk*p.KVWORDS,[1,1,hk,p.KVWORDS],[0,0,p.KVWORDS,1])
            wait(range(4))
    return '\n'.join(lines+['    }','  }','}'])+'\n'


def build(out,shape):
    pl=plan(*shape); out.mkdir(parents=True,exist_ok=True)
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}',f'-I{p.attention.UPSTREAM/"aie_runtime_lib/AIE2"}',
      '-Wno-attributes','-Wno-parentheses','-DATTENTION_SHARED_MATMUL=1']
    clang=p.attention.PEANO/'bin/clang++.exe'; objects=[]
    for name,src in [('mlp',HERE/'mlp.cc'),('gemm',HERE/'gemm.cc'),('dispatch',HERE/'dispatch.cc'),
      ('attention',HERE.parent/'laya_attention/kernel_shared.cc'),('norm',HERE.parent/'laya_norm/kernel.cc'),
      ('lut',p.attention.UPSTREAM/'aie_runtime_lib/AIE2/lut_based_ops.cpp')]:
        obj=out/f'{name}.o'; objects.append(obj); p.attention.command([clang,*flags,'-c',src,'-o',obj],out)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved helper')
    (out/'harness.mlir').write_text(core_prefix()+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=STACK
        audits=p.audit(out)
    finally: p.STACK=before
    meta=dict(sources=sources(),plan=pl,audits=audits,stack_bytes=STACK,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin'))
    p.attention.dump(out/'build.json',meta)
    print('MLP_BUILD '+json.dumps(meta),flush=True)


def fixture(shape,seed,kind):
    m,k,h,n=shape; rng=np.random.default_rng(seed)
    x=p.attention.rounded(rng.normal(0,.5,(m,k)))
    wi=p.attention.rounded(rng.normal(0,.04,(k,2*h)))
    wo=p.attention.rounded(rng.normal(0,.04,(h,n)))
    if kind=='zero': x.fill(0)
    elif kind=='basis':
        x.fill(0); x[np.arange(m),(np.arange(m)*37+seed)%k]=1
    return x,wi,wo


def make_arena(pl,x,wi,wo,poison):
    m,k,h,n=pl['shape']; off=pl['offsets']; sizes=pl['sizes']
    data=np.full(pl['words'],poison,np.uint32)
    data[off['input']:off['input']+sizes['input']]=p.gemm_packet(x,wi,poison)
    # Only static weight packets are used from this independent packing.
    packed=p.gemm_packet(np.zeros((1,h),np.float32),wo,poison)
    weights=packed[4*pl['hk']*p.QWORDS:]
    assert len(weights)==sizes['weight_out']
    data[off['weight_out']:off['weight_out']+len(weights)]=weights
    return data


def matrix(data,pl,name,n):
    raw=data[pl['offsets'][name]:pl['offsets'][name]+pl['sizes'][name]].reshape(-1,p.OUT)
    return p.gemm_output(raw[:,16:2064].copy().view(np.float32),n,pl['shape'][0])


def gated(data,pl):
    raw=data[pl['offsets']['geglu']:pl['offsets']['geglu']+pl['sizes']['geglu']].reshape(pl['hk'],pl['mg'],4,p.OUT)
    bits=raw[:,:,:,16:1040].copy().view(np.uint16)
    tiles=bits.reshape(pl['hk'],pl['mg'],4,8,8,4,8).transpose(0,1,2,3,5,4,6).reshape(pl['hk'],pl['mg'],4,32,64)
    full=tiles.transpose(1,2,3,0,4).reshape(pl['mg']*128,pl['shape'][2])
    if np.any(full[pl['shape'][0]:]): raise ValueError('GeGLU row padding changed')
    return (full[:pl['shape'][0]].astype(np.uint32)<<16).view(np.float32)


def model_pairs(path,shape):
    import onnx
    model=onnx.load(str(path),load_external_data=False)
    initial={t.name:t for t in model.graph.initializer}
    products=[node for node in model.graph.node if node.op_type=='MatMul' and node.input[1] in initial]
    pairs=[]
    for i,node in enumerate(products[:-1]):
        if tuple(initial[node.input[1]].dims)!=(1024,5248): continue
        next_node=products[i+1]
        if tuple(initial[next_node.input[1]].dims)!=(2624,1024): raise ValueError('model MLP pair changed')
        pairs.append((initial[node.input[1]],initial[next_node.input[1]]))
    if len(pairs)!=28 or shape!=[512,1024,2624,1024]: raise ValueError('model MLP shape/count changed')
    return pairs


def isolate(gotwi,gotg,gotout,refwi,refg,wo):
    """Post-dispatch diagnosis only: none of these tensors feeds the device."""
    erf=np.vectorize(math.erf,otypes=[np.float64]); hidden=gotg.shape[1]
    gate=gotwi[:,:hidden].astype(np.float64); value=gotwi[:,hidden:].astype(np.float64)
    from_device_wi=p.attention.rounded(gate*.5*(1+erf(gate/np.sqrt(2)))*value)
    from_exact_wi=p.attention.rounded(refg)
    delta_wi=(from_device_wi.astype(np.float64)-from_exact_wi.astype(np.float64))@wo.astype(np.float64)
    delta_gelu=(gotg.astype(np.float64)-from_device_wi.astype(np.float64))@wo.astype(np.float64)
    refout=from_exact_wi.astype(np.float64)@wo.astype(np.float64)
    device_product_ref=gotg.astype(np.float64)@wo.astype(np.float64)
    residual=gotout.astype(np.float64)-device_product_ref
    total=gotout.astype(np.float64)-refout
    error=np.max(np.abs(total-(delta_wi+delta_gelu+residual)))
    if error>1e-10: raise ValueError('error decomposition does not close')
    def ordered(a):
        bits=p.attention.bits(a).astype(np.int32)
        return np.where(bits&0x8000,0x8000-(bits&0x7fff),0x8000+bits)
    def transition(a,b):
        distance=np.abs(ordered(a)-ordered(b))
        return dict(changed=int(np.count_nonzero(distance)),over_one_ulp=int(np.count_nonzero(distance>1)),max_ulp=int(distance.max()))
    def metrics(delta):
        return dict(maxabs=float(np.max(np.abs(delta))),l2=float(np.sqrt(np.sum(delta*delta))))
    return dict(elements=gotg.size,linear_rounding=transition(from_exact_wi,from_device_wi),
      nonlinear_rounding=transition(from_device_wi,gotg),first_product_contribution=metrics(delta_wi),
      nonlinear_contribution=metrics(delta_gelu),last_product_contribution=metrics(residual),
      total=metrics(total),decomposition_maxabs=float(error),
      scope='diagnostic decomposition; independent complete-chain gate unchanged')


def run(out,smoke=False,model=None,collect_chain_failures=False,isolation=False):
    meta=json.loads((out/'build.json').read_text()); pl=meta['plan']
    if meta['sources']!=sources(): raise ValueError('stale source build')
    for path,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
        if p.attention.sha(out/path)!=meta[key]: raise ValueError('changed artifact')
    sys.path.insert(0,str(p.ROOT/'tools'))
    from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    erf=np.vectorize(math.erf,otypes=[np.float64])
    pairs=model_pairs(model,pl['shape']) if model else []
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python')
    import pyxrt as xrt
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    words=np.frombuffer((out/'insts.bin').read_bytes(),np.uint32)
    ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1))
    ib.write(words.tobytes(),0); S=xrt.xclBOSyncDirection; ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    # Keep the full allocation group ID. Only bank bits are compared.
    ga,go=kernel.group_id(3),kernel.group_id(4)
    if ga&0xffff!=go&0xffff: raise ValueError('arena bank connectivity differs')
    arena=xrt.bo(dev,pl['words']*4,xrt.bo.host_only,ga); mapped=np.frombuffer(arena.map(),np.uint32)
    checks=0; timings=[]; hrun=None; chain_failures=[]
    print('MLP_RUN '+json.dumps(dict(build=meta,scope='one composed MLP, not complete Laya',
      interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,kernel_calls_per_chain=1,
      gates=dict(linear_atol=.0002,linear_rtol=.0002,linear_l2=.00001,geglu_atol=.000002,geglu_rtol=.0078125,
        geglu_l2=.004,chain_atol=.0002,chain_rtol=.002,chain_l2=.002),
      model_sha256=p.attention.sha(model) if model else None,model_weight_pairs=len(pairs),collect_chain_failures=collect_chain_failures,isolation=isolation,
      activations='synthetic; model weights tested separately when requested')),flush=True)
    def check_numeric(got,ref,atol,rtol,l2):
        diff=np.abs(got-ref); rel=float(np.sqrt(np.sum(diff*diff))/max(np.sqrt(np.sum(ref*ref)),1e-30))
        return dict(ok=bool(np.isfinite(got).all() and np.all(diff<=atol+rtol*np.abs(ref)) and rel<=l2),maxabs=float(diff.max()),relative_l2=rel)
    try:
        for seed in (((11,) if smoke else (11,29,47))+((71,) if pairs else ())):
            kinds=tuple(f'model-layer-{i}' for i in range(len(pairs))) if seed==71 else (('random',) if smoke else ('random','zero','basis'))
            for kind in kinds:
                x,wi,wo=fixture(pl['shape'],seed,kind); hidden=pl['shape'][2]
                provenance=None
                if seed==71:
                    import onnx
                    index=int(kind.rsplit('-',1)[1]); tensors=pairs[index]
                    originals=[onnx.numpy_helper.to_array(t,base_dir=str(model.parent)) for t in tensors]
                    provenance={t.name:hashlib.sha256(a.tobytes()).hexdigest() for t,a in zip(tensors,originals)}
                    wi,wo=map(p.attention.rounded,originals)
                refwi=x.astype(np.float64)@wi.astype(np.float64)
                gate,value=refwi[:,:hidden],refwi[:,hidden:]
                refg=gate*.5*(1+erf(gate/np.sqrt(2)))*value
                refout=p.attention.rounded(refg).astype(np.float64)@wo.astype(np.float64)
                previous=None
                # Runtime poison is compiled. Two initial arena poisons expose
                # stale/unwritten payloads, while output guards use POISON.
                for fill in (0x7fc12345,0x7fc54321):
                    payload=make_arena(pl,x,wi,wo,fill); mapped[:]=payload
                    arena.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
                    start=time.perf_counter_ns(); hrun=kernel(3,ib,words.nbytes,arena,arena)
                    if hrun.wait(10000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('chain timeout; no retry')
                    ms=(time.perf_counter_ns()-start)/1e6
                    # The first host read/sync follows the complete three-stage dispatch.
                    arena.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE); observed=mapped.copy()
                    changed=np.zeros(pl['words'],bool)
                    for name in ('wi','geglu','output'):
                        offset,size=pl['offsets'][name],pl['sizes'][name]; changed[offset:offset+size]=True
                        packets=observed[offset:offset+size].reshape(-1,p.OUT)
                        used=1024 if name=='geglu' else 2048
                        if not np.all(packets[:,:16]==POISON) or not np.all(packets[:,16+used:]==POISON): raise ValueError(name+' guard/unwritten mismatch')
                        if np.any(packets[:,16:16+used]==fill) or np.any(packets[:,16:16+used]==POISON): raise ValueError(name+' unwritten payload')
                    if not np.array_equal(observed[~changed],payload[~changed]): raise ValueError('input, static weights or arena gap changed')
                    gotwi=matrix(observed,pl,'wi',2*hidden); gotg=gated(observed,pl); gotout=matrix(observed,pl,'output',pl['shape'][3])
                    # This reference uses device GeGLU only after completion,
                    # isolating the second product; it never supplies an input.
                    secondref=gotg.astype(np.float64)@wo.astype(np.float64)
                    verdicts=dict(wi=check_numeric(gotwi,refwi,.0002,.0002,.00001),
                      geglu=check_numeric(gotg,refg,.000002,.0078125,.004),
                      wo=check_numeric(gotout,secondref,.0002,.0002,.00001),
                      chain=check_numeric(gotout,refout,.0002,.002,.002))
                    if isolation and previous is None:
                        print('MLP_ISOLATION '+json.dumps(dict(seed=seed,kind=kind,**isolate(gotwi,gotg,gotout,refwi,refg,wo))),flush=True)
                    if not verdicts['geglu']['ok']:
                        pos=np.unravel_index(np.argmax(np.abs(gotg-refg)),gotg.shape)
                        print('MLP_GEGLU_DIAGNOSTIC '+json.dumps(dict(position=list(map(int,pos)),
                          got=float(gotg[pos]),reference=float(refg[pos]),gate=float(gotwi[pos]),value=float(gotwi[pos[0],pos[1]+hidden]),
                          first_got=gotg[0,:16].tolist(),first_ref=refg[0,:16].tolist(),
                          first_gate=gotwi[0,:16].tolist(),first_value=gotwi[0,hidden:hidden+16].tolist())),flush=True)
                    if previous is not None and not np.array_equal(gotout,previous): raise ValueError('poison dependence')
                    print('MLP_CHECK '+json.dumps(dict(seed=seed,kind=kind,fill=fill,model_weights=provenance,dispatch_ms_correctness_only=ms,verdicts=verdicts)),flush=True)
                    if not all(verdicts[key]['ok'] for key in ('wi','geglu','wo')): raise ValueError('per-stage numerical gate')
                    if not verdicts['chain']['ok']:
                        chain_failures.append(dict(seed=seed,kind=kind,fill=fill,**verdicts['chain']))
                        if not collect_chain_failures: raise ValueError('composed numerical gate')
                    previous=gotout; checks+=1; timings.append(ms)
        print('MLP_RESULT '+json.dumps(dict(checks=checks,contexts_created=1,kernel_calls_per_chain=1,
          interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,shape=pl['shape'],
          model_weight_pairs=len(pairs),strict_chain_failures=chain_failures,strict_chain_pass=not chain_failures,
          scope='correctness only; synthetic activations, no full-model accuracy or speed claim')),flush=True)
        if chain_failures: raise ValueError('strict chain gate failed; collected all cases without changing tolerances')
    finally:
        hrun=None
        del mapped,arena,ib,kernel,ctx,xb,dev
        gc.collect()
        print('MLP_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--shape',nargs=4,type=int,default=[512,1024,2624,1024]); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--model',type=Path)
    ap.add_argument('--collect-chain-failures',action='store_true'); ap.add_argument('--isolation',action='store_true'); args=ap.parse_args()
    if args.command=='build': build(args.out.resolve(),args.shape)
    else: run(args.out.resolve(),args.smoke,args.model,args.collect_chain_failures,args.isolation)
