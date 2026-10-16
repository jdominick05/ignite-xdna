#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone 16-core fused residual/LayerNorm qualification; no model execution."""
import argparse
import gc
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'laya_attention'))
import probe as common

IN,OUT,PARAM,ROWS,STACK=2064,2080,2048,512,4096
ATOL,RTOL,L2=2e-4,2e-4,2e-4
KINDS=('random','constant','near_constant','cancel','offset','no_residual','tiny','large','mixed_exponents',
       'signed_zero','rounding_edges','subnormal')


def harness():
    lines=['module {','  aie.device(npu1) {',
      f'    func.func private @residual_norm(memref<{IN}xi32>,memref<{PARAM}xf32>,memref<{OUT}xi32>)']
    workers=[]
    for c in range(4):
        lines += [f'    %s{c} = aie.tile({c},0)',f'    %m{c} = aie.tile({c},1)']
        for r in range(4): lines.append(f'    %t{c*4+r} = aie.tile({c},{r+2})')
        targets=', '.join(f'%t{c*4+r}' for r in range(4))
        lines += [
          f'    aie.objectfifo @pL{c}(%s{c},{{%m{c}}},1 : i32) : !aie.objectfifo<memref<{PARAM}xf32>>',
          f'    aie.objectfifo @pF{c}(%m{c},{{{targets}}},1 : i32) : !aie.objectfifo<memref<{PARAM}xf32>>',
          f'    aie.objectfifo.link [@pL{c}] -> [@pF{c}]([] [0])',
          f'    aie.objectfifo @iL{c}(%s{c},{{%m{c}}},1 : i32) : !aie.objectfifo<memref<{4*IN}xi32>>',
          f'    aie.objectfifo @oL{c}(%m{c},{{%s{c}}},1 : i32) : !aie.objectfifo<memref<{4*OUT}xi32>>']
        for r in range(4):
            i=c*4+r
            lines += [
              f'    aie.objectfifo @iF{i}(%m{c},{{%t{i}}},1 : i32) : !aie.objectfifo<memref<{IN}xi32>>',
              f'    aie.objectfifo @oF{i}(%t{i},{{%m{c}}},1 : i32) : !aie.objectfifo<memref<{OUT}xi32>>']
            workers.append(f'''    %core{i} = aie.core(%t{i}) {{
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %rows = arith.constant 32 : index
      %forever = arith.constant 4294967295 : index
      scf.for %it = %z to %forever step %one {{
        %ps = aie.objectfifo.acquire @pF{c}(Consume,1) : !aie.objectfifosubview<memref<{PARAM}xf32>>
        %p = aie.objectfifo.subview.access %ps[0] : !aie.objectfifosubview<memref<{PARAM}xf32>> -> memref<{PARAM}xf32>
        scf.for %j = %z to %rows step %one {{
          %is = aie.objectfifo.acquire @iF{i}(Consume,1) : !aie.objectfifosubview<memref<{IN}xi32>>
          %a = aie.objectfifo.subview.access %is[0] : !aie.objectfifosubview<memref<{IN}xi32>> -> memref<{IN}xi32>
          %os = aie.objectfifo.acquire @oF{i}(Produce,1) : !aie.objectfifosubview<memref<{OUT}xi32>>
          %o = aie.objectfifo.subview.access %os[0] : !aie.objectfifosubview<memref<{OUT}xi32>> -> memref<{OUT}xi32>
          func.call @residual_norm(%a,%p,%o) : (memref<{IN}xi32>,memref<{PARAM}xf32>,memref<{OUT}xi32>) -> ()
          aie.objectfifo.release @oF{i}(Produce,1)
          aie.objectfifo.release @iF{i}(Consume,1)
        }}
        aie.objectfifo.release @pF{c}(Consume,1)
      }}
      aie.end
    }} {{link_with = "kernel.o",stack_size = {STACK} : i32}}''')
        lines += [
          f'    aie.objectfifo.link [@iL{c}] -> ['+', '.join(f'@iF{c*4+r}' for r in range(4))+']([] ['+', '.join(str(r*IN) for r in range(4))+'])',
          '    aie.objectfifo.link ['+', '.join(f'@oF{c*4+r}' for r in range(4))+f'] -> [@oL{c}](['+', '.join(str(r*OUT) for r in range(4))+'] [])']
    lines.extend(workers)
    ni=PARAM+ROWS*IN; no=ROWS*OUT
    lines.append(f'    aie.runtime_sequence(%a: memref<{ni}xi32>,%o: memref<{no}xi32>) {{')
    for c in range(4):
        lines += [
          f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,0][1,1,{PARAM//16},16][0,0,16,1]) {{id = 0 : i64,metadata = @pL{c}}} : memref<{ni}xi32>',
          f'      aiex.npu.dma_memcpy_nd(%a[0,0,0,{PARAM+c*128*IN}][1,1,{128*IN//16},16][0,0,16,1]) {{id = 1 : i64,metadata = @iL{c}}} : memref<{ni}xi32>',
          f'      aiex.npu.dma_memcpy_nd(%o[0,0,0,{c*128*OUT}][1,1,{128*OUT//16},16][0,0,16,1]) {{id = 2 : i64,metadata = @oL{c},issue_token = true}} : memref<{no}xi32>']
    for c in range(4): lines.append(f'      aiex.npu.dma_wait {{symbol = @oL{c}}}')
    return '\n'.join(lines+['    }','  }','}'])+'\n'


def sources():
    return {str(p.relative_to(HERE.parents[1])).replace('\\','/'):common.sha(p) for p in (
      HERE/'kernel.cc',HERE/'probe.py',HERE.parent/'laya_attention/probe.py',HERE.parents[1]/'tools/laya_hybrid.py')}


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    clang=common.PEANO/'bin/clang++.exe'
    flags=['-O2','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
           '-D__AIE_API_AIE_ADF_HPP__',f'-I{common.INSTALL/"include"}','-Wno-attributes','-Wno-parentheses']
    common.command([clang,*flags,'-c',HERE/'kernel.cc','-o',out/'kernel.o'],out)
    undefined=common.command([common.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out)
    if undefined.strip(): raise ValueError(f'unresolved helpers: {undefined}')
    (out/'harness.mlir').write_text(harness(),encoding='utf-8')
    common.command([common.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={common.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    audits=[]
    for elf in sorted((out/'prj').rglob('*.elf')):
        c,r=map(int,re.search(r'core_(\d+)_(\d+)\.elf$',elf.name).groups())
        symbols=common.command([common.PEANO/'bin/llvm-nm.exe',elf],out)
        if common.software_float_helpers(symbols): raise ValueError('software float')
        sections=common.command([common.PEANO/'bin/llvm-size.exe','-A',elf],out)
        code=sum(int(s) for s in re.findall(r'^\.text\S*\s+(\d+)\s+',sections,re.M))
        stack=common.command([common.PEANO/'bin/llvm-readobj.exe','--stack-sizes',elf],out)
        frames=[int(s,16) for s in re.findall(r'Size: (0x[0-9a-fA-F]+)',stack)]
        ld=(out/f'prj/ldScripts_main_core_{c}_{r}.ld.script').read_text()
        spans=sorted((int(a,16)-0x70000,int(a,16)-0x70000+int(n,16),name) for a,name,n in re.findall(
          r'\. = (0x[0-9A-Fa-f]+);\s*\n\s*(\w+) = \.;\s*\n\. \+= (0x[0-9A-Fa-f]+);',ld))
        if not 0<code<=16384 or not frames or sum(frames)+256>STACK: raise ValueError('code/stack bounds')
        if f'. += 0x{STACK:X}; /* stack */'.lower() not in ld.lower(): raise ValueError('missing stack')
        if len(spans)<4 or any(a < -0x30000 or b>65536 or a//65536!=(b-1)//65536 for a,b,_ in spans): raise ValueError('memory bounds')
        if any(a[1]>b[0] for a,b in zip(spans,spans[1:])): raise ValueError('memory overlap')
        audits.append(dict(core=[c,r],code_bytes=code,frames=frames,spans=spans))
    if len(audits)!=16: raise ValueError('expected sixteen cores')
    meta=dict(sources=sources(),audits=audits,xclbin=common.sha(out/'final.xclbin'),instructions=common.sha(out/'insts.bin'),
              gates=dict(atol=ATOL,rtol=RTOL,relative_l2=L2,residual='exact FP32 RNE'),shape=[ROWS,1024],epsilon=1e-5,
              seeds=[11,29,47],cases=KINDS,poisons=['7fc12345','7fc54321'])
    common.dump(out/'build.json',meta)
    print('NORM_BUILD '+json.dumps(meta),flush=True)


def fixture(seed,kind):
    rng=np.random.default_rng(seed)
    x,r=[rng.normal(size=(ROWS,1024)).astype(np.float32) for _ in range(2)]
    g=rng.uniform(.25,1.75,1024).astype(np.float32)
    b=rng.normal(0,.1,1024).astype(np.float32)
    if kind=='constant': x.fill(3); r.fill(-1)
    if kind=='near_constant': x[:]=np.float32(20)+x*np.float32(1e-4); r.fill(0)
    if kind=='cancel': r=-x.copy()
    if kind=='offset': x+=np.float32(1000)
    if kind=='no_residual': r.fill(0); b.fill(0)
    if kind=='tiny': x*=np.float32(1e-6); r*=np.float32(1e-6)
    if kind=='large': x*=np.float32(1e6); r*=np.float32(1e6)
    if kind=='mixed_exponents':
        x=np.ldexp(x,rng.integers(-30,21,x.shape)).astype(np.float32)
        r=np.ldexp(r,rng.integers(-30,21,r.shape)).astype(np.float32)
    if kind=='signed_zero':
        x.fill(0); r.fill(0); x[:,::2]=np.float32(-0.0); r[:,::3]=np.float32(-0.0)
    if kind=='rounding_edges':
        xb=np.array([0x3f800000,0x3f800001,0xbf800000,0xbf800001,0x3f800000,0xbf800000,0x00800000,0x80800000],np.uint32)
        rb=np.array([0x33800000,0x33800000,0xb3800000,0xb3800000,0xbf7fffff,0x3f7fffff,1,0x80000001],np.uint32)
        x[:]=np.tile(xb.view(np.float32),128); r[:]=np.tile(rb.view(np.float32),128)
    if kind=='subnormal':
        for dest in (x,r):
            mag=rng.integers(0,0x1000000,dest.shape,dtype=np.uint32)
            sign=rng.integers(0,2,dest.shape,dtype=np.uint32)<<31
            dest.view(np.uint32)[:]=mag|sign
    return x,r,g,b


def oracle(x,r,g,b):
    # Residual addition is FP32 in the exported graph, then FP32 LayerNorm.
    z=np.add(x,r,dtype=np.float32)
    d=z.astype(np.float64)-z.astype(np.float64).mean(axis=-1,keepdims=True)
    return z,d/np.sqrt(np.mean(d*d,axis=-1,keepdims=True)+1e-5)*g+b


def pack(x,r,g,b,poison):
    a=np.zeros(PARAM+ROWS*IN,np.uint32)
    a[:PARAM].view(np.float32)[:]=np.concatenate((g,b))
    rows=a[PARAM:].reshape(4,32,4,IN)
    rows[:,:,:,0]=poison
    # Column-major core groups; groups of four consecutive rows are split/joined.
    rows[:,:,:,16:1040].view(np.float32)[:]=x.reshape(4,32,4,1024)
    rows[:,:,:,1040:].view(np.float32)[:]=r.reshape(4,32,4,1024)
    return a


def model_parameters(model):
    import onnx
    m=onnx.load(str(model),load_external_data=False)
    initial={t.name:t for t in m.graph.initializer}
    params=[]
    for n in m.graph.node:
        if n.op_type!='LayerNormalization': continue
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}
        if attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5):
            raise ValueError('unsupported normalization semantics')
        g=onnx.numpy_helper.to_array(initial[n.input[1]],base_dir=str(model.parent)).copy()
        b=onnx.numpy_helper.to_array(initial[n.input[2]],base_dir=str(model.parent)).copy() if len(n.input)>2 else np.zeros_like(g)
        if g.shape!=(1024,) or b.shape!=(1024,) or g.dtype!=np.float32 or b.dtype!=np.float32:
            raise ValueError('unsupported affine shape/type')
        params.append((n.name,g,b))
    if len(params)!=62: raise ValueError('expected 62 model norms')
    return params


def cpu_session():
    import onnx
    import onnxruntime as ort
    h=onnx.helper
    graph=h.make_graph([h.make_node('Add',['x','r'],['sum']),
      h.make_node('LayerNormalization',['sum','g','b'],['y'],axis=-1,epsilon=1e-5)],'residual_norm',
      [h.make_tensor_value_info(n,onnx.TensorProto.FLOAT,shape) for n,shape in (
        ('x',[512,1024]),('r',[512,1024]),('g',[1024]),('b',[1024]))],
      [h.make_tensor_value_info(n,onnx.TensorProto.FLOAT,[512,1024]) for n in ('sum','y')])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=9)
    so=ort.SessionOptions(); so.intra_op_num_threads=8; so.inter_op_num_threads=1
    so.add_session_config_entry('session.intra_op.allow_spinning','0')
    return ort.InferenceSession(model.SerializeToString(),so,providers=['CPUExecutionProvider'])


def run(out,calls,diagnostic=False,model=None):
    meta=json.loads((out/'build.json').read_text())
    if meta['sources']!=sources() or meta['xclbin']!=common.sha(out/'final.xclbin') or meta['instructions']!=common.sha(out/'insts.bin'):
        raise ValueError('stale build')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python')
    import pyxrt as xrt
    sys.path.insert(0,str(HERE.parents[1]/'tools'))
    from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign NPU context')
    params=model_parameters(model) if model else []
    cpu=cpu_session()
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kern=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    words=np.frombuffer((out/'insts.bin').read_bytes(),np.uint32)
    ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kern.group_id(1)); ib.write(words.tobytes(),0)
    S=xrt.xclBOSyncDirection
    ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    a=xrt.bo(dev,(PARAM+ROWS*IN)*4,xrt.bo.host_only,0)
    o=xrt.bo(dev,ROWS*OUT*4,xrt.bo.host_only,0)
    av=np.frombuffer(a.map(),np.uint32); ov=np.frombuffer(o.map(),np.uint32).reshape(ROWS,OUT)
    def execute(payload,poison):
        np.copyto(av,payload); ov.fill(poison)
        a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE); o.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        h=kern(3,ib,words.nbytes,a,o)
        if h.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('dispatch did not complete')
        o.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
        if not np.array_equal(av,payload): raise ValueError('input changed')
        if not np.all(ov[:,:16]==poison) or not np.all(ov[:,2064:]==poison): raise ValueError('guard changed')
        if np.any(ov[:,16:2064]==poison): raise ValueError('unwritten output')
        return ov[:,16:1040].view(np.float32).copy(),ov[:,1040:2064].view(np.float32).copy()
    count=0; failed=0; h=None
    try:
        print('NORM_RUN '+json.dumps(dict(build=meta,description='synthetic standalone; not full Laya',seeds=[11,29,47],
          cases=KINDS,poisons=['7fc12345','7fc54321'],model_sha256=common.sha(model) if model else None,
          model_affine_sha256={name:common.hashlib.sha256(g.tobytes()+b.tobytes()).hexdigest() for name,g,b in params})),flush=True)
        cases=[(seed,kind,fixture(seed,kind)) for seed in (11,29,47) for kind in KINDS]
        real=fixture(71,'random')
        cases.extend((71,name,(*real[:2],g,b)) for name,g,b in params)
        for seed,kind,vals in cases:
                z,ref=oracle(*vals); previous=None
                for poison in (0x7fc12345,0x7fc54321):
                    gotz,got=execute(pack(*vals,poison),poison)
                    diff=np.abs(got-ref); maxabs=float(diff.max())
                    l2=float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-30))
                    ok=np.array_equal(gotz.view(np.uint32),z.view(np.uint32)) and np.isfinite(got).all() and np.all(diff<=ATOL+RTOL*np.abs(ref)) and l2<=L2
                    if previous is not None: ok=ok and np.array_equal(got,previous)
                    print('NORM_CHECK '+json.dumps(dict(seed=seed,kind=kind,poison=poison,maxabs=maxabs,relative_l2=l2,
                      residual_maxabs=float(np.abs(gotz-z).max()),residual_mismatches=int(np.count_nonzero(gotz.view(np.uint32)!=z.view(np.uint32))),
                      mismatch_examples=[[float(gotz.flat[i]),float(z.flat[i])] for i in np.flatnonzero(gotz!=z)[:4]],passed=bool(ok))),flush=True)
                    if not ok:
                        failed+=1
                        if not diagnostic: raise ValueError('numerical gate')
                    previous=got; count+=1
        if failed: raise ValueError(f'numerical gate: {failed} failures, timing refused')
        # Correctness first. Timing below excludes Python checking/output copies.
        vals=fixture(11,'random'); payload=pack(*vals,0x7fc12345); np.copyto(av,payload)
        a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        samples=[]; end_to_end=[]; cpu_samples=[]
        for i in range(calls+3):
            start=time.perf_counter_ns(); h=kern(3,ib,words.nbytes,a,o)
            if h.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('timed dispatch')
            if i>=3: samples.append((time.perf_counter_ns()-start)/1e6)
        for i in range(calls+3):
            start=time.perf_counter_ns()
            np.copyto(av,pack(*vals,0x7fc12345)); a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
            h=kern(3,ib,words.nbytes,a,o)
            if h.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('end-to-end dispatch')
            o.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
            detached=ov[:,16:2064].copy()
            if i>=3: end_to_end.append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns(); cpu_result=cpu.run(None,dict(zip(('x','r','g','b'),vals)))
            if i>=3: cpu_samples.append((time.perf_counter_ns()-start)/1e6)
        refz,ref=oracle(*vals)
        if not np.array_equal(cpu_result[0],refz) or not np.all(np.abs(cpu_result[1]-ref)<=ATOL+RTOL*np.abs(ref)):
            raise ValueError('CPU comparison disagrees with oracle')
        print('NORM_RESULT '+json.dumps(dict(checks=count,dispatch_ms=samples,median_dispatch_ms=float(np.median(samples)),
          end_to_end_ms=end_to_end,median_end_to_end_ms=float(np.median(end_to_end)),cpu_ms=cpu_samples,
          median_cpu_ms=float(np.median(cpu_samples)),scope='512x1024 standalone residual add plus LayerNorm; no full-model claim')),flush=True)
    finally:
        h=None
        del av,ov,a,o,ib,kern,ctx,xb,dev
        gc.collect()
        print('NORM_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--calls',type=int,default=20); ap.add_argument('--diagnostic',action='store_true')
    ap.add_argument('--model',type=Path); args=ap.parse_args()
    if args.calls<1: ap.error('--calls must be positive')
    if args.command=='build': build(args.out.resolve())
    else: run(args.out.resolve(),args.calls,args.diagnostic,args.model)
