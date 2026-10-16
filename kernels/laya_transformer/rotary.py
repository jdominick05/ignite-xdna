#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""QKV projection -> model RoPE -> packet conversion -> attention, one dispatch.

Fixed trigonometric coefficients are constant-folded from the exported model
before inference. Learned activations remain device-owned between all stages.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import mlp
p=mlp.p
HERE=Path(__file__).resolve().parent


def sources():
    return dict(mlp.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f)
      for f in (HERE/'rotary.py',HERE/'rotary.cc')})


def coefficients(model,out):
    import onnx
    import onnxruntime as ort
    initial={t.name:t for t in model.graph.initializer}
    nodes=list(model.graph.node[33:45])
    if [n.op_type for n in nodes]!=['Unsqueeze','Cast','MatMul','Transpose','Concat','Cos','Sin','MatMul','Transpose','Concat','Cos','Sin']:
        raise ValueError('rotary graph changed')
    produced={name for n in nodes for name in n.output}
    needed={name for n in nodes for name in n.input if name and name not in produced}
    if needed-set(initial)!={'unsqueeze'}: raise ValueError('unexpected rotary graph input')
    graph=onnx.helper.make_graph(nodes,'fixed_rotary',[onnx.helper.make_tensor_value_info('unsqueeze',onnx.TensorProto.INT64,[1,512])],
      [onnx.helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None) for n in ('cos','sin','cos_1','sin_1')],
      [initial[n] for n in needed if n in initial])
    tiny=onnx.helper.make_model(graph,opset_imports=model.opset_import,ir_version=model.ir_version)
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.add_session_config_entry('session.intra_op.allow_spinning','0')
    sess=ort.InferenceSession(tiny.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    values=sess.run(None,{'unsqueeze':np.arange(512,dtype=np.int64)[None,:]})
    tables={}
    for mode,start in [('local',0),('global',2)]:
        c,s=(v.reshape(512,64) for v in values[start:start+2])
        if not np.array_equal(c[:,:32],c[:,32:]) or not np.array_equal(s[:,:32],s[:,32:]): raise ValueError('rotary halves differ')
        packed=np.zeros((16,p.KVWORDS),np.uint32)
        packed[:,:1024]=c[:,:32].copy().reshape(16,1024).view(np.uint32)
        packed[:,1024:2048]=s[:,:32].copy().reshape(16,1024).view(np.uint32)
        tables[mode]=packed
    np.savez(out/'coefficients.npz',**tables)


def plan():
    sizes=dict(input=p.geometry('gemm',1024,3072)[0],projection=768*p.OUT,coeff=16*p.KVWORDS,
      queries=512*p.OUT,keys=256*p.OUT,kv=256*p.OUT,attention=512*p.OUT)
    offsets={}; cursor=16
    for name,size in sizes.items(): offsets[name]=cursor; cursor+=size+16
    return dict(sizes=sizes,offsets=offsets,words=cursor)


class Sequence:
    def __init__(self,words):
        self.words=words; self.lines=[f'    aie.runtime_sequence(%a: memref<{words}xi32>,%o: memref<{words}xi32>) {{']; self.constants={}; self.current={}
    def constant(self,value):
        if value not in self.constants:
            name=f'%value{len(self.constants)}'; self.constants[value]=name
            signed=value if value<0x80000000 else value-0x100000000
            self.lines.append(f'      {name} = arith.constant {signed} : i32')
        return self.constants[value]
    def control(self,mode,kblocks=1,position=0,valid=512,local=0):
        if mode not in (0,1,2,4,5,6,7): raise ValueError('operation unavailable in the attention image')
        for index,value in enumerate((mode,kblocks,mlp.POISON,None,position,valid,local)):
            if index in self.current and self.current[index]==value: continue
            self.current[index]=value
            for i in range(16):
                name=self.constant(i%4 if value is None else value)
                self.lines.append(f'      "aiex.npu.rtp_write"({name}) {{buffer = @chain_control_{i}, index = {index} : ui32}} : (i32) -> ()')
    def dma(self,col,kind,offset,dims,strides,output=False,bid=None):
        if dims[1]==1 and dims[3]>1023 and strides[3]==1 and strides[2]!=dims[3]:
            if dims[3]%16: raise ValueError('packet span cannot be split into vector words')
            dims=[dims[0],dims[2],dims[3]//16,16]; strides=[strides[0],strides[2],16,1]
        var='%o' if output else '%a'; bid=({'q':0,'kv':3,'out':2}[kind] if bid is None else bid)
        attr=',issue_token = true' if output else ''
        self.lines.append(f'      aiex.npu.dma_memcpy_nd({var}[0,0,0,{offset}][{",".join(map(str,dims))}][{",".join(map(str,strides))}]) {{id = {bid} : i64, metadata = @{kind}L2_{col}{attr}}} : memref<{self.words}xi32>')
    def wait(self):
        for col in range(4): self.lines.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')
    def finish(self): return '\n'.join(self.lines+['    }','  }','}'])+'\n'


def prefix():
    text=mlp.core_prefix()
    text=text.replace('    func.func private @chain_geglu(',
      f'    func.func private @chain_elementwise(memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32,i32,i32,i32,i32,memref<{p.STATE}xi32>)\n    func.func private @chain_geglu(')
    # This composed harness controls every operation through RTP. Dropping
    # legacy packet-command dispatch also saves scarce instruction memory.
    text=text.replace('%legacy = arith.cmpi eq, %override,%zero : i32','%legacy = arith.constant false')
    text=text.replace('%is_geglu = arith.cmpi eq, %command,%three : i32','%is_geglu = arith.cmpi uge, %command,%three : i32')
    text=text.replace('%not_geglu = arith.cmpi ne, %command,%three : i32','%not_geglu = arith.cmpi ult, %command,%three : i32')
    for i in range(16):
        start=text.index(f'    %core{i} = aie.core'); end=text.index(' {link_with',start); body=text[start:end]
        body=body.replace('      %c3 =', '      %c5 = arith.constant 5 : index\n      %c6 = arith.constant 6 : index\n      %thirtytwo = arith.constant 32 : i32\n      %c3 =')
        body=body.replace('        %row = arith.index_cast',f'''        %base = memref.load %control{i}[%c4] : memref<16xi32>
        %valid = memref.load %control{i}[%c5] : memref<16xi32>
        %local = memref.load %control{i}[%c6] : memref<16xi32>
        %row_offset = arith.muli %rowi,%thirtytwo : i32
        %position = arith.addi %base,%row_offset : i32
        %row = arith.index_cast''')
        body=body.replace(f'func.call @chain_geglu(%q,%kv,%go,%fill) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32) -> ()',
          f'func.call @chain_elementwise(%q,%kv,%go,%fill,%command,%position,%valid,%local,%state{i}) : (memref<{p.QWORDS}xi32>,memref<{p.KVWORDS}xi32>,memref<{p.OUT}xi32>,i32,i32,i32,i32,i32,memref<{p.STATE}xi32>) -> ()')
        text=text[:start]+body+text[end:]
    return text


def runtime(pl,local,valid):
    seq=Sequence(pl['words']); off=pl['offsets']; size=16*p.OUT
    seq.control(2,16)
    for group in range(12):
        for col in range(4):
            seq.dma(col,'q',off['input'],[1,4,16,4*p.QWORDS],[0,64*p.QWORDS,4*p.QWORDS,1])
            seq.dma(col,'out',off['projection']+(col*12+group)*size,[1,1,size//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',off['input']+256*p.QWORDS+(col*12+group)*16*p.KVWORDS,[4,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
        seq.wait()
    for mode in (4,5,6,7):
        for which in range(4):
            for mg in range(4):
                seq.control(mode,position=mg*128,valid=valid,local=int(local))
                for col in range(4):
                    head=col*4+which
                    feature=head+(16 if mode==6 else 32 if mode==7 else 0)
                    proj=off['projection']+((feature%4)*12+feature//4)*size+mg*4*p.OUT
                    if mode==7:
                        qoffset=off['keys']+(head*16+mg*4)*p.OUT; voffset=proj
                    else:
                        qoffset=proj; voffset=off['coeff']+mg*4*p.KVWORDS
                    seq.dma(col,'q',qoffset,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                    if mode<6:
                        target=off['queries']+(head*32+mg*8+(mode-4))*p.OUT
                        seq.dma(col,'out',target,[1,1,4,p.OUT],[0,0,2*p.OUT,1],True)
                    else:
                        target=off['keys' if mode==6 else 'kv']+(head*16+mg*4)*p.OUT
                        seq.dma(col,'out',target,[1,1,p.OUT//4,16],[0,0,16,1],True)
                    seq.dma(col,'kv',voffset,[1,1,4,p.KVWORDS],[0,0,p.OUT if mode==7 else p.KVWORDS,1])
                seq.wait()
    seq.control(0)
    for col in range(4):
        seq.dma(col,'q',off['queries']+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['attention']+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        for head in range(4):
            seq.dma(col,'kv',off['kv']+(col*4+head)*16*p.OUT+16,[8,1,16,p.KVWORDS],[0,0,p.OUT,1],bid=head+3)
    seq.wait()
    return seq.finish()


def build(out,model_path,all_masks=False):
    import onnx
    out.mkdir(parents=True,exist_ok=True); model=onnx.load(str(model_path),load_external_data=False)
    coefficients(model,out); pl=plan()
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}',f'-I{p.attention.UPSTREAM/"aie_runtime_lib/AIE2"}',
      '-Wno-attributes','-Wno-parentheses','-DATTENTION_SHARED_MATMUL=1']
    clang=p.attention.PEANO/'bin/clang++.exe'; objects=[]
    for name,src in [('rotary',HERE/'rotary.cc'),('gemm',HERE/'gemm.cc'),('dispatch',HERE/'dispatch.cc'),
      ('attention',HERE.parent/'laya_attention/kernel_shared.cc'),('norm',HERE.parent/'laya_norm/kernel.cc'),
      ('lut',p.attention.UPSTREAM/'aie_runtime_lib/AIE2/lut_based_ops.cpp')]:
        obj=out/f'{name}.o'; objects.append(obj); p.attention.command([clang,*flags,'-c',src,'-o',obj],out)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved helper')
    import shutil
    requested=[(mode,valid) for mode in ('global','local') for valid in (512,193,1,0)] if all_masks else [('global',512)]
    layouts={}
    for mode,valid in requested:
        name=f'{mode}_{valid}'; folder=out/name; folder.mkdir(exist_ok=True); shutil.copyfile(out/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(prefix()+runtime(pl,mode=='local',valid),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        before=p.STACK
        try:
            p.STACK=mlp.STACK; audits=p.audit(folder)
        finally: p.STACK=before
        cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if len(cdo)!=3: raise ValueError('missing initialization payloads')
        layouts[name]=dict(mode=mode,valid=valid,audits=audits,cdo=cdo,xclbin=p.attention.sha(folder/'final.xclbin'),instructions=p.attention.sha(folder/'insts.bin'))
        if layouts[name]['audits']!=layouts['global_512']['audits'] or cdo!=layouts['global_512']['cdo']: raise ValueError('persistent core/init image differs')
        print('ROTARY_LAYOUT '+json.dumps(dict(name=name,code_bytes=sorted(set(a['code_bytes'] for a in audits)))),flush=True)
    meta=dict(sources=sources(),plan=pl,layouts=layouts,model_sha256=p.attention.sha(model_path),
      coefficients_sha256=p.attention.sha(out/'coefficients.npz'))
    p.attention.dump(out/'build.json',meta); print('ROTARY_BUILD '+json.dumps(meta),flush=True)


def packet_matrix(raw,n):
    return p.gemm_output(raw[:,16:2064].copy().view(np.float32),n)


def unpack_query(raw):
    tiles=raw[:,16:528].copy().view(np.uint16).reshape(16,32,4,8,4,8)
    bits=tiles.transpose(0,1,2,4,3,5).reshape(16,512,64)
    return (bits.astype(np.uint32)<<16).view(np.float32)


def unpack_kv(raw):
    bits=raw[:,16:2064].copy().view(np.uint16).reshape(16,16,4096)
    kt=bits[:,:,:2048].reshape(16,16,8,8,8,4).transpose(0,1,2,4,3,5).reshape(16,16,64,32)
    key=kt.transpose(0,1,3,2).reshape(16,512,64)
    value=bits[:,:,2048:].reshape(16,16,4,16,8,4).transpose(0,1,2,4,3,5).reshape(16,512,64)
    return tuple((v.astype(np.uint32)<<16).view(np.float32) for v in (key,value))


def reference_rotary(projected,table):
    heads=projected.reshape(512,3,16,64).transpose(1,2,0,3)
    coeff=table[:,:2048].copy().view(np.float32)
    c=coeff[:,:1024].reshape(512,32).astype(np.float64); s=coeff[:,1024:].reshape(512,32).astype(np.float64)
    result=[]
    for x in heads[:2]:
        lo=x[:,:,:32].astype(np.float64); hi=x[:,:,32:].astype(np.float64)
        result.append(p.attention.rounded(np.concatenate((lo*c-hi*s,hi*c+lo*s),axis=2)))
    return (*result,p.attention.rounded(heads[2]))


def run(out,masks,smoke=False,model=None):
    meta=json.loads((out/'build.json').read_text()); pl=meta['plan']; off=pl['offsets']; sizes=pl['sizes']
    if meta['sources']!=sources(): raise ValueError('stale source build')
    for name,layout in meta['layouts'].items():
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(out/name/file)!=layout[key]: raise ValueError('changed artifact')
    if p.attention.sha(out/'coefficients.npz')!=meta['coefficients_sha256']: raise ValueError('changed coefficients')
    tables=np.load(out/'coefficients.npz'); masks_data=np.load(masks)
    model_weight=None; weight_provenance=None
    if model:
        import onnx
        if p.attention.sha(model)!=meta['model_sha256']: raise ValueError('model graph changed')
        graph=onnx.load(str(model),load_external_data=False); initial={t.name:t for t in graph.graph.initializer}
        tensor=next(initial[node.input[1]] for node in graph.graph.node if node.op_type=='MatMul' and node.input[1] in initial and tuple(initial[node.input[1]].dims)==(1024,3072))
        original=onnx.numpy_helper.to_array(tensor,base_dir=str(model.parent))
        model_weight=p.attention.rounded(original)
        weight_provenance=dict(name=tensor.name,sha256=hashlib.sha256(original.tobytes()).hexdigest())
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'global_512/final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    S=xrt.xclBOSyncDirection; instructions={}
    for name in meta['layouts']:
        words=np.frombuffer((out/name/'insts.bin').read_bytes(),np.uint32)
        ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(words.tobytes(),0); ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
        instructions[name]=(ib,words.nbytes)
    ib=None
    ga,go=kernel.group_id(3),kernel.group_id(4)
    if ga&0xffff!=go&0xffff: raise ValueError('arena bank mismatch')
    arena=xrt.bo(dev,pl['words']*4,xrt.bo.host_only,ga); av=np.frombuffer(arena.map(),np.uint32); hrun=None
    checks=0
    print('ROTARY_RUN '+json.dumps(dict(build=meta,masks_sha256=p.attention.sha(masks),
      kernel_calls_per_chain=1,interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
      model_weight=weight_provenance,scope='projection through attention, synthetic activations; not full Laya')) ,flush=True)
    try:
        for seed in (((11,) if smoke else (11,29,47))+((71,) if model_weight is not None else ())):
            kinds=('actual-first-qkv',) if seed==71 else (('random',) if smoke else ('random','zero','basis'))
            cases=[(kind,name) for kind in kinds for name in meta['layouts']]
            for kind,name in cases:
                layout=meta['layouts'][name]; table=tables[layout['mode']]; mask=masks_data[name]
                local=int(layout['mode']=='local'); valid=layout['valid']; ib,nb=instructions[name]
                a,b=p.gemm_case(seed,kind,1024,3072)
                # Keep finite-mask oracle assumptions and attention tolerance
                # consistent with the existing synthetic attention qualification.
                b*=np.float32(.0625)
                if seed==71: b=model_weight
                projected=a.astype(np.float64)@b.astype(np.float64)
                qr,kr,vr=reference_rotary(projected,table)
                qr[:,np.all(mask==2,axis=1)]=0
                ref=p.attention.oracle(qr,kr,vr,mask)
                previous=None
                for fill in (0x7fc12345,0x7fc54321):
                    payload=np.full(pl['words'],fill,np.uint32)
                    payload[off['input']:off['input']+sizes['input']]=p.gemm_packet(a,b,fill)
                    payload[off['coeff']:off['coeff']+sizes['coeff']]=table.ravel()
                    av[:]=payload; arena.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
                    start=time.perf_counter_ns(); hrun=kernel(3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('chain dispatch failed; no retry')
                    ms=(time.perf_counter_ns()-start)/1e6
                    arena.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE); observed=av.copy(); regions={}
                    written=np.zeros(pl['words'],bool)
                    for region_name in ('projection','queries','keys','kv','attention'):
                        regions[region_name]=observed[off[region_name]:off[region_name]+sizes[region_name]].reshape(-1,p.OUT)
                        written[off[region_name]:off[region_name]+sizes[region_name]]=True
                    if not np.array_equal(observed[~written],payload[~written]): raise ValueError('input/weights/coefficients/gaps changed')
                    expected_headers=np.full((512,16),mlp.POISON,np.uint32)
                    expected_headers[:,0]=np.tile(np.arange(32)*16,16); expected_headers[:,1]=valid
                    expected_headers[:,2]=mlp.POISON; expected_headers[:,3]=local; expected_headers[:,4]=0
                    if not np.array_equal(regions['queries'][:,:16],expected_headers): raise ValueError('query metadata')
                    for region_name,used in [('projection',2048),('queries',512),('keys',1024),('kv',2064),('attention',1024)]:
                        rows=regions[region_name]
                        if region_name!='queries' and not np.all(rows[:,:16]==mlp.POISON): raise ValueError(region_name+' prefix guard')
                        if not np.all(rows[:,16+used:]==mlp.POISON): raise ValueError(region_name+' tail guard')
                        if np.any(rows[:,16:16+used]==mlp.POISON) or np.any(rows[:,16:16+used]==fill): raise ValueError(region_name+' unwritten payload')
                    ramp=regions['kv'][:,2064:].copy().view(np.uint16)
                    if not np.all(ramp==np.arange(32,dtype=np.uint16)): raise ValueError('key ramp')
                    gotp=packet_matrix(regions['projection'],3072)
                    gotq=unpack_query(regions['queries']); gotk,gotv=unpack_kv(regions['kv'])
                    got=regions['attention'][:,16:1040].copy().view(np.float32).reshape(16,512,64)
                    # Diagnose the attention core against its actual device-
                    # prepared operands only after the complete chain finishes.
                    attention_ref=p.attention.oracle(gotq,gotk,gotv,mask)
                    verdicts=dict(projection=p.attention.numeric(gotp,projected),q=p.attention.numeric(gotq,qr),
                      k=p.attention.numeric(gotk,kr),v=p.attention.numeric(gotv,vr),
                      attention=p.attention.numeric(got,attention_ref),chain=p.attention.numeric(got,ref))
                    if previous is not None and not np.array_equal(got,previous): raise ValueError('poison dependence')
                    print('ROTARY_CHECK '+json.dumps(dict(seed=seed,kind=kind,layout=name,fill=fill,dispatch_ms_correctness_only=ms,verdicts=verdicts)),flush=True)
                    if not all(v['ok'] for v in verdicts.values()): raise ValueError('composed attention numerical gate')
                    previous=got; checks+=1
        print('ROTARY_RESULT '+json.dumps(dict(checks=checks,contexts_created=1,kernel_calls_per_chain=1,
          interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
          layouts=list(meta['layouts']),scope='projection/RoPE/attention only; full model unverified')),flush=True)
    finally:
        hrun=None; instructions.clear(); ib=None; del av,arena,kernel,ctx,xb,dev
        gc.collect(); print('ROTARY_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path); ap.add_argument('--masks',type=Path); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--all-masks',action='store_true'); args=ap.parse_args()
    if args.command=='build': build(args.out.resolve(),args.model,args.all_masks)
    else: run(args.out.resolve(),args.masks,args.smoke,args.model)
