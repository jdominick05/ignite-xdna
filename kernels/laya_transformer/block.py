#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two serial NPU programs, one arena: attention and MLP with residual carry.

The original input, weights and instructions are uploaded before the first
dispatch. No host activation access occurs between the two programs.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import rotary as r
p=r.p
HERE=Path(__file__).resolve().parent


def sources():
    return dict(r.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in (HERE/'block.py',HERE/'block.cc')})


def plan():
    pl=r.plan(); cursor=pl['words']
    extra=dict(mlp_wi_weight=84*16*p.KVWORDS,mlp_wi=84*16*p.OUT,
      mlp_geglu=41*16*p.OUT,mlp_wo_weight=16*41*p.KVWORDS,
      mlp_affine=p.KVWORDS,mlp_norm=512*p.OUT)
    for name,size in extra.items():
        pl['sizes'][name]=size; pl['offsets'][name]=cursor; cursor+=size+16
    pl['words']=cursor
    return pl


def mlp_prefix():
    text=r.mlp.core_prefix()
    text=text.replace('    func.func private @gemm_finish(',
      f'    func.func private @chain_finish(memref<{p.STATE}xi32>,memref<{p.OUT}xi32>,i32)\n    func.func private @gemm_finish(')
    text=text.replace('%legacy = arith.cmpi eq, %override,%zero : i32','%legacy = arith.constant false')
    text=text.replace('      %three = arith.constant 3 : i32','      %nine = arith.constant 9 : i32\n      %three = arith.constant 3 : i32')
    text=text.replace('%command = arith.select %legacy,%inband,%override : i32',
      '%is_tiled = arith.cmpi eq, %override,%nine : i32\n        %tiled = arith.extui %is_tiled : i1 to i32\n        %command = arith.select %is_tiled,%two,%override : i32')
    for i in range(16):
        text=text.replace(f'func.call @gemm_finish(%state{i},%o) : (memref<{p.STATE}xi32>,memref<{p.OUT}xi32>) -> ()',
          f'func.call @chain_finish(%state{i},%o,%tiled) : (memref<{p.STATE}xi32>,memref<{p.OUT}xi32>,i32) -> ()')
    return text


class MlpSequence(r.Sequence):
    def control(self,mode,kblocks=1,position=0,valid=512,local=0):
        if mode not in (1,2,3,9): raise ValueError('operation unavailable in the MLP image')
        for index,value in enumerate((mode,kblocks,r.mlp.POISON,None)):
            if index in self.current and self.current[index]==value: continue
            self.current[index]=value
            for i in range(16):
                name=self.constant(i%4 if value is None else value)
                self.lines.append(f'      "aiex.npu.rtp_write"({name}) {{buffer = @chain_control_{i}, index = {index} : ui32}} : (i32) -> ()')
    def wait_columns(self,columns):
        for col in columns: self.lines.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')


def mlp_runtime(pl):
    off=pl['offsets']; seq=MlpSequence(pl['words']); size=16*p.OUT
    seq.control(2,16)
    for group in range(21):
        for mg in range(4):
            for col in range(4):
                seq.dma(col,'q',off['packed_norm']+mg*4*p.OUT,[1,16,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                seq.dma(col,'out',off['mlp_wi']+((col*21+group)*4+mg)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off['mlp_wi_weight']+(col*21+group)*16*p.KVWORDS,[1,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    seq.control(3)
    for first in range(0,41,4):
        columns=[]
        for feature in range(first,min(first+4,41)):
            col=feature%4; columns.append(col); value=feature+41
            qoff=off['mlp_wi']+((feature%4)*21+feature//4)*size
            voff=off['mlp_wi']+((value%4)*21+value//4)*size
            seq.dma(col,'q',qoff,[1,4,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            seq.dma(col,'out',off['mlp_geglu']+feature*size,[1,1,size//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',voff,[1,4,4,p.KVWORDS],[0,4*p.OUT,p.OUT,1])
        seq.wait_columns(columns)
    seq.control(9,41)
    for group in range(4):
        for mg in range(4):
            for col in range(4):
                seq.dma(col,'q',off['mlp_geglu']+mg*4*p.OUT,[1,41,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                # Reuse the attention norm output as this norm input. Its
                # residual at 1040 is untouched; old Y is no longer needed.
                seq.dma(col,'out',off['norm']+mg*128*p.OUT+(group*4+col)*65,[1,4,32,65],[0,32*p.OUT,p.OUT,1],True)
                seq.dma(col,'kv',off['mlp_wo_weight']+(col*4+group)*41*p.KVWORDS,[1,1,41,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    seq.control(1)
    for col in range(4):
        seq.dma(col,'q',off['norm']+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['mlp_norm']+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['mlp_affine'],[32,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait()
    return seq.finish()


def build(out,model,all_masks=False):
    import onnx
    out.mkdir(parents=True,exist_ok=True); pl=plan()
    r.coefficients(onnx.load(str(model),load_external_data=False),out)
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}',f'-I{p.attention.UPSTREAM/"aie_runtime_lib/AIE2"}',
      '-Wno-attributes','-Wno-parentheses','-DATTENTION_SHARED_MATMUL=1','-DTRANSFORMER_NORM_TILED=1']
    clang=p.attention.PEANO/'bin/clang++.exe'; families={}
    for family in ('attention','mlp'):
        folder=out/family; folder.mkdir(exist_ok=True); objects=[]
        choices=[('rotary',HERE/'rotary.cc')] if family=='attention' else [('block',HERE/'block.cc'),('mlp',HERE/'mlp.cc')]
        choices += [('gemm',HERE/'gemm.cc'),('dispatch',HERE/'dispatch.cc'),('attention',HERE.parent/'laya_attention/kernel_shared.cc'),
          ('norm',HERE.parent/'laya_norm/kernel.cc'),('lut',p.attention.UPSTREAM/'aie_runtime_lib/AIE2/lut_based_ops.cpp')]
        for name,source in choices:
            obj=folder/f'{name}.o'; objects.append(obj); p.attention.command([clang,*flags,'-c',source,'-o',obj],folder)
        dis=p.attention.command([p.attention.PEANO/'bin/llvm-objdump.exe','-d',folder/'dispatch.o'],folder)
        if not re.search(r'mov\s+crRnd,\s+#0x0\b',dis): raise ValueError('rounding fence removed')
        p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',folder/'kernel.o'],folder)
        if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',folder/'kernel.o'],folder).strip(): raise ValueError('unresolved helper')
        requested=[(mode,valid) for mode in ('global','local') for valid in (512,193,1,0)] if family=='attention' and all_masks else [('global',512)]
        layouts={}
        for mode,valid in requested:
            name=f'{mode}_{valid}'; layout=folder/name; layout.mkdir(exist_ok=True)
            import shutil
            shutil.copyfile(folder/'kernel.o',layout/'kernel.o')
            prefix=r.prefix() if family=='attention' else mlp_prefix()
            runtime=r.runtime(pl,mode=='local',valid) if family=='attention' else mlp_runtime(pl)
            (layout/'harness.mlir').write_text(prefix+runtime,encoding='utf-8')
            p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',layout/'harness.mlir',f'--peano={p.attention.PEANO}',
              '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={layout/"prj"}'],layout)
            before=p.STACK
            try:
                p.STACK=r.mlp.STACK; audits=p.audit(layout)
            finally: p.STACK=before
            cdo={f.name:p.attention.sha(f) for f in (layout/'prj').rglob('*_aie_cdo_*.bin')}
            if len(cdo)!=3: raise ValueError('missing initialization')
            layouts[name]=dict(mode=mode,valid=valid,audits=audits,cdo=cdo,xclbin=p.attention.sha(layout/'final.xclbin'),instructions=p.attention.sha(layout/'insts.bin'))
            if layouts[name]['audits']!=layouts['global_512']['audits'] or cdo!=layouts['global_512']['cdo']: raise ValueError('family image differs across layouts')
            print('BLOCK_LAYOUT '+json.dumps(dict(family=family,layout=name,code_bytes=sorted(set(a['code_bytes'] for a in audits)))),flush=True)
        families[family]=layouts
    meta=dict(sources=sources(),plan=pl,families=families,model_sha256=p.attention.sha(model),
      coefficients_sha256=p.attention.sha(out/'coefficients.npz'),rounding_fence_machine_verified=True)
    p.attention.dump(out/'build.json',meta); print('BLOCK_BUILD '+json.dumps(meta),flush=True)


def unpack_operand(raw,blocks):
    bits=raw[:,16:1040].copy().view(np.uint16).reshape(blocks,4,4,8,8,4,8)
    tiles=bits.transpose(0,1,2,3,5,4,6).reshape(blocks,4,4,32,64)
    full=tiles.transpose(1,2,3,0,4).reshape(512,blocks*64)
    return (full.astype(np.uint32)<<16).view(np.float32)


def numeric(got,ref,atol,rtol,l2):
    diff=np.abs(got-ref); rel=float(np.sqrt(np.sum(diff*diff))/max(np.sqrt(np.sum(ref*ref)),1e-30))
    return dict(ok=bool(np.isfinite(got).all() and np.all(diff<=atol+rtol*np.abs(ref)) and rel<=l2),max_abs=float(diff.max()),relative_l2=rel)


def gelu(value):
    erf=np.vectorize(math.erf,otypes=[np.float64])
    return value*.5*(1+erf(value/np.sqrt(2)))


def model_parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); initial={t.name:t for t in graph.graph.initializer}
    products=[node for node in graph.graph.node if node.op_type=='MatMul' and node.input[1] in initial]
    start=next(i for i,node in enumerate(products) if tuple(initial[node.input[1]].dims)==(1024,3072))
    tensors=[initial[node.input[1]] for node in products[start:start+4]]
    if [tuple(t.dims) for t in tensors]!=[(1024,3072),(1024,1024),(1024,5248),(2624,1024)]: raise ValueError('encoder product order changed')
    originals=[onnx.numpy_helper.to_array(t,base_dir=str(model.parent)) for t in tensors]
    hashes={t.name:hashlib.sha256(a.tobytes()).hexdigest() for t,a in zip(tensors,originals)}
    affine=[]
    for name in ('m.encoder.layers.0.mlp_norm.weight','m.encoder.layers.1.attn_norm.weight'):
        node=next(node for node in graph.graph.node if node.op_type=='LayerNormalization' and node.input[1]==name)
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5): raise ValueError('norm semantics changed')
        g=onnx.numpy_helper.to_array(initial[name],base_dir=str(model.parent)).copy()
        b=onnx.numpy_helper.to_array(initial[node.input[2]],base_dir=str(model.parent)).copy() if len(node.input)>2 else np.zeros_like(g)
        affine.append((g,b)); hashes[name]=hashlib.sha256(g.tobytes()).hexdigest()
    return [p.attention.rounded(a) for a in originals],affine,hashes


def weight_packet(weight,poison):
    k=weight.shape[0]
    raw=p.gemm_packet(np.zeros((1,k),np.float32),weight,poison)
    return raw[4*(p.padded(k,64)//64)*p.QWORDS:]


def run(out,masks,model,smoke=False,collect_composition_failures=False):
    meta=json.loads((out/'build.json').read_text()); pl=meta['plan']; off=pl['offsets']; sizes=pl['sizes']
    if meta['sources']!=sources() or meta['model_sha256']!=p.attention.sha(model): raise ValueError('stale source or model')
    if meta['coefficients_sha256']!=p.attention.sha(out/'coefficients.npz'): raise ValueError('changed coefficients')
    for family,layouts in meta['families'].items():
        for name,layout in layouts.items():
            for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
                if p.attention.sha(out/family/name/file)!=layout[key]: raise ValueError('changed artifact')
    real_weights,real_affine,hashes=model_parameters(model)
    tables=np.load(out/'coefficients.npz'); masks_data=np.load(masks)
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); states={}; hrun=None; arena=None; av=None; failed=False
    S=xrt.xclBOSyncDirection
    try:
        for family,layouts in meta['families'].items():
            xb=xrt.xclbin(str(out/family/'global_512/final.xclbin')); dev.register_xclbin(xb)
            ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name()); instructions={}
            for name in layouts:
                words=np.frombuffer((out/family/name/'insts.bin').read_bytes(),np.uint32)
                ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(words.tobytes(),0); ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
                instructions[name]=(ib,words.nbytes)
            states[family]=dict(xb=xb,ctx=ctx,kernel=kernel,instructions=instructions,groups=[kernel.group_id(3),kernel.group_id(4)])
        del xb,ctx,kernel,ib,instructions
        groups=[group for state in states.values() for group in state['groups']]
        if len(set(group&0xffff for group in groups))!=1: raise ValueError('cross-image bank connectivity differs')
        arena=xrt.bo(dev,pl['words']*4,xrt.bo.host_only,groups[0]); av=np.frombuffer(arena.map(),np.uint32)
        print('BLOCK_RUN '+json.dumps(dict(build=meta,model_weights=hashes,allocation_group=groups[0],argument_groups=groups,
          contexts_created=2,dispatches_per_block=2,interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
          gates=dict(linear=[.0002,.0002,.00001],geglu=[.000002,.0078125,.004],
            attention=[p.attention.ATOL,p.attention.RTOL,p.attention.L2],norm=[p.norm.ATOL,p.norm.RTOL,p.norm.L2],
            retained_strict_mlp=[.0002,.002,.002],residual='FP32 RNE bits exact'),
          scope='encoder block including next attention norm, synthetic input; not full Laya')),flush=True)
        checks=0; strict_failures=[]; composition_failures=[]
        for seed in ((11,71) if smoke else (11,29,47,71)):
            kinds=('actual-first-block',) if seed==71 else (('random',) if smoke else ('random','zero','basis'))
            for kind in kinds:
                a,b=p.gemm_case(seed,kind,1024,3072); b*=np.float32(.0625)
                if kind in ('random','actual-first-block'): a=np.add(a,np.random.default_rng(seed+207).normal(0,.000173,a.shape).astype(np.float32),dtype=np.float32)
                _,out_weight=p.gemm_case(seed+101,'random',1024,1024); out_weight*=np.float32(.0625)
                _,wi,wo=r.mlp.fixture([512,1024,2624,1024],seed,'random')
                _,_,g0,b0=p.norm.fixture(seed,'random'); _,_,g1,b1=p.norm.fixture(seed+31,'random')
                if seed==71:
                    b,out_weight,wi,wo=real_weights; (g0,b0),(g1,b1)=real_affine
                for name,layout in meta['families']['attention'].items():
                    table=tables[layout['mode']]; mask=masks_data[name]; rounded=p.attention.rounded(a)
                    projected=rounded.astype(np.float64)@b.astype(np.float64)
                    q,k,v=r.reference_rotary(projected,table); q[:,np.all(mask==2,axis=1)]=0
                    attref=p.attention.oracle(q,k,v,mask)
                    aref=p.attention.rounded(attref.transpose(1,0,2).reshape(512,1024)).astype(np.float64)@out_weight.astype(np.float64)
                    zref,yref=p.norm.oracle(aref.astype(np.float32),a,g0,b0)
                    wiref=p.attention.rounded(yref).astype(np.float64)@wi.astype(np.float64)
                    gref=gelu(wiref[:,:2624])*wiref[:,2624:]
                    woref=p.attention.rounded(gref).astype(np.float64)@wo.astype(np.float64)
                    finalzref,finalref=p.norm.oracle(woref.astype(np.float32),zref,g1,b1)
                    previous=None
                    for fill in (0x7fc12345,0x7fc54321):
                        data=np.full(pl['words'],fill,np.uint32)
                        data[off['input']:off['input']+sizes['input']]=p.gemm_packet(rounded,b,fill)
                        data[off['coeff']:off['coeff']+sizes['coeff']]=table.ravel()
                        for field,weight in [('projection_weight',out_weight),('mlp_wi_weight',wi),('mlp_wo_weight',wo)]:
                            packet=weight_packet(weight,fill)
                            if len(packet)!=sizes[field]: raise ValueError('weight packet size')
                            data[off[field]:off[field]+sizes[field]]=packet
                        norm_input=data[off['norm_input']:off['norm_input']+sizes['norm_input']].reshape(512,p.OUT)
                        norm_input[:,1040:2064]=a.copy().view(np.uint32)
                        for field,gamma,beta in [('affine',g0,b0),('mlp_affine',g1,b1)]:
                            affine=data[off[field]:off[field]+sizes[field]]; affine.fill(0)
                            affine[:1024]=gamma.copy().view(np.uint32); affine[1024:2048]=beta.copy().view(np.uint32)
                        av[:]=data; arena.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
                        start=time.perf_counter_ns()
                        # No host tensor operations or cache syncs between these
                        # waits and the following dispatch. Both see the same BO.
                        for family,layout_name in [('attention',name),('mlp','global_512')]:
                            state=states[family]; ib,nb=state['instructions'][layout_name]
                            hrun=state['kernel'](3,ib,nb,arena,arena)
                            if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                                failed=True; raise RuntimeError('dispatch failed; no retry')
                            hrun=None
                        ms=(time.perf_counter_ns()-start)/1e6
                        arena.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE); observed=av.copy(); regions={}; written=np.zeros(pl['words'],bool)
                        for field in ('projection','queries','keys','kv','attention','packed_attention','norm','packed_norm','mlp_wi','mlp_geglu','mlp_norm'):
                            regions[field]=observed[off[field]:off[field]+sizes[field]].reshape(-1,p.OUT)
                            written[off[field]:off[field]+sizes[field]]=True
                        written[off['norm_input']:off['norm_input']+sizes['norm_input']].reshape(512,p.OUT)[:,:1040]=True
                        if not np.array_equal(observed[~written],data[~written]): raise ValueError('input, weight, affine, residual or gap modified')
                        header=np.full((512,16),r.mlp.POISON,np.uint32)
                        header[:,0]=np.tile(np.arange(32)*16,16); header[:,1]=layout['valid']
                        header[:,2]=r.mlp.POISON; header[:,3]=int(layout['mode']=='local'); header[:,4]=0
                        if not np.array_equal(regions['queries'][:,:16],header) or not np.all(regions['queries'][:,528:]==r.mlp.POISON): raise ValueError('query packet metadata or guards')
                        if not np.all(regions['kv'][:,2064:].copy().view(np.uint16)==np.arange(32,dtype=np.uint16)): raise ValueError('key ramp')
                        for field,used in [('projection',2048),('keys',1024),('kv',2064),('attention',1024),('packed_attention',1024),('packed_norm',1024),('mlp_wi',2048),('mlp_geglu',1024)]:
                            raw=regions[field]
                            if not np.all(raw[:,:16]==r.mlp.POISON) or not np.all(raw[:,16+used:]==r.mlp.POISON): raise ValueError(field+' guards')
                            if np.any(raw[:,16:16+used]==fill) or np.any(raw[:,16:16+used]==r.mlp.POISON): raise ValueError(field+' unwritten')
                        for field in ('norm_input','norm'):
                            raw=observed[off[field]:off[field]+sizes[field]].reshape(512,p.OUT)
                            tail=fill if field=='norm_input' else r.mlp.POISON
                            if not np.all(raw[:,:1040].reshape(512,16,65)[:,:,64]==r.mlp.POISON) or not np.all(raw[:,2064:]==tail): raise ValueError(field+' scatter guards')
                        final=regions['mlp_norm']
                        if not np.all(final[:,1024:1040]==r.mlp.POISON) or not np.all(final[:,2064:]==r.mlp.POISON): raise ValueError('final norm guards')
                        actual_input=unpack_operand(regions['packed_norm'],16)
                        actual_wi=r.packet_matrix(regions['mlp_wi'],5248)
                        actual_g=unpack_operand(regions['mlp_geglu'],41)
                        actual_wo=regions['norm'][:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
                        residual=regions['norm'][:,1040:2064].copy().view(np.float32)
                        final_y=final[:,:1024].copy().view(np.float32); final_z=final[:,1040:2064].copy().view(np.float32)
                        attn_out=observed[off['norm_input']:off['norm_input']+sizes['norm_input']].reshape(512,p.OUT)[:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
                        expected_residual,expected_y=p.norm.oracle(attn_out,a,g0,b0)
                        direct_wi=actual_input.astype(np.float64)@wi.astype(np.float64)
                        direct_g=gelu(direct_wi[:,:2624])*direct_wi[:,2624:]
                        strict_wo=p.attention.rounded(direct_g).astype(np.float64)@wo.astype(np.float64)
                        direct_wo=actual_g.astype(np.float64)@wo.astype(np.float64)
                        direct_z,direct_y=p.norm.oracle(actual_wo,residual,g1,b1)
                        verdicts=dict(attention_residual=dict(ok=bool(np.array_equal(residual.view(np.uint32),expected_residual.view(np.uint32)))),
                          packed_input=p.attention.numeric(actual_input,p.attention.rounded(expected_y)),
                          wi=numeric(actual_wi,direct_wi,.0002,.0002,.00001),geglu=numeric(actual_g,direct_g,.000002,.0078125,.004),
                          wo=numeric(actual_wo,direct_wo,.0002,.0002,.00001),norm=r.norm_verdict(final_z,final_y,direct_z,direct_y),
                          block=p.attention.numeric(final_y,finalref),block_residual=p.attention.numeric(final_z,finalzref))
                        strict=numeric(actual_wo,strict_wo,.0002,.002,.002)
                        if not strict['ok']: strict_failures.append(dict(seed=seed,kind=kind,layout=name,fill=fill,**strict))
                        if previous is not None and not np.array_equal(final,previous): raise ValueError('poison dependence')
                        print('BLOCK_CHECK '+json.dumps(dict(seed=seed,kind=kind,layout=name,fill=fill,dispatch_ms_correctness_only=ms,verdicts=verdicts,retained_mlp_strict=strict)),flush=True)
                        if not all(v['ok'] for key,v in verdicts.items() if key not in ('block','block_residual')): raise ValueError('block operator gate')
                        if not verdicts['block']['ok'] or not verdicts['block_residual']['ok']:
                            composition_failures.append(dict(seed=seed,kind=kind,layout=name,fill=fill,block=verdicts['block'],residual=verdicts['block_residual']))
                            if not collect_composition_failures: raise ValueError('complete block composition gate')
                        previous=final.copy(); checks+=1
        print('BLOCK_RESULT '+json.dumps(dict(checks=checks,contexts_created=2,dispatches_per_block=2,interstage_host_reads=0,interstage_host_writes=0,
          interstage_host_syncs=0,strict_mlp_failures=strict_failures,strict_mlp_pass=not strict_failures,composition_failures=composition_failures,
          scope='encoder block on synthetic input; whole-model accuracy and speed unverified')),flush=True)
        if strict_failures or composition_failures: raise ValueError('retained pointwise gates failed; tolerances unchanged')
    finally:
        if failed:
            # An indeterminate run may still reference its resources. Keep
            # them alive until process teardown; never dispatch a retry.
            globals()['_quarantined']=(dev,states,arena,av,hrun)
        else:
            # Release children explicitly before parents. Clearing a dict that
            # holds a context before its kernel can crash this PyXRT binding.
            hrun=None; state=None; ib=None; av=None; arena=None
            print('BLOCK_RELEASE instruction_bos',flush=True)
            for family in reversed(tuple(states)): states[family]['instructions'].clear()
            print('BLOCK_RELEASE kernels',flush=True)
            for family in reversed(tuple(states)): states[family]['kernel']=None
            print('BLOCK_RELEASE contexts',flush=True)
            for family in reversed(tuple(states)): states[family]['ctx']=None
            print('BLOCK_RELEASE xclbins',flush=True)
            for family in reversed(tuple(states)): states[family]['xb']=None
            states.clear(); dev=None; gc.collect()
            print('BLOCK_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True); ap.add_argument('--masks',type=Path); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--all-masks',action='store_true'); ap.add_argument('--collect-composition-failures',action='store_true'); args=ap.parse_args()
    if args.command=='build': build(args.out.resolve(),args.model,args.all_masks)
    else: run(args.out.resolve(),args.masks,args.model,args.smoke,args.collect_composition_failures)
