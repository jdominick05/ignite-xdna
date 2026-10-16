# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Execute a bounded encoder prefix without host activation access between stages."""
import gc
import json
from pathlib import Path
import sys

import numpy as np
import chain as c
b=c.b; p=b.p


def references(initial,parameters,spec,tables,masks,valid,initial_residual=None):
    y=initial; residual=initial if initial_residual is None else initial_residual; result=[]
    for entry,values in zip(spec,parameters):
        qkv,out,wi,wo=values['weights']; (g0,b0),(g1,b1)=values['affine']
        projected=p.attention.rounded(y).astype(np.float64)@qkv.astype(np.float64)
        mask=masks[f"{entry['mode']}_{valid}"]
        q,k,v=b.r.reference_rotary(projected,tables[entry['mode']]); q[:,np.all(mask==2,axis=1)]=0
        attention=p.attention.oracle(q,k,v,mask)
        projection=p.attention.rounded(attention.transpose(1,0,2).reshape(512,1024)).astype(np.float64)@out.astype(np.float64)
        residual,y=p.norm.oracle(projection.astype(np.float32),residual,g0,b0)
        hidden=p.attention.rounded(y).astype(np.float64)@wi.astype(np.float64)
        activation=b.gelu(hidden[:,:2624])*hidden[:,2624:]
        projection=p.attention.rounded(activation).astype(np.float64)@wo.astype(np.float64)
        residual,y=p.norm.oracle(projection.astype(np.float32),residual,g1,b1)
        result.append((residual,y)); print('CHAIN_REFERENCE_LAYER '+str(entry['layer']),flush=True)
    return result


def initialize(data,pl,initial,parameters,tables,spec,fill):
    data.fill(fill); off=pl['offsets']; sizes=pl['sizes']
    data[off['input']:off['input']+sizes['input']]=p.gemm_packet(p.attention.rounded(initial),parameters[0]['weights'][0],fill)
    norm=data[off['norm_input']:off['norm_input']+sizes['norm_input']].reshape(512,p.OUT)
    norm[:,1040:2064]=initial.copy().view(np.uint32)
    initialize_constants(data,pl,parameters,tables,fill)


def initialize_constants(data,pl,parameters,tables,fill):
    for index,values in enumerate(parameters):
        bank=pl['banks'][index]
        for key,weight in zip(('qkv','projection','wi','wo'),values['weights']):
            start,size=bank[key]; packet=b.weight_packet(weight,fill)
            if packet.size!=size: raise ValueError('constant bank packet size')
            data[start:start+size]=packet
        for key,(gamma,beta) in zip(('affine','next_affine'),values['affine']):
            start,size=bank[key]; data[start:start+size]=0
            data[start:start+1024]=gamma.copy().view(np.uint32); data[start+1024:start+2048]=beta.copy().view(np.uint32)
    for mode,(start,size) in pl['coefficients'].items(): data[start:start+size]=tables[mode].ravel()


def run(out,model,masks,valid,seed,collect):
    out=out.resolve(); model=model.resolve(); meta=json.loads((out/'build.json').read_text())
    if meta['sources']!=c.sources() or meta['model_sha256']!=p.attention.sha(model): raise ValueError('stale chain build/model')
    if meta['coefficients_sha256']!=p.attention.sha(out/'coefficients.npz'): raise ValueError('changed coefficient table')
    for name,artifact in meta['artifacts'].items():
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(out/name/file)!=artifact[key]: raise ValueError('chain artifact changed')
    for stage in meta['stages']:
        if p.attention.sha(out/stage['instructions'])!=stage['sha256']: raise ValueError('changed stage instructions')
    graph,spec=c.l.model_spec(model,len(meta['spec']))
    if spec!=meta['spec']: raise ValueError('changed model layer structure')
    parameters,weight_hashes=c.l.constants(model,graph,spec)
    # numpy_helper materializes external FP32 raw_data in this graph. The
    # independent rounded constants own their storage; drop the original
    # graph before allocating the device arena and verification snapshots.
    graph=None; gc.collect()
    tables=np.load(out/'coefficients.npz'); mask_values=np.load(masks)
    if any(f"{s['mode']}_{valid}" not in mask_values for s in spec): raise ValueError('missing independent mask reference')
    initial,_=p.gemm_case(seed,'random',1024,3072)
    initial=np.add(initial,np.random.default_rng(seed+207).normal(0,.000173,initial.shape).astype(np.float32),dtype=np.float32)
    refs=references(initial,parameters,spec,tables,mask_values,valid)
    pl=meta['plan']; off=pl['offsets']; sizes=pl['sizes']; count=len(spec)
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); states={}; arena=None; av=None; hrun=None; failed=False
    sync=xrt.xclBOSyncDirection; failures=[]; strict_failures=[]; snapshots={}; checks=0
    try:
        for family,name in [('attention','first'),('mlp','mlp')]:
            xb=xrt.xclbin(str(out/name/'final.xclbin')); dev.register_xclbin(xb)
            ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            states[family]=dict(xb=xb,ctx=ctx,kernel=kernel,instructions={},groups=[kernel.group_id(3),kernel.group_id(4)])
        groups=[g for state in states.values() for g in state['groups']]
        if len({g&0xffff for g in groups})!=1: raise ValueError('incompatible activation arena banks')
        prepared=[]
        for stage in meta['stages']:
            family=stage['family']; data=(out/stage['instructions']).read_bytes(); changes=[]
            if family=='attention': data,changes=c.txn.patch(data,meta['artifacts']['first']['audits'],valid,spec[stage['layer']]['mode']=='local')
            words=np.frombuffer(data,np.uint32); kernel=states[family]['kernel']
            ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
            key=(stage['layer'],family); states[family]['instructions'][key]=(ib,words.nbytes)
            import hashlib
            prepared.append(dict(layer=stage['layer'],family=family,sha256=hashlib.sha256(data).hexdigest(),control_changes=changes))
        del xb,ctx,kernel,ib
        arena=xrt.bo(dev,pl['words']*4,xrt.bo.host_only,groups[0]); av=np.frombuffer(arena.map(),np.uint32)
        print('CHAIN_RUN '+json.dumps(dict(build=meta,runner_sha256=p.attention.sha(Path(__file__)),weights=weight_hashes,
          seed=seed,valid=valid,prepared=prepared,allocation_group=groups[0],argument_groups=groups,contexts_created=2,
          interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
          scope='supplied synthetic normalized encoder input/residual; no embedding or decision head; correctness only')),flush=True)
        prefixes=sorted({1,2,count})
        for prefix in prefixes:
            previous=None
            for fill in (0x7fc12345,0x7fc54321):
                data=np.empty(pl['words'],np.uint32); initialize(data,pl,initial,parameters,tables,spec,fill)
                av[:]=data; arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                # Every constant and instruction is resident before this loop.
                # No host tensor calculation or activation transfer occurs here.
                for stage in meta['stages'][:prefix*2]:
                    family=stage['family']; key=(stage['layer'],family)
                    ib,nb=states[family]['instructions'][key]
                    failed=True; hrun=states[family]['kernel'](3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('chain dispatch failed; no retry')
                    failed=False; hrun=None
                arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE); observed=av.copy()
                final=observed[off['mlp_norm']:off['mlp_norm']+sizes['mlp_norm']].reshape(512,p.OUT)
                if not np.all(final[:,1024:1040]==b.r.mlp.POISON) or not np.all(final[:,2064:]==b.r.mlp.POISON): raise ValueError('final norm guards')
                y=final[:,:1024].copy().view(np.float32); residual=final[:,1040:2064].copy().view(np.float32)
                if previous is not None and not np.array_equal(final,previous): raise ValueError('chain poison dependence')
                previous=final.copy()
                norm_input=observed[off['norm']:off['norm']+sizes['norm']].reshape(512,p.OUT)
                wo=norm_input[:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
                prior=norm_input[:,1040:2064].copy().view(np.float32)
                gamma,beta=parameters[prefix-1]['affine'][1]; direct_r,direct_y=p.norm.oracle(wo,prior,gamma,beta)
                geglu=b.unpack_operand(observed[off['mlp_geglu']:off['mlp_geglu']+sizes['mlp_geglu']].reshape(-1,p.OUT),41)
                mlp_input=b.unpack_operand(observed[off['packed_norm']:off['packed_norm']+sizes['packed_norm']].reshape(-1,p.OUT),16)
                wi=b.r.packet_matrix(observed[off['mlp_wi']:off['mlp_wi']+sizes['mlp_wi']].reshape(-1,p.OUT),5248)
                direct_wi=mlp_input.astype(np.float64)@parameters[prefix-1]['weights'][2].astype(np.float64)
                direct_geglu=b.gelu(direct_wi[:,:2624])*direct_wi[:,2624:]
                direct_wo=geglu.astype(np.float64)@parameters[prefix-1]['weights'][3].astype(np.float64)
                strict_wo=p.attention.rounded(direct_geglu).astype(np.float64)@parameters[prefix-1]['weights'][3].astype(np.float64)
                verdicts=dict(final_norm=b.r.norm_verdict(residual,y,direct_r,direct_y),
                  final_wi=b.numeric(wi,direct_wi,.0002,.0002,.00001),
                  final_geglu=b.numeric(geglu,direct_geglu,.000002,.0078125,.004),
                  final_wo=b.numeric(wo,direct_wo,.0002,.0002,.00001),
                  retained_mlp_strict=b.numeric(wo,strict_wo,.0002,.002,.002),
                  chain_y=p.attention.numeric(y,refs[prefix-1][1]),chain_residual=p.attention.numeric(residual,refs[prefix-1][0]))
                if prefix==2:
                    projected=b.r.packet_matrix(observed[off['projection']:off['projection']+sizes['projection']].reshape(-1,p.OUT),3072)
                    independent=p.attention.rounded(snapshots[1]).astype(np.float64)@parameters[1]['weights'][0].astype(np.float64)
                    verdicts['handoff_qkv']=b.numeric(projected,independent,.0002,.0002,.00001)
                mutable=np.zeros(pl['words'],bool)
                for field in ('projection','queries','keys','kv','attention','packed_attention','norm_input','norm','packed_norm','mlp_wi','mlp_geglu','mlp_norm'):
                    start=off[field]; mutable[start:start+sizes[field]]=True
                if not np.array_equal(observed[~mutable],data[~mutable]): raise ValueError('constant, input or arena border overwritten')
                if not all(v['ok'] for k,v in verdicts.items() if k not in ('chain_y','chain_residual','retained_mlp_strict')): raise ValueError('direct operator/handoff gate')
                print('CHAIN_CHECK '+json.dumps(dict(prefix=prefix,fill=fill,dispatches=prefix*2,verdicts=verdicts)),flush=True)
                if not verdicts['retained_mlp_strict']['ok']: strict_failures.append(dict(prefix=prefix,fill=fill,**verdicts['retained_mlp_strict']))
                if not verdicts['chain_y']['ok'] or not verdicts['chain_residual']['ok']:
                    failures.append(dict(prefix=prefix,fill=fill,verdicts=verdicts))
                    if not collect: raise ValueError('composed encoder-prefix gate')
                snapshots[prefix]=y.copy(); checks+=1
        print('CHAIN_RESULT '+json.dumps(dict(checks=checks,layers=count,prefixes=prefixes,composition_failures=failures,strict_mlp_failures=strict_failures,
          contexts_created=2,interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
          scope='encoder prefix; model accuracy and eligible performance unverified')),flush=True)
        if failures or strict_failures: raise ValueError('retained prefix gates failed; thresholds unchanged')
    finally:
        if failed: globals()['_quarantined']=(dev,states,arena,av,hrun)
        else:
            hrun=None; ib=None; av=None; arena=None; state=None; kernel=None; ctx=None; xb=None
            for family in reversed(tuple(states)): states[family]['instructions'].clear()
            for family in reversed(tuple(states)): states[family]['kernel']=None
            for family in reversed(tuple(states)): states[family]['ctx']=None
            for family in reversed(tuple(states)): states[family]['xb']=None
            states.clear(); dev=None; gc.collect(); print('CHAIN_IDLE '+str(idle()),flush=True)
