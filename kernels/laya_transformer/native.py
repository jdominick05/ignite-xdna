#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resident complete Laya inference. No CPU neural operations or reference oracles.

Experimental research runtime, batch one, static 512-token device geometry.
The host prepares integer token/marker addresses and option metadata only.
"""
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import action
import chain as c
import chain_run as cr
import embedding_carry as carry
import embedded
import stem as stem
import stem_run
import vector

p=c.p
DEFAULT_BUILDS=dict(chain='chain_v4_28',vector='vector_v6',head_pre='head_pre_v3',
  head_attention='head_attention_v4',head_post='head_post_v2',head_second='head_second_v2',
  scorer='scorer_v2',tail='tail_v4',action='action_v2',carry='embedding_carry_v1')


def request(ids,markers,qtype,mask=None):
    ids=np.asarray(ids); markers=np.asarray(markers)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not 1<=ids.size<=512 or np.any(ids<0) or np.any(ids>=stem.VOCAB): raise ValueError('token vector/count/range')
    if isinstance(qtype,(bool,np.bool_)) or not isinstance(qtype,(int,np.integer)) or not 0<=qtype<3: raise ValueError('question type')
    action.s.markers(markers)
    if np.any(markers>=ids.size): raise ValueError('marker beyond actual token extent')
    if mask is None: mask=np.ones(markers.size,bool)
    mask=np.asarray(mask); action.metadata(mask)
    if mask.size!=markers.size: raise ValueError('marker mask extent')
    padded=np.full(512,50283,np.int64); padded[:ids.size]=ids
    return padded,np.maximum(markers,0).astype(np.int64),int(qtype),mask.copy(),int(ids.size)


class Bundle:
    """Offline source, model and image checks before a hardware context exists."""
    def __init__(self,base,model,builds=None):
        self.base=Path(base).resolve(); self.model=Path(model).resolve()
        self.folders={key:self.base/name for key,name in (DEFAULT_BUILDS if builds is None else builds).items()}
        self.meta={key:json.loads((folder/'build.json').read_text()) for key,folder in self.folders.items()}
        modules=dict(chain=c,vector=vector,head_pre=action.s.second.post.a.h,head_attention=action.s.second.post.a,
          head_post=action.s.second.post,head_second=action.s.second,scorer=action.s,tail=action.t,action=action,carry=carry)
        self.files={}; self.sources={}
        for name,module in modules.items():
            meta=self.meta[name]
            if meta['sources']!=module.sources(): raise ValueError('stale native build: '+name)
            self.sources.update(meta['sources']); self.files[name+'/build.json']=p.attention.sha(self.folders[name]/'build.json')
        self.sources.update({str(Path(x.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(x.__file__)) for x in (sys.modules[__name__],cr,embedded,stem_run)})
        cm=self.meta['chain']; vm=self.meta['vector']; tm=self.meta['tail']
        vector.validate_compatibility(self.folders['vector'],vm,cm['artifacts']['mlp'])
        relationships=dict(head_pre=dict(vector='vector_build_sha256',chain='chain_build_sha256'),
          head_attention=dict(head_pre='head_pre_build_sha256',chain='chain_build_sha256'),
          head_post=dict(head_attention='head_attention_build_sha256',vector='vector_build_sha256'),
          head_second=dict(head_post='head_post_build_sha256',vector='vector_build_sha256'),
          scorer=dict(head_second='head_second_build_sha256',vector='vector_build_sha256'),
          action=dict(scorer='scorer_build_sha256',tail='tail_build_sha256'),
          carry=dict(vector='vector_build_sha256',action='plan_build_sha256'))
        for name,parents in relationships.items():
            for parent,field in parents.items():
                if self.meta[name][field]!=self.files[parent+'/build.json']: raise ValueError('native build parent changed: '+name)
        self.plan=self.meta['action']['plan']
        if self.meta['carry']['plan']!=self.plan: raise ValueError('carry plan differs')
        for name in ('head_pre','head_post','scorer','carry'):
            if self.meta[name]['audits']!=vm['audits'] or self.meta[name]['cdo']!=vm['cdo']: raise ValueError('vector image ABI differs: '+name)
        if self.meta['head_attention']['audits']!=cm['artifacts']['first']['audits'] or self.meta['head_attention']['cdo']!=cm['artifacts']['first']['cdo']: raise ValueError('attention image ABI differs')
        if self.meta['action']['audits']!=tm['audits'] or self.meta['action']['cdo']!=tm['cdo']: raise ValueError('tail image ABI differs')
        if cm['model_sha256']!=p.attention.sha(self.model) or self.meta['carry']['model_sha256']!=cm['model_sha256']: raise ValueError('native model graph changed')
        if self.meta['carry']['contract']!=carry.graph_contract(self.model): raise ValueError('first residual graph changed')
        self.files['model/'+self.model.name]=cm['model_sha256']
        import onnx
        graph=onnx.load(str(self.model),load_external_data=False)
        locations=sorted({entry.value for tensor in graph.graph.initializer for entry in tensor.external_data if entry.key=='location'})
        for location in locations:
            filename=(self.model.parent/location).resolve()
            if not filename.is_relative_to(self.model.parent): raise ValueError('external weight path leaves model directory')
            self.files['model/'+location]=p.attention.sha(filename)
        for name,meta in self.meta.items():
            artifacts=meta['artifacts'] if name in ('chain','head_second') else {'':meta}
            for variant,artifact in artifacts.items():
                folder=self.folders[name]/variant
                for filename,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
                    if p.attention.sha(folder/filename)!=artifact[key]: raise ValueError('changed native image/instructions')
                    self.files['/'.join(x for x in (name,variant,filename) if x)]=artifact[key]
                if name=='head_second' and (artifact['audits']!=vm['audits'] or artifact['cdo']!=vm['cdo']): raise ValueError('second head image ABI differs')
        coefficients=self.folders['chain']/'coefficients.npz'
        if p.attention.sha(coefficients)!=cm['coefficients_sha256']: raise ValueError('coefficient table changed')
        self.files['chain/coefficients.npz']=cm['coefficients_sha256']
        for stage in cm['stages']:
            if p.attention.sha(self.folders['chain']/stage['instructions'])!=stage['sha256']: raise ValueError('changed stage instructions')
            self.files['chain/'+stage['instructions']]=stage['sha256']
        self.identity=dict(files=self.files,sources=self.sources,policy=dict(batch=1,tokens_padded=512,pad_id=50283,
          initial_residual='device normalized embedding Y',host_work='integer token/marker address and option metadata preparation',
          dispatches=66,contexts=3,cpu_neural_fallback=False,readback='only model logits and act_probs'))


class Instruction:
    def __init__(self,owner,family,raw):
        self.owner=owner; self.family=family; self.raw=bytearray(raw)
        kernel=owner.states[family]['kernel']; xrt=owner.xrt
        self.bo=xrt.bo(owner.device,len(raw),xrt.bo.cacheable,kernel.group_id(1)); self.length=len(raw)
        self.upload()
    def upload(self):
        self.bo.write(self.raw,0); self.bo.sync(self.owner.sync.XCL_BO_SYNC_BO_TO_DEVICE)
    def integers(self,edits):
        changed=False
        for offset,value in edits:
            if offset<0 or offset+4>len(self.raw) or offset%4 or not 0<=int(value)<2**32: raise ValueError('instruction integer patch')
            if struct.unpack_from('<I',self.raw,offset)[0]!=int(value): struct.pack_into('<I',self.raw,offset,int(value)); changed=True
        if changed: self.upload()
    def close(self): self.bo=None; self.owner=None


class NativeSession:
    def __init__(self,bundle):
        self.bundle=bundle; self.plan=bundle.plan; self.states={}; self.instructions={}; self.steps=[]
        self.device=self.arena=self.av=self.embedding=self.ev=None; self.failed=False; self.closed=False; self.busy=False
        self.frame_count=0; self.last_stats=None
        sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
        if not idle(): raise RuntimeError('foreign NPU context')
        sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
        self.xrt=xrt; self.sync=xrt.xclBOSyncDirection
        try:
            self.device=xrt.device(0)
            for family,folder in [('attention',bundle.folders['chain']/'first'),('mlp',bundle.folders['vector']),('tail',bundle.folders['action'])]:
                xb=xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
                ctx=xrt.hw_context(self.device,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
                self.states[family]=dict(xb=xb,ctx=ctx,kernel=kernel,groups=[kernel.group_id(3),kernel.group_id(4)])
            groups=[g for state in self.states.values() for g in state['groups']]
            if len({g&0xffff for g in groups})!=1: raise ValueError('native activation bank mismatch')
            self.groups=groups
            self._instructions()
            self.arena=xrt.bo(self.device,self.plan['words']*4,xrt.bo.host_only,groups[0]); self.av=np.frombuffer(self.arena.map(),np.uint32)
            self.embedding=xrt.bo(self.device,stem.INPUT_WORDS*4,xrt.bo.host_only,self.states['mlp']['groups'][0]); self.ev=np.frombuffer(self.embedding.map(),np.uint32)
            self._constants()
            self.last_metadata=None
        except BaseException:
            self.close(); raise

    def _add(self,name,family,raw):
        inst=Instruction(self,family,raw); self.instructions[name]=inst; self.steps.append(name); return inst

    def _instructions(self):
        bundle=self.bundle; meta=bundle.meta; folders=bundle.folders; pl=self.plan
        raw=folders['vector'].joinpath('insts.bin').read_bytes(); tokens=np.asarray(meta['vector']['tokens'],np.int64)
        raw,token_fields=stem.prepare(raw,tokens,tokens)
        raw,_=embedded.relocate_output(raw,meta['vector']['target'],pl['offsets']['mlp_norm'],pl['words'])
        self._add('stem','mlp',raw)
        emission=[col*128+group*4+row for group in range(32) for col in range(4) for row in range(4)]
        self.token_patches=list(zip(token_fields,emission))
        self._add('carry','mlp',(folders['carry']/'insts.bin').read_bytes()); self.mask_patches={}
        for stage in meta['chain']['stages']:
            family=stage['family']; index=stage['layer']; raw=(folders['chain']/stage['instructions']).read_bytes()
            if family=='attention' and index==0:
                raw=(folders['chain']/'repeat/insts.bin').read_bytes()
                raw,_,used=c.l.relocate(raw,c.l.bindings(pl,1,meta['chain']['spec'][1]['mode']),c.l.bindings(pl,0,meta['chain']['spec'][0]['mode']))
                if used!={'qkv','projection','affine','coeff'}: raise ValueError('packed first-layer QKV relocation')
            name=f'{family}_{index}'
            if family=='attention':
                raw,changes=c.txn.patch(raw,meta['chain']['artifacts']['first']['audits'],512,meta['chain']['spec'][index]['mode']=='local')
                self.mask_patches[name]=[row['value_offset'] for row in changes if row['field']=='valid']
            self._add(name,family,raw)
        raw=(folders['head_pre']/'insts.bin').read_bytes(); action.s.second.post.a.h.prepare(raw,pl,0)
        type_offset=pl['offsets']['head_type']*4
        self.type_patches=[op['offset']+40 for op in c.txn.parse(raw) if op['op']=='DDR_PATCH' and op['arg_idx']==0 and op['arg_offset']==type_offset]
        self._add('head_pre','mlp',raw)
        raw=(folders['head_attention']/'insts.bin').read_bytes(); raw,changes=c.txn.patch(raw,meta['head_attention']['audits'],512,False)
        self.mask_patches['head_attention']=[row['value_offset'] for row in changes if row['field']=='valid']
        self._add('head_attention','attention',raw)
        self._add('head_post','mlp',(folders['head_post']/'insts.bin').read_bytes())
        self._add('second_pre','mlp',(folders['head_second']/'pre/insts.bin').read_bytes())
        self.steps.append('head_attention')
        self._add('second_post','mlp',(folders['head_second']/'post/insts.bin').read_bytes())
        raw=(folders['scorer']/'insts.bin').read_bytes(); action.s.prepare(raw,pl,np.arange(128,dtype=np.int64))
        source=pl['offsets']['mlp_norm']*4; end=source+512*p.OUT*4
        order=[col*32+group*4+row for group in range(8) for col in range(4) for row in range(4)]
        fields=[op['offset']+40 for op in c.txn.parse(raw) if op['op']=='DDR_PATCH' and op['arg_idx']==0 and source<=op['arg_offset']<end]
        self.marker_patches=list(zip(fields,order)); self._add('scorer','mlp',raw)
        self._add('action','tail',(folders['action']/'insts.bin').read_bytes())
        if len(self.steps)!=66 or len(self.token_patches)!=512 or len(self.marker_patches)!=128 or len(self.type_patches)!=4 or any(len(x)!=16 for x in self.mask_patches.values()): raise ValueError('native operation/patch coverage')

    def _constants(self):
        bundle=self.bundle; model=bundle.model; pl=self.plan; fill=0x7fc12345
        graph,spec=c.l.model_spec(model,28)
        if spec!=bundle.meta['chain']['spec']: raise ValueError('native layer structure changed')
        parameters,_=c.l.constants(model,graph,spec); graph=None; gc.collect()
        data=np.full(pl['words'],fill,np.uint32)
        with np.load(bundle.folders['chain']/'coefficients.npz') as tables: cr.initialize_constants(data,pl,parameters,tables,fill)
        parameters=None
        for module in (action.s.second.post.a.h,action.s.second.post,action.s.second,action.s,action):
            values,_=module.parameters(model)
            if module is action: module.initialize(data,pl,values,np.ones(2,bool),fill)
            else: module.initialize(data,pl,values,fill)
            values=None
        action.s.second.post.a.initialize(data,pl)
        self.av[:]=data; self.arena.sync(self.sync.XCL_BO_SYNC_BO_TO_DEVICE); data=None
        table,gamma=stem_run.parameters(model); packet=stem_run.constant_packet(table,gamma,fill)
        self.ev[:]=packet; self.embedding.sync(self.sync.XCL_BO_SYNC_BO_TO_DEVICE)
        table=gamma=packet=None; gc.collect()

    def run(self,ids,markers,qtype,mask=None,*,full_readback=False,profile=False):
        if self.closed or self.failed: raise RuntimeError('native session closed or failed')
        if self.busy: raise RuntimeError('native session is single tenant')
        started=time.perf_counter(); tokens,markers,qtype,mask,valid=request(ids,markers,qtype,mask)
        self.busy=True
        try:
            self.instructions['stem'].integers((offset,int(tokens[index])*4096) for offset,index in self.token_patches)
            for name,offsets in self.mask_patches.items(): self.instructions[name].integers((offset,valid) for offset in offsets)
            type_value=(self.plan['offsets']['head_type']+qtype*p.KVWORDS)*4
            self.instructions['head_pre'].integers((offset,type_value) for offset in self.type_patches)
            padded=action.s.markers(markers); source=self.plan['offsets']['mlp_norm']
            self.instructions['scorer'].integers((offset,(source+int(padded[index])*p.OUT)*4) for offset,index in self.marker_patches)
            metadata=action.metadata(mask); metadata_bytes=metadata.tobytes()
            if metadata_bytes!=self.last_metadata:
                start=self.plan['offsets']['tail_metadata']; self.av[start:start+p.QWORDS]=metadata
                self.arena.sync(self.sync.XCL_BO_SYNC_BO_TO_DEVICE,p.QWORDS*4,start*4); self.last_metadata=metadata_bytes
            prepared=time.perf_counter()
            stages=[]
            for name in self.steps:
                inst=self.instructions[name]; kernel=self.states[inst.family]['kernel']
                if profile: stage_start=time.perf_counter()
                self.failed=True
                hrun=kernel(3,inst.bo,inst.length,self.embedding if name=='stem' else self.arena,self.arena)
                if hrun.wait(15000)!=self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('native dispatch failed: '+name)
                self.failed=False; hrun=None
                if profile: stages.append(dict(name=name,family=inst.family,ms=(time.perf_counter()-stage_start)*1000))
            completed=time.perf_counter(); off=self.plan['offsets']; n=len(markers)
            if full_readback: self.arena.sync(self.sync.XCL_BO_SYNC_BO_FROM_DEVICE)
            else:
                self.arena.sync(self.sync.XCL_BO_SYNC_BO_FROM_DEVICE,n*4,(off['tail_features']+16)*4)
                self.arena.sync(self.sync.XCL_BO_SYNC_BO_FROM_DEVICE,8,(off['action_probs']+16)*4)
            logits,probabilities=action.outputs(self.av,self.plan,n)
            if not np.isfinite(logits).all() or not np.isfinite(probabilities).all(): raise ValueError('nonfinite native model output')
            stopped=time.perf_counter(); self.frame_count+=1
            self.last_stats=dict(prepare_ms=(prepared-started)*1000,dispatch_ms=(completed-prepared)*1000,
              readback_ms=(stopped-completed)*1000,total_ms=(stopped-started)*1000,dispatches=len(self.steps),
              output_sync_bytes=self.plan['words']*4 if full_readback else (n+2)*4,full_readback=full_readback)
            if profile: self.last_stats['stages']=stages
            return dict(logits=logits.reshape(1,n),act_probs=probabilities.reshape(1,2))
        finally: self.busy=False

    def close(self):
        if self.closed: return
        if self.failed:
            # Keep in-flight BO/context ownership until bounded process exit.
            globals().setdefault('_quarantined',[]).append(self); self.closed=True; return
        for inst in self.instructions.values(): inst.close()
        self.instructions.clear(); self.av=self.ev=None; self.arena=self.embedding=None
        for state in self.states.values(): state['kernel']=None
        for state in self.states.values(): state['ctx']=None; state['xb']=None
        self.states.clear(); self.device=None; self.closed=True; gc.collect()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
