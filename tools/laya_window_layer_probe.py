#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capture qualified layer inputs, then compare isolated complete window MLPs.

Host copies are explicit diagnostic setup, excluded from serving/timing claims.
No CPU numerical operator supplies any captured or candidate activation.
"""
import gc
import json
from pathlib import Path
import sys

import numpy as np

import laya_pack_full_probe as full
import laya_window_vector_primitive as vector_runner
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import window_mlp_transport as transport
candidate=transport.candidate;c=full.c;p=full.p;n=full.n


class Captured(Exception):pass


class CaptureKernel:
    def __init__(self,owner,delegate,selected,records):
        self.owner=owner;self.delegate=delegate;self.selected=selected;self.records=records
    def __getattr__(self,name):return getattr(self.delegate,name)
    def __call__(self,*args):
        if args[1] is not self.selected.bo:return self.delegate(*args)
        owner=self.owner;owner.arena.sync(owner.sync.XCL_BO_SYNC_BO_FROM_DEVICE);before=owner.av.copy()
        handle=self.delegate(*args);records=self.records
        class CaptureWait:
            def wait(self,timeout):
                state=handle.wait(timeout)
                if state!=owner.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:return state
                owner.arena.sync(owner.sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                records.append((before,owner.av.copy()))
                # The dispatch is authoritatively complete. This is an intentional
                # diagnostic stop; no in-flight context is being marked healthy.
                owner.failed=False
                raise Captured()
        return CaptureWait()


def prerequisite(path,tag,variant):
    lines=path.read_text().splitlines()
    rows=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(tag+' ')]
    terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
    if (len(rows)!=1 or rows[0]['variant']!=variant or not rows[0]['idle_after_close'] or
        len(terminal)!=1 or terminal[0]['exit_code']!=0):raise ValueError('incomplete primitive prerequisite')
    key='all_checks_pass' if tag=='WINDOW_VECTOR_RESULT' else 'all_output_bits_and_guards_pass'
    if not rows[0][key] or rows[0]['cases']!=6:raise ValueError('primitive prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,files,_=full.identity();folder=paths.base/'window_lookup_v2_m128'
    meta=json.loads((folder/'build.json').read_text())
    expected=dict(m=128,width=2,qwords=1040,kvwords=2064,q_depth=2,kv_depth=1,l2_kv_depth=2,l2_q_depth=2,stack_bytes=3328)
    if meta['sources']!=candidate.sources() or meta['base_identity']!=bundle.identity or meta['policy']!=expected:
        raise ValueError('window full MLP identity differs')
    for name,sha in meta['generated_sources'].items():
        if c.sha(folder/(name+'.cc'))!=sha:raise ValueError('window generated source changed')
    for name,sha in meta['unchanged_objects'].items():
        if c.sha(bundle.folders['vector']/(name+'.o'))!=sha:raise ValueError('window arithmetic object changed')
    raw=full.op.artifacts(folder,meta)
    primitive_folder=paths.base/'window_vector_primitive_v2';vector_meta=vector_runner.qualify(primitive_folder)
    row=vector_meta['layouts']['window']
    if meta['audits']!=row['audits'] or meta['cdo']!=row['cdo'] or c.sha(primitive_folder/'window/kernel.o')!=meta['kernel_object_sha256']:
        raise ValueError('complete MLP changes qualified primitive core/init')
    proofs={}
    for file,tag,variant in [('window_lookup_gemm_primitive_window_v1_desktop2_20261016.log','WINDOW_PRIMITIVE_RESULT','window'),
                            ('window_vector_primitive_window_v1_desktop2_20261016.log','WINDOW_VECTOR_RESULT','window')]:
        proofs[file]=prerequisite(c.ROOT/'results/laya'/file,tag,variant)
    proof=transport.verify(raw,bundle.plan,meta['audits'])
    for file in ('build.json','kernel.o','insts.bin','final.xclbin'):files['window/'+file]=c.sha(folder/file)
    for module in (transport,candidate,vector_runner,sys.modules[__name__]):
        file=Path(module.__file__);files['window_source/'+str(file.relative_to(c.ROOT)).replace('\\','/')]=c.sha(file)
    files['vector_build']=c.sha(primitive_folder/'build.json')
    return paths,bundle,fixtures,folder,meta,raw,files,dict(transport=proof,prerequisites=proofs)


def main():
    paths,bundle,fixtures,folder,meta,raw,files,proof=identity()
    c.emit('WINDOW_LAYER_IDENTITY',dict(files=files,proof=proof,arena_words=bundle.plan['words'],
        scope='captured complete MLP layers0/1; host copies for diagnostic setup; no full-model or speed claim'))
    if not c.hybrid.idle():raise ValueError('device occupied')
    calls=[];captures=[]
    with full.guards(calls):
        with full.make(paths,bundle,fixtures,True) as session:
            for layer in (0,1):
                name=f'mlp_{layer}_m128';selected=session.buffered_mlp[name];family=selected.family
                for fixture,item in enumerate(fixtures):
                    if candidate.window.bm.bucket(len(item['ids']))!=128:raise ValueError('fixture is not M128')
                    records=[];delegate=session.states[family]['kernel']
                    proxy=CaptureKernel(session,delegate,selected,records);session.states[family]['kernel']=proxy
                    try:
                        try:session.run(**full.feed(item))
                        except Captured:pass
                        else:raise ValueError('target layer was not captured')
                    finally:
                        session.states[family]['kernel']=delegate;proxy=delegate=None
                    if len(records)!=1 or session.failed:raise ValueError('layer capture completion differs')
                    captures.append((layer,fixture,*records[0]));records=None
                    c.emit('WINDOW_LAYER_CAPTURE',dict(layer=layer,fixture=fixture,source='qualified complete model prefix',
                        contexts=len(session.states),stopped_after_completed_layer=True))
        session=None;selected=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('capture session context cleanup')
        sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
        device=xb=ctx=kernel=arena=av=handle=None;instructions={};poisoned=False
        try:
            device=xrt.device(0);xb=xrt.xclbin(str(folder/'final.xclbin'));device.register_xclbin(xb)
            ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('MLP operand banks differ')
            sync=xrt.xclBOSyncDirection;arena=xrt.bo(device,bundle.plan['words']*4,xrt.bo.host_only,groups[0]);av=np.frombuffer(arena.map(),np.uint32)
            for layer in (0,1):
                spec=bundle.meta['chain']['spec'][layer]
                relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(bundle.plan,0,'global'),n.c.l.bindings(bundle.plan,layer,spec['mode']))
                if used!={'wi','wo','next_affine'}:raise ValueError('window layer constant relocation differs')
                bo=xrt.bo(device,len(relocated),xrt.bo.cacheable,kernel.group_id(1));bo.write(relocated,0);bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                instructions[layer]=(bo,len(relocated))
            bo=None
            for layer,fixture,before,reference in captures:
                av[:]=before;arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);bo,length=instructions[layer]
                poisoned=True;handle=kernel(3,bo,length,arena,arena)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('complete MLP layer timeout; no retry')
                poisoned=False;handle=None;arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                mismatch=np.flatnonzero(av!=reference)
                c.emit('WINDOW_LAYER_CHECK',dict(layer=layer,fixture=fixture,all_arena_bits_exact=not len(mismatch),
                    mismatch_words=int(len(mismatch)),first_mismatch_indices=mismatch[:8].tolist(),contexts=1))
                if len(mismatch):raise ValueError('complete MLP layer/guard mismatch')
        finally:
            if poisoned:globals()['_quarantined']=(device,xb,ctx,kernel,arena,av,handle,instructions)
            else:
                c.emit('WINDOW_LAYER_CLEANUP',dict(phase='begin',instruction_bos=len(instructions),device_alive=device is not None))
                av=handle=bo=None
                instructions.clear()
                c.emit('WINDOW_LAYER_CLEANUP',dict(phase='instruction_bos_released',device_alive=device is not None))
                arena=None;kernel=None;ctx=None;xb=None;gc.collect()
                c.emit('WINDOW_LAYER_CLEANUP',dict(phase='context_released',device_alive=device is not None))
                device=None;gc.collect()
                c.emit('WINDOW_LAYER_CLEANUP',dict(phase='device_released'))
    case_count=len(captures);captures.clear();before=reference=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[6]!=files:raise ValueError('layer CPU/identity/cleanup gate')
    c.emit('WINDOW_LAYER_RESULT',dict(cases=case_count,all_arena_bits_exact=True,cpu_neural_oracle_calls=len(calls),idle_after_close=True,
        scope='MLP layers0/1 on three fixtures; explicit diagnostic host copies; complete model and speed still pending'))


if __name__=='__main__':main()
