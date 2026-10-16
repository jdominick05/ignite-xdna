#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay four phase dispatches on complete qualified NPU-produced MLP inputs."""
import gc
import json
from pathlib import Path
import sys

import laya_small_lookup_compare as comparison
import laya_window_layer_probe as capture
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import small_lookup_phases as phases
c=comparison.c
full=comparison.full
n=phases.n


def identity():
    paths,bundle,fixtures,parent,files,proof=comparison.identity()
    folder=paths.base/'small_lookup_phases_v1_m128';meta=json.loads((folder/'build.json').read_text())
    reference=json.loads((parent/'build.json').read_text())
    if meta['parent_build_sha256']!=c.sha(parent/'build.json') or meta['base_identity']!=bundle.identity:
        raise ValueError('phase parent identity differs')
    transport=phases.validate(folder,meta,bundle.plan,reference)
    files=dict(files);files['phase_build']=c.sha(folder/'build.json')
    for phase in phases.PHASES:
        for filename in ('kernel.o','insts.bin','final.xclbin'):
            files[f'phase/{phase}/{filename}']=c.sha(folder/phase/filename)
    for module in (phases,sys.modules[__name__],capture):
        filename=Path(module.__file__);files['phase_source/'+str(filename.relative_to(c.ROOT)).replace('\\','/')]=c.sha(filename)
    return paths,bundle,fixtures,parent,folder,files,transport


def main():
    paths,bundle,fixtures,parent,folder,files,proof=identity()
    c.emit('SMALL_PHASE_LAYER_IDENTITY',dict(files=files,transport=proof,
        scope='layers0/1 three fixtures; diagnostic host captures; four dispatches versus one complete MLP'))
    if not c.hybrid.idle():raise ValueError('device occupied before phase layer gate')
    calls=[];captures=[]
    with full.guards(calls):
        with comparison.candidate.make(paths,bundle,fixtures,parent,True) as session:
            for layer in (0,1):
                selected=session.buffered_mlp[f'mlp_{layer}_m128'];family=selected.family
                for fixture,item in enumerate(fixtures):
                    records=[];delegate=session.states[family]['kernel']
                    proxy=capture.CaptureKernel(session,delegate,selected,records)
                    session.states[family]['kernel']=proxy
                    try:
                        try:session.run(**full.feed(item))
                        except capture.Captured:pass
                        else:raise ValueError('selected complete MLP was not captured')
                    finally:session.states[family]['kernel']=delegate;proxy=delegate=None
                    if len(records)!=1 or session.failed:raise ValueError('capture did not complete selected layer')
                    captures.append((layer,fixture,*records[0]));records=None
                    c.emit('SMALL_PHASE_LAYER_CAPTURE',dict(layer=layer,fixture=fixture,contexts=len(session.states)))
        session=selected=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('capture contexts not released')
        sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
        device=xb=ctx=kernel=arena=av=handle=bo=None;instructions={};pending=False
        try:
            device=xrt.device(0);xb=xrt.xclbin(str(parent/'final.xclbin'));device.register_xclbin(xb)
            ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('phase operand banks differ')
            sync=xrt.xclBOSyncDirection
            arena=xrt.bo(device,bundle.plan['words']*4,xrt.bo.host_only,groups[0]);av=c.np.frombuffer(arena.map(),c.np.uint32)
            for layer in (0,1):
                spec=bundle.meta['chain']['spec'][layer]
                for phase in phases.PHASES:
                    raw=(folder/phase/'insts.bin').read_bytes()
                    raw,_,used=n.c.l.relocate(raw,n.c.l.bindings(bundle.plan,0,'global'),n.c.l.bindings(bundle.plan,layer,spec['mode']))
                    if used!={'wi':{'wi'},'geglu':set(),'wo':{'wo'},'norm':{'next_affine'}}[phase]:
                        raise ValueError('phase constant relocation coverage')
                    bo=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1))
                    bo.write(raw,0);bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                    instructions[layer,phase]=(bo,len(raw))
            bo=None
            for layer,fixture,before,reference in captures:
                av[:]=before;arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                for phase in phases.PHASES:
                    bo,length=instructions[layer,phase];pending=True
                    handle=kernel(3,bo,length,arena,arena)
                    if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('phase dispatch timeout; no retry')
                    pending=False;handle=None
                arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE);mismatch=c.np.flatnonzero(av!=reference)
                c.emit('SMALL_PHASE_LAYER_CHECK',dict(layer=layer,fixture=fixture,all_arena_bits_exact=not len(mismatch),
                    mismatch_words=int(len(mismatch)),first_mismatch_indices=mismatch[:8].tolist(),contexts=1,dispatches=4))
                if len(mismatch):raise ValueError('split phase whole-layer/guard mismatch')
        finally:
            if pending:globals()['_quarantined']=(device,xb,ctx,kernel,arena,av,handle,instructions)
            else:
                av=handle=bo=None;instructions.clear();arena=kernel=ctx=xb=None;gc.collect();device=None;gc.collect()
    count=len(captures);captures.clear();before=reference=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[5]!=files:raise ValueError('phase CPU/source/cleanup gate')
    c.emit('SMALL_PHASE_LAYER_RESULT',dict(cases=count,all_arena_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,registered_files_unchanged=True,scope='complete isolated MLP layers; full model and timing pending'))


if __name__=='__main__':main()
