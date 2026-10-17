#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare cached Wi with captured real NPU inputs, one column and one context."""
import gc
import json
from pathlib import Path
import sys

import laya_small_lookup_phase_full as phase_full
import laya_window_layer_probe as capture
import laya_cached_wi_rearm_primitive as primitive

rearm=primitive.rearm
c=phase_full.c
np=c.np
n=phase_full.n


def output_slices(plan):
    for group in range(21):
        first=plan['offsets']['mlp_wi']+group*16*rearm.p.OUT
        yield slice(first,first+4*rearm.p.OUT)


def main():
    paths,bundle,fixtures,parent,phase_folder,files,proof=phase_full.identity()
    folder=paths.base/'cached_wi_v5_col0';warm=paths.base/'cached_wi_rearm_v1_col0'
    rearm_proof=rearm.verify(parent,folder,warm)
    directory=paths.base/'cached_wi_layer_v1_captures';directory.mkdir(exist_ok=False)
    source_files={str(Path(module.__file__).relative_to(c.ROOT)).replace('\\','/'):c.sha(Path(module.__file__))
        for module in (rearm,primitive,sys.modules[__name__])}
    c.emit('CACHED_WI_LAYER_START',dict(reference_files_digest=c.na.digest(files),restart_proof=rearm_proof,
        source_files=source_files,scope='one-column Wi on real captured layer0/1 inputs; full-arena comparison; no timing claim'))
    if not c.hybrid.idle():raise ValueError('device occupied before cached Wi layer gate')
    calls=[];captures=[]
    with phase_full.full.guards(calls):
        with phase_full.Session(paths,bundle,fixtures,parent,phase_folder) as session:
            for layer in (0,1):
                selected=session.instructions[f'mlp_{layer}_m128_wi'];family=selected.family
                for fixture,item in enumerate(fixtures):
                    records=[];delegate=session.states[family]['kernel']
                    proxy=capture.CaptureKernel(session,delegate,selected,records)
                    session.states[family]['kernel']=proxy
                    try:
                        try:session.run(**phase_full.full.feed(item))
                        except capture.Captured:pass
                        else:raise ValueError('real Wi layer was not captured')
                    finally:session.states[family]['kernel']=delegate;proxy=delegate=None
                    if len(records)!=1 or session.failed:raise ValueError('real Wi capture completion')
                    before,reference=records.pop()
                    outputs=np.concatenate([reference[index] for index in output_slices(bundle.plan)])
                    name=f'layer{layer}_fixture{fixture}';path=directory/(name+'.npy')
                    np.save(path,before,allow_pickle=False)
                    captures.append((layer,fixture,path,c.sha(path),outputs))
                    before=reference=records=None;gc.collect()
                    c.emit('CACHED_WI_LAYER_CAPTURE',dict(layer=layer,fixture=fixture,input_sha256=captures[-1][3],
                        output_sha256=primitive.hashlib.sha256(outputs).hexdigest(),contexts=len(session.states)))
        session=selected=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('real Wi capture contexts not released')
        sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
        device=xb=ctx=kernel=arena=av=handle=bo=before=None;instructions={};pending=False
        try:
            device=xrt.device(0);xb=xrt.xclbin(str(folder/'final.xclbin'));device.register_xclbin(xb)
            ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('real Wi bank mismatch')
            sync=xrt.xclBOSyncDirection
            arena=xrt.bo(device,bundle.plan['words']*4,xrt.bo.host_only,groups[0]);av=np.frombuffer(arena.map(),np.uint32)
            for layer in (0,1):
                spec=bundle.meta['chain']['spec'][layer]
                for index,directory in enumerate((folder,warm)):
                    raw=(directory/'insts.bin').read_bytes()
                    raw,_,used=n.c.l.relocate(raw,n.c.l.bindings(bundle.plan,0,'global'),n.c.l.bindings(bundle.plan,layer,spec['mode']))
                    if used!={'wi'}:raise ValueError('real Wi constant relocation')
                    bo=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1));bo.write(raw,0);bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                    instructions[layer,index]=(bo,len(raw))
            bo=None
            for case,(layer,fixture,path,digest,outputs) in enumerate(captures):
                if c.sha(path)!=digest:raise ValueError('real Wi captured input changed')
                before=np.load(path,mmap_mode='r',allow_pickle=False);av[:]=before
                arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);bo,length=instructions[layer,int(case>0)]
                pending=True;handle=kernel(3,bo,length,arena,arena)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('real Wi timeout; no retry')
                pending=False;handle=None;arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                # Compare every arena word, including unexecuted columns/stages.
                cursor=0;offset=0
                for index in output_slices(bundle.plan):
                    for start in range(cursor,index.start,1<<22):
                        end=min(index.start,start+(1<<22))
                        if not np.array_equal(av[start:end],before[start:end]):raise ValueError('real Wi outside-output guard')
                    count=index.stop-index.start
                    if not np.array_equal(av[index],outputs[offset:offset+count]):raise ValueError('real Wi output packet mismatch')
                    cursor=index.stop;offset+=count
                for start in range(cursor,len(av),1<<22):
                    if not np.array_equal(av[start:start+(1<<22)],before[start:start+(1<<22)]):raise ValueError('real Wi final guard')
                before=None
                c.emit('CACHED_WI_LAYER_CHECK',dict(case=case,layer=layer,fixture=fixture,cold=case==0,
                    arena_words_checked=len(av),output_and_packet_words_exact=offset,all_arena_bits_exact=True))
        finally:
            if pending:globals()['_quarantined']=(device,xb,ctx,kernel,arena,av,handle,instructions,before)
            else:
                av=handle=bo=before=None;instructions.clear();arena=kernel=ctx=xb=None;gc.collect();device=None;gc.collect()
    if calls or not c.hybrid.idle() or rearm.verify(parent,folder,warm)!=rearm_proof:raise ValueError('real Wi cleanup/identity/CPU gate')
    if any(c.sha(c.ROOT/name)!=digest for name,digest in source_files.items()):raise ValueError('real Wi source changed')
    c.emit('CACHED_WI_LAYER_RESULT',dict(cases=len(captures),all_arena_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_while_process_alive=True,scope='one column of real Wi layers only; full MLP/model, all columns and timing pending'))


if __name__=='__main__':main()
