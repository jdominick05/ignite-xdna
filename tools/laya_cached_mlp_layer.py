#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete four-column cache/normal MLP parity on real captured NPU inputs."""
import gc
import json
from pathlib import Path
import sys

import laya_small_lookup_phase_layer as reference
import laya_window_layer_probe as capture
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import cached_mlp_checks as checks

c=reference.c
np=c.np
n=reference.n


def qualify(parent,folder,warm):
    proof=checks.verify(parent,folder)
    meta=json.loads((warm/'build.json').read_text(encoding='utf-8'))
    if meta['checker_sha256']!=c.sha(Path(checks.__file__)) or meta['cache_build_sha256']!=c.sha(folder/'build.json'):
        raise ValueError('cached MLP warm build provenance')
    if meta['warm_instructions_sha256']!=c.sha(warm/'insts.bin') or meta['cold_proof']!=proof:
        raise ValueError('cached MLP warm artifact/proof changed')
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    old=checks.c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    original=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    warm_proof=checks.runtime_proof((warm/'insts.bin').read_bytes(),text,old,checks.q.sq.bm.b.plan(),original['audits'],False)
    if meta['warm_proof']!=warm_proof:raise ValueError('cached MLP warm operation proof changed')
    return dict(cold=proof,warm=warm_proof)


def main():
    paths,bundle,fixtures,parent,_,files,_=reference.identity()
    folder=paths.base/'cached_mlp_v3_m128';warm=paths.base/'cached_mlp_warm_v1_m128'
    proof=qualify(parent,folder,warm)
    directory=paths.base/'cached_mlp_v3_layer_captures';directory.mkdir(exist_ok=False)
    sources={str(Path(module.__file__).relative_to(c.ROOT)).replace('\\','/'):c.sha(Path(module.__file__))
        for module in (checks,checks.c,sys.modules[__name__])}
    c.emit('CACHED_MLP_LAYER_START',dict(proof=proof,reference_files_digest=c.na.digest(files),sources=sources,
        scope='first complete four-column cached MLP layers0/1; one cold then five warm, no timing claim'))
    if not c.hybrid.idle():raise ValueError('device occupied before cached full MLP layer gate')
    calls=[];captures=[]
    with reference.full.guards(calls):
        with reference.comparison.candidate.make(paths,bundle,fixtures,parent,True) as session:
            for layer in (0,1):
                selected=session.buffered_mlp[f'mlp_{layer}_m128'];family=selected.family
                for fixture,item in enumerate(fixtures):
                    records=[];delegate=session.states[family]['kernel'];proxy=capture.CaptureKernel(session,delegate,selected,records)
                    session.states[family]['kernel']=proxy
                    try:
                        try:session.run(**reference.full.feed(item))
                        except capture.Captured:pass
                        else:raise ValueError('complete MLP input was not captured')
                    finally:session.states[family]['kernel']=delegate;proxy=delegate=None
                    if len(records)!=1 or session.failed:raise ValueError('complete MLP reference capture completion')
                    before,after=records.pop();name=f'layer{layer}_fixture{fixture}'
                    input_path=directory/(name+'_before.npy');output_path=directory/(name+'_after.npy')
                    np.save(input_path,before,allow_pickle=False);np.save(output_path,after,allow_pickle=False)
                    digests=(c.sha(input_path),c.sha(output_path));captures.append((layer,fixture,input_path,output_path,digests))
                    before=after=records=None;gc.collect()
                    c.emit('CACHED_MLP_LAYER_CAPTURE',dict(layer=layer,fixture=fixture,input_sha256=digests[0],output_sha256=digests[1],contexts=len(session.states)))
        session=selected=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('reference cached layer contexts not released')
        sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
        device=xb=ctx=kernel=arena=av=handle=bo=before=after=None;instructions={};pending=False
        try:
            device=xrt.device(0);xb=xrt.xclbin(str(folder/'final.xclbin'));device.register_xclbin(xb)
            ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('cached full MLP bank compatibility')
            sync=xrt.xclBOSyncDirection
            arena=xrt.bo(device,bundle.plan['words']*4,xrt.bo.host_only,groups[0]);av=np.frombuffer(arena.map(),np.uint32)
            for layer in (0,1):
                spec=bundle.meta['chain']['spec'][layer]
                for index,directory in enumerate((folder,warm)):
                    raw=(directory/'insts.bin').read_bytes()
                    raw,_,used=n.c.l.relocate(raw,n.c.l.bindings(bundle.plan,0,'global'),n.c.l.bindings(bundle.plan,layer,spec['mode']))
                    if used!={'wi','wo','next_affine'}:raise ValueError('cached full MLP constant coverage')
                    bo=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1));bo.write(raw,0);bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                    instructions[layer,index]=(bo,len(raw))
            bo=None
            for case,(layer,fixture,input_path,output_path,digests) in enumerate(captures):
                if (c.sha(input_path),c.sha(output_path))!=digests:raise ValueError('cached full MLP capture changed')
                before=np.load(input_path,mmap_mode='r',allow_pickle=False);av[:]=before;before=None
                arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);bo,length=instructions[layer,int(case>0)]
                c.emit('CACHED_MLP_LAYER_DISPATCH',dict(case=case,layer=layer,fixture=fixture,cold=case==0))
                pending=True;handle=kernel(3,bo,length,arena,arena)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('cached full MLP timeout; no retry')
                pending=False;handle=None;arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                after=np.load(output_path,mmap_mode='r',allow_pickle=False);mismatch=0;first=[]
                for start in range(0,len(av),1<<22):
                    bad=np.flatnonzero(av[start:start+(1<<22)]!=after[start:start+(1<<22)])
                    mismatch+=len(bad)
                    if len(first)<8:first.extend((bad[:8-len(first)]+start).tolist())
                after=None
                c.emit('CACHED_MLP_LAYER_CHECK',dict(case=case,layer=layer,fixture=fixture,arena_words_checked=len(av),
                    all_arena_bits_exact=mismatch==0,mismatch_words=mismatch,first_mismatch_indices=first))
                if mismatch:raise ValueError('cached full MLP arena mismatch')
        finally:
            if pending:globals()['_quarantined']=(device,xb,ctx,kernel,arena,av,handle,instructions,before,after)
            else:
                av=handle=bo=before=after=None;instructions.clear();arena=kernel=ctx=xb=None;gc.collect();device=None;gc.collect()
    if calls or not c.hybrid.idle() or qualify(parent,folder,warm)!=proof:raise ValueError('cached MLP CPU/cleanup/identity gate')
    if any(c.sha(c.ROOT/name)!=digest for name,digest in sources.items()):raise ValueError('cached full MLP source changed')
    c.emit('CACHED_MLP_LAYER_RESULT',dict(cases=len(captures),all_arena_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,scope='complete isolated MLP layers; full-model and original frozen gates and speed remain open'))


if __name__=='__main__':main()
