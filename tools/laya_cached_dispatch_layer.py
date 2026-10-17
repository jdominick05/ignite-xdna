#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repeat self-contained cached MLP on immutable qualified NPU layer captures."""
import gc
import json
from pathlib import Path
import sys

import laya_cached_mlp_layer as previous
import laya_cached_mlp_full as failed
import cached_mlp_dispatch as dispatch
c=previous.c
n=previous.n
np=c.np


def captures(root,evidence):
    failed.layer_gate(evidence)
    rows=[json.loads(line.split(' ',1)[1]) for line in evidence.read_text(encoding='utf-8').splitlines()
          if line.startswith('CACHED_MLP_LAYER_CAPTURE ')]
    if {(row['layer'],row['fixture']) for row in rows}!={(a,b) for a in (0,1) for b in range(3)} or len(rows)!=6:
        raise ValueError('incomplete or repeated cached layer captures')
    result=[]
    for row in rows:
        stem=f'layer{row["layer"]}_fixture{row["fixture"]}'
        before=root/(stem+'_before.npy');after=root/(stem+'_after.npy')
        if c.sha(before)!=row['input_sha256'] or c.sha(after)!=row['output_sha256']:raise ValueError('qualified capture changed')
        result.append((row['layer'],row['fixture'],before,after,row['input_sha256'],row['output_sha256']))
    return result


def identity():
    paths,bundle,fixtures,parent,_,files,_=previous.reference.identity()
    folder=paths.base/'cached_mlp_dispatch_v3_m128';proof=dispatch.verify(parent,folder)
    evidence=c.ROOT/'results/laya/cached_mlp_v3_layer_desktop2_20261017.log'
    rows=[json.loads(line.split(' ',1)[1]) for line in evidence.read_text(encoding='utf-8').splitlines()
          if line.startswith('CACHED_MLP_LAYER_START ')]
    if len(rows)!=1 or rows[0]['reference_files_digest']!=c.na.digest(files):raise ValueError('captured layer reference identity changed')
    return paths,bundle,fixtures,parent,folder,evidence,proof


def main():
    paths,bundle,_,parent,folder,evidence,proof=identity()
    records=captures(paths.base/'cached_mlp_v3_layer_captures',evidence)
    sources={str(Path(module.__file__).relative_to(c.ROOT)).replace('\\','/'):c.sha(Path(module.__file__))
        for module in (dispatch,sys.modules[__name__])}
    c.emit('CACHED_DISPATCH_LAYER_START',dict(proof=proof,capture_evidence_sha256=c.sha(evidence),sources=sources,
        scope='six real complete-layer replays; same self-contained query setup each dispatch; no timing'))
    if not c.hybrid.idle():raise ValueError('device occupied before self-contained layer gate')
    calls=[]
    with previous.reference.full.guards(calls):
        sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
        device=xb=ctx=kernel=arena=av=bo=handle=before=after=None;instructions={};pending=False
        try:
            device=xrt.device(0);xb=xrt.xclbin(str(folder/'final.xclbin'));device.register_xclbin(xb)
            ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('self-contained layer bank compatibility')
            sync=xrt.xclBOSyncDirection
            arena=xrt.bo(device,bundle.plan['words']*4,xrt.bo.host_only,groups[0]);av=np.frombuffer(arena.map(),np.uint32)
            for layer in (0,1):
                spec=bundle.meta['chain']['spec'][layer];raw=(folder/'insts.bin').read_bytes()
                raw,_,used=n.c.l.relocate(raw,n.c.l.bindings(bundle.plan,0,'global'),n.c.l.bindings(bundle.plan,layer,spec['mode']))
                if used!={'wi','wo','next_affine'}:raise ValueError('self-contained layer binding coverage')
                bo=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1));bo.write(raw,0);bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                instructions[layer]=(bo,len(raw))
            bo=None
            for case,(layer,fixture,input_path,output_path,input_sha,output_sha) in enumerate(records):
                if (c.sha(input_path),c.sha(output_path))!=(input_sha,output_sha):raise ValueError('capture changed during replay')
                before=np.load(input_path,mmap_mode='r',allow_pickle=False);av[:]=before;before=None
                arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);bo,length=instructions[layer]
                c.emit('CACHED_DISPATCH_LAYER_ISSUE',dict(case=case,layer=layer,fixture=fixture))
                pending=True;handle=kernel(3,bo,length,arena,arena)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('self-contained layer timeout; no retry')
                pending=False;handle=None;arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                after=np.load(output_path,mmap_mode='r',allow_pickle=False);mismatch=0;first=[]
                for start in range(0,len(av),1<<22):
                    bad=np.flatnonzero(av[start:start+(1<<22)]!=after[start:start+(1<<22)]);mismatch+=len(bad)
                    if len(first)<8:first.extend((bad[:8-len(first)]+start).tolist())
                after=None
                c.emit('CACHED_DISPATCH_LAYER_CHECK',dict(case=case,layer=layer,fixture=fixture,
                    arena_words_checked=len(av),all_arena_bits_exact=mismatch==0,mismatch_words=mismatch,first_mismatch_indices=first))
                if mismatch:raise ValueError('self-contained complete MLP arena mismatch')
        finally:
            if pending:globals()['_quarantined']=(device,xb,ctx,kernel,arena,av,bo,handle,instructions,before,after)
            else:
                av=bo=handle=before=after=None;instructions.clear();arena=kernel=ctx=xb=None;gc.collect();device=None;gc.collect()
    if calls or not c.hybrid.idle() or dispatch.verify(parent,folder)!=proof:raise ValueError('self-contained CPU/cleanup/image gate')
    if any(c.sha(c.ROOT/name)!=sha for name,sha in sources.items()):raise ValueError('self-contained replay source changed')
    c.emit('CACHED_DISPATCH_LAYER_RESULT',dict(cases=len(records),all_arena_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,scope='complete isolated MLP correctness only; interleaving/full/frozen/timing gates pending'))


if __name__=='__main__':main()
