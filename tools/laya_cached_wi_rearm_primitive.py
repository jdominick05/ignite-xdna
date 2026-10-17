#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded changed-input cache requests in one context; no performance claim."""
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import cached_wi_rearm as rearm
import laya_cached_wi_primitive as cold
from laya_hybrid import idle

cache=rearm.cache
p=rearm.p


def operands(seed,row,group,k):
    return ((row+1)*(k+1)*((k+seed)%3-1), (group+1)*((k+seed)%5-2))


def expected(seed,row,group):
    return 64*sum(a*b for a,b in (operands(seed,row,group,k) for k in range(16)))


def prepare(av,plan,seed,poison):
    cold.prepare(av,plan,poison);off=plan['offsets']
    for k in range(16):
        for row in range(4):
            start=off['packed_norm']+k*16*p.OUT+row*p.OUT
            av[start+16:start+1040]=cold.bf16_pair(operands(seed,row,0,k)[0])
    for group in range(21):
        for k in range(16):
            start=off['mlp_wi_weight']+(group*16+k)*p.KVWORDS
            av[start:start+2048]=cold.bf16_pair(operands(seed,0,group,k)[1])


def verify_output(ov,plan,seed,poison):
    cursor=0;count=0
    for group in range(21):
        start=plan['offsets']['mlp_wi']+group*16*p.OUT;end=start+4*p.OUT
        for first in range(cursor,start,1<<22):
            if not np.all(ov[first:min(start,first+(1<<22))]==poison):raise ValueError('cache rearm outside-output guard')
        packets=np.full((4,p.OUT),cache.q.sq.bm.b.r.mlp.POISON,np.uint32)
        for row in range(4):packets[row,16:2064]=np.array(expected(seed,row,group),np.float32).view(np.uint32)
        if not np.array_equal(ov[start:end].reshape(4,p.OUT),packets):raise ValueError('cache rearm output/packet mismatch')
        count+=packets.size;cursor=end
    for first in range(cursor,len(ov),1<<22):
        if not np.all(ov[first:first+(1<<22)]==poison):raise ValueError('cache rearm final guard')
    return count


def main():
    root=Path.cwd()/'scratch/laya_transformer';parent=root/'small_lookup_v3_m128'
    folder=root/'cached_wi_v5_col0';warm=root/'cached_wi_rearm_v1_col0'
    proof=rearm.verify(parent,folder,warm);plan=cache.q.sq.bm.b.plan()
    sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__))
        for module in (cold,rearm,sys.modules[__name__])}
    cold.emit('CACHED_WI_REARM_START',dict(proof=proof,sources=sources,seeds=[11,29,47],poisons=[0xdeadbeef,0x13579bdf],
        scope='one column/four cores; cold first then five finite-task restarts; no timing claim'))
    if not idle():raise ValueError('device occupied before cache rearm primitive')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
    device=xb=ctx=kernel=source=target=av=ov=handle=None;instructions=[];pending=False;cases=0
    try:
        device=xrt.device(0);xb=xrt.xclbin(str(folder/'final.xclbin'));device.register_xclbin(xb)
        ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
        groups=[kernel.group_id(3),kernel.group_id(4)]
        if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('cache rearm bank connectivity')
        sync=xrt.xclBOSyncDirection
        raw=[(folder/'insts.bin').read_bytes(),(warm/'insts.bin').read_bytes()]
        for data in raw:
            instruction=xrt.bo(device,len(data),xrt.bo.cacheable,kernel.group_id(1));instruction.write(data,0)
            instruction.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);instructions.append(instruction)
        instruction=None
        source=xrt.bo(device,plan['words']*4,xrt.bo.host_only,groups[0]);target=xrt.bo(device,plan['words']*4,xrt.bo.host_only,groups[1])
        av=np.frombuffer(source.map(),np.uint32);ov=np.frombuffer(target.map(),np.uint32)
        # Changed seed on every adjacent request exposes stale query/weight reuse.
        for poison in (0xdeadbeef,0x13579bdf):
            for seed in (11,29,47):
                prepare(av,plan,seed,poison);before=hashlib.sha256(memoryview(av)).hexdigest();ov.fill(poison)
                source.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);target.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                index=int(cases>0)
                cold.emit('CACHED_WI_REARM_DISPATCH',dict(case=cases,seed=seed,poison=poison,cold=not index,source_sha256=before))
                pending=True;handle=kernel(3,instructions[index],len(raw[index]),source,target)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('cache rearm timeout; no retry')
                pending=False;handle=None
                target.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE);source.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                if hashlib.sha256(memoryview(av)).hexdigest()!=before:raise ValueError('cache rearm source changed')
                words=verify_output(ov,plan,seed,poison)
                cold.emit('CACHED_WI_REARM_CHECK',dict(case=cases,seed=seed,poison=poison,output_and_packet_words_exact=words,
                    source_unchanged=True,all_outside_output_unchanged=True))
                cases+=1
    finally:
        if pending:globals()['_quarantined']=(device,xb,ctx,kernel,instructions,source,target,av,ov,handle)
        else:
            av=ov=handle=None;instructions.clear();source=target=kernel=ctx=xb=None;gc.collect();device=None;gc.collect()
    if not idle() or rearm.verify(parent,folder,warm)!=proof:raise ValueError('cache rearm cleanup/identity gate')
    if any(p.attention.sha(p.ROOT/name)!=sha for name,sha in sources.items()):raise ValueError('cache rearm runner changed')
    cold.emit('CACHED_WI_REARM_RESULT',dict(cases=cases,contexts=1,warm_requests=cases-1,
        all_output_bits_and_guards_pass=True,idle_while_process_alive=True,
        scope='integer primitive only; general matrix, real-layer/full-model parity and speed unproven'))


if __name__=='__main__':main()
