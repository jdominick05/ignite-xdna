#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One cold-context cache run with exact integer operands and whole-arena guards."""
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import cached_wi_checks as checks
from laya_hybrid import idle
cache=checks.cache
p=cache.p


def emit(tag,value):print(tag+' '+json.dumps(value),flush=True)


def bf16_pair(value):
    bits=int(np.array(value,np.float32).view(np.uint32))>>16
    return np.uint32(bits|(bits<<16))


def prepare(av,plan,poison):
    av.fill(poison);off=plan['offsets']
    for k in range(16):
        for row in range(4):
            start=off['packed_norm']+k*16*p.OUT+row*p.OUT
            av[start:start+16]=0;av[start+2]=cache.q.sq.bm.b.r.mlp.POISON
            av[start+4]=2;av[start+6]=int(k==0);av[start+7]=int(k==15)
            av[start+16:start+1040]=bf16_pair((row+1)*(k+1))
    for group in range(21):
        for k in range(16):
            start=off['mlp_wi_weight']+(group*16+k)*p.KVWORDS
            av[start:start+2048]=bf16_pair((group+1)*(k%3-1))


def verify_output(ov,plan,poison):
    cursor=0;output_words=0
    for group in range(21):
        start=plan['offsets']['mlp_wi']+group*16*p.OUT;end=start+4*p.OUT
        for first in range(cursor,start,1<<22):
            if not np.all(ov[first:min(start,first+(1<<22))]==poison):raise ValueError('cache primitive outside-output guard')
        packets=np.full((4,p.OUT),cache.q.sq.bm.b.r.mlp.POISON,np.uint32)
        for row in range(4):
            # Exactly representable integer products/sums; no learned-model oracle.
            total=64*(row+1)*(group+1)*sum((k+1)*(k%3-1) for k in range(16))
            packets[row,16:2064]=np.array(total,np.float32).view(np.uint32)
        observed=ov[start:end].reshape(4,p.OUT)
        if not np.array_equal(observed,packets):
            mismatch=np.flatnonzero(observed.reshape(-1)!=packets.reshape(-1))
            emit('CACHED_WI_MISMATCH',dict(group=group,words=int(mismatch.size),first=mismatch[:8].tolist()))
            raise ValueError('cached Wi integer output/packet mismatch')
        output_words+=4*p.OUT;cursor=end
    for first in range(cursor,len(ov),1<<22):
        if not np.all(ov[first:first+(1<<22)]==poison):raise ValueError('cache primitive final guard')
    return output_words


def main():
    root=Path.cwd()/'scratch/laya_transformer';parent=root/'small_lookup_v3_m128';folder=root/'cached_wi_v5_col0'
    proof=checks.verify(parent,folder);plan=cache.q.sq.bm.b.plan()
    sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__))
        for module in (checks,sys.modules[__name__])}
    emit('CACHED_WI_PRIMITIVE_START',dict(proof=proof,sources=sources,
        scope='one initialized context, one column/four cores, 21 output groups and 16 K blocks; integer-valued fixture; timing ineligible'))
    if not idle():raise ValueError('device occupied before cache primitive')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
    device=xb=ctx=kernel=instructions=source=target=av=ov=handle=None;pending=False
    try:
        device=xrt.device(0);xb=xrt.xclbin(str(folder/'final.xclbin'));device.register_xclbin(xb)
        ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
        groups=[kernel.group_id(3),kernel.group_id(4)]
        if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('cache primitive bank connectivity')
        sync=xrt.xclBOSyncDirection;raw=(folder/'insts.bin').read_bytes()
        instructions=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1));instructions.write(raw,0)
        instructions.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        source=xrt.bo(device,plan['words']*4,xrt.bo.host_only,groups[0]);target=xrt.bo(device,plan['words']*4,xrt.bo.host_only,groups[1])
        av=np.frombuffer(source.map(),np.uint32);ov=np.frombuffer(target.map(),np.uint32)
        poison=0xdeadbeef;prepare(av,plan,poison);before=hashlib.sha256(memoryview(av)).hexdigest();ov.fill(poison)
        source.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);target.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        emit('CACHED_WI_PRIMITIVE_DISPATCH',dict(cold_context=True,source_sha256=before,poison=poison))
        pending=True;handle=kernel(3,instructions,len(raw),source,target)
        if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('cache primitive timeout; no retry')
        pending=False;handle=None
        target.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE);source.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
        if hashlib.sha256(memoryview(av)).hexdigest()!=before:raise ValueError('cache primitive source changed')
        output_words=verify_output(ov,plan,poison)
        emit('CACHED_WI_PRIMITIVE_CHECK',dict(output_and_packet_words_exact=output_words,source_unchanged=True,all_outside_output_unchanged=True))
    finally:
        if pending:globals()['_quarantined']=(device,xb,ctx,kernel,instructions,source,target,av,ov,handle)
        else:
            av=ov=handle=None;instructions=source=target=kernel=ctx=xb=None;gc.collect();device=None;gc.collect()
    if not idle() or checks.verify(parent,folder)!=proof:raise ValueError('cache primitive cleanup/identity gate')
    if any(p.attention.sha(p.ROOT/name)!=sha for name,sha in sources.items()):raise ValueError('cache runner changed')
    emit('CACHED_WI_PRIMITIVE_RESULT',dict(cases=1,all_output_bits_and_guards_pass=True,idle_while_process_alive=True,
        scope='single cold-context integer primitive; no repeated serving, model parity or speed claim'))


if __name__=='__main__':main()
