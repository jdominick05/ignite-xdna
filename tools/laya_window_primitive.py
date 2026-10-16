#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guarded GEMM query-reuse primitive; no complete-model speed claim."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import window_lookup_primitive as primitive
import window_lock_check as locks
import window_core_check as control
from laya_hybrid import idle
p=primitive.p


def emit(key,value):print(key+' '+json.dumps(value),flush=True)


def qualify(folder):
    meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=primitive.sources() or meta['plan']!=primitive.plan():raise ValueError('stale primitive source/plan')
    for name,row in meta['layouts'].items():
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(folder/name/file)!=row[key]:raise ValueError('changed primitive artifact')
        for file in (folder/name/'prj').rglob('*_aie_cdo_*.bin'):
            if p.attention.sha(file)!=row['cdo'][file.name]:raise ValueError('changed primitive CDO')
        primitive.verify_stream((folder/name/'insts.bin').read_bytes(),name=='window',int(name[-1]) if name!='window' else 0,row['audits'])
        control.verify((folder/name/'prj/input_with_addresses.mlir').read_text(),name=='window')
    relation=locks.compare(folder/'original0',folder/'window')
    emit('WINDOW_PRIMITIVE_OFFLINE_GATES',dict(build_sha256=p.attention.sha(folder/'build.json'),compiled_lock_relation=relation,
        address_packet_controls_and_repeated_core_ledger=True))
    return meta


def expected_output(window,slot,golden,poison):
    plan=primitive.plan();expected=np.full(plan['words'],poison,np.uint32);designated=np.zeros(plan['words'],bool)
    for index in (range(2) if window else [slot]):
        packets=np.full((4,p.OUT),primitive.candidate.window.bm.b.r.mlp.POISON,np.uint32)
        packets[:,16:2064]=golden[index].view(np.uint32)
        start=plan['output']+index*4*p.OUT
        expected[start:start+4*p.OUT]=packets.reshape(-1);designated[start:start+4*p.OUT]=True
    return expected,designated


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--variant',choices=('original0','original1','window'),required=True)
    ap.add_argument('--offline',action='store_true');args=ap.parse_args()
    meta=qualify(args.build)
    if args.offline:
        emit('WINDOW_PRIMITIVE_RUNNER_OFFLINE',dict(passed=True,hardware_contexts_opened=0));return
    if not idle():raise ValueError('device occupied before primitive')
    window=args.variant=='window';slot=0 if window else int(args.variant[-1])
    emit('WINDOW_PRIMITIVE_START',dict(machine='Desktop 2',variant=args.variant,runner_sha256=p.attention.sha(Path(__file__)),
        scope='one column/four cores; two K blocks; exact integer-valued fixture; timing ineligible'))
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
    device=xb=ctx=kernel=instructions=source=target=av=ov=handle=None;poisoned=False;cases=0
    try:
        device=xrt.device(0);xb=xrt.xclbin(str(args.build/args.variant/'final.xclbin'));device.register_xclbin(xb)
        ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
        groups=[kernel.group_id(3),kernel.group_id(4)]
        if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('primitive operand banks differ')
        sync=xrt.xclBOSyncDirection;plan=meta['plan'];raw=(args.build/args.variant/'insts.bin').read_bytes()
        instructions=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1));instructions.write(raw,0)
        instructions.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        source=xrt.bo(device,plan['words']*4,xrt.bo.host_only,groups[0]);target=xrt.bo(device,plan['words']*4,xrt.bo.host_only,groups[1])
        av=np.frombuffer(source.map(),np.uint32);ov=np.frombuffer(target.map(),np.uint32)
        for seed in (11,29,47):
            payload,golden=primitive.fixture(seed)
            for poison in (0xdeadbeef,0x13579bdf):
                av.fill(poison);av[plan['input']:plan['input']+payload.size]=payload
                original=av.copy();ov.fill(poison);expected,designated=expected_output(window,slot,golden,poison)
                source.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);target.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                emit('WINDOW_PRIMITIVE_DISPATCH',dict(variant=args.variant,seed=seed,poison=poison))
                poisoned=True
                handle=kernel(3,instructions,len(raw),source,target)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    poisoned=True;raise RuntimeError('window GEMM primitive timeout; no retry')
                poisoned=False;handle=None;target.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE);source.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                checks=dict(output_and_packet_bits_exact=bool(np.array_equal(ov[designated],expected[designated])),
                    source_unchanged=bool(np.array_equal(av,original)),
                    all_outside_output_unchanged=bool(np.array_equal(ov[~designated],expected[~designated])))
                emit('WINDOW_PRIMITIVE_CHECK',dict(variant=args.variant,seed=seed,poison=poison,checks=checks))
                if not all(checks.values()):raise ValueError('primitive output/guard failure')
                cases+=1
    finally:
        if poisoned:globals()['_quarantined']=(device,xb,ctx,kernel,instructions,source,target,av,ov,handle)
        else:
            av=ov=handle=instructions=source=target=kernel=ctx=xb=device=None;gc.collect()
    if not idle():raise ValueError('primitive context cleanup failed')
    if meta['sources']!=primitive.sources():raise ValueError('primitive sources changed during run')
    emit('WINDOW_PRIMITIVE_RESULT',dict(variant=args.variant,cases=cases,contexts=1,idle_after_close=True,
        all_output_bits_and_guards_pass=True,scope='GEMM primitive only; paired-query vector and model gates pending; no speed claim'))


if __name__=='__main__':main()
