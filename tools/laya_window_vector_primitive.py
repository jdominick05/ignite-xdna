#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare paired-query vector operators against the qualified lookup image."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import window_vector_primitive as primitive
from laya_hybrid import idle
p=primitive.p
CASES=[(seed,poison) for seed in (11,29,47) for poison in (0xdeadbeef,0x13579bdf)]


def emit(key,value):print(key+' '+json.dumps(value),flush=True)


def qualify(folder):
    meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=primitive.sources() or meta['plan']!=primitive.plan():raise ValueError('stale vector source/plan')
    for name,row in meta['layouts'].items():
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(folder/name/file)!=row[key]:raise ValueError('changed vector artifact')
        for file in (folder/name/'prj').rglob('*_aie_cdo_*.bin'):
            if p.attention.sha(file)!=row['cdo'][file.name]:raise ValueError('changed vector CDO')
        if primitive.verify_stream((folder/name/'insts.bin').read_bytes(),name=='window',row['audits'])!=row['transport']:
            raise ValueError('changed vector transport gate')
        if primitive.control.verify_vector((folder/name/'prj/input_with_addresses.mlir').read_text(),name=='window')!=row['ledger']:
            raise ValueError('changed vector control ledger')
    relation=primitive.locks.compare(folder/'original',folder/'window')
    if relation!=meta['lock_relation']:raise ValueError('changed vector lock relation')
    emit('WINDOW_VECTOR_OFFLINE_GATES',dict(build_sha256=p.attention.sha(folder/'build.json'),all_passed=True))
    return meta


def packets(target):
    off=primitive.plan()['offsets']
    return np.stack([target[off[name]:off[name]+4*2080].reshape(4,2080).copy() for name in ('norm_output','geglu_output')])


def sanity(result):
    if result.shape!=(2,4,2080) or result.dtype!=np.uint32:raise ValueError('vector output geometry')
    norm,geglu=result;poison=0x7fc12345
    return dict(norm_payload_finite=bool(np.all(np.isfinite(norm[:,:1024].copy().view(np.float32))) and
                                         np.all(np.isfinite(norm[:,1040:2064].copy().view(np.float32)))),
        norm_packet_poison_exact=bool(np.all(norm[:,1024:1040]==poison) and np.all(norm[:,2064:]==poison)),
        geglu_payload_finite=bool(np.all(np.isfinite((geglu[:,16:1040].copy().view(np.uint16).astype(np.uint32)<<16).view(np.float32)))),
        geglu_packet_poison_exact=bool(np.all(geglu[:,:16]==poison) and np.all(geglu[:,1040:]==poison)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--variant',choices=('original','window'),required=True);ap.add_argument('--out',type=Path)
    ap.add_argument('--reference',type=Path);ap.add_argument('--reference-sha256');ap.add_argument('--offline',action='store_true')
    args=ap.parse_args();meta=qualify(args.build);build_sha=p.attention.sha(args.build/'build.json');reference=None
    if args.offline:emit('WINDOW_VECTOR_RUNNER_OFFLINE',dict(passed=True,hardware_contexts_opened=0));return
    if args.variant=='original':
        if args.out is None or args.out.exists():raise ValueError('new baseline output path required')
    else:
        if args.reference is None or not args.reference_sha256 or p.attention.sha(args.reference)!=args.reference_sha256:
            raise ValueError('pinned baseline reference required')
        with np.load(args.reference,allow_pickle=False) as saved:
            if str(saved['build_sha256'])!=build_sha or saved['case_keys'].tolist()!=[list(row) for row in CASES]:
                raise ValueError('reference build/cases differ')
            reference=saved['outputs'].copy()
        if reference.shape!=(6,2,4,2080) or reference.dtype!=np.uint32:raise ValueError('reference output geometry')
    if not idle():raise ValueError('device occupied before vector primitive')
    emit('WINDOW_VECTOR_START',dict(machine='Desktop 2',variant=args.variant,runner_sha256=p.attention.sha(Path(__file__)),
        reference_sha256=args.reference_sha256,scope='norm then GeGLU; one column/four cores; correctness only'))
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python');import pyxrt as xrt
    device=xb=ctx=kernel=instructions=source=target=av=ov=handle=None;poisoned=False;outputs=[]
    try:
        device=xrt.device(0);xb=xrt.xclbin(str(args.build/args.variant/'final.xclbin'));device.register_xclbin(xb)
        ctx=xrt.hw_context(device,xb.get_uuid());kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
        groups=[kernel.group_id(3),kernel.group_id(4)]
        if (groups[0]&0xffff)!=(groups[1]&0xffff):raise ValueError('vector operand banks differ')
        sync=xrt.xclBOSyncDirection;pl=meta['plan'];raw=(args.build/args.variant/'insts.bin').read_bytes()
        instructions=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1));instructions.write(raw,0)
        instructions.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        source=xrt.bo(device,pl['words']*4,xrt.bo.host_only,groups[0]);target=xrt.bo(device,pl['words']*4,xrt.bo.host_only,groups[1])
        av=np.frombuffer(source.map(),np.uint32);ov=np.frombuffer(target.map(),np.uint32)
        designated=np.zeros(pl['words'],bool)
        for name in ('norm_output','geglu_output'):designated[pl['offsets'][name]:pl['offsets'][name]+4*2080]=True
        for index,(seed,poison) in enumerate(CASES):
            av[:]=primitive.fixture(seed);original=av.copy();ov.fill(poison)
            source.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE);target.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
            emit('WINDOW_VECTOR_DISPATCH',dict(variant=args.variant,seed=seed,poison=poison))
            poisoned=True;handle=kernel(3,instructions,len(raw),source,target)
            if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('vector primitive timeout; no retry')
            poisoned=False;handle=None;target.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE);source.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
            result=packets(ov);checks=sanity(result)
            checks.update(source_unchanged=bool(np.array_equal(av,original)),all_outside_output_unchanged=bool(np.all(ov[~designated]==poison)))
            if reference is not None:checks.update(norm_bits_exact=bool(np.array_equal(result[0],reference[index,0])),
                                                 geglu_bits_exact=bool(np.array_equal(result[1],reference[index,1])))
            emit('WINDOW_VECTOR_CHECK',dict(variant=args.variant,seed=seed,poison=poison,checks=checks))
            if not all(checks.values()):raise ValueError('vector output/reference/guard failure')
            outputs.append(result)
    finally:
        if poisoned:globals()['_quarantined']=(device,xb,ctx,kernel,instructions,source,target,av,ov,handle)
        else:av=ov=handle=instructions=source=target=kernel=ctx=xb=device=None;gc.collect()
    if not idle():raise ValueError('vector primitive context cleanup failed')
    if meta['sources']!=primitive.sources():raise ValueError('vector source changed during run')
    out_sha=None
    if args.variant=='original':
        args.out.parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.out,outputs=np.stack(outputs),case_keys=np.array(CASES,dtype=np.uint64),build_sha256=np.array(build_sha))
        out_sha=p.attention.sha(args.out)
    emit('WINDOW_VECTOR_RESULT',dict(variant=args.variant,cases=len(outputs),contexts=1,idle_after_close=True,
        reference_sha256=args.reference_sha256,output_sha256=out_sha,all_checks_pass=True,
        scope='paired-query vector primitive only; full MLP/model and speed gates pending'))


if __name__=='__main__':main()
