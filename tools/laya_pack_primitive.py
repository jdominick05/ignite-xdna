#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guarded one-column transport qualification; no complete-model speed claim."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import pack_primitive as primitive
import numpy as np
from laya_hybrid import idle
p=primitive.p


def emit(key,value): print(key+' '+json.dumps(value),flush=True)


def main():
    global primitive
    ap=argparse.ArgumentParser(); ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--sweep',action='store_true')
    ap.add_argument('--variant',choices=('original','compact'),required=True); args=ap.parse_args()
    if args.sweep:
        import pack_sweep
        primitive=pack_sweep
    meta=json.loads((args.build/'build.json').read_text()); layout=meta['layouts'][args.variant]
    if meta['sources']!=primitive.sources() or meta['plan']!=primitive.plan():
        raise ValueError('primitive source/plan changed')
    for variant,row in meta['layouts'].items():
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(args.build/variant/file)!=row[key]: raise ValueError('primitive artifact changed')
        for file in (args.build/variant/'prj').rglob('*_aie_cdo_*.bin'):
            if p.attention.sha(file)!=row['cdo'][file.name]: raise ValueError('primitive initialization changed')
    if primitive.topology_proof()!=meta['topology_proof']: raise ValueError('topology gate changed')
    if primitive.init_proof(args.build/'original/prj/cdo_main/main_aie_cdo_init.bin',
        args.build/'compact/prj/cdo_main/main_aie_cdo_init.bin',meta['layouts']['original']['audits'],
        meta['layouts']['compact']['audits'])!=meta['init_proof']: raise ValueError('initialization gate changed')
    if primitive.shim_proof((args.build/'original/insts.bin').read_bytes(),(args.build/'compact/insts.bin').read_bytes(),
        meta['layouts']['original']['audits'],meta['layouts']['compact']['audits'])!=meta['shim_proof']:
        raise ValueError('Shim gate changed')
    if not idle(): raise ValueError('device occupied before primitive')
    emit('PACK_PRIMITIVE_IDENTITY',dict(variant=args.variant,build_sha256=p.attention.sha(args.build/'build.json'),
        runner_sha256=p.attention.sha(Path(__file__)),scope=meta['scope']))
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    device=xrt.device(0); xb=xrt.xclbin(str(args.build/args.variant/'final.xclbin')); device.register_xclbin(xb)
    ctx=xrt.hw_context(device,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    groups=[kernel.group_id(3),kernel.group_id(4)]
    if (groups[0]&0xffff)!=(groups[1]&0xffff): raise ValueError('primitive operand banks differ')
    sync=xrt.xclBOSyncDirection; words=meta['plan']['words']; out_start=meta['plan']['offsets']['packed_norm']
    raw=(args.build/args.variant/'insts.bin').read_bytes()
    instructions=xrt.bo(device,len(raw),xrt.bo.cacheable,kernel.group_id(1))
    instructions.write(raw,0); instructions.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
    source=xrt.bo(device,words*4,xrt.bo.host_only,groups[0]); target=xrt.bo(device,words*4,xrt.bo.host_only,groups[1])
    av=np.frombuffer(source.map(),np.uint32); ov=np.frombuffer(target.map(),np.uint32)
    cases=0; poisoned=False; handle=None
    try:
        for seed in (11,29,47):
            rng=np.random.default_rng(seed)
            values=rng.normal(0,1,(128,p.OUT)).astype(np.float32)
            # Include positive/negative round-to-even tie cases in every core's rows.
            ties=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000],np.uint32).view(np.float32)
            for feature in range(16): values[:,feature*64:feature*64+4]=ties
            regions=primitive.output_regions() if args.sweep else [(out_start,0)]
            for poison in (0xdeadbeef,0x13579bdf):
                av.fill(poison); av[16:16+values.size]=values.reshape(-1).view(np.uint32)
                original=av.copy(); ov.fill(poison)
                expected=np.full(words,poison,np.uint32)
                designated=np.zeros(words,bool)
                for start,feature in regions:
                    feature_values=values.copy()
                    feature_values[:,:64]=values[:,feature*64:feature*64+64]
                    packet=np.full((4,p.OUT),primitive.ba.r.mlp.POISON,np.uint32)
                    packet[:,16:1040]=primitive.expected(feature_values)
                    expected[start:start+4*p.OUT]=packet.reshape(-1)
                    designated[start:start+4*p.OUT]=True
                source.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE); target.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                handle=kernel(3,instructions,len(raw),source,target)
                if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    poisoned=True; raise RuntimeError('packing primitive dispatch did not complete; no retry')
                handle=None
                target.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE); source.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                checks=dict(payload_and_packet_bits_exact=bool(np.array_equal(ov[designated],expected[designated])),
                    input_unchanged=bool(np.array_equal(av,original)),
                    outside_output_unchanged=bool(np.array_equal(ov[~designated],expected[~designated])))
                emit('PACK_PRIMITIVE_CHECK',dict(variant=args.variant,seed=seed,poison=poison,checks=checks))
                if not all(checks.values()): raise ValueError('packing primitive output/guard failure')
                cases+=1
    finally:
        if poisoned:
            globals()['_quarantined']=(device,xb,ctx,kernel,instructions,source,target,av,ov,handle)
        else:
            av=None; ov=None; handle=None; instructions=None; source=None; target=None
            kernel=None; ctx=None; xb=None; device=None; gc.collect()
    if not idle(): raise ValueError('primitive context cleanup failed')
    if meta['sources']!=primitive.sources(): raise ValueError('primitive source changed during execution')
    emit('PACK_PRIMITIVE_RESULT',dict(variant=args.variant,cases=cases,contexts=1,idle_after_close=True,
        all_guards_and_output_bits_pass=True,features=len(regions),scope=meta['scope']))


if __name__=='__main__':
    main()
