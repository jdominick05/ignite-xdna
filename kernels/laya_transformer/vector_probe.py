#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device qualification for head-vector commands; synthetic inputs, no timing."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import vector as v
b=v.b; p=v.p
MODES=(10,13,14,15,16,17,18)


def plan():
    banks={}; cursor=16
    for mode in MODES:
        parts={}
        for name,size in [('q',16*p.QWORDS),('kv',4*128*p.KVWORDS),('out',16*p.OUT)]:
            parts[name]=cursor; cursor+=size+16
        banks[mode]=parts
    return banks,cursor


def runtime():
    banks,words=plan(); seq=v.Sequence(words)
    for mode,off in banks.items():
        seq.control(mode,position=(0,1,2,3))
        count=128 if mode==10 else 1 if mode in (17,18) else 4
        for col in range(4):
            seq.dma(col,'q',off['q']+col*4*p.QWORDS,[1,1,4,p.QWORDS],[0,0,p.QWORDS,1])
            seq.dma(col,'out',off['out']+col*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
            seq.dma(col,'kv',off['kv']+col*128*p.KVWORDS,[1,1,count,p.KVWORDS],[0,0,p.KVWORDS,1])
        seq.wait()
    return seq.finish()


def build(source,out):
    meta=json.loads((source/'build.json').read_text())
    if meta['sources']!=v.sources(): raise ValueError('stale vector program')
    out.mkdir(parents=True,exist_ok=False); shutil.copyfile(source/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(v.prefix()+runtime(),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=meta['audits'] or cdo!=meta['cdo']: raise ValueError('vector qualification core/layout drift')
    p.attention.dump(out/'build.json',dict(sources=v.sources(),runner_sha256=p.attention.sha(Path(__file__)),audits=audits,
      cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('VECTOR_PROBE_BUILD '+json.dumps(dict(code_bytes=sorted({a['code_bytes'] for a in audits}),
      stack_bound=max(sum(a['frames'])+256 for a in audits),same_core_and_init=True)),flush=True)


def packed(value):
    # Independent 4x8 physical layout and integer round-to-nearest-even.
    u=np.asarray(value,np.float32).view(np.uint32)
    bits=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
    return bits.reshape(8,4,8,8).transpose(0,2,1,3).copy().ravel().view(np.uint32)


def inputs(seed,fill):
    banks,words=plan(); arena=np.full(words,fill,np.uint32); expected={}; rng=np.random.default_rng(seed)
    for mode,off in banks.items():
        results=np.full((16,p.OUT),b.r.mlp.POISON,np.uint32)
        for col in range(4):
            q=arena[off['q']+col*4*p.QWORDS:off['q']+(col+1)*4*p.QWORDS].reshape(4,p.QWORDS)
            kv=arena[off['kv']+col*128*p.KVWORDS:off['kv']+(col+1)*128*p.KVWORDS].reshape(128,p.KVWORDS)
            if mode==10:
                raw=rng.normal(0,2,(128,1024)).astype(np.float32)
                kv[:,:1024]=raw.view(np.uint32)
                for row in range(4): results[col*4+row,16:1040]=packed(raw[row*32:(row+1)*32,col*64:(col+1)*64])
            elif mode in (17,18):
                raw=rng.normal(0,2,(4,1024)).astype(np.float32); residual=rng.normal(0,1,(4,1024)).astype(np.float32)
                kv[0,:1024]=residual[0].view(np.uint32)
                for row in range(4):
                    result=results[col*4+row]
                    if mode==17:
                        q[row,:1024]=raw[row].view(np.uint32)
                        result[:1040].reshape(16,65)[:,:64]=raw[row].reshape(16,64).view(np.uint32)
                        result[1040:2064]=residual[0].view(np.uint32)
                    else:
                        q[row,:1040].reshape(16,65)[:,:64]=raw[row].reshape(16,64).view(np.uint32)
                        q[row,1040:2064]=residual[row].view(np.uint32)
                        summed=(raw[row].astype(np.float64)+residual[row].astype(np.float64)).astype(np.float32)
                        result[:1024]=summed.view(np.uint32); result[1040:2064]=summed.view(np.uint32)
            else:
                raw=rng.normal(0,3,(4,32,64)).astype(np.float32); bias=rng.normal(0,1,(4,64)).astype(np.float32)
                # Cancellation, signed zero, tiny numbers, GELU tails and RNE ties.
                raw[:,:,0]=0; bias[:,0]=0; raw[:,:,1]=-0.0; bias[:,1]=-0.0
                raw[:,:,2]=np.linspace(-12,12,32); bias[:,2]=0
                for row in range(4):
                    q[row,16:]=raw[row].ravel().view(np.uint32); kv[row,:64]=bias[row].view(np.uint32)
                    value=(raw[row].astype(np.float64)+bias[row].astype(np.float64)).astype(np.float32)
                    result=results[col*4+row]
                    if mode==13: result[16:1040]=packed(np.maximum(value,0))
                    elif mode==14: result[16:1040]=packed(b.gelu(value).astype(np.float32))
                    elif mode==15: result[16:2064]=value.ravel().view(np.uint32)
                    else: result[:2080].reshape(32,65)[:,:64]=value.view(np.uint32)
        expected[mode]=results
    return arena,expected


def run(out):
    meta=json.loads((out/'build.json').read_text())
    if meta['sources']!=v.sources() or meta['runner_sha256']!=p.attention.sha(Path(__file__)): raise ValueError('stale vector probe')
    if meta['xclbin']!=p.attention.sha(out/'final.xclbin') or meta['instructions']!=p.attention.sha(out/'insts.bin'): raise ValueError('changed vector artifact')
    banks,words=plan()
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    bi=bo=ib=iv=ov=hrun=None; failed=False; sync=xrt.xclBOSyncDirection
    try:
        raw=(out/'insts.bin').read_bytes(); ib=xrt.bo(dev,len(raw),xrt.bo.cacheable,kernel.group_id(1)); ib.write(raw,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        bi=xrt.bo(dev,words*4,xrt.bo.host_only,kernel.group_id(3)); bo=xrt.bo(dev,words*4,xrt.bo.host_only,kernel.group_id(4))
        iv=np.frombuffer(bi.map(),np.uint32); ov=np.frombuffer(bo.map(),np.uint32)
        print('VECTOR_PROBE_RUN '+json.dumps(dict(build=meta,scope='synthetic operator qualification only; no timing',seeds=[71,193,0],poisons=[0x7fc12345,0x7fc54321])),flush=True)
        count=0; failures=0
        for seed in (71,193,0):
            previous=None
            for fill in (0x7fc12345,0x7fc54321):
                original,expected=inputs(seed,fill); iv[:]=original; ov.fill(fill)
                bi.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE); bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                failed=True; hrun=kernel(3,ib,len(raw),bi,bo)
                if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('vector dispatch failed; no retry')
                failed=False; hrun=None; bo.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE); observed=ov.copy()
                writable=np.zeros(words,bool)
                for mode,off in banks.items():
                    actual=observed[off['out']:off['out']+16*p.OUT].reshape(16,p.OUT)
                    ref=expected[mode]; bad=int(np.count_nonzero(actual!=ref)); good=bad==0
                    # GELU is the existing approximate activation: compare bf16 values
                    # with its original pointwise gate, but keep all guards exact.
                    max_abs=0.0
                    if mode==14:
                        a=(actual[:,16:1040].copy().view(np.uint16).astype(np.uint32)<<16).view(np.float32)
                        e=(ref[:,16:1040].copy().view(np.uint16).astype(np.uint32)<<16).view(np.float32)
                        diff=np.abs(a-e); max_abs=float(diff.max())
                        guard=np.ones(p.OUT,bool); guard[16:1040]=False
                        verdict=b.numeric(a,e,.000002,.0078125,.004)
                        good=bool(verdict['ok'] and np.array_equal(actual[:,guard],ref[:,guard]))
                    print('VECTOR_CHECK '+json.dumps(dict(seed=seed,fill=fill,mode=mode,ok=good,bit_mismatches=bad,max_abs=max_abs,
                      gate='exact bits' if mode!=14 else dict(atol=.000002,rtol=.0078125,l2=.004))),flush=True)
                    failures+=int(not good); count+=1; writable[off['out']:off['out']+16*p.OUT]=True
                if not np.all(observed[~writable]==fill): raise ValueError('write outside output banks')
                if previous is not None and not np.array_equal(previous[writable],observed[writable]): raise ValueError('poison dependence')
                previous=observed
                bi.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                if not np.array_equal(iv,original): raise ValueError('input modified')
        print('VECTOR_PROBE_RESULT '+json.dumps(dict(checks=count,failures=failures,scope='operator qualification; not complete head or full-model accuracy')),flush=True)
        if failures: raise ValueError('vector qualification gates')
    finally:
        if failed: globals()['_quarantined']=(dev,xb,ctx,kernel,bi,bo,ib,iv,ov,hrun)
        else:
            hrun=iv=ov=bi=bo=ib=kernel=ctx=xb=dev=None; gc.collect(); print('VECTOR_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--source',type=Path)
    args=ap.parse_args()
    if args.command=='build': build(args.source.resolve(),args.out.resolve())
    else: run(args.out.resolve())
