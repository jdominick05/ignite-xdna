#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic tail math, padding and action-input qualification, no timing."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import tail as t
import vector_probe as vp
p=t.p; b=t.b
GATES=(2e-6,2e-5,2e-5)


def plan():
    bank={}; cursor=16
    for mode in (19,20,21):
        for name,size in [('q',16*p.QWORDS),('kv',4*p.KVWORDS),('out',(17 if mode==20 else 1)*16*p.OUT)]:
            bank[f'{mode}_{name}']=[cursor,size]; cursor+=size+16
    return bank,cursor


def runtime():
    banks,words=plan(); seq=t.Sequence(words)
    for mode in (19,20,21):
        for feature in (range(17) if mode==20 else (0,)):
            seq.control(mode,position=feature)
            for col in range(4):
                seq.dma(col,'q',banks[f'{mode}_q'][0]+col*4*p.QWORDS,[1,1,4,p.QWORDS],[0,0,p.QWORDS,1])
                seq.dma(col,'out',banks[f'{mode}_out'][0]+((feature if mode==20 else 0)*16+col*4)*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',banks[f'{mode}_kv'][0],[1,1,4,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    return seq.finish()


def cases():
    rng=np.random.default_rng(71); values=rng.normal(0,5,128).astype(np.float32)
    result=[]
    for n in (2,3,77,128):
        result.append((f'random_{n}',n,np.ones(n,bool),values.copy()))
        result.append((f'all_masked_{n}',n,np.zeros(n,bool),values.copy()))
    result.append(('one_valid',77,np.arange(77)==51,values.copy()))
    result.append(('holes',77,np.arange(77)%3!=0,values.copy()))
    result.append(('ties',77,np.ones(77,bool),np.zeros(128,np.float32)))
    low=np.full(128,-30000,np.float32); low[:3]=[-20000,-20001,-20002]
    result.append(('exclude_padding_above_real_max',3,np.ones(3,bool),low))
    extreme=np.linspace(-10000,10000,128,dtype=np.float32)
    result.append(('extreme',128,np.ones(128,bool),extreme))
    return result


def oracle(n,mask,scores):
    logits=np.full(128,-10000,np.float32); logits[:n]=np.where(mask,scores[:n],np.float32(-10000))
    probability=np.exp(logits[:n].astype(np.float64)-float(logits[:n].max())); probability/=probability.sum()
    clipped=np.maximum(probability,float(np.float32(1e-9))); logs=np.log(clipped)
    ordered=np.sort(probability)
    count=max(int(mask.sum()),2)
    features=np.array([ordered[-1],ordered[-1]-ordered[-2],-np.sum(probability*logs)/np.log(count),count/255],np.float64)
    return logits,probability,features,logs


def inputs(index,n,mask,scores,fill):
    banks,words=plan(); data=np.full(words,fill,np.uint32)
    q=data[banks['19_q'][0]:sum(banks['19_q'])].reshape(16,p.QWORDS); q[:,0]=n; q[:,16:144]=0; q[:,16:16+n]=mask.astype(np.uint32)
    kv=data[banks['19_kv'][0]:sum(banks['19_kv'])].reshape(4,p.KVWORDS)
    for packet in range(4):
        for row in range(32): kv[packet,16+row*64]=scores[packet*32+row].view(np.uint32)
    rng=np.random.default_rng(200+index); cls=rng.normal(0,3,1024).astype(np.float32)
    cls[:5]=[0,-0.0,1.00390625,-1.00390625,1e-20]
    features=oracle(n,mask,scores)[2].astype(np.float32)
    q=data[banks['20_q'][0]:sum(banks['20_q'])].reshape(16,p.QWORDS); q[:,:1024]=cls.view(np.uint32)
    kv=data[banks['20_kv'][0]:sum(banks['20_kv'])].reshape(4,p.KVWORDS); kv[:,144:148]=features.view(np.uint32)
    two=np.array([scores[0],scores[1]],np.float32)
    q=data[banks['21_q'][0]:sum(banks['21_q'])].reshape(16,p.QWORDS); q[:,16:18]=two.view(np.uint32)
    return data,cls,features,two


def build(source,out):
    base=json.loads((source/'build.json').read_text())
    if base['sources']!=t.sources(): raise ValueError('stale tail image')
    out.mkdir(parents=True,exist_ok=False); shutil.copyfile(source/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(t.prefix()+runtime(),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=base['audits'] or cdo!=base['cdo']: raise ValueError('tail probe core/init drift')
    p.attention.dump(out/'build.json',dict(sources=t.sources(),runner_sha256=p.attention.sha(Path(__file__)),audits=audits,cdo=cdo,
      words=plan()[1],xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('TAIL_PROBE_BUILD '+json.dumps(dict(same_core_and_init=True)),flush=True)


def run(out):
    meta=json.loads((out/'build.json').read_text())
    if meta['sources']!=t.sources() or meta['runner_sha256']!=p.attention.sha(Path(__file__)): raise ValueError('stale tail probe')
    if meta['xclbin']!=p.attention.sha(out/'final.xclbin') or meta['instructions']!=p.attention.sha(out/'insts.bin'): raise ValueError('changed tail probe artifact')
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    raw=(out/'insts.bin').read_bytes(); banks,words=plan(); sync=xrt.xclBOSyncDirection
    ib=bi=bo=iv=ov=hrun=None; failed=False; failures=0; checks=0
    try:
        ib=xrt.bo(dev,len(raw),xrt.bo.cacheable,kernel.group_id(1)); ib.write(raw,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        bi=xrt.bo(dev,words*4,xrt.bo.host_only,kernel.group_id(3)); bo=xrt.bo(dev,words*4,xrt.bo.host_only,kernel.group_id(4))
        iv=np.frombuffer(bi.map(),np.uint32); ov=np.frombuffer(bo.map(),np.uint32)
        print('TAIL_PROBE_RUN '+json.dumps(dict(build=meta,gates=GATES,log_gate=[2e-5,2e-6,2e-5],scope='synthetic only; no model decisions or timing')),flush=True)
        for index,(name,n,mask,scores) in enumerate(cases()):
            logits,prob,features,logs=oracle(n,mask,scores); previous=None
            for fill in (0x7fc12345,0x7fc54321):
                original,cls,packed_features,two=inputs(index,n,mask,scores,fill); iv[:]=original; ov.fill(fill)
                bi.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE); bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                failed=True; hrun=kernel(3,ib,len(raw),bi,bo)
                if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('tail dispatch failed; no retry')
                failed=False; hrun=None; bo.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE); observed=ov.copy()
                rows=observed[banks['19_out'][0]:sum(banks['19_out'])].reshape(16,p.OUT); values=rows.copy().view(np.float32)
                verdicts=dict(logits_bits=bool(np.all(rows[:,16:144]==logits.view(np.uint32))),
                  probabilities=b.numeric(values[:,160:160+n],np.broadcast_to(prob,(16,n)),*GATES),
                  features=b.numeric(values[:,144:148],np.broadcast_to(features,(16,4)),*GATES),
                  logarithms=b.numeric(values[:,288:288+n],np.broadcast_to(logs,(16,n)),2e-5,2e-6,2e-5))
                allowed=np.zeros(p.OUT,bool); allowed[16:144]=True; allowed[144:148]=True; allowed[160:416]=True
                verdicts['collector_guards']=bool(np.all(rows[:,~allowed]==b.r.mlp.POISON))
                packed=observed[banks['20_out'][0]:sum(banks['20_out'])].reshape(17,16,p.OUT); packing=True
                for feature in range(17):
                    for core in range(16):
                        expected=np.full(p.OUT,b.r.mlp.POISON,np.uint32); tile=np.zeros((32,64),np.float32)
                        if core%4==0:
                            if feature<16: tile[0]=cls[feature*64:(feature+1)*64]
                            else: tile[0,:4]=packed_features
                        expected[16:1040]=vp.packed(tile); packing &= bool(np.array_equal(packed[feature,core],expected))
                verdicts['action_pack_bits']=packing
                result=observed[banks['21_out'][0]:sum(banks['21_out'])].reshape(16,p.OUT)
                expected=np.exp(two.astype(np.float64)-float(two.max())); expected/=expected.sum()
                verdicts['action_softmax']=b.numeric(result[:,16:18].copy().view(np.float32),np.broadcast_to(expected,(16,2)),*GATES)
                guard=np.ones(p.OUT,bool); guard[16:18]=False; verdicts['action_guards']=bool(np.all(result[:,guard]==b.r.mlp.POISON))
                good=all(value.get('ok',False) if isinstance(value,dict) else value for value in verdicts.values())
                writable=np.zeros(words,bool)
                for mode in (19,20,21): start,size=banks[f'{mode}_out']; writable[start:start+size]=True
                if not np.all(observed[~writable]==fill): raise ValueError('tail writes outside output banks')
                if previous is not None and not np.array_equal(previous[writable],observed[writable]): raise ValueError('tail poison dependence')
                previous=observed; bi.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                if not np.array_equal(iv,original): raise ValueError('tail input modified')
                print('TAIL_CHECK '+json.dumps(dict(case=name,n=n,fill=fill,ok=good,verdicts=verdicts)),flush=True)
                failures+=int(not good); checks+=1
        print('TAIL_PROBE_RESULT '+json.dumps(dict(checks=checks,failures=failures,scope='operator qualification only')),flush=True)
        if failures: raise ValueError('tail numeric gates')
    finally:
        if failed: globals()['_quarantined']=(dev,xb,ctx,kernel,ib,bi,bo,iv,ov,hrun)
        else:
            hrun=iv=ov=bi=bo=ib=kernel=ctx=xb=dev=None; gc.collect(); print('TAIL_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--source',type=Path); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='build': build(args.source.resolve(),args.out.resolve())
    else: run(args.out.resolve())
