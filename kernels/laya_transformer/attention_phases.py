#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split qualified attention at existing completion waits for wall-time diagnosis."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import attention_fifo as af
ba=af.ba; n=af.n; p=af.p
PHASES=('prepare','attend','finish')
BaseSequence=ba.r.Sequence


class PhaseSequence(BaseSequence):
    def __init__(self,words,phase):
        super().__init__(words)
        self.phase=phase; self.active=False; self.after_attention=False

    def control(self,mode,*args,**kwargs):
        if mode==0:
            selected='attend'; self.after_attention=True
        else:
            selected='finish' if self.after_attention else 'prepare'
        self.active=selected==self.phase
        if self.active:
            super().control(mode,*args,**kwargs)

    def dma(self,*args,**kwargs):
        if self.active:
            super().dma(*args,**kwargs)

    def wait(self):
        if self.active:
            super().wait()


def runtime(plan,m,phase):
    if phase not in PHASES:
        raise ValueError('unknown attention phase')
    with patch.object(ba.r,'Sequence',lambda words:PhaseSequence(words,phase)):
        return ba.runtime(plan,m)


def sources():
    return dict(af.sources(),**{'kernels/laya_transformer/attention_phases.py':p.attention.sha(Path(__file__))})


def execution_signature(data,audits):
    """Keep every descriptor/queue/wait byte and effective RTP at operand patches."""
    registers={}
    for audit in audits:
        col,row=audit['core']; core=col*4+row-2
        spans=[(start,end) for start,end,name in audit['spans'] if name==f'chain_control_{core}']
        if len(spans)!=1 or spans[0][1]-spans[0][0]!=64:
            raise ValueError('phase control allocation differs')
        base=(col<<25)|(row<<20)|spans[0][0]
        for index in range(7): registers[base+index*4]=(core,index)
    if len(registers)!=112:
        raise ValueError('incomplete phase control map')
    state={}; result=[]
    for op in n.c.txn.parse(data):
        target=registers.get(op.get('addr'))
        if target is not None:
            if op['op']!='WRITE' or op['size']!=24 or op['val_low']!=0:
                raise ValueError('unexpected phase control encoding')
            state[target]=op['val']; continue
        snapshot=None
        if op['op']=='DDR_PATCH':
            if len(state)!=112:
                raise ValueError('phase operand used before all controls initialized')
            snapshot=tuple(sorted(state.items()))
        result.append((data[op['offset']:op['offset']+op['size']],snapshot))
    return result


def verify_streams(candidate,out,audits):
    original=execution_signature((candidate/'insts.bin').read_bytes(),audits)
    phases=[execution_signature((out/phase/'insts.bin').read_bytes(),audits) for phase in PHASES]
    if sum(phases,[])!=original:
        raise ValueError('phase descriptors, packet/queue/wait bytes or effective controls differ')
    return dict(non_rtp_operations=len(original),phase_operations=[len(phase) for phase in phases],
        descriptors_queues_waits_byte_identical=True,effective_controls_identical=True)


def build(base,model,candidate,out):
    out=out.absolute(); candidate=candidate.absolute()
    frozen=sources(); bundle=n.Bundle(base,model)
    meta=json.loads((candidate/'build.json').read_text())
    if meta['sources']!=af.sources() or meta['base_identity']!=bundle.identity:
        raise ValueError('stale qualified attention build')
    if p.attention.sha(candidate/'kernel.o')!=meta['kernel_object_sha256']:
        raise ValueError('qualified attention object changed')
    policy=meta['policy']; m=policy['m']
    prefix=af.prefix(policy['q_depth'],policy['kv_depth'],policy['l2_q_depth'])
    out.mkdir(parents=True,exist_ok=False); layouts={}
    for phase in PHASES:
        folder=out/phase; folder.mkdir()
        shutil.copyfile(candidate/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(prefix+runtime(bundle.plan,m,phase),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        before=p.STACK
        try:
            p.STACK=4864; audits=json.loads(json.dumps(p.audit(folder,{'fp_exp_ab':2048,'fp_exp_cd':2048})))
        finally:
            p.STACK=before
        cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=meta['audits'] or cdo!=meta['cdo']:
            raise ValueError('phase changes qualified core or lock/buffer initialization')
        layouts[phase]=dict(audits=audits,cdo=cdo,instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('ATTENTION_PHASE_LAYOUT '+json.dumps(dict(phase=phase,core_and_init_identical=True)),flush=True)
    equivalence=verify_streams(candidate,out,meta['audits'])
    print('ATTENTION_PHASE_TRANSPORT_EQUIVALENCE '+json.dumps(equivalence),flush=True)
    if sources()!=frozen:
        raise ValueError('phase sources changed during build')
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,policy=policy,
        candidate_build_sha256=p.attention.sha(candidate/'build.json'),layouts=layouts,equivalence=equivalence,
        scope='diagnostic streams; no device qualification or speed claim'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','candidate','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args(); build(args.base,args.model,args.candidate,args.out)
