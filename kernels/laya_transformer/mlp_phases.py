#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split encoder MLP streams at existing phase waits, without changing core code."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import bucket_mlp as bm
n=bm.native; p=bm.p
PHASES=dict(wi=2,geglu=3,wo=9,norm=1)
BaseSequence=bm.b.MlpSequence


class PhaseSequence(BaseSequence):
    def __init__(self,words,mode):
        super().__init__(words); self.selected=mode; self.active=False
    def control(self,mode,*args,**kwargs):
        self.active=mode==self.selected
        if self.active: super().control(mode,*args,**kwargs)
    def dma(self,*args,**kwargs):
        if self.active: super().dma(*args,**kwargs)
    def wait(self):
        if self.active: super().wait()
    def wait_columns(self,*args):
        if self.active: super().wait_columns(*args)


def runtime(pl,m,phase):
    if phase not in PHASES: raise ValueError('unknown MLP phase')
    # This is compiler dependency injection in a single process, not simulated inference.
    # No frozen source file or artifact is modified, and the original class is restored.
    with patch.object(bm.b,'MlpSequence',lambda words:PhaseSequence(words,PHASES[phase])):
        return bm.runtime(pl,m)


def sources():
    return dict(bm.sources(),**{'kernels/laya_transformer/mlp_phases.py':p.attention.sha(Path(__file__))})


def build(base,model,out,m):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model); vm=bundle.meta['vector']
    out.mkdir(parents=True,exist_ok=False); layouts={}
    for phase in PHASES:
        folder=out/phase; folder.mkdir()
        shutil.copyfile(bundle.folders['vector']/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(n.vector.prefix()+runtime(bundle.plan,m,phase),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        before=p.STACK
        try:
            p.STACK=5376; audits=json.loads(json.dumps(p.audit(folder,{})))
        finally: p.STACK=before
        cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('MLP phase changes core/init image')
        layouts[phase]=dict(audits=audits,cdo=cdo,instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('MLP_PHASE_LAYOUT '+json.dumps(dict(phase=phase,m=m,core_and_init_identical=True,instructions=layouts[phase]['instructions'])),flush=True)
    if sources()!=frozen: raise ValueError('phase builder source changed')
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,m=m,layouts=layouts,
      scope='offline diagnostic streams only; phase timings and complete-output equivalence unverified'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m)
