#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete inference with split MLP phases, for serialized device diagnostics."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import bucket_attention as ba
import mlp_phases as mp

n=ba.n; p=ba.p


class Session(ba.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,phase_folder):
        folder=Path(phase_folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=mp.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale phase diagnostic build')
        if set(meta['layouts'])!=set(mp.PHASES): raise ValueError('incomplete MLP phases')
        for phase,layout in meta['layouts'].items():
            if layout['audits']!=bundle.meta['vector']['audits'] or layout['cdo']!=bundle.meta['vector']['cdo']: raise ValueError('phase core/init differs')
            for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
                if p.attention.sha(folder/phase/file)!=layout[key]: raise ValueError('phase artifact differs')
        super().__init__(bundle,mlp_folder,attention_folder)
        try:
            self.phase_m=meta['m']
            for phase in mp.PHASES:
                raw=(folder/phase/'insts.bin').read_bytes()
                for index,spec in enumerate(bundle.meta['chain']['spec']):
                    relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                    expected={'wi':{'wi'},'geglu':set(),'wo':{'wo'},'norm':{'next_affine'}}[phase]
                    if used!=expected: raise ValueError('phase constant coverage differs: '+phase)
                    name=f'mlp_{index}_m{self.phase_m}_{phase}'
                    self.instructions[name]=n.Instruction(self,'mlp',relocated)
        except BaseException:
            self.close(); raise

    def run(self,ids,markers,qtype,mask=None,*,split=True,profile=False):
        valid=n.request(ids,markers,qtype,mask)[-1]; m=ba.bm.bucket(valid)
        if not split or m!=self.phase_m: return super().run(ids,markers,qtype,mask,profile=profile)
        self.steps=[]
        for name in self.baseline_steps:
            if name.startswith('mlp_'): self.steps.extend(name+f'_m{m}_'+phase for phase in mp.PHASES)
            elif name.startswith('attention_'): self.steps.append(name+f'_m{m}')
            else: self.steps.append(name)
        if len(self.steps)!=150: raise ValueError('split phase dispatch coverage')
        self.mask_patches=dict(self.short_masks[m],head_attention=self.original_masks['head_attention'])
        return n.NativeSession.run(self,ids,markers,qtype,mask,profile=profile)


def run(args):
    if not 3<=args.reps<=20: raise ValueError('bounded repetition count')
    prereg=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(args.items)!=prereg['items']['file_sha256']: raise ValueError('frozen items changed')
    items=[json.loads(row) for row in args.items.read_text().splitlines()][:3]
    if any(item['gold']!=-1 or item['task']!='fixture' for item in items): raise ValueError('unlabelled fixtures only')
    meta=json.loads((args.phase_folder/'build.json').read_text())
    if any(ba.bm.bucket(len(item['ids']))!=meta['m'] for item in items): raise ValueError('phase row bucket differs from fixture')
    bundle=n.Bundle(args.base,args.model); calls=[]
    def forbidden(*args,**kwargs): calls.append(True); raise AssertionError('CPU neural/oracle call')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    emit=lambda key,value: print(key+' '+json.dumps(value),flush=True)
    emit('MLP_PHASE_RUN',dict(bundle=bundle.identity,phase_build_sha256=p.attention.sha(args.phase_folder/'build.json'),
      runner_sha256=p.attention.sha(Path(__file__)),scope='phase wall times include extra dispatch/wait overhead; no core cycle or bandwidth claim'))
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        with Session(bundle,args.mlp_folder,args.attention_folder,args.phase_folder) as session:
            for index,item in enumerate(items):
                observed=session.run(item['ids'],item['markers'],item['qtype'],split=True)
                original=session.run(item['ids'],item['markers'],item['qtype'],split=False)
                exact={key:bool(np.array_equal(observed[key].view(np.uint32),original[key].view(np.uint32))) for key in original}
                emit('MLP_PHASE_CHECK',dict(fixture=index,exact=exact))
                if not all(exact.values()): raise ValueError('phase split changed complete outputs')
            item=items[0]; samples={'unsplit':[],'split':[]}; groups={phase:[] for phase in mp.PHASES}
            for rep in range(args.reps):
                for name in (('unsplit','split') if rep%2==0 else ('split','unsplit')):
                    started=time.perf_counter(); session.run(item['ids'],item['markers'],item['qtype'],split=name=='split')
                    samples[name].append((time.perf_counter()-started)*1000)
                session.run(item['ids'],item['markers'],item['qtype'],split=True,profile=True)
                totals={phase:0. for phase in mp.PHASES}; counts={phase:0 for phase in mp.PHASES}
                for stage in session.last_stats['stages']:
                    if stage['name'].startswith('mlp_'):
                        phase=stage['name'].rsplit('_',1)[-1]; totals[phase]+=stage['ms']; counts[phase]+=1
                if set(counts.values())!={28}: raise ValueError('missing per-layer phase profile')
                for phase,value in totals.items(): groups[phase].append(value)
            summary=lambda values:dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)
            emit('MLP_PHASE_RESULT',dict(complete_model={key:summary(values) for key,values in samples.items()},
              encoder_phase_groups={key:summary(values) for key,values in groups.items()},cpu_neural_oracle_calls=len(calls),
              unsplit_dispatches=66,split_dispatches=150,contexts=len(session.states),fixtures_bit_exact=True,
              scope='M128 diagnostic; phase sums across28layers; driver/dispatch perturbation measured by paired complete calls; labelled metrics not evaluated'))
    from laya_hybrid import idle
    emit('MLP_PHASE_IDLE',dict(idle=idle()))
    if calls or not idle(): raise ValueError('CPU oracle/cleanup gate')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','phase-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); run(ap.parse_args())
