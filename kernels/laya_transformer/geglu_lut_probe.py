#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unlabelled device development gates for a separate GELU approximation arm."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import fast_buffered_probe as fb
import geglu_lut as gl
n=fb.n; p=fb.p


class Session(fb.Session):
    def validate_candidate(self,meta,bundle):
        if meta['sources']!=gl.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale GeGLU lookup build')
        policy=meta['policy']
        if any(policy[key]!=value for key,value in dict(m=128,q_depth=2,kv_depth=2,l2_q_depth=2).items()): raise ValueError('lookup row/depth policy differs')
        if policy['approximation']!='512 FP32 Gaussian tail-CDF entries at step1/32; Q16 fractional indexing; linear interpolation; asymptote at abs15.9375': raise ValueError('lookup approximation differs')


def stats(values):
    return dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)


def run(args,*,session_factory=None):
    if not 3<=args.reps<=20: raise ValueError('bounded repetitions')
    prereg=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(args.items)!=prereg['items']['file_sha256']: raise ValueError('frozen items changed')
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('unlabelled fixtures only')
    bundle=n.Bundle(args.base,args.model)
    sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__))
      for module in (fb,fb.buffered,fb.buffered.apb,fb.buffered.compact,gl)}
    print('GEGLU_LUT_RUN '+json.dumps(dict(runner_sha256=p.attention.sha(Path(__file__)),sources=sources,
      build_sha256=p.attention.sha(args.fast_folder/'build.json'),
      development_gates=dict(atol=p.attention.ATOL,rtol=p.attention.RTOL,relative_l2=p.attention.L2),
      scope='new approximation development only; qualified native NPU reference, no labelled scoring; original full-set decision/calibration gates still required')),flush=True)
    calls=[]
    def forbidden(*unused,**kwargs): calls.append(True); raise AssertionError('CPU neural/oracle call')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        factory=Session if session_factory is None else session_factory
        with factory(bundle,args.mlp_folder,args.attention_folder,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures,args.fast_folder) as session:
            cases=[(f'fixture{i}',item['ids'],item['markers'],item['qtype']) for i,item in enumerate(fixtures)]
            if not args.quick:
                for length in (1,31,32,33,63,64,65,91,127,128,129,255,256,257,383,384,385,511,512,91):
                    ids=np.full(length,50283,np.int64); keep=min(length,len(fixtures[0]['ids'])); ids[:keep]=fixtures[0]['ids'][:keep]
                    cases.append((f'synthetic{length}',ids,[0,length-1],length%3))
            for key,ids,markers,qtype in cases:
                observed=session.run(ids,markers,qtype)
                reference=session.run(ids,markers,qtype,candidate=False)
                comparisons={name:p.attention.numeric(observed[name],reference[name]) for name in reference}
                exact={name:bool(np.array_equal(observed[name].view(np.uint32),reference[name].view(np.uint32))) for name in reference}
                print('GEGLU_LUT_CHECK '+json.dumps(dict(case=key,valid=len(ids),comparisons=comparisons,exact=exact)),flush=True)
                if not all(result['ok'] for result in comparisons.values()): raise ValueError('lookup development gate failed: '+key)
            item=fixtures[0]; variants=dict(short_original=dict(candidate=False),lookup_with_attention=dict(),lookup_mlp_only=dict(attention_candidate=False))
            timings={key:[] for key in variants}
            for rep in range(args.reps):
                order=list(variants)
                if rep%2: order.reverse()
                for key in order:
                    started=time.perf_counter(); session.run(item['ids'],item['markers'],item['qtype'],**variants[key])
                    timings[key].append((time.perf_counter()-started)*1000)
            session.run(item['ids'],item['markers'],item['qtype'],profile=True)
            groups={}
            for stage in session.last_stats['stages']:
                key='encoder_mlp' if stage['name'].startswith('mlp_') else 'encoder_attention' if stage['name'].startswith('attention_') else stage['name']
                groups[key]=groups.get(key,0)+stage['ms']
            print('GEGLU_LUT_RESULT '+json.dumps(dict(cases=len(cases),development_gates_pass=True,timings={key:stats(value) for key,value in timings.items()},
              profile_groups_ms=groups,cpu_neural_oracle_calls=len(calls),contexts=len(session.states),dispatches=66,
              scope='unlabelled/synthetic development only; full frozen criteria unverified; no direct paired comparison against prior buffered image')),flush=True)
    from laya_hybrid import idle
    print('GEGLU_LUT_IDLE '+str(idle()),flush=True)
    if calls or not idle(): raise ValueError('CPU neural/oracle or cleanup gate')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','fast-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true')
    run(ap.parse_args())
