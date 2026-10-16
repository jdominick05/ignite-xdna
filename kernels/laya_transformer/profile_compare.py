#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded A/B/A serving comparison with serialized, separately warmed contexts."""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import optimized_probe as op
import laya_accuracy as la
from laya_hybrid import idle
n=op.n; p=op.p


def run(args):
    if not 3<=args.reps<=20: raise ValueError('bounded repetitions')
    pre=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(args.items)!=pre['items']['file_sha256']: raise ValueError('frozen items changed')
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(row['gold']!=-1 or row['task']!='fixture' for row in fixtures): raise ValueError('unlabelled fixtures required')
    for name,sha in pre['reference']['files'].items():
        if p.attention.sha(args.model.parent/name)!=sha: raise ValueError('CPU reference graph changed')
    bundle=n.Bundle(args.base,args.model); item=fixtures[0]
    modules=(op,op.hp,op.hp.buffered,op.fp,op.lp,op.cw,op.sq)
    manifest={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__)) for module in modules}
    manifest[str(Path(__file__).relative_to(p.ROOT)).replace('\\','/')]=p.attention.sha(Path(__file__))
    builds={name:p.attention.sha(getattr(args,name)/'build.json') for name in
      ('mlp_folder','attention_folder','mlp_candidate','attention_candidate','compact_folder','head_folder','fast_folder')}
    print('PROFILE_COMPARE_RUN '+json.dumps(dict(sources=manifest,builds=builds,base_identity=bundle.identity,
      order=['buffered_short_heads_before',args.mlp_kind+'_short_heads','buffered_short_heads_after'],
      valid=len(item['ids']),ort=ort.__version__,cpu_threads=8,reps=args.reps,warmups=2,
      scope='A/B/A complete fixture inference, setup excluded; CPU measured before and after NPU; full frozen accuracy unverified')),flush=True)
    def cpu_samples():
        cpu=la.lb.session(args.model.parent/'laya.onnx','cpu',8)
        feed=la.feed(item['ids'],item['markers'],item['qtype'])
        for _ in range(2): cpu.run(None,feed)
        samples=[]
        for _ in range(args.reps):
            started=time.perf_counter(); cpu.run(None,feed); samples.append((time.perf_counter()-started)*1000)
        cpu=None; gc.collect()
        return op.lp.stats(samples)
    cpu_before=cpu_samples(); calls=[]; reference=None; results=[]
    def forbidden(*unused,**kwargs): calls.append(True); raise AssertionError('CPU neural/oracle call during NPU arm')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        for index in range(3):
            candidate=index==1
            common=(bundle,args.mlp_folder,args.attention_folder,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures,args.head_folder)
            factory=op.Session if candidate else op.hp.Session
            extra=dict(fast_folder=args.fast_folder,mlp_kind=args.mlp_kind) if candidate else {}
            with factory(*common,**extra) as session:
                outputs=session.run(item['ids'],item['markers'],item['qtype'])
                if reference is None: reference={key:value.copy() for key,value in outputs.items()}
                exact={key:bool(np.array_equal(value.view(np.uint32),reference[key].view(np.uint32))) for key,value in outputs.items()}
                numeric={key:p.attention.numeric(value,reference[key]) for key,value in outputs.items()}
                if not all(row['ok'] for row in numeric.values()): raise ValueError('profile development numeric gate')
                if (not candidate or args.mlp_kind!='lookup') and not all(exact.values()): raise ValueError('same-math profile differs')
                for _ in range(2): session.run(item['ids'],item['markers'],item['qtype'])
                samples=[]
                for _ in range(args.reps):
                    started=time.perf_counter(); session.run(item['ids'],item['markers'],item['qtype'])
                    samples.append((time.perf_counter()-started)*1000)
                if session.last_stats['dispatches']!=66 or len(session.states)!=5: raise ValueError('execution policy changed')
                row=dict(index=index,kind=args.mlp_kind if candidate else 'buffered',timings=op.lp.stats(samples),
                  exact=exact,comparisons=numeric,contexts=len(session.states),dispatches=66)
            session=None; gc.collect()
            if not idle(): raise ValueError('profile context cleanup')
            row['idle_after_close']=True; results.append(row)
            print('PROFILE_COMPARE_ARM '+json.dumps(row),flush=True)
    cpu_after=cpu_samples()
    if calls or not idle(): raise ValueError('CPU neural/oracle or final cleanup')
    for filename,sha in manifest.items():
        if p.attention.sha(p.ROOT/filename)!=sha: raise ValueError('comparison source changed during run')
    for name,sha in builds.items():
        if p.attention.sha(getattr(args,name)/'build.json')!=sha: raise ValueError('comparison build changed during run')
    print('PROFILE_COMPARE_RESULT '+json.dumps(dict(arms=results,cpu_before=cpu_before,cpu_after=cpu_after,
      cpu_neural_oracle_calls=len(calls),device_idle_while_process_alive=True,
      scope='fixture performance/development comparison; original full frozen gates remain required')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','fast-folder','items'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mlp-kind',choices=('polynomial','lookup','small-query'),required=True)
    ap.add_argument('--reps',type=int,default=7)
    run(ap.parse_args())
