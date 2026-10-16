#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete-model short attention qualification and same-sitting CPU comparison."""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import bucket_attention as ba

n=ba.n; p=ba.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_accuracy as la


def stats(values):
    return dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)


def run(base,model,mlp_folder,folder,items_path,reps,quick):
    if not 3<=reps<=20: raise ValueError('bounded timing repetitions')
    prereg=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(items_path)!=prereg['items']['file_sha256']: raise ValueError('frozen items changed')
    items=[json.loads(row) for row in items_path.read_text().splitlines()][:3]
    if any(item['gold']!=-1 or item['task']!='fixture' for item in items): raise ValueError('only unlabelled fixtures')
    bundle=n.Bundle(base,model); original=model.parent/'laya.onnx'
    for name,digest in prereg['reference']['files'].items():
        if p.attention.sha(original.parent/name)!=digest: raise ValueError('CPU model changed')
    print('SHORT_ATTENTION_RUN '+json.dumps(dict(base_identity=bundle.identity,attention_build_sha256=p.attention.sha(folder/'build.json'),
      mlp_build_sha256=p.attention.sha(mlp_folder/'build.json'),source_sha256=p.attention.sha(Path(__file__)),
      ort=ort.__version__,cpu_threads=8,scope='unlabelled fixtures and synthetic inputs; no native labelled scores')),flush=True)
    item=items[0]; feed=la.feed(item['ids'],item['markers'],item['qtype']); cpu=la.lb.session(original,'cpu',8)
    for _ in range(2): cpu.run(None,feed)
    elapsed=[]
    for _ in range(reps):
        started=time.perf_counter(); cpu_values=cpu.run(None,feed); elapsed.append((time.perf_counter()-started)*1000)
    cpu_stats=stats(elapsed); cpu_names=[out.name for out in cpu.get_outputs()]; cpu_outputs=dict(zip(cpu_names,cpu_values))
    cpu=None; gc.collect()
    calls=[]
    def forbidden(*args,**kwargs): calls.append(True); raise AssertionError('CPU neural/oracle call')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        with ba.Session(bundle,mlp_folder,folder) as session:
            cases=[(f'fixture{i}',item['ids'],item['markers'],item['qtype']) for i,item in enumerate(items)]
            if not quick:
                for length in (512,1,511,31,385,32,384,33,383,63,257,64,256,65,255,91,193,127,129,128,512,91):
                    ids=np.full(length,50283,np.int64); keep=min(length,len(items[0]['ids'])); ids[:keep]=items[0]['ids'][:keep]
                    cases.append((f'synthetic{length}',ids,[0,length-1],length%3))
            # The very first dispatch uses short attention, before any baseline filled padding.
            for key,ids,markers,qtype in cases:
                short=session.run(ids,markers,qtype)
                original=session.run(ids,markers,qtype,baseline=True)
                exact={name:bool(np.array_equal(original[name].view(np.uint32),short[name].view(np.uint32))) for name in original}
                comparisons={name:p.attention.numeric(short[name],original[name]) for name in original}
                print('SHORT_ATTENTION_CHECK '+json.dumps(dict(case=key,valid=len(ids),m=ba.bm.bucket(len(ids)),exact=exact,comparisons=comparisons)),flush=True)
                if not all(exact.values()): raise ValueError('short attention changed complete outputs: '+key)
            item=items[0]; timings={key:[] for key in ('baseline','mlp_only','short_encoder')}; outputs={}
            for rep in range(reps):
                order=list(timings)
                if rep%2: order.reverse()
                for key in order:
                    started=time.perf_counter()
                    result=session.run(item['ids'],item['markers'],item['qtype'],baseline=key=='baseline',mlp_only=key=='mlp_only')
                    timings[key].append((time.perf_counter()-started)*1000)
                    outputs[key]=result
            session.run(item['ids'],item['markers'],item['qtype'],profile=True)
            groups={}
            for stage in session.last_stats['stages']:
                name='encoder_mlp' if stage['name'].startswith('mlp_') else 'encoder_attention_projection_norm' if stage['name'].startswith('attention_') else stage['name']
                groups[name]=groups.get(name,0)+stage['ms']
            timings={key:stats(values) for key,values in timings.items()}; timings['cpu_actual_length']=cpu_stats
            output_comparison={key:p.attention.numeric(value,cpu_outputs[key]) for key,value in outputs['short_encoder'].items()}
            print('SHORT_ATTENTION_RESULT '+json.dumps(dict(cases=len(cases),exact=True,cpu_neural_oracle_calls=len(calls),
              timings=timings,speedup=timings['baseline']['median_ms']/timings['short_encoder']['median_ms'],
              npu_over_actual_cpu=timings['short_encoder']['median_ms']/cpu_stats['median_ms'],
              profile_groups_ms=groups,output_comparison_to_cpu=output_comparison,contexts=len(session.states),dispatches_per_frame=66,
              scope='complete model fixture timings; setup excluded; CPU FP32 vs NPU BF16 matrix path; labelled accuracy unverified')),flush=True)
    from laya_hybrid import idle
    print('SHORT_ATTENTION_IDLE '+str(idle()),flush=True)
    if calls or not idle(): raise ValueError('oracle/cleanup gate')
    if not all(value['ok'] for value in output_comparison.values()): raise ValueError('CPU fixture numeric comparison')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--mlp-folder',type=Path,required=True); ap.add_argument('--folder',type=Path,required=True)
    ap.add_argument('--items',type=Path,required=True); ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true')
    args=ap.parse_args(); run(args.base,args.model,args.mlp_folder,args.folder,args.items,args.reps,args.quick)
