#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Alternate complete baseline/short-MLP calls in the same resident contexts."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import bucket_mlp as bm

n=bm.native; p=bm.p


def run(base,model,folder,items_path,reps):
    prereg=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(items_path)!=prereg['items']['file_sha256']: raise ValueError('frozen items changed')
    items=[json.loads(row) for row in items_path.read_text().splitlines()][:3]
    if any(item['gold']!=-1 for item in items): raise ValueError('unlabelled fixtures only')
    bundle=n.Bundle(base,model); calls=[]
    def forbidden(*args,**kwargs): calls.append(True); raise AssertionError('CPU neural/oracle operation')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    print('BUCKET_PROBE_RUN '+json.dumps(dict(base_identity=bundle.identity,bucket_build_sha256=p.attention.sha(folder/'build.json'),
      source_sha256=p.attention.sha(Path(__file__)),scope='unlabelled fixture and synthetic boundary qualification; no labelled scores')),flush=True)
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        with bm.Session(bundle,folder) as session:
            cases=[(f'fixture{i}',item['ids'],item['markers'],item['qtype']) for i,item in enumerate(items)]
            # Deliberately alternate large/small extents to expose stale rows.
            for length in (512,1,511,31,385,32,384,33,383,63,257,64,256,65,255,91,193,127,129,128,512,91):
                ids=np.full(length,50283,np.int64); keep=min(length,len(items[0]['ids'])); ids[:keep]=items[0]['ids'][:keep]
                cases.append((f'synthetic{length}',ids,[0,length-1],length%3))
            for key,ids,markers,qtype in cases:
                original=session.run(ids,markers,qtype,baseline=True)
                short=session.run(ids,markers,qtype)
                exact={name:bool(np.array_equal(original[name].view(np.uint32),short[name].view(np.uint32))) for name in original}
                print('BUCKET_PROBE_CHECK '+json.dumps(dict(case=key,valid=len(ids),m=bm.bucket(len(ids)),exact=exact)),flush=True)
                if not all(exact.values()): raise ValueError('short MLP changed complete outputs: '+key)
            item=items[0]; samples={'baseline':[],'short_mlp':[]}
            # Paired order alternates to reduce ordering bias. Both paths already warm.
            for rep in range(reps):
                for label in (('baseline','short_mlp') if rep%2==0 else ('short_mlp','baseline')):
                    started=time.perf_counter()
                    session.run(item['ids'],item['markers'],item['qtype'],baseline=label=='baseline')
                    samples[label].append((time.perf_counter()-started)*1000)
            session.run(item['ids'],item['markers'],item['qtype'],profile=True)
            groups={}
            for stage in session.last_stats['stages']:
                name='encoder_mlp' if stage['name'].startswith('mlp_') else 'other'
                groups[name]=groups.get(name,0)+stage['ms']
            stats={key:dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values) for key,values in samples.items()}
            print('BUCKET_PROBE_RESULT '+json.dumps(dict(cases=len(cases),exact=True,cpu_neural_oracle_calls=len(calls),timings=stats,
              speedup=stats['baseline']['median_ms']/stats['short_mlp']['median_ms'],profile_groups_ms=groups,
              frame_count=session.frame_count,dispatches_per_frame=66,contexts=3,scope='complete model fixture only; labelled accuracy unverified')),flush=True)
    from laya_hybrid import idle
    print('BUCKET_PROBE_IDLE '+str(idle()),flush=True)
    if calls or not idle(): raise ValueError('CPU oracle/cleanup gate')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--folder',type=Path,required=True); ap.add_argument('--items',type=Path,required=True); ap.add_argument('--reps',type=int,default=5)
    args=ap.parse_args(); run(args.base,args.model,args.folder,args.items,args.reps)
