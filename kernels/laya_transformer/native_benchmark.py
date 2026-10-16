#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same-sitting CPU/native baseline and native wall-stage profile, fixture only."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import native
p=native.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_accuracy as la


def summary(values):
    return dict(median_ms=float(np.median(values)),min_ms=float(min(values)),max_ms=float(max(values)),samples_ms=values)


def run(base,model,items_path,reps):
    if not 3<=reps<=30: raise ValueError('bounded benchmark reps')
    prereg=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(items_path)!=prereg['items']['file_sha256']: raise ValueError('changed fixture items')
    item=json.loads(items_path.read_text().splitlines()[0])
    if item['task']!='fixture' or item['gold']!=-1: raise ValueError('benchmark only unlabelled fixture')
    bundle=native.Bundle(base,model); original=Path(model).parent/'laya.onnx'
    for name,digest in prereg['reference']['files'].items():
        if p.attention.sha(original.parent/name)!=digest: raise ValueError('original CPU model changed')
    print('NATIVE_BENCHMARK_RUN '+json.dumps(dict(bundle=bundle.identity,cpu_model=prereg['reference']['files'],
      valid=len(item['ids']),options=item['k'],reps=reps,warmup=2,cpu_threads=8,ort=la.lb.versions()['onnxruntime'] if hasattr(la.lb,'versions') else __import__('onnxruntime').__version__,
      source_sha256=p.attention.sha(Path(__file__)),scope='same-sitting fixture; setup excluded; CPU EP versus complete native NPU')),flush=True)
    results={}; outputs={}
    cpu=la.lb.session(original,'cpu',8)
    for label,padded in [('cpu_actual_length',False),('cpu_padded_512',True)]:
        feed=la.feed(item['ids'],item['markers'],item['qtype'])
        if padded:
            ids=np.full((1,512),50283,np.int64); ids[0,:len(item['ids'])]=item['ids']; feed['input_ids']=ids
            mask=np.zeros((1,512),np.int64); mask[0,:len(item['ids'])]=1; feed['attention_mask']=mask
        for _ in range(2): cpu.run(None,feed)
        elapsed=[]
        for _ in range(reps):
            start=time.perf_counter(); out=cpu.run(None,feed); elapsed.append((time.perf_counter()-start)*1000)
        outputs[label]=dict(zip([x.name for x in cpu.get_outputs()],out)); results[label]=summary(elapsed)
        print('NATIVE_BENCHMARK_CONFIG '+json.dumps(dict(config=label,**results[label])),flush=True)
    cpu=None; gc.collect()
    with native.NativeSession(bundle) as session:
        for _ in range(2): session.run(item['ids'],item['markers'],item['qtype'])
        elapsed=[]; breakdown=[]
        for _ in range(reps):
            start=time.perf_counter(); out=session.run(item['ids'],item['markers'],item['qtype']); elapsed.append((time.perf_counter()-start)*1000)
            breakdown.append(session.last_stats)
        outputs['npu_padded_512']=out; results['npu_padded_512']=summary(elapsed)
        print('NATIVE_BENCHMARK_CONFIG '+json.dumps(dict(config='npu_padded_512',**results['npu_padded_512'],breakdown=breakdown)),flush=True)
        session.run(item['ids'],item['markers'],item['qtype'],profile=True)
        groups={}
        for stage in session.last_stats['stages']:
            name=stage['name']
            group='encoder_attention_projection_norm' if name.startswith('attention_') else 'encoder_mlp' if name.startswith('mlp_') else name
            groups[group]=groups.get(group,0)+stage['ms']
        print('NATIVE_STAGE_PROFILE '+json.dumps(dict(groups_ms=groups,**session.last_stats,scope='wall time includes driver/wait/context transitions; not pure core time')),flush=True)
    from laya_hybrid import idle
    print('NATIVE_BENCHMARK_IDLE '+str(idle()),flush=True)
    comparisons={label:{key:p.attention.numeric(value.ravel(),outputs['cpu_actual_length'][key].ravel()) for key,value in output.items()} for label,output in outputs.items()}
    print('NATIVE_BENCHMARK_RESULT '+json.dumps(dict(configs=results,outputs_comparison_to_cpu_actual=comparisons,
      ratio_npu_over_cpu=results['npu_padded_512']['median_ms']/results['cpu_actual_length']['median_ms'],
      whole_model_outputs=['logits','act_probs'],scope='fixture only; frozen labelled accuracy remains unverified')),flush=True)
    if not idle(): raise ValueError('NPU not idle after native benchmark')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--items',type=Path,required=True); ap.add_argument('--reps',type=int,default=5)
    args=ap.parse_args(); run(args.base,args.model,args.items,args.reps)
