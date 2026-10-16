#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repeated complete serving checks with CPU neural/oracle calls forbidden."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import native
import onnxruntime as ort

p=native.p


def run(base,model,items_path,reference,qualified):
    prereg=json.loads((p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json').read_text())
    if p.attention.sha(items_path)!=prereg['items']['file_sha256'] or p.attention.sha(reference)!=prereg['reference']['logits_sha256']: raise ValueError('changed frozen reference/items')
    items=[json.loads(line) for line in items_path.read_text().splitlines()][:3]
    if any(row['task']!='fixture' or row['gold']!=-1 for row in items): raise ValueError('only unlabelled fixtures allowed in this probe')
    frozen=np.load(reference)['logits']; original=np.load(qualified)
    bundle=native.Bundle(base,model); forbidden_calls=[]
    def forbidden(*args,**kwargs): forbidden_calls.append('called'); raise AssertionError('CPU neural/oracle call in native serving')
    functions=[(ort.InferenceSession,'run'),(native.cr,'references'),(native.embedded.mask_reference,'extract'),
      (native.p.norm,'oracle'),(native.p.attention,'oracle'),(native.action,'reference'),(native.action,'verify'),
      (native.action.s,'reference'),(native.action.s,'verify'),(native.action.s.second,'reference'),
      (native.action.s.second.post,'reference')]
    previous=None; failures=[]
    print('NATIVE_PROBE_RUN '+json.dumps(dict(bundle=bundle.identity,qualified_output_sha256=p.attention.sha(qualified),
      source_sha256=p.attention.sha(Path(__file__)),scope='three unlabelled fixtures; not labelled accuracy; timing ineligible')),flush=True)
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        with native.NativeSession(bundle) as session:
            for index,full in ((0,True),(0,False),(1,False),(2,False),(0,False)):
                item=items[index]; result=session.run(item['ids'],item['markers'],item['qtype'],full_readback=full)
                logits=result['logits'].ravel(); probabilities=result['act_probs'].ravel()
                comparison=p.attention.numeric(logits,frozen[index,:item['k']])
                exact=True
                if index==0:
                    exact=bool(np.array_equal(logits.view(np.uint32),original['logits'].view(np.uint32)) and np.array_equal(probabilities.view(np.uint32),original['act_probs'].view(np.uint32)))
                    if previous is not None: exact &= all(np.array_equal(result[key],previous[key]) for key in result)
                    previous=result
                if not exact: raise ValueError('serving differs from qualified fixture or repeated/partial readback')
                if not comparison['ok']: failures.append(dict(fixture=index,comparison=comparison))
                print('NATIVE_PROBE_CHECK '+json.dumps(dict(fixture=index,qtype=item['qtype'],valid=len(item['ids']),full_readback=full,
                  logits=logits.tolist(),act_probs=probabilities.tolist(),qualified_and_repeat_exact=exact,
                  frozen_logits=comparison,stats=session.last_stats)),flush=True)
            print('NATIVE_PROBE_RESULT '+json.dumps(dict(frames=session.frame_count,cpu_neural_oracle_calls=len(forbidden_calls),
              contexts=3,dispatches_per_frame=66,partial_output_sync_qualified=True,initial_residual='normalized embedding Y',
              fixture_numeric_failures=failures,scope='serving qualification; frozen labelled criteria and eligible speed unverified')),flush=True)
    from laya_hybrid import idle
    print('NATIVE_IDLE '+str(idle()),flush=True)
    if forbidden_calls or not idle(): raise ValueError('native serving cleanup/oracle gate')
    if failures: raise ValueError('retained fixture numeric comparisons failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--items',type=Path,required=True); ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--qualified',type=Path,required=True)
    args=ap.parse_args(); run(args.base,args.model,args.items,args.reference,args.qualified)
