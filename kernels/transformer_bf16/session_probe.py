"""Check cached XRT packets against CPU fixtures and A/B/C/A replay.

Checks only; includes input preservation, output guards, cache eviction, rejected
use after close, and deterministic replay after different input data.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16 import Bf16Ops
from kernels.transformer_bf16.common import bf16_values,save,sha
from kernels.transformer_bf16.persistent import PacketSession
from kernels.transformer_bf16.probe import GATES


def run(folder,cache,output,precise_projection,single_image=False,check_drift=False):
    if output.exists():raise FileExistsError('Refusing to replace session evidence')
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['gates']!=GATES or len(manifest['cases'])<3:raise ValueError('Unexpected reference criteria or coverage')
    cases=[]
    for case in manifest['cases']:
        path=folder/case['file']
        if sha(path)!=case['sha256']:raise ValueError('Reference fixture changed')
        with np.load(path,allow_pickle=False) as archive:
            cases.append((case['seed'],{key:archive[key].copy() for key in archive.files}))
    operations={
        'projection':lambda ops,a:ops.projection(a['projection.x'],a['projection.w']),
        'norm':lambda ops,a:ops.norm(a['norm.x'],a['norm.w']),
        'rope':lambda ops,a:ops.rope(a['rope.q'],a['rope.cosine'],a['rope.sine']),
        'softmax':lambda ops,a:ops.softmax(a['softmax.x']),
        'geglu':lambda ops,a:ops.geglu(a['geglu.gate'],a['geglu.up'],a['table']),
        'residual':lambda ops,a:ops.residual(a['residual.x'],a['residual.r'],np.uint16(0x3f60)),
        'topk':lambda ops,a:ops.topk(a['topk.x'],32),
    }
    rows=[]
    session=PacketSession(max_contexts=2,share_program=single_image)
    with session:
        ops=Bf16Ops(cache,precise_projection=precise_projection,transport=session,single_image=single_image)
        for name,operation in operations.items():
            first=None
            for index,(seed,a) in enumerate(cases+[cases[0]]):
                value=operation(ops,a);reference=a[name+'.expected']
                if name=='topk':metric=dict(passed=bool(np.array_equal(value,reference)))
                else:
                    x,y=bf16_values(value).astype(np.float64),bf16_values(reference).astype(np.float64)
                    error=np.abs(x-y);relative=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
                    metric=dict(passed=bool(np.isfinite(x).all() and
                        np.all(error<=GATES['atol']+GATES['rtol']*np.abs(y)) and relative<=GATES['relative_l2']),
                        relative_l2=relative,max_abs=float(error.max()))
                if index==0:first=value.copy()
                replay=index==len(cases)
                if replay:metric['replay_bitwise_equal']=bool(np.array_equal(first,value))
                metric['passed']=metric['passed'] and (not replay or metric['replay_bitwise_equal'])
                row=dict(operation=name,seed=seed,replay=replay,**metric);rows.append(row)
                print(json.dumps(row),flush=True)
    # A closed session must reject dispatch before touching device state.
    try:session.execute(cache,np.empty((0,0),np.uint32))
    except RuntimeError:closed_rejected=True
    else:closed_rejected=False
    expected_contexts=(session.contexts_created<=2 and session.evictions==0) if single_image else (
        session.contexts_created==len(operations) and session.evictions==len(operations)-2)
    cache_ok=(session.cache_hits>=len(operations)*len(cases) and expected_contexts and not session.images and not session.active)
    drift=None
    if check_drift:
        # Corrupt only this probe's cached manifest after a completed dispatch.
        # The next exit must release contexts and reject the changed identity.
        drift_session=PacketSession(share_program=single_image)
        altered=None;original=None;detected=False
        try:
            try:
                with drift_session:
                    drift_ops=Bf16Ops(cache,precise_projection=precise_projection,transport=drift_session,single_image=single_image)
                    operations['norm'](drift_ops,cases[0][1])
                    altered=Path(next(iter(drift_session._verified)))/'build.json'
                    original=altered.read_bytes();altered.write_bytes(original+b'\n')
            except ValueError as error:
                if 'Admitted packet image changed' not in str(error):raise
                detected=True
        finally:
            if altered is not None and original is not None:altered.write_bytes(original)
        drift=dict(detected=detected,contexts_released=not drift_session.active and not drift_session.images,
                   restored=altered is not None and altered.read_bytes()==original)
        if not all(drift.values()):raise ValueError('Artifact drift check failed')
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
             for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cc')}
    report=dict(passed=all(row['passed'] for row in rows) and cache_ok and closed_rejected,cases=len(rows),
        gates=GATES,sources=sources,fixture_manifest_sha256=sha(folder/'manifest.json'),
        precise_projection=precise_projection,single_image=single_image,cache_ok=cache_ok,closed_rejected=closed_rejected,
        drift=drift,
        cache_hits=session.cache_hits,contexts_created=session.contexts_created,evictions=session.evictions,
        pre_witness=session.pre_witness,post_witness=session.post_witness,results=rows,dispatches=ops.dispatches)
    save(output,report)
    print(json.dumps(dict(passed=report['passed'],cases=len(rows),cache_hits=session.cache_hits)),flush=True)
    if not report['passed']:raise ValueError('Cached packet transport qualification failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--precise-projection',action='store_true')
    parser.add_argument('--single-image',action='store_true')
    parser.add_argument('--check-drift',action='store_true')
    args=parser.parse_args();run(args.fixtures,args.cache,args.out,args.precise_projection,args.single_image,args.check_drift)
