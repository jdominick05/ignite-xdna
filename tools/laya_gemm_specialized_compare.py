#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preregister and time qualified packing against the specialized MLP candidate."""
import argparse
import gc
import json
from pathlib import Path
import time

import laya_gemm_specialized_full as candidate
c=candidate.c;full=candidate.full


def gates(path):
    lines=path.read_text().splitlines()
    rows=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('WINDOW_FULL_RESULT ')]
    terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
    if (len(rows)!=1 or rows[0]['cases']!=25 or not rows[0]['all_output_bits_exact'] or
        rows[0]['cpu_neural_oracle_calls'] or not rows[0]['idle_after_close'] or not rows[0]['registered_files_unchanged'] or
        len(terminal)!=1 or terminal[0]['exit_code']!=0 or terminal[0]['stop'] is not None):
        raise ValueError('specialized complete-model development qualification incomplete')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,folder,files,proof=candidate.identity()
    evidence=c.ROOT/'results/laya/gemm_specialized_mlp_v1_full_desktop2_20261016.log'
    files['specialized_full_qualification']=gates(evidence)
    files['specialized_source/tools/laya_gemm_specialized_compare.py']=c.sha(Path(__file__))
    return paths,bundle,fixtures,folder,files,proof


def protocol():
    return dict(order=[['packing','specialized'],['specialized','packing'],['packing','specialized']],
        warmups=2,reps=5,policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        scope='same91-token/M128 complete inference; fresh serialized sessions; setup excluded; no CPU/hybrid ratio',
        disclosure='development parity only for specialized candidate; original frozen gates still required before adoption')


def register(path):
    _,_,fixtures,_,files,proof=identity()
    pre=dict(version=1,files=files,proof=proof,fixture=fixtures[0],**protocol())
    with path.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(pre,indent=2)+'\n')
    c.emit('GEMM_COMPARE_REGISTER',dict(sha256=c.sha(path),identity=c.na.digest(pre),files=len(files)))


def benchmark(path):
    pre=json.loads(path.read_text());c.na.committed_registration(path,pre)
    paths,bundle,fixtures,folder,files,proof=identity()
    if pre['files']!=files or pre['proof']!=proof or pre['fixture']!=fixtures[0] or any(pre[k]!=v for k,v in protocol().items()):
        raise ValueError('specialized comparison preregistration changed')
    rows=[];golden=None;calls=[]
    c.emit('GEMM_COMPARE_START',dict(prereg_sha256=c.sha(path),scope=pre['scope']))
    with full.guards(calls):
        for round_index,order in enumerate(pre['order']):
            for arm in order:
                if not c.hybrid.idle():raise ValueError('device occupied before profile')
                with candidate.make(paths,bundle,fixtures,folder,arm=='specialized') as session:
                    for _ in range(pre['warmups']):output=session.run(**full.feed(fixtures[0]))
                    if golden is None:golden={k:v.copy() for k,v in output.items()}
                    samples=[]
                    for _ in range(pre['reps']):
                        start=time.perf_counter();output=session.run(**full.feed(fixtures[0]));samples.append((time.perf_counter()-start)*1000)
                        if not all(full.exact(output,golden).values()):raise ValueError('timed output changed')
                    if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:
                        raise ValueError('timed execution policy changed')
                session=None;gc.collect()
                if not c.hybrid.idle():raise ValueError('profile context cleanup')
                row=dict(round=round_index,arm=arm,**c.stats(samples),idle_after_close=True)
                rows.append(row);c.emit('GEMM_COMPARE_PROFILE',row)
    if calls or identity()[4]!=files:raise ValueError('comparison source/CPU gate')
    c.emit('GEMM_COMPARE_RESULT',dict(arms={arm:c.stats([value for row in rows if row['arm']==arm for value in row['samples_ms']])
        for arm in ('packing','specialized')},cpu_neural_oracle_calls=len(calls),idle_after_close=c.hybrid.idle(),
        registered_files_unchanged=True,scope='incremental native comparison; frozen candidate accuracy required before adoption'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('register','benchmark'));ap.add_argument('--prereg',type=Path,required=True)
    args=ap.parse_args();(register if args.command=='register' else benchmark)(args.prereg)
