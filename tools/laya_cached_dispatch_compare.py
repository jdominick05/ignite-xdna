#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One frozen-qualified, same-sitting end-to-end Wi-cache adoption comparison."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import laya_cached_dispatch_full as candidate
c=candidate.c
full=candidate.full
ARMS=('control','cache')


def protocol():
    return dict(order=[['control','cache'],['cache','control'],['control','cache']],warmups=2,reps=5,
        expected_saved_ms=20,kill_saved_ms=10,require_every_round_win=True,
        policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        metric='external wall time around complete Session.run; setup and output validation excluded',
        scope='same 91-token/M128 request, fresh serialized sessions; incremental native control comparison; no CPU/hybrid ratio')


def frozen_gate(prereg,score,report):
    pre=json.loads(prereg.read_text(encoding='utf-8'));c.na.committed_registration(prereg,pre)
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in report.read_text(encoding='utf-8').splitlines() if line.startswith(tag+' ')]
    verdict=get('NATIVE_ACCURACY_VERDICT');exact=get('CACHED_ACCURACY_EQUIVALENCE');end=get('RESEARCH_PROCESS_RESULT')
    if (len(verdict)!=1 or len(exact)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        not verdict[0]['passed'] or verdict[0]['criteria']!={f'C{i}':True for i in range(6)} or verdict[0]['labelled']!=2400 or
        verdict[0]['registration_id']!=c.na.digest(pre) or verdict[0]['output_sha256']!=c.sha(score) or
        exact[0]['exact']!={'logits':True,'act_probs':True}):
        raise ValueError('complete original frozen accuracy and equivalence must pass before timing')
    return dict(prereg_sha256=c.sha(prereg),output_sha256=c.sha(score),report_sha256=c.sha(report),registration_id=c.na.digest(pre))


def identity():
    paths,bundle,fixtures,parent,folder,files,proof=candidate.identity()
    gate=frozen_gate(c.ROOT/'results/laya/cached_dispatch_accuracy_v1_prereg_desktop2_20261017.json',
        c.ROOT/'scratch/laya/native_accuracy/native_cached_wi_dispatch_bf16_v1.npz',
        c.ROOT/'results/laya/cached_dispatch_accuracy_v1_report_desktop2_20261017.log')
    control=candidate.layer.previous.reference.comparison.identity()
    files=dict(files);files.update({'qualified_control/'+key:value for key,value in control[4].items()})
    files['comparison_source']=c.sha(Path(__file__))
    return paths,bundle,fixtures,parent,folder,files,dict(development=proof,frozen=gate)


def decision(rows):
    expected={(index,arm) for index in range(3) for arm in ARMS}
    if len(rows)!=6 or {(row['round'],row['arm']) for row in rows}!=expected:raise ValueError('comparison coverage')
    if any(len(row['samples_ms'])!=5 or any(not math.isfinite(value) or value<=0 for value in row['samples_ms']) for row in rows):
        raise ValueError('comparison samples')
    arms={arm:c.stats([value for row in rows if row['arm']==arm for value in row['samples_ms']]) for arm in ARMS}
    rounds=[]
    for index in range(3):
        medians={row['arm']:c.stats(row['samples_ms'])['median_ms'] for row in rows if row['round']==index}
        rounds.append(medians['control']-medians['cache'])
    saved=arms['control']['median_ms']-arms['cache']['median_ms']
    win=saved>=protocol()['kill_saved_ms'] and all(value>0 for value in rounds)
    return dict(arms=arms,pooled_saved_ms=saved,round_median_saved_ms=rounds,
        adoption_from_samples='adopt' if win else 'kill',requires_timing_eligible_wrapper=True)


def timing_verdict(path):
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in path.read_text(encoding='utf-8').splitlines() if line.startswith(tag+' ')]
    rows=get('CACHED_COMPARE_PROFILE');results=get('CACHED_COMPARE_RESULT');end=get('RESEARCH_PROCESS_RESULT')
    if (len(results)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        not end[0]['timing_eligible'] or results[0]['cpu_neural_oracle_calls'] or
        not results[0]['idle_after_close'] or not results[0]['registered_files_unchanged'] or
        not all(row['idle_after_close'] for row in rows)):
        raise ValueError('comparison is not an eligible complete same-sitting result')
    verdict=decision(rows)
    if any(results[0][key]!=value for key,value in verdict.items()):raise ValueError('comparison summary differs from samples')
    return verdict


def register(path):
    _,_,fixtures,_,_,files,proof=identity()
    if len(fixtures[0]['ids'])!=91:raise ValueError('comparison request changed')
    pre=dict(version=1,files=files,proof=proof,fixture=fixtures[0],**protocol())
    with path.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(pre,indent=2)+'\n')
    c.emit('CACHED_COMPARE_REGISTER',dict(sha256=c.sha(path),identity=c.na.digest(pre),files=len(files)))


def benchmark(path):
    pre=json.loads(path.read_text(encoding='utf-8'));c.na.committed_registration(path,pre)
    paths,bundle,fixtures,parent,folder,files,proof=identity()
    if pre['files']!=files or pre['proof']!=proof or pre['fixture']!=fixtures[0] or any(pre[key]!=value for key,value in protocol().items()):
        raise ValueError('comparison registration changed')
    rows=[];golden=None;calls=[]
    c.emit('CACHED_COMPARE_START',dict(prereg_sha256=c.sha(path),scope=pre['scope'],metric=pre['metric']))
    with full.guards(calls):
        for index,order in enumerate(pre['order']):
            for arm in order:
                if not c.hybrid.idle():raise ValueError('device occupied before cached comparison')
                session=(candidate.Session(paths,bundle,fixtures,parent,folder) if arm=='cache' else
                    candidate.layer.previous.reference.comparison.candidate.make(paths,bundle,fixtures,parent,True))
                with session:
                    for _ in range(pre['warmups']):output=session.run(**full.feed(fixtures[0]))
                    if golden is None:golden={key:value.copy() for key,value in output.items()}
                    samples=[];internal=[]
                    for _ in range(pre['reps']):
                        start=time.perf_counter();output=session.run(**full.feed(fixtures[0]));samples.append((time.perf_counter()-start)*1000)
                        internal.append(dict(session.last_stats))
                        if not all(full.exact(output,golden).values()):raise ValueError('cached timed output changed')
                        if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:raise ValueError('cached timed execution policy')
                session=None;gc.collect()
                if not c.hybrid.idle():raise ValueError('cached comparison cleanup')
                row=dict(round=index,arm=arm,**c.stats(samples),native_internal=internal,idle_after_close=True)
                rows.append(row);c.emit('CACHED_COMPARE_PROFILE',row)
    if calls or identity()[5]!=files:raise ValueError('cached comparison source/CPU gate')
    c.emit('CACHED_COMPARE_RESULT',dict(**decision(rows),cpu_neural_oracle_calls=len(calls),idle_after_close=c.hybrid.idle(),
        registered_files_unchanged=True,scope='M128 incremental adoption only; matched512 speed goal remains separate'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','benchmark'));parser.add_argument('--prereg',type=Path,required=True)
    args=parser.parse_args();(register if args.command=='register' else benchmark)(args.prereg)
