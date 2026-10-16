#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preregister complete split/unsplit costs and diagnostic MLP phase wall times."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import time

import laya_small_lookup_phase_full as candidate
c=candidate.c
full=candidate.full


def phase_totals(stats):
    expected=Counter(f'mlp_{i}_m128_{phase}' for i in range(28) for phase in candidate.phases.PHASES)
    expected.update(f'attention_{i}_m128' for i in range(28))
    expected.update(['stem','carry','head_pre','head_attention','head_attention','head_post',
        'second_pre','second_post','scorer','action'])
    if stats['dispatches']!=150 or Counter(row['name'] for row in stats['stages'])!=expected:
        raise ValueError('incomplete/duplicate physical phase profile')
    totals={phase:0. for phase in candidate.phases.PHASES};elapsed=0.
    for row in stats['stages']:
        value=row['ms']
        if not math.isfinite(value) or value<0:raise ValueError('invalid phase stage time')
        elapsed+=value
        if row['name'].startswith('mlp_'):totals[row['name'].rsplit('_',1)[-1]]+=value
    if elapsed>stats['dispatch_ms']+0.001:raise ValueError('phase stage time exceeds dispatch interval')
    return totals


def full_gate(path):
    lines=path.read_text().splitlines()
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(tag+' ')]
    verdict=get('SMALL_PHASE_FULL_RESULT');end=get('RESEARCH_PROCESS_RESULT');checks=get('SMALL_PHASE_FULL_CHECK')
    if (len(verdict)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        verdict[0]['cases']!=25 or len(checks)!=25 or verdict[0]['unsplit_controls']!=3 or
        not verdict[0]['all_output_bits_exact'] or verdict[0]['cpu_neural_oracle_calls'] or
        not verdict[0]['idle_after_close'] or not verdict[0]['registered_files_unchanged'] or
        not all(all(row['exact'].values()) and row['contexts']==5 and
            row['dispatches']==(150 if row['valid']<=128 else 66) for row in checks)):
        raise ValueError('full phase parity prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,parent,folder,files,proof=candidate.identity()
    evidence=c.ROOT/'results/laya/small_lookup_phase_full_v1_desktop2_20261016.log'
    files=dict(files);files['full_phase_parity']=full_gate(evidence);files['phase_compare_source']=c.sha(Path(__file__))
    return paths,bundle,fixtures,parent,folder,files,proof


def protocol():
    return dict(version=1,order=[['unsplit','split'],['split','unsplit'],['unsplit','split']],warmups=2,reps=5,
        scope='same91-token/M128 complete request; three fresh serialized sessions; two arm profiles within each session; setup excluded',
        policy=dict(contexts=5,unsplit_dispatches=66,split_dispatches=150,cpu_neural_oracle_calls=0),
        phase_profile='one additional profiled split request after both arm profiles in each session; retain all150 stages; require28 of each MLP phase',
        interpretation='phase host wall times include driver, scheduling, DMA and arithmetic; extra84 dispatches perturb the split path; not pure core times or new serving speedup',
        disclosure='unlabelled full development cases and previous whole-model timings observed; unchanged original accuracy gates remain binding')


def register(path):
    _,_,fixtures,_,_,files,proof=identity();pre=dict(files=files,proof=proof,fixture=fixtures[0],**protocol())
    with path.open('x',encoding='utf-8',newline='\n') as output:output.write(json.dumps(pre,indent=2)+'\n')
    c.emit('SMALL_PHASE_COMPARE_REGISTER',dict(sha256=c.sha(path),registration_id=c.na.digest(pre),files=len(files)))


def benchmark(path):
    pre=json.loads(path.read_text());c.na.committed_registration(path,pre)
    paths,bundle,fixtures,parent,folder,files,proof=identity()
    if pre['files']!=files or pre['proof']!=proof or pre['fixture']!=fixtures[0] or any(pre[k]!=v for k,v in protocol().items()):
        raise ValueError('phase timing registration changed')
    rows=[];profiles=[];calls=[];golden=None
    c.emit('SMALL_PHASE_COMPARE_START',dict(prereg_sha256=c.sha(path),scope=pre['scope']))
    with full.guards(calls):
        for index,order in enumerate(pre['order']):
            if not c.hybrid.idle():raise ValueError('device occupied before phase comparison')
            with candidate.Session(paths,bundle,fixtures,parent,folder) as session:
                for arm in order:
                    split=arm=='split'
                    for _ in range(pre['warmups']):output=session.run(**full.feed(fixtures[0]),split=split)
                    if golden is None:golden={key:value.copy() for key,value in output.items()}
                    samples=[]
                    for _ in range(pre['reps']):
                        start=time.perf_counter();output=session.run(**full.feed(fixtures[0]),split=split)
                        samples.append((time.perf_counter()-start)*1000)
                        if not all(full.exact(output,golden).values()):raise ValueError('phase timed output changed')
                        if session.last_stats['dispatches']!=(150 if split else 66) or len(session.states)!=5 or calls:
                            raise ValueError('phase timed execution policy')
                    row=dict(round=index,arm=arm,**c.stats(samples));rows.append(row);c.emit('SMALL_PHASE_COMPARE_PROFILE',row)
                output=session.run(**full.feed(fixtures[0]),split=True,profile=True)
                if not all(full.exact(output,golden).values()):raise ValueError('phase profiled output changed')
                totals=phase_totals(session.last_stats);profiles.append(totals)
                c.emit('SMALL_PHASE_PROFILE',dict(round=index,phase_totals_ms=totals,stats=session.last_stats))
            session=None;gc.collect()
            if not c.hybrid.idle():raise ValueError('phase comparison cleanup')
            c.emit('SMALL_PHASE_COMPARE_CLOSED',dict(round=index,idle_after_close=True))
    if calls or identity()[5]!=files:raise ValueError('phase comparison source/CPU gate')
    c.emit('SMALL_PHASE_COMPARE_RESULT',dict(
        arms={arm:c.stats([sample for row in rows if row['arm']==arm for sample in row['samples_ms']]) for arm in ('unsplit','split')},
        encoder_phases_ms={phase:c.stats([row[phase] for row in profiles]) for phase in candidate.phases.PHASES},
        cpu_neural_oracle_calls=len(calls),registered_files_unchanged=True,idle_after_close=c.hybrid.idle(),
        interpretation=pre['interpretation']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','benchmark'))
    parser.add_argument('--prereg',type=Path,required=True);args=parser.parse_args()
    (register if args.command=='register' else benchmark)(args.prereg)
