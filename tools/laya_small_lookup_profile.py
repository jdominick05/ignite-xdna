#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preregister host-observed stage timing of the complete small-query model."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import re
import time

import laya_small_lookup_compare as comparison
c=comparison.c
full=comparison.full


def stage_groups(stats):
    """Require every dispatch, including both uses of head_attention."""
    expected=Counter(['stem','carry','head_pre','head_attention','head_attention',
        'head_post','second_pre','second_post','scorer','action'])
    expected.update(f'{family}_{i}' for family in ('attention','mlp') for i in range(28))
    seen=Counter();groups={}
    if stats['dispatches']!=66:raise ValueError('profile dispatch count')
    for row in stats['stages']:
        name=re.sub(r'_m128$','',row['name']);seen[name]+=1
        value=row['ms']
        if not math.isfinite(value) or value<0:raise ValueError('invalid stage duration')
        key='encoder_mlp' if name.startswith('mlp_') else 'encoder_attention_projection_norm' if name.startswith('attention_') else name
        groups[key]=groups.get(key,0)+value
    if seen!=expected:raise ValueError('profile stage coverage')
    elapsed=sum(groups.values())
    if elapsed>stats['dispatch_ms']+0.001:raise ValueError('stage time exceeds enclosing dispatch time')
    return dict(groups_ms=groups,stage_sum_ms=elapsed,
        dispatch_loop_unattributed_ms=stats['dispatch_ms']-elapsed,
        prepare_ms=stats['prepare_ms'],readback_ms=stats['readback_ms'],total_ms=stats['total_ms'])


def identity():
    paths,bundle,fixtures,folder,files,proof=comparison.identity()
    files=dict(files);files['stage_profile_source']=c.sha(Path(__file__))
    return paths,bundle,fixtures,folder,files,proof


def protocol():
    return dict(version=1,sessions=3,warmups=2,
        order=[False,True,True,False,True],
        policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        scope='same91-token/M128 complete candidate request; fresh serialized sessions; setup excluded',
        interpretation='host wall time around each submit and completed wait; includes driver, scheduling and DMA; not core cycles or arithmetic-only time',
        overhead='unprofiled requests bracket profiled requests within each session; retain individual totals; do not infer instrumentation overhead from a cross-sitting baseline',
        adoption='diagnostic only; original frozen qualification remains required before serving adoption')


def register(path):
    _,_,fixtures,_,files,proof=identity()
    pre=dict(files=files,proof=proof,fixture=fixtures[0],**protocol())
    with path.open('x',encoding='utf-8',newline='\n') as output:output.write(json.dumps(pre,indent=2)+'\n')
    c.emit('SMALL_PROFILE_REGISTER',dict(sha256=c.sha(path),registration_id=c.na.digest(pre),files=len(files)))


def benchmark(path):
    pre=json.loads(path.read_text());c.na.committed_registration(path,pre)
    paths,bundle,fixtures,folder,files,proof=identity()
    if pre['files']!=files or pre['proof']!=proof or pre['fixture']!=fixtures[0] or any(pre[k]!=v for k,v in protocol().items()):
        raise ValueError('stage profile registration changed')
    calls=[];rows=[];golden=None
    c.emit('SMALL_PROFILE_START',dict(prereg_sha256=c.sha(path),interpretation=pre['interpretation']))
    with full.guards(calls):
        for index in range(pre['sessions']):
            if not c.hybrid.idle():raise ValueError('device occupied before stage profile')
            with comparison.candidate.make(paths,bundle,fixtures,folder,True) as session:
                for _ in range(pre['warmups']):output=session.run(**full.feed(fixtures[0]))
                if golden is None:golden={key:value.copy() for key,value in output.items()}
                for sample,profile in enumerate(pre['order']):
                    start=time.perf_counter();output=session.run(**full.feed(fixtures[0]),profile=profile)
                    wall_ms=(time.perf_counter()-start)*1000
                    if not all(full.exact(output,golden).values()):raise ValueError('profile output changed')
                    if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:raise ValueError('profile execution policy')
                    row=dict(session=index,sample=sample,profile=profile,wall_ms=wall_ms,stats=session.last_stats)
                    if profile:row['summary']=stage_groups(session.last_stats)
                    rows.append(row);c.emit('SMALL_PROFILE_SAMPLE',row)
            session=None;gc.collect()
            if not c.hybrid.idle():raise ValueError('profile context cleanup')
            c.emit('SMALL_PROFILE_SESSION_CLOSED',dict(session=index,idle_after_close=True))
    if calls or identity()[4]!=files:raise ValueError('profile CPU/source gate')
    groups=sorted(rows[1]['summary']['groups_ms'])
    c.emit('SMALL_PROFILE_RESULT',dict(samples=len(rows),
        wall_ms={str(mode):c.stats([row['wall_ms'] for row in rows if row['profile']==mode]) for mode in (False,True)},
        profiled_groups_ms={key:c.stats([row['summary']['groups_ms'][key] for row in rows if row['profile']]) for key in groups},
        cpu_neural_oracle_calls=len(calls),registered_files_unchanged=True,idle_after_close=c.hybrid.idle(),
        interpretation=pre['interpretation'],scope='diagnostic stage attribution; no new whole-model speedup claim'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','benchmark'))
    parser.add_argument('--prereg',type=Path,required=True);args=parser.parse_args()
    (register if args.command=='register' else benchmark)(args.prereg)
