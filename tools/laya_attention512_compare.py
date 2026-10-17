#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One full-qualified attention buffering adoption timing, with a matched hybrid witness."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import laya_attention512_full as candidate
import laya_polynomial512_compare as control_comparison
c=candidate.c
full=candidate.polynomial.full
import attention512_accuracy as accuracy
ARMS=('control','attention','hybrid512')


def protocol():
    return dict(order=[['control','attention','hybrid512'],['hybrid512','attention','control'],
                       ['control','attention','hybrid512']],warmups=2,reps=5,
        expected_saved_ms=250,kill_saved_ms=100,require_every_round_win=True,
        metric='external wall around complete Session.run or hybrid Plan.run; setup and output validation excluded',
        scope='same logical91-token request;512 encoder and head rows in both native arms and hybrid;fresh serialized sessions',
        native_policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        hybrid_policy=dict(contexts=1,npu_products=118,host_products=2),
        goal_rule='candidate pooled median and every round median below matched hybrid, plus native adoption and eligible wrapper')


def records(path,tag):
    return [json.loads(line.split(' ',1)[1]) for line in path.read_text(encoding='utf-8').splitlines()
            if line.startswith(tag+' ')]


def frozen_gate(prereg,score,report):
    pre=json.loads(prereg.read_text(encoding='utf-8'))
    c.na.committed_registration(prereg,pre)
    verdict=records(report,'NATIVE_ACCURACY_VERDICT')
    end=records(report,'RESEARCH_PROCESS_RESULT')
    if (len(verdict)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        not verdict[0]['passed'] or verdict[0]['criteria']!={f'C{i}':True for i in range(6)} or
        verdict[0]['labelled']!=2400 or verdict[0]['registration_id']!=c.na.digest(pre) or
        verdict[0]['output_sha256']!=c.sha(score)):
        raise ValueError('original full frozen gates must pass before adoption timing')
    return dict(prereg_sha256=c.sha(prereg),output_sha256=c.sha(score),report_sha256=c.sha(report),
                registration_id=c.na.digest(pre))


def identity():
    paths,bundle,fixtures,folder,attention_folder,files=candidate.identity()
    prereg=c.ROOT/'results/laya/attention_fifo_m512_accuracy_v1_prereg_desktop2_20261017.json'
    score=c.ROOT/'scratch/laya/native_accuracy/native_attention_fifo_full512_bf16_v1.npz'
    report=c.ROOT/'results/laya/attention_fifo_m512_accuracy_v1_report_desktop2_20261017.log'
    gate=frozen_gate(prereg,score,report)
    pre=json.loads(prereg.read_text(encoding='utf-8'))
    _,current=accuracy.identity()
    if current!=pre['arms'][accuracy.NAME]['files']:
        raise ValueError('qualified attention512 source/artifact map changed')
    _,items=c.na.inputs(paths.items,paths.reference)
    with patch.object(c.na,'NAME',accuracy.NAME):
        c.na.validate_scored(score,pre,items)
    files=dict(files)
    files['comparison_source']=c.sha(Path(__file__))
    # Revalidate the adopted polynomial control's original frozen proof as well.
    _,_,control_fixtures,_,_,control_gate=control_comparison.identity()
    if control_fixtures!=fixtures:
        raise ValueError('control and candidate fixtures differ')
    files['control_comparison_source']=c.sha(Path(control_comparison.__file__))
    files['control_frozen_proof']=c.na.digest(control_gate)
    gate=dict(attention=gate,control=control_gate)
    return paths,bundle,fixtures,(folder,attention_folder),files,gate


def decision(rows):
    expected={(index,arm) for index in range(3) for arm in ARMS}
    if len(rows)!=9 or {(row['round'],row['arm']) for row in rows}!=expected:
        raise ValueError('comparison coverage')
    if any(len(row['samples_ms'])!=5 or any(not math.isfinite(value) or value<=0
        for value in row['samples_ms']) for row in rows):
        raise ValueError('comparison samples')
    arms={arm:c.stats([value for row in rows if row['arm']==arm for value in row['samples_ms']]) for arm in ARMS}
    saved=[]
    hybrid_margin=[]
    for index in range(3):
        medians={row['arm']:c.stats(row['samples_ms'])['median_ms'] for row in rows if row['round']==index}
        saved.append(medians['control']-medians['attention'])
        hybrid_margin.append(medians['hybrid512']-medians['attention'])
    pooled=arms['control']['median_ms']-arms['attention']['median_ms']
    adopt=pooled>=protocol()['kill_saved_ms'] and all(value>0 for value in saved)
    goal=adopt and arms['attention']['median_ms']<arms['hybrid512']['median_ms'] and all(value>0 for value in hybrid_margin)
    return dict(arms=arms,pooled_saved_ms=pooled,round_median_saved_ms=saved,
        round_hybrid_minus_attention_ms=hybrid_margin,adoption_from_samples='adopt' if adopt else 'kill',
        matched512_goal_from_samples=goal,requires_timing_eligible_wrapper=True)


def timing_verdict(path,prereg=None):
    starts=records(path,'ATTENTION512_COMPARE_START')
    rows=records(path,'ATTENTION512_COMPARE_PROFILE')
    results=records(path,'ATTENTION512_COMPARE_RESULT')
    end=records(path,'RESEARCH_PROCESS_RESULT')
    if (len(starts)!=1 or len(results)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        not end[0]['timing_eligible'] or end[0].get('foreign_contention_observed',True) or
        results[0]['cpu_neural_oracle_calls'] or not results[0]['idle_after_close'] or
        not results[0]['registered_files_unchanged'] or not all(row['idle_after_close'] for row in rows)):
        raise ValueError('comparison is not complete and timing eligible')
    if starts[0]['scope']!=protocol()['scope'] or starts[0]['metric']!=protocol()['metric']:
        raise ValueError('comparison timing scope changed')
    if [(row['round'],row['arm']) for row in rows]!=[
        (index,arm) for index,order in enumerate(protocol()['order']) for arm in order]:
        raise ValueError('comparison arm order changed')
    if prereg is not None:
        pre=json.loads(prereg.read_text(encoding='utf-8'))
        c.na.committed_registration(prereg,pre)
        if starts[0]['prereg_sha256']!=c.sha(prereg) or starts[0]['frozen']!=pre['frozen']:
            raise ValueError('comparison log belongs to another registration')
        if any(pre.get(key)!=value for key,value in protocol().items()):
            raise ValueError('comparison registered protocol changed')
    for row in rows:
        expected=protocol()['hybrid_policy' if row['arm']=='hybrid512' else 'native_policy']
        if row['policy']!=expected:
            raise ValueError('timed execution policy changed')
    verdict=decision(rows)
    if any(results[0][key]!=value for key,value in verdict.items()):
        raise ValueError('comparison summary differs from samples')
    return verdict


def register(path):
    _,_,fixtures,_,files,gate=identity()
    if len(fixtures[0]['ids'])!=91:
        raise ValueError('comparison fixture changed')
    pre=dict(version=1,files=files,frozen=gate,fixture=fixtures[0],**protocol())
    with path.open('x',encoding='utf-8',newline='\n') as output:
        output.write(json.dumps(pre,indent=2)+'\n')
    c.emit('ATTENTION512_COMPARE_REGISTER',dict(sha256=c.sha(path),identity=c.na.digest(pre),files=len(files)))


def native_profile(paths,bundle,fixtures,folder,arm,pre,golden,calls):
    if arm=='attention':
        session=candidate.Session(paths,bundle,fixtures,folder[0],folder[1])
    elif arm=='control':
        # The control is the adopted full512 polynomial, never the older slow image.
        session=candidate.polynomial.Session(paths,bundle,fixtures,folder[0])
    else:
        raise ValueError('unknown native comparison arm')
    samples=[]
    internal=[]
    try:
        with session:
            for index in range(pre['warmups']+pre['reps']):
                start=time.perf_counter()
                output=session.run(**full.feed(fixtures[0]))
                wall=(time.perf_counter()-start)*1000
                if golden is None:
                    golden={key:value.copy() for key,value in output.items()}
                if not all(full.exact(output,golden).values()):
                    raise ValueError('native comparison outputs not bit-exact')
                if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:
                    raise ValueError('native comparison execution policy')
                if index>=pre['warmups']:
                    samples.append(wall)
                    internal.append(dict(session.last_stats))
    finally:
        session=None
        gc.collect()
    return dict(**c.stats(samples),native_internal=internal,policy=pre['native_policy']),golden


def hybrid_profile(paths,fixtures,pre,golden):
    plan=c.hybrid.Plan(paths.model,threads=8)
    products=c.hybrid.NpuProducts(plan,paths.gemm_dir)
    feed=c.hybrid.padded_feed(fixtures[0]['ids'],fixtures[0]['markers'],fixtures[0]['qtype'])
    samples=[]
    reference=None
    host_calls=0
    original=c.ort.InferenceSession.run
    def host_run(session,*args,**kwargs):
        nonlocal host_calls
        host_calls+=1
        return original(session,*args,**kwargs)
    try:
        with patch.object(c.ort.InferenceSession,'run',host_run):
            for index in range(pre['warmups']+pre['reps']):
                start=time.perf_counter()
                output=dict(zip(plan.graph_outputs,plan.run(feed,products),strict=True))
                wall=(time.perf_counter()-start)*1000
                if reference is None:
                    reference={key:value.copy() for key,value in output.items()}
                if not all(full.exact(output,reference).values()):
                    raise ValueError('hybrid outputs do not repeat bit-exactly')
                if index>=pre['warmups']:
                    samples.append(wall)
        c.check(output,golden)
        total=pre['warmups']+pre['reps']
        if products.counts!={'npu':118*total,'host':2*total} or host_calls!=total*sum(seg is not None for seg in plan.segments):
            raise ValueError('hybrid comparison execution policy')
        observed=dict(product_counts=products.counts.copy(),host_segment_calls=host_calls,calls_including_warmups=total)
    finally:
        products.close()
        products=None
        plan=None
        gc.collect()
    return dict(**c.stats(samples),policy=pre['hybrid_policy'],observed=observed)


def benchmark(path):
    pre=json.loads(path.read_text(encoding='utf-8'))
    c.na.committed_registration(path,pre)
    paths,bundle,fixtures,folder,files,gate=identity()
    if pre['files']!=files or pre['frozen']!=gate or pre['fixture']!=fixtures[0] or any(pre[key]!=value for key,value in protocol().items()):
        raise ValueError('comparison registration changed')
    rows=[]
    golden=None
    calls=[]
    c.emit('ATTENTION512_COMPARE_START',dict(prereg_sha256=c.sha(path),frozen=gate,
        scope=pre['scope'],metric=pre['metric']))
    for index,order in enumerate(pre['order']):
        for arm in order:
            if not c.hybrid.idle():
                raise ValueError('device occupied before comparison arm')
            if arm=='hybrid512':
                if golden is None:
                    raise ValueError('native reference required before hybrid')
                result=hybrid_profile(paths,fixtures,pre,golden)
            else:
                with full.guards(calls):
                    result,golden=native_profile(paths,bundle,fixtures,folder,arm,pre,golden,calls)
            if not c.hybrid.idle():
                raise ValueError('comparison cleanup failure')
            row=dict(round=index,arm=arm,**result,idle_after_close=True)
            rows.append(row)
            c.emit('ATTENTION512_COMPARE_PROFILE',row)
    if calls or identity()[4:]!=(files,gate):
        raise ValueError('comparison source or CPU-neural gate')
    c.emit('ATTENTION512_COMPARE_RESULT',dict(**decision(rows),cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,registered_files_unchanged=True,scope=pre['scope']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('register','benchmark','verify'))
    parser.add_argument('--prereg',type=Path)
    parser.add_argument('--log',type=Path)
    args=parser.parse_args()
    if args.command=='verify':
        if args.log is None or args.prereg is None:
            parser.error('--log and --prereg required')
        c.emit('ATTENTION512_TIMING_VERDICT',timing_verdict(args.log,args.prereg))
    else:
        if args.prereg is None:
            parser.error('--prereg required')
        (register if args.command=='register' else benchmark)(args.prereg)
