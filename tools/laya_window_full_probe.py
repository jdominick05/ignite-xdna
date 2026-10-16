#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete-model window MLP qualification and serialized native comparison."""
import argparse
import gc
import json
from pathlib import Path
import time

import laya_window_layer_probe as layer
full=layer.full;c=full.c;op=full.op;candidate=layer.candidate


class Session(full.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                 window_folder,pack_folder):
        folder=Path(window_folder);meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=candidate.sources() or meta['base_identity']!=bundle.identity:
            raise ValueError('stale complete window MLP')
        raw=op.artifacts(folder,meta)
        # Keep one replaceable MLP context. Installing lookup first would have
        # already replaced buffered_mlp and would require an extra context.
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                         fast_folder=None,pack_folder=pack_folder,mlp_kind='lookup')
        try:op.Session.install(self,'mlp',folder,meta,raw)
        except BaseException:self.close();raise


def identity():
    paths,bundle,fixtures,folder,meta,raw,files,proof=layer.identity()
    evidence=c.ROOT/'results/laya/window_mlp_layer_v1_qualification_and_cleanup_desktop2_20261016.log'
    lines=evidence.read_text().splitlines()
    results=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('WINDOW_LAYER_RESULT ')]
    terminals=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
    if (len(results)!=1 or results[0]['cases']!=6 or not results[0]['all_arena_bits_exact'] or
        results[0]['cpu_neural_oracle_calls'] or not results[0]['idle_after_close'] or not terminals or terminals[-1]['exit_code']!=0):
        raise ValueError('complete layer prerequisite not qualified')
    files['window_layer_qualification']=c.sha(evidence)
    files['window_source/tools/laya_window_full_probe.py']=c.sha(Path(__file__))
    return paths,bundle,fixtures,folder,files,proof


def make(paths,bundle,fixtures,folder,window):
    if not window:return full.make(paths,bundle,fixtures,True)
    return Session(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
                   paths.compact_folder,fixtures,paths.head_folder,folder,paths.base/'compact_pack_wide_v2_m128')


def qualify():
    paths,bundle,fixtures,folder,files,proof=identity();requests=full.cases(fixtures);references=[];calls=[]
    c.emit('WINDOW_FULL_START',dict(files_count=len(files),files_digest=c.na.digest(files),proof=proof,cases=len(requests),
        runner_sha256=c.sha(Path(__file__)),scope='all28 M128 MLPs and packing; full model development correctness only'))
    if not c.hybrid.idle():raise ValueError('device occupied')
    with full.guards(calls):
        with make(paths,bundle,fixtures,folder,False) as session:
            for row in requests:references.append({key:value.copy() for key,value in session.run(**full.feed(row)).items()})
        session=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('reference context cleanup')
        with make(paths,bundle,fixtures,folder,True) as session:
            for row,reference in zip(requests,references):
                output=session.run(**full.feed(row));exact=full.exact(output,reference)
                c.emit('WINDOW_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=exact,contexts=len(session.states),dispatches=session.last_stats['dispatches']))
                if not all(exact.values()) or len(session.states)!=5 or session.last_stats['dispatches']!=66:
                    raise ValueError('complete window fixture/boundary/execution gate')
        session=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[4]!=files:raise ValueError('window source/CPU/cleanup gate')
    c.emit('WINDOW_FULL_RESULT',dict(cases=len(requests),all_output_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,registered_files_unchanged=True,scope='development correctness only; frozen accuracy and speed unqualified'))


def protocol():
    return dict(order=[['packing','window'],['window','packing'],['packing','window']],warmups=2,reps=5,
        policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        scope='same-sitting complete native serving arms; same91-token request M128; setup excluded; no CPU/hybrid ratio',
        disclosure='unlabelled development outputs observed; window frozen known-set accuracy remains unqualified')


def register(args):
    _,_,fixtures,_,files,proof=identity()
    pre=dict(version=1,files=files,proof=proof,fixture=fixtures[0],**protocol())
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(pre,indent=2)+'\n')
    c.emit('WINDOW_COMPARE_REGISTER',dict(sha256=c.sha(args.prereg),identity=c.na.digest(pre),files=len(files)))


def benchmark(args):
    pre=json.loads(args.prereg.read_text());c.na.committed_registration(args.prereg,pre)
    paths,bundle,fixtures,folder,files,proof=identity()
    if pre['files']!=files or pre['fixture']!=fixtures[0] or pre['proof']!=proof or any(pre[k]!=v for k,v in protocol().items()):
        raise ValueError('window comparison preregistration changed')
    calls=[];reference=None;rows=[]
    c.emit('WINDOW_COMPARE_START',dict(prereg_sha256=c.sha(args.prereg),scope=pre['scope']))
    with full.guards(calls):
        for round_index,order in enumerate(pre['order']):
            for name in order:
                if not c.hybrid.idle():raise ValueError('device occupied before profile')
                with make(paths,bundle,fixtures,folder,name=='window') as session:
                    for _ in range(pre['warmups']):output=session.run(**full.feed(fixtures[0]))
                    if reference is None:reference={key:value.copy() for key,value in output.items()}
                    samples=[]
                    for _ in range(pre['reps']):
                        start=time.perf_counter();output=session.run(**full.feed(fixtures[0]));samples.append((time.perf_counter()-start)*1000)
                        if not all(full.exact(output,reference).values()):raise ValueError('timed output changed')
                    if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:raise ValueError('timed execution policy changed')
                session=None;gc.collect()
                if not c.hybrid.idle():raise ValueError('profile context cleanup')
                row=dict(round=round_index,arm=name,**c.stats(samples),idle_after_close=True);rows.append(row);c.emit('WINDOW_COMPARE_PROFILE',row)
    if calls or identity()[4]!=files:raise ValueError('comparison source/CPU gate')
    c.emit('WINDOW_COMPARE_RESULT',dict(arms={name:c.stats([value for row in rows if row['arm']==name for value in row['samples_ms']]) for name in ('packing','window')},
        cpu_neural_oracle_calls=len(calls),idle_after_close=c.hybrid.idle(),registered_files_unchanged=True,
        scope='incremental native comparison; full window accuracy qualification still required before adoption'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('qualify','register','benchmark'))
    ap.add_argument('--prereg',type=Path,default=c.ROOT/'results/laya/window_compare_v1_prereg_desktop2_20261016.json')
    args=ap.parse_args()
    if args.command=='qualify':qualify()
    elif args.command=='register':register(args)
    else:benchmark(args)
