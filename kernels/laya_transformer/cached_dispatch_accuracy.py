#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen original gates and exact qualified-control bits for cached Wi dispatch."""
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import small_lookup_accuracy as parent
ba=parent.ba
na=parent.na
p=parent.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_cached_dispatch_full as candidate
NAME='native_cached_wi_dispatch_bf16_v1'


def previous(args):
    old_args=argparse.Namespace(**vars(args));old_args.prereg=args.small_prereg
    with patch.object(na,'NAME',parent.NAME):
        arm,items,_=na.registered(old_args)
        na.committed_registration(args.small_prereg,arm);na.validate_scored(args.small_reference,arm,items)
    return arm,dict(file_sha256=p.attention.sha(args.small_reference),prereg_sha256=p.attention.sha(args.small_prereg),
        registration_id=na.digest(arm))


def full_gate(path):
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in path.read_text(encoding='utf-8').splitlines() if line.startswith(tag+' ')]
    rows=get('CACHED_DISPATCH_FULL_CHECK');verdict=get('CACHED_DISPATCH_FULL_RESULT');end=get('RESEARCH_PROCESS_RESULT')
    if (len(rows)!=25 or len(verdict)!=1 or len(end)!=1 or verdict[0]['cases']!=25 or
        not verdict[0]['all_output_bits_exact'] or verdict[0]['cpu_neural_oracle_calls'] or not verdict[0]['idle_after_close'] or
        end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        not all(all(row['exact'].values()) and row['contexts']==5 and row['dispatches']==66 and
            row['cached_mlp_stages']==(28 if row['valid']<=128 else 0) for row in rows)):
        raise ValueError('cached Wi full development prerequisite failed')
    return p.attention.sha(path)


def identity(args,base,model,mlp_folder,attention_folder):
    bundle,files=parent.identity(args,base,model,mlp_folder,attention_folder)
    _,other,_,control,folder,extra,_=candidate.identity()
    if args.cache_folder.resolve()!=folder.resolve() or args.small_folder.resolve()!=control.resolve() or other.identity!=bundle.identity:
        raise ValueError('cached accuracy artifacts differ from development arm')
    files.update({'cached_dispatch_qualification/'+key:value for key,value in extra.items()})
    for module in (candidate.layer.previous,candidate.layer.failed,candidate.layer.previous.reference):
        path=Path(module.__file__)
        files['source/'+str(path.relative_to(p.ROOT)).replace('\\','/')]=p.attention.sha(path)
    for name in ('build.json','insts.bin','final.xclbin'):
        files['cached_dispatch/'+name]=p.attention.sha(folder/name)
    files['cached_dispatch_full_gate']=full_gate(p.ROOT/'results/laya/cached_dispatch_full_v1_desktop2_20261017.log')
    files['source/kernels/laya_transformer/cached_dispatch_accuracy.py']=p.attention.sha(Path(__file__))
    files['equivalence/small_prereg']=p.attention.sha(args.small_prereg)
    files['equivalence/small_scores']=p.attention.sha(args.small_reference)
    previous(args)
    return bundle,files


def document(old,files,prior):
    arm=copy.deepcopy(old)
    arm.update(study='Self-contained cached Wi revalidation; original frozen gates unchanged',
        registered_utc=datetime.now(timezone.utc).isoformat(),tool='kernels/laya_transformer/cached_dispatch_accuracy.py',
        frozen='before this cached-dispatch arm has any labelled final outputs',
        arms={NAME:dict(files=files,what='M128 Wi query residency with self-contained finite query DMA setup; qualified arithmetic unchanged')},
        small_equivalence_reference=prior,
        small_equivalence_requirement='all scored logits/action probabilities including unused padding must be bit-exact to qualified smaller-query control',
        development_disclosure=dict(labelled_final_scores_before_registration=0,
            observed='unlabelled isolated layers and full development boundaries; earlier resident-context integration failed before a complete fixture; prior frozen outcomes observed',
            heldout='known-set revalidation on the disclosed original frozen items, not an independent held-out test',
            criteria='original C0-C5 thresholds, items, FP32 reference and temperatures retained; exact qualified control bits are additional'))
    arm['native_policy']['cached_wi_mlp']='M128 only; same 66 dispatches/five contexts; each MLP installs query BDs then preloads and replays Wi; no CPU neural work or lock resets'
    for key in ('criteria','criteria_text','items','reference','decision'):
        if arm[key]!=old[key]:raise ValueError('cached addendum changed original '+key)
    return arm


def register(args):
    if args.out.exists() or args.out.with_name(args.out.stem+'.checkpoint.npz').exists():raise ValueError('cached labelled outputs already observed')
    old,prior=previous(args);_,files=identity(args,args.base,args.model,args.mlp_folder,args.attention_folder)
    arm=document(old,files,prior)
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(arm,indent=2)+'\n')
    na.emit('CACHED_ACCURACY_REGISTER',dict(name=NAME,registration_id=na.digest(arm),file_sha256=p.attention.sha(args.prereg),
        hashes=len(files),criteria=arm['criteria'],small_equivalence=prior,policy=arm['native_policy']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','score','report'))
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','fast-folder',
        'pack-folder','items','reference','prereg','out','native-reference','native-prereg','lookup-reference','lookup-prereg',
        'small-folder','packing-reference','packing-prereg','small-reference','small-prereg','cache-folder'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    def factory(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures):
        paths=SimpleNamespace(base=args.base,mlp_folder=mlp,attention_folder=attention,mlp_candidate=mlp_candidate,
            attention_candidate=attention_candidate,compact_folder=compact,head_folder=args.head_folder)
        return candidate.Session(paths,bundle,fixtures,args.small_folder,args.cache_folder)
    proxy=SimpleNamespace(**vars(ba.buffered));proxy.Session=factory
    with patch.object(na,'NAME',NAME),patch.object(ba,'NAME',NAME),patch.object(na,'identity',lambda *values:identity(args,*values)),patch.object(ba,'buffered',proxy):
        if args.command=='register':register(args)
        elif args.command=='score':ba.score(args)
        else:
            arm,_,_=na.registered(args,current=True);_,prior=previous(args)
            if prior!=arm['small_equivalence_reference']:raise ValueError('qualified small-control reference changed')
            na.report(args)
            with ba.np.load(args.out,allow_pickle=False) as observed,ba.np.load(args.small_reference,allow_pickle=False) as reference:
                exact={key:bool(ba.np.array_equal(observed[key].view(ba.np.uint32),reference[key].view(ba.np.uint32))) for key in ('logits','act_probs')}
            na.emit('CACHED_ACCURACY_EQUIVALENCE',dict(exact=exact,scope='all frozen scored output bits; additional to original C0-C5'))
            if not all(exact.values()):raise SystemExit(1)


if __name__=='__main__':main()
