#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Original frozen gates plus exact qualified-packing outputs for small-query KV2."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pack_accuracy as parent
ba=parent.ba;na=parent.na;p=parent.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_small_lookup_full as small
import laya_small_lookup_compare as comparison
NAME='native_small_lookup_kv2_bf16_v1'


def previous(args):
    old_args=argparse.Namespace(**vars(args));old_args.prereg=args.packing_prereg
    with patch.object(na,'NAME',parent.NAME):
        arm,items,_=na.registered(old_args)
        na.committed_registration(args.packing_prereg,arm);na.validate_scored(args.packing_reference,arm,items)
    return arm,dict(file_sha256=p.attention.sha(args.packing_reference),prereg_sha256=p.attention.sha(args.packing_prereg),registration_id=na.digest(arm))


def identity(args,base,model,mlp_folder,attention_folder):
    bundle,files=parent.identity(args,base,model,mlp_folder,attention_folder)
    paths,_,_,folder,extra,_=comparison.identity()
    if args.small_folder.resolve()!=folder.resolve():raise ValueError('frozen small-query image differs from timed candidate')
    files.update({'small_lookup_qualification/'+key:value for key,value in extra.items()})
    files['source/kernels/laya_transformer/small_lookup_accuracy.py']=p.attention.sha(Path(__file__))
    files['equivalence/packing_prereg']=p.attention.sha(args.packing_prereg)
    files['equivalence/packing_scores']=p.attention.sha(args.packing_reference)
    files['small_lookup_repeat_timing']=p.attention.sha(p.ROOT/'results/laya/small_lookup_compare_repeat_v1_timing_desktop2_20261016.log')
    previous(args)
    return bundle,files


def document(old,files,prior):
    arm=copy.deepcopy(old)
    arm.update(study='One-state smaller-query lookup/KV2 revalidation; original frozen gates unchanged',
        registered_utc=datetime.now(timezone.utc).isoformat(),tool='kernels/laya_transformer/small_lookup_accuracy.py',
        frozen='before this smaller-query arm has any labelled frozen outputs',
        arms={NAME:dict(files=files,what='qualified packing plus M128 smaller-query lookup, one output state and two KV buffers')},
        packing_equivalence_reference=prior,
        packing_equivalence_requirement='all scored logits and action probabilities, including unused padding, must be bit-exact to qualified packing',
        development_disclosure=dict(labelled_final_scores_before_registration=0,
            observed='unlabelled primitives, complete layers, full development boundaries and two preregistered timing sittings; prior full frozen outcomes observed',
            heldout='known-set revalidation on the disclosed original frozen items, not an independent held-out test',
            criteria='original C0-C5 thresholds, items, FP32 reference and temperatures retained; exact packing equivalence is additional'))
    arm['native_policy']['small_query_mlp']='M128 only; lookup arithmetic unchanged; physical Q3/KV2 and one output state; other buckets retain controls'
    for key in ('criteria','criteria_text','items','reference','decision'):
        if arm[key]!=old[key]:raise ValueError('small lookup addendum changed original '+key)
    return arm


def register(args):
    if args.out.exists() or args.out.with_name(args.out.stem+'.checkpoint.npz').exists():raise ValueError('small-query outputs already observed')
    old,prior=previous(args);_,files=identity(args,args.base,args.model,args.mlp_folder,args.attention_folder)
    arm=document(old,files,prior)
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(arm,indent=2)+'\n')
    na.emit('SMALL_ACCURACY_REGISTER',dict(name=NAME,registration_id=na.digest(arm),file_sha256=p.attention.sha(args.prereg),
        hashes=len(files),criteria=arm['criteria'],packing_equivalence=prior,policy=arm['native_policy']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','score','report'))
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','fast-folder',
        'pack-folder','items','reference','prereg','out','native-reference','native-prereg','lookup-reference','lookup-prereg',
        'small-folder','packing-reference','packing-prereg'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    def factory(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures):
        return small.Session(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures,args.head_folder,args.small_folder,args.pack_folder)
    proxy=SimpleNamespace(**vars(ba.buffered));proxy.Session=factory
    with patch.object(na,'NAME',NAME),patch.object(ba,'NAME',NAME),patch.object(na,'identity',lambda *values:identity(args,*values)),patch.object(ba,'buffered',proxy):
        if args.command=='register':register(args)
        elif args.command=='score':ba.score(args)
        else:
            arm,_,_=na.registered(args,current=True);_,prior=previous(args)
            if prior!=arm['packing_equivalence_reference']:raise ValueError('packing equivalence reference changed')
            na.report(args)
            with ba.np.load(args.out,allow_pickle=False) as observed,ba.np.load(args.packing_reference,allow_pickle=False) as reference:
                exact={key:bool(ba.np.array_equal(observed[key].view(ba.np.uint32),reference[key].view(ba.np.uint32))) for key in ('logits','act_probs')}
            na.emit('SMALL_ACCURACY_EQUIVALENCE',dict(exact=exact,scope='all frozen scored output bits including unused padding; additional to original C0-C5'))
            if not all(exact.values()):raise SystemExit(1)


if __name__=='__main__':main()
