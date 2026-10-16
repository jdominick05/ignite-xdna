#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen packing arm: original decision gates plus exact qualified-lookup outputs."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import lookup_head_accuracy as lookup
ba=lookup.ba;na=lookup.na;p=lookup.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_pack_full_probe as full
NAME='native_lookup_wide_pack_bf16_v1'
BASE_IDENTITY=lookup.identity


def previous_lookup(args):
    old_args=argparse.Namespace(**vars(args));old_args.prereg=args.lookup_prereg
    with patch.object(na,'NAME',lookup.NAME):
        arm,items,_=na.registered(old_args)
        na.committed_registration(args.lookup_prereg,arm)
        na.validate_scored(args.lookup_reference,arm,items)
    return arm,dict(file_sha256=p.attention.sha(args.lookup_reference),prereg_sha256=p.attention.sha(args.lookup_prereg),registration_id=na.digest(arm))


def identity(args,base,model,mlp_folder,attention_folder):
    bundle,files=BASE_IDENTITY(args,base,model,mlp_folder,attention_folder)
    paths,_,_,extra,_=full.identity()
    if args.pack_folder.resolve()!=(paths.base/'compact_pack_wide_v2_m128').resolve():
        raise ValueError('packing accuracy folder differs from qualified development arm')
    files.update({'packing_qualification/'+key:value for key,value in extra.items()})
    files['source/kernels/laya_transformer/pack_accuracy.py']=p.attention.sha(Path(__file__))
    files['equivalence/lookup_prereg']=p.attention.sha(args.lookup_prereg)
    files['equivalence/lookup_scores']=p.attention.sha(args.lookup_reference)
    previous_lookup(args)
    return bundle,files


def document(old,files,prior_lookup,prior_native):
    arm=copy.deepcopy(old)
    arm.update(study='Single-BD native Laya packing addendum; original frozen decision/calibration gates unchanged',
        registered_utc=datetime.now(timezone.utc).isoformat(),tool='kernels/laya_transformer/pack_accuracy.py',
        frozen='before this packing arm has any labelled frozen outputs',
        arms={NAME:dict(files=files,what='qualified lookup/short heads plus M128 single-BD packing across all encoder layers')},
        native_equivalence_reference=prior_native,
        native_equivalence_reference_use='provenance of original native scores; original-native bit equivalence is not imposed on the inherited lookup approximation',
        lookup_equivalence_reference=prior_lookup,
        lookup_equivalence_requirement='all scored logits and action probabilities, including unused padding, must be bit-exact to qualified lookup',
        development_disclosure=dict(labelled_final_scores_before_registration=0,
            observed='unlabelled fixtures and synthetic boundaries, individual/full-layer exactness and preregistered fixture timing; prior full frozen outcomes observed',
            heldout='known-set revalidation on the disclosed original frozen items, not an independent held-out test',
            criteria='original C0-C5 thresholds, items, FP32 reference and temperatures retained; exact lookup equivalence is an additional gate'))
    arm['native_policy']['packing']='M128 attention uses single-BD FP32[32,66] gathers with unchanged live operands; other encoder buckets unchanged'
    for key in ('criteria','criteria_text','items','reference','decision'):
        if arm[key]!=old[key]:raise ValueError('packing addendum changed original '+key)
    return arm


def register(args):
    if args.out.exists() or args.out.with_name(args.out.stem+'.checkpoint.npz').exists():raise ValueError('packing outputs already observed')
    old,prior=previous_lookup(args)
    _,files=identity(args,args.base,args.model,args.mlp_folder,args.attention_folder)
    arm=document(old,files,prior,ba.previous(args))
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(arm,indent=2)+'\n')
    na.emit('PACK_ACCURACY_REGISTER',dict(name=NAME,registration_id=na.digest(arm),file_sha256=p.attention.sha(args.prereg),
        hashes=len(files),criteria=arm['criteria'],lookup_equivalence=prior,policy=arm['native_policy']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','score','report'))
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','fast-folder',
        'pack-folder','items','reference','prereg','out','native-reference','native-prereg','lookup-reference','lookup-prereg'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    def factory(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures):
        return full.Session(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures,args.head_folder,
            fast_folder=args.fast_folder,pack_folder=args.pack_folder,mlp_kind='lookup')
    proxy=SimpleNamespace(**vars(ba.buffered));proxy.Session=factory
    with patch.object(na,'NAME',NAME),patch.object(ba,'NAME',NAME),patch.object(na,'identity',lambda *values:identity(args,*values)),patch.object(ba,'buffered',proxy):
        if args.command=='register':register(args)
        elif args.command=='score':ba.score(args)
        else:
            arm,_,_=na.registered(args,current=True)
            _,prior=previous_lookup(args)
            if prior!=arm['lookup_equivalence_reference']:raise ValueError('lookup equivalence reference changed')
            na.report(args)
            with ba.np.load(args.out,allow_pickle=False) as observed,ba.np.load(args.lookup_reference,allow_pickle=False) as reference:
                exact={key:bool(ba.np.array_equal(observed[key].view(ba.np.uint32),reference[key].view(ba.np.uint32))) for key in ('logits','act_probs')}
            na.emit('PACK_ACCURACY_EQUIVALENCE',dict(exact=exact,scope='all frozen outputs including unused padding; additional to original C0-C5'))
            if not all(exact.values()):raise SystemExit(1)


if __name__=='__main__':main()
