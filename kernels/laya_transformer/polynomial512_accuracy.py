#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen full-set validation of original polynomial O3 at fixed 512 rows."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import buffered_accuracy as ba
na=ba.na
p=ba.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_polynomial512_full as candidate
c=candidate.c
NAME='native_polynomial_o3_full512_bf16_v1'


def development_gate(path):
    rows=path.read_text(encoding='utf-8').splitlines()
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in rows if line.startswith(tag+' ')]
    result=get('POLYNOMIAL512_FULL_RESULT')
    checks=get('POLYNOMIAL512_FULL_CHECK')
    end=get('RESEARCH_PROCESS_RESULT')
    if (len(result)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        len(checks)!=25 or result[0]['cases']!=25 or not result[0]['all_output_bits_exact'] or
        result[0]['cpu_neural_oracle_calls'] or not result[0]['idle_after_close'] or
        not result[0]['registered_files_unchanged'] or not all(all(row['exact'].values()) and
        row['contexts']==5 and row['dispatches']==66 and row['polynomial_stages']==28 for row in checks)):
        raise ValueError('complete polynomial512 development gate missing')
    return c.sha(path)


def identity(*unused):
    _,bundle,_,_,files=candidate.identity()
    files=dict(files)
    evidence=p.ROOT/'results/laya/geglu_polynomial_m512_v1_full_desktop2_20261017.log'
    files['polynomial512_full_development']=development_gate(evidence)
    files['source/kernels/laya_transformer/polynomial512_accuracy.py']=c.sha(Path(__file__))
    return bundle,files


def document(original,files,prior):
    arm=copy.deepcopy(original)
    arm.update(study='Original polynomial O3 full512 addendum; original frozen criteria unchanged',
        base_preregistration=dict(path=str(na.ORIGINAL.relative_to(p.ROOT)).replace('\\','/'),sha256=c.sha(na.ORIGINAL)),
        registered_utc=datetime.now(timezone.utc).isoformat(),tool='kernels/laya_transformer/polynomial512_accuracy.py',
        frozen='before this fixed512 polynomial arm has labelled frozen outputs',
        arms={NAME:dict(files=files,what='Original polynomial source at O3 and depth2, fixed512 encoder and original full heads')},
        native_equivalence_reference=prior,
        native_equivalence_reference_use='provenance and diagnostic bit comparison to the prior bucketed native arm; original C0-C5 decide accuracy',
        native_C0_definition='Source/artifact map equals committed registration before and after scoring; original C1-C5 retained.',
        native_policy=dict(batch=1,padded_tokens=512,pad_id=50283,contexts=5,dispatches=66,
            encoder_row_bucket='fixed512 for EVERY item, including short logical inputs',
            embedding_and_head_schedule=512,polynomial_stages=28,approximation='original polynomial, unchanged source and constants',
            codegen='O3, no fast-math; Q/KV/L2 Q FIFO depth2',cpu_neural_fallback=False,
            host_work='integer token/marker addresses, question type and boolean option metadata',
            readback='logits and act_probs only',calibration='original T=1 gates, no fitting'),
        development_disclosure=dict(labelled_final_scores_before_registration=0,
            observed='unlabelled complete development cases and prior native/lookup/cache frozen outcomes',
            heldout='known-set revalidation, not an independent held-out test',
            criteria='original C0-C5 items, FP32 reference, thresholds and temperatures unchanged',
            prior_kill='distinct lookup approximation failed development and is not this polynomial image'))
    for key in ('criteria','criteria_text','items','reference','decision'):
        if arm[key]!=original[key]:
            raise ValueError('polynomial512 addendum changed original '+key)
    return arm


def register(args):
    if args.out.exists() or args.out.with_name(args.out.stem+'.checkpoint.npz').exists():
        raise ValueError('polynomial512 labelled outputs already observed')
    original,_=na.inputs(args.items,args.reference)
    _,files=identity()
    prior=ba.previous(args)
    for filename,sha in original['reference']['files'].items():
        if c.sha(args.model.parent/filename)!=sha:
            raise ValueError('original reference model changed')
    arm=document(original,files,prior)
    with args.prereg.open('x',encoding='utf-8',newline='\n') as output:
        output.write(json.dumps(arm,indent=2)+'\n')
    na.emit('POLYNOMIAL512_ACCURACY_REGISTER',dict(name=NAME,registration_id=na.digest(arm),
        file_sha256=c.sha(args.prereg),hashes=len(files),criteria=arm['criteria'],policy=arm['native_policy']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('register','score','report'))
    parser.add_argument('--prereg',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parsed=parser.parse_args()
    paths=c.paths()
    args=SimpleNamespace(**vars(paths))
    args.prereg=parsed.prereg
    args.out=parsed.out
    args.native_reference=p.ROOT/'scratch/laya/native_accuracy/native_short_bf16_v1.npz'
    args.native_prereg=p.ROOT/'results/laya/native_accuracy_v1_prereg_desktop2_20261016.json'
    def factory(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures):
        return candidate.Session(paths,bundle,fixtures,paths.base/'geglu_fifo_v1_double_m512')
    proxy=SimpleNamespace(**vars(ba.buffered))
    proxy.Session=factory
    with patch.object(na,'NAME',NAME),patch.object(ba,'NAME',NAME),patch.object(na,'identity',identity),patch.object(ba,'buffered',proxy):
        if parsed.command=='register':
            register(args)
        elif parsed.command=='score':
            ba.score(args)
        else:
            na.report(args)
            with ba.np.load(args.out,allow_pickle=False) as observed,ba.np.load(args.native_reference,allow_pickle=False) as reference:
                exact={key:bool(ba.np.array_equal(observed[key].view(ba.np.uint32),reference[key].view(ba.np.uint32)))
                    for key in ('logits','act_probs')}
            na.emit('POLYNOMIAL512_PRIOR_BITS',dict(exact=exact,scope='diagnostic vs prior bucketed schedule; original gates remain binding'))


if __name__=='__main__':
    main()
