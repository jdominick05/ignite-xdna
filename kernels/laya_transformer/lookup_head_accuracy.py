#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""New frozen arm for shorter heads and the independently compiled GELU lookup."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import buffered_accuracy as ba
import optimized_probe as op
na=ba.na; p=ba.p
NAME='native_lookup_short_heads_bf16_v1'


def identity(args,base,model,mlp_folder,attention_folder):
    bundle,files=ba.identity(args,base,model,mlp_folder,attention_folder)
    head=json.loads((args.head_folder/'build.json').read_text())
    if head['sources']!=op.hp.bh.sources() or head['base_identity']!=bundle.identity: raise ValueError('stale frozen head build')
    if head['attention_build_sha256']!=p.attention.sha(attention_folder/'build.json'): raise ValueError('head attention parent changed')
    if set(head['layouts'])!={'128','256','384','512'}: raise ValueError('missing frozen head buckets')
    parent=json.loads((attention_folder/'build.json').read_text())
    files['short_heads/build.json']=p.attention.sha(args.head_folder/'build.json')
    for m,layouts in head['layouts'].items():
        if set(layouts)!={'head_pre','head_attention','head_post','second_pre','second_post'}: raise ValueError('missing frozen head stage')
        for name,layout in layouts.items():
            expected=parent['layouts']['512'] if name=='head_attention' else bundle.meta['vector']
            if layout['audits']!=expected['audits'] or layout['cdo']!=expected['cdo']: raise ValueError('frozen head core/init changed')
            for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
                actual=p.attention.sha(args.head_folder/m/name/file)
                if actual!=layout[key]: raise ValueError('frozen head artifact changed')
                files[f'short_heads/{m}/{name}/{file}']=actual
    lut=json.loads((args.fast_folder/'build.json').read_text())
    op.lp.Session.validate_candidate(None,lut,bundle)
    for name,sha in lut['unchanged_objects'].items():
        if p.attention.sha(bundle.folders['vector']/(name+'.o'))!=sha: raise ValueError('lookup changes another helper')
    files['lookup/build.json']=p.attention.sha(args.fast_folder/'build.json')
    for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
        actual=p.attention.sha(args.fast_folder/file)
        if actual!=lut[key]: raise ValueError('frozen lookup artifact changed')
        files['lookup/'+file]=actual
    files.update({'source/'+key:value for meta in (head,lut) for key,value in meta['sources'].items()})
    for module in (op,op.hp,op.fp,op.lp,op.cp,op.cw,op.sq):
        filename=Path(module.__file__)
        files['source/'+str(filename.relative_to(p.ROOT)).replace('\\','/')]=p.attention.sha(filename)
    files['source/kernels/laya_transformer/lookup_head_accuracy.py']=p.attention.sha(Path(__file__))
    return bundle,files


def register(args):
    pre,items=na.inputs(args.items,args.reference)
    _,files=identity(args,args.base,args.model,args.mlp_folder,args.attention_folder)
    for name,sha in pre['reference']['files'].items():
        if p.attention.sha(args.model.parent/name)!=sha: raise ValueError('frozen CPU model changed')
    prior=ba.previous(args)
    arm=dict(pre)
    arm.update(study='Native Laya lookup/short-head addendum; original frozen decision and calibration gates unchanged',
      base_preregistration=dict(path=str(na.ORIGINAL.relative_to(p.ROOT)).replace('\\','/'),sha256=p.attention.sha(na.ORIGINAL)),
      registered_utc=datetime.now(timezone.utc).isoformat(),tool='kernels/laya_transformer/lookup_head_accuracy.py',
      frozen='before this lookup/short-head arm has native final scores on labelled frozen items',
      arms={NAME:dict(files=files,what='M128 buffered encoder with GELU lookup; shorter heads for all row buckets; other encoder buckets retain original arithmetic')},
      native_equivalence_reference=prior,
      native_equivalence_reference_use='provenance of prior qualified native scores only; bit equivalence is not required for this changed approximation',
      native_policy=dict(batch=1,padded_tokens=512,pad_id=50283,contexts=5,dispatches=66,
        encoder_row_bucket='ceil(actual_length/128)*128',buffered_lookup_bucket=128,other_encoder_buckets='original short images',
        head_row_bucket='same selected row bucket for both complete head blocks',
        initial_residual='device normalized embedding Y',cpu_neural_fallback=False,
        host_work='integer token/marker addresses, question type and boolean option metadata',readback='logits and act_probs only',
        learned_matrix_operands='BF16 RNE',accumulation='FP32',calibration='original T=1 gates; no fitting',
        setup_qualification='three unlabelled fixtures for attention-context consolidation; excluded from scored count'),
      native_C0_definition='Actual served source/artifact map matches committed registration before and after scoring; NPZ carries the same map and identity. Original C1-C5 unchanged.',
      development_disclosure=dict(labelled_final_scores_before_registration=0,
        observed='three unlabelled fixtures, synthetic boundaries and fixture performance; prior native and buffered full-set outcomes have been observed',
        heldout='reuses original frozen set, not an independent held-out study',criteria='original decision/calibration thresholds unchanged; lookup changes logits'),
      item_buckets={str(k):v for k,v in sorted(Counter(op.hp.buffered.apb.ba.bm.bucket(len(item['ids'])) for item in items).items())})
    with args.prereg.open('x',encoding='utf-8',newline='\n') as output: output.write(json.dumps(arm,indent=2)+'\n')
    na.emit('LOOKUP_HEAD_ACCURACY_REGISTER',dict(name=NAME,registration_id=na.digest(arm),file_sha256=p.attention.sha(args.prereg),
      artifact_and_source_hashes=len(files),items=len(items),criteria=arm['criteria'],policy=arm['native_policy']))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('register','score','report'))
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','fast-folder',
      'items','reference','prereg','out','native-reference','native-prereg'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    def factory(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures):
        return op.Session(bundle,mlp,attention,mlp_candidate,attention_candidate,compact,fixtures,args.head_folder,
          fast_folder=args.fast_folder,mlp_kind='lookup')
    # Replace this adapter's module binding, not buffered_probe.Session: that
    # class uses its original module-global Session name in a super() call.
    proxy=SimpleNamespace(**vars(ba.buffered)); proxy.Session=factory
    with patch.object(na,'NAME',NAME),patch.object(ba,'NAME',NAME),\
      patch.object(na,'identity',lambda *values:identity(args,*values)),patch.object(ba,'buffered',proxy):
        if args.command=='register': register(args)
        elif args.command=='score': ba.score(args)
        else:
            na.registered(args,current=True)
            na.report(args)


if __name__=='__main__': main()
