#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Separate frozen accuracy registration for the M128 buffering arm."""
import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import native_accuracy as na
import buffered_probe as buffered

n=na.n; p=na.p
NAME='native_buffered_m128_bf16_v1'
BASE_IDENTITY=na.identity


def identity(args,base,model,mlp_folder,attention_folder):
    bundle,files=BASE_IDENTITY(base,model,mlp_folder,attention_folder)
    for name,folder,module in [('buffered_mlp',args.mlp_candidate,buffered.mf),
      ('buffered_attention',args.attention_candidate,buffered.apb.af),('compact_head',args.compact_folder,buffered.compact)]:
        meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=module.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale buffered accuracy build')
        if name!='compact_head' and meta['policy']['m']!=128: raise ValueError('registered buffering is M128 only')
        files[name+'/build.json']=p.attention.sha(folder/'build.json')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            actual=p.attention.sha(folder/file)
            if actual!=meta[key]: raise ValueError('buffered artifact changed')
            files[name+'/'+file]=actual
        if 'kernel_object_sha256' in meta:
            actual=p.attention.sha(folder/'kernel.o')
            if actual!=meta['kernel_object_sha256']: raise ValueError('buffered arithmetic object changed')
            files[name+'/kernel.o']=actual
        files.update({'source/'+key:value for key,value in meta['sources'].items()})
    for module in (buffered,buffered.apb,buffered.bp):
        filename=Path(module.__file__)
        files['source/'+str(filename.relative_to(p.ROOT)).replace('\\','/')]=p.attention.sha(filename)
    files['source/kernels/laya_transformer/buffered_accuracy.py']=p.attention.sha(Path(__file__))
    return bundle,files


def previous(args):
    """Validate the completed qualified native arm before using its output bits."""
    old_args=argparse.Namespace(**vars(args)); old_args.prereg=args.native_prereg
    with patch.object(na,'NAME','native_short_bf16_v1'):
        arm,items,_=na.registered(old_args)
        na.committed_registration(args.native_prereg,arm)
        na.validate_scored(args.native_reference,arm,items)
    return dict(file_sha256=p.attention.sha(args.native_reference),
      prereg_sha256=p.attention.sha(args.native_prereg),registration_id=na.digest(arm))


def register(args):
    pre,items=na.inputs(args.items,args.reference)
    _,files=identity(args,args.base,args.model,args.mlp_folder,args.attention_folder)
    prior=previous(args)
    for filename,sha in pre['reference']['files'].items():
        if p.attention.sha(args.model.parent/filename)!=sha: raise ValueError('original FP32 reference model changed')
    arm=dict(pre)
    arm.update(study='Buffered native Laya addendum; original frozen items, FP32 reference and C1-C5 unchanged',
      base_preregistration=dict(path=str(na.ORIGINAL.relative_to(p.ROOT)).replace('\\','/'),sha256=p.attention.sha(na.ORIGINAL)),
      registered_utc=datetime.now(timezone.utc).isoformat(),tool='kernels/laya_transformer/buffered_accuracy.py',
      frozen='before this buffering arm has native final scores on labelled frozen items',
      arms={NAME:dict(files=files,what='Original neural arithmetic; M128 encoder attention/MLP FIFOs deeper; head attention consolidated into the variable-KV context')},
      native_equivalence_reference=prior,
      native_policy=dict(batch=1,padded_tokens=512,pad_id=50283,contexts=5,dispatches=66,
        encoder_row_bucket='ceil(actual_length/128)*128',buffered_bucket=128,other_buckets='original short images',
        head_attention='same full-512 head stream, explicit sixteen KV blocks in the short context',
        initial_residual='device normalized embedding Y',cpu_neural_fallback=False,
        host_work='integer token/marker addresses, question type and boolean option metadata',readback='logits and act_probs only',
        learned_matrix_operands='BF16 RNE',accumulation='FP32',calibration='original T=1 gates, no fitting',
        setup_qualification='three unlabelled fixtures before/after head-context consolidation; excluded from item count'),
      native_C0_definition='Actual served source/artifact map must match committed registration before and after scoring; NPZ carries the same registration and map. Original C1-C5 unchanged.',
      development_disclosure=dict(buffered_final_outputs='three unlabelled fixtures and synthetic sequence boundaries',
        buffered_labelled_final_scores_before_registration=0,
        previous_native_arm='original full native scores and report have been observed; this is not an independent held-out study',
        criteria='unchanged original decision/calibration gates; additionally report and require bit equivalence to qualified native outputs'),
      item_buckets={str(k):v for k,v in sorted(Counter(buffered.apb.ba.bm.bucket(len(item['ids'])) for item in items).items())})
    with args.prereg.open('x',encoding='utf-8',newline='\n') as output: output.write(json.dumps(arm,indent=2)+'\n')
    na.emit('BUFFERED_ACCURACY_REGISTER',dict(name=NAME,registration_id=na.digest(arm),file_sha256=p.attention.sha(args.prereg),
      artifact_and_source_hashes=len(files),items=len(items),criteria=arm['criteria'],policy=arm['native_policy']))


def score(args):
    if args.out.suffix!='.npz' or args.out.exists(): raise ValueError('output must be a new NPZ')
    checkpoint=args.out.with_name(args.out.stem+'.checkpoint.npz')
    if checkpoint.exists() or checkpoint.with_name(checkpoint.name+'.tmp').exists(): raise ValueError('observed checkpoint exists; never overwrite/restart')
    arm,items,bundle=na.registered(args,current=True); commit=na.committed_registration(args.prereg,arm)
    if previous(args)!=arm['native_equivalence_reference']: raise ValueError('qualified native reference changed')
    fixtures=items[:3]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('setup requires unlabelled fixtures')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    logits=np.full((len(items),na.la.KMAX),np.nan,np.float32); probabilities=np.full((len(items),2),np.nan,np.float32)
    calls=[]
    def forbidden(*unused,**keywords): calls.append(True); raise AssertionError('CPU neural/oracle call')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)); completed=0; started=time.monotonic()
    na.emit('BUFFERED_ACCURACY_START',dict(name=NAME,registration_id=na.digest(arm),registration_commit=commit,
      items=len(items),policy=arm['native_policy'],keepawake_held=held,scope='full frozen accuracy; timing ineligible'))
    try:
        with ExitStack() as stack:
            for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
            with buffered.Session(bundle,args.mlp_folder,args.attention_folder,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures) as session:
                for j,item in enumerate(items):
                    output=session.run(item['ids'],item['markers'],item['qtype'])
                    if session.last_stats['dispatches']!=66 or len(session.states)!=5: raise ValueError('registered execution policy differs')
                    logits[j,:item['k']]=output['logits'][0]; probabilities[j]=output['act_probs'][0]; completed=j+1
                    if completed%25==0 or completed==len(items):
                        na.save_snapshot(checkpoint,logits,probabilities,items,arm,commit,completed,False,len(calls))
                        na.emit('BUFFERED_ACCURACY_PROGRESS',dict(completed=completed,total=len(items),elapsed_seconds=round(time.monotonic()-started,1),
                          checkpoint_sha256=p.attention.sha(checkpoint),scope='progress; not eligible latency'))
        current,_,_=na.registered(args,current=True)
        if na.digest(current)!=na.digest(arm): raise ValueError('registration changed during run')
        from laya_hybrid import idle
        if calls or not idle(): raise ValueError('CPU neural/oracle or cleanup gate')
        na.save_snapshot(args.out,logits,probabilities,items,arm,commit,completed,True,len(calls))
        na.validate_scored(args.out,arm,items)
        na.emit('BUFFERED_ACCURACY_FINISHED',dict(completed=completed,registration_id=na.digest(arm),output_sha256=p.attention.sha(args.out),
          cpu_neural_oracle_calls=len(calls),device_idle_while_process_alive=True))
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        na.emit('BUFFERED_ACCURACY_KEEPAWAKE',dict(released=True,completed=completed))


def report(args):
    arm,_,_=na.registered(args,current=True)
    if previous(args)!=arm['native_equivalence_reference']: raise ValueError('qualified native reference changed')
    na.report(args)
    with np.load(args.out,allow_pickle=False) as observed,np.load(args.native_reference,allow_pickle=False) as reference:
        exact={name:bool(np.array_equal(observed[name].view(np.uint32),reference[name].view(np.uint32))) for name in ('logits','act_probs')}
    na.emit('BUFFERED_ACCURACY_EQUIVALENCE',dict(exact=exact,scope='full scored output bits including unused logit padding'))
    if not all(exact.values()): raise SystemExit(1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('register','score','report'))
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder',
      'items','reference','prereg','out','native-reference','native-prereg'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    with patch.object(na,'NAME',NAME),patch.object(na,'identity',lambda *values:identity(args,*values)):
        globals()[args.command](args)


if __name__=='__main__': main()
