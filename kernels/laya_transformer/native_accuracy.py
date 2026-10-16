#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register, score and report a native arm without replacing the original study."""
import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import bucket_attention as ba

n=ba.n; p=ba.p
sys.path.insert(0,str(p.ROOT/'tools'))
import laya_accuracy as la

ORIGINAL=p.ROOT/'results/laya/laya_accuracy_prereg_desktop2_20260926.json'
NAME='native_short_bf16_v1'


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def emit(tag,value): print(tag+' '+json.dumps(value,allow_nan=False),flush=True)


def inputs(items_path,reference):
    pre=json.loads(ORIGINAL.read_text(encoding='utf-8'))
    if p.attention.sha(items_path)!=pre['items']['file_sha256'] or p.attention.sha(reference)!=pre['reference']['logits_sha256']:
        raise ValueError('frozen items/reference changed')
    items=la.read_items(items_path)
    if len(items)!=pre['items']['labelled']+pre['items']['fixture_questions']: raise ValueError('frozen item count')
    if sum(x['gold']>=0 for x in items)!=pre['items']['labelled']: raise ValueError('frozen labelled count')
    for item in items:
        n.request(item['ids'],item['markers'],item['qtype'])
        if not 2<=item['k']<=la.KMAX or item['k']!=len(item['markers']) or not (item['gold']==-1 or 0<=item['gold']<item['k']): raise ValueError('item option/label contract')
    return pre,items


def identity(base,model,mlp_folder,attention_folder):
    bundle=n.Bundle(base,model); files={'native/'+key:value for key,value in bundle.files.items()}
    sources=dict(bundle.sources)
    for name,folder,module in [('short_mlp',mlp_folder,ba.bm),('short_attention',attention_folder,ba)]:
        folder=Path(folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=module.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale accuracy arm bucket build')
        if set(meta['layouts'])!={'128','256','384','512'}: raise ValueError('incomplete accuracy bucket layouts')
        files[name+'/build.json']=p.attention.sha(folder/'build.json'); sources.update(meta['sources'])
        for m,layout in meta['layouts'].items():
            for filename,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
                actual=p.attention.sha(folder/m/filename)
                if actual!=layout[key]: raise ValueError('accuracy arm artifact changed')
                files[name+'/'+m+'/'+filename]=actual
    for filename in [Path(__file__),Path(la.__file__),Path(la.lb.__file__)]:
        sources[str(filename.relative_to(p.ROOT)).replace('\\','/')]=p.attention.sha(filename)
    files.update({'source/'+key:value for key,value in sources.items()})
    files['model/laya_config.json']=p.attention.sha(model.parent/'laya_config.json')
    return bundle,files


def register(args):
    pre,items=inputs(args.items,args.reference)
    _,files=identity(args.base,args.model,args.mlp_folder,args.attention_folder)
    for filename,sha in pre['reference']['files'].items():
        if p.attention.sha(args.model.parent/filename)!=sha: raise ValueError('original reference model changed')
    arm=dict(pre)
    arm['study']='Native Laya accuracy addendum; original frozen criteria and reference unchanged'
    arm['base_preregistration']=dict(path=str(ORIGINAL.relative_to(p.ROOT)).replace('\\','/'),sha256=p.attention.sha(ORIGINAL))
    arm['registered_utc']=datetime.now(timezone.utc).isoformat()
    arm['tool']='kernels/laya_transformer/native_accuracy.py'
    arm['frozen']='before native final class outputs are scored on any labelled frozen item'
    arm['arms']={NAME:dict(files=files,what='Complete native BF16 matrix path with FP32 accumulation/residual/norm and qualified device nonlinear/attention approximations; not the prior CPU bf16 emulation')}
    arm['native_policy']=dict(batch=1,padded_tokens=512,pad_id=50283,encoder_row_bucket='ceil(actual_length/128)*128',
      attention_kv_blocks='selected_row_bucket/32',embedding_and_head_schedule=512,contexts=4,dispatches=66,
      initial_residual='device normalized embedding Y',cpu_neural_fallback=False,
      host_work='integer token/marker addresses, question type and boolean option metadata',readback='logits and act_probs only',
      learned_matrix_operands='BF16 RNE',accumulation='FP32',attention_probability_product='two BF16 parts',
      calibration='original T=1 gates; no fitting or temperature changes',frozen_accuracy_reference='original fp32.npz; not refreshed with current ORT')
    arm['native_C0_definition']='The served source/artifact map must equal this committed registration before and after scoring; scored outputs carry that same map and registration identity. Original C1-C5 are unchanged.'
    arm['development_disclosure']=dict(final_output_development='three unlabelled fixture items',
      prior_labelled_intermediates='first labelled item used for earlier intermediate activation qualification, no native final class scores',
      native_labelled_final_scores_before_registration=0,retained_failures='earlier intermediate composition and strict MLP gates remain documented')
    arm['item_buckets']={str(k):v for k,v in sorted(Counter(ba.bm.bucket(len(x['ids'])) for x in items).items())}
    args.prereg.parent.mkdir(parents=True,exist_ok=True)
    with args.prereg.open('x',encoding='utf-8',newline='\n') as output: output.write(json.dumps(arm,indent=2)+'\n')
    emit('NATIVE_ACCURACY_REGISTER',dict(name=NAME,registration_id=digest(arm),registration_file_sha256=p.attention.sha(args.prereg),
      original_prereg_sha256=arm['base_preregistration']['sha256'],items=len(items),labelled=pre['items']['labelled'],buckets=arm['item_buckets'],
      criteria=arm['criteria'],artifact_and_source_hashes=len(files),scope='registration only; no native labelled final scores'))


def registered(args,*,current=False):
    arm=json.loads(args.prereg.read_text(encoding='utf-8')); pre,items=inputs(args.items,args.reference)
    if arm['base_preregistration']['sha256']!=p.attention.sha(ORIGINAL): raise ValueError('original preregistration changed')
    for key in ('criteria','criteria_text','items','reference','decision'):
        if arm[key]!=pre[key]: raise ValueError('native addendum changes original '+key)
    if set(arm['arms'])!={NAME}: raise ValueError('wrong native arm')
    for filename,key in [(args.model.parent/'laya_config.json','model/laya_config.json'),
      (Path(__file__),'source/kernels/laya_transformer/native_accuracy.py'),
      (Path(la.__file__),'source/tools/laya_accuracy.py'),(Path(la.lb.__file__),'source/tools/laya_baseline.py')]:
        if p.attention.sha(filename)!=arm['arms'][NAME]['files'][key]: raise ValueError('registered metric/runner source changed')
    if current:
        bundle,files=identity(args.base,args.model,args.mlp_folder,args.attention_folder)
        if arm['arms'][NAME]['files']!=files: raise ValueError('registered native sources/artifacts differ')
    else: bundle=None
    return arm,items,bundle


def committed_registration(path,arm):
    relative=str(path.resolve().relative_to(p.ROOT)).replace('\\','/')
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=p.ROOT,text=True).strip()
    if not commit: raise ValueError('registration must be committed before labelled scoring')
    stored=json.loads(subprocess.check_output(['git','show',commit+':'+relative],cwd=p.ROOT,text=True))
    if canonical(stored)!=canonical(arm): raise ValueError('working registration differs from committed one')
    return commit


def save_snapshot(path,logits,act_probs,items,arm,commit,completed,complete,oracle_calls):
    # Checkpoints may replace only this run's checkpoint path; final scored files are exclusive.
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as output:
        np.savez(output,logits=logits,act_probs=act_probs,k=np.array([x['k'] for x in items]),
          items_sha256=arm['items']['file_sha256'],model=NAME,model_shas=canonical(arm['arms'][NAME]['files']),
          registration_id=digest(arm),registration_commit=commit,completed=np.int64(completed),complete=np.bool_(complete),
          cpu_neural_oracle_calls=np.int64(oracle_calls))
        output.flush()
    if path.exists() and path.name.endswith('.checkpoint.npz'): temporary.replace(path)
    else: temporary.rename(path)


def validate_scored(path,arm,items):
    with np.load(path,allow_pickle=False) as data:
        if not bool(data['complete']) or int(data['completed'])!=len(items): raise ValueError('incomplete scored arm')
        if int(data['cpu_neural_oracle_calls'])!=0: raise ValueError('CPU neural/oracle calls in scored arm')
        if str(data['registration_id'])!=digest(arm): raise ValueError('scored registration differs')
        if str(data['items_sha256'])!=arm['items']['file_sha256'] or str(data['model'])!=NAME: raise ValueError('scored item/arm identity')
        if json.loads(str(data['model_shas']))!=arm['arms'][NAME]['files']: raise ValueError('scored artifact identities differ')
        logits=data['logits'].copy(); probabilities=data['act_probs'].copy()
        if logits.shape!=(len(items),la.KMAX) or probabilities.shape!=(len(items),2): raise ValueError('scored output shapes')
        if not np.array_equal(data['k'],[x['k'] for x in items]): raise ValueError('scored option counts')
        if not np.isfinite(probabilities).all() or not all(np.isfinite(logits[j,:x['k']]).all() for j,x in enumerate(items)): raise ValueError('nonfinite scored outputs')
        if not all(np.isnan(logits[j,x['k']:]).all() for j,x in enumerate(items)): raise ValueError('unused logit padding')
    return logits


def score(args):
    if args.out.suffix!='.npz' or args.out.exists(): raise ValueError('scored output must be a new .npz file')
    checkpoint=args.out.with_name(args.out.stem+'.checkpoint.npz')
    if checkpoint.exists() or checkpoint.with_name(checkpoint.name+'.tmp').exists(): raise ValueError('checkpoint already exists; never restart over an observed run')
    arm,items,bundle=registered(args,current=True); commit=committed_registration(args.prereg,arm)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    logits=np.full((len(items),la.KMAX),np.nan,np.float32); probabilities=np.full((len(items),2),np.nan,np.float32)
    calls=[]
    def forbidden(*unused,**keywords): calls.append(True); raise AssertionError('CPU neural/oracle operation in native accuracy scoring')
    functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
      (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
      (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    emit('NATIVE_ACCURACY_START',dict(name=NAME,registration_id=digest(arm),registration_file_sha256=p.attention.sha(args.prereg),
      registration_commit=commit,items=len(items),labelled=arm['items']['labelled'],outputs=['logits','act_probs'],
      policy=arm['native_policy'],scope='full frozen accuracy only; timing ineligible'))
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)); completed=0; started=time.monotonic()
    emit('NATIVE_ACCURACY_KEEPAWAKE',dict(held=held))
    try:
        with ExitStack() as stack:
            for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
            with ba.Session(bundle,args.mlp_folder,args.attention_folder) as session:
                for j,item in enumerate(items):
                    output=session.run(item['ids'],item['markers'],item['qtype'])
                    if session.last_stats['dispatches']!=66 or len(session.states)!=4: raise ValueError('registered execution policy differs')
                    logits[j,:item['k']]=output['logits'][0]; probabilities[j]=output['act_probs'][0]; completed=j+1
                    if completed%25==0 or completed==len(items):
                        save_snapshot(checkpoint,logits,probabilities,items,arm,commit,completed,False,len(calls))
                        emit('NATIVE_ACCURACY_PROGRESS',dict(completed=completed,total=len(items),elapsed_seconds=round(time.monotonic()-started,1),
                          checkpoint_sha256=p.attention.sha(checkpoint),scope='progress only; not eligible latency'))
        current,_,_=registered(args,current=True)
        if digest(current)!=digest(arm): raise ValueError('registration changed during scoring')
        from laya_hybrid import idle
        if calls or not idle(): raise ValueError('CPU neural/oracle or device cleanup gate')
        save_snapshot(args.out,logits,probabilities,items,arm,commit,completed,True,len(calls))
        validate_scored(args.out,arm,items)
        emit('NATIVE_ACCURACY_FINISHED',dict(completed=completed,registration_id=digest(arm),output_sha256=p.attention.sha(args.out),
          cpu_neural_oracle_calls=len(calls),device_idle_while_process_alive=True,scope='full frozen outputs; criteria reported separately'))
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        emit('NATIVE_ACCURACY_KEEPAWAKE',dict(released=True,completed=completed))


def gates(arm,items,reference,observed):
    tasks=[task for task in la.PUBLISHED if any(x['task']==task for x in items)]
    by={task:[j for j,x in enumerate(items) if x['task']==task] for task in tasks}
    labelled=[j for task in tasks for j in by[task]]
    ref=la.probs(items,reference,{},False); got=la.probs(items,observed,{},False)
    rows={}
    for task,indices in by.items():
        r=la.task_metrics(items,ref,indices); a=la.task_metrics(items,got,indices)
        rows[task]=dict(n=len(indices),agreement=sum(np.argmax(ref[j])==np.argmax(got[j]) for j in indices)/len(indices),
          reference=r,observed=a,dacc=a['acc']-r['acc'],dece1=a['ece']-r['ece'])
    r=la.task_metrics(items,ref,labelled); a=la.task_metrics(items,got,labelled)
    agreement=sum(np.argmax(ref[j])==np.argmax(got[j]) for j in labelled)/len(labelled)
    c=arm['criteria']; eps=1e-9
    checks=dict(C0=True,C1=bool(agreement>=c['agree_pooled_min']-eps),C2=bool(a['acc']-r['acc']>=c['dacc_pooled_min']-eps),
      C3=all(x['dacc']>=c['dacc_task_min']-eps for x in rows.values()),C4=bool(a['ece']-r['ece']<=c['dece1_pooled_max']+eps),
      C5=all(x['dece1']<=c['dece1_task_max']+eps for x in rows.values()))
    return dict(criteria=checks,passed=all(checks.values()),labelled=len(labelled),agreement=float(agreement),
      reference=r,observed=a,dacc=a['acc']-r['acc'],dece1=a['ece']-r['ece'],tasks=rows)


def report(args):
    arm,items,_=registered(args); committed_registration(args.prereg,arm)
    observed=validate_scored(args.out,arm,items)
    with np.load(args.reference,allow_pickle=False) as data: reference=data['logits'].copy()
    # Original report functions and thresholds, with the explicit new native C0 file map.
    la.cmd_report(SimpleNamespace(model_dir=str(args.model.parent),items=str(args.items),ref=str(args.reference),arm=[str(args.out)],prereg=str(args.prereg)))
    result=gates(arm,items,reference,observed)
    emit('NATIVE_ACCURACY_VERDICT',dict(registration_id=digest(arm),output_sha256=p.attention.sha(args.out),**result))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('register','score','report'))
    for name in ('base','model','mlp-folder','attention-folder','items','reference','prereg','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args(); globals()[args.command](args)
