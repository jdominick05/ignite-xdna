#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preregistered native/hybrid comparison; raw dispatch logs stay in scratch."""
import argparse
from contextlib import ExitStack
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'kernels/laya_transformer'))
import numpy as np
import onnx
import onnxruntime as ort
import lookup_head_accuracy as accuracy
import laya_hybrid as hybrid
op=accuracy.op; na=accuracy.na; p=accuracy.p


def emit(tag,value): print(tag+' '+json.dumps(value,allow_nan=False),flush=True)
def sha(path): return p.attention.sha(path)


def paths():
    base=ROOT/'scratch/laya_transformer'
    return SimpleNamespace(base=base,model=ROOT/'scratch/laya/laya-onnx/laya_az0_noguard.onnx',
      mlp_folder=base/'bucket_mlp_v3',attention_folder=base/'bucket_attention_v2',
      mlp_candidate=base/'mlp_fifo_v1_double_m128',attention_candidate=base/'attention_fifo_v1_double_m128',
      compact_folder=base/'compact_attention_v1',head_folder=base/'bucket_head_v2',fast_folder=base/'geglu_lut_v1_double_m128',
      items=ROOT/'scratch/laya/step2/items.jsonl',reference=ROOT/'scratch/laya/step2/fp32.npz',
      prereg=ROOT/'results/laya/lookup_head_accuracy_v1_prereg_desktop2_20261016.json',gemm_dir=ROOT/'scratch/gemm_generic/aot6')


def identity():
    args=paths(); bundle,files=accuracy.identity(args,args.base,args.model,args.mlp_folder,args.attention_folder)
    with patch.object(na,'NAME',accuracy.NAME),patch.object(na,'identity',lambda *unused:(bundle,files)):
        arm,items,_=na.registered(args,current=True); na.committed_registration(args.prereg,arm)
    if hybrid.SEQ!=512: raise ValueError('hybrid support audit changed')
    relative=lambda file:str(file.relative_to(ROOT)).replace('\\','/')
    for file in (Path(__file__),ROOT/'tools/laya_hybrid.py',ROOT/'tools/laya_baseline.py',ROOT/'tools/research_process.py',ROOT/'scripts/research-lowlevel.sh'):
        files['comparison/'+relative(file)]=sha(file)
    files['qualified_lookup_registration']=sha(args.prereg)
    files['qualified_lookup_scores']=sha(ROOT/'scratch/laya/native_accuracy/native_lookup_short_heads_bf16_v1.npz')
    files['qualified_buffered_registration']=sha(ROOT/'results/laya/buffered_accuracy_v1_prereg_desktop2_20261016.json')
    files['qualified_buffered_scores']=sha(ROOT/'scratch/laya/native_accuracy/native_buffered_m128_bf16_v1.npz')
    for K,N in sorted(hybrid.GEMM_SHAPES):
        file=args.gemm_dir/f'g_{K}_{N}.bin'; files['hybrid/'+file.name]=sha(file)
    file=args.gemm_dir/'g_1024_3072.xclbin'; files['hybrid/'+file.name]=sha(file)
    model=onnx.load(str(args.model),load_external_data=False); init={value.name:value for value in model.graph.initializer}
    products=[model.graph.node[index] for index in hybrid.la.weight_products(model) if model.graph.node[index].op_type=='MatMul']
    for value in model.graph.input:
        for index,dim in enumerate(value.type.tensor_type.shape.dim):
            dim.dim_value=1 if index==0 else 512 if value.name in ('input_ids','attention_mask') else len(items[0]['markers'])
    del model.graph.value_info[:]
    inferred=onnx.shape_inference.infer_shapes(model)
    shapes={value.name:[dim.dim_value if dim.HasField('dim_value') else None for dim in value.type.tensor_type.shape.dim]
      for value in list(inferred.graph.value_info)+list(inferred.graph.input)}
    rejected=[]; npu=0
    for node in products:
        shape=shapes.get(node.input[0]); known_rows=int(np.prod(shape[:-1])) if shape and all(value is not None for value in shape[:-1]) else None
        if tuple(init[node.input[1]].dims) not in hybrid.GEMM_SHAPES or (known_rows is not None and known_rows!=512): rejected.append(node.name)
        else: npu+=1
    support=dict(seq=512,weight_matmuls=len(products),npu_products=npu,host_products=len(products)-npu,
      shorter_bucket_supported=False,
      predicted_host_product_nodes=rejected,
      placement_gate='static shape prediction; actual per-call NPU/host counters must match, including warmups, before accepting results',
      evidence='laya_hybrid.SEQ=512, fixed A/C allocations and x.shape[0]!=SEQ host branch; no new transaction compilation or SEQ patch',
      matched_control='same 91-token fixture padded to512, with native baseline=True for all512 schedules',
      serving_comparison='native actual request uses M128; current hybrid closest supported schedule is512; no claim of matched M128 kernels')
    if len(products)!=120: raise ValueError('hybrid product graph changed')
    return args,bundle,files,items,support


def make_order(rounds):
    names=['hybrid512','native_full512','native_buffered','native_lookup']
    return [names[index:]+names[:index] for index in range(rounds)]


def register(args):
    _,_,files,items,support=identity()
    if not (1<=args.rounds<=4 and 3<=args.reps<=10): raise ValueError('bounded protocol')
    item=items[0]
    if item['task']!='fixture' or item['gold']!=-1 or len(item['ids'])!=91: raise ValueError('comparison fixture changed')
    pre=dict(version=1,files=files,fixture=item,support=support,
      timing=dict(rounds=args.rounds,reps=args.reps,cpu_reps=3,warmups=2,order=make_order(args.rounds)),
      qualification=dict(rounds=1,reps=1,cpu_reps=1,warmups=1,order=make_order(1)),
      cpu=dict(configs=[dict(name='one_no_spin',threads=1,spinning=False),dict(name='eight_no_spin',threads=8,spinning=False),
        dict(name='eight_default',threads=8,spinning=None)],schedules=['actual91','padded512'],
        interleaving='one CPU block before the first NPU profile and after every profile; config and schedule order rotate',
        sessions='fresh sessions per config/block; release before NPU; hybrid host pools retained with spinning disabled',
        stability_max_median_ratio=1.10,ratio_policy='withhold all NPU/CPU ratios if any CPU config/schedule exceeds stability threshold',
        load='system CPU utilization sampled every100ms; process CPU seconds and thread count; no inferred clock or thermal cause'),
      correctness=dict(native_cpu_neural_calls=0,native_contexts=5,native_dispatches=66,
        logits_and_action_probs='existing development numeric gates, finite/shape checks and same choice against first CPU actual-length reference',
        atol=p.attention.ATOL,rtol=p.attention.RTOL,relative_l2=p.attention.L2),
      comparison_priority=['native versus hybrid with matched512 schedule','native served configurations on same actual91 request, hybrid pads512',
        'CPU secondary, conditional on drift investigation'],
      disclosure='native and hybrid historical timings and full accuracy outcomes already observed; new performance comparison, no accuracy threshold changes',
      outputs='summaries and failures in results; verbose raw logs in git-ignored scratch, SHA256 in summary',
      device_policy='one profile at a time; idle before and after every profile; at most five contexts; outer process timeout and per-dispatch waits')
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file: file.write(json.dumps(pre,indent=2)+'\n')
    emit('COMPARE_REGISTER',dict(prereg=str(args.prereg.relative_to(ROOT)),sha256=sha(args.prereg),identity=na.digest(pre),support=support,protocol=pre['timing']))


def load_registration(path):
    pre=json.loads(path.read_text()); args,bundle,files,items,support=identity()
    if pre['files']!=files or pre['fixture']!=items[0] or pre['support']!=support: raise ValueError('comparison registration differs from current artifacts')
    na.committed_registration(path,pre)
    return pre,args,bundle,items[:3]


def stats(values): return dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)


def system_times():
    idle=wintypes.FILETIME(); kernel=wintypes.FILETIME(); user=wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user)): raise OSError('GetSystemTimes failed')
    value=lambda field:(field.dwHighDateTime<<32)|field.dwLowDateTime
    return value(kernel)+value(user),value(idle)


def native_thread_count():
    class Entry(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('thread',wintypes.DWORD),('owner',wintypes.DWORD),
          ('base',wintypes.LONG),('delta',wintypes.LONG),('flags',wintypes.DWORD)]
    api=ctypes.windll.kernel32
    api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    api.Thread32First.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    api.Thread32Next.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.CreateToolhelp32Snapshot(4,0)
    if handle==ctypes.c_void_p(-1).value: raise OSError('thread snapshot failed')
    try:
        entry=Entry(); entry.size=ctypes.sizeof(entry); more=api.Thread32First(handle,ctypes.byref(entry)); count=0
        while more:
            count+=entry.owner==os.getpid(); more=api.Thread32Next(handle,ctypes.byref(entry))
        return count
    finally: api.CloseHandle(handle)


def timed(function,reps,warmups):
    for _ in range(warmups): function()
    stop=threading.Event(); loads=[]; first=system_times()
    def sample():
        previous=first
        while not stop.wait(.1):
            current=system_times(); total=current[0]-previous[0]; idle=current[1]-previous[1]
            if total>0: loads.append(100*(total-idle)/total)
            previous=current
    thread=threading.Thread(target=sample,daemon=True); thread.start()
    threads=native_thread_count(); before=time.process_time(); started=time.perf_counter()
    values=[]; output=None
    try:
        for _ in range(reps):
            tick=time.perf_counter(); output=function(); values.append((time.perf_counter()-tick)*1000)
    finally:
        wall=time.perf_counter()-started; after=time.process_time(); stop.set(); thread.join()
    seconds=after-before
    return output,dict(timing=stats(values),load=dict(system_percent_samples=loads,
      process_cpu_seconds=seconds,wall_seconds=wall,derived_process_cores=seconds/wall,threads=threads))


def check(outputs,reference):
    comparisons={}; choices={}
    for name in ('logits','act_probs'):
        value=np.asarray(outputs[name]); expected=np.asarray(reference[name])
        if value.shape!=expected.shape or not np.isfinite(value).all(): raise ValueError('nonfinite or incorrect output shape: '+name)
        comparisons[name]=p.attention.numeric(value,expected)
        choices[name]=bool(np.argmax(value)==np.argmax(expected))
    emit('COMPARE_CHECK',dict(comparisons=comparisons,same_choice=choices))
    if not all(row['ok'] for row in comparisons.values()) or not all(choices.values()): raise ValueError('comparison correctness gate')


def cpu_block(index,pre,args,protocol,reference):
    item=pre['fixture']; feeds=dict(actual91=hybrid.la.feed(item['ids'],item['markers'],item['qtype']),
      padded512=hybrid.padded_feed(item['ids'],item['markers'],item['qtype']))
    configs=pre['cpu']['configs']; configs=configs[index%len(configs):]+configs[:index%len(configs)]
    rows=[]
    for config in configs:
        options=ort.SessionOptions(); options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads=config['threads']; options.inter_op_num_threads=1
        if config['spinning'] is False: options.add_session_config_entry('session.intra_op.allow_spinning','0')
        session=ort.InferenceSession(str(args.model.parent/'laya.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        names=[out.name for out in session.get_outputs()]
        schedules=pre['cpu']['schedules'][::(-1 if index%2 else 1)]
        for schedule in schedules:
            output,result=timed(lambda:dict(zip(names,session.run(None,feeds[schedule]))),protocol['cpu_reps'],protocol['warmups'])
            if reference is None: reference={key:value.copy() for key,value in output.items()}
            check(output,reference)
            row=dict(block=index,config=config['name'],schedule=schedule,**result); rows.append(row); emit('COMPARE_CPU',row)
        session=None; gc.collect()
    return rows,reference


def stability(rows,limit):
    groups={}
    for row in rows: groups.setdefault((row['config'],row['schedule']),[]).append(row['timing']['median_ms'])
    return {config+'/'+schedule:dict(block_medians_ms=values,derived_max_over_min=max(values)/min(values),stable=max(values)/min(values)<=limit)
      for (config,schedule),values in groups.items()}


def run(args):
    pre,paths_,bundle,fixtures=load_registration(args.prereg)
    protocol=pre['qualification' if args.qualification else 'timing']
    emit('COMPARE_START',dict(prereg_sha256=sha(args.prereg),identity=na.digest(pre),qualification=args.qualification,
      ort=ort.__version__,cpu_monitor='Win32 GetSystemTimes, time.process_time and Toolhelp32 thread snapshot',protocol=protocol,support=pre['support']))
    plan=hybrid.Plan(paths_.model,threads=8)
    reference=None; cpu_rows=[]; profiles=[]; index=0
    rows,reference=cpu_block(index,pre,paths_,protocol,reference); cpu_rows+=rows
    item=pre['fixture']; feed=hybrid.padded_feed(item['ids'],item['markers'],item['qtype'])
    for round_index,order in enumerate(protocol['order']):
        for name in order:
            if not hybrid.idle(): raise ValueError('NPU not idle before profile')
            emit('COMPARE_IDLE',dict(phase='before',round=round_index,arm=name,idle=True))
            if name=='hybrid512':
                products=hybrid.NpuProducts(plan,paths_.gemm_dir)
                try:
                    count=0
                    original=ort.InferenceSession.run
                    def host_run(session,*a,**kw):
                        nonlocal count
                        count+=1; return original(session,*a,**kw)
                    with patch.object(ort.InferenceSession,'run',host_run):
                        output,result=timed(lambda:dict(zip(plan.graph_outputs,plan.run(feed,products))),protocol['reps'],protocol['warmups'])
                    calls=protocol['reps']+protocol['warmups']; expected=pre['support']
                    if products.counts!={'npu':calls*expected['npu_products'],'host':calls*expected['host_products']}: raise ValueError('hybrid product placement changed')
                    policy=dict(contexts=1,product_counts=products.counts.copy(),host_segment_calls=count,calls_including_warmups=calls)
                finally: products.close()
                products=None; gc.collect()
            else:
                calls=[]
                def forbidden(*unused,**kw): calls.append(True); raise AssertionError('CPU neural/oracle call during native inference')
                n=op.n
                functions=[(ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
                  (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
                  (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
                common=(bundle,paths_.mlp_folder,paths_.attention_folder,paths_.mlp_candidate,paths_.attention_candidate,paths_.compact_folder,fixtures)
                with ExitStack() as stack:
                    for module,method in functions: stack.enter_context(patch.object(module,method,forbidden))
                    if name=='native_buffered': session=op.hp.buffered.Session(*common)
                    elif name=='native_full512': session=op.hp.Session(*common,paths_.head_folder)
                    else: session=op.Session(*common,paths_.head_folder,fast_folder=paths_.fast_folder,mlp_kind='lookup')
                    with session:
                        kwargs=dict(baseline=True) if name=='native_full512' else {}
                        output,result=timed(lambda:session.run(item['ids'],item['markers'],item['qtype'],**kwargs),protocol['reps'],protocol['warmups'])
                        policy=dict(contexts=len(session.states),dispatches=session.last_stats['dispatches'],cpu_neural_oracle_calls=len(calls))
                        if policy!=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0): raise ValueError('native execution policy changed')
                session=None; gc.collect()
            if not hybrid.idle(): raise ValueError('NPU not idle after profile')
            check(output,reference)
            row=dict(round=round_index,arm=name,**result,policy=policy,idle_after_close=True); profiles.append(row); emit('COMPARE_PROFILE',row)
            index+=1; rows,reference=cpu_block(index,pre,paths_,protocol,reference); cpu_rows+=rows
    plan=None; gc.collect()
    load_registration(args.prereg)
    drift=stability(cpu_rows,pre['cpu']['stability_max_median_ratio'])
    native={name:stats([sample for row in profiles if row['arm']==name for sample in row['timing']['samples_ms']]) for name in protocol['order'][0]}
    emit('COMPARE_RESULT',dict(profile_aggregate=native,cpu_stability=drift,cpu_ratios_withheld=not all(row['stable'] for row in drift.values()),
      matched512_native_beats_hybrid=native['native_full512']['median_ms']<native['hybrid512']['median_ms'],
      served_native_beats_current_hybrid=native['native_lookup']['median_ms']<native['hybrid512']['median_ms'],
      serving_schedule_caveat=pre['support']['serving_comparison'],device_idle=hybrid.idle(),
      cpu_cause='unexplained unless separately established; load samples do not establish clocks or thermals'))


def summarize(args):
    pre=json.loads(args.prereg.read_text()); lines=args.raw.read_text(encoding='utf-8').splitlines()
    emit('COMPARE_RAW_EVIDENCE',dict(raw_path=str(args.raw.relative_to(ROOT)),raw_sha256=sha(args.raw),
      prereg_path=str(args.prereg.relative_to(ROOT)),prereg_sha256=sha(args.prereg),identity=na.digest(pre)))
    tags=('RESEARCH_PLATFORM ','PRE_NPU_CONTEXT_WITNESS ','POST_NPU_CONTEXT_WITNESS ','RESEARCH_PROCESS_RESULT ','COMPARE_')
    for line in lines:
        if line.startswith(tags): print(line,flush=True)
    terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
    starts=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('COMPARE_START ')]
    completed=any(line.startswith('COMPARE_RESULT ') for line in lines)
    accepted=bool(completed and terminal and terminal[-1]['exit_code']==0 and terminal[-1]['timing_eligible'] and
      starts and not starts[-1]['qualification'] and starts[-1]['prereg_sha256']==sha(args.prereg))
    emit('COMPARE_SUMMARY_STATUS',dict(terminal=terminal[-1] if terminal else None,
      completed=completed,accepted_timing=accepted,
      note='raw log is retained in scratch; failure tracebacks remain in raw evidence, correctness failures prevent successful completion'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    registration=sub.add_parser('register'); registration.add_argument('--prereg',type=Path,required=True)
    registration.add_argument('--rounds',type=int,default=3); registration.add_argument('--reps',type=int,default=5)
    execution=sub.add_parser('run'); execution.add_argument('--prereg',type=Path,required=True); execution.add_argument('--qualification',action='store_true')
    summary=sub.add_parser('summarize'); summary.add_argument('--prereg',type=Path,required=True); summary.add_argument('--raw',type=Path,required=True)
    args=parser.parse_args()
    args.prereg=args.prereg.absolute()
    if args.command=='summarize': args.raw=args.raw.absolute()
    try: globals()[args.command](args)
    except Exception as error:
        emit('COMPARE_FAILURE',dict(type=type(error).__name__,message=str(error)))
        raise
