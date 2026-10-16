#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serial ABBA comparison of registered persistent GEMM implementations."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args,cwd):
    return subprocess.check_output([str(x) for x in args],cwd=cwd,text=True).strip()


def register(args):
    import program as p
    base=args.baseline.resolve(); candidate=args.candidate.resolve(); base_root=args.baseline_root.resolve()
    if args.registration.exists(): raise ValueError('registration already exists')
    commit=command(['git','rev-parse','HEAD'],base_root)
    if not commit.startswith('14c55c9') or command(['git','status','--porcelain'],base_root):
        raise ValueError('baseline must be a clean checkout of the registered control commit')
    arms={}
    for name,folder,root,layout in [('baseline',base,base_root,'gemm'),
      ('candidate',candidate,ROOT,args.candidate_layout)]:
        meta=json.loads((folder/'build.json').read_text())
        for path,digest in meta['sources'].items():
            if sha(root/path)!=digest: raise ValueError('stale '+name+' source '+path)
        lm=meta['layouts'][layout]
        shape=lm.get('shape') or meta['gemm_shape']
        if list(shape)!=[512,1024,1024]: raise ValueError('comparison fixes M512,K1024,N1024')
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if sha(folder/layout/file)!=lm[key]: raise ValueError('changed artifact')
        arms[name]=dict(root=str(root),build=str(folder),layout=layout,metadata_sha256=sha(folder/'build.json'),
                       source_hashes=meta['sources'],image=lm['xclbin'],instructions=lm['instructions'])
    args.registration.parent.mkdir(parents=True,exist_ok=True)
    a,b=p.gemm_case(11,'random',1024,1024)
    fixture=args.registration.with_suffix('.npz')
    np.savez(fixture,a=a,b=b,reference=a.astype(np.float64)@b.astype(np.float64))
    reg=dict(arms=arms,script_sha256=sha(__file__),baseline_commit=commit,shape=[512,1024,1024],
      fixture_sha256=sha(fixture),fixture=fixture.name,blocks=args.blocks,reps=args.reps,warmups=3,
      poisons=[0x7fc12345,0x7fc54321],gates=dict(atol=.01,rtol=.02,l2=.02),
      order=['baseline','candidate','candidate','baseline'],
      targets=dict(boundary_reduction=.20,dispatch_reduction=.30,positive_boundary_ci=True),
      scope='one bf16 GEMM, packing through detached output; not complete Laya; contexts serialized, creation excluded')
    args.registration.write_text(json.dumps(reg,indent=2)+'\n',encoding='utf-8')
    print('GEMM_PAIR_REGISTER '+json.dumps(dict(registration_sha256=sha(args.registration),baseline_commit=commit,
      shape=reg['shape'],blocks=reg['blocks'],reps=reg['reps'],targets=reg['targets'],scope=reg['scope'])),flush=True)


def validate(registration):
    reg=json.loads(registration.read_text())
    if reg['script_sha256']!=sha(__file__): raise ValueError('changed benchmark script')
    if sha(registration.parent/reg['fixture'])!=reg['fixture_sha256']: raise ValueError('fixture changed')
    for name,arm in reg['arms'].items():
        root=Path(arm['root']); folder=Path(arm['build'])
        if sha(folder/'build.json')!=arm['metadata_sha256']: raise ValueError('build metadata changed')
        for path,digest in arm['source_hashes'].items():
            if sha(root/path)!=digest: raise ValueError('changed '+name+' source')
        for file,key in [('final.xclbin','image'),('insts.bin','instructions')]:
            if sha(folder/arm['layout']/file)!=arm[key]: raise ValueError('changed artifact')
    return reg


def worker(registration,name):
    reg=validate(registration); arm=reg['arms'][name]
    root=Path(arm['root']); folder=Path(arm['build']); layout=arm['layout']
    sys.path.insert(0,str(root/'kernels/laya_transformer'))
    import program as p
    sys.path.insert(0,str(root/'tools'))
    from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign NPU context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python')
    import pyxrt as xrt
    fixture=np.load(registration.parent/reg['fixture']); ga,gb,reference=[fixture[n] for n in ('a','b','reference')]
    meta=json.loads((folder/'build.json').read_text())['layouts'][layout]
    used=512 if name=='baseline' else 2048
    aq=32*16 if name=='baseline' else 16*16
    words=np.frombuffer((folder/layout/'insts.bin').read_bytes(),np.uint32)
    dev=xrt.device(0); xb=xrt.xclbin(str(folder/layout/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    S=xrt.xclBOSyncDirection
    ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(words.tobytes(),0); ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    a=xrt.bo(dev,meta['input_words']*4,xrt.bo.host_only,0); o=xrt.bo(dev,meta['output_words']*4,xrt.bo.host_only,0)
    av=np.frombuffer(a.map(),np.uint32); ov=np.frombuffer(o.map(),np.uint32).reshape(-1,2080)
    h=None; previous=None; checks=[]
    def dispatch():
        handle=kernel(3,ib,words.nbytes,a,o)
        if handle.wait(5000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('dispatch failure; no retry')
    try:
        for poison in reg['poisons']:
            payload=p.gemm_packet(ga,gb,poison)
            if len(payload)!=len(av): raise ValueError('packet geometry')
            np.copyto(av,payload); ov.fill(poison)
            a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE); o.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
            dispatch(); o.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
            raw=ov[:,16:16+used].view(np.float32).copy(); got=p.gemm_output(raw,1024)
            err=np.abs(got-reference); l2=float(np.sqrt(np.sum(err*err)/np.sum(reference*reference)))
            ok=np.isfinite(got).all() and np.all(err<=.01+.02*np.abs(reference)) and l2<=.02
            ok=ok and np.all(ov[:,:16]==poison) and np.all(ov[:,16+used:]==poison) and not np.any(ov[:,16:16+used]==poison)
            ok=ok and np.array_equal(av,payload)
            if previous is not None: ok=ok and np.array_equal(got,previous)
            if not ok: raise ValueError('qualification gate')
            checks.append(dict(poison=poison,maxabs=float(err.max()),relative_l2=l2,passed=True)); previous=got
        dispatch_ms=[]; boundary_ms=[]
        for i in range(reg['warmups']+reg['reps']):
            start=time.perf_counter_ns(); dispatch()
            if i>=reg['warmups']: dispatch_ms.append((time.perf_counter_ns()-start)/1e6)
        for i in range(reg['warmups']+reg['reps']):
            start=time.perf_counter_ns()
            payload=p.gemm_packet(ga,gb,reg['poisons'][-1]); np.copyto(av,payload)
            a.sync(S.XCL_BO_SYNC_BO_TO_DEVICE); dispatch(); o.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
            detached=p.gemm_output(ov[:,16:16+used].view(np.float32).copy(),1024)
            # The result owns storage independently of the next dispatch.
            if not detached.flags.c_contiguous: detached=np.ascontiguousarray(detached)
            if i>=reg['warmups']: boundary_ms.append((time.perf_counter_ns()-start)/1e6)
        print('GEMM_PAIR_WORKER '+json.dumps(dict(arm=name,checks=checks,dispatch_ms=dispatch_ms,boundary_ms=boundary_ms,
          input_bytes=av.nbytes,output_bytes=ov.nbytes,contexts_created=1)),flush=True)
    finally:
        h=None
        del av,ov,a,o,ib,kernel,ctx,xb,dev
        gc.collect()
        if not idle(): raise RuntimeError('context not released while alive')


def run(registration):
    reg=validate(registration)
    print('GEMM_PAIR_RUN '+json.dumps(dict(registration_sha256=sha(registration),baseline_commit=reg['baseline_commit'],
      shape=reg['shape'],blocks=reg['blocks'],reps=reg['reps'],targets=reg['targets'],scope=reg['scope'])),flush=True)
    blocks=[]
    for i in range(reg['blocks']):
        samples=[]
        for name in reg['order']:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--registration',str(registration),'--arm',name],
              capture_output=True,text=True,timeout=45)
            if result.returncode:
                print(result.stdout+result.stderr,flush=True)
                raise RuntimeError('worker failed, stopping comparison')
            row=next(json.loads(line.split(' ',1)[1]) for line in result.stdout.splitlines() if line.startswith('GEMM_PAIR_WORKER '))
            samples.append(row)
        print('GEMM_PAIR_BLOCK '+json.dumps(dict(block=i,samples=samples)),flush=True)
        blocks.append(samples)
    summary={}
    for metric in ('dispatch_ms','boundary_ms'):
        arms={name:[statistics.mean([v for row in block if row['arm']==name for v in row[metric]]) for block in blocks]
              for name in ('baseline','candidate')}
        deltas=[a-b for a,b in zip(arms['baseline'],arms['candidate'])]
        saving=statistics.mean(deltas); margin=1.96*statistics.stdev(deltas)/np.sqrt(len(deltas))
        baseline=statistics.mean(arms['baseline']); candidate=statistics.mean(arms['candidate'])
        summary[metric]=dict(baseline_mean=baseline,candidate_mean=candidate,saving=saving,reduction=1-candidate/baseline,
          saving_ci95_normal_approx=[saving-margin,saving+margin])
    passed=summary['boundary_ms']['reduction']>=.20 and summary['dispatch_ms']['reduction']>=.30 and summary['boundary_ms']['saving_ci95_normal_approx'][0]>0
    print('GEMM_PAIR_RESULT '+json.dumps(dict(blocks=len(blocks),contexts_serialized=True,summary=summary,target_pass=bool(passed),
      scope=reg['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('register','run','worker'))
    ap.add_argument('--registration',type=Path,required=True); ap.add_argument('--baseline',type=Path)
    ap.add_argument('--baseline-root',type=Path); ap.add_argument('--candidate',type=Path)
    ap.add_argument('--candidate-layout',default='gemm_512_1024_1024'); ap.add_argument('--blocks',type=int,default=16)
    ap.add_argument('--reps',type=int,default=8); ap.add_argument('--arm',choices=('baseline','candidate')); args=ap.parse_args()
    if args.blocks<2 or args.reps<1: ap.error('need at least two blocks and one repetition')
    if args.command=='register': register(args)
    elif args.command=='run': run(args.registration.resolve())
    else: worker(args.registration.resolve(),args.arm)
