"""Compare a timed GPU worker with the preserved binary, then profile decode.

Fixed public tokens, FP32 state hashes and greedy predictions are compared for
identical batch shapes across binaries. This is diagnostic, not a speed gate.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import sha,save
from npu.gemma4.runtime import admit_hardware,HardwareBusy
from tools.ggml_gpu_target.client import Target


def fingerprint(response):
    return dict(position=response['position'],predictions=response['predictions'],
        tensors={k:dict(shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())
                 for k,v in sorted(response['tensors'].items())})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    progress=a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():raise FileExistsError(a.out)
    prompt=a.artifacts/'npu/live-public-dictionary-20261018.i32'
    model=a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    paths=[Path(__file__).resolve(),ROOT/'tools/ggml_gpu_target/worker.cpp',ROOT/'tools/ggml_gpu_target/client.py',
           ROOT/'npu/gemma4/runtime.py',ROOT/'npu/contention.py',a.control,a.candidate,prompt,model,
           a.control.with_name('worker.cpp'),a.control.with_name('client.py'),ROOT/'tools/ggml_gpu_target/build.cmd']
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    sources={str(path.resolve()):sha(path) for path in paths}
    ids=np.fromfile(prompt,np.int32).tolist()
    env=os.environ.copy();env['GGML_CUDA_DISABLE_FUSION']='1'
    rows=[];preflights=[];reference=[];initial=None;workers=[];rejections=[]
    def admit(gpu=None):
        deadline=time.monotonic()+60
        while True:
            try:return admit_hardware(gpu)
            except HardwareBusy as error:
                rejections.append(error.report)
                print(json.dumps(dict(admission_rejected=error.report)),flush=True)
                if time.monotonic()>=deadline:raise
                time.sleep(1)
    try:
        for label,exe in (('control',a.control),('candidate',a.candidate)):
            preflights.append(admit())
            with Target(exe,a.runtime,model,prompt,env=env) as gpu:
                workers.append(gpu.process)
                if label=='control':
                    initial=fingerprint(gpu.initial);current=gpu.initial
                    for i in range(33):
                        token=current['predictions'][-1];reference.append(token)
                        if i<32:current=gpu.decode([token])
                else:assert fingerprint(gpu.initial)==initial
                for repeat in range(2):
                    for batch in (1,2,3,5):
                        preflights.append(admit(gpu))
                        current=gpu.prepare(ids)
                        assert current['predictions'][-1]==reference[0]
                        before=gpu.timing() if label=='candidate' else None
                        snapshots=[];start=time.perf_counter()
                        for offset in range(0,32,batch):
                            tokens=reference[offset:min(offset+batch,32)]
                            decoded=gpu.decode(tokens)
                            assert decoded['predictions']==reference[offset+1:offset+1+len(tokens)]
                            snapshots.append(fingerprint(decoded))
                        loop_ms=(time.perf_counter()-start)*1000
                        after=gpu.timing() if label=='candidate' else None
                        delta=None
                        if before is not None:
                            old=before['groups'].get('1',{})
                            delta={k:v-old.get(k,0) for k,v in after['groups']['1'].items()}
                            assert delta['tokens']==32 and delta['calls']==len(snapshots)
                            assert delta['capture_ms']<=delta['process_ms']
                        row=dict(binary=label,repeat=repeat,batch=batch,snapshots=snapshots,
                                 loop_ms=loop_ms,worker_decode=delta,gpu_greedy_equal=True)
                        if label=='candidate':
                            control=next(r for r in rows if r['binary']=='control' and r['repeat']==repeat and r['batch']==batch)
                            assert row['snapshots']==control['snapshots']
                            row['control_state_bitwise_equal']=True
                        rows.append(row)
                        print(json.dumps({k:v for k,v in row.items() if k!='snapshots'}),flush=True)
        assert all(worker.poll() is not None for worker in workers)
        assert all(sha(path)==expected for path,expected in sources.items())
        save(a.out,dict(passed=True,scope=__doc__,timing_eligible=False,source_sha256=sources,cases=rows,
                        preflights=preflights,rejections=rejections,reference=reference,workers_closed=True))
    finally:
        save(progress,dict(passed=False,scope='Partial progress only; final report establishes completion',
                           source_sha256=sources,cases=rows,preflights=preflights,rejections=rejections,reference=reference,
                           workers_closed=all(worker.poll() is not None for worker in workers)))


if __name__=='__main__':main()
