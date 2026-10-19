"""Check real streamed cancellation during activity and recovery in one worker.

Separate GPU controls verify completed requests. Serving performs no replay.
This is bounded functionality, not energy or unattended-operation qualification.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Ignition/src'));sys.path.insert(0,str(ROOT))
from ignition.experimental import HybridAssistant
from kernels.transformer_bf16.common import sha,save
from npu.gemma4.runtime import HardwareBusy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();progress=a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():raise FileExistsError(a.out)
    prompt_path=a.artifacts/'npu/live-public-dictionary-20261018.i32'
    prompt=np.fromfile(prompt_path,np.int32).tolist()
    sources={str(path.resolve()):sha(path) for path in (Path(__file__).resolve(),prompt_path,ROOT/'Ignition/src/ignition/experimental/hybrid.py')}
    rows=[];rejections=[];worker=activity_worker=None
    try:
        with HybridAssistant(study_root=ROOT,artifacts=a.artifacts,llama_runtime=a.runtime,
                             input_depth=2,vector_copy=True,collect_trace=True,gpu_activity_helper=a.helper) as assistant:
            backend=assistant._backend
            for mode,limit in (('complete',8),('cancel_active',64),('resume',16)):
                deadline=time.monotonic()+60;seen=threading.Event()
                while True:
                    stop=threading.Event();watcher=None;chunks=[]
                    stream=assistant.stream_tokens(prompt,max_new_tokens=limit,depth=2)
                    try:
                        def observe():
                            while not stop.is_set():
                                activity=backend._activity
                                if activity is not None and activity.active:
                                    seen.set();assistant.cancel();return
                                stop.wait(.002)
                        for chunk in stream:
                            chunks.append(chunk)
                            if mode=='cancel_active' and watcher is None:
                                watcher=threading.Thread(target=observe,daemon=True);watcher.start()
                        break
                    except HardwareBusy as error:
                        rejections.append(dict(mode=mode,preflight=error.report))
                        if time.monotonic()>=deadline:raise
                        time.sleep(1)
                    finally:
                        stop.set()
                        if watcher is not None:watcher.join(timeout=2);assert not watcher.is_alive()
                        stream.close()
                result=dict(assistant.last_result)
                assert result['npu_released'] and result['npu_idle'] and result['gpu_activity']['idle']
                assert result['gpu_activity']['windows'] and all(w['gpu_validated'] for w in result['gpu_activity']['windows'])
                assert backend.model_loads==1 and backend.gpu.process.poll() is None and backend._activity.process.poll() is None
                if worker is None:worker=backend.gpu.process;activity_worker=backend._activity.process
                assert worker is backend.gpu.process and activity_worker is backend._activity.process
                row=dict(mode=mode,result=result,cancel_observed_active=seen.is_set())
                rows.append(row)
                save(progress,dict(passed=False,source_sha256=sources,cases=rows,rejections=rejections))
                if mode=='cancel_active':
                    assert seen.is_set() and result['cancelled'] and not result['completed']
                else:
                    assert result['completed']
                    deadline=time.monotonic()+60
                    while True:
                        try:row['control_preflight']=backend._preflight();break
                        except HardwareBusy as error:
                            rejections.append(dict(mode=mode,phase='control',preflight=error.report))
                            if time.monotonic()>=deadline:raise
                            time.sleep(1)
                    current=backend.gpu.prepare(prompt);reference=[]
                    for i in range(len(result['tokens'])):
                        token=current['predictions'][-1];reference.append(token)
                        if i+1<len(result['tokens']):current=backend.gpu.decode([token])
                    assert reference==result['tokens']==[t for c in chunks for t in c.token_ids]
                    row['gpu_control_equal']=True
                print(mode+': PASS, NPU released and activity idle',flush=True)
        assert worker.poll() is not None and activity_worker.poll()==0
        assert all(sha(path)==expected for path,expected in sources.items())
        save(a.out,dict(passed=True,scope=__doc__,source_sha256=sources,timing_eligible=False,cases=rows,
                        rejections=rejections,worker_closed=True,activity_worker_closed=True))
    finally:
        save(progress,dict(passed=False,scope='Partial qualification only',source_sha256=sources,cases=rows,rejections=rejections,
                           worker_closed=worker is None or worker.poll() is not None,
                           activity_worker_closed=activity_worker is None or activity_worker.poll() is not None))


if __name__=='__main__':main()
