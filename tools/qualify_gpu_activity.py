"""Bounded Phoenix activity helper smoke check; no model inference or speed claim."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import sha,save
from npu.gemma4.runtime import admit_hardware
from tools.ggml_gpu_target.activity import ActivityHelper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    paths=[Path(__file__).resolve(),a.helper,ROOT/'tools/ggml_gpu_target/gpu_activity.cpp',
           ROOT/'tools/ggml_gpu_target/activity.py',ROOT/'tools/ggml_gpu_target/build_activity.cmd',
           ROOT/'npu/gemma4/runtime.py',ROOT/'npu/contention.py',Path('C:/Windows/System32/d3dcompiler_47.dll'),
           a.runtime/'amdhip64_7.dll']
    sources={str(path.resolve()):sha(path) for path in paths};preflight=admit_hardware()
    with ActivityHelper(a.helper,a.runtime) as activity:
        process=activity.process;identity=activity.identity;rows=[]
        rejected=[]
        for invalid in (0,101,-1,True,1.5):
            try:activity.start(invalid)
            except ValueError:rejected.append(invalid)
            else:raise AssertionError('Unbounded activity accepted')
        for milliseconds in (1,40,100):
            activity.start(milliseconds);row=activity.wait();row['requested_ms']=milliseconds;rows.append(row)
        activity.start(100);time.sleep(.02);row=activity.wait(stop=True);row['stopped_early']=True;rows.append(row)
        assert process.poll() is None
    assert process.poll()==0 and all(sha(path)==expected for path,expected in sources.items())
    with ActivityHelper(a.helper,a.runtime) as activity:
        cancel_process=activity.process;activity.start(100)
    assert cancel_process.poll()==0
    save(a.out,dict(passed=True,scope=__doc__,timing_eligible=False,source_sha256=sources,
                    preflight=preflight,identity=identity,cases=rows,rejected=rejected,worker_closed=True,cancel_worker_closed=True))
    print('PASS: bounded GPU activity, exact integer results and clean exit',flush=True)


if __name__=='__main__':main()
