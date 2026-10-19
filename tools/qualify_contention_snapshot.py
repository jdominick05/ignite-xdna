"""Bounded native sensor checks: idle readout, owned CPU burn and GPU control.

No foreign workload is stopped. Both GPU sensors must identify the owned worker
while it runs a public-prompt GPU-only control, then the worker is closed.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from npu.contention import snapshot,owned_process_ids
from tools.ggml_gpu_target.client import Target
from kernels.transformer_bf16.common import sha


def main():
    if sys.argv[1:]==['--cpu-burn']:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:pass
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--llama-runtime',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    paths=[Path(__file__).resolve(),ROOT/'npu/contention.py',ROOT/'tools/check-inference-contention.ps1',
           ROOT/'tools/ggml_gpu_target/client.py',args.artifacts/'bin/ggml_gpu_target.exe']
    sources={str(p):sha(p) for p in paths}
    idle=snapshot()
    assert idle['gpu_telemetry_known'] and idle['cpu_telemetry_known'] and idle['npu_idle'],idle
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--cpu-burn'])
    try:
        time.sleep(.1);cpu=snapshot()
        assert any(p['pid']==child.pid and p['active_cpu_cores']>=.1 for p in cpu['inference_processes']),cpu
    finally:
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    # Recheck before the owned GPU work, keeping the same background limits.
    preflight=snapshot()
    owned=owned_process_ids(ROOT);preflight['owned_pids']=sorted(owned)
    foreign_cpu=[p for p in preflight['inference_processes'] if p['pid'] not in owned and p['active_cpu_cores']>=.1]
    foreign_gpu=[p for p in preflight['busy_gpu_engines'] if p['utilization_percent']>2]
    assert preflight['npu_idle'] and preflight['gpu_telemetry_known'] and not foreign_cpu and not foreign_gpu,preflight
    env=os.environ.copy();env['GGML_CUDA_DISABLE_FUSION']='1'
    errors=[]
    with Target(args.artifacts/'bin/ggml_gpu_target.exe',args.llama_runtime,
                args.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf',
                args.artifacts/'npu/live-public-dictionary-20261018.i32',env=env) as gpu:
        def run_control():
            try:
                current=gpu.initial
                for _ in range(64):current=gpu.decode([current['predictions'][-1]])
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=run_control)
        thread.start()
        try:
            native_gpu=snapshot()
            legacy=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-File',
                                   str(ROOT/'tools/check-inference-contention.ps1')],capture_output=True,text=True,check=True,timeout=15)
            legacy_gpu=json.loads(legacy.stdout)
        finally:
            thread.join(timeout=15)
        assert not thread.is_alive() and not errors,errors
        for report in (native_gpu,legacy_gpu):
            assert report['gpu_telemetry_known'],report
            assert any(p['pid']==gpu.process.pid and p['utilization_percent']>2 for p in report['busy_gpu_engines']),report
        worker_pid=gpu.process.pid
    after=snapshot()
    assert after['npu_idle'] and gpu.process.poll() is not None
    assert all(sha(Path(p))==value for p,value in sources.items())
    args.out.write_text(json.dumps(dict(passed=True,scope=__doc__,source_sha256=sources,
                                       idle=idle,cpu_burn=cpu,gpu_worker_pid=worker_pid,
                                       native_gpu=native_gpu,legacy_gpu=legacy_gpu,after=after),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,native_idle_ms=idle['snapshot_seconds']*1000,
                         native_busy_ms=native_gpu['snapshot_seconds']*1000,gpu_worker_pid=worker_pid)),flush=True)


if __name__=='__main__':main()
