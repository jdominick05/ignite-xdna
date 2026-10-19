"""Isolate GPU decode costs with resident or interleaved BF16 NPU drafting.

GPU inputs are fixed checked reference tokens; NPU proposals are discarded.
This measures coexistence diagnostics, not speculative decoding speed or power.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from contextlib import ExitStack,nullcontext,redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import sha,save
from kernels.transformer_bf16.persistent import PacketSession
from npu.gemma4.runtime import PersistentAssistant,HardwareBusy,admit_hardware
from npu.gemma4.draft import SharedState
from tools.profile_gemma4_target import fingerprint
from tools.ggml_gpu_target.activity import ActivityHelper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--disable-graphs',action='store_true',help='Request graph scheduling off in this child only')
    p.add_argument('--graph-debug',action='store_true')
    p.add_argument('--reference-report',type=Path,help='Require all GPU states and reference tokens to match a prior raw report')
    p.add_argument('--suite',choices=('coexistence','priming','activity'),default='coexistence')
    p.add_argument('--activity-helper',type=Path)
    p.add_argument('--activity-ms',type=int,choices=(40,80,100),default=100)
    p.add_argument('--activity-stop-before-verify',action='store_true')
    a=p.parse_args();progress=a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():raise FileExistsError(a.out)
    if a.suite=='activity' and a.activity_helper is None:p.error('Activity suite requires an explicit helper executable')
    if a.disable_graphs:os.environ['GGML_CUDA_DISABLE_GRAPHS']='1'
    if a.graph_debug:os.environ['GGML_CUDA_GRAPH_DEBUG']='1'
    environment={k:os.environ.get(k) for k in ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_GRAPH_DEBUG')}
    environment['GGML_CUDA_DISABLE_FUSION']='1'
    prompt_path=a.artifacts/'npu/live-public-dictionary-20261018.i32'
    prompt=np.fromfile(prompt_path,np.int32).tolist()
    sources={str(path.resolve()):sha(path) for path in (Path(__file__).resolve(),
             ROOT/'tools/profile_gemma4_target.py',a.candidate,prompt_path)}
    rows=[];rejections=[];preflights=[];reference=[];baseline=None;worker=None
    activity=None;activity_process=None
    if a.suite=='activity':
        for path in (a.activity_helper,ROOT/'tools/ggml_gpu_target/activity.py',ROOT/'tools/ggml_gpu_target/gpu_activity.cpp',
                     ROOT/'tools/ggml_gpu_target/build_activity.cmd',a.runtime/'amdhip64_7.dll',Path('C:/Windows/System32/d3dcompiler_47.dll')):
            sources[str(path.resolve())]=sha(path)
    expected=None
    if a.reference_report:
        sources[str(a.reference_report.resolve())]=sha(a.reference_report)
        expected=json.loads(a.reference_report.read_text())
        if expected.get('passed') is not True:raise ValueError('Reference report must be a complete pass')
        baseline=expected['cases'][0]['snapshots']
    def admit(gpu=None):
        deadline=time.monotonic()+60
        while True:
            try:preflights.append(admit_hardware(gpu));return
            except HardwareBusy as error:
                rejections.append(error.report)
                if time.monotonic()>=deadline:raise
                time.sleep(1)
    try:
        admit()
        with PersistentAssistant(a.artifacts,a.runtime,input_depth=2,vector_copy=True) as backend,ExitStack() as resources:
            # Test-only candidate selection. The serving executable is untouched.
            backend.executable=a.candidate;sources.update(backend.sources)
            admit();current=backend._prepare(prompt);gpu=backend.gpu;worker=gpu.process
            for i in range(33):
                token=current['predictions'][-1];reference.append(token)
                if i<32:current=gpu.decode([token])
            if expected is not None:assert reference==expected['reference']
            if a.suite=='activity':
                admit(gpu);activity=resources.enter_context(ActivityHelper(a.activity_helper,a.runtime));activity_process=activity.process
            modes=('gpu_before','gpu_delayed','npu_resident','gpu_between','npu_interleaved','gpu_after')
            if a.suite=='priming':
                modes=('gpu_before','gpu_delayed','gpu_delayed_primed','npu_interleaved','npu_interleaved_primed','gpu_after')
            if a.suite=='activity':
                modes=('gpu_before','gpu_delayed','gpu_delayed_activity','gpu_between','npu_interleaved','npu_interleaved_activity','gpu_after')
            for repeat in range(2):
                for mode in modes:
                    admit(gpu);current=gpu.prepare(prompt);state=SharedState(current)
                    backend.drafter.invalidate_prefix()
                    session=PacketSession(share_program=True,resident_inputs=True) if mode.startswith('npu_') else None
                    dispatches=[];snapshots=[];decode_ms=priming_ms=0;compiler_log=io.StringIO()
                    activity_windows=[];activity_rpc_ms=0
                    with session if session is not None else nullcontext():
                        def draft(token,hidden):
                            with redirect_stdout(compiler_log):
                                proposed,projected,evidence=backend.drafter.step(gpu,session,state,token,hidden)
                            dispatches.append(evidence);return proposed,projected
                        if mode=='npu_resident':draft(reference[0],state.hidden)
                        before=gpu.timing();start=time.perf_counter()
                        for offset in range(0,32,3):
                            tokens=reference[offset:min(offset+3,32)]
                            if mode.endswith('_activity'):
                                rpc_start=time.perf_counter();activity.start(a.activity_ms)
                                activity_rpc_ms+=(time.perf_counter()-rpc_start)*1000
                            if mode.startswith('gpu_delayed'):time.sleep(.04)
                            if mode.startswith('npu_interleaved'):
                                proposed,hidden=draft(tokens[0],state.hidden)
                                draft(proposed,hidden)
                            if mode.endswith('_primed'):
                                prime_start=time.perf_counter();gpu.embedding(tokens[0])
                                priming_ms+=(time.perf_counter()-prime_start)*1000
                            if mode.endswith('_activity') and a.activity_stop_before_verify:
                                rpc_start=time.perf_counter();activity_windows.append(activity.wait(stop=True))
                                activity_rpc_ms+=(time.perf_counter()-rpc_start)*1000
                            decode_start=time.perf_counter();decoded=gpu.decode(tokens)
                            decode_ms+=(time.perf_counter()-decode_start)*1000
                            assert decoded['predictions']==reference[offset+1:offset+1+len(tokens)]
                            snapshots.append(fingerprint(decoded))
                            state.append(decoded,len(tokens))
                            if mode.endswith('_activity') and not a.activity_stop_before_verify:
                                rpc_start=time.perf_counter();activity_windows.append(activity.wait())
                                activity_rpc_ms+=(time.perf_counter()-rpc_start)*1000
                        loop_ms=(time.perf_counter()-start)*1000;after=gpu.timing()
                    if baseline is None:baseline=snapshots
                    assert snapshots==baseline
                    delta={k:v-before['groups'].get('1',{}).get(k,0) for k,v in after['groups']['1'].items()}
                    assert delta['calls']==11 and delta['tokens']==32
                    row=dict(repeat=repeat,mode=mode,gpu_decode_ms=decode_ms,loop_ms=loop_ms,
                             delay_before_decode_ms=40 if mode.startswith('gpu_delayed') else 0,
                             priming_ms=priming_ms,
                             activity_rpc_ms=activity_rpc_ms,activity_windows=activity_windows,
                             worker_decode=delta,target_state_bitwise_equal=True,snapshots=snapshots,
                             dispatches=dispatches,compiler_log_tail=compiler_log.getvalue()[-65536:],
                             contexts_created=session.contexts_created if session else 0,
                             npu_released=session is None or not session.active and not session.images,
                             post_witness=getattr(session,'post_witness',None))
                    rows.append(row);compiler_log.close()
                    print(json.dumps({k:v for k,v in row.items() if k not in ('snapshots','dispatches','compiler_log_tail','post_witness')}),flush=True)
        assert worker.poll() is not None and (activity_process is None or activity_process.poll()==0)
        assert all(sha(path)==expected for path,expected in sources.items())
        save(a.out,dict(passed=True,scope=__doc__,source_sha256=sources,timing_eligible=False,cases=rows,
                        preflights=preflights,rejections=rejections,reference=reference,worker_closed=True,
                        backend_environment=environment,target_stderr_tail=gpu.stderr_tail,
                        test_suite=a.suite,
                        activity_identity=activity.identity if activity else None,activity_closed=activity_process is None or activity_process.poll()==0,
                        activity_stop_before_verify=a.activity_stop_before_verify,
                        graph_debug_witness='CUDA Graph id ' in gpu.stderr_tail or 'CUDA graph warmup complete' in gpu.stderr_tail,
                        prior_report_state_equal=expected is not None))
    finally:
        save(progress,dict(passed=False,scope='Partial progress only',source_sha256=sources,cases=rows,
                           preflights=preflights,rejections=rejections,backend_environment=environment,
                           activity_closed=activity_process is None or activity_process.poll() is not None,
                           worker_closed=worker is None or worker.poll() is not None))


if __name__=='__main__':main()
