"""Diagnostic request latency through Ignition, including per-request admission.

Uses frozen public prompt IDs, a continuously consuming client, and separate
GPU control replays outside the served request. No tokenization is timed.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Ignition/src'))
sys.path.insert(0,str(ROOT))
import ignition
from ignition.experimental import HybridAssistant
from kernels.transformer_bf16.common import sha,save
from npu.gemma4.runtime import HardwareBusy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--llama-runtime',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tokens',type=int,choices=range(1,65),default=32)
    parser.add_argument('--depth',type=int,choices=(1,2,3,4),default=2)
    parser.add_argument('--repeats',type=int,choices=range(2,9),default=4)
    parser.add_argument('--input-depth',type=int,choices=(1,2),default=2)
    parser.add_argument('--vector-copy',action='store_true')
    parser.add_argument('--prompt-file',type=Path,help='Local pinned int32 prompt IDs; defaults to the dictionary control')
    parser.add_argument('--context-tokens',type=int,choices=(512,2560),default=512)
    parser.add_argument('--gpu-activity-helper',type=Path)
    args=parser.parse_args()
    progress=args.out.with_suffix('.progress.json')
    if args.out.exists() or progress.exists():raise FileExistsError(args.out)
    prompt_path=args.prompt_file or args.artifacts/'npu/live-public-dictionary-20261018.i32'
    prompt_bytes=prompt_path.stat().st_size
    prompt_bound=256 if args.context_tokens==512 else 2048
    if not 4<=prompt_bytes<=prompt_bound*4 or prompt_bytes%4:
        raise ValueError(f'Prompt file requires 1..{prompt_bound} complete int32 tokens')
    source_paths=[Path(__file__).resolve(),ROOT/'Ignition/src/ignition/experimental/hybrid.py',prompt_path.resolve()]
    sources={str(p):sha(p) for p in source_paths}
    assert Path(ignition.__file__).resolve().is_relative_to(ROOT/'Ignition/src')
    prompt=np.fromfile(prompt_path,np.int32).tolist()
    rows=[];rejections=[];start=time.perf_counter();activity_worker=None
    def persist(pending=None):
        save(progress,dict(passed=False,scope='Partial request diagnostics; final report establishes qualification',
                           source_sha256=sources,cases=rows,rejections=rejections,pending_result=pending))
    persist()
    with HybridAssistant(study_root=ROOT,artifacts=args.artifacts,llama_runtime=args.llama_runtime,
                         collect_trace=True,input_depth=args.input_depth,vector_copy=args.vector_copy,
                         context_tokens=args.context_tokens,gpu_activity_helper=args.gpu_activity_helper) as assistant:
        initialization_ms=(time.perf_counter()-start)*1000
        backend=assistant._backend;worker=None
        for index in range(args.repeats):
            deadline=time.monotonic()+60
            while True:
                start=time.perf_counter();first_token=first_text=None;chunks=[];arrivals=[]
                stream=assistant.stream_tokens(prompt,max_new_tokens=args.tokens,depth=args.depth)
                try:
                    for chunk in stream:
                        elapsed=(time.perf_counter()-start)*1000
                        if first_token is None:first_token=elapsed
                        if first_text is None and chunk.text:first_text=elapsed
                        chunks.append(chunk)
                        arrivals.append(dict(elapsed_ms=elapsed,tokens=len(chunk.token_ids),finished=chunk.finished))
                    break
                except HardwareBusy as error:
                    rejections.append(dict(index=index,preflight=error.report))
                    print(json.dumps(dict(admission_rejected=index,preflight=error.report)),flush=True)
                    if time.monotonic()>=deadline:raise
                    time.sleep(1)
                except BaseException:
                    persist(assistant.last_result);raise
                finally:
                    stream.close()
            request_ms=(time.perf_counter()-start)*1000
            result=dict(assistant.last_result)
            assert result['completed'] and result['npu_released'] and result['npu_idle']
            assert result['gpu_activity']['idle']
            if args.gpu_activity_helper is not None:
                assert result['gpu_activity']['windows'] and all(w['gpu_validated'] for w in result['gpu_activity']['windows'])
                if activity_worker is None:activity_worker=backend._activity.process
                assert activity_worker is backend._activity.process and activity_worker.poll() is None
            assert backend.model_loads==1 and backend.gpu.process.poll() is None
            if worker is None:worker=backend.gpu.process
            assert worker is backend.gpu.process
            tokens=[token for chunk in chunks for token in chunk.token_ids]
            assert tokens==result['tokens']
            steps=[dispatch for round in result['rounds'] for dispatch in round['dispatches']]
            phases={name:sum(step['body']['step_phases_ms'][name] for step in steps)
                    for name in ('gpu_embedding','body','head')}
            row=dict(index=index,first_request=index==0,request_ms=request_ms,first_token_ms=first_token,
                     first_text_ms=first_text,arrivals=arrivals,generated_tokens=len(tokens),step_phases_ms=phases,
                     gpu_control_equal=False,result=result,operation_ms=dict(result.get('operation_ms',{})),
                     unattributed_ms=request_ms-sum(result.get('operation_ms',{}).values()))
            rows.append(row);persist()
            # This reference generation is test-only and outside request time.
            deadline=time.monotonic()+60
            while True:
                try:control_preflight=backend._preflight();break
                except HardwareBusy as error:
                    rejections.append(dict(index=index,phase='control',preflight=error.report));persist()
                    if time.monotonic()>=deadline:raise
                    time.sleep(1)
            control_start=time.perf_counter();current=backend.gpu.prepare(prompt);reference=[]
            control_prefill_ms=(time.perf_counter()-control_start)*1000
            control_start=time.perf_counter()
            for _ in range(len(tokens)):
                token=current['predictions'][-1];reference.append(token)
                if backend.gpu.render([token])['eog'][0]:break
                if len(reference)<len(tokens):current=backend.gpu.decode([token])
            control_ms=(time.perf_counter()-control_start)*1000
            assert tokens==reference
            row.update(
                     control_prefill_ms=control_prefill_ms,control_loop_ms=control_ms,
                     control_preflight=control_preflight,gpu_control_equal=True)
            persist()
            print(json.dumps({k:v for k,v in row.items() if k not in ('result','arrivals','control_preflight')}),flush=True)
        assert len(backend.gpu._stderr_bytes)<=65536
    assert worker.poll() is not None
    assert activity_worker is None or activity_worker.poll()==0
    assert all(sha(Path(p))==value for p,value in sources.items())
    args.out.write_text(json.dumps(dict(passed=True,scope=__doc__,timing_eligible=False,source_sha256=sources,
                                       initialization_ms=initialization_ms,cases=rows,rejections=rejections,
                                       latency_scope='Accepted attempts only; admission rejections listed separately'),indent=2),encoding='utf-8')
    print('PASS: persistent request latency diagnostics and GPU equality',flush=True)


if __name__=='__main__':main()
