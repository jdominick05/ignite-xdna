"""Compare fixed-block cached prepares with cold blocks and unchanged serving.

Cache consistency is an experiment, not permission to replace serving precision
or its numerical baseline. The overall pass still requires exact serving state.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save, sha
from npu.gemma4.runtime import HardwareBusy, admit_hardware
from tools.ggml_gpu_target.client import Target
from tools.ggml_gpu_target.prefix_client import PrefixTarget
from tools.qualify_gemma4_prefix_worker import compare
from tools.profile_gemma4_target import fingerprint


def evaluate(cases):
    exact = lambda key: all(row[key]['prepare']['passed'] and
        all(item['passed'] for item in row[key]['continuations']) for row in cases)
    consistent = bool(cases) and exact('cached_vs_cold_blocks')
    original = bool(cases) and exact('cold_blocks_vs_serving')
    avoided = any(row['name'] in ('extended','diverged_tail','shorter','restored') and
        row['evaluated_rows'] < row['cold_block_rows'] for row in cases)
    return dict(cache_consistency_passed=consistent, original_state_exact=original,
                changed_prompt_work_avoided=avoided, passed=consistent and original and avoided)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--prompt-file',type=Path,required=True)
    p.add_argument('--context-tokens',type=int,choices=(512,2560),default=2560)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--block-rows',type=int,choices=(128,256),default=128)
    a = p.parse_args()
    progress = a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():
        raise FileExistsError(a.out)
    prompt = np.fromfile(a.prompt_file,np.int32).tolist()
    bound = 256 if a.context_tokens == 512 else 2048
    if not 2 <= len(prompt) <= bound-3:
        raise ValueError('Leave three extension tokens within the prompt bound')
    unrelated_path = a.artifacts/'npu/live-public-ok-20261018.i32'
    unrelated = np.fromfile(unrelated_path,np.int32).tolist()
    model = a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    serving = a.artifacts/'bin/ggml_gpu_target.exe'
    paths = [Path(__file__).resolve(),ROOT/'tools/qualify_gemma4_prefix_worker.py',
        ROOT/'tools/profile_gemma4_target.py',ROOT/'tools/ggml_gpu_target/worker.cpp',
        ROOT/'tools/ggml_gpu_target/worker_blocks.cpp',ROOT/'tools/ggml_gpu_target/build_blocks.cmd',
        ROOT/'tools/ggml_gpu_target/prefix_history.hpp',ROOT/'tools/ggml_gpu_target/client.py',
        ROOT/'tools/ggml_gpu_target/prefix_client.py',ROOT/'tools/ggml_gpu_embedding/embedding.cpp',
        ROOT/'npu/gemma4/runtime.py',ROOT/'npu/contention.py',a.prompt_file,unrelated_path,
        model,serving,a.candidate]
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    paths += [Path('L:/llm-work/src/llama.cpp')/name for name in
        ('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h',
         'ggml/include/ggml-alloc.h','ggml/include/gguf.h')]
    report = dict(passed=False,diagnostic_completed=False,scope=__doc__,cases=[],rejections=[],
        source_sha256={str(path.resolve()):sha(path) for path in paths},
        block_rows=a.block_rows,timing_eligible=False,energy_eligible=False,
        criteria='Exact cached/cold block state plus exact unchanged serving state and changed-prompt work avoidance')
    env = os.environ.copy()
    env['GGML_CUDA_DISABLE_FUSION'] = '1'
    candidate_env = dict(env,IGNITION_GPU_PREFILL_BLOCK=str(a.block_rows))
    report['candidate_environment'] = dict(GGML_CUDA_DISABLE_FUSION='1',IGNITION_GPU_PREFILL_BLOCK=str(a.block_rows))
    workers = []
    controls = []
    cold_blocks = []

    def admit(gpu=None):
        deadline = time.monotonic()+60
        while True:
            try:
                return admit_hardware(gpu)
            except HardwareBusy as error:
                report['rejections'].append(error.report)
                save(progress,report)
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(1,max(0,deadline-time.monotonic())))

    def responses(gpu,ids,method,tokens=None):
        admit(gpu)
        started = time.perf_counter()
        prepared = method(ids)
        elapsed = (time.perf_counter()-started)*1000
        outputs = []
        generated = []
        current = prepared
        for index in range(2):
            token = current['predictions'][-1] if tokens is None else tokens[index]
            generated.append(token)
            current = gpu.decode([token])
            outputs.append(current)
        gpu.rollback(len(ids)+1)
        return dict(prepare=prepared,outputs=outputs,tokens=generated,prepare_ms=elapsed)

    def comparison(actual,expected):
        return dict(prepare=compare(actual['prepare'],expected['prepare']),
            continuations=[compare(v,c) for v,c in zip(actual['outputs'],expected['outputs'])])

    save(progress,report)
    try:
        admit()
        with Target(serving,a.runtime,model,a.prompt_file,env=env,context_tokens=a.context_tokens) as gpu:
            workers.append(gpu.process)
            current = gpu.initial
            extension = []
            for _ in range(3):
                token = current['predictions'][-1]
                extension.append(token)
                current = gpu.decode([token])
            plans = [('identical',prompt),('extended',prompt+extension),
                ('diverged_tail',prompt+extension[:2]+[(extension[-1]+1)%262144]),
                ('shorter',prompt[:-1]),('restored',prompt),('unrelated',unrelated),
                ('restored_again',prompt),('identical_again',prompt)]
            for name,ids in plans:
                result = responses(gpu,ids,gpu.prepare)
                controls.append(result)
                report['cases'].append(dict(name=name,prompt_length=len(ids),
                    serving_ms=result['prepare_ms'],serving_snapshot=fingerprint(result['prepare'])))
                save(progress,report)
        admit()
        with Target(a.candidate,a.runtime,model,a.prompt_file,env=candidate_env,context_tokens=a.context_tokens) as gpu:
            workers.append(gpu.process)
            for index,(name,ids) in enumerate(plans):
                result = responses(gpu,ids,gpu.prepare,controls[index]['tokens'])
                cold_blocks.append(result)
                report['cases'][index].update(cold_blocks_ms=result['prepare_ms'],
                    cold_blocks_vs_serving=comparison(result,controls[index]))
                save(progress,report)
        admit()
        with PrefixTarget(a.candidate,a.runtime,model,a.prompt_file,env=candidate_env,context_tokens=a.context_tokens) as gpu:
            workers.append(gpu.process)
            for token in extension:
                gpu.decode([token])
            for index,(name,ids) in enumerate(plans):
                before = gpu.timing()['groups'].get('7',{}).get('tokens',0)
                result = responses(gpu,ids,gpu.prepare_cached,controls[index]['tokens'])
                after = gpu.timing()['groups']['7']['tokens']
                row = report['cases'][index]
                row.update(cached_ms=result['prepare_ms'],evaluated_rows=after-before,
                    cold_block_rows=((len(ids)+a.block_rows-1)//a.block_rows)*a.block_rows,
                    cached_vs_cold_blocks=comparison(result,cold_blocks[index]))
                save(progress,report)
                print('BLOCK_PREFILL '+str(dict(name=name,rows=row['evaluated_rows'],
                    cold_rows=row['cold_block_rows'],cache_exact=row['cached_vs_cold_blocks']['prepare']['passed'],
                    serving_exact=row['cold_blocks_vs_serving']['prepare']['passed'])),flush=True)
        if not all(worker.poll() is not None for worker in workers):
            raise ValueError('Owned worker did not close')
        if any(sha(Path(path)) != expected for path,expected in report['source_sha256'].items()):
            raise ValueError('Source or artifact changed')
        report.update(diagnostic_completed=True,worker_closed=True,**evaluate(report['cases']))
        save(a.out,report)
        return 0 if report['passed'] else 2
    except BaseException as error:
        report.update(passed=False,error=repr(error),worker_closed=all(worker.poll() is not None for worker in workers))
        save(a.out,report)
        raise
    finally:
        save(progress,report)


if __name__ == '__main__':
    raise SystemExit(main())
