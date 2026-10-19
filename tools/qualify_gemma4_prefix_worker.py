"""Compare cached GPU prepares with independent full-prefill controls.

Exact FP32 tensor bits and greedy predictions remain the gate. Numerical
differences are retained as diagnostics, never converted into a passing result.
No NPU inference or CPU model inference runs in this tool.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.transformer_bf16.common import bf16_bits, save, sha
from npu.gemma4.runtime import admit_hardware, HardwareBusy
from tools.ggml_gpu_target.client import Target
from tools.ggml_gpu_target.prefix_client import PrefixTarget
from tools.profile_gemma4_target import fingerprint


def compare(actual, expected):
    result = dict(position_equal=actual['position']==expected['position'],
                  predictions_equal=actual['predictions']==expected['predictions'], tensors={})
    result['tensor_names_equal'] = actual['tensors'].keys()==expected['tensors'].keys()
    for name, control in expected['tensors'].items():
        value = actual['tensors'].get(name)
        if value is None or value.shape != control.shape:
            result['tensors'][name] = dict(shape_equal=False)
            continue
        different = int(np.count_nonzero(value.view(np.uint32)!=control.view(np.uint32)))
        result['tensors'][name] = dict(shape_equal=True, different_elements=different,
            max_abs=float(np.max(np.abs(value-control))),
            bf16_different=int(np.count_nonzero(bf16_bits(value)!=bf16_bits(control))))
    result['passed'] = (result['position_equal'] and result['predictions_equal'] and result['tensor_names_equal']
        and all(row['shape_equal'] and row['different_elements']==0 for row in result['tensors'].values()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--history-test', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context-tokens', type=int, choices=(512,2560), default=512)
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--padding-rows', type=int, choices=(0,9,129), default=0)
    p.add_argument('--disable-graphs', action='store_true', help='Disable graphs in the candidate only; preserve the default control')
    p.add_argument('--align-prefix', action='store_true', help='Test aligned retained prefix and leading future padding')
    p.add_argument('--align-trailing', action='store_true', help='Align the retained prefix and batch length with trailing padding')
    a = p.parse_args()
    if a.align_prefix and a.align_trailing:
        p.error('Choose one alignment mode')
    progress = a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():
        raise FileExistsError(a.out)
    prompt_path = a.prompt_file or a.artifacts/'npu/live-public-dictionary-20261018.i32'
    prompt = np.fromfile(prompt_path, np.int32).tolist()
    prompt_bound = 256 if a.context_tokens==512 else 2048
    if not 2 <= len(prompt) <= prompt_bound-3:
        raise ValueError('Prompt must leave room for three extension tokens')
    unrelated_path = a.artifacts/'npu/live-public-ok-20261018.i32'
    unrelated = np.fromfile(unrelated_path, np.int32).tolist()
    model = a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    control = a.artifacts/'bin/ggml_gpu_target.exe'
    paths = [Path(__file__).resolve(), ROOT/'tools/profile_gemma4_target.py',
        ROOT/'tools/ggml_gpu_target/worker.cpp', ROOT/'tools/ggml_gpu_target/worker_prefix.cpp',
        ROOT/'tools/ggml_gpu_target/prefix_history.hpp', ROOT/'tools/ggml_gpu_target/prefix_history_test.cpp',
        ROOT/'tools/ggml_gpu_target/build_prefix.cmd', ROOT/'tools/ggml_gpu_target/build_prefix_test.cmd',
        ROOT/'tools/ggml_gpu_target/client.py', ROOT/'tools/ggml_gpu_target/prefix_client.py',
        ROOT/'tools/ggml_gpu_embedding/embedding.cpp', ROOT/'npu/contention.py', ROOT/'npu/gemma4/runtime.py',
        Path('L:/llm-work/src/llama.cpp/include/llama.h'), model, control, a.candidate, a.history_test,
        prompt_path, unrelated_path]
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    report = dict(passed=False, scope=__doc__, timing_eligible=False, energy_eligible=False,
        source_sha256={str(path.resolve()):sha(path) for path in paths}, context_tokens=a.context_tokens,
        prompt_length=len(prompt), cases=[], rejections=[])
    report['prefix_padding_rows'] = a.padding_rows
    check = subprocess.run([str(a.history_test)], capture_output=True, text=True, timeout=10)
    report['host_history_test'] = dict(exit_code=check.returncode, stdout=check.stdout, stderr=check.stderr)
    assert check.returncode==0
    env = os.environ.copy()
    env['GGML_CUDA_DISABLE_FUSION'] = '1'
    env['IGNITION_GPU_PREFIX_PADDING'] = str(a.padding_rows)
    candidate_env = dict(env)
    if a.disable_graphs:
        candidate_env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    if a.align_prefix:
        candidate_env['IGNITION_GPU_PREFIX_ALIGN'] = '1'
    if a.align_trailing:
        candidate_env['IGNITION_GPU_PREFIX_ALIGN'] = '2'
    report['candidate_environment'] = {k:candidate_env.get(k) for k in
        ('GGML_CUDA_DISABLE_FUSION','GGML_CUDA_DISABLE_GRAPHS','IGNITION_GPU_PREFIX_PADDING','IGNITION_GPU_PREFIX_ALIGN')}
    controls = []
    workers = []

    def admit(gpu=None):
        deadline = time.monotonic()+60
        while True:
            try:
                return admit_hardware(gpu)
            except HardwareBusy as error:
                report['rejections'].append(error.report)
                save(progress, report)
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    save(progress, report)
    try:
        admit()
        with Target(control, a.runtime, model, prompt_path, env=env, context_tokens=a.context_tokens) as gpu:
            workers.append(gpu.process)
            initial = gpu.initial
            current = initial
            generated = []
            for _ in range(3):
                token = current['predictions'][-1]
                generated.append(token)
                current = gpu.decode([token])
            altered = (generated[-1]+1)%262144
            cases = [('identical',prompt), ('extended',prompt+generated),
                     ('diverged_tail',prompt+generated[:2]+[altered]), ('shorter',prompt[:-1]),
                     ('restored',prompt), ('unrelated',unrelated), ('restored_again',prompt),
                     ('identical_again',prompt)]
            for name, ids in cases:
                admit(gpu)
                started = time.perf_counter()
                prepared = gpu.prepare(ids)
                elapsed = (time.perf_counter()-started)*1000
                tokens, outputs = [], []
                current = prepared
                for _ in range(2):
                    token = current['predictions'][-1]
                    tokens.append(token)
                    current = gpu.decode([token])
                    outputs.append(current)
                gpu.rollback(len(ids)+1)
                controls.append(dict(prepare=prepared, tokens=tokens, outputs=outputs))
                report['cases'].append(dict(name=name, prompt_length=len(ids), control_ms=elapsed,
                    control_snapshot=fingerprint(prepared)))
                save(progress, report)
        assert workers[-1].poll() is not None
        admit()
        with PrefixTarget(a.candidate, a.runtime, model, prompt_path, env=candidate_env, context_tokens=a.context_tokens) as gpu:
            workers.append(gpu.process)
            report['initial_comparison'] = compare(gpu.initial, initial)
            for token in generated:
                gpu.decode([token])
            for index, (name, ids) in enumerate(cases):
                admit(gpu)
                before = gpu.timing()['groups'].get('7',{}).get('tokens',0)
                started = time.perf_counter()
                prepared = gpu.prepare_cached(ids)
                elapsed = (time.perf_counter()-started)*1000
                after = gpu.timing()['groups']['7']['tokens']
                comparison = compare(prepared, controls[index]['prepare'])
                outputs = [gpu.decode([token]) for token in controls[index]['tokens']]
                continuations = [compare(v,c) for v,c in zip(outputs,controls[index]['outputs'])]
                gpu.rollback(len(ids)+1)
                row = report['cases'][index]
                row.update(candidate_ms=elapsed, gpu_evaluated_tokens=after-before,
                    avoided_evaluation_equivalents=len(ids)-(after-before), comparison=comparison,
                    continuation_comparisons=continuations, candidate_snapshot=fingerprint(prepared))
                save(progress, report)
                print('PREFIX_CASE '+json.dumps(dict(name=name, control_ms=row['control_ms'],
                    candidate_ms=elapsed, gpu_evaluated_tokens=after-before,
                    prepare_passed=comparison['passed'], continuations_passed=all(c['passed'] for c in continuations))), flush=True)
        assert all(worker.poll() is not None for worker in workers)
        assert all(sha(Path(path))==expected for path,expected in report['source_sha256'].items())
        report.update(workers_closed=True,
            passed=report['initial_comparison']['passed'] and all(row['comparison']['passed'] and
                all(c['passed'] for c in row['continuation_comparisons']) for row in report['cases']))
        save(a.out, report)
        print('PREFIX_GATE '+str(report['passed']), flush=True)
        return 0 if report['passed'] else 2
    except BaseException as error:
        report['error'] = repr(error)
        report['workers_closed'] = all(worker.poll() is not None for worker in workers)
        if 'gpu' in locals():
            report['last_target_stderr_tail'] = gpu.stderr_tail
        save(a.out, report)
        raise
    finally:
        save(progress, report)


if __name__ == '__main__':
    raise SystemExit(main())
