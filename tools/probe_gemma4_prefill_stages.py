"""Locate shared-prefix divergence while gating stage capture against serving."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.transformer_bf16.common import save, sha
from npu.gemma4.runtime import admit_hardware, HardwareBusy
from tools.ggml_gpu_target.client import Target
from tools.profile_gemma4_target import fingerprint
from tools.probe_gemma4_prefix_invariance import compare_rows

STAGES = ('attn_norm', 'Qcur_pos', 'Kcur_pos', 'Vcur_normed', 'attn_post_norm',
          'attn_out', 'ffn_out', 'ffn_post_norm', 'l_out', 'kq', 'kq_soft_max', 'kqv', 'kqv_out')


def token_rows(array, length):
    axes = [i for i, n in enumerate(array.shape) if n == length]
    if array.dtype != np.float32 or len(axes) != 1:
        raise ValueError('One unambiguous FP32 token axis required')
    return np.ascontiguousarray(np.moveaxis(array, axes[0], 0).reshape(length, -1))


def serving_view(response, stage):
    required = {'result_norm', 'Kcur_pos-13', 'Vcur_normed-13', 'Kcur_pos-14', 'Vcur_normed-14'}
    return fingerprint(dict(response, tensors={k: v for k, v in response['tensors'].items()
                                              if k != stage or k in required}))


def causal_rows(array, length, common):
    """Compare only common causal scores; padding/future keys are excluded."""
    if (array.dtype != np.float32 or array.ndim != 4 or array.shape[1] != length
            or array.shape[0] < common or not 0 < common <= length):
        raise ValueError('Expected key-by-query-by-head FP32 attention scores')
    value = array[:common,:common,:,:].copy()
    for query in range(common):
        value[query+1:,query,:,:] = 0
    return np.ascontiguousarray(np.moveaxis(value,1,0).reshape(common,-1))


def masked_probability_stats(array, length, common):
    if (array.dtype != np.float32 or array.ndim != 4 or array.shape[1] != length
            or array.shape[0] < common or not 0 < common <= length):
        raise ValueError('Expected key-by-query-by-head FP32 probabilities')
    nonzero = 0
    for query in range(common):
        value = np.ascontiguousarray(array[query+1:,query,:,:])
        nonzero += int(np.count_nonzero(value.view(np.uint32)))
    return dict(future_and_padding_bits_nonzero=nonzero, all_positive_zero=nonzero == 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--suite', choices=('layer','attention'), default='layer')
    a = p.parse_args()
    progress = a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():
        raise FileExistsError(a.out)
    ids = np.fromfile(a.prompt_file, np.int32).tolist()
    if not 129 <= len(ids) <= 2045:
        raise ValueError('Use 129..2045 prompt tokens')
    model = a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    original = a.artifacts/'bin/ggml_gpu_target.exe'
    paths = [Path(__file__).resolve(), ROOT/'tools/ggml_gpu_target/worker_stages.cpp',
        ROOT/'tools/ggml_gpu_target/worker.cpp', ROOT/'tools/ggml_gpu_target/client.py',
        ROOT/'tools/ggml_gpu_target/build_stages.cmd', ROOT/'tools/profile_gemma4_target.py',
        ROOT/'tools/probe_gemma4_prefix_invariance.py', ROOT/'npu/gemma4/runtime.py',
        ROOT/'npu/contention.py', ROOT/'tools/ggml_gpu_embedding/embedding.cpp',
        original, a.candidate, a.prompt_file, model]
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    paths += [Path('L:/llm-work/src/llama.cpp')/name for name in
        ('include/llama.h', 'ggml/include/ggml.h', 'ggml/include/ggml-backend.h',
         'ggml/include/ggml-alloc.h', 'ggml/include/gguf.h')]
    report = dict(passed=False, diagnostic_completed=False, scope=__doc__, suite=a.suite, stages=[], rejections=[],
        source_sha256={str(path.resolve()): sha(path) for path in paths},
        timing_eligible=False, energy_eligible=False)
    env = os.environ.copy()
    env['GGML_CUDA_DISABLE_FUSION'] = '1'
    workers = []
    save(progress, report)
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
                time.sleep(min(1, max(0, deadline-time.monotonic())))
    try:
        admit()
        with Target(original, a.runtime, model, a.prompt_file, env=env, context_tokens=2560) as gpu:
            workers.append(gpu.process)
            base = gpu.initial
            current = base
            extension = []
            for _ in range(3):
                token = current['predictions'][-1]
                extension.append(token)
                current = gpu.decode([token])
            admit(gpu)
            extended = gpu.prepare(ids+extension)
            controls = [fingerprint(base), fingerprint(extended)]
        report.update(base_length=len(ids), extension=extension, controls=controls)
        admit()
        with Target(a.candidate, a.runtime, model, a.prompt_file, env=env, context_tokens=2560) as gpu:
            workers.append(gpu.process)
            if serving_view(gpu.initial, 'attn_norm-0') != controls[0]:
                raise ValueError('Initial instrumented serving state changed')

            def check(stage, layer):
                name = f'{stage}-{layer}'
                gpu.request(10, [STAGES.index(stage), layer])
                responses = []
                masked = []
                for prompt, control in zip((ids, ids+extension), controls):
                    admit(gpu)
                    response = gpu.prepare(prompt)
                    if name not in response['tensors'] or serving_view(response, name) != control:
                        raise ValueError(f'Instrumentation changes serving state or missing {name}')
                    tensor = response['tensors'][name]
                    if stage == 'kq_soft_max':
                        stats = masked_probability_stats(tensor,len(prompt),len(ids))
                        masked.append(stats)
                        if not stats['all_positive_zero']:
                            raise ValueError('Future/padding probabilities are not bitwise positive zero')
                    responses.append(causal_rows(tensor,len(prompt),len(ids)) if stage in ('kq','kq_soft_max')
                                     else token_rows(tensor,len(prompt)))
                comparison = compare_rows(responses[1][:len(ids)], responses[0])
                row = dict(stage=stage, layer=layer, serving_states_exact=True, comparison=comparison,
                    comparison_domain='common causal keys only' if stage in ('kq','kq_soft_max') else 'all common token features')
                if masked:
                    row['masked_probabilities'] = masked
                report['stages'].append(row)
                save(progress, report)
                print('PREFILL_STAGE '+str(dict(stage=name, equal=comparison['bitwise_equal'],
                    changed_rows=comparison['changed_rows'])), flush=True)
                return comparison['bitwise_equal']

            if a.suite == 'attention':
                for stage in STAGES[9:]:
                    check(stage, 0)
            else:
                first_layer = None
                for layer in range(14):
                    if not check('l_out', layer):
                        first_layer = layer
                        break
                report['first_changed_layer_output'] = first_layer
                if first_layer is not None:
                    for stage in STAGES[:8]:
                        check(stage, first_layer)
        report.update(diagnostic_completed=True, passed=True,
            worker_closed=all(worker.poll() is not None for worker in workers))
        if not report['worker_closed'] or any(sha(Path(path)) != expected for path, expected in report['source_sha256'].items()):
            raise ValueError('Worker leak or source identity changed')
        save(a.out, report)
        return 0
    except BaseException as error:
        report.update(passed=False, error=repr(error), worker_closed=all(worker.poll() is not None for worker in workers))
        save(a.out, report)
        raise
    finally:
        save(progress, report)


if __name__ == '__main__':
    raise SystemExit(main())
