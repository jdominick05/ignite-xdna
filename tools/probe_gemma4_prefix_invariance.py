"""Test shared-prefix KV invariance using only the registered full-prefill worker.

This isolates the control's batch-length behavior from the experimental GPU
cache. Exact FP32 bits remain the gate; changed row indices are retained.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.transformer_bf16.common import bf16_bits, save, sha
from npu.gemma4.runtime import admit_hardware, HardwareBusy
from tools.ggml_gpu_target.client import Target
from tools.profile_gemma4_target import fingerprint

KV_NAMES = ('Kcur_pos-13','Vcur_normed-13','Kcur_pos-14','Vcur_normed-14')


def shared_length(left, right):
    count = 0
    for a, b in zip(left,right):
        if a != b:
            break
        count += 1
    return count


def compare_rows(value, control):
    if value.dtype != np.float32 or control.dtype != np.float32 or value.shape != control.shape or value.ndim != 2:
        raise ValueError('Matching FP32 token-by-feature matrices required')
    mask = value.view(np.uint32) != control.view(np.uint32)
    rows = np.flatnonzero(mask.any(axis=1)).tolist()
    return dict(bitwise_equal=not rows, changed_rows=rows, changed_elements=int(mask.sum()),
        max_abs=float(np.max(np.abs(value-control))) if value.size else 0.,
        bf16_different=int(np.count_nonzero(bf16_bits(value)!=bf16_bits(control))),
        actual_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        control_sha256=hashlib.sha256(control.tobytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--context-tokens', type=int, choices=(512,2560), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    progress = a.out.with_suffix('.progress.json')
    if a.out.exists() or progress.exists():
        raise FileExistsError(a.out)
    ids = np.fromfile(a.prompt_file,np.int32).tolist()
    bound = 256 if a.context_tokens==512 else 2048
    if not 9 <= len(ids) <= bound-8:
        raise ValueError('Leave room for eight extension tokens')
    model = a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    executable = a.artifacts/'bin/ggml_gpu_target.exe'
    paths = [Path(__file__).resolve(), ROOT/'tools/ggml_gpu_target/worker.cpp',
        ROOT/'tools/ggml_gpu_target/client.py', ROOT/'tools/profile_gemma4_target.py',
        ROOT/'npu/contention.py', ROOT/'npu/gemma4/runtime.py', a.prompt_file, model, executable]
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    report = dict(passed=False, diagnostic_completed=False, scope=__doc__,
        source_sha256={str(path.resolve()):sha(path) for path in paths},
        context_tokens=a.context_tokens, base_length=len(ids), cases=[], rejections=[],
        timing_eligible=False, energy_eligible=False)
    env = os.environ.copy()
    env['GGML_CUDA_DISABLE_FUSION'] = '1'

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
                time.sleep(1)

    worker = None
    save(progress,report)
    try:
        admit()
        with Target(executable,a.runtime,model,a.prompt_file,env=env,context_tokens=a.context_tokens) as gpu:
            worker = gpu.process
            initial = gpu.initial
            current = initial
            extension = []
            for _ in range(8):
                token = current['predictions'][-1]
                extension.append(token)
                current = gpu.decode([token])
            report['extension'] = extension
            plans = [('repeat',ids), ('plus_one',ids+extension[:1]), ('plus_three',ids+extension[:3]),
                     ('plus_eight',ids+extension), ('minus_one',ids[:-1]), ('minus_three',ids[:-3]),
                     ('repeat_after',ids)]
            initial_fingerprint = fingerprint(initial)
            for name, prompt in plans:
                admit(gpu)
                response = gpu.prepare(prompt)
                common = shared_length(ids,prompt)
                comparisons = {}
                for tensor in KV_NAMES:
                    left = np.ascontiguousarray(response['tensors'][tensor][:,0,:common,0].T)
                    right = np.ascontiguousarray(initial['tensors'][tensor][:,0,:common,0].T)
                    comparisons[tensor] = compare_rows(left,right)
                row = dict(name=name, prompt_length=len(prompt), common_length=common,
                    prefix_bitwise_equal=all(c['bitwise_equal'] for c in comparisons.values()),
                    comparisons=comparisons, snapshot=fingerprint(response))
                if name.startswith('repeat'):
                    row['whole_response_equal'] = fingerprint(response)==initial_fingerprint
                report['cases'].append(row)
                save(progress,report)
                print('CONTROL_PREFIX '+str(dict(name=name, common=common, equal=row['prefix_bitwise_equal'],
                    changed_rows={k:v['changed_rows'] for k,v in comparisons.items()})),flush=True)
        assert worker.poll() is not None
        assert all(sha(Path(path))==expected for path,expected in report['source_sha256'].items())
        report.update(diagnostic_completed=True, worker_closed=True,
            passed=all(row['prefix_bitwise_equal'] and row.get('whole_response_equal',True) for row in report['cases']))
        save(a.out,report)
        return 0 if report['passed'] else 2
    except BaseException as error:
        report['error'] = repr(error)
        report['worker_closed'] = worker is None or worker.poll() is not None
        save(a.out,report)
        raise
    finally:
        save(progress,report)


if __name__ == '__main__':
    raise SystemExit(main())
