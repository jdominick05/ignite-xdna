"""Check research logits export rows, repeatability and rollback invalidation."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha
from npu.gemma4.runtime import admit_hardware
from tools.ggml_gpu_target.logits_client import LogitTarget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--serving-export',type=Path,required=True)
    p.add_argument('--block-export',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    prompt = a.artifacts/'npu/live-public-dictionary-20261018.i32'
    model = a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    paths = [Path(__file__).resolve(),ROOT/'tools/ggml_gpu_target/logits_client.py',
        ROOT/'tools/ggml_gpu_target/client.py',ROOT/'tools/ggml_gpu_target/worker_logits.cpp',
        ROOT/'tools/ggml_gpu_target/worker_blocks_logits.cpp',ROOT/'tools/ggml_gpu_target/prefix_history.hpp',
        ROOT/'tools/ggml_gpu_embedding/embedding.cpp',ROOT/'npu/gemma4/runtime.py',ROOT/'npu/contention.py',
        a.serving_export,a.block_export,prompt,model]
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    report = dict(passed=False,scope=__doc__,source_sha256={str(path.resolve()):sha(path) for path in paths},
        arms=[],timing_eligible=False,energy_eligible=False)
    env = dict(os.environ,GGML_CUDA_DISABLE_FUSION='1',IGNITION_GPU_PREFILL_BLOCK='128')
    workers = []
    try:
        for executable in (a.serving_export,a.block_export):
            admit_hardware()
            with LogitTarget(executable,a.runtime,model,prompt,env=env,context_tokens=2560) as gpu:
                workers.append(gpu.process)
                row = dict(executable=str(executable),exports=[])
                before = gpu.timing()
                first = gpu.logits()
                repeated = gpu.logits()
                assert np.array_equal(first.view(np.uint32),repeated.view(np.uint32))
                assert int(np.argmax(first[0])) == gpu.initial['predictions'][-1]
                assert gpu.timing() == before
                row['initial_repeat_exact_without_evaluation'] = True
                for count in (1,16):
                    admit_hardware(gpu)
                    response = gpu.decode([gpu.initial['predictions'][-1]]*count)
                    values = gpu.logits(count)
                    assert np.argmax(values,axis=1).tolist() == response['predictions']
                    row['exports'].append(dict(rows=count,shape=list(values.shape),all_greedy_rows_exact=True))
                gpu.rollback(gpu.initial['position']+1)
                try:
                    gpu.logits()
                except RuntimeError as error:
                    if 'Logits export requires' not in str(error):
                        raise
                    row['rollback_export_rejected'] = True
                else:
                    raise ValueError('Rollback did not invalidate logits export')
                report['arms'].append(row)
            save(a.out,report)
        assert all(worker.poll() is not None for worker in workers)
        assert all(sha(Path(path)) == value for path,value in report['source_sha256'].items())
        report.update(passed=True,worker_closed=True)
        print('PASS: both workers export 1/16 rows exactly and reject stale rollback output',flush=True)
        return 0
    except BaseException as error:
        report.update(error=repr(error),worker_closed=all(worker.poll() is not None for worker in workers))
        raise
    finally:
        save(a.out,report)


if __name__ == '__main__':
    raise SystemExit(main())
