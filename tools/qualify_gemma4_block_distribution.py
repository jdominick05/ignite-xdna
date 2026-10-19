"""Frozen held-out distribution diagnostic; does not waive the serving state gate.

Uses only token IDs from the existing BF16 reference file. Its clipped stored
probabilities are not used as a distribution oracle or a perplexity baseline.
All neural computation is GPU-only; FP64 host statistics are offline analysis.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha
from npu.gemma4.runtime import admit_hardware,HardwareBusy
from tools.gemma4_distribution import GATES,compare_distribution,summarize
from tools.ggml_gpu_target.client import Target
from tools.ggml_gpu_target.logits_client import LogitTarget
from tools.profile_gemma4_target import fingerprint

LENGTHS = (256,511,512,513,768,1023)
ARMS = ['serving_base','serving_export','blocks128_base','blocks128_export','blocks256_base','blocks256_export']


def reference_tokens(reference):
    with reference.open('rb') as source:
        header = source.read(20)
        if len(header) != 20 or header[:8] != b'_logits_':
            raise ValueError('Unknown reference file header')
        context,vocabulary,chunks = struct.unpack('<III',header[8:])
        if (context,vocabulary,chunks) != (1024,262144,4):
            raise ValueError('Frozen corpus requires four 1024-token windows')
        tokens = np.frombuffer(source.read(4*context*chunks),'<i4').reshape(chunks,context).copy()
    if reference.stat().st_size != 20+4*context*chunks+chunks*(context//2-1)*(vocabulary+4)*2:
        raise ValueError('Unexpected reference file size')
    if tokens[0,0] != 2 or np.any(tokens < 0) or np.any(tokens >= vocabulary):
        raise ValueError('Unexpected BOS or invalid token IDs')
    tokens[:,0] = 2
    return tokens


def prepare_cases(reference,folder):
    tokens = reference_tokens(reference)
    folder.mkdir()
    cases = []
    for window in range(4):
        # Explicit per-window BOS, matching the model's existing token ID.
        tokens[window,0] = 2
        for length in LENGTHS:
            path = folder/f'window{window}-prefix{length}.i32'
            path.write_bytes(tokens[window,:length].astype('<i4').tobytes())
            cases.append(dict(name=path.stem,window=window,prompt_tokens=length,
                prompt_file=str(path.resolve()),prompt_sha256=sha(path),target=int(tokens[window,length])))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--serving-export',type=Path,required=True)
    p.add_argument('--block-export',type=Path,required=True)
    p.add_argument('--blocks',type=Path,required=True)
    p.add_argument('--prereg',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a = p.parse_args()
    reference = a.artifacts/'quality/e2b-reference-ROCm0-1024-4.bin'
    model = a.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
    serving = a.artifacts/'bin/ggml_gpu_target.exe'
    paths = [Path(__file__).resolve(),ROOT/'tools/gemma4_distribution.py',
        ROOT/'tools/ggml_gpu_target/client.py',ROOT/'tools/ggml_gpu_target/logits_client.py',
        ROOT/'tools/ggml_gpu_target/worker.cpp',ROOT/'tools/ggml_gpu_target/worker_blocks.cpp',
        ROOT/'tools/ggml_gpu_target/worker_logits.cpp',ROOT/'tools/ggml_gpu_target/worker_blocks_logits.cpp',
        ROOT/'tools/ggml_gpu_target/build_logits.cmd',ROOT/'tools/ggml_gpu_target/prefix_history.hpp',
        ROOT/'tools/ggml_gpu_embedding/embedding.cpp',ROOT/'tools/profile_gemma4_target.py',
        ROOT/'npu/contention.py',ROOT/'npu/gemma4/runtime.py',model,serving,a.blocks,
        a.serving_export,a.block_export,reference,reference.with_suffix('.bin.json'),
        a.artifacts/'quality/heldout-engineering-v1.txt',a.artifacts/'quality/heldout-engineering-v1.json']
    paths += [a.runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
    paths += [Path('L:/llm-work/src/llama.cpp')/name for name in
        ('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h',
         'ggml/include/ggml-alloc.h','ggml/include/gguf.h')]
    sources = {str(path.resolve()):sha(path) for path in paths}
    if a.prepare_only:
        if a.prereg.exists():
            raise FileExistsError(a.prereg)
        cases = prepare_cases(reference,a.prereg.with_suffix('.fixtures'))
        protocol = dict(scope=__doc__,gates=GATES,cases=cases,source_sha256=sources,
            context_tokens=2560,neural_backend='ROCm GPU only',host_math='offline FP64 statistics only',
            adoption_qualified=False,exact_serving_gate_waived=False,
            arms=ARMS)
        save(a.prereg,protocol)
        print('FROZEN: 24 held-out states; distribution diagnostic only',flush=True)
        return 0
    protocol = json.loads(a.prereg.read_text())
    if protocol['source_sha256'] != sources or protocol['gates'] != GATES:
        raise ValueError('Frozen source/artifact identity or gates changed')
    if protocol['arms'] != ARMS or protocol['context_tokens'] != 2560:
        raise ValueError('Frozen arm order or context changed')
    cases = protocol['cases']
    if len(cases) != 24 or any(sha(Path(case['prompt_file'])) != case['prompt_sha256'] for case in cases):
        raise ValueError('Frozen cases changed')
    tokens = reference_tokens(reference)
    for index,case in enumerate(cases):
        window,length = index//len(LENGTHS),LENGTHS[index%len(LENGTHS)]
        if (case['window'] != window or case['prompt_tokens'] != length or case['target'] != int(tokens[window,length])
                or Path(case['prompt_file']).read_bytes() != tokens[window,:length].astype('<i4').tobytes()):
            raise ValueError('Frozen target or prompt no longer matches held-out corpus')
    data = a.out.with_suffix('.logits')
    progress = a.out.with_suffix('.progress.json')
    if a.out.exists() or data.exists() or progress.exists():
        raise FileExistsError(a.out)
    data.mkdir()
    report = dict(passed=False,diagnostic_completed=False,scope=__doc__,prereg_sha256=sha(a.prereg),
        source_sha256=sources,protocol=protocol,arms={},summary={},rejections=[],
        adoption_qualified=False,exact_serving_gate_waived=False,timing_eligible=False,energy_eligible=False)
    workers = []

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

    save(progress,report)
    try:
        for arm in protocol['arms']:
            is_export = arm.endswith('_export')
            block = 128 if arm.startswith('blocks128') else 256 if arm.startswith('blocks256') else None
            executable = a.block_export if block and is_export else a.blocks if block else a.serving_export if is_export else serving
            target_type = LogitTarget if is_export else Target
            env = os.environ.copy()
            env['GGML_CUDA_DISABLE_FUSION'] = '1'
            if block:
                env['IGNITION_GPU_PREFILL_BLOCK'] = str(block)
            admit()
            report['arms'][arm] = dict(cases=[],initial=None)
            with target_type(executable,a.runtime,model,Path(cases[0]['prompt_file']),env=env,context_tokens=2560) as gpu:
                workers.append(gpu.process)
                report['arms'][arm]['initial'] = fingerprint(gpu.initial)
                base_arm = arm.removesuffix('_export')+'_base'
                if is_export and fingerprint(gpu.initial) != report['arms'][base_arm]['initial']:
                    raise ValueError('Logits export changes initial serving state')
                for index,case in enumerate(cases):
                    ids = np.fromfile(case['prompt_file'],'<i4').tolist()
                    admit(gpu)
                    response = gpu.prepare(ids)
                    row = dict(name=case['name'],snapshot=fingerprint(response))
                    if is_export:
                        if row['snapshot'] != report['arms'][base_arm]['cases'][index]['snapshot']:
                            raise ValueError('Logits export changes regular serving state')
                        logits = gpu.logits()[0]
                        if int(np.argmax(logits)) != response['predictions'][-1]:
                            raise ValueError('Logits layout or greedy prediction mismatch')
                        path = data/f'{arm}-{case["name"]}.npy'
                        np.save(path,logits,allow_pickle=False)
                        row.update(logits_file=str(path.resolve()),logits_sha256=sha(path),export_state_exact=True)
                        if block:
                            control_row = report['arms']['serving_export']['cases'][index]
                            control_path = control_row['logits_file']
                            if sha(control_path) != control_row['logits_sha256']:
                                raise ValueError('Serving distribution artifact changed')
                            control = np.load(control_path,allow_pickle=False)
                            row['distribution'] = compare_distribution(logits,control,case['target'])
                    report['arms'][arm]['cases'].append(row)
                    save(progress,report)
                    print('DISTRIBUTION '+str(dict(arm=arm,case=case['name'],
                        metrics=row.get('distribution'))),flush=True)
            if block and is_export:
                report['summary'][arm] = summarize([r['distribution'] for r in report['arms'][arm]['cases']],24)
        if not all(worker.poll() is not None for worker in workers):
            raise ValueError('Owned worker did not close')
        if any(sha(Path(path)) != value for path,value in sources.items()):
            raise ValueError('Frozen source or artifact changed')
        for arm in report['arms'].values():
            for row in arm['cases']:
                if 'logits_file' in row and sha(row['logits_file']) != row['logits_sha256']:
                    raise ValueError('Recorded logits changed')
        if set(report['summary']) != {'blocks128_export','blocks256_export'}:
            raise ValueError('Incomplete candidate summaries')
        report.update(diagnostic_completed=True,worker_closed=True,
            passed=all(row['diagnostic_passed'] for row in report['summary'].values()))
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
