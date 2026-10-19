"""Bounded hardware lifecycle checks for Ignition's experimental Gemma runtime."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Ignition/src'))
sys.path.insert(0, str(ROOT))
import ignition
from ignition.experimental import HybridAssistant
from npu.gemma4.runtime import HardwareBusy
from kernels.transformer_bf16.common import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--llama-runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input-depth',type=int,choices=(1,2),default=1)
    parser.add_argument('--vector-copy',action='store_true')
    parser.add_argument('--context-tokens',type=int,choices=(512,2560),default=512)
    parser.add_argument('--prompt-suite',type=Path,help='Prepared public context prompts; requires context 2560')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert Path(ignition.__file__).resolve().is_relative_to(ROOT / 'Ignition/src')
    prompts = {name: np.fromfile(args.artifacts / ('npu/live-public-' + name + '-20261018.i32'),
                                 dtype=np.int32).tolist()
               for name in ('dictionary', 'ok', 'unicode')}
    cases=[('dictionary',True,16),('ok',False,16),('unicode',False,16),('dictionary',False,16)]
    if args.prompt_suite:
        if args.context_tokens!=2560:raise ValueError('Context prompts require context 2560')
        for length in (511,513,1024,2048):
            path=args.prompt_suite/f'public-{length}.i32'
            metadata=json.loads(path.with_suffix('.json').read_text())
            if sha(path)!=metadata['sha256'] or metadata['tokens']!=length:raise ValueError('Context prompt identity changed')
            ids=np.fromfile(path,np.int32).tolist()
            if len(ids)!=length:raise ValueError('Context prompt length changed')
            prompts[str(length)]=ids
        cases=[('511',False,8),('513',False,8),('1024',False,8),('2048',False,8),('513',True,8),('ok',False,16)]
    records = []
    rejections = []
    with HybridAssistant(study_root=ROOT, artifacts=args.artifacts,
                         llama_runtime=args.llama_runtime, collect_trace=True,
                         input_depth=args.input_depth,vector_copy=args.vector_copy,context_tokens=args.context_tokens) as assistant:
        backend = assistant._backend
        for invalid in ([], [True], [-1], [262144],[1]*(assistant.max_prompt_tokens+1)):
            try:
                assistant.stream_tokens(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid prompt accepted')
        # Closing a stream before its first iteration must release admission.
        pending = assistant.stream_tokens(prompts['ok'])
        pending.close()
        assert not backend._lock.locked()
        assert backend.gpu is None
        worker = None
        for name,cancel,limit in cases:
            deadline=time.monotonic()+60
            while True:
                stream = assistant.stream_tokens(prompts[name], max_new_tokens=limit)
                chunks = []
                try:
                    for chunk in stream:
                        chunks.append(chunk)
                        if cancel and len(chunks) == 2:
                            stream.close()
                            break
                    break
                except HardwareBusy as error:
                    rejection=dict(rejected_case=name,preflight=error.report)
                    rejections.append(rejection)
                    print(json.dumps(rejection),flush=True)
                    if time.monotonic()>=deadline:
                        args.out.write_text(json.dumps(dict(passed=False,cases=records,
                                                            rejections=rejections),indent=2),encoding='utf-8')
                        raise
                    # This checks-only runner waits for admission; the chat
                    # runtime itself reports busy immediately. Never evict work.
                    time.sleep(1)
            result = dict(assistant.last_result)
            assert result['npu_released'] and result['npu_idle'], result
            assert backend.gpu.process.poll() is None
            assert backend.model_loads == 1
            if worker is None:
                worker = backend.gpu.process
            assert backend.gpu.process is worker
            tokens = [token for chunk in chunks for token in chunk.token_ids]
            # Test-only control: the serving path never runs this replay.
            current = backend.gpu.prepare(prompts[name])
            reference = []
            for _ in range(len(tokens)):
                token = current['predictions'][-1]
                reference.append(token)
                if backend.gpu.render([token])['eog'][0]:
                    break
                current = backend.gpu.decode([token])
            assert tokens == reference, (name, tokens, reference)
            result['gpu_control_equal'] = True
            result['case'] = name
            records.append(result)
            print(json.dumps(dict(case=name, tokens=tokens, completed=result['completed'],
                                  npu_idle=result['npu_idle'], model_loads=backend.model_loads)), flush=True)
        assert len(backend.gpu._stderr_bytes) <= 65536
    assert worker.poll() is not None
    args.out.write_text(json.dumps(dict(scope='checks-only; no performance qualification',
                                        passed=True, cases=records,rejections=rejections), indent=2), encoding='utf-8')
    print('PASS: persistent weights, reset, GPU equality, cancellation and NPU release', flush=True)


if __name__ == '__main__':
    main()
