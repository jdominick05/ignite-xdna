"""Audit exact NPU packets, then compare an isolated host prefix-cache candidate.

Serving sources and kernels stay unchanged. Packet/input/output hashes are
compared in separate qualification requests, outside latency observations.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'Ignition/src'))
from ignition.experimental import HybridAssistant
from kernels.transformer_bf16.common import save, sha
from kernels.transformer_bf16.persistent import PacketSession
from npu.gemma4 import runtime
from npu.gemma4.runtime import HardwareBusy
from tools.gemma4_prefix_candidate import PrefixCachedDraft


class AuditedSession(PacketSession):
    audit = False
    records = []

    def execute(self, folder, packets, *args, **kwargs):
        before = hashlib.sha256(packets).hexdigest() if self.audit else None
        result, evidence = super().execute(folder, packets, *args, **kwargs)
        if self.audit:
            self.records.append(dict(build=str(Path(folder)/'build.json'), input_sha256=before,
                output_sha256=hashlib.sha256(result).hexdigest(),
                packet_count=len(packets), input_fully_verified=evidence['input_fully_verified']))
        return result, evidence


def run_request(assistant, prompt, cached, reference, rejections):
    assistant._backend.drafter.prefix_cache_enabled = cached
    deadline = time.monotonic()+60
    while True:
        start = time.perf_counter()
        stream = assistant.stream_tokens(prompt, max_new_tokens=len(reference), depth=2)
        try:
            tokens = [t for chunk in stream for t in chunk.token_ids]
            break
        except HardwareBusy as error:
            rejections.append(error.report)
            print('ADMISSION_REJECTED '+json.dumps(error.report), flush=True)
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
        finally:
            stream.close()
    elapsed = (time.perf_counter()-start)*1000
    result = assistant.last_result
    assert tokens == reference and result['tokens'] == reference
    assert result['completed'] and result['npu_released'] and result['npu_idle']
    assert result['gpu_activity']['idle']
    cache = [r['body']['prefix_cache'] for round in result['rounds'] for r in round['dispatches']
             if 'prefix_cache' in r['body']]
    return dict(cached=cached, request_ms=elapsed, operation_ms=result['operation_ms'],
                cache_hits=sum(r['hit'] for r in cache), cache_misses=sum(not r['hit'] for r in cache),
                key_ms=sum(r['key_ms'] for r in cache), result=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--llama-runtime', type=Path, required=True)
    p.add_argument('--gpu-activity-helper', type=Path)
    p.add_argument('--bind-prefix', action='store_true', help='Bind owned prefix packets instead of content-key caching')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--context-tokens', type=int, choices=(512, 2560), default=512)
    p.add_argument('--tokens', type=int, choices=range(1, 65), default=32)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    progress = args.out.with_suffix('.progress.json')
    if args.out.exists() or progress.exists():
        raise FileExistsError(args.out)
    prompt_path = args.prompt_file or args.artifacts/'npu/live-public-dictionary-20261018.i32'
    bound = 256 if args.context_tokens == 512 else 2048
    if not 4 <= prompt_path.stat().st_size <= bound*4 or prompt_path.stat().st_size % 4:
        raise ValueError('Complete int32 prompt IDs within the context bound required')
    prompt = np.fromfile(prompt_path, np.int32).tolist()
    paths = [Path(__file__).resolve(), ROOT/'tools/gemma4_prefix_candidate.py',
             ROOT/'kernels/transformer_bf16/prefix_cache.py', ROOT/'kernels/transformer_bf16/state.py',
             ROOT/'kernels/transformer_bf16/prefix_binding.py',
             ROOT/'kernels/transformer_bf16/persistent.py', prompt_path]
    if args.prompt_file is not None:
        metadata_path = prompt_path.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        if metadata['sha256'] != sha(prompt_path) or metadata['tokens'] != len(prompt):
            raise ValueError('Prepared prompt identity changed')
        paths.append(metadata_path)
    report = dict(passed=False, timing_eligible=False, energy_eligible=False, scope=__doc__,
                  candidate_kind='prefix_binding' if args.bind_prefix else 'prefix_cache',
                  context_tokens=args.context_tokens, prompt_length=len(prompt), output_limit=args.tokens,
                  source_sha256={str(path.resolve()): sha(path) for path in paths},
                  cases=[], audits={}, rejections=[])
    save(progress, report)
    # The instrumentation exists only in this experimental child process.
    original_session = runtime.PacketSession
    runtime.PacketSession = AuditedSession
    try:
        assistant_type = HybridAssistant
        if args.bind_prefix:
            from ignition.experimental.prefix_bound import PrefixBoundAssistant
            assistant_type = PrefixBoundAssistant
        with assistant_type(study_root=ROOT, artifacts=args.artifacts, llama_runtime=args.llama_runtime,
                             collect_trace=True, input_depth=2, vector_copy=True,
                             context_tokens=args.context_tokens,
                             gpu_activity_helper=args.gpu_activity_helper) as assistant:
            backend = assistant._backend
            report['source_sha256'].update(backend.sources)
            if not args.bind_prefix:
                backend.drafter = PrefixCachedDraft.from_draft(backend.drafter)
            deadline = time.monotonic()+60
            while True:
                try:
                    report['initial_preflight'] = backend._preflight()
                    break
                except HardwareBusy as error:
                    report['rejections'].append(dict(phase='initial_control', preflight=error.report))
                    save(progress, report)
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
            current = backend._prepare(prompt)
            reference = []
            for _ in range(args.tokens):
                token = current['predictions'][-1]
                reference.append(token)
                assert not backend.gpu.render([token])['eog'][0]
                if len(reference) < args.tokens:
                    current = backend.gpu.decode([token])
            report['reference'] = reference
            worker = backend.gpu.process
            report['warmup'] = run_request(assistant, prompt, False, reference, report['rejections'])
            for name, cached in (('control', False), ('candidate', True)):
                AuditedSession.audit = True
                AuditedSession.records = []
                row = run_request(assistant, prompt, cached, reference, report['rejections'])
                report['audits'][name] = dict(request=row, packets=AuditedSession.records)
                save(progress, report)
            control = report['audits']['control']['packets']
            candidate = report['audits']['candidate']['packets']
            assert control and control == candidate
            assert all(r['input_fully_verified'] for r in control)
            report['packet_inputs_and_outputs_bitwise_equal'] = True
            AuditedSession.audit = False
            print(f'PACKET_AUDIT PASS: {len(control)} dispatches, complete input/output hashes equal', flush=True)
            for index, cached in enumerate((False, True, True, False)):
                row = run_request(assistant, prompt, cached, reference, report['rejections'])
                row['index'] = index
                report['cases'].append(row)
                save(progress, report)
                print('REQUEST_RESULT '+json.dumps({k: v for k, v in row.items() if k != 'result'}), flush=True)
            assert backend.model_loads == 1 and backend.gpu.process is worker and worker.poll() is None
            activity = backend._activity.process if backend._activity else None
        assert worker.poll() is not None and (activity is None or activity.poll() == 0)
        assert all(sha(Path(path)) == expected for path, expected in report['source_sha256'].items())
        report.update(passed=True, worker_closed=True, activity_closed=True)
        save(args.out, report)
        print('PASS: exact prefix-cache candidate; latency observations remain diagnostic', flush=True)
    finally:
        runtime.PacketSession = original_session
        save(progress, report)


if __name__ == '__main__':
    main()
