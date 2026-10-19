"""Matched package-energy diagnostics for GPU and hybrid E2B requests.

Retains raw typeperf CSVs and freezes the protocol before hardware admission.
This measures package energy for one public prompt, not wall energy, an NPU
power domain, general task quality, or unattended operation.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Ignition/src'))
from ignition.experimental import HybridAssistant
from kernels.transformer_bf16.common import save, sha
from npu.contention import _cpu_times, _GpuQuery, owned_process_ids
from tools import llm_energy as energy
from tools.research_process import context_report, context_pids

PROTOCOL = dict(
    order=['gpu', 'hybrid', 'hybrid_activity', 'hybrid_activity', 'hybrid', 'gpu'],
    idle_seconds=60, active_min_seconds=70, settle_seconds=10,
    counter_lead_seconds=2.5, sample_seconds=1, trim_seconds=1,
    min_rows=50, min_sample_coverage=.90, max_sample_gap_seconds=2.5,
    idle_max_cpu_percent=12, idle_max_stdev_w=2.5, idle_max_shift_w=2,
    repeat_max_relative_difference=.10, tokens=32, depth=2,
    input_depth=2, vector_copy=True, context_tokens=512,
    workload='Complete repeated requests: admission, full prefill, decode, rendering and request cleanup',
    baseline='Same arm/model resident and idle; warm-up and constructor excluded',
    alignment='Discard edge requests; count only complete interior requests, and full one-second power intervals inside their span',
    failure='Stop on observed foreign inference CPU >=0.1 core, GPU >2%, foreign NPU context, unknown telemetry or a failed gate; retain partial evidence',
    scope=__doc__,
)


def summarize_window(rows, start, end):
    """Power rows represent the preceding second, as in llm_energy.inside."""
    selected = [r for r in rows if start + 1 <= r[0] <= end]
    if len(selected) < PROTOCOL['min_rows']:
        raise ValueError(f'Only {len(selected)} valid power rows')
    if any(not all(math.isfinite(v) and v >= 0 for v in r[:4]) or r[1] <= 0 for r in selected):
        raise ValueError('Invalid power/CPU sample')
    stamps = [r[0] for r in selected]
    if any(not 0 < b-a <= PROTOCOL['max_sample_gap_seconds'] for a, b in zip(stamps, stamps[1:])):
        raise ValueError('Nonmonotonic or missing power intervals')
    duration = end-start
    coverage = len(selected)/duration
    if duration <= 0 or not PROTOCOL['min_sample_coverage'] <= coverage <= 1.02:
        raise ValueError(f'Power coverage {coverage}')
    return dict(rows=len(selected), start=start, end=end, seconds=duration,
                sample_coverage=coverage, package_w=statistics.fmean(r[1] for r in selected)/1000,
                package_stdev_w=statistics.pstdev(r[1] for r in selected)/1000,
                cores_w=statistics.fmean(r[2] for r in selected)/1000,
                cpu_percent=statistics.fmean(r[3] for r in selected))


def arm_metrics(idle_rows, active_rows, requests):
    first, last = requests[0]['start'], requests[-1]['end']
    interior = [r for r in requests if r['start'] >= first+1 and r['end'] <= last-1]
    if not interior:
        raise ValueError('No complete interior requests')
    active = summarize_window(active_rows, interior[0]['start'], interior[-1]['end'])
    idle = summarize_window(idle_rows['rows'], idle_rows['start']+1, idle_rows['end']-1)
    if idle['cpu_percent'] > PROTOCOL['idle_max_cpu_percent'] or idle['package_stdev_w'] > PROTOCOL['idle_max_stdev_w']:
        raise ValueError('Disturbed loaded-idle baseline')
    tokens = sum(len(r['tokens']) for r in interior)
    rate = tokens/active['seconds']
    delta = active['package_w']-idle['package_w']
    if delta <= 0:
        raise ValueError('Nonpositive above-idle package power')
    return dict(idle=idle, active=active, complete_requests=len(interior), tokens=tokens,
                tokens_per_second=rate, delta_package_w=delta,
                package_j_per_token=active['package_w']/rate,
                above_idle_j_per_token=delta/rate)


def evaluate(arms):
    if len(arms) != len(PROTOCOL['order']):
        raise ValueError('Incomplete mirrored comparison')
    baselines = [a['metrics']['idle']['package_w'] for a in arms]
    center = statistics.median(baselines)
    if max(abs(v-center) for v in baselines) > PROTOCOL['idle_max_shift_w']:
        raise ValueError('Loaded-idle baseline drift exceeds frozen gate')
    repeats = {}
    for mode in ('gpu', 'hybrid', 'hybrid_activity'):
        pair = [a['metrics'] for a in arms if a['mode'] == mode]
        differences = {name: abs(pair[0][name]-pair[1][name])/statistics.fmean(p[name] for p in pair)
                       for name in ('package_j_per_token', 'above_idle_j_per_token')}
        if max(differences.values()) > PROTOCOL['repeat_max_relative_difference']:
            raise ValueError(f'{mode} energy repeat disagreement: {differences}')
        repeats[mode] = differences
    return dict(repeat_relative_differences=repeats, baseline_median_w=center)


class ContentionMonitor:
    """Independent CPU/GPU/NPU witnesses throughout idle and active windows."""
    def __init__(self, backend):
        self.backend = backend
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.ready = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def own(self):
        gpu = self.backend.gpu
        own = owned_process_ids(ROOT, gpu.process.pid if gpu and gpu.process.poll() is None else None)
        helper = self.backend._activity
        if helper and helper.process.poll() is None:
            own.add(helper.process.pid)
        return own

    def run(self):
        try:
            gpu = _GpuQuery()
            try:
                before, known = _cpu_times()
                while not self.stop.wait(1):
                    t0 = time.perf_counter()
                    after, known_after = _cpu_times()
                    own = self.own()
                    # One-second interval plus the previous ownership query.
                    elapsed = t0-getattr(self, '_last', t0-1)
                    foreign_cpu = []
                    for key, (name, value) in after.items():
                        old = before.get(key)
                        if old is None:
                            if key[0] not in own:
                                raise RuntimeError('New inference process; no complete CPU interval')
                            continue
                        cores = max(0, (value-old[1])/elapsed)
                        if key[0] not in own and cores >= .1:
                            foreign_cpu.append(dict(pid=key[0], name=name, cores=cores))
                    gpu_known, engines, details = gpu.sample()
                    foreign_gpu = [e for e in engines if e['pid'] not in own and e['utilization_percent'] > 2]
                    witness = context_report()
                    foreign_npu = sorted(context_pids(witness)-{os.getpid()})
                    row = dict(utc=time.time(), foreign_cpu=foreign_cpu, foreign_gpu=foreign_gpu,
                               foreign_npu=foreign_npu, gpu_details=details,
                               telemetry_known=known and known_after and gpu_known)
                    self.rows.append(row)
                    if not row['telemetry_known'] or foreign_cpu or foreign_gpu or foreign_npu:
                        raise RuntimeError(f'Contention or unavailable telemetry: {row}')
                    self.ready.set()
                    before, known, self._last = after, known_after, t0
            finally:
                gpu.close()
        except BaseException as error:
            self.rows.append(dict(utc=time.time(), error=repr(error)))
            self.failed.set()
            self.ready.set()

    def check(self):
        if self.failed.is_set():
            raise RuntimeError(f'Energy window rejected: {self.rows[-1]}')

    def wait(self, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.check()
            self.failed.wait(min(.25, deadline-time.monotonic()))
        self.check()

    def __enter__(self):
        self.thread.start()
        try:
            if not self.ready.wait(20):
                raise TimeoutError('Contention monitor did not become ready')
            self.check()
        except BaseException:
            self.stop.set()
            self.thread.join(20)
            raise
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(20)
        if self.thread.is_alive():
            raise RuntimeError('Contention monitor did not close')
        if exc[0] is None:
            self.check()


@contextmanager
def sampler(path):
    if path.exists():
        raise FileExistsError(path)
    process = energy.typeperf(180, path)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=15)


def gpu_request(backend, prompt):
    backend._preflight()
    state = backend._prepare(prompt)
    tokens = []
    for _ in range(PROTOCOL['tokens']):
        token = state['predictions'][-1]
        tokens.append(token)
        if backend.gpu.render([token])['eog'][0]:
            break
        if len(tokens) < PROTOCOL['tokens']:
            state = backend.gpu.decode([token])
    backend.gpu.render(tokens)
    return tokens


def request(assistant, mode, prompt, reference, *, retain_result=False):
    start, mono = time.time(), time.perf_counter()
    evidence = {}
    if mode == 'gpu':
        tokens = gpu_request(assistant._backend, prompt)
    else:
        stream = assistant.stream_tokens(prompt, max_new_tokens=PROTOCOL['tokens'], depth=PROTOCOL['depth'])
        try:
            tokens = [t for chunk in stream for t in chunk.token_ids]
        finally:
            stream.close()
        result = assistant.last_result
        if not all(result.get(k) for k in ('completed', 'npu_released', 'npu_idle')):
            raise RuntimeError('Request cleanup or completion failed')
        activity = result['gpu_activity']
        if not activity['idle']:
            raise RuntimeError('GPU activity did not join')
        if mode == 'hybrid_activity' and (not activity['windows'] or not all(w['gpu_validated'] for w in activity['windows'])):
            raise RuntimeError('Unchecked activity work')
        evidence = dict(operation_ms=result['operation_ms'], gpu_activity=activity,
                        accepted=result['accepted'], proposed=result['proposed'])
        if retain_result:
            evidence['runtime_result'] = result
    end = time.time()
    elapsed = time.perf_counter()-mono
    if abs((end-start)-elapsed) > .1:
        raise RuntimeError('Wall clock changed during request')
    if tokens != reference or len(tokens) != PROTOCOL['tokens']:
        raise RuntimeError('Greedy token control mismatch or early end')
    return dict(start=start, end=end, elapsed_seconds=elapsed, tokens=tokens, **evidence)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--llama-runtime', type=Path, required=True)
    p.add_argument('--gpu-activity-helper', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prereg-only', action='store_true')
    p.add_argument('--prereg', type=Path, help='Run against an existing, byte-pinned protocol; never replace it')
    args = p.parse_args()
    prereg = args.prereg or args.out.with_suffix('.prereg.json')
    data = args.out.with_suffix('.samples')
    if args.out.exists() or data.exists() or (prereg.exists() and args.prereg is None):
        raise FileExistsError(args.out)
    prompt_path = args.artifacts/'npu/live-public-dictionary-20261018.i32'
    paths = [Path(__file__).resolve(), Path(energy.__file__), ROOT/'tools/power_probe.py',
             ROOT/'npu/contention.py', ROOT/'tools/research_process.py',
             ROOT/'Ignition/src/ignition/experimental/hybrid.py', prompt_path]
    paths += list((ROOT/'npu/gemma4').glob('*.py'))
    paths += [args.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf',
              args.artifacts/'npu/gemma-4-E2B-it-assistant.weights.npz',
              args.artifacts/'bin/ggml_gpu_target.exe', args.gpu_activity_helper]
    sources = {str(path.resolve()): sha(path) for path in paths}
    frozen = dict(protocol=PROTOCOL, source_sha256=sources)
    if args.prereg is not None:
        if json.loads(prereg.read_text()) != frozen:
            raise ValueError('Frozen protocol or source identity changed')
    else:
        save(prereg, frozen)
    if args.prereg_only:
        print(f'Frozen protocol: {prereg}', flush=True)
        return
    prompt = np.fromfile(prompt_path, np.int32).tolist()
    data.mkdir()
    report = dict(passed=False, energy_eligible=False, timing_eligible=False,
                  scope=__doc__, protocol=PROTOCOL, prereg_sha256=sha(prereg),
                  source_sha256=sources, arms=[])
    save(args.out, report)
    try:
        for index, mode in enumerate(PROTOCOL['order']):
            arm = dict(index=index, mode=mode, requests=[], monitor=[])
            report['arms'].append(arm)
            with HybridAssistant(study_root=ROOT, artifacts=args.artifacts, llama_runtime=args.llama_runtime,
                                 input_depth=2, vector_copy=True, collect_trace=True,
                                 gpu_activity_helper=args.gpu_activity_helper if mode == 'hybrid_activity' else None) as assistant:
                backend = assistant._backend
                reference = gpu_request(backend, prompt)
                arm['source_sha256'] = backend.sources
                arm['warmup'] = request(assistant, mode, prompt, reference, retain_result=True)
                worker = backend.gpu.process
                monitor = ContentionMonitor(backend)
                arm['monitor'] = monitor.rows
                with monitor:
                    monitor.wait(PROTOCOL['settle_seconds'])
                    idle_path = data/f'{index}_{mode}_idle.csv'
                    with sampler(idle_path):
                        monitor.wait(PROTOCOL['counter_lead_seconds'])
                        start = time.time()
                        mono = time.perf_counter()
                        monitor.wait(PROTOCOL['idle_seconds'])
                        end = time.time()
                        if abs(end-start-(time.perf_counter()-mono)) > .1:
                            raise RuntimeError('Wall clock changed during idle')
                    idle = dict(start=start, end=end, rows=energy.read_csv(idle_path))
                    arm.update(idle=idle, idle_csv=str(idle_path), idle_csv_sha256=sha(idle_path))
                    # Fail before active work when the baseline is already invalid.
                    idle_stats = summarize_window(idle['rows'], start+1, end-1)
                    if idle_stats['cpu_percent'] > PROTOCOL['idle_max_cpu_percent'] or idle_stats['package_stdev_w'] > PROTOCOL['idle_max_stdev_w']:
                        raise RuntimeError('Disturbed loaded-idle baseline')
                    save(args.out, report)
                    print(f'ARM {index} {mode}: loaded-idle baseline accepted', flush=True)
                    backend._preflight()
                    active_path = data/f'{index}_{mode}_active.csv'
                    with sampler(active_path):
                        monitor.wait(PROTOCOL['counter_lead_seconds'])
                        deadline = time.monotonic()+PROTOCOL['active_min_seconds']
                        while time.monotonic() < deadline:
                            monitor.check()
                            arm['requests'].append(request(assistant, mode, prompt, reference))
                        monitor.check()
                    arm.update(active_rows=energy.read_csv(active_path), active_csv=str(active_path),
                               active_csv_sha256=sha(active_path))
                    arm['metrics'] = arm_metrics(idle, arm['active_rows'], arm['requests'])
                if backend.model_loads != 1 or worker.poll() is not None:
                    raise RuntimeError('Model worker was reloaded or exited')
            if worker.poll() is None:
                raise RuntimeError('Owned GPU worker did not close')
            if any(sha(Path(path)) != value for path, value in arm['source_sha256'].items()):
                raise RuntimeError('Runtime source identity changed')
            save(args.out, report)
            print('ARM_RESULT '+json.dumps(dict(index=index, mode=mode, **arm['metrics'])), flush=True)
        if any(sha(Path(path)) != value for path, value in sources.items()):
            raise RuntimeError('Measurement source identity changed')
        report['evaluation'] = evaluate(report['arms'])
        report.update(passed=True, energy_eligible=True)
        print('PASS: matched package-energy diagnostics; scope remains one public workload', flush=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        save(args.out, report)


if __name__ == '__main__':
    main()
