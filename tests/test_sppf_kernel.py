"""Standalone silicon qualification for SPPF; use scripts/sppf.sh.

CPU oracle: torch.nn.MaxPool2d(5, 1, 2), exact integer-valued float32.
Hardware mode compares every byte of all four concatenands, with poisoned output.
The latency gate covers the whole 16-core chain, including DMA and XRT start/wait;
input writes, output readback, oracle calculation and initialization are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
_STAGE_PATH = ROOT / 'kernels' / 'aie2' / 'sppf' / 'sppf_stage.py'
_STAGE_SPEC = importlib.util.spec_from_file_location('ignite_xdna_sppf_stage', _STAGE_PATH)
if _STAGE_SPEC is None or _STAGE_SPEC.loader is None:
    raise ImportError(f'cannot load SPPF stage module from {_STAGE_PATH}')
_STAGE = importlib.util.module_from_spec(_STAGE_SPEC)
_STAGE_SPEC.loader.exec_module(_STAGE)
INPUT_BYTES = _STAGE.INPUT_BYTES
OUTPUT_BYTES = _STAGE.OUTPUT_BYTES
compile_design = _STAGE.compile_design


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), sort_keys=True), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: digest(ROOT / 'kernels/aie2/sppf' / name)
            for name in ('maxpool5x5.cc', 'sppf_stage.py')}


def artifact_dir(base, trace=False, repeats=1):
    identity = dict(sources=sources(), trace=trace, repeats=repeats)
    return base.resolve() / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def fixture(seed, case):
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (20, 20, 256), dtype=np.int16).astype(np.int8)
    if case == 'negative':
        x = rng.integers(-128, 0, x.shape, dtype=np.int16).astype(np.int8)
    elif case == 'minimum':
        x.fill(-128)
    elif case == 'spatial':
        yy, xx, cc = np.indices(x.shape)
        x = ((yy * 7 + xx * 11 + cc * 3 + seed) % 256 - 128).astype(np.int8)
    elif case == 'impulses':
        x.fill(-128)
        for ch in range(256):
            x[(ch + seed) % 20, (ch // 20 + seed) % 20, ch] = ch % 255 - 127
    elif case != 'random':
        raise ValueError(case)
    return x


def oracle(x):
    import torch
    torch.set_num_threads(1)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).float()
    pool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    stages = [t]
    for _ in range(3):
        stages.append(pool(stages[-1]))
    return torch.cat(stages, dim=1).squeeze(0).permute(1, 2, 0).numpy().astype(np.int8)


def offline():
    # A separate direct 25-tap integer oracle also checks negative padding and layout.
    for case in ('random', 'negative', 'minimum', 'spatial', 'impulses'):
        x = fixture(17, case)
        stages = [x]
        for _ in range(3):
            p = np.pad(stages[-1], ((2, 2), (2, 2), (0, 0)), constant_values=-128)
            stages.append(np.maximum.reduce([p[y:y+20, z:z+20] for y in range(5) for z in range(5)]))
        np.testing.assert_array_equal(oracle(x), np.concatenate(stages, axis=2))
    emit('offline_pass', cases=5, oracle='torch.nn.MaxPool2d(5,1,2)',
         output_bytes_per_case=OUTPUT_BYTES, hardware=False)


def preflight():
    for command, expected in (([r'C:\Windows\System32\AMD\xrt-smi.exe', 'examine', '-r', 'aie-partitions'],
                               'No hardware contexts running'),
                              (['powershell', '-NoProfile', '-File', str(ROOT / 'tools/host_load.ps1')],
                               'HOST_LOAD_VERDICT CLEAR')):
        run = subprocess.run(command, capture_output=True, text=True, check=True, timeout=25)
        print(run.stdout, flush=True)
        if expected not in run.stdout:
            raise RuntimeError('Host/device preflight failed')
        if command[0].endswith('xrt-smi.exe') and '[003d:00:01.1] : NPU Phoenix' not in run.stdout:
            raise RuntimeError('Expected physical Phoenix Device 0 [003d:00:01.1]')


def compile_bundle(directory, trace=False, repeats=1):
    from aie.utils import config
    compile_design(directory, trace, repeats)
    peano = Path(config.peano_install_dir()) / 'bin'
    elfs = list((directory / 'design.prj').rglob('*.elf'))
    if len(elfs) != 16:
        raise AssertionError(f'Expected 16 core ELFs, got {len(elfs)}')
    obj = directory / 'design.prj/maxpool5x5.o'
    dump = subprocess.check_output([str(peano / 'llvm-objdump.exe'), '-d', str(obj)], text=True)
    obj.with_suffix('.disassembly.txt').write_text(dump, encoding='utf-8')
    if not re.search(r'\bvmax', dump):
        raise AssertionError('No native vector maximum in compiled object')
    manifest = {'sources': sources(), 'trace': trace, 'repeats': repeats, 'compiler': subprocess.check_output(
        [str(peano / 'clang++.exe'), '--version'], text=True),
        'artifacts': {p.relative_to(directory).as_posix(): digest(p)
                      for p in directory.rglob('*') if p.suffix in ('.elf', '.o', '.bin', '.xclbin')}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    emit('compile_pass', cores=len(elfs), vector_max_instructions=len(re.findall(r'\bvmax', dump)),
         manifest=manifest)


def hardware(args, directory):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from ignite_xdna.runtime.splice import shim_patches
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['sources'] != sources():
        raise RuntimeError('Sources changed; recompile')
    for name, expected in manifest['artifacts'].items():
        if digest(directory / name) != expected:
            raise RuntimeError(f'Artifact mismatch: {name}')
    emit('identity', machine='Desktop 2 / Ryzen 7 8700G', device=0, bdf='003d:00:01.1',
         xclbin_sha256=digest(directory / 'design.xclbin'),
         transaction_sha256=digest(directory / 'insts.bin'), sources=sources(),
         test_sha256=digest(__file__),
         shim_patches=[dict(argument=p.argument, offset=p.offset, register=p.register)
                       for p in shim_patches((directory / 'insts.bin').read_bytes())])
    h = XrtSiliconHarness(0)
    h.load_xclbin(str(directory / 'design.xclbin'))
    instr, length = h.create_instruction_bo(str(directory / 'insts.bin'))
    inp = h.create_host_bo(INPUT_BYTES, 3)
    out = h.create_host_bo(OUTPUT_BYTES, 4)
    trace_bo = h.create_host_bo(65536, 5) if args.trace else None
    run = h.pyxrt.run(h.kernel)
    for argument, value in enumerate((3, instr, length, inp, out)):
        run.set_arg(argument, value)
    if trace_bo is not None:
        run.set_arg(5, trace_bo)
    todev = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    fromdev = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    samples = []
    witness = directory / ('run_' + str(time.time_ns()))
    witness.mkdir()
    cases = [(seed, case, False) for seed in (17, 91)
             for case in ('random', 'negative', 'minimum', 'spatial', 'impulses')]
    cases += [(200 + i, 'random', i >= args.warmup) for i in range(args.warmup + args.iters)]
    if args.trace:
        cases = [(17, 'spatial', True)]
    for index, (seed, case, measured) in enumerate(cases):
        x = fixture(seed, case)
        expected = oracle(x)
        wire_input = x.reshape(20, 20, 4, 4, 16).transpose(2, 3, 0, 1, 4).copy()
        wire_expected = expected.reshape(20, 20, 4, 4, 64).transpose(3, 0, 1, 2, 4).copy()
        inp.write(wire_input.tobytes(), 0)
        inp.sync(todev)
        out.write(np.bitwise_xor(wire_expected.view(np.uint8), np.uint8(255)).tobytes(), 0)
        out.sync(todev)
        if trace_bo is not None:
            trace_bo.write(bytes(65536), 0)
            trace_bo.sync(todev)
        start = time.perf_counter_ns()
        run.start()
        state = run.wait(5000)
        elapsed = (time.perf_counter_ns() - start) / 1000
        if str(state) != 'ert_cmd_state.ERT_CMD_STATE_COMPLETED':
            raise RuntimeError(f'Hardware completion failed: {state}')
        out.sync(fromdev)
        wire_actual = np.frombuffer(out.read(OUTPUT_BYTES, 0), dtype=np.int8).reshape(4, 20, 20, 4, 64)
        actual = wire_actual.transpose(1, 2, 3, 0, 4).copy().reshape(20, 20, 1024)
        mismatches = np.count_nonzero((actual != expected).reshape(20, 20, 4, 4, 64), axis=(0, 1, 4))
        emit('comparison', index=index, seed=seed, case=case, measured=measured,
             mismatches=int(mismatches.sum()), per_stage_column_mismatches=mismatches.tolist(),
             bytes_checked=OUTPUT_BYTES, dispatch_wait_us=elapsed,
             input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
             output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
        if index < 10 or mismatches.any():
            np.savez_compressed(witness / f'{index}_{case}.npz', input=x, expected=expected, actual=actual)
        if mismatches.any():
            raise AssertionError('Full SPPF output failed bit-exact PyTorch parity')
        if measured:
            samples.append(elapsed)
        if trace_bo is not None:
            trace_bo.sync(fromdev)
            words = np.frombuffer(trace_bo.read(65536, 0), dtype=np.uint32)
            np.save(witness / 'trace.npy', words)
            from aie.utils.trace.utils import convert_to_byte_stream, trace_pkts_de_interleave
            from kernels.clock_probe.clock_probe import decode_event_time
            valid = np.flatnonzero(words)
            streams = convert_to_byte_stream(trace_pkts_de_interleave(
                [f'{int(w):08x}' for w in words[:((int(valid[-1]) // 8) + 1)*8]]))
            columns = []
            for loc, bs in streams[2].items():
                events = decode_event_time(bs)
                begins = [stamp for slot, stamp in events if slot == 0]
                ends = [stamp for slot, stamp in events if slot == 1]
                starts = [i for i, b in enumerate(bs) if (b & 0xFB) == 0xF0]
                emit('trace_decode', location=loc, begins=begins, ends=ends,
                     first_bytes=bs[:16], starts=starts[:5], event_count=len(events))
                if len(begins) != args.repeats or len(ends) != args.repeats:
                    raise AssertionError('Incomplete Shim DMA event coverage')
                columns.append(dict(location=loc, cycles=ends[-1]-begins[0],
                                    per_frame_cycles=[e-b for b, e in zip(begins, ends)]))
            if len(columns) != 4:
                raise AssertionError(f'Expected four traced Shim columns, got {columns}')
            emit('trace_result', columns=columns, host_us=elapsed, repeats=args.repeats,
                 trace_sha256=digest(witness / 'trace.npy'), qualification='cycles only; frequency calibration required')
            return
    result = dict(correctness='PASS', cores=16, samples=len(samples),
                  min_us=min(samples), median_us=statistics.median(samples), max_us=max(samples),
                  target_us=150, timing_scope='whole array XRT dispatch through completion, includes DMA',
                  latency_verdict='PASS' if max(samples) < 150 else 'FAIL')
    emit('hardware_result', **result)
    (witness / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if max(samples) >= 150:
        raise AssertionError('Total SPPF execution exceeded 150 us')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--build-dir', type=Path, default=ROOT / 'build/sppf')
    args = p.parse_args()
    if args.iters < 1 or args.warmup < 0:
        p.error('Need positive iters and nonnegative warmup')
    offline()
    directory = artifact_dir(args.build_dir, args.trace, args.repeats)
    if args.compile:
        preflight()
        compile_bundle(directory, args.trace, args.repeats)
    if args.hardware:
        preflight()
        hardware(args, directory)


if __name__ == '__main__':
    main()
