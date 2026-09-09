"""Test whether local operand placement transfers to tiled GEMM, with all C checked.

This is a single-core INT8 matrix kernel. It does not replace the repository's
whole-array GEMM or measure stream bandwidth, peak MAC rate, or DMA overlap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent

from probe import disassemble_function, TraceReader, fit_line, platform_report, _npu_time_ns


@iron.jit
def gemm_design(params: In, result: Out, *, b_address: CompileTime[int] = 0x8000,
                trace_size: CompileTime[int] = 65536):
    p_ty = np.ndarray[(64,), np.dtype[np.int32]]
    c_ty = np.ndarray[(4096,), np.dtype[np.int32]]
    ab_ty = np.ndarray[(8192,), np.dtype[np.int8]]
    fn = ExternalFunction('gemm_placement', object_file_name='gemm_placement.o',
            source_file=str(Path(__file__).with_name('gemm_placement.cc')),
            arg_types=[p_ty, c_ty, ab_ty, ab_ty, ab_ty], include_dirs=[config.cxx_header_path()])
    aa = Buffer(ab_ty, name='operand_a', address=0x4000)
    bb = Buffer(ab_ty, name='operand_b', address=b_address)
    reserve = Buffer(ab_ty, name='bank2_reserve', address=0xA000)
    inp = ObjectFifo(p_ty, name='parameters', depth=1)
    out = ObjectFifo(c_ty, name='gemm_result', depth=1)
    def core(ih, oh, av, bv, rv, kernel):
        i, o = ih.acquire(1), oh.acquire(1)
        kernel(i, o, av, bv, rv)
        ih.release(1)
        oh.release(1)
    worker = Worker(core, [inp.cons(), out.prod(), aa, bb, reserve, fn], tile=Tile(0, 2), trace=1)
    def sequence(p, c, ih, oh):
        ih.fill(p)
        oh.drain(c, wait=True)
    runtime = Runtime(sequence, [p_ty, c_ty, inp.prod(tile=Tile(0, 0)), out.cons(tile=Tile(0, 0))])
    program = Program(iron.get_current_device(), runtime, workers=[worker])
    program.enable_trace(trace_size=trace_size, workers=[worker],
                         coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1])
    return program.resolve_program()


def reference(seed):
    j = np.arange(8192, dtype=np.int64)
    aa = ((j+seed) % 7-3).reshape(2, 4096)
    bb = ((j*3+seed+1) % 7-3).reshape(2, 4096)
    products = []
    for a, b in zip(aa, bb):
        # Memory has row-major tiles of A(4x8), B(8x8), C(4x8).
        a = a.reshape(16, 8, 4, 8).transpose(0, 2, 1, 3).reshape(64, 64)
        b = b.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(64, 64)
        c = a @ b
        products.append(c.reshape(16, 4, 8, 8).transpose(0, 2, 1, 3).copy().ravel())
    return products


def artifacts(out, address):
    handles = list(gemm_design._kernel_cache.values())
    if len(handles) != 1:
        raise RuntimeError('expected one runtime specialization')
    directory = Path(handles[0].xclbin_path).resolve().parent
    if directory.parent != Path(os.environ['NPU_CACHE_HOME']).resolve():
        raise RuntimeError('artifact escaped private cache')
    report = {'a_address': 0x4000, 'b_address': address, 'reserved_address': 0xA000,
              'cache_directory': str(directory), 'final_functions': [], 'buffers': []}
    obj = directory / 'gemm_placement.o'
    report['object_sha256'] = hashlib.sha256(obj.read_bytes()).hexdigest()
    for path in [obj, *directory.rglob('*.elf')]:
        dump, binary = disassemble_function(path, 'gemm_placement')
        if dump is None:
            continue
        shutil.copyfile(path, out / path.name)
        (out / (path.name+'.txt')).write_text(dump, encoding='utf-8')
        if path.suffix == '.elf':
            report['final_functions'].append({'elf': path.name, 'function_bytes': len(binary),
                'function_sha256': hashlib.sha256(binary).hexdigest()})
    for path in directory.rglob('*.mlir'):
        for line in path.read_text(encoding='utf-8').splitlines():
            if 'aie.buffer' in line and 'mem_bank' in line and line.strip() not in report['buffers']:
                report['buffers'].append(line.strip())
    c_lines = [s for s in report['buffers'] if 'sym_name = "gemm_result' in s]
    if not c_lines or any('address = 49152 ' not in s for s in c_lines):
        raise RuntimeError('C placement is not fixed at 0xC000: '+repr(c_lines))
    if not report['final_functions']:
        raise RuntimeError('missing final function')
    print('GEMM_PLACEMENT_ARTIFACTS', json.dumps(report), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layout', choices=['same', 'separate'], default='separate')
    ap.add_argument('--alternate', action='store_true')
    ap.add_argument('--targets', default='1,2,4,8,16,32')
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--pause-ms', type=float, default=100, help='Host pacing outside every measured event bracket')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    a = ap.parse_args()
    targets = [int(x) for x in a.targets.split(',')]
    if a.out.exists() or a.iters <= 0 or not all(1 <= n <= 128 for n in targets) or not 0 <= a.seed < 128 or not 0 <= a.pause_ms <= 1000:
        ap.error('need new output, positive iters, targets 1..128, seed 0..127')
    a.out.mkdir(parents=True)
    tc = TraceConfig(trace_size=65536, trace_file=str(a.out / 'trace.txt'))
    gemm_design.trace_config = tc
    reader = TraceReader(tc)
    address = 0x6000 if a.layout == 'same' else 0x8000
    inp = iron.zeros(64, dtype=np.int32, device='npu')
    out = iron.zeros(4096, dtype=np.int32, device='npu')
    prod = reference(a.seed)
    raw = {}
    def call(n):
        inp[0], inp[1], inp[2] = n, int(a.alternate), a.seed
        t = time.perf_counter()
        runtime = gemm_design(inp, out, b_address=address)
        wall = time.perf_counter()-t
        actual = out.numpy().copy()
        expected = prod[0]*n if not a.alternate else prod[0]*((n+1)//2)+prod[1]*(n//2)
        if not np.array_equal(actual, expected):
            raise AssertionError(f'GEMM mismatch: panels={n}, elements={np.count_nonzero(actual != expected)}')
        first, last, pairs, _ = reader.stamps()
        if first is None or last is None or last <= first or pairs < 200 or reader.tile != (2, 1):
            raise AssertionError('trace timing, flush, or physical tile check failed')
        raw[str(n)] = actual
        time.sleep(a.pause_ms / 1000)
        return {'panels': n, 'cycles': last-first, 'wall_seconds': wall,
                'hardware_ns': _npu_time_ns(runtime), 'checked_elements': actual.size}
    print('GEMM_PLACEMENT_PLATFORM', platform_report(), flush=True)
    call(1)
    art = artifacts(a.out, address)
    if a.reference:
        prev = json.loads(a.reference.read_text(encoding='utf-8'))['artifacts']
        if art['object_sha256'] != prev['object_sha256'] or art['final_functions'] != prev['final_functions']:
            raise AssertionError('placement changed GEMM function bytes')
    rows = []
    for n in targets:
        call(n)
        samples = [call(n) for _ in range(a.iters)]
        rows.extend(samples)
        print('GEMM_PLACEMENT_SAMPLES', json.dumps(samples), flush=True)
    means = [statistics.mean(r['cycles'] for r in rows if r['panels'] == n) for n in targets]
    intercept, slope, r2 = fit_line(targets, means)
    report = {'layout': a.layout, 'alternate_panels': a.alternate, 'seed': a.seed,
              'pause_ms': a.pause_ms,
              'shape': '64 x (64*panels) x 64', 'dma_during_bracket': False,
              'fit': {'intercept': intercept, 'slope': slope, 'r2': r2},
              'rows': rows, 'artifacts': art}
    np.savez_compressed(a.out / 'outputs.npz', **raw)
    (a.out / 'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('GEMM_PLACEMENT_RESULT', json.dumps({k:v for k,v in report.items() if k not in ('rows','artifacts')}), flush=True)


if __name__ == '__main__':
    main()
