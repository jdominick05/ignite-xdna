"""Causal AIE operand-placement probe, using the existing event-time decoder.

Run with scripts/research-lowlevel.sh --npu -- bash scripts/research-iron.sh ...
Same/disjoint allocator-bank placement is an intervention, not a claim about the
physical banking function. The final external function's instruction bytes must
match across variants before a cycle difference is attributed to memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'kernels/clock_probe'))
from clock_probe import TraceReader, fit_line, _npu_time_ns, platform_report


@iron.jit
def memory_design(a_in: In, c_out: Out, *, b_address: CompileTime[int] = 0x8000,
                  a_offset: CompileTime[int] = 0, trace_size: CompileTime[int] = 65536):
    io_ty = np.ndarray[(64,), np.dtype[np.int32]]
    operand_ty = np.ndarray[(512,), np.dtype[np.int16]]
    fn = ExternalFunction('memory_probe', object_file_name='memory_kernels.o',
            source_file=str(Path(__file__).with_name('memory_kernels.cc')),
            arg_types=[io_ty, io_ty, operand_ty, operand_ty], include_dirs=[config.cxx_header_path()])
    aa = Buffer(operand_ty, name='operand_a', address=0x4000+a_offset)
    bb = Buffer(operand_ty, name='operand_b', address=b_address)
    inp = ObjectFifo(io_ty, name='parameters')
    out = ObjectFifo(io_ty, name='result')
    def core(inh, outh, av, bv, kernel):
        i = inh.acquire(1)
        o = outh.acquire(1)
        kernel(i, o, av, bv)
        inh.release(1)
        outh.release(1)
    worker = Worker(core, [inp.cons(), out.prod(), aa, bb, fn], tile=Tile(0, 2),
                    trace=1 if trace_size else None)
    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)
    runtime = Runtime(sequence, [io_ty, io_ty, inp.prod(tile=Tile(0, 0)), out.cons(tile=Tile(0, 0))])
    program = Program(iron.get_current_device(), runtime, workers=[worker])
    if trace_size:
        program.enable_trace(trace_size=trace_size, workers=[worker],
                             coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1])
    return program.resolve_program()


def expected(iterations, mode, seed):
    j = np.arange(512, dtype=np.int64)
    aa = (j+seed) % 7-3
    bb = (j*3+seed+1) % 7-3
    values = aa*bb if mode == 0 else aa
    rounds, tail = divmod(iterations, 32)
    return int(values.sum()*rounds + values[:tail*16].sum())


def disassemble_function(path, function='memory_probe'):
    bindir = Path(config.peano_install_dir()) / 'bin'
    symbols = subprocess.check_output([str(bindir / 'llvm-readelf.exe'), '--symbols', '--wide', str(path)], text=True)
    match = re.search(r'^\s*\d+:\s+([0-9a-fA-F]+)\s+(\d+)\s+FUNC\s+.*\s'+re.escape(function)+r'$', symbols, re.M)
    if not match:
        return None, None
    start, size = int(match[1], 16), int(match[2])
    args = ['-d', '--section=.text.'+function] if path.suffix == '.o' else [
        '-d', f'--start-address={start}', f'--stop-address={start+size}']
    # --disassemble-symbols alone stops at the first local basic-block label on
    # this Peano build. Use the ELF symbol size so the measured loop is included.
    result = subprocess.check_output([str(bindir / 'llvm-objdump.exe'), *args, str(path)], text=True)
    raw = ''.join(re.findall(r'^\s*[0-9a-fA-F]+:\s+((?:[0-9a-fA-F]{2}\s+)+)', result, re.M))
    binary = bytes.fromhex(raw)
    if len(binary) != size:
        raise RuntimeError(f'incomplete function byte extraction: {len(binary)} != {size}')
    return result, binary


def artifact_report(out, b_address, a_offset):
    cache = Path(os.environ['NPU_CACHE_HOME']).resolve()
    # This probe calls exactly one specialization. Its live runtime handle is
    # authoritative; the newest filesystem cache can belong to another layout.
    handles = list(memory_design._kernel_cache.values())
    if len(handles) != 1:
        raise RuntimeError('expected exactly one compiled runtime specialization')
    directory = Path(handles[0].xclbin_path).resolve().parent
    if directory.parent != cache:
        raise RuntimeError('runtime artifact escaped the private cache')
    obj = directory / 'memory_kernels.o'
    text, object_bytes = disassemble_function(obj)
    if not text or '<memory_probe>:' not in text:
        raise RuntimeError('objdump did not find the measured function')
    if not any('vlda' in line and 'vldb' in line for line in text.splitlines()):
        raise RuntimeError('compiler did not bundle the two operand loads together')
    (out / 'kernel_object_disassembly.txt').write_text(text, encoding='utf-8')
    shutil.copyfile(obj, out / obj.name)
    report = {'object_sha256': hashlib.sha256(obj.read_bytes()).hexdigest(),
              'cache_directory': str(directory), 'a_address_requested': 0x4000+a_offset,
              'b_address_requested': b_address, 'final_functions': [], 'buffer_declarations': []}
    for path in directory.rglob('*.elf'):
        result, binary = disassemble_function(path)
        if not result:
            continue
        target = out / ('core_'+path.name)
        shutil.copyfile(path, target)
        target.with_suffix('.disassembly.txt').write_text(result, encoding='utf-8')
        report['final_functions'].append({'elf': path.name, 'function_bytes': len(binary),
              'function_sha256': hashlib.sha256(binary).hexdigest()})
    for path in directory.rglob('*.mlir'):
        for line in path.read_text(encoding='utf-8').splitlines():
            if 'aie.buffer' in line and ('operand_a' in line or 'operand_b' in line):
                if line.strip() not in report['buffer_declarations']:
                    report['buffer_declarations'].append(line.strip())
    if not report['final_functions']:
        raise RuntimeError('no final core ELF with memory_probe found')
    print('PLACEMENT_ARTIFACTS', json.dumps(report), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layout', choices=['same', 'separate'], default='separate')
    ap.add_argument('--a-offset', type=int, choices=[0, 64, 128, 256], default=0)
    ap.add_argument('--targets', default='512,1024,2048,4096,8192,16384')
    ap.add_argument('--mode', choices=['dual', 'single'], default='dual')
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--trace-size', type=int, choices=[0, 65536], default=65536)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', type=Path, help='Other variant result.json; require identical final instruction bytes')
    ap.add_argument('--compile-only', action='store_true', help='Compile just the external object; no NPU or performance claim')
    a = ap.parse_args()
    targets = [int(x) for x in a.targets.split(',')]
    if a.out.exists() or a.iters < 1 or a.warmup < 0 or not all(1 <= n <= 2**20 for n in targets) or not 0 <= a.seed < 128:
        ap.error('need new output, positive iterations/targets <=2^20, nonnegative warmup and seed <128')
    a.out.mkdir(parents=True)
    if a.compile_only:
        from aie.utils.compile.utils import compile_cxx_core_function
        target = (a.out / 'memory_kernels.o').resolve()
        compile_cxx_core_function(str(Path(__file__).with_name('memory_kernels.cc')), 'aie2', str(target))
        dump, binary = disassemble_function(target)
        (a.out / 'object_disassembly.txt').write_text(dump, encoding='utf-8')
        print('PLACEMENT_COMPILE_ONLY_PASS', json.dumps({'object_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'function_bytes': len(binary), 'bundled_dual_load': any('vlda' in line and 'vldb' in line for line in dump.splitlines()),
              'disassembly': dump}), flush=True)
        return
    mode = 0 if a.mode == 'dual' else 1
    address = 0x4800 if a.layout == 'same' else 0x8000
    reader = None
    if a.trace_size:
        tc = TraceConfig(trace_size=a.trace_size, trace_file=str(a.out / 'trace.txt'))
        memory_design.trace_config = tc
        reader = TraceReader(tc)
    print('PLACEMENT_PLATFORM', platform_report(), flush=True)
    inp = iron.zeros(64, dtype=np.int32, device='npu')
    out = iron.zeros_like(inp)
    def call(n):
        inp[0], inp[1], inp[2] = n, mode, a.seed
        t = time.perf_counter()
        result = memory_design(inp, out, b_address=address, a_offset=a.a_offset, trace_size=a.trace_size)
        wall = time.perf_counter()-t
        y = out.numpy()
        if tuple(int(v) for v in y[:3]) != (n, mode, expected(n, mode, a.seed)) or int(y[5]) != a.seed:
            raise AssertionError('kernel checksum or input echo failed')
        cycles = None
        if reader:
            s, e, pairs, count = reader.stamps()
            if s is None or e is None or e <= s or pairs < 200:
                raise AssertionError('missing, reversed or incomplete trace event pairs')
            if reader.tile != (int(y[3]), int(y[4])):
                raise AssertionError('trace/core tile identity mismatch')
            cycles = e-s
        ns = _npu_time_ns(result)
        return {'target': n, 'cycles': cycles, 'hardware_ns': ns, 'wall_seconds': wall,
                'checksum': int(y[2]), 'tile': [int(y[3]), int(y[4])]}
    call(128)
    artifacts = artifact_report(a.out, address, a.a_offset)
    if a.reference:
        other = json.loads(a.reference.read_text(encoding='utf-8'))['artifacts']
        if artifacts['object_sha256'] != other['object_sha256'] or artifacts['final_functions'] != other['final_functions']:
            raise AssertionError('address intervention changed the compiled function')
    rows = []
    for n in targets:
        for _ in range(a.warmup):
            call(n)
        samples = [call(n) for _ in range(a.iters)]
        rows.extend(samples)
        print('PLACEMENT_SAMPLES', json.dumps(samples), flush=True)
    means = [statistics.mean(r['cycles'] if reader else r['hardware_ns'] for r in rows if r['target'] == n) for n in targets]
    intercept, slope, r2 = fit_line(targets, means)
    report = {'layout': a.layout, 'a_offset': a.a_offset, 'mode': a.mode, 'trace_size': a.trace_size,
              'seed': a.seed, 'rows': rows, 'artifacts': artifacts, 'fit': {'intercept': intercept,
              'slope': slope, 'r2': r2, 'units': 'core_cycles_per_iteration' if reader else 'hardware_ns_per_iteration'},
              'reference_function_identical': bool(a.reference), 'dma_during_bracket': False}
    (a.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('PLACEMENT_RESULT', json.dumps({k: v for k, v in report.items() if k not in ('rows', 'artifacts')}), flush=True)


if __name__ == '__main__':
    main()
