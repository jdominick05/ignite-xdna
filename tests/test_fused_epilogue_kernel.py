"""Standalone Phoenix Conv+Residual+SiLU validation (no mocked device).

--offline checks the complete Q8 SiLU domain and packet/reference coverage.
--compile builds Peano objects and 16-core ELF/xclbin/transaction bundles.
--hardware requires those exact artifacts, checks every output byte, and
measures trace cycles in a separate one-core build with identical kernel bytes.
Use scripts/fused-epilogue.sh for host/device witnesses and new evidence logs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
PACKET, OUTPUT, CORES = 11904, 256, 16


def input_bytes(cin):
    return 3264 * (cin // 8) if cin > 32 else PACKET


def emit(event, **values):
    print(json.dumps({'event': event, **values}, sort_keys=True), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded_shift(x, shift):
    # Integer ties-to-even, including negative halves. Does not use float.
    q, r = np.divmod(np.asarray(x, dtype=np.int64), 1 << shift)
    return q + ((r > (1 << (shift - 1))) | ((r == (1 << (shift - 1))) & (q & 1 != 0)))


def polynomial(q):
    x = np.clip(np.asarray(q, dtype=np.int64), -32768, 32767)
    t = np.minimum(np.abs(x), 2048)
    r = 2048 - t
    r2 = rounded_shift(r * r, 11)
    r4 = rounded_shift(r2 * r2, 11)
    r5 = rounded_shift(r4 * r, 11)
    d = rounded_shift(r5 * t, 12)
    return np.clip(rounded_shift(np.maximum(x, 0) - d, 4), -128, 127).astype(np.int8)


def float_silu(q):
    x = np.asarray(q, dtype=np.float32) / np.float32(256)
    # Stable independent float32 sigmoid, with no polynomial shared with kernel.
    z = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
    return np.clip(np.rint(x * sigmoid * np.float32(16)), -128, 127).astype(np.int8)


def pack_output(x):
    return x.reshape(2, 4, 4, 8).transpose(0, 2, 1, 3).copy().ravel()


def fixture(seed, case='random', cin=32):
    rng = np.random.default_rng(seed)
    packets, raw, exact, floating = [], [], [], []
    for core in range(CORES):
        feature = rng.integers(-24, 25, (4, 6, cin), dtype=np.int8)
        weights = rng.integers(-12, 13, (3, 3, cin, 32), dtype=np.int8)
        bias = rng.integers(-256, 257, 32, dtype=np.int32)
        skip = rng.integers(-128, 128, (8, 32), dtype=np.int16).astype(np.int8)
        if case == 'boundaries':
            feature.fill(0)
            weights.fill(0)
            edges = np.array([-32768, -4096, -2049, -2048, -1537, -1536, -1025,
                              -1024, -513, -512, -257, -256, -17, -16, -9, -8,
                              -1, 0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 511,
                              512, 1024, 2048, 32767], dtype=np.int32)
            bias = np.roll(edges, core)
        elif case == 'channel_basis':
            feature.fill(0)
            weights.fill(0)
            # Every Cin and Cout gets a distinct contribution; neither half
            # of M=2 nor any high channel can silently disappear.
            for ch in range(cin):
                feature[:, :, ch] = (ch + core) % 11 - 5
                weights[1, 1, ch, ch % 32] = ch % 7 + 1
            bias.fill(0)
        elif case == 'cancellation':
            feature.fill(0)
            weights.fill(0)
            # Conv alone saturates INT8, but the sum comes back into range.
            bias[:] = np.where(np.arange(32) % 2, 2200, -2200) + core
            skip[:] = np.where(np.arange(32) % 2, -127, 127)
        patches = np.stack([feature[y:y+3, x:x+3] for y in range(2) for x in range(4)])
        # Independent direct convolution over original NHWC feature maps.
        conv = np.stack([np.einsum('hwc,hwco->o', feature[y:y+3, x:x+3].astype(np.int64),
                                     weights.astype(np.int64))
                         for y in range(2) for x in range(4)]) + bias
        conv_float = np.stack([np.einsum('hwc,hwco->o',
                    feature[y:y+3, x:x+3].astype(np.float32) / 16,
                    weights.astype(np.float32) / 16)
                    for y in range(2) for x in range(4)]) + bias.astype(np.float32) / 256
        if not np.array_equal(conv_float * 256, conv):
            raise AssertionError('fixture float32 Conv is not exactly representable')
        # [group, tap, Cin block, pixel, channel] and [tap, Cin block, Cout block, ci, co].
        if cin == 32:
            pa = patches.reshape(2, 4, 9, 4, 8).transpose(0, 2, 3, 1, 4).copy().ravel()
            pw = weights.reshape(9, 4, 8, 4, 8).transpose(0, 1, 3, 2, 4).copy().ravel()
            packed = np.concatenate([pa, pw, bias.view(np.int8), pack_output(skip)])
        else:
            pa = patches.reshape(2, 4, 9, cin // 8, 8).transpose(3, 0, 2, 1, 4).copy().reshape(cin // 8, -1)
            pw = weights.reshape(9, cin // 8, 8, 4, 8).transpose(1, 0, 3, 2, 4).copy().reshape(cin // 8, -1)
            metadata = np.concatenate([bias.view(np.int8), pack_output(skip)])
            packed = np.concatenate([pa, pw, np.tile(metadata, (cin // 8, 1))], axis=1).ravel()
        assert packed.size == input_bytes(cin)
        packets.append(packed)
        raw.append(pack_output(np.clip(rounded_shift(conv, 4), -128, 127).astype(np.int8)))
        q = conv + skip.astype(np.int64) * 16
        exact.append(pack_output(polynomial(q)))
        floating.append(pack_output(float_silu((conv_float + skip.astype(np.float32) / 16) * 256)))
    return tuple(np.stack(x) for x in (packets, raw, exact, floating))


def offline(cin):
    q = np.arange(-32768, 32768, dtype=np.int64)
    err = np.abs(polynomial(q).astype(np.int16) - float_silu(q).astype(np.int16))
    assert err.max() <= 1
    for case in ('random', 'boundaries', 'channel_basis', 'cancellation'):
        p, raw, exact, ref = fixture(17, case, cin)
        assert p.shape == (16, input_bytes(cin)) and raw.shape == (16, OUTPUT)
        assert len({row.tobytes() for row in p}) == 16
        assert np.max(np.abs(exact.astype(np.int16) - ref.astype(np.int16))) <= 1
    emit('offline_pass', q8_inputs_checked=q.size, max_error_output_lsb=int(err.max()),
         scope='CPU arithmetic and fixture validation; no hardware execution')


def compile_bundles(directory, cin):
    import aie.iron as iron
    from aie.iron.device import NPU1
    from kernels.aie2.fused_conv_epilogue.design import fused_design
    from aie.utils import config
    from aie.utils.compile.utils import compile_mlir_module
    from kernels.aie2.fused_conv_epilogue.transport import chunk_output, stream_input
    iron.set_current_device(NPU1())
    manifest = {'source_sha256': digest(ROOT / 'kernels/aie2/fused_conv_epilogue/conv2d_fused.cc'),
                'design_sha256': digest(ROOT / 'kernels/aie2/fused_conv_epilogue/design.py'),
                'transport_sha256': digest(ROOT / 'kernels/aie2/fused_conv_epilogue/transport.py'),
                'cin': cin,
                'stages': {}}
    peano = Path(config.peano_install_dir()) / 'bin'
    manifest['compiler'] = subprocess.check_output([str(peano / 'clang++.exe'), '--version'], text=True)
    for cores, fused in ((16, 0), (16, 1), (1, 0), (1, 1)):
        name = ('fused' if fused else 'raw') + str(cores)
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        emit('compile_begin', variant=name)
        specialization = fused_design.compilable.specialize(fused=fused, cores=cores, cin=cin,
                                                            trace_size=65536 if cores == 1 else 0)
        bootstrap = target / '_bootstrap'
        bootstrap.mkdir(exist_ok=True)
        specialization.compile(xclbin_path=bootstrap / 'design.xclbin', inst_path=bootstrap / 'insts.bin')
        lowered = (bootstrap / 'design.prj/input_with_addresses.mlir').read_text(encoding='utf-8')
        if cin > 32:
            lowered = stream_input(lowered, cores)
        lowered = chunk_output(lowered, cores)
        work = target / 'design.prj'
        work.mkdir(exist_ok=True)
        for obj in (bootstrap / 'design.prj').glob('conv2d_*.o'):
            shutil.copyfile(obj, work / obj.name)
        (target / 'chunked_input.mlir').write_text(lowered, encoding='utf-8')
        compile_mlir_module(lowered, insts_path=target / 'insts.bin',
                            xclbin_path=target / 'design.xclbin', work_dir=work)
        # Retain all compiler outputs, including final ELFs; no newest-cache guessing.
        elfs = list(work.rglob('*.elf'))
        if len(elfs) != cores:
            raise RuntimeError(f'Expected {cores} core ELFs, got {len(elfs)}')
        emit('compile_end', variant=name, elf_count=len(elfs), directory=str(target))
        manifest['stages'][name] = {p.relative_to(directory).as_posix(): digest(p)
            for p in target.rglob('*') if p.is_file() and '_bootstrap' not in p.parts and p.suffix in ('.o', '.elf', '.xclbin', '.bin')}
        for p in work.rglob('conv2d_*.o'):
            dump = subprocess.check_output([str(peano / 'llvm-objdump.exe'), '-d', str(p)], text=True)
            p.with_suffix('.disassembly.txt').write_text(dump, encoding='utf-8')
            emit('disassembly', variant=name, file=p.name,
                 vmac=len(re.findall(r'\bvmac', dump)), srs=len(re.findall(r'\bsrs|\bvsrs|vst\.srs', dump)),
                 stack_references=len(re.findall(r'\bsp\b', dump)),
                 memory_stalls='requires trace', dma_overlap='requires trace')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def audit_bundles(directory):
    from kernels.memory_placement.probe import disassemble_function
    from ignite_xdna.runtime.splice import shim_patches
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    functions = {}
    for stage in ('raw16', 'fused16', 'raw1', 'fused1'):
        target = directory / stage
        symbol = 'conv2d_fused' if stage.startswith('fused') else 'conv2d_raw'
        cores = 16 if stage.endswith('16') else 1
        function_hashes = set()
        elfs = list((target / 'design.prj').rglob('*.elf'))
        if len(elfs) != cores:
            raise AssertionError('Final ELF coverage does not match the core count')
        for elf in elfs:
            dump, code = disassemble_function(elf, symbol)
            if dump is None:
                raise AssertionError(f'{elf.name} has no {symbol}')
            stack_lines = [line for line in dump.splitlines() if re.search(r'\bsp\b', line)]
            # Peano may preserve a callee-saved pointer register at function
            # entry/exit. Permit only scalar ABI saves outside ALL event brackets;
            # any stack access in the compute/epilogue or vector spill is rejected.
            marked = dump[dump.index('event\t#0x0'):dump.rindex('event\t#0x1')]
            if re.search(r'\bsp\b', marked) or any(
                    not re.search(r'paddb\s+\[sp\]|\b(?:st|lda)\s+p[0-7], \[sp,', line) for line in stack_lines):
                raise AssertionError(f'{symbol} spills within compute or spills vectors in {elf.name}')
            if 'vst.srs.s8.s32' not in dump or 'vmac' not in dump:
                raise AssertionError(f'{symbol} lacks vector MAC / INT8 SRS stores')
            if stage.startswith('fused') and ('vadd' not in dump or 'vmul' not in dump):
                raise AssertionError('Fused ELF lacks vector add / polynomial multiply')
            events = list(re.finditer(r'\bevent\s+#0x([01])\b', dump))
            count = 6 if manifest['cin'] > 32 else 2
            if [int(e[1]) for e in events[:count]] != [0, 1] * (count // 2):
                raise AssertionError('Unexpected static event order')
            epi = dump[events[count-2].end():events[count-1].start()]
            if len(re.findall(r'\brel\s+#0x31,', epi)) != 4:
                raise AssertionError('Output-ready must release core-side selector 49 four times')
            parts = re.split(r'\brel\s+#0x31,', epi)
            if any(len(re.findall(r'vst\.srs\.s8\.s32', part)) != 2 for part in parts[:4]):
                raise AssertionError('Each DMA credit must follow exactly two INT8 stores')
            if manifest['cin'] > 32:
                mac_region = dump[events[2].end():events[3].start()]
                if len(re.findall(r'\bvmac\b', mac_region)) != 16 or re.search(r'\bvmac\b', epi):
                    raise AssertionError('Expected MAC loop and peeled tail entirely inside its bracket')
                if set(re.findall(r'\bacq\s+#0x([0-9a-f]+),', dump)) != {'33'}:
                    raise AssertionError('Streaming input must acquire core-side selector 51')
                if set(re.findall(r'\brel\s+#0x([0-9a-f]+),', dump)) != {'31', '32'}:
                    raise AssertionError('Streaming release selectors must be 49 and 50')
            elf.with_suffix('.disassembly.txt').write_text(dump, encoding='utf-8')
            function_hashes.add(hashlib.sha256(code).hexdigest())
        if len(function_hashes) != 1:
            raise AssertionError(f'Core function bytes differ within {stage}')
        functions[stage] = next(iter(function_hashes))
        mlir = (target / 'design.prj/input_with_addresses.mlir').read_text(encoding='utf-8')
        inputs = [line for line in mlir.splitlines() if 'aie.buffer' in line and '_split' in line and 'mem_tile' not in line]
        activations = [line for line in mlir.splitlines() if 'aie.buffer' in line and '%activations_' in line]
        streamed = manifest['cin'] > 32
        expected_input_addresses = {1024, 32768} if streamed else {1024}
        actual_input_addresses = {int(re.search(r'address = (\d+)', line)[1]) for line in inputs}
        if len(inputs) != cores * (2 if streamed else 1) or actual_input_addresses != expected_input_addresses:
            raise AssertionError('Input packet placement changed from bank 0')
        if len(activations) != cores or not all('address = 16384 ' in line for line in activations):
            raise AssertionError('Activation placement changed from bank 1')
        outputs = [line for line in mlir.splitlines() if 'aie.buffer' in line and '_join' in line and 'mem_tile' not in line]
        if streamed and (len(outputs) != cores or not all('address = 49152 ' in line for line in outputs)):
            raise AssertionError('Streaming output placement changed from bank 3')
        wrappers = list((target / 'design.prj').glob('peano-linked_main_core_*.ll'))
        if len(wrappers) != cores:
            raise AssertionError('Missing core LLVM wrapper')
        for wrapper in wrappers:
            ir = wrapper.read_text(encoding='utf-8')
            acq = re.findall(r'call void @llvm.aie2.acquire\(i32 (\d+), i32 -1\)', ir)
            rel = re.findall(r'call void @llvm.aie2.release\(i32 (\d+), i32 1\)', ir)
            if acq != (['48'] if streamed else ['51', '48']) or rel != ([] if streamed else ['50']):
                raise AssertionError('Compiler wrapper lock namespace/ownership changed')
        transaction = (target / 'insts.bin').read_bytes()
        patches = shim_patches(transaction)
        expected_args = {0, 1, 2} if cores == 1 else {0, 1}
        if {p.argument for p in patches} != expected_args:
            raise AssertionError('Unexpected Shim BO argument mapping')
        # Reproducible C transaction header, suitable for a native XRT host.
        words = np.frombuffer(transaction, dtype='<u4')
        header = '#pragma once\n#include <stdint.h>\n'
        header += f'static const uint32_t {stage}_instructions[] = {{\n'
        header += '\n'.join('    ' + ', '.join(f'0x{int(w):08x}' for w in words[i:i+8]) + ','
                            for i in range(0, words.size, 8))
        header += f'\n}};\nstatic const unsigned {stage}_instruction_bytes = {len(transaction)};\n'
        (target / 'transaction.h').write_text(header, encoding='utf-8')
        emit('elf_audit', stage=stage, core_elves=cores, function_sha256=functions[stage],
             compute_stack_references=0, vector_spills=0, abi_stack_references=len(stack_lines),
             input_addresses=sorted(expected_input_addresses), activation_address=16384,
             output_chunk_bytes=64, output_ready_selector=49, output_credits=4,
             shim_arguments=sorted(expected_args), instruction_bytes=len(transaction),
             dma_overlap='requires port trace', scope='static instruction and ABI audit; stalls require trace')
    if functions['raw16'] != functions['raw1'] or functions['fused16'] != functions['fused1']:
        raise AssertionError('Single-core trace build changed the measured kernel bytes')
    manifest['function_sha256'] = functions
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def preflight():
    smi = subprocess.run([r'C:\Windows\System32\AMD\xrt-smi.exe', 'examine', '-r', 'aie-partitions'],
                         text=True, capture_output=True, timeout=20, check=True)
    print(smi.stdout, flush=True)
    if '[003d:00:01.1] : NPU Phoenix' not in smi.stdout or 'No hardware contexts running' not in smi.stdout:
        raise RuntimeError('Physical Phoenix device must be idle')
    host = subprocess.run(['powershell', '-NoProfile', '-File', str(ROOT / 'tools/host_load.ps1')],
                          text=True, capture_output=True, timeout=25, check=True)
    print(host.stdout, flush=True)
    if 'HOST_LOAD_VERDICT CLEAR' not in host.stdout:
        raise RuntimeError('Host is not clear for compilation / measurement')


def hardware(args):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from ignite_xdna.runtime.splice import shim_patches
    sys.path.insert(0, str(ROOT / 'kernels/clock_probe'))
    from clock_probe import TraceReader, decode_event_time
    from aie.utils.trace import TraceConfig
    directory = artifact_directory(args.build_dir, args.cin)
    audit_bundles(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['stages']) != {'raw16', 'fused16', 'raw1', 'fused1'}:
        raise RuntimeError('Incomplete build manifest')
    if manifest['source_sha256'] != digest(ROOT / 'kernels/aie2/fused_conv_epilogue/conv2d_fused.cc'):
        raise RuntimeError('Kernel source changed; rebuild')
    if manifest['design_sha256'] != digest(ROOT / 'kernels/aie2/fused_conv_epilogue/design.py'):
        raise RuntimeError('Transport changed; rebuild')
    if manifest['transport_sha256'] != digest(ROOT / 'kernels/aie2/fused_conv_epilogue/transport.py') or manifest['cin'] != args.cin:
        raise RuntimeError('Lowered transport or channel count changed; rebuild')
    emit('build_identity', **{k: v for k, v in manifest.items() if k != 'stages'},
         test_sha256=digest(__file__))
    for stage, paths in manifest['stages'].items():
        for path, expected in paths.items():
            if digest(directory / path) != expected:
                raise RuntimeError(f'Artifact hash mismatch: {path}')
        emit('artifact', stage=stage, xclbin_sha256=digest(directory / stage / 'design.xclbin'),
             transaction_sha256=digest(directory / stage / 'insts.bin'),
             shim_patches=len(shim_patches((directory / stage / 'insts.bin').read_bytes())))
    outdir = directory / ('run_' + str(time.time_ns()))
    outdir.mkdir(exist_ok=False)
    rows = []

    def run_stage(cores, fused, fixtures):
        stage = ('fused' if fused else 'raw') + str(cores)
        target = directory / stage
        harness = XrtSiliconHarness(device_idx=0)
        harness.load_xclbin(str(target / 'design.xclbin'))
        instr, instr_bytes = harness.create_instruction_bo(str(target / 'insts.bin'))
        inp = harness.create_host_bo(cores * input_bytes(args.cin), 3)
        out = harness.create_host_bo(cores * OUTPUT, 4)
        trace = harness.create_host_bo(65536, 5) if cores == 1 else None
        tc = TraceConfig(65536, str(outdir / (stage + '_trace.txt'))) if trace else None
        reader = TraceReader(tc) if tc else None
        px = harness.pyxrt
        todev, fromdev = px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, px.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        for index, (seed, case, measured) in enumerate(fixtures):
            packet, raw, exact, ref = fixture(seed, case, args.cin)
            expected = (exact if fused else raw)[:cores].ravel()
            golden = (ref if fused else raw)[:cores].ravel()
            host_input = packet[:cores]
            if args.cin > 32 and cores == 16:
                # Shim input is column/panel/row, not column/row/panel.
                host_input = host_input.reshape(4, 4, args.cin // 8, 3264).transpose(0, 2, 1, 3).copy()
            inp.write(host_input.tobytes(), 0)
            inp.sync(todev)
            # Complement of the expected output catches every unwritten byte.
            out.write(np.bitwise_xor(expected.view(np.uint8), np.uint8(255)).tobytes(), 0)
            out.sync(todev)
            bos = [inp, out]
            if trace:
                trace.write(bytes(65536), 0)
                trace.sync(todev)
                bos.append(trace)
            run, state = harness.dispatch_kernel(instr, instr_bytes, *bos, timeout_ms=5000)
            if str(state) != 'ert_cmd_state.ERT_CMD_STATE_COMPLETED':
                raise RuntimeError(f'{stage}: hardware completion failed: {state}')
            out.sync(fromdev)
            actual = np.frombuffer(out.read(cores * OUTPUT, 0), dtype=np.int8).copy()
            errors = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            float_errors = np.abs(actual.astype(np.int16) - golden.astype(np.int16))
            witness = outdir / f'{stage}_{seed}_{case}_{index}.npz'
            np.savez_compressed(witness, input=packet[:cores],
                                actual=actual, fixed=expected, float32=golden)
            emit('comparison', stage=stage, seed=seed, case=case, elements=actual.size,
                 fixed_mismatches=int(np.count_nonzero(errors)), fixed_max_error=int(errors.max()),
                 float_max_error_lsb=int(float_errors.max()),
                 input_sha256=hashlib.sha256(packet[:cores].tobytes()).hexdigest(),
                 wire_input_sha256=hashlib.sha256(host_input.tobytes()).hexdigest(),
                 witness_file=str(witness), witness_sha256=digest(witness),
                 output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                 per_core_mismatches=np.count_nonzero(errors.reshape(cores, OUTPUT), axis=1).tolist())
            if errors.any() or float_errors.max() > 1:
                emit('mismatch_detail', actual=actual[:64].tolist(), expected=expected[:64].tolist())
                raise AssertionError(f'{stage}: numerical parity failed')
            if trace:
                trace.sync(fromdev)
                trace_words = np.frombuffer(trace.read(65536, 0), dtype=np.uint32)
                tc.write_trace(trace_words)
                first, last, pairs, _ = reader.stamps()
                if first is None or last is None or last <= first or pairs < 200 or reader.tile != (2, 1):
                    raise AssertionError(f'Invalid/truncated trace: {first}, {last}, {pairs}, {reader.tile}')
                hits = decode_event_time(reader.core_stream())
                count = args.cin // 8 + 2 if args.cin > 32 else 1
                markers = [(slot, stamp) for slot, stamp in hits if slot in (0, 1)]
                if len(markers) < 2 * (count + 200) or any(slot != i % 2 for i, (slot, _) in enumerate(markers[:2*count])):
                    raise AssertionError('Missing or reordered compute brackets')
                intervals = [(markers[2*i][1], markers[2*i+1][1]) for i in range(count)]
                if any(end <= begin for begin, end in intervals):
                    raise AssertionError('Invalid cycle interval')
                stalls = sum(slot == 2 and any(begin <= stamp <= end for begin, end in intervals) for slot, stamp in hits)
                drain = sum(slot == 3 and intervals[-1][0] < stamp < intervals[-1][1] for slot, stamp in hits)
                row = {'stage': stage, 'cycles': sum(end-begin for begin, end in intervals),
                       'elapsed_cycles': intervals[-1][1] - intervals[0][0],
                       'bracket_cycles': [end-begin for begin, end in intervals], 'memory_stall_events': stalls,
                       'output_dma_active_events': drain, 'seed': seed, 'measured': measured}
                rows.append(row)
                emit('cycle_sample', **row)
                trace_witness = outdir / f'{stage}_{seed}_{index}.trace.txt'
                trace_witness.write_text(Path(tc.trace_file).read_text(), encoding='utf-8')
                emit('trace_witness', stage=stage, seed=seed, file=str(trace_witness),
                     sha256=digest(trace_witness), decoded_pairs=pairs, physical_tile=list(reader.tile))
            del run
        del bos, inp, out, trace, instr
        harness.kernel = None
        harness.context = None
        del harness
        gc.collect()

    correctness = [(seed, case, False) for seed in (17, 91)
                   for case in ('random', 'boundaries', 'channel_basis', 'cancellation')]
    for fused in (0, 1):
        run_stage(16, fused, correctness)
    # Sequential contexts, paired alternating A/B order, identical input seeds.
    for repeat in range(args.iters):
        for fused in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            # The AIE trace stream can retain a partial packet across dispatches;
            # restarting the transaction mid-packet corrupts the next capture.
            # Each sample therefore has its own fresh hardware context.
            run_stage(1, fused, [(101 + repeat, 'random', True)])
    raw_cycles = [r['cycles'] for r in rows if r['measured'] and r['stage'] == 'raw1']
    fused_cycles = [r['cycles'] for r in rows if r['measured'] and r['stage'] == 'fused1']
    overhead = 100 * (statistics.median(fused_cycles) / statistics.median(raw_cycles) - 1)
    report = {'correctness': 'PASS', 'cin': args.cin, 'cout': 32, 'array_cores_checked': 16, 'outputs_per_array_run': 4096,
              'raw_cycles': raw_cycles, 'fused_cycles': fused_cycles, 'overhead_percent': overhead,
              'overhead_target': 'PASS' if overhead < 8 else 'FAIL',
              'cycle_scope': 'one core, sum of accumulator initialization/MAC/epilogue brackets; input DMA waits, input and metadata staging, function entry/exit and trace flush excluded',
              'output_dma_overlap': all(r['output_dma_active_events'] > 0 for r in rows if r['stage'] == 'fused1'),
              'output_dma_active_events': [r['output_dma_active_events'] for r in rows],
              'memory_stall_events': [r['memory_stall_events'] for r in rows if r['measured']]}
    (outdir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    emit('hardware_result', **report)
    if overhead >= 8:
        raise AssertionError(f'Fused cycle overhead {overhead:.2f}% exceeds the <8% target')
    if not report['output_dma_overlap']:
        raise AssertionError('No output DMA activity during the epilogue')
    if any(r['memory_stall_events'] for r in rows if r['stage'] == 'fused1'):
        raise AssertionError('Fused compute brackets contain memory stall events')


def artifact_directory(base, cin):
    sources = [ROOT / 'kernels/aie2/fused_conv_epilogue' / name for name in ('conv2d_fused.cc', 'design.py', 'transport.py')]
    key = hashlib.sha256(str(cin).encode() + b''.join(path.read_bytes() for path in sources)).hexdigest()[:16]
    return base.resolve() / key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/fused_conv_epilogue')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--cin', type=int, choices=(32, 512), default=512,
                        help='Real Conv input channels; the performance target is shape-specific')
    parser.add_argument('--require-overhead', action='store_true', help='Compatibility flag; hardware always enforces the <8 percent target')
    args = parser.parse_args()
    if args.iters < 1:
        parser.error('--iters must be positive')
    offline(args.cin)
    if args.compile or args.hardware:
        preflight()
    if args.compile:
        compile_bundles(artifact_directory(args.build_dir, args.cin), args.cin)
    if args.compile or args.audit:
        audit_bundles(artifact_directory(args.build_dir, args.cin))
    if args.hardware:
        if args.compile:
            preflight()
        hardware(args)


if __name__ == '__main__':
    main()
