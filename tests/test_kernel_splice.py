"""Native Phoenix BO-splice validation; run via scripts/kernel-splice.sh.

Default mode tests only transaction/ABI rejection. --hardware runs real silicon:
Conv2D(1x1, depthwise) -> existing bf16 GroupNorm(32) -> Conv2D(1x1, depthwise).
Bit-exact parity means the same AIE kernels executed serially with host copies.
An independent CPU mathematical oracle checks Conv exactly and GroupNorm within
bf16 tolerance. It is not a claim of bit-exact fp32 GroupNorm equivalence.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ignite_xdna.runtime.splice import KernelSplicer, KernelStage, TensorSpec, memory_bank, shim_patches


def txn(patches):
    body = b"".join(struct.pack("<12I", 0x81, 48, 0, 0, 0, 0, reg, 0, arg, 0, offset, 0)
                    for reg, arg, offset in patches)
    return struct.pack("<6B2xII", 0, 1, 3, 6, 5, 1, len(patches), 16 + len(body)) + body


class TestTransactionABI(unittest.TestCase):
    def test_context_slot_is_not_memory_bank(self):
        self.assertEqual(memory_bank(8192000), memory_bank(8257536))
        self.assertNotEqual(memory_bank(8192000), memory_bank(8192001))

    def test_relocations_and_offsets(self):
        patches = shim_patches(txn([(0x1D004, 0, 0), (0x601D024, 2, 2048)]))
        self.assertEqual([(p.register, p.argument, p.offset) for p in patches],
                         [(0x1D004, 0, 0), (0x601D024, 2, 2048)])

    def test_reject_corrupt_transaction(self):
        good = txn([(0x1D004, 0, 0)])
        for data in (good[:-4], good + b"\0" * 4, good[:2] + b"\4" + good[3:],
                     good[:8] + struct.pack("<I", 2) + good[12:],
                     good[:16] + struct.pack("<I", 99) + good[20:],
                     good[:20] + struct.pack("<I", 0) + good[24:]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                shim_patches(data)

    def test_reject_wrong_tile_or_abi(self):
        for reg, arg in ((0x11D004, 0), (0x1D000, 0), (0xA01D004, 0), (0x1D004, 5)):
            with self.subTest(reg=reg, arg=arg), self.assertRaises(ValueError):
                shim_patches(txn([(reg, arg, 0)]))

    def test_spec_rejects_invalid_storage(self):
        self.assertEqual(TensorSpec((1, 32, 16, 16), "bf16", "NCHW").nbytes, 16384)
        for shape, dtype, layout in (((1, 0), "bf16", "NCHW"), ((1,), "float64", "NCHW"),
                                      ((1,), "bf16", "")):
            with self.subTest(shape=shape, dtype=dtype, layout=layout), self.assertRaises(ValueError):
                TensorSpec(shape, dtype, layout)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def context_witness(*, idle=False):
    run = subprocess.run([r"C:\Windows\System32\AMD\xrt-smi.exe", "examine", "-r", "aie-partitions"],
                         capture_output=True, text=True, timeout=20, check=True)
    print(run.stdout, flush=True)
    if "[003d:00:01.1] : NPU Phoenix" not in run.stdout:
        raise RuntimeError("Expected physical Phoenix Device 0 [003d:00:01.1]")
    if idle and "No hardware contexts running" not in run.stdout:
        raise RuntimeError("NPU context preflight failed")
    if not idle:
        pids = set(re.findall(r"^\s*\|(\d+)\s*\|\d+\s*\|", run.stdout, re.MULTILINE))
        if pids != {str(os.getpid())}:
            raise RuntimeError(f"Expected only this process's contexts, got PIDs {sorted(pids)}")


def host_witness():
    run = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
                         capture_output=True, text=True, timeout=25, check=True)
    print(run.stdout, flush=True)
    if "HOST_LOAD_VERDICT CLEAR" not in run.stdout:
        raise RuntimeError("Host-load preflight failed")


def summary(values):
    return {"min_us": min(values), "median_us": statistics.median(values),
            "mean_us": statistics.mean(values), "max_us": max(values)}


def run_hardware(args):
    import numpy as np
    import pyxrt
    from ml_dtypes import bfloat16
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from ignite_xdna.runtime.session import InferenceSession
    from kernels.dispatch_floor.splice_fixture import compile_fixtures
    from kernels.groupnorm_bf16.groupnorm import pack_params, reference

    if args.L % args.chunk or args.chunk < 16 or args.chunk % 16 or args.L % 16:
        raise ValueError("L must be divisible by chunk and 16; chunk must be a positive multiple of 16")
    if args.iters < 1 or args.warmup < 0:
        raise ValueError("Need positive iters and nonnegative warmup")
    context_witness(idle=True)
    host_witness()
    emit("environment", machine="Desktop 2 / Ryzen 7 8700G / Phoenix", device_index=0,
         python=sys.version, pid=os.getpid(), L=args.L, chunk=args.chunk,
         iters=args.iters, warmup=args.warmup, scope="native graph BOs; system DDR remains")
    directory = ROOT / "build" / "kernel_splice" / f"L{args.L}_c{args.chunk}"
    if args.compile:
        compile_fixtures(directory, args.L, args.chunk)
    artifacts = {name: (directory / name / "design.xclbin", directory / name / "insts.bin")
                 for name in ("conv_up", "groupnorm", "conv_down")}
    for name, (xclbin, insts) in artifacts.items():
        patches = shim_patches(insts.read_bytes())
        emit("artifact", stage=name, xclbin_sha256=sha(xclbin.read_bytes()),
             transaction_sha256=sha(insts.read_bytes()), instruction_bytes=insts.stat().st_size,
             shim_patch_count=len(patches))
    if args.compile_only:
        emit("compile_pass", hardware_execution=False)
        return
    # Compilation can take minutes. Refresh both witnesses before owning contexts.
    context_witness(idle=True)
    host_witness()
    spec = TensorSpec((1, 32, 16, args.L // 16), "bf16", "NCHW")
    prm_spec = TensorSpec((8, args.chunk), "bf16", "GroupNorm32 packed fp32 parameters")
    with ExitStack() as stack:
        stages = []
        for name in ("conv_up", "groupnorm", "conv_down"):
            xclbin, insts = artifacts[name]
            if name.startswith("conv"):
                session = stack.enter_context(InferenceSession(
                    {"exec": str(insts)}, device_index=0, ring_depth=1,
                    xclbin_path=xclbin, num_cores=8, in_bytes=spec.nbytes, out_bytes=spec.nbytes))
                stage = KernelStage.from_session(name, session, spec, spec)
            else:
                harness = XrtSiliconHarness(0)
                harness.load_xclbin(str(xclbin))
                stage = KernelStage.from_files(name, harness, insts, {3: spec, 4: prm_spec, 5: spec},
                                               input_arg=3, output_arg=5)
            stages.append(stage)
            stack.callback(stage.close)
            emit("stage_buffers", stage=name,
                 group_ids={a: b.group_id for a, b in stage.buffers.items()})
        # Preserve distinct destination BOs for a real host-copy control.
        serial_inputs = [s.input for s in stages]
        serial_runs = []
        for stage in stages:
            serial_runs.append(stage._prepare())
        stack.callback(serial_runs.clear)
        splicer = KernelSplicer(stages)
        stack.callback(splicer.close)
        for record in splicer.bindings():
            emit("binding", **record)
        context_witness()

        # Exercise real BO ownership rejection without dispatch or fake hardware.
        with stages[0].input._access_lock:
            for operation in (lambda: splicer.run(), lambda: stages[0].input.read(),
                              lambda: stages[0].input.write(b"\0" * spec.nbytes)):
                try:
                    operation()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Active-buffer access was not rejected")
        try:
            KernelSplicer([stages[0]])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate stage ownership was not rejected")
        emit("ownership_guards", verdict="PASS", dispatches=0)

        rng = np.random.default_rng(args.seed)
        scale = rng.uniform(0.5, 1.5, 32).astype(np.float32)
        bias = rng.uniform(-0.5, 0.5, 32).astype(np.float32)
        params = pack_params(scale, bias, args.chunk).tobytes()
        stages[1].buffers[4].write(params)
        emit("parameters", sha256=sha(params))
        timings, host_timings, handoffs, stage_timings = [], [], [], [[], [], []]
        previous_output = None
        coverage = 0
        for iteration in range(args.warmup + args.iters):
            # Independent data in every group, different input every iteration.
            x = (rng.normal(size=(32, args.L)) + np.arange(32)[:, None] / 32).astype(bfloat16)
            payload = x.tobytes()
            serial_inputs[0].write(payload)

            def serial():
                outputs = []
                t0 = time.perf_counter_ns()
                for i, stage in enumerate(stages):
                    if i:
                        serial_inputs[i].write(outputs[-1])
                    run = serial_runs[i]
                    run.start()
                    KernelSplicer._wait(run, stage, 2000)
                    outputs.append(stage.output.read())
                return outputs, (time.perf_counter_ns() - t0) / 1000

            def direct():
                # Poison every output first. Every byte must be overwritten by DMA.
                for stage in stages:
                    stage.output.write(b"\xA5" * spec.nbytes)
                t0 = time.perf_counter_ns()
                result = splicer.run()
                final = result.output.read()
                wall = (time.perf_counter_ns() - t0) / 1000
                if (result.host_read_bytes, result.host_write_bytes, result.host_sync_calls) != (0, 0, 0):
                    raise AssertionError("Inter-stage host transfer detected")
                # Intermediate readbacks only AFTER the entire chain, outside timing.
                outputs = [stages[0].output.read(), stages[1].output.read(), final]
                return outputs, wall, result

            # Alternate order to avoid systematically favoring a warm context.
            if iteration % 2:
                actual, wall, result = direct()
                expected, host_wall = serial()
            else:
                expected, host_wall = serial()
                actual, wall, result = direct()
            for stage, a, e in zip(stages, actual, expected):
                if len(a) != spec.nbytes or a != e:
                    diff = np.count_nonzero(np.frombuffer(a, np.uint8) != np.frombuffer(e, np.uint8))
                    raise AssertionError(f"{stage.name}: serial parity failed in {diff} bytes")
                coverage += len(a)
            conv_ref = (x.astype(np.float32) * 0.5 + 0.25).astype(bfloat16)
            if actual[0] != conv_ref.tobytes():
                raise AssertionError("Upstream Conv2D CPU oracle failed")
            gn = np.frombuffer(actual[1], dtype=bfloat16).reshape(32, args.L)
            ref = reference(conv_ref, scale, bias, 1e-5)
            err = np.abs(gn.astype(np.float64) - ref)
            if not np.all(err <= 0.016 + 0.008 * np.abs(ref)):
                raise AssertionError(f"GroupNorm CPU oracle failed: max error={err.max()}")
            down_ref = (gn.astype(np.float32) * 2 + 1).astype(bfloat16)
            if actual[2] != down_ref.tobytes():
                raise AssertionError("Downstream Conv2D CPU oracle failed")
            if actual[-1] == previous_output:
                raise AssertionError("Stale pipeline output for changing input")
            previous_output = actual[-1]
            emit("parity", iteration=iteration, input_sha256=sha(payload),
                 output_sha256=[sha(v) for v in actual], bytes_per_stage=spec.nbytes,
                 all_stages_bit_exact=True, groupnorm_cpu_max_abs=float(err.max()),
                 interstage_host_read_bytes=result.host_read_bytes,
                 interstage_host_write_bytes=result.host_write_bytes,
                 interstage_host_sync_calls=result.host_sync_calls,
                 direct_wall_us=wall, serial_wall_us=host_wall,
                 handoff_us=result.handoff_us, stage_us=result.stage_us)
            if iteration >= args.warmup:
                timings.append(wall)
                host_timings.append(host_wall)
                handoffs.extend(result.handoff_us)
                for records, value in zip(stage_timings, result.stage_us):
                    records.append(value)
        context_witness()
        emit("summary", verdict="PASS", compared_bytes=coverage,
             direct_submit_through_final_read=summary(timings),
             serial_submit_through_final_read=summary(host_timings),
             completion_to_next_submit=summary(handoffs),
             stage_submit_wait=[summary(v) for v in stage_timings],
             interstage_host_read_bytes=0, interstage_host_write_bytes=0, interstage_host_sync_calls=0,
             traffic_scope="DeviceBuffer API, not physical DDR bus counters",
             parity_scope="all bytes at all stages vs serial host-copy AIE execution",
             cpu_oracle="Conv bit-exact; GroupNorm bf16 tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.hardware:
        run_hardware(args)
    else:
        unittest.main(argv=[sys.argv[0]])
