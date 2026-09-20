"""Double-buffered shim/MemTile passthrough and same-sitting trace-clock calibration."""
import argparse
import gc
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/silicon_stream_probe_large_buffers"
SIZES = (1 << 20, 2 << 20, 4 << 20, 8 << 20)  # words per channel
CHUNK = 65536


def design(n, channels):
    lines = ["module { aie.device(npu1) {"]
    for c in range(channels):
        lines += [f"%s{c} = aie.tile({c}, 0)", f"%m{c} = aie.tile({c}, 1)",
                  f"aie.flow(%s{c}, DMA : 0, %m{c}, DMA : 0)",
                  f"aie.flow(%m{c}, DMA : 0, %s{c}, DMA : 0)",
                  f"aie.shim_dma_allocation @in{c}(%s{c}, MM2S, 0)",
                  f"aie.shim_dma_allocation @out{c}(%s{c}, S2MM, 0)"]
        for j in range(2):
            lines += [f'%b{c}_{j} = aie.buffer(%m{c}) {{address = {j*CHUNK*4} : i32}} : memref<{CHUNK}xi32>',
                      f'%f{c}_{j} = aie.lock(%m{c}, {2*j}) {{init = 1 : i32}}',
                      f'%r{c}_{j} = aie.lock(%m{c}, {2*j+1}) {{init = 0 : i32}}']
        lines += [f"aie.memtile_dma(%m{c}) {{", "%one = arith.constant 1 : i32",
                  'aie.dma_start("S2MM", 0, ^in0, ^out)', "^out:",
                  'aie.dma_start("MM2S", 0, ^out0, ^end)']
        for direction, acquire, release in (("in", "f", "r"), ("out", "r", "f")):
            for j in range(2):
                lines += [f"^{direction}{j}:", f"aie.use_lock(%{acquire}{c}_{j}, AcquireGreaterEqual, %one)",
                          f"aie.dma_bd(%b{c}_{j} : memref<{CHUNK}xi32> offset = 0 len = {CHUNK})",
                          f"aie.use_lock(%{release}{c}_{j}, Release, %one)", f"aie.next_bd ^{direction}{1-j}"]
        lines += ["^end:", "aie.end", "}"]
    total = n * channels
    lines += [f"aie.runtime_sequence(%x: memref<{total}xi32>, %y: memref<{total}xi32>) {{"]
    for c in range(channels):
        for direction, arg, token in (("in", "x", ""), ("out", "y", " {issue_token = true}")):
            lines += [f"%{direction}{c} = aiex.dma_configure_task_for @{direction}{c} {{",
                      f"aie.dma_bd(%{arg} : memref<{total}xi32> offset = {c*n} len = {n})", "aie.end", "}" + token]
        lines += [f"aiex.dma_start_task(%out{c})", f"aiex.dma_start_task(%in{c})"]
    for c in range(channels):
        lines += [f"aiex.dma_await_task(%out{c})", f"aiex.dma_free_task(%in{c})", f"aiex.dma_free_task(%out{c})"]
    lines += ["}", "}}"]
    return "\n".join(lines)


def build():
    from aie.utils.compile.utils import compile_mlir_module
    for channels in (1, 2):
        for n in SIZES:
            d = BUILD / f"c{channels}_n{n}"
            (d / "design.prj").mkdir(parents=True, exist_ok=True)
            module = design(n, channels)
            (d / "probe.mlir").write_text(module, encoding="utf-8")
            print("BUILD", d.name, flush=True)
            compile_mlir_module(module, insts_path=d / "insts.bin", xclbin_path=d / "probe.xclbin", work_dir=d / "design.prj")


def run(n, channels):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / f"c{channels}_n{n}"
    print("ARTIFACT", d.name, "chunk_bytes", CHUNK * 4, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest())
    total = n * channels
    with XrtSiliconHarness(0) as h:
        xrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        instr, count = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        inp, out = h.create_host_bo(total * 4, 3), h.create_host_bo(total * 4, 4)
        task = xrt.run(h.kernel)
        for i, val in enumerate((3, instr, count, inp, out)):
            task.set_arg(i, val)
        times = []
        for iteration in range(18):
            # New sentinel/data every call proves execution, outside timing.
            data = np.arange(total, dtype=np.uint32) ^ np.uint32(0x12345678 + iteration * 7654321)
            inp.write(data, 0)
            out.write(np.bitwise_not(data), 0)
            inp.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            t0 = time.perf_counter_ns()
            task.start()
            state = task.wait(3000)
            dt = (time.perf_counter_ns() - t0) / 1e6
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch failed {state}")
            out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            actual = np.frombuffer(out.read(total * 4, 0), dtype=np.uint32)
            assert np.array_equal(actual, data), "readback mismatch"
            if iteration >= 3:
                times.append(dt)
        del task, inp, out, instr
    gc.collect()
    rec = {"words_per_channel": n, "channels": channels, "bytes_per_direction": total * 4,
           "times_ms": times, "median_ms": statistics.median(times), "mean_ms": statistics.fmean(times),
           "verified_calls": 18, "mismatches": 0}
    print("RESULT_JSON", json.dumps(rec), flush=True)


def clock():
    sys.path.insert(0, str(ROOT / "kernels/clock_probe"))
    import clock_probe as cp
    import aie.iron as iron
    from aie.utils.trace import TraceConfig
    tc = TraceConfig(trace_size=65536, trace_file=str(BUILD / "clock_trace.txt"))
    cp.clock_probe.trace_config = tc
    reader = cp.TraceReader(tc)
    a = iron.zeros(cp.N_IO, dtype=np.int32, device="npu")
    c = iron.zeros_like(a)
    clocks = []
    for mode in ("scalar", "vector"):
        xs, ys = [], []
        targets = (1 << 20, 1 << 22, 1 << 24) if mode == "scalar" else (1 << 22, 1 << 24, 1 << 26)
        for target in targets:
            samples, cycles = [], []
            for iteration in range(11):
                _, hw, cyc, decoded, reason = cp.one_call(a, c, cp.MODES[mode], target, reader)
                assert reason is None, reason
                if iteration >= 2:
                    samples.append(hw)
                    cycles.append(cyc)
            assert len(set(cycles)) == 1
            xs.append(cycles[0])
            ys.append(statistics.median(samples))
            print("CLOCK_POINT", json.dumps({"mode": mode, "iterations": target, "cycles": cycles[0],
                  "hardware_ms": samples, "tile": [decoded["col"], decoded["row"]]}), flush=True)
        intercept, slope, r2 = cp.fit_line(xs, ys)
        assert r2 > 0.999
        ghz = 1e-6 / slope
        clocks.append(ghz)
        print("CLOCK", mode, "GHz", ghz, "R2", r2, flush=True)
    assert abs(clocks[0] / clocks[1] - 1) < .01
    print("CLOCK_JSON", json.dumps({"ghz": statistics.fmean(clocks), "method": "trace event cycle/time slopes, scalar and vector"}), flush=True)


def child(args, script=__file__):
    from silicon_probe_record import witness
    print("PRE_CHILD", args, flush=True)
    witness()
    try:
        result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180)
        print(result.stdout, result.stderr, flush=True)
    finally:
        print("POST_CHILD", args, flush=True)
        witness()
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def suite():
    host = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")], capture_output=True, text=True)
    # The generic guard labels every node.exe a build peer, even when idle.
    # Gate on its measured CPU usage and preserve that evidence without local
    # application identities in a public research log.
    summary = next(s for s in host.stdout.splitlines() if s.startswith("HOST_LOAD busy_cores="))
    peers = [float(v) for v in re.findall(r"HOST_LOAD_PEER ([\d.]+)", host.stdout)]
    print(summary, "peer_busy_cores", peers, flush=True)
    assert float(re.search(r"busy_cores=([\d.]+)", summary)[1]) < 2
    assert max(peers, default=0) < .5
    child(["suite"], ROOT / "tools/silicon_onchip_stream_probe.py")
    output = child(["clock"])
    clock_rec = json.loads(next(s.split(" ", 1)[1] for s in output.splitlines() if s.startswith("CLOCK_JSON ")))
    ghz = clock_rec["ghz"]
    rows = []
    for n in SIZES:
        for channels in (1, 2):
            output = child(["run", "--words", str(n), "--channels", str(channels)])
            rows.append(json.loads(next(s.split(" ", 1)[1] for s in output.splitlines() if s.startswith("RESULT_JSON "))))
    for channels in (1, 2):
        subset = [r for r in rows if r["channels"] == channels]
        xs = np.array([r["bytes_per_direction"] for r in subset])
        ys = np.array([r["median_ms"] for r in subset])
        slope, intercept = np.polyfit(xs, ys, 1)
        predicted = intercept + slope * xs
        r2 = 1 - sum((ys-predicted)**2) / sum((ys-ys.mean())**2)
        gbps = 1e-6 / slope
        print("FIT_JSON", json.dumps({"channels": channels, "GBps_per_direction": gbps,
              "bytes_per_core_cycle_per_channel": gbps / ghz / channels,
              "intercept_ms": intercept, "R2": r2}), flush=True)
    print("Scope: host submit/wait slope for correct full-duplex transfers; bytes/cycle derived using same-sitting trace clock.")
    print("Shared DRAM vs NoC ceiling is not attributed by this one/two-channel probe.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build", "run", "clock", "suite"))
    ap.add_argument("--words", type=int, default=SIZES[0])
    ap.add_argument("--channels", type=int, choices=(1, 2), default=1)
    a = ap.parse_args()
    if a.mode == "build":
        build()
    elif a.mode == "run":
        run(a.words, a.channels)
    elif a.mode == "clock":
        clock()
    else:
        suite()


if __name__ == "__main__":
    main()
