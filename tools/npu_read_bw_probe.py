#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Clean NPU read-only DRAM bandwidth: shim MM2S -> mem tile sink, no write-back stream.

SILICON 1.6 has three off-chip figures and says none of them is "a clean single-direction read
test": each shared the bus with a write stream. Here every enabled shim MM2S channel streams one
contiguous DDR region into its own mem-tile S2MM channel, whose single BD loops on itself with no
locks, so the data is overwritten in place and nothing flows back. Only the MM2S tasks exist in
the runtime sequence and each carries the completion token. An MM2S task completes once its last
word has entered the stream, and a sink that never stalls leaves at most a few words in flight,
so a completed dispatch has read all its bytes from DDR. Rate = bytes / median submit-to-wait
wall time of one dispatch.

    bash scripts/research-iron.sh tools/npu_read_bw_probe.py build     # compile only
    python tools/npu_read_bw_probe.py prereg                            # the pre-registration text
    bash scripts/research-iron.sh tools/npu_read_bw_probe.py suite     # the sitting (NPU)
    python tools/npu_read_bw_probe.py verdict <suite log>               # the mechanical verdict

Layouts are (columns, MM2S channels per column) over IRON's logical columns 0-3 (physical 1-4).
The LLM study's decode reopens only at >= 59.7 GB/s (results/llm/, commit 1cf1dc8).
"""
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

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "npu_read_bw_probe"
LAYOUTS = ((1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (3, 2), (4, 2))
SIZES_MIB = (256, 1024)            # target bytes per dispatch; each channel reads an equal share
SMOKE = ((1, 1), 32)               # first child, not deciding: the largest size SILICON 1.5's round trip drove
CHUNK = 32768                      # words in each mem-tile sink buffer (128 KiB)
WARMUP, REPS = 3, 15
WAIT_MS = 10000

# Pre-registered constants (DERIVED from MEASURED figures unless tagged)
STREAM_BPC = 3.999756              # SILICON 1.5: on-chip stream, bytes per cycle (MEASURED)
CLOCK_GHZ_REF = 1.796301           # SILICON 1.5: trace clock in the S1 sitting (MEASURED)
CHANNEL_CEIL = STREAM_BPC * CLOCK_GHZ_REF        # 7.185 GB/s per MM2S stream (DERIVED)
ALL8_CEIL = 8 * CHANNEL_CEIL                     # 57.48 GB/s: every drivable MM2S at 1 word/cycle
REOPEN_GBPS = 59.7                 # the LLM decode reopen line (DERIVED, llm_decode_verdict)
CAP_LO, CAP_HI = 26.0, 28.8        # SILICON 1.6's derived shared cap (26-28), and 16 B/cycle at 1.8 GHz
DEVICE_YAML_GBPS = 25.6            # device.yaml dram_read_bandwidth 16 x 1.6 GHz (unit not stated)
CAP_BINDS_MAX = CAP_HI * 1.05      # 30.24 GB/s: the cap binds reads if the best total is at or below this
FIXED_MS = (-1.0, 3.0)             # plausible fixed cost per dispatch from the 256 MiB / 1 GiB line
COLUMN_RATIO = 1.10                # R3: across-columns / within-a-column rate at the same channel count


def words_per_channel(mib: int, channels: int) -> int:
    return -(-(mib << 20) // 4 // channels // 1024) * 1024


def name(layout, mib):
    return f"c{layout[0]}x{layout[1]}_m{mib}"


def design(layout, n: int) -> str:
    cols, per = layout
    total = n * cols * per
    L = ["module { aie.device(npu1) {"]
    for c in range(cols):
        L += [f"%s{c} = aie.tile({c}, 0)", f"%m{c} = aie.tile({c}, 1)"]
        for ch in range(per):
            L += [f"aie.flow(%s{c}, DMA : {ch}, %m{c}, DMA : {ch})",
                  f"aie.shim_dma_allocation @in{c}_{ch}(%s{c}, MM2S, {ch})",
                  f"%b{c}_{ch} = aie.buffer(%m{c}) {{address = {ch * CHUNK * 4} : i32}} : memref<{CHUNK}xi32>"]
        L += [f"aie.memtile_dma(%m{c}) {{"]
        for ch in range(per):
            nxt = f"^s{ch + 1}" if ch + 1 < per else "^end"
            if ch:
                L += [f"^s{ch}:"]
            # the sink: one BD looping on itself, no locks, so the stream never back-pressures
            L += [f'aie.dma_start("S2MM", {ch}, ^k{ch}, {nxt})', f"^k{ch}:",
                  f"aie.dma_bd(%b{c}_{ch} : memref<{CHUNK}xi32> offset = 0 len = {CHUNK})", f"aie.next_bd ^k{ch}"]
        L += ["^end:", "aie.end", "}"]
    L += [f"aie.runtime_sequence(%x: memref<{total}xi32>) {{"]
    idx = 0
    for c in range(cols):
        for ch in range(per):
            L += [f"%t{c}_{ch} = aiex.dma_configure_task_for @in{c}_{ch} {{",
                  f"aie.dma_bd(%x : memref<{total}xi32> offset = {idx * n} len = {n})", "aie.end",
                  "} {issue_token = true}", f"aiex.dma_start_task(%t{c}_{ch})"]
            idx += 1
    for c in range(cols):
        for ch in range(per):
            L += [f"aiex.dma_await_task(%t{c}_{ch})", f"aiex.dma_free_task(%t{c}_{ch})"]
    L += ["}", "}}"]
    return "\n".join(L)


def artifacts():
    return [SMOKE] + [(layout, mib) for mib in SIZES_MIB for layout in LAYOUTS]


def build(only=None):
    from aie.utils.compile.utils import compile_mlir_module
    for layout, mib in artifacts():
        nm = name(layout, mib)
        if only and nm != only:
            continue
        d = BUILD / nm
        if (d / "insts.bin").exists():
            print("EXISTS", nm, flush=True)
            continue
        (d / "design.prj").mkdir(parents=True, exist_ok=True)
        module = design(layout, words_per_channel(mib, layout[0] * layout[1]))
        (d / "probe.mlir").write_text(module, encoding="utf-8")
        print("BUILD", nm, flush=True)
        compile_mlir_module(module, insts_path=d / "insts.bin", xclbin_path=d / "probe.xclbin",
                            work_dir=d / "design.prj")
    for layout, mib in artifacts():
        d = BUILD / name(layout, mib)
        if (d / "insts.bin").exists():
            print("ARTIFACT", d.name, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest(),
                  "mlir_sha256", hashlib.sha256((d / "probe.mlir").read_bytes()).hexdigest(), flush=True)


def run(layout, mib):
    import numpy as np
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / name(layout, mib)
    channels = layout[0] * layout[1]
    n = words_per_channel(mib, channels)
    total = n * channels
    print("ARTIFACT", d.name, "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest(), flush=True)
    with XrtSiliconHarness(0) as h:
        xrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        instr, count = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        inp = h.create_host_bo(total * 4, 3)
        # page every byte in once, outside timing; the content does not matter to a sink
        step = 1 << 26
        for off in range(0, total * 4, step):
            k = min(step, total * 4 - off)
            inp.write(np.full(k // 4, 0x5A5A5A5A, dtype=np.uint32), off)
        inp.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        task = xrt.run(h.kernel)
        for i, val in enumerate((3, instr, count, inp)):
            task.set_arg(i, val)
        times = []
        for it in range(WARMUP + REPS):
            t0 = time.perf_counter_ns()
            task.start()
            state = task.wait(WAIT_MS)
            dt = (time.perf_counter_ns() - t0) / 1e6
            if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"dispatch {it} did not complete: {state}")
            if it >= WARMUP:
                times.append(dt)
        del task, inp, instr
    gc.collect()
    med = statistics.median(times)
    rec = {"layout": list(layout), "columns": layout[0], "per_column": layout[1], "channels": channels,
           "target_mib": mib, "words_per_channel": n, "bytes": total * 4, "reps": REPS, "warmup": WARMUP,
           "times_ms": times, "median_ms": med, "min_ms": min(times), "max_ms": max(times),
           "gbps": total * 4 / med / 1e6, "gbps_best": total * 4 / min(times) / 1e6}
    print("RESULT_JSON", json.dumps(rec), flush=True)


def gpu_engines():
    ps = ("$s = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples "
          "| Where-Object CookedValue -gt 1; 'GPU_ENGINES_OVER_1PCT ' + @($s).Count; "
          "$s | ForEach-Object { 'GPU_ENGINE ' + $_.InstanceName + ' ' + [math]::Round($_.CookedValue, 1) }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    print(out.strip(), flush=True)


def child(args, script=__file__, timeout=300):
    sys.path.insert(0, str(ROOT / "tools"))
    from silicon_probe_record import witness
    print("PRE_CHILD", args, flush=True)
    witness()
    try:
        result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=timeout)
        print(result.stdout, result.stderr, flush=True)
    finally:
        print("POST_CHILD", args, flush=True)
        witness()
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def suite():
    host = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
                          capture_output=True, text=True)
    summary = next(s for s in host.stdout.splitlines() if s.startswith("HOST_LOAD busy_cores="))
    peers = [float(v) for v in re.findall(r"HOST_LOAD_PEER ([\d.]+)", host.stdout)]
    print(summary, "peer_busy_cores", peers, flush=True)
    assert float(re.search(r"busy_cores=([\d.]+)", summary)[1]) < 2, "host busy: the CPU would share DRAM"
    assert max(peers, default=0) < .5, "a heavy peer holds the CPU"
    gpu_engines()
    (ROOT / "build" / "silicon_stream_probe_large_buffers").mkdir(parents=True, exist_ok=True)
    out = child(["clock"], ROOT / "tools/silicon_stream_probe.py")
    ghz = json.loads(next(s.split(" ", 1)[1] for s in out.splitlines() if s.startswith("CLOCK_JSON ")))["ghz"]
    for layout, mib in artifacts():
        child(["run", "--layout", f"{layout[0]}x{layout[1]}", "--mib", str(mib)])
    gpu_engines()
    print("SITTING_CLOCK_GHZ", ghz, flush=True)


PREREG = f"""NPU read-only DRAM bandwidth (LLM study stage 1), pre-registered before any sitting

Question
  How fast can Phoenix's NPU read DDR when nothing is written back? SILICON 1.6 has no clean
  single-direction read test. It derives a shared cap of 26-28 GB/s per direction above ~4 channels
  (DRAM or NoC side, unattributed); device.yaml's unitless 16 reads as {DEVICE_YAML_GBPS} GB/s at 1.6 GHz and
  28.8 at 1.8. The LLM study's decode reopens only if the NPU reads >= {REOPEN_GBPS} GB/s.

Design (tools/npu_read_bw_probe.py)
  Every enabled shim MM2S channel streams one contiguous region of one host-only DDR buffer into
  its own mem-tile S2MM channel. The sink is one {CHUNK * 4 // 1024} KiB BD looping on itself with no locks.
  The runtime sequence holds only the MM2S tasks, each with a completion token. No stream leaves
  the array. Layouts (columns x channels per column), logical columns from 0:
  {', '.join(f'{c}x{p}' for c, p in LAYOUTS)}
  Sizes: {SIZES_MIB[0]} MiB and {SIZES_MIB[1]} MiB per dispatch (at least; split equally over the channels, rounded up
  to 1024 words each). Each (layout, size) runs in a fresh process: {WARMUP} warmup then {REPS} timed
  dispatches; rate = bytes / median submit-to-wait wall time. The 1 GiB rows decide.
  Smoke row, first and not deciding: {SMOKE[0][0]}x{SMOKE[0][1]} at {SMOKE[1]} MiB, the size SILICON 1.5's round trip drove at
  6.899 GB/s per direction. Two mechanisms are new here: SILICON 1.5's on-chip probe ran a
  lockless self-looping mem-tile BD on this npu1, but as an MM2S source, not an S2MM sink (the
  sink pattern's only mlir-aie examples target npu2); and the repo's designs await S2MM tokens,
  not MM2S ones. A hang from either shows here first. The verdict requires this row to complete.
  Fallback, pre-stated: if a 1 GiB host-only buffer fails to allocate (untested at that size
  here), the 1 GiB rows re-run on a 256 MiB buffer read 4 times per dispatch
  (repeat_count = 3; N means N+1 runs), with the same layouts, labelled as an amendment and
  compiled and committed before that sitting. The largest cache AMD lists for the 8700G is the
  CPU's 16 MiB L3 (SPEC), far below 256 MiB.
  The sitting: host-load gate (busy cores < 2, no peer >= 0.5), GPU engine snapshot, xrt-smi idle
  before and after every child, the same-sitting trace clock (tools/silicon_stream_probe.py clock),
  announced to the other sessions, nothing else reading DRAM.

Constants (DERIVED from MEASURED unless tagged)
  per-stream ceiling  {STREAM_BPC} B/cycle (SILICON 1.5) x {CLOCK_GHZ_REF} GHz = {CHANNEL_CEIL:.3f} GB/s
  all 8 drivable MM2S at that rate: {ALL8_CEIL:.2f} GB/s, BELOW the {REOPEN_GBPS} GB/s reopen line
  cap binds reads if the best 1 GiB total <= {CAP_BINDS_MAX:.2f} GB/s ({CAP_HI} x 1.05)

Rules (mechanical; the rate is the 1 GiB row's bytes / median wall time, fixed cost included)
  R1 reopen   decode reopens iff the best total read rate >= {REOPEN_GBPS} GB/s.
  R2 cap      BINDS if the best total <= {CAP_BINDS_MAX:.2f} GB/s: reads alone, with no write stream on
              the bus, stay within 5% of the 26-28.8 GB/s cap. REFUTED otherwise: the cap does
              not hold for reads alone. Either way this names no cause for the earlier figures.
  R3 where    at equal channel counts, compare 2x1 with 1x2 and 4x1 with 2x2. A ratio (across columns
              / within a column) >= {COLUMN_RATIO} names the per-column shim as a limiter; below {COLUMN_RATIO} at
              both, the limit is not per column.
  R4 scaling  the 1x1 rate in bytes per cycle at the sitting clock, against 1 word/cycle.
  Reported, not deciding: the slope rate and fixed cost per dispatch from the line through each
  layout's 256 MiB and 1 GiB medians.
  VOID        a fixed cost outside {FIXED_MS[0]} to {FIXED_MS[1]} ms (time not linear in bytes); a dispatch that does not
              complete; an xrt-smi witness that is not idle (the suite stops); a missing row.

Written predictions (stated expectations; the rules decide)
  Q1  1x1 reads 6.7-7.2 GB/s (S1 measured 6.899 per direction on one channel, in a round trip).
      If it undershoots, one named alternative to the stream limit is address translation over
      a 1 GiB mapping; a fixed cost that grows with size would show it, and a clean line
      through 256 MiB and 1 GiB argues against it.
  Q2  1x2 and 2x1 read 13-14.4 GB/s.
  Q3  2x2 reads 26-29 GB/s (memcpy moved 28.1 per direction on 2 columns x 2 channels), 4x1
      reads the same within {COLUMN_RATIO}x, and 4x2 reads less than {COLUMN_RATIO}x the better of them: R2 BINDS.
      The alternative, named because the evidence for
      the cap is thin: SILICON 1.6's only 8-channel figure is GroupNorm's 25.9 GB/s of reads, a
      compute kernel whose 3.2 GB/s per channel may be its own limit, and memcpy moved 56.2 GB/s
      read + write together. If that is so, 4x2 reads up to {ALL8_CEIL:.2f} GB/s and R2 is REFUTED.
  Q4  R1: decode does NOT reopen; {REOPEN_GBPS} GB/s exceeds even {ALL8_CEIL:.2f}, all 8 streams at 1 word/cycle.
      Reopening would need a shim MM2S stream faster than 1 word/cycle, which SILICON 1.5 measured
      it is not.

Unverified by design: reads through the core tiles (the sink is the mem tile); packet-switched
streams; the host-to-NPU coherence path (host-only BOs); power modes other than the default;
column 0, which the driver never exposes.
"""


def verdict(log: Path) -> int:
    text = log.read_text(encoding="utf-8")
    rows = [json.loads(s.split(" ", 1)[1]) for s in text.splitlines() if s.startswith("RESULT_JSON ")]
    ghz = float(re.search(r"^SITTING_CLOCK_GHZ ([\d.]+)", text, flags=re.M)[1])
    problems = []
    if "TIMEOUT" in text or "did not complete" in text:
        problems.append("a dispatch did not complete")
    by = {(tuple(r["layout"]), r["target_mib"]): r for r in rows}
    print(f"sitting clock {ghz:.4f} GHz; per-stream ceiling at it {STREAM_BPC * ghz:.3f} GB/s")
    print(f"{'layout':6s} {'ch':>3s} {'GiB':>6s} {'median ms':>10s} {'GB/s':>7s} {'per ch':>7s} {'B/cycle/ch':>10s} "
          f"{'256 MiB GB/s':>12s} {'slope GB/s':>10s} {'fixed ms':>8s}")
    smoke = by.get(SMOKE)
    if smoke is None:
        problems.append("the smoke row did not complete")
    else:
        print(f"smoke {SMOKE[0][0]}x{SMOKE[0][1]} {SMOKE[1]} MiB: {smoke['gbps']:.3f} GB/s read-only "
              f"(SILICON 1.5: 6.899 per direction in a round trip, slope method)")
    big = {}
    for layout in LAYOUTS:
        r, s = by.get((layout, SIZES_MIB[1])), by.get((layout, SIZES_MIB[0]))
        if r is None or s is None:
            problems.append(f"missing row {layout}")
            continue
        big[layout] = r
        per = r["gbps"] / r["channels"]
        # two-point line through the medians: the slope rate excludes the fixed cost per dispatch
        slope = (r["median_ms"] - s["median_ms"]) / (r["bytes"] - s["bytes"])
        fixed = s["median_ms"] - slope * s["bytes"]
        if not FIXED_MS[0] <= fixed <= FIXED_MS[1]:
            problems.append(f"{layout}: fixed cost {fixed:.2f} ms outside {FIXED_MS} ms, time is not linear in bytes")
        print(f"{layout[0]}x{layout[1]:<4d} {r['channels']:3d} {r['bytes'] / 2**30:6.3f} {r['median_ms']:10.2f} "
              f"{r['gbps']:7.2f} {per:7.2f} {per / ghz:10.3f} {s['gbps']:12.2f} {1e-6 / slope:10.2f} {fixed:8.2f}")
    for w in re.findall(r"^GPU_ENGINES_OVER_1PCT (\d+)", text, flags=re.M):
        if w != "0":
            print(f"witness: {w} GPU engine(s) over 1% at a snapshot")
    if problems:
        for p in problems:
            print("PROBLEM", p)
        print("VERDICT INCOMPLETE")
        return 2
    best = max(big.values(), key=lambda r: r["gbps"])
    bl = f"{best['layout'][0]}x{best['layout'][1]}"
    print(f"\nbest total {best['gbps']:.2f} GB/s at {bl}; device.yaml reading {DEVICE_YAML_GBPS}; SILICON 1.6 cap {CAP_LO}-{CAP_HI}")
    print("R1 " + (f"REOPEN: {best['gbps']:.2f} >= {REOPEN_GBPS} GB/s" if best["gbps"] >= REOPEN_GBPS else
                   f"NOT REOPENED: {best['gbps']:.2f} < {REOPEN_GBPS} GB/s; NPU-only decode stays killed"))
    print("R2 " + (f"BINDS: reads alone top out at {best['gbps']:.2f} <= {CAP_BINDS_MAX:.2f} GB/s, with no write stream on the bus"
                   if best["gbps"] <= CAP_BINDS_MAX else
                   f"REFUTED for reads alone: {best['gbps']:.2f} GB/s at {bl} > {CAP_BINDS_MAX:.2f}"))
    for across, within in (((2, 1), (1, 2)), ((4, 1), (2, 2))):
        ratio = big[across]["gbps"] / big[within]["gbps"]
        print(f"R3 {across[0]}x{across[1]} / {within[0]}x{within[1]} = {ratio:.3f} -> "
              + ("the per-column shim limits" if ratio >= COLUMN_RATIO else "not a per-column limit"))
    one = big[(1, 1)]
    print(f"R4 1x1 {one['gbps']:.3f} GB/s = {one['gbps'] / ghz:.3f} B/cycle at {ghz:.4f} GHz "
          f"({one['gbps'] / (STREAM_BPC * ghz):.1%} of 1 word/cycle)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "prereg", "run", "suite", "verdict"))
    ap.add_argument("log", nargs="?")
    ap.add_argument("--layout", default="1x1")
    ap.add_argument("--mib", type=int, default=SIZES_MIB[1])
    ap.add_argument("--only", help="build: one artifact name")
    a = ap.parse_args()
    if a.mode == "build":
        build(a.only)
    elif a.mode == "prereg":
        print(PREREG)
        print("Artifacts the sitting will load (compiled before it; the sitting prints each insts_sha256 again)")
        for layout, mib in artifacts():
            d = BUILD / name(layout, mib)
            if not (d / "insts.bin").exists():
                raise SystemExit(f"{d.name} is not built: bash scripts/research-iron.sh {Path(__file__).name} build")
            print("ARTIFACT", d.name, "words_per_channel", words_per_channel(mib, layout[0] * layout[1]),
                  "insts_sha256", hashlib.sha256((d / "insts.bin").read_bytes()).hexdigest(),
                  "mlir_sha256", hashlib.sha256((d / "probe.mlir").read_bytes()).hexdigest())
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        print("git HEAD", head, "(plus this log's own commit)")
        print("PREREG_JSON " + json.dumps({"layouts": LAYOUTS, "sizes_mib": SIZES_MIB, "smoke": SMOKE, "chunk_words": CHUNK,
                                          "warmup": WARMUP, "reps": REPS, "channel_ceil_gbps": CHANNEL_CEIL,
                                          "all8_ceil_gbps": ALL8_CEIL, "reopen_gbps": REOPEN_GBPS,
                                          "cap_binds_max_gbps": CAP_BINDS_MAX, "fixed_ms": FIXED_MS,
                                          "column_ratio": COLUMN_RATIO,
                                          "git_head": head}))
    elif a.mode == "run":
        c, p = (int(v) for v in a.layout.split("x"))
        run((c, p), a.mib)
    elif a.mode == "suite":
        suite()
    else:
        return verdict(Path(a.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
