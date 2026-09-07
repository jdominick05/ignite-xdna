#!/usr/bin/env python3
"""Measure the AIE core clock through the trace unit's timer -- docs/SILICON.md objective S0.

WHY THIS EXISTS
---------------
Every per-second ceiling this repo derives for the XDNA1 array (TOPS per column, bytes
per cycle on a stream, MACs per cycle per core) multiplies a per-cycle figure by a clock
that nothing on this machine has measured. RESEARCH.md cites 1.6 GHz from a web search;
results/aie/bottleneck_spatial_sweep_npu.log reasons at 1 GHz and says so. xrt-smi prints
no clock (results/aie/xrt_smi_platform_pmode.log). docs/SILICON.md section 1.7 carries the
clock as "unmeasured" and objective S0 is this script.

WHAT IT MEASURES
----------------
One Worker on one core tile runs kernels/clock_probe/clock_kernels.cc: event0(), a
DMA-free loop of `target` iterations, event1(). The tile's trace unit stamps both events
with its 64-bit timer and streams the packets to a host buffer (IRON's
Program.enable_trace); the parser turns the deltas into a cycle timestamp per event. The
host times the same call and fits, across a sweep of loop lengths,

    hw_ms = intercept + slope * (ts_event1 - ts_event0)      ->  f = 1 / slope

so the fixed per-dispatch cost (169.8 us hardware / 617.0 us wall,
results/aie/dispatch_floor_npu.log) lands in the intercept and drops out of the
frequency. The raw ratio cycles / hw_ms is printed per point too: a lower bound on f that
converges to the fit as the loop grows. A second loop (vector add instead of scalar add)
must fit to the same frequency; its different cycles-per-iteration is the check that the
stamps are counting core cycles and not something coarser.

Why not aie::tile::current().cycles(): Peano (llvm-aie 22) declares get_cycles() but never
defines it (ld.lld: undefined symbol), does not lower __builtin_readcyclecounter for
aie2, and rejects inline asm. The trace unit is the one path to the tile timer that the
open toolchain exposes -- which also makes this the first end-to-end hardware trace on
this machine (objective S2's first step).

Power mode is NOT switched here. `@iron.jit` keeps one hardware context alive for the
life of the process and a pmode change made under it may not take, so each power mode is
a fresh process: `xrt-smi configure --pmode X`, then this script with `--label X`. The
script captures `xrt-smi examine -r platform` and XRT's max_clock_frequency_mhz itself at
startup so the mode a run was made under is recorded in its own output.

CORRECTNESS GATE
----------------
Every call is verified before its timing is kept: the kernel must echo mode and target,
the checksum must equal the host's sum_{i<target} i mod 2^32 (x16 for the vector loop),
the reported tile row must be a core-tile row, and the trace must contain the real
event0/event1 pair (the first of each; the kernel emits filler pairs after it so the
packet reaches host memory) with event1 after event0. A call that fails is reported and
excluded from the fit rather than averaged in.

`--trace-size 0` runs the same sweep untraced: no cycles, so no clock, but the hardware
bracket is fitted against iterations instead. Its intercept against a traced run's at the
same points measures what trace configuration and the trace-buffer DMA add to submit+wait.

USAGE (from the mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/clock_probe/clock_probe.py
    python3 kernels/clock_probe/clock_probe.py --label performance --iters 20
    python3 kernels/clock_probe/clock_probe.py --idle-sleep 5 --idle-repeats 3
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent
from aie.utils.trace.parse import parse_trace
from aie.utils.trace.utils import convert_to_byte_stream, trace_pkts_de_interleave

N_IO = 64  # int32 elements each way (256 B); parameters in, echo/checksum out
CORE_ROWS = (2, 3, 4, 5)  # physical rows of the four core tiles in a column (row 0 shim, 1 memtile)
MODES = {"scalar": 0, "vector": 1}

_KERNEL_SRC = Path(__file__).resolve().parent / "clock_kernels.cc"
_XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"


@iron.jit
def clock_probe(a_in: In, c_out: Out, *, trace_size: CompileTime[int] = 0):
    """One Worker on Tile(0, 2); parameters in, echo out; event0/event1 traced."""
    io_ty = np.ndarray[(N_IO,), np.dtype[np.int32]]
    probe = ExternalFunction(
        "clock_probe",
        source_file=str(_KERNEL_SRC),
        object_file_name="clock_kernels.o",
        arg_types=[io_ty, io_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DN_OUT={N_IO}"],
    )
    of_in = ObjectFifo(io_ty, name="in")
    of_out = ObjectFifo(io_ty, name="out")

    def core_fn(of_in, of_out, fn):
        e = of_in.acquire(1)
        o = of_out.acquire(1)
        fn(e, o)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_in.cons(), of_out.prod(), probe],
        tile=Tile(0, 2),
        trace=1 if trace_size > 0 else None,
    )

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [io_ty, io_ty, of_in.prod(tile=Tile(0, 0)), of_out.cons(tile=Tile(0, 0))],
    )
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_size > 0:
        prog.enable_trace(
            trace_size=trace_size,
            workers=[worker],
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
        )
    return prog.resolve_program()


# --------------------------------------------------------------------------- helpers


def _npu_time_ns(res):
    for cand in (res, *(res if isinstance(res, tuple) else ())):
        t = getattr(cand, "npu_time", None)
        if isinstance(t, (int, float)):
            return t
    return None


def _u32(v):
    return int(v) & 0xFFFFFFFF


def decode(out):
    return dict(
        mode=int(out[0]),
        target=_u32(out[1]),
        row=int(out[2]),
        col=int(out[3]),
        check=_u32(out[4]),
    )


def expected_check(mode, target):
    s = (target * (target - 1) // 2) & 0xFFFFFFFF
    return s if mode == MODES["scalar"] else (16 * s) & 0xFFFFFFFF


SYNC_CYCLES = 1 << 18  # one event-sync frame: the 18-bit delta counter of a Single2 frame wrapped once


def decode_event_time(bs):
    """Decode a core tile's Event-Time trace byte stream into [(event_slot, timestamp)].

    Frame formats follow mlir-aie's aie.utils.trace.utils.convert_to_commands (Start,
    Single0/1/2, Multiple0/1/2, Repeat0/1, Event_Sync, fillers) with ONE difference that
    matters here: an Event_Sync frame (0xff) advances the timer by 2^18 cycles, and a
    Repeat frame repeats whatever frame preceded it -- a Sync as well as an event frame.
    mlir-aie v1.4.2's parse.py treats 0xff as a no-op for the timer and a Repeat as a
    re-issue of the last *event*, so any gap over 2^18 cycles between two events comes out
    wrong there (measured on this machine: a 2,097,172-cycle gap read as 45,017). The
    hardware encodes that gap as 0xff, Repeat0(7), i.e. 8 x 2^18 + the remainder.
    Timestamps use the parser's "+1 per event frame" convention so the two agree exactly
    whenever no sync frame is present.
    """
    t = 0
    cursor = 0
    out = []
    prev = None  # ("sync",) or ("event", event_bits, cycles)
    n = len(bs)

    def emit(event_bits, cycles):
        nonlocal t
        t += cycles + 1
        for i in range(8):
            if (event_bits >> i) & 1:
                out.append((i, t))

    while cursor < n:
        b = bs[cursor]
        if (b & 0xFB) == 0xF0:  # Start: 7-byte absolute timer follows
            cursor += 8
            prev = None
            continue
        if (b & 0xFC) == 0xDC:  # 4-byte frame the upstream parser skips
            cursor += 4
            continue
        if b == 0xFE:  # filler
            cursor += 1
            continue
        if b == 0xFF:  # Event_Sync: the delta counter wrapped
            t += SYNC_CYCLES
            prev = ("sync",)
            cursor += 1
            continue
        if (b & 0x80) == 0:  # Single0
            ev, cyc, w = 1 << ((b >> 4) & 7), b & 0xF, 1
        elif (b & 0xE0) == 0x80:  # Single1
            ev, cyc, w = 1 << ((b >> 2) & 7), ((b & 3) << 8) | bs[cursor + 1], 2
        elif (b & 0xE0) == 0xA0:  # Single2
            ev, cyc, w = 1 << ((b >> 2) & 7), ((b & 3) << 16) | (bs[cursor + 1] << 8) | bs[cursor + 2], 3
        elif (b & 0xF0) == 0xC0:  # Multiple0
            ev, cyc, w = ((b & 0xF) << 4) | (bs[cursor + 1] >> 4), bs[cursor + 1] & 0xF, 2
        elif (b & 0xFC) == 0xD0:  # Multiple1
            ev = ((b & 3) << 6) | (bs[cursor + 1] >> 2)
            cyc, w = ((bs[cursor + 1] & 3) << 8) | bs[cursor + 2], 3
        elif (b & 0xFC) == 0xD4:  # Multiple2
            ev = ((b & 3) << 6) | (bs[cursor + 1] >> 2)
            cyc, w = ((bs[cursor + 1] & 3) << 16) | (bs[cursor + 2] << 8) | bs[cursor + 3], 4
        elif (b & 0xF0) == 0xE0 or (b & 0xFC) == 0xD8:  # Repeat0 / Repeat1
            if (b & 0xF0) == 0xE0:
                r, w = b & 0xF, 1
            else:
                r, w = ((b & 3) << 8) | bs[cursor + 1], 2
            if prev is None:
                raise ValueError(f"Repeat frame at byte {cursor} with nothing to repeat")
            if prev[0] == "sync":
                t += r * SYNC_CYCLES
            else:
                for _ in range(r):
                    emit(prev[1], prev[2])
            cursor += w
            continue
        else:
            raise ValueError(f"unknown trace frame byte 0x{b:02x} at {cursor}")
        emit(ev, cyc)
        prev = ("event", ev, cyc)
        cursor += w
    return out


class TraceReader:
    """Read the raw trace IRON wrote after a call and pull out the event0/event1 stamps."""

    def __init__(self, trace_config):
        self.tc = trace_config
        self.mlir_str = None
        self.mlir_path = None
        self.tile = None  # (row, col) from the trace packet header

    def _find_mlir(self):
        """Lowered MLIR with the trace-register writes, for the upstream-parser cross-check."""
        if self.mlir_str is not None:
            return self.mlir_str is not False
        candidates = []
        if self.tc.physical_mlir_path:
            candidates.append(Path(self.tc.physical_mlir_path))
        cache = Path.home() / ".npu" / "cache"
        dirs = sorted(
            (d for d in cache.glob("*") if (d / "clock_kernels.o").exists()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for d in dirs[:3]:
            for name in ("input_with_addresses.mlir", "npu_lowered.mlir", "npu_dma_lowered.mlir"):
                candidates.append(d / name)
        for p in candidates:
            if p.exists():
                s = p.read_text()
                if "340D0" in s or "340d0" in s or "213200" in s:
                    self.mlir_str, self.mlir_path = s, p
                    return True
        self.mlir_str = False
        return False

    def core_stream(self):
        """Byte stream of the (single) core tile's trace packets, plus its header location."""
        buf = self.tc.read_trace()
        words = [f"{int(w):08x}" for w in buf]
        end = len(words)
        while end > 0 and int(words[end - 1], 16) == 0:
            end -= 1
        sorted_pkts = trace_pkts_de_interleave(words[:end])
        streams = convert_to_byte_stream(sorted_pkts)
        core = streams[0]  # packet type 0 = core tile
        if not core:
            return None
        loc, bs = next(iter(core.items()))
        row, col = (int(x) for x in loc.split(","))
        self.tile = (row, col)
        return bs

    def stamps(self):
        """Return (ts_event0, ts_event1, n_pairs, n_events) from the last trace file."""
        bs = self.core_stream()
        if bs is None:
            return None, None, 0, 0
        events = decode_event_time(bs)
        e0 = [t for slot, t in events if slot == 0]
        e1 = [t for slot, t in events if slot == 1]
        if not e0 or not e1:
            return None, None, 0, len(events)
        return e0[0], e1[0], min(len(e0), len(e1)), len(events)

    def upstream_stamps(self):
        """The same pair through mlir-aie's own parser, for the no-sync cross-check."""
        if not self._find_mlir():
            return None
        try:
            events = parse_trace(self.tc.read_trace(), self.mlir_str)
        except Exception:  # noqa: BLE001
            return None
        e0 = [x["ts"] for x in events if x.get("name") == "INSTR_EVENT_0" and x.get("ph") == "B"]
        e1 = [x["ts"] for x in events if x.get("name") == "INSTR_EVENT_1" and x.get("ph") == "B"]
        if not e0 or not e1:
            return None
        return e1[0] - e0[0]


def verify(d, mode, target, cycles, traced):
    if d["mode"] != mode or d["target"] != target:
        return f"echo mismatch (mode {d['mode']}, target {d['target']})"
    if d["row"] not in CORE_ROWS:
        return f"row {d['row']} is not a core-tile row"
    if d["check"] != expected_check(mode, target):
        return f"checksum {d['check']} != {expected_check(mode, target)}"
    if traced:
        if cycles is None:
            return "trace has no event0/event1 pair"
        if cycles <= 0:
            return f"event1 not after event0 ({cycles})"
    return None


def one_call(a, c, mode, target, reader):
    """Set parameters, dispatch, return (wall_ms, hw_ms, cycles, decoded, verify_reason)."""
    a[0] = mode
    a[1] = target
    t_start = time.perf_counter()
    res = clock_probe(a, c)
    wall = (time.perf_counter() - t_start) * 1e3
    npu_ns = _npu_time_ns(res)
    hw = npu_ns / 1e6 if npu_ns is not None else float("nan")
    d = decode(c.numpy())
    cycles = None
    if reader is not None:
        t0, t1, pairs, n_ev = reader.stamps()
        if t0 is not None:
            cycles = t1 - t0
        d["pairs"], d["n_ev"] = pairs, n_ev
    return wall, hw, cycles, d, verify(d, mode, target, cycles, reader is not None)


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return intercept, slope, r2


def platform_report():
    """Verbatim xrt-smi platform report plus XRT's own clock field, best effort."""
    lines = []
    try:
        out = subprocess.run(
            [_XRT_SMI, "--batch", "examine", "-r", "platform"],
            capture_output=True, text=True, timeout=30,
        )
        lines.append(((out.stdout or "") + (out.stderr or "")).rstrip())
    except Exception as e:  # noqa: BLE001 - reporting only
        lines.append(f"(xrt-smi not captured: {e})")
    try:
        import pyxrt

        dev = pyxrt.device(0)
        mhz = dev.get_info(pyxrt.xrt_info_device.max_clock_frequency_mhz)
        lines.append(f"pyxrt device.get_info(max_clock_frequency_mhz) = {mhz}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"(pyxrt max_clock_frequency_mhz not captured: {e})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- sweeps


def sweep(a, c, mode_name, targets, iters, warmup, reader):
    """Time one loop over `targets`; x for the fit is stamped cycles (traced) or iterations (not)."""
    mode = MODES[mode_name]
    traced = reader is not None
    print(f"\n=== loop: {mode_name}{'' if traced else ' (no trace: x = iterations)'} ===")
    print(
        f"{'iters':>10} {'cycles':>12} {'cyc/iter':>9} {'hw ms':>10} {'hw min':>10} "
        f"{'wall ms':>10} {'cyc/hw GHz':>10} {'first GHz':>10}  tile  ok"
    )
    print("-" * 112)
    xs, ys, wxs, wys, failures = [], [], [], [], []
    for target in targets:
        try:
            first = float("nan")
            for k in range(warmup):
                w, h, cyc, d, why = one_call(a, c, mode, target, reader)
                if k == 0 and cyc and h == h and h > 0:
                    first = cyc / (h * 1e6)
            walls, hws, cycs, reasons = [], [], [], []
            tile = None
            for _ in range(iters):
                w, h, cyc, d, why = one_call(a, c, mode, target, reader)
                if why is not None:
                    reasons.append(why)
                    continue
                walls.append(w)
                hws.append(h)
                cycs.append(cyc)
                tile = (d["row"], d["col"])
        except Exception as e:  # noqa: BLE001 - the driver may reject a long run
            print(f"{target:>10}  !! call raised {type(e).__name__}: {e}")
            failures.append((target, f"{type(e).__name__}: {e}"))
            break
        if reasons or not walls:
            print(f"{target:>10}  !! {len(reasons)}/{iters} calls failed verification: {reasons[0] if reasons else 'none ran'}")
            failures.append((target, reasons[0] if reasons else "no verified calls"))
            continue
        h_mean, h_min, w_mean = statistics.mean(hws), min(hws), statistics.mean(walls)
        if traced:
            cyc_mean = statistics.mean(cycs)
            ratio = cyc_mean / (h_mean * 1e6)  # cycles per ns == GHz
            print(
                f"{target:>10} {cyc_mean:>12.0f} {cyc_mean / target:>9.3f} {h_mean:>10.4f} {h_min:>10.4f} "
                f"{w_mean:>10.4f} {ratio:>10.4f} {first:>10.4f}  r{tile[0]}c{tile[1]}  yes"
            )
            x = cyc_mean
        else:
            print(
                f"{target:>10} {'n/a':>12} {'n/a':>9} {h_mean:>10.4f} {h_min:>10.4f} "
                f"{w_mean:>10.4f} {'n/a':>10} {'n/a':>10}  r{tile[0]}c{tile[1]}  yes"
            )
            x = target
        xs.append(x)
        ys.append(h_mean)
        wxs.append(x)
        wys.append(w_mean)
    return xs, ys, wxs, wys, failures


def report_fit(label, xs, ys, traced):
    """Fit time against cycles (traced: slope gives the clock) or iterations (not: ns per iteration)."""
    icept, slope, r2 = fit_line(xs, ys)
    unit = "cycles" if traced else "iterations"
    print(f"\n{label}:  time_ms = intercept + slope * {unit}   (R^2 = {r2:.6f})")
    print(f"  intercept (per-dispatch cost, cancels)  : {icept * 1e3:.1f} us")
    print(f"  slope                                   : {slope * 1e6:.6f} ns/{unit[:-1] if traced else 'iteration'}")
    ghz = float("nan")
    if traced:
        ghz = 1e-6 / slope if slope > 0 else float("nan")
        print(f"  CORE CLOCK  1/slope                     : {ghz:.4f} GHz")
    return ghz, icept * 1e3


def idle_probe(a, c, target, sleep_s, repeats, burst, reader):
    """Is the clock lower right after idling? Sleep, one call, repeat; then a burst."""
    mode = MODES["scalar"]
    print(f"\n=== idle probe: sleep {sleep_s}s -> 1 call, x{repeats}; then {burst} back-to-back ===")
    print(f"{'call':>6} {'cycles':>12} {'hw ms':>10} {'cyc/hw GHz':>10}")
    for i in range(repeats):
        time.sleep(sleep_s)
        w, h, cyc, d, why = one_call(a, c, mode, target, reader)
        tag = "" if why is None else f"  !! {why}"
        print(f"{'idle' + str(i):>6} {cyc or 0:>12d} {h:>10.4f} {(cyc or 0) / (h * 1e6):>10.4f}{tag}")
    for i in range(burst):
        w, h, cyc, d, why = one_call(a, c, mode, target, reader)
        tag = "" if why is None else f"  !! {why}"
        print(f"{'b' + str(i):>6} {cyc or 0:>12d} {h:>10.4f} {(cyc or 0) / (h * 1e6):>10.4f}{tag}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", default="default", help="free text, e.g. the pmode this run was made under")
    p.add_argument("--iters", type=int, default=10, help="timed calls per point")
    p.add_argument("--warmup", type=int, default=2, help="untimed calls per point")
    p.add_argument(
        "--scalar-targets",
        default="262144,524288,1048576,2097152,4194304,8388608,16777216,33554432",
        help="comma-separated iteration counts for the scalar loop (2^18..2^25)",
    )
    p.add_argument(
        "--vector-targets",
        default="262144,524288,1048576,2097152,4194304,8388608,16777216,33554432",
        help="comma-separated iteration counts for the vector loop (2^18..2^25)",
    )
    p.add_argument("--loops", default="scalar,vector", help="subset of scalar,vector")
    p.add_argument("--trace-size", type=int, default=65536, help="trace buffer bytes (0 = no trace, hw time only)")
    p.add_argument("--trace-file", default=None, help="where the raw trace is written (default: temp file)")
    p.add_argument("--idle-sleep", type=float, default=0.0, help="seconds to idle before a probe call (0 = skip)")
    p.add_argument("--idle-repeats", type=int, default=3)
    p.add_argument("--idle-burst", type=int, default=10)
    p.add_argument("--idle-target", type=int, default=8388608, help="scalar iterations for the idle probe")
    args = p.parse_args()

    loops = [m.strip() for m in args.loops.split(",") if m.strip()]
    for m in loops:
        if m not in MODES:
            sys.exit(f"unknown loop {m}")

    print(f"AIE core clock via trace-unit event stamps -- one Worker, scalar / vector loops, label={args.label}")
    print(f"iters={args.iters} warmup={args.warmup} trace_size={args.trace_size} (compile excluded, kernel cached)")
    print("\n--- xrt-smi --batch examine -r platform, and XRT's clock field (captured by this run) ---")
    print(platform_report())
    print("--- end ---")

    reader = None
    if args.trace_size > 0:
        trace_file = args.trace_file or os.path.join(tempfile.gettempdir(), f"clock_probe_trace_{os.getpid()}.txt")
        tc = TraceConfig(trace_size=args.trace_size, trace_file=trace_file)
        clock_probe.trace_config = tc
        reader = TraceReader(tc)

    a = iron.zeros(N_IO, dtype=np.int32, device="npu")
    c = iron.zeros_like(a)

    t_compile = time.perf_counter()
    w, h, cyc, d, why = one_call(a, c, MODES["scalar"], 1 << 13, reader)
    print(
        f"\nfirst call (includes compile/cache lookup): {time.perf_counter() - t_compile:.2f} s wall; "
        f"tile r{d['row']}c{d['col']} (get_coreid), cycles={cyc}, hw {h:.4f} ms, "
        f"trace events={d.get('n_ev')}, pairs={d.get('pairs')}, verify={'ok' if why is None else why}"
    )
    if reader is not None:
        print(f"trace packet header says the traced core is row {reader.tile[0]}, col {reader.tile[1]}")
        # Cross-check the in-script frame decoder against mlir-aie's parser on a run
        # short enough (< 2^18 cycles) to contain no sync frame: they must agree exactly.
        up = reader.upstream_stamps()
        print(
            f"decoder cross-check on a no-sync run: this script {cyc} cycles, "
            f"mlir-aie parse.py {up} cycles ({'agree' if up == cyc else 'DISAGREE'}; "
            f"parsed against {reader.mlir_path})"
        )

    results = {}
    all_failures = []
    for m in loops:
        targets = [int(t) for t in (args.scalar_targets if m == "scalar" else args.vector_targets).split(",")]
        xs, ys, wxs, wys, failures = sweep(a, c, m, targets, args.iters, args.warmup, reader)
        all_failures += [(m, *f) for f in failures]
        traced = reader is not None
        if len(xs) >= 3:
            ghz_hw, icept_hw = report_fit(f"{m.upper()} / HARDWARE bracket (submit+wait)", xs, ys, traced)
            if not traced:
                ghz_wall, icept_wall = report_fit(
                    f"{m.upper()} / WALL (perf_counter around @iron.jit call)", wxs, wys, traced
                )
            else:
                ghz_wall, icept_wall = float("nan"), float("nan")
                print(
                    f"\n{m.upper()} / WALL: not fitted. With trace on, every call allocates a "
                    f"{args.trace_size}-byte trace buffer and dumps it to disk inside the wall "
                    f"bracket (tens of ms); only the hardware bracket is meaningful here."
                )
            results[m] = (ghz_hw, ghz_wall, icept_hw, icept_wall)
        else:
            print(f"\n{m}: not enough verified points to fit ({len(xs)})")

    if all_failures:
        print("\n!! points excluded from the fits:")
        for m, t, why in all_failures:
            print(f"   {m} iters {t}: {why}")

    if args.idle_sleep > 0 and reader is not None:
        idle_probe(a, c, args.idle_target, args.idle_sleep, args.idle_repeats, args.idle_burst, reader)

    print(f"\nSUMMARY label={args.label} trace={'on' if reader is not None else 'off'}")
    for m, (ghz_hw, ghz_wall, icept_hw, icept_wall) in results.items():
        if reader is not None:
            print(f"  {m:>6}: core clock {ghz_hw:.4f} GHz (hw bracket); hw intercept {icept_hw:.1f} us")
        else:
            print(f"  {m:>6}: no clock (untraced); hw intercept {icept_hw:.1f} us, wall intercept {icept_wall:.1f} us")
    if reader is not None and "scalar" in results and "vector" in results:
        s, v = results["scalar"][0], results["vector"][0]
        print(f"  scalar vs vector agreement: {abs(s - v) / s * 100:.2f}%")


if __name__ == "__main__":
    main()
