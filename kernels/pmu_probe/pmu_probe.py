#!/usr/bin/env python3
"""Where a core tile's cycles go: stall taxonomy and instruction mix from the trace unit.

WHY THIS EXISTS
---------------
results/aie/aie2_isa_static.log established that a hardware loop's bundle count is its
cycle count, so the cost of an inner loop is readable off the disassembly. That holds
only while the core is issuing. A loop that waits -- on a lock, a stream, the memory
system or a cascade -- takes longer than its bundle count, and nothing static can say by
how much. The AIE2 trace unit exposes exactly that difference: alongside the two
instruction events S0 used for the clock, it can route a stall taxonomy
(MEMORY_STALL, STREAM_STALL, LOCK_STALL, CASCADE_STALL), an occupancy signal (ACTIVE,
DISABLED) and an instruction mix (INSTR_VECTOR, INSTR_LOAD, INSTR_STORE, ...), up to
eight events at a time per tile.

This script is the harness for that. It reuses kernels/clock_probe/'s kernel and design
unchanged, so the loops it runs are the two whose cycle counts are already known on this
machine -- 9.000 and 2.000 cycles per iteration -- and swaps only the event list. That
makes the first run a calibration rather than a measurement: if the decoded occupancy
does not reproduce 9 and 2, the instrument is wrong and nothing built on it would be
trustworthy.

WHAT IS UNKNOWN AND WHY --raw EXISTS
------------------------------------
S0 only ever traced two point events, INSTR_EVENT_0 and INSTR_EVENT_1, which fire once
each. A level event such as ACTIVE is true for many consecutive cycles, and how the trace
unit encodes that -- one frame per cycle, a frame per transition, or a Repeat frame
carrying a run length -- is not documented anywhere this project has found and is not
inferable from the S0 decoder. `--raw` prints the decoded frame stream so the encoding
can be read off the hardware rather than assumed. Everything else here depends on that
answer, so run --raw first on any event you have not traced before.

USAGE (from the mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/pmu_probe/pmu_probe.py --raw --target 4096
    python3 kernels/pmu_probe/pmu_probe.py --events ACTIVE,MEMORY_STALL,LOCK_STALL
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent
from aie.utils.trace.utils import convert_to_byte_stream, trace_pkts_de_interleave

# Reuse the clock probe's kernel and its verified frame decoder rather than forking them.
_CLOCK_DIR = Path(__file__).resolve().parent.parent / "clock_probe"
sys.path.insert(0, str(_CLOCK_DIR))
import clock_probe as cp  # noqa: E402

N_IO = cp.N_IO
CORE_ROWS = cp.CORE_ROWS
MODES = cp.MODES
_KERNEL_SRC = _CLOCK_DIR / "clock_kernels.cc"

# The eight-event budget per tile, defaulting to the calibration set: the two instruction
# events S0 already understands, occupancy, and the four stall categories.
DEFAULT_EVENTS = [
    "INSTR_EVENT_0",
    "INSTR_EVENT_1",
    "ACTIVE",
    "MEMORY_STALL",
    "STREAM_STALL",
    "LOCK_STALL",
    "CASCADE_STALL",
    "DISABLED",
]


def resolve_events(names):
    out = []
    for n in names:
        n = n.strip()
        if not n:
            continue
        if not hasattr(CoreEvent, n):
            raise SystemExit(
                f"unknown core event {n!r}. Names come from aie.utils.trace.events.CoreEvent."
            )
        out.append(getattr(CoreEvent, n))
    if not out:
        raise SystemExit("no events given")
    if len(out) > 8:
        raise SystemExit(f"the trace unit takes at most 8 events per tile, got {len(out)}")
    return out


@iron.jit
def pmu_probe(a_in: In, c_out: Out, *, trace_size: CompileTime[int] = 0,
              events: CompileTime[str] = ",".join(DEFAULT_EVENTS)):
    """kernels/clock_probe's design, with the traced event list as a parameter."""
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
            coretile_events=resolve_events(events.split(",")),
        )
    return prog.resolve_program()


def core_byte_stream(tc):
    """The core tile's raw trace bytes, via the same path clock_probe.TraceReader uses."""
    buf = tc.read_trace()
    words = [f"{int(w):08x}" for w in buf]
    end = len(words)
    while end > 0 and words[end - 1] == "00000000":
        end -= 1
    if end == 0:
        raise ValueError("Invalid trace data: empty or all zeros")
    streams = convert_to_byte_stream(trace_pkts_de_interleave(words[:end]))
    core = streams[0]
    loc, bs = next(iter(core.items()))
    row, col = (int(x) for x in loc.split(","))
    return row, col, bs


def frame_dump(bs, limit):
    """Walk the frames the way clock_probe.decode_event_time does, printing each one.

    Only the shape of the stream is printed, not an interpretation of it: the point is to
    see how the hardware encodes an event that is true for many consecutive cycles.
    """
    t = 0
    cursor = 0
    n = len(bs)
    shown = 0
    counts = {}
    while cursor < n and shown < limit:
        b = bs[cursor]
        kind = width = None
        detail = ""
        if (b & 0xFB) == 0xF0:
            kind, width, detail = "Start", 8, "absolute timer"
        elif (b & 0xFC) == 0xDC:
            kind, width = "Skip4", 4
        elif b == 0xFE:
            kind, width = "Filler", 1
        elif b == 0xFF:
            kind, width = "Sync", 1
            t += cp.SYNC_CYCLES
        elif (b & 0x80) == 0:
            ev, cyc, width = 1 << ((b >> 4) & 7), b & 0xF, 1
            kind, detail = "Single0", f"slots={ev:08b} +{cyc}"
        elif (b & 0xE0) == 0x80:
            ev, cyc, width = 1 << ((b >> 2) & 7), ((b & 3) << 8) | bs[cursor + 1], 2
            kind, detail = "Single1", f"slots={ev:08b} +{cyc}"
        elif (b & 0xE0) == 0xA0:
            ev = 1 << ((b >> 2) & 7)
            cyc = ((b & 3) << 16) | (bs[cursor + 1] << 8) | bs[cursor + 2]
            width, kind, detail = 3, "Single2", f"slots={ev:08b} +{cyc}"
        elif (b & 0xF0) == 0xC0:
            ev, cyc, width = ((b & 0xF) << 4) | (bs[cursor + 1] >> 4), bs[cursor + 1] & 0xF, 2
            kind, detail = "Multiple0", f"slots={ev:08b} +{cyc}"
        elif (b & 0xFC) == 0xD0:
            ev = ((b & 3) << 6) | (bs[cursor + 1] >> 2)
            cyc, width = ((bs[cursor + 1] & 3) << 8) | bs[cursor + 2], 3
            kind, detail = "Multiple1", f"slots={ev:08b} +{cyc}"
        elif (b & 0xFC) == 0xD4:
            ev = ((b & 3) << 6) | (bs[cursor + 1] >> 2)
            cyc, width = ((bs[cursor + 1] & 3) << 16) | (bs[cursor + 2] << 8) | bs[cursor + 3], 4
            kind, detail = "Multiple2", f"slots={ev:08b} +{cyc}"
        elif (b & 0xF0) == 0xE0:
            r, width = b & 0xF, 1
            kind, detail = "Repeat0", f"x{r}"
        elif (b & 0xFC) == 0xD8:
            r, width = ((b & 3) << 8) | bs[cursor + 1], 2
            kind, detail = "Repeat1", f"x{r}"
        else:
            kind, width, detail = f"UNKNOWN(0x{b:02x})", 1, ""
        counts[kind] = counts.get(kind, 0) + 1
        print(f"    {cursor:6d}  0x{b:02x}  {kind:<10} {detail}")
        cursor += width
        shown += 1
    print(f"    ... {n} bytes total, showed {shown} frames")
    print("    frame kinds:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return counts


def event_counts(bs, names):
    """How many times each traced event appears, by slot index in the event list."""
    hits = cp.decode_event_time(bs)
    counts = {i: 0 for i in range(len(names))}
    first = {}
    last = {}
    for slot, ts in hits:
        if slot in counts:
            counts[slot] += 1
            first.setdefault(slot, ts)
            last[slot] = ts
    return counts, first, last


def one_run(a, c_out, mode, target, names, trace_size, tc):
    """Dispatch once and return the decoded per-event picture, or None if verify fails."""
    a[0] = MODES[mode]
    a[1] = target
    res = pmu_probe(a, c_out, trace_size=trace_size, events=",".join(names))
    d = cp.decode(c_out.numpy())
    if not (
        d["mode"] == MODES[mode]
        and d["target"] == target
        and d["row"] in CORE_ROWS
        and d["check"] == cp.expected_check(MODES[mode], target)
    ):
        return None
    row, col, bs = core_byte_stream(tc)
    counts, first, last = event_counts(bs, names)
    idx = {n: i for i, n in enumerate(names)}
    loop = None
    if idx.get("INSTR_EVENT_0") in first and idx.get("INSTR_EVENT_1") in first:
        loop = first[idx["INSTR_EVENT_1"]] - first[idx["INSTR_EVENT_0"]]
    return {
        "mode": mode,
        "target": target,
        "bytes": len(bs),
        "hw_ms": (cp._npu_time_ns(res) or float("nan")) / 1e6,
        "counts": counts,
        "first": first,
        "last": last,
        "loop": loop,
        "idx": idx,
    }


def calibrate(a, c_out, names, trace_size, tc, points):
    """Reproduce the two loops S0 measured, then close the cycle accounting.

    Gate 1: the decoded span between the two instruction events must give the cycles per
    iteration S0 measured on this machine, 9.000 scalar and 2.000 vector.
    Gate 2: a stall term must be non-zero somewhere, or the decoder has only been shown
    to count zeros. Nothing artificial is needed -- the core waits on the input
    ObjectFifo's lock every dispatch.
    """
    expect = {"scalar": 9.0, "vector": 2.0}
    stall_names = [n for n in names if n.endswith("_STALL")]
    print()
    print("CALIBRATION -- the two loops results/aie/clock_probe_npu.log already measured")
    print()
    hdr = (f"{'loop':>7} {'iters':>8} {'span':>10} {'cyc/iter':>9} {'expect':>7} "
           f"{'err %':>7} {'active':>9} {'stalled':>9} {'issuing':>9} {'bytes':>6}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    ok = True
    for mode, target in points:
        r = one_run(a, c_out, mode, target, names, trace_size, tc)
        if r is None:
            print(f"{mode:>7} {target:>8}  VERIFY FAILED, excluded")
            ok = False
            continue
        per = r["loop"] / target if r["loop"] else float("nan")
        err = 100.0 * (per - expect[mode]) / expect[mode]
        active = r["counts"].get(r["idx"].get("ACTIVE", -1), 0)
        stalled = sum(r["counts"].get(r["idx"][n], 0) for n in stall_names)
        print(
            f"{mode:>7} {target:>8} {r['loop']:>10} {per:>9.4f} {expect[mode]:>7.3f} "
            f"{err:>+7.2f} {active:>9} {stalled:>9} {active - stalled:>9} {r['bytes']:>6}"
        )
        if abs(err) > 1.0:
            ok = False
        rows.append((r, per, active, stalled))

    print()
    print("CYCLE ACCOUNTING -- does the traced work add up to the cycles the core was alive?")
    print()
    hdr2 = (f"{'loop':>7} {'iters':>8} {'active':>9} {'stalled':>9} {'issuing':>9} "
            f"{'loop+flush':>11} {'unaccounted':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    for r, per, active, stalled in rows:
        # The kernel emits FLUSH_PAIRS=256 event0/event1 pairs after the loop; the
        # disassembly in results/aie/aie2_isa_static.log shows that flush loop is 8
        # bundles for 4 pairs, so 512 cycles.
        accounted = (r["loop"] or 0) + 512
        print(
            f"{r['mode']:>7} {r['target']:>8} {active:>9} {stalled:>9} "
            f"{active - stalled:>9} {accounted:>11} {active - stalled - accounted:>11}"
        )

    any_stall = any(s > 0 for _, _, _, s in rows)
    print()
    print(f"GATE 1 cycles per iteration within 1% of the measured values: "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"GATE 2 at least one stall term non-zero: {'PASS' if any_stall else 'FAIL'}")
    return 0 if (ok and any_stall) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default=",".join(DEFAULT_EVENTS),
                    help="comma-separated CoreEvent names, at most 8")
    ap.add_argument("--mode", choices=sorted(MODES), default="vector")
    ap.add_argument("--target", type=int, default=4096, help="loop iterations")
    ap.add_argument("--trace-size", type=int, default=65536)
    ap.add_argument("--trace-file", default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="run the two loops S0 measured and gate on reproducing them")
    ap.add_argument("--raw", action="store_true", help="dump the frame stream")
    ap.add_argument("--raw-limit", type=int, default=60)
    ap.add_argument("--label", default="pmu")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.events.split(",") if n.strip()]
    resolve_events(names)  # fail early on a bad name

    print(f"== pmu_probe [{args.label}] mode={args.mode} target={args.target} "
          f"trace_size={args.trace_size}")
    print("   events:", ", ".join(f"{i}:{n}" for i, n in enumerate(names)))
    print(cp.platform_report())

    a = iron.zeros(N_IO, dtype=np.int32, device="npu")
    c_out = iron.zeros_like(a)
    a[0] = MODES[args.mode]
    a[1] = args.target

    trace_file = args.trace_file or os.path.join(
        os.environ.get("TEMP", "/tmp"), f"pmu_probe_{os.getpid()}.txt"
    )
    tc = TraceConfig(trace_size=args.trace_size, trace_file=trace_file)
    pmu_probe.trace_config = tc

    if args.calibrate:
        points = [
            ("vector", 4096), ("vector", 16384), ("vector", 65536),
            ("scalar", 2048), ("scalar", 8192), ("scalar", 32768),
        ]
        return calibrate(a, c_out, names, args.trace_size, tc, points)

    res = pmu_probe(a, c_out, trace_size=args.trace_size, events=",".join(names))
    d = cp.decode(c_out.numpy())
    ok = (
        d["mode"] == MODES[args.mode]
        and d["target"] == args.target
        and d["row"] in CORE_ROWS
        and d["check"] == cp.expected_check(MODES[args.mode], args.target)
    )
    hw_ns = cp._npu_time_ns(res)
    print(f"   kernel echo: mode={d['mode']} target={d['target']} tile=({d['row']},{d['col']}) "
          f"checksum {'ok' if d['check'] == cp.expected_check(MODES[args.mode], args.target) else 'BAD'}")
    print(f"   verify: {'PASS' if ok else 'FAIL'}   hw {hw_ns / 1e6 if hw_ns else float('nan'):.3f} ms")
    if not ok:
        return 1

    row, col, bs = core_byte_stream(tc)
    print(f"   trace: {len(bs)} bytes from physical tile ({row},{col})")

    if args.raw:
        print("   frame stream:")
        frame_dump(bs, args.raw_limit)

    counts, first, last = event_counts(bs, names)
    print()
    print(f"   {'slot':>4}  {'event':<16} {'hits':>8}  {'first':>12}  {'last':>12}  {'span':>12}")
    print("   " + "-" * 72)
    for i, n in enumerate(names):
        f = first.get(i)
        l = last.get(i)
        span = (l - f) if (f is not None and l is not None) else None
        print(f"   {i:>4}  {n:<16} {counts[i]:>8}  "
              f"{f if f is not None else '-':>12}  {l if l is not None else '-':>12}  "
              f"{span if span is not None else '-':>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
