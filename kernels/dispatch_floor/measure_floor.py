#!/usr/bin/env python3
"""Measure the NPU's per-dispatch fixed cost in isolation.

WHY THIS EXISTS
---------------
Two hand-written kernels in this repo (kernels/groupnorm_bf16, kernels/attention_bf16)
lost to the Zen4 CPU, and both write-ups attribute the loss to a per-dispatch fixed cost
of "~185-200us". That number was never measured on its own -- it was inferred from one
96KB probe during the GroupNorm work and then reused as a constant. Every isolated-op
verdict in this repo rests on it, so it is worth measuring directly.

WHAT IT MEASURES
----------------
A design with NO COMPUTE TILE AT ALL: data flows shim -> memtile -> shim via
ObjectFifo.forward(), the same structure as mlir-aie's basic/passthrough_dmas. There is
no kernel math to attribute time to, so wall time per call is dispatch + DMA transfer
and nothing else.

Sweeping the payload and fitting  time = intercept + slope * bytes  separates them:

    intercept -> the fixed per-call cost (the number this repo has been quoting)
    slope     -> the streaming bandwidth (1/slope = GB/s)

The intercept is the honest floor: the cost a kernel pays before it has moved a single
byte or done any math. Any op whose CPU time is below it can never win on this hardware
no matter how good the kernel is, which makes it a go/no-go test to run BEFORE writing
a kernel rather than after.

WHAT IS INCLUDED IN THE NUMBER
------------------------------
This times the full `@iron.jit` call path -- the same path kernels/attention_bf16 and
kernels/groupnorm_bf16 use, and therefore the cost those results actually paid. That
includes IRON's Python wrapper, XRT buffer sync, and the hardware dispatch. It is the
cost of a real design, NOT a theoretical hardware minimum; a lower-level raw-pyxrt
submission path could plausibly be cheaper (--raw-hint prints what to compare against).
Compile is excluded: every payload is warmed up before timing, and @iron.jit caches the
built kernel, so the timed loop recompiles nothing.

CORRECTNESS GATE
----------------
A floor measured from a kernel that silently did nothing would be worthless, so every
payload is verified (output == input) before its timings are kept. A payload that fails
verification is reported and excluded from the fit rather than quietly averaged in.

USAGE (from the mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/dispatch_floor/measure_floor.py
    python3 kernels/dispatch_floor/measure_floor.py --iters 200 --warmup 10
"""

import argparse
import statistics
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyShimTile

LINE_SIZE = 1024  # transfer chunk; every payload must be a multiple of this


@iron.jit
def passthrough(a_in: In, _unused: In, c_out: Out, *, n: CompileTime[int] = 4096):
    """shim -> memtile -> shim. No compute tile, so no math to attribute time to."""
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    def sequence(a, _, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            vector_ty,
            vector_ty,
            vector_ty,
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


def time_payload(n, iters, warmup):
    """Return (wall times ms, hw times ms, verified_ok) for one payload size.

    Two layers are timed per call, which is the whole point of this harness:

      wall -- perf_counter around the @iron.jit call. What a design actually pays.
      hw   -- the runtime's OWN measurement, which brackets only
              `kernel(3, insts_bo, nbytes, *bufs)` + `wait()` (see
              xrtruntime/hostruntime.py run()). It is returned per call as
              KernelResult.npu_time in nanoseconds. This is submit+wait with no
              Python bookkeeping in it.

    wall - hw is IRON's per-call host-side cost. Note what that does and does NOT
    mean: the hw bracket is NARROW -- it starts after the hw_context cache lookup,
    the pyxrt kernel-handle retrieval and any insts_bo/buffer coherence work, so
    all of that lands in the host column despite being XRT/driver time rather than
    interpreter time. A cProfile of this path attributes its single largest entry
    (XRTRuntime.load) with tottime == cumtime, i.e. time inside a C call the
    profiler cannot see into. Treat wall-hw as "host-side, mostly XRT outside the
    submit bracket", not as "Python overhead".
    """
    a = iron.arange(1, n + 1, dtype=np.int32, device="npu")
    b = iron.zeros_like(a)
    c = iron.zeros_like(a)

    for _ in range(warmup):
        passthrough(a, b, c, n=n)

    # Correctness gate: a floor measured from a no-op kernel would be meaningless.
    verified_ok = bool(np.array_equal(c.numpy(), a.numpy()))

    wall, hw = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = passthrough(a, b, c, n=n)
        wall.append((time.perf_counter() - t0) * 1e3)
        npu_ns = _npu_time_ns(res)
        if npu_ns is not None:
            hw.append(npu_ns / 1e6)
    return wall, hw, verified_ok


def _npu_time_ns(res):
    """Dig KernelResult.npu_time out of whatever load_and_run returned."""
    for cand in (res, *(res if isinstance(res, tuple) else ())):
        t = getattr(cand, "npu_time", None)
        if isinstance(t, (int, float)):
            return t
    return None


def fit_line(xs, ys):
    """Least-squares fit y = intercept + slope*x. Returns (intercept, slope, r2)."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=100, help="timed calls per payload")
    p.add_argument("--warmup", type=int, default=5, help="untimed calls per payload")
    p.add_argument(
        "--sizes",
        type=str,
        default="1024,2048,4096,8192,16384,32768,65536,131072,262144",
        help="comma-separated element counts (int32), each a multiple of 1024",
    )
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    for n in sizes:
        if n % LINE_SIZE:
            sys.exit(f"size {n} is not a multiple of {LINE_SIZE}")

    print(f"NPU per-dispatch floor -- no-compute passthrough (shim->memtile->shim)")
    print(f"iters={args.iters} warmup={args.warmup} (compile excluded, kernel cached)\n")
    print(
        f"{'elems':>8} {'KB moved':>9} {'wall ms':>9} {'wall min':>9} "
        f"{'hw ms':>9} {'hw min':>9} {'host ms':>9}  ok"
    )
    print("-" * 80)

    xs, ys, hw_xs, hw_ys, failures = [], [], [], [], []
    for n in sizes:
        wall, hw, ok = time_payload(n, args.iters, args.warmup)
        kb = (n * 4 * 2) / 1024.0  # int32 in + int32 out
        w_mean = statistics.mean(wall)
        h_mean = statistics.mean(hw) if hw else float("nan")
        print(
            f"{n:>8} {kb:>9.1f} {w_mean:>9.4f} {min(wall):>9.4f} "
            f"{h_mean:>9.4f} {(min(hw) if hw else float('nan')):>9.4f} "
            f"{w_mean - h_mean:>9.4f}  {'yes' if ok else 'NO'}"
        )
        if ok:
            xs.append(n * 4 * 2)  # bytes moved
            ys.append(w_mean)
            if hw:
                hw_xs.append(n * 4 * 2)
                hw_ys.append(h_mean)
        else:
            failures.append(n)

    if failures:
        print(f"\n!! payloads FAILED verification and were excluded: {failures}")
    if len(xs) < 3:
        sys.exit("\nnot enough verified points to fit")

    # slope is ms/byte -> bytes/s is 1/(slope*1e-3); /1e9 for GB/s == 1e-6/slope
    def report(label, ax, ay):
        icept, slope, r2 = fit_line(ax, ay)
        gbps = (1e-6 / slope) if slope > 0 else float("nan")
        print(f"\n{label}:  time_ms = intercept + slope * bytes   (R^2 = {r2:.4f})")
        print(f"  intercept (FIXED PER-DISPATCH COST) : {icept * 1e3:.1f} us")
        print(f"  slope                               : {slope * 1e6:.4f} ns/byte")
        print(f"  implied streaming bandwidth         : {gbps:.2f} GB/s")
        return icept

    wall_icept = report("WALL (what a design pays today)", xs, ys)
    hw_icept = None
    if len(hw_xs) >= 3:
        hw_icept = report("HARDWARE ONLY (submit+wait, no Python)", hw_xs, hw_ys)

    print("\nGo/no-go threshold this implies:")
    print(f"  An op whose CPU time is under {wall_icept * 1e3:.1f} us cannot win through")
    print("  the IRON call path regardless of kernel quality.")
    if hw_icept is not None:
        print(
            f"  Of that, {hw_icept * 1e3:.1f} us is hardware and "
            f"{(wall_icept - hw_icept) * 1e3:.1f} us is host-side overhead that a"
        )
        print("  leaner submission path could in principle remove.")
    print("  Compare against the measured CPU times before writing a kernel:")
    print("    attention_bf16 stage 4: CPU 12 us   stage 3: CPU 34 us   stage 2: CPU 240 us")


if __name__ == "__main__":
    main()
