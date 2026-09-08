"""
Follow-up to measure_handoff_floor.py, testing a specific challenge to its
"floor" claim: that floor used ml_dtypes.astype() for fp32<->bf16 conversion,
which is a scalar per-element loop (bfloat16 isn't a native numpy dtype, so
there's no SIMD path) -- so the floor may have measured a slow conversion
function, not physics. This script replaces that conversion with a strided
view: the high 16 bits of a little-endian fp32 word ARE bf16 by truncation
(no arithmetic), so pack is `x_f32.view(uint16)[1::2]` and unpack is the
mirror (write into the upper half of a zeroed uint32 buffer, view as float32).
Buffers are preallocated once and reused via out=/in-place writes, so nothing
allocates in the hot loop -- both changes were named explicitly as untried in
the prior log's "What this does NOT show" section.

Trade: truncation drops up to ~1 ULP more than round-to-nearest-even, on top
of the ~0.17% rel-L2 bf16 rounding error the kernel already carries (checked
against real golden tensors in groupnorm.py's own tolerance check) -- not
re-verified numerically here, since this script (like the v1 floor) never
touches the real kernel's output, only IPC+conversion cost.

Everything else (shared-memory ping-pong, busy-spin control block, identity
copy standing in for the kernel) is unchanged from measure_handoff_floor.py --
see that file's docstring for the full protocol. Adds --no-convert to isolate
the busy-spin/shared-memory protocol cost alone (raw uint16 passthrough, zero
conversion work), since the prior log's residual-after-conversion estimate
(~2.1ms at L=301056, extrapolated not measured at other shapes) was flagged
as an assumption, not a measurement, at every shape but L=301056.

Usage: identical to measure_handoff_floor.py (--role ep / --role kernel, one
process per role, in the env each role would actually use).
"""
import argparse
import time
from multiprocessing import shared_memory

import numpy as np

CTRL_SLOTS = 4
CTRL_DTYPE = np.int64
IDLE, INPUT_READY, OUTPUT_READY, STOP = 0, 1, 2, 3
GROUPS = 32


def attach_ctrl_and_data(name, nbytes, attach_timeout):
    deadline = time.perf_counter() + attach_timeout
    last_err = None
    while time.perf_counter() < deadline:
        try:
            ctrl = shared_memory.SharedMemory(name=f"{name}_ctrl")
            din = shared_memory.SharedMemory(name=f"{name}_in")
            dout = shared_memory.SharedMemory(name=f"{name}_out")
            return ctrl, din, dout
        except FileNotFoundError as e:
            last_err = e
            time.sleep(0.05)
    raise TimeoutError(
        f"never found shared memory '{name}_*' within {attach_timeout}s "
        f"-- start the kernel-role process first (or increase --attach-timeout)"
    ) from last_err


def create_ctrl_and_data(name, nbytes):
    for suffix in ("ctrl", "in", "out"):
        try:
            stale = shared_memory.SharedMemory(name=f"{name}_{suffix}")
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
    ctrl = shared_memory.SharedMemory(name=f"{name}_ctrl", create=True,
                                       size=CTRL_SLOTS * np.dtype(CTRL_DTYPE).itemsize)
    din = shared_memory.SharedMemory(name=f"{name}_in", create=True, size=nbytes)
    dout = shared_memory.SharedMemory(name=f"{name}_out", create=True, size=nbytes)
    return ctrl, din, dout


def run_ep(args):
    n_elems = GROUPS * args.L
    nbytes = n_elems * 2  # bf16
    ctrl, din, dout = attach_ctrl_and_data(args.name, nbytes, args.attach_timeout)
    ctrl_arr = np.ndarray((CTRL_SLOTS,), dtype=CTRL_DTYPE, buffer=ctrl.buf)
    in_arr = np.ndarray((n_elems,), dtype=np.uint16, buffer=din.buf)
    out_arr = np.ndarray((n_elems,), dtype=np.uint16, buffer=dout.buf)

    # Preallocated scratch, reused every iteration -- nothing allocates in the
    # timed region. u32_scratch stays zeroed except the upper half, which is
    # overwritten in full every call.
    u32_scratch = np.zeros(n_elems, dtype=np.uint32)

    rng = np.random.default_rng(args.seed)
    total = args.warmup + args.iters
    times_ns = np.empty(args.iters, dtype=np.int64)
    raw_src = rng.integers(0, 2**16, size=n_elems, dtype=np.uint16)  # for --no-convert: no fp32 involved at all

    for i in range(total):
        if not args.no_convert:
            x_f32 = rng.standard_normal(n_elems, dtype=np.float32)

        while ctrl_arr[0] != IDLE:
            pass
        t0 = time.perf_counter_ns()
        if args.no_convert:
            in_arr[:] = raw_src  # raw passthrough, zero conversion work -- protocol floor only
        else:
            in_arr[:] = x_f32.view(np.uint16)[1::2]  # truncating pack: high half of each fp32 word IS bf16-by-truncation
        ctrl_arr[1] = i
        ctrl_arr[0] = INPUT_READY

        while ctrl_arr[0] != OUTPUT_READY:
            pass
        if args.no_convert:
            y_f32 = out_arr  # skip unpack entirely for the protocol-floor isolation run
        else:
            u32_scratch.view(np.uint16)[1::2] = out_arr
            y_f32 = u32_scratch.view(np.float32)
        t1 = time.perf_counter_ns()
        ctrl_arr[0] = IDLE

        if i >= args.warmup:
            times_ns[i - args.warmup] = t1 - t0

    ctrl_arr[0] = STOP

    us = times_ns / 1e3
    mode = "no-convert (protocol floor only)" if args.no_convert else "fast truncating pack/unpack (preallocated)"
    print(f"handoff floor v2 [{mode}] (ep role), L={args.L}, n={args.iters} iters "
          f"(after {args.warmup} warmup):")
    print(f"  round-trip us (avg/min/max): {us.mean():.1f} / {us.min():.1f} / {us.max():.1f}")
    print(f"  payload per direction: {nbytes/1e6:.2f} MB bf16")
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(f"mode={'noconv' if args.no_convert else 'fastconv'} L={args.L} iters={args.iters} "
                    f"warmup={args.warmup} avg_us={us.mean():.1f} min_us={us.min():.1f} "
                    f"max_us={us.max():.1f} p50_us={np.median(us):.1f} p90_us={np.percentile(us,90):.1f}\n")

    ctrl.close()
    din.close()
    dout.close()


def run_kernel(args):
    n_elems = GROUPS * args.L
    nbytes = n_elems * 2
    ctrl, din, dout = create_ctrl_and_data(args.name, nbytes)
    ctrl_arr = np.ndarray((CTRL_SLOTS,), dtype=CTRL_DTYPE, buffer=ctrl.buf)
    ctrl_arr[:] = 0
    ctrl_arr[2] = args.L
    in_arr = np.ndarray((n_elems,), dtype=np.uint16, buffer=din.buf)
    out_arr = np.ndarray((n_elems,), dtype=np.uint16, buffer=dout.buf)

    print(f"kernel role ready, waiting for ep role (L={args.L}) ...")
    n_serviced = 0
    while True:
        while ctrl_arr[0] != INPUT_READY and ctrl_arr[0] != STOP:
            pass
        state = ctrl_arr[0]
        if state == STOP:
            break
        out_arr[:] = in_arr[:]  # identity copy stands in for gn_stats/gn_finalize/gn_affine, same as v1
        ctrl_arr[0] = OUTPUT_READY
        n_serviced += 1

    print(f"kernel role serviced {n_serviced} iterations, stopping.")
    ctrl.close()
    ctrl.unlink()
    din.close()
    din.unlink()
    dout.close()
    dout.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["ep", "kernel"])
    ap.add_argument("--L", type=int, default=301056)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--name", default="gn_handoff_v2")
    ap.add_argument("--attach-timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-convert", action="store_true",
                     help="ep role: skip bf16 pack/unpack, send/receive raw bytes -- isolates the busy-spin/shared-memory protocol cost alone")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.role == "ep":
        run_ep(args)
    else:
        run_kernel(args)


if __name__ == "__main__":
    main()
