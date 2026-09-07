"""
Floor measurement for the two-process splice this repo has not yet built: an
EP-side process (resnet_env17, python 3.12) handing a real GroupNorm(32) input
tensor to a kernel-side process (ironenv, python 3.13) and getting the result
back. results/bit/profile_instancenorm_splice_feasibility.log already showed
pyxrt can never load inside resnet_env17 (hard ABI dependency on python313.dll)
-- so any splice is two OS processes, not an in-process custom op. This script
measures the cheapest possible version of that handoff: shared-memory ping-
pong of the exact byte volume and fp32<->bf16 conversion the real splice would
need, with NO onnxruntime and NO actual NPU dispatch on the kernel side (an
identity copy stands in for gn_stats/gn_finalize/gn_affine). Whatever this
floor costs, a real splice can only be slower -- so if the floor already
exceeds a shape's CPU-vs-kernel margin from
results/aie/groupnorm_bf16_kernel_npu.log, that shape's win does not survive
a real splice and no further wiring can rescue it.

Run one instance per role, in the env each role would actually use in a real
splice (though nothing here imports onnxruntime or iron -- both roles are
plain numpy + multiprocessing.shared_memory, so this floor is also every
process's minimum, not the env's own minimum):

    # terminal 1 (resnet_env17), start first -- this is the client, retries
    # attach for up to --attach-timeout seconds
    conda activate resnet_env17
    python kernels/groupnorm_bf16/measure_handoff_floor.py --role ep --L 301056

    # terminal 2 (ironenv), the server -- creates the shared-memory segments
    . C:\\Users\\<user>\\mlir-aie\\iron_env.ps1
    python kernels/groupnorm_bf16/measure_handoff_floor.py --role kernel --L 301056

Order doesn't matter in practice: the ep role retries its attach, and the
kernel role clears any stale same-named segment before creating its own.
"""
import argparse
import sys
import time
from multiprocessing import shared_memory

import numpy as np

try:
    from ml_dtypes import bfloat16 as _BF16_DTYPE
except ImportError:
    _BF16_DTYPE = None

# Control-block layout: 4 int64 slots in one small shared segment.
#   [0] state: 0 = idle/ready-for-input, 1 = input ready, 2 = output ready, 3 = stop
#   [1] iteration counter (written by ep, read by kernel for a sanity check)
#   [2] L this run was started with (written once by whichever role creates it)
#   [3] unused
CTRL_SLOTS = 4
CTRL_DTYPE = np.int64

IDLE, INPUT_READY, OUTPUT_READY, STOP = 0, 1, 2, 3

GROUPS = 32


def bf16_pack(x_f32):
    """fp32 -> raw bf16 bits (uint16), round-to-nearest-even -- matches the
    aie::set_rounding(conv_even) store the real kernel uses, so this floor's
    conversion cost is the same operation, not a cheaper stand-in. Prefers
    ml_dtypes (measured ~2.5x faster than the hand-rolled bit-trick path
    below, which it turns out IS installed in resnet_env17) so the reported
    floor is the best readily-available conversion, not an artificially slow
    one -- falls back to the bit trick only if ml_dtypes is absent."""
    if _BF16_DTYPE is not None:
        return x_f32.astype(_BF16_DTYPE).view(np.uint16)
    u = x_f32.view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) >> 16).astype(np.uint16)


def bf16_unpack(bits_u16):
    """raw bf16 bits -> fp32 (exact, zero-extend)."""
    if _BF16_DTYPE is not None:
        return bits_u16.view(_BF16_DTYPE).astype(np.float32)
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)


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

    rng = np.random.default_rng(args.seed)
    total = args.warmup + args.iters
    times_ns = np.empty(args.iters, dtype=np.int64)

    for i in range(total):
        x_f32 = rng.standard_normal(n_elems, dtype=np.float32)  # stand-in for ORT's real fp32 tensor

        while ctrl_arr[0] != IDLE:
            pass
        t0 = time.perf_counter_ns()
        in_arr[:] = bf16_pack(x_f32)
        ctrl_arr[1] = i
        ctrl_arr[0] = INPUT_READY

        while ctrl_arr[0] != OUTPUT_READY:
            pass
        y_f32 = bf16_unpack(out_arr)
        t1 = time.perf_counter_ns()
        ctrl_arr[0] = IDLE

        if i >= args.warmup:
            times_ns[i - args.warmup] = t1 - t0

    ctrl_arr[0] = STOP

    us = times_ns / 1e3
    print(f"handoff floor (ep role), L={args.L}, n={args.iters} iters "
          f"(after {args.warmup} warmup):")
    print(f"  round-trip us (avg/min/max): {us.mean():.1f} / {us.min():.1f} / {us.max():.1f}")
    print(f"  payload per direction: {nbytes/1e6:.2f} MB bf16 "
          f"({n_elems} elems from a [1,{GROUPS},{args.L}] tensor)")
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(f"L={args.L} iters={args.iters} warmup={args.warmup} "
                    f"avg_us={us.mean():.1f} min_us={us.min():.1f} max_us={us.max():.1f} "
                    f"p50_us={np.median(us):.1f} p90_us={np.percentile(us,90):.1f}\n")

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
        # wait specifically for INPUT_READY/STOP -- "not IDLE" is also true
        # right after we set OUTPUT_READY ourselves, which would otherwise
        # make us re-process our own stale flag before the ep role resets it.
        while ctrl_arr[0] != INPUT_READY and ctrl_arr[0] != STOP:
            pass
        state = ctrl_arr[0]
        if state == STOP:
            break
        # identity copy stands in for the real gn_stats/gn_finalize/gn_affine
        # NPU dispatch -- this floor measures IPC + conversion only.
        out_arr[:] = in_arr[:]
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
    ap.add_argument("--name", default="gn_handoff")
    ap.add_argument("--attach-timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="append ep-role summary line to this file")
    args = ap.parse_args()

    if args.role == "ep":
        run_ep(args)
    else:
        run_kernel(args)


if __name__ == "__main__":
    main()
