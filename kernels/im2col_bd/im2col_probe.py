#!/usr/bin/env python3
"""Can the MemTile's 4-D BD emit overlapping KxK receptive fields on its own?

WHY THIS EXISTS
---------------
results/aie/conv_issue_rate_decomposed.log closed objective K1's first question without a
trace: the int8 conv's 11.3x shortfall is ISSUE RATE, not data movement. The int8 GEMM
issues 0.889 vmac/cycle; the 3x3 conv's main loop manages 0.222, and it does so because
SIX of its eighteen bundles are `vshift` and FOUR more are `vmov` -- ten bundles of
sliding-window realignment spent in issue slots. docs/SILICON.md names the fix in K1's
tooling section: "im2col or row shifting done by the mem tile's 4-D BDs rather than by
core code", and SILICON 2.6 calls the 4-D BD "the one address generator on the chip that
can do an im2col or a transpose in flight without a core touching the data."

That is a claim about the silicon that this repo has never tested. This probe tests it,
and ONLY it. There is no compute tile and no kernel: data goes shim -> memtile -> shim,
and the memtile's MM2S applies the access pattern on the way out. If the pattern comes
back byte-for-byte equal to a numpy im2col, the addressing engine can express overlap and
the ten bundles are removable in principle. If it cannot, the whole K1 tooling plan needs
a different mechanism and no kernel should be written against it.

WHY OVERLAP IS THE QUESTION
---------------------------
A BD walks (size, stride) pairs from highest to lowest dimension. Nothing requires the
outer step to clear the inner span, so an outer dimension stepping by ONE element while an
inner dimension reads K elements RE-READS the same addresses -- which is exactly im2col.
For a KxK window with unit stride over an HxW plane the pattern is four-dimensional:

    [(out_h, W), (out_w, 1), (K, W), (K, 1)]

emitting out_h*out_w*K*K elements from an H*W buffer. The expansion factor is K*K (9 for
3x3), and that is the cost this trades for the ten issue slots.

WHAT IS CHECKED BEFORE THE RUN
------------------------------
docs/SILICON.md 2.2 gives the mem tile BD fields as a 10-bit wrap and a 17-bit step, so
every size must be < 1024 and every stride < 131072. The probe asserts that against the
shape it was given rather than discovering it as a compile error, and prints the pattern.

USAGE (from the mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/im2col_bd/im2col_probe.py
    python3 kernels/im2col_bd/im2col_probe.py --height 32 --width 32 --k 3
    python3 kernels/im2col_bd/im2col_probe.py --k 1     # degenerate: no overlap
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyMemTile, AnyShimTile

# docs/SILICON.md 2.2: mem tile BD is 10-bit wrap, 17-bit step.
BD_MAX_SIZE = 1 << 10
BD_MAX_STRIDE = 1 << 17


def im2col_dims(h, w, k):
    """The 4-D (size, stride) pattern for a KxK unit-stride window over HxW.

    Highest dimension first, strides in elements -- the order dims_to_stream takes.
    """
    out_h, out_w = h - k + 1, w - k + 1
    return [(out_h, w), (out_w, 1), (k, w), (k, 1)], out_h, out_w


def im2col_reference(x, k):
    """What the stream must equal, computed on the host with no AIE involvement."""
    h, w = x.shape
    out_h, out_w = h - k + 1, w - k + 1
    out = np.empty((out_h, out_w, k, k), dtype=x.dtype)
    for oh in range(out_h):
        for ow in range(out_w):
            out[oh, ow] = x[oh:oh + k, ow:ow + k]
    return out.reshape(-1)


@iron.jit
def im2col_forward(a_in: In, c_out: Out, *, h: CompileTime[int] = 16,
                   w: CompileTime[int] = 16, k: CompileTime[int] = 3,
                   wide_obj: CompileTime[int] = 0):
    """shim -> memtile -> shim, with the memtile's MM2S doing the im2col.

    No compute tile and no kernel: whatever comes back was produced by the address
    generator alone, which is the whole point of the probe.
    """
    dims, out_h, out_w = im2col_dims(h, w, k)
    in_ty = np.ndarray[(h * w,), np.dtype[np.int32]]
    out_ty = np.ndarray[(out_h * out_w * k * k,), np.dtype[np.int32]]

    of_in = ObjectFifo(in_ty, name="im2col_in")
    # The 4-D pattern is applied HERE, on the memtile's outbound stream.
    # wide_obj decides whether the FORWARDED fifo's object is the input tile (default,
    # so the memtile streams out of an H*W buffer) or the expanded stream length. It is a
    # flag because it separates two candidate causes of the k=3 timeout: an ObjectFifo
    # that binds transfer length to object size, versus the address generator itself
    # refusing to re-read.
    of_out = of_in.cons().forward(
        tile=AnyMemTile,
        name="im2col_out",
        obj_type=out_ty if wide_obj else None,
        dims_to_stream=dims,
    )

    def sequence(a, c, in_h, out_h_):
        in_h.fill(a)
        out_h_.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            in_ty,
            out_ty,
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--k", type=int, default=3, help="window size (3 for a 3x3 conv)")
    ap.add_argument("--wide-obj", action="store_true",
                    help="give the forwarded fifo the EXPANDED object type instead of "
                         "the input tile's; separates a length-binding limit from an "
                         "addressing limit")
    ap.add_argument("--show", type=int, default=27,
                    help="elements of the stream to print on a mismatch")
    args = ap.parse_args(argv)

    h, w, k = args.height, args.width, args.k
    if k > h or k > w:
        sys.exit(f"k={k} does not fit in {h}x{w}")

    dims, out_h, out_w = im2col_dims(h, w, k)
    n_out = out_h * out_w * k * k

    print(f"== im2col_bd  {h}x{w}  window {k}x{k}  ->  {out_h}x{out_w}x{k}x{k}")
    print(f"   dims_to_stream (size, stride), highest first: {dims}")
    print(f"   input {h*w} elements -> stream {n_out} elements "
          f"(expansion {n_out / (h*w):.2f}x, ideal {k*k})")

    # Gate on the BD field widths BEFORE compiling, so an over-wide shape is a stated
    # limit rather than an opaque compiler error (docs/SILICON.md 2.2).
    for i, (size, stride) in enumerate(dims):
        if size >= BD_MAX_SIZE:
            sys.exit(f"dim {i} size {size} exceeds the mem tile's 10-bit wrap "
                     f"({BD_MAX_SIZE}) -- docs/SILICON.md 2.2")
        if stride >= BD_MAX_STRIDE:
            sys.exit(f"dim {i} stride {stride} exceeds the mem tile's 17-bit step "
                     f"({BD_MAX_STRIDE}) -- docs/SILICON.md 2.2")
    print("   BD field widths: OK (all sizes < 1024, all strides < 131072)")

    # Distinct values so a wrong element is identifiable, not merely unequal.
    # iron.arange is the allocation path measure_runlist.py uses; constructing the tensor
    # from a numpy int32 array instead raises on the int32->uint32 cast.
    src = np.arange(1, h * w + 1, dtype=np.int32)
    a = iron.arange(1, h * w + 1, dtype=np.int32, device="npu")
    c = iron.zeros(n_out, dtype=np.int32, device="npu")

    im2col_forward(a, c, h=h, w=w, k=k, wide_obj=1 if args.wide_obj else 0)

    got = np.asarray(c.numpy()).reshape(-1)[:n_out]
    want = im2col_reference(src.reshape(h, w), k)

    if np.array_equal(got, want):
        print(f"\nMATCH: all {n_out} elements equal the host im2col.")
        print("The mem tile's 4-D BD CAN emit overlapping receptive fields with no core")
        print("involvement, so conv2dk3's vshift/vmov realignment is removable in")
        print("principle. This says nothing yet about the RATE -- see the log.")
        return 0

    bad = int(np.flatnonzero(got != want)[0])
    print(f"\nMISMATCH at element {bad} of {n_out}: got {got[bad]}, want {want[bad]}")
    print(f"  first {args.show} got : {got[:args.show].tolist()}")
    print(f"  first {args.show} want: {want[:args.show].tolist()}")
    print("The addressing engine did NOT reproduce im2col for this pattern. Record the")
    print("shape and the first divergence; do not write a kernel against this mechanism")
    print("until it is understood.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
