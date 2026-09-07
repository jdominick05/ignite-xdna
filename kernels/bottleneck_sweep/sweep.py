#!/usr/bin/env python3
"""Sweep the spatial size of mlir-aie's int8 ResNet bottleneck and fit its GOPS.

WHY THIS EXISTS
---------------
results/aie/conv2x_int8_cpu_baseline.log closed *one design at one shape*: the chained
3-bottleneck `ml/resnet/layers_conv2_x` at 32x32x256 loses to Zen4 CPU ORT int8 by
6.3-8.5x. That log named the experiment needed to widen or narrow the verdict:

    run standalone `ml/bottleneck` at 32x32 against a larger spatial and watch whether
    GOPS scales.

If throughput is flat in problem size, the array is already saturated and conv on this
chip simply loses. If it climbs, 32x32 was underutilizing the array and the verdict is
shape-specific, not op-class-specific.

WHICH AXIS SCALES
-----------------
`tensor_h` is the free axis; `tensor_w` is not. Every L1 buffer in bottleneck.py is
sized `tensor_w * channels` bytes (`l1_in_ty` is (w,1,256) int8), while rows stream
through `range_(tensor_h)` one at a time. At w=128 the depth-2 input FIFO alone is
64 KB -- all of an AIE2 tile's local memory. So the sweep walks h far and w only a
little, and lets the first failure bound w rather than guessing it.

WHAT DECIDES THE QUESTION
-------------------------
Not per-point GOPS. Per-point end-to-end GOPS *must* climb with size regardless of the
array, because the ~450us fixed host cost measured in results/aie/dispatch_floor_npu.log
shrinks as a fraction of a bigger job. The honest metric is a least-squares fit of

    hw_time = intercept + slope * FLOPs

  1/slope  -> marginal GOPS: steady-state array throughput on this kernel, shape-free
  intercept-> fixed cost inside the hardware bracket

Marginal GOPS below the CPU's measured int8 rate means the op class loses on this chip
no matter the shape. Marginal GOPS above it means 32x32 was the problem, not conv.

CORRECTNESS GATE
----------------
Throughput from a kernel that computed the wrong thing is worthless, so every shape is
checked against the same torch int8 golden mlir-aie's own test.py uses (the scales are
per-element and size-independent), with the identical DataShaper layouts. A shape that
fails is reported and excluded from the fit, never quietly averaged in.

USAGE (mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/bottleneck_sweep/sweep.py
    python3 kernels/bottleneck_sweep/sweep.py --shapes 32x32,32x64,56x56 --iters 50

The CPU side of the comparison lives in kernels/bottleneck_sweep/cpu_sweep.py, which
needs onnxruntime and therefore runs in resnet_env. Run both in the same sitting: NPU
latency on this machine drifts between sessions.

NOTE: aiecc also needs xclbinutil, which is NOT in ironenv/Scripts. Put the XRT SDK on
PATH too or the build dies at the final link with "tool 'xclbinutil' not found":
    export PATH=/c/Xilinx/XRT/xrt_sdk/xrt:$PATH
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import aie.iron as iron
from aie.utils.ml import DataShaper

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

# bottleneck.py lives in the mlir-aie checkout, not in this repo. Import it rather than
# copying it, so the sweep measures the upstream design and not a fork of it.
_MLIR_AIE = os.environ.get(
    "MLIR_AIE_DIR", os.path.join(os.path.expanduser("~"), "mlir-aie")
)
_BOTTLENECK_DIR = os.path.join(_MLIR_AIE, "programming_examples", "ml", "bottleneck")
sys.path.insert(0, _BOTTLENECK_DIR)
try:
    from bottleneck import bottleneck  # noqa: E402
except ImportError as e:  # pragma: no cover - environment problem, not a code path
    sys.exit(f"cannot import bottleneck from {_BOTTLENECK_DIR}: {e}")

CHANNELS = 256  # l1_in_c; the design derives 64 / 64 / 256 from it
IN_LAYOUT = ("YCXC8", "CYX")
WTS_LAYOUT = ("OIYXI8O8", "OIYX")
OUT_REORDER = ("CDYX", "YCXD")

# Scales lifted verbatim from mlir-aie's ml/bottleneck/test.py -- they are per-element
# shifts, so they carry to any spatial size unchanged.
INP_SCALE = 0.5
WEIGHT_SCALE = 0.5


def flops(h, w):
    """MACs*2 for the three convs. Channels are fixed by the design (256/64/64/256)."""
    c_in, c_mid = CHANNELS, CHANNELS // 4
    macs = h * w * (c_in * c_mid + 9 * c_mid * c_mid + c_mid * c_in)
    return 2 * macs


def build_golden(int_inp, w1, w2, w3):
    """The int8 bottleneck torch reference from mlir-aie's test.py, weights loaded."""

    class BottleneckInt8(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(
                64, 64, kernel_size=3, padding=1, padding_mode="zeros", bias=False
            )
            self.conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            c1 = self.conv1(x) * INP_SCALE * WEIGHT_SCALE
            r1 = torch.clamp(torch.round(self.relu1(c1) / INP_SCALE), 0, 255)
            c2 = self.conv2(r1) * INP_SCALE * WEIGHT_SCALE
            r2 = torch.clamp(torch.round(self.relu2(c2) / INP_SCALE), 0, 255)
            c3 = self.conv3(r2) * INP_SCALE * WEIGHT_SCALE
            same_scale = torch.clamp(torch.round(c3 / INP_SCALE), -128, 127)
            skip = INP_SCALE * (same_scale + int_inp)
            return INP_SCALE * torch.clamp(torch.round(skip / INP_SCALE), 0, 255)

    m = BottleneckInt8()
    m.conv1.weight.data.copy_(w1)
    m.conv2.weight.data.copy_(w2)
    m.conv3.weight.data.copy_(w3)
    m.eval()
    return m


def make_buffers(h, w):
    """Torch inputs plus the AIE-layout int8 buffers the design expects."""
    int_inp = torch.randint(1, 100, (1, CHANNELS, h, w)).type(torch.FloatTensor)
    w1 = torch.randint(50, 100, (64, 256, 1, 1)).type(torch.FloatTensor)
    w2 = torch.randint(50, 100, (64, 64, 3, 3)).type(torch.FloatTensor)
    w3 = torch.randint(50, 100, (256, 64, 1, 1)).type(torch.FloatTensor)

    ds = DataShaper()
    ifm = ds.reorder_mat(int_inp.squeeze().data.numpy().astype(np.int8), *IN_LAYOUT)
    wts = np.concatenate(
        [
            ds.reorder_mat(t.data.numpy().astype(np.int8), *WTS_LAYOUT)
            for t in (w1, w2, w3)
        ],
        axis=None,
    )
    golden = build_golden(int_inp, w1, w2, w3)(int_inp).detach().numpy()
    return ifm, wts, golden, ds


def check(out_tensor, ds, golden, h, w):
    """AIE output buffer -> torch layout -> allclose against the golden."""
    raw = out_tensor.numpy().view(np.uint8).astype(np.float32) * INP_SCALE
    temp = raw.reshape((h, CHANNELS // 8, w, 8))
    temp = ds.reorder_mat(temp, *OUT_REORDER).reshape((CHANNELS, h, w))
    return bool(np.allclose(temp[None, ...], golden, rtol=0, atol=INP_SCALE))


def _npu_time_ns(res):
    """Dig KernelResult.npu_time out of whatever the jit call returned."""
    for cand in (res, *(res if isinstance(res, tuple) else ())):
        t = getattr(cand, "npu_time", None)
        if isinstance(t, (int, float)):
            return t
    return None


def time_shape(h, w, iters, warmup):
    """Return (wall ms list, hw ms list, verified_ok) for one (h, w)."""
    ifm, wts, golden, ds = make_buffers(h, w)
    a = iron.tensor(ifm, dtype=np.int8)
    b = iron.tensor(wts, dtype=np.int8)
    c = iron.zeros(h * w * CHANNELS, dtype=np.int8)

    for _ in range(warmup):
        bottleneck(a, b, c, tensor_w=w, tensor_h=h, tensor_in_c=CHANNELS)

    ok = check(c, ds, golden, h, w)

    wall, hw = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = bottleneck(a, b, c, tensor_w=w, tensor_h=h, tensor_in_c=CHANNELS)
        wall.append((time.perf_counter() - t0) * 1e3)
        ns = _npu_time_ns(res)
        if ns is not None:
            hw.append(ns / 1e6)
    return wall, hw, ok


def fit_line(xs, ys):
    """Least-squares y = intercept + slope*x. Returns (intercept, slope, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return intercept, slope, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=50, help="timed calls per shape")
    p.add_argument("--warmup", type=int, default=3, help="untimed calls per shape")
    p.add_argument(
        "--shapes",
        type=str,
        default="32x32,64x32,128x32,256x32,512x32,56x56,32x64,64x64,128x64",
        help="comma-separated HxW list; H is the free axis, W is L1-bound",
    )
    args = p.parse_args()

    shapes = []
    for tok in args.shapes.split(","):
        hs, ws = tok.lower().split("x")
        shapes.append((int(hs), int(ws)))

    print("mlir-aie ml/bottleneck (int8, 1 column) -- spatial sweep")
    print(f"iters={args.iters} warmup={args.warmup} (compile excluded, jit-cached)")
    print(f"design: {_BOTTLENECK_DIR}\n")
    head = (
        f"{'HxW':>9} {'MFLOP':>9} {'wall ms':>9} {'hw ms':>9} {'host ms':>9} "
        f"{'hw GOPS':>9} {'e2e GOPS':>9}  ok"
    )
    print(head)
    print("-" * 82)

    xs, ys, failures = [], [], []
    for h, w in shapes:
        f = flops(h, w)
        label = f"{h}x{w}"
        try:
            wall, hw, ok = time_shape(h, w, args.iters, args.warmup)
        except Exception as e:  # a shape that will not compile bounds the sweep
            print(f"{label:>9} {f/1e6:>9.2f}   {type(e).__name__}: {e}")
            failures.append((h, w, f"{type(e).__name__}: {e}"))
            continue
        w_mean = statistics.mean(wall)
        h_mean = statistics.mean(hw) if hw else float("nan")
        print(
            f"{label:>9} {f/1e6:>9.2f} {w_mean:>9.4f} {h_mean:>9.4f} "
            f"{w_mean - h_mean:>9.4f} {f/(h_mean*1e6):>9.1f} "
            f"{f/(w_mean*1e6):>9.1f}  {'yes' if ok else 'NO'}"
        )
        if ok and hw:
            xs.append(f)
            ys.append(h_mean)
        elif not ok:
            failures.append((h, w, "output does not match torch golden"))

    if failures:
        print("\nEXCLUDED FROM THE FIT")
        for h, w, why in failures:
            print(f"  {h}x{w}: {why}")

    if len(xs) < 2:
        print("\nnot enough verified shapes to fit")
        return

    intercept, slope, r2 = fit_line(xs, ys)
    marginal_gops = 1e-6 / slope if slope > 0 else float("nan")
    print("\nFIT  hw_time_ms = intercept + slope * FLOPs   (verified shapes only)")
    print(f"  points        : {len(xs)}")
    print(f"  intercept     : {intercept*1e3:.1f} us   fixed cost in the hw bracket")
    print(f"  marginal GOPS : {marginal_gops:.1f}   steady-state array throughput")
    print(f"  r^2           : {r2:.5f}")
    print("\nCompare marginal GOPS against the CPU's measured int8 rate from")
    print("kernels/conv2x_baseline/cpu_baseline.py --single (resnet_env, same sitting).")


if __name__ == "__main__":
    main()
