#!/usr/bin/env python3
"""CPU side of the ResNet-bottleneck spatial sweep: ORT QDQ int8, same shapes.

WHY THIS EXISTS
---------------
`kernels/bottleneck_sweep/sweep.py` measures one mlir-aie int8 bottleneck on the NPU
across spatial sizes and fits its marginal GOPS. A marginal-GOPS number alone decides
nothing -- it has to be read against what the CPU does with the same arithmetic at the
same shapes, because the CPU also gets more efficient as the problem grows (bigger GEMM
tiles, better cache reuse, less per-call ORT overhead). Comparing an NPU number that
scales against a CPU number frozen at 32x32 would manufacture a crossover that isn't
there.

WHAT IS BEING COMPARED
----------------------
Exactly one bottleneck, matching the AIE design (no downsample shortcut -- the skip is
identity, so Cin == Cout == 256):

    conv1  256 -> 64   1x1
    conv2   64 -> 64   3x3, padding 1
    conv3   64 -> 256  1x1
    out = relu(conv3 + x)

Only the ORT QDQ int8 line is reported. `results/aie/conv2x_int8_cpu_baseline.log`
already established why: torch fp32 landed within 1% of the NPU and would have read as
parity, while the like-for-like int8 line (which reaches VNNI) was 6.3x faster. int8 vs
int8 is the honest comparison and the only one this sweep needs.

Calibration is random data. That is indefensible for an accuracy claim and fine for a
latency one -- the QDQ graph shape and the int8 kernels ORT selects do not depend on what
the scales happen to be, and nothing here is ever used for an accuracy number.

ENVIRONMENT
-----------
resnet_env has torch, onnx and onnxruntime, so the whole thing runs in one activation:

    conda activate resnet_env
    python kernels/bottleneck_sweep/cpu_sweep.py --workdir scratch/bottleneck_sweep

Do NOT write the intermediate ONNX into models/ -- that directory is a Syncthing folder
shared with two other machines (see CLAUDE.md). scratch/ is git-ignored and local.

Run this in the same sitting as the NPU sweep: NPU latency on this machine drifts
between sessions independently of any code change.
"""

import argparse
import os
import statistics
import time

import numpy as np

CHANNELS = 256
MID = 64


def flops(h, w):
    """MACs*2 for the three convs -- the identical formula sweep.py uses."""
    macs = h * w * (CHANNELS * MID + 9 * MID * MID + MID * CHANNELS)
    return 2 * macs


def build_model():
    import torch.nn as nn

    class Bottleneck(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(CHANNELS, MID, 1, bias=False)
            self.conv2 = nn.Conv2d(MID, MID, 3, padding=1, bias=False)
            self.conv3 = nn.Conv2d(MID, CHANNELS, 1, bias=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            y = self.relu(self.conv1(x))
            y = self.relu(self.conv2(y))
            return self.relu(self.conv3(y) + x)

    return Bottleneck().eval()


def export_fp32(path, h, w):
    import torch

    torch.manual_seed(0)
    model = build_model()
    x = torch.randn(1, CHANNELS, h, w)
    # Repo invariant: static batch 1, no dynamic_axes, opset 17, dynamo=False.
    torch.onnx.export(
        model,
        x,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )


def quantize(fp32_path, int8_path, h, w):
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class RandomCalib(CalibrationDataReader):
        def __init__(self, n=16):
            self.data = iter(
                [
                    {"input": np.random.randn(1, CHANNELS, h, w).astype(np.float32)}
                    for _ in range(n)
                ]
            )

        def get_next(self):
            return next(self.data, None)

    quantize_static(
        fp32_path,
        int8_path,
        RandomCalib(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def run(int8_path, h, w, iters, warmup):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(int8_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, CHANNELS, h, w).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


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
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--shapes",
        type=str,
        default="32x32,64x32,128x32,256x32,512x32,56x56,32x64,64x64,128x64",
        help="comma-separated HxW list; keep identical to sweep.py's",
    )
    p.add_argument(
        "--workdir",
        default=os.path.join("scratch", "bottleneck_sweep"),
        help="where the intermediate ONNX goes -- never models/ (Syncthing)",
    )
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    shapes = []
    for tok in args.shapes.split(","):
        hs, ws = tok.lower().split("x")
        shapes.append((int(hs), int(ws)))

    import onnxruntime as ort

    print("ResNet bottleneck (256-64-64-256 + identity skip) -- CPU spatial sweep")
    print(f"onnxruntime {ort.__version__}, ORT CPU EP, QDQ int8 (VNNI)")
    print(f"iters={args.iters} warmup={args.warmup}\n")
    print(f"{'HxW':>9} {'MFLOP':>9} {'int8 ms':>9} {'min ms':>9} {'GOPS':>9}")
    print("-" * 50)

    xs, ys = [], []
    for h, w in shapes:
        f = flops(h, w)
        fp32 = os.path.join(args.workdir, f"bneck_{h}x{w}_fp32.onnx")
        int8 = os.path.join(args.workdir, f"bneck_{h}x{w}_int8.onnx")
        export_fp32(fp32, h, w)
        quantize(fp32, int8, h, w)
        ts = run(int8, h, w, args.iters, args.warmup)
        mean = statistics.mean(ts)
        print(
            f"{f'{h}x{w}':>9} {f/1e6:>9.2f} {mean:>9.4f} {min(ts):>9.4f} "
            f"{f/(mean*1e6):>9.1f}"
        )
        xs.append(f)
        ys.append(mean)

    intercept, slope, r2 = fit_line(xs, ys)
    print("\nFIT  cpu_time_ms = intercept + slope * FLOPs")
    print(f"  intercept     : {intercept*1e3:.1f} us")
    print(f"  marginal GOPS : {1e-6/slope:.1f}")
    print(f"  r^2           : {r2:.5f}")


if __name__ == "__main__":
    main()
