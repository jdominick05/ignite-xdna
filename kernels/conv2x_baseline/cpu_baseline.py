#!/usr/bin/env python3
"""CPU baseline for mlir-aie's ml/resnet/layers_conv2_x, the chained int8 design.

WHY THIS EXISTS
---------------
`ml/resnet/layers_conv2_x` — three ResNet bottleneck blocks chained core-to-core across
three columns, int8, ObjectFifo to ObjectFifo, ONE dispatch for the whole chain — already
runs on this machine and PASSes at 1888.5 us NPU time
(`results/aie/mlir_aie_magika_mobilenet_npu.log:18`). Nobody ever measured what the CPU
does the same work in, so "is it faster than CPU" has never been answered for the one
design on this hardware that is both big enough to amortize the dispatch floor and built
on the real int8 MAC path.

That matters because the two hand-written kernels in this repo both lost, and their shared
flaw was building before measuring the primitive at the target shape
(`results/aie/dispatch_floor_npu.log`). This measures first.

WHAT IS BEING COMPARED
----------------------
The exact network in the design's own torch reference (its test.py): input 1x64x32x32,
stride 1 and 3x3 padding 1 throughout, so spatial stays 32x32.

    shortcut   64 -> 256  1x1
    block 0:   64 -> 64   1x1  |  64 -> 64  3x3  |  64 -> 256  1x1   (+ skip)
    block 1:  256 -> 64   1x1  |  64 -> 64  3x3  |  64 -> 256  1x1   (+ skip)
    block 2:  256 -> 64   1x1  |  64 -> 64  3x3  |  64 -> 256  1x1   (+ skip)

TWO CPU LINES, DELIBERATELY
---------------------------
The splice work already established that the CPU kernel choice, not the NPU, can decide a
verdict (numpy vs torch attention was 9x, `results/mobilevit/splice_wall_clock_npu.log`).
So this reports both and lets the reader see the spread:

  torch fp32   -- what you would actually run if you never quantized.
  ORT CPU int8 -- QDQ int8 under the ONNX Runtime CPU EP, which reaches VNNI and is the
                  like-for-like comparison against an int8 NPU design. This is also the
                  line that matches this repo's whole XINT8 thesis.

Neither is "the" baseline on its own. Report both, and say which one a claim uses.

ENVIRONMENTS (they differ, and that is not optional here)
--------------------------------------------------------
torch lives in the mlir-aie ironenv (py3.13); onnxruntime lives in resnet_env17 (py3.12).
pyxrt.pyd hard-links python313.dll with no abi3 marker, so it cannot be imported in
resnet_env17 at all — a measured fact, not a guess. Hence the modes:

    # in ironenv  (has torch)
    python3 kernels/conv2x_baseline/cpu_baseline.py --mode torch --export models/conv2x_fp32.onnx
    # in resnet_env17  (has onnxruntime)
    python  kernels/conv2x_baseline/cpu_baseline.py --mode ort --onnx models/conv2x_fp32.onnx

Capture the NPU number in the same sitting as these — NPU latency drifts between sessions
on this machine independently of any code change.
"""

import argparse
import statistics
import sys
import time

import numpy as np

# 1x64x32x32 in, stride 1, 3x3 padded to keep 32x32. FLOPs = 2*H*W*Cin*Cout*k*k.
HW = 32 * 32
CONVS = [
    ("shortcut", 64, 256, 1),
    ("b0_conv1", 64, 64, 1),
    ("b0_conv2", 64, 64, 3),
    ("b0_conv3", 64, 256, 1),
    ("b1_conv1", 256, 64, 1),
    ("b1_conv2", 64, 64, 3),
    ("b1_conv3", 64, 256, 1),
    ("b2_conv1", 256, 64, 1),
    ("b2_conv2", 64, 64, 3),
    ("b2_conv3", 64, 256, 1),
]


def flop_table():
    rows, total = [], 0
    for name, cin, cout, k in CONVS:
        f = 2 * HW * cin * cout * k * k
        rows.append((name, cin, cout, k, f))
        total += f
    return rows, total


def build_torch_model():
    import torch.nn as nn

    class Conv2x(nn.Module):
        """The design's three chained bottlenecks, fp32, plain (no quant emulation).

        The reference test.py interleaves round/clamp scaling to emulate int8. That is
        for numerical verification, not latency -- emulating quantization in fp32 would
        make the CPU line slower than any CPU you would actually deploy, which would
        flatter the NPU. Structure is identical; only the fake-quant ops are dropped.
        """

        def __init__(self, in_planes=64, planes=64, expansion=4):
            super().__init__()
            oc = expansion * planes
            self.shortcut = nn.Conv2d(in_planes, oc, 1, bias=False)
            self.b0 = nn.ModuleList([
                nn.Conv2d(in_planes, planes, 1, bias=False),
                nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                nn.Conv2d(planes, oc, 1, bias=False),
            ])
            self.b1 = nn.ModuleList([
                nn.Conv2d(oc, planes, 1, bias=False),
                nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                nn.Conv2d(planes, oc, 1, bias=False),
            ])
            self.b2 = nn.ModuleList([
                nn.Conv2d(oc, planes, 1, bias=False),
                nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                nn.Conv2d(planes, oc, 1, bias=False),
            ])
            self.relu = nn.ReLU()

        def _block(self, x, convs, skip):
            y = self.relu(convs[0](x))
            y = self.relu(convs[1](y))
            y = convs[2](y)
            return self.relu(y + skip)

        def forward(self, x):
            out = self._block(x, self.b0, self.shortcut(x))
            out = self._block(out, self.b1, out)
            out = self._block(out, self.b2, out)
            return out

    return Conv2x()


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def summarize(label, ts, total_flop):
    mean = statistics.mean(ts)
    print(
        f"  {label:<28} mean {mean:8.3f} ms   median {statistics.median(ts):8.3f}   "
        f"min {min(ts):8.3f}   -> {total_flop / (mean * 1e-3) / 1e9:7.2f} GFLOPS"
    )
    return mean


def mode_torch(args, total_flop):
    import torch

    torch.manual_seed(0)
    model = build_torch_model().eval()
    x = torch.randn(1, 64, 32, 32)

    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}\n")
    with torch.inference_mode():
        ts = timeit(lambda: model(x), args.iters, args.warmup)
        summarize("torch fp32 (CPU)", ts, total_flop)

    if args.export:
        # Repo invariant: static batch 1, no dynamic_axes, opset 17, dynamo=False.
        torch.onnx.export(
            model,
            x,
            args.export,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
        print(f"\nexported fp32 ONNX -> {args.export}")
        print("Now run --mode ort in resnet_env17 for the int8 CPU line.")


def mode_quant(args):
    """Static QDQ int8 for the CPU VNNI line. LATENCY ONLY -- accuracy is irrelevant here.

    Calibration is random data, which would be indefensible for an accuracy claim and is
    fine for a latency one: the QDQ graph's shape and the int8 kernels ORT selects do not
    depend on what the scales happen to be. Nothing from this path is ever used for a
    top-1 number, and it is not Quark -- it is ORT's own quantizer, because the question
    is what the CPU does with int8, not what the NPU would.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class RandomCalib(CalibrationDataReader):
        def __init__(self, name, n=16):
            self.data = iter(
                [{name: np.random.randn(1, 64, 32, 32).astype(np.float32)} for _ in range(n)]
            )

        def get_next(self):
            return next(self.data, None)

    import onnxruntime as ort

    name = ort.InferenceSession(
        args.onnx, providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    quantize_static(
        args.onnx,
        args.out,
        RandomCalib(name),
        quant_format=__import__(
            "onnxruntime.quantization", fromlist=["QuantFormat"]
        ).QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"QDQ int8 written -> {args.out}  (random calibration; latency use only)")


def mode_ort(args, total_flop):
    import onnxruntime as ort

    x = np.random.randn(1, 64, 32, 32).astype(np.float32)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"onnxruntime {ort.__version__}\n")

    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ts = timeit(lambda: sess.run(None, {name: x}), args.iters, args.warmup)
    summarize("ORT CPU EP, fp32", ts, total_flop)

    if args.int8:
        sess8 = ort.InferenceSession(args.int8, so, providers=["CPUExecutionProvider"])
        n8 = sess8.get_inputs()[0].name
        ts8 = timeit(lambda: sess8.run(None, {n8: x}), args.iters, args.warmup)
        summarize("ORT CPU EP, QDQ int8", ts8, total_flop)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("torch", "ort", "quant", "flops"), default="flops")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--export", default=None, help="torch mode: write fp32 ONNX here")
    p.add_argument("--onnx", default=None, help="ort mode: fp32 ONNX to run")
    p.add_argument("--int8", default=None, help="ort mode: QDQ int8 ONNX to run")
    p.add_argument("--out", default=None, help="quant mode: QDQ int8 output path")
    args = p.parse_args()

    rows, total = flop_table()
    print("mlir-aie ml/resnet/layers_conv2_x -- CPU baseline")
    print("3 chained ResNet bottlenecks, 1x64x32x32, stride 1, spatial 32x32 throughout\n")
    print(f"  {'conv':<10} {'Cin':>4} {'Cout':>5} {'k':>2}   {'MFLOP':>8}")
    for name, cin, cout, k, f in rows:
        print(f"  {name:<10} {cin:>4} {cout:>5} {k:>2}   {f / 1e6:>8.2f}")
    print(f"  {'TOTAL':<10} {'':>4} {'':>5} {'':>2}   {total / 1e6:>8.2f} MFLOP\n")

    if args.mode == "flops":
        print("(--mode torch | ort to measure; see the module docstring for envs)")
        return
    if args.mode == "quant":
        if not (args.onnx and args.out):
            sys.exit("--mode quant needs --onnx and --out")
        mode_quant(args)
        return
    if args.mode == "torch":
        mode_torch(args, total)
    else:
        if not args.onnx:
            sys.exit("--mode ort needs --onnx")
        mode_ort(args, total)


if __name__ == "__main__":
    main()
