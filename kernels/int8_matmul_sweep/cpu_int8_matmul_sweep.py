#!/usr/bin/env python3
"""CPU side of the int8 GEMM sweep: torch._int_mm, ORT MatMulInteger, torch bf16/fp32.

WHY THIS EXISTS
---------------
`kernels/bf16_matmul_sweep/cpu_matmul_sweep.py` gave this repo its first CPU bf16/fp32
GEMM baseline. There is still no CPU *int8* GEMM baseline -- every CPU int8 number here
is ORT QDQ conv (MLAS) -- so an NPU int8 GEMM figure would have nothing like-for-like to
be read against. This script is that baseline, and it deliberately reports two CPU int8
kernels rather than one, because this repo has twice been wrong by picking the slower CPU
kernel (`results/aie/conv2x_int8_cpu_baseline.log`, the MobileViT splice):

  torch._int_mm       int8 x int8 -> int32, torch's own CPU int8 GEMM (oneDNN/VNNI path)
  ORT MatMulInteger   uint8 x int8 -> int32, MLAS -- the library behind every other CPU
                      int8 number in this repo (QLinearConv/QLinearMatMul use it too)

The faster of the two is the verdict line; both are printed. torch bf16 and fp32 matmul
are timed in the same sitting so the int8-vs-bf16 ratio on the CPU side comes from one
run, not from a table measured hours earlier.

Both int8 outputs are checked exactly: against an int64 reference when M*K*N <= 2^30,
and against each other at every shape (the uint8 operand is the int8 one plus 128, so
ORT's result minus 128 * colsum(B) must equal torch's, element for element).

USAGE (resnet_env -- has torch, onnx, onnxruntime; NOT ironenv)
    conda activate resnet_env
    python kernels/int8_matmul_sweep/cpu_int8_matmul_sweep.py \\
        --shapes 512x512x512,2048x2048x2048 --iters 20 --warmup 5

Run in the same sitting as npu_matmul_sweep.py (ironenv).
"""

import argparse
import platform
import statistics
import time

import numpy as np
import torch

torch.manual_seed(0)


def cpu_name():
    try:
        import winreg

        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(times), min(times)


def ort_session(M, K, N):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    A = helper.make_tensor_value_info("A", TensorProto.UINT8, [M, K])
    B = helper.make_tensor_value_info("B", TensorProto.INT8, [K, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [M, N])
    node = helper.make_node("MatMulInteger", ["A", "B"], ["Y"])
    g = helper.make_graph([node], "mmi", [A, B], [Y])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess, so


def one_shape(M, K, N, iters, warmup, verify_limit):
    ops = 2.0 * M * K * N
    a8 = torch.randint(-128, 128, (M, K), dtype=torch.int8)
    b8 = torch.randint(-128, 128, (K, N), dtype=torch.int8)

    # torch int8 GEMM
    t_mean, t_min = timeit(lambda: torch._int_mm(a8, b8), iters, warmup)
    y_torch = torch._int_mm(a8, b8)

    # ORT MatMulInteger u8s8, A shifted by +128 into uint8 (a zero-point-128 activation)
    sess, so = ort_session(M, K, N)
    au = (a8.to(torch.int16) + 128).to(torch.uint8).numpy()
    bs = b8.numpy()
    feed = {"A": au, "B": bs}
    o_mean, o_min = timeit(lambda: sess.run(None, feed), iters, warmup)
    y_ort = sess.run(None, feed)[0]

    # Exact checks
    colsum = b8.to(torch.int64).sum(dim=0)  # (N,)
    cross = bool(((torch.from_numpy(y_ort).to(torch.int64) - 128 * colsum) == y_torch.to(torch.int64)).all())
    if M * K * N <= verify_limit:
        ref = a8.to(torch.int64) @ b8.to(torch.int64)
        exact_torch = bool((y_torch.to(torch.int64) == ref).all())
        exact_ort = bool((torch.from_numpy(y_ort).to(torch.int64) == (a8.to(torch.int64) + 128) @ b8.to(torch.int64)).all())
        check = f"int64-ref torch={'OK' if exact_torch else 'MISMATCH'} ort={'OK' if exact_ort else 'MISMATCH'} cross={'OK' if cross else 'MISMATCH'}"
    else:
        check = f"cross-check torch-vs-ort={'OK' if cross else 'MISMATCH'} (int64 ref skipped, > 2^30 MACs)"

    # torch bf16 / fp32 for the same-sitting reference
    af = torch.randn(M, K, dtype=torch.float32)
    bf = torch.randn(K, N, dtype=torch.float32)
    ab, bb = af.to(torch.bfloat16), bf.to(torch.bfloat16)
    bf_mean, bf_min = timeit(lambda: torch.matmul(ab, bb), iters, warmup)
    f_mean, f_min = timeit(lambda: torch.matmul(af, bf), iters, warmup)

    return {
        "shape": f"{M}x{K}x{N}",
        "ops": ops,
        "torch_i8": (t_mean, t_min, ops / (t_mean * 1e6)),
        "ort_u8s8": (o_mean, o_min, ops / (o_mean * 1e6)),
        "bf16": (bf_mean, bf_min, ops / (bf_mean * 1e6)),
        "fp32": (f_mean, f_min, ops / (f_mean * 1e6)),
        "check": check,
        "ort_threads": so.intra_op_num_threads,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--shapes", type=str, required=True, help="comma-separated MxKxN list")
    p.add_argument("--verify-limit", type=int, default=2**30, help="max M*K*N for the int64 reference check")
    args = p.parse_args()

    import onnxruntime as ort

    shapes = []
    for tok in args.shapes.split(","):
        m, k, n = tok.lower().strip().split("x")
        shapes.append((int(m), int(k), int(n)))

    print("CPU int8 GEMM sweep -- torch._int_mm vs ORT MatMulInteger (u8s8), plus torch bf16/fp32")
    print(f"CPU: {cpu_name()}")
    print(f"torch {torch.__version__} (threads {torch.get_num_threads()}), onnxruntime {ort.__version__} "
          f"(intra_op_num_threads 0 = ORT default), numpy {np.__version__}")
    print(f"iters={args.iters} warmup={args.warmup}; GOPS/GFLOPS = 2MKN / mean wall ms\n")

    head = (f"{'MxKxN':>16} {'GOP':>8} | {'torch _int_mm ms':>17} {'GOPS':>8} | {'ORT u8s8 ms':>13} {'GOPS':>8} | "
            f"{'best i8':>8} | {'bf16 ms':>9} {'GFLOPS':>8} | {'fp32 GFLOPS':>11} | {'i8/bf16':>7}")
    print(head)
    print("-" * len(head))
    rows = []
    for M, K, N in shapes:
        r = one_shape(M, K, N, args.iters, args.warmup, args.verify_limit)
        rows.append(r)
        ti, oi, bfr, fr = r["torch_i8"], r["ort_u8s8"], r["bf16"], r["fp32"]
        best = max(ti[2], oi[2])
        print(f"{r['shape']:>16} {r['ops']/1e9:8.2f} | {ti[0]:9.3f}/{ti[1]:7.3f} {ti[2]:8.1f} | "
              f"{oi[0]:6.2f}/{oi[1]:6.2f} {oi[2]:8.1f} | {best:8.1f} | {bfr[0]:9.3f} {bfr[2]:8.1f} | "
              f"{fr[2]:11.1f} | {best/bfr[2]:7.2f}")
        print(f"{'':>16} {'':>8}   {r['check']}")
    print("\n(ms columns are mean/min over the timed iterations; 'best i8' is the faster of the two int8 kernels)")


if __name__ == "__main__":
    main()
