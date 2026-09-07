#!/usr/bin/env python3
"""NPU side of the int8-vs-bf16 GEMM sweep: drives mlir-aie's whole_array.py per shape.

WHY THIS EXISTS
---------------
This project is named for int8 (Quark XINT8, the only dtype the VitisAI X1 backend
takes), and XDNA1's headline TOPS figure is an int8 figure -- yet every int8 number
under kernels/ came from the chained conv design (`ml/bottleneck`, `layers_conv2_x`),
whose two kernels each carried a width-32 bug and which lost to the CPU 5.7-12.75x. The
one NPU *win* in this repo is bf16 GEMM (`bf16_matmul_sweep/`). Nobody had run the same
upstream `whole_array` GEMM design in int8, so "is int8 GEMM this chip's real strength,
or is bf16?" had no measurement behind it. This driver runs that sweep. It is a sweep of
the upstream design (unmodified, `--dtype_in i8 --dtype_out i32` and
`--dtype_in bf16 --dtype_out f32`), not a hand-written kernel -- the same footing as
`bf16_matmul_sweep/`.

WHAT IT DOES
------------
For each `MxKxN[@m]` row it runs whole_array.py once (JIT compile on first use of a
shape, then `--warmup`/`--iters` timed runs), parses the script's own benchmark tail
(NPU-time bracket around `kernel.wait()`, end-to-end bracket around the Python call, the
GFLOPS it derives from the NPU bracket, and its PASS!/FAIL! verdict against numpy A@B --
exact `np.array_equal` for integer dtypes, a 5%/0.5 tolerance for float), prints the raw
tail per run so the log stays traceable, and ends with one table. A FAIL or a compile
error is recorded and the sweep continues.

Tile row size `m` defaults to min(64, M // 8): whole_array needs M % (m*4) == 0 and
(M/m/4) % 2 == 0, so M=128 forces m=16 and M=256 forces m=32. An explicit `@m` on a row
overrides that, which is how the m-tile control rows are run (the real-shape log,
results/aie/bf16_matmul_ffn_real_shape_npu.log, showed m=16 alone halves throughput, so
any small-M row has to be read against a same-M row at a different m).

USAGE (ironenv -- NOT resnet_env / resnet_env17)
------------------------------------------------
    . C:\\Users\\<user>\\mlir-aie\\iron_env.ps1
    $env:PATH = "C:\\Xilinx\\XRT\\xrt_sdk\\xrt;$env:PATH"
    python kernels/int8_matmul_sweep/npu_matmul_sweep.py --dtype i8 \\
        --shapes 512x512x512,2048x2048x2048,512x4096x4096@16 --iters 10 --warmup 3

Pair it with cpu_int8_matmul_sweep.py (resnet_env) in the same sitting -- NPU latency on
this machine drifts between sessions independently of any code change.
"""

import argparse
import os
import re
import subprocess
import sys
import time

DEFAULT_WHOLE_ARRAY_DIR = os.path.join(
    os.path.expanduser("~"),
    "mlir-aie",
    "programming_examples",
    "basic",
    "matrix_multiplication",
    "whole_array",
)

DTYPES = {
    # --dtype -> (dtype_in, dtype_out). i32 out because the design's K-reduction
    # accumulates in a dtype_out-typed buffer (see bf16_matmul_k_limit_diagnosed_npu.log);
    # i16 would wrap and i8 would saturate after the first k-tile.
    "i8": ("i8", "i32"),
    "bf16": ("bf16", "f32"),
    "i16": ("i16", "i32"),
}

RE_NPU = re.compile(r"NPU time\s+\(avg/min/max us\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)")
RE_E2E = re.compile(r"End-to-end\s+\(avg/min/max us\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)")
RE_GFLOPS = re.compile(r"NPU GFLOPS\s*:\s*([\d.]+)")


def parse_rows(spec):
    rows = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m_override = None
        if "@" in tok:
            tok, m_str = tok.split("@", 1)
            m_override = int(m_str)
        M, K, N = (int(x) for x in tok.lower().split("x"))
        rows.append((M, K, N, m_override))
    return rows


def default_m(M):
    # M % (m*4) == 0 and (M/m/4) % 2 == 0  ->  m divides M/8; cap at the design default.
    m = 64
    while m > 4 and (M % (m * 8)) != 0:
        m //= 2
    return m


def run_one(args, M, K, N, m):
    dtype_in, dtype_out = DTYPES[args.dtype]
    if args.dtype_out:
        dtype_out = args.dtype_out
    cmd = [
        sys.executable,
        "whole_array.py",
        "--dev",
        "npu",
        "-M", str(M), "-K", str(K), "-N", str(N),
        "-m", str(m), "-k", str(args.k), "-n", str(args.n),
        "--n-aie-cols", str(args.cols),
        "--dtype_in", dtype_in,
        "--dtype_out", dtype_out,
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]
    print(f"\n=== {M}x{K}x{N}  m={m} k={args.k} n={args.n}  {dtype_in}->{dtype_out} ===")
    print("$ " + " ".join(cmd[1:]))
    sys.stdout.flush()
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            cwd=args.whole_array_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = "TIMEOUT"
    wall = time.perf_counter() - t0

    npu = RE_NPU.search(out)
    e2e = RE_E2E.search(out)
    gf = RE_GFLOPS.search(out)
    passed = "PASS!" in out
    failed = "FAIL!" in out or "FAIL!" in err
    # Raw tail: the benchmark block from stdout, plus stderr's last lines on failure.
    tail_lines = [ln for ln in out.splitlines() if ln.strip()][-8:]
    print("--- stdout tail ---")
    for ln in tail_lines:
        print("  " + ln)
    if rc != 0 or failed or not passed:
        err_lines = [ln for ln in err.splitlines() if ln.strip()][-12:]
        print(f"--- stderr tail (rc={rc}) ---")
        for ln in err_lines:
            print("  " + ln[:300])
    print(f"(wall incl. compile: {wall:.1f} s)")
    sys.stdout.flush()

    verdict = "PASS" if (passed and not failed and rc == 0) else ("FAIL" if failed else f"ERROR(rc={rc})")
    return {
        "shape": f"{M}x{K}x{N}",
        "m": m,
        "verdict": verdict,
        "npu_avg": float(npu.group(1)) if npu else None,
        "npu_min": float(npu.group(2)) if npu else None,
        "npu_max": float(npu.group(3)) if npu else None,
        "e2e_avg": float(e2e.group(1)) if e2e else None,
        "gops": float(gf.group(1)) if gf else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dtype", choices=sorted(DTYPES), default="i8")
    p.add_argument("--dtype-out", default=None, help="override the output dtype (default per --dtype)")
    p.add_argument("--shapes", required=True, help="comma-separated MxKxN[@m] rows")
    p.add_argument("-k", type=int, default=64, help="tile k (whole_array default 64)")
    p.add_argument("-n", type=int, default=32, help="tile n (whole_array default 32)")
    p.add_argument("--cols", type=int, default=4, help="--n-aie-cols (4 = the whole Phoenix array)")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1500, help="seconds per run, compile included")
    p.add_argument("--whole-array-dir", default=DEFAULT_WHOLE_ARRAY_DIR)
    args = p.parse_args()

    if not os.path.isfile(os.path.join(args.whole_array_dir, "whole_array.py")):
        sys.exit(f"whole_array.py not found in {args.whole_array_dir}")

    rows = parse_rows(args.shapes)
    dtype_in, dtype_out = DTYPES[args.dtype]
    if args.dtype_out:
        dtype_out = args.dtype_out
    print(f"whole_array sweep  dtype {dtype_in}->{dtype_out}  cols={args.cols} k={args.k} n={args.n}  "
          f"iters={args.iters} warmup={args.warmup}  python={sys.executable}")
    print(f"whole_array dir: {args.whole_array_dir}")

    results = []
    for M, K, N, m_override in rows:
        m = m_override if m_override is not None else default_m(M)
        results.append(run_one(args, M, K, N, m))

    unit = "GOPS" if dtype_in.startswith("i") else "GFLOPS"
    print(f"\nSUMMARY  ({dtype_in}->{dtype_out}, {args.cols} cols, k={args.k} n={args.n}; "
          f"{unit} = 2MKN / NPU-bracket avg)")
    head = f"{'MxKxN':>16} {'m':>3} {'verdict':>10} {'NPU avg us':>12} {'NPU min us':>12} {'e2e avg us':>12} {unit:>9}"
    print(head)
    print("-" * len(head))
    for r in results:
        fmt = lambda v: f"{v:12.1f}" if v is not None else f"{'-':>12}"
        g = f"{r['gops']:9.2f}" if r["gops"] is not None else f"{'-':>9}"
        print(f"{r['shape']:>16} {r['m']:>3} {r['verdict']:>10} {fmt(r['npu_avg'])} {fmt(r['npu_min'])} {fmt(r['e2e_avg'])} {g}")


if __name__ == "__main__":
    main()
