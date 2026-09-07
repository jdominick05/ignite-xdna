"""Microbenchmark for CPU threading in Quark AdaRound FastFinetune.

Compares three configurations:
1. 1 pinned core: process pinned to Core 0 (mask 0x1), 1 thread
2. 4 real cores: process pinned to 4 physical cores (mask 0x55: cores 0,2,4,6), 4 threads
3. 16 threads: unconstrained all logical cores (mask 0xFFFF), 16 threads

Usage:
    conda activate resnet_env
    python tools/bench_quant_threads.py
"""

import ctypes
from ctypes import wintypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Win32 affinity setup
k32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
if k32:
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL

CONFIGS = [
    {
        "name": "1 pinned core",
        "mask": 0x1,
        "threads": 1,
        "desc": "Pinned to physical Core 0 (mask 0x1), 1 thread",
    },
    {
        "name": "4 real cores",
        "mask": 0x55,  # 0b01010101 -> cores 0, 2, 4, 6 (physical Zen 4 cores without SMT siblings)
        "threads": 4,
        "desc": "Pinned to 4 physical cores (mask 0x55), 4 threads",
    },
    {
        "name": "8 real cores",
        "mask": 0x5555,  # 0b0101010101010101 -> cores 0, 2, 4, 6, 8, 10, 12, 14 (all 8 physical Zen 4 cores without SMT siblings)
        "threads": 8,
        "desc": "Pinned to 8 physical cores (mask 0x5555), 8 threads",
    },
    {
        "name": "16 threads",
        "mask": 0xFFFF,  # all 16 logical cores
        "threads": 16,
        "desc": "Unconstrained 16 logical threads (mask 0xFFFF, default behavior)",
    },
]


def run_one(cfg, in_model, cfg_path, iters=100, limit=32):
    name = cfg["name"]
    mask = cfg["mask"]
    threads = cfg["threads"]
    out_model = REPO_ROOT / f"scratch/bench_threads_{threads}.onnx"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "pipelines/resnet50/3_quantize.py"),
        "--calib-dir", str(REPO_ROOT / "data/calib"),
        "--config", "XINT8_ADAROUND",
        "--in-model", str(in_model),
        "--cfg-path", str(cfg_path),
        "--limit", str(limit),
        "--iters", str(iters),
        "--threads", str(threads),
        "--out", str(out_model),
    ]

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_WAIT_POLICY"] = "PASSIVE"

    print(f"\n=======================================================")
    print(f"Running: {name} (threads={threads}, mask=0x{mask:X})")
    print(f"=======================================================")

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if k32 and mask:
        ret = k32.SetProcessAffinityMask(proc._handle, mask)
        if not ret:
            print(f"WARNING: SetProcessAffinityMask failed with err {k32.GetLastError()}")
        else:
            print(f"Set process affinity mask to 0x{mask:X}")

    stdout, _ = proc.communicate()
    t1 = time.perf_counter()
    wall_clock = t1 - t0

    if proc.returncode != 0:
        print(f"ERROR: process failed with return code {proc.returncode}")
        print(stdout[-1000:])
        return None

    # Parse Quark latency profiler outputs
    torch_time = None
    onnx_time = None
    fast_ft_time = None
    e2e_time = None

    m = re.search(r"ONNX inference costs ([\d\.]+)s and Torch training costs ([\d\.]+)s", stdout)
    if m:
        onnx_time = float(m.group(1))
        torch_time = float(m.group(2))

    m = re.search(r"The FastFinetune algorithm has been applied\. It took ([\d\.]+)s to complete\.", stdout)
    if m:
        fast_ft_time = float(m.group(1))

    m = re.search(r"Quark_latency_profiler: e2e quantization time consumed:([\d\.]+)", stdout)
    if m:
        e2e_time = float(m.group(1))

    # Clean up output model
    if out_model.exists():
        try:
            out_model.unlink()
        except OSError:
            pass

    return {
        "name": name,
        "threads": threads,
        "mask": f"0x{mask:X}",
        "wall_clock": wall_clock,
        "torch_time": torch_time,
        "onnx_time": onnx_time,
        "fast_ft_time": fast_ft_time,
        "e2e_time": e2e_time,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["1", "4", "8", "16"], default=None,
                    help="run only this thread configuration")
    cli_args = ap.parse_args()

    in_model = REPO_ROOT / "models/mobilenetv2_fp32.onnx"
    cfg_path = REPO_ROOT / "models/preprocess_config_mobilenetv2.json"

    if not in_model.exists():
        raise SystemExit(f"Missing {in_model}")
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}")

    configs = [c for c in CONFIGS if str(c["threads"]) == cli_args.only] if cli_args.only else CONFIGS

    print("Benchmarking Quark AdaRound FastFinetune CPU Thread Scaling")
    print(f"Model: {in_model.name} (52 Conv layers)")
    print(f"Calibration: 32 images, 100 iterations/layer, batch_size=2")

    results = []
    for cfg in configs:
        res = run_one(cfg, in_model, cfg_path, iters=100, limit=32)
        if res:
            results.append(res)

    print("\n" + "=" * 80)
    print("                      THREAD BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Configuration':<18} {'Mask':<8} {'Threads':<8} {'FastFT (s)':<12} {'Torch (s)':<12} {'ONNX (s)':<10} {'Wall (s)':<10} {'Speedup vs 16T':<14}"
    print(header)
    print("-" * 80)

    base_time = None
    for r in results:
        if r["threads"] == 16:
            base_time = r["fast_ft_time"] or r["wall_clock"]
            break

    for r in results:
        t = r["fast_ft_time"] or r["wall_clock"]
        speedup_str = f"{base_time / t:.2f}x" if base_time and t else "—"
        ft_str = f"{r['fast_ft_time']:.1f}s" if r["fast_ft_time"] else "—"
        torch_str = f"{r['torch_time']:.1f}s" if r["torch_time"] else "—"
        onnx_str = f"{r['onnx_time']:.1f}s" if r["onnx_time"] else "—"
        wall_str = f"{r['wall_clock']:.1f}s"
        print(f"{r['name']:<18} {r['mask']:<8} {r['threads']:<8} {ft_str:<12} {torch_str:<12} {onnx_str:<10} {wall_str:<10} {speedup_str:<14}")
    print("=" * 80)


if __name__ == "__main__":
    main()
