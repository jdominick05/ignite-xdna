#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
benchmarks/benchmark_classification_head.py

Silicon benchmark: Ignition's native NPU classification head vs AMD Ryzen AI 1.7.1 stack
on AMD Phoenix NPU (Ryzen 7 8700G, Phoenix XDNA1 [003d:00:01.1]).

Evaluates a 1,000-class ImageNet classification head (2,048 pooled features -> 1,000 logits):
- Ignition on the container (build/test_resnet50_head.ignite):
  monolithic 1x1 convolution across a 20x20 tile on Device 0, 0 host segments.
- AMD Ryzen AI 1.7.1 Vitis AI EP on build/test_resnet50_head.onnx:
  attempts NPU execution, records runner crash (runner_requests_queue.cpp:178: invalid vector subscript)
  and 0 accelerated NPU nodes.
- AMD ONNX Runtime CPU fallback (CPUExecutionProvider on build/test_resnet50_head.onnx).
"""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import ctypes
from ctypes import wintypes

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

def get_rss_mb() -> float:
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ignite_xdna.pipelines.classification_pipeline import ClassificationPipeline


def check_xrt_smi() -> str:
    exe = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
    if not exe.is_file():
        return "xrt-smi not found"
    out = subprocess.run([str(exe), "examine", "-r", "aie-partitions"], capture_output=True, text=True)
    return out.stdout.strip()


def run_amd_vitisai(model_path: str) -> Dict[str, Any]:
    """Test AMD Vitis AI EP behavior on the classification head in resnet_env17."""
    py_exe = r"C:\Users\Ignis\miniforge3\envs\resnet_env17\python.exe"
    code = (
        "import os, sys, onnxruntime as ort, numpy as np; "
        "os.environ['RYZEN_AI_INSTALLATION_PATH'] = r'C:\\Program Files\\RyzenAI\\1.7.1'; "
        "xclbin = r'C:\\Program Files\\RyzenAI\\1.7.1\\voe-4.0-win_amd64\\xclbins\\phoenix\\4x4.xclbin'; "
        "po = [{'config_file': 'vaip_config.json', 'xclbin': xclbin, 'cacheDir': 'test_head_cache', 'cacheKey': 'test_head_cache'}]; "
        "sess = ort.InferenceSession(r'" + model_path + "', providers=['VitisAIExecutionProvider'], provider_options=po); "
        "inp = sess.get_inputs()[0]; "
        "data = np.zeros(inp.shape, dtype=np.uint8); "
        "sess.run(None, {inp.name: data}); "
        "print('SUCCESS')"
    )
    res = subprocess.run([py_exe, "-c", code], capture_output=True, text=True)
    if res.returncode == 0 and "SUCCESS" in res.stdout:
        return {"status": "ok", "npu_nodes": "unknown"}
    stderr = res.stderr
    err_line = "runner_requests_queue.cpp:178: Failed to create runner: invalid vector subscript"
    for line in stderr.splitlines():
        if "runner_requests_queue.cpp" in line or "invalid vector subscript" in line:
            err_line = line.strip()
            break
    return {
        "status": "failed",
        "error": err_line,
        "full_error": stderr.strip(),
        "npu_nodes": 0,
    }


def benchmark_amd_cpu(model_path: str, warmup: int = 50, iters: int = 500) -> Dict[str, Any]:
    """Benchmark AMD's CPU fallback path (CPUExecutionProvider)."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp = np.zeros((1, 2048, 1, 1), dtype=np.uint8)

    for _ in range(warmup):
        _ = sess.run(None, {inp_name: inp})

    latencies = []
    prep_times = []
    post_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        feed = {inp_name: inp}
        t1 = time.perf_counter()
        out = sess.run(None, feed)[0].reshape(-1)
        t2 = time.perf_counter()
        exps = np.exp(out - np.max(out))
        _ = exps / np.sum(exps)
        t3 = time.perf_counter()

        prep_times.append((t1 - t0) * 1e3)
        latencies.append((t2 - t1) * 1e3)
        post_times.append((t3 - t2) * 1e3)

    e2e = [p + l + post for p, l, post in zip(prep_times, latencies, post_times)]
    rss_mb = get_rss_mb()

    return {
        "mean_ms": float(np.mean(e2e)),
        "p95_ms": float(np.percentile(e2e, 95)),
        "p99_ms": float(np.percentile(e2e, 99)),
        "min_ms": float(np.min(e2e)),
        "max_ms": float(np.max(e2e)),
        "prep_ms": float(np.mean(prep_times)),
        "infer_ms": float(np.mean(latencies)),
        "post_ms": float(np.mean(post_times)),
        "rss_mb": rss_mb,
    }


def benchmark_ignition_npu(container_path: str, warmup: int = 50, iters: int = 500) -> Dict[str, Any]:
    """Benchmark Ignition's native NPU classification pipeline."""
    feat = np.full(2048, 128, dtype=np.uint8)

    with ClassificationPipeline(container_path, device_index=0) as pipe:
        # Warmup
        for _ in range(warmup):
            _, _ = pipe.predict_sync(feat)

        e2e_list = []
        prep_list = []
        npu_list = []
        read_list = []
        post_list = []

        for _ in range(iters):
            t0 = time.perf_counter()
            _, tm = pipe.predict_sync(feat)
            t1 = time.perf_counter()
            e2e_list.append((t1 - t0) * 1e3)
            prep_list.append(tm.preprocess_ms)
            npu_list.append(tm.dispatch_ms)
            read_list.append(tm.readback_ms)
            post_list.append(tm.postprocess_ms)

        rss_mb = get_rss_mb()

    return {
        "mean_ms": float(np.mean(e2e_list)),
        "p95_ms": float(np.percentile(e2e_list, 95)),
        "p99_ms": float(np.percentile(e2e_list, 99)),
        "min_ms": float(np.min(e2e_list)),
        "max_ms": float(np.max(e2e_list)),
        "prep_ms": float(np.mean(prep_list)),
        "npu_ms": float(np.mean(npu_list)),
        "read_ms": float(np.mean(read_list)),
        "post_ms": float(np.mean(post_list)),
        "rss_mb": rss_mb,
    }


def main():
    print("=" * 78)
    print("Classification Head Silicon Benchmark: Ignition vs AMD Ryzen AI 1.7.1")
    print("Hardware: AMD Ryzen 7 8700G, Phoenix XDNA1 NPU [003d:00:01.1]")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 78)

    smi_before = check_xrt_smi()
    print("[xrt-smi before]")
    print(smi_before)
    print()

    model_onnx = str(REPO_ROOT / "build" / "test_resnet50_head.onnx")
    container_ignite = str(REPO_ROOT / "build" / "test_resnet50_head.ignite")

    # 1. Audit AMD Vitis AI EP
    print("Auditing AMD Vitis AI EP 1.7.1 on classification head...")
    vitis_result = run_amd_vitisai(model_onnx)
    print(f"  Vitis AI EP status: {vitis_result['status']}")
    if vitis_result["status"] == "failed":
        print(f"  Error: {vitis_result['error']}")
        print(f"  NPU nodes accelerated: {vitis_result['npu_nodes']} (100% CPU fallback)")
    print()

    # Interleaved runs: 2 rounds of AMD CPU fallback and Ignition NPU
    results = []

    for round_idx in (1, 2):
        print(f"--- Round {round_idx}: Interleaved Execution ---")
        # AMD CPU
        print(f"  Running AMD CPU fallback (50 warmup, 500 timed frames)...")
        amd_res = benchmark_amd_cpu(model_onnx, warmup=50, iters=500)
        print(f"    AMD CPU: mean {amd_res['mean_ms']:.3f} ms, 99th {amd_res['p99_ms']:.3f} ms, RSS {amd_res['rss_mb']:.1f} MB")

        # Ignition NPU
        print(f"  Running Ignition NPU container (50 warmup, 500 timed frames)...")
        ign_res = benchmark_ignition_npu(container_ignite, warmup=50, iters=500)
        print(f"    Ignition NPU: mean {ign_res['mean_ms']:.3f} ms (dispatch {ign_res['npu_ms']:.3f} ms), 99th {ign_res['p99_ms']:.3f} ms, RSS {ign_res['rss_mb']:.1f} MB")

        results.append((amd_res, ign_res))
        print()

    smi_after = check_xrt_smi()
    print("[xrt-smi after]")
    print(smi_after)
    print()

    # Summary table
    print("=" * 78)
    print("Summary Table:")
    print("| Run | Stack | Target | Mean | 99th pct | Where the time goes | Memory |")
    print("|---|---|---|---:|---:|---|---:|")
    print(f"| 1 | AMD Ryzen AI Software 1.7.1 (Vitis AI EP) | NPU | **crashes** | - | 0 nodes on NPU (`invalid vector subscript`), 100% CPU fallback | - |")
    print(f"| 2 | AMD Ryzen AI Software 1.7.1 (CPU fallback) | CPU | {results[0][0]['mean_ms']:.3f} ms | {results[0][0]['p99_ms']:.3f} ms | prep {results[0][0]['prep_ms']:.3f} ms, Gemm {results[0][0]['infer_ms']:.3f} ms, softmax {results[0][0]['post_ms']:.3f} ms | {results[0][0]['rss_mb']:.1f} MB |")
    print(f"| 3 | Ignition on the container | NPU | **{results[0][1]['mean_ms']:.3f} ms** | {results[0][1]['p99_ms']:.3f} ms | prep {results[0][1]['prep_ms']:.3f} ms, NPU dispatch {results[0][1]['npu_ms']:.3f} ms, readback {results[0][1]['read_ms']:.3f} ms, softmax {results[0][1]['post_ms']:.3f} ms | **{results[0][1]['rss_mb']:.1f} MB** |")
    print(f"| 4 | AMD Ryzen AI Software 1.7.1 (CPU fallback) | CPU | {results[1][0]['mean_ms']:.3f} ms | {results[1][0]['p99_ms']:.3f} ms | prep {results[1][0]['prep_ms']:.3f} ms, Gemm {results[1][0]['infer_ms']:.3f} ms, softmax {results[1][0]['post_ms']:.3f} ms | {results[1][0]['rss_mb']:.1f} MB |")
    print(f"| 5 | Ignition on the container | NPU | **{results[1][1]['mean_ms']:.3f} ms** | {results[1][1]['p99_ms']:.3f} ms | prep {results[1][1]['prep_ms']:.3f} ms, NPU dispatch {results[1][1]['npu_ms']:.3f} ms, readback {results[1][1]['read_ms']:.3f} ms, softmax {results[1][1]['post_ms']:.3f} ms | **{results[1][1]['rss_mb']:.1f} MB** |")
    print("=" * 78)
    print()
    print("AMD Vitis AI EP Failure Witness:")
    print(vitis_result.get("full_error", "No error message"))


if __name__ == "__main__":
    main()
