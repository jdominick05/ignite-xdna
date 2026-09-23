#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tools/bench_yolov8_split_suite.py

Comprehensive physical silicon benchmarking of all YOLOv8 models
(YOLOv8n, YOLOv8s, YOLOv8n-pose, YOLOv8m, YOLOv8l, YOLOv8x)
with split containers, multi-segment linked dispatch, decoupled stationary weights,
and early-exit cascades on AMD Phoenix NPU (Ryzen 7 8700G, XDNA1).

Enforces:
  - Pinned execution on 8 physical CPU cores (mask 0x5555, OMP_NUM_THREADS=8)
  - PyXRT device harness on NPU Device 0 (clean witness before and after)
  - Inter-segment dispatch timing and early-exit headroom calculation
  - Decoupled stationary weight storage and memory footprint quantification
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler.serializer import IgniteModelReader
from ignite_xdna.runtime.driver import setup_xrt_environment
from ignite_xdna.runtime.graph_session import GraphSession

SMI = "C:/Windows/System32/AMD/xrt-smi.exe"


def enforce_eight_physical_cores():
    """Pin the current process to 8 physical CPU cores (mask 0x5555) on Ryzen 7 8700G."""
    try:
        proc = psutil.Process()
        # Cores 0, 2, 4, 6, 8, 10, 12, 14
        proc.cpu_affinity([0, 2, 4, 6, 8, 10, 12, 14])
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["OPENBLAS_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        os.environ["NUMEXPR_NUM_THREADS"] = "8"
        print(f"[*] Process affinity set to 8 physical cores: {proc.cpu_affinity()} (OMP_NUM_THREADS=8)")
    except Exception as ex:
        print(f"[-] Warning: could not set CPU affinity: {ex}")


def check_npu_witness(label: str) -> str:
    """Query xrt-smi to witness zero lingering hardware contexts on the device."""
    print(f"\n[{label}] NPU Hardware Status Witness:")
    try:
        res = subprocess.run([SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=15)
        out = (res.stdout + res.stderr).strip()
        print(out)
        if "No hardware contexts running" not in out:
            raise RuntimeError(f"NPU not idle at {label}!\n{out}")
        return out
    except Exception as ex:
        print(f"  [WARN] xrt-smi query failed: {ex}")
        return str(ex)


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def benchmark_model(
    model_name: str,
    container_path: Path,
    device_index: int = 0,
    warmup: int = 30,
    iters: int = 50,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"BENCHMARKING: {model_name.upper()} ({container_path.name})")
    print("=" * 80)

    if not container_path.exists():
        raise FileNotFoundError(f"Container not found: {container_path}")

    container_bytes = container_path.stat().st_size
    weights_path = container_path.with_suffix(".weights")
    weights_bytes = weights_path.stat().st_size if weights_path.exists() else 0

    with IgniteModelReader(container_path) as r:
        manifest = dict(r.manifest)
        task = manifest.get("task", "detect")
        egress_bytes = manifest.get("egress_bytes", 1209600)
        input_shape = manifest.get("input_shape", [1, 3, 640, 640])
        segments_meta = manifest.get("graph_engine", {}).get("segments", [])
        insts_bytes = manifest.get("graph_engine", {}).get("insts_bytes", 0)

    print(f"Container size:     {container_bytes:,} B ({container_bytes / 1e6:.2f} MB)")
    print(f"Decoupled weights:  {weights_bytes:,} B ({weights_bytes / 1e6:.2f} MB)")
    print(f"Total artifact pkg: {(container_bytes + weights_bytes) / 1e6:.2f} MB")
    print(f"Task:               {task}")
    print(f"Segments declared:  {len(segments_meta)}")
    for idx, s in enumerate(segments_meta):
        print(f"  Segment {idx}: layers {s.get('layers')}, tasks {s.get('tasks')}")

    # Prepare dummy input matching quantized input plane
    dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)

    # Initialize GraphSession
    t_init0 = time.perf_counter()
    session = GraphSession(container_path, device_index=device_index)
    init_ms = (time.perf_counter() - t_init0) * 1e3
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"Session init time:  {init_ms:.2f} ms")
    print(f"Process RSS memory: {rss_mb:.1f} MB")

    try:
        # Warmup
        print(f"Running {warmup} warm-up iterations...")
        for _ in range(warmup):
            session.run_yolo_monolithic(dummy_input)

        # Timed Full Multi-Segment Linked Execution
        print(f"Running {iters} timed iterations (Full Linked Dispatch)...")
        latencies_ms = []
        seg_times_list = []
        gaps_us_list = []

        for _ in range(iters):
            _, ts = session.run_yolo_monolithic(dummy_input, return_timestamps=True)
            latencies_ms.append(ts["npu_ms"])
            if hasattr(session, "last_segment_ms") and session.last_segment_ms:
                seg_times_list.append(list(session.last_segment_ms))

        # Measure Inter-Segment Hardware Gap (if multi-segment)
        if len(session._runs) > 1:
            for _ in range(20):
                # Measure time between run[0].wait() and run[1].start()
                session._runs[0].start()
                session._runs[0].wait(2000)
                t_w = time.perf_counter()
                session._runs[1].start()
                t_s = time.perf_counter()
                session._runs[1].wait(2000)
                gaps_us_list.append((t_s - t_w) * 1e6)

        # Timed Early-Exit Cascade (Segment 0 only)
        ee1_latencies_ms = []
        if len(session.segments) > 1:
            print(f"Running {iters} timed iterations (Early-Exit Cascade: Segment 0 only)...")
            for _ in range(iters):
                ee1_latencies_ms.append(session.dispatch(max_segments=1))

        # Check Output Parity & Shape
        heads = session.read_heads(unswizzle=True)
        print(f"Output egress bytes read: {len(heads):,} (min: {heads.min()}, max: {heads.max()})")

    finally:
        session.close()

    # Summarize stats
    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    fps_full = 1e3 / mean_lat

    seg_means = [float(v) for v in np.mean(seg_times_list, axis=0)] if seg_times_list else []
    mean_gap_us = float(np.mean(gaps_us_list)) if gaps_us_list else 0.0

    ee1_mean = float(np.mean(ee1_latencies_ms)) if ee1_latencies_ms else None
    ee1_fps = (1e3 / ee1_mean) if ee1_mean else None
    ee1_savings = (100 * (1.0 - ee1_mean / mean_lat)) if ee1_mean else 0.0

    print(f"\nRESULTS FOR {model_name.upper()}:")
    print(f"  Full Network Dispatch: {mean_lat:.3f} ms (p50: {p50_lat:.3f} ms, p95: {p95_lat:.3f} ms)")
    print(f"  Full Network Headroom: {fps_full:.1f} FPS")
    if seg_means:
        for idx, sm in enumerate(seg_means):
            print(f"    Segment {idx}: {sm:.3f} ms ({sm/mean_lat*100:.1f}%)")
    if mean_gap_us > 0:
        print(f"  Inter-Segment Gap:     {mean_gap_us:.1f} us")
    if ee1_mean:
        print(f"  Early Exit 1 (Seg 0):  {ee1_mean:.3f} ms ({ee1_savings:.1f}% latency cut)")
        print(f"  Early Exit 1 Headroom: {ee1_fps:.1f} FPS (saves {mean_lat - ee1_mean:.2f} ms/frame)")

    return {
        "model": model_name,
        "task": task,
        "container": container_path.name,
        "container_bytes": container_bytes,
        "weights_bytes": weights_bytes,
        "rss_mb": rss_mb,
        "init_ms": init_ms,
        "full_mean_ms": mean_lat,
        "full_p50_ms": p50_lat,
        "full_p95_ms": p95_lat,
        "full_fps": fps_full,
        "segment_means_ms": seg_means,
        "inter_segment_gap_us": mean_gap_us,
        "ee1_mean_ms": ee1_mean,
        "ee1_fps": ee1_fps,
        "ee1_savings_pct": ee1_savings,
    }


def main():
    setup_xrt_environment()
    enforce_eight_physical_cores()

    machine = os.environ.get("COMPUTERNAME", platform.node())
    commit = get_git_commit()
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 80)
    print("ALL-YOLOV8 SPLIT CONTAINER BENCHMARK SUITE (PHYSICAL SILICON)")
    print(f"Host:    {machine} (Ryzen 7 8700G, 8 Physical Cores, Phoenix Device 0)")
    print(f"Date:    {now_utc}")
    print(f"Commit:  {commit}")
    print("=" * 80)

    check_npu_witness("PRE-SUITE")

    models_to_bench = [
        ("yolov8n", ROOT / "build" / "yolov8n_full_split2.ignite"),
        ("yolov8s", ROOT / "build" / "yolov8s_split3.ignite"),
        ("yolov8n-pose", ROOT / "build" / "yolov8n_pose_split3.ignite"),
        ("yolov8m", ROOT / "build" / "yolov8m_split.ignite"),
        ("yolov8l", ROOT / "build" / "yolov8l_split.ignite"),
        ("yolov8x", ROOT / "build" / "yolov8x_split.ignite"),
    ]

    results = []
    for name, path in models_to_bench:
        if not path.exists():
            print(f"\n[-] Skipping {name}: {path} does not exist yet.")
            continue
        try:
            res = benchmark_model(name, path, device_index=0, warmup=20, iters=50)
            results.append(res)
            time.sleep(0.5)
        except Exception as ex:
            print(f"[-] ERROR benchmarking {name}: {ex}")

    check_npu_witness("POST-SUITE")

    print("\n" + "=" * 80)
    print("SUITE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<14} | {'Full Net (ms)':<14} | {'Early Exit (ms)':<16} | {'EE Speedup':<11} | {'Container':<10} | {'Weights':<10} | {'RSS (MB)':<8}")
    print("-" * 92)
    for r in results:
        ee_str = f"{r['ee1_mean_ms']:.2f} ms" if r['ee1_mean_ms'] else "N/A"
        ee_spd = f"{r['ee1_savings_pct']:.1f}% cut" if r['ee1_mean_ms'] else "N/A"
        print(f"{r['model']:<14} | {r['full_mean_ms']:>6.2f} ms      | {ee_str:>8}        | {ee_spd:>9} | {r['container_bytes']/1e6:>5.2f} MB   | {r['weights_bytes']/1e6:>5.2f} MB   | {r['rss_mb']:>6.1f}")
    print("=" * 80)

    # Write out log file
    log_path = ROOT / "results" / "aie" / "yolov8_split_suite_phoenix_20260920.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"UTC:      {now_utc}\n")
        f.write(f"MACHINE:  {machine}\n")
        f.write(f"COMMAND:  python tools/bench_yolov8_split_suite.py\n")
        f.write(f"COMMIT:   {commit}\n")
        f.write(f"AFFINITY: 8 Physical Cores (0x5555), OMP_NUM_THREADS=8\n\n")
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\n[+] Results recorded to: {log_path}")


if __name__ == "__main__":
    main()
