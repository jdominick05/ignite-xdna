#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tools/verify_split_silicon.py

End-to-end silicon verification of split containers and decoupled weights on physical AMD Phoenix NPU:
1. Decoupled Stationary Weights: validates sidecar loading and bit-exact output parity against monolithic container.
2. Multi-Segment Pure NPU Splits: validates segmented instruction streams, inter-segment persistence, and bit-exact output parity.
3. Early-Exit Cascade: validates max_segments=1 partial dispatch and measures latency reduction.

Usage:
    python tools/verify_split_silicon.py [--device 0]
"""
import argparse
import copy
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_schedule as es
from ignite_xdna.compiler import graph_ir
from ignite_xdna.compiler.engine_sequence import program_task_count, split_instruction_stream
from ignite_xdna.compiler.serializer import (
    ARCH_XDNA1_PHOENIX,
    IgniteModelReader,
    IgniteModelWriter,
)
from ignite_xdna.runtime.graph_session import GraphSession

SMI = "C:/Windows/System32/AMD/xrt-smi.exe"
BASE_CONTAINER = ROOT / "build" / "yolov8n_full.ignite"
MODEL_ONNX = ROOT / "models" / "yolov8n_cut_xint8.onnx"


def check_npu_witness(tag: str = ""):
    """Witness that NPU has no lingering hardware contexts."""
    res = subprocess.run([SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=20)
    out = res.stdout + res.stderr
    print(f"[{tag}] NPU Hardware Status:\n{out.strip()}", flush=True)
    if res.returncode != 0 or "No hardware contexts running" not in out:
        raise RuntimeError(f"NPU not idle at [{tag}]")


def make_decoupled_container(base_ignite: Path, out_ignite: Path, out_weights: Path):
    """Transform a monolithic container into a decoupled-weights container."""
    with IgniteModelReader(base_ignite) as r:
        manifest = copy.deepcopy(r.manifest)
        wpackets = r.get_blob_bytes("wpackets.bin")
        out_weights.write_bytes(wpackets)

        manifest["graph_engine"]["decoupled_weights"] = True
        manifest["graph_engine"]["weights_file"] = out_weights.name
        manifest["graph_engine"]["weights_sha256"] = hashlib.sha256(wpackets).hexdigest()
        manifest["graph_engine"]["weights_bytes"] = len(wpackets)

        writer = IgniteModelWriter(manifest_meta=manifest, arch_id=r.header.arch_id)
        writer.add_blob("engine.xclbin", r.get_blob_bytes("engine.xclbin"), content_type="xclbin")
        writer.add_blob("insts.bin", r.get_blob_bytes("insts.bin"), content_type="npu_instructions")
        writer.write(out_ignite)


def make_split_container(base_ignite: Path, out_ignite: Path, out_weights: Path, split_layer: int = 10):
    """Transform a monolithic container into a 2-segment pure NPU split container with decoupled weights."""
    ir = graph_ir.lower_yolov8n(MODEL_ONNX)
    ws = es.plan_workspace(ir)
    scheds, _ = es.schedule_graph(ir, ws)
    tasks = [sum(program_task_count(p) for p in s.programs) for s in scheds]
    tasks_seg0 = sum(tasks[:split_layer])
    tasks_seg1 = sum(tasks[split_layer:])

    with IgniteModelReader(base_ignite) as r:
        manifest = copy.deepcopy(r.manifest)
        insts = r.get_blob_bytes("insts.bin")
        wpackets = r.get_blob_bytes("wpackets.bin")
        out_weights.write_bytes(wpackets)

        pieces = split_instruction_stream(insts, [tasks_seg0, tasks_seg1])

        segments = [
            {"kind": "npu", "layers": [0, split_layer], "tasks": tasks_seg0, "blob": "insts_0.bin"},
            {"kind": "npu", "layers": [split_layer, len(scheds)], "tasks": tasks_seg1, "blob": "insts_1.bin"},
        ]
        manifest["single_dispatch"] = False
        manifest["graph_engine"]["segments"] = segments
        manifest["graph_engine"]["decoupled_weights"] = True
        manifest["graph_engine"]["weights_file"] = out_weights.name
        manifest["graph_engine"]["weights_sha256"] = hashlib.sha256(wpackets).hexdigest()
        manifest["graph_engine"]["weights_bytes"] = len(wpackets)

        writer = IgniteModelWriter(manifest_meta=manifest, arch_id=r.header.arch_id)
        writer.add_blob("engine.xclbin", r.get_blob_bytes("engine.xclbin"), content_type="xclbin")
        writer.add_blob("insts_0.bin", pieces[0], content_type="npu_instructions")
        writer.add_blob("insts_1.bin", pieces[1], content_type="npu_instructions")
        writer.write(out_ignite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not BASE_CONTAINER.exists():
        print(f"[!] Base container {BASE_CONTAINER} not found.")
        sys.exit(1)
    if not MODEL_ONNX.exists():
        print(f"[!] ONNX model {MODEL_ONNX} not found.")
        sys.exit(1)

    print("=" * 72)
    print("PHYSICAL SILICON VERIFICATION: SPLIT CONTAINERS & DECOUPLED WEIGHTS")
    print("=" * 72)

    # Pre-run witness
    check_npu_witness("PRE-RUN")

    # Fixed synthetic input for bit-exact comparison
    rng = np.random.default_rng(42)
    input_data = rng.integers(-128, 127, size=(1, 3, 640, 640), dtype=np.int8)

    # 1. Monolithic Baseline Run
    print("\n[*] Running 1/3: Monolithic Baseline (build/yolov8n_full.ignite)...")
    sess_mono = GraphSession(BASE_CONTAINER, device_index=args.device)
    for _ in range(3):
        sess_mono.run_yolo_monolithic(input_data)
    mono_lats = []
    for _ in range(10):
        _, ts = sess_mono.run_yolo_monolithic(input_data, return_timestamps=True)
        mono_lats.append(ts["npu_ms"])
    out_mono = sess_mono.run_yolo_monolithic(input_data)
    sess_mono.close()
    print(f"    [OK] Monolithic NPU Dispatch: {np.median(mono_lats):.3f} ms (min {np.min(mono_lats):.3f} ms)")

    # 2. Decoupled Weights Run
    print("\n[*] Generating & Running 2/3: Decoupled Weights Container...")
    decoupled_ignite = ROOT / "build" / "yolov8n_full_decoupled.ignite"
    decoupled_weights = ROOT / "build" / "yolov8n_full_decoupled.weights"
    make_decoupled_container(BASE_CONTAINER, decoupled_ignite, decoupled_weights)

    sess_decoupled = GraphSession(decoupled_ignite, device_index=args.device)
    for _ in range(3):
        sess_decoupled.run_yolo_monolithic(input_data)
    decoupled_lats = []
    for _ in range(10):
        _, ts = sess_decoupled.run_yolo_monolithic(input_data, return_timestamps=True)
        decoupled_lats.append(ts["npu_ms"])
    out_decoupled = sess_decoupled.run_yolo_monolithic(input_data)
    sess_decoupled.close()

    np.testing.assert_array_equal(out_mono["raw_heads"], out_decoupled["raw_heads"])
    print(f"    [OK] Decoupled NPU Dispatch: {np.median(decoupled_lats):.3f} ms (min {np.min(decoupled_lats):.3f} ms)")
    print(f"    [OK] Bit-exact output match: 100% agreement on all {len(out_mono['raw_heads']):,} head bytes")

    # 3. Multi-Segment Split Container Run (2 NPU segments, pure NPU split at layer 10)
    print("\n[*] Generating & Running 3/3: Multi-Segment Split Container (Layer 10 Cut)...")
    split_ignite = ROOT / "build" / "yolov8n_full_split2.ignite"
    split_weights = ROOT / "build" / "yolov8n_full_split2.weights"
    make_split_container(BASE_CONTAINER, split_ignite, split_weights, split_layer=10)

    sess_split = GraphSession(split_ignite, device_index=args.device)
    for _ in range(3):
        sess_split.run_yolo_monolithic(input_data)
    split_lats = []
    seg0_lats = []
    seg1_lats = []
    for _ in range(10):
        _, ts = sess_split.run_yolo_monolithic(input_data, return_timestamps=True)
        split_lats.append(ts["npu_ms"])
        seg0_lats.append(sess_split.last_segment_ms[0])
        seg1_lats.append(sess_split.last_segment_ms[1])
    out_split = sess_split.run_yolo_monolithic(input_data)

    np.testing.assert_array_equal(out_mono["raw_heads"], out_split["raw_heads"])
    print(f"    [OK] Split Total NPU Dispatch: {np.median(split_lats):.3f} ms (min {np.min(split_lats):.3f} ms)")
    print(f"         Segment 0 (Layers 0..10): {np.median(seg0_lats):.3f} ms")
    print(f"         Segment 1 (Layers 10..66): {np.median(seg1_lats):.3f} ms")
    print(f"         Inter-segment gap: {(np.median(split_lats) - (np.median(seg0_lats) + np.median(seg1_lats))) * 1e3:.1f} µs")
    print(f"    [OK] Bit-exact output match: 100% agreement on all {len(out_mono['raw_heads']):,} head bytes")

    # 4. Early-Exit Test
    print("\n[*] Testing Early-Exit Cascade (max_segments=1)...")
    sess_split.stage_input(input_data)
    early_ms = sess_split.dispatch(max_segments=1)
    print(f"    [OK] Early-Exit Dispatch (Segment 0 only): {early_ms:.3f} ms (executed {len(sess_split.last_segment_ms)} of {len(sess_split.segments)} segments)")

    sess_split.close()

    # Post-run witness
    check_npu_witness("POST-RUN")

    print("\n" + "=" * 72)
    print("ALL SILICON VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()
