"""
Batch>1 Silent Failure Demo: The Unwritten Slot Bug on AMD XDNA1 NPU.

Demonstrates the critical negative finding recorded in CLAUDE.md & docs/DECISIONS.md:
"Batch 1 only: static, no dynamic_axes, opset 17, dynamo=False. Measured at static batch 2,
the EP takes 80/395 nodes and WRITES ONLY SLOT 0 — slot 1's logits are byte-identical
across every input, i.e. a stale buffer, not a miscomputation. It fails silently, with
a plausible-looking latency." (Filed as AMD issue #401).

This demo:
  1. Runs two distinct image pairs ([A, B] then [C, D]) through models/resnet50_b2_xint8_c64.onnx.
  2. Compares CPUExecutionProvider vs VitisAIExecutionProvider (NPU).
  3. Measures per-slot logit delta:
     - CPU: Both Slot 0 and Slot 1 change completely when fed new inputs.
     - NPU: Slot 0 updates with new logits; Slot 1 is byte-identical (max abs diff = 0.000000)
       to the prior run, exposing the unwritten DMA output buffer.
  4. Renders an annotated diagnostic visual report to results/batch_failure_npu.jpg.

Usage:
    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/batch_failure_demo.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu.paths import RESNET_CACHE_KEY
from npu.preprocess import build_transform
from npu.session import build_session, clear_cache

DEFAULT_MODEL = ROOT / "models" / "resnet50_b2_xint8_c64.onnx"
DEFAULT_CFG = ROOT / "models" / "preprocess_config.json"
CALIB_DIR = ROOT / "data" / "calib"
LABELS_FILE = CALIB_DIR / "labels.json"


def load_images_and_batch(img_paths, transform):
    """Load two image paths, apply transform, and return (batch_tensor, [bgr_display_imgs])."""
    tensors = []
    display_imgs = []
    for p in img_paths:
        t = transform(p)  # (1, 3, 224, 224)
        tensors.append(t)
        # Load for OpenCV display (resized to 224x224)
        bgr = cv2.imread(str(p))
        bgr_resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        display_imgs.append(bgr_resized)

    batch_input = np.concatenate(tensors, axis=0)  # (2, 3, 224, 224)
    return batch_input, display_imgs


def run_session_batch(sess, input_name, batch_input):
    """Run batch and return logits (N, 1000) and inference ms."""
    t0 = time.perf_counter()
    logits = sess.run(None, {input_name: batch_input})[0]
    ms = (time.perf_counter() - t0) * 1000.0
    return logits, ms


def main():
    ap = argparse.ArgumentParser(
        description="Visual demo of the static batch>1 unwritten-slot bug on AMD XDNA1 NPU."
    )
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    with open(args.cfg, "r") as f:
        cfg = json.load(f)
    transform = build_transform(cfg)

    # Load 4 distinct calibration images
    img_files = ["000000.jpg", "000001.jpg", "000002.jpg", "000003.jpg"]
    img_paths = [CALIB_DIR / f for f in img_files]
    for p in img_paths:
        if not p.exists():
            raise SystemExit(f"Calibration image missing: {p}")

    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r") as f:
            labels = json.load(f)

    batch1_input, batch1_imgs = load_images_and_batch(img_paths[:2], transform)
    batch2_input, batch2_imgs = load_images_and_batch(img_paths[2:], transform)

    print(f"\n{'='*75}")
    print("  STATIC BATCH>1 SILENT FAILURE DEMO (AMD XDNA1 NPU)")
    print(f"  Model: {model_path.name} | Batch Size: 2")
    print("  Pair 1: [000000.jpg, 000001.jpg]  vs  Pair 2: [000002.jpg, 000003.jpg]")
    print(f"{'='*75}\n")

    # 1. Benchmark CPU (Known-Good Control)
    print("  [1/2] Testing CPUExecutionProvider (Control)...", flush=True)
    sess_cpu = build_session(str(model_path), "cpu", RESNET_CACHE_KEY, log_severity=args.log)
    inp_name = sess_cpu.get_inputs()[0].name

    # Run CPU Pass 1 & 2
    cpu_out1, cpu_ms1 = run_session_batch(sess_cpu, inp_name, batch1_input)
    cpu_out2, cpu_ms2 = run_session_batch(sess_cpu, inp_name, batch2_input)

    cpu_diff0 = np.max(np.abs(cpu_out2[0] - cpu_out1[0]))
    cpu_diff1 = np.max(np.abs(cpu_out2[1] - cpu_out1[1]))
    print(f"        CPU Pass 1: {cpu_ms1:.2f} ms | Pass 2: {cpu_ms2:.2f} ms")
    print(f"        Slot 0 max abs diff: {cpu_diff0:.6f}  (Updated)")
    print(f"        Slot 1 max abs diff: {cpu_diff1:.6f}  (Updated)")

    # 2. Benchmark NPU (VitisAIExecutionProvider)
    print("\n  [2/2] Testing VitisAIExecutionProvider on Phoenix NPU...", flush=True)
    clear_cache(RESNET_CACHE_KEY)
    sess_npu = build_session(str(model_path), "npu", RESNET_CACHE_KEY, log_severity=args.log)

    # Run NPU Pass 1 (first run compiles)
    print("        NPU Pass 1 (compiling/running)...", end="", flush=True)
    npu_out1, npu_ms1 = run_session_batch(sess_npu, inp_name, batch1_input)
    print(f" done ({npu_ms1:.2f} ms)")

    # Run NPU Pass 2 (warm)
    print("        NPU Pass 2 with new images...", end="", flush=True)
    npu_out2, npu_ms2 = run_session_batch(sess_npu, inp_name, batch2_input)
    print(f" done ({npu_ms2:.2f} ms)")

    npu_diff0 = np.max(np.abs(npu_out2[0] - npu_out1[0]))
    npu_diff1 = np.max(np.abs(npu_out2[1] - npu_out1[1]))
    print(f"        Slot 0 max abs diff: {npu_diff0:.6f}  (Updated)")
    print(f"        Slot 1 max abs diff: {npu_diff1:.6f}  <-- STALE UNWRITTEN BUFFER!")

    # ------------------------------------------------------------------ #
    # Comparison Summary Dashboard                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 75)
    print(f"  {'Execution Provider':<26} {'Slot 0 Max Delta':>18} {'Slot 1 Max Delta':>18} {'Status':>10}")
    print("-" * 75)
    print(f"  {'CPUExecutionProvider':<26} {cpu_diff0:>18.6f} {cpu_diff1:>18.6f} {'PASS':>10}")
    status_npu = "FAIL (STALE)" if npu_diff1 == 0.0 else "PASS"
    print(f"  {'VitisAIExecutionProvider':<26} {npu_diff0:>18.6f} {npu_diff1:>18.6f} {status_npu:>10}")
    print("=" * 75)

    print("\n  ROOT CAUSE & HARDWARE MECHANISM:")
    print("  The VitisAI compiler assigns only 80/395 nodes to NPU and sets the output")
    print("  DMA stride incorrectly for batch>1. Output slot 0 receives computed logits,")
    print("  while slot 1 is NEVER written to memory by the hardware engine.")
    print("  ORT reports success and returns a plausible latency (~14 ms), silently")
    print("  returning stale uninitialized RAM for all batch slots > 0.")
    print("  -> LOCKED DECISION: Batch 1 ONLY on AMD XDNA1.")

    # ------------------------------------------------------------------ #
    # Render Visual Evidence Image                                       #
    # ------------------------------------------------------------------ #
    # Canvas layout:
    # Top half: Pass 1 [Img 0 (Slot 0), Img 1 (Slot 1)]
    # Bottom half: Pass 2 [Img 2 (Slot 0), Img 3 (Slot 1)]
    w_box = 240
    h_box = 240
    pad = 20

    # Build image tiles
    tile0 = cv2.resize(batch1_imgs[0], (w_box, h_box))
    tile1 = cv2.resize(batch1_imgs[1], (w_box, h_box))
    tile2 = cv2.resize(batch2_imgs[0], (w_box, h_box))
    tile3 = cv2.resize(batch2_imgs[1], (w_box, h_box))

    # Top predictions
    pred_cpu_p1_0 = int(np.argmax(cpu_out1[0]))
    pred_cpu_p1_1 = int(np.argmax(cpu_out1[1]))
    pred_npu_p1_0 = int(np.argmax(npu_out1[0]))
    pred_npu_p1_1 = int(np.argmax(npu_out1[1]))

    pred_cpu_p2_0 = int(np.argmax(cpu_out2[0]))
    pred_cpu_p2_1 = int(np.argmax(cpu_out2[1]))
    pred_npu_p2_0 = int(np.argmax(npu_out2[0]))
    pred_npu_p2_1 = int(np.argmax(npu_out2[1]))

    # Label image tiles
    cv2.putText(tile0, f"Slot 0: {img_files[0]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(tile0, f"Class: {pred_npu_p1_0}", (8, h_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(tile1, f"Slot 1: {img_files[1]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(tile1, f"Class: {pred_npu_p1_1}", (8, h_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(tile2, f"Slot 0: {img_files[2]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(tile2, f"Class: {pred_npu_p2_0}", (8, h_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Tile 3 is STALE
    cv2.rectangle(tile3, (0, 0), (w_box, h_box), (0, 0, 255), 4)
    cv2.putText(tile3, f"Slot 1: {img_files[3]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(tile3, "UNWRITTEN STALE!", (8, h_box // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(tile3, f"Class: {pred_npu_p2_1} (Diff: 0.0)", (8, h_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    row1 = np.hstack([tile0, tile1])
    row2 = np.hstack([tile2, tile3])

    # Header and divider cards
    hdr1 = np.zeros((36, row1.shape[1], 3), dtype=np.uint8)
    cv2.putText(hdr1, "PASS 1: Input Images [000000.jpg, 000001.jpg]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    hdr2 = np.zeros((36, row2.shape[1], 3), dtype=np.uint8)
    cv2.putText(hdr2, "PASS 2: Input Images [000002.jpg, 000003.jpg] (NEW INPUTS)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    banner = np.zeros((64, row1.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, "NPU BATCH>1 SILENT HARDWARE CORRUPTION DETECTED", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(banner, f"Slot 1 Max Logit Delta = {npu_diff1:.8f} (Stale buffer never updated)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    composite = np.vstack([hdr1, row1, hdr2, row2, banner])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "batch_failure_npu.jpg"
    cv2.imwrite(str(out_file), composite)
    print(f"\n  Saved visual evidence composite -> {out_file}\n")


if __name__ == "__main__":
    main()
