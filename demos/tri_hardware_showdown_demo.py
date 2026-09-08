"""
Tri-Hardware Showdown Demo: CPU vs DirectML (Radeon 780M iGPU) vs Ryzen AI NPU (XDNA1).

Demonstrates the central trade-off documented in docs/BENCHMARKS.md:
"iGPU vs NPU: is Ryzen AI worth it over DirectML?"

Four pathways evaluated back-to-back on the exact same test image:
  1. CPU FP32                 — 8-core Zen 4 baseline (zero quantization effort)
  2. DirectML FP32 (iGPU)     — Radeon 780M running export ONNX as-is
  3. DirectML FP16 (iGPU)     — Radeon 780M running FP16 (one-line convert, 0 mAP loss)
  4. Ryzen AI NPU (XDNA1)     — 4x4 AIE2 tile running XINT8 + AdaRound (head-cut)

Timing follows the CLAUDE.md invariant:
  * "infer" is sess.run strictly alone (hardware compute)
  * "post" is DFL decode (cut models) + NMS
  * Measured in a single sitting to eliminate cross-session latency drift

Saves a 4-quadrant side-by-side comparison JPEG to results/tri_hardware_showdown.jpg.

Usage:
    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/tri_hardware_showdown_demo.py
    python tools/tri_hardware_showdown_demo.py --runs 50 --source assets/test_image.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import npu.yolo as yc
from npu.paths import YOLO_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolo_decode import decode_heads, head_order

# Four comparison configs: (label, device_name, model_relpath, ep, cache_key, published_map50_95, effort_note)
CONFIGS = [
    (
        "CPU FP32",
        "Zen 4 CPU (8-core)",
        "models/yolov8n.onnx",
        "cpu",
        None,
        36.69,
        "None (baseline export)",
    ),
    (
        "DML FP32",
        "Radeon 780M iGPU",
        "models/yolov8n.onnx",
        "dml",
        None,
        36.69,
        "None (runs ONNX as-is)",
    ),
    (
        "DML FP16",
        "Radeon 780M iGPU",
        "models/yolov8n_fp16.onnx",
        "dml",
        None,
        36.72,
        "One-line convert (lossless)",
    ),
    (
        "NPU XINT8",
        "Phoenix XDNA1 (4x4)",
        "models/yolov8n_cut_xint8_adaround.onnx",
        "npu",
        YOLO_CUT_CACHE_KEY,
        32.19,
        "Head-cut + AdaRound FastFT",
    ),
]


def _bar(frac, width=20):
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def run_pathway(model_path, ep, cache_key, source_img, runs, conf, log_sev):
    """Run one hardware pathway and return (dets, infer_ms, post_ms, annotated_frame)."""
    if ep == "npu" and cache_key:
        clear_cache(cache_key)

    sess = build_session(str(model_path), ep, cache_key, log_severity=log_sev)
    inp = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, str(model_path))
    n_out = len(sess.get_outputs())

    if n_out == len(yc.HEAD_OUTS):
        order = head_order(sess, imgsz)

        def run_raw(x):
            return sess.run(None, {inp: x})

        def decode_raw(r):
            return decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=conf)
    elif n_out == 1:
        def run_raw(x):
            return sess.run(None, {inp: x})

        def decode_raw(r):
            return r[0]
    else:
        raise SystemExit(f"Unexpected output count {n_out} for {model_path}")

    x, pad, scale = yc.letterbox(source_img, imgsz)

    # Warm-up (3 passes, not timed)
    for _ in range(3):
        run_raw(x)

    # Timed inference (sess.run strictly alone per CLAUDE.md invariant)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        raw = run_raw(x)
        ts.append((time.perf_counter() - t0) * 1000.0)

    # Timed post-process (decode + NMS)
    t1 = time.perf_counter()
    out = decode_raw(raw)
    dets = yc.postprocess(out, pad, scale, conf_thres=conf, iou_thres=0.5)
    post_ms = (time.perf_counter() - t1) * 1000.0

    annotated = yc.draw(source_img.copy(), dets)
    return dets, float(np.mean(ts)), float(np.median(ts)), post_ms, annotated


def main():
    ap = argparse.ArgumentParser(
        description="Head-to-head hardware showdown: CPU vs DirectML iGPU vs Ryzen AI NPU."
    )
    ap.add_argument("--source", default="assets/test_image.jpg",
                    help="path to input image (default: assets/test_image.jpg)")
    ap.add_argument("--runs", type=int, default=30,
                    help="timed benchmark runs per pathway (default: 30)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence threshold (default: 0.25)")
    ap.add_argument("--out-dir", default="results",
                    help="output directory for composite image (default: results)")
    ap.add_argument("--log", type=int, default=2,
                    help="ORT log severity (default 2=warning)")
    args = ap.parse_args()

    src = ROOT / args.source if not os.path.isabs(args.source) else Path(args.source)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"Cannot read image: {src}")

    print(f"\n{'='*75}")
    print("  TRI-HARDWARE SHOWDOWN: CPU vs DirectML (780M iGPU) vs Ryzen AI NPU")
    print(f"  Image: {src.name} ({img.shape[1]}x{img.shape[0]}) | Runs: {args.runs} | Conf: {args.conf}")
    print(f"{'='*75}\n")

    results = []
    panels = []

    for label, dev_name, relpath, ep, cache_key, pub_map, effort in CONFIGS:
        model_path = ROOT / relpath
        if not model_path.exists():
            print(f"  [{label}] {relpath} not found, skipping...")
            continue

        print(f"  [{label:<9}] Benchmarking on {dev_name}...", end="", flush=True)
        dets, mean_ms, median_ms, post_ms, annotated = run_pathway(
            model_path, ep, cache_key, img, args.runs, args.conf, args.log
        )
        fps = 1000.0 / mean_ms
        print(f"  {mean_ms:6.2f} ms infer (median {median_ms:6.2f} ms, {fps:5.1f} fps) | {len(dets)} dets")

        # Create header card for image panel
        h, w = img.shape[:2]
        hdr = np.zeros((46, w, 3), dtype=np.uint8)
        hdr_text = f"{label} ({dev_name}): {mean_ms:.1f} ms ({fps:.0f} fps) | {len(dets)} dets"
        cv2.putText(hdr, hdr_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        panel = np.vstack([hdr, annotated])
        panels.append(panel)

        results.append({
            "label": label,
            "device": dev_name,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "fps": fps,
            "post_ms": post_ms,
            "n_dets": len(dets),
            "pub_map": pub_map,
            "effort": effort,
        })

    if not results:
        raise SystemExit("No benchmark configurations succeeded.")

    # ------------------------------------------------------------------ #
    # Print the Comparison Dashboard                                     #
    # ------------------------------------------------------------------ #
    cpu_ms = results[0]["mean_ms"]
    max_ms = max(r["mean_ms"] for r in results)

    print("\n" + "=" * 80)
    print(f"  {'Hardware Pathway':<14} {'Device':<18} {'Infer (ms)':>10} {'FPS':>7} {'vs CPU':>7} {'mAP@50-95':>10}  Latency Bar")
    print("-" * 80)
    for r in results:
        speedup = f"{cpu_ms / r['mean_ms']:.2f}x"
        bar = _bar(r["mean_ms"] / max_ms, width=18)
        print(f"  {r['label']:<14} {r['device']:<18} {r['mean_ms']:>9.2f}ms {r['fps']:>6.1f} {speedup:>7} {r['pub_map']:>9.2f}%  {bar}")
    print("=" * 80)

    # Key Engineering Insights (Direct from docs/BENCHMARKS.md)
    dml_fp16 = next((r for r in results if r["label"] == "DML FP16"), None)
    npu_row = next((r for r in results if r["label"] == "NPU XINT8"), None)

    print("\n  KEY ENGINEERING TAKEAWAYS (docs/BENCHMARKS.md:250):")
    if dml_fp16 and npu_row:
        ratio = dml_fp16["mean_ms"] / npu_row["mean_ms"]
        map_delta = dml_fp16["pub_map"] - npu_row["pub_map"]
        print(f"  1. RAW SPEED: NPU wins by {ratio:.2f}x over DirectML FP16 ({npu_row['mean_ms']:.1f}ms vs {dml_fp16['mean_ms']:.1f}ms).")
        print(f"  2. ACCURACY:  DirectML FP16 retains full FP32 accuracy (+{map_delta:.2f} mAP@50-95 over AdaRound INT8).")
        print("  3. TRADEOFF:  DirectML requires 0 quantization effort (one-line convert, no calibration/head-cut).")
        print("                NPU is worth it when the last 30-40% of latency is critical; otherwise iGPU FP16 is optimal.")

    # ------------------------------------------------------------------ #
    # Assemble 2x2 Composite Image                                       #
    # ------------------------------------------------------------------ #
    if len(panels) == 4:
        top_row = np.hstack([panels[0], panels[1]])
        bot_row = np.hstack([panels[2], panels[3]])
        composite = np.vstack([top_row, bot_row])

        # Bottom legend banner
        foot_h = 40
        footer = np.zeros((foot_h, composite.shape[1], 3), dtype=np.uint8)
        foot_text = "Tri-Hardware Showdown: Zen 4 CPU vs Radeon 780M iGPU (DirectML) vs Phoenix XDNA1 NPU (Ryzen AI)"
        cv2.putText(footer, foot_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        composite = np.vstack([composite, footer])

        out_dir = ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "tri_hardware_showdown.jpg"
        cv2.imwrite(str(out_file), composite)
        print(f"\n  Saved 4-quadrant visual showdown -> {out_file}")
    print()


if __name__ == "__main__":
    main()
