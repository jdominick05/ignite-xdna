"""
ResNet50 Resolution Ladder Demo on AMD XDNA1 NPU.

Demonstrates the architectural finding documented in docs/BENCHMARKS.md:
"ResNet50 input resolution: does the fixed cost story hold for a classifier?"

Sweeps input resolutions across the exact same model architecture:
  128² (1.35 GMACs) -> 160² (2.11 GMACs) -> 224² (4.14 GMACs) -> 288² (6.84 GMACs) -> 384² (12.18 GMACs)

Key Insight:
  * 128² -> 224² is a 3.06x arithmetic increase (1.35 -> 4.14 GMACs), but latency only increases
    by ~1.16x (~4.5 ms -> ~5.2 ms).
  * Why? The NPU has a fixed hardware execution floor (~4.0 ms) across its 393 on-NPU nodes
    governed by AIE overlay dispatch, DMA descriptor programming, and graph synchronization.
  * At small resolutions (<224²), arithmetic is practically free because the hardware spends
    most of its time in the fixed dispatch floor. Past 288², compute finally overtakes the floor
    and scales proportionally to pixel area O(r²).

Renders an annotated resolution comparison strip to results/resolution_ladder_npu.jpg.

Usage:
    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/resolution_ladder_demo.py
    python tools/resolution_ladder_demo.py --runs 40 --source data/calib/000000.jpg
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

DEFAULT_IMG = ROOT / "data" / "calib" / "000000.jpg"
LABELS_FILE = ROOT / "data" / "calib" / "labels.json"

RESOLUTIONS = [
    (128, "models/resnet50_r128_xint8_c64.onnx", "models/preprocess_config_r128.json", 1.35),
    (160, "models/resnet50_r160_xint8_c64.onnx", "models/preprocess_config_r160.json", 2.11),
    (224, "models/resnet50_xint8_c64.onnx", "models/preprocess_config.json", 4.14),
    (288, "models/resnet50_r288_xint8_c64.onnx", "models/preprocess_config_r288.json", 6.84),
    (384, "models/resnet50_r384_xint8_c64.onnx", "models/preprocess_config_r384.json", 12.18),
]


def _bar(val, max_val, width=20):
    frac = min(val / max_val, 1.0) if max_val > 0 else 0
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def run_resolution_point(model_path, cfg_path, img_path, runs, ep, log_sev):
    """Compile, warm-up, and benchmark one resolution point on NPU."""
    clear_cache(RESNET_CACHE_KEY)
    sess = build_session(str(model_path), ep, RESNET_CACHE_KEY, log_severity=log_sev)
    inp_name = sess.get_inputs()[0].name

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    transform = build_transform(cfg)
    tensor = transform(img_path)  # (1, 3, r, r)

    # Warm-up (3 passes)
    for _ in range(3):
        sess.run(None, {inp_name: tensor})

    # Timed inference (sess.run strictly alone per CLAUDE.md invariant)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        raw = sess.run(None, {inp_name: tensor})
        ts.append((time.perf_counter() - t0) * 1000.0)

    top_cls = int(np.argmax(raw[0][0]))
    conf_score = float(np.max(raw[0][0]))

    return float(np.mean(ts)), float(np.median(ts)), top_cls, conf_score, cfg["input_size"][1]


def main():
    ap = argparse.ArgumentParser(
        description="ResNet50 resolution ladder & fixed hardware dispatch floor demo on NPU."
    )
    ap.add_argument("--source", default=str(DEFAULT_IMG))
    ap.add_argument("--ep", choices=["npu", "cpu"], default="npu")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Input image not found: {src}")

    bgr_orig = cv2.imread(str(src))

    print(f"\n{'='*78}")
    print(f"  RESNET50 INPUT RESOLUTION SWEEP: FIXED HARDWARE FLOOR DEMO ({args.ep.upper()})")
    print(f"  Image: {src.name} | Timed Runs: {args.runs} | Backend: AMD XDNA1 NPU")
    print(f"{'='*78}\n")

    records = []
    panels = []

    for res, rel_model, rel_cfg, gmacs in RESOLUTIONS:
        model_path = ROOT / rel_model
        cfg_path = ROOT / rel_cfg
        if not model_path.exists() or not cfg_path.exists():
            print(f"  [{res}x{res}] Files missing, skipping...")
            continue

        print(f"  [{res}x{res:<3}] Compiling and benchmarking ({gmacs:5.2f} GMACs)...", end="", flush=True)
        mean_ms, med_ms, top_cls, top_val, side = run_resolution_point(
            model_path, cfg_path, src, args.runs, args.ep, args.log
        )
        fps = 1000.0 / mean_ms
        print(f"  {mean_ms:5.2f} ms ({fps:5.1f} fps) | Class: {top_cls}")

        # Crop preview thumbnail
        crop_thumb = cv2.resize(bgr_orig, (220, 220), interpolation=cv2.INTER_AREA)
        cv2.putText(crop_thumb, f"{res}x{res}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(crop_thumb, f"{mean_ms:.2f} ms", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(crop_thumb, f"{gmacs:.2f} GMACs", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        panels.append(crop_thumb)

        records.append({
            "res": res,
            "gmacs": gmacs,
            "mean_ms": mean_ms,
            "med_ms": med_ms,
            "fps": fps,
            "top_cls": top_cls,
        })

    if not records:
        raise SystemExit("No resolution configurations succeeded.")

    # ------------------------------------------------------------------ #
    # Print Architectural Comparison Table                               #
    # ------------------------------------------------------------------ #
    base_ms = records[0]["mean_ms"]
    base_gmacs = records[0]["gmacs"]
    max_ms = max(r["mean_ms"] for r in records)

    print("\n" + "=" * 80)
    print(f"  {'Resolution':<12} {'GMACs':>7} {'GMACs vs 128':>14} {'Latency':>10} {'Lat vs 128':>12} {'FPS':>7}  Hardware Bar")
    print("-" * 80)
    for r in records:
        gmac_ratio = f"{r['gmacs'] / base_gmacs:.2f}x"
        lat_ratio = f"{r['mean_ms'] / base_ms:.2f}x"
        bar = _bar(r["mean_ms"], max_ms, width=16)
        print(f"  {r['res']}x{r['res']:<7} {r['gmacs']:>7.2f} {gmac_ratio:>14} {r['mean_ms']:>8.2f}ms {lat_ratio:>12} {r['fps']:>6.1f}  {bar}")
    print("=" * 80)

    # Key Architectural Takeaways
    p128 = next((r for r in records if r["res"] == 128), None)
    p224 = next((r for r in records if r["res"] == 224), None)
    p384 = next((r for r in records if r["res"] == 384), None)

    print("\n  ARCHITECTURAL INSIGHTS (docs/BENCHMARKS.md:200):")
    if p128 and p224:
        g_scale = p224["gmacs"] / p128["gmacs"]
        l_scale = p224["mean_ms"] / p128["mean_ms"]
        print(f"  1. DISPATCH FLOOR (128² -> 224²): {g_scale:.2f}x FLOPs increase costs only {l_scale:.2f}x latency!")
        print(f"     ~80% of execution time at 128² ({p128['mean_ms']:.2f} ms) is fixed hardware dispatch/DMA floor.")
    if p224 and p384:
        g_scale2 = p384["gmacs"] / p224["gmacs"]
        l_scale2 = p384["mean_ms"] / p224["mean_ms"]
        print(f"  2. COMPUTE KNEE (224² -> 384²): {g_scale2:.2f}x FLOPs scales latency by {l_scale2:.2f}x.")
        print("     Past 288², compute arithmetic overtakes the dispatch floor and scales linearly with pixels.")

    # ------------------------------------------------------------------ #
    # Render Visual Strip                                                #
    # ------------------------------------------------------------------ #
    if len(panels) >= 3:
        composite = np.hstack(panels)
        hdr = np.zeros((36, composite.shape[1], 3), dtype=np.uint8)
        cv2.putText(hdr, "ResNet50 Resolution Sweep: 128x128 -> 384x384 on Phoenix XDNA1 NPU",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        full_strip = np.vstack([hdr, composite])

        out_dir = ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "resolution_ladder_npu.jpg"
        cv2.imwrite(str(out_file), full_strip)
        print(f"\n  Saved resolution comparison strip -> {out_file}\n")


if __name__ == "__main__":
    main()
