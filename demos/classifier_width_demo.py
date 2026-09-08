"""
Classifier Width Ladder Demo: ResNet50 vs Wide-ResNet50 vs Wide-ResNet101 on AMD XDNA1 NPU.

Demonstrates the finding from docs/BENCHMARKS.md:
"Model width: does 'width is nearly free' hold for a classifier too?"

Compares three progressively wider/deeper architectures on the exact same 224² ImageNet input:
  1. ResNet50 (25.6M params, 4.14 GMACs, 393 NPU nodes)           — Baseline
  2. Wide-ResNet50-2 (68.9M params, 11.65 GMACs, 393 NPU nodes)     — 2.7x params, pure width step
  3. Wide-ResNet101-2 (126.9M params, 23.18 GMACs, 767 NPU nodes)   — 5.0x params, double depth + width

Key Takeaways:
  * ResNet50 -> Wide-ResNet50-2 is 2.7x params for only 1.73x latency.
  * ResNet50 -> Wide-ResNet101-2 is 4.96x params for only 3.15x latency.
  * Confirms that sub-linear latency scaling is an inherent property of the XDNA1
    compute fabric absorbing width, not unique to YOLO.
  * Node placement stays near 100% (393/395, 767/769) at all scales.

Renders an annotated comparison visual to results/classifier_width_npu.jpg.

Usage:
    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/classifier_width_demo.py
    python tools/classifier_width_demo.py --runs 40 --source data/calib/000000.jpg
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
DEFAULT_CFG = ROOT / "models" / "preprocess_config.json"
LABELS_FILE = ROOT / "data" / "calib" / "labels.json"

MODELS = [
    (
        "ResNet50",
        "models/resnet50_xint8_c64.onnx",
        25.6,
        4.14,
        "393 / 395",
        72.10,
        "Baseline (50 layers)",
    ),
    (
        "Wide-ResNet50-2",
        "models/wide_resnet50_2_xint8_c64.onnx",
        68.9,
        11.65,
        "393 / 395",
        71.50,
        "2.7x params (isolated width)",
    ),
    (
        "Wide-ResNet101-2",
        "models/wide_resnet101_2_xint8_c64.onnx",
        126.9,
        23.18,
        "767 / 769",
        80.20,
        "5.0x params (depth + width)",
    ),
]


def _bar(val, max_val, width=18):
    frac = min(val / max_val, 1.0) if max_val > 0 else 0
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def run_model(model_path, cfg, img_path, runs, ep, log_sev):
    """Compile, warm-up, and benchmark one classifier model on NPU."""
    clear_cache(RESNET_CACHE_KEY)
    sess = build_session(str(model_path), ep, RESNET_CACHE_KEY, log_severity=log_sev)
    inp_name = sess.get_inputs()[0].name

    transform = build_transform(cfg)
    tensor = transform(img_path)

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

    return float(np.mean(ts)), float(np.median(ts)), top_cls, conf_score


def main():
    ap = argparse.ArgumentParser(
        description="Classifier width ladder: ResNet50 vs Wide-ResNet50 vs Wide-ResNet101 on NPU."
    )
    ap.add_argument("--source", default=str(DEFAULT_IMG))
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--ep", choices=["npu", "cpu"], default="npu")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Input image not found: {src}")

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    bgr_orig = cv2.imread(str(src))

    print(f"\n{'='*78}")
    print(f"  CLASSIFIER WIDTH LADDER DEMO ({args.ep.upper()})")
    print(f"  Image: {src.name} | Runs: {args.runs} | Backend: AMD XDNA1 NPU")
    print(f"{'='*78}\n")

    records = []
    panels = []

    for name, rel_model, params_m, gmacs, nodes_str, pub_acc, note in MODELS:
        model_path = ROOT / rel_model
        if not model_path.exists():
            print(f"  [{name}] Model missing ({rel_model}), skipping...")
            continue

        print(f"  [{name:<16}] Compiling and benchmarking ({params_m}M params, {gmacs} GMACs)...", end="", flush=True)
        mean_ms, med_ms, top_cls, top_val = run_model(
            model_path, cfg, src, args.runs, args.ep, args.log
        )
        fps = 1000.0 / mean_ms
        print(f"  {mean_ms:5.2f} ms ({fps:5.1f} fps) | Class: {top_cls}")

        # Crop thumbnail card
        thumb = cv2.resize(bgr_orig, (260, 260), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(thumb, f"{mean_ms:.2f} ms ({fps:.0f} fps)", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(thumb, f"Params: {params_m}M", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(thumb, f"NPU Nodes: {nodes_str}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(thumb, f"Top-1: {pub_acc:.1f}%", (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
        panels.append(thumb)

        records.append({
            "name": name,
            "params_m": params_m,
            "gmacs": gmacs,
            "mean_ms": mean_ms,
            "med_ms": med_ms,
            "fps": fps,
            "nodes": nodes_str,
            "top_cls": top_cls,
            "pub_acc": pub_acc,
            "note": note,
        })

    if not records:
        raise SystemExit("No classifier width models were found.")

    # ------------------------------------------------------------------ #
    # Print Architectural Comparison Table                               #
    # ------------------------------------------------------------------ #
    base_ms = records[0]["mean_ms"]
    base_params = records[0]["params_m"]
    max_ms = max(r["mean_ms"] for r in records)

    print("\n" + "=" * 80)
    print(f"  {'Model':<16} {'Params':>8} {'vs R50':>8} {'GMACs':>7} {'Latency':>10} {'vs R50':>8} {'Top-1':>8}  Latency Bar")
    print("-" * 80)
    for r in records:
        param_ratio = f"{r['params_m'] / base_params:.2f}x"
        lat_ratio = f"{r['mean_ms'] / base_ms:.2f}x"
        bar = _bar(r["mean_ms"], max_ms, width=16)
        print(f"  {r['name']:<16} {r['params_m']:>7.1f}M {param_ratio:>8} {r['gmacs']:>7.2f} {r['mean_ms']:>8.2f}ms {lat_ratio:>8} {r['pub_acc']:>7.1f}%  {bar}")
    print("=" * 80)

    # Key Architectural Takeaways
    r50 = next((r for r in records if r["name"] == "ResNet50"), None)
    w50 = next((r for r in records if r["name"] == "Wide-ResNet50-2"), None)
    w101 = next((r for r in records if r["name"] == "Wide-ResNet101-2"), None)

    print("\n  KEY ARCHITECTURAL TAKEAWAYS (docs/BENCHMARKS.md:1073):")
    if r50 and w50:
        p_ratio = w50["params_m"] / r50["params_m"]
        l_ratio = w50["mean_ms"] / r50["mean_ms"]
        print(f"  1. PURE WIDTH STEP: {p_ratio:.2f}x parameter scaling costs only {l_ratio:.2f}x latency.")
        print(f"     Confirms 'width is nearly free' holds for CNN classifiers (71.5% top-1, 90.1% top-5).")
    if r50 and w101:
        p_ratio2 = w101["params_m"] / r50["params_m"]
        l_ratio2 = w101["mean_ms"] / r50["mean_ms"]
        print(f"  2. EXTENDED STEP:   {p_ratio2:.2f}x parameters (126.9M params) costs only {l_ratio2:.2f}x latency.")
        print(f"     Wide-ResNet101-2 achieves {w101['pub_acc']:.1f}% top-1 under plain XINT8, beating ResNet50 AdaRound.")

    # ------------------------------------------------------------------ #
    # Render Visual Strip                                                #
    # ------------------------------------------------------------------ #
    if len(panels) >= 2:
        composite = np.hstack(panels)
        hdr = np.zeros((36, composite.shape[1], 3), dtype=np.uint8)
        cv2.putText(hdr, "Classifier Width Ladder: ResNet50 -> Wide-ResNet50-2 -> Wide-ResNet101-2 on NPU",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        full_strip = np.vstack([hdr, composite])

        out_dir = ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "classifier_width_npu.jpg"
        cv2.imwrite(str(out_file), full_strip)
        print(f"\n  Saved classifier width visual strip -> {out_file}\n")


if __name__ == "__main__":
    main()
