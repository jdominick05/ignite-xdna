"""
Step 5: Evaluate MODNet portrait matting quality (MAD, SAD, MSE) and latency
against the FP32 reference model.

Run in resnet_env17:
    python pipelines/modnet/5_eval.py --ep npu --model models/modnet/modnet_xint8.onnx \
        --val-dir data/modnet_val --fresh
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.modnet import postprocess_matte, preprocess
from npu.paths import MODELS, modnet_cache_key
from npu.session import build_session, clear_cache





DEFAULT_MODEL = MODELS / "modnet" / "modnet_cut_xint8.onnx"
DEFAULT_REF = MODELS / "modnet" / "modnet_fp32.onnx"
DEFAULT_VAL = Path("data/modnet_val")


def main():
    ap = argparse.ArgumentParser(description="Evaluate MODNet matting accuracy and latency.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider for evaluation")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Quantized model to test")
    ap.add_argument("--ref-model", default=str(DEFAULT_REF), help="FP32 reference model for ground truth")
    ap.add_argument("--val-dir", default=str(DEFAULT_VAL), help="Validation images directory")
    ap.add_argument("--limit", type=int, default=50, help="Evaluation image limit (default: 50)")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache before test")
    ap.add_argument("--xclbin", default=None, help="Explicit xclbin path")
    args = ap.parse_args()

    val_files = sorted(list(Path(args.val_dir).glob("*.jpg")) + list(Path(args.val_dir).glob("*.png")))[:args.limit]
    if not val_files:
        raise SystemExit(f"No validation images found in {args.val_dir}")

    cache_key = modnet_cache_key(args.model)
    if args.fresh:
        clear_cache(cache_key)

    print(f"[5_eval] Loading reference model on CPU: {args.ref_model}...")
    ref_session = build_session(args.ref_model, "cpu", "modnet_ref_cache")
    ref_in = ref_session.get_inputs()[0].name
    ref_out = ref_session.get_outputs()[0].name

    print(f"[5_eval] Loading test model on {args.ep.upper()}: {args.model}, Cache={cache_key}...")
    test_session = build_session(args.model, args.ep, cache_key, args.xclbin)
    test_in = test_session.get_inputs()[0].name
    test_out = test_session.get_outputs()[0].name

    # Warmup
    dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
    ref_session.run([ref_out], {ref_in: dummy})
    test_session.run([test_out], {test_in: dummy})

    mads = []
    sads = []
    mses = []
    latencies = []

    print(f"[5_eval] Evaluating {len(val_files)} validation images...")
    for i, p in enumerate(val_files):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        tensor, orig_shape = preprocess(img_bgr, 512)

        # Reference (FP32 CPU)
        ref_raw = ref_session.run([ref_out], {ref_in: tensor})[0]
        ref_matte = postprocess_matte(ref_raw, orig_shape)

        # Test model
        t0 = time.perf_counter()
        test_raw = test_session.run([test_out], {test_in: tensor})[0]
        t_infer = (time.perf_counter() - t0) * 1000
        latencies.append(t_infer)

        test_matte = postprocess_matte(test_raw, orig_shape)

        diff = np.abs(test_matte - ref_matte)
        mad = float(np.mean(diff))
        sad = float(np.sum(diff) / 1000.0)  # Standard SAD scaling (in thousands)
        mse = float(np.mean((test_matte - ref_matte) ** 2))

        mads.append(mad)
        sads.append(sad)
        mses.append(mse)

        if (i + 1) % 10 == 0 or (i + 1) == len(val_files):
            print(f"  [{i+1:2d}/{len(val_files)}] MAD: {np.mean(mads):.5f} | SAD: {np.mean(sads):.2f}k | Latency: {np.mean(latencies):.2f} ms")

    print("\n=======================================================")
    print(f"MODNet Evaluation Summary (EP={args.ep.upper()}, N={len(latencies)})")
    print("=======================================================")
    print(f"Mean Absolute Difference (MAD): {np.mean(mads):.5f}")
    print(f"Sum of Absolute Diff (SAD, 1e3): {np.mean(sads):.2f}")
    print(f"Mean Squared Error (MSE):       {np.mean(mses):.6f}")
    print(f"Inference Latency Mean:         {np.mean(latencies):.2f} ms")
    print(f"Inference Latency Median (P50): {np.percentile(latencies, 50):.2f} ms")
    print(f"Inference Latency P90:          {np.percentile(latencies, 90):.2f} ms")
    print(f"Throughput:                     {1000.0 / np.mean(latencies):.1f} fps")
    print("=======================================================\n")

    # If NPU was used, inspect vitisai_ep_report.json
    if args.ep == "npu":
        report_path = Path(cache_key) / "vitisai_ep_report.json"
        if not report_path.exists():
            report_path = Path("vitisai_ep_report.json")
        if report_path.exists():
            print(f"[5_eval] Reading {report_path}:")
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                dev_stat = rep.get("deviceStat", [])
                npu_nodes = sum(d.get("nodeNum", 0) for d in dev_stat if d.get("name") == "NPU")
                cpu_nodes = sum(d.get("nodeNum", 0) for d in dev_stat if "CPU" in d.get("name", ""))
                total = npu_nodes + cpu_nodes
                pct = (npu_nodes / total * 100) if total > 0 else 0
                print(f"  NPU nodes: {npu_nodes}/{total} ({pct:.1f}%)")
                print(f"  CPU fallback nodes: {cpu_nodes}/{total}")
            except Exception as e:
                print(f"  Could not parse report: {e}")


if __name__ == "__main__":
    main()
