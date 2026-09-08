"""
Step 5: Quantitative evaluation of MiDaS monocular depth estimation.

Compares quantized XINT8 models (stock bilinear or NPU-optimized nearest) against
the FP32 reference model across a dataset of test scenes (data/midas_val).

Metrics evaluated:
  - Pearson correlation (r) -- preserves depth ordering and structural geometry
  - Mean Absolute Difference (MAD) in 8-bit normalized depth space [0, 255]
  - Root Mean Squared Error (RMSE)
  - Threshold accuracy (delta < 1.25 and delta < 1.25^2)
  - Per-frame inference latency (sess.run only)

Run in resnet_env17:
    python pipelines/midas/5_eval.py --val-dir data/midas_val --ep npu
    python pipelines/midas/5_eval.py --val-dir data/midas_val --ep cpu
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.midas import INPUT_SIZE, input_size, postprocess_depth, preprocess
from npu.paths import MIDAS_CACHE_KEY, MIDAS_NEAREST_CACHE_KEY, MODELS
from npu.session import build_session, clear_cache

DEFAULT_REF = MODELS / "midas_small_cut.onnx"
DEFAULT_TEST = MODELS / "midas_small_nearest_cut_xint8.onnx"


def compute_metrics(ref_depth, pred_depth):
    """Computes relative depth fidelity metrics between ref and pred 2D depth maps."""
    r_flat = ref_depth.ravel().astype(np.float64)
    p_flat = pred_depth.ravel().astype(np.float64)

    # Pearson correlation
    corr = float(np.corrcoef(r_flat, p_flat)[0, 1]) if np.std(r_flat) > 1e-6 and np.std(p_flat) > 1e-6 else 0.0

    # Min-max normalized 8-bit scale [0, 255] for MAD and RMSE
    r_norm = postprocess_depth(ref_depth, normalize=True).astype(np.float64)
    p_norm = postprocess_depth(pred_depth, normalize=True).astype(np.float64)

    mad = float(np.mean(np.abs(r_norm - p_norm)))
    rmse = float(np.sqrt(np.mean((r_norm - p_norm) ** 2)))

    # Threshold accuracy: ratio max(p/r, r/p) < 1.25
    # Avoid div by zero by adding epsilon
    eps = 1e-3
    r_pos = np.clip(r_norm, eps, 255.0)
    p_pos = np.clip(p_norm, eps, 255.0)
    ratio = np.maximum(p_pos / r_pos, r_pos / p_pos)
    d1 = float(np.mean(ratio < 1.25)) * 100.0
    d2 = float(np.mean(ratio < (1.25 ** 2))) * 100.0

    return corr, mad, rmse, d1, d2


def main():
    ap = argparse.ArgumentParser(description="Evaluate MiDaS monocular depth estimation fidelity.")
    ap.add_argument("--ref-model", default=str(DEFAULT_REF), help="FP32 reference ONNX model")
    ap.add_argument("--test-model", default=str(DEFAULT_TEST), help="Quantized ONNX model to evaluate")
    ap.add_argument("--val-dir", default=str(Path("data") / "midas_val"), help="Validation images directory")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider for test model")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache before evaluation")
    ap.add_argument("--n", type=int, default=50, help="Number of images to evaluate (default: 50)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.val_dir, "*.jpg")))[:args.n]
    if not files:
        # Fallback to data/calib if midas_val empty
        files = sorted(glob.glob("data/calib/*.jpg"))[:args.n]
    if not files:
        raise SystemExit(f"No validation JPGs found in {args.val_dir} or data/calib")

    print(f"Evaluating {len(files)} validation images...")
    print(f"Reference (FP32 CPU): {args.ref_model}")
    print(f"Test model ({args.ep.upper()}): {args.test_model}")

    # Build reference session on CPU
    ref_sess = build_session(args.ref_model, ep="cpu", cache_key="midas_ref")
    ref_in = ref_sess.get_inputs()[0].name

    # Build test session
    cache_key = MIDAS_NEAREST_CACHE_KEY if "nearest" in Path(args.test_model).stem else MIDAS_CACHE_KEY
    if args.fresh and args.ep == "npu":
        clear_cache(cache_key)
    test_sess = build_session(args.test_model, ep=args.ep, cache_key=cache_key)
    test_in = test_sess.get_inputs()[0].name

    target_size = input_size(test_sess.get_inputs()[0].shape, where="test_session")

    # Warmup
    dummy, _ = preprocess(cv2.imread(files[0]), target_size)
    for _ in range(5):
        test_sess.run(None, {test_in: dummy})

    corrs, mads, rmses, d1s, d2s = [], [], [], [], []
    infer_times = []

    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        tensor, orig_shape = preprocess(img, target_size)

        # Ref FP32 inference
        ref_out = ref_sess.run(None, {ref_in: tensor})[0]

        # Test model inference (timed)
        t0 = time.perf_counter()
        test_out = test_sess.run(None, {test_in: tensor})[0]
        t_infer = time.perf_counter() - t0
        infer_times.append(t_infer)

        corr, mad, rmse, d1, d2 = compute_metrics(ref_out, test_out)
        corrs.append(corr)
        mads.append(mad)
        rmses.append(rmse)
        d1s.append(d1)
        d2s.append(d2)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{i + 1}/{len(files)}] Corr: {np.mean(corrs):.4f} | MAD: {np.mean(mads):.2f} | "
                  f"Latency: {np.mean(infer_times) * 1000.0:.2f} ms")

    mean_latency = float(np.mean(infer_times)) * 1000.0
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0

    print("\n=======================================================")
    print(f"MiDaS v2.1 Small Evaluation Results ({args.ep.upper()})")
    print(f"Images evaluated      : {len(corrs)}")
    print(f"Mean Latency (infer)  : {mean_latency:.2f} ms ({fps:.1f} fps)")
    print(f"Pearson Correlation r : {np.mean(corrs):.4f}")
    print(f"Mean Absolute Diff MAD: {np.mean(mads):.2f} / 255")
    print(f"Root Mean Squared RMSE: {np.mean(rmses):.2f} / 255")
    print(f"Delta < 1.25 Accuracy : {np.mean(d1s):.2f} %")
    print(f"Delta < 1.25^2 Acc.   : {np.mean(d2s):.2f} %")
    print("=======================================================")


if __name__ == "__main__":
    main()
