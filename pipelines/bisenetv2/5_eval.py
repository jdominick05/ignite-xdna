"""Evaluate BiSeNetV2 segmentation fidelity against FP32 CPU reference across 50 scenes.

Calculates:
  1. Mean Pixel Accuracy (class agreement)
  2. Mean Intersection over Union (mIoU) across 19 Cityscapes classes
  3. Mean Absolute Difference (MAD) and RMSE on Softmax probabilities
  4. Per-scene inference latency
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from npu.bisenetv2 import (
    CITYSCAPES_CLASSES,
    INPUT_SIZE,
    postprocess_mask,
    preprocess,
)
from npu.paths import BISENETV2_CACHE_KEY, BISENETV2_BILINEAR_CACHE_KEY, MODELS
from npu.session import build_session, clear_cache


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_iou(mask_pred, mask_ref, num_classes=19):
    ious = []
    for c in range(num_classes):
        pred_c = (mask_pred == c)
        ref_c = (mask_ref == c)
        intersection = np.logical_and(pred_c, ref_c).sum()
        union = np.logical_or(pred_c, ref_c).sum()
        if union > 0:
            ious.append(float(intersection) / float(union))
    return np.mean(ious) if ious else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "bisenetv2_fp32_xint8.onnx"))
    ap.add_argument("--ref", default=str(MODELS / "bisenetv2_fp32.onnx"))
    ap.add_argument("--val-dir", default=str(REPO_ROOT / "data" / "bisenetv2_val"))
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--cache-key", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"Model {args.model} not found")
    if not os.path.isfile(args.ref):
        raise SystemExit(f"Reference model {args.ref} not found")

    cache_key = args.cache_key or (
        BISENETV2_BILINEAR_CACHE_KEY if "bilinear" in args.model else BISENETV2_CACHE_KEY
    )
    if args.fresh and args.ep == "npu":
        clear_cache(cache_key)

    print(f"Building test session: model={args.model}, ep={args.ep}, cache_key={cache_key}")
    sess_test = build_session(args.model, ep=args.ep, cache_key=cache_key, xclbin=args.xclbin)

    print(f"Building reference session on CPU: model={args.ref}")
    sess_ref = build_session(args.ref, ep="cpu", cache_key="bisenetv2_cpu_ref")

    images = sorted(glob.glob(os.path.join(args.val_dir, "*.jpg")))[:args.limit]
    if not images:
        raise SystemExit(f"No JPG images found in {args.val_dir}")
    print(f"Evaluating across {len(images)} validation scenes...")

    input_name_test = sess_test.get_inputs()[0].name
    input_name_ref = sess_ref.get_inputs()[0].name

    pixel_accs = []
    mious = []
    prob_mads = []
    prob_rmses = []
    infer_times = []

    for i, img_path in enumerate(images):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        tensor, orig_shape = preprocess(img_bgr, target_size=INPUT_SIZE)

        # Reference CPU inference
        ref_out = sess_ref.run(None, {input_name_ref: tensor})[0]

        # Test model inference
        t0 = time.perf_counter()
        test_out = sess_test.run(None, {input_name_test: tensor})[0]
        t1 = time.perf_counter()
        infer_times.append((t1 - t0) * 1000.0)

        # Postprocess masks
        mask_test = postprocess_mask(test_out)
        mask_ref = postprocess_mask(ref_out)

        # Pixel agreement
        pixel_acc = float(np.mean(mask_test == mask_ref) * 100.0)
        pixel_accs.append(pixel_acc)

        # mIoU
        miou = compute_iou(mask_test, mask_ref, num_classes=len(CITYSCAPES_CLASSES)) * 100.0
        mious.append(miou)

        # Softmax probability fidelity
        prob_test = softmax(test_out, axis=1)
        prob_ref = softmax(ref_out, axis=1)
        diff = np.abs(prob_test - prob_ref)
        prob_mads.append(float(np.mean(diff)))
        prob_rmses.append(float(np.sqrt(np.mean(diff ** 2))))

        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            print(f"[{i+1:2d}/{len(images)}] Mean Acc: {np.mean(pixel_accs):.2f}%, mIoU: {np.mean(mious):.2f}%, Infer: {np.mean(infer_times):.2f} ms")

    print("\n=======================================================")
    print("      BiSeNetV2 Quantitative Segmentation Fidelity     ")
    print("=======================================================")
    print(f"Test Model:      {args.model}")
    print(f"Execution Target:{args.ep.upper()}")
    print(f"Reference Model: {args.ref} (CPU FP32)")
    print(f"Evaluated Scenes:{len(images)}")
    print(f"Pixel Accuracy:  {np.mean(pixel_accs):.2f}% +/- {np.std(pixel_accs):.2f}%")
    print(f"Mean IoU (mIoU): {np.mean(mious):.2f}% +/- {np.std(mious):.2f}%")
    print(f"Prob MAD:        {np.mean(prob_mads):.5f}")
    print(f"Prob RMSE:       {np.mean(prob_rmses):.5f}")
    print(f"Mean Latency:    {np.mean(infer_times):.2f} ms ({1000.0 / np.mean(infer_times):.1f} fps)")
    print(f"Median Latency:  {np.median(infer_times):.2f} ms")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
