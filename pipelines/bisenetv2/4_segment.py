"""Benchmark BiSeNetV2 inference latency across CPU, iGPU (DirectML), and NPU.

Measures sess.run alone (excluding preprocess and postprocess) over multiple
iterations to record precise hardware latency distributions.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from npu.bisenetv2 import (
    INPUT_SIZE,
    input_size,
    overlay_segmentation,
    postprocess_mask,
    preprocess,
)
from npu.paths import BISENETV2_CACHE_KEY, BISENETV2_BILINEAR_CACHE_KEY, MODELS
from npu.session import build_session, clear_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "bisenetv2_fp32_xint8.onnx"))
    ap.add_argument("--image", default=str(REPO_ROOT / "data" / "bisenetv2_val" / "000000000139.jpg"))
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--cache-key", default=None)
    ap.add_argument("--out-image", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"Model {args.model} not found")

    cache_key = args.cache_key or (
        BISENETV2_BILINEAR_CACHE_KEY if "bilinear" in args.model else BISENETV2_CACHE_KEY
    )
    if args.fresh and args.ep == "npu":
        clear_cache(cache_key)

    print(f"Building session: model={args.model}, ep={args.ep}, cache_key={cache_key}")
    sess = build_session(args.model, ep=args.ep, cache_key=cache_key, xclbin=args.xclbin)

    img_path = args.image
    if not os.path.isfile(img_path):
        # Fallback to any JPG in val or coco
        cands = list((REPO_ROOT / "data" / "bisenetv2_val").glob("*.jpg")) or list((REPO_ROOT / "data" / "coco" / "val2017").glob("*.jpg"))
        if not cands:
            raise SystemExit("No test image found")
        img_path = str(cands[0])

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise SystemExit(f"Failed to read image {img_path}")

    input_name = sess.get_inputs()[0].name
    tensor, orig_shape = preprocess(img_bgr, target_size=INPUT_SIZE)

    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        _ = sess.run(None, {input_name: tensor})

    print(f"Benchmarking ({args.iters} runs, sess.run only)...")
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: tensor})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    mean_ms = float(np.mean(times))
    median_ms = float(np.median(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms

    print("\n--- BiSeNetV2 Inference Benchmark Results ---")
    print(f"Model:      {args.model}")
    print(f"EP:         {args.ep.upper()}")
    print(f"Iterations: {args.iters}")
    print(f"Mean:       {mean_ms:.2f} ms ({fps:.1f} fps)")
    print(f"Median:     {median_ms:.2f} ms")
    print(f"Min / Max:  {min_ms:.2f} ms / {max_ms:.2f} ms")
    print(f"StdDev:     {std_ms:.2f} ms")

    # Save output visualization
    out_img_path = args.out_image or str(REPO_ROOT / "results" / f"bisenetv2_segment_{args.ep}.jpg")
    mask = postprocess_mask(out[0], orig_shape=orig_shape)
    overlay = overlay_segmentation(img_bgr, mask, alpha=0.5)
    os.makedirs(os.path.dirname(out_img_path) or ".", exist_ok=True)
    cv2.imwrite(out_img_path, overlay)
    print(f"Saved visual segmentation overlay to {out_img_path}")


if __name__ == "__main__":
    main()
