"""
Step 4: Run SESR-M7 2x super-resolution on CPU, DirectML (iGPU), or Ryzen AI NPU.

Supports single-image upscaling, arbitrary-size tiled super-resolution with
seamless overlap reconstruction, and isolated latency benchmarking
(sess.run timed alone, separated from pre/post, per the project invariant).

Run in resnet_env17 (has ONNX Runtime with VitisAI EP + DML):
    # Benchmark on NPU:
    python pipelines/sesr/4_upscale.py --ep npu --fresh --n 50

    # Upscale a test image on NPU:
    python pipelines/sesr/4_upscale.py --ep npu --image data/sesr_val/Set5_LR_x2/butterfly.png --out results/butterfly_sesr_npu.png

    # CPU FP32 baseline:
    python pipelines/sesr/4_upscale.py --ep cpu --model models/sesr_m7_fp32.onnx --n 50

    # DirectML (Radeon 780M iGPU):
    python pipelines/sesr/4_upscale.py --ep dml --model models/sesr_m7_fp32.onnx --n 50
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.paths import MODELS, RESULTS, SESR_CACHE_KEY, SESR_ADAROUND_CACHE_KEY
from npu.sesr import (
    INPUT_SIZE,
    SCALE,
    compute_psnr,
    compute_ssim,
    input_size,
    merge_tiles,
    postprocess,
    preprocess,
    split_into_tiles,
)
from npu.session import build_session, clear_cache

DEFAULT_MODEL = MODELS / "sesr_m7_xint8.onnx"


def upscale_image(sess, input_name, img_bgr, target_tile_size=INPUT_SIZE, overlap=8):
    """Upscale arbitrary-sized image. If larger than tile size, splits and merges tiles."""
    h, w = img_bgr.shape[:2]
    if (h, w) == (target_tile_size, target_tile_size):
        tensor, _ = preprocess(img_bgr, target_tile_size)
        out_raw = sess.run(None, {input_name: tensor})[0]
        return postprocess(out_raw)

    tiles, orig_hw, padded_hw, grid_hw = split_into_tiles(
        img_bgr, patch_size=(target_tile_size, target_tile_size), overlap=overlap
    )
    sr_tiles = []
    for tile in tiles:
        tensor, _ = preprocess(tile, target_tile_size)
        out_raw = sess.run(None, {input_name: tensor})[0]
        sr_tiles.append(postprocess(out_raw))

    return merge_tiles(sr_tiles, orig_hw, padded_hw, grid_hw, scale=SCALE, overlap=overlap)


def main():
    ap = argparse.ArgumentParser(description="Run SESR-M7 super-resolution.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to ONNX model")
    ap.add_argument("--image", default=None, help="Path to input image")
    ap.add_argument("--out", default=None, help="Path to save super-resolved output")
    ap.add_argument("--n", type=int, default=1, help="Number of benchmark iterations (default: 1)")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache before building session")
    ap.add_argument("--overlap", type=int, default=8, help="Tile overlap in pixels for seamless tiling")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model file not found: {model_path}")

    cache_key = SESR_ADAROUND_CACHE_KEY if "adaround" in model_path.lower() else SESR_CACHE_KEY
    if args.fresh and args.ep == "npu":
        clear_cache(cache_key)

    print(f"Building session (ep={args.ep}, cache_key={cache_key})...")
    sess = build_session(model_path, ep=args.ep, cache_key=cache_key)

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    target_size = input_size(input_shape, where="session")
    print(f"Session ready. Model input: {input_name} {input_shape}")

    # Benchmark mode: isolated sess.run on static tile shape
    dummy_input, _ = preprocess(np.zeros((target_size, target_size, 3), dtype=np.uint8), target_size)

    # Warmup
    warmup_n = 10 if args.n > 1 else 1
    for _ in range(warmup_n):
        sess.run(None, {input_name: dummy_input})

    if args.n > 1:
        print(f"\nBenchmarking {args.n} iterations on {args.ep.upper()} (sess.run alone, {target_size}x{target_size})...")
        latencies = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            sess.run(None, {input_name: dummy_input})
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p90 = np.percentile(latencies, 90)
        mean_lat = np.mean(latencies)
        fps = 1000.0 / mean_lat
        print(f"  Mean latency: {mean_lat:.3f} ms ({fps:.1f} fps)")
        print(f"  P50 latency:  {p50:.3f} ms")
        print(f"  P90 latency:  {p90:.3f} ms")
        print(f"  Min / Max:    {latencies.min():.3f} ms / {latencies.max():.3f} ms")

    # Image inference mode
    img_path = args.image
    if img_path is None:
        # Default test image if none provided
        candidate = Path("data") / "sesr_val" / "Set5_LR_x2" / "butterfly.png"
        if candidate.exists():
            img_path = str(candidate)

    if img_path and os.path.isfile(img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise SystemExit(f"Failed to read image: {img_path}")
        h, w = img_bgr.shape[:2]
        print(f"\nProcessing image {img_path} ({w}x{h})...")

        t0 = time.perf_counter()
        sr_bgr = upscale_image(sess, input_name, img_bgr, target_tile_size=target_size, overlap=args.overlap)
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        sr_h, sr_w = sr_bgr.shape[:2]
        print(f"Upscaled to {sr_w}x{sr_h} in {total_time_ms:.2f} ms")

        # If ground truth HR counterpart exists, compute PSNR/SSIM
        hr_candidate = Path("data") / "sesr_val" / "Set5_HR" / f"{Path(img_path).stem}.png"
        if not hr_candidate.exists():
            hr_candidate = Path("data") / "sesr_val" / "Set14_HR" / f"{Path(img_path).stem}.png"
        if hr_candidate.exists():
            hr_bgr = cv2.imread(str(hr_candidate))
            if hr_bgr is not None:
                # Align if needed
                min_h = min(sr_bgr.shape[0], hr_bgr.shape[0])
                min_w = min(sr_bgr.shape[1], hr_bgr.shape[1])
                sr_crop = sr_bgr[:min_h, :min_w]
                hr_crop = hr_bgr[:min_h, :min_w]
                psnr_y = compute_psnr(sr_crop, hr_crop, y_channel=True)
                ssim_y = compute_ssim(sr_crop, hr_crop, y_channel=True)
                print(f"Fidelity against HR ground truth ({hr_candidate.name}):")
                print(f"  PSNR (Y): {psnr_y:.2f} dB")
                print(f"  SSIM (Y): {ssim_y:.4f}")

        out_path = args.out or str(RESULTS / f"{Path(img_path).stem}_sesr_{args.ep}.png")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, sr_bgr)
        print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
