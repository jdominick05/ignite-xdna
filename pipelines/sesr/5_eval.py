"""
Step 5: Quantitative fidelity and accuracy evaluation for SESR-M7.

Evaluates Super-Resolution models against high-resolution ground truth:
  - Bicubic interpolation baseline
  - FP32 reference model
  - Plain Quark XINT8 model
  - AdaRound-recovered XINT8 model

Metrics computed:
  - PSNR (dB) on luminance Y-channel (SISR standard) and RGB
  - SSIM on luminance Y-channel and RGB
  - Per-tile inference latency (ms) and throughput (fps)

Benchmarks:
  - Set5:  5 standard images (baby, bird, butterfly, head, woman)
  - Set14: 14 standard images (baboon, barbara, bridge, comic, lenna...)

Run in resnet_env17:
    python pipelines/sesr/5_eval.py --ep npu --dataset Set5
    python pipelines/sesr/5_eval.py --ep npu --dataset Set14
    python pipelines/sesr/5_eval.py --ep npu --dataset all
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.paths import DATA, MODELS, SESR_CACHE_KEY, SESR_ADAROUND_CACHE_KEY
from npu.sesr import (
    INPUT_SIZE,
    SCALE,
    bgr2y,
    compute_psnr,
    compute_ssim,
    input_size,
    merge_tiles,
    postprocess,
    preprocess,
    split_into_tiles,
)
from npu.session import build_session, clear_cache

DEFAULT_FP32 = MODELS / "sesr_m7_fp32.onnx"
DEFAULT_XINT8 = MODELS / "sesr_m7_xint8.onnx"
DEFAULT_ADAROUND = MODELS / "sesr_m7_xint8_adaround.onnx"
VAL_DIR = DATA / "sesr_val"


def get_image_pairs(dataset_name: str):
    """Load matching (LR_path, HR_path) pairs for a given dataset."""
    hr_dir = VAL_DIR / f"{dataset_name}_HR"
    lr_dir = VAL_DIR / f"{dataset_name}_LR_x2"

    if not hr_dir.exists() or not lr_dir.exists():
        raise SystemExit(f"Dataset directories not found: {hr_dir} or {lr_dir}. Run 2_fetch_data.py first.")

    hr_files = sorted(list(hr_dir.glob("*.png")) + list(hr_dir.glob("*.jpg")))
    lr_files = sorted(list(lr_dir.glob("*.png")) + list(lr_dir.glob("*.jpg")))

    pairs = []
    for hr_p in hr_files:
        stem = hr_p.stem
        # Look for matching LR file
        match = next((lr for lr in lr_files if lr.stem == stem or lr.stem.startswith(stem)), None)
        if match is not None:
            pairs.append((match, hr_p))

    if not pairs:
        raise SystemExit(f"No matching image pairs found in {dataset_name}!")
    return pairs


def run_bicubic_eval(pairs):
    """Evaluate standard Bicubic interpolation baseline."""
    psnr_y_list, ssim_y_list = [], []
    psnr_rgb_list, ssim_rgb_list = [], []

    for lr_p, hr_p in pairs:
        lr = cv2.imread(str(lr_p))
        hr = cv2.imread(str(hr_p))
        h, w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        bic = cv2.resize(lr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
        # Crop to align with HR
        min_h = min(bic.shape[0], hr_h)
        min_w = min(bic.shape[1], hr_w)
        bic = bic[:min_h, :min_w]
        hr_crop = hr[:min_h, :min_w]

        psnr_y_list.append(compute_psnr(bic, hr_crop, y_channel=True))
        ssim_y_list.append(compute_ssim(bic, hr_crop, y_channel=True))
        psnr_rgb_list.append(compute_psnr(bic, hr_crop, y_channel=False))
        ssim_rgb_list.append(compute_ssim(bic, hr_crop, y_channel=False))

    return {
        "psnr_y": float(np.mean(psnr_y_list)),
        "ssim_y": float(np.mean(ssim_y_list)),
        "psnr_rgb": float(np.mean(psnr_rgb_list)),
        "ssim_rgb": float(np.mean(ssim_rgb_list)),
        "lat_ms": 0.0,
        "fps": 0.0,
    }


def run_model_eval(sess, input_name, pairs, tile_size=INPUT_SIZE, overlap=8):
    """Evaluate an ONNX model session on image pairs."""
    psnr_y_list, ssim_y_list = [], []
    psnr_rgb_list, ssim_rgb_list = [], []
    tile_times = []

    for lr_p, hr_p in pairs:
        lr = cv2.imread(str(lr_p))
        hr = cv2.imread(str(hr_p))
        h, w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        if (h, w) == (tile_size, tile_size):
            tensor, _ = preprocess(lr, tile_size)
            t0 = time.perf_counter()
            out_raw = sess.run(None, {input_name: tensor})[0]
            tile_times.append((time.perf_counter() - t0) * 1000.0)
            sr = postprocess(out_raw)
        else:
            tiles, orig_hw, padded_hw, grid_hw = split_into_tiles(
                lr, patch_size=(tile_size, tile_size), overlap=overlap
            )
            sr_tiles = []
            for tile in tiles:
                tensor, _ = preprocess(tile, tile_size)
                t0 = time.perf_counter()
                out_raw = sess.run(None, {input_name: tensor})[0]
                tile_times.append((time.perf_counter() - t0) * 1000.0)
                sr_tiles.append(postprocess(out_raw))
            sr = merge_tiles(sr_tiles, orig_hw, padded_hw, grid_hw, scale=SCALE, overlap=overlap)

        min_h = min(sr.shape[0], hr_h)
        min_w = min(sr.shape[1], hr_w)
        sr_crop = sr[:min_h, :min_w]
        hr_crop = hr[:min_h, :min_w]

        psnr_y_list.append(compute_psnr(sr_crop, hr_crop, y_channel=True))
        ssim_y_list.append(compute_ssim(sr_crop, hr_crop, y_channel=True))
        psnr_rgb_list.append(compute_psnr(sr_crop, hr_crop, y_channel=False))
        ssim_rgb_list.append(compute_ssim(sr_crop, hr_crop, y_channel=False))

    mean_tile_ms = float(np.mean(tile_times)) if tile_times else 0.0
    fps = 1000.0 / mean_tile_ms if mean_tile_ms > 0 else 0.0

    return {
        "psnr_y": float(np.mean(psnr_y_list)),
        "ssim_y": float(np.mean(ssim_y_list)),
        "psnr_rgb": float(np.mean(psnr_rgb_list)),
        "ssim_rgb": float(np.mean(ssim_rgb_list)),
        "lat_ms": mean_tile_ms,
        "fps": fps,
    }


def eval_dataset(dataset_name: str, ep: str, fresh: bool = False):
    pairs = get_image_pairs(dataset_name)
    print(f"\n===========================================================")
    print(f"  Dataset: {dataset_name} ({len(pairs)} image pairs)")
    print(f"===========================================================")

    results = {}

    # 1. Bicubic baseline
    print("Evaluating Bicubic baseline...")
    results["Bicubic"] = run_bicubic_eval(pairs)
    print(f"  Bicubic:       PSNR (Y): {results['Bicubic']['psnr_y']:.2f} dB, SSIM (Y): {results['Bicubic']['ssim_y']:.4f}")

    models_to_eval = [
        ("FP32 Reference", str(DEFAULT_FP32), ep if ep != "npu" else "cpu"),
        ("XINT8", str(DEFAULT_XINT8), ep),
    ]
    if DEFAULT_ADAROUND.exists():
        models_to_eval.append(("XINT8 + AdaRound", str(DEFAULT_ADAROUND), ep))

    for name, model_path, target_ep in models_to_eval:
        if not os.path.isfile(model_path):
            print(f"Skipping {name} ({model_path} not found)")
            continue

        cache_key = SESR_ADAROUND_CACHE_KEY if "adaround" in model_path.lower() else SESR_CACHE_KEY
        if fresh and target_ep == "npu":
            clear_cache(cache_key)
        sess = build_session(model_path, ep=target_ep, cache_key=cache_key)
        input_name = sess.get_inputs()[0].name
        target_size = input_size(sess.get_inputs()[0].shape, where="session")

        res = run_model_eval(sess, input_name, pairs, tile_size=target_size)
        results[f"{name} ({target_ep.upper()})"] = res
        print(f"  {name:18s}: PSNR (Y): {res['psnr_y']:.2f} dB, SSIM (Y): {res['ssim_y']:.4f}, Lat: {res['lat_ms']:.2f} ms ({res['fps']:.1f} fps)")

    print(f"\n--- Summary Table for {dataset_name} ---")
    print(f"| Model | EP | PSNR (Y) [dB] | SSIM (Y) | PSNR (RGB) [dB] | SSIM (RGB) | Latency [ms] | FPS |")
    print(f"|---|---|---|---|---|---|---|---|")
    for k, v in results.items():
        ep_label = "CPU" if "Bicubic" in k else (k.split("(")[-1].replace(")", "") if "(" in k else ep.upper())
        model_label = k.split("(")[0].strip()
        lat_str = f"{v['lat_ms']:.2f}" if v['lat_ms'] > 0 else "-"
        fps_str = f"{v['fps']:.1f}" if v['fps'] > 0 else "-"
        print(f"| {model_label} | {ep_label} | {v['psnr_y']:.2f} | {v['ssim_y']:.4f} | {v['psnr_rgb']:.2f} | {v['ssim_rgb']:.4f} | {lat_str} | {fps_str} |")

    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate SESR-M7 SISR models.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider")
    ap.add_argument("--dataset", choices=["Set5", "Set14", "all"], default="all", help="Dataset to evaluate")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache on first build")
    args = ap.parse_args()

    datasets = ["Set5", "Set14"] if args.dataset == "all" else [args.dataset]
    for d in datasets:
        eval_dataset(d, ep=args.ep, fresh=args.fresh)


if __name__ == "__main__":
    main()
