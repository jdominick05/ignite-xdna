"""Step 5: Quantitative fidelity and accuracy evaluation for Real-ESRGAN.

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
    python pipelines/realesrgan/5_eval.py --ep npu --dataset Set5
    python pipelines/realesrgan/5_eval.py --ep npu --dataset Set14
    python pipelines/realesrgan/5_eval.py --ep npu --dataset all
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.paths import DATA, MODELS, realesrgan_cache_key
from npu.realesrgan import (
    DEFAULT_INPUT_SIZE,
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

VAL_DIR = DATA / "realesrgan_val"


def get_image_pairs(dataset_name: str):
    """Load matching (LR_path, HR_path) pairs for a given dataset."""
    hr_dir = VAL_DIR / f"{dataset_name}_HR"
    lr_dir = VAL_DIR / f"{dataset_name}_LR_x4"

    if not hr_dir.exists() or not lr_dir.exists():
        raise SystemExit(f"Dataset directories not found: {hr_dir} or {lr_dir}. Run 2_fetch_data.py first.")

    hr_files = sorted(list(hr_dir.glob("*.png")) + list(hr_dir.glob("*.jpg")))
    lr_files = sorted(list(lr_dir.glob("*.png")) + list(lr_dir.glob("*.jpg")))

    pairs = []
    for hr_p in hr_files:
        stem = hr_p.stem
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
        min_h = min(bic.shape[0], hr_h)
        min_w = min(bic.shape[1], hr_w)
        bic = bic[:min_h, :min_w]
        hr_crop = hr[:min_h, :min_w]

        psnr_y_list.append(compute_psnr(bic, hr_crop, y_channel=True))
        ssim_y_list.append(compute_ssim(bic, hr_crop, y_channel=True))
        psnr_rgb_list.append(compute_psnr(bic, hr_crop, y_channel=False))
        ssim_rgb_list.append(compute_ssim(bic, hr_crop, y_channel=False))

    return {
        "psnr_y": np.mean(psnr_y_list),
        "ssim_y": np.mean(ssim_y_list),
        "psnr_rgb": np.mean(psnr_rgb_list),
        "ssim_rgb": np.mean(ssim_rgb_list),
        "latency_ms": 0.0,
        "fps": 0.0,
    }


def eval_model(sess, input_name, target_tile_size, pairs, overlap=8):
    """Run model across all image pairs and evaluate PSNR, SSIM, and per-tile latency."""
    psnr_y_list, ssim_y_list = [], []
    psnr_rgb_list, ssim_rgb_list = [], []
    tile_latencies = []

    for lr_p, hr_p in pairs:
        lr = cv2.imread(str(lr_p))
        hr = cv2.imread(str(hr_p))
        h, w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        # Upscale
        if (h, w) == (target_tile_size, target_tile_size):
            tensor, _ = preprocess(lr, target_tile_size)
            t0 = time.perf_counter()
            out_raw = sess.run(None, {input_name: tensor})[0]
            tile_latencies.append((time.perf_counter() - t0) * 1000.0)
            sr = postprocess(out_raw)
        else:
            tiles, orig_hw, padded_hw, grid_hw = split_into_tiles(
                lr, patch_size=(target_tile_size, target_tile_size), overlap=overlap
            )
            sr_tiles = []
            for tile in tiles:
                tensor, _ = preprocess(tile, target_tile_size)
                t0 = time.perf_counter()
                out_raw = sess.run(None, {input_name: tensor})[0]
                tile_latencies.append((time.perf_counter() - t0) * 1000.0)
                sr_tiles.append(postprocess(out_raw))
            sr = merge_tiles(sr_tiles, orig_hw, padded_hw, grid_hw, scale=SCALE, overlap=overlap)

        min_h = min(sr.shape[0], hr_h)
        min_w = min(sr.shape[1], hr_w)
        sr = sr[:min_h, :min_w]
        hr_crop = hr[:min_h, :min_w]

        psnr_y_list.append(compute_psnr(sr, hr_crop, y_channel=True))
        ssim_y_list.append(compute_ssim(sr, hr_crop, y_channel=True))
        psnr_rgb_list.append(compute_psnr(sr, hr_crop, y_channel=False))
        ssim_rgb_list.append(compute_ssim(sr, hr_crop, y_channel=False))

    mean_lat = np.mean(tile_latencies) if tile_latencies else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    return {
        "psnr_y": np.mean(psnr_y_list),
        "ssim_y": np.mean(ssim_y_list),
        "psnr_rgb": np.mean(psnr_rgb_list),
        "ssim_rgb": np.mean(ssim_rgb_list),
        "latency_ms": mean_lat,
        "fps": fps,
    }


def print_table(results_dict, dataset_name, ep):
    print(f"\n==========================================================================")
    print(f" Real-ESRGAN Evaluation Benchmark: {dataset_name} (Scale 4x) on {ep.upper()}")
    print(f"==========================================================================")
    print(f"| Model Variant       | PSNR (Y)  | SSIM (Y)  | PSNR (RGB)| SSIM (RGB)| Latency (ms)| Throughput |")
    print(f"|---------------------|-----------|-----------|-----------|-----------|-------------|------------|")
    for name, m in results_dict.items():
        lat_str = f"{m['latency_ms']:.2f}" if m['latency_ms'] > 0 else "—"
        fps_str = f"{m['fps']:.1f} fps" if m['fps'] > 0 else "—"
        print(
            f"| {name:<19} | {m['psnr_y']:>7.2f} dB | {m['ssim_y']:>8.4f}  | {m['psnr_rgb']:>7.2f} dB | {m['ssim_rgb']:>8.4f}  | {lat_str:>10}  | {fps_str:>10} |"
        )
    print(f"==========================================================================\n")


def main():
    ap = argparse.ArgumentParser(description="Evaluate Real-ESRGAN super-resolution fidelity.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider")
    ap.add_argument("--dataset", choices=["Set5", "Set14", "all"], default="Set5", help="Benchmark dataset")
    ap.add_argument("--model", default=None, help="Explicit model to evaluate (optional)")
    ap.add_argument("--arch", choices=["compact", "rrdb"], default="compact", help="Architecture to evaluate")
    ap.add_argument("--res", type=int, default=DEFAULT_INPUT_SIZE, help="Spatial input tile size")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache before building session")
    ap.add_argument("--overlap", type=int, default=8, help="Tile overlap in pixels")
    args = ap.parse_args()

    datasets = ["Set5", "Set14"] if args.dataset == "all" else [args.dataset]

    # Models to evaluate
    if args.model:
        target_ep = "cpu" if (args.ep == "npu" and "fp32" in args.model.lower()) else args.ep
        models_to_test = [("Custom Model", Path(args.model), target_ep)]
    else:
        prefix = f"realesrgan_{args.arch}_r{args.res}"
        fp32_m = MODELS / f"{prefix}_fp32.onnx"
        xint8_m = MODELS / f"{prefix}_xint8.onnx"
        adaround_m = MODELS / f"{prefix}_xint8_adaround.onnx"
        models_to_test = []
        if fp32_m.exists():
            fp32_ep = "cpu" if args.ep == "npu" else args.ep
            models_to_test.append(("FP32 Reference", fp32_m, fp32_ep))
        if xint8_m.exists():
            models_to_test.append(("XINT8 (Plain)", xint8_m, args.ep))
        if adaround_m.exists():
            models_to_test.append(("XINT8 + AdaRound", adaround_m, args.ep))

    for ds in datasets:
        pairs = get_image_pairs(ds)
        print(f"Loaded {len(pairs)} image pairs for {ds}.")

        results = {}
        print(f"Running Bicubic interpolation baseline on {ds}...")
        results["Bicubic Baseline"] = run_bicubic_eval(pairs)

        for label, m_path, target_ep in models_to_test:
            if not m_path.exists():
                print(f"Skipping {label} (not found: {m_path})")
                continue
            print(f"Evaluating {label} ({m_path.name}) on {target_ep.upper()}...")
            ckey = realesrgan_cache_key(m_path)
            if args.fresh and target_ep == "npu":
                clear_cache(ckey)
            sess = build_session(str(m_path), ep=target_ep, cache_key=ckey)
            input_name = sess.get_inputs()[0].name
            target_size = input_size(sess.get_inputs()[0].shape, where="session")
            metrics = eval_model(sess, input_name, target_size, pairs, overlap=args.overlap)
            results[f"{label} ({target_ep.upper()})"] = metrics

        print_table(results, ds, args.ep)


if __name__ == "__main__":
    main()
