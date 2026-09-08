"""
Step 4: Run MODNet portrait matting on CPU or Ryzen AI XDNA1 NPU.

Supports single-image matting, live webcam background removal / bokeh blur,
and benchmark profiling.

Run in resnet_env17 (ONNX Runtime with VitisAI EP):
    # Benchmark on NPU with fresh compile:
    python pipelines/modnet/4_matte.py --ep npu --image data/modnet_val/000000000139.jpg --fresh --n 50

    # Image cutout on CPU baseline:
    python pipelines/modnet/4_matte.py --ep cpu --model models/modnet/modnet_fp32.onnx \
        --image data/modnet_val/000000000139.jpg --out results/modnet/cutout_cpu.png

    # Webcam with bokeh background blur on NPU:
    python pipelines/modnet/4_matte.py --ep npu --webcam 0 --blur
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.modnet import postprocess_matte, preprocess
from npu.paths import MODELS, modnet_cache_key
from npu.session import build_session, clear_cache

DEFAULT_MODEL = MODELS / "modnet" / "modnet_cut_xint8.onnx"
DEFAULT_CFG = MODELS / "modnet" / "preprocess_config.json"






def apply_background_blur(frame_bgr, alpha, blur_ksize=25):
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    alpha_3ch = np.repeat(np.expand_dims(alpha, axis=-1), 3, axis=-1)
    composite = (alpha_3ch * frame_bgr + (1.0 - alpha_3ch) * blurred).astype(np.uint8)
    return composite


def apply_virtual_background(frame_bgr, alpha, bg_bgr):
    orig_h, orig_w = frame_bgr.shape[:2]
    bg_resized = cv2.resize(bg_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    alpha_3ch = np.repeat(np.expand_dims(alpha, axis=-1), 3, axis=-1)
    composite = (alpha_3ch * frame_bgr + (1.0 - alpha_3ch) * bg_resized).astype(np.uint8)
    return composite


def main():
    ap = argparse.ArgumentParser(description="Run MODNet portrait matting.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to ONNX model")
    ap.add_argument("--cfg", default=str(DEFAULT_CFG), help="Preprocess config path")
    ap.add_argument("--image", default=None, help="Path to single image")
    ap.add_argument("--webcam", type=int, default=None, help="Webcam device ID (e.g. 0)")
    ap.add_argument("--blur", action="store_true", help="Apply background blur")
    ap.add_argument("--blur-ksize", type=int, default=25, help="Blur kernel size (default: 25)")
    ap.add_argument("--bg-image", default=None, help="Path to replacement background image")
    ap.add_argument("--out", default=None, help="Path to save result")
    ap.add_argument("--n", type=int, default=1, help="Benchmark iteration count (default: 1)")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache")
    ap.add_argument("--xclbin", default=None, help="Explicit xclbin path")
    ap.add_argument("--verbose", action="store_true", help="Verbose ORT logging")
    args = ap.parse_args()

    cache_key = modnet_cache_key(args.model)
    if args.fresh:
        clear_cache(cache_key)

    print(f"[4_matte] Building session: EP={args.ep}, Model={args.model}, Cache={cache_key}...")
    session = build_session(
        args.model,
        args.ep,
        cache_key,
        args.xclbin,
        log_severity=0 if args.verbose else 1,
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open webcam ID {args.webcam}")
        print(f"[4_matte] Streaming from webcam {args.webcam}. Press 'q' to quit.")

        bg_img = None
        if args.bg_image and os.path.exists(args.bg_image):
            bg_img = cv2.imread(args.bg_image)

        fps_history = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            tensor, orig_shape = preprocess(frame, 512)
            t_pre = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            raw_matte = session.run([output_name], {input_name: tensor})[0]
            t_infer = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            alpha = postprocess_matte(raw_matte, orig_shape)
            if bg_img is not None:
                display = apply_virtual_background(frame, alpha, bg_img)
            elif args.blur:
                display = apply_background_blur(frame, alpha, args.blur_ksize)
            else:
                alpha_vis = (alpha * 255).astype(np.uint8)
                display = cv2.cvtColor(alpha_vis, cv2.COLOR_GRAY2BGR)
            t_post = (time.perf_counter() - t2) * 1000

            t_total = (time.perf_counter() - t0) * 1000
            fps_history.append(1000.0 / max(t_total, 1e-3))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            cv2.putText(display, f"EP: {args.ep.upper()} | Infer: {t_infer:.1f}ms | FPS: {avg_fps:.1f}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("MODNet Portrait Matting", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # Image mode
    if args.image is None:
        # Default test image if none provided
        val_imgs = list(Path("data/modnet_val").glob("*.jpg"))
        if not val_imgs:
            raise SystemExit("No image specified and data/modnet_val is empty.")
        args.image = str(val_imgs[0])
        print(f"[4_matte] Using default image: {args.image}")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"Cannot read image from {args.image}")

    # Warmup
    warmup_tensor, orig_shape = preprocess(img_bgr, 512)
    session.run([output_name], {input_name: warmup_tensor})

    latencies = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        tensor, _ = preprocess(img_bgr, 512)
        t_pre = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        raw_matte = session.run([output_name], {input_name: tensor})[0]
        t_infer = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        alpha = postprocess_matte(raw_matte, orig_shape)
        t_post = (time.perf_counter() - t2) * 1000

        latencies.append((t_pre, t_infer, t_post))

    infer_times = [x[1] for x in latencies]
    mean_infer = np.mean(infer_times)
    p50_infer = np.percentile(infer_times, 50)
    p90_infer = np.percentile(infer_times, 90)

    print(f"\n--- [4_matte] Benchmark Results (N={args.n}, EP={args.ep}) ---")
    print(f"Inference latency (ms): Mean={mean_infer:.2f}, P50={p50_infer:.2f}, P90={p90_infer:.2f}")
    print(f"Preprocess latency (ms): {np.mean([x[0] for x in latencies]):.2f}")
    print(f"Postprocess latency (ms): {np.mean([x[2] for x in latencies]):.2f}")
    print(f"Total frame latency (ms): {np.mean([sum(x) for x in latencies]):.2f} (~{1000.0/np.mean([sum(x) for x in latencies]):.1f} FPS)")

    out_path = args.out
    if out_path is None:
        out_dir = Path("results/modnet")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        out_path = str(out_dir / f"{stem}_{args.ep}_composite.png")

    if args.blur:
        composite = apply_background_blur(img_bgr, alpha, args.blur_ksize)
    elif args.bg_image and os.path.exists(args.bg_image):
        bg = cv2.imread(args.bg_image)
        composite = apply_virtual_background(img_bgr, alpha, bg)
    else:
        # Save RGBA cutout
        alpha_u8 = (alpha * 255).astype(np.uint8)
        composite = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        composite[:, :, 3] = alpha_u8

    cv2.imwrite(out_path, composite)
    print(f"[4_matte] Saved composite output to {out_path}")


if __name__ == "__main__":
    main()
