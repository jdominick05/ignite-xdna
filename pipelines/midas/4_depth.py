"""
Step 4: Run MiDaS v2.1 Small monocular depth estimation on CPU, DirectML (iGPU), or Ryzen AI NPU.

Supports single-image inference, directory sweeps, live webcam depth estimation,
and clean latency benchmarking (sess.run timed alone, separated from pre/post).

Run in resnet_env17 (has ONNX Runtime with VitisAI EP + DML):
    # Benchmark on NPU:
    python pipelines/midas/4_depth.py --ep npu --image data/midas_val/000000000139.jpg --fresh --n 50

    # CPU FP32 baseline:
    python pipelines/midas/4_depth.py --ep cpu --model models/midas_small_cut.onnx \
        --image data/midas_val/000000000139.jpg --out results/midas_depth_cpu.jpg

    # DirectML (Radeon 780M iGPU):
    python pipelines/midas/4_depth.py --ep dml --model models/midas_small_cut.onnx \
        --image data/midas_val/000000000139.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.midas import INPUT_SIZE, colorize_depth, input_size, postprocess_depth, preprocess
from npu.paths import MIDAS_CACHE_KEY, MIDAS_NEAREST_CACHE_KEY, MODELS, RESULTS
from npu.session import build_session, clear_cache

DEFAULT_MODEL = MODELS / "midas_small_nearest_cut_xint8.onnx"


def main():
    ap = argparse.ArgumentParser(description="Run MiDaS monocular depth estimation.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to ONNX model")
    ap.add_argument("--image", default=None, help="Path to single image")
    ap.add_argument("--webcam", type=int, default=None, help="Webcam device ID (e.g. 0)")
    ap.add_argument("--out", default=str(RESULTS / "midas_depth.jpg"), help="Path to save depth map")
    ap.add_argument("--n", type=int, default=1, help="Number of benchmark iterations (default: 1)")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache before building session")
    ap.add_argument("--colormap", default="inferno", choices=["inferno", "magma", "plasma", "viridis", "turbo"],
                    help="Color map for depth visualization")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model file not found: {model_path}")

    # Select appropriate cache key based on model architecture
    cache_key = MIDAS_NEAREST_CACHE_KEY if "nearest" in Path(model_path).stem else MIDAS_CACHE_KEY

    if args.fresh and args.ep == "npu":
        clear_cache(cache_key)

    print(f"Building session (ep={args.ep}, cache_key={cache_key})...")
    sess = build_session(model_path, ep=args.ep, cache_key=cache_key)

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    target_size = input_size(input_shape, where="session")
    print(f"Session ready. Model input: {input_name} {input_shape}")

    colormap_dict = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    cmap = colormap_dict.get(args.colormap, cv2.COLORMAP_INFERNO)

    # Webcam mode
    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam {args.webcam}")
        print("Starting webcam live depth estimation. Press 'q' to exit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                t_in, orig_shape = preprocess(frame, target_size)
                t0 = time.perf_counter()
                raw_depth = sess.run(None, {input_name: t_in})[0]
                t_infer = (time.perf_counter() - t0) * 1000.0

                depth_u8 = postprocess_depth(raw_depth, orig_shape=orig_shape, normalize=True)
                colored = colorize_depth(depth_u8, cmap)

                fps = 1000.0 / t_infer if t_infer > 0 else 0
                cv2.putText(colored, f"{args.ep.upper()} infer: {t_infer:.1f}ms ({fps:.1f} fps)",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("MiDaS Depth", colored)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    # Image inference mode
    img_path = args.image
    if img_path is None:
        # Fallback to first image in data/midas_val or data/coco_calib
        candidates = list(Path("data/midas_val").glob("*.jpg")) or list(Path("data/calib").glob("*.jpg"))
        if candidates:
            img_path = str(candidates[0])
        else:
            raise SystemExit("No test image provided and none found in data directories.")

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")

    # Benchmark warmup and timing
    t_in, orig_shape = preprocess(img, target_size)

    # Warmup
    for _ in range(min(5, args.n)):
        sess.run(None, {input_name: t_in})

    infer_times = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        raw_depth = sess.run(None, {input_name: t_in})[0]
        infer_times.append(time.perf_counter() - t0)

    t_prep_start = time.perf_counter()
    _, _ = preprocess(img, target_size)
    t_prep = (time.perf_counter() - t_prep_start) * 1000.0

    t_post_start = time.perf_counter()
    depth_u8 = postprocess_depth(raw_depth, orig_shape=orig_shape, normalize=True)
    colored = colorize_depth(depth_u8, cmap)
    t_post = (time.perf_counter() - t_post_start) * 1000.0

    mean_infer = float(np.mean(infer_times)) * 1000.0
    median_infer = float(np.median(infer_times)) * 1000.0
    fps = 1000.0 / mean_infer if mean_infer > 0 else 0

    print(f"\n=== MiDaS v2.1 Small Benchmark ({args.ep.upper()}) ===")
    print(f"Model       : {model_path}")
    print(f"Image       : {img_path} ({orig_shape[1]}x{orig_shape[0]})")
    print(f"Iterations  : {args.n}")
    print(f"Preprocess  : {t_prep:.2f} ms")
    print(f"Infer (mean): {mean_infer:.2f} ms ({fps:.1f} fps)")
    print(f"Infer (med) : {median_infer:.2f} ms")
    print(f"Postprocess : {t_post:.2f} ms")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), colored)
        print(f"Saved depth visualization to {out_p}")


if __name__ == "__main__":
    main()
