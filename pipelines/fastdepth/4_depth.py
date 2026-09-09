"""
Step 4: Run FastDepth monocular depth estimation on CPU, DirectML (iGPU), or Ryzen AI NPU.

Supports single-image inference, directory sweeps, live webcam depth estimation,
and clean latency benchmarking (sess.run timed alone, separated from pre/post).

Run in resnet_env17 (has ONNX Runtime with VitisAI EP + DML):
    # Benchmark on NPU:
    python pipelines/fastdepth/4_depth.py --ep npu --image data/fastdepth_val/000000000139.jpg --fresh --n 50

    # CPU FP32 baseline:
    python pipelines/fastdepth/4_depth.py --ep cpu --model models/fastdepth_fp32.onnx \
        --image data/fastdepth_val/000000000139.jpg --out results/fastdepth_depth_cpu.jpg

    # DirectML (Radeon 780M iGPU):
    python pipelines/fastdepth/4_depth.py --ep dml --model models/fastdepth_fp32.onnx \
        --image data/fastdepth_val/000000000139.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from npu.fastdepth import INPUT_SIZE, colorize_depth, input_size, postprocess_depth, preprocess
from npu.paths import FASTDEPTH_CACHE_KEY, MODELS, RESULTS
from npu.session import build_session, clear_cache

DEFAULT_MODEL = MODELS / "fastdepth_fp32_xint8.onnx"


def main():
    ap = argparse.ArgumentParser(description="Run FastDepth monocular depth estimation.")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="npu", help="Execution provider")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to ONNX model")
    ap.add_argument("--image", default=None, help="Path to single image")
    ap.add_argument("--webcam", type=int, default=None, help="Webcam device ID (e.g. 0)")
    ap.add_argument("--out", default=str(RESULTS / "fastdepth_depth.jpg"), help="Path to save depth map")
    ap.add_argument("--n", type=int, default=1, help="Number of benchmark iterations (default: 1)")
    ap.add_argument("--fresh", action="store_true", help="Clear compile cache before building session")
    ap.add_argument("--colormap", default="inferno", choices=["inferno", "magma", "plasma", "viridis", "turbo"],
                    help="Color map for depth visualization")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model file not found: {model_path}")

    cache_key = FASTDEPTH_CACHE_KEY

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
                cv2.imshow("FastDepth", colored)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    # Single-image / benchmark mode
    if args.image is None:
        # Fallback to test image in data
        cands = list((Path("data") / "fastdepth_val").glob("*.jpg")) + list((Path("data") / "midas_val").glob("*.jpg"))
        if not cands:
            raise SystemExit("No test image provided and none found in data/fastdepth_val or data/midas_val.")
        img_path = str(cands[0])
    else:
        img_path = args.image

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    t_in, orig_shape = preprocess(img_bgr, target_size)

    # Warmup
    for _ in range(5):
        sess.run(None, {input_name: t_in})

    # Benchmark loop (sess.run alone timed)
    latencies = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        raw_depth = sess.run(None, {input_name: t_in})[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    min_lat = np.min(latencies)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0

    print(f"Benchmark results ({args.ep.upper()}, {args.n} runs):")
    print(f"  Mean latency:   {mean_lat:.2f} ms ({fps:.1f} fps)")
    print(f"  Median latency: {median_lat:.2f} ms")
    print(f"  Min latency:    {min_lat:.2f} ms")

    # Postprocessing
    depth_u8 = postprocess_depth(raw_depth, orig_shape=orig_shape, normalize=True)
    colored = colorize_depth(depth_u8, cmap)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), colored)
    print(f"Saved depth visualization to {out_path}")


if __name__ == "__main__":
    main()
