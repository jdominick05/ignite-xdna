"""
Step 4: Run a model on CPU or on the Ryzen AI NPU and report top-1 / top-5
accuracy + latency.

    python pipelines/resnet50/4_run.py --ep cpu --images data/eval --n 1000 \
        --model models/resnet50_fp32.onnx
    python pipelines/resnet50/4_run.py --ep npu --images data/eval --n 1000 \
        --model models/resnet50_xint8_adaround.onnx --fresh

Or just run ./scripts/resnet-bench.sh, which does all three rows of the table.

Hawk Point (HPT / XDNA1) needs target=X1 plus the Phoenix xclbin, which only
the 1.7.1 install ships. Set RYZEN_AI_INSTALLATION_PATH to the 1.7.1 tree (or
pass --xclbin); npu.session refuses to build an NPU session without one,
because driver-resolved firmware compiles for Strix and DPU-timeouts.
Use --fresh whenever you change model or xclbin, or the stale compile gets
reused -- the cache is keyed by cacheKey, not by model hash.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import npu.preprocess as q  # same preprocessing as calibration, no quark import
from npu.paths import MODELS, RESNET_CACHE_KEY
from npu.session import build_session, clear_cache

MODEL = MODELS / "resnet50_int8.onnx"
CFG_PATH = MODELS / "preprocess_config.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", choices=["cpu", "npu"], default="cpu")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--verbose", action="store_true", help="ORT verbose logs")
    ap.add_argument("--model", default=str(MODEL),
                    help="ONNX model to run (default: the INT8 one). "
                         "Use models/resnet50_fp32.onnx for the FP32 baseline on CPU.")
    ap.add_argument("--xclbin", default=None,
                    help="explicit NPU binary (PHX/HPT). Omit to resolve it from "
                         "RYZEN_AI_INSTALLATION_PATH; there is no driver fallback.")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the compile cache first (do this when changing xclbin)")
    ap.add_argument("--cfg-path", default=None,
                    help="preprocess config (default: models/preprocess_config.json) -- "
                         "must match the resolution --model was exported/quantized at")
    ap.add_argument("--batch", type=int, default=1,
                    help="must match the static batch --model was exported with; "
                         "images are grouped into batches of this size per session.run")
    args = ap.parse_args()

    with open(args.cfg_path or str(CFG_PATH)) as f:
        cfg = json.load(f)
    transform = q.build_transform(cfg)

    files = []
    for ext in q.IMG_EXTS:
        files.extend(glob.glob(os.path.join(args.images, "**", ext), recursive=True))
    files = sorted(files)[: args.n]
    if not files:
        raise SystemExit(f"No images under {args.images}")
    n_batches = len(files) // args.batch
    files = files[: n_batches * args.batch]  # drop remainder: every batch must be full

    labels_path = os.path.join(args.images, "labels.json")
    labels = json.load(open(labels_path)) if os.path.isfile(labels_path) else None
    if labels is None:
        print("no labels.json found - will print predictions only, no accuracy")

    if args.fresh:
        clear_cache(RESNET_CACHE_KEY)

    session = build_session(args.model, args.ep, RESNET_CACHE_KEY, args.xclbin,
                            log_severity=0 if args.verbose else 1)
    input_name = session.get_inputs()[0].name

    B = args.batch
    batches = [files[i:i + B] for i in range(0, len(files), B)]

    # Warm-up: first inference pays one-time setup cost, keep it out of timing
    warmup = np.concatenate([transform(p) for p in batches[0]], axis=0)
    session.run(None, {input_name: warmup})

    top1 = top5 = 0
    times = []  # per-BATCH latency; divided by B below for per-image numbers
    for k, chunk in enumerate(batches):
        x = np.concatenate([transform(p) for p in chunk], axis=0)
        t0 = time.perf_counter()
        out = session.run(None, {input_name: x})[0]
        times.append(time.perf_counter() - t0)

        for i, path in enumerate(chunk):
            pred5 = np.argsort(out[i])[::-1][:5]
            if labels is not None:
                y = labels.get(os.path.basename(path))
                if y is not None:
                    top1 += int(pred5[0] == y)
                    top5 += int(y in pred5)
            elif k == 0 and i < 10:
                print(f"{os.path.basename(path):24s} top5: {pred5.tolist()}")

        if (k + 1) % max(1, 100 // B) == 0:
            print(f"  {(k + 1) * B}/{len(files)}")

    n = len(files)
    times = np.array(times) * 1000
    per_image = times / B
    print(f"\n=== {args.ep.upper()} | {n} images | batch {B} | {args.model} ===")
    if labels is not None:
        print(f"top-1: {100 * top1 / n:.2f}%    top-5: {100 * top5 / n:.2f}%")
    print(f"latency/batch  mean {times.mean():.2f} ms   median {np.median(times):.2f} ms   "
          f"p95 {np.percentile(times, 95):.2f} ms")
    print(f"latency/image  mean {per_image.mean():.2f} ms")
    print(f"throughput ~{1000 * B / times.mean():.1f} img/s")


if __name__ == "__main__":
    main()
