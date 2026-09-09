"""Does the shift-cut hazard predictor actually predict DPU failure?

The predictor in `quant/shift_cut.py` (branch `main`, unmerged at the time of writing)
flags convolutions whose scale triple cannot be represented by the DPU's
multiplier-plus-shift requantizer, and its author's evidence was retrodictive: every
model it had been pointed at already had a known outcome. This runs the other direction.
Predictions are recorded for a fixed candidate set FIRST, then the artifacts are executed
on the device and compared against their own CPU reference.

The failure mode being hunted is specific and is one this repo has hit before: a model
that is correct under CPU execution and wrong on the DPU. So the comparison is CPU vs NPU
on identical inputs, not accuracy against a dataset. Agreement is reported as the maximum
and mean absolute difference over the output tensor, plus the correlation, which separates
"slightly requantized" from "structurally wrong".

Placement is reported too, from the EP's own report, because a model that quietly falls
back to CPU would agree perfectly and mean nothing.

Usage (resnet_env17, NOT resnet_env):
    python tools/shift_cut_forward_test.py models/sesr_m7_fp32_xint8.onnx --cache-key sesrfwd
    python tools/shift_cut_forward_test.py <model> --cache-key <key> --seed 3

Always passes a fresh cache: the compile cache is keyed by name alone, so a stale entry is
reused silently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu.session import build_session, clear_cache, resolve_xclbin  # noqa: E402


#: Input conventions taken from this repo's own preprocessing modules in npu/. Feeding
#: the wrong one saturates the network and makes BOTH paths produce nonsense, which then
#: disagree for reasons that have nothing to do with the DPU -- so this is not cosmetic.
RANGES = {
    "byte": (0.0, 255.0),        # npu/sesr.py subtracts 128 from an RGB 0-255 array
    "unit": (0.0, 1.0),          # npu/realesrgan.py and the yolo modules scale by 1/255
    "imagenet": (-2.5, 2.5),     # mean/std normalized, e.g. npu/bisenetv2.py
}


def make_inputs(sess, seed: int, rng_name: str) -> dict:
    """Deterministic inputs shaped to the model, in the range it was calibrated for."""
    rng = np.random.default_rng(seed)
    lo, hi = RANGES[rng_name]
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if "float" in inp.type:
            feeds[inp.name] = rng.uniform(lo, hi, size=shape).astype(np.float32)
        elif "uint8" in inp.type:
            feeds[inp.name] = rng.integers(0, 256, size=shape, dtype=np.uint8)
        else:
            feeds[inp.name] = rng.integers(-128, 128, size=shape).astype(np.int8)
    return feeds


def load_image(sess, path: str, rng_name: str) -> dict:
    """Feed a real image, resized to the model's own input shape."""
    import cv2
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    h, w = shape[2], shape[3]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image {path}")
    img = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB).astype(np.float32)
    lo, hi = RANGES[rng_name]
    arr = img if hi > 2 else img / 255.0
    if rng_name == "imagenet":
        arr = (img / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) /               np.array([0.229, 0.224, 0.225], np.float32)
    return {inp.name: np.ascontiguousarray(arr.transpose(2, 0, 1)[None])}


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    diff = np.abs(a - b)
    denom = max(float(np.abs(a).max()), 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 else float("nan")
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel_to_peak": float(diff.max() / denom),
        "corr": corr,
        "elements": int(a.size),
    }


def placement(cache_key: str) -> str:
    from npu.paths import CACHE_DIR
    p = Path(str(CACHE_DIR)) / cache_key / "vitisai_ep_report.json"
    if not p.exists():
        return "no EP report written"
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return f"EP report unreadable at {p}"
    txt = json.dumps(rep)
    for key in ("deviceStat", "nodeStat", "stat"):
        if key in txt:
            break
    return txt[:400]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--cache-key", required=True,
                    help="cache directory name; never reuse another family's key")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--input-range", choices=sorted(RANGES), default="byte",
                    help="the value range this model was calibrated for; see RANGES")
    ap.add_argument("--image", help="a real image to feed instead of noise. Random input "
                    "is a fair probe for a smooth pixel mapping such as super-resolution, "
                    "and a poor one for a detection or segmentation head, whose outputs on "
                    "noise are degenerate and disagree for reasons unrelated to the DPU.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="skip the fresh-cache delete (do not use for a real result)")
    args = ap.parse_args(argv)

    model = str(Path(args.model).resolve())
    print(f"model      {model}")
    print(f"cache key  {args.cache_key}")
    print(f"seed       {args.seed}")
    print(f"input rng  {args.input_range} {RANGES[args.input_range]}")
    print()

    if not args.keep_cache:
        clear_cache(args.cache_key)

    cpu = build_session(model, "cpu", args.cache_key, log_severity=3)
    feeds = (load_image(cpu, args.image, args.input_range) if args.image
             else make_inputs(cpu, args.seed, args.input_range))
    for name, arr in feeds.items():
        print(f"input      {name} {arr.shape} {arr.dtype}")
    ref = cpu.run(None, feeds)
    del cpu

    xclbin = resolve_xclbin()
    print(f"xclbin     {xclbin}")
    npu = build_session(model, "npu", args.cache_key, xclbin=xclbin, log_severity=3)
    got = npu.run(None, feeds)
    del npu

    print()
    print("EP report (truncated):")
    print("  " + placement(args.cache_key))
    print()

    print(f"{'output':<24} {'elements':>10} {'max abs':>12} {'mean abs':>12} "
          f"{'max/peak':>10} {'corr':>9}")
    print("-" * 82)
    worst = 0.0
    for i, (r, g) in enumerate(zip(ref, got)):
        m = agreement(np.asarray(r), np.asarray(g))
        worst = max(worst, m["max_rel_to_peak"])
        print(f"{('out[' + str(i) + ']'):<24} {m['elements']:>10} {m['max_abs']:>12.4f} "
              f"{m['mean_abs']:>12.4f} {m['max_rel_to_peak']:>10.4f} {m['corr']:>9.5f}")

    print()
    print(f"WORST relative deviation across outputs: {worst:.4f} of peak")
    print("  A structurally wrong DPU result shows a large max/peak AND a fallen")
    print("  correlation. Ordinary requantization noise moves max/peak a little and")
    print("  leaves the correlation at essentially 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
