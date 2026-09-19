#!/usr/bin/env python3
"""Prove that two graph-engine containers produce the same SESR image, frame by frame.

    bash scripts/research-iron.sh tools/sesr_identity_probe.py \
        --container-a build/sesr_m7_base.ignite --container-b build/sesr_m7.ignite \
        [--images data/sesr_val/Set5_LR_x2] [--limit 24] [--json out.json]

Each container owns one hardware context in turn (never two at once). Every frame is fed through
``SuperResolutionPipeline.predict_sync`` — the shipped path, so the comparison covers ingress
staging and the host depth-to-space as well as the NPU stages — and the resulting uint8 image is
compared byte for byte against the other container's image for the same frame. A differing frame
also reports its max absolute code difference and PSNR. Exit 0 means every frame was identical.

The combined digest is a SHA-256 over the concatenated output images in frame order, so a log line
from this tool pins the whole run's output, not just a per-frame coincidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--container-a", required=True)
ap.add_argument("--container-b", required=True)
ap.add_argument("--images", default="data/sesr_val/Set5_LR_x2")
ap.add_argument("--limit", type=int, default=24)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--json", default=None)
args = ap.parse_args()

from ignite_xdna.pipelines.sr_pipeline import SuperResolutionPipeline  # noqa: E402

paths = sorted(Path(args.images).glob("*.png")) + sorted(Path(args.images).glob("*.jpg"))
if not paths:
    raise SystemExit(f"no images under {args.images}")
paths = paths[: args.limit]
frames = []
for p in paths:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is not None:
        frames.append((p.name, img))
print(f"[probe] {len(frames)} frames from {args.images} (of {len(paths)} candidate files)")


def run(container):
    """Return the list of output images and the per-frame SHA-256 of each."""
    out, digests = [], []
    with SuperResolutionPipeline(container, device_index=args.device) as pipe:
        for name, img in frames:
            image, t = pipe.predict_sync(img)
            image = np.ascontiguousarray(image, dtype=np.uint8)
            out.append(image)
            digests.append(hashlib.sha256(image.tobytes()).hexdigest())
            print(f"[probe]   {Path(container).name} {name} {image.shape} "
                  f"g2g {t.glass_to_glass_ms:.3f} ms sha={digests[-1][:12]}", flush=True)
    return out, digests


a, da = run(args.container_a)
b, db = run(args.container_b)

bad = 0
worst = 0
worst_psnr = float("inf")
for i, (name, _) in enumerate(frames):
    ia, ib, hsha_a, hsha_b = a[i], b[i], da[i], db[i]
    if ia.shape != ib.shape:
        print(f"[probe] L{i} {name}: SHAPE {ia.shape} vs {ib.shape}")
        bad += 1
        continue
    if hsha_a == hsha_b:
        print(f"[probe] L{i} {name}: identical max_diff=0")
        continue
    diff = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)
    bad += 1
    worst = max(worst, int(diff.max()))
    worst_psnr = min(worst_psnr, psnr)
    print(f"[probe] L{i} {name}: DIFFER bytes {int(np.count_nonzero(diff))}/{diff.size} "
          f"max_diff={int(diff.max())} psnr={psnr:.2f} dB")

digest_a = hashlib.sha256(b"".join(x.tobytes() for x in a)).hexdigest()
digest_b = hashlib.sha256(b"".join(x.tobytes() for x in b)).hexdigest()
print(f"[probe] RESULT frames={len(frames)} identical={len(frames) - bad} differing={bad} "
      f"max_diff={worst} worst_psnr={'inf' if worst_psnr == float('inf') else f'{worst_psnr:.2f} dB'}")
print(f"[probe] combined sha256 {Path(args.container_a).name}={digest_a}")
print(f"[probe] combined sha256 {Path(args.container_b).name}={digest_b}")
if args.json:
    Path(args.json).write_text(json.dumps({
        "frames": [f[0] for f in frames], "identical": len(frames) - bad, "differing": bad,
        "max_diff": worst, "sha256_a": digest_a, "sha256_b": digest_b,
        "per_frame_a": da, "per_frame_b": db,
    }, indent=2), encoding="utf-8")
raise SystemExit(0 if bad == 0 else 1)
