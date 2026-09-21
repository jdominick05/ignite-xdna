#!/usr/bin/env python
"""Count a dense container's ONNX Runtime calls per frame, and check them against its manifest.

An `.ignite` container reaches ONNX Runtime only for the host segments its manifest declares. That
number is a contract: an accidental host round trip produces exactly the same output and shows up
only as latency, so it has to be counted rather than assumed. YOLOv8n and YOLOv8n-pose make 0 calls
per frame, YOLO11n makes 1, YOLO26n 2; a dense container makes one per host region, which is about
half its segments.

This runs the container through Ignition's own pipeline - the same `DensePipeline` that
`--task segment|matte` reaches - so it also checks that the consumer surface works, not just the
engine underneath it.

Needs the NPU and the conda `mlir-aie-iron` environment (Ignition imports Pillow, which the mlir-aie
ironenv does not carry). Run it through scripts/research-lowlevel.sh --npu for the idle witnesses:

    ./scripts/research-lowlevel.sh --log results/dense/<name>.log --checks-only --npu -- \
        python tools/dense_host_call_count.py --container build/bisenetv2_dense.ignite \
        --task segment --image data/bisenetv2_val/000000032570.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "Ignition" / "src")]

import cv2
import numpy as np
import onnxruntime as ort

from ignite_xdna.compiler.serializer import IgniteModelReader


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--task", choices=("segment", "matte"), required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1,
                    help="timed frames after one warm-up; the count is reported per frame")
    args = ap.parse_args()

    with IgniteModelReader(args.container) as reader:
        manifest = reader.manifest
    segments = manifest["graph_engine"]["segments"]
    declared = sum(1 for s in segments if s.get("kind") == "host")
    print(json.dumps({"container": args.container.name, "task": manifest.get("task"),
                      "segments": len(segments), "declared_host_segments": declared}))
    if manifest.get("task") != args.task:
        raise SystemExit(f"container declares task {manifest.get('task')!r}, not {args.task!r}")

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"could not read {args.image}")

    calls = {"n": 0}
    real_run = ort.InferenceSession.run

    def counting_run(self, *a, **kw):
        calls["n"] += 1
        return real_run(self, *a, **kw)

    from ignition.pipelines.dense import DensePipeline

    ort.InferenceSession.run = counting_run
    pipeline = DensePipeline(str(args.container), args.task, device_id=args.device)
    try:
        pipeline.predict(frame)          # warm up: session construction runs too
        calls["n"] = 0
        for _ in range(args.frames):
            result = pipeline.predict(frame)
        per_frame = calls["n"] / args.frames
    finally:
        ort.InferenceSession.run = real_run
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    out = result.mask if getattr(result, "mask", None) is not None else result.alpha
    print(json.dumps({"host_calls_per_frame": per_frame, "declared_host_segments": declared,
                      "matches_manifest": per_frame == declared,
                      "output_shape": list(np.shape(out)), "output_dtype": str(np.asarray(out).dtype)}))
    if per_frame != declared:
        print("MISMATCH: an unexpected host round trip, or a declared segment that never ran")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
