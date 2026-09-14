#!/usr/bin/env python3
"""Split a graph-engine container's NPU dispatch into core compute and everything else.

    bash scripts/research-iron.sh tools/engine_dispatch_floor.py build/sesr_m7.ignite [--iters 300] [--json out.json]

Writes a copy of the container with every weight packet's op set to NOP (``<stem>_nop.ignite``
next to it). The copy has the same xclbin, instruction stream and packets, so the cores still
acquire every weight packet, consume every activation packet and emit every output object; only
the arithmetic is skipped. Both containers then dispatch the same staged frame ``--iters`` times
on Device 0, one hardware context at a time. The NOP dispatch time is the non-compute floor (DMA
tasks, transport, fixed per-dispatch cost); the difference is core compute.
"""
import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler.serializer import ARCH_XDNA1_PHOENIX, IgniteModelReader, IgniteModelWriter  # noqa: E402


def write_nop_copy(src: Path, dst: Path) -> dict:
    with IgniteModelReader(src) as reader:
        manifest = dict(reader.manifest)
        blobs = {name: reader.get_blob_bytes(name) for name in ("engine.xclbin", "insts.bin", "wpackets.bin")}
    wp = np.frombuffer(blobs["wpackets.bin"], dtype=np.uint8).copy()
    n = wp.size // em.W_BYTES
    packets = wp[:n * em.W_BYTES].reshape(n, em.W_BYTES)
    words = packets[:, :em.HDR_BYTES].copy().view(np.int32)
    ops = np.bincount(words[:, em.H_OP], minlength=4).tolist()
    words[:, em.H_OP] = em.OP_NOP
    packets[:, :em.HDR_BYTES] = words.view(np.uint8).reshape(n, em.HDR_BYTES)
    manifest["model_name"] = manifest["model_name"] + "_nop"
    writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
    writer.add_blob("engine.xclbin", blobs["engine.xclbin"], content_type="xclbin")
    writer.add_blob("insts.bin", blobs["insts.bin"], content_type="npu_instructions")
    writer.add_blob("wpackets.bin", wp.tobytes(), content_type="weight_packets")
    return {"packets": n, "ops_before": ops, "bytes": writer.write(dst), "instruction_bytes": len(blobs["insts.bin"])}


def time_dispatches(path: Path, image: Path, iters: int, warmup: int) -> dict:
    from ignite_xdna.runtime.graph_session import DenseGraphSession, GraphSession

    with IgniteModelReader(path) as reader:
        task = reader.manifest.get("task", "detect")
    session = (DenseGraphSession if task == "super_resolution" else GraphSession)(path, device_index=0)
    try:
        session.stage_image(cv2.imread(str(image)))
        for _ in range(warmup):
            session.dispatch()
        times = []
        t0 = time.perf_counter()
        for _ in range(iters):
            times.append(session.dispatch())
        wall = time.perf_counter() - t0
    finally:
        session.close()
    a = np.asarray(times)
    return {"container": path.name, "task": task, "iters": iters, "mean_ms": float(a.mean()),
            "p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "min_ms": float(a.min()), "max_ms": float(a.max()),
            "dispatches_per_s": iters / wall}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("container", type=Path)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--image", type=Path, default=ROOT / "assets" / "bus.jpg")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    nop = args.container.with_name(args.container.stem + "_nop.ignite")
    copy = write_nop_copy(args.container, nop)
    print(f"[floor] {nop.name}: {copy['packets']} weight packets, ops before {copy['ops_before']} -> all NOP, "
          f"{copy['instruction_bytes']:,} instruction bytes", flush=True)
    rows = []
    for path in (args.container, nop):
        r = time_dispatches(path, args.image, args.iters, args.warmup)
        rows.append(r)
        print(f"[dispatch] {r['container']} ({r['task']}): {r['iters']} dispatches mean {r['mean_ms']:.3f} ms "
              f"p50 {r['p50_ms']:.3f} p95 {r['p95_ms']:.3f} p99 {r['p99_ms']:.3f} min {r['min_ms']:.3f} "
              f"max {r['max_ms']:.3f} | {r['dispatches_per_s']:.1f} dispatches/s", flush=True)
    real, floor = rows
    print(f"[floor] {args.container.name}: dispatch {real['mean_ms']:.3f} ms = non-compute floor "
          f"{floor['mean_ms']:.3f} ms + core compute {real['mean_ms'] - floor['mean_ms']:.3f} ms", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"nop_copy": copy, "real": real, "nop": floor,
                                         "compute_ms": real["mean_ms"] - floor["mean_ms"]}, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
