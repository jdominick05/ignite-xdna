#!/usr/bin/env python3
"""Split a SPLIT container's dispatch into core compute and everything else, per NPU segment.

    bash scripts/research-iron.sh tools/engine_dispatch_floor_split.py build/yolov8s_split3.ignite \
        [--iters 300] [--warmup 20] [--json out.json]

tools/engine_dispatch_floor.py answers this for a monolithic container, but it reads exactly
``insts.bin`` and ``wpackets.bin``; a container built with ``--split-layer`` carries ``insts_<i>.bin``
per NPU segment, and one built with ``--decouple-weights`` keeps no ``wpackets.bin`` at all -- its
weights live in a ``.weights`` sidecar whose sha256 the runtime checks at load. This tool handles
both shapes and reports every segment separately, which a monolithic dispatch cannot: it is the one
thing a split container measures that a whole-network one does not.

The NOP copy sets every weight packet's op to NOP, so the cores still acquire every weight packet,
consume every activation packet and emit every output object; only the arithmetic is skipped. The
NOP time is the non-compute floor (DMA tasks, transport, fixed per-dispatch cost) and the difference
is core compute. When the weights are decoupled the copy writes its own ``_nop.weights`` sidecar and
rewrites ``graph_engine.weights_file``/``weights_sha256`` to match, or the runtime refuses to load it.

Per-segment times come from ``GraphSession.dispatch()``'s own ``last_segment_ms``. Offline sizing
cannot produce these: the floor is a shim-timing property that exists only on silicon.
"""
import argparse
import hashlib
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


def nop_weights(raw: bytes):
    """Return (nop_bytes, packet_count, op_histogram_before)."""
    wp = np.frombuffer(raw, dtype=np.uint8).copy()
    n = wp.size // em.W_BYTES
    packets = wp[:n * em.W_BYTES].reshape(n, em.W_BYTES)
    words = packets[:, :em.HDR_BYTES].copy().view(np.int32)
    ops = np.bincount(words[:, em.H_OP], minlength=8).tolist()
    words[:, em.H_OP] = em.OP_NOP
    packets[:, :em.HDR_BYTES] = words.view(np.uint8).reshape(n, em.HDR_BYTES)
    return wp.tobytes(), n, ops


def write_nop_copy(src: Path, dst: Path) -> dict:
    with IgniteModelReader(src) as reader:
        manifest = json.loads(json.dumps(reader.manifest))
        names = list(reader.blobs.keys())
        blobs = {name: reader.get_blob_bytes(name) for name in names}
    ge = manifest.setdefault("graph_engine", {})
    decoupled = "wpackets.bin" not in blobs
    if decoupled:
        side = src.parent / (ge.get("weights_file") or f"{src.stem}.weights")
        if not side.exists():
            raise FileNotFoundError(f"weights sidecar not found: {side}")
        raw = side.read_bytes()
    else:
        raw = blobs["wpackets.bin"]
    nop, n, ops = nop_weights(raw)

    manifest["model_name"] = manifest.get("model_name", src.stem) + "_nop"
    writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
    side_out = None
    if decoupled:
        side_out = dst.parent / f"{dst.stem}.weights"
        side_out.write_bytes(nop)
        ge["weights_file"] = side_out.name
        ge["weights_sha256"] = hashlib.sha256(nop).hexdigest()
    for name in names:
        if name == "wpackets.bin":
            writer.add_blob(name, nop, content_type="weight_packets")
        elif name.startswith("insts"):
            writer.add_blob(name, blobs[name], content_type="npu_instructions")
        elif name == "engine.xclbin":
            writer.add_blob(name, blobs[name], content_type="xclbin")
        else:
            writer.add_blob(name, blobs[name])
    size = writer.write(dst)
    return {"packets": n, "ops_before": ops, "bytes": size, "decoupled": decoupled,
            "sidecar": side_out.name if side_out else None,
            "instruction_blobs": {k: len(v) for k, v in blobs.items() if k.startswith("insts")}}


def time_segments(path: Path, image: Path, iters: int, warmup: int) -> dict:
    from ignite_xdna.runtime.graph_session import DenseGraphSession, GraphSession
    with IgniteModelReader(path) as reader:
        task = reader.manifest.get("task", "detect")
    session = (DenseGraphSession if task == "super_resolution" else GraphSession)(path, device_index=0)
    try:
        nseg = len(session.segments)
        session.stage_image(cv2.imread(str(image)))
        for _ in range(warmup):
            session.dispatch()
        per_seg = [[] for _ in range(nseg)]
        total = []
        t0 = time.perf_counter()
        for _ in range(iters):
            total.append(session.dispatch())
            for i, ms in enumerate(session.last_segment_ms):
                per_seg[i].append(ms)
        wall = time.perf_counter() - t0
    finally:
        session.close()
    a = np.asarray(total)
    segs = []
    for i, v in enumerate(per_seg):
        s = np.asarray(v)
        segs.append({"segment": i, "mean_ms": float(s.mean()), "p50_ms": float(np.median(s)),
                     "min_ms": float(s.min()), "max_ms": float(s.max())})
    return {"container": path.name, "task": task, "iters": iters, "segments": segs,
            "total_mean_ms": float(a.mean()), "total_p50_ms": float(np.median(a)),
            "total_min_ms": float(a.min()), "dispatches_per_s": iters / wall}


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
    print(f"[floor] {nop.name}: {copy['packets']:,} weight packets -> all NOP, decoupled={copy['decoupled']}"
          + (f", sidecar {copy['sidecar']}" if copy["sidecar"] else "")
          + f", segments {copy['instruction_blobs']}", flush=True)

    rows = []
    for path in (args.container, nop):
        r = time_segments(path, args.image, args.iters, args.warmup)
        rows.append(r)
        segtxt = "  ".join(f"s{s['segment']} {s['mean_ms']:.3f}" for s in r["segments"])
        print(f"[dispatch] {r['container']}: {r['iters']}x total mean {r['total_mean_ms']:.3f} ms "
              f"p50 {r['total_p50_ms']:.3f} | per segment {segtxt}", flush=True)

    real, floor = rows
    print(f"[floor] {args.container.name}: per-segment floor and compute", flush=True)
    print(f"  {'seg':>4} {'real ms':>9} {'floor ms':>9} {'compute ms':>11} {'floor %':>8}", flush=True)
    for a, b in zip(real["segments"], floor["segments"]):
        comp = a["mean_ms"] - b["mean_ms"]
        pct = 100.0 * b["mean_ms"] / a["mean_ms"] if a["mean_ms"] else 0.0
        print(f"  {a['segment']:>4} {a['mean_ms']:>9.3f} {b['mean_ms']:>9.3f} {comp:>11.3f} {pct:>7.1f}%", flush=True)
    tc = real["total_mean_ms"] - floor["total_mean_ms"]
    print(f"  {'all':>4} {real['total_mean_ms']:>9.3f} {floor['total_mean_ms']:>9.3f} {tc:>11.3f} "
          f"{100.0 * floor['total_mean_ms'] / real['total_mean_ms']:>7.1f}%", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"nop_copy": copy, "real": real, "nop": floor}, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())