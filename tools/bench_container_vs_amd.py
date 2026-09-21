#!/usr/bin/env python3
"""Glass-to-glass on any detect container: the engine against AMD's Vitis AI EP.

    python tools/bench_container_vs_amd.py --arm ignite --container build/M.ignite --onnx M.onnx
    python tools/bench_container_vs_amd.py --arm amd    --onnx M.onnx

One arm per invocation, because they need different environments: the engine arm needs
mlir-aie-iron for pyxrt, AMD's needs resnet_env17, the only one that can run the Vitis AI EP.
Run them alternately with xrt-smi idle between and interleave the rounds; a mean from two
rounds either side of the other arm is what makes a comparison same-sitting on this machine.

Nothing here is model-specific. The head tensors are the cut ONNX's own graph outputs, in
their own order, and the container's placements are looked up by those names with the
quantizer's suffix. The decoder is ``--decoder module:function`` with the same contract
``npu.yolo_decode.decode_heads`` has - it takes the head arrays and ``imgsz`` and returns
(1, 4 + nc, N) - so a family with a different head only needs its own decode module, not its
own benchmark. YOLO26 needs one because it dropped DFL; YOLOv8, YOLOv6 and pose already have
theirs.

Both arms share the letterbox, the decoder and the NMS, and time the same span: a decoded
frame from an image already in memory. Neither uses a native decode path, so the host tail is
numpy on both sides and is NOT optimized - the engine arm additionally pays one readback per
head, where the EP returns all of them from one call, so a margin measured here is a floor.
"""
import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_decoder(spec):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "decode_heads")


def head_names(onnx_path):
    import onnx
    return [o.name for o in onnx.load(str(onnx_path)).graph.output]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("ignite", "amd"), required=True)
    ap.add_argument("--onnx", required=True, help="the head-cut QDQ model; its outputs are the heads")
    ap.add_argument("--container", default=None, help="required for --arm ignite")
    ap.add_argument("--decoder", default="npu.yolo_decode:decode_heads")
    ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--cache-key", default="benchcachekey")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import cv2
    from npu.yolo import letterbox, postprocess
    decode_heads = load_decoder(args.decoder)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    heads = head_names(args.onnx)
    g2g, stages, dets = [], {"preprocess": [], "forward": [], "decode": []}, 0

    if args.arm == "amd":
        from npu.session import build_session
        sess = build_session(args.onnx, "npu", args.cache_key)
        names = [o.name for o in sess.get_outputs()]
        order = [names.index(h) for h in heads]
        feed = sess.get_inputs()[0].name

        def forward(x):
            outs = sess.run(None, {feed: x})
            return [np.asarray(outs[i], dtype=np.float32) for i in order]
        close = lambda: None
    else:
        if not args.container:
            raise SystemExit("--container is required for --arm ignite")
        from ignite_xdna.compiler import graph_reference as gr
        from ignite_xdna.runtime.graph_session import EngineSession
        s = EngineSession(args.container, device_index=0)
        pl_all = s.ge["placements"]
        # the quantizer renames a head; take whichever spelling the container placed
        qheads = [h if h in pl_all else h + "_QuantizeLinear_Output" for h in heads]
        missing = [h for h in qheads if h not in pl_all]
        if missing:
            raise SystemExit(f"{args.container} has no placement for {missing}")
        pls = [pl_all[h] for h in qheads]
        p = s.input_placement
        halo, base = int(p["halo"]), int(p["base"])

        def forward(x):
            q = gr.quantize_input(x[0], float(p["scale"]), int(p["zero_point"]))
            s._input_plane[halo:halo + p["height"], halo:halo + p["width"], :q.shape[0]] = \
                np.transpose(q, (1, 2, 0))
            if s._ws_map is None:
                s.bo_ws.write(s._input_plane, base)
            s.bo_ws.sync(s.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         s._input_bytes, base)
            s.dispatch()
            return [((s.read_tensor(h).astype(np.float32) - float(q_["zero_point"])) * float(q_["scale"]))[None]
                    for h, q_ in zip(qheads, pls)]
        close = s.close

    try:
        for i in range(args.warmup + args.frames):
            t0 = time.perf_counter()
            x, pad, scale = letterbox(img, args.imgsz)
            t1 = time.perf_counter()
            outs = forward(x)
            t2 = time.perf_counter()
            det = postprocess(decode_heads(outs, imgsz=args.imgsz), pad, scale,
                              conf_thres=args.conf, iou_thres=args.iou)
            t3 = time.perf_counter()
            if i >= args.warmup:
                g2g.append((t3 - t0) * 1e3)
                stages["preprocess"].append((t1 - t0) * 1e3)
                stages["forward"].append((t2 - t1) * 1e3)
                stages["decode"].append((t3 - t2) * 1e3)
                dets = len(det)
    finally:
        close()

    g = sorted(g2g)
    n = len(g)
    out = {"arm": args.arm, "onnx": Path(args.onnx).name, "frames": n,
           "mean": statistics.fmean(g), "p50": g[n // 2], "p95": g[int(n * 0.95)],
           "p99": g[int(n * 0.99)], "max": g[-1], "detections": dets,
           "stages_ms": {k: statistics.fmean(v) for k, v in stages.items()}}
    print(f"[{args.arm}] G2G mean {out['mean']:.3f} ms | P50 {out['p50']:.3f} | P95 {out['p95']:.3f} "
          f"| P99 {out['p99']:.3f} | max {out['max']:.3f} over {n} timed frames")
    print(f"[{args.arm}] stage means (ms): " +
          " | ".join(f"{k} {v:.3f}" for k, v in out["stages_ms"].items()))
    print(f"[{args.arm}] {dets} detections per frame")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())