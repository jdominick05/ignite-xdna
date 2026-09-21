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

Both arms share the letterbox - the same cv2 resize and pad, from npu.yolo.letterbox_canvas -
the decoder and the NMS, and time the same span: a decoded frame from an image already in
memory. Neither uses a native decode path, so the host tail is numpy on both sides and is NOT
optimized - the engine arm must also dequantize its int8 heads to float32, where the EP's model
returns float already, so a margin measured here is a floor.

Where the arms necessarily differ is the ingress AFTER that shared letterbox, because the two
stacks want different things: AMD's model takes float32 NCHW RGB, the container takes a
channel-blocked uint8 plane. Each arm pays its own stack's real cost for that step.
  amd     the float conversion in npu.yolo.letterbox (astype, /255, transpose).
  ignite  FusedPreprocessor.preprocess_to_plane, the shipped AVX2 ingress, writing the plane
          directly. It is handed the already-letterboxed square canvas, so its own resize is an
          identity pass and the bytes it writes are the ones the numpy quantize wrote.

That last point is why this is a fair change and not a thumb on the scale: it removes a host tax
the benchmark was charging the engine arm that the shipped pipeline never pays. The container is
not faster, and AMD's arm is untouched. The engine arm asserts its ingress LUT equals the numpy
quantize it replaces for all 256 pixel values, so a container whose input scale is not a power of
two fails loudly instead of quietly measuring a different frame.

Egress is the shipped path too: read_heads merges the device syncs, takes zero-copy views of a
mapped workspace and unswizzles with the native kernel, where this arm used to sync, copy and
transpose once per head out of the read_tensor DEBUG helper. What is left is the float32
dequantize the shared numpy decoder needs, and that is a real cost of this arm rather than an
artefact of the tool. It is also small: on yolov8s the dispatch is 17.4 ms of a 20.0 ms forward.

Stage buckets differ per arm and are reported as such:
  preprocess  everything up to a model-ready input, the plane write included on the engine arm.
  forward     amd = session.run; ignite = sync + dispatch + the shipped read_heads + dequantize.
  decode      the shared numpy decode and NMS.
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
    from npu.yolo import letterbox, letterbox_canvas, postprocess
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

        def prepare(im):
            return letterbox(im, args.imgsz)

        def forward(x):
            outs = sess.run(None, {feed: x})
            return [np.asarray(outs[i], dtype=np.float32) for i in order]
        close = lambda: None
        ingress = "numpy letterbox + float32 NCHW, what the EP's model takes"
    else:
        if not args.container:
            raise SystemExit("--container is required for --arm ignite")
        from ignite_xdna.compiler import graph_reference as gr
        from ignite_xdna.runtime.graph_session import GraphSession
        # GraphSession, not the bare EngineSession, because it carries the declared head layout
        # and the shipped read_heads; read_tensor is a debug helper and this arm used to hand-roll
        # the readback out of it, one device sync and one numpy transpose per head.
        s = GraphSession(args.container, device_index=0)
        pl_all = s.ge["placements"]
        # the quantizer renames a head; take whichever spelling the container placed
        qheads = [h if h in pl_all else h + "_QuantizeLinear_Output" for h in heads]
        missing = [h for h in qheads if h not in pl_all]
        if missing:
            raise SystemExit(f"{args.container} has no placement for {missing}")
        pls = [pl_all[h] for h in qheads]
        p = s.input_placement
        halo, base = int(p["halo"]), int(p["base"])

        from ignite_xdna.pipelines.preprocess import FusedPreprocessor
        from ignite_xdna.runtime.graph_session import input_lut

        in_scale, in_zp = float(p["scale"]), int(p["zero_point"])
        lut = input_lut(in_scale, in_zp)
        # The ingress LUT has to BE the numpy quantize it replaces, or this stops being a pure
        # host-tax removal and starts changing what the model sees. letterbox hands
        # quantize_input float32, so the comparison is made in that dtype. The two agree when
        # the input scale is a power of two, which is not a property any container has to have.
        _pix = np.arange(256, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
        _ref = gr.quantize_input(_pix, in_scale, in_zp)
        if not np.array_equal(lut, _ref):
            n = int(np.count_nonzero(lut != _ref))
            raise SystemExit(
                f"{args.container}: the native ingress LUT differs from quantize_input on {n} of 256 "
                f"pixel values (input scale {in_scale!r}, zero point {in_zp}), so it would change the "
                f"model's input. Refusing rather than quietly measuring a different frame.")

        pre = FusedPreprocessor(imgsz=args.imgsz)
        if not pre.has_plane_ingress:
            raise SystemExit("the native preprocessor has no plane ingress; rebuild preprocess_simd")
        want = (int(p["height"]) + 2 * halo, int(p["width"]) + 2 * halo, 8)
        if s._input_plane.shape != want:
            raise SystemExit(f"input plane is {s._input_plane.shape}, expected {want}")

        def prepare(im):
            canvas, pad_, scale_ = letterbox_canvas(im, args.imgsz)
            # the canvas is already square at imgsz, so the native resize is an identity pass
            # and the plane bytes are the ones the numpy quantize wrote
            pre.preprocess_to_plane(canvas, s._input_plane, halo, lut)
            if s._ws_map is None:
                s.bo_ws.write(s._input_plane, base)
            return None, pad_, scale_

        # The readback goes through the shipped path too. read_heads merges the device syncs into
        # one span per run of adjacent heads, takes zero-copy views when the workspace is mapped,
        # and unswizzles with the native kernel into one reused buffer - where read_tensor synced,
        # copied and transposed per head. The egress is int8 with zero point 0 (read_heads flips
        # the uint8 codes' top bit), which is why the layout's own zero points are 0 and not the
        # containers' 128, and why dequantizing it is a multiply.
        status = s.head_status
        if not status.present:
            raise SystemExit(
                f"{args.container} declares no head layout, so the shipped readback cannot be "
                f"used and this arm would not measure what the pipeline measures.")
        layout = status.layout

        # The layout names heads canonically (p3_box, p3_cls, ...); the ONNX names them by their
        # convolution and in a different order. Map on (scale, channels, height, width): the scale
        # alone repeats across strides and the shape alone can collide when a class head happens to
        # be as wide as a box head, so the pair is the key. Refuse if it is not one-to-one rather
        # than fall back quietly - a silent fallback would measure a different thing per model.
        spec_by_key = {}
        for hs in layout.heads:
            c, hh, ww = (int(v) for v in hs.shape[1:])
            spec_by_key.setdefault((float(hs.scale), c, hh, ww), []).append(hs.name)
        order = []
        for h, q_ in zip(qheads, pls):
            key = (float(q_["scale"]), int(q_["channels"]), int(q_["height"]), int(q_["width"]))
            hit = spec_by_key.get(key, [])
            if len(hit) != 1:
                raise SystemExit(
                    f"{args.container}: head {h} matches {len(hit)} entries of the declared head "
                    f"layout on (scale, channels, height, width)={key}. Refusing rather than "
                    f"guessing which egress region is which head.")
            order.append(hit[0])

        head_views = {}

        def forward(_x):
            s.bo_ws.sync(s.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         s._input_bytes, base)
            s.dispatch()
            egress = s.read_heads(unswizzle=True)
            if not head_views:
                # the egress buffer is allocated once, so its views are too
                head_views.update(layout.unpack(egress))
            deq = layout.dequantize(head_views)
            return [deq[k] for k in order]
        close = s.close
        ingress = "shared letterbox + native AVX2 plane ingress (what YoloPipeline ships)"

    try:
        for i in range(args.warmup + args.frames):
            t0 = time.perf_counter()
            x, pad, scale = prepare(img)
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
           "ingress": ingress,
           "stages_ms": {k: statistics.fmean(v) for k, v in stages.items()}}
    print(f"[{args.arm}] G2G mean {out['mean']:.3f} ms | P50 {out['p50']:.3f} | P95 {out['p95']:.3f} "
          f"| P99 {out['p99']:.3f} | max {out['max']:.3f} over {n} timed frames")
    print(f"[{args.arm}] ingress: {ingress}")
    print(f"[{args.arm}] stage means (ms): " +
          " | ".join(f"{k} {v:.3f}" for k, v in out["stages_ms"].items()))
    print(f"[{args.arm}] {dets} detections per frame")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())