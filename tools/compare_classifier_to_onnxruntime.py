#!/usr/bin/env python3
"""Run a whole classifier's lowering through the integer reference and compare it with ONNX Runtime.

    python tools/compare_classifier_to_onnxruntime.py --model models/yolov8n-cls_640_cut_xint8.onnx \
        [--image assets/bus.jpg] [--topk 5]

Closes the second link of the chain the silicon verifier cannot: ``verify_engine_container.py`` proves
the device matches ``graph_reference.run_direct``, and this proves ``run_direct`` matches the model on
a real image -- same quantized input, reference on one side, ONNX Runtime CPU on the other, logits
compared after the container's own egress dequantization. It reports top-1, top-k and the largest
logit error; it does not measure accuracy against labels, and it needs no device.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ignite_xdna.compiler import engine_schedule as es, graph_ir, graph_reference as gr  # noqa: E402
from ignite_xdna.runtime.heads import resolve_classification_layout  # noqa: E402
import verify_engine_container as verify  # noqa: E402  (same input staging as the device verifier)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--container", help="optional .ignite whose manifest supplies the egress layout")
    args = ap.parse_args(argv)

    import cv2
    import onnxruntime as ort

    ir = graph_ir.lower_yolov8n(args.model)
    hosts = sum(1 for L in ir.layers if isinstance(L, graph_ir.HostLayer))
    print(f"[compare] {Path(args.model).name}: {len(ir.layers)} layers, {hosts} on the host")

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2
    in_hw = (ir.tensors[ir.input].height, ir.tensors[ir.input].width)
    rgb = cv2.cvtColor(cv2.resize(bgr, (in_hw[1], in_hw[0])), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    ref = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0],
                      dtype=np.float32).reshape(-1)

    q_in = verify.quantized_input(ir, Path(args.image), "classify")
    out = gr.run_direct(ir, q_in)
    ws = es.plan_workspace(ir)
    scheds, store = es.schedule_graph(ir, ws)
    from ignite_xdna.compiler.engine_compile import build_manifest  # noqa: PLC0415
    mf = build_manifest(ir, ws, scheds, store, Path(args.model).stem, 0, "x", "y", 0.0)
    layout = resolve_classification_layout(mf)
    if layout is None:
        print("[compare] this graph does not lower to a classification egress", file=sys.stderr)
        return 2
    plane = out[ir.layers[-1].output]
    q = plane[:, 0, 0][:layout.num_classes]
    got = layout.dequantize(q).astype(np.float32)

    n = min(got.size, ref.size)
    g, r = got[:n], ref[:n]
    print(f"[compare] egress: {layout.num_classes} classes, scale {layout.scale}, unpacked from "
          f"{plane.shape[0]} channels at pixel (0, 0)")
    print(f"  reference top-{args.topk}: {g.argsort()[-args.topk:][::-1].tolist()}")
    print(f"  onnxruntime top-{args.topk}: {r.argsort()[-args.topk:][::-1].tolist()}")
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))
    print(f"  top-1 {int(g.argmax())} vs {int(r.argmax())} (agree={int(g.argmax()) == int(r.argmax())}) | "
          f"max|diff| {np.abs(g - r).max():.4f} | mean|diff| {np.abs(g - r).mean():.4f} | cosine {cos:.5f}")
    same = bool(np.array_equal(g.argsort()[-args.topk:][::-1], r.argsort()[-args.topk:][::-1]))
    ok = same and int(g.argmax()) == int(r.argmax())
    print(f"[compare] {'MATCH' if ok else 'DIVERGES'}: the integer reference "
          f"{'reproduces' if ok else 'does not reproduce'} the model's own ranking")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
