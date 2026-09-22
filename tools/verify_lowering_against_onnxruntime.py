#!/usr/bin/env python3
"""Check a lowered graph's integer reference against ONNX Runtime on the untouched model.

    bash scripts/research-iron.sh tools/verify_lowering_against_onnxruntime.py \
        --model models/yolov8n_cut_xint8.onnx [--recipe modnet_cut] [--image assets/bus.jpg]

This answers "does the lowering still compute the model", not "does the device compute the
lowering" - there is no device here. ``graph_reference.run_direct`` walks the lowered IR in
integers and ``graph_reference.host_session`` runs the original ONNX; every graph output the
reference produces is compared element for element. Exit 1 on any difference.

It also prints the Conv -> Clip -> QuantizeLinear census, because whether a ReLU6 bound can bind
decides whether that layer is allowed on the engine at all: the epilogue is max(q, ZP) with the
store saturating at 255, so a bound at or above quantum 255 is already performed by saturation
and one below it is a clamp the integer epilogue cannot express. A layer listed as BINDS is one
the lowering refuses on purpose.

``--recipe`` selects the dense path (BiSeNetV2, MODNet-Cut); without it the whole-model path runs,
where a single refused convolution fails the build rather than falling back to the host.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from ignite_xdna.compiler import graph_reference as gr  # noqa: E402
from ignite_xdna.compiler.graph_ir import ConvLayer, HostLayer  # noqa: E402

ZP = 128


def const_of(g, name):
    """An initializer or Constant-node value, or None when the tensor is computed."""
    for t in g.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    for n in g.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def clip_census(model):
    """Every Conv -> Clip -> QuantizeLinear, and where its upper bound lands in output quanta."""
    g = model.graph
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    rows = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        cons = consumers.get(n.output[0], [])
        if not cons or cons[0].op_type != "Clip":
            continue
        clip = cons[0]
        qs = [c for c in consumers.get(clip.output[0], []) if c.op_type == "QuantizeLinear"]
        if not qs:
            continue
        lo = const_of(g, clip.input[1]) if len(clip.input) > 1 else None
        hi = const_of(g, clip.input[2]) if len(clip.input) > 2 else None
        scale = const_of(g, qs[0].input[1])
        zp = const_of(g, qs[0].input[2]) if len(qs[0].input) > 2 else np.array(0)
        s = float(np.ravel(scale)[0])
        z = int(np.ravel(zp)[0])
        hi_v = float(np.ravel(hi)[0]) if hi is not None else float("inf")
        lo_v = float(np.ravel(lo)[0]) if lo is not None else float("-inf")
        group = next((int(a.i) for a in n.attribute if a.name == "group"), 1)
        rows.append((n.name, group > 1, lo_v, hi_v, s, z + hi_v / s))
    return rows


def make_input(model, image):
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if not image:
        # Correctness here is input-independent, so a fixed seed keeps the run reproducible.
        return np.random.default_rng(136).normal(size=shape).astype(np.float32), shape
    import cv2
    img = cv2.imread(str(ROOT / image))
    if img is None:
        raise SystemExit(f"cannot read {image}")
    img = cv2.resize(img, (shape[3], shape[2]))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None].astype(np.float32), shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a quantized ONNX, relative to the repo")
    ap.add_argument("--recipe", default=None, choices=["bisenetv2", "modnet_cut"],
                    help="lower through dense_regions instead of the whole-model path")
    ap.add_argument("--image", default=None, help="real image instead of a fixed pseudo-random input")
    args = ap.parse_args()

    path = ROOT / args.model
    if not path.exists():
        raise SystemExit(f"missing {path}")
    model = onnx.load(str(path))
    print(f"model    {args.model}", flush=True)
    print(f"input    {model.graph.input[0].name} "
          f"{[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}", flush=True)

    rows = clip_census(model)
    binds = [r for r in rows if r[5] < 255]
    print(f"\nConv -> Clip -> QuantizeLinear: {len(rows)} layers, {len(binds)} with a binding bound")
    if rows:
        print("  a bound at or above quantum 255 is already performed by uint8 saturation, so the")
        print("  layer is a plain ReLU; below 255 it is a clamp this epilogue cannot express.")
        for name, dw, lo, hi, s, q_hi in binds:
            print(f"  BINDS  {name[:70]:<70s} dw={'y' if dw else 'n'} "
                  f"[{lo:.0f},{hi:.0f}] scale {s:.6f} -> quantum {q_hi:.1f}")

    t0 = time.time()
    if args.recipe:
        from ignite_xdna.compiler.dense_regions import lower_dense
        ir = lower_dense(model, args.recipe)
    else:
        from ignite_xdna.compiler.graph_ir import lower_yolov8n
        ir = lower_yolov8n(str(path))
    conv = [ly for ly in ir.layers if isinstance(ly, ConvLayer)]
    host = [ly for ly in ir.layers if isinstance(ly, HostLayer)]
    print(f"\nlowered in {time.time() - t0:.1f}s: {len(conv)} engine layers, {len(host)} host layers",
          flush=True)

    # Depthwise is lowered as the dense convolution whose off-diagonal taps are zero, so a model
    # with depthwise layers on the engine carries weight packets that are entirely zero. Both
    # counts belong in the same log: one says how much of the model got there, the other how much
    # of what it streams is structurally nothing.
    dw = {n.name for n in model.graph.node if n.op_type == "Conv"
          and next((int(a.i) for a in n.attribute if a.name == "group"), 1) > 1}
    if dw:
        on_engine = [ly.name for ly in conv if ly.name in dw]
        print(f"depthwise convolutions on the engine: {len(on_engine)} of {len(dw)}")

    from ignite_xdna.compiler import engine_emulator as em, engine_schedule as es
    ws = es.plan_workspace(ir)
    _scheds, store = es.schedule_graph(ir, ws)
    blob = store.blob()
    npkt = blob.size // em.W_BYTES
    if npkt:
        pkts = blob[:npkt * em.W_BYTES].reshape(npkt, em.W_BYTES)
        hdr = pkts[:, :em.HDR_BYTES].copy().view(np.int32)
        is_conv = hdr[:, em.H_OP] == em.OP_CONV
        zero = ~pkts[:, em.W_OFFSET:].any(axis=1)
        holds = (hdr[:, em.H_FLAGS] & (em.F_EMIT | em.F_HOLD)) != 0
        drop = int((is_conv & zero & ~holds).sum())
        print(f"weight packets {npkt:,} ({int(is_conv.sum()):,} conv), "
              f"all-zero conv {int((is_conv & zero).sum()):,}, "
              f"droppable {drop:,} ({drop / npkt:.1%})", flush=True)

    x, _ = make_input(model, args.image)
    # run_direct takes the already-quantized plane on the whole-model path, where the model's own
    # ingress QuantizeLinear is not part of the lowered graph; the dense path keeps the ingress
    # inside its first host region and so takes the float image.
    if args.recipe:
        staged = x[0]
    else:
        t_in = ir.tensors[ir.input]
        staged = gr.quantize_input(x[0], t_in.scale, t_in.zero_point)
    t0 = time.time()
    direct = gr.run_direct(ir, staged)
    t_ref = time.time() - t0

    names = [ly.output for ly in conv]
    t0 = time.time()
    ort = gr.ort_intermediates(str(path), x, names)
    t_ort = time.time() - t0
    print(f"integer reference {t_ref:.1f}s, onnxruntime {t_ort:.1f}s", flush=True)

    bad = []
    for ly in conv:
        c = ir.tensors[ly.output].channels
        got, ref = direct[ly.output][:c], ort[ly.output]
        if not np.array_equal(got, ref):
            d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            bad.append((ly.name, int((d > 0).sum()), d.size, d.max()))
    print()
    for name, n, size, mx in bad[:20]:
        print(f"  MISMATCH {name[:70]:<70s} {n:,} of {size:,}, max |delta| {mx}")
    print(f"{len(conv) - len(bad)}/{len(conv)} engine layer tensors byte-exact against ONNX Runtime")

    # For a dense lowering the interesting output is the model's, downstream of the host regions.
    tail = 0
    if args.recipe:
        outputs = gr.host_session(model.SerializeToString()).run(
            None, {model.graph.input[0].name: x})
        for vi, ref in zip(model.graph.output, outputs):
            if vi.name not in direct:
                continue
            got = direct[vi.name]
            if got.ndim == ref.ndim - 1:
                got = got[None]
            ok = np.array_equal(got, ref)
            tail += 0 if ok else 1
            print(f"  graph output {vi.name:<30s} {'EXACT' if ok else 'MISMATCH'} "
                  f"{tuple(ref.shape)} {ref.dtype}")
    return 1 if (bad or tail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
