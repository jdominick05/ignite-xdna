#!/usr/bin/env python3
"""Build the W8A16 CPU oracle for a QDQ model: its lowered int8 weights, dequantized, in bf16 arithmetic.

    python tools/w8a16_oracle.py --qdq models/sesr_m7_xint8.onnx --fp32 models/sesr_m7_fp32.onnx \
        --out scratch/sesr_m7_xint8_w8a16_oracle.onnx

WHAT IT IS. The external truth a bf16 engine running int8 work (design B, W8A16) is developed
against. The bf16 packer dequantizes ``ConvLayer.weights * weight_scale`` and
``bias_q * bias_scale`` from the lowered IR at pack time, so this oracle reads exactly those fields
- from ``graph_ir.lower_yolov8n``, the entry point the compiler uses - and NOT the original floats of
the fp32 export. Those are a different model: the quantizer changed them, and a comparison against
them would measure quantization, not the engine.

HOW. The fp32 export supplies the graph (its Relu, Add, DepthToSpace and layout). Each Conv's weight
and bias initializer is replaced BY NODE NAME with the IR's dequantized values, and the mapping is
asserted one to one with matching shapes, so a renamed or missing node raises rather than being
patched by position. Then ``onnx_bf16_cast.cast_model`` rounds the weights to bf16 once and wraps
every Conv's input and output in a Cast(BFLOAT16)->Cast(FLOAT) pair - the core's store-and-reload.

WHAT IT PRINTS, as measured outputs rather than constants (all on one seeded integer input in the
model's activation range, or on ``--input-npy``):

  * ``rel_l2(dequant_fp32, qdq)`` - the oracle without the bf16 step against the QDQ model itself.
    Only activation quantization separates them. On a real image it should sit BELOW the original
    fp32 export's distance to the QDQ model, which a wrong weight mapping cannot manage. The seeded
    default input is uniform noise, which saturates activation quantization and inflates this
    figure (SESR-M7: 0.139 on noise against 0.018-0.020 on Set5 tiles), so pass ``--input-npy`` for
    a number worth quoting; the default is a smoke check only.
  * ``rel_l2(oracle, dequant_fp32)`` - the bf16 rounding gap on its own.
  * the Cast count in the saved model against the count inserted. The session runs with
    ORT_DISABLE_ALL (graph_reference), without which the pairs fold away and the oracle is fp32.

It models precision, not the kernel: the fidelity caveats in onnx_bf16_cast.py's docstring apply.
Where the engine fuses a residual Add into its epilogue, this oracle adds in fp32 and rounds at the
next Conv's input; that difference is inside the measured tolerance, not asserted away.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler.graph_ir import ConvLayer  # noqa: E402
from onnx_bf16_cast import cast_model  # noqa: E402


def dequantized_convs(qdq_path, host_regions=(), silu_sigmoid=False) -> dict:
    """node name -> (float32 weights, float32 bias) exactly as the bf16 packer will read them."""
    ir = graph_ir.lower_yolov8n(str(qdq_path), host_regions=host_regions, silu_sigmoid=silu_sigmoid)
    out = {}
    for L in ir.layers:
        if not isinstance(L, ConvLayer):
            raise ValueError(f"{type(L).__name__} {L.name}: only ConvLayer is on the W8A16 path")
        w = L.weights.astype(np.float32) * np.float32(L.weight_scale)
        b = L.bias_q.astype(np.float32) * np.float32(L.bias_scale)
        out[L.name] = (w, b)
    return out


def patch_weights(model: onnx.ModelProto, convs: dict) -> None:
    """Write the IR's dequantized weights into ``model``'s Conv initializers by node name, in place."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    nodes = [n for n in g.node if n.op_type == "Conv"]
    names = [n.name for n in nodes]
    if len(set(names)) != len(names):
        raise ValueError("duplicate Conv node names in the float model")
    missing, extra = sorted(set(convs) - set(names)), sorted(set(names) - set(convs))
    if missing or extra:
        raise ValueError(f"Conv names do not map one to one: IR only {missing}, float model only {extra}")
    uses = {}
    for n in g.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1
    for n in nodes:
        w, b = convs[n.name]
        wname = n.input[1]
        if wname not in inits or uses[wname] != 1:
            raise ValueError(f"{n.name}: weight {wname} is not a private initializer")
        old = numpy_helper.to_array(inits[wname])
        if old.shape != w.shape:
            raise ValueError(f"{n.name}: float weights {old.shape}, IR weights {w.shape}")
        inits[wname].CopyFrom(numpy_helper.from_array(w, wname))
        bname = n.input[2] if len(n.input) > 2 else ""
        if bname:
            if bname not in inits or uses[bname] != 1:
                raise ValueError(f"{n.name}: bias {bname} is not a private initializer")
            if numpy_helper.to_array(inits[bname]).shape != b.shape:
                raise ValueError(f"{n.name}: bias shape mismatch")
            inits[bname].CopyFrom(numpy_helper.from_array(b, bname))
        elif np.any(b != 0):
            raise ValueError(f"{n.name}: the float Conv has no bias but the IR's is nonzero")


QDQ_SUFFIX = "_QuantizeLinear_Output"


def layer_tensor_names(model: onnx.ModelProto, ir) -> dict:
    """IR tensor name -> the float model's tensor holding the same values, for the input and every layer.

    ``model`` is the float export BEFORE ``cast_model`` (the oracle republishes every name, so the
    result reads the oracle too). The walk is driven by the IR, never by searching for a likely op: from
    the layer's Conv node, a ``relu`` layer steps through exactly one Relu, and a residual layer then
    through exactly one Add whose OTHER input must be the residual producer's own mapped tensor. That
    is the engine's order - the conv's epilogue activates, then OP_RESIDUAL adds with no activation -
    so a graph that adds before activating, clips instead of rectifying, or feeds the Add from
    elsewhere is refused rather than compared against the wrong tensor. Where the IR name carries the
    QDQ suffix, stripping it must give the walked name too.
    """
    g = model.graph
    nodes = {n.name: n for n in g.node}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    names = {ir.input: g.input[0].name}

    def step(layer, tensor, op):
        nxt = consumers.get(tensor, [])
        if len(nxt) != 1 or nxt[0].op_type != op:
            raise ValueError(f"{layer.name}: expected {tensor} to feed exactly one {op}, "
                             f"it feeds {[c.op_type for c in nxt]}")
        return nxt[0]

    for L in ir.layers:
        n = nodes.get(L.name)
        if n is None or n.op_type != "Conv":
            raise ValueError(f"{L.name}: the float model has no Conv node of that name")
        if len(L.inputs) == 1 and L.inputs[0].block_offset == 0 and n.input[0] != names.get(L.inputs[0].tensor):
            raise ValueError(f"{L.name}: the Conv reads {n.input[0]}, the IR's input maps to "
                             f"{names.get(L.inputs[0].tensor)}")
        t = n.output[0]
        if L.act == "relu":
            t = step(L, t, "Relu").output[0]
        elif L.act is not None:
            raise ValueError(f"{L.name}: activation {L.act!r} is not on the W8A16 path")
        if L.residual is not None:
            add = step(L, t, "Add")
            other = [i for i in add.input if i != t]
            want = names.get(L.residual.tensor)
            if want is None or other != [want]:
                raise ValueError(f"{L.name}: the Add's other input is {other}, the residual is {want}")
            t = add.output[0]
        if L.output.endswith(QDQ_SUFFIX) and L.output[:-len(QDQ_SUFFIX)] != t:
            raise ValueError(f"{L.name}: walked to {t}, the IR names {L.output}")
        names[L.output] = t
    return names


def seeded_input(qdq, seed=0) -> np.ndarray:
    """A seeded integer input in the model's activation range, shaped like its graph input."""
    dims = [d.dim_value for d in onnx.load(str(qdq)).graph.input[0].type.tensor_type.shape.dim]
    return np.random.default_rng(seed).integers(-128, 128, size=dims).astype(np.float32)


def build(qdq, fp32, bf16=True, host_regions=(), silu_sigmoid=False):
    """Return (oracle ModelProto, cast stats or None)."""
    model = onnx.load(str(fp32))
    patch_weights(model, dequantized_convs(qdq, host_regions, silu_sigmoid))
    stats = cast_model(model, {"Conv"}) if bf16 else None
    model.metadata_props.add(key="w8a16_oracle_qdq_source", value=Path(qdq).name)
    return model, stats


def run(model, x: np.ndarray) -> np.ndarray:
    from ignite_xdna.compiler.graph_reference import host_session
    m = model if isinstance(model, bytes) else (model.SerializeToString() if isinstance(model, onnx.ModelProto)
                                               else Path(model).read_bytes())
    sess = host_session(m)  # ORT_DISABLE_ALL: the Cast pairs must survive
    return np.asarray(sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0], dtype=np.float64)


def rel_l2(a, ref) -> float:
    return float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qdq", type=Path, required=True, help="the QDQ model the engine container is built from")
    ap.add_argument("--fp32", type=Path, required=True, help="the float export that QDQ model was quantized from")
    ap.add_argument("--out", type=Path, required=True, help="where to write the oracle (scratch/, never models/)")
    ap.add_argument("--host-regions", nargs="*", default=[])
    ap.add_argument("--silu-sigmoid", action="store_true")
    ap.add_argument("--input-npy", type=Path, help="a preprocessed NCHW float input; default a seeded integer one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if "models" in args.out.resolve().parts:
        raise SystemExit("refusing to write under models/ (receive-only Syncthing)")

    oracle, stats = build(args.qdq, args.fp32, True, args.host_regions, args.silu_sigmoid)
    plain, _ = build(args.qdq, args.fp32, False, args.host_regions, args.silu_sigmoid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(oracle, str(args.out))
    saved_casts = sum(n.op_type == "Cast" for n in onnx.load(str(args.out)).graph.node)
    inserted = 2 * (stats["input_cast_chains"] + stats["output_cast_chains"])

    x = np.load(args.input_npy) if args.input_npy else seeded_input(args.qdq, args.seed)
    y_qdq, y_plain, y_oracle = run(args.qdq, x), run(plain, x), run(oracle, x)

    print("W8A16_ORACLE " + json.dumps({
        "qdq": args.qdq.name, "fp32": args.fp32.name, "output": str(args.out), **stats,
        "casts_inserted": inserted, "casts_saved": saved_casts,
        "input": str(args.input_npy) if args.input_npy else f"seeded integers [-128,127], seed {args.seed}",
        "rel_l2_dequant_fp32_vs_qdq": rel_l2(y_plain, y_qdq),
        "rel_l2_oracle_vs_dequant_fp32": rel_l2(y_oracle, y_plain),
        "rel_l2_oracle_vs_qdq": rel_l2(y_oracle, y_qdq),
    }, sort_keys=True), flush=True)
    if saved_casts != inserted:
        print(f"FAIL: {inserted} Casts inserted, {saved_casts} in the saved model", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
