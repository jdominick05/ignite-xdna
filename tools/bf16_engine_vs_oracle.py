#!/usr/bin/env python3
"""The bf16 engine's numerics against the W8A16 oracle, layer by layer. Offline: no device is opened.

    python tools/bf16_engine_vs_oracle.py --qdq models/sesr_m7_xint8.onnx --fp32 models/sesr_m7_fp32.onnx \
        --input-npy scratch/bf16_vs_oracle/baby.npy --input-npy scratch/bf16_vs_oracle/bird.npy --seed 0

Run it again with ``--mac-model wide --no-schedule`` as the control: "wide" sums each instruction's
nine operands exactly, so what the gap loses between the two runs is the core's aligned accumulate.

WHAT IT MEASURES. How far the bf16 engine's arithmetic, as the core emulator models it, sits from
tools/w8a16_oracle.py: the same dequantized int8 weights run in ONNX Runtime with a bf16 store and
reload around every Conv. Both sides round weights, biases and every stored activation to bf16 the
same way, so what is left between them is the ACCUMULATION. The oracle sums in the order of ORT's
fp32 kernel; the engine sums in the core's one-instruction multiply-accumulate
(``engine_bf16_emulator.mac``, by default the "aligned" model the accumulate probe selected). The
two differ only where that pushes a value across a bf16 rounding boundary, and the difference then
propagates.

THREE COMPARISONS PER INPUT:
  chained    ``graph_reference_bf16.run_direct`` from the input to the last layer, against the
             oracle's tensor for every layer: the gap a whole-network run shows.
  isolated   each layer fed the ORACLE's input tensors, rounded to bf16, so a layer's own
             contribution is separated from what it inherited. Here a third arm, ``exact``, sums the
             same bf16 operands in float64 and rounds once to bf16 - a numpy convolution sharing no
             code with the packer or the emulator - and both sides are counted against it. The oracle
             is ORT's fp32 sum, not the truth about accumulation, so where the two differ this says
             which one left the exact sum.
  schedule   ``engine_schedule_bf16``'s schedule for the whole graph, emulated through its
             workspace, DMA patterns and streams on a NaN background, against the chained reference.
             It must agree to the bit on every layer, and the tool exits 1 if it does not. That
             agreement is what makes the chained gap the schedule's gap too.

EVERY ORACLE TENSOR IS ROUNDED TO bf16 BEFORE ANY COMPARISON. The oracle wraps only Conv in Cast
pairs, so its residual Add output is an unrounded fp32 sum, while the engine's residual packet rounds
on store. On a Conv or Relu output the rounding must change nothing, because a Cast pair already
did it. The tool checks that and exits 1 otherwise, since a change there means the pairs were folded
away and the oracle is fp32.

METRICS, per layer:
  * elements whose bf16 value differs;
  * max_abs in ULPs of the tensor's PEAK - the largest difference divided by the bf16 spacing at
    the largest magnitude either side holds. This is the scale-free figure: 1.0 means one step of
    the coarsest resolution the tensor has;
  * the largest ELEMENTWISE distance in bf16 ULPs, taking the 16-bit patterns as ordered integers;
  * rel_l2 and max_abs.
The elementwise ULP distance is NOT scale-free, and the first SESR run showed why. A difference
inherited from a large value lands on a small one as hundreds of that value's ULPs, and a ReLU edge
- 0.0 on one side, 0.0256 on the other - is 15,570 of them, while both are a fraction of one ULP at
the tensor's peak. It is kept because a large figure there is still a reason to look. Patterns that
differ only in the sign of a zero are counted apart, since they are equal values. At the graph
output, the oracle's own distance to the same model without the bf16 step is printed beside the
engine's, for scale.

WHAT IT IS NOT. It compares the emulator against the oracle, on the CPU. It predicts the device
against the oracle only if the device matches the emulator, which is a separate, bit-exact contract
checked on silicon. The oracle's fp32 summation order is ORT's MLAS kernel for THIS CPU, so the gap
can move from one machine to another, and the output names the machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import onnx  # noqa: E402

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule_bf16 as eb  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler import graph_reference_bf16 as gr16  # noqa: E402
from ignite_xdna.compiler.graph_reference import ort_intermediates  # noqa: E402
import w8a16_oracle as wo  # noqa: E402

NAN_BITS = 0x7FC0


def ordered(bits: np.ndarray) -> np.ndarray:
    """bf16 patterns as integers in value order, so a ULP distance is a difference. +0 and -0 are both 0."""
    b = np.asarray(bits, np.uint16).astype(np.int32)
    mag = b & 0x7FFF
    return np.where(b & 0x8000, -mag, mag)


def bf16_spacing(v: float) -> float:
    """The gap between adjacent bf16 values at magnitude ``v``: eight significant bits."""
    if v == 0 or not np.isfinite(v):
        return 0.0
    _, e = np.frexp(abs(v))                      # v = m * 2**e, m in [0.5, 1)
    return float(np.ldexp(1.0, int(e) - 8))


def round_bf16_f64(v: np.ndarray) -> np.ndarray:
    """float64 to the nearest bf16 value, ties to even, in ONE rounding. Going through float32 first
    would round twice, and a double rounding flips about one value in 2^16 - the size of the effect
    being measured."""
    m, e = np.frexp(np.asarray(v, np.float64))
    return np.ldexp(np.rint(m * 256.0), e - 8)


def compare(engine: np.ndarray, oracle: np.ndarray) -> dict:
    """Two float32 arrays of bf16 values, same shape: how far apart they are."""
    e, o = np.asarray(engine, np.float32), np.asarray(oracle, np.float32)
    if e.shape != o.shape:
        raise ValueError(f"engine {e.shape} against oracle {o.shape}")
    be, bo = em.bf16_bits(e), em.bf16_bits(o)
    ulp = np.abs(ordered(be) - ordered(bo))
    d = e.astype(np.float64) - o.astype(np.float64)
    finite = bool(np.isfinite(e).all() and np.isfinite(o).all())
    max_abs = float(np.abs(d).max())
    peak = float(max(np.abs(e).max(), np.abs(o).max())) if finite else float("nan")
    step = bf16_spacing(peak)
    return {"n": int(e.size), "differ": int(np.count_nonzero(ulp)),
            "zero_sign_only": int(np.count_nonzero((be != bo) & (ulp == 0))),
            "peak": peak, "max_abs_peak_ulps": max_abs / step if step else 0.0,
            "max_ulp": int(ulp.max()), "rel_l2": wo.rel_l2(e.astype(np.float64), o.astype(np.float64)),
            "max_abs": max_abs, "finite": finite}


def attribute(engine: np.ndarray, oracle: np.ndarray, exact: np.ndarray) -> dict:
    """Against the exact sum: how often each side leaves it, and at the elements where engine and oracle
    disagree, which of them kept it."""
    ke, ko, kx = (ordered(em.bf16_bits(a)) for a in (engine, oracle, exact))
    split = ke != ko
    return {"engine_differs": int(np.count_nonzero(ke != kx)), "oracle_differs": int(np.count_nonzero(ko != kx)),
            "split_engine_exact": int(np.count_nonzero(split & (ke == kx))),
            "split_oracle_exact": int(np.count_nonzero(split & (ko == kx))),
            "split_neither": int(np.count_nonzero(split & (ke != kx) & (ko != kx)))}


def exact_layer(ir, layer, feed: dict):
    """float32 ``[Cout][H][W]``: the layer on the same bf16 operands in float64, rounded to bf16 once.

    Independent of the packer and the core emulator: the weights come straight from the IR's int8
    values and scale, the bias is the IR's ``bias_q * bias_scale`` rounded to bf16 as both sides store
    it, and the epilogue is the model's (round, then ReLU, then a residual add rounded again). float64
    carries 53 bits against bf16's 8, so this is the exact sum for every value not sitting within
    2^-45 of a rounding boundary. Single-input layers only; None otherwise.
    """
    if len(layer.inputs) != 1 or layer.inputs[0].block_offset != 0:
        return None
    t = ir.tensors[layer.output]
    x = np.asarray(feed[layer.inputs[0].tensor], np.float64)[:layer.cin]
    w = layer.weights.astype(np.float64) * float(layer.weight_scale)
    b = em.to_bf16((layer.bias_q.astype(np.float64) * float(layer.bias_scale)).astype(np.float32)).astype(np.float64)
    k, s, p = layer.k, layer.stride, layer.pad
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    acc = np.repeat(b[:, None, None], t.height * t.width, axis=1).reshape(-1, t.height, t.width)
    for ky in range(k):
        for kx in range(k):
            win = xp[:, ky:ky + s * (t.height - 1) + 1:s, kx:kx + s * (t.width - 1) + 1:s]
            acc = acc + np.einsum("oc,chw->ohw", w[:, :, ky, kx], win)
    y = round_bf16_f64(acc)
    if layer.act == "relu":
        y = np.maximum(y, 0.0)
    if layer.residual is not None:
        y = round_bf16_f64(y + np.asarray(feed[layer.residual.tensor], np.float64)[:t.channels])
    return (y + 0.0).astype(np.float32)


def depth_to_space(x: np.ndarray, r: int, mode: str) -> np.ndarray:
    """ONNX DepthToSpace on one ``[C][H][W]`` image."""
    c, h, w = x.shape
    if mode == "DCR":
        return x.reshape(r, r, c // (r * r), h, w).transpose(2, 3, 0, 4, 1).reshape(c // (r * r), h * r, w * r)
    return x.reshape(c // (r * r), r, r, h, w).transpose(0, 3, 1, 4, 2).reshape(c // (r * r), h * r, w * r)


def graph_output(model: onnx.ModelProto, names: dict, engine: dict):
    """(output name, engine value or None) when the float graph's output is a layer output or a
    DepthToSpace of one. Anything else is reported as not compared rather than guessed at."""
    out = model.graph.output[0].name
    by_float = {v: k for k, v in names.items()}
    if out in by_float:
        return out, engine[by_float[out]]
    prod = next((n for n in model.graph.node if out in n.output), None)
    if prod is not None and prod.op_type == "DepthToSpace" and prod.input[0] in by_float:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in prod.attribute}
        mode = attrs.get("mode", b"DCR")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        return out, depth_to_space(engine[by_float[prod.input[0]]], int(attrs["blocksize"]), mode)
    return out, None


def provenance() -> dict:
    """Where the engine modules were imported from, and the sha256 of each file that made the numbers.

    An editable install elsewhere would shadow this tree, and the numbers would be another tree's.
    """
    mods = {m.__name__.rsplit(".", 1)[-1]: Path(m.__file__).resolve() for m in (eb, gr16, em)}
    src = (ROOT / "src").resolve()
    stray = {k: str(v) for k, v in mods.items() if src not in v.parents}
    if stray:
        raise SystemExit(f"engine modules imported from outside {src}: {stray}")
    files = [Path(__file__).resolve(), Path(wo.__file__).resolve(), *mods.values()]
    return {p.relative_to(ROOT.resolve()).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files}


def measure(qdq, fp32, inputs, mac_model=None, emulate=True, isolated=True, host_regions=(), silu_sigmoid=False):
    """Compare engine and oracle on each ``(label, NCHW float32)`` input.

    Returns one report per input. A report's ``ok`` is False when the schedule departs from the
    chained reference by a single pattern, when the oracle's rounding changed a tensor that a Cast
    pair should already have rounded, or when anything is non-finite. ``emulate`` and ``isolated``
    each cost about as much as the chained run itself.
    """
    mac_model = em.MAC_MODEL if mac_model is None else mac_model
    ir = graph_ir.lower_yolov8n(str(qdq), host_regions=host_regions, silu_sigmoid=silu_sigmoid)
    eb.check_ir(ir)
    float_model = onnx.load(str(fp32))
    names = wo.layer_tensor_names(float_model, ir)
    oracle, _ = wo.build(qdq, fp32, True, host_regions, silu_sigmoid)
    plain, _ = wo.build(qdq, fp32, False, host_regions, silu_sigmoid)
    out_name = float_model.graph.output[0].name
    if emulate:
        ws = eb.plan_workspace(ir)
        scheds, store = eb.schedule_graph(ir, ws)
    channels = {n: t.channels for n, t in ir.tensors.items()}

    reports = []
    for label, x in inputs:
        x = np.asarray(x, np.float32)
        want = sorted({names[L.output] for L in ir.layers} | {out_name})
        raw = ort_intermediates(oracle, x, want, elem_type=onnx.TensorProto.FLOAT)
        ref = {ir.input: em.to_bf16(x[0])}
        rounding = {}
        for L in ir.layers:
            r = np.ascontiguousarray(raw[names[L.output]], np.float32)
            ref[L.output] = em.to_bf16(r)
            # Elements the rounding changed, i.e. that were not bf16 values already.
            rounding[L.name] = int(np.count_nonzero(ref[L.output].view(np.uint32) != r.view(np.uint32)))
        bad_rounding = {L.name: rounding[L.name] for L in ir.layers if L.residual is None and rounding[L.name]}

        chained = gr16.run_direct(ir, x[0], mac_model=mac_model)
        engine = {n: chained[n][:channels[n]] for n in chained}
        layers = []
        for L in ir.layers:
            row = {"layer": L.name, "oracle_tensor": names[L.output], "channels": channels[L.output],
                   "oracle_rounded_by_to_bf16": rounding[L.name],
                   "chained": compare(engine[L.output], ref[L.output])}
            if isolated:
                feed = {s.tensor: ref[s.tensor] for s in L.inputs + ([L.residual] if L.residual else [])}
                iso = gr16.direct_layer(ir, L, feed, mac_model=mac_model)[:channels[L.output]]
                row["isolated"] = compare(iso, ref[L.output])
                ex = exact_layer(ir, L, feed)
                if ex is not None:
                    row["exact"] = attribute(iso, ref[L.output], ex)
            layers.append(row)

        sched_differ = None
        if emulate:
            arr = ws.halo_fill(background=NAN_BITS)
            ws.write_values(arr, ir.input, chained[ir.input])
            sched_differ = {}
            for s, L, row in zip(scheds, ir.layers, layers):
                eb.emulate_layer(s, store, arr, mac_model=mac_model)
                got = ws.read_tensor(arr, L.output)
                sched_differ[L.name] = int(np.count_nonzero(got != em.bf16_bits(chained[L.output])))
                row["schedule_differs_from_chained"] = sched_differ[L.name]

        oname, oval = graph_output(float_model, names, engine)
        output = {"name": oname}
        if oval is not None:
            y_oracle = np.asarray(raw[out_name], np.float32)[None]
            y_plain = wo.run(plain, x)
            output.update(compare(oval, em.to_bf16(y_oracle[0])))
            output["rel_l2_engine_vs_model_without_bf16"] = wo.rel_l2(oval[None].astype(np.float64), y_plain)
            output["rel_l2_oracle_vs_model_without_bf16"] = wo.rel_l2(y_oracle.astype(np.float64), y_plain)
        else:
            output["compared"] = False

        finite = all(r[k]["finite"] for r in layers for k in ("chained", "isolated") if k in r)
        ok = finite and not bad_rounding and (sched_differ is None or not any(sched_differ.values()))
        reports.append({"input": label, "mac_model": mac_model, "layers": layers, "output": output,
                        "oracle_rounding_changed_a_rounded_tensor": bad_rounding,
                        "schedule_emulated": emulate, "ok": ok})
    return reports


def _input_label(path: Path, x: np.ndarray) -> str:
    return f"{path.as_posix()} sha256 {hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]}"


def _table(rep: dict) -> str:
    """Columns: differ = elements with another bf16 value; pk = max_abs in ULPs of the tensor's peak;
    ulp = the largest elementwise ULP distance; exact = elements where engine / oracle leave the
    float64 sum, then of the split elements how many the engine / oracle / neither kept exact."""
    rows = [f"input {rep['input']}  mac model {rep['mac_model']}",
            f"  {'layer':22s} {'ch':>2s} | {'chained: differ':>15s} {'pk':>5s} {'ulp':>5s} {'rel_l2':>8s}"
            f" | {'isolated: differ':>16s} {'pk':>5s} {'ulp':>4s} | {'exact: eng/orc':>14s} {'split e/o/-':>11s}"
            f" | {'sched':>5s} {'rnd':>6s}"]
    for r in rep["layers"]:
        c, i, x = r["chained"], r.get("isolated"), r.get("exact")
        iso = (f"{i['differ']:9d}/{i['n']:<7d}{i['max_abs_peak_ulps']:5.2f} {i['max_ulp']:4d}" if i
               else f"{'-':>16s} {'-':>5s} {'-':>4s}")
        exa = (f"{x['engine_differs']:6d}/{x['oracle_differs']:<7d} {x['split_engine_exact']:3d}/"
               f"{x['split_oracle_exact']:3d}/{x['split_neither']:<3d}" if x else f"{'-':>14s} {'-':>11s}")
        rows.append(f"  {r['layer'][-22:]:22s} {r['channels']:2d} | {c['differ']:7d}/{c['n']:<7d} "
                    f"{c['max_abs_peak_ulps']:5.2f} {c['max_ulp']:5d} {c['rel_l2']:8.1e} | {iso} | {exa} | "
                    f"{str(r.get('schedule_differs_from_chained', '-')):>5s} {r['oracle_rounded_by_to_bf16']:6d}")
    o = rep["output"]
    if o.get("compared", True):
        rows.append(f"  output {o['name']}: differ {o['differ']}/{o['n']}, max_abs {o['max_abs']:.3e} = "
                    f"{o['max_abs_peak_ulps']:.2f} ULP of its peak {o['peak']:g}, max elementwise ulp {o['max_ulp']}, "
                    f"rel_l2 {o['rel_l2']:.3e}; against the model without the bf16 step: engine "
                    f"{o['rel_l2_engine_vs_model_without_bf16']:.4e}, oracle {o['rel_l2_oracle_vs_model_without_bf16']:.4e}")
    else:
        rows.append(f"  output {o['name']}: not compared (not a layer output or a DepthToSpace of one)")
    rows.append(f"  ok {rep['ok']}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qdq", type=Path, required=True, help="the QDQ model the engine container is built from")
    ap.add_argument("--fp32", type=Path, required=True, help="the float export that QDQ model was quantized from")
    ap.add_argument("--input-npy", type=Path, action="append", default=[],
                    help="a preprocessed NCHW float input; repeatable")
    ap.add_argument("--seed", type=int, action="append", default=[],
                    help="add w8a16_oracle's seeded integer input; repeatable; seed 0 when no input is given")
    ap.add_argument("--mac-model", choices=em.MAC_MODELS, default=em.MAC_MODEL)
    ap.add_argument("--no-schedule", action="store_true", help="skip the whole-graph schedule emulation")
    ap.add_argument("--no-isolated", action="store_true", help="skip the per-layer isolated comparison")
    ap.add_argument("--host-regions", nargs="*", default=[])
    ap.add_argument("--silu-sigmoid", action="store_true")
    args = ap.parse_args()

    inputs = [(_input_label(p, x), x) for p in args.input_npy for x in [np.load(p)]]
    for seed in args.seed or ([] if inputs else [0]):
        x = wo.seeded_input(args.qdq, seed)
        inputs.append((f"seeded integers [-128,127], seed {seed}", x))
    print("BF16_VS_ORACLE_SETUP " + json.dumps({
        "machine": platform.node(), "qdq": args.qdq.name, "fp32": args.fp32.name, "mac_model": args.mac_model,
        "mac_model_in_force": em.MAC_MODEL, "modules": provenance(), "inputs": [lbl for lbl, _ in inputs],
        "schedule_background": hex(NAN_BITS)}, sort_keys=True), flush=True)

    reports = measure(args.qdq, args.fp32, inputs, mac_model=args.mac_model, emulate=not args.no_schedule,
                      isolated=not args.no_isolated, host_regions=args.host_regions,
                      silu_sigmoid=args.silu_sigmoid)
    for rep in reports:
        print(_table(rep), flush=True)
        print("BF16_VS_ORACLE " + json.dumps(rep, sort_keys=True), flush=True)
    keys = ("differ", "max_abs_peak_ulps", "max_ulp", "rel_l2", "max_abs")
    exact = [row["exact"] for r in reports for row in r["layers"] if "exact" in row]
    worst = {"chained_any_layer": {k: max(row["chained"][k] for r in reports for row in r["layers"]) for k in keys},
             "output": {k: max(r["output"].get(k, 0) for r in reports) for k in keys},
             "isolated_exact_totals": {k: sum(e[k] for e in exact) for k in exact[0]} if exact else None,
             "all_ok": all(r["ok"] for r in reports)}
    print("BF16_VS_ORACLE_WORST " + json.dumps(worst, sort_keys=True), flush=True)
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
