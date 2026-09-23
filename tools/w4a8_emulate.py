#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""W4A8 weight emulation of a shipped XINT8 YOLOv8 model, for a CPU accuracy check.

Every activation Q/DQ, bias and scale of the shipped model stays as it is. Only the Conv weights
are re-quantized to 4 bits, round-to-nearest from the float model, on the grid [-7, 7] (-8 is left
out because its int8 image, -128, is outside the XINT8 dialect).

``emit`` writes one variant:
  w8check  the 8-bit path of this tool; it must rebuild the shipped file exactly (control)
  e1       one pow2 scale per weight tensor (today's engine can run it)
  e2       one pow2 scale per 32-output-channel group, the engine's packet group
  e1h      e1 with the stem and the 6 head-output convs kept at 8 bits (diagnostic)
  u        one pow2 scale per output channel (context: needs a kernel change)

Each group's 4-bit position pos4 is Quark's MinMSE (quant/calib.py choose_pow2_minmse) on the
float weights with bounds [-7, 7] and a --window of candidates around the min/max position.
Two storage forms:
  b  the 4-bit value q4 is stored as q4 * 2**k in the int8 container, k = pos8 - pos4 in [0, 4],
     under the shipped per-tensor scale. The float weights are identical to form a, and the file
     is a plain XINT8 file that the engine lowers.
  a  q4 stored as is, with a per-output-channel scale vector and axis=0 on the weight DQ. ONNX
     Runtime only: graph_ir.scale_zp reads channel 0's scale of a vector and would silently
     miscompile it, so a form-a file must be named *_ortonly.onnx and never reach ignite-compile.
--form auto uses b for a conv whose groups all have k in [0, 4], else a.

After saving, emit checks the file: graph_diff against the shipped model (w8check: identical;
form b: only the Conv weights differ), the grid (every stored weight is a multiple of 2**k and
its q4 is in [-7, 7]) and float identity (stored * scale == q4 * 2**-pos4, exactly).

``emit-float`` is a weight-only diagnostic outside the pre-registered gate: the float model, float
activations and Sigmoid SiLU kept, with each Conv weight replaced by the same q * 2**-pos.

``compare`` runs two files through ONNX Runtime with ORT_DISABLE_ALL on the first --n val2017
images and requires bit-identical outputs (form a against form b, or the shipped file against
w8check).

    python tools/w4a8_emulate.py emit --float models/yolov8n_cut.onnx \\
        --xint8 models/yolov8n_cut_xint8.onnx --variant e2 \\
        --out scratch/int4_w4a8/ --sidecar scratch/int4_w4a8/yolov8n_e2.json
    python tools/w4a8_emulate.py compare A.onnx B.onnx --n 16

scripts/w4a8-eval.sh runs every variant, control and evaluation of the gate.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

from quant.calib import choose_pow2_minmse  # noqa: E402
from quant.graph import Graph  # noqa: E402
from quant.pow2 import pos2scale, scale2pos  # noqa: E402
from quant.verify import graph_diff  # noqa: E402

GRIDS = {"sym7": (-7, 7)}
W8 = (-127, 127)
W8_WINDOW = (-1, 3)      # Quark's MinMSE candidates, quant/calib.py
K_MAX = 4                # q4 * 2**4 = +-112 still fits int8 [-127, 127]
GROUP = 32               # output channels per engine packet (engine_schedule.py, OUT_BLOCKS * 8)
VARIANTS = ("w8check", "e1", "e2", "e1h", "u")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shown(path) -> str:
    """A path as given, with forward slashes; never resolved, so no profile path reaches a log."""
    return str(path).replace("\\", "/")


def quantize_bounded(x: np.ndarray, pos: int, lo: int, hi: int) -> np.ndarray:
    """quant/pow2.quantize with explicit bounds and zp 0: float32 divide, ties to even, clip."""
    scale = pos2scale(pos)
    value = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore"):
        rounded = np.rint(value / scale)
    return np.asarray(np.clip(rounded, lo, hi), dtype=np.int8)


def sqerr_bounded(x: np.ndarray, pos: int, lo: int, hi: int) -> float:
    """quant/pow2.sqerr with explicit bounds: float32 restore and float32 sum."""
    value = np.asarray(x, dtype=np.float32)
    restored = np.asarray(quantize_bounded(value, pos, lo, hi).astype(np.float32) * pos2scale(pos),
                          dtype=np.float32)
    return float(np.sum((restored - value) ** 2, dtype=np.float32))


def choose_pos(x: np.ndarray, lo: int, hi: int, window: tuple[int, int]) -> tuple[int, int, list[int]]:
    """quant/calib.choose_pow2_minmse for symmetric weights, with explicit bounds and window.

    Returns (chosen position, min/max base position, candidates). First minimum wins ties."""
    value = np.asarray(x, dtype=np.float32)
    if not value.size or not np.all(np.isfinite(value)):
        raise ValueError("MinMSE requires nonempty finite float32 samples")
    vmin, vmax = value.min(), value.max()
    extent = np.minimum(max(abs(vmin), abs(vmax)), np.finfo(np.float32).max / 2)
    span = np.float32(np.float32(extent) - np.float32(-extent))
    scale = np.float32(np.float64(span) / (hi - lo))
    if scale < np.finfo(np.float32).tiny:
        scale = np.float32(1)
    base = scale2pos(scale)
    candidates = list(range(base + window[0], base + window[1] + 1))
    if min(candidates) < -127 or max(candidates) > 127:
        raise ValueError("MinMSE candidate range exceeds supported positions")
    errors = [sqerr_bounded(value, pos, lo, hi) for pos in candidates]
    if not all(np.isfinite(errors)):
        raise ValueError("MinMSE float32 error accumulation overflowed")
    return candidates[min(range(len(errors)), key=errors.__getitem__)], base, candidates


def scalar_pos(scale: np.ndarray, what: str) -> int:
    s = np.asarray(scale)
    if s.shape != ():
        raise ValueError(f"{what}: expected a per-tensor scale, got shape {s.shape}")
    pos = scale2pos(float(s))
    if pos2scale(pos) != np.float32(s):
        raise ValueError(f"{what}: scale {float(s)} is not a power of two")
    return pos


def sqnr_db(w: np.ndarray, w_hat: np.ndarray) -> float:
    w64 = w.astype(np.float64)
    noise = float(np.sum((w64 - w_hat.astype(np.float64)) ** 2))
    return math.inf if noise == 0 else 10 * math.log10(float(np.sum(w64 ** 2)) / noise)


def conv_weights(g: Graph, gf: Graph) -> list[dict]:
    """Each Conv of the XINT8 graph with its weight DQ and float weight, the stored int8 checked
    against clip(rint(w / s8), -127, 127) and the scale checked against Quark's MinMSE."""
    by_output = {o: n for n in g.nodes() for o in n.output}
    inits = {t.name: numpy_helper.to_array(t) for t in g.model.graph.initializer}
    finits = {t.name: numpy_helper.to_array(t) for t in gf.model.graph.initializer}
    fconvs = {n.name: n for n in gf.nodes() if n.op_type == "Conv"}
    out = []
    for conv in (n for n in g.nodes() if n.op_type == "Conv"):
        dq = by_output[conv.input[1]]
        if dq.op_type != "DequantizeLinear" or any(a.name == "axis" for a in dq.attribute):
            raise ValueError(f"{conv.name}: weight is not a per-tensor DequantizeLinear")
        qname = dq.input[0]
        if not qname.endswith("_quantized") or qname not in inits:
            raise ValueError(f"{conv.name}: weight initializer {qname} is not *_quantized")
        stored = inits[qname]
        if stored.dtype != np.int8:
            raise ValueError(f"{conv.name}: {qname} is {stored.dtype}, not int8")
        zp = inits[dq.input[2]]
        if zp.shape != () or int(zp) != 0 or zp.dtype != np.int8:
            raise ValueError(f"{conv.name}: weight zero point is not scalar int8 0")
        pos8 = scalar_pos(inits[dq.input[1]], f"{conv.name} weight")
        fname = qname[: -len("_quantized")]
        fconv = fconvs.get(conv.name)
        if fconv is None or fconv.input[1] != fname or fname not in finits:
            raise ValueError(f"{conv.name}: no float weight {fname} on the same Conv of the float model")
        w = np.asarray(finits[fname], dtype=np.float32)
        if w.shape != stored.shape or w.ndim != 4:
            raise ValueError(f"{conv.name}: float {w.shape} vs stored {stored.shape}")
        if not np.array_equal(quantize_bounded(w, pos8, *W8), stored):
            raise ValueError(f"{conv.name}: stored int8 is not clip(rint(w / s8)) of the float weight")
        mine, _, _ = choose_pos(w, *W8, W8_WINDOW)
        quark, _ = choose_pow2_minmse(w, "int8")
        if not mine == quark == pos8:
            raise ValueError(f"{conv.name}: 8-bit MinMSE mismatch: choose_pos {mine}, "
                             f"choose_pow2_minmse {quark}, shipped {pos8}")

        def scale_of(tensor, want):
            node = by_output.get(tensor)
            return None if node is None or node.op_type != want else inits.get(node.input[1])
        sx = scale_of(conv.input[0], "DequantizeLinear")
        sb = scale_of(conv.input[2], "DequantizeLinear") if len(conv.input) > 2 else None
        qs = [n for n in g.consumers(conv.output[0]) if n.op_type == "QuantizeLinear"]
        so = inits.get(qs[0].input[1]) if len(qs) == 1 else None
        shift_out = e_bias = None
        if sx is not None and so is not None:
            shift_out = scalar_pos(sx, "in") + pos8 - scalar_pos(so, "out")
        if sx is not None and sb is not None:
            e_bias = scalar_pos(sx, "in") + pos8 - scalar_pos(sb, "bias")
        out.append({"conv": conv, "dq": dq, "qname": qname, "w": w, "stored": stored, "pos8": pos8,
                    "shift_out": shift_out, "e_bias": e_bias})
    if len(out) != 63:
        raise ValueError(f"expected 63 Convs, found {len(out)}")
    return out


def stem_and_head(g: Graph) -> list[str]:
    """Names of the stem Conv (fed by the quantized graph input) and the 6 head-output Convs (each
    graph output is DQ <- Q <- Conv)."""
    by_output = {o: n for n in g.nodes() for o in n.output}
    graph_inputs = {i.name for i in g.model.graph.input}

    def back(tensor, *ops):
        node = None
        for op in ops:
            node = by_output.get(tensor)
            if node is None or node.op_type != op:
                return None
            tensor = node.input[0]
        return node
    heads = [back(o.name, "DequantizeLinear", "QuantizeLinear", "Conv") for o in g.model.graph.output]
    if len(heads) != 6 or any(h is None for h in heads):
        raise ValueError("expected 6 graph outputs, each DQ <- Q <- Conv")
    stems = [n for n in g.nodes() if n.op_type == "Conv"
             and (dq := by_output.get(n.input[0])) is not None and dq.op_type == "DequantizeLinear"
             and (q := by_output.get(dq.input[0])) is not None and q.op_type == "QuantizeLinear"
             and q.input[0] in graph_inputs]
    names = {n.name for n in heads + stems}
    if len(stems) != 1 or len(names) != 7:
        raise ValueError(f"expected 1 stem and 6 head Convs, 7 distinct; got {len(stems)} stems, {len(names)}")
    return sorted(names)


def groups(cout: int, variant: str) -> list[slice]:
    if variant in ("e1", "e1h", "w8check"):
        return [slice(0, cout)]
    if variant == "e2":
        return [slice(c, min(c + GROUP, cout)) for c in range(0, cout, GROUP)]
    if variant == "u":
        return [slice(c, c + 1) for c in range(cout)]
    raise ValueError(variant)


def quantize_conv(cw: dict, variant: str, lo: int, hi: int, window: tuple[int, int]) -> dict:
    """Per group: pos4, k = pos8 - pos4, q4. An all-zero group takes pos4 = pos8 (k = 0)."""
    w, pos8 = cw["w"], cw["pos8"]
    q4 = np.zeros(w.shape, dtype=np.int8)
    gs = []
    for sl in groups(w.shape[0], variant):
        x = w[sl]
        if not np.any(x):
            gs.append({"slice": sl, "pos4": pos8, "k": 0, "edge": None, "zero": True})
            continue
        pos4, base, cand = choose_pos(x, lo, hi, window)
        q4[sl] = quantize_bounded(x, pos4, lo, hi)
        edge = "low" if pos4 == cand[0] else "high" if pos4 == cand[-1] else None
        gs.append({"slice": sl, "pos4": pos4, "k": pos8 - pos4, "edge": edge, "zero": False,
                   "clipped": int(np.count_nonzero(np.abs(np.rint(x / pos2scale(pos4))) > hi)),
                   "levels": int(np.unique(q4[sl]).size)})
    w_hat = np.zeros(w.shape, dtype=np.float32)
    for gr in gs:
        w_hat[gr["slice"]] = q4[gr["slice"]].astype(np.float32) * pos2scale(gr["pos4"])
    return {"q4": q4, "groups": gs, "w_hat": w_hat,
            "form_b_ok": all(0 <= gr["k"] <= K_MAX for gr in gs)}


def channel_scales(q: dict, cout: int) -> np.ndarray:
    s = np.zeros(cout, dtype=np.float32)
    for gr in q["groups"]:
        s[gr["slice"]] = pos2scale(gr["pos4"])
    return s


def emit_b(g: Graph, cw: dict, q: dict) -> None:
    stored = np.zeros(cw["w"].shape, dtype=np.int16)
    for gr in q["groups"]:
        stored[gr["slice"]] = q["q4"][gr["slice"]].astype(np.int16) << gr["k"]
    if stored.min() < W8[0] or stored.max() > W8[1]:
        raise ValueError(f"{cw['conv'].name}: form b value outside int8 [-127, 127]")
    g.set_initializer(cw["qname"], stored.astype(np.int8))


def form_a_names(cw: dict) -> tuple[str, str]:
    return cw["dq"].input[1] + "_w4pc", cw["dq"].input[2] + "_w4pc"


def rewire_a(g: Graph, cw: dict, scale: np.ndarray, zp: np.ndarray) -> None:
    """Point the weight DQ at new per-channel scale/zp initializers with axis=0; drop the old
    scalars only if nothing else reads them (no global clean, so w8check stays byte-exact)."""
    sname, zname = form_a_names(cw)
    g.set_initializer(sname, scale)
    g.set_initializer(zname, zp)
    dq = next(n for n in g.model.graph.node if n.output[0] == cw["dq"].output[0])
    old = list(dq.input[1:3])
    dq.input[1], dq.input[2] = sname, zname
    dq.attribute.append(helper.make_attribute("axis", 0))
    for name in old:
        if not any(name in n.input for n in g.model.graph.node):
            g.remove_initializer(name)


def emit_a(g: Graph, cw: dict, q: dict) -> None:
    """quant/probe.py weights_per_channel, with the 4-bit scales and new initializer names."""
    cout = cw["w"].shape[0]
    g.set_initializer(cw["qname"], q["q4"])
    rewire_a(g, cw, channel_scales(q, cout), np.zeros(cout, dtype=np.int8))


def expected_graph(shipped: Graph, emitted: Graph, rows: list[dict]) -> Graph:
    """The shipped graph with exactly the intended rewrite applied, values taken from the emitted
    file: graph_diff(expected, emitted) must then be empty."""
    exp = Graph(copy.deepcopy(shipped.model))
    for r in rows:
        if r["form"] is None:
            continue
        exp.set_initializer(r["qname"], emitted.initializer(r["qname"]))
        if r["form"] == "a":
            sname, zname = form_a_names(r["cw"])
            rewire_a(exp, r["cw"], emitted.initializer(sname), emitted.initializer(zname))
    return exp


def check_emitted(path: Path, shipped: Graph, rows: list[dict], variant: str, lo: int, hi: int) -> list[str]:
    """Post-save controls; returns failure strings (empty = pass)."""
    fails = []
    em = Graph.load(path, strict=True)
    changed = {r["qname"] for r in rows if r["form"] is not None}
    any_a = any(r["form"] == "a" for r in rows)
    if not any_a:
        d = graph_diff(shipped, em)
        if d.node_delta or d.init_exact_mismatch:
            fails.append(f"graph_diff: node_delta {d.node_delta[:3]} init {d.init_exact_mismatch[:3]}")
        if variant == "w8check":
            if d.weight_lsb:
                fails.append(f"w8check differs from shipped in {len(d.weight_lsb)} weights")
        elif set(d.weight_lsb) != changed:
            fails.append(f"weight_lsb keys {len(d.weight_lsb)} != re-quantized convs {len(changed)}")
        print(f"[check] graph_diff vs shipped: node_delta 0, init mismatches 0, weights changed "
              f"{len(d.weight_lsb)}/{d.compared_int8_initializers} int8 initializers")
    d = graph_diff(expected_graph(shipped, em, rows), em)
    if d.node_delta or d.init_exact_mismatch or d.weight_lsb:
        fails.append(f"graph_diff vs shipped + intended rewrite: node_delta {len(d.node_delta)} "
                     f"init {d.init_exact_mismatch[:3]} weights {list(d.weight_lsb)[:3]}")
    print(f"[check] graph_diff vs shipped + intended rewrite only: "
          f"{'empty' if not (d.node_delta or d.init_exact_mismatch or d.weight_lsb) else 'NOT EMPTY'}")
    grid_bad = 0
    for r in rows:
        if r["form"] is None:
            continue
        stored = em.initializer(r["qname"]).astype(np.int16)
        dq = next(n for n in em.nodes() if n.output[0] == r["cw"]["dq"].output[0])
        scale = em.initializer(dq.input[1])
        for gr in r["q"]["groups"]:
            s = stored[gr["slice"]]
            q4 = s >> gr["k"] if r["form"] == "b" else s
            ok = (r["form"] == "a" or not np.any(s % (1 << gr["k"]))) and q4.min() >= lo and q4.max() <= hi \
                and np.unique(q4).size <= hi - lo + 1 and (r["form"] == "a" or 0 <= gr["k"] <= K_MAX)
            grid_bad += not ok
        eff = stored.astype(np.float32) * (scale.reshape(-1, 1, 1, 1) if scale.ndim else scale)
        if not np.array_equal(eff, r["q"]["w_hat"]):
            fails.append(f"{r['cw']['conv'].name}: stored * scale != q * 2**-pos")
    print(f"[check] grid [{lo}, {hi}], multiple of 2**k with k in [0, {K_MAX}] (form b), <= {hi - lo + 1} "
          f"levels per group: {'all groups' if not grid_bad else f'{grid_bad} groups FAIL'}; float identity "
          f"stored * scale == q * 2**-pos on {len(changed)} convs")
    if grid_bad:
        fails.append(f"grid check failed in {grid_bad} groups")
    return fails


def out_name(xint8: str, variant: str, form: str, any_a: bool) -> str:
    """The auto name when --out is a directory: <model>_w8check.onnx or
    <model>_w4a8_<variant>[_forma][_ortonly].onnx, <model> being the XINT8 stem before _cut."""
    stem = Path(xint8).stem.split("_cut")[0]
    if variant == "w8check":
        return f"{stem}_w8check.onnx"
    return f"{stem}_w4a8_{variant}{'_forma' if form == 'a' else ''}{'_ortonly' if any_a else ''}.onnx"


def cmd_emit(args) -> int:
    auto = args.out.endswith(("/", "\\")) or Path(args.out).is_dir()
    if auto and not args.sidecar:
        raise SystemExit("--out is a directory: pass --sidecar")
    side = Path(args.sidecar or Path(args.out).with_suffix(".json"))
    for p in ([] if auto else [Path(args.out)]) + [side]:
        if p.exists():
            raise SystemExit(f"refusing to overwrite {shown(p)}")
    lo, hi = GRIDS[args.grid] if args.variant != "w8check" else W8
    window = tuple(int(v) for v in args.window.split(":")) if args.variant != "w8check" else W8_WINDOW
    if len(window) != 2 or window[0] > 0 or window[1] < 0:
        raise SystemExit("--window is lo:hi around the min/max position, lo <= 0 <= hi")
    form = "b" if args.variant == "w8check" else args.form

    shipped = Graph.load(Path(args.xint8), strict=True)
    g = Graph(copy.deepcopy(shipped.model))
    gf = Graph.load(Path(args.float), strict=False)
    print(f"[emit] variant {args.variant}  grid [{lo}, {hi}]  window {window}  form {args.form}")
    print(f"[emit] float  {shown(args.float)}  sha256 {sha256(args.float)}")
    print(f"[emit] xint8  {shown(args.xint8)}  sha256 {sha256(args.xint8)}")
    cws = conv_weights(shipped, gf)
    print(f"[check] 63 Convs: stored int8 == clip(rint(w / s8), -127, 127) of the float weight, and "
          f"choose_pos(-127..127, window -1..3) == choose_pow2_minmse == shipped pos8, on all 63")
    keep = set(stem_and_head(shipped)) if args.variant == "e1h" else set()
    if keep:
        kept = sum(c["w"].size for c in cws if c["conv"].name in keep)
        total = sum(c["w"].size for c in cws)
        print(f"[emit] e1h keeps 7 convs at W8 ({kept} of {total} weights, {100 * kept / total:.2f}%): "
              f"{', '.join(sorted(keep))}")

    rows = []
    for cw in cws:
        if cw["conv"].name in keep:
            rows.append({"cw": cw, "qname": cw["qname"], "form": None, "q": None})
            continue
        q = quantize_conv(cw, args.variant if args.variant != "e1h" else "e1", lo, hi, window)
        f = form if form != "auto" else ("b" if q["form_b_ok"] else "a")
        if f == "b" and not q["form_b_ok"]:
            ks = sorted({gr["k"] for gr in q["groups"]})
            raise SystemExit(f"{cw['conv'].name}: k {ks} outside [0, {K_MAX}], form b impossible; use --form auto or a")
        rows.append({"cw": cw, "qname": cw["qname"], "form": f, "q": q})
    any_a = any(r["form"] == "a" for r in rows)
    for r in rows:
        if r["form"] == "a":
            ks = [gr["k"] for gr in r["q"]["groups"]]
            print(f"[emit] form a: {r['cw']['conv'].name}: k {min(ks)}..{max(ks)}, "
                  f"{sum(not 0 <= k <= K_MAX for k in ks)} of {len(ks)} groups outside [0, {K_MAX}]")
    out = Path(args.out) / out_name(args.xint8, args.variant, form, any_a) if auto else Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {shown(out)}")
    if any_a != out.name.endswith("_ortonly.onnx"):
        raise SystemExit(f"{sum(r['form'] == 'a' for r in rows)} convs are form a; a file holds a form-a conv "
                         f"exactly when its name ends in _ortonly.onnx (got {out.name})")
    for r in rows:
        if r["form"] == "b":
            emit_b(g, r["cw"], r["q"])
        elif r["form"] == "a":
            emit_a(g, r["cw"], r["q"])
    out.parent.mkdir(parents=True, exist_ok=True)
    g.save(out)
    digest = sha256(out)

    fails = check_emitted(out, shipped, rows, args.variant, lo, hi)
    convs = []
    print(f"{'conv':44s} {'shape':>16s} {'form':>4s} {'grp':>4s} {'pos8':>4s} {'pos4':>7s} {'k':>5s} "
          f"{'edge':>4s} {'SQNR8':>6s} {'SQNR4':>6s} {'clip%':>6s} {'lvl':>3s} {'sh-k':>4s} {'e-k':>4s}")
    for r in rows:
        cw, q = r["cw"], r["q"]
        w = cw["w"]
        w8 = cw["stored"].astype(np.float32) * pos2scale(cw["pos8"])
        rec = {"conv": cw["conv"].name, "weight": cw["qname"], "shape": list(w.shape), "form": r["form"],
               "pos8": cw["pos8"], "shift_out": cw["shift_out"], "e_bias": cw["e_bias"],
               "sqnr_w8_db": round(sqnr_db(w, w8), 3)}
        if q is None:
            rec["kept_w8"] = True
            convs.append(rec)
            print(f"{cw['conv'].name:44s} {str(tuple(w.shape)):>16s} {'W8':>4s}")
            continue
        real = [gr for gr in q["groups"] if not gr["zero"]]
        ks = [gr["k"] for gr in q["groups"]]
        p4 = [gr["pos4"] for gr in q["groups"]]
        clipped = sum(gr["clipped"] for gr in real)
        kmax = max(ks)
        rec.update({
            "groups": len(q["groups"]), "zero_groups": len(q["groups"]) - len(real),
            "pos4_min": min(p4), "pos4_max": max(p4), "k_min": min(ks), "k_max": kmax,
            "k_hist": {str(k): ks.count(k) for k in sorted(set(ks))},
            "edge_low": sum(gr["edge"] == "low" for gr in real),
            "edge_high": sum(gr["edge"] == "high" for gr in real),
            "sqnr_w4_db": round(sqnr_db(w, q["w_hat"]), 3),
            "clipped_frac": clipped / w.size,
            "levels_max": max((gr["levels"] for gr in real), default=0),
            "shift_out_minus_kmax": None if cw["shift_out"] is None else cw["shift_out"] - kmax,
            "e_bias_minus_kmax": None if cw["e_bias"] is None else cw["e_bias"] - kmax,
        })
        convs.append(rec)
        edge = ("L" if rec["edge_low"] else "") + ("H" if rec["edge_high"] else "")
        pos4s = f"{min(p4)}" if min(p4) == max(p4) else f"{min(p4)}..{max(p4)}"
        kr = f"{min(ks)}" if min(ks) == kmax else f"{min(ks)}..{kmax}"
        print(f"{cw['conv'].name:44s} {str(tuple(w.shape)):>16s} {r['form']:>4s} {len(q['groups']):>4d} "
              f"{cw['pos8']:>4d} {pos4s:>7s} {kr:>5s} {edge:>4s} {rec['sqnr_w8_db']:>6.2f} "
              f"{rec['sqnr_w4_db']:>6.2f} {100 * rec['clipped_frac']:>6.3f} {rec['levels_max']:>3d} "
              f"{str(rec['shift_out_minus_kmax']):>4s} {str(rec['e_bias_minus_kmax']):>4s}")

    q_rows = [c for c in convs if not c.get("kept_w8")]
    all_k = [int(k) for c in q_rows for k, n in c.get("k_hist", {}).items() for _ in range(n)]
    summary = {
        "variant": args.variant, "grid": [lo, hi], "window": list(window), "form_requested": args.form,
        "convs_form_b": sum(r["form"] == "b" for r in rows), "convs_form_a": sum(r["form"] == "a" for r in rows),
        "convs_kept_w8": sum(r["form"] is None for r in rows),
        "k_hist": {str(k): all_k.count(k) for k in sorted(set(all_k))},
        "groups_at_window_edge": {"low": sum(c.get("edge_low", 0) for c in q_rows),
                                  "high": sum(c.get("edge_high", 0) for c in q_rows)},
        "convs_shift_out_minus_kmax_negative": sum(1 for c in q_rows if (c["shift_out_minus_kmax"] or 0) < 0),
        "convs_e_bias_minus_kmax_negative": sum(1 for c in q_rows if (c["e_bias_minus_kmax"] or 0) < 0),
        "float": {"path": shown(args.float), "sha256": sha256(args.float)},
        "xint8": {"path": shown(args.xint8), "sha256": sha256(args.xint8)},
        "out": {"path": shown(out), "sha256": digest},
        "controls_pass": not fails, "control_failures": fails,
    }
    side.write_text(json.dumps({"summary": summary, "convs": convs}, indent=1) + "\n", encoding="utf-8")
    print(f"[emit] forms: b {summary['convs_form_b']}, a {summary['convs_form_a']}, kept W8 "
          f"{summary['convs_kept_w8']}; k over all groups {summary['k_hist']}; groups at the window edge "
          f"{summary['groups_at_window_edge']}")
    print(f"[emit] headroom if lowered as true int4 (acc << k): convs with shift_out - k_max < 0: "
          f"{summary['convs_shift_out_minus_kmax_negative']}, with e_bias - k_max < 0: "
          f"{summary['convs_e_bias_minus_kmax_negative']}")
    print(f"[emit] wrote {shown(out)}  sha256 {digest}")
    print(f"[emit] sidecar {shown(side)}")
    print(f"[emit] {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return 0 if not fails else 1


def cmd_emit_float(args) -> int:
    """Weight-only diagnostic: the FLOAT model (float activations, Sigmoid SiLU) with every Conv weight
    replaced by the variant's dequantized weight, q * 2**-pos, exactly as emit computes it. Not part of
    the pre-registered gate; it separates the 4-bit weights from the frozen W8A8 activation pipeline."""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {shown(out)}")
    lo, hi = GRIDS[args.grid] if args.variant != "w8check" else W8
    window = tuple(int(v) for v in args.window.split(":")) if args.variant != "w8check" else W8_WINDOW
    shipped = Graph.load(Path(args.xint8), strict=True)
    gf = Graph.load(Path(args.float), strict=False)
    print(f"[emit-float] variant {args.variant}  grid [{lo}, {hi}]  window {window}  (weight-only, float activations)")
    print(f"[emit-float] float  {shown(args.float)}  sha256 {sha256(args.float)}")
    print(f"[emit-float] xint8  {shown(args.xint8)}  sha256 {sha256(args.xint8)}")
    cws = conv_weights(shipped, gf)
    keep = set(stem_and_head(shipped)) if args.variant == "e1h" else set()
    model = copy.deepcopy(gf.model)
    inits = {t.name: t for t in model.graph.initializer}
    replaced = {}
    for cw in cws:
        if args.variant == "w8check" or cw["conv"].name in keep:
            w_hat = cw["stored"].astype(np.float32) * pos2scale(cw["pos8"])
        else:
            w_hat = quantize_conv(cw, args.variant if args.variant != "e1h" else "e1", lo, hi, window)["w_hat"]
        fname = cw["qname"][: -len("_quantized")]
        inits[fname].CopyFrom(numpy_helper.from_array(np.asarray(w_hat, dtype=np.float32), fname))
        replaced[fname] = w_hat
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out))
    back = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(out)).graph.initializer}
    orig = {t.name: numpy_helper.to_array(t) for t in gf.model.graph.initializer}
    bad = [n for n, w in replaced.items() if not np.array_equal(back[n], w)]
    other = [n for n in orig if n not in replaced and (n not in back or back[n].tobytes() != orig[n].tobytes())]
    extra = [n for n in back if n not in orig]
    print(f"[check] {len(replaced) - len(bad)}/{len(replaced)} Conv weights equal the variant's q * 2**-pos; "
          f"{len(orig) - len(replaced) - len(other)}/{len(orig) - len(replaced)} other initializers byte-identical "
          f"to the float model; {len(extra)} added")
    digest = sha256(out)
    print(f"[emit-float] wrote {shown(out)}  sha256 {digest}")
    ok = not (bad or other or extra)
    print(f"[emit-float] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_compare(args) -> int:
    import cv2
    import onnxruntime as ort
    images = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    if len(images) != args.n:
        raise SystemExit(f"found {len(images)} images in {shown(args.images)}, wanted {args.n}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = args.threads
    sessions = [ort.InferenceSession(str(m), so, providers=["CPUExecutionProvider"]) for m in (args.a, args.b)]
    names = [o.name for o in sessions[0].get_outputs()]
    if names != [o.name for o in sessions[1].get_outputs()]:
        raise SystemExit("the two models have different outputs")
    inp = sessions[0].get_inputs()[0]
    h, w = inp.shape[2], inp.shape[3]
    print(f"[compare] onnxruntime {ort.__version__}  ORT_DISABLE_ALL  threads {args.threads}")
    for m in (args.a, args.b):
        print(f"[compare] {shown(m)}  sha256 {sha256(m)}")
    bad = 0
    for p in images:
        img = cv2.resize(cv2.imread(str(p)), (w, h), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1))[None]
        ya, yb = (s.run(names, {inp.name: x}) for s in sessions)
        same = all(np.array_equal(u, v) for u, v in zip(ya, yb))
        bad += not same
        if not same:
            print(f"[compare] {p.name}: outputs DIFFER")
    print(f"[compare] {len(images) - bad}/{len(images)} images bit-identical on all {len(names)} outputs")
    print(f"[compare] {'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="write one W4A8 variant and check it")
    e.add_argument("--float", required=True, help="the float model the XINT8 file was quantized from")
    e.add_argument("--xint8", required=True, help="the shipped XINT8 QDQ model")
    e.add_argument("--variant", required=True, choices=VARIANTS)
    e.add_argument("--grid", default="sym7", choices=sorted(GRIDS))
    e.add_argument("--window", default="-1:5", help="MinMSE candidates around the min/max position (default -1:5)")
    e.add_argument("--form", default="auto", choices=("auto", "a", "b"))
    e.add_argument("--out", required=True, help="the .onnx to write, or a directory (trailing /) to name it "
                   "<model>_w4a8_<variant>[_forma][_ortonly].onnx there")
    e.add_argument("--sidecar", default=None, help="per-conv JSON (default: --out with .json; required with a "
                   "directory --out)")
    f = sub.add_parser("emit-float", help="diagnostic: the float model with the variant's dequantized weights")
    f.add_argument("--float", required=True)
    f.add_argument("--xint8", required=True)
    f.add_argument("--variant", required=True, choices=VARIANTS)
    f.add_argument("--grid", default="sym7", choices=sorted(GRIDS))
    f.add_argument("--window", default="-1:5")
    f.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="bit-compare two models' outputs under ORT_DISABLE_ALL")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--images", default=str(ROOT / "data" / "coco" / "val2017"))
    c.add_argument("--n", type=int, default=16)
    c.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    return {"emit": cmd_emit, "emit-float": cmd_emit_float, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
