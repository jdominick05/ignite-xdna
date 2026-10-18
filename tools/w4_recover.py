#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""W4V: post-training W4 recovery of the shipped XINT8 YOLOv8n/s, written through gate D's emitter.

Every arm file is the shipped XINT8 file with exactly one rewrite: the Conv weights on the grid [-7, 7] (form b,
q4 << k under the shipped per-tensor scale, where every group allows it, else form a, a per-channel scale vector,
*_ortonly.onnx), plus an int32 Conv bias at s_x * s_w for BC and VC. The files are built by tools/w4a8_emulate.py's
own emit_b / emit_a / rewire_a (read only), so an RTN arm is gate D's file byte for byte (C2). The pre-registered
constants live in tools/w4v_verdict.py; every mode prints its frozen triple first.

    python tools/w4_recover.py selftest
    python tools/w4_recover.py controls --model yolov8n --sidecar-dir DIR --tag desktop2_YYYYMMDD
    python tools/w4_recover.py sensitivity --model yolov8n
    python tools/w4_recover.py emit --model yolov8n --method gptq --form e1 --tier 1 --sidecar-dir DIR \\
        --tag desktop2_YYYYMMDD [--twin] [--repeat]

Arm files go to scratch/w4v/arms/<model>_w4v_<method>_<form>[_ortonly].onnx, the y_f statistics and the MX sets
to scratch/w4v/captures/ (--out-dir, --captures); every mode but selftest runs from the repo root. Markers
"W4V_CONTROL <C#> <scope> PASS|FAIL" are what tools/w4v_verdict.py --verdict reads.

Methods (section 3 of the accepted text): rtn, bc and vc (on RTN), ar (quant/adaround.finetune with a WeightGrid
per layer), gptq, the tier-2 stacks arbc and gptqbc (BC over the tier-1 file), and mxar / mxgptq (the W8 set from
sensitivity kept as shipped). Forms: e1, e2, u, and f (float per-channel scale, RTN and GPTQ only).

Calibration: the first 64 images of data/coco_calib by name, letterboxed as 5_eval_map.py does
(quant.sources.CocoSource). GPTQ, BC, VC and the MX sensitivity capture the current QDQ graph up to each Conv's
input under ORT_DISABLE_ALL at 8 intra-op threads; AdaRound runs quant/adaround's own sessions (ORT's default
threads) with torch at 8. y_f, for BC and VC, is the float head-cut graph's Conv output; the MX sensitivity compares
the float and the RTN-W4 weights on B8's captured input. Nothing here reads a mAP.
"""
import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MKL_CBWR", "COMPATIBLE")   # before numpy/torch: reproducible BLAS reductions (C7)

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnx.utils  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

import w4a8_emulate as g4  # noqa: E402  (gate D's emitter, read only)
import w4v_verdict as vv  # noqa: E402
from quant.calib import choose_pow2_minmse  # noqa: E402
from quant.graph import Graph  # noqa: E402
from quant.pow2 import pos2scale  # noqa: E402
from quant.verify import graph_diff  # noqa: E402

GRID, WINDOW = vv.GRID, vv.WINDOW
ORT_THREADS = int(vv.EVAL["threads"])
TORCH_THREADS = vv.HYPER["torch_threads"]
AR = vv.HYPER["adaround"]
GP = vv.HYPER["gptq"]
MX_BUDGET = vv.HYPER["mx"]["budget"]
F_ALPHAS = [np.float32(a / 100) for a in range(50, 101)]
GPTQ_HITS_FLOOR = 30   # selftest: GPTQ at the brute-force optimum in at least 30 of its 40 seeded cases (36 when set)
VARIANT = {"e1": "e1", "e2": "e2", "u": "u"}


def shown(path) -> str:
    return vv.shown(path)


def marker(name: str, scope: str, ok: bool, detail: str = "") -> None:
    print(f"W4V_CONTROL {name} {scope} {'PASS' if ok else 'FAIL'}{' ' + detail if detail else ''}", flush=True)


def torch_mod():
    import torch
    torch.set_num_threads(TORCH_THREADS)
    return torch


def thread_facts() -> dict:
    """AdaRound's threads as run (section 3, order and inputs): quant/adaround's ORT sessions keep a fresh
    SessionOptions' intra-op setting, 0 = ORT's own default; torch as set by torch_mod()."""
    import onnxruntime as ort
    import torch
    return {"ort_intra_op_num_threads_setting": ort.SessionOptions().intra_op_num_threads,
            "ort_intra_op_note": "0 is ORT's default; quant/adaround sets no thread count",
            "torch_get_num_threads": torch.get_num_threads(), "os_cpu_count": os.cpu_count()}


# ------------------------------------------------------------------------------------------------- the model

class Model:
    """The shipped XINT8 file, its float head-cut model and gate D's per-Conv records (conv_weights)."""

    def __init__(self, name: str, shipped: Graph, gf: Graph, cws: list[dict], float_path=None, xint8_path=None):
        self.name, self.shipped, self.gf, self.cws = name, shipped, gf, cws
        self.by_name = {cw["conv"].name: cw for cw in cws}
        self.order = [cw["conv"].name for cw in cws]
        self.float_path, self.xint8_path = float_path, xint8_path   # repo-relative: printed and passed on as is
        self.input_name = shipped.model.graph.input[0].name
        by_output = {o: n for n in shipped.nodes() for o in n.output}
        inits = {t.name: numpy_helper.to_array(t) for t in shipped.model.graph.initializer}
        finits = {t.name: numpy_helper.to_array(t) for t in gf.model.graph.initializer}
        self.fconv = {n.name: n for n in gf.nodes() if n.op_type == "Conv"}
        for cw in cws:
            conv = cw["conv"]
            cw["s_x"] = np.float32(inits[by_output[conv.input[0]].input[1]])
            bq, bs, bz = by_output[conv.input[2]].input[:3]
            cw["bias_float"] = (inits[bq].astype(np.float64) - inits[bz].astype(np.float64)) * np.float64(inits[bs])
            fb = self.fconv[conv.name].input[2] if len(self.fconv[conv.name].input) > 2 else ""
            if fb and fb not in finits:
                raise SystemExit(f"{conv.name}: the float Conv's bias {fb} is not an initializer")
            # the float graph's own bias (MX sensitivity's float reference on B8's input)
            cw["bias_f"] = (np.asarray(finits[fb], dtype=np.float32) if fb
                            else np.zeros(cw["w"].shape[0], dtype=np.float32))
            cw["attrs"] = conv_attrs(conv, cw["w"].shape)

    @classmethod
    def load(cls, name: str) -> "Model":
        fpath, fsha = vv.MODEL_PIN[name]["float"]
        qpath, qsha = vv.MODEL_PIN[name]["xint8"]
        for p, want in ((fpath, fsha), (qpath, qsha)):
            got = vv.sha_file(ROOT / p)
            print(f"[model] {p}  sha256 {got}  {'== pinned' if got == want else 'PIN MISMATCH, STOP'}", flush=True)
            if got != want:
                raise SystemExit(f"{p}: sha256 {got} is not the pinned {want}")
        shipped = Graph.load(ROOT / qpath, strict=True)
        gf = Graph.load(ROOT / fpath, strict=False)
        return cls(name, shipped, gf, g4.conv_weights(shipped, gf), Path(fpath), Path(qpath))


def conv_attrs(node, wshape) -> dict:
    a = {x.name: helper.get_attribute_value(x) for x in node.attribute}
    pads = list(a.get("pads", [0, 0, 0, 0]))
    half = len(pads) // 2
    if pads[:half] != pads[half:] or int(a.get("group", 1)) != 1 or a.get("auto_pad", b"NOTSET") not in (b"NOTSET",
                                                                                                         "NOTSET"):
        raise ValueError(f"{node.name}: only group-1 Convs with symmetric explicit pads are supported")
    return {"stride": tuple(int(v) for v in a.get("strides", (1, 1))), "padding": tuple(int(v) for v in pads[:half]),
            "dilation": tuple(int(v) for v in a.get("dilations", (1, 1))), "kernel": tuple(int(v) for v in wshape[2:])}


# ------------------------------------------------------------------------------------------ rows and emission

def q_with(q4: np.ndarray, like: dict) -> dict:
    """A q-dict (gate D's shape) for new integers on an existing group layout (pos4, k, or F's float scales)."""
    w_hat = np.zeros(q4.shape, dtype=np.float32)
    if "scales" in like:
        w_hat[...] = q4.astype(np.float32) * like["scales"].reshape(-1, 1, 1, 1)
    else:
        for gr in like["groups"]:
            w_hat[gr["slice"]] = q4[gr["slice"]].astype(np.float32) * pos2scale(gr["pos4"])
    out = {**like, "q4": np.asarray(q4, dtype=np.int8), "w_hat": w_hat}
    return out


def f_quant(w: np.ndarray) -> dict:
    """F: s_c = alpha * max|w_c| / 7, alpha in 0.50..1.00, the first minimum of the float32 squared error."""
    cout = w.shape[0]
    q4 = np.zeros(w.shape, dtype=np.int8)
    scales = np.ones(cout, dtype=np.float32)
    groups = []
    for c in range(cout):
        x = np.asarray(w[c], dtype=np.float32)
        m = np.float32(np.abs(x).max())
        if m == 0:
            groups.append({"slice": slice(c, c + 1), "pos4": None, "k": 0, "zero": True, "alpha": None})
            continue
        best = None
        for a in F_ALPHAS:
            s = np.float32(a * m / np.float32(7))
            q = np.clip(np.rint(x / s), GRID[0], GRID[1])
            err = float(np.sum((x - s * q.astype(np.float32)) ** 2, dtype=np.float32))
            if best is None or err < best[0]:
                best = (err, s, q, a)
        scales[c], q4[c] = best[1], best[2].astype(np.int8)
        groups.append({"slice": slice(c, c + 1), "pos4": None, "k": 0, "zero": False, "alpha": float(best[3])})
    q = {"q4": q4, "groups": groups, "scales": scales, "form_b_ok": False}
    return q_with(q4, q)


def rtn_row(cw: dict, form: str) -> dict:
    if form == "f":
        return {"cw": cw, "qname": cw["qname"], "form": "a", "q": f_quant(cw["w"])}
    q = g4.quantize_conv(cw, VARIANT[form], GRID[0], GRID[1], WINDOW)
    return {"cw": cw, "qname": cw["qname"], "form": "b" if q["form_b_ok"] else "a", "q": q}


def rtn_rows(model: Model, form: str, keep=()) -> list[dict]:
    """Gate D's rows: RTN per Conv, form auto; a Conv named in keep stays W8 as shipped (form None)."""
    return [{"cw": cw, "qname": cw["qname"], "form": None, "q": None} if cw["conv"].name in keep else rtn_row(cw, form)
            for cw in model.cws]


def w_scale(row: dict):
    """The Conv weight's dequantization scale in the emitted file: scalar (form b / kept W8) or per channel."""
    cw = row["cw"]
    if row["form"] in (None, "b"):
        return pos2scale(cw["pos8"])
    if "scales" in row["q"]:
        return row["q"]["scales"]
    return g4.channel_scales(row["q"], cw["w"].shape[0])


def w_hat(row: dict) -> np.ndarray:
    if row["form"] is None:
        return row["cw"]["stored"].astype(np.float32) * pos2scale(row["cw"]["pos8"])
    return row["q"]["w_hat"]


def emit_row(g: Graph, r: dict) -> None:
    if r["form"] == "b":
        g4.emit_b(g, r["cw"], r["q"])
    elif r["form"] == "a":
        if "scales" in r["q"]:
            cout = r["cw"]["w"].shape[0]
            g.set_initializer(r["qname"], r["q"]["q4"])
            g4.rewire_a(g, r["cw"], r["q"]["scales"], np.zeros(cout, dtype=np.int8))
        else:
            g4.emit_a(g, r["cw"], r["q"])


def bias_names(g: Graph, conv_name: str) -> tuple:
    conv = next(n for n in g.model.graph.node if n.name == conv_name)
    dq = next(n for n in g.model.graph.node if n.output[0] == conv.input[2])
    return dq, list(dq.input[:3])


def rewire_bias(g: Graph, conv_name: str, i32: np.ndarray, scale) -> None:
    """The Conv bias as int32 at s_x * s_w (3c_gptq_cv2.py's form): new initializers <old>_w4v, the bias DQ
    pointed at them (axis 0 for a per-channel scale), the old ones dropped when nothing else reads them."""
    dq, old = bias_names(g, conv_name)
    names = [n + "_w4v" for n in old]
    scale = np.asarray(scale, dtype=np.float32)
    g.set_initializer(names[0], np.asarray(i32, dtype=np.int32))
    g.set_initializer(names[1], scale)
    g.set_initializer(names[2], np.zeros(scale.shape, dtype=np.int32))
    dq.input[0], dq.input[1], dq.input[2] = names
    if scale.ndim:
        dq.attribute.append(helper.make_attribute("axis", 0))
    for name in old:
        if not any(name in n.input for n in g.model.graph.node):
            g.remove_initializer(name)


def build(model: Model, rows: list[dict], biases: dict | None = None) -> Graph:
    """Gate D's cmd_emit sequence (deep copy of the shipped file, emit per Conv in order), then the biases."""
    g = Graph(copy.deepcopy(model.shipped.model))
    for r in rows:
        emit_row(g, r)
    for name in model.order:
        if biases and name in biases:
            rewire_bias(g, name, biases[name]["i32"], biases[name]["scale"])
    return g


def make_bias(b: np.ndarray, s_x, scale_w) -> dict:
    scale = (np.float32(s_x) * np.asarray(scale_w, dtype=np.float32)).astype(np.float32)
    q = np.rint(np.asarray(b, dtype=np.float64) / scale.astype(np.float64))
    if np.abs(q).max(initial=0) >= 2 ** 31:
        raise ValueError("an int32 bias overflows")
    return {"i32": q.astype(np.int32), "scale": scale, "float": np.asarray(b, dtype=np.float64)}


def bias_now(row: dict, biases: dict) -> np.ndarray:
    name = row["cw"]["conv"].name
    if name in biases:
        b = biases[name]
        return b["i32"].astype(np.float64) * np.asarray(b["scale"], dtype=np.float64)
    return row["cw"]["bias_float"]


def out_name(model: str, arm: str, rows: list[dict]) -> str:
    return f"{model}_w4v_{arm}{'_ortonly' if any(r['form'] == 'a' for r in rows) else ''}.onnx"


# ------------------------------------------------------------------------------------------------- the checks

def check_arm(path: Path, model: Model, rows: list[dict], biases: dict, arm: str, rescaled: dict) -> list[str]:
    """C5: graph_diff against the shipped file plus exactly the intended rewrite; the grid; float identity; pos4
    == MinMSE (on w' for VC; not for F); the int32 biases. Gate D's check_emitted runs too wherever it applies."""
    fails = []
    em = Graph.load(path, strict=True)
    exp = g4.expected_graph(model.shipped, em, rows)
    for name in model.order:
        if name in biases:
            _, old = bias_names(model.shipped, name)
            rewire_bias(exp, name, em.initializer(old[0] + "_w4v"), em.initializer(old[1] + "_w4v"))
    d = graph_diff(exp, em)
    empty = not (d.node_delta or d.init_exact_mismatch or d.weight_lsb)
    if not empty:
        fails.append(f"graph_diff vs shipped + intended rewrite: node_delta {d.node_delta[:2]} "
                     f"init {d.init_exact_mismatch[:3]} weights {list(d.weight_lsb)[:3]}")
    print(f"[check] graph_diff vs shipped + intended rewrite (weights{', int32 biases' if biases else ''}): "
          f"{'empty' if empty else 'NOT EMPTY'}", flush=True)
    if not biases:
        fails += g4.check_emitted(path, model.shipped, rows, arm, GRID[0], GRID[1])
    else:
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
                ok = (r["form"] == "a" or not np.any(s % (1 << gr["k"]))) and q4.min() >= GRID[0] \
                    and q4.max() <= GRID[1] and (r["form"] == "a" or 0 <= gr["k"] <= g4.K_MAX)
                grid_bad += not ok
            eff = stored.astype(np.float32) * (scale.reshape(-1, 1, 1, 1) if scale.ndim else scale)
            if not np.array_equal(eff, r["q"]["w_hat"]):
                fails.append(f"{r['cw']['conv'].name}: stored * scale != the method's dequantized weights")
        print(f"[check] grid [{GRID[0]}, {GRID[1]}] (form b: multiple of 2**k, k in [0, {g4.K_MAX}]): "
              f"{'all groups' if not grid_bad else f'{grid_bad} groups FAIL'}; float identity on "
              f"{sum(r['form'] is not None for r in rows)} convs", flush=True)
        if grid_bad:
            fails.append(f"grid check failed in {grid_bad} groups")
    pos_bad, pos_n = 0, 0
    for r in rows:
        if r["form"] is None or "scales" in r["q"]:
            continue
        name = r["cw"]["conv"].name
        w = rescaled.get(name, r["cw"]["w"])
        for gr in r["q"]["groups"]:
            if gr["zero"]:
                pos_bad += gr["pos4"] != r["cw"]["pos8"]
                continue
            pos_n += 1
            mine = g4.choose_pos(w[gr["slice"]], GRID[0], GRID[1], WINDOW)[0]
            quant = choose_pow2_minmse(w[gr["slice"]], "int8", grid=GRID, window=WINDOW)[0]
            pos_bad += not (mine == quant == gr["pos4"])
    print(f"[check] pos4 == choose_pos == choose_pow2_minmse(grid, window) on {pos_n} groups"
          f"{' (on w for VC)' if rescaled else ''}: {'all' if not pos_bad else f'{pos_bad} FAIL'}", flush=True)
    if pos_bad:
        fails.append(f"pos4 != MinMSE in {pos_bad} groups")
    bias_bad = 0
    for name, b in biases.items():
        row = next(r for r in rows if r["cw"]["conv"].name == name)
        _, old = bias_names(model.shipped, name)
        i32, scale = em.initializer(old[0] + "_w4v"), em.initializer(old[1] + "_w4v")
        want = (np.float32(row["cw"]["s_x"]) * np.asarray(w_scale(row), dtype=np.float32)).astype(np.float32)
        err = np.abs(i32.astype(np.float64) * scale.astype(np.float64) - b["float"])
        ok = i32.dtype == np.int32 and np.array_equal(scale, want) and bool(np.all(err <= scale.astype(np.float64)
                                                                                     / 2 * (1 + 1e-9)))
        bias_bad += not ok
    if biases:
        print(f"[check] int32 biases at s_x * s_w, within half a step of the method's float bias: "
              f"{len(biases) - bias_bad}/{len(biases)}", flush=True)
    if bias_bad:
        fails.append(f"{bias_bad} int32 biases wrong")
    return fails


# --------------------------------------------------------------------------------------------------- captures

def extract(model_proto: onnx.ModelProto, starts: list[str], ends: list[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model_proto)
    return onnx.utils.Extractor(inferred).extract_model(starts, ends)


def ort_session(model_proto: onnx.ModelProto):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = ORT_THREADS
    return ort.InferenceSession(model_proto.SerializeToString(), so, providers=["CPUExecutionProvider"])


def capture(g: Graph, input_name: str, tensor: str, images: list[np.ndarray]) -> np.ndarray:
    """A tensor of the current QDQ graph for every calibration image, stacked (N, ...)."""
    sess = ort_session(extract(g.model, [input_name], [tensor]))
    out = None
    for i, image in enumerate(images):
        y = sess.run([tensor], {input_name: image})[0]
        if out is None:
            out = np.empty((len(images), *y.shape[1:]), dtype=y.dtype)
        out[i] = y[0]
    return out


def conv_sums(x: np.ndarray, w: np.ndarray, b: np.ndarray, attrs: dict, chunk: int = 4) -> tuple:
    """Per output channel: the sum and the sum of squares (float64) of conv(x, w) + b, and the element count."""
    torch = torch_mod()
    import torch.nn.functional as F
    wt = torch.from_numpy(np.array(w, dtype=np.float32, order="C"))   # copies: an ONNX initializer is read-only
    bt = torch.from_numpy(np.array(b, dtype=np.float32))
    s = torch.zeros(w.shape[0], dtype=torch.float64)
    ss = torch.zeros(w.shape[0], dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            y = F.conv2d(torch.from_numpy(np.ascontiguousarray(x[i:i + chunk])), wt, bt, stride=attrs["stride"],
                         padding=attrs["padding"], dilation=attrs["dilation"])
            yd = y.double()
            s += yd.sum(dim=(0, 2, 3))
            ss += (yd * yd).sum(dim=(0, 2, 3))
            n += y.shape[0] * y.shape[2] * y.shape[3]
    return s.numpy(), ss.numpy(), n


def conv_out(x: np.ndarray, w: np.ndarray, b: np.ndarray, attrs: dict) -> np.ndarray:
    torch = torch_mod()
    import torch.nn.functional as F
    with torch.no_grad():   # w and b are copied: an ONNX initializer (the float weights, bias_f) is read-only
        return F.conv2d(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                        torch.from_numpy(np.array(w, dtype=np.float32, order="C")),
                        torch.from_numpy(np.array(b, dtype=np.float32)), stride=attrs["stride"],
                        padding=attrs["padding"], dilation=attrs["dilation"]).numpy()


def hessian(x: np.ndarray, attrs: dict, chunk: int = 1) -> tuple:
    """GPTQ's H = 2 X X^T / N (float64) over every output position; X is im2col in ONNX (cin, kh, kw) order."""
    torch = torch_mod()
    import torch.nn.functional as F
    k = x.shape[1] * attrs["kernel"][0] * attrs["kernel"][1]
    h = torch.zeros((k, k), dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            cols = F.unfold(torch.from_numpy(np.ascontiguousarray(x[i:i + chunk])), kernel_size=attrs["kernel"],
                            dilation=attrs["dilation"], padding=attrs["padding"], stride=attrs["stride"])
            xm = cols.permute(1, 0, 2).reshape(k, -1).double()
            h += xm @ xm.T
            n += xm.shape[1]
    return (2 * h / n).numpy(), n


def float_reference(model: Model, images: list[np.ndarray]) -> dict:
    """y_f per Conv (the float graph's Conv output on the float network's own input): per-channel sum, sum of
    squares and count, float64."""
    outs = [model.fconv[n].output[0] for n in model.order]
    sess = ort_session(extract(model.gf.model, [model.input_name], outs))
    acc = {n: [0.0, 0.0, 0] for n in model.order}
    for image in images:
        ys = sess.run(outs, {model.input_name: image})
        for n, y in zip(model.order, ys):
            yd = y.astype(np.float64)
            a = acc[n]
            a[0] = a[0] + yd.sum(axis=(0, 2, 3))
            a[1] = a[1] + (yd * yd).sum(axis=(0, 2, 3))
            a[2] += yd.shape[0] * yd.shape[2] * yd.shape[3]
    return {n: {"sum": a[0], "sumsq": a[1], "count": a[2]} for n, a in acc.items()}


def mean_std(s, ss, n) -> tuple:
    mean = np.asarray(s, dtype=np.float64) / n
    var = np.maximum(np.asarray(ss, dtype=np.float64) / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


# ------------------------------------------------------------------------------------------------------- GPTQ

def gptq_quantize(w: np.ndarray, h: np.ndarray, s: np.ndarray, lo: int, hi: int, damp: float = GP["damp"],
                  block: int = GP["block"], act_order: bool = GP["act_order"]) -> tuple:
    """GPTQ (Frantar et al. 2022) in float64: dead columns, act-order, damping, the Cholesky-inverse update with
    lazy blocks. w [rows, cols]; s the fixed per-row scale; returns the integer Q and the column permutation."""
    w = np.array(w, dtype=np.float64)
    h = np.array(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    cols = w.shape[1]
    dead = np.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    perm = np.argsort(-np.diag(h), kind="stable") if act_order else np.arange(cols)
    w = w[:, perm]
    h = h[np.ix_(perm, perm)]
    h[np.diag_indices(cols)] += damp * np.mean(np.diag(h))
    lower = np.linalg.cholesky(h)
    linv = np.linalg.inv(lower)
    hinv = linv.T @ linv
    u = np.linalg.cholesky(hinv).T
    q = np.zeros_like(w)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        w1 = w[:, i1:i2].copy()
        q1 = np.zeros_like(w1)
        e1 = np.zeros_like(w1)
        u1 = u[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            qi = np.clip(np.rint(col / s), lo, hi)
            q1[:, i] = qi
            err = (col - qi * s) / u1[i, i]
            w1[:, i:] -= np.outer(err, u1[i, i:])
            e1[:, i] = err
        q[:, i1:i2] = q1
        w[:, i2:] -= e1 @ u[i1:i2, i2:]
    return q[:, np.argsort(perm)], perm


def gptq_reference(w: np.ndarray, h: np.ndarray, s: np.ndarray, lo: int, hi: int, damp: float, act_order: bool):
    """The unblocked OBQ reference for the selftest, written apart from gptq_quantize (its own dead-column,
    ordering and damping code, and no Cholesky factor or inverse). Step t re-solves the free columns R given the
    columns F already fixed: v_R = w_R + H_RR^-1 H_RF (w_F - q_F s), the minimiser of (w - v)^T H (w - v) with
    v_F = q_F s; the next column in order takes round-to-nearest of its v."""
    rows, cols = len(w), len(w[0])
    wr = [[float(w[r][j]) for j in range(cols)] for r in range(rows)]
    hr = [[float(h[i][j]) for j in range(cols)] for i in range(cols)]
    for j in range(cols):                     # a dead input: weight 0, unit curvature
        if hr[j][j] == 0.0:
            hr[j][j] = 1.0
            for r in range(rows):
                wr[r][j] = 0.0
    order = sorted(range(cols), key=lambda j: -hr[j][j]) if act_order else list(range(cols))   # stable
    lam = damp * sum(hr[j][j] for j in range(cols)) / cols
    for j in range(cols):
        hr[j][j] += lam
    hm, w0 = np.array(hr), np.array(wr)
    sr = [float(v) for v in np.asarray(s).reshape(-1)]
    q = np.zeros((rows, cols))
    fixed = []
    for t, j in enumerate(order):
        free = order[t:]
        if fixed:
            d = w0[:, fixed] - q[:, fixed] * np.array(sr).reshape(-1, 1)          # [rows, |F|]
            v = w0[:, free] + np.linalg.solve(hm[np.ix_(free, free)], hm[np.ix_(free, fixed)] @ d.T).T
        else:
            v = w0[:, free]
        for r in range(rows):
            q[r, j] = min(max(float(np.rint(v[r, 0] / sr[r])), lo), hi)
        fixed.append(j)
    return q


def recon_loss(w: np.ndarray, q: np.ndarray, s: np.ndarray, h: np.ndarray) -> float:
    e = np.asarray(w, dtype=np.float64) - np.asarray(q, dtype=np.float64) * np.asarray(s, dtype=np.float64).reshape(-1, 1)
    return float(np.einsum("ij,jk,ik->", e, h, e))


def row_scales(row: dict) -> np.ndarray:
    if "scales" in row["q"]:
        return row["q"]["scales"].astype(np.float64)
    return g4.channel_scales(row["q"], row["cw"]["w"].shape[0]).astype(np.float64)


# --------------------------------------------------------------------------------------------- the sequential loop

def sequential(model: Model, rows: list[dict], biases: dict, images: list[np.ndarray], step, log) -> None:
    """Visit every Conv in the XINT8 graph's Conv order; each sees the rewrites of every Conv before it (its input
    is captured from the current QDQ graph, rebuilt from the shipped file)."""
    by_name = {r["cw"]["conv"].name: r for r in rows}
    for i, name in enumerate(model.order):
        row = by_name[name]
        if row["form"] is None:
            log(f"[seq] {i:2d} {name}: W8 kept as shipped")
            continue
        t0 = time.perf_counter()
        g = build(model, rows, biases)
        x = capture(g, model.input_name, row["cw"]["conv"].input[0], images)
        info = step(row, x)
        log(f"[seq] {i:2d} {name} {json.dumps(info, sort_keys=True)} ({time.perf_counter() - t0:.1f} s)")
        del x


def step_bc(fref: dict, biases: dict, record: dict):
    def step(row, x):
        name, cw = row["cw"]["conv"].name, row["cw"]
        b = bias_now(row, biases)
        s, ss, n = conv_sums(x, w_hat(row), b, cw["attrs"])
        mean_q, _ = mean_std(s, ss, n)
        mean_f, _ = mean_std(fref[name]["sum"], fref[name]["sumsq"], fref[name]["count"])
        new = make_bias(b + mean_f - mean_q, cw["s_x"], w_scale(row))
        biases[name] = new
        after = mean_q + (new["i32"].astype(np.float64) * new["scale"].astype(np.float64) - b)
        info = {"mean_err_before": float(np.abs(mean_f - mean_q).max()),
                "mean_err_after": float(np.abs(mean_f - after).max())}
        record[name] = {"bc": info}
        return info
    return step


def step_vc(fref: dict, biases: dict, record: dict, rescaled: dict, form: str):
    def step(row, x):
        name, cw = row["cw"]["conv"].name, row["cw"]
        b = bias_now(row, biases)
        s, ss, n = conv_sums(x, w_hat(row), b, cw["attrs"])
        _, std_q = mean_std(s, ss, n)
        mean_f, std_f = mean_std(fref[name]["sum"], fref[name]["sumsq"], fref[name]["count"])
        a = np.where(std_q > 0, std_f / np.where(std_q > 0, std_q, 1.0), 1.0)
        w2 = (a.reshape(-1, 1, 1, 1) * cw["w"].astype(np.float64)).astype(np.float32)
        q2 = g4.quantize_conv({**cw, "w": w2}, VARIANT[form], GRID[0], GRID[1], WINDOW)
        row["q"], row["form"] = q2, "b" if q2["form_b_ok"] else "a"
        rescaled[name] = w2
        s2, ss2, n2 = conv_sums(x, q2["w_hat"], b, cw["attrs"])
        mean_q2, std_q2 = mean_std(s2, ss2, n2)
        new = make_bias(b + mean_f - mean_q2, cw["s_x"], w_scale(row))
        biases[name] = new
        info = {"a_min": float(a.min()), "a_max": float(a.max()), "form": row["form"],
                "std_ratio_after_max_dev": float(np.abs(std_q2 / np.where(std_f > 0, std_f, 1.0) - 1)[std_f > 0].max(
                    initial=0))}
        record[name] = {"vc": info}
        return info
    return step


def step_gptq(record: dict):
    def step(row, x):
        name, cw = row["cw"]["conv"].name, row["cw"]
        h, n = hessian(x, cw["attrs"])
        w = cw["w"].reshape(cw["w"].shape[0], -1)
        s = row_scales(row)
        q, _ = gptq_quantize(w, h, s, GRID[0], GRID[1])
        before = recon_loss(w, row["q"]["q4"].reshape(w.shape[0], -1), s, h)
        after = recon_loss(w, q, s, h)
        row["q"] = q_with(q.reshape(cw["w"].shape).astype(np.int8), row["q"])
        info = {"cols": int(w.shape[1]), "positions": int(n), "dead": int(np.sum(np.diag(h) == 0)),
                "recon_rtn": before, "recon_gptq": after}
        record[name] = {"gptq": info}
        return info
    return step


# ---------------------------------------------------------------------------------------------------- AdaRound

def prepared_float(model: Model) -> Graph:
    """quant's AdaRound reference: the float graph after the vendor's pre-process (Split -> Slice); no CLE, which
    finds 0 patterns on these graphs (results/quant/prepare_probe_yolov8n_cut.log:269, :296)."""
    from quant.quantize import graph_family, prepare, simplify_for
    fg = Graph.load(ROOT / model.float_path, strict=True)   # quant/cli.py's adaround path, less CLE
    family = graph_family(fg)
    if simplify_for(fg, family):
        fg.infer_shapes()
    prepare(fg, family)
    fg.infer_shapes()
    return fg


def stored_of(row: dict) -> np.ndarray:
    if row["form"] == "b":
        st = np.zeros(row["cw"]["w"].shape, dtype=np.int16)
        for gr in row["q"]["groups"]:
            st[gr["slice"]] = row["q"]["q4"][gr["slice"]].astype(np.int16) << gr["k"]
        return st.astype(np.int8)
    return row["q"]["q4"]


def run_adaround(model: Model, rows: list[dict], source, float_graph: Graph, log, record: dict,
                 iterations: int = AR["NumIterations"]):
    from quant.adaround import FastFinetuneConfig, WeightGrid, finetune
    qg = build(model, rows)
    by_name = {r["cw"]["conv"].name: r for r in rows}
    grids = {}
    for name, r in by_name.items():
        grids[name] = None if r["form"] is None else WeightGrid(
            scales=row_scales(r).astype(np.float32), low=GRID[0], high=GRID[1], q4=r["q"]["q4"].copy())

    def write_back(name, q):
        r = by_name[name]
        r["q"] = q_with(np.asarray(q, dtype=np.int8), r["q"])
        qg.set_initializer(r["qname"], stored_of(r))
    cfg = FastFinetuneConfig(DataSize=AR["DataSize"], FixedSeed=AR["FixedSeed"], BatchSize=AR["BatchSize"],
                             NumIterations=iterations, LearningRate=AR["LearningRate"], EarlyStop=AR["EarlyStop"],
                             RegParam=AR["RegParam"], BetaRange=tuple(AR["BetaRange"]), WarmStart=AR["WarmStart"],
                             DropRatio=AR["DropRatio"])
    torch_mod()   # torch at the protocol's 8 threads; quant/adaround sets none and its ORT sessions keep the default
    rep = finetune(float_graph, qg, source, cfg, log=log, weight_grid=grids, write_back=write_back)
    for lr in rep.layers:
        record[lr.name] = {"ar": {"recons_before": lr.recons_before, "recons_after": lr.recons_after,
                                  "iterations": lr.iterations, "early_stop": lr.early_stop,
                                  "changed": lr.changed_elements}}
    return rep


# --------------------------------------------------------------------------------------------- MX sensitivity

def sensitivity(model: Model, images: list[np.ndarray], log) -> tuple:
    """Per form and Conv: sum ||conv(W_f, x) - conv(W4, x)||^2 / sum ||conv(W_f, x)||^2, both on B8's captured input
    x with the float bias (W_f the float weights, W4 RTN's); and, apart, y_f's per-channel statistics for BC and VC
    (the float network's own Conv outputs)."""
    ins = list(dict.fromkeys(model.by_name[n]["conv"].input[0] for n in model.order))   # heads share inputs
    outs = [model.fconv[n].output[0] for n in model.order]
    qs = ort_session(extract(model.shipped.model, [model.input_name], ins))
    fs = ort_session(extract(model.gf.model, [model.input_name], outs))
    forms = ("e1", "e2", "u")
    rows = {f: {r["cw"]["conv"].name: r for r in rtn_rows(model, f)} for f in forms}
    err = {f: {n: 0.0 for n in model.order} for f in forms}
    ref = {n: 0.0 for n in model.order}
    fref = {n: [0.0, 0.0, 0] for n in model.order}
    for k, image in enumerate(images):
        xin = dict(zip(ins, qs.run(ins, {model.input_name: image})))
        ys = fs.run(outs, {model.input_name: image})
        for n, yf in zip(model.order, ys):
            cw = model.by_name[n]
            yd = yf.astype(np.float64)
            a = fref[n]
            a[0] = a[0] + yd.sum(axis=(0, 2, 3))
            a[1] = a[1] + (yd * yd).sum(axis=(0, 2, 3))
            a[2] += yd.shape[0] * yd.shape[2] * yd.shape[3]
            x = xin[cw["conv"].input[0]]
            y8 = conv_out(x, cw["w"], cw["bias_f"], cw["attrs"]).astype(np.float64)
            ref[n] += float((y8 * y8).sum())
            for f in forms:
                yq = conv_out(x, rows[f][n]["q"]["w_hat"], cw["bias_f"], cw["attrs"]).astype(np.float64)
                err[f][n] += float(((y8 - yq) ** 2).sum())
        if (k + 1) % 16 == 0 or k + 1 == len(images):
            log(f"[sensitivity] {k + 1}/{len(images)} images")
    sizes = {n: model.by_name[n]["w"].size for n in model.order}
    sets = {}
    for f in forms:
        rel = {n: err[f][n] / ref[n] if ref[n] > 0 else math.inf for n in model.order}
        sets[f] = {**mx_select(model.order, sizes, rel, MX_BUDGET), "rel": rel}
    fstats = {n: {"sum": a[0], "sumsq": a[1], "count": a[2]} for n, a in fref.items()}
    return sets, fstats


def mx_select(order: list[str], sizes: dict, rel: dict, budget: float) -> dict:
    """Most sensitive first (ties in Conv order); stop at the first Conv that would take the W8 share of the
    Conv weight elements over the budget (no later, smaller Conv is taken)."""
    total = sum(sizes[n] for n in order)
    ranked = sorted(order, key=lambda n: (-rel[n], order.index(n)))
    keep, used = [], 0
    for n in ranked:
        if (used + sizes[n]) / total > budget:
            break
        keep.append(n)
        used += sizes[n]
    return {"w8": keep, "share": used / total, "ranked": ranked}


def save_captures(model: Model, sets: dict, fstats: dict, cap: Path) -> tuple:
    cap.mkdir(parents=True, exist_ok=True)
    mx_path, yf_path = cap / f"{model.name}_mx.json", cap / f"{model.name}_yf.npz"
    for p in (mx_path, yf_path):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {shown(p)}")
    mx = {"model": model.name, "budget": MX_BUDGET, "rule": vv.HYPER["mx"]["rule"],
          "forms": {f: {"w8": v["w8"], "share": v["share"],
                        "rel": {n: v["rel"][n] for n in v["ranked"]}} for f, v in sets.items()}}
    mx_path.write_text(json.dumps(mx, indent=1) + "\n", encoding="utf-8", newline="\n")
    arrays = {}
    for i, n in enumerate(model.order):
        arrays[f"sum_{i}"], arrays[f"sumsq_{i}"] = fstats[n]["sum"], fstats[n]["sumsq"]
        arrays[f"count_{i}"] = np.array(fstats[n]["count"], dtype=np.int64)
    arrays["order"] = np.array(model.order)
    np.savez(yf_path, **arrays)
    return mx_path, yf_path


def capture_file(path: Path, require_pin: bool, log) -> Path:
    """A capture file as pinned: a pinned one must match wherever it is read; tier 2 (after the prereg log) also
    refuses an unpinned one, since the y_f statistics set the BC biases and the MX file the W8 Convs. The pins come
    from committed, clean logs only; before any prereg or pins log exists, tier 1 reads an empty table."""
    got, want = vv.sha_file(path), vv.pins(allow_none=not require_pin).get(path.name)
    log(f"[captures] {shown(path)}  sha256 {got}  "
        f"{'== pinned' if want == got else 'NOT the pinned ' + want if want else 'not pinned'}")
    if (want is not None and want != got) or (require_pin and want is None):
        raise SystemExit(f"{shown(path)}: {'not the pinned file' if want else 'no committed prereg log pins it'}")
    return path


def load_fref(model: Model, cap: Path, require_pin: bool = False, log=print) -> dict:
    z = np.load(capture_file(cap / f"{model.name}_yf.npz", require_pin, log))
    if list(z["order"]) != model.order:
        raise SystemExit(f"{model.name}_yf.npz: Conv order differs")
    return {n: {"sum": z[f"sum_{i}"], "sumsq": z[f"sumsq_{i}"], "count": int(z[f"count_{i}"])}
            for i, n in enumerate(model.order)}


def load_mx(model: Model, cap: Path, form: str, require_pin: bool = True, log=print) -> list[str]:
    path = capture_file(cap / f"{model.name}_mx.json", require_pin, log)
    return json.loads(path.read_text(encoding="utf-8"))["forms"][form]["w8"]


# ------------------------------------------------------------------------------------------------ calibration

def calib_source(model: Model):
    from npu.yolo import input_size
    from quant.sources import CocoSource
    shape = model.shipped.value_shape(model.input_name)
    imgsz = input_size(list(shape), str(model.xint8_path))
    return CocoSource(ROOT / vv.CALIB["dir"], vv.CALIB["n"], imgsz, model.input_name)


def calib_listing_ok(source) -> bool:
    names = [p.name for p in source.listing()]
    text = "".join(f"{n} {vv.sha_file(ROOT / vv.CALIB['dir'] / n)}\n" for n in names)
    got = vv.sha_bytes(text.encode())
    ok = (len(names) == vv.CALIB["n"] and names[0] == vv.CALIB["first"] and names[-1] == vv.CALIB["last"]
          and got == vv.CALIB["listing_sha256"])
    print(f"[calib] {len(names)} images {names[0]} .. {names[-1]}, listing sha256 {got} "
          f"{'== pinned' if ok else 'MISMATCH'}", flush=True)
    return ok


def rows_from_file(model: Model, path: Path, form: str) -> list[dict]:
    """A tier-1 AdaRound or GPTQ file read back as rows on RTN's group layout (its pos4, k and forms)."""
    em = Graph.load(path, strict=True)
    rows = rtn_rows(model, form)
    for r in rows:
        stored = em.initializer(r["qname"])
        dq = next(n for n in em.nodes() if n.output[0] == r["cw"]["dq"].output[0])
        is_a = any(a.name == "axis" for a in dq.attribute)
        if is_a != (r["form"] == "a"):
            raise SystemExit(f"{shown(path)}: {r['cw']['conv'].name} is not in RTN's form")
        if r["form"] == "b":
            q4 = np.zeros(stored.shape, dtype=np.int16)
            for gr in r["q"]["groups"]:
                s = stored[gr["slice"]].astype(np.int16)
                if np.any(s % (1 << gr["k"])):
                    raise SystemExit(f"{shown(path)}: {r['cw']['conv'].name} is not q4 << k")
                q4[gr["slice"]] = s >> gr["k"]
        else:
            q4 = stored
        r["q"] = q_with(q4.astype(np.int8), r["q"])
    rebuilt = build(model, rows)
    d = graph_diff(rebuilt, em)
    if d.node_delta or d.init_exact_mismatch or d.weight_lsb:
        raise SystemExit(f"{shown(path)}: the rows read back do not rebuild the file")
    return rows


# --------------------------------------------------------------------------------------------------- the arms

def run_method(model: Model, arm: str, images, source, cap: Path, out_dir: Path, log) -> dict:
    method, form = arm.split("_")
    record, biases, rescaled, extra = {}, {}, {}, {}
    if method in ("rtn",):
        rows = rtn_rows(model, form)
    elif method in ("bc", "vc"):
        rows = rtn_rows(model, form)
        fref = load_fref(model, cap)
        step = step_bc(fref, biases, record) if method == "bc" else step_vc(fref, biases, record, rescaled, form)
        sequential(model, rows, biases, images, step, log)
    elif method in ("gptq", "mxgptq"):
        keep = load_mx(model, cap, form) if method == "mxgptq" else ()
        rows = rtn_rows(model, form, keep)
        sequential(model, rows, biases, images, step_gptq(record), log)
    elif method in ("ar", "mxar"):
        keep = load_mx(model, cap, form) if method == "mxar" else ()
        rows = rtn_rows(model, form, keep)
        rep = run_adaround(model, rows, source, prepared_float(model), log, record)
        extra["adaround"] = {"data_seconds": rep.data_seconds, "train_seconds": rep.train_seconds,
                             "peak_rss_bytes": rep.peak_rss_bytes, "onnxruntime": rep.onnxruntime_version,
                             "torch": rep.torch_version, "torch_threads": rep.torch_threads, **thread_facts()}
    elif method in ("arbc", "gptqbc"):
        base = arm_file(out_dir, model.name, f"{'ar' if method == 'arbc' else 'gptq'}_{form}")
        pinned = vv.pins().get(base.name)
        got = vv.sha_file(base)
        print(f"[base] {shown(base)}  sha256 {got}  {'== pinned' if pinned == got else 'NOT PINNED'}", flush=True)
        if pinned != got:
            raise SystemExit("the tier-2 base must be the pinned tier-1 file")
        rows = rows_from_file(model, base, form)
        fref = load_fref(model, cap, require_pin=True)
        sequential(model, rows, biases, images, step_bc(fref, biases, record), log)
        extra["base"] = {"file": shown(base), "sha256": got}
    else:
        raise SystemExit(f"unknown method {method}")
    return {"rows": rows, "biases": biases, "record": record, "rescaled": rescaled, "extra": extra}


def emit_arm(model: Model, arm: str, res: dict, out_dir: Path, name: str | None = None) -> Path:
    out = out_dir / (name or out_name(model.name, arm, res["rows"]))
    if out.exists():
        raise SystemExit(f"refusing to overwrite {shown(out)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    g = build(model, res["rows"], res["biases"])
    g.save(out)
    return out


def conv_record(r: dict, res: dict) -> dict:
    """Gate D's per-conv sidecar fields (w4a8_emulate cmd_emit), from the method's final integers, plus the
    method's own per-layer figures and the int32 bias range."""
    cw, q = r["cw"], r["q"]
    w, name = cw["w"], cw["conv"].name
    rec = {"conv": name, "weight": cw["qname"], "shape": list(w.shape), "form": r["form"], "pos8": cw["pos8"],
           "shift_out": cw["shift_out"], "e_bias": cw["e_bias"],
           "sqnr_w8_db": round(g4.sqnr_db(w, cw["stored"].astype(np.float32) * pos2scale(cw["pos8"])), 3)}
    if q is None:
        rec["kept_w8"] = True
        return rec
    real = [gr for gr in q["groups"] if not gr["zero"]]
    ks = [gr["k"] for gr in q["groups"]]
    kmax = max(ks)
    rec.update({"groups": len(q["groups"]), "zero_groups": len(q["groups"]) - len(real), "k_min": min(ks),
                "k_max": kmax, "k_hist": {str(k): ks.count(k) for k in sorted(set(ks))},
                "sqnr_w4_db": round(g4.sqnr_db(w, q["w_hat"]), 3),
                "levels_max": max((int(np.unique(q["q4"][gr["slice"]]).size) for gr in real), default=0),
                "shift_out_minus_kmax": None if cw["shift_out"] is None else cw["shift_out"] - kmax,
                "e_bias_minus_kmax": None if cw["e_bias"] is None else cw["e_bias"] - kmax})
    if "scales" in q:
        alphas = [gr["alpha"] for gr in real]
        rec.update({"alpha_min": min(alphas, default=None), "alpha_max": max(alphas, default=None)})
    else:
        p4 = [gr["pos4"] for gr in q["groups"]]
        rec.update({"pos4_min": min(p4), "pos4_max": max(p4),
                    "edge_low": sum(gr.get("edge") == "low" for gr in real),
                    "edge_high": sum(gr.get("edge") == "high" for gr in real),
                    "clipped_frac": sum(gr.get("clipped", 0) for gr in real) / w.size})
    if name in res["biases"]:
        b = res["biases"][name]["i32"]
        rec["int32_bias"] = {"min": int(b.min()), "max": int(b.max())}
    rec.update(res["record"].get(name, {}))
    return rec


def sidecar(model: Model, arm: str, res: dict, out: Path, fails: list[str], seconds: float) -> dict:
    convs = [conv_record(r, res) for r in res["rows"]]
    q_rows = [c for c in convs if not c.get("kept_w8")]
    all_k = [int(k) for c in q_rows for k, n in c["k_hist"].items() for _ in range(n)]

    def pinned(p):
        return None if p is None else {"path": Path(p).as_posix(), "sha256": vv.sha_file(ROOT / p)}
    summary = {
        "model": model.name, "arm": arm, "grid": list(GRID), "window": list(WINDOW), "form_requested": "auto",
        "convs_form_b": sum(r["form"] == "b" for r in res["rows"]),
        "convs_form_a": sum(r["form"] == "a" for r in res["rows"]),
        "convs_kept_w8": sum(r["form"] is None for r in res["rows"]),
        "k_hist": {str(k): all_k.count(k) for k in sorted(set(all_k))},
        "groups_at_window_edge": {"low": sum(c.get("edge_low", 0) for c in q_rows),
                                  "high": sum(c.get("edge_high", 0) for c in q_rows)},
        "convs_shift_out_minus_kmax_negative": sum(1 for c in q_rows if (c["shift_out_minus_kmax"] or 0) < 0),
        "convs_e_bias_minus_kmax_negative": sum(1 for c in q_rows if (c["e_bias_minus_kmax"] or 0) < 0),
        "int32_bias_convs": len(res["biases"]),
        "float": pinned(model.float_path), "xint8": pinned(model.xint8_path),
        "out": {"path": shown(out), "sha256": vv.sha_file(out)},
        "controls_pass": not fails, "control_failures": fails, "seconds": round(seconds, 1),
        "triple": vv.frozen_values(), **res["extra"]}
    return {"summary": summary, "convs": convs}


def compare(a: Path, b: Path, n: int) -> int:
    """gate D's compare (it prints the paths as given, so they stay repo-relative; main() runs from the root)."""
    return g4.cmd_compare(argparse.Namespace(a=Path(a).as_posix(), b=Path(b).as_posix(), images="data/coco/val2017",
                                             n=n, threads=ORT_THREADS))


def arm_file(out_dir: Path, model: str, arm: str) -> Path:
    found = [p for p in (out_dir / f"{model}_w4v_{arm}.onnx", out_dir / f"{model}_w4v_{arm}_ortonly.onnx")
             if p.is_file()]
    if len(found) != 1:
        raise SystemExit(f"want exactly one file for {model} {arm}, found {[shown(p) for p in found]}")
    return found[0]


def emit_one(model: Model, arm: str, images, source, cap: Path, out_dir: Path, side_dir: Path, tag: str,
             twin: bool, repeat: bool) -> bool:
    print(f"\n### emit {model.name} {arm}", flush=True)
    side = side_dir / f"w4v_sidecar_{model.name}_{arm}_{tag}.json"
    if side.exists():
        raise SystemExit(f"refusing to overwrite {shown(side)}")
    t0 = time.perf_counter()
    res = run_method(model, arm, images, source, cap, out_dir, print)
    out = emit_arm(model, arm, res, out_dir)
    fails = check_arm(out, model, res["rows"], res["biases"], arm, res["rescaled"])
    seconds = time.perf_counter() - t0
    digest = vv.sha_file(out)
    rows = res["rows"]
    print(f"[emit] {model.name} {arm}: forms b {sum(r['form'] == 'b' for r in rows)}, a "
          f"{sum(r['form'] == 'a' for r in rows)}, kept W8 {sum(r['form'] is None for r in rows)}; int32 biases "
          f"{len(res['biases'])}; {seconds:.1f} s", flush=True)
    print(f"[emit] wrote {shown(out)}  sha256 {digest}", flush=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(sidecar(model, arm, res, out, fails, seconds), indent=1) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"[emit] sidecar {shown(side)}  sha256 {vv.sha_file(side)}", flush=True)
    marker("C5", f"{model.name}_{arm}", not fails, "; ".join(fails))
    ok = not fails
    if twin:
        forced = [dict(r, form="a") if r["form"] == "b" else r for r in rows]
        twin_path = emit_arm(model, arm, {**res, "rows": forced}, out_dir,
                             f"{model.name}_w4v_{arm}_forma_ortonly.onnx")
        print(f"[twin] {shown(twin_path)}  sha256 {vv.sha_file(twin_path)}", flush=True)
        rc = compare(out, twin_path, 16)
        marker("C3", f"{model.name}_{arm}", rc == 0)
        ok &= rc == 0
    if repeat:
        res2 = run_method(model, arm, images, source, cap, out_dir, lambda *a, **k: None)
        rep_path = emit_arm(model, arm, res2, out_dir, out_name(model.name, f"{arm}_repeat", res2["rows"]))
        same = vv.sha_file(rep_path) == digest
        print(f"[repeat] {shown(rep_path)}  sha256 {vv.sha_file(rep_path)}  {'==' if same else '!='} the first",
              flush=True)
        marker("C7", f"{model.name}_{arm}", same)
        ok &= same
    return ok


# ------------------------------------------------------------------------------------------------------ modes

def cmd_controls(args) -> int:
    """C0, ORDER, MINMSE, C1, C2 (with each RTN arm's C5), C3, C4 in that order; the first FAIL is a STOP: nothing
    after it runs."""
    vv.print_hashes()
    import onnxruntime as ort
    model = Model.load(args.model)
    out_dir, side_dir = Path(args.out_dir), Path(args.sidecar_dir)

    def stopped(what: str) -> int:
        print(f"[controls] {model.name} FAIL (STOP at {what}; no later control ran)", flush=True)
        return 1

    # C0: the pins
    h = vv.hashes()
    src = calib_source(model)
    ann_path = ROOT / vv.ANN["path"]
    ann_sha = vv.sha_file(ann_path)
    ids = sorted(int(i["id"]) for i in json.loads(ann_path.read_text(encoding="utf-8"))["images"])
    calib_all = sorted(p.name for p in (ROOT / vv.CALIB["dir"]).glob("*.jpg"))
    cids = {int(Path(n).stem) for n in calib_all}
    h5000, h500 = vv.h_ids(ids, cids), vv.h_ids(ids[:500], cids)
    checks = {
        "text == the accepted v2.4": h["PREREG_TEXT_SHA256"] == vv.PREREG_SHA,
        f"onnxruntime {ort.__version__} == {vv.ORT_VERSION}": ort.__version__ == vv.ORT_VERSION,
        f"env {os.environ.get('CONDA_DEFAULT_ENV')} == {vv.ENV}": os.environ.get("CONDA_DEFAULT_ENV") == vv.ENV,
        "calibration listing": calib_listing_ok(src),
        f"coco_calib {len(calib_all)} names": (len(calib_all) == vv.CALIB["all_count"] and vv.sha_bytes(
            "".join(f"{n}\n" for n in calib_all).encode()) == vv.CALIB["all_names_sha256"]),
        f"annotations sha256 {ann_sha}": ann_sha == vv.ANN["sha256"],
        f"H_5000 {len(h5000)} ids": (len(h5000), vv.id_text_sha(h5000)) == vv.H_PIN[5000],
        f"H_500 {len(h500)} ids": (len(h500), vv.id_text_sha(h500)) == vv.H_PIN[500],
    }
    for k, v in checks.items():
        print(f"[C0] {'ok' if v else 'FAIL'} {k}", flush=True)
    marker("C0", model.name, all(checks.values()))
    if not all(checks.values()):
        return stopped("C0")

    # the order fact: AdaRound's vendor_order against the XINT8 Conv order (both topological)
    from quant.adaround import layer_targets
    vorder = [L.name for L in layer_targets(model.shipped, prepared_float(model))]
    diff = [i for i, (a, b) in enumerate(zip(model.order, vorder)) if a != b]
    upstream, anc = {}, {}   # tensor -> the Convs it depends on; the nodes are topological (strict load)
    for node in model.shipped.nodes():
        s = frozenset().union(*(upstream.get(i, frozenset()) for i in node.input))
        if node.op_type == "Conv":
            anc[node.name] = s
            s = s | {node.name}
        for o in node.output:
            upstream[o] = s
    topo = all(order.index(a) < order.index(n) for order in (model.order, vorder) for n in model.order
               for a in anc[n])
    order_ok = set(vorder) == set(model.order) and len(vorder) == len(model.order) and topo
    moved = [f"{model.order[i]} -> {vorder[i]}" for i in diff[:3]]
    print(f"[order] AdaRound's vendor_order vs the XINT8 Conv order: same set {set(vorder) == set(model.order)}, "
          f"{len(diff)} positions differ (from index {diff[0] if diff else None}: {'; '.join(moved)}), every Conv "
          f"after all its Conv ancestors in both {topo}", flush=True)
    marker("ORDER", model.name, order_ok)
    if not order_ok:
        return stopped("ORDER")

    # the MinMSE cross-check on every RTN group (C5's pos4 rule, both implementations)
    bad = n_groups = 0
    for f in ("e1", "e2", "u"):
        for cw in model.cws:
            for sl in g4.groups(cw["w"].shape[0], f):
                x = cw["w"][sl]
                if not np.any(x):
                    continue
                n_groups += 1
                bad += g4.choose_pos(x, GRID[0], GRID[1], WINDOW)[0] != choose_pow2_minmse(
                    x, "int8", grid=GRID, window=WINDOW)[0]
    print(f"[minmse] choose_pos == quant.calib.choose_pow2_minmse(grid [-7, 7], window -1:5) on {n_groups} groups "
          f"(E1, E2, U): {n_groups - bad}/{n_groups}", flush=True)
    marker("MINMSE", model.name, bad == 0)
    if bad:
        return stopped("MINMSE")

    # C1: w8check through this tool's builder
    rows = [{"cw": cw, "qname": cw["qname"], "form": "b",
             "q": g4.quantize_conv(cw, "w8check", g4.W8[0], g4.W8[1], g4.W8_WINDOW)} for cw in model.cws]
    p = out_dir / f"{model.name}_w4v_c1_w8check.onnx"
    if p.exists():
        raise SystemExit(f"refusing to overwrite {shown(p)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    build(model, rows).save(p)
    fails = g4.check_emitted(p, model.shipped, rows, "w8check", g4.W8[0], g4.W8[1])
    got = vv.sha_file(p)
    ok = not fails and got == vv.GATE_D_FILES[model.name]["w8check"]
    print(f"[C1] {shown(p)}  sha256 {got}  {'== gate D' if ok else '!= gate D ' + vv.GATE_D_FILES[model.name]['w8check']}")
    marker("C1", model.name, ok)
    if not ok:
        return stopped("C1")

    # C2 (and the RTN arms E1, E2, U, which the screen and C9 use): byte for byte gate D's files
    c2 = True
    for form in ("e1", "e2", "u"):
        if not emit_one(model, f"rtn_{form}", None, None, Path(args.captures), out_dir, side_dir, args.tag, False,
                        False):
            return stopped(f"C5 {model.name}_rtn_{form}")
        f = arm_file(out_dir, model.name, f"rtn_{form}")
        same = vv.sha_file(f) == vv.GATE_D_FILES[model.name][form]
        print(f"[C2] {shown(f)} {'==' if same else '!='} gate D's {form} {vv.GATE_D_FILES[model.name][form][:12]}")
        c2 &= same
    rows = rtn_rows(model, "e1", keep=set(g4.stem_and_head(model.shipped)))
    p = out_dir / f"{model.name}_w4v_c2_e1h.onnx"
    if p.exists():
        raise SystemExit(f"refusing to overwrite {shown(p)}")
    build(model, rows).save(p)
    same = vv.sha_file(p) == vv.GATE_D_FILES[model.name]["e1h"]
    print(f"[C2] {shown(p)} {'==' if same else '!='} gate D's e1h {vv.GATE_D_FILES[model.name]['e1h'][:12]}")
    c2 &= same
    marker("C2", model.name, c2)
    if not c2:
        return stopped("C2")

    # C3 (RTN E1, form a == form b) and C4 (the negative control)
    e1 = arm_file(out_dir, model.name, "rtn_e1")
    rows = rtn_rows(model, "e1")
    forced = [dict(r, form="a") for r in rows]
    twin = out_dir / f"{model.name}_w4v_rtn_e1_forma_ortonly.onnx"
    if twin.exists():
        raise SystemExit(f"refusing to overwrite {shown(twin)}")
    build(model, forced).save(twin)
    rc = compare(e1, twin, 16)
    marker("C3", f"{model.name}_rtn_e1", rc == 0)
    if rc != 0:
        return stopped("C3")
    rc = compare(model.xint8_path, e1, 2)
    marker("C4", model.name, rc == 1, "compare tells the shipped file from RTN E1" if rc == 1 else f"exit {rc}")
    if rc != 1:
        return stopped("C4")
    print(f"[controls] {model.name} PASS", flush=True)
    return 0


def cmd_sensitivity(args) -> int:
    vv.print_hashes()
    model = Model.load(args.model)
    src = calib_source(model)
    if not calib_listing_ok(src):
        raise SystemExit("the calibration listing is not the pinned one")
    images = list(src)
    t0 = time.perf_counter()
    sets, fstats = sensitivity(model, images, print)
    mx_path, yf_path = save_captures(model, sets, fstats, Path(args.captures))
    for f, v in sets.items():
        print(f"[mx] {model.name} {f}: W8 {len(v['w8'])} convs, {100 * v['share']:.3f}% of conv weight elements "
              f"(budget {100 * MX_BUDGET:.0f}%): {', '.join(v['w8'])}")
    print(f"[captures] {shown(mx_path)}  sha256 {vv.sha_file(mx_path)}")
    print(f"[captures] {shown(yf_path)}  sha256 {vv.sha_file(yf_path)}")
    print(f"[sensitivity] {time.perf_counter() - t0:.1f} s")
    print(f"W4V_DONE sensitivity {model.name}", flush=True)   # the stage's success line (tools/w4v_verdict.py ready)
    return 0


def cmd_emit(args) -> int:
    vv.print_hashes()
    arm = f"{args.method}_{args.form}"
    if arm not in vv.arms(args.tier):
        raise SystemExit(f"{arm} is not a tier-{args.tier} arm of the matrix")
    model = Model.load(args.model)
    src = calib_source(model)
    if not calib_listing_ok(src):
        raise SystemExit("the calibration listing is not the pinned one")
    images = list(src)
    ok = emit_one(model, arm, images, src, Path(args.captures), Path(args.out_dir), Path(args.sidecar_dir), args.tag,
                  args.twin, args.repeat)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------- selftest

def synth(rng, cin=3, c1=40, c2=6, hw=12, grid2=False):
    """A two-Conv model in the XINT8 dialect (uint8/zp128 activations, pow2 int8 weights and biases) and its float
    twin, with gate D's per-Conv records built by hand (conv_weights wants the 63 Convs of YOLOv8). grid2: conv2's
    float weights are q4 * 2^-6 with every output channel holding +7 and -7, so RTN W4 keeps them exactly in E1, E2
    and U (MinMSE's first zero-error position is 6 in each group)."""
    from onnx import TensorProto
    from quant.pow2 import quantize as pquant
    w1 = (rng.standard_normal((c1, cin, 3, 3)) * 0.2).astype(np.float32)
    w1[5] = 0.0   # an all-zero channel
    b1 = (rng.standard_normal(c1) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((c2, c1, 1, 1)) * 0.1).astype(np.float32)
    b2 = (rng.standard_normal(c2) * 0.05).astype(np.float32)
    if grid2:
        q2 = rng.integers(-7, 8, (c2, c1, 1, 1))
        q2[:, 0], q2[:, 1] = 7, -7
        w2 = (q2 * pos2scale(6)).astype(np.float32)

    def init(name, arr):
        return numpy_helper.from_array(np.asarray(arr), name)
    nodes, inits = [], []
    fnodes, finits = [], []
    act = {"images": 7, "y1": 5, "y2": 5}
    for t, pos in act.items():
        inits += [init(f"{t}_scale", np.float32(pos2scale(pos))), init(f"{t}_zero_point", np.uint8(128))]
    nodes += [helper.make_node("QuantizeLinear", ["images", "images_scale", "images_zero_point"], ["images_q"],
                               name="q_in"),
              helper.make_node("DequantizeLinear", ["images_q", "images_scale", "images_zero_point"], ["images_dq"],
                               name="dq_in")]
    cws, x_in = [], "images"
    for i, (w, b, k, out) in enumerate(((w1, b1, 3, "y1"), (w2, b2, 1, "y2")), 1):
        pos8, _ = choose_pow2_minmse(w, "int8")
        stored = pquant(w, pos8, 0, "int8")
        posb, _ = choose_pow2_minmse(b, "int8")
        bq = pquant(b, posb, 0, "int8")
        wn = f"conv{i}.weight"
        inits += [init(wn + "_quantized", stored), init(wn + "_scale", pos2scale(pos8)),
                  init(wn + "_zero_point", np.int8(0)), init(f"conv{i}.bias_quantized", bq),
                  init(f"conv{i}.bias_scale", pos2scale(posb)), init(f"conv{i}.bias_zero_point", np.int8(0))]
        nodes += [helper.make_node("DequantizeLinear", [wn + "_quantized", wn + "_scale", wn + "_zero_point"],
                                   [wn + "_dq"], name=f"dqw{i}"),
                  helper.make_node("DequantizeLinear", [f"conv{i}.bias_quantized", f"conv{i}.bias_scale",
                                                        f"conv{i}.bias_zero_point"], [f"conv{i}.bias_dq"],
                                   name=f"dqb{i}")]
        pads = [k // 2] * 4
        nodes.append(helper.make_node("Conv", [f"{x_in}_dq", wn + "_dq", f"conv{i}.bias_dq"], [out], name=f"/conv{i}",
                                      kernel_shape=[k, k], pads=pads, strides=[1, 1]))
        nodes += [helper.make_node("QuantizeLinear", [out, f"{out}_scale", f"{out}_zero_point"], [f"{out}_q"],
                                   name=f"q_{out}"),
                  helper.make_node("DequantizeLinear", [f"{out}_q", f"{out}_scale", f"{out}_zero_point"],
                                   [f"{out}_dq"], name=f"dq_{out}")]
        finits += [init(wn, w), init(f"conv{i}.bias", b)]
        fnodes.append(helper.make_node("Conv", [x_in, wn, f"conv{i}.bias"], [out], name=f"/conv{i}",
                                       kernel_shape=[k, k], pads=pads, strides=[1, 1]))
        cws.append({"qname": wn + "_quantized", "w": w, "stored": stored, "pos8": pos8, "shift_out": None,
                    "e_bias": None})
        x_in = out
    vin = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, cin, hw, hw])
    qout = helper.make_tensor_value_info("y2_dq", TensorProto.FLOAT, [1, c2, hw, hw])
    fout = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, c2, hw, hw])
    opset = [helper.make_opsetid("", 17)]
    qm = helper.make_model(helper.make_graph(nodes, "synth_q", [vin], [qout], inits), opset_imports=opset,
                           ir_version=8)
    fm = helper.make_model(helper.make_graph(fnodes, "synth_f", [vin], [fout], finits), opset_imports=opset,
                           ir_version=8)
    shipped, gf = Graph(qm), Graph(fm)
    shipped.check()
    gf.check()
    by_output = {o: n for n in shipped.nodes() for o in n.output}
    for i, cw in enumerate(cws, 1):
        cw["conv"] = next(n for n in shipped.nodes() if n.name == f"/conv{i}")
        cw["dq"] = by_output[cw["conv"].input[1]]
    return Model("synth", shipped, gf, cws), qm, fm


def selftest() -> int:
    """Synthetic only: random tensors and a two-Conv model built here; no model file, image or W4V file is read."""
    vv.print_hashes()
    torch = torch_mod()
    import torch.nn.functional as F
    from quant.pow2 import bounds, quantize as pquant, sqerr
    fails = []

    def expect(what, ok):
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            fails.append(what)

    rng = np.random.default_rng(20260925)

    # first, before any other torch call (torch warns once per process): an ONNX initializer, as the real models'
    # float weights and bias arrive, is read-only, and torch must get a writable copy of it
    import warnings
    w_ro = numpy_helper.to_array(numpy_helper.from_array(np.arange(108, dtype=np.float32).reshape(4, 3, 3, 3) / 100,
                                                         "w"))
    b_ro = numpy_helper.to_array(numpy_helper.from_array(np.ones(4, dtype=np.float32), "b"))
    at = {"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "kernel": (3, 3)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conv_out(np.ones((2, 3, 8, 8), np.float32), w_ro, b_ro, at)
        conv_sums(np.ones((2, 3, 8, 8), np.float32), w_ro, b_ro, at)
    expect(f"read-only ONNX initializers (writeable {w_ro.flags.writeable}, {b_ro.flags.writeable}) reach torch as "
           f"copies: no 'not writable' warning", not w_ro.flags.writeable and not b_ro.flags.writeable
           and not any("not writable" in str(c.message) for c in caught))

    # quant/'s additive grid: the defaults are today's arithmetic, the grid is gate D's
    same_w8 = same_w4 = True
    for _ in range(200):
        x = (rng.standard_normal(int(rng.integers(1, 400))) * rng.uniform(1e-3, 3)).astype(np.float32)
        same_w8 &= choose_pow2_minmse(x, "int8")[0] == g4.choose_pos(x, g4.W8[0], g4.W8[1], g4.W8_WINDOW)[0]
        same_w4 &= choose_pow2_minmse(x, "int8", grid=GRID, window=WINDOW)[0] == g4.choose_pos(x, GRID[0], GRID[1],
                                                                                                WINDOW)[0]
    expect("choose_pow2_minmse's default == gate D's W8 choose_pos on 200 random tensors", same_w8)
    expect("choose_pow2_minmse(grid [-7, 7], window -1:5) == gate D's W4 choose_pos on 200 random tensors", same_w4)
    x = (rng.standard_normal(1000) * 2).astype(np.float32)
    expect("pow2.quantize's default is clip(rint(x / s) + zp) to the producer range",
           np.array_equal(pquant(x, 3, 0, "int8"), np.clip(np.rint(x / pos2scale(3)), -127, 127).astype(np.int8))
           and np.array_equal(pquant(x, 3, 128, "uint8"),
                              np.clip(np.rint(x / pos2scale(3)) + 128, 0, 255).astype(np.uint8)))
    expect("pow2.quantize(grid) == gate D's quantize_bounded",
           np.array_equal(pquant(x, 2, 0, "int8", GRID), g4.quantize_bounded(x, 2, GRID[0], GRID[1])))
    expect("pow2.sqerr(grid) == gate D's sqerr_bounded", sqerr(x, 2, 0, "int8", GRID) == g4.sqerr_bounded(x, 2, *GRID))
    bad_grid = False
    try:
        bounds("int8", (-128, 127))
    except ValueError:
        bad_grid = True
    bad_window = False
    try:
        choose_pow2_minmse(x, "int8", window=(1, 3))
    except ValueError:
        bad_window = True
    expect("a grid outside the producer range and a window without 0 are refused", bad_grid and bad_window)

    # im2col layout and GPTQ
    xi = rng.standard_normal((2, 5, 7, 7)).astype(np.float32)
    wi = rng.standard_normal((4, 5, 3, 3)).astype(np.float32)
    attrs = {"stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "kernel": (3, 3)}
    ref = F.conv2d(torch.from_numpy(xi), torch.from_numpy(wi), stride=2, padding=1).numpy()
    cols = F.unfold(torch.from_numpy(xi), kernel_size=(3, 3), padding=1, stride=2)
    mine = (torch.from_numpy(wi.reshape(4, -1)) @ cols).reshape(ref.shape).numpy()
    expect("im2col in ONNX (cin, kh, kw) order reproduces the convolution", np.allclose(ref, mine, atol=1e-4))
    h_full, _ = hessian(xi, attrs)
    expect("the Hessian is symmetric positive semidefinite", np.allclose(h_full, h_full.T)
           and np.linalg.eigvalsh(h_full).min() > -1e-9)
    ok_ref = ok_diag = True
    off_rtn = off_order = 0
    for case in range(30):
        rows, n_cols = int(rng.integers(1, 5)), int(rng.integers(2, 40))
        xs = rng.standard_normal((n_cols, 200)) @ rng.standard_normal((200, 200)) * 0.1
        h = 2 * xs @ xs.T / 200
        if case % 3 == 0:
            h[1, :] = h[:, 1] = 0          # a dead column
        w = rng.standard_normal((rows, n_cols))
        s = rng.uniform(0.05, 0.4, rows)
        q_ref = {act: gptq_reference(w, h, s, *GRID, GP["damp"], act) for act in (True, False)}
        for act in (True, False):
            for block in (1, 3, 128):
                ok_ref &= np.array_equal(gptq_quantize(w, h, s, *GRID, GP["damp"], block, act)[0], q_ref[act])
        off_rtn += not np.array_equal(q_ref[True], np.clip(np.rint(w / s[:, None]), *GRID))
        off_order += not np.array_equal(q_ref[True], q_ref[False])
        d = np.diag(np.abs(np.diag(h)) + 0.1)
        ok_diag &= np.array_equal(gptq_quantize(w, d, s, *GRID)[0], np.clip(np.rint(w / s[:, None]), *GRID))
    expect("blocked Cholesky GPTQ (blocks 1, 3, 128; act-order on and off) == an independent unblocked OBQ "
           "reference (a re-solve per column, its own dead-column, order and damping code) on 30 seeded cases",
           ok_ref)
    expect(f"the reference is not RTN: it leaves round-to-nearest in {off_rtn}/30 cases, and act-order changes "
           f"its integers in {off_order}/30", off_rtn > 0 and off_order > 0)
    expect("with a diagonal Hessian GPTQ is round-to-nearest", ok_diag)
    import itertools
    grid_vals = np.arange(GRID[0], GRID[1] + 1, dtype=np.float64)
    combos = np.array(list(itertools.product(grid_vals, repeat=4)))
    g_sum, r_sum, hits, le = 0.0, 0.0, 0, 0
    for case in range(40):
        xs = rng.standard_normal((4, 50)) @ rng.standard_normal((50, 50)) * 0.2
        h = 2 * xs @ xs.T / 50
        w = rng.standard_normal((1, 4)) * 0.6
        s = np.array([rng.uniform(0.08, 0.2)])
        e = w[0] - combos * s[0]
        best = float(np.einsum("ij,jk,ik->i", e, h, e).min())
        qg, _ = gptq_quantize(w, h, s, *GRID)
        lg = recon_loss(w, qg, s, h)
        lr = recon_loss(w, np.clip(np.rint(w / s[:, None]), *GRID), s, h)
        g_sum, r_sum = g_sum + lg, r_sum + lr
        hits += abs(lg - best) <= 1e-12
        le += lg <= lr + 1e-12
    expect(f"GPTQ's loss <= RTN's in each of the 40 cases ({le}/40)", le == 40)
    expect(f"GPTQ reaches the brute-force optimum (all 15^4 grid points) in {hits}/40 cases, floor "
           f"{GPTQ_HITS_FLOOR}/40", hits >= GPTQ_HITS_FLOOR)
    expect(f"GPTQ's mean loss <= RTN's over the 40 cases ({g_sum / 40:.5f} vs {r_sum / 40:.5f})", g_sum <= r_sum)

    # F
    wf = (rng.standard_normal((6, 4, 3, 3)) * 0.3).astype(np.float32)
    wf[2] = 0
    qf = f_quant(wf)
    first_min = True
    for c in (0, 1, 3):
        x = wf[c]
        m = np.float32(np.abs(x).max())
        errs = [float(np.sum((x - np.float32(a * m / np.float32(7)) * np.clip(np.rint(x / np.float32(
            a * m / np.float32(7))), *GRID).astype(np.float32)) ** 2, dtype=np.float32)) for a in F_ALPHAS]
        first_min &= qf["groups"][c]["alpha"] == float(F_ALPHAS[int(np.argmin(errs))])
    expect("F takes the first minimum over alpha 0.50..1.00, and w_hat == q4 * s",
           first_min and np.array_equal(qf["w_hat"], qf["q4"].astype(np.float32) * qf["scales"].reshape(-1, 1, 1, 1))
           and not np.any(qf["q4"][2]))

    # MX: the stopping rule
    order = ["a", "b", "c", "d", "e"]
    sizes = {"a": 400, "b": 10, "c": 30, "d": 5, "e": 555}
    rel = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2, "e": 0.5}
    sel = mx_select(order, sizes, rel, 0.05)
    expect("MX takes the most sensitive first, ties in Conv order, and stops at the first that overflows "
           "(b, c kept; e would overflow; d is not taken after the stop)",
           sel["w8"] == ["b", "c"] and sel["ranked"][:3] == ["b", "c", "e"] and sel["share"] == 0.04)

    # the synthetic two-Conv model: every form through the builder and the checks
    import onnxruntime as ort
    model, qm, fm = synth(rng)
    with tempfile_dir() as tmp:
        tmp = Path(tmp)
        imgs = [rng.uniform(0, 0.95, (1, 3, 12, 12)).astype(np.float32) for _ in range(6)]

        def run(path, x):
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.intra_op_num_threads = 2
            return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
        for form in ("e1", "e2", "u", "f"):
            rows = rtn_rows(model, form)
            p = tmp / f"rtn_{form}.onnx"
            build(model, rows).save(p)
            fl = check_arm(p, model, rows, {}, f"rtn_{form}", {})
            expect(f"RTN {form}: C5 passes (gate D's check_emitted and this tool's)", not fl)
        rows = rtn_rows(model, "e1")
        expect(f"RTN E1 of the synthetic model has form-b Convs ({sum(r['form'] == 'b' for r in rows)}/2)",
               any(r["form"] == "b" for r in rows))
        pa, pb = tmp / "e1_a.onnx", tmp / "e1_b.onnx"
        build(model, [dict(r, form="a") for r in rows]).save(pa)
        build(model, rows).save(pb)
        expect("form a == form b, bit for bit, on 6 inputs", all(np.array_equal(run(pa, x), run(pb, x)) for x in imgs))
        expect("the negative control: RTN E1 differs from the shipped model",
               any(not np.array_equal(run(pb, x), ort.InferenceSession(qm.SerializeToString(), providers=[
                   "CPUExecutionProvider"]).run(None, {"images": x})[0]) for x in imgs))

        # sensitivity's y_f statistics (what BC and VC read) == an independent pass; the files round-trip
        fref = float_reference(model, imgs)
        sets, fstats = sensitivity(model, imgs, lambda *a: None)
        expect("sensitivity's y_f statistics == a separate float pass, bit for bit",
               all(np.array_equal(fstats[n][k], fref[n][k]) for n in model.order for k in ("sum", "sumsq", "count")))
        mx_path, _ = save_captures(model, sets, fstats, tmp / "cap")
        quiet = lambda *a: None  # noqa: E731  (the temp dir's name is random; the log stays reproducible)
        back = load_fref(model, tmp / "cap", log=quiet)
        expect("the y_f file and the MX file read back unchanged",
               all(np.array_equal(back[n][k], fref[n][k]) for n in model.order for k in ("sum", "sumsq", "count"))
               and all(load_mx(model, tmp / "cap", f, require_pin=False, log=quiet) == sets[f]["w8"] for f in sets))
        refused = False
        try:
            load_mx(model, tmp / "cap", "e1", require_pin=True, log=quiet)
        except SystemExit:
            refused = True
        expect("tier 2 refuses a capture file that no prereg log pins", refused)
        kept = ", ".join(f"{f} {len(v['w8'])}" for f, v in sets.items())
        expect(f"MX: the budget rule keeps W8 at most {100 * MX_BUDGET:.0f}% of the weights (W8 Convs: {kept})",
               all(v["share"] <= MX_BUDGET for v in sets.values()))

        # MX's ranking: ||conv(W_f, x) - conv(W4, x)||^2 / ||conv(W_f, x)||^2, both on B8's captured x, float bias
        model_g, _, _ = synth(np.random.default_rng(20260926), grid2=True)
        sets_g, _ = sensitivity(model_g, imgs, lambda *a: None)
        c1n, c2n = model_g.order
        expect("MX: a Conv whose float weights sit on the W4 grid scores exactly 0 in E1, E2 and U, the other Conv "
               "above 0 (the two terms share B8's input and the float bias)",
               all(v["rel"][c2n] == 0.0 and v["rel"][c1n] > 0 for v in sets_g.values()))

        def ratio(x, w, w4, b, attrs):
            yf = conv_out(x, w, b, attrs).astype(np.float64)
            yq = conv_out(x, w4, b, attrs).astype(np.float64)
            return float(((yf - yq) ** 2).sum() / (yf * yf).sum())
        same_b8, gap_f = True, 0.0
        for m_, sets_ in ((model, sets), (model_g, sets_g)):
            for n in m_.order:
                cw = m_.by_name[n]
                x8 = capture(m_.shipped, "images", cw["conv"].input[0], imgs)
                fin = m_.fconv[n].input[0]
                xf = np.concatenate(imgs) if fin == "images" else capture(m_.gf, "images", fin, imgs)
                for f in ("e1", "e2", "u"):
                    w4 = next(r for r in rtn_rows(m_, f) if r["cw"]["conv"].name == n)["q"]["w_hat"]
                    got, r8 = sets_[f]["rel"][n], ratio(x8, cw["w"], w4, cw["bias_f"], cw["attrs"])
                    same_b8 &= math.isclose(got, r8, rel_tol=1e-6, abs_tol=1e-15)
                    if m_ is model:
                        rf = ratio(xf, cw["w"], w4, cw["bias_f"], cw["attrs"])
                        gap_f = max(gap_f, abs(got - rf) / rf)
        expect("MX: each rel == the same ratio recomputed on B8's captured input (both models, every form)", same_b8)
        expect(f"MX: on the float network's own input the ratio differs (largest relative gap {gap_f:.2e}), so the "
               f"input is B8's", gap_f > 1e-4)

        # BC, VC, GPTQ end to end: the same sequential loop, captures and builder as the real models
        for method in ("bc", "vc", "gptq"):
            for form in ("e1", "u"):
                rows = rtn_rows(model, form)
                biases, record, rescaled = {}, {}, {}
                step = {"bc": lambda: step_bc(fref, biases, record),
                        "vc": lambda: step_vc(fref, biases, record, rescaled, form),
                        "gptq": lambda: step_gptq(record)}[method]()
                sequential(model, rows, biases, imgs, step, lambda *a: None)
                p = tmp / f"{method}_{form}.onnx"
                build(model, rows, biases).save(p)
                fl = check_arm(p, model, rows, biases, f"{method}_{form}", rescaled)
                expect(f"{method.upper()} {form}: C5 passes ({len(biases)} int32 biases)", not fl)
                side = json.loads(json.dumps(sidecar(model, f"{method}_{form}", {
                    "rows": rows, "biases": biases, "record": record, "rescaled": rescaled, "extra": {}}, p, fl, 0.0)))
                expect(f"{method.upper()} {form}: the sidecar serializes, with gate D's fields and the method's",
                       len(side["convs"]) == 2 and all(method in c and "sqnr_w4_db" in c and "k_hist" in c
                                                       for c in side["convs"])
                       and side["summary"]["int32_bias_convs"] == len(biases))
                p2 = tmp / f"{method}_{form}_2.onnx"
                rows2 = rtn_rows(model, form)
                b2, r2, s2 = {}, {}, {}
                step2 = {"bc": lambda: step_bc(fref, b2, r2), "vc": lambda: step_vc(fref, b2, r2, s2, form),
                         "gptq": lambda: step_gptq(r2)}[method]()
                sequential(model, rows2, b2, imgs, step2, lambda *a: None)
                build(model, rows2, b2).save(p2)
                expect(f"{method.upper()} {form} twice gives the same file (C7)", vv.sha_file(p) == vv.sha_file(p2))
                if method == "bc":
                    worst = max(r["bc"]["mean_err_after"] / np.max(biases[n]["scale"]) for n, r in record.items())
                    expect(f"BC {form}: every channel mean within half an int32 step of y_f's (worst "
                           f"{worst:.3f} step)", worst <= 0.5 + 1e-6)
                    run(p, imgs[0])
                if method == "vc":
                    expect(f"VC {form}: a_c in (0, inf) and the file runs", all(
                        0 < r["vc"]["a_min"] <= r["vc"]["a_max"] < np.inf for r in record.values())
                        and run(p, imgs[0]).shape == (1, 6, 12, 12))

        # VC's identity before re-quantization: std(conv(x, a * w)) == std(y_f)
        cw = model.cws[0]
        xq = capture(build(model, rtn_rows(model, "u")), "images", cw["conv"].input[0], imgs)
        s, ss, n = conv_sums(xq, cw["w"], np.zeros(cw["w"].shape[0], np.float32), cw["attrs"])
        _, std_w = mean_std(s, ss, n)
        target = std_w * rng.uniform(0.5, 2.0, std_w.shape)
        a = np.where(std_w > 0, target / np.where(std_w > 0, std_w, 1), 1.0)
        s2, ss2, n2 = conv_sums(xq, (a.reshape(-1, 1, 1, 1) * cw["w"]).astype(np.float32),
                                np.zeros(cw["w"].shape[0], np.float32), cw["attrs"])
        _, std_a = mean_std(s2, ss2, n2)
        live = std_w > 0
        expect("VC's rescale sets each channel's std to the target (before re-quantization)",
               np.allclose(std_a[live], target[live], rtol=1e-4))

        # AdaRound's override: the file's own scale on the int8 range reproduces the transcribed path exactly
        from quant.adaround import FastFinetuneConfig, WeightGrid, finetune
        cfg = FastFinetuneConfig(DataSize=len(imgs), NumIterations=40)
        g_def = Graph(copy.deepcopy(qm))
        finetune(Graph(copy.deepcopy(fm)), g_def, imgs, cfg, log=lambda *a: None)
        g_ovr = Graph(copy.deepcopy(qm))
        grids = {}
        for cw in model.cws:
            grids[cw["conv"].name] = WeightGrid(scales=np.full(cw["w"].shape[0], pos2scale(cw["pos8"]), np.float32),
                                                low=-128, high=127, q4=cw["stored"].copy())
        finetune(Graph(copy.deepcopy(fm)), g_ovr, imgs, FastFinetuneConfig(DataSize=len(imgs), NumIterations=40),
                 log=lambda *a: None, weight_grid=grids,
                 write_back=lambda name, q: g_ovr.set_initializer(model.by_name[name]["qname"], q, np.int8))
        expect("WeightGrid with the file's scale on [-128, 127] == the transcribed AdaRound, bit for bit",
               all(np.array_equal(g_def.initializer(cw["qname"]), g_ovr.initializer(cw["qname"])) for cw in model.cws)
               and any(not np.array_equal(g_def.initializer(cw["qname"]), cw["stored"]) for cw in model.cws))
        for form in ("e1", "u"):
            for keep in ((), (model.order[0],)):
                rows = rtn_rows(model, form, keep)
                rec = {}
                run_adaround(model, rows, imgs, Graph(copy.deepcopy(fm)), lambda *a: None, rec, iterations=40)
                p = tmp / f"ar_{form}_{len(keep)}.onnx"
                build(model, rows).save(p)
                fl = check_arm(p, model, rows, {}, f"ar_{form}", {})
                kept_ok = all(np.array_equal(Graph.load(p).initializer(model.by_name[k]["qname"]),
                                             model.by_name[k]["stored"]) for k in keep)
                expect(f"AdaRound W4 {form}{' with one Conv kept W8' if keep else ''}: C5 passes, q4 in [-7, 7], "
                       f"{len(rec)} layers optimised{', the kept Conv as shipped' if keep else ''}",
                       not fl and kept_ok and len(rec) == 2 - len(keep)
                       and all(r["form"] is None or np.abs(r["q"]["q4"]).max() <= 7 for r in rows))
        rows_a, rows_b = rtn_rows(model, "e1"), rtn_rows(model, "e1")
        torch.set_num_threads(1)
        run_adaround(model, rows_a, imgs, Graph(copy.deepcopy(fm)), lambda *a: None, {}, iterations=40)
        tf = thread_facts()
        expect(f"run_adaround puts torch at {TORCH_THREADS} threads before finetune (1 before the call, "
               f"{tf['torch_get_num_threads']} after); ORT's intra-op setting is recorded as "
               f"{tf['ort_intra_op_num_threads_setting']} (the default)",
               tf["torch_get_num_threads"] == TORCH_THREADS and tf["ort_intra_op_num_threads_setting"] == 0)
        run_adaround(model, rows_b, imgs, Graph(copy.deepcopy(fm)), lambda *a: None, {}, iterations=40)
        expect("AdaRound W4 twice gives the same integers (C7)",
               all(np.array_equal(a["q"]["q4"], b["q"]["q4"]) for a, b in zip(rows_a, rows_b)))
        rb = rows_from_file(model, tmp / "gptq_e1.onnx", "e1")
        expect("a tier-1 file reads back as rows that rebuild it (the tier-2 base)", len(rb) == 2)
    print(f"[selftest] {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return 0 if not fails else 1


def tempfile_dir():
    """Inside the git-ignored scratch/, so a printed path is repo-relative, never a profile path."""
    import tempfile
    base = ROOT / "scratch" / "w4v"
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="selftest-", dir=base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")

    def tag(v):
        if not re.fullmatch(r"[a-z0-9]+_\d{8}", v):
            raise argparse.ArgumentTypeError("the sidecar tag is <machine>_<YYYYMMDD>, as the logs are named")
        return v
    c = sub.add_parser("controls")
    c.add_argument("--model", required=True, choices=vv.MODELS)
    c.add_argument("--out-dir", default="scratch/w4v/arms")
    c.add_argument("--captures", default="scratch/w4v/captures")
    c.add_argument("--sidecar-dir", required=True)
    c.add_argument("--tag", required=True, type=tag)
    s = sub.add_parser("sensitivity")
    s.add_argument("--model", required=True, choices=vv.MODELS)
    s.add_argument("--captures", default="scratch/w4v/captures")
    e = sub.add_parser("emit")
    e.add_argument("--model", required=True, choices=vv.MODELS)
    e.add_argument("--method", required=True, choices=[m for _, m, _ in vv.MATRIX])
    e.add_argument("--form", required=True, choices=list(vv.FORM_TAG))
    e.add_argument("--tier", required=True, type=int, choices=(1, 2))
    e.add_argument("--out-dir", default="scratch/w4v/arms")
    e.add_argument("--captures", default="scratch/w4v/captures")
    e.add_argument("--sidecar-dir", required=True)
    e.add_argument("--tag", required=True, type=tag)
    e.add_argument("--twin", action="store_true", help="C3: also a forced form-a file, compared bit for bit")
    e.add_argument("--repeat", action="store_true", help="C7: run the method again and compare the file's sha")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit("run from the repo root: paths are printed and passed on repo-relative")
    return {"controls": cmd_controls, "sensitivity": cmd_sensitivity, "emit": cmd_emit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
