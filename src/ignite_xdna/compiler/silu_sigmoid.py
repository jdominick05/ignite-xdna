"""SiLU through a four-line piecewise-linear sigmoid instead of Quark's HardSigmoid form.

Quark's XINT8 writes every SiLU as  q1 -> DQ(s1) -> HardSigmoid -> Mul(k) -> Q(1/128) -> DQ -> Mul(x) -> Q(s2), and
the engine reproduces that chain exactly (``graph_ir.fit_hardswish``). On the first 500 COCO val2017 images the form
is most of the XINT8 accuracy loss of YOLOv8n, YOLOv8s and YOLOv8n-pose (docs/BENCHMARKS.md). With ``silu_sigmoid``
the compiler instead gives each such layer the core's sigmoid epilogue (``engine_emulator.sigmoid_epilogue``,
F_SIGMOID), fitted per layer to the exact quantized SiLU:

    u = |t|;  g = clip(min_i rne((u * A_i + B_i) >> S), 0, 64);  y = rne(((t << 6) + u * g) >> YSH)

The network then no longer computes Quark's graph, so the exactness reference becomes ``reference_model``: the same
QDQ model with each SiLU's HardSigmoid, Mul(k) and their Q/DQ replaced by a ``Gather`` over the uint8 conv output of
the table qh / 128 (qh = 64 + sign(t) * g), read by the SiLU's own Mul(x) -> Q(s2). ONNX Runtime's float path through
that model equals the integer epilogue on all 256 inputs of every layer (asserted when the model is built).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ignite_xdna.compiler.engine_emulator import SIGMOID_LINES, SigmoidParams, rne_shift, sat_i16, sigmoid_epilogue

T = np.arange(-128, 128, dtype=np.int64)


def _log2_exact(scale: float) -> int:
    e = math.log2(scale)
    if abs(e - round(e)) > 1e-9:
        raise ValueError(f"scale {scale} is not a power of two")
    return int(round(e))


def silu_exact_y(s1: float, s2: float) -> np.ndarray:
    """The exact quantized SiLU for t = -128..127: clip(round(silu(t * s1) / s2), -128, 127)."""
    x = T * s1
    return np.clip(np.round(x / (1 + np.exp(-x)) / s2), -128, 127).astype(np.int64)


def ysh_of(s1: float, s2: float) -> int:
    return 7 - _log2_exact(s1) + _log2_exact(s2)


def g_pl(lines, S) -> np.ndarray:
    u = np.abs(T)
    g = np.minimum.reduce([sat_i16(rne_shift(u * a + b, S)) for a, b in lines])
    return np.clip(g, 0, 64)


def y_pl(lines, S, s1: float, s2: float) -> np.ndarray:
    y = sat_i16(rne_shift((T << 6) + np.abs(T) * g_pl(lines, S), ysh_of(s1, s2)))
    return np.clip(y, -128, 127)


def qh_pl(lines, S) -> np.ndarray:
    g = g_pl(lines, S)
    return np.where(T >= 0, 64 + g, 64 - g)


def score(y: np.ndarray, ye: np.ndarray) -> Tuple[int, float]:
    d = np.abs(y - ye)
    return int(d.max()), float(d.mean())


def float_pl(K: int, R: float):
    """Minimax K-chord fit of sigmoid(u) - 0.5 on [0, R]; returns [(a, b)] in u units."""
    u = np.linspace(0, R, 4001)
    g = 1 / (1 + np.exp(-u)) - 0.5

    def build(bp):
        lines = []
        for i in range(K):
            u0, u1 = bp[i], bp[i + 1]
            g0, g1 = 1 / (1 + np.exp(-u0)) - 0.5, 1 / (1 + np.exp(-u1)) - 0.5
            a = (g1 - g0) / (u1 - u0)
            lines.append((a, g0 - a * u0))
        ap = np.minimum.reduce([a * u + b for a, b in lines] + [np.full_like(u, 0.5)])
        err = g - ap
        sh = (err.max() + err.min()) / 2
        lines = [(a, b + sh) for a, b in lines]
        ap = np.minimum.reduce([a * u + b for a, b in lines] + [np.full_like(u, 0.5)])
        return lines, float(np.abs(g - ap).max())

    top = min(R, 8.0)
    bp = list(np.linspace(0, top, K + 1))
    best_lines, best = build(bp)
    for _ in range(60):
        improved = False
        for i in range(1, K + 1):
            for step in (0.2, 0.05, 0.01):
                for d in (-step, step):
                    cand = list(bp)
                    cand[i] += d
                    if not all(cand[j] < cand[j + 1] for j in range(K)) or cand[-1] > R:
                        continue
                    lines, e = build(cand)
                    if e < best - 1e-9:
                        bp, best, best_lines, improved = cand, e, lines, True
        if not improved:
            break
    return best_lines


def fit_pl(s1: float, s2: float, K: int):
    """Integer lines [[A, B], ...] and shift S for one (s1, s2) pair, with (max, mean) output-LSB error."""
    ye = silu_exact_y(s1, s2)
    fl = float_pl(K, 128 * s1)
    S = max(s for s in range(6, 15) if all(abs(round(128 * a * s1 * (1 << s))) < 1 << 15 for a, _ in fl))
    lines = [[int(round(128 * a * s1 * (1 << S))), int(round(128 * b * (1 << S)))] for a, b in fl]
    best = score(y_pl(lines, S, s1, s2), ye)
    steps = [1 << e for e in range(S, -1, -1)]
    improved = True
    while improved:
        improved = False
        for i in range(K):
            for j in range(2):
                for st in steps:
                    for sgn in (-1, 1):
                        cand = [list(l) for l in lines]
                        cand[i][j] += sgn * st
                        if j == 0 and not 0 < cand[i][j] < 1 << 15:
                            continue
                        sc = score(y_pl(cand, S, s1, s2), ye)
                        if sc < best:
                            lines, best, improved = cand, sc, True
    return lines, S, best


@dataclass
class SigmoidFit:
    params: SigmoidParams
    table: np.ndarray        # uint8[256], indexed by the linear conv output q1: the epilogue as the core computes it
    max_error: int           # output LSBs against the exact quantized SiLU
    mean_error: float


_FITS: Dict[Tuple[float, float], SigmoidFit] = {}


def fit_sigmoid(s1: float, s2: float) -> SigmoidFit:
    """The core's four-line sigmoid epilogue for a SiLU with input scale ``s1`` and output scale ``s2``."""
    key = (float(s1), float(s2))
    if key not in _FITS:
        ysh = ysh_of(s1, s2)
        if ysh < 0:
            raise ValueError(f"SiLU output scale {s2} finer than the sigmoid epilogue supports for input scale {s1}")
        lines, S, (max_err, mean_err) = fit_pl(s1, s2, SIGMOID_LINES)
        params = SigmoidParams(tuple((int(a), int(b)) for a, b in lines), int(S), int(ysh))
        table = sigmoid_epilogue(np.arange(256, dtype=np.int64), params)
        if not np.array_equal(table.astype(np.int64) - 128, y_pl(lines, S, s1, s2)):
            raise AssertionError(f"sigmoid epilogue and its fit disagree for s1={s1} s2={s2}")
        _FITS[key] = SigmoidFit(params, table, max_err, mean_err)
    return _FITS[key]


def silu_sites(model) -> List[tuple]:
    """(HardSigmoid, Mul(k), DQ(q1), s1, s2, k, Q(1/128), DQ, Mul(x)) for every SiLU of a Quark XINT8 graph."""
    from onnx import numpy_helper
    g = model.graph
    const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    prod = {o: n for n in g.node for o in n.output}
    cons: Dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    sites = []
    for hs in [n for n in g.node if n.op_type == "HardSigmoid"]:
        dq = prod[hs.input[0]]
        assert dq.op_type == "DequantizeLinear", dq.op_type
        s1 = float(const[dq.input[1]])
        assert int(const[dq.input[2]]) == 128
        mk = cons[hs.output[0]]
        assert len(mk) == 1 and mk[0].op_type == "Mul", [c.op_type for c in mk]
        mk = mk[0]
        k = float(np.asarray(const[mk.input[1]]).flatten()[0])
        qk = cons[mk.output[0]]
        assert len(qk) == 1 and qk[0].op_type == "QuantizeLinear"
        assert abs(float(const[qk[0].input[1]]) - 1 / 128) < 1e-12 and int(const[qk[0].input[2]]) == 128
        assert len(cons[qk[0].output[0]]) == 1
        dqk = cons[qk[0].output[0]][0]
        assert dqk.op_type == "DequantizeLinear"
        mx = [c for c in cons[dqk.output[0]] if c.op_type == "Mul"][0]
        qy = [c for c in cons[mx.output[0]] if c.op_type == "QuantizeLinear"][0]
        s2 = float(const[qy.input[1]])
        sites.append((hs, mk, dq, s1, s2, k, qk[0], dqk, mx))
    return sites


def reference_model(model):
    """A copy of the QDQ ``model`` whose every SiLU computes the core's sigmoid epilogue in ONNX Runtime."""
    import copy

    import onnx
    from onnx import helper, numpy_helper
    m = copy.deepcopy(model)
    g = m.graph
    sites = silu_sites(m)
    new_inits, repl, drop = [], {}, set()
    for idx, (hs, mk, dq, s1, s2, k, qk, dqk, mx) in enumerate(sites):
        fit = fit_sigmoid(s1, s2)
        lines, S = [list(l) for l in fit.params.lines], fit.params.s
        qh = qh_pl(lines, S)
        x = (T.astype(np.float32) * np.float32(s1)).astype(np.float32)
        yf = np.clip(np.round(((x * (qh.astype(np.float32) / np.float32(128))).astype(np.float32)).astype(np.float64)
                              / s2), -128, 127).astype(np.int64)
        if not np.array_equal(yf, y_pl(lines, S, s1, s2)):
            raise AssertionError(f"{hs.name}: ONNX Runtime's float path differs from the integer epilogue")
        tab = (qh.astype(np.float32) / np.float32(128)).astype(np.float32)
        tname = f"sigtab_{idx}"
        new_inits.append(numpy_helper.from_array(tab, tname))
        ci = f"{tname}_idx"
        repl[hs.name] = [helper.make_node("Cast", [dq.input[0]], [ci], to=onnx.TensorProto.INT64, name=ci),
                         helper.make_node("Gather", [tname, ci], [dqk.output[0]], axis=0, name=f"{tname}_gather")]
        drop.update({hs.name, mk.name, qk.name, dqk.name})
    nodes = []
    for n in g.node:
        if n.name in repl:
            nodes += repl[n.name]
        elif n.name not in drop:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_inits)
    return m
