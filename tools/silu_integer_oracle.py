"""CPU only: an XINT8 model whose SiLU sigmoids are replaced by the integer function a core epilogue would compute, so
ONNX Runtime evaluates exactly the arithmetic of a candidate kernel design on COCO.

In a Quark XINT8 head-cut model every SiLU is  q1 -> DQ(s1) -> HardSigmoid -> Mul(k) -> Q(1/128) -> DQ -> Mul(x) -> Q(s2),
and the graph engine reproduces it with the HardSwish epilogue (graph_ir.fit_hardswish). The Mul(x) -> Q(s2) stage is
bit-exact with the core's y = rne((t * qh) >> YSH) for power-of-two scales. The table is indexed by q1 = t + 128
through Gather(table, Cast(q1, int64)):

  hardsigmoid  qh = the current epilogue at fit_hardswish's constants; the table feeds Q(1/128). Validates the surgery:
               outputs must equal the unmodified model's.
  sigmoid      qh = min(rne(128 * sigmoid(t * s1)), 127): a real sigmoid at the same 1/128 int8 quantization (what
               Quark's XINT8 keeps with ConvertSigmoidToHardSigmoid=False); the table feeds Q(1/128).
  silu         the exact quantized SiLU, y = rne(silu(t * s1) / s2), a 256-entry output table per layer; the table feeds
               Q(s2) and HardSigmoid, Mul(k), its Q/DQ and Mul(x) are removed.
  pl           a K-line piecewise-linear sigmoid mirroring the candidate core expression step for step:
                   u = |t|;  g = clip(min_i sat16(rne((u * A_i + B_i) >> S)), 0, 64);
                   y = sat16(rne(((t << 6) + u * g) >> YSH))
               which is t * qh with qh = 64 + sign(t) * g in [0, 128] (sigmoid 1.0 representable, no 127 clamp).
               A_i, B_i, S are fitted per (s1, s2) pair to the exact quantized SiLU (max output-LSB error, then mean).
               qh = 128 cannot pass Q(1/128), so that Q/DQ pair is removed and Mul(x) reads table[q1] = qh / 128; the
               float path x * qh / 128 -> Q(s2) is asserted equal to the integer y on all 256 inputs of every pair.

Always prints, per (s1, s2) pair, the output-LSB error (max/mean over the 256 inputs) of each form against the exact
quantized SiLU.

    python tools/silu_integer_oracle.py models/yolov8n_cut_xint8.onnx --mode pl --lines 4 --report-only
    python tools/silu_integer_oracle.py models/yolov8n_cut_xint8.onnx out.onnx --mode pl --lines 4
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ignite_xdna.compiler.engine_emulator import rne_shift, sat_i16  # noqa: E402
from ignite_xdna.compiler.graph_ir import _log2_exact, fit_hardswish  # noqa: E402

T = np.arange(-128, 128, dtype=np.int64)


def silu_exact_y(s1, s2):
    x = T * s1
    return np.clip(np.round(x / (1 + np.exp(-x)) / s2), -128, 127).astype(np.int64)


def ysh_of(s1, s2):
    return 7 - _log2_exact(s1) + _log2_exact(s2)


def y_from_qh(qh, s1, s2):
    return np.clip(sat_i16(rne_shift(T * qh, ysh_of(s1, s2))), -128, 127)


def qh_hardsigmoid(s1, s2, k):
    p = fit_hardswish(s1, s2, k).params
    h = np.clip(sat_i16(rne_shift(T * p.a1 + p.b1, p.s1)), 0, p.qmax)
    return np.minimum(sat_i16(rne_shift(h * p.k2, p.s2)), 127)


def qh_sigmoid(s1):
    return np.minimum(np.round(128 / (1 + np.exp(-T * s1))).astype(np.int64), 127)


def g_pl(lines, S):
    u = np.abs(T)
    g = np.minimum.reduce([sat_i16(rne_shift(u * a + b, S)) for a, b in lines])
    return np.clip(g, 0, 64)


def y_pl(lines, S, s1, s2):
    y = sat_i16(rne_shift((T << 6) + np.abs(T) * g_pl(lines, S), ysh_of(s1, s2)))
    return np.clip(y, -128, 127)


def qh_pl(lines, S):
    g = g_pl(lines, S)
    return np.where(T >= 0, 64 + g, 64 - g)


def score(y, ye):
    d = np.abs(y - ye)
    return int(d.max()), float(d.mean())


def float_pl(K, R):
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


def fit_pl(s1, s2, K):
    """Integer lines (A, B) and shift S for one (s1, s2) pair, with (max, mean) output-LSB error."""
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


def silu_sites(m):
    """(HardSigmoid, Mul(k), DQ(q1), s1, s2, k, Q(1/128), DQ, Mul(x)) for every SiLU of a Quark XINT8 graph."""
    g = m.graph
    const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--mode", choices=("hardsigmoid", "sigmoid", "silu", "pl"), required=True)
    ap.add_argument("--lines", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    sites = silu_sites(m)
    pairs = Counter((s1, s2, k) for _, _, _, s1, s2, k, _, _, _ in sites)
    print(f"{Path(args.src).name}: {len(sites)} SiLU sites, {len(pairs)} (s1, s2) pairs; "
          f"output-LSB error against the exact quantized SiLU, max/mean over the 256 inputs")
    cache = {}
    for (s1, s2, k), cnt in sorted(pairs.items()):
        ye = silu_exact_y(s1, s2)
        hs_sc = score(y_from_qh(qh_hardsigmoid(s1, s2, k), s1, s2), ye)
        sg_sc = score(y_from_qh(qh_sigmoid(s1), s1, s2), ye)
        line = f"  s1 1/{round(1 / s1)} s2 1/{round(1 / s2)} x{cnt}:  HardSigmoid {hs_sc[0]}/{hs_sc[1]:.3f}" \
               f"  sigmoid@1/128 {sg_sc[0]}/{sg_sc[1]:.3f}"
        for K in sorted({2, 3, 4, 5, args.lines}):
            lines, S, sc = fit_pl(s1, s2, K)
            cache[(s1, s2, K)] = (lines, S)
            line += f"  PL{K} {sc[0]}/{sc[1]:.3f}"
        print(line)
        if args.mode == "pl":
            lines, S = cache[(s1, s2, args.lines)]
            print(f"      PL{args.lines} S={S} lines (A, B) = {lines}")
    if args.report_only:
        return

    new_inits, new_nodes, drop = [], [], set()
    for idx, (hs, mk, dq, s1, s2, k, qk, dqk, mx) in enumerate(sites):
        if args.mode == "silu":
            tab = (silu_exact_y(s1, s2).astype(np.float32) * np.float32(s2)).astype(np.float32)
            tname = f"silutab_{idx}"
            new_inits.append(numpy_helper.from_array(tab, tname))
            ci = f"{tname}_idx"
            new_nodes.append((hs.name, [helper.make_node("Cast", [dq.input[0]], [ci], to=onnx.TensorProto.INT64, name=ci),
                                        helper.make_node("Gather", [tname, ci], [mx.output[0]], axis=0,
                                                         name=f"{tname}_gather")]))
            drop.update({hs.name, mk.name, qk.name, dqk.name, mx.name})
            continue
        if args.mode == "hardsigmoid":
            qh = qh_hardsigmoid(s1, s2, k)
        elif args.mode == "sigmoid":
            qh = qh_sigmoid(s1)
        else:
            lines, S = cache[(s1, s2, args.lines)]
            qh = qh_pl(lines, S)
            x = (T.astype(np.float32) * np.float32(s1)).astype(np.float32)
            yf = np.clip(np.round(((x * (qh.astype(np.float32) / np.float32(128))).astype(np.float32)).astype(np.float64)
                                  / s2), -128, 127).astype(np.int64)
            assert np.array_equal(yf, y_pl(lines, S, s1, s2)), (s1, s2)
        tab = (qh.astype(np.float32) / np.float32(128)).astype(np.float32)
        tname = f"sigtab_{idx}"
        new_inits.append(numpy_helper.from_array(tab, tname))
        ci = f"{tname}_idx"
        out = dqk.output[0] if args.mode == "pl" else mk.output[0]
        new_nodes.append((hs.name, [helper.make_node("Cast", [dq.input[0]], [ci], to=onnx.TensorProto.INT64, name=ci),
                                    helper.make_node("Gather", [tname, ci], [out], axis=0, name=f"{tname}_gather")]))
        drop.update({hs.name, mk.name})
        if args.mode == "pl":
            drop.update({qk.name, dqk.name})
    repl = dict(new_nodes)
    nodes = []
    for n in g.node:
        if n.name in repl:
            nodes += repl[n.name]
        elif n.name not in drop:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_inits)
    onnx.save(m, args.dst)
    print(f"mode {args.mode}{f' K={args.lines}' if args.mode == 'pl' else ''}: replaced {len(sites)} sites -> "
          f"{Path(args.dst).name}")


if __name__ == "__main__":
    main()
