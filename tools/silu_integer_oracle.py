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
from ignite_xdna.compiler.graph_ir import fit_hardswish  # noqa: E402
from ignite_xdna.compiler.silu_sigmoid import (T, fit_pl, qh_pl, score, silu_exact_y,  # noqa: E402
                                               silu_sites, y_pl, ysh_of)


def y_from_qh(qh, s1, s2):
    return np.clip(sat_i16(rne_shift(T * qh, ysh_of(s1, s2))), -128, 127)


def qh_hardsigmoid(s1, s2, k):
    p = fit_hardswish(s1, s2, k).params
    h = np.clip(sat_i16(rne_shift(T * p.a1 + p.b1, p.s1)), 0, p.qmax)
    return np.minimum(sat_i16(rne_shift(h * p.k2, p.s2)), 127)


def qh_sigmoid(s1):
    return np.minimum(np.round(128 / (1 + np.exp(-T * s1))).astype(np.int64), 127)


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
