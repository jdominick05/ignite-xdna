"""CPU only: an FP32 model whose SiLU activations use another sigmoid, to isolate what an approximation costs.

Every Sigmoid whose output feeds a Mul together with the Sigmoid's own input (x * sigmoid(x), SiLU) is replaced;
other Sigmoids are left alone. Weights stay FP32; nothing is quantized.

  --form hardsigmoid   HardSigmoid(alpha=1/6, beta=0.5) then Mul by HARD_SIGMOID_SCALE = (2731/16384) / (1/6),
                       exactly what quark.onnx's SimulateDPU writes for XINT8 (convert_sigmoid_to_hard_sigmoid, then
                       convert_hard_sigmoid_to_dpu_version). This is the form every graph-engine container computes.
  --form pl --lines K  a K-line piecewise-linear sigmoid: 0.5 + sign(x) * min(line_1(|x|), ..., line_K(|x|), 0.5),
                       the lines K chords of sigmoid(u) - 0.5 raised by the half-error that makes the fit minimax,
                       breakpoints from a coordinate search on [0, 12] (ONNX Abs, Sign, Mul, Add, Min).

    python tools/silu_fp32_forms.py models/yolov8n_cut.onnx out.onnx --form hardsigmoid
    python tools/silu_fp32_forms.py models/yolov8n_cut.onnx out.onnx --form pl --lines 3
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper

HARD_SIGMOID_SCALE = (2731.0 / 16384.0) / (1.0 / 6.0)


def fit_pl(K):
    u = np.linspace(0, 12, 24001)
    g = 1 / (1 + np.exp(-u)) - 0.5

    def build(bp):
        lines = []
        for i in range(K):
            u0, u1 = bp[i], bp[i + 1]
            g0, g1 = 1 / (1 + np.exp(-u0)) - 0.5, 1 / (1 + np.exp(-u1)) - 0.5
            a = (g1 - g0) / (u1 - u0)
            lines.append((a, g0 - a * u0))
        approx = np.minimum.reduce([a * u + b for a, b in lines] + [np.full_like(u, 0.5)])
        err = g - approx   # >= 0 inside the chords of a concave function
        shift = (err.max() + err.min()) / 2
        lines = [(a, b + shift) for a, b in lines]
        approx = np.minimum.reduce([a * u + b for a, b in lines] + [np.full_like(u, 0.5)])
        return lines, float(np.abs(g - approx).max())

    bp = list(np.linspace(0, 6, K + 1))
    best_lines, best = build(bp)
    for _ in range(60):
        improved = False
        for i in range(1, K + 1):
            for step in (0.2, 0.05, 0.01):
                for d in (-step, step):
                    cand = list(bp)
                    cand[i] = cand[i] + d
                    if not all(cand[j] < cand[j + 1] for j in range(K)) or cand[-1] > 12:
                        continue
                    lines, e = build(cand)
                    if e < best - 1e-9:
                        bp, best, best_lines, improved = cand, e, lines, True
        if not improved:
            break
    hs_err = float(np.abs(g - np.minimum(u / 6, 0.5)).max())
    print(f"{K} lines: breakpoints {[round(b, 3) for b in bp]}, max |sigmoid error| {best:.4f} "
          f"(HardSigmoid form {hs_err:.4f})")
    for a, b in best_lines:
        print(f"   line {a:.5f} * u + {b:.5f}")
    return best_lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--form", choices=("hardsigmoid", "pl"), required=True)
    ap.add_argument("--lines", type=int, default=3)
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    if args.form == "hardsigmoid":
        g.initializer.append(numpy_helper.from_array(np.array(HARD_SIGMOID_SCALE, np.float32), "silu_hardsigmoid_dpu_scale"))
    else:
        lines = fit_pl(args.lines)
        g.initializer.append(numpy_helper.from_array(np.array(0.5, np.float32), "plsig_half"))
        for i, (a, b) in enumerate(lines):
            g.initializer.extend([numpy_helper.from_array(np.array(a, np.float32), f"plsig_a{i}"),
                                  numpy_helper.from_array(np.array(b, np.float32), f"plsig_b{i}")])
    new_nodes, swapped, left = [], 0, 0
    for n in g.node:
        silu = n.op_type == "Sigmoid" and any(c.op_type == "Mul" and n.input[0] in c.input
                                              for c in consumers.get(n.output[0], []))
        if not silu:
            left += n.op_type == "Sigmoid"
            new_nodes.append(n)
            continue
        x = n.input[0]
        if args.form == "hardsigmoid":
            mid = n.output[0] + "_hard"
            new_nodes += [helper.make_node("HardSigmoid", [x], [mid], name=n.name + "_hard", alpha=1.0 / 6.0, beta=0.5),
                          helper.make_node("Mul", [mid, "silu_hardsigmoid_dpu_scale"], [n.output[0]],
                                           name=n.name + "_dpu_scale")]
        else:
            p = n.output[0] + "_pl"
            new_nodes += [helper.make_node("Abs", [x], [p + "_abs"]), helper.make_node("Sign", [x], [p + "_sign"])]
            terms = []
            for i in range(args.lines):
                new_nodes += [helper.make_node("Mul", [p + "_abs", f"plsig_a{i}"], [f"{p}_m{i}"]),
                              helper.make_node("Add", [f"{p}_m{i}", f"plsig_b{i}"], [f"{p}_l{i}"])]
                terms.append(f"{p}_l{i}")
            new_nodes += [helper.make_node("Min", terms + ["plsig_half"], [p + "_g"]),
                          helper.make_node("Mul", [p + "_g", p + "_sign"], [p + "_sg"]),
                          helper.make_node("Add", [p + "_sg", "plsig_half"], [n.output[0]])]
        swapped += 1
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, args.dst)
    form = "HardSigmoid(1/6, 0.5) * %.9f" % HARD_SIGMOID_SCALE if args.form == "hardsigmoid" else f"{args.lines}-line sigmoid"
    print(f"{args.src}: {swapped} SiLU Sigmoids -> {form}; {left} other Sigmoids left -> {args.dst}")


if __name__ == "__main__":
    main()
