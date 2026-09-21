#!/usr/bin/env python3
"""Swap one activation op for another in a float ONNX graph, changing nothing else.

    python tools/swap_activation.py IN.onnx OUT.onnx --from Sigmoid --to HardSigmoid --attr alpha=0.16666667

This exists to separate two costs that quantization confounds. A quantizer may both narrow the
arithmetic to int8 AND substitute a cheaper activation form - Quark replaces SiLU's sigmoid with
HardSigmoid - and a mAP drop measured against the original float model charges the whole loss to
"quantization" without saying which half did it. Running the FLOAT graph with only the activation
swapped isolates the activation's share, with no calibration, no int8 and no device involved.

That is the shape scripts/quant-cle-probe.sh uses for CLE: equalize everything except the one
suspected factor, so the verdict does not rest on the number that raised the question.

Nothing here is model-specific: it matches by op type and rewrites in place, so it works on any
graph. It does NOT check that the substitution is meaningful for the model - ``--from Sigmoid``
on a graph whose sigmoids are attention weights rather than SiLU would happily produce nonsense,
so read the report it prints.
"""
import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper


def parse_attr(items):
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"--attr wants name=value, got {it!r}")
        out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--from", dest="src", required=True, help="op type to replace, e.g. Sigmoid")
    ap.add_argument("--to", dest="dst", required=True, help="op type to insert, e.g. HardSigmoid")
    ap.add_argument("--attr", action="append", metavar="NAME=VALUE",
                    help="float attribute for the new op; repeatable. Defaults are the ONNX "
                         "operator's own, which are often NOT what a quantizer used - read them "
                         "off the quantized graph rather than assuming")
    ap.add_argument("--expect", type=int, default=None,
                    help="fail unless exactly this many nodes are swapped")
    args = ap.parse_args()

    attrs = parse_attr(args.attr)
    m = onnx.load(args.inp)
    g = m.graph

    before = Counter(n.op_type for n in g.node)
    swapped = 0
    for i, n in enumerate(g.node):
        if n.op_type != args.src:
            continue
        new = helper.make_node(args.dst, list(n.input), list(n.output), name=n.name, **attrs)
        g.node[i].CopyFrom(new)
        swapped += 1

    if args.expect is not None and swapped != args.expect:
        raise SystemExit(f"swapped {swapped} {args.src} nodes, expected {args.expect}")
    if not swapped:
        raise SystemExit(f"no {args.src} nodes in {args.inp}")

    onnx.checker.check_model(m)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, args.out)

    after = Counter(n.op_type for n in g.node)
    print(f"swapped {swapped} {args.src} -> {args.dst}"
          + (f" {attrs}" if attrs else " (operator defaults)"))
    print(f"  {args.src:14s} {before.get(args.src, 0)} -> {after.get(args.src, 0)}")
    print(f"  {args.dst:14s} {before.get(args.dst, 0)} -> {after.get(args.dst, 0)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
