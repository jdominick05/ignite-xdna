"""What exactly differs between two producers' XINT8 artifacts for the same model?

Ignition's parity claim against Quark is structural plus initializer-exact, never whole-file
identity, and the two files do differ in bytes. This says what that difference consists of.

The distinction matters because of what comes next. The VitisAI EP compiles the ONNX file
into a DPU program, and whether two files that agree on every number but disagree on names
and metadata compile to the SAME program is not something the parity gate can answer. This
tool establishes the premise for that test: it separates the differences that are numeric
(there should be none) from the ones that are naming and metadata (there are).

Uses the repo's own gate, `quant.verify.graph_diff`, rather than a second opinion, so a pass
here means exactly what a pass means in `scripts/quant-validate.sh`.

Usage:
    python tools/quant_pair_diff.py <a.onnx> <b.onnx>
    python tools/quant_pair_diff.py models/resnet50_ignition_cle_c64.onnx \\
                                    models/resnet50_quark_cle_c64.onnx

Run it from `resnet_env17` or `resnet_env`; it needs onnx and the `quant` package, and it
touches neither the NPU nor a session.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import onnx  # noqa: E402

from quant.graph import Graph  # noqa: E402
from quant.verify import graph_diff  # noqa: E402


def sha256_16(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", help="first ONNX artifact")
    ap.add_argument("b", help="second ONNX artifact")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args(argv)

    print("file bytes")
    for label, p in ((args.label_a, args.a), (args.label_b, args.b)):
        print(f"  {label:<10} {os.path.basename(p):<40} {os.path.getsize(p):>10} B  "
              f"sha256:{sha256_16(p)}")
    print()

    ma, mb = onnx.load(args.a), onnx.load(args.b)

    print("model metadata")
    for label, m in ((args.label_a, ma), (args.label_b, mb)):
        print(f"  {label:<10} producer={m.producer_name!r} version={m.producer_version!r} "
              f"ir={m.ir_version} opset={[o.version for o in m.opset_import]}")
    print()

    print("graph shape")
    for label, m in ((args.label_a, ma), (args.label_b, mb)):
        print(f"  {label:<10} nodes={len(m.graph.node)} "
              f"initializers={len(m.graph.initializer)} "
              f"inputs={len(m.graph.input)} outputs={len(m.graph.output)}")
    print()

    d = graph_diff(
        Graph.load(Path(args.a), strict=False),
        Graph.load(Path(args.b), strict=False),
    )
    print("graph_diff -- the same gate scripts/quant-validate.sh applies")
    print(f"  structural node delta      : {len(d.node_delta)}")
    print(f"  initializers byte-mismatch : {len(d.init_exact_mismatch)}")
    print(f"  int8 initializers compared : {d.compared_int8_initializers}")
    print(f"  int8 weights over 0 LSB    : {len(d.weight_lsb)}")
    print(f"  ok(0 LSB)                  : {d.ok(0)}")
    print(f"  ok(1 LSB)                  : {d.ok(1)}")
    for line in d.node_delta[:5]:
        print(f"    {line}")
    print()

    na = [n.name for n in ma.graph.node]
    nb = [n.name for n in mb.graph.node]
    same = sum(1 for x, y in zip(na, nb) if x == y)
    ia = sorted(i.name for i in ma.graph.initializer)
    ib = sorted(i.name for i in mb.graph.initializer)
    print("names, which the numbers do not depend on")
    print(f"  node names identical in order : {same}/{min(len(na), len(nb))}")
    print(f"  initializer name sets equal   : {ia == ib}")
    if ia != ib:
        only_a, only_b = set(ia) - set(ib), set(ib) - set(ia)
        print(f"    only in {args.label_a}: {len(only_a)} e.g. {sorted(only_a)[:3]}")
        print(f"    only in {args.label_b}: {len(only_b)} e.g. {sorted(only_b)[:3]}")
    print()
    print(f"byte-size difference: {abs(os.path.getsize(args.a) - os.path.getsize(args.b))} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
