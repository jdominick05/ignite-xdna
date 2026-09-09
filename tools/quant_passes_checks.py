"""Check quant/passes.py rewrites against ONNX fixtures whose answer is known.

passes.py transcribes Quark's own graph rewrites, and its docstring promises that names,
attribute values and Constant payloads follow the source "so the emitted graph diffs empty
against an oracle". That promise is checked today only by diffing whole quantized models
against a fresh Quark run -- which is the right final gate, but it cannot say WHICH pass
drifted when the diff is non-empty, and it needs a producer run to say anything at all.

These fixtures pin each pass separately on a graph small enough to assert an exact node
list. They do NOT establish vendor parity; that remains the oracle diff. They establish
that a pass does what its docstring says, which is what a later refactor breaks first.

Static ONNX only. No onnxruntime session, no NPU, no hardware context.

    python tools/quant_passes_checks.py
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# See quant_shift_cut_checks.py: site-packages ships an unrelated `tools` package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant.graph import Graph
from quant.passes import (HARD_SIGMOID_ALPHA, HARD_SIGMOID_SCALE, avgpool_dpu_scale, avgpool_scale,
                          check_hard_sigmoid, hardsigmoid_dpu_scale, sigmoid_to_hardsigmoid,
                          split_to_slice)
import quant_fixtures as fx


class Checks:
    def __init__(self):
        self.passed = self.failed = 0

    def check(self, label, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failed += 1
            print(f"  FAIL  {label}   {detail}")

    def equal(self, label, got, want):
        self.check(label, got == want, f"got {got!r}, want {want!r}")

    def close(self, label, got, want, tol=1e-9):
        self.check(label, abs(got - want) < tol, f"got {got!r}, want {want!r}")

    def raises(self, label, fn, exc=Exception):
        try:
            fn()
        except exc:
            self.passed += 1
            print(f"  PASS  {label}")
            return
        except Exception as e:  # noqa: BLE001 - the wrong exception type is a failure
            self.failed += 1
            print(f"  FAIL  {label}   raised {type(e).__name__}, want {exc.__name__}")
            return
        self.failed += 1
        print(f"  FAIL  {label}   did not raise")


def constant_value(node):
    return helper.get_attribute_value(next(a for a in node.attribute if a.name == "value"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    c = Checks()

    print("split_to_slice: one Slice plus four INT64 Constants per output")
    g = Graph(fx.split_model(axis=1, splits=(2, 3), channels=5))
    c.equal("one Split converted", split_to_slice(g), 1)
    nodes = g.nodes()
    slices = [n for n in nodes if n.op_type == "Slice"]
    constants = [n for n in nodes if n.op_type == "Constant"]
    c.equal("no Split survives", sum(1 for n in nodes if n.op_type == "Split"), 0)
    c.equal("one Slice per output", len(slices), 2)
    c.equal("four Constants per output", len(constants), 8)
    c.equal("Slice is named <output>_<i>", sorted(n.name for n in slices), ["out0_0", "out1_1"])
    c.equal("the split-sizes initializer is gone", g.initializer("split_sizes"), None)
    by_output = {n.output[0]: n for n in slices}
    payload = {n.output[0]: int(np.asarray(onnx.numpy_helper.to_array(constant_value(n))).reshape(-1)[0])
               for n in constants}
    c.equal("out0 starts", payload[by_output["out0"].input[1]], 0)
    c.equal("out0 ends", payload[by_output["out0"].input[2]], 2)
    c.equal("out1 starts", payload[by_output["out1"].input[1]], 2)
    c.equal("out1 ends", payload[by_output["out1"].input[2]], 5)
    c.equal("axes is the Split axis", payload[by_output["out1"].input[3]], 1)
    c.equal("steps is 1", payload[by_output["out1"].input[4]], 1)
    c.check("Constant payloads are INT64",
            all(constant_value(n).data_type == onnx.TensorProto.INT64 for n in constants))

    print("\nsplit_to_slice: the attribute form of Split")
    g = Graph(fx.split_model_attribute(axis=1, splits=(2, 3), channels=5))
    c.equal("attribute-form Split converted", split_to_slice(g), 1)
    c.equal("attribute form gives the same Slice count",
            sum(1 for n in g.nodes() if n.op_type == "Slice"), 2)

    print("\nsplit_to_slice: a Split it cannot read is rejected, not silently kept")
    bad = fx.split_model_attribute(splits=(2, 3))
    del bad.graph.node[0].attribute[:]          # no axis, no split
    c.raises("Split without an axis raises", lambda: split_to_slice(Graph(bad)), ValueError)
    no_sizes = fx.split_model_attribute(splits=(2, 3))
    for a in list(no_sizes.graph.node[0].attribute):
        if a.name == "split":
            no_sizes.graph.node[0].attribute.remove(a)
    c.raises("Split without sizes raises", lambda: split_to_slice(Graph(no_sizes)), ValueError)
    mismatched = fx.split_model(splits=(2, 3))
    mismatched.graph.initializer[0].CopyFrom(
        onnx.numpy_helper.from_array(np.array([1, 1, 3], np.int64), name="split_sizes"))
    c.raises("size/output count mismatch raises", lambda: split_to_slice(Graph(mismatched)), ValueError)

    print("\navgpool_scale: the sourced table, exactly")
    for (kh, kw), want in {(3, 3): 9.0 * 7.0 / 64.0, (5, 5): 25.0 * 10.0 / 256.0,
                           (6, 6): 36.0 * 7.0 / 256.0, (7, 7): 49.0 * 21.0 / 1024.0,
                           (14, 14): 196.0 * 21.0 / 4096.0}.items():
        c.close(f"avgpool_scale({kh},{kw}) is the table entry", avgpool_scale(kh, kw), want)
    c.close("avgpool_scale is 1.0 above 255", avgpool_scale(256, 256), 1.0)
    # Off the table the source runs a dyadic search for k/2^n closest to 1/rec; the scale it
    # returns is that ratio times rec, so it must sit near 1.0 without being exactly 1.0.
    got = avgpool_scale(4, 4)
    c.close("avgpool_scale(4,4) is exact for a power-of-two window", got, 1.0)
    c.check("avgpool_scale(9,9) lands near 1.0", 0.9 < avgpool_scale(9, 9) < 1.1, f"got {avgpool_scale(9, 9)}")

    print("\navgpool_dpu_scale: a Constant and a Mul on a square GlobalAveragePool")
    g = Graph(fx.global_avgpool_model(7, 7))
    report = avgpool_dpu_scale(g)
    c.equal("one pool scaled", report["scaled"], 1)
    c.equal("nothing skipped", report["skipped_not_square"], [])
    muls = [n for n in g.nodes() if n.op_type == "Mul"]
    consts = [n for n in g.nodes() if n.op_type == "Constant"]
    c.equal("one Mul inserted", len(muls), 1)
    c.equal("Mul is named <out>_Mul", muls[0].name, "y_Mul")
    c.equal("Constant feeds the Mul", muls[0].input[1], "y_Scale")
    c.close("the Constant carries the 7x7 table scale",
            float(onnx.numpy_helper.to_array(constant_value(consts[0])).reshape(-1)[0]),
            49.0 * 21.0 / 1024.0, tol=1e-7)
    c.equal("the pool now writes the pre-Mul tensor",
            [n for n in g.nodes() if n.op_type == "GlobalAveragePool"][0].output[0], "y_Mul")

    print("\navgpool_dpu_scale: a non-square pool is skipped and counted, not rewritten")
    g = Graph(fx.global_avgpool_model(7, 5))
    report = avgpool_dpu_scale(g)
    c.equal("nothing scaled", report["scaled"], 0)
    c.equal("the skip is reported by name", report["skipped_not_square"], ["gap"])
    c.equal("no Mul inserted", sum(1 for n in g.nodes() if n.op_type == "Mul"), 0)

    print("\navgpool_dpu_scale: AveragePool has no measured instance and raises")
    model = fx.global_avgpool_model(7, 7)
    model.graph.node[0].op_type = "AveragePool"
    model.graph.node[0].attribute.append(helper.make_attribute("kernel_shape", [7, 7]))
    c.raises("AveragePool raises rather than guessing", lambda: avgpool_dpu_scale(Graph(model)),
             NotImplementedError)

    print("\nsigmoid_to_hardsigmoid: same name, inputs and outputs; alpha 1/6 and no beta")
    g = Graph(fx.sigmoid_model(count=2))
    c.equal("both Sigmoids converted", sigmoid_to_hardsigmoid(g), 2)
    hs = [n for n in g.nodes() if n.op_type == "HardSigmoid"]
    c.equal("no Sigmoid survives", sum(1 for n in g.nodes() if n.op_type == "Sigmoid"), 0)
    c.equal("names are preserved", sorted(n.name for n in hs), ["sig0", "sig1"])
    c.equal("inputs are preserved", [list(n.input) for n in hs], [["x"], ["s0"]])
    c.equal("outputs are preserved", [list(n.output) for n in hs], [["s0"], ["s1"]])
    c.close("alpha is 1/6", next(a.f for a in hs[0].attribute if a.name == "alpha"), HARD_SIGMOID_ALPHA, 1e-7)
    c.equal("no beta is written", sum(1 for a in hs[0].attribute if a.name == "beta"), 0)

    print("\ncheck_hard_sigmoid: the attribute test the DPU rescale gates on")
    c.check("alpha 1/6, no beta qualifies", check_hard_sigmoid(fx.hardsigmoid_model().graph.node[0]))
    c.check("alpha 1/6, beta 0.5 qualifies",
            check_hard_sigmoid(fx.hardsigmoid_model(beta=0.5).graph.node[0]))
    c.check("alpha 1/6, beta 0.25 does not",
            not check_hard_sigmoid(fx.hardsigmoid_model(beta=0.25).graph.node[0]))
    c.check("alpha 0.2 does not", not check_hard_sigmoid(fx.hardsigmoid_model(alpha=0.2).graph.node[0]))

    print("\nhardsigmoid_dpu_scale: a Mul by (2731/16384)/(1/6) after each qualifying node")
    g = Graph(fx.hardsigmoid_model())
    c.equal("one node scaled", hardsigmoid_dpu_scale(g), 1)
    consts = [n for n in g.nodes() if n.op_type == "Constant"]
    c.close("the Constant carries HARD_SIGMOID_SCALE",
            float(onnx.numpy_helper.to_array(constant_value(consts[0])).reshape(-1)[0]),
            HARD_SIGMOID_SCALE, tol=1e-7)
    g = Graph(fx.hardsigmoid_model(alpha=0.2))
    c.equal("a non-qualifying HardSigmoid is left alone", hardsigmoid_dpu_scale(g), 0)
    c.equal("and gets no Mul", sum(1 for n in g.nodes() if n.op_type == "Mul"), 0)

    print("\nsigmoid_to_hardsigmoid then hardsigmoid_dpu_scale compose")
    g = Graph(fx.sigmoid_model(count=3))
    sigmoid_to_hardsigmoid(g)
    c.equal("every converted node is then scaled", hardsigmoid_dpu_scale(g), 3)
    c.equal("three Muls", sum(1 for n in g.nodes() if n.op_type == "Mul"), 3)

    print(f"\n{c.passed} passed, {c.failed} failed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
