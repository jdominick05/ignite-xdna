"""Check quant/shift_cut.py against ONNX fixtures whose answer is known by construction.

The analyzer had three defects fixed in 311a672 and two more here, and every one of them
was found by re-reading real models rather than by a check that could have caught it. These
fixtures place an operation at a chosen sigma and assert the analyzer reads that sigma back,
classifies it in the right band, and refuses to score an operation whose scales it cannot
read.

Static ONNX inspection only. No onnxruntime session, no NPU, no hardware context; safe to
run while another session holds the device.

    python tools/quant_shift_cut_checks.py
"""
import argparse
from pathlib import Path
import sys
import tempfile

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# This directory, not the `tools` package: resnet_env17's site-packages ships an unrelated
# module of that name, and it wins the import when tools/ is addressed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant.shift_cut import (CONTRACT_SHIFT_CUT, CONTRACT_SIGMA, STATED_SIGMA,
                             analyze_branch_divergence, analyze_model_shift_cut,
                             compute_shift_cut, repair_model_shift_cut)
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


def save(model, directory, name):
    path = Path(directory) / f"{name}.onnx"
    onnx.save(model, str(path))
    return str(path)


def systolic(hazards):
    return [h for h in hazards if h.op_type in ("Conv", "Gemm", "MatMul")]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    c = Checks()

    with tempfile.TemporaryDirectory() as tmp:
        print("sigma is read back as pos_x + pos_w - pos_y + 14")
        for pos_x, pos_w, pos_y in ((8, 8, 2), (4, 4, 1), (7, 9, 6), (2, 2, 3)):
            path = save(fx.qdq_conv_model(pos_x, pos_w, pos_y), tmp, f"conv_{pos_x}_{pos_w}_{pos_y}")
            ops = systolic(analyze_model_shift_cut(path))
            want = pos_x + pos_w - pos_y + 14
            c.equal(f"pos({pos_x},{pos_w},{pos_y}) -> sigma", ops[0].sigma if ops else None, want)
            c.equal(f"pos({pos_x},{pos_w},{pos_y}) -> shift_cut", ops[0].shift_cut if ops else None, want - 14)
            c.check(f"pos({pos_x},{pos_w},{pos_y}) resolved", bool(ops) and ops[0].resolved)

        print("\nthe producer contract sigma in [14, 30] is the vendor's shift_cut in [0, 16]")
        c.equal("CONTRACT_SIGMA is CONTRACT_SHIFT_CUT + 14", CONTRACT_SIGMA,
                (CONTRACT_SHIFT_CUT[0] + 14, CONTRACT_SHIFT_CUT[1] + 14))
        # sigma 14 (lower edge), 30 (upper edge), 31 (outside contract, inside stated bound).
        for pos_x, pos_w, pos_y, want_sigma, in_contract in (
                (8, 8, 16, 14, True), (8, 8, 0, 30, True), (9, 8, 0, 31, False)):
            path = save(fx.qdq_conv_model(pos_x, pos_w, pos_y), tmp, f"band_{want_sigma}")
            op = systolic(analyze_model_shift_cut(path))[0]
            c.equal(f"sigma={want_sigma} value", op.sigma, want_sigma)
            c.equal(f"sigma={want_sigma} in_contract", op.in_contract, in_contract)
            c.check(f"sigma={want_sigma} is not a hazard (inside {list(STATED_SIGMA)})", not op.is_hazard,
                    f"reason={op.reason}")

        print("\ncontract edges are reported as edges")
        low = systolic(analyze_model_shift_cut(save(fx.qdq_conv_model(8, 8, 16), tmp, "edge_low")))[0]
        high = systolic(analyze_model_shift_cut(save(fx.qdq_conv_model(8, 8, 0), tmp, "edge_high")))[0]
        mid = systolic(analyze_model_shift_cut(save(fx.qdq_conv_model(8, 8, 8), tmp, "edge_mid")))[0]
        c.equal("sigma 14 is the low edge", low.at_contract_edge, "low")
        c.equal("sigma 30 is the high edge", high.at_contract_edge, "high")
        c.equal("sigma 22 is not an edge", mid.at_contract_edge, None)

        print("\nthe stated (unvalidated) bound still fires outside [0, 31]")
        over = systolic(analyze_model_shift_cut(save(fx.qdq_conv_model(16, 16, 0), tmp, "over")))[0]
        c.equal("pos(16,16,0) -> sigma", over.sigma, 46)
        c.check("sigma 46 is a hazard", over.is_hazard, f"reason={over.reason}")
        c.check("sigma 46 names the shift clamp", "SHIFT CLAMP" in over.reason, over.reason)
        under = systolic(analyze_model_shift_cut(save(fx.qdq_conv_model(0, 0, 20), tmp, "under")))[0]
        c.equal("pos(0,0,20) -> sigma", under.sigma, -6)
        c.check("sigma -6 is a hazard", under.is_hazard, f"reason={under.reason}")
        c.check("sigma -6 names the accumulator", "ACCUMULATOR" in under.reason, under.reason)

        print("\nan operation whose scales cannot be read is UNRESOLVED, never a pass")
        path = save(fx.unresolved_mul_model(), tmp, "unresolved_mul")
        hazards = analyze_model_shift_cut(path)
        muls = [h for h in hazards if h.op_type == "Mul"]
        c.equal("the DPU-simulation Mul is found", len(muls), 1)
        c.check("it is not resolved", bool(muls) and not muls[0].resolved)
        c.check("it is not a hazard either", bool(muls) and not muls[0].is_hazard)
        c.check("it says why", bool(muls) and "not DequantizeLinear" in muls[0].unresolved_reason,
                muls[0].unresolved_reason if muls else "")
        c.equal("no resolved operation is claimed", sum(1 for h in hazards if h.resolved), 0)

        print("\nrepair skips what the analyzer could not score")
        out = str(Path(tmp) / "repaired.onnx")
        _, count, repairs = repair_model_shift_cut(path, out)
        c.equal("nothing repaired on an unresolvable graph", count if count is not None else len(repairs), 0)
        before = onnx.load(path).SerializeToString()
        after = onnx.load(out).SerializeToString()
        c.check("the model is byte-identical after a no-op repair", before == after)

        print("\nrepair leaves a feasible model untouched")
        good = save(fx.qdq_conv_model(8, 8, 2), tmp, "good")
        out_good = str(Path(tmp) / "good_repaired.onnx")
        _, _, repairs = repair_model_shift_cut(good, out_good)
        c.equal("no repair on an in-contract Conv", len(repairs), 0)
        c.check("in-contract model byte-identical after repair",
                onnx.load(good).SerializeToString() == onnx.load(out_good).SerializeToString())

        print("\nmulti-branch divergence reports the position spread")
        for pos_a, pos_b in ((4, 9), (6, 6), (2, 11)):
            path = save(fx.branch_add_model(pos_a, pos_b), tmp, f"branch_{pos_a}_{pos_b}")
            rows = analyze_branch_divergence(path)
            adds = [r for r in rows if r["op_type"] == "Add"]
            c.equal(f"Add({pos_a},{pos_b}) spread", adds[0]["spread"] if adds else None, abs(pos_a - pos_b))

        print("\ncompute_shift_cut keeps M in the 15-bit multiplier range")
        for pos_x, pos_w, pos_y in ((8, 8, 2), (4, 4, 1), (16, 16, 0), (0, 0, 20)):
            A = (2.0 ** -pos_x * 2.0 ** -pos_w) / 2.0 ** -pos_y
            M, _ = compute_shift_cut(A)
            c.check(f"M in [16384, 32767] for pos({pos_x},{pos_w},{pos_y})", 16384 <= M <= 32767, f"M={M}")

    print(f"\n{c.passed} passed, {c.failed} failed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
