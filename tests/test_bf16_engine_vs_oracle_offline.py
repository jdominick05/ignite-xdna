"""Offline checks of tools/bf16_engine_vs_oracle.py, and the bf16 engine's measured gap to the W8A16
oracle on SESR-M7, pinned.

The helper checks always run. The pinned gap is model-gated, about half a minute an input.

WHERE THE BOUNDS COME FROM. results/aie/bf16_engine_vs_oracle_desktop2_20260922.log chained all nine
SESR-M7 layers from the five Set5 images and seed 0, with the aligned MAC model, and measured the worst
over those six inputs: 151 differing values in one tensor of 1,048,576 (body.4 on head.png), rel_l2
1.18e-4 in a layer and 1.59e-5 at the output, a largest difference of 1.0 bf16 step at the tensor's
peak (the output on butterfly.png), and an output gap 0.0091 of what the bf16 step itself does to the
model. THE MARGIN RULE IS TWICE THOSE WORST FIGURES, applied to every tensor. The margin is for what
moves between machines: the oracle sums in ORT's MLAS order for the CPU it runs on.

TWO INPUTS, AND WHAT EACH CAN SEE. Seed 0 needs no data/ and runs everywhere the model does, but its
own gap is at most one differing value per layer and none at the output, and the MAC model makes no
difference to it. So it pins the packet layout, the weights, the oracle's Cast pairs and its wiring
- one weight one int8 step off lands every tensor far outside the bound - and it cannot see the
accumulation. head.png, preprocessed in memory through npu.sesr.preprocess, is the input behind the
worst figures, so there the bound is a real assertion at a factor of two; it is skipped where
data/sesr_val is absent.

The MAC model is "aligned", the one in force. It is not what the gap is made of: the wide-model
control (bf16_engine_vs_oracle_wide_desktop2_20260922.log) leaves the worst figures unchanged, and
the aligned grid accounts for about 8 of the 80 places where an isolated layer leaves the exact sum.
The rest is fp32 summation order, and why ORT's order keeps the exact value more often than the
engine's is unexplained.

It is the emulator against the oracle. It says nothing about the device, which must equal the
emulator to the bit; that is a separate contract, and checking it on silicon is what this bound will
be applied to.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

from ignite_xdna.compiler import engine_bf16_emulator as em
from ignite_xdna.compiler import graph_reference_bf16 as gr16
from tests.test_engine_schedule_bf16_offline import chain_ir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
_spec = importlib.util.spec_from_file_location("bf16_engine_vs_oracle", ROOT / "tools" / "bf16_engine_vs_oracle.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

QDQ = ROOT / "models" / "sesr_m7_xint8.onnx"
FP32 = ROOT / "models" / "sesr_m7_fp32.onnx"


class Helpers(unittest.TestCase):
    def test_ordered_patterns_follow_value_order_and_merge_the_zeros(self):
        v = np.array([-2.0, -1.0078125, -1.0, -0.0, 0.0, 1.0, 1.0078125, 2.0], np.float32)
        k = tool.ordered(em.bf16_bits(v))
        self.assertTrue(np.all(np.diff(k) >= 0))
        self.assertEqual(k[3], k[4])                     # -0.0 and +0.0: equal values, no distance
        self.assertEqual(k[6] - k[5], 1)                 # adjacent bf16 values are one apart
        self.assertEqual(k[2] - k[1], 1)

    def test_rounding_from_float64_is_single(self):
        # Through float32 the value first lands on the bf16 tie 1 + 2**-8, and ties-to-even then takes
        # it down to 1.0: a double rounding. Rounded once, it is above the tie and goes up.
        v = 1.0 + 2.0 ** -8 + 2.0 ** -30
        self.assertEqual(float(tool.round_bf16_f64(np.array([v]))[0]), 1.0 + 2.0 ** -7)
        self.assertEqual(float(em.to_bf16(np.array([v], np.float32))[0]), 1.0)
        x = em.to_bf16(np.random.default_rng(0).normal(scale=100.0, size=4096).astype(np.float32))
        np.testing.assert_array_equal(tool.round_bf16_f64(x.astype(np.float64)), x)

    def test_spacing_is_one_bf16_step_at_that_magnitude(self):
        self.assertEqual([tool.bf16_spacing(v) for v in (136.0, 644.0, 1.0, 0.99, 0.0)],
                         [1.0, 4.0, 2.0 ** -7, 2.0 ** -8, 0.0])

    def test_attribution_counts_which_side_kept_the_exact_value(self):
        e, o, x = (np.array(a, np.float32) for a in ([1, 2, 3, 5], [1, 2, 4, 6], [1, 2, 4, 7]))
        self.assertEqual(tool.attribute(e, o, x), {"engine_differs": 2, "oracle_differs": 1, "split_engine_exact": 0,
                                                   "split_oracle_exact": 1, "split_neither": 1})

    def test_depth_to_space_matches_onnx_runtime_in_both_modes(self):
        import onnxruntime as ort
        x = np.arange(12 * 3 * 5, dtype=np.float32).reshape(1, 12, 3, 5)
        for mode in ("DCR", "CRD"):
            node = helper.make_node("DepthToSpace", ["x"], ["y"], blocksize=2, mode=mode)
            g = helper.make_graph([node], "d2s", [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
            want = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
            np.testing.assert_array_equal(tool.depth_to_space(x[0], 2, mode), want[0], mode)

    def test_the_exact_arm_equals_the_engine_where_every_sum_is_exact(self):
        # Small integer activations times power-of-two weights sum exactly in the core's aligned
        # accumulate, so the engine must equal the float64 arm to the bit on every layer. This is what
        # checks exact_layer's padding, stride, ReLU and residual, which share no code with the packer.
        ir = chain_ir()
        rng = np.random.default_rng(40)
        feed = {n: rng.integers(-3, 4, (t.channels, t.height, t.width)).astype(np.float32)
                for n, t in ir.tensors.items()}
        for L in ir.layers:
            want = tool.exact_layer(ir, L, feed)
            got = gr16.direct_layer(ir, L, feed)[:ir.tensors[L.output].channels]
            np.testing.assert_array_equal(em.bf16_bits(got), em.bf16_bits(want), L.name)
            self.assertGreater(np.count_nonzero(want), want.size // 4, L.name)


# Twice the worst over the six inputs of results/aie/bf16_engine_vs_oracle_desktop2_20260922.log.
DIFFER_FRACTION = 2 * 151 / 1048576      # body.4, head.png
PEAK_STEPS = 2 * 1.0                     # the output, butterfly.png: 0.5 at a peak of 118.5
LAYER_REL_L2 = 2 * 1.18e-4               # body.4, head.png
OUTPUT_REL_L2 = 2 * 1.59e-5              # head.png
OUTPUT_TO_BF16_STEP = 2 * 0.0091         # head.png: 1.594e-5 against 1.751e-3


HEAD_PNG = ROOT / "data" / "sesr_val" / "Set5_LR_x2" / "head.png"


class _Gap:
    """The pinned gap on one input; subclasses say which."""

    @classmethod
    def load(cls):
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        cls.rep = tool.measure(QDQ, FP32, [cls.load()], emulate=False, isolated=False)[0]
        if not cls.rep["output"].get("compared", True):
            raise AssertionError("the graph output was not compared, so the output bounds cannot apply")

    def test_the_comparison_is_not_vacuous(self):
        self.assertTrue(self.rep["ok"])
        self.assertEqual(self.rep["mac_model"], "aligned")
        rounded = {r["layer"]: r["oracle_rounded_by_to_bf16"] for r in self.rep["layers"]}
        # Only the residual Add is an unrounded fp32 sum in the oracle; a Conv or Relu output changing
        # under to_bf16 would mean the Cast pairs were folded and the oracle is fp32.
        self.assertGreater(rounded.pop("/body.6/body.6.0/Conv"), 0)
        self.assertEqual(set(rounded.values()), {0})
        out = self.rep["output"]
        self.assertGreater(out["rel_l2_oracle_vs_model_without_bf16"], 0.0)
        self.assertGreater(out["rel_l2_engine_vs_model_without_bf16"], 0.0)

    def test_every_tensor_is_inside_twice_the_measured_gap(self):
        rule = "bound = 2x the worst of bf16_engine_vs_oracle_desktop2_20260922.log"
        for r in self.rep["layers"] + [dict(self.rep["output"], layer="output", chained=self.rep["output"])]:
            c = r["chained"]
            self.assertLessEqual(c["differ"] / c["n"], DIFFER_FRACTION, f"{r['layer']}: {rule}")
            self.assertLessEqual(c["max_abs_peak_ulps"], PEAK_STEPS, f"{r['layer']}: {rule}")
            self.assertLessEqual(c["rel_l2"], OUTPUT_REL_L2 if r["layer"] == "output" else LAYER_REL_L2,
                                 f"{r['layer']}: {rule}")

    def test_the_gap_is_small_beside_what_bf16_itself_does(self):
        out = self.rep["output"]
        self.assertLessEqual(out["rel_l2"], OUTPUT_TO_BF16_STEP * out["rel_l2_oracle_vs_model_without_bf16"])


@unittest.skipUnless(QDQ.exists() and FP32.exists(), "sesr_m7 models not present")
class SesrGapSeed(_Gap, unittest.TestCase):
    @classmethod
    def load(cls):
        return "seed 0", tool.wo.seeded_input(QDQ, 0)


@unittest.skipUnless(QDQ.exists() and FP32.exists() and HEAD_PNG.exists(), "sesr_m7 models or Set5 not present")
class SesrGapHead(_Gap, unittest.TestCase):
    @classmethod
    def load(cls):
        import cv2
        from npu import sesr
        x, _ = sesr.preprocess(cv2.imread(str(HEAD_PNG)))
        return "head.png", x

    def test_this_input_sees_the_accumulation(self):
        # The reason this arm exists: seed 0 is blind to summation order and this image is not. On
        # Desktop 2 it measured 151 in its worst layer and 58 at the output; the count moves with the
        # CPU's ORT kernel, so only its being nonzero is asserted.
        self.assertGreater(max(r["chained"]["differ"] for r in self.rep["layers"]), 0)
