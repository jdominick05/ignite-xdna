"""Offline checks of tools/w8a16_oracle.py: the oracle carries the IR's dequantized weights, mapped by
node name and refused on any mismatch, and its bf16 Cast pairs survive into the session.

Skipped when the SESR-M7 models are absent, like the other model-gated tests.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
_spec = importlib.util.spec_from_file_location("w8a16_oracle", ROOT / "tools" / "w8a16_oracle.py")
wo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wo)

QDQ = ROOT / "models" / "sesr_m7_xint8.onnx"
FP32 = ROOT / "models" / "sesr_m7_fp32.onnx"


@unittest.skipUnless(QDQ.exists() and FP32.exists(), "sesr_m7 models not present")
class TestW8A16Oracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.convs = wo.dequantized_convs(QDQ)
        cls.plain, _ = wo.build(QDQ, FP32, bf16=False)
        cls.oracle, cls.stats = wo.build(QDQ, FP32, bf16=True)
        dims = [d.dim_value for d in onnx.load(str(QDQ)).graph.input[0].type.tensor_type.shape.dim]
        cls.x = np.random.default_rng(0).integers(-128, 128, size=dims).astype(np.float32)

    def test_weights_are_the_irs_dequantized_values(self):
        inits = {t.name: numpy_helper.to_array(t) for t in self.plain.graph.initializer}
        convs = [n for n in self.plain.graph.node if n.op_type == "Conv"]
        self.assertEqual(len(convs), len(self.convs))
        for n in convs:
            w, _ = self.convs[n.name]
            np.testing.assert_array_equal(inits[n.input[1]], w)

    def test_a_renamed_node_is_refused(self):
        model = onnx.load(str(FP32))
        next(n for n in model.graph.node if n.op_type == "Conv").name = "not_in_the_ir"
        with self.assertRaisesRegex(ValueError, "one to one"):
            wo.patch_weights(model, self.convs)

    def test_casts_are_inserted_and_change_the_output(self):
        casts = sum(n.op_type == "Cast" for n in self.oracle.graph.node)
        self.assertEqual(casts, 2 * (self.stats["input_cast_chains"] + self.stats["output_cast_chains"]))
        # If ORT folded the pairs, the bf16 oracle would equal the fp32 one bit for bit.
        y_bf16, y_fp32 = wo.run(self.oracle, self.x), wo.run(self.plain, self.x)
        self.assertGreater(wo.rel_l2(y_bf16, y_fp32), 0.0)
        self.assertLess(wo.rel_l2(y_bf16, y_fp32), 0.05)

    def test_uint8_intermediates_unchanged_by_the_elem_type_keyword(self):
        from ignite_xdna.compiler.graph_reference import ort_intermediates
        name = next(n.output[0] for n in onnx.load(str(QDQ)).graph.node if n.op_type == "QuantizeLinear")
        a = ort_intermediates(QDQ, self.x, [name])[name]
        b = ort_intermediates(QDQ, self.x, [name], elem_type=onnx.TensorProto.UINT8)[name]
        self.assertEqual(a.dtype, np.uint8)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
