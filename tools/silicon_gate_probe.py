"""Offline diagnosis of three pre-existing suite failures; changes no test files.

The workspace trial overrides only the fixture's allocation choice in this
process. The compilation trial preserves full-model rejection and separately
checks that a small graph can still produce a container. Neither
trial is a substitute for an unmodified, passing pytest run.
"""
import argparse
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def workspace():
    from tests import test_engine_host_layer as tests

    original = tests.es.plan_workspace
    allocations = []

    def allocate(ir, *args, **kwargs):
        kwargs["reuse"] = False
        ws = original(ir, *args, **kwargs)
        allocations.append(ws.nbytes)
        return ws

    suite = unittest.TestSuite([
        tests.ConcatViewHostInput("test_emulated_host_steps_read_the_view"),
        tests.ResidualHardSwish("test_packet_emulation_of_the_residual_hardswish_layers_matches_direct"),
    ])
    with mock.patch.object(tests.es, "plan_workspace", allocate):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun == 2 and not result.skipped
    assert result.wasSuccessful()
    print("DIAGNOSIS: both unchanged assertions pass with distinct storage for the preloaded golden tensors.")
    print("FIXTURE_WORKSPACE_BYTES:", allocations)


def compilation():
    import json
    import tempfile
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from ignite_xdna.compiler.cli import compile_model
    from ignite_xdna.quantization import PTQEngine, QuantizationConfig

    with tempfile.TemporaryDirectory(prefix="silicon_gate_") as tmp:
        d = Path(tmp)
        quantized, scales = d / "quant.onnx", d / "scales.json"
        summary = PTQEngine(QuantizationConfig(
            use_cle=True, use_adaround=True, num_calib=8, adaround_iterations=30,
        )).quantize(
            model_path=ROOT / "models/yolov8n_cut.onnx",
            calib_data_dir=ROOT / "data/coco128",
            output_onnx_path=quantized, output_scales_path=scales,
        )
        assert summary["status"] == "SUCCESS"
        assert json.loads(scales.read_text())["producer"] == "Ignite-PTQ"
        full = d / "full.ignite"
        try:
            compile_model(quantized, full, quant_scales_path=scales)
        except ValueError as exc:
            message = str(exc)
            print("EXPECTED_FULL_MODEL_REJECTION:", message, flush=True)
            assert "63 resident parameter sets" in message
            assert "data memory ends at 0x10000" in message
            assert "at most 16 sets fit" in message
        else:
            raise AssertionError("Oversized resident schedule unexpectedly accepted")
        assert not full.exists()

        # The legacy compiler requires all nine named stages to be nonempty.
        # One convolution per stage fits the resident layout; this tests its
        # container-writing path without weakening the full-model guard.
        prefixes = ["/model.0", "/model.4", "/model.6", "/model.8", "/model.10", "/model.16",
                    "/model.22/cv2.0", "/model.22/cv2.1", "/model.22/cv2.2"]
        nodes, initializers = [], []
        rng = np.random.default_rng(2901)
        previous = "input"
        for i, prefix in enumerate(prefixes):
            channels_in = 3 if i == 0 else 32
            weight = rng.normal(0, .01, (32, channels_in, 3, 3)).astype(np.float32)
            initializers.append(numpy_helper.from_array(weight, f"w{i}"))
            initializers.append(numpy_helper.from_array(np.zeros(32, dtype=np.float32), f"b{i}"))
            output = f"v{i}"
            nodes.append(helper.make_node("Conv", [previous, f"w{i}", f"b{i}"], [output],
                                          name=prefix + "/conv/Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
            previous = output
        graph = helper.make_graph(nodes, "resident_capacity_fixture",
                                  [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 16, 16])],
                                  [helper.make_tensor_value_info(previous, onnx.TensorProto.FLOAT, [1, 32, 16, 16])],
                                  initializers)
        small = d / "nine_conv.onnx"
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        onnx.checker.check_model(model)
        onnx.save(model, small)
        small_quantized, small_scales = d / "nine_quant.onnx", d / "nine_scales.json"
        small_summary = PTQEngine(QuantizationConfig(
            use_cle=True, use_adaround=True, num_calib=8, adaround_iterations=30,
            input_shape=(1, 3, 16, 16),
        )).quantize(
            model_path=small, calib_data_dir=ROOT / "data/coco128",
            output_onnx_path=small_quantized, output_scales_path=small_scales,
        )
        assert small_summary["status"] == "SUCCESS"
        assert json.loads(small_scales.read_text())["producer"] == "Ignite-PTQ"
        container = d / "nine_conv.ignite"
        count = compile_model(small_quantized, container, quant_scales_path=small_scales)
        assert 0 < count == container.stat().st_size < 10 * 1024 * 1024
        print("SMALL_MODEL_CONTAINER_BYTES:", count)
        print("DIAGNOSIS: full-model safety rejection and positive small-model container coverage can coexist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("workspace", "compilation"))
    args = parser.parse_args()
    workspace() if args.mode == "workspace" else compilation()
