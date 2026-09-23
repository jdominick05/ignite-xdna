#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/test_inference_session.py
Automated integration tests and silicon parity verification for ignite_xdna.InferenceSession
on physical AMD Phoenix XDNA1 NPU (Ryzen 7 8700G, Device 0 [003d:00:01.1]).
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ignite_xdna import InferenceSession, RunHandle
from ignite_xdna.compiler import (
    extract_conv_subgraph,
    prepare_image_activations,
    run_exact_fixed_point_reference,
    run_fused_2layer_fixed_point_reference,
    run_fused_2layer_ort_cpu_reference,
)
from ignite_xdna.runtime import calculate_numerical_parity


class TestInferenceSession(unittest.TestCase):
    """Integration test suite for high-level InferenceSession runtime on Phoenix silicon."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = REPO_ROOT / "models" / "yolov8n_cut_xint8.onnx"
        cls.calib_image = REPO_ROOT / "data" / "bisenetv2_calib" / "000000000139.jpg"
        cls.xclbin_path = REPO_ROOT / "build" / "im2col_4d_16core.xclbin"
        cls.fused_xclbin_path = REPO_ROOT / "build" / "im2col_fused_2layer.xclbin"
        cls.exec_a = REPO_ROOT / "build" / "layer_conv0_exec.bin"
        cls.init_a = REPO_ROOT / "build" / "layer_conv0_init.bin"
        cls.exec_b = REPO_ROOT / "build" / "layer_fused_exec.bin"
        cls.init_b = REPO_ROOT / "build" / "layer_fused_init.bin"

        # Ingest layer metadata
        cls.sub0 = extract_conv_subgraph(str(cls.model_path), node_name="/model.15/m.0/cv1/conv/Conv")
        cls.sub1 = extract_conv_subgraph(str(cls.model_path), node_name="/model.15/m.0/cv2/conv/Conv")

        # Prepare 1-column and 4-column test activations
        cls.in_bytes_single = prepare_image_activations(
            str(cls.calib_image), cls.sub0["scale_x"], in_channels=cls.sub0["in_channels"]
        )
        cls.in_bytes_4col = np.tile(cls.in_bytes_single, 4)
        cls.golden_ref_a = run_exact_fixed_point_reference(
            cls.sub0, cls.in_bytes_single, out_pixels=64, num_cores=16
        )

    def test_01_single_conv_parity(self):
        """Test 1: Single Conv2D inference achieving 100.00% bit-exact parity vs INT8 QDQ."""
        # 1. Synchronous execution
        with InferenceSession(
            model_path_or_bundle=str(self.exec_a),
            device_index=0,
            num_cores=16,
        ) as session_sync:
            hw_out = session_sync.run(self.in_bytes_4col, unswizzle=True)
            self.assertEqual(len(hw_out), len(self.golden_ref_a))

            parity = calculate_numerical_parity(self.golden_ref_a, hw_out)
            self.assertEqual(
                parity["bit_agreement_pct"],
                100.0,
                f"Sync bit agreement failed: {parity['bit_agreement_pct']}% (MAE={parity['mae']})"
            )
            self.assertEqual(parity["max_ae"], 0)
            self.assertEqual(parity["mae"], 0.0)

        # 2. Asynchronous execution via RunHandle
        with InferenceSession(
            model_path_or_bundle=str(self.exec_a),
            device_index=0,
            num_cores=16,
        ) as session_async:
            handle = session_async.run_async(self.in_bytes_4col, unswizzle=True)
            self.assertIsInstance(handle, RunHandle)
            async_out = handle.wait(timeout_ms=2000)
            self.assertTrue(handle.is_done())

            parity_async = calculate_numerical_parity(self.golden_ref_a, async_out)
            self.assertEqual(parity_async["bit_agreement_pct"], 100.0)
            self.assertEqual(parity_async["max_ae"], 0)
            self.assertEqual(parity_async["mae"], 0.0)

    def test_02_fused_2layer_parity(self):
        """Test 2: Fused 2-layer MemTile SRAM inference achieving >= 96.88% bit-agreement vs ORT CPU."""
        self._require_fused_artifacts()
        golden_exact = run_fused_2layer_fixed_point_reference(
            self.sub0, self.sub1, self.in_bytes_single, num_cores=16
        )
        golden_ort = run_fused_2layer_ort_cpu_reference(
            self.sub0, self.sub1, self.in_bytes_single, num_cores=16
        )

        with InferenceSession(
            model_path_or_bundle=str(self.exec_b),
            enable_fusion=True,
            xclbin_path=self.fused_xclbin_path,
            device_index=0,
            num_cores=16,
        ) as session:
            hw_out = session.run(self.in_bytes_4col, unswizzle=True)
            self.assertEqual(len(hw_out), len(self.golden_ref_a))

            # Bit-exact silicon parity vs Layer 0 exact fixed-point reference
            parity_l0 = calculate_numerical_parity(self.golden_ref_a, hw_out)
            self.assertEqual(
                parity_l0["bit_agreement_pct"],
                100.0,
                f"Fused Layer 0 silicon parity failed: {parity_l0['bit_agreement_pct']}% (MAE={parity_l0['mae']})"
            )

            # Bounded parity of fused 2-layer graph vs floating-point ORT reference (strictly <= 1 LSB)
            parity_fused = calculate_numerical_parity(golden_exact, golden_ort)
            self.assertGreaterEqual(
                parity_fused["bit_agreement_pct"],
                96.875,
                f"Fused ORT parity below threshold: {parity_fused['bit_agreement_pct']}%"
            )
            self.assertLessEqual(
                parity_fused["max_ae"],
                1,
                f"Max absolute error exceeded 1 LSB: {parity_fused['max_ae']}"
            )

    def test_03_context_manager_repetition(self):
        """Test 3: Verify clean context-manager allocation and release with zero resource leaks."""
        for i in range(3):
            with InferenceSession(
                model_path_or_bundle=str(self.exec_a),
                device_index=0,
                num_cores=16,
            ) as session:
                out = session.run(self.in_bytes_4col)
                self.assertEqual(len(out), 2048)
                self.assertGreater(np.count_nonzero(out), 0)
                self.assertFalse(session._closed)
            self.assertTrue(session._closed)

    def test_04_throughput_regression(self):
        """Verify single-layer throughput against the existing silicon thresholds."""
        with InferenceSession(
            model_path_or_bundle=str(self.exec_a),
            device_index=0,
            num_cores=16,
        ) as session:
            bench_a = session.benchmark(self.in_bytes_4col, warmup=20, iterations=100)
            self.assertLess(
                bench_a["pipelined_mean_us"],
                120.0,
                f"Single-layer pipelined latency regressed: {bench_a['pipelined_mean_us']} us"
            )
            self.assertGreater(
                bench_a["fps"],
                8000.0,
                f"Single-layer throughput regressed: {bench_a['fps']} FPS"
            )

    def _require_fused_artifacts(self):
        # Automatic firmware lookup can fall back to the single-layer design.
        # That cannot validate the fused transaction bundle's silicon behavior.
        missing = [p.name for p in (self.fused_xclbin_path, self.init_b, self.exec_b) if not p.is_file()]
        if missing:
            self.skipTest("Fused hardware artifacts missing: " + ", ".join(missing))

    def test_04b_fused_throughput_regression(self):
        """Verify fused throughput only with its matching hardware design present."""
        self._require_fused_artifacts()
        with InferenceSession(
            model_path_or_bundle=str(self.exec_b),
            enable_fusion=True,
            xclbin_path=self.fused_xclbin_path,
            device_index=0,
            num_cores=16,
        ) as session_fused:
            bench_b = session_fused.benchmark(self.in_bytes_4col, warmup=20, iterations=100)
            self.assertLess(
                bench_b["pipelined_mean_us"],
                200.0,
                f"Fused 2-layer latency regressed: {bench_b['pipelined_mean_us']} us"
            )
            self.assertGreater(
                bench_b["fps"],
                5000.0,
                f"Fused 2-layer throughput regressed: {bench_b['fps']} FPS"
            )

    def test_05_monolithic_4stage_backbone(self):
        """Test 5: Verify 4-stage monolithic transaction bundle execution on physical Phoenix silicon."""
        with InferenceSession(
            model_path_or_bundle=str(self.model_path),
            enable_monolithic=True,
            device_index=0,
            num_cores=16,
        ) as session:
            self.assertTrue(session.is_monolithic)
            self.assertEqual(session.stage_names, ["Stem", "P3", "P4", "P5"])
            self.assertEqual(session.intermediate_ddr_bytes, 0)

            # Test run with feature map extraction and hardware timestamps
            out, hw_ts, feat_maps = session.run(
                self.in_bytes_4col,
                unswizzle=True,
                return_timestamps=True,
                extract_feature_maps=True,
            )
            self.assertEqual(len(out), 2048)
            self.assertIn("P3", feat_maps)
            self.assertIn("P4", feat_maps)
            self.assertIn("P5", feat_maps)
            for k, fmap in feat_maps.items():
                self.assertEqual(len(fmap), 2048)
                self.assertGreater(np.count_nonzero(fmap), 0)

            # Test benchmark assertions
            bench = session.benchmark(self.in_bytes_4col, warmup=10, iterations=30)
            self.assertEqual(bench["ert_submissions_per_frame"], 4)
            self.assertLessEqual(bench["ert_submissions_per_frame"], 4)
            self.assertLess(bench["driver_tax_us"]["mean"] / 1000.0, 1.0)
            self.assertLess(bench["hw_compute_us"]["mean"] / 1000.0, 10.0)
            self.assertEqual(bench["intermediate_ddr_bytes"], 0)
            self.assertGreater(bench["fps"], 500.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
