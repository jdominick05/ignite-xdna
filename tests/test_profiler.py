#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/test_profiler.py
Unit tests and integration verification for ignite_xdna.runtime.profiler
and InferenceSession profiling instrumentation on AMD Phoenix silicon.
"""

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

from ignite_xdna.runtime.profiler import (
    HardwareTimestamps,
    PartitionProfileRecord,
    HardwareEventProfiler,
    map_yolo_node_to_stage,
    CAT_HOST_DISPATCH,
    CAT_DDR_BOUNCE,
    CAT_SPATIAL_STEM,
    CAT_HIGH_CHANNEL,
    CAT_CPU_FALLBACK,
)
from ignite_xdna import InferenceSession


class TestProfiler(unittest.TestCase):
    """Test suite for hardware profiling and timing breakdown components."""

    def test_01_hardware_timestamps(self):
        """Test nanosecond timestamp conversions to microsecond latencies."""
        ts = HardwareTimestamps(
            ingress_marshal_ns=10_000,       # 10 us
            bo_in_sync_ns=5_000,             # 5 us
            dispatch_submission_ns=75_000,   # 75 us
            device_execution_ns=130_000,     # 130 us
            bo_out_sync_ns=5_000,            # 5 us
            egress_unswizzle_ns=10_000,      # 10 us
            total_partition_ns=235_000,      # 235 us
        )
        self.assertAlmostEqual(ts.ingress_marshal_us, 10.0)
        self.assertAlmostEqual(ts.bo_in_sync_us, 5.0)
        self.assertAlmostEqual(ts.dispatch_submission_us, 75.0)
        self.assertAlmostEqual(ts.device_execution_us, 130.0)
        self.assertAlmostEqual(ts.bo_out_sync_us, 5.0)
        self.assertAlmostEqual(ts.egress_unswizzle_us, 10.0)
        self.assertAlmostEqual(ts.total_partition_us, 235.0)

    def test_02_map_yolo_node_to_stage(self):
        """Test YOLOv8n DAG stage mapping for stem, C2f, downsamples, neck, and head."""
        # Stem
        g0, l0 = map_yolo_node_to_stage("/model.0/conv/Conv")
        self.assertIn("Stem Layer 0", g0)
        g1, l1 = map_yolo_node_to_stage("/model.1/conv/Conv")
        self.assertIn("Stem Layer 1", g1)

        # C2f Stages
        g2, l2 = map_yolo_node_to_stage("/model.2/cv1/conv/Conv")
        self.assertIn("Stage 2 C2f", g2)
        g4, l4 = map_yolo_node_to_stage("/model.4/m.0/cv1/conv/Conv")
        self.assertIn("Stage 3 C2f", g4)
        g6, l6 = map_yolo_node_to_stage("/model.6/cv2/conv/Conv")
        self.assertIn("Stage 4 C2f", g6)
        g8, l8 = map_yolo_node_to_stage("/model.8/m.0/cv2/conv/Conv")
        self.assertIn("Stage 5 C2f", g8)

        # Downsample Convolutions
        g3, l3 = map_yolo_node_to_stage("/model.3/conv/Conv")
        self.assertIn("Downsample Conv (32->64", g3)
        g5, l5 = map_yolo_node_to_stage("/model.5/conv/Conv")
        self.assertIn("Downsample Conv (64->128", g5)
        g7, l7 = map_yolo_node_to_stage("/model.7/conv/Conv")
        self.assertIn("Downsample Conv (128->256", g7)

        # SPPF & Head
        g9, l9 = map_yolo_node_to_stage("/model.9/cv1/conv/Conv")
        self.assertIn("Neck SPPF", g9)
        g22, l22 = map_yolo_node_to_stage("/model.22/cv3.0/cv3.0.1/conv/Conv")
        self.assertIn("Detect Head", g22)

    def test_03_hardware_event_profiler_aggregation(self):
        """Test multi-iteration trace aggregation, Pareto table sorting, and category decomposition."""
        profiler = HardwareEventProfiler(target_device="Test Phoenix Silicon")
        profiler.enable()

        for it in range(3):
            profiler.start_iteration(it)
            # Record Stem Conv (Spatial Stem)
            profiler.record_partition(PartitionProfileRecord(
                partition_id=0,
                partition_type="NPU",
                stage_group="0 - Stem Layer 0 (3->16, s=2, 320x320)",
                stage_label="Stem Conv0",
                node_names=["/model.0/conv/Conv"],
                op_types=["Conv"],
                duration_us=500.0,
                dispatch_submission_us=75.0,
                device_execution_us=300.0,
                input_bytes=100_000,
                output_bytes=200_000,
            ))
            # Record High Channel Conv
            profiler.record_partition(PartitionProfileRecord(
                partition_id=1,
                partition_type="NPU",
                stage_group="8 - Stage 5 C2f (c=256, 20x20)",
                stage_label="C2f-8 cv1",
                node_names=["/model.8/cv1/conv/Conv"],
                op_types=["Conv"],
                duration_us=800.0,
                dispatch_submission_us=80.0,
                device_execution_us=600.0,
                input_bytes=50_000,
                output_bytes=50_000,
            ))
            # Record CPU fallback
            profiler.record_partition(PartitionProfileRecord(
                partition_id=2,
                partition_type="CPU",
                stage_group="Misc / Preamble",
                stage_label="CPU Mul",
                node_names=["/model.0/act/Mul"],
                op_types=["Mul"],
                duration_us=100.0,
                cpu_eval_us=100.0,
            ))
            profiler.end_iteration(1400.0)

        summary = profiler.summarize()
        self.assertEqual(summary["iterations_profiled"], 3)
        self.assertEqual(summary["fragmentation_audit"]["total_partitions"], 3)
        self.assertAlmostEqual(summary["summary_latencies_us"]["wall_mean_us"], 1400.0)

        # Check Pareto sorting: partition 1 (800 us) should be first
        pareto = summary["pareto_partitions"]
        self.assertEqual(pareto[0]["partition_id"], 1)
        self.assertEqual(pareto[1]["partition_id"], 0)
        self.assertEqual(pareto[2]["partition_id"], 2)

        # Check category decomposition
        cats = summary["category_breakdown"]
        self.assertIn(CAT_HOST_DISPATCH, cats)
        self.assertIn(CAT_SPATIAL_STEM, cats)
        self.assertIn(CAT_HIGH_CHANNEL, cats)
        self.assertIn(CAT_CPU_FALLBACK, cats)

    def test_04_session_profiling_integration_on_silicon(self):
        """Test InferenceSession enable_profiling and get_profile_report on physical Device 0."""
        exec_a = str(REPO_ROOT / "build" / "layer_conv0_exec.bin")
        if not Path(exec_a).exists():
            self.skipTest("layer_conv0_exec.bin not present in build/")

        with InferenceSession(
            model_path_or_bundle=exec_a,
            device_index=0,
            num_cores=16,
        ) as session:
            session.enable_profiling()
            dummy_in = np.zeros(session.in_bytes, dtype=np.int8)

            # Warmup and test runs
            for _ in range(5):
                session.run(dummy_in, unswizzle=True)

            report = session.get_profile_report()
            self.assertIsNotNone(report)
            self.assertGreaterEqual(report["iterations_profiled"], 5)
            self.assertGreater(report["summary_latencies_us"]["wall_mean_us"], 0.0)

            # Verify hardware timestamps were captured
            p0 = report["chronological_partitions"][0]
            self.assertGreater(p0["sub_timings_us"]["dispatch_submission"], 0.0)
            self.assertGreater(p0["sub_timings_us"]["device_execution"], 0.0)

            session.disable_profiling()
            self.assertFalse(session.profiler.is_enabled)


if __name__ == "__main__":
    unittest.main()
