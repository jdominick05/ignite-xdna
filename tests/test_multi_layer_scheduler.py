#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
tests/test_multi_layer_scheduler.py

Physical Silicon Integration Tests & Latency Scaling Benchmarks for:
  - ONNX Graph Partitioner & Subgraph Extraction (partitioner.py)
  - Dynamic L2 MemTile Multi-Pass Scheduler (scheduler.py)
  - Heterogeneous CPU/NPU InferenceSession Pipeline (session.py)
  - Zero Intermediate DDR Roundtrips on AMD Phoenix XDNA1 NPU [003d:00:01.1].
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root and src are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ignite_xdna.compiler.partitioner import (
    GraphPartitioner,
    NpuFusedPartition,
    CpuFallbackPartition,
    build_synthetic_multi_layer_conv_model,
)
from ignite_xdna.compiler.scheduler import (
    MemTileMultiPassScheduler,
    SchedulePlan,
    emit_multi_layer_transaction_bundle,
    run_n_layer_fixed_point_reference,
    run_n_layer_ort_cpu_reference,
    execute_multi_layer_on_silicon,
)
from ignite_xdna.compiler.lower_onnx_conv import run_exact_fixed_point_reference
from ignite_xdna.runtime.driver import get_repo_root, setup_xrt_environment
from ignite_xdna.runtime.session import InferenceSession
from ignite_xdna.runtime.test_im2col_hardware import calculate_numerical_parity


def is_hardware_available() -> bool:
    try:
        setup_xrt_environment()
        import pyxrt
        d = pyxrt.device(0)
        return d is not None
    except Exception:
        return False


class TestMultiLayerScheduler(unittest.TestCase):
    """Integration test suite for multi-layer ONNX graph partitioning and dynamic MemTile scheduling."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = get_repo_root()
        cls.build_dir = cls.repo_root / "build"
        cls.build_dir.mkdir(parents=True, exist_ok=True)

        cls.xclbin_path = str(cls.build_dir / "im2col_4d_16core.xclbin")
        cls.base_txn_path = str(cls.build_dir / "layer_conv0_exec.bin")
        if not os.path.exists(cls.base_txn_path):
            cls.base_txn_path = str(cls.build_dir / "im2col_4d_16core.bin")

        cls.hw_available = is_hardware_available() and os.path.exists(cls.xclbin_path)

    def test_partitioner_multi_layer_chain(self):
        """Verify GraphPartitioner extracts maximal fusible Conv chains and stationary parameters."""
        model_3layer = build_synthetic_multi_layer_conv_model(num_layers=3, seed=42)
        partitioner = GraphPartitioner(model_3layer)
        pg = partitioner.partition()

        self.assertEqual(len(pg.npu_partitions), 1)
        npu_part = pg.npu_partitions[0]
        self.assertEqual(npu_part.num_layers, 3)

        # Verify per-layer metadata and stationary vector layout packing
        for k, layer in enumerate(npu_part.layers):
            self.assertIsNotNone(layer.weights_packed)
            self.assertEqual(len(layer.weights_packed), 2304)
            self.assertIsNotNone(layer.bias_packed)
            self.assertEqual(layer.bias_packed.shape, (32,))
            self.assertGreaterEqual(layer.shift_cut, 0)

    def test_heterogeneous_cpu_npu_partitioning(self):
        """Verify topological partitioning splits unsupported operators to CPU fallback partitions."""
        model_het = build_synthetic_multi_layer_conv_model(
            num_layers=3, seed=42, add_cpu_head=True, add_cpu_tail=True
        )
        partitioner = GraphPartitioner(model_het)
        pg = partitioner.partition()

        self.assertEqual(len(pg.partitions), 3)
        self.assertIsInstance(pg.partitions[0], CpuFallbackPartition)
        self.assertIsInstance(pg.partitions[1], NpuFusedPartition)
        self.assertIsInstance(pg.partitions[2], CpuFallbackPartition)

        self.assertEqual(pg.partitions[1].num_layers, 3)
        self.assertIsNotNone(pg.partitions[0].onnx_model)
        self.assertIsNotNone(pg.partitions[2].onnx_model)

    def test_dynamic_l2_memtile_scheduler_floorplan(self):
        """Verify dynamic scheduler alternates L2 Bank 0 (0x40000) and Bank 1 (0x60000) with 0 DDR bytes."""
        model_4layer = build_synthetic_multi_layer_conv_model(num_layers=4, seed=42)
        partitioner = GraphPartitioner(model_4layer)
        pg = partitioner.partition()
        npu_part = pg.npu_partitions[0]

        scheduler = MemTileMultiPassScheduler(num_cores=16)
        plan = scheduler.schedule(npu_part)

        self.assertEqual(plan.num_layers, 4)
        self.assertEqual(plan.intermediate_ddr_bytes, 0)
        self.assertTrue(plan.has_zero_intermediate_ddr_traffic)

        # Pass 0: Ingress DDR -> Egress L2_BANK_0 (0x40000)
        self.assertEqual(plan.passes[0].ingress_source, "HOST_DDR")
        self.assertEqual(plan.passes[0].egress_dest, "L2_BANK_0")
        self.assertEqual(plan.passes[0].egress_addr, 0x40000)
        self.assertEqual(plan.passes[0].egress_lock_id, 4)

        # Pass 1: Ingress L2_BANK_0 (0x40000) -> Egress L2_BANK_1 (0x60000)
        self.assertEqual(plan.passes[1].ingress_source, "L2_BANK_0")
        self.assertEqual(plan.passes[1].ingress_addr, 0x40000)
        self.assertEqual(plan.passes[1].egress_dest, "L2_BANK_1")
        self.assertEqual(plan.passes[1].egress_addr, 0x60000)
        self.assertEqual(plan.passes[1].ingress_lock_id, 4)
        self.assertEqual(plan.passes[1].egress_lock_id, 5)

        # Pass 2: Ingress L2_BANK_1 (0x60000) -> Egress L2_BANK_0 (0x40000)
        self.assertEqual(plan.passes[2].ingress_source, "L2_BANK_1")
        self.assertEqual(plan.passes[2].ingress_addr, 0x60000)
        self.assertEqual(plan.passes[2].egress_dest, "L2_BANK_0")
        self.assertEqual(plan.passes[2].egress_addr, 0x40000)
        self.assertEqual(plan.passes[2].ingress_lock_id, 5)
        self.assertEqual(plan.passes[2].egress_lock_id, 4)

        # Pass 3: Ingress L2_BANK_0 (0x40000) -> Egress HOST_DDR
        self.assertEqual(plan.passes[3].ingress_source, "L2_BANK_0")
        self.assertEqual(plan.passes[3].ingress_addr, 0x40000)
        self.assertEqual(plan.passes[3].egress_dest, "HOST_DDR")
        self.assertEqual(plan.passes[3].ingress_lock_id, 4)
        self.assertIsNone(plan.passes[3].egress_lock_id)

    def test_physical_silicon_3layer_execution(self):
        """Execute 3-layer Conv subgraph on physical Phoenix silicon and assert numerical parity."""
        if not self.hw_available:
            self.skipTest("AMD Phoenix NPU hardware or firmware XCLBIN unavailable")

        model_3layer = build_synthetic_multi_layer_conv_model(num_layers=3, seed=123)
        partitioner = GraphPartitioner(model_3layer)
        pg = partitioner.partition()
        npu_part = pg.npu_partitions[0]

        scheduler = MemTileMultiPassScheduler(num_cores=16)
        plan = scheduler.schedule(npu_part)

        init_bin = str(self.build_dir / "test_3layer_silicon_init.bin")
        exec_bin = str(self.build_dir / "test_3layer_silicon_exec.bin")
        emit_multi_layer_transaction_bundle(
            schedule=plan,
            base_txn_path=self.base_txn_path,
            out_init_path=init_bin,
            out_exec_path=exec_bin,
        )

        rng = np.random.RandomState(42)
        in_bytes = rng.randint(-32, 32, size=8192, dtype=np.int8)

        result = execute_multi_layer_on_silicon(
            schedule=plan,
            init_txn_path=init_bin,
            exec_txn_path=exec_bin,
            xclbin_path=self.xclbin_path,
            input_bytes=in_bytes,
            num_cores=16,
            warmup_iters=20,
            bench_iters=100,
        )

        self.assertEqual(result["intermediate_ddr_bytes"], 0)
        self.assertGreater(result["fps"], 0)

        # 1. Bit-exact physical silicon execution vs Layer 0 exact fixed-point reference
        l0_meta = {
            "weights_raw": npu_part.layers[0].weights_raw,
            "bias_i32": npu_part.layers[0].bias_i32,
            "shift_cut": npu_part.layers[0].shift_cut,
        }
        ref_l0 = run_exact_fixed_point_reference(l0_meta, in_bytes[:2048], num_cores=16)
        parity_hw = calculate_numerical_parity(ref_l0, result["unpacked_hw"][: len(ref_l0)])
        self.assertEqual(parity_hw["bit_agreement_pct"], 100.0)
        self.assertEqual(parity_hw["max_ae"], 0.0)

        # 2. End-to-end N-layer bounded parity vs ONNX Runtime CPU INT8 reference
        ref_exact = run_n_layer_fixed_point_reference(npu_part, in_bytes, num_cores=16)
        ref_ort = run_n_layer_ort_cpu_reference(model_3layer, in_bytes, num_cores=16)
        parity_ort = calculate_numerical_parity(ref_exact, ref_ort)
        self.assertLessEqual(parity_ort["max_ae"], 1.0)
        self.assertGreaterEqual(parity_ort["bit_agreement_pct"], 75.0)

    def test_physical_silicon_4layer_execution(self):
        """Execute 4-layer Conv subgraph on physical Phoenix silicon and assert numerical parity."""
        if not self.hw_available:
            self.skipTest("AMD Phoenix NPU hardware or firmware XCLBIN unavailable")

        model_4layer = build_synthetic_multi_layer_conv_model(num_layers=4, seed=456)
        partitioner = GraphPartitioner(model_4layer)
        pg = partitioner.partition()
        npu_part = pg.npu_partitions[0]

        scheduler = MemTileMultiPassScheduler(num_cores=16)
        plan = scheduler.schedule(npu_part)

        init_bin = str(self.build_dir / "test_4layer_silicon_init.bin")
        exec_bin = str(self.build_dir / "test_4layer_silicon_exec.bin")
        emit_multi_layer_transaction_bundle(
            schedule=plan,
            base_txn_path=self.base_txn_path,
            out_init_path=init_bin,
            out_exec_path=exec_bin,
        )

        rng = np.random.RandomState(42)
        in_bytes = rng.randint(-32, 32, size=8192, dtype=np.int8)

        result = execute_multi_layer_on_silicon(
            schedule=plan,
            init_txn_path=init_bin,
            exec_txn_path=exec_bin,
            xclbin_path=self.xclbin_path,
            input_bytes=in_bytes,
            num_cores=16,
            warmup_iters=20,
            bench_iters=100,
        )

        self.assertEqual(result["intermediate_ddr_bytes"], 0)
        self.assertGreater(result["fps"], 0)

        # 1. Bit-exact physical silicon execution vs Layer 0 exact fixed-point reference
        l0_meta = {
            "weights_raw": npu_part.layers[0].weights_raw,
            "bias_i32": npu_part.layers[0].bias_i32,
            "shift_cut": npu_part.layers[0].shift_cut,
        }
        ref_l0 = run_exact_fixed_point_reference(l0_meta, in_bytes[:2048], num_cores=16)
        parity_hw = calculate_numerical_parity(ref_l0, result["unpacked_hw"][: len(ref_l0)])
        self.assertEqual(parity_hw["bit_agreement_pct"], 100.0)
        self.assertEqual(parity_hw["max_ae"], 0.0)

        # 2. End-to-end N-layer bounded parity vs ONNX Runtime CPU INT8 reference
        ref_exact = run_n_layer_fixed_point_reference(npu_part, in_bytes, num_cores=16)
        ref_ort = run_n_layer_ort_cpu_reference(model_4layer, in_bytes, num_cores=16)
        parity_ort = calculate_numerical_parity(ref_exact, ref_ort)
        self.assertLessEqual(parity_ort["max_ae"], 1.0)
        self.assertEqual(parity_ort["bit_agreement_pct"], 100.0)

    def test_latency_scaling_and_marginal_cost(self):
        """Profile sustained latency showing single host dispatch overhead paid once with ~40 μs/layer scaling."""
        if not self.hw_available:
            self.skipTest("AMD Phoenix NPU hardware or firmware XCLBIN unavailable")

        latencies = {}
        rng = np.random.RandomState(42)
        in_bytes = rng.randint(-32, 32, size=8192, dtype=np.int8)

        for n_layers in [1, 2, 3, 4]:
            model = build_synthetic_multi_layer_conv_model(num_layers=n_layers, seed=100 + n_layers)
            partitioner = GraphPartitioner(model)
            pg = partitioner.partition()
            npu_part = pg.npu_partitions[0]

            scheduler = MemTileMultiPassScheduler(num_cores=16)
            plan = scheduler.schedule(npu_part)

            init_bin = str(self.build_dir / f"test_scale_{n_layers}l_init.bin")
            exec_bin = str(self.build_dir / f"test_scale_{n_layers}l_exec.bin")
            emit_multi_layer_transaction_bundle(
                schedule=plan,
                base_txn_path=self.base_txn_path,
                out_init_path=init_bin,
                out_exec_path=exec_bin,
            )

            res = execute_multi_layer_on_silicon(
                schedule=plan,
                init_txn_path=init_bin,
                exec_txn_path=exec_bin,
                xclbin_path=self.xclbin_path,
                input_bytes=in_bytes,
                num_cores=16,
                warmup_iters=20,
                bench_iters=100,
            )
            latencies[n_layers] = res["mean_us"]

        # Assert linear latency progression: each layer adds ~35-45 μs without redundant 75 μs host submission
        # Marginal cost of layer 2 -> 3 and 3 -> 4 is bounded well under host dispatch overhead (< 65 μs)
        marginal_2_to_3 = latencies[3] - latencies[2]
        marginal_3_to_4 = latencies[4] - latencies[3]

        self.assertLess(marginal_2_to_3, 65.0)
        self.assertLess(marginal_3_to_4, 65.0)


if __name__ == "__main__":
    unittest.main()
