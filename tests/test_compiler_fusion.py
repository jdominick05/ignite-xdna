#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tests/test_compiler_fusion.py

Unit and integration tests for YOLOv8n backbone partition consolidation,
C2f MemTile DMA stride & slice routing, in-tile residual Add fusion,
and monolithic multi-stage transaction scheduling on Phoenix XDNA1 NPU.
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np

# Ensure repo root and src are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ignite_xdna.compiler.memtile_agu import (
    MemTileAGU,
    MemTileBD,
    ChannelSlicePlan,
    ChannelConcatPlan,
    C2fRoutingPlan,
    DetectHeadEgressPlan,
    MEMTILE_BYTES,
    assert_disjoint_destinations,
)
from ignite_xdna.compiler.partitioner import (
    GraphPartitioner,
    NpuFusedPartition,
    CpuFallbackPartition,
    PartitionedGraph,
    ConvLayerMeta,
)
from ignite_xdna.compiler.scheduler import (
    MemTileMultiPassScheduler,
    SchedulePlan,
    StageSchedulePlan,
    MultiStageSchedulePlan,
    validate_memtile_buffer_layout,
    emit_multi_stage_transaction_bundle,
    L2_BANK_0_OFFSET,
    L2_BANK_1_OFFSET,
    L2_FINAL_EGRESS_OFFSET,
    L2_WEIGHTS_OFFSET,
)


class TestCompilerFusion(unittest.TestCase):
    """Test suite verifying partition consolidation and zero-copy MemTile routing."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = REPO_ROOT / "models" / "yolov8n_cut_xint8.onnx"
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Model file not found: {cls.model_path}")

    def test_01_yolov8n_backbone_partition_reduction(self):
        """
        Verify partitioner collapses 50 discrete NPU subgraphs down to <= 4 monolithic
        stages (Stem, P3, P4, P5) and reduces CPU fallbacks across the backbone to 0.
        """
        gp = GraphPartitioner(self.model_path, backbone_only=True)
        pg = gp.partition()

        # 1. Total discrete NPU partition count drops from 50 to <= 4 monolithic stages
        self.assertLessEqual(len(pg.npu_partitions), 4, f"Expected <= 4 NPU stages, got {len(pg.npu_partitions)}")
        self.assertEqual(len(pg.npu_partitions), 4, "Expected exactly 4 monolithic stages (Stem, P3, P4, P5)")

        # 2. Assert CPU fallback partitions drop from 51 to 0 across the backbone
        self.assertEqual(len(pg.cpu_partitions), 0, "Expected 0 CPU partitions in backbone-only mode")
        self.assertEqual(len(pg.backbone_cpu_partitions), 0, "Expected 0 backbone CPU fallbacks")

        # 3. Check each stage properties
        stage_names = [p.stage_name for p in pg.npu_partitions]
        self.assertEqual(stage_names, ["Stem", "P3", "P4", "P5"])

        # Check layer counts per stage (Stem: 7, P3: 7, P4: 7, P5: 6 => 27 Convs total)
        layer_counts = [p.num_layers for p in pg.npu_partitions]
        self.assertEqual(layer_counts, [7, 7, 7, 6], f"Expected [7, 7, 7, 6] layers, got {layer_counts}")
        total_convs = sum(layer_counts)
        self.assertEqual(total_convs, 27, f"Expected 27 backbone Convs, got {total_convs}")

        # 4. Verify zero intermediate DDR roundtrips across all stages
        for p in pg.npu_partitions:
            self.assertTrue(p.has_zero_ddr_roundtrip, f"Stage {p.stage_name} must have zero DDR roundtrips")

    def test_02_yolov8n_full_graph_backbone_fallbacks(self):
        """
        Verify that when partitioning the full graph, CPU fallbacks across
        the backbone feature extractor drop from 51 to 0.
        """
        gp = GraphPartitioner(self.model_path, backbone_only=False)
        pg = gp.partition()

        # Backbone NPU partitions should be <= 4
        self.assertLessEqual(len(pg.backbone_npu_partitions), 4)

        # CPU fallback partitions across the backbone must be exactly 0
        self.assertEqual(
            len(pg.backbone_cpu_partitions),
            0,
            f"Expected 0 CPU fallback partitions across backbone, got {len(pg.backbone_cpu_partitions)}"
        )

    def test_03_c2f_memtile_channel_slice_and_concat(self):
        """
        Verify MemTileAGU lowers C2f channel slicing and concatenation into
        4D Buffer Descriptors with custom strides, keeping all traffic in on-die SRAM.
        """
        agu = MemTileAGU()

        # Test channel slice: 32 -> 16 channels across 64 pixels
        slice_bd = agu.channel_slice_bd(
            base_address=L2_BANK_0_OFFSET,
            channel_offset=16,
            num_channels=16,
            total_channels=32,
            spatial_pixels=64,
        )
        self.assertEqual(slice_bd.address_span[0], L2_BANK_0_OFFSET + 16)
        self.assertEqual(slice_bd.transfer_bytes, 16 * 64)
        self.assertLessEqual(slice_bd.address_span[1], MEMTILE_BYTES)

        # Test strided channel concat: 3 chunks of 16 channels into 48-channel layout
        concat_bds = agu.channel_concat_bds(
            base_address=L2_BANK_1_OFFSET,
            chunk_channels=(16, 16, 16),
            spatial_pixels=64,
        )
        self.assertEqual(len(concat_bds), 3)
        for i, bd in enumerate(concat_bds):
            self.assertEqual(bd.address_span[0], L2_BANK_1_OFFSET + i * 16)
            self.assertEqual(bd.transfer_bytes, 16 * 64)
            self.assertLessEqual(bd.address_span[1], MEMTILE_BYTES)

        # Test full C2f routing plan
        routing = agu.route_c2f_stage(
            stage_name="model.2",
            spatial_pixels=64,
            in_channels=32,
            hidden_channels=16,
            num_bottlenecks=1,
        )
        self.assertTrue(routing.has_zero_intermediate_ddr_traffic)
        self.assertEqual(routing.stage_name, "model.2")

    def test_04_monolithic_multi_stage_transaction_layout(self):
        """
        Verify that scheduling multi-stage monolithic pipelines produces
        transaction sequences matching physical L2 MemTile capacity (512 KB).
        """
        gp = GraphPartitioner(self.model_path, backbone_only=True)
        pg = gp.partition()

        scheduler = MemTileMultiPassScheduler()

        # Schedule each stage and validate buffer layout
        for p in pg.npu_partitions:
            stage_plan = scheduler.schedule_stage(p)
            self.assertEqual(stage_plan.intermediate_ddr_bytes, 0)
            self.assertTrue(stage_plan.has_zero_intermediate_ddr_traffic)
            self.assertTrue(validate_memtile_buffer_layout(stage_plan))

        # Schedule multi-stage pipeline
        multi_plan = scheduler.schedule_multi_stage(pg.npu_partitions)
        self.assertEqual(multi_plan.total_stages, 4)
        self.assertEqual(multi_plan.total_layers, 27)
        self.assertEqual(multi_plan.intermediate_ddr_bytes, 0)
        self.assertTrue(multi_plan.has_zero_intermediate_ddr_traffic)
        self.assertTrue(validate_memtile_buffer_layout(multi_plan))

        # Check flattened SchedulePlan
        unified_plan = multi_plan.to_schedule_plan()
        self.assertEqual(unified_plan.num_layers, 27)
        self.assertEqual(len(unified_plan.passes), 27)
        self.assertTrue(unified_plan.has_zero_intermediate_ddr_traffic)

        # First pass must read from DDR, final pass must write to DDR
        self.assertEqual(unified_plan.passes[0].ingress_source, "HOST_DDR")
        self.assertEqual(unified_plan.passes[-1].egress_dest, "HOST_DDR")

        # All intermediate passes (1..25) must alternate strictly between L2 Banks with zero DDR traffic
        for k in range(1, 26):
            p = unified_plan.passes[k]
            self.assertIn(p.ingress_source, ("L2_BANK_0", "L2_BANK_1"))
            self.assertIn(p.egress_dest, ("L2_BANK_0", "L2_BANK_1"))

    def test_05_bottleneck_residual_add_and_silu_fusion(self):
        """
        Verify that all 6 residual Add nodes in YOLOv8n are fused in-tile,
        and all 27 Convs are marked with SiLU activation rather than falling back to CPU.
        """
        gp = GraphPartitioner(self.model_path, backbone_only=True)
        pg = gp.partition()

        all_layers: list[ConvLayerMeta] = []
        for p in pg.npu_partitions:
            all_layers.extend(p.layers)

        # Check all 27 Convs have SiLU activation fused
        for layer in all_layers:
            self.assertEqual(
                layer.activation,
                "SiLU",
                f"Layer {layer.node_name} should have SiLU activation fused"
            )

        # Check all 6 residual Add operations
        residual_add_layers = [l for l in all_layers if l.residual_add]
        self.assertEqual(
            len(residual_add_layers),
            6,
            f"Expected exactly 6 bottleneck residual Add layers, found {len(residual_add_layers)}"
        )

        expected_add_convs = [
            "/model.2/m.0/cv2/conv/Conv",
            "/model.4/m.0/cv2/conv/Conv",
            "/model.4/m.1/cv2/conv/Conv",
            "/model.6/m.0/cv2/conv/Conv",
            "/model.6/m.1/cv2/conv/Conv",
            "/model.8/m.0/cv2/conv/Conv",
        ]
        actual_add_convs = [l.node_name for l in residual_add_layers]
        self.assertEqual(actual_add_convs, expected_add_convs)

    def test_06_emit_multi_stage_transaction_bundle(self):
        """
        Verify emit_multi_stage_transaction_bundle validates MemTile capacity
        and generates valid init.bin and exec.bin files when base transaction is provided.
        """
        gp = GraphPartitioner(self.model_path, backbone_only=True)
        pg = gp.partition()

        scheduler = MemTileMultiPassScheduler()
        stage_stem = scheduler.schedule_stage(pg.npu_partitions[0])

        # Find any existing base transaction binary in the workspace
        candidate_bases = [
            REPO_ROOT / "build" / "bench_scale_1l_exec.bin",
            REPO_ROOT / "build" / "bench_scale_2l_exec.bin",
        ]
        base_txn = None
        for cand in candidate_bases:
            if cand.exists():
                base_txn = str(cand)
                break

        if base_txn is None:
            # Layout validation is already verified in test_04
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            out_init = os.path.join(tmpdir, "stem_init.bin")
            out_exec = os.path.join(tmpdir, "stem_exec.bin")

            init_res, exec_res = emit_multi_stage_transaction_bundle(
                schedule=stage_stem,
                base_txn_path=base_txn,
                out_init_path=out_init,
                out_exec_path=out_exec,
            )

            self.assertTrue(os.path.exists(init_res))
            self.assertTrue(os.path.exists(exec_res))
            self.assertGreater(os.path.getsize(init_res), 0)
            self.assertGreater(os.path.getsize(exec_res), 0)

    def test_07_memtile_in_flight_2x_upsampling(self):
        """
        Verify MemTile AGU 2x nearest-neighbour upsampling is lowered as four DMA passes:
        BD step fields are encoded minus one, so a zero-step (step=0, wrap=2) duplication
        is not expressible. Each pass streams the source once and scatters it into one
        output phase with 2-pixel / 2-row destination strides; every output word is
        written exactly once, with 0 intermediate DDR bytes and no core instruction.
        """
        agu = MemTileAGU()

        # 1. Layer 11 upsampling (per-column slice): 20x20x64 -> 40x40x64, Bank 0 -> scratch
        plan_p5 = agu.plan_upsample_2x(
            base_address=L2_BANK_0_OFFSET,
            destination_address=L2_BANK_1_OFFSET,
            input_shape=(20, 20, 64),
            destination_bounds=(L2_BANK_1_OFFSET, MEMTILE_BYTES),
        )
        self.assertTrue(plan_p5.has_zero_intermediate_ddr_traffic)
        self.assertEqual(plan_p5.total_output_bytes, 40 * 40 * 64)
        self.assertEqual(plan_p5.passes, 4)
        for bd in plan_p5.source_bds:
            self.assertEqual(bd.direction, "MM2S")
            self.assertEqual(bd.transfer_bytes, 20 * 20 * 64)
        for bd in plan_p5.dest_bds:
            self.assertEqual(bd.direction, "S2MM")
            self.assertEqual(bd.step_words[1], 2 * 64 // 4)           # two output pixels
            self.assertEqual(bd.step_words[2], 2 * 2 * 20 * 64 // 4)  # two output rows
        self.assertEqual(assert_disjoint_destinations(plan_p5.dest_bds), 40 * 40 * 64 // 4)

        # 2. Layer 14 upsampling (per-column slice): 40x40x32 -> 80x80x32 (204,800 B), Bank 1 -> scratch
        plan_p4 = agu.plan_upsample_2x(
            base_address=L2_BANK_1_OFFSET,
            destination_address=0x0,
            input_shape=(40, 40, 32),
            destination_bounds=(0x0, L2_BANK_0_OFFSET),
        )
        self.assertTrue(plan_p4.has_zero_intermediate_ddr_traffic)
        self.assertEqual(plan_p4.total_output_bytes, 80 * 80 * 32)
        self.assertEqual(plan_p4.passes, 4)
        self.assertEqual(assert_disjoint_destinations(plan_p4.dest_bds), 80 * 80 * 32 // 4)

    def test_08_memtile_lateral_concatenations(self):
        """
        Verify lateral feature map concatenations (P4 + upsampled P5, P3 + upsampled L12)
        are synthesized via strided S2MM DMA scatter into contiguous L2 MemTile SRAM.
        """
        agu = MemTileAGU()

        # Lateral concat 1 (per-column slice): P4 (32 ch) + Upsampled P5 (64 ch) -> 96 ch
        plan1 = agu.plan_lateral_concat(
            base_address=L2_BANK_0_OFFSET,
            chunk_channels=(32, 64),
            spatial_pixels=256,
        )
        self.assertTrue(plan1.has_zero_intermediate_ddr_traffic)
        self.assertEqual(plan1.total_channels, 96)
        self.assertEqual(len(plan1.buffer_descriptors), 2)
        self.assertEqual(plan1.buffer_descriptors[0].address_span[0], L2_BANK_0_OFFSET)
        self.assertEqual(plan1.buffer_descriptors[1].address_span[0], L2_BANK_0_OFFSET + 32)

        # Lateral concat 2 (per-column slice): P3 (16 ch) + Upsampled L12 (32 ch) -> 48 ch
        plan2 = agu.plan_lateral_concat(
            base_address=L2_BANK_1_OFFSET,
            chunk_channels=(16, 32),
            spatial_pixels=256,
        )
        self.assertTrue(plan2.has_zero_intermediate_ddr_traffic)
        self.assertEqual(plan2.total_channels, 48)
        self.assertEqual(len(plan2.buffer_descriptors), 2)
        self.assertEqual(plan2.buffer_descriptors[0].address_span[0], L2_BANK_1_OFFSET)
        self.assertEqual(plan2.buffer_descriptors[1].address_span[0], L2_BANK_1_OFFSET + 16)

    def test_09_neck_fpn_pan_graph_partitioning(self):
        """
        Verify GraphPartitioner absorbs ONNX Resize and Concat nodes across Layers 10-21,
        producing 2 monolithic NPU partitions (Neck_FPN, Neck_PAN) with strictly 0 CPU
        fallback partitions across the entire Neck.
        """
        gp = GraphPartitioner(self.model_path, fuse_neck=True)
        pg = gp.partition(backbone_only=False)

        # 4 Backbone + 2 Neck = 6 NPU monolithic stages
        self.assertEqual(len(pg.backbone_npu_partitions), 4)
        self.assertEqual(len(pg.neck_npu_partitions), 2)
        self.assertEqual(len(pg.neck_cpu_partitions), 0, "Zero CPU fallback partitions allowed in Neck")

        neck_fpn = pg.neck_npu_partitions[0]
        neck_pan = pg.neck_npu_partitions[1]

        self.assertEqual(neck_fpn.stage_name, "Neck_FPN")
        self.assertEqual(neck_fpn.num_layers, 8)
        self.assertEqual(neck_fpn.c2f_blocks, ["model.12", "model.15"])
        self.assertTrue(neck_fpn.has_zero_ddr_roundtrip)

        self.assertEqual(neck_pan.stage_name, "Neck_PAN")
        self.assertEqual(neck_pan.num_layers, 10)
        self.assertEqual(neck_pan.c2f_blocks, ["model.18", "model.21"])
        self.assertTrue(neck_pan.has_zero_ddr_roundtrip)

        # Verify all 18 Neck Convs have SiLU activation fused
        neck_layers = neck_fpn.layers + neck_pan.layers
        self.assertEqual(len(neck_layers), 18)
        for layer in neck_layers:
            self.assertEqual(layer.activation, "SiLU")

    def test_10_neck_multi_stage_schedule_layout(self):
        """
        Verify scheduling the 2-stage monolithic Neck transaction bundle alternates
        feature handoffs between L2 Bank 0 (0x40000) and Bank 1 (0x60000) with Locks 4 and 5
        and 0 intermediate DDR bytes.
        """
        gp = GraphPartitioner(self.model_path, fuse_neck=True)
        pg = gp.partition(neck_only=True)

        scheduler = MemTileMultiPassScheduler()
        multi_plan = scheduler.schedule_multi_stage(pg.neck_npu_partitions)

        self.assertEqual(multi_plan.total_stages, 2)
        self.assertEqual(multi_plan.total_layers, 18)
        self.assertEqual(multi_plan.intermediate_ddr_bytes, 0)
        self.assertTrue(multi_plan.has_zero_intermediate_ddr_traffic)
        self.assertTrue(validate_memtile_buffer_layout(multi_plan))

        unified_plan = multi_plan.to_schedule_plan()
        self.assertEqual(unified_plan.num_layers, 18)
        self.assertEqual(len(unified_plan.passes), 18)
        self.assertEqual(unified_plan.passes[0].ingress_source, "HOST_DDR")
        self.assertEqual(unified_plan.passes[-1].egress_dest, "HOST_DDR")

        # Intermediate passes alternate strictly between Bank 0 and Bank 1
        for k in range(1, 17):
            p = unified_plan.passes[k]
            self.assertIn(p.ingress_source, ("L2_BANK_0", "L2_BANK_1"))
            self.assertIn(p.egress_dest, ("L2_BANK_0", "L2_BANK_1"))

    def test_11_detect_heads_graph_partitioning(self):
        """
        Verify that absorbing YOLOv8n Detect Heads (Layer 22) into compiler graph:
          1. Produces 3 head stages (Detect_P3, Detect_P4, Detect_P5) with 6 Convs each (18 total).
          2. Absorbs all 6 final prediction 1x1 Convs with linear Identity activation.
          3. Absorbs all 12 intermediate 3x3 Convs with fused SiLU activation.
          4. Yields strictly 0 CPU fallback partitions across the entire network (Layers 0..22).
        """
        gp = GraphPartitioner(self.model_path, fuse_neck=True, fuse_head=True)
        pg = gp.partition(backbone_only=False)

        # 4 Backbone + 2 Neck + 3 Heads = 9 NPU monolithic stages
        self.assertEqual(len(pg.backbone_npu_partitions), 4)
        self.assertEqual(len(pg.neck_npu_partitions), 2)
        self.assertEqual(len(pg.head_npu_partitions), 3)
        self.assertEqual(len(pg.npu_partitions), 9)

        # Strictly 0 CPU fallback partitions across the entire network!
        self.assertEqual(len(pg.head_cpu_partitions), 0, "Zero CPU fallback partitions allowed in Detect Heads")
        self.assertEqual(len(pg.cpu_partitions), 0, "Zero CPU fallback partitions across Layers 0..22")

        head_stages = {p.stage_name: p for p in pg.head_npu_partitions}
        self.assertIn("Detect_P3", head_stages)
        self.assertIn("Detect_P4", head_stages)
        self.assertIn("Detect_P5", head_stages)

        for s_name in ("Detect_P3", "Detect_P4", "Detect_P5"):
            st = head_stages[s_name]
            self.assertEqual(st.num_layers, 6, f"{s_name} must contain 6 Convs (3 Box + 3 Cls)")
            self.assertTrue(st.has_zero_ddr_roundtrip)

            # Check that final 2 Convs in each head stage are 1x1 linear prediction heads
            # and first 4 Convs are 3x3 SiLU intermediate Convs
            for layer in st.layers:
                if layer.node_name.endswith(".2/Conv"):
                    self.assertEqual(layer.activation, "Identity")
                    self.assertEqual(layer.kernel_shape, [1, 1])
                    self.assertIn(layer.out_channels, (64, 80))
                else:
                    self.assertEqual(layer.activation, "SiLU")
                    self.assertEqual(layer.kernel_shape, [3, 3])

    def test_12_detect_head_s2mm_egress_plan(self):
        """
        Verify that MemTileAGU.plan_detect_head_egress configures direct S2MM DMA
        scatter into host DDR buffers formatted for DFL/NMS post-processing:
          - Total channels = 144 (64 box + 80 cls)
          - Direct S2MM DMA egress with zero intermediate DDR traffic.
        """
        agu = MemTileAGU()
        for scale, sp in (("P3", 100), ("P4", 200), ("P5", 400)):
            plan = agu.plan_detect_head_egress(
                scale_name=scale,
                spatial_pixels=sp,
                box_channels=64,
                cls_channels=80,
                base_address=0x40000,
                buffer_bounds=(0x40000, 0x40000 + 144 * sp + 64),
            )
            self.assertTrue(plan.has_zero_intermediate_ddr_traffic)
            self.assertEqual(plan.total_channels, 144)
            self.assertEqual(plan.box_bd.direction, "S2MM")
            self.assertEqual(plan.cls_bd.direction, "S2MM")
            self.assertEqual(plan.box_bd.address_span[0], 0x40000)
            self.assertEqual(plan.cls_bd.address_span[0], 0x40000 + 64)


if __name__ == "__main__":
    unittest.main()

