#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/profiler.py

High-resolution hardware and host event profiler for AMD Phoenix XDNA1 (AIE2).
Instruments per-layer and per-partition execution timings, hardware PyXRT dispatch
events (submission, device execution, drain), and host DDR data-marshalling overhead.
Maps YOLOv8n DAG nodes to architectural stages and classifies performance bottlenecks.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# Categorical bottleneck taxonomy
CAT_HOST_DISPATCH = "Host Dispatch Overhead"
CAT_DDR_BOUNCE = "Intermediate DDR Bounce"
CAT_SPATIAL_STEM = "Spatial Stem Compute"
CAT_HIGH_CHANNEL = "High-Channel Bottleneck Compute"
CAT_CPU_FALLBACK = "CPU Fallback Compute"


@dataclass
class HardwareTimestamps:
    """Hardware and driver event timestamps (nanoseconds) for a single execution pulse."""
    ingress_marshal_ns: int = 0
    bo_in_sync_ns: int = 0
    dispatch_submission_ns: int = 0
    device_execution_ns: int = 0
    bo_out_sync_ns: int = 0
    egress_unswizzle_ns: int = 0
    total_partition_ns: int = 0

    @property
    def ingress_marshal_us(self) -> float:
        return self.ingress_marshal_ns / 1000.0

    @property
    def bo_in_sync_us(self) -> float:
        return self.bo_in_sync_ns / 1000.0

    @property
    def dispatch_submission_us(self) -> float:
        return self.dispatch_submission_ns / 1000.0

    @property
    def device_execution_us(self) -> float:
        return self.device_execution_ns / 1000.0

    @property
    def bo_out_sync_us(self) -> float:
        return self.bo_out_sync_ns / 1000.0

    @property
    def egress_unswizzle_us(self) -> float:
        return self.egress_unswizzle_ns / 1000.0

    @property
    def total_partition_us(self) -> float:
        return self.total_partition_ns / 1000.0


@dataclass
class PartitionProfileRecord:
    """Detailed profile record for a single graph partition across an execution iteration."""
    partition_id: int
    partition_type: str                  # "NPU" or "CPU"
    stage_group: str                     # e.g., "0 - Stem Layer 0 (3->16, s=2, 320x320)"
    stage_label: str                     # e.g., "Stem Conv0"
    node_names: List[str] = field(default_factory=list)
    op_types: List[str] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_bytes: int = 0
    output_bytes: int = 0
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None

    # Fine-grained microsecond metrics
    duration_us: float = 0.0
    ingress_marshal_us: float = 0.0
    bo_in_sync_us: float = 0.0
    dispatch_submission_us: float = 0.0
    device_execution_us: float = 0.0
    bo_out_sync_us: float = 0.0
    egress_unswizzle_us: float = 0.0
    cpu_eval_us: float = 0.0

    # Bottleneck classification
    primary_category: str = CAT_HOST_DISPATCH

    @property
    def ddr_bounce_bytes(self) -> int:
        """Host DDR bytes transferred across heterogeneous execution boundary."""
        return self.input_bytes + self.output_bytes

    def classify_bottleneck(self):
        """Classifies the primary bottleneck category based on timing breakdown."""
        if self.partition_type == "CPU":
            if "Stem" in self.stage_group or "Stage 2" in self.stage_group:
                self.primary_category = CAT_SPATIAL_STEM
            elif any(c in self.stage_group for c in ["Stage 4", "Stage 5", "SPPF", "Neck"]):
                self.primary_category = CAT_HIGH_CHANNEL
            else:
                self.primary_category = CAT_CPU_FALLBACK
            return

        # For NPU partitions: evaluate hardware vs driver vs DDR marshalling
        ddr_overhead = self.ingress_marshal_us + self.bo_in_sync_us + self.bo_out_sync_us + self.egress_unswizzle_us
        dispatch_overhead = self.dispatch_submission_us

        if dispatch_overhead > self.device_execution_us and dispatch_overhead > ddr_overhead:
            self.primary_category = CAT_HOST_DISPATCH
        elif ddr_overhead > self.device_execution_us:
            self.primary_category = CAT_DDR_BOUNCE
        elif "Stem" in self.stage_group or "Stage 2" in self.stage_group:
            self.primary_category = CAT_SPATIAL_STEM
        elif any(c in self.stage_group for c in ["Stage 4", "Stage 5", "SPPF", "Neck"]):
            self.primary_category = CAT_HIGH_CHANNEL
        else:
            self.primary_category = CAT_HOST_DISPATCH


def map_yolo_node_to_stage(node_name: str, op_types: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Maps an ONNX node name to its architectural stage in the YOLOv8n DAG:
      - Stem convolutions: Layer 0 (3->16, s=2, 320x320) and Layer 1 (16->32, s=2, 160x160)
      - C2f stages: Layers 2, 4, 6, 8 (tracking internal split, bottleneck Convs, skip Adds, and Concat)
      - Downsample convolutions: Layers 3 (32->64, s=2), 5 (64->128, s=2), 7 (128->256, s=2)
      - Neck/SPPF pooling: Layer 9 (SPPF 5x5 pooling chains)
      - Neck FPN/PAN: Layers 10-21
      - Detect Head: Layer 22
    """
    ops = op_types or []
    ops_str = f" [{','.join(ops[:2])}]" if ops else ""

    if "/model.0/" in node_name:
        return ("0 - Stem Layer 0 (3->16, s=2, 320x320)", f"Stem Conv0{ops_str}")
    elif "/model.1/" in node_name:
        return ("1 - Stem Layer 1 (16->32, s=2, 160x160)", f"Stem Conv1{ops_str}")
    elif "/model.2/" in node_name:
        sub = "cv1" if "cv1" in node_name else ("cv2" if "cv2" in node_name else ("m.0" if "m.0" in node_name else "split/add"))
        return ("2 - Stage 2 C2f (c=32, 160x160)", f"C2f-2 {sub}{ops_str}")
    elif "/model.3/" in node_name:
        return ("3 - Downsample Conv (32->64, s=2, 80x80)", f"Downsample Conv3{ops_str}")
    elif "/model.4/" in node_name:
        sub = "cv1" if "cv1" in node_name else ("cv2" if "cv2" in node_name else ("m.0" if "m.0" in node_name else ("m.1" if "m.1" in node_name else "split/add")))
        return ("4 - Stage 3 C2f (c=64, 80x80)", f"C2f-4 {sub}{ops_str}")
    elif "/model.5/" in node_name:
        return ("5 - Downsample Conv (64->128, s=2, 40x40)", f"Downsample Conv5{ops_str}")
    elif "/model.6/" in node_name:
        sub = "cv1" if "cv1" in node_name else ("cv2" if "cv2" in node_name else ("m.0" if "m.0" in node_name else ("m.1" if "m.1" in node_name else "split/add")))
        return ("6 - Stage 4 C2f (c=128, 40x40)", f"C2f-6 {sub}{ops_str}")
    elif "/model.7/" in node_name:
        return ("7 - Downsample Conv (128->256, s=2, 20x20)", f"Downsample Conv7{ops_str}")
    elif "/model.8/" in node_name:
        sub = "cv1" if "cv1" in node_name else ("cv2" if "cv2" in node_name else ("m.0" if "m.0" in node_name else "split/add"))
        return ("8 - Stage 5 C2f (c=256, 20x20)", f"C2f-8 {sub}{ops_str}")
    elif "/model.9/" in node_name:
        sub = "cv1" if "cv1" in node_name else ("cv2" if "cv2" in node_name else ("pool" if "MaxPool" in str(ops) else "concat"))
        return ("9 - Neck SPPF (c=256, 20x20)", f"SPPF {sub}{ops_str}")
    elif any(f"/model.{i}/" in node_name for i in range(10, 22)):
        idx = next((i for i in range(10, 22) if f"/model.{i}/" in node_name), 10)
        return (f"10-21 - Neck FPN/PAN (Layer {idx})", f"Neck L{idx}{ops_str}")
    elif "/model.22/" in node_name:
        return ("22 - Detect Head (Decodes)", f"Head Conv/Decode{ops_str}")
    else:
        return ("Misc / Preamble", f"Node{ops_str}")


@dataclass
class IterationProfile:
    """Trace records for all partitions in one complete model inference pass."""
    iteration_idx: int
    wall_duration_us: float = 0.0
    partitions: List[PartitionProfileRecord] = field(default_factory=list)

    @property
    def total_npu_us(self) -> float:
        return sum(p.duration_us for p in self.partitions if p.partition_type == "NPU")

    @property
    def total_cpu_us(self) -> float:
        return sum(p.duration_us for p in self.partitions if p.partition_type == "CPU")

    @property
    def total_ddr_bounce_bytes(self) -> int:
        return sum(p.ddr_bounce_bytes for p in self.partitions)


class HardwareEventProfiler:
    """
    Active execution profiler for InferenceSession.
    Captures microsecond event timelines, accumulates multi-iteration statistics,
    and generates Pareto bottleneck analyses.
    """

    def __init__(self, target_device: str = "AMD Phoenix [003d:00:01.1]"):
        self.target_device = target_device
        self.iterations: List[IterationProfile] = []
        self._current_iteration: Optional[IterationProfile] = None
        self._enabled = False

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def start_iteration(self, iteration_idx: int):
        if not self._enabled:
            return
        self._current_iteration = IterationProfile(iteration_idx=iteration_idx)

    def end_iteration(self, wall_duration_us: float):
        if not self._enabled or self._current_iteration is None:
            return
        self._current_iteration.wall_duration_us = wall_duration_us
        self.iterations.append(self._current_iteration)
        self._current_iteration = None

    def record_partition(self, record: PartitionProfileRecord):
        if not self._enabled or self._current_iteration is None:
            return
        record.classify_bottleneck()
        self._current_iteration.partitions.append(record)

    def clear(self):
        self.iterations.clear()
        self._current_iteration = None

    def summarize(self) -> Dict[str, Any]:
        """
        Aggregates multi-iteration traces into steady-state statistics,
        Pareto timing tables, and categorical bottleneck decompositions.
        """
        if not self.iterations:
            return {"error": "No profiled iterations recorded"}

        num_iters = len(self.iterations)
        wall_times = [it.wall_duration_us for it in self.iterations]
        npu_times = [it.total_npu_us for it in self.iterations]
        cpu_times = [it.total_cpu_us for it in self.iterations]

        # Use first iteration to get partition metadata
        sample_parts = self.iterations[0].partitions
        num_partitions = len(sample_parts)
        num_npu = sum(1 for p in sample_parts if p.partition_type == "NPU")
        num_cpu = sum(1 for p in sample_parts if p.partition_type == "CPU")

        # Per-partition mean metrics across iterations
        partition_stats: List[Dict[str, Any]] = []
        for p_idx in range(num_partitions):
            p_sample = sample_parts[p_idx]
            durs = [it.partitions[p_idx].duration_us for it in self.iterations if p_idx < len(it.partitions)]
            mean_dur = float(np.mean(durs)) if durs else 0.0
            median_dur = float(np.median(durs)) if durs else 0.0
            p95_dur = float(np.percentile(durs, 95)) if durs else 0.0

            sub_marshal = float(np.mean([it.partitions[p_idx].ingress_marshal_us for it in self.iterations if p_idx < len(it.partitions)]))
            sub_sync_in = float(np.mean([it.partitions[p_idx].bo_in_sync_us for it in self.iterations if p_idx < len(it.partitions)]))
            sub_dispatch = float(np.mean([it.partitions[p_idx].dispatch_submission_us for it in self.iterations if p_idx < len(it.partitions)]))
            sub_device = float(np.mean([it.partitions[p_idx].device_execution_us for it in self.iterations if p_idx < len(it.partitions)]))
            sub_sync_out = float(np.mean([it.partitions[p_idx].bo_out_sync_us for it in self.iterations if p_idx < len(it.partitions)]))
            sub_unswizzle = float(np.mean([it.partitions[p_idx].egress_unswizzle_us for it in self.iterations if p_idx < len(it.partitions)]))
            sub_cpu_eval = float(np.mean([it.partitions[p_idx].cpu_eval_us for it in self.iterations if p_idx < len(it.partitions)]))

            partition_stats.append({
                "partition_id": p_sample.partition_id,
                "partition_type": p_sample.partition_type,
                "stage_group": p_sample.stage_group,
                "stage_label": p_sample.stage_label,
                "node_names": p_sample.node_names,
                "op_types": p_sample.op_types,
                "primary_category": p_sample.primary_category,
                "mean_us": round(mean_dur, 2),
                "median_us": round(median_dur, 2),
                "p95_us": round(p95_dur, 2),
                "sub_timings_us": {
                    "ingress_marshal": round(sub_marshal, 2),
                    "bo_in_sync": round(sub_sync_in, 2),
                    "dispatch_submission": round(sub_dispatch, 2),
                    "device_execution": round(sub_device, 2),
                    "bo_out_sync": round(sub_sync_out, 2),
                    "egress_unswizzle": round(sub_unswizzle, 2),
                    "cpu_eval": round(sub_cpu_eval, 2),
                },
                "input_bytes": p_sample.input_bytes,
                "output_bytes": p_sample.output_bytes,
                "ddr_bounce_bytes": p_sample.ddr_bounce_bytes,
            })

        # Cumulative driver dispatch floor
        dispatch_floor_us = num_npu * 75.0

        # Cumulative DDR traffic
        total_ddr_bounce_bytes = sum(p["ddr_bounce_bytes"] for p in partition_stats)

        # Categorical aggregation
        category_totals: Dict[str, float] = {
            CAT_HOST_DISPATCH: 0.0,
            CAT_DDR_BOUNCE: 0.0,
            CAT_SPATIAL_STEM: 0.0,
            CAT_HIGH_CHANNEL: 0.0,
            CAT_CPU_FALLBACK: 0.0,
        }

        for p in partition_stats:
            cat = p["primary_category"]
            category_totals[cat] = category_totals.get(cat, 0.0) + p["mean_us"]

        total_mean_us = float(np.mean(wall_times))

        category_breakdown = {
            cat: {
                "total_us": round(us, 2),
                "pct_of_total": round((us / total_mean_us * 100.0) if total_mean_us > 0 else 0.0, 2),
            }
            for cat, us in category_totals.items()
        }

        # Pareto table: sorted descending by mean duration
        pareto_table = sorted(partition_stats, key=lambda x: x["mean_us"], reverse=True)

        return {
            "target_device": self.target_device,
            "iterations_profiled": num_iters,
            "summary_latencies_us": {
                "wall_mean_us": round(float(np.mean(wall_times)), 2),
                "wall_median_us": round(float(np.median(wall_times)), 2),
                "wall_min_us": round(float(np.min(wall_times)), 2),
                "wall_max_us": round(float(np.max(wall_times)), 2),
                "wall_p95_us": round(float(np.percentile(wall_times, 95)), 2),
                "npu_total_mean_us": round(float(np.mean(npu_times)), 2),
                "cpu_total_mean_us": round(float(np.mean(cpu_times)), 2),
            },
            "fragmentation_audit": {
                "total_partitions": num_partitions,
                "num_npu_subgraphs": num_npu,
                "num_cpu_fallback_subgraphs": num_cpu,
                "driver_dispatch_floor_us": round(dispatch_floor_us, 2),
                "total_intermediate_ddr_bounce_bytes": total_ddr_bounce_bytes,
                "total_intermediate_ddr_bounce_mb": round(total_ddr_bounce_bytes / (1024 * 1024), 2),
            },
            "category_breakdown": category_breakdown,
            "pareto_partitions": pareto_table,
            "chronological_partitions": partition_stats,
        }
