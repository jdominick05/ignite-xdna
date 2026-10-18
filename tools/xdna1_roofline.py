#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Closed-Form Roofline Model and Micro-Architectural Ceilings for AMD XDNA1 (Phoenix).

Hardware constant tags:
- MEASURED: Log path under results/ or line reference under docs/BENCHMARKS.md
- SPEC: Hardware target model or vendor specification document
- DERIVED: Formula derived from tagged inputs

Hardware Matrix:
- 16 reachable compute cores across 4 columns (Rows 2..5, Cols 1..4) [SPEC: AIETargetModel.cpp; docs/SILICON.md:62]
- 1.80 GHz core clock [MEASURED: results/aie/clock_probe_npu.log]
- 3.0 MB reachable on-chip SRAM (1,024 KB L1 + 2,048 KB MemTile L2) [DERIVED: docs/SILICON.md:70,104]
- DDR 8-channel MM2S read: 50.36 GB/s; 1 channel: 6.57-6.88 GB/s [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]
- Fill transport rate: 26.8 GB/s [MEASURED: docs/BENCHMARKS.md:12558]; whether concurrent R/W share a cap is OPEN [docs/SILICON.md:179]
- Compute ceilings:
  - INT8xINT8: 14.75 TOPS [DERIVED: 16 cores * 256 MACs * 2 * 1.80 GHz; SPEC: device.yaml]
  - INT8xINT4: SPEC shape ceiling 29.49 TOPS (512 MACs/vmac shape, SPEC: AIE-API 2024.1; docs/SILICON.md:74);
               MEASURED array ceiling 6,195.33 GOPS (~6.20 TOPS, MEASURED: results/aie/w4a8_array_npu.log:23;
               best k loop 0.73-0.75 vmac/cycle, MEASURED: results/aie/w4a8_probe_npu.log:178)
  - BF16xBF16: 7.37 TFLOPS [DERIVED: 16 cores * 128 MACs * 2 * 1.80 GHz; SPEC: device.yaml]
  - INT16xINT8: 7.37 TOPS [DERIVED: 16 cores * 128 MACs * 2 * 1.80 GHz; SPEC: device.yaml]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class Precision(str, Enum):
    INT8_INT8 = "INT8xINT8"
    INT8_INT4 = "INT8xINT4"
    BF16_BF16 = "BF16xBF16"
    INT16_INT8 = "INT16xINT8"


@dataclass(frozen=True)
class HardwareMatrix:
    """Micro-architectural hardware constants for AMD Phoenix AIE2 (XDNA1)."""

    # Clock frequency
    f_clk_ghz: float = 1.80  # [MEASURED: results/aie/clock_probe_npu.log]

    # Array topology
    total_physical_cols: int = 5  # [SPEC: device.yaml; docs/SILICON.md:60]
    total_physical_rows: int = 4  # [SPEC: device.yaml; docs/SILICON.md:60]
    total_physical_cores: int = 20  # [DERIVED: 5 cols * 4 rows; docs/SILICON.md:61]

    reachable_cols: int = 4  # [MEASURED: results/gops_yolov8n.log; docs/SILICON.md:62]
    reachable_rows: int = 4  # [SPEC: AIETargetModel.cpp; docs/SILICON.md:62]
    reachable_cores: int = 16  # [DERIVED: 4 cols * 4 rows; docs/SILICON.md:61]

    # SRAM hierarchy
    l1_kb_per_core: int = 64  # [SPEC: device.yaml; docs/SILICON.md:70]
    l1_banks_per_core: int = 4  # [SPEC: device.yaml; docs/SILICON.md:86]
    l1_bank_kb: int = 16  # [DERIVED: 64 KB / 4 banks; docs/SILICON.md:86]
    core_stack_bytes: int = 3328  # [MEASURED: results/aie/gemm_tile_sweep_c_single_buffer_npu.log; docs/SILICON.md:1038]
    l1_total_kb: int = 1024  # [DERIVED: 16 cores * 64 KB; docs/SILICON.md:70]

    memtile_count: int = 4  # [SPEC: device.yaml; docs/SILICON.md:34]
    memtile_kb_per_col: int = 512  # [SPEC: device.yaml; docs/SILICON.md:104]
    memtile_banks_per_tile: int = 8  # [SPEC: device.yaml; docs/SILICON.md:104]
    memtile_bank_kb: int = 64  # [DERIVED: 512 KB / 8 banks; docs/SILICON.md:104]
    memtile_total_kb: int = 2048  # [DERIVED: 4 tiles * 512 KB; docs/SILICON.md:104]

    total_sram_kb: int = 3072  # [DERIVED: 1024 KB L1 + 2048 KB MemTile; docs/SILICON.md:70,104]

    # Core L1 bus
    l1_bus_bytes_per_cycle: int = 64  # [SPEC: AIETargetModel.cpp (256-bit load + 256-bit store); docs/SILICON.md:79,85]
    l1_bandwidth_per_core_gbs: float = 115.2  # [DERIVED: 64 B * 1.80 GHz]
    l1_aggregate_bandwidth_gbs: float = 1843.2  # [DERIVED: 16 cores * 115.2 GB/s]

    # MemTile-to-Core stream interconnect
    memtile_stream_bytes_per_cycle: int = 4  # [MEASURED: results/aie/silicon_stream_width_desktop2_20260919.log]
    memtile_stream_bandwidth_gbs: float = 7.20  # [DERIVED: 4 B * 1.80 GHz]
    memtile_mm2s_channels_per_col: int = 6  # [SPEC: docs/SILICON.md:105]
    # Default to 4 streams per column (one activation serve MM2S per core row: src/ignite_xdna/compiler/engine_sequence.py:488)
    memtile_streams_per_col: int = 4  # [DERIVED: src/ignite_xdna/compiler/engine_sequence.py:488]

    # Shim DMA off-chip streaming (MM2S = Memory-Mapped to Stream = DDR read)
    shim_mm2s_single_channel_min_gbs: float = 6.57  # [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]
    shim_mm2s_single_channel_max_gbs: float = 6.88  # [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]
    shim_mm2s_single_channel_gbs: float = 6.88  # [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]
    shim_mm2s_read_channels: int = 8  # 2 MM2S per column * 4 columns [SPEC: docs/SILICON.md:114]
    shim_mm2s_8ch_aggregate_gbs: float = 50.36  # [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]

    # Engine fill transport rate (cite: docs/BENCHMARKS.md:12558, line 2300; note: whether concurrent R/W share a cap is OPEN, docs/SILICON.md:179)
    fill_transport_rate_gbs: float = 26.80  # [MEASURED: docs/BENCHMARKS.md:12558]


@dataclass(frozen=True)
class ComputeRates:
    """Theoretical and measured compute metrics per precision mode."""

    precision: Precision
    ops_per_mac: int
    macs_per_cycle_per_core_spec: int
    spec_core_peak_gops: float
    spec_column_peak_gops: float
    spec_array_peak_gops: float

    # Measured peak (where hardware measurements exist under results/aie/)
    measured_array_peak_gops: Optional[float] = None
    measured_k_loop_vmac_per_cycle: Optional[float] = None

    @property
    def spec_array_peak_tops(self) -> float:
        return self.spec_array_peak_gops / 1000.0

    @property
    def default_array_peak_gops(self) -> float:
        """Defaults to measured array rate if available, else spec ceiling."""
        return self.measured_array_peak_gops if self.measured_array_peak_gops is not None else self.spec_array_peak_gops

    @property
    def default_array_peak_tops(self) -> float:
        return self.default_array_peak_gops / 1000.0


# Pre-calculated compute rates at 1.80 GHz
COMPUTE_TABLE: Dict[Precision, ComputeRates] = {
    Precision.INT8_INT8: ComputeRates(
        precision=Precision.INT8_INT8,
        ops_per_mac=2,
        macs_per_cycle_per_core_spec=256,  # [SPEC: device.yaml; docs/SILICON.md:72]
        spec_core_peak_gops=921.6,  # [DERIVED: 256 * 2 * 1.80 GHz]
        spec_column_peak_gops=3686.4,  # [DERIVED: 4 cores * 921.6 GOPS]
        spec_array_peak_gops=14745.6,  # [DERIVED: 16 cores * 921.6 GOPS = 14.7456 TOPS]
        measured_array_peak_gops=4906.23,  # [MEASURED: results/aie/w4a8_array_npu.log:311, 2048^3 int8 upstream]
        measured_k_loop_vmac_per_cycle=0.889,  # [MEASURED: results/aie/w4a8_probe_npu.log:214; docs/SILICON.md:74]
    ),
    Precision.INT8_INT4: ComputeRates(
        precision=Precision.INT8_INT4,
        ops_per_mac=2,
        # 512 is the 4x16x8 vmac shape, not a sustained cycle count (docs/SILICON.md:74)
        macs_per_cycle_per_core_spec=512,  # [SPEC: AIE-API 2024.1; docs/SILICON.md:74]
        spec_core_peak_gops=1843.2,  # [DERIVED: 512 * 2 * 1.80 GHz]
        spec_column_peak_gops=7372.8,  # [DERIVED: 4 cores * 1843.2 GOPS]
        spec_array_peak_gops=29491.2,  # [DERIVED: 16 cores * 1843.2 GOPS = 29.4912 TOPS (SPEC shape ceiling)]
        measured_array_peak_gops=6195.33,  # [MEASURED: results/aie/w4a8_array_npu.log:23, 2048^3 native:unroll2:1]
        measured_k_loop_vmac_per_cycle=0.727,  # [MEASURED: results/aie/w4a8_probe_npu.log:178; 372.4/512 = 0.727 vmac/cyc]
    ),
    Precision.BF16_BF16: ComputeRates(
        precision=Precision.BF16_BF16,
        ops_per_mac=2,
        macs_per_cycle_per_core_spec=128,  # [SPEC: device.yaml; docs/SILICON.md:72]
        spec_core_peak_gops=460.8,  # [DERIVED: 128 * 2 * 1.80 GHz]
        spec_column_peak_gops=1843.2,  # [DERIVED: 4 cores * 460.8 GOPS]
        spec_array_peak_gops=7372.8,  # [DERIVED: 16 cores * 460.8 GOPS = 7.3728 TFLOPS]
        measured_array_peak_gops=2477.23,  # [MEASURED: results/aie/bf16_matmul_n64_single_buffer_npu.log; docs/SILICON.md:1036]
    ),
    Precision.INT16_INT8: ComputeRates(
        precision=Precision.INT16_INT8,
        ops_per_mac=2,
        macs_per_cycle_per_core_spec=128,  # [SPEC: device.yaml; docs/SILICON.md:72]
        spec_core_peak_gops=460.8,
        spec_column_peak_gops=1843.2,
        spec_array_peak_gops=7372.8,  # [DERIVED: 16 cores * 460.8 GOPS = 7.3728 TOPS]
    ),
}


@dataclass(frozen=True)
class RidgePoints:
    """Operational intensity ridge points (OPs/Byte) for each memory boundary."""

    precision: Precision
    l1_ridge_ops_per_byte: float  # [DERIVED: Array Peak / Aggregate L1 BW]
    memtile_ridge_ops_per_byte: float  # [DERIVED: Column Peak / (streams_per_col * 7.20 GB/s)]
    ddr_8ch_read_ridge_ops_per_byte: float  # [DERIVED: Array Peak / 50.36 GB/s]
    fill_transport_ridge_ops_per_byte: float  # [DERIVED: Array Peak / 26.80 GB/s]


def compute_ridge_points(
    hw: HardwareMatrix = HardwareMatrix(),
    use_spec_ceiling: bool = False,
    memtile_streams_per_col: Optional[int] = None,
) -> Dict[Precision, RidgePoints]:
    """Calculate roofline ridge points for each precision against hardware bandwidth tiers."""
    streams_per_col = memtile_streams_per_col if memtile_streams_per_col is not None else hw.memtile_streams_per_col
    col_memtile_bw = streams_per_col * hw.memtile_stream_bandwidth_gbs

    ridge_dict: Dict[Precision, RidgePoints] = {}
    for prec, rates in COMPUTE_TABLE.items():
        array_gops = rates.spec_array_peak_gops if use_spec_ceiling else rates.default_array_peak_gops
        col_gops = rates.spec_column_peak_gops if use_spec_ceiling else (array_gops / hw.reachable_cols)

        l1_ridge = array_gops / hw.l1_aggregate_bandwidth_gbs
        memtile_ridge = col_gops / col_memtile_bw
        ddr_8ch_ridge = array_gops / hw.shim_mm2s_8ch_aggregate_gbs
        fill_ridge = array_gops / hw.fill_transport_rate_gbs

        ridge_dict[prec] = RidgePoints(
            precision=prec,
            l1_ridge_ops_per_byte=round(l1_ridge, 2),
            memtile_ridge_ops_per_byte=round(memtile_ridge, 2),
            ddr_8ch_read_ridge_ops_per_byte=round(ddr_8ch_ridge, 2),
            fill_transport_ridge_ops_per_byte=round(fill_ridge, 2),
        )
    return ridge_dict


@dataclass
class MicroArchEfficiency:
    """Micro-architectural efficiency factors decomposing kernel execution gap."""

    eta_vliw: float = 1.000  # VLIW issue slot density (GEMM: 0.889 i8; 0.727-0.749 w4a8 probe)
    eta_bank: float = 1.000  # Paired-load bank conflict factor 1/(1+p_hazard)
    eta_nest: float = 1.000  # Outer loop nest / spill efficiency (0.75-0.95)
    eta_spatial: float = 1.000  # Spatial array & channel padding utilization (0.40-0.98)

    @property
    def composite_efficiency(self) -> float:
        return self.eta_vliw * self.eta_bank * self.eta_nest * self.eta_spatial


class BottleneckRegime(str, Enum):
    COMPUTE_BOUND = "Compute-Bound"
    DDR_READ_BOUND = "DDR-Read-Bound"
    MEMTILE_STREAM_BOUND = "MemTile-to-Core-Stream-Bound"
    DISPATCH_BOUND = "Host-Dispatch-Bound"


@dataclass
class LatencyPrediction:
    """Analytical latency prediction with distinct DDR and MemTile streaming terms."""

    t_compute_ms: float
    t_ddr_read_ms: float  # Separate DDR transfer term
    t_memtile_to_core_ms: float  # Separate MemTile-to-core stream term
    t_dispatch_ms: float
    t_total_ms: float
    bottleneck: BottleneckRegime
    sustained_tops: float
    peak_tops: float
    hardware_utilization_pct: float
    operational_intensity_ops_per_byte: float


class TheoreticalRooflineModel:
    """Analytical roofline model for AMD Phoenix AIE2."""

    def __init__(
        self,
        hw: HardwareMatrix = HardwareMatrix(),
        use_spec_ceiling_int4: bool = False,
        memtile_streams_per_col: Optional[int] = None,
    ):
        self.hw = hw
        self.use_spec_ceiling_int4 = use_spec_ceiling_int4
        self.memtile_streams_per_col = (
            memtile_streams_per_col if memtile_streams_per_col is not None else hw.memtile_streams_per_col
        )
        self.compute_table = COMPUTE_TABLE
        self.ridge_table = compute_ridge_points(
            hw=hw,
            use_spec_ceiling=use_spec_ceiling_int4,
            memtile_streams_per_col=self.memtile_streams_per_col,
        )

    def get_peak_tops(self, precision: Precision) -> float:
        if precision == Precision.INT8_INT4 and not self.use_spec_ceiling_int4:
            return self.compute_table[precision].default_array_peak_tops
        return self.compute_table[precision].spec_array_peak_tops

    def get_ridge_points(self, precision: Precision) -> RidgePoints:
        return self.ridge_table[precision]

    def predict_layer_or_model(
        self,
        total_macs: int,
        ddr_read_bytes: int,
        memtile_stream_bytes: int,
        precision: Precision = Precision.INT8_INT8,
        efficiency: Optional[MicroArchEfficiency] = None,
        dispatch_overhead_ms: float = 0.080,
    ) -> LatencyPrediction:
        """Predict latency keeping DDR read and MemTile-to-core streaming as separate terms.

        Parameters:
            total_macs: Total multiply-accumulate operations (Ops = 2 * MACs).
            ddr_read_bytes: Bytes read from off-chip DDR across Shim DMA MM2S channels.
            memtile_stream_bytes: Bytes streamed on-chip from MemTile to Core tiles.
            precision: Target numerical format.
            efficiency: Micro-architectural efficiency terms.
            dispatch_overhead_ms: Driver / ring dispatch floor (~80 us).
        """
        if efficiency is None:
            efficiency = MicroArchEfficiency()

        rates = self.compute_table[precision]
        peak_gops = (
            rates.spec_array_peak_gops
            if (precision != Precision.INT8_INT4 or self.use_spec_ceiling_int4)
            else rates.default_array_peak_gops
        )
        total_ops = total_macs * rates.ops_per_mac

        # 1. Compute execution time
        effective_gops = peak_gops * efficiency.composite_efficiency
        t_compute_ms = (total_ops / (effective_gops * 1e9)) * 1000.0

        # 2. Separate DDR read transfer time (via 8-ch MM2S aggregate rate)
        t_ddr_read_ms = (ddr_read_bytes / (self.hw.shim_mm2s_8ch_aggregate_gbs * 1e9)) * 1000.0

        # 3. Separate MemTile-to-core stream time
        # Aggregate MemTile stream bandwidth across 4 columns
        agg_memtile_stream_bw = (
            self.hw.reachable_cols * self.memtile_streams_per_col * self.hw.memtile_stream_bandwidth_gbs
        )
        t_memtile_to_core_ms = (memtile_stream_bytes / (agg_memtile_stream_bw * 1e9)) * 1000.0

        # 4. Host dispatch overhead
        t_dispatch_ms = dispatch_overhead_ms

        # Master analytical formulation: T = max(T_compute, T_ddr, T_memtile) + T_dispatch
        t_hardware_ms = max(t_compute_ms, t_ddr_read_ms, t_memtile_to_core_ms)
        t_total_ms = t_hardware_ms + t_dispatch_ms

        if t_dispatch_ms > t_hardware_ms:
            bottleneck = BottleneckRegime.DISPATCH_BOUND
        elif t_ddr_read_ms >= t_compute_ms and t_ddr_read_ms >= t_memtile_to_core_ms:
            bottleneck = BottleneckRegime.DDR_READ_BOUND
        elif t_memtile_to_core_ms >= t_compute_ms and t_memtile_to_core_ms >= t_ddr_read_ms:
            bottleneck = BottleneckRegime.MEMTILE_STREAM_BOUND
        else:
            bottleneck = BottleneckRegime.COMPUTE_BOUND

        external_bytes = ddr_read_bytes
        op_intensity = (total_ops / external_bytes) if external_bytes > 0 else float("inf")

        sustained_tops = (total_ops / (t_total_ms * 1e-3)) / 1e12
        hw_utilization = (sustained_tops / (peak_gops / 1000.0)) * 100.0

        return LatencyPrediction(
            t_compute_ms=t_compute_ms,
            t_ddr_read_ms=t_ddr_read_ms,
            t_memtile_to_core_ms=t_memtile_to_core_ms,
            t_dispatch_ms=t_dispatch_ms,
            t_total_ms=t_total_ms,
            bottleneck=bottleneck,
            sustained_tops=sustained_tops,
            peak_tops=peak_gops / 1000.0,
            hardware_utilization_pct=hw_utilization,
            operational_intensity_ops_per_byte=op_intensity,
        )


def main():
    parser = argparse.ArgumentParser(description="AMD XDNA1 Theoretical Roofline Ceilings")
    parser.add_argument("--spec-int4", action="store_true", help="Use 29.49 TOPS SPEC shape ceiling for INT4 instead of measured 6.20 TOPS")
    parser.add_argument("--streams-per-col", type=int, default=4, help="MemTile MM2S streams per column (default: 4 from engine_sequence.py:488)")
    args = parser.parse_args()

    hw = HardwareMatrix(memtile_streams_per_col=args.streams_per_col)
    model = TheoreticalRooflineModel(hw=hw, use_spec_ceiling_int4=args.spec_int4, memtile_streams_per_col=args.streams_per_col)

    print("=== AMD XDNA1 (Phoenix) Theoretical Roofline Ceilings ===")
    print(f"Core Clock:              {hw.f_clk_ghz:.2f} GHz [MEASURED: results/aie/clock_probe_npu.log]")
    print(f"Reachable Cores:         {hw.reachable_cores} (4 cols x 4 rows) [SPEC: AIETargetModel.cpp; docs/SILICON.md:62]")
    print(f"On-Chip SRAM:            {hw.total_sram_kb / 1024:.2f} MB ({hw.l1_total_kb} KB L1 + {hw.memtile_total_kb} KB MemTile) [DERIVED]")
    print(f"Shim MM2S 8-ch Read:     {hw.shim_mm2s_8ch_aggregate_gbs:.2f} GB/s [MEASURED: stream_ceiling_phoenix_desktop2_20261013T060213Z.log]")
    print(f"Engine Fill Transport:   {hw.fill_transport_rate_gbs:.2f} GB/s [MEASURED: docs/BENCHMARKS.md:12558; shared cap OPEN]")
    print(f"MemTile Streams / Col:   {args.streams_per_col} streams ({args.streams_per_col * hw.memtile_stream_bandwidth_gbs:.2f} GB/s/col) [DERIVED]\n")

    print(f"{'Precision':<14} | {'Array Peak':<14} | {'L1 Ridge':<12} | {'MemTile Ridge':<14} | {'DDR 8ch Ridge':<14} | Status")
    print("-" * 92)
    for prec in Precision:
        rates = model.compute_table[prec]
        ridge = model.get_ridge_points(prec)
        if prec == Precision.INT8_INT4:
            status = "SPEC shape ceiling" if args.spec_int4 else "MEASURED (w4a8_array_npu.log:23)"
            peak_str = f"{rates.spec_array_peak_tops:.2f} TOPS (SPEC)" if args.spec_int4 else f"{rates.measured_array_peak_gops/1000.0:.2f} TOPS (MEAS)"
        elif prec == Precision.BF16_BF16:
            status = "SPEC (7.37 TFLOPS)"
            peak_str = f"{rates.spec_array_peak_tops:.2f} TFLOPS"
        else:
            status = f"SPEC ({rates.spec_array_peak_tops:.2f} TOPS)"
            peak_str = f"{rates.spec_array_peak_tops:.2f} TOPS"

        print(
            f"{prec.value:<14} | {peak_str:<14} | {ridge.l1_ridge_ops_per_byte:>8.2f} OPs/B | "
            f"{ridge.memtile_ridge_ops_per_byte:>10.2f} OPs/B | {ridge.ddr_8ch_read_ridge_ops_per_byte:>10.2f} OPs/B | {status}"
        )


if __name__ == "__main__":
    main()
