#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Analytical Model for INT4xINT8 Weight-Only LLM/SLM Prefill on AMD XDNA1 (Phoenix).

DISCLAIMER: ALL NUMBERS COMPUTED OR PRINTED BY THIS FILE ARE MODEL OUTPUTS,
NOT HARDWARE MEASUREMENTS, EXCEPT WHERE EXPLICITLY CITING A LOG UNDER results/aie/.

Hardware Constants & References:
- 16 reachable compute cores @ 1.80 GHz [MEASURED: results/aie/clock_probe_npu.log]
- 3.0 MB reachable on-chip SRAM (1,024 KB L1 + 2,048 KB MemTile L2) [DERIVED: docs/SILICON.md:70,104]
- INT4xINT8 SPEC Shape Ceiling: 29.49 TOPS (512 MACs/vmac 4x16x8 shape) [SPEC: AIE-API 2024.1; docs/SILICON.md:74]
- INT4xINT8 Measured Array Peak: 6,195.33 GOPS at 2048^3 tile 64/128/64 [MEASURED: results/aie/w4a8_array_npu.log:23]
- INT4xINT8 Measured k-loop MAC rate: 372.4-383.7 MAC/cycle/core (0.727-0.749 vmac/cycle) [MEASURED: results/aie/w4a8_probe_npu.log:178]
- Shim DMA 8-channel MM2S read rate: 50.36 GB/s [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]

Real Model Projection Shapes from Hugging Face config.json:
- google/gemma-3-4b-it: hidden_size=2560, intermediate_size=10240, num_heads=8, num_kv_heads=4, head_dim=256
  q_proj: 2560 -> 2048 (8*256), k_proj: 2560 -> 1024 (4*256), v_proj: 2560 -> 1024 (4*256), o_proj: 2048 -> 2560
  gate_up_proj: 2560 -> 20480 (2*10240), down_proj: 10240 -> 2560
- answerdotai/ModernBERT-base: hidden_size=768, intermediate_size=1152, num_heads=12, head_dim=64
  qkv_proj: 768 -> 2304 (3*768), o_proj: 768 -> 768, wi_proj: 768 -> 2304 (2*1152 GeGLU), wo_proj: 1152 -> 768
- answerdotai/ModernBERT-large: hidden_size=1024, intermediate_size=2624, num_heads=16, head_dim=64
  qkv_proj: 1024 -> 3072 (3*1024), o_proj: 1024 -> 1024, wi_proj: 1024 -> 5248 (2*2624 GeGLU), wo_proj: 2624 -> 1024
- meta-llama/Meta-Llama-3-8B (unsloth/llama-3-8b): hidden_size=4096, intermediate_size=14336, num_heads=32, num_kv_heads=8, head_dim=128
  q_proj: 4096 -> 4096, k_proj: 4096 -> 1024, v_proj: 4096 -> 1024, o_proj: 4096 -> 4096
  gate_up_proj: 4096 -> 28672 (2*14336), down_proj: 14336 -> 4096
- Qwen/Qwen2.5-3B: hidden_size=2048, intermediate_size=11008, num_heads=16, num_kv_heads=2, head_dim=128
  q_proj: 2048 -> 2048, k_proj: 2048 -> 256, v_proj: 2048 -> 256, o_proj: 2048 -> 2048
  gate_up_proj: 2048 -> 22016 (2*11008), down_proj: 11008 -> 2048
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from tools.xdna1_roofline import (
    BottleneckRegime,
    HardwareMatrix,
    MicroArchEfficiency,
    Precision,
    TheoreticalRooflineModel,
)


class ModelArchitecture(str, Enum):
    GEMMA_3_4B = "google/gemma-3-4b-it"
    MODERNBERT_BASE = "answerdotai/ModernBERT-base"
    MODERNBERT_LARGE = "answerdotai/ModernBERT-large"
    LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
    QWEN_2_5_3B = "Qwen/Qwen2.5-3B"


@dataclass(frozen=True)
class LLMProjectionShape:
    """Dimensions of a transformer projection GEMM (K -> N)."""

    name: str
    d_in: int  # K dimension
    d_out: int  # N dimension
    description: str


# Projection shapes verified directly against config.json from Hugging Face
MODEL_PROJECTIONS: Dict[ModelArchitecture, List[LLMProjectionShape]] = {
    ModelArchitecture.GEMMA_3_4B: [
        # Verified from google/gemma-3-4b-it: hidden=2560, intermediate=10240, heads=8, kv_heads=4, head_dim=256
        LLMProjectionShape("q_proj", 2560, 2048, "Query Projection (8 * 256)"),
        LLMProjectionShape("k_proj", 2560, 1024, "Key Projection (4 * 256)"),
        LLMProjectionShape("v_proj", 2560, 1024, "Value Projection (4 * 256)"),
        LLMProjectionShape("o_proj", 2048, 2560, "Attention Output Projection"),
        LLMProjectionShape("gate_up_proj", 2560, 20480, "Fused Gated/Up MLP Projection (2 * 10240)"),
        LLMProjectionShape("down_proj", 10240, 2560, "MLP Down Projection"),
    ],
    ModelArchitecture.MODERNBERT_BASE: [
        # Verified from answerdotai/ModernBERT-base: hidden=768, intermediate=1152, heads=12, head_dim=64
        LLMProjectionShape("qkv_proj", 768, 2304, "Fused QKV Projection (3 * 768)"),
        LLMProjectionShape("o_proj", 768, 768, "Attention Output Projection"),
        LLMProjectionShape("wi_proj", 768, 2304, "GeGLU MLP Input Projection (2 * 1152)"),
        LLMProjectionShape("wo_proj", 1152, 768, "MLP Output Projection"),
    ],
    ModelArchitecture.MODERNBERT_LARGE: [
        # Verified from answerdotai/ModernBERT-large: hidden=1024, intermediate=2624, heads=16, head_dim=64
        LLMProjectionShape("qkv_proj", 1024, 3072, "Fused QKV Projection (3 * 1024)"),
        LLMProjectionShape("o_proj", 1024, 1024, "Attention Output Projection"),
        LLMProjectionShape("wi_proj", 1024, 5248, "GeGLU MLP Input Projection (2 * 2624)"),
        LLMProjectionShape("wo_proj", 2624, 1024, "MLP Output Projection"),
    ],
    ModelArchitecture.LLAMA_3_8B: [
        # Verified from meta-llama/Meta-Llama-3-8B (unsloth/llama-3-8b): hidden=4096, intermediate=14336, heads=32, kv_heads=8, head_dim=128
        LLMProjectionShape("q_proj", 4096, 4096, "Query Projection (32 * 128)"),
        LLMProjectionShape("k_proj", 4096, 1024, "Key Projection (8 * 128)"),
        LLMProjectionShape("v_proj", 4096, 1024, "Value Projection (8 * 128)"),
        LLMProjectionShape("o_proj", 4096, 4096, "Attention Output Projection"),
        LLMProjectionShape("gate_up_proj", 4096, 28672, "Fused Gated/Up MLP Projection (2 * 14336)"),
        LLMProjectionShape("down_proj", 14336, 4096, "MLP Down Projection"),
    ],
    ModelArchitecture.QWEN_2_5_3B: [
        # Verified from Qwen/Qwen2.5-3B: hidden=2048, intermediate=11008, heads=16, kv_heads=2, head_dim=128
        LLMProjectionShape("q_proj", 2048, 2048, "Query Projection (16 * 128)"),
        LLMProjectionShape("k_proj", 2048, 256, "Key Projection (2 * 128)"),
        LLMProjectionShape("v_proj", 2048, 256, "Value Projection (2 * 128)"),
        LLMProjectionShape("o_proj", 2048, 2048, "Attention Output Projection"),
        LLMProjectionShape("gate_up_proj", 2048, 22016, "Fused Gated/Up MLP Projection (2 * 11008)"),
        LLMProjectionShape("down_proj", 11008, 2048, "MLP Down Projection"),
    ],
}


@dataclass
class PrefillRooflineMetrics:
    """Model output metrics for an LLM prefill projection."""

    model: ModelArchitecture
    proj_name: str
    seq_len_m: int
    k_dim: int
    n_dim: int
    total_macs: int

    # Weight storage footprints
    int4_weight_bytes: int
    int8_weight_bytes: int

    # Data movement including SRAM-aware tiling re-reads
    int4_ddr_read_bytes: int
    int8_ddr_read_bytes: int
    weight_re_read_passes: int
    activation_re_read_passes: int

    # Modelled latencies (milliseconds)
    t_compute_int4_ms: float
    t_compute_int8_ms: float
    t_dram_int4_ms: float
    t_dram_int8_ms: float
    t_total_int4_ms: float
    t_total_int8_ms: float

    # Modelled performance
    modelled_tops_int4: float
    modelled_tops_int8: float
    modelled_speedup_int4_over_int8: float
    dram_bandwidth_reduction_pct: float
    bottleneck_int4: BottleneckRegime
    bottleneck_int8: BottleneckRegime


class Int4WeightPacker:
    """Packs and unpacks signed 4-bit integer weights (2 elements per byte, low nibble first)."""

    @staticmethod
    def pack(weights_int4: np.ndarray) -> np.ndarray:
        assert weights_int4.shape[-1] % 2 == 0, "Last dimension must be even"
        clipped = np.clip(weights_int4, -8, 7).astype(np.int8)
        u4 = clipped.astype(np.uint8) & 0x0F
        low = u4[..., 0::2]
        high = u4[..., 1::2]
        packed = (high << 4) | low
        return packed.astype(np.uint8)

    @staticmethod
    def unpack(packed_bytes: np.ndarray) -> np.ndarray:
        orig_shape = list(packed_bytes.shape)
        orig_shape[-1] = orig_shape[-1] * 2
        unpacked = np.zeros(orig_shape, dtype=np.int8)
        low = packed_bytes & 0x0F
        high = (packed_bytes >> 4) & 0x0F
        low_signed = np.where(low >= 8, low - 16, low).astype(np.int8)
        high_signed = np.where(high >= 8, high - 16, high).astype(np.int8)
        unpacked[..., 0::2] = low_signed
        unpacked[..., 1::2] = high_signed
        return unpacked


class Int4MmulReference:
    """NumPy reference implementation of 4x16x8 uint8 x int4 matrix multiply-accumulate.

    NOTE: This is a software reference checked only against NumPy arithmetic.
    Evidence for hardware bit-exactness on Phoenix silicon is documented in
    results/aie/w4a8_probe_npu.log (0 of 4,096 mismatches against int64 reference).
    """

    M: int = 4
    K: int = 8
    N: int = 16

    @classmethod
    def execute(
        cls,
        a_uint8: np.ndarray,
        b_packed_int4: np.ndarray,
        c_acc32: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert a_uint8.shape == (cls.M, cls.K), f"Expected A shape (4, 8), got {a_uint8.shape}"
        assert b_packed_int4.shape == (cls.K, cls.N // 2), f"Expected B packed shape (8, 8), got {b_packed_int4.shape}"
        b_int4 = Int4WeightPacker.unpack(b_packed_int4)
        prod = np.matmul(a_uint8.astype(np.int32), b_int4.astype(np.int32))
        if c_acc32 is not None:
            assert c_acc32.shape == (cls.M, cls.N)
            return c_acc32 + prod
        return prod


class PrefillGEMMEngine:
    """Analytical GEMM prefill engine accounting for on-chip SRAM capacity and tiling re-reads."""

    def __init__(self, hw: HardwareMatrix = HardwareMatrix()):
        self.hw = hw
        self.roofline = TheoreticalRooflineModel(hw=hw)

    def analyze_projection_prefill(
        self,
        model: ModelArchitecture,
        projection: LLMProjectionShape,
        seq_len_m: int = 512,
        ddr_read_bw_gbs: float = 50.36,  # [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]
        dispatch_overhead_ms: float = 0.080,
    ) -> PrefillRooflineMetrics:
        """Model prefill projection accounting for SRAM tiling and operand re-reads."""
        m = seq_len_m
        k = projection.d_in
        n = projection.d_out
        total_macs = m * k * n
        total_ops = 2 * total_macs

        int4_w_bytes = (k * n) // 2
        int8_w_bytes = k * n

        act_in_bytes = m * k  # uint8 activations
        act_out_bytes = m * n  # uint8 outputs

        # SRAM Capacity Accounting (MemTile L2 is 2,048 KB total = 2,097,152 B)
        # S_onchip: available SRAM buffer budget for stationary operand
        s_onchip_bytes = self.hw.memtile_total_kb * 1024  # 2.0 MB

        # Tiling policy:
        # If weights exceed MemTile SRAM, weights cannot be kept stationary.
        # We tile along N: MemTile holds as many columns of W as fit.
        # N_block = min(N, floor(S_onchip / (K * bytes_per_weight)))
        # Then activations A must be streamed once per N_block pass!
        w_int4_tile_n = max(1, s_onchip_bytes // (k // 2))
        w_int8_tile_n = max(1, s_onchip_bytes // k)

        passes_a_int4 = max(1, int(math.ceil(n / w_int4_tile_n)))
        passes_a_int8 = max(1, int(math.ceil(n / w_int8_tile_n)))

        # If activations A exceed on-chip memory, weights W must be re-read across M-passes
        passes_w_int4 = 1
        passes_w_int8 = 1
        if act_in_bytes > s_onchip_bytes:
            m_block = max(1, s_onchip_bytes // k)
            passes_w_int4 = max(1, int(math.ceil(m / m_block)))
            passes_w_int8 = max(1, int(math.ceil(m / m_block)))

        # Total DDR traffic including re-reads
        int4_ddr_bytes = (int4_w_bytes * passes_w_int4) + (act_in_bytes * passes_a_int4) + act_out_bytes
        int8_ddr_bytes = (int8_w_bytes * passes_w_int8) + (act_in_bytes * passes_a_int8) + act_out_bytes

        # DRAM transfer times
        t_dram_int4_ms = (int4_ddr_bytes / (ddr_read_bw_gbs * 1e9)) * 1000.0
        t_dram_int8_ms = (int8_ddr_bytes / (ddr_read_bw_gbs * 1e9)) * 1000.0

        # Compute execution times using measured k-loop issue rates
        # INT4 measured k loop: 0.727 vmac/cycle (372.4 MAC/cycle) [results/aie/w4a8_probe_npu.log:178]
        # INT8 measured k loop: 0.889 vmac/cycle (227.6 MAC/cycle) [results/aie/w4a8_probe_npu.log:214]
        eff_int4 = MicroArchEfficiency(eta_vliw=0.727, eta_bank=1.0, eta_nest=0.90, eta_spatial=0.95)
        eff_int8 = MicroArchEfficiency(eta_vliw=0.889, eta_bank=1.0, eta_nest=0.90, eta_spatial=0.95)

        # Model compute rate using SPEC shape ceilings modulated by measured efficiency
        spec_peak_int4_gops = self.roofline.compute_table[Precision.INT8_INT4].spec_array_peak_gops
        spec_peak_int8_gops = self.roofline.compute_table[Precision.INT8_INT8].spec_array_peak_gops

        t_compute_int4_ms = (total_ops / (spec_peak_int4_gops * eff_int4.composite_efficiency * 1e9)) * 1000.0
        t_compute_int8_ms = (total_ops / (spec_peak_int8_gops * eff_int8.composite_efficiency * 1e9)) * 1000.0

        t_total_int4_ms = max(t_compute_int4_ms, t_dram_int4_ms) + dispatch_overhead_ms
        t_total_int8_ms = max(t_compute_int8_ms, t_dram_int8_ms) + dispatch_overhead_ms

        modelled_tops_int4 = (total_ops / (t_total_int4_ms * 1e-3)) / 1e12
        modelled_tops_int8 = (total_ops / (t_total_int8_ms * 1e-3)) / 1e12

        speedup = t_total_int8_ms / t_total_int4_ms
        bw_reduction = ((int8_ddr_bytes - int4_ddr_bytes) / int8_ddr_bytes) * 100.0

        bot_int4 = BottleneckRegime.COMPUTE_BOUND if t_compute_int4_ms >= t_dram_int4_ms else BottleneckRegime.DDR_READ_BOUND
        bot_int8 = BottleneckRegime.COMPUTE_BOUND if t_compute_int8_ms >= t_dram_int8_ms else BottleneckRegime.DDR_READ_BOUND

        return PrefillRooflineMetrics(
            model=model,
            proj_name=projection.name,
            seq_len_m=m,
            k_dim=k,
            n_dim=n,
            total_macs=total_macs,
            int4_weight_bytes=int4_w_bytes,
            int8_weight_bytes=int8_w_bytes,
            int4_ddr_read_bytes=int4_ddr_bytes,
            int8_ddr_read_bytes=int8_ddr_bytes,
            weight_re_read_passes=passes_w_int4,
            activation_re_read_passes=passes_a_int4,
            t_compute_int4_ms=t_compute_int4_ms,
            t_compute_int8_ms=t_compute_int8_ms,
            t_dram_int4_ms=t_dram_int4_ms,
            t_dram_int8_ms=t_dram_int8_ms,
            t_total_int4_ms=t_total_int4_ms,
            t_total_int8_ms=t_total_int8_ms,
            modelled_tops_int4=modelled_tops_int4,
            modelled_tops_int8=modelled_tops_int8,
            modelled_speedup_int4_over_int8=speedup,
            dram_bandwidth_reduction_pct=bw_reduction,
            bottleneck_int4=bot_int4,
            bottleneck_int8=bot_int8,
        )

    def calibrate_against_w4a8_array_log(
        self,
        m: int = 2048,
        k: int = 2048,
        n: int = 2048,
        l3_traffic_rate_gbs: float = 23.96,  # [MEASURED: results/aie/w4a8_array_npu.log:52]
    ) -> Dict[str, float]:
        """Calibrate model against the best tile from results/aie/w4a8_array_npu.log:23.

        Tile: 64/128/64 at 2048^3. Moves 64 MiB (A 32 MiB, B 16 MiB int4, C 16 MiB).
        Measured time: 2773.0 us -> 6,195.33 GOPS.
        """
        total_ops = 2 * m * k * n  # 17,179,869,184 ops
        l3_bytes = 64 * 1024 * 1024  # 64 MiB = 67,108,864 bytes [w4a8_array_npu.log:34]

        # Modeled time based on measured L3 bandwidth
        t_dram_sec = l3_bytes / (l3_traffic_rate_gbs * 1e9)
        t_dram_ms = t_dram_sec * 1000.0

        # Modeled compute time based on measured k-loop issue rate (372.4 MAC/cycle/core)
        # [MEASURED: results/aie/w4a8_probe_npu.log:178]
        macs_per_sec = 16 * 372.4 * 1.80e9
        ops_per_sec = 2 * macs_per_sec
        t_comp_sec = total_ops / ops_per_sec
        t_comp_ms = t_comp_sec * 1000.0

        t_total_ms = max(t_dram_ms, t_comp_ms)
        predicted_gops = (total_ops / (t_total_ms * 1e-3)) / 1e9

        measured_gops = 6195.33  # [MEASURED: results/aie/w4a8_array_npu.log:23]
        gap_pct = ((predicted_gops - measured_gops) / measured_gops) * 100.0

        return {
            "predicted_gops": predicted_gops,
            "measured_gops": measured_gops,
            "gap_pct": gap_pct,
            "predicted_time_us": t_total_ms * 1000.0,
            "measured_time_us": 2773.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Modelled INT4xINT8 LLM/SLM Prefill on AMD XDNA1")
    parser.add_argument("--prompt-length", type=int, default=512, help="Sequence length M (default: 512)")
    args = parser.parse_args()

    engine = PrefillGEMMEngine()
    print("=== MODEL OUTPUT: INT4xINT8 LLM/SLM Prefill Projection Model ===")
    print("NOTE: ALL FIGURES BELOW ARE MODEL OUTPUTS, NOT ON-DEVICE MEASUREMENTS.\n")

    for model_arch in ModelArchitecture:
        print(f"--- Model: {model_arch.value} (Prompt M={args.prompt_length}) ---")
        for proj in MODEL_PROJECTIONS[model_arch]:
            res = engine.analyze_projection_prefill(model_arch, proj, seq_len_m=args.prompt_length)
            print(
                f"  {proj.name:<13} ({proj.d_in:>5}x{proj.d_out:<5}): "
                f"W_int4={res.int4_weight_bytes / (1024*1024):>5.2f} MB | "
                f"T_int4={res.t_total_int4_ms:>6.3f} ms | "
                f"T_int8={res.t_total_int8_ms:>6.3f} ms | "
                f"Speedup={res.modelled_speedup_int4_over_int8:>4.2f}x | "
                f"DRAM BW Red={res.dram_bandwidth_reduction_pct:>4.1f}% | "
                f"Re-reads (W={res.weight_re_read_passes}, A={res.activation_re_read_passes})"
            )
        print()


if __name__ == "__main__":
    main()
