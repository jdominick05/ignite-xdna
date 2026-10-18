#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""L1 Tile Buffer and MemTile SRAM Sizing Calculator for Audio DSP on AMD XDNA1.

Sizing Calculator ONLY: Computes per-core L1 bank allocations, SRAM residency,
and causal receptive fields without latency, RTF, or dispatch claims.

Hardware Constants & Sizing Constraints:
- 64 KB Core Data Memory (L1) per core [SPEC: device.yaml; docs/SILICON.md:70]
- 4 banks of 16 KB (16,384 B) each [SPEC: device.yaml; docs/SILICON.md:86]
- Bank 0 carries the compiler stack of 3,328 B [MEASURED: results/aie/gemm_tile_sweep_c_single_buffer_npu.log; docs/SILICON.md:1038]
  -> Usable Bank 0 buffer budget: 16,384 B - 3,328 B = 13,056 B [DERIVED]
  -> Usable Banks 1, 2, 3 buffer budget: 16,384 B each [SPEC]
  -> Total usable L1 per core: 65,536 B - 3,328 B = 62,208 B [DERIVED]
- 4 MemTiles x 512 KB = 2,048 KB (2,097,152 B) L2 SRAM [SPEC: device.yaml; docs/SILICON.md:104]
- Total reachable on-chip SRAM: 3,072 KB (3,145,728 B) [DERIVED: docs/SILICON.md:70,104]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Hardware SRAM limits
CORE_L1_TOTAL_BYTES: int = 65536  # 64 KB [SPEC: device.yaml; docs/SILICON.md:70]
BANK_SIZE_BYTES: int = 16384  # 16 KB per bank [SPEC: device.yaml; docs/SILICON.md:86]
STACK_SIZE_BYTES: int = 3328  # Stack in Bank 0 [MEASURED: results/aie/gemm_tile_sweep_c_single_buffer_npu.log; docs/SILICON.md:1038]
BANK0_USABLE_BYTES: int = BANK_SIZE_BYTES - STACK_SIZE_BYTES  # 13,056 B [DERIVED]
TOTAL_USABLE_L1_BYTES: int = CORE_L1_TOTAL_BYTES - STACK_SIZE_BYTES  # 62,208 B [DERIVED]

MEMTILE_TOTAL_BYTES: int = 2048 * 1024  # 2,048 KB (2.0 MB) [SPEC: device.yaml; docs/SILICON.md:104]
REACHABLE_SRAM_TOTAL_BYTES: int = 3072 * 1024  # 3,072 KB (3.0 MB) [DERIVED: docs/SILICON.md:70,104]


@dataclass(frozen=True)
class AudioSizingConfig:
    """User-specified parameters for audio feature extraction and temporal modeling."""

    sample_rate: int = 16000  # Samples per second (e.g. 16,000 or 48,000)
    frame_ms: float = 10.0  # Frame duration in milliseconds (e.g. 10.0 or 20.0)
    channels: int = 64  # Feature channel width (C)
    kernel_size: int = 3  # 1D conv kernel width (K)
    dilations: Tuple[int, ...] = (1, 2, 4, 8)  # Dilation rate per layer
    kv_context_frames: int = 32  # Past history frames retained in rolling attention cache
    pcm_bits: int = 16  # Linear PCM bit depth (16-bit)

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0))

    @property
    def frame_pcm_bytes(self) -> int:
        return self.samples_per_frame * (self.pcm_bits // 8)

    @property
    def layers(self) -> int:
        return len(self.dilations)


@dataclass
class CoreL1BankAllocation:
    """Per-core L1 bank byte allocations against physical bank limits."""

    bank0_weights_bytes: int  # Weights tile placed in Bank 0
    bank1_context_bytes: int  # Rolling context / KV cache placed in Bank 1
    bank2_activation_bytes: int  # Output activation frame placed in Bank 2
    bank3_scratch_bytes: int  # Ping-pong / scratch state placed in Bank 3

    stack_bytes: int = STACK_SIZE_BYTES

    def fits_in_bank(self, bank_idx: int) -> bool:
        """Verifies if a specific bank allocation does not exceed physical capacity."""
        if bank_idx == 0:
            return self.bank0_weights_bytes <= BANK0_USABLE_BYTES
        elif bank_idx == 1:
            return self.bank1_context_bytes <= BANK_SIZE_BYTES
        elif bank_idx == 2:
            return self.bank2_activation_bytes <= BANK_SIZE_BYTES
        elif bank_idx == 3:
            return self.bank3_scratch_bytes <= BANK_SIZE_BYTES
        raise ValueError(f"Invalid bank index: {bank_idx}. Must be 0..3.")

    def all_banks_fit(self) -> bool:
        return (
            self.fits_in_bank(0)
            and self.fits_in_bank(1)
            and self.fits_in_bank(2)
            and self.fits_in_bank(3)
        )

    def total_bytes(self) -> int:
        return (
            self.bank0_weights_bytes
            + self.bank1_context_bytes
            + self.bank2_activation_bytes
            + self.bank3_scratch_bytes
            + self.stack_bytes
        )


@dataclass
class AudioDSPReport:
    """Sizing report detailing L1 bank allocations, SRAM residency, and receptive field."""

    config: AudioSizingConfig
    receptive_field_samples: int
    receptive_field_ms: float
    total_model_weights_bytes: int
    core_allocations: Dict[int, CoreL1BankAllocation]

    def fits_in_memtile_sram(self) -> bool:
        return self.total_model_weights_bytes <= MEMTILE_TOTAL_BYTES

    def fits_in_reachable_sram(self) -> bool:
        return self.total_model_weights_bytes <= REACHABLE_SRAM_TOTAL_BYTES


class AudioL1SizingCalculator:
    """Calculator for L1 banking, MemTile weight storage, and temporal receptive fields."""

    @staticmethod
    def compute_receptive_field(
        sample_rate: int,
        kernel_size: int,
        dilations: Tuple[int, ...],
    ) -> Tuple[int, float]:
        """Compute the causal receptive field in samples and milliseconds.

        Formula: 1 + sum((kernel_size - 1) * d for d in dilations)
        """
        rf_samples = 1 + sum((kernel_size - 1) * d for d in dilations)
        rf_ms = (rf_samples / sample_rate) * 1000.0
        return rf_samples, rf_ms

    @classmethod
    def compute_sizing(cls, config: AudioSizingConfig = AudioSizingConfig()) -> AudioDSPReport:
        c = config.channels
        k = config.kernel_size
        dilations = config.dilations
        t_ctx = config.kv_context_frames

        # 1. Receptive field
        rf_samples, rf_ms = cls.compute_receptive_field(config.sample_rate, k, dilations)

        # 2. Total Model Weight Footprint (in INT8 bytes):
        # Analysis encoder 1D conv (K=16, C_in=1, C_out=C): 16 * 1 * C
        w_enc = 16 * 1 * c
        # Causal dilated conv layers (each layer has K * C * C): len(dilations) * (K * C * C)
        w_conv = len(dilations) * (k * c * c)
        # Streaming causal self-attention projections (Q, K, V, Out): 4 * (C * C)
        w_attn = 4 * (c * c)
        # Synthesis decoder 1D conv (K=16, C_in=C, C_out=1): 16 * C * 1
        w_dec = 16 * c * 1
        total_weights = w_enc + w_conv + w_attn + w_dec

        # 3. Per-Column Core L1 Bank Allocations (for 4 columns):
        # Column 0: Analysis Filterbank
        col0_alloc = CoreL1BankAllocation(
            bank0_weights_bytes=w_enc,
            bank1_context_bytes=config.frame_pcm_bytes,  # Input PCM samples
            bank2_activation_bytes=config.samples_per_frame * c // 16,  # Subsampled feature frames
            bank3_scratch_bytes=1024,
        )

        # Column 1: Causal Dilated Convolutions
        # Per-core weight partition (total conv weights divided across 4 compute cores)
        conv_weights_per_core = w_conv // 4
        # Rolling temporal context: receptive field history in frames * C bytes
        conv_history_bytes = (sum(dilations) * (k - 1)) * c
        col1_alloc = CoreL1BankAllocation(
            bank0_weights_bytes=conv_weights_per_core,
            bank1_context_bytes=conv_history_bytes,
            bank2_activation_bytes=c * 4,  # Current output activation quad
            bank3_scratch_bytes=c * 4,
        )

        # Column 2: Streaming Causal Self-Attention
        attn_weights_per_core = w_attn // 4
        # Rolling KV cache: 2 (K and V) * t_ctx frames * C bytes
        kv_cache_bytes = 2 * t_ctx * c
        col2_alloc = CoreL1BankAllocation(
            bank0_weights_bytes=attn_weights_per_core,
            bank1_context_bytes=kv_cache_bytes,
            bank2_activation_bytes=c * 4,  # Attn out
            bank3_scratch_bytes=512,  # Softmax scratch
        )

        # Column 3: Synthesis Decoder
        col3_alloc = CoreL1BankAllocation(
            bank0_weights_bytes=w_dec,
            bank1_context_bytes=config.samples_per_frame * c // 16,  # Feature input
            bank2_activation_bytes=config.frame_pcm_bytes,  # Reconstructed PCM output
            bank3_scratch_bytes=config.frame_pcm_bytes,  # Overlap-add synthesis state
        )

        allocations = {0: col0_alloc, 1: col1_alloc, 2: col2_alloc, 3: col3_alloc}

        return AudioDSPReport(
            config=config,
            receptive_field_samples=rf_samples,
            receptive_field_ms=rf_ms,
            total_model_weights_bytes=total_weights,
            core_allocations=allocations,
        )


def main():
    parser = argparse.ArgumentParser(description="L1 Tile Buffer and MemTile Sizing Calculator for Audio DSP")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--frame-ms", type=float, default=10.0, help="Frame duration in ms (default: 10.0)")
    parser.add_argument("--channels", type=int, default=64, help="Channel dimension C (default: 64)")
    parser.add_argument("--kernel-size", type=int, default=3, help="1D Conv kernel size K (default: 3)")
    parser.add_argument("--dilations", type=str, default="1,2,4,8", help="Comma-separated dilations (default: 1,2,4,8)")
    parser.add_argument("--kv-context", type=int, default=32, help="Rolling attention context frames (default: 32)")
    args = parser.parse_args()

    dilations = tuple(int(x.strip()) for x in args.dilations.split(","))
    config = AudioSizingConfig(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dilations=dilations,
        kv_context_frames=args.kv_context,
    )

    report = AudioL1SizingCalculator.compute_sizing(config)

    print("=== MODEL OUTPUT: Audio DSP L1 Buffer & MemTile Sizing Report ===")
    print(f"Sample Rate:               {config.sample_rate} Hz")
    print(f"Frame Duration:            {config.frame_ms:.1f} ms ({config.samples_per_frame} samples, {config.frame_pcm_bytes} bytes PCM16)")
    print(f"Feature Channels (C):      {config.channels}")
    print(f"Conv Layers:               {len(dilations)} layers (K={config.kernel_size}, dilations={dilations})")
    print(f"Receptive Field:           {report.receptive_field_samples} samples ({report.receptive_field_ms:.2f} ms)")
    print(f"Total Model Weights:       {report.total_model_weights_bytes} bytes ({report.total_model_weights_bytes / 1024:.2f} KB)")
    print(f"Fits in MemTile (2.0 MB):  {report.fits_in_memtile_sram()} (capacity {MEMTILE_TOTAL_BYTES} B)")
    print(f"Fits in Array SRAM (3 MB): {report.fits_in_reachable_sram()} (capacity {REACHABLE_SRAM_TOTAL_BYTES} B)\n")

    print(f"{'Col':<4} | {'Bank 0 (W)':<14} | {'Bank 1 (Ctx)':<14} | {'Bank 2 (Act)':<14} | {'Bank 3 (Scr)':<14} | {'Stack':<8} | {'Total':<9} | Fits All Banks")
    print("-" * 105)
    for col_idx, alloc in report.core_allocations.items():
        b0_str = f"{alloc.bank0_weights_bytes} B"
        b1_str = f"{alloc.bank1_context_bytes} B"
        b2_str = f"{alloc.bank2_activation_bytes} B"
        b3_str = f"{alloc.bank3_scratch_bytes} B"
        stack_str = f"{alloc.stack_bytes} B"
        total_str = f"{alloc.total_bytes()} B"
        fits_str = "YES (fits_in_bank: True)" if alloc.all_banks_fit() else "OVERFLOW"
        print(f"{col_idx:<4} | {b0_str:<14} | {b1_str:<14} | {b2_str:<14} | {b3_str:<14} | {stack_str:<8} | {total_str:<9} | {fits_str}")


if __name__ == "__main__":
    main()
