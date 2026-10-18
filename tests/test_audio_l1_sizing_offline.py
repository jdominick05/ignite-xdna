"""Offline unit tests for Audio DSP L1 Buffer & MemTile Sizing in tools/audio_l1_sizing.py.

Validates:
- Hand-computed exact byte allocations per L1 bank against 16 KB boundaries.
- Stack reservation of 3,328 B in Bank 0 (leaving 13,056 B usable buffer space).
- Total weights residency against 2,048 KB MemTile and 3,072 KB aggregate SRAM.
- Hand-computed causal receptive fields in samples and milliseconds.
- fits_in_bank checks on every core buffer partition.
"""

import pytest

from tools.audio_l1_sizing import (
    BANK0_USABLE_BYTES,
    BANK_SIZE_BYTES,
    CORE_L1_TOTAL_BYTES,
    MEMTILE_TOTAL_BYTES,
    REACHABLE_SRAM_TOTAL_BYTES,
    STACK_SIZE_BYTES,
    TOTAL_USABLE_L1_BYTES,
    AudioL1SizingCalculator,
    AudioSizingConfig,
    CoreL1BankAllocation,
)


class TestHardwareMemoryLimits:
    """Verifies physical memory sizing limits and compiler stack deduction."""

    def test_bank_and_stack_constants(self):
        assert CORE_L1_TOTAL_BYTES == 65536  # 64 KB
        assert BANK_SIZE_BYTES == 16384  # 16 KB per bank
        assert STACK_SIZE_BYTES == 3328  # Stack in Bank 0 [docs/SILICON.md:1038]
        assert BANK0_USABLE_BYTES == 16384 - 3328  # 13,056 B
        assert TOTAL_USABLE_L1_BYTES == 65536 - 3328  # 62,208 B
        assert MEMTILE_TOTAL_BYTES == 2048 * 1024  # 2,048 KB (2.0 MB)
        assert REACHABLE_SRAM_TOTAL_BYTES == 3072 * 1024  # 3,072 KB (3.0 MB)

    def test_fits_in_bank_boundary(self):
        # Bank 0: 13,056 B limit
        alloc_exact_b0 = CoreL1BankAllocation(
            bank0_weights_bytes=13056,
            bank1_context_bytes=1000,
            bank2_activation_bytes=1000,
            bank3_scratch_bytes=1000,
        )
        assert alloc_exact_b0.fits_in_bank(0) is True

        alloc_overflow_b0 = CoreL1BankAllocation(
            bank0_weights_bytes=13057,
            bank1_context_bytes=1000,
            bank2_activation_bytes=1000,
            bank3_scratch_bytes=1000,
        )
        assert alloc_overflow_b0.fits_in_bank(0) is False

        # Banks 1..3: 16,384 B limit
        alloc_exact_b1 = CoreL1BankAllocation(
            bank0_weights_bytes=1000,
            bank1_context_bytes=16384,
            bank2_activation_bytes=1000,
            bank3_scratch_bytes=1000,
        )
        assert alloc_exact_b1.fits_in_bank(1) is True

        alloc_overflow_b1 = CoreL1BankAllocation(
            bank0_weights_bytes=1000,
            bank1_context_bytes=16385,
            bank2_activation_bytes=1000,
            bank3_scratch_bytes=1000,
        )
        assert alloc_overflow_b1.fits_in_bank(1) is False


class TestHandComputedReceptiveFields:
    """Verifies analytical causal receptive field formulas against hand-computed values."""

    def test_k3_dilations_1_2_4_8(self):
        # K=3, dilations=(1, 2, 4, 8)
        # RF_samples = 1 + (3 - 1) * (1 + 2 + 4 + 8) = 1 + 2 * 15 = 31 samples
        rf_samples, rf_ms = AudioL1SizingCalculator.compute_receptive_field(
            sample_rate=16000,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
        )
        assert rf_samples == 31
        # 31 / 16000 * 1000 = 1.9375 ms
        assert rf_ms == pytest.approx(1.9375, rel=1e-5)

    def test_k5_dilations_1_2_4(self):
        # K=5, dilations=(1, 2, 4)
        # RF_samples = 1 + (5 - 1) * (1 + 2 + 4) = 1 + 4 * 7 = 29 samples
        rf_samples, rf_ms = AudioL1SizingCalculator.compute_receptive_field(
            sample_rate=48000,
            kernel_size=5,
            dilations=(1, 2, 4),
        )
        assert rf_samples == 29
        # 29 / 48000 * 1000 = 0.604166... ms
        assert rf_ms == pytest.approx(29000.0 / 48000.0, rel=1e-5)


class TestHandComputedByteCounts:
    """Verifies hand-computed weight and buffer byte allocations."""

    def test_hand_computed_model_weights(self):
        cfg = AudioSizingConfig(
            sample_rate=16000,
            frame_ms=10.0,
            channels=64,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            kv_context_frames=32,
        )
        report = AudioL1SizingCalculator.compute_sizing(cfg)

        # Hand calculations:
        # w_enc = 16 * 1 * 64 = 1,024 B
        # w_conv = 4 layers * (3 * 64 * 64) = 49,152 B
        # w_attn = 4 * (64 * 64) = 16,384 B
        # w_dec = 16 * 64 * 1 = 1,024 B
        # Total = 1024 + 49152 + 16384 + 1024 = 67,584 B
        assert report.total_model_weights_bytes == 67584
        assert report.fits_in_memtile_sram() is True
        assert report.fits_in_reachable_sram() is True

    def test_hand_computed_column_allocations(self):
        cfg = AudioSizingConfig(
            sample_rate=16000,
            frame_ms=10.0,  # 160 samples, 320 B PCM16
            channels=64,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            kv_context_frames=32,
        )
        report = AudioL1SizingCalculator.compute_sizing(cfg)

        # Column 0: Analysis Filterbank
        col0 = report.core_allocations[0]
        assert col0.bank0_weights_bytes == 1024
        assert col0.bank1_context_bytes == 320  # 160 samples * 2 B
        assert col0.bank2_activation_bytes == 160 * 64 // 16  # 640 B
        assert col0.bank3_scratch_bytes == 1024
        assert col0.all_banks_fit() is True

        # Column 1: Causal Dilated Convolutions
        col1 = report.core_allocations[1]
        # w_conv // 4 = 49152 // 4 = 12,288 B in Bank 0
        assert col1.bank0_weights_bytes == 12288
        assert col1.fits_in_bank(0) is True  # 12,288 <= 13,056 B
        # conv_history_bytes = (1+2+4+8) * (3-1) * 64 = 15 * 2 * 64 = 1,920 B in Bank 1
        assert col1.bank1_context_bytes == 1920
        assert col1.fits_in_bank(1) is True  # 1,920 <= 16,384 B
        assert col1.bank2_activation_bytes == 64 * 4  # 256 B
        assert col1.bank3_scratch_bytes == 64 * 4  # 256 B
        assert col1.all_banks_fit() is True

        # Column 2: Streaming Causal Attention
        col2 = report.core_allocations[2]
        # w_attn // 4 = 16384 // 4 = 4,096 B in Bank 0
        assert col2.bank0_weights_bytes == 4096
        assert col2.fits_in_bank(0) is True  # 4,096 <= 13,056 B
        # kv_cache_bytes = 2 * 32 * 64 = 4,096 B in Bank 1
        assert col2.bank1_context_bytes == 4096
        assert col2.fits_in_bank(1) is True  # 4,096 <= 16,384 B
        assert col2.bank2_activation_bytes == 256
        assert col2.bank3_scratch_bytes == 512
        assert col2.all_banks_fit() is True

        # Column 3: Synthesis Decoder
        col3 = report.core_allocations[3]
        assert col3.bank0_weights_bytes == 1024
        assert col3.bank1_context_bytes == 640
        assert col3.bank2_activation_bytes == 320
        assert col3.bank3_scratch_bytes == 320
        assert col3.all_banks_fit() is True
