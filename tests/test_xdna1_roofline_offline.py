"""Offline unit tests for AMD XDNA1 (Phoenix) Roofline Model in tools/xdna1_roofline.py.

Validates:
- Physical hardware constants with strict epistemic tags from docs/SILICON.md.
- Separate DDR MM2S read and MemTile-to-core stream latency terms.
- Parameterized MemTile MM2S streams per column (default 4 from engine_sequence.py:488).
- INT8xINT4 dual ceilings: SPEC shape ceiling (29.49 TOPS) vs MEASURED array ceiling (6.20 TOPS / 6,195.33 GOPS).
- Operational intensity ridge points across all precision modes and memory tiers.
"""

import pytest

from tools.xdna1_roofline import (
    BottleneckRegime,
    HardwareMatrix,
    MicroArchEfficiency,
    Precision,
    TheoreticalRooflineModel,
    compute_ridge_points,
)


class TestHardwareMatrixConstants:
    """Verifies hardware constants against docs/SILICON.md tags."""

    def test_core_clock_and_counts(self):
        hw = HardwareMatrix()
        # [MEASURED: results/aie/clock_probe_npu.log]
        assert hw.f_clk_ghz == 1.80
        # [SPEC: device.yaml; docs/SILICON.md:60]
        assert hw.total_physical_cores == 20
        # [SPEC: AIETargetModel.cpp; docs/SILICON.md:62]
        assert hw.reachable_cols == 4
        assert hw.reachable_rows == 4
        # [DERIVED: 4 cols * 4 rows; docs/SILICON.md:61]
        assert hw.reachable_cores == 16

    def test_sram_capacities(self):
        hw = HardwareMatrix()
        # [SPEC: device.yaml; docs/SILICON.md:70]
        assert hw.l1_kb_per_core == 64
        # [DERIVED: 16 cores * 64 KB; docs/SILICON.md:70]
        assert hw.l1_total_kb == 1024
        # [SPEC: device.yaml; docs/SILICON.md:104]
        assert hw.memtile_count == 4
        assert hw.memtile_kb_per_col == 512
        # [DERIVED: 4 tiles * 512 KB; docs/SILICON.md:104]
        assert hw.memtile_total_kb == 2048
        # [DERIVED: 1024 KB + 2048 KB; docs/SILICON.md:70,104]
        assert hw.total_sram_kb == 3072

    def test_interconnect_and_shim_rates(self):
        hw = HardwareMatrix()
        # [DERIVED: 64 B * 1.80 GHz]
        assert hw.l1_bandwidth_per_core_gbs == pytest.approx(115.2, rel=1e-3)
        assert hw.l1_aggregate_bandwidth_gbs == pytest.approx(1843.2, rel=1e-3)

        # [DERIVED: 4 B * 1.80 GHz; MEASURED: results/aie/silicon_stream_width_desktop2_20260919.log]
        assert hw.memtile_stream_bandwidth_gbs == pytest.approx(7.20, rel=1e-3)
        # Default MemTile streams per col is 4 [DERIVED: src/ignite_xdna/compiler/engine_sequence.py:488]
        assert hw.memtile_streams_per_col == 4

        # [MEASURED: results/aie/stream_ceiling_phoenix_desktop2_20261013T060213Z.log]
        assert hw.shim_mm2s_single_channel_min_gbs == pytest.approx(6.57, rel=1e-3)
        assert hw.shim_mm2s_single_channel_max_gbs == pytest.approx(6.88, rel=1e-3)
        assert hw.shim_mm2s_8ch_aggregate_gbs == pytest.approx(50.36, rel=1e-3)

        # [MEASURED: docs/BENCHMARKS.md:12558; shared cap OPEN, docs/SILICON.md:179]
        assert hw.fill_transport_rate_gbs == pytest.approx(26.80, rel=1e-3)


class TestDualInt4Ceilings:
    """Verifies INT8xINT4 spec shape ceiling vs measured array ceiling."""

    def test_spec_vs_measured_int4_rates(self):
        hw = HardwareMatrix()
        model_default = TheoreticalRooflineModel(hw=hw, use_spec_ceiling_int4=False)
        model_spec = TheoreticalRooflineModel(hw=hw, use_spec_ceiling_int4=True)

        rates = model_default.compute_table[Precision.INT8_INT4]
        # SPEC shape ceiling: 512 MACs/vmac shape * 2 * 1.80 GHz * 16 cores = 29.4912 TOPS
        assert rates.spec_array_peak_tops == pytest.approx(29.4912, rel=1e-3)
        # MEASURED array ceiling: 6,195.33 GOPS (~6.20 TOPS) [results/aie/w4a8_array_npu.log:23]
        assert rates.measured_array_peak_gops == pytest.approx(6195.33, rel=1e-3)
        assert rates.default_array_peak_tops == pytest.approx(6.19533, rel=1e-3)

        # By default, INT4 model peak TOPS must return the measured rate
        assert model_default.get_peak_tops(Precision.INT8_INT4) == pytest.approx(6.19533, rel=1e-3)
        # With flag, returns SPEC ceiling
        assert model_spec.get_peak_tops(Precision.INT8_INT4) == pytest.approx(29.4912, rel=1e-3)


class TestRidgePointsParameterization:
    """Verifies ridge points with parameterized MemTile streams per column."""

    def test_memtile_streams_ridge_scaling(self):
        hw = HardwareMatrix()
        # 1 stream per column (7.20 GB/s)
        ridge_1stream = compute_ridge_points(hw=hw, memtile_streams_per_col=1, use_spec_ceiling=True)
        # 4 streams per column (28.80 GB/s, engine default)
        ridge_4streams = compute_ridge_points(hw=hw, memtile_streams_per_col=4, use_spec_ceiling=True)

        # INT8xINT8 column peak = 3,686.4 GOPS
        # 1 stream: 3686.4 / 7.20 = 512.00 OPs/B
        assert ridge_1stream[Precision.INT8_INT8].memtile_ridge_ops_per_byte == pytest.approx(512.00, abs=0.01)
        # 4 streams: 3686.4 / (4 * 7.20) = 3686.4 / 28.80 = 128.00 OPs/B
        assert ridge_4streams[Precision.INT8_INT8].memtile_ridge_ops_per_byte == pytest.approx(128.00, abs=0.01)

    def test_default_measured_ridge_points(self):
        hw = HardwareMatrix()
        ridge_default = compute_ridge_points(hw=hw, use_spec_ceiling=False)

        # INT8xINT8: array peak 4906.23 GOPS (measured)
        # L1 ridge: 4906.23 / 1843.2 = 2.66 OPs/B
        assert ridge_default[Precision.INT8_INT8].l1_ridge_ops_per_byte == pytest.approx(4906.23 / 1843.2, abs=0.01)
        # DDR 8-ch read ridge: 4906.23 / 50.36 = 97.42 OPs/B
        assert ridge_default[Precision.INT8_INT8].ddr_8ch_read_ridge_ops_per_byte == pytest.approx(4906.23 / 50.36, abs=0.01)


class TestSeparateDDRAndMemTileLatencyTerms:
    """Verifies that DDR read and MemTile-to-core streams are computed with distinct rates."""

    def test_separate_stream_and_ddr_latencies(self):
        model = TheoreticalRooflineModel()
        total_macs = 100_000_000  # 100 MMAC = 200 MOP
        ddr_bytes = 10_000_000  # 10 MB from DDR
        memtile_stream_bytes = 5_000_000  # 5 MB MemTile-to-core stream

        pred = model.predict_layer_or_model(
            total_macs=total_macs,
            ddr_read_bytes=ddr_bytes,
            memtile_stream_bytes=memtile_stream_bytes,
        )

        # DDR rate is 50.36 GB/s
        expected_t_ddr = (10_000_000 / (50.36 * 1e9)) * 1000.0
        assert pred.t_ddr_read_ms == pytest.approx(expected_t_ddr, rel=1e-3)

        # MemTile stream rate is 4 cols * 4 streams/col * 7.20 GB/s = 115.2 GB/s
        expected_t_memtile = (5_000_000 / (115.2 * 1e9)) * 1000.0
        assert pred.t_memtile_to_core_ms == pytest.approx(expected_t_memtile, rel=1e-3)
        assert pred.t_ddr_read_ms != pred.t_memtile_to_core_ms
