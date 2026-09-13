#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tests/test_low_level_invariants.py

Low-level invariants of the transaction emitter, the MemTile AGU descriptors,
the .ignite container, the DFL fixed-point arithmetic, the C ingress
preprocessor and the XRT harness lifecycle. Everything here is self-contained:
synthetic transaction templates, in-memory containers and bit-exact Python
models of the kernel arithmetic. Nothing under build/ or models/ is needed and
no device is opened; a silicon check is a separate, measured step.

Run with:  python -m pytest tests/test_low_level_invariants.py -q
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Pin the checkout under test ahead of any installed copy of the package.
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import ignite_xdna  # noqa: E402
from ignite_xdna.compiler import memtile_agu as agu_mod  # noqa: E402
from ignite_xdna.compiler import scheduler as sched  # noqa: E402
from ignite_xdna.compiler import serializer as ser  # noqa: E402
from ignite_xdna.runtime.driver import XrtSiliconHarness  # noqa: E402


def _load_disasm_tool():
    # tools/ has no __init__.py; a regular 'tools' package installed in the
    # environment shadows it, so load the module by path.
    import importlib.util
    path = REPO_ROOT / "tools" / "disasm_txn.py"
    spec = importlib.util.spec_from_file_location("ignite_xdna_disasm_txn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.disassemble_transaction


disassemble_transaction = _load_disasm_tool()


def test_imports_resolve_to_this_checkout():
    """An installed ignite_xdna would silently test another checkout."""
    for mod in (ignite_xdna, sched, agu_mod, ser):
        assert Path(mod.__file__).resolve().is_relative_to(REPO_ROOT), mod.__file__


# ---------------------------------------------------------------------------
# Synthetic XDNA1 transaction template (same op classes and order as the
# single-layer template build/layer_conv0_exec.bin, without its payloads).
# ---------------------------------------------------------------------------
CORES = [(c, r) for c in range(4) for r in range(2, 6)]


def _addr(col, row, reg):
    return (col << 25) | (row << 20) | reg


def _col_row(col, row):
    return (col & 0xFF) | ((row & 0xFF) << 8)


def _write(col, row, reg, value):
    return struct.pack("<6I", 0, _col_row(col, row), _addr(col, row, reg), 0, value, 24)


def _maskwrite(col, row, reg, mask, value):
    return struct.pack("<7I", 3, _col_row(col, row), _addr(col, row, reg), 0, mask, value, 28)


def _blockwrite(col, row, reg, words):
    return struct.pack(f"<4I{len(words)}I", 1, _col_row(col, row), _addr(col, row, reg),
                       16 + 4 * len(words), *words)


def _ddr_patch(col, reg, arg_idx, arg_offset=0):
    return struct.pack("<12I", 0x81, 48, 0, 0, 0, 0, (col << 25) | reg, 0, arg_idx, 0, arg_offset, 0)


def _tct():
    return struct.pack("<4I", 0x80, 16, 0, 0x00010000)


def _stream(ops, num_ops=None, size=None):
    payload = b"".join(ops)
    return struct.pack("<4I", 0, 0, len(ops) if num_ops is None else num_ops,
                       16 + len(payload) if size is None else size) + payload


def synthetic_template(with_tct=True):
    ops = []
    for c, r in CORES:
        ops.append(_maskwrite(c, r, sched.CORE_CONTROL_REG, 0x3, 2))     # core reset
    for c in range(4):
        ops.append(_write(c, 0, 0x14000, 1))
        ops.append(_write(c, 0, 0x14010, 1))
    for c in range(4):
        for lock, value in ((0, 1), (1, 0), (2, 4), (3, 0)):             # MemTile locks
            ops.append(_write(c, 1, sched.memtile_lock_reg(lock), value))
    for c, r in CORES:
        for lock in (0, 2, 4):
            ops.append(_write(c, r, sched.CORE_LOCK_BASE + 0x10 * lock, 1))
    for c, r in CORES:
        ops.append(_blockwrite(c, r, sched.CORE_BD_BASE, [0x80000200, 0, 0, 0, 0, 0]))
        ops.append(_write(c, r, 0x1DE04, 0))
        ops.append(_write(c, r, 0x1DE00, 1))
    for c in range(4):
        ops.append(_blockwrite(c, 1, sched.MEMTILE_BD_BASE, list(range(8))))
        ops.append(_write(c, 1, 0xA0604, 0))
        ops.append(_write(c, 1, 0xA0600, 1))
    for c in range(4):
        ops.append(_blockwrite(c, 0, sched.SHIM_BD_BASE, list(range(16))))
        ops.append(_ddr_patch(c, 0x1D004, 0, c * 2048))
        ops.append(_ddr_patch(c, 0x1D024, 1, c * 1024))
        ops.append(_write(c, 0, 0x1D214, 0))
        ops.append(_write(c, 0, 0x1D210, 1))
        ops.append(_write(c, 0, 0x1D204, 0x80000001))
        ops.append(_write(c, 0, 0x1D200, 1))
    for c in range(4):
        ops.append(_maskwrite(c, 0, 0x1F000, 0xC00, 0xC00))
    for c, r in CORES:
        ops.append(_maskwrite(c, r, sched.CORE_CONTROL_REG, 0x3, 1))     # core enable
    if with_tct:
        ops.append(_tct())
    return _stream(ops)


# core reset, shim locks, MemTile locks, core locks, core BD+queue, MemTile BD+queue,
# shim BD+patches+queue, shim mask, core enable, TCT
TEMPLATE_OPS = 16 + 8 + 16 + 48 + 48 + 12 + 28 + 4 + 16 + 1


def _fake_layers(n, weight_bytes=sched.TEMPLATE_WEIGHT_BYTES):
    rng = np.random.default_rng(n)
    return [SimpleNamespace(
        weights_packed=rng.integers(0, 256, weight_bytes, dtype=np.uint8),
        bias_packed=rng.integers(-1000, 1000, 32, dtype=np.int32),
        shift_cut=4 + k % 3, node_name=f"conv{k}") for k in range(n)]


def _legacy_chain(stage_txns, stage_names):
    """The chained-stream builder as it was before validation was added (byte oracle)."""
    def build_barrier_ops(stage_idx):
        ops_bytes = []
        barrier_reg = 0x1C0060 if stage_idx % 2 == 0 else 0x1C0070
        for c in range(4):
            col_row = (c & 0xFF) | (1 << 8)
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0020, 0, 4, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | barrier_reg, 0, 1, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0040, 0, 1, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0050, 0, 1, 24))
        return b"".join(ops_bytes), 16

    def build_frame_prologue_ops():
        ops_bytes = []
        for c in range(4):
            col_row = (c & 0xFF) | (1 << 8)
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0020, 0, 4, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0060, 0, 0, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0070, 0, 0, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0040, 0, 1, 24))
            ops_bytes.append(struct.pack("<6I", 0, col_row, (c << 25) | (1 << 20) | 0x1C0050, 0, 1, 24))
        return b"".join(ops_bytes), 20

    prologue_bytes, prologue_ops = build_frame_prologue_ops()
    chained = [prologue_bytes]
    total = prologue_ops
    n_stages = len(stage_txns)
    for s_idx, s_b in enumerate(stage_txns):
        ops_data = s_b[16:]
        is_last = s_idx == n_stages - 1
        has_tct = len(ops_data) >= 16 and ops_data[-16:-12] == struct.pack("<I", 0x80)
        if is_last:
            chained.append(ops_data)
            total += struct.unpack("<I", s_b[8:12])[0]
        else:
            if has_tct:
                chained.append(ops_data[:-16])
                total += struct.unpack("<I", s_b[8:12])[0] - 1
            else:
                chained.append(ops_data)
                total += struct.unpack("<I", s_b[8:12])[0]
            bar_bytes, bar_ops = build_barrier_ops(s_idx)
            chained.append(bar_bytes)
            total += bar_ops
    payload = b"".join(chained)
    size = 16 + len(payload)
    rem = size % 64
    pad = b"\x00" * (64 - rem) if rem else b""
    return struct.pack("<4I", 0, 0, total, size + len(pad)) + payload + pad


class TestTransactionStreams:
    def test_parser_matches_disasm_field_by_field(self):
        template = synthetic_template()
        ours = sched.parse_transaction_stream(template)
        theirs = disassemble_transaction(template)
        assert len(ours) == len(theirs) == TEMPLATE_OPS
        for a, b in zip(ours, theirs):
            for key, value in b.items():
                assert a[key] == value, (key, a, b)

    def test_parser_is_count_bounded_and_rejects_truncation(self):
        template = synthetic_template()
        padded = template + b"\x00" * 32
        padded = struct.pack("<4I", 0, 0, TEMPLATE_OPS, len(padded)) + padded[16:]
        assert len(sched.parse_transaction_stream(padded)) == TEMPLATE_OPS
        with pytest.raises((ValueError, struct.error)):
            disassemble_transaction(padded)          # the tool decodes the pad as an op
        with pytest.raises(ValueError):
            sched.parse_transaction_stream(template[:-8])
        bogus = _stream([_write(0, 1, 0xC0000, 1), struct.pack("<4I", 0x77, 16, 0, 0)])
        with pytest.raises(ValueError, match="unknown opcode"):
            sched.parse_transaction_stream(bogus)

    def test_validator_accepts_template(self):
        summary = sched.validate_transaction_stream(synthetic_template(), name="template")
        assert summary["num_ops"] == TEMPLATE_OPS
        assert summary["tct_count"] == 1
        assert summary["pad_bytes"] == 0
        assert summary["op_counts"]["DDR_PATCH"] == 8

    def test_validator_accepts_zero_pad_inside_declared_size(self):
        template = synthetic_template()
        rem = len(template) % 64
        padded = template + b"\x00" * (64 - rem if rem else 0)
        padded = struct.pack("<4I", 0, 0, TEMPLATE_OPS, len(padded)) + padded[16:]
        assert sched.validate_transaction_stream(padded)["pad_bytes"] == (64 - rem) % 64

    @pytest.mark.parametrize("case", [
        "size_mismatch", "no_tct", "program_memory", "data_overrun", "bad_tile",
        "long_tail", "dirty_tail", "patch_target",
    ])
    def test_validator_rejects(self, case):
        template = synthetic_template()
        ops = [_write(0, 1, sched.memtile_lock_reg(2), 4)]
        if case == "size_mismatch":
            bad = template[:-1]
        elif case == "no_tct":
            bad = synthetic_template(with_tct=False)
        elif case == "program_memory":
            bad = _stream(ops + [_blockwrite(0, 2, sched.CORE_PROGRAM_MEMORY_BASE + 0x400, [1] * 64), _tct()])
        elif case == "data_overrun":
            bad = _stream(ops + [_blockwrite(1, 3, 0xFF00, [1] * 128), _tct()])
        elif case == "bad_tile":
            bad = _stream(ops + [_write(5, 2, 0x1F000, 1), _tct()])
        elif case == "long_tail":
            bad = template + b"\x00" * 64
            bad = struct.pack("<4I", 0, 0, TEMPLATE_OPS, len(bad)) + bad[16:]
        elif case == "dirty_tail":
            bad = template + b"\x00" * 15 + b"\x01"
            bad = struct.pack("<4I", 0, 0, TEMPLATE_OPS, len(bad)) + bad[16:]
        else:
            bad = _stream(ops + [_ddr_patch(0, 0x1F004, 0), _tct()])
        with pytest.raises(ValueError):
            sched.validate_transaction_stream(bad, name=case)

    def test_validator_flags_out_of_memory_parameter_sets(self):
        """Layer 16 at 0x400 + 16 * 0x1000 is the first window outside core data memory."""
        window = 16 * sched.L1_PARAM_STRIDE
        bad = _stream([_blockwrite(0, 2, sched.L1_WEIGHTS_OFFSET + window, [0] * 576), _tct()])
        with pytest.raises(ValueError, match="outside data memory"):
            sched.validate_transaction_stream(bad)
        good = _stream([_blockwrite(0, 2, sched.L1_WEIGHTS_OFFSET + 15 * sched.L1_PARAM_STRIDE, [0] * 576), _tct()])
        sched.validate_transaction_stream(good)

    def test_chained_stream_is_byte_identical_to_legacy_builder(self):
        stages = [synthetic_template() for _ in range(3)] + [synthetic_template(with_tct=False)]
        for n in (1, 2, 3):
            new = sched.chain_stage_transaction_streams(stages[:n])
            assert new == _legacy_chain(stages[:n], None)
        # A stage without a terminal TCT is kept whole when it is not last.
        mixed = [stages[3], stages[0]]
        assert sched.chain_stage_transaction_streams(mixed) == _legacy_chain(mixed, None)

    def test_chained_stream_arithmetic_and_lock_addresses(self):
        n = 4
        stream = sched.chain_stage_transaction_streams([synthetic_template() for _ in range(n)])
        ops = sched.parse_transaction_stream(stream)
        declared = struct.unpack("<I", stream[8:12])[0]
        assert declared == len(ops) == 20 + n * TEMPLATE_OPS - (n - 1) + 16 * (n - 1)
        assert len(stream) % 64 == 0
        assert ops[-1]["op"] == "TCT" and sum(o["op"] == "TCT" for o in ops) == 1
        prologue = ops[:20]
        regs = [o["addr"] & 0xFFFFF for o in prologue]
        assert regs[:5] == [0xC0020, 0xC0060, 0xC0070, 0xC0040, 0xC0050]
        assert [o["val"] for o in prologue[:5]] == [4, 0, 0, 1, 1]
        assert all(((o["addr"] >> 20) & 0x1F) == 1 for o in prologue)
        summary = sched.validate_transaction_stream(stream)
        assert summary["op_counts"]["WRITE"] == n * (8 + 16 + 48 + 32 + 8 + 16) + 20 + 16 * (n - 1)

    def test_chained_stream_rejects_missing_final_tct(self):
        with pytest.raises(ValueError, match="TCT"):
            sched.chain_stage_transaction_streams([synthetic_template(), synthetic_template(with_tct=False)])

    def test_bundle_emits_lock_4_5_init_and_skips_params_in_exec(self, tmp_path):
        n = 3
        plan = sched.MemTileMultiPassScheduler().schedule(_fake_layers(n))
        template_path = tmp_path / "template.bin"
        template_path.write_bytes(synthetic_template())
        init_path, exec_path = sched.emit_multi_layer_transaction_bundle(
            plan, str(template_path), str(tmp_path / "init.bin"), str(tmp_path / "exec.bin"))
        init_ops = sched.parse_transaction_stream(Path(init_path).read_bytes())
        exec_ops = sched.parse_transaction_stream(Path(exec_path).read_bytes())
        sched.validate_transaction_stream(Path(init_path).read_bytes())
        sched.validate_transaction_stream(Path(exec_path).read_bytes())

        def lock_writes(ops, lock):
            reg = sched.memtile_lock_reg(lock)
            return [o for o in ops if o["op"] == "WRITE" and ((o["addr"] >> 20) & 0x1F) == 1
                    and (o["addr"] & 0xFFFFF) == reg]

        # The template's 4 Lock-2 inits are each replaced by Lock 2/4/5 writes.
        for ops in (init_ops, exec_ops):
            assert [o["val"] for o in lock_writes(ops, 2)] == [4] * 4
            assert [o["val"] for o in lock_writes(ops, 4)] == [1] * 4
            assert [o["val"] for o in lock_writes(ops, 5)] == [0] * 4
        param_regs = {sched.L1_SHIFT_CUT_OFFSET, sched.L1_BIAS_OFFSET, sched.L1_WEIGHTS_OFFSET}

        def param_writes(ops):
            return [o for o in ops if o["op"] == "BLOCKWRITE" and ((o["addr"] >> 20) & 0x1F) >= 2
                    and ((o["addr"] & 0xFFFFF) % sched.L1_PARAM_STRIDE) in param_regs]

        assert len(param_writes(init_ops)) == 16 * n * 3
        assert param_writes(exec_ops) == []
        # Init = template + 16*n*3 parameter writes + 2 patches per shim BD + 2 extra lock writes per column.
        assert len(init_ops) == TEMPLATE_OPS + 16 * n * 3 + 8 + 8
        assert len(exec_ops) == TEMPLATE_OPS + 8
        # Parameters are injected before the first core enable.
        first_enable = next(i for i, o in enumerate(init_ops)
                            if o["op"] == "MASKWRITE" and (o["addr"] & 0xFFFFF) == sched.CORE_CONTROL_REG
                            and o["val"] == 1)
        last_param = max(i for i, o in enumerate(init_ops) if o in param_writes(init_ops))
        assert last_param < first_enable

    def test_l1_parameter_layout_limit(self, tmp_path):
        assert sched.max_resident_l1_layers() == 16
        plan16 = sched.MemTileMultiPassScheduler().schedule(_fake_layers(16))
        assert sched.validate_l1_parameter_layout(plan16) == 15 * 0x1000 + 0x400 + 2304
        plan17 = sched.MemTileMultiPassScheduler().schedule(_fake_layers(17))
        with pytest.raises(ValueError, match="core data memory"):
            sched.validate_l1_parameter_layout(plan17)
        template_path = tmp_path / "template.bin"
        template_path.write_bytes(synthetic_template())
        with pytest.raises(ValueError, match="at most 16 sets"):
            sched.emit_multi_layer_transaction_bundle(
                plan17, str(template_path), str(tmp_path / "i.bin"), str(tmp_path / "e.bin"))
        assert not (tmp_path / "i.bin").exists() and not (tmp_path / "e.bin").exists()

    def test_unified_bundle_names_stages_when_refusing(self, tmp_path):
        scheduler = sched.MemTileMultiPassScheduler()
        stages = {}
        for name, n in (("Stem", 7), ("P3", 7), ("P4", 4)):
            base = scheduler.schedule(_fake_layers(n))
            stages[name] = sched.StageSchedulePlan(
                num_layers=n, passes=base.passes, intermediate_ddr_bytes=0,
                total_l1_param_bytes_per_core=base.total_l1_param_bytes_per_core, stage_name=name)
        template_path = tmp_path / "template.bin"
        template_path.write_bytes(synthetic_template())
        with pytest.raises(ValueError, match=r"flattened.*Stem=7, P3=7, P4=4"):
            sched.emit_unified_monolithic_transaction_bundle(
                stages, str(template_path), str(tmp_path / "init.bin"), str(tmp_path / "exec.bin"))
        # Two stages of 7 (14 resident sets) still fit and produce a chained stream.
        small = {k: v for k, v in stages.items() if k != "P4"}
        init_path, exec_path = sched.emit_unified_monolithic_transaction_bundle(
            small, str(template_path), str(tmp_path / "init.bin"), str(tmp_path / "exec.bin"))
        stream = Path(exec_path).read_bytes()
        summary = sched.validate_transaction_stream(stream)
        assert summary["tct_count"] == 1 and len(stream) % 64 == 0
        assert sched.validate_transaction_stream(Path(init_path).read_bytes())["tct_count"] == 1

    def test_lock_helpers_match_absolute_addresses(self):
        assert sched.memtile_lock_reg(2) == 0xC0020
        assert sched.memtile_lock_address(3, 7) == (3 << 25) | (1 << 20) | 0xC0070
        assert sched.memtile_lock_address(1, 2) == (1 << 25) | 0x1C0020  # the former literal, row bit included
        with pytest.raises(ValueError):
            sched.memtile_lock_reg(64)


# ---------------------------------------------------------------------------
# MemTile AGU descriptors
# ---------------------------------------------------------------------------
def _simulate_stream(source_bd, dest_bd, memory):
    """Move words from memory through an MM2S/S2MM pair in traversal order (in place)."""
    src = list(source_bd.iter_word_offsets())
    dst = list(dest_bd.iter_word_offsets())
    assert len(src) == len(dst), (len(src), len(dst))
    words = [bytes(memory[o:o + 4]) for o in src]
    for o, w in zip(dst, words):
        memory[o:o + 4] = w


class TestMemTileAGU:
    def test_zero_step_is_rejected(self):
        agu = agu_mod.MemTileAGU()
        with pytest.raises(ValueError, match="not encodable"):
            agu.synthesize(base_address=0, sizes=(4, 2, 1, 1), steps=(4, 0, 4, 4))
        with pytest.raises(ValueError, match="iteration_step"):
            agu.synthesize(base_address=0, sizes=(4, 1, 1, 1), steps=(4, 4, 4, 4),
                           iteration_count=2, iteration_step=0)
        # A dimension that never advances may carry a zero step.
        bd = agu.synthesize(base_address=0, sizes=(4, 1, 1, 1), steps=(4, 0, 0, 0))
        assert bd.step_words == (1, 1, 1, 1) and bd.sizes == (4, 1, 1, 1)
        assert list(bd.iter_word_offsets()) == [0, 4, 8, 12]

    def test_encoded_step_fields_are_minus_one(self):
        bd = agu_mod.MemTileAGU().synthesize(base_address=64, sizes=(2, 3, 1, 1), steps=(4, 32, 4, 4))
        assert bd.steps[:2] == (0, 7)           # 1 word -> 0, 8 words -> 7
        assert bd.step_words[:2] == (1, 8)
        assert list(bd.iter_word_offsets()) == [64, 68, 96, 100, 128, 132]
        assert bd.memory_bytes == 6 * 4 == len(list(bd.iter_word_offsets())) * 4

    def test_upsample_2x_is_exact_nearest_neighbour(self):
        h, w, c = 4, 6, 8
        agu = agu_mod.MemTileAGU()
        plan = agu.plan_upsample_2x(input_shape=(h, w, c), base_address=0x40000,
                                    destination_address=0x60000)
        assert plan.passes == 4 and plan.transfer_bytes == 4 * h * w * c
        assert agu_mod.assert_disjoint_destinations(plan.dest_bds) == h * w * c
        source_counts = agu_mod.touched_words(plan.source_bds)
        assert set(source_counts.values()) == {4} and len(source_counts) == h * w * c // 4
        memory = bytearray(agu_mod.MEMTILE_BYTES)
        x = np.random.default_rng(3).integers(-128, 128, (h, w, c), dtype=np.int8)
        memory[0x40000:0x40000 + x.nbytes] = x.tobytes()
        for src_bd, dst_bd in zip(plan.source_bds, plan.dest_bds):
            _simulate_stream(src_bd, dst_bd, memory)
        out = np.frombuffer(bytes(memory[0x60000:0x60000 + 4 * x.nbytes]), dtype=np.int8).reshape(2 * h, 2 * w, c)
        np.testing.assert_array_equal(out, np.repeat(np.repeat(x, 2, axis=0), 2, axis=1))

    def test_lateral_concat_scatter_is_exact_and_disjoint(self):
        px, chunks = 10, (16, 32, 16)
        agu = agu_mod.MemTileAGU()
        plan = agu.plan_lateral_concat(base_address=0x60000, spatial_pixels=px, chunk_channels=chunks)
        assert agu_mod.assert_disjoint_destinations(plan.bds) == px * sum(chunks) // 4
        memory = bytearray(agu_mod.MEMTILE_BYTES)
        rng = np.random.default_rng(5)
        parts = [rng.integers(-128, 128, (px, ch), dtype=np.int8) for ch in chunks]
        for k, (part, bd) in enumerate(zip(parts, plan.bds)):
            staging = 0x10000 + k * 0x4000
            memory[staging:staging + part.nbytes] = part.tobytes()
            linear = agu.synthesize(base_address=staging, sizes=(part.shape[1] // 4, px, 1, 1),
                                    steps=(4, part.shape[1], 4, 4))
            _simulate_stream(linear, bd, memory)
        out = np.frombuffer(bytes(memory[0x60000:0x60000 + px * sum(chunks)]), dtype=np.int8).reshape(px, -1)
        np.testing.assert_array_equal(out, np.concatenate(parts, axis=1))

    def test_channel_slice_gathers_exact_channels(self):
        px, total, off, n = 7, 64, 32, 16
        agu = agu_mod.MemTileAGU()
        x = np.random.default_rng(9).integers(-128, 128, (px, total), dtype=np.int8)
        memory = bytearray(agu_mod.MEMTILE_BYTES)
        memory[0x40000:0x40000 + x.nbytes] = x.tobytes()
        bd = agu.channel_slice_bd(base_address=0x40000, spatial_pixels=px, total_channels=total,
                                  channel_offset=off, num_channels=n)
        gathered = b"".join(bytes(memory[o:o + 4]) for o in bd.iter_word_offsets())
        np.testing.assert_array_equal(np.frombuffer(gathered, dtype=np.int8).reshape(px, n), x[:, off:off + n])
        assert bd.address_span[0] == 0x40000 + off      # word-granular offset inside the buffer

    def test_detect_head_egress_box_and_cls_are_disjoint(self):
        plan = agu_mod.MemTileAGU().plan_detect_head_egress(scale_name="P5", spatial_pixels=400)
        assert agu_mod.assert_disjoint_destinations((plan.box_bd, plan.cls_bd)) == 400 * 144 // 4
        assert plan.box_bd.address_span[1] <= 0x40000 + 0x10000

    def test_plan_defaults_stay_inside_their_bank(self):
        agu = agu_mod.MemTileAGU()
        # P4 head: 1600 px * 144 ch = 230,400 B runs from Bank 0 (0x40000) into Bank 1 (0x60000).
        with pytest.raises(ValueError, match="exceeds buffer bounds"):
            agu.plan_detect_head_egress(scale_name="P4", spatial_pixels=1600)
        with pytest.raises(ValueError, match="exceeds buffer bounds"):
            agu.plan_lateral_concat(base_address=0x40000, spatial_pixels=1600, chunk_channels=(64, 80))
        # Explicit bounds keep the previous whole-MemTile behaviour available.
        agu.plan_lateral_concat(base_address=0x40000, spatial_pixels=1600, chunk_channels=(64, 80),
                                buffer_bounds=(0x40000, agu_mod.MEMTILE_BYTES))
        assert agu_mod.bank_bounds(0x4FFFF) == (0x40000, 0x50000)
        assert agu_mod.bank_bounds(0x60000) == (0x60000, 0x70000)
        assert agu_mod.bank_bounds(0x10000) == (0, agu_mod.MEMTILE_BYTES)

    def test_plan_bases_must_be_cacheline_aligned(self):
        agu = agu_mod.MemTileAGU()
        with pytest.raises(ValueError, match="64-byte aligned"):
            agu.channel_slice_bd(base_address=0x40020, spatial_pixels=4, total_channels=32,
                                 channel_offset=0, num_channels=16)
        with pytest.raises(ValueError, match="64-byte aligned"):
            agu.plan_lateral_concat(base_address=0x40010, spatial_pixels=4, chunk_channels=(16, 16))
        with pytest.raises(ValueError, match="64-byte aligned"):
            agu.route_c2f_stage(stage_name="x", spatial_pixels=4, in_channels=32, hidden_channels=16,
                                num_bottlenecks=1, ping_buffer_addr=0x40010)
        # Word-granular offsets inside a buffer stay legal.
        agu.channel_slice_bd(base_address=0x40000, spatial_pixels=4, total_channels=32,
                             channel_offset=4, num_channels=16)

    def test_channel_bd_ownership_parity_rule(self):
        for channel, bd_id in ((0, 0), (0, 23), (1, 24), (2, 3), (3, 25), (4, 5), (5, 47)):
            agu_mod.validate_channel_bd(channel, bd_id)
        for channel, bd_id in ((1, 3), (0, 24), (3, 0), (5, 5), (4, 24)):
            with pytest.raises(ValueError, match="own BD"):
                agu_mod.validate_channel_bd(channel, bd_id)
        bd = agu_mod.MemTileAGU().synthesize(base_address=0, sizes=(4, 1, 1, 1), steps=(4, 4, 4, 4))
        assert bd.queue_word(24, channel=1) == 24
        with pytest.raises(ValueError):
            bd.queue_word(24, channel=0)

    def test_receptive_fields_still_compile(self):
        plan = agu_mod.MemTileAGU().receptive_fields((8, 8, 8), kernel_size=3, padding=1, base_address=0)
        assert plan.output_shape == (8, 8, 8)
        assert plan.transfer_bytes == plan.logical_bytes
        for region in plan.regions:
            list(region.bd.iter_word_offsets())


# ---------------------------------------------------------------------------
# .ignite container
# ---------------------------------------------------------------------------
def _legacy_single_relayout_overruns(manifest, blobs):
    """True when the pre-fix writer would have written a manifest longer than its pad."""
    m = dict(manifest)
    padded = ser.align_up(len(json.dumps(m, indent=2).encode()))

    def table(start):
        entries, cur = [], start
        for name, data in blobs:
            cur = ser.align_up(cur)
            entries.append({"name": name, "offset": cur, "size": len(data), "content_type": "raw", "crc32": 0})
            cur += ser.align_up(len(data))
        return entries

    m["blobs"] = table(64 + padded)
    final = json.dumps(m, indent=2).encode()
    if len(final) > padded:
        padded = ser.align_up(len(final))
        m["blobs"] = table(64 + padded)
        final = json.dumps(m, indent=2).encode()
    return len(final) > padded


class TestIgniteContainer:
    def _roundtrip(self, tmp_path, manifest, blobs, name="c.ignite"):
        writer = ser.IgniteModelWriter(manifest)
        for blob_name, data in blobs:
            writer.add_blob(blob_name, data)
        path = tmp_path / name
        total = writer.write(path)
        assert path.stat().st_size == total and total % 64 == 0
        with ser.IgniteModelReader(path) as reader:
            assert reader.verify_checksum()
            assert all(reader.verify_all_blobs().values())
            assert reader.header.blob_offset == 64 + ser.align_up(reader.header.manifest_size)
            for blob_name, data in blobs:
                entry = reader.blobs[blob_name]
                assert entry.offset % 64 == 0 and entry.offset >= reader.header.blob_offset
                assert reader.get_blob_bytes(blob_name) == data
        return path

    def test_layout_stress_round_trips(self, tmp_path):
        rng = np.random.default_rng(11)
        for trial in range(40):
            n = int(rng.integers(0, 12))
            blobs = [(f"blob_{trial}_{i}_{'x' * int(rng.integers(0, 40))}", rng.bytes(int(rng.integers(0, 5000))))
                     for i in range(n)]
            manifest = {"model_name": "t", "note": "n" * int(rng.integers(0, 300)), "stages": {"s": {"index": 0}}}
            self._roundtrip(tmp_path, manifest, blobs, name=f"t{trial}.ignite")

    def test_layout_fixed_point_covers_digit_growth(self, tmp_path):
        """Find a configuration where a single re-layout overruns the manifest pad, then round-trip it."""
        found = None
        for note in range(0, 512):
            for first in (8000, 8192, 8704, 9216):
                blobs = [("a", b"\x01" * first)] + [(f"b{i}", b"\x02" * 640) for i in range(6)]
                manifest = {"model_name": "digits", "note": "n" * note}
                if _legacy_single_relayout_overruns(manifest, blobs):
                    found = (manifest, blobs)
                    break
            if found:
                break
        assert found is not None, "no digit-growth configuration found in the search space"
        self._roundtrip(tmp_path, *found, name="digits.ignite")

    def test_writer_rejects_duplicate_and_empty_names(self):
        writer = ser.IgniteModelWriter({})
        writer.add_blob("a", b"1")
        with pytest.raises(ValueError, match="Duplicate"):
            writer.add_blob("a", b"2")
        with pytest.raises(ValueError):
            writer.add_blob("", b"3")

    def test_reader_rejects_truncation_and_bad_directory(self, tmp_path):
        path = self._roundtrip(tmp_path, {"model_name": "m"}, [("a", b"\x11" * 1234), ("b", b"\x22" * 100)])
        data = path.read_bytes()
        with pytest.raises(ValueError, match="truncated"):
            ser.IgniteModelReader(data[:-1])
        with pytest.raises(ValueError, match="truncated"):
            ser.IgniteModelReader(data + b"\x00")
        # Same-length edit of the directory: blob "b" claims 9999 bytes at its offset.
        tampered = data.replace(b'"size": 100,', b'"size": 999,', 1)
        assert len(tampered) == len(data)
        with pytest.raises(ValueError, match="outside the blob section"):
            ser.IgniteModelReader(tampered)
        bad_version = bytearray(data)
        bad_version[4:6] = struct.pack("<H", 2)
        with pytest.raises(ValueError, match="version"):
            ser.IgniteModelReader(bytes(bad_version))

    def test_blob_crc_localises_tamper(self, tmp_path):
        path = self._roundtrip(tmp_path, {"model_name": "m"}, [("a", b"\x11" * 1234), ("b", b"\x22" * 100)])
        data = bytearray(path.read_bytes())
        with ser.IgniteModelReader(bytes(data)) as reader:
            offset = reader.blobs["b"].offset
        data[offset + 3] ^= 0xFF
        with ser.IgniteModelReader(bytes(data)) as reader:
            assert reader.verify_blob("a") and not reader.verify_blob("b")
            assert not reader.verify_checksum()

    def test_header_layout_matches_native_struct(self):
        assert struct.calcsize(ser.HEADER_STRUCT_FORMAT) == 64 == ser.HEADER_SIZE
        hdr = ser.IgniteHeader(total_file_size=640, manifest_size=10, blob_offset=128, blob_size=512, num_blobs=1)
        packed = hdr.pack()
        assert packed[:4] == b"IGNT" and struct.unpack_from("<H", packed, 4)[0] == 1
        assert ser.IgniteHeader.unpack(packed) == hdr


# ---------------------------------------------------------------------------
# DFL fixed-point arithmetic (bit-exact model of kernels/aie2/dfl/dfl_decode.cc)
# ---------------------------------------------------------------------------
Q15_ONE = 32767


def _sat16(x):
    return max(-32768, min(32767, int(x)))


def _wrap16(x):
    return ((int(x) + 32768) & 0xFFFF) - 32768


def _mul_q15(a, b):
    """aie::mul(int16, int16).to_vector<int16>(15): round to nearest even, saturate."""
    product = int(a) * int(b)
    q, rem = divmod(abs(product), 1 << 15)
    if rem > (1 << 14) or (rem == (1 << 14) and (q & 1)):
        q += 1
    return _sat16(-q if product < 0 else q)


def _reciprocal_fixed(denominator, numerator_bits):
    return 0 if denominator == 0 else (1 << numerator_bits) // int(denominator)


def _exp_negative_q15(zs):
    out = []
    for z in zs:
        z = max(0, int(z))
        z_q8 = z * 16
        q = (z * 23 + 128) >> 8
        while q * 177 > z_q8:
            q -= 1
        while (q + 1) * 177 <= z_q8:
            q += 1
        q = max(q, 0)
        residual = (q * 177 - z_q8) * 128
        r2 = _mul_q15(residual, residual)
        r3 = _mul_q15(r2, residual)
        r4 = _mul_q15(r2, r2)
        poly = _sat16(_sat16(_sat16(Q15_ONE + residual) + _mul_q15(r2, 16384))
                      + _sat16(_mul_q15(r3, 5461) + _mul_q15(r4, 1365)))
        out.append(0 if q >= 15 else poly >> q)
    return out


def _dfl_expectation(logits, clamp):
    logits = [int(v) for v in logits]
    exp = _exp_negative_q15([max(logits) - v for v in logits])
    total = sum(exp)
    reduced = (total >> 4) if total > 16 else 1
    reciprocal = _reciprocal_fixed(reduced, 26)
    if clamp:
        reciprocal = min(reciprocal, Q15_ONE)
    reciprocal16 = _wrap16(reciprocal)              # static_cast<int16_t>
    probabilities = [_mul_q15(e, reciprocal16) for e in exp]
    return sum(k * p for k, p in enumerate(probabilities)) / 32768.0


def _sigmoid_q15(logit, clamp):
    magnitude = abs(int(logit))
    exp = _exp_negative_q15([magnitude])[0]
    inverse = _reciprocal_fixed(32768 + exp, 30)
    if clamp:
        inverse = min(inverse, Q15_ONE)
    if logit >= 0:
        return _wrap16(inverse) / 32768.0
    return _wrap16((exp * inverse + (1 << 14)) >> 15) / 32768.0


def _float_expectation(logits):
    x = np.asarray(logits, dtype=np.float32) / np.float32(16)
    p = np.exp(x - x.max())
    return float(np.dot(p / p.sum(), np.arange(16, dtype=np.float32)))


class TestDflFixedPoint:
    def test_lone_maximum_bin_overflows_int16_without_clamp(self):
        """Only the maximum bin survives: sum >> 4 = 2047, 2^26 // 2047 = 32784 > int16."""
        for k in range(16):
            logits = [-128] * 16
            logits[k] = 127
            unclamped = _dfl_expectation(logits, clamp=False)
            clamped = _dfl_expectation(logits, clamp=True)
            assert abs(clamped - k) < 1e-3, (k, clamped)
            if k:
                assert unclamped < 0, (k, unclamped)        # sign flip before the fix
            assert abs(_float_expectation(logits) - k) < 1e-4

    def test_second_overflow_denominator_reachable(self):
        """sum in [32768, 32783] (reduced sum 2048, reciprocal exactly 32768) is reachable from int8."""
        hits = []
        for z in range(0, 256):
            e = _exp_negative_q15([z])[0]
            if 1 <= e <= 16:
                hits.append((z, e))
        assert hits, "no second-bin magnitude yields 1..16 Q15 counts"
        z, e = hits[0]
        logits = [-128] * 16
        logits[0] = 127
        logits[3] = 127 - z
        assert 127 - z >= -128
        exp = _exp_negative_q15([max(logits) - v for v in logits])
        assert 32768 <= sum(exp) <= 32783 and (sum(exp) >> 4) == 2048
        assert _reciprocal_fixed(2048, 26) == 32768 and _wrap16(32768) == -32768
        assert _dfl_expectation(logits, clamp=False) < 0 < _dfl_expectation(logits, clamp=True)

    def test_clamped_expectation_tracks_float_reference(self):
        rng = np.random.default_rng(17)
        worst = 0.0
        for logits in rng.integers(-128, 128, (3000, 16), dtype=np.int16):
            worst = max(worst, abs(_dfl_expectation(logits, True) - _float_expectation(logits)))
        assert worst < 0.06, worst
        # Sanity: the clamp only acts on the overflow cases (all other inputs are unchanged).
        for logits in rng.integers(-40, 41, (500, 16), dtype=np.int16):
            assert _dfl_expectation(logits, True) == _dfl_expectation(logits, False)

    def test_exponent_argument_is_never_positive(self):
        """z = max - logit >= 0 for every int8 pair; the exp polynomial never sees a positive argument."""
        for a in (-128, -1, 0, 1, 127):
            for b in (-128, -1, 0, 1, 127):
                assert max(a, b) - min(a, b) >= 0
        assert max(_exp_negative_q15(list(range(0, 256)))) == Q15_ONE  # z = 0 gives exactly one

    def test_sigmoid_has_no_int8_overflow_but_is_clamped_anyway(self):
        worst = 0.0
        for logit in range(-128, 128):
            clamped = _sigmoid_q15(logit, True)
            assert clamped == _sigmoid_q15(logit, False)
            worst = max(worst, abs(clamped - 1.0 / (1.0 + np.exp(-logit / 16.0))))
        assert worst < 0.02, worst
        assert _reciprocal_fixed(32768 + 0, 30) == 32768   # the latent case: an exp of 0 would wrap

    def test_kernel_source_carries_the_clamps_and_model_constants(self):
        text = (REPO_ROOT / "kernels/aie2/dfl/dfl_decode.cc").read_text(encoding="utf-8")
        assert "if (reciprocal > kQ15One)" in text
        assert "if (inverse > kQ15One)" in text
        for constant in ("z * 23 + 128", "q * 177", "5461", "1365", "16384", "reciprocal_fixed(reduced_sum, 26)",
                         "reciprocal_fixed(denominator, 30)"):
            assert constant in text, constant

    def test_anchor_partition_has_no_tail(self):
        assert 8400 == 16 * 35 * 15
        # A 15-anchor chunk can straddle the P3/P4 and P4/P5 boundaries; geometry is per anchor.
        assert 6400 % 15 != 0 and 8000 % 15 != 0


# ---------------------------------------------------------------------------
# C ingress preprocessor (compiled from the checkout into a scratch DLL)
# ---------------------------------------------------------------------------
VCVARS = [
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
]


def _compile_preprocess_dll(work: Path) -> Path | None:
    source = REPO_ROOT / "src/ignite_xdna/pipelines/preprocess_simd.c"
    (work / "preprocess_simd.c").write_bytes(source.read_bytes())
    if platform.system() == "Windows":
        for vcvars in VCVARS:
            if not vcvars.exists():
                continue
            # Same flags as ignite_xdna.pipelines.preprocess._compile_simd_dll, built in a scratch dir.
            bat = work / "build.bat"
            bat.write_text(
                "@echo off\r\n"
                f'call "{vcvars}" >nul\r\n'
                f'cd /d "{work}"\r\n'
                "cl.exe /nologo /O2 /fp:fast /openmp /LD preprocess_simd.c /Fe:preprocess_simd_test.dll\r\n",
                encoding="ascii")
            res = subprocess.run(["cmd.exe", "/c", str(bat)], capture_output=True, text=True, timeout=120)
            out = work / "preprocess_simd_test.dll"
            if res.returncode == 0 and out.exists():
                return out
    for cc in ("clang", "gcc"):
        out = work / "preprocess_simd_test.so"
        try:
            res = subprocess.run([cc, "-O3", "-shared", "-fPIC", "-fopenmp", str(work / "preprocess_simd.c"),
                                  "-o", str(out)], capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            continue
        if res.returncode == 0 and out.exists():
            return out
    return None


@pytest.fixture(scope="module")
def preprocess_lib(tmp_path_factory):
    work = tmp_path_factory.mktemp("preprocess")
    dll = _compile_preprocess_dll(work)
    if dll is None:
        pytest.skip("no C compiler for preprocess_simd.c")
    lib = ctypes.CDLL(str(dll))
    fn = lib.fused_preprocess_bgr_to_chw_int8
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                   ctypes.POINTER(ctypes.c_float)]
    fn.restype = ctypes.c_int
    return fn


def _run_preprocess(fn, img, dst_w, dst_h, stride=None):
    img = np.ascontiguousarray(img)
    src_h, src_w = img.shape[:2]
    stride = img.strides[0] if stride is None else stride
    dst = np.full(3 * dst_w * dst_h, 99, dtype=np.int8)
    pad_top, pad_left, scale = ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_float(0)
    rc = fn(img.ctypes.data, src_w, src_h, stride, dst.ctypes.data, dst_w, dst_h,
            ctypes.byref(pad_top), ctypes.byref(pad_left), ctypes.byref(scale))
    return rc, dst.reshape(3, dst_h, dst_w), pad_top.value, pad_left.value, scale.value


def _reference_preprocess(img, dst_w, dst_h):
    """The C kernel's Q11 arithmetic in NumPy (float32 letterbox maths, integer filtering)."""
    f32 = np.float32
    src_h, src_w = img.shape[:2]
    scale = min(f32(dst_w) / f32(src_w), f32(dst_h) / f32(src_h))
    nw = min(int(np.floor(f32(src_w) * scale + f32(0.5))), dst_w)
    nh = min(int(np.floor(f32(src_h) * scale + f32(0.5))), dst_h)
    pad_top, pad_left = (dst_h - nh) // 2, (dst_w - nw) // 2

    def table(n, size):
        f = f32(size) / f32(n)
        s = (np.arange(n, dtype=f32) + f32(0.5)) * f - f32(0.5)
        i0 = np.floor(s).astype(np.int64)
        i0 = np.maximum(i0, 0)
        i1 = np.minimum(i0 + 1, size - 1)
        alpha = np.clip(s - i0.astype(f32), f32(0), f32(1))
        w1 = np.floor(alpha * f32(2048) + f32(0.5)).astype(np.int64)
        return i0, i1, 2048 - w1, w1

    x0, x1, bx0, bx1 = table(nw, src_w)
    y0, y1, by0, by1 = table(nh, src_h)
    p = img.astype(np.int64)
    top = (p[y0][:, x0] * bx0[None, :, None] + p[y0][:, x1] * bx1[None, :, None] + 1024) >> 11
    bottom = (p[y1][:, x0] * bx0[None, :, None] + p[y1][:, x1] * bx1[None, :, None] + 1024) >> 11
    v = (top * by0[:, None, None] + bottom * by1[:, None, None] + 1024) >> 11
    v = np.clip(v, 0, 255)
    out = np.full((3, dst_h, dst_w), -14, dtype=np.int8)
    for plane, channel in ((0, 2), (1, 1), (2, 0)):        # BGR -> RGB planes
        out[plane, pad_top:pad_top + nh, pad_left:pad_left + nw] = (v[:, :, channel] - 128).astype(np.int8)
    return out, pad_top, pad_left, float(scale)


class TestPreprocessSimd:
    def test_rejects_short_stride_and_bad_arguments(self, preprocess_lib):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert _run_preprocess(preprocess_lib, img, 16, 16, stride=8 * 3 - 1)[0] == -3
        assert _run_preprocess(preprocess_lib, img, 0, 16)[0] == -1

    def test_letterbox_geometry_1080p(self, preprocess_lib):
        img = np.random.default_rng(1).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        rc, out, pad_top, pad_left, scale = _run_preprocess(preprocess_lib, img, 640, 640)
        assert rc == 0 and (pad_top, pad_left) == (140, 0)
        assert abs(scale - 1 / 3) < 1e-6
        assert np.all(out[:, :140, :] == -14) and np.all(out[:, 500:, :] == -14)
        assert not np.all(out[:, 140:500, :] == -14)

    def test_matches_numpy_model_of_the_kernel(self, preprocess_lib):
        rng = np.random.default_rng(2)
        for src_h, src_w, dst in ((97, 61, 640), (640, 640, 640), (33, 129, 96), (300, 400, 320)):
            img = rng.integers(0, 256, (src_h, src_w, 3), dtype=np.uint8)
            rc, out, pad_top, pad_left, scale = _run_preprocess(preprocess_lib, img, dst, dst)
            ref, ref_top, ref_left, ref_scale = _reference_preprocess(img, dst, dst)
            assert rc == 0 and (pad_top, pad_left) == (ref_top, ref_left)
            diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
            assert diff.max() <= 1, (src_h, src_w, dst, diff.max())
            assert np.count_nonzero(diff) <= out.size // 1000, "more than 0.1% of pixels differ from the model"

    def test_target_widths_beyond_1024_use_the_heap_table(self, preprocess_lib):
        img = np.random.default_rng(4).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        rc, out, pad_top, pad_left, scale = _run_preprocess(preprocess_lib, img, 1280, 1280)
        assert rc == 0 and (pad_top, pad_left) == (280, 0)
        assert np.all(out[:, :280, :] == -14) and not np.all(out[:, 280:1000, :] == -14)

    def test_report_opencv_bilinear_agreement(self, preprocess_lib, capsys):
        """Informational: the header claims OpenCV INTER_LINEAR parity; measure it, do not assert it."""
        cv2 = pytest.importorskip("cv2")
        img = np.random.default_rng(6).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        rc, out, pad_top, pad_left, scale = _run_preprocess(preprocess_lib, img, 640, 640)
        nh, nw = 480, 640
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ours = (out[:, pad_top:pad_top + nh, pad_left:pad_left + nw].astype(np.int16) + 128).astype(np.uint8)
        theirs = resized[:, :, ::-1].transpose(2, 0, 1)
        diff = np.abs(ours.astype(np.int16) - theirs.astype(np.int16))
        with capsys.disabled():
            print(f"\n[preprocess] identity resize vs cv2.INTER_LINEAR: max|d|={diff.max()} "
                  f"mismatches={np.count_nonzero(diff)}/{diff.size}")
        img2 = np.random.default_rng(7).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        rc, out2, pad_top2, pad_left2, scale2 = _run_preprocess(preprocess_lib, img2, 640, 640)
        resized2 = cv2.resize(img2, (640, 360), interpolation=cv2.INTER_LINEAR)
        ours2 = (out2[:, pad_top2:pad_top2 + 360, :].astype(np.int16) + 128).astype(np.uint8)
        theirs2 = resized2[:, :, ::-1].transpose(2, 0, 1)
        diff2 = np.abs(ours2.astype(np.int16) - theirs2.astype(np.int16))
        with capsys.disabled():
            print(f"[preprocess] 1920x1080->640x360 vs cv2.INTER_LINEAR: max|d|={diff2.max()} "
                  f"mismatches={np.count_nonzero(diff2)}/{diff2.size}")


# ---------------------------------------------------------------------------
# XRT harness lifecycle (bookkeeping only; no device is opened)
# ---------------------------------------------------------------------------
class TestHarnessLifecycle:
    def test_close_releases_kernel_context_xclbin_device_in_order(self):
        order = []

        class Sentinel:
            def __init__(self, name):
                self.name = name

            def __del__(self):
                order.append(self.name)

        h = XrtSiliconHarness.__new__(XrtSiliconHarness)
        h.pyxrt, h.device_idx = None, 0
        h.kernel, h.context, h.xclbin, h.dev = (Sentinel("kernel"), Sentinel("context"),
                                              Sentinel("xclbin"), Sentinel("device"))
        h.close()
        assert order == ["kernel", "context", "xclbin", "device"]
        assert h.kernel is None and h.context is None and h.xclbin is None and h.dev is None
        h.close()          # idempotent

    def test_context_manager_closes(self):
        h = XrtSiliconHarness.__new__(XrtSiliconHarness)
        h.pyxrt, h.device_idx = None, 0
        h.kernel = h.context = h.xclbin = h.dev = object()
        with h as inside:
            assert inside is h
        assert h.dev is None
