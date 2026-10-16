# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A full candidate must not bypass the completed layer/qualification gates."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_gemm_specialized_full as candidate
import laya_gemm_specialized_compare as compare


def test_complete_layer_gate_rejects_bad_numeric_result(tmp_path):
    path=Path.cwd()/'results/laya/gemm_specialized_mlp_v1_layer_desktop2_20261016.log'
    if not path.exists():pytest.skip('layer evidence unavailable')
    assert candidate.layer_gate(path)==candidate.c.sha(path)
    bad=tmp_path/'bad.log';bad.write_text(path.read_text().replace('"all_arena_bits_exact": true','"all_arena_bits_exact": false'))
    with pytest.raises(ValueError,match='prerequisite failed'):candidate.layer_gate(bad)


def test_comparison_gate_requires_full_model_result(tmp_path):
    path=tmp_path/'incomplete.log';path.write_text('')
    with pytest.raises(ValueError,match='qualification incomplete'):compare.gates(path)


def test_complete_model_gate_rejects_failed_process(tmp_path):
    path=Path.cwd()/'results/laya/gemm_specialized_mlp_v1_full_desktop2_20261016.log'
    if not path.exists():pytest.skip('full development evidence unavailable')
    assert compare.gates(path)==compare.c.sha(path)
    bad=tmp_path/'bad.log';bad.write_text(path.read_text().replace('"exit_code": 0','"exit_code": 1'))
    with pytest.raises(ValueError,match='qualification incomplete'):compare.gates(bad)
