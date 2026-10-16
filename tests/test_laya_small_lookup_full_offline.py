# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Block full testing/timing when the corresponding device gate is incomplete."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_small_lookup_full as candidate
import laya_small_lookup_compare as compare


def test_failed_layer_result_blocks_full_candidate(tmp_path):
    path=Path.cwd()/'results/laya/small_lookup_v3_layer_desktop2_20261016.log'
    if not path.exists():pytest.skip('layer evidence unavailable')
    assert candidate.layer_gate(path)==candidate.c.sha(path)
    bad=tmp_path/'bad.log';bad.write_text(path.read_text().replace('"all_arena_bits_exact": true','"all_arena_bits_exact": false'))
    with pytest.raises(ValueError):candidate.layer_gate(bad)


def test_missing_complete_model_gate_blocks_timing(tmp_path):
    path=tmp_path/'incomplete.log';path.write_text('')
    with pytest.raises(ValueError):compare.gates(path)


def test_failed_complete_process_blocks_timing(tmp_path):
    path=Path.cwd()/'results/laya/small_lookup_v3_full_desktop2_20261016.log'
    if not path.exists():pytest.skip('complete development evidence unavailable')
    assert compare.gates(path)==compare.c.sha(path)
    bad=tmp_path/'bad.log';bad.write_text(path.read_text().replace('"exit_code": 0','"exit_code": 1'))
    with pytest.raises(ValueError):compare.gates(bad)
