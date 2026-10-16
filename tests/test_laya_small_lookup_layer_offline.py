# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind complete-layer testing to all four finished primitive profiles."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_small_lookup_layer as layer


def evidence():
    root=Path.cwd();path=root/'results/laya/small_lookup_v3_primitives_desktop2_20261016.log'
    build=root/'scratch/laya_transformer/small_lookup_primitive_v1/build.json'
    if not path.exists() or not build.exists():pytest.skip('primitive evidence unavailable')
    return path,layer.c.sha(build)


def test_all_four_profiles_bind_to_current_primitive_build():
    path,digest=evidence()
    assert layer.gates(path,digest)==layer.c.sha(path)
    with pytest.raises(ValueError,match='prerequisite failed'):layer.gates(path,'0'*64)


def test_missing_output_group_blocks_layer_gate(tmp_path):
    path,digest=evidence();changed=tmp_path/'incomplete.log'
    changed.write_text('\n'.join(line for line in path.read_text().splitlines()
        if not(line.startswith('WINDOW_PRIMITIVE_RESULT ') and 'original1' in line)))
    with pytest.raises(ValueError,match='all four'):layer.gates(changed,digest)


def test_failed_numeric_check_blocks_layer_gate(tmp_path):
    path,digest=evidence();changed=tmp_path/'failed.log'
    changed.write_text(path.read_text().replace('"norm_bits_exact": true','"norm_bits_exact": false'))
    with pytest.raises(ValueError,match='prerequisite failed'):layer.gates(changed,digest)
