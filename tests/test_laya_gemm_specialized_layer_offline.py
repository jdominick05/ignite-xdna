# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Require both device primitive verdicts before testing complete MLP layers."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_gemm_specialized_layer as layer


def evidence():
    path=Path.cwd()/'results/laya/gemm_specialized_primitive_v2_desktop2_20261016.log'
    if not path.exists():pytest.skip('device primitive evidence unavailable')
    return path


def test_both_complete_device_groups_are_required(tmp_path):
    path=evidence();assert layer.primitive_gates(path)==layer.c.sha(path)
    bad=tmp_path/'incomplete.log'
    lines=path.read_text().splitlines()
    bad.write_text('\n'.join(line for line in lines if not (line.startswith('WINDOW_PRIMITIVE_RESULT ') and 'original1' in line)))
    with pytest.raises(ValueError,match='both guarded'):
        layer.primitive_gates(bad)


@pytest.mark.parametrize('old,new',[('"exit_code": 0','"exit_code": 1'),
                                   ('"output_and_packet_bits_exact": true','"output_and_packet_bits_exact": false')])
def test_bad_exit_or_numeric_result_blocks_layer_gate(tmp_path,old,new):
    bad=tmp_path/'failed.log';bad.write_text(evidence().read_text().replace(old,new))
    with pytest.raises(ValueError,match='qualification failed'):
        layer.primitive_gates(bad)
