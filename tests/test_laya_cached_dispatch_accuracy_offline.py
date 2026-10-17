# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cached registration retains the original frozen criteria and complete gate."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import cached_dispatch_accuracy as accuracy


def test_frozen_decision_reference_and_criteria_unchanged():
    path=Path.cwd()/'results/laya/small_lookup_accuracy_v1_prereg_desktop2_20261016.json'
    old=json.loads(path.read_text(encoding='utf-8'));snapshot=copy.deepcopy(old)
    arm=accuracy.document(old,{'new':'source-hash'},{'reference':'old-score-hash'})
    for key in ('criteria','criteria_text','items','reference','decision'):
        assert arm[key]==old[key]
    assert old==snapshot
    assert set(arm['arms'])=={accuracy.NAME}
    assert arm['native_policy']['dispatches']==66
    assert arm['native_policy']['contexts']==5


def test_actual_full_development_prerequisite():
    path=Path.cwd()/'results/laya/cached_dispatch_full_v1_desktop2_20261017.log'
    if not path.exists():pytest.skip('complete cached development evidence unavailable')
    assert accuracy.full_gate(path)==accuracy.p.attention.sha(path)


def test_missing_full_case_blocks_registration(tmp_path):
    source=Path.cwd()/'results/laya/cached_dispatch_full_v1_desktop2_20261017.log'
    if not source.exists():pytest.skip('complete cached development evidence unavailable')
    rows=source.read_text(encoding='utf-8').splitlines();removed=False;output=[]
    for row in rows:
        if row.startswith('CACHED_DISPATCH_FULL_CHECK ') and not removed:removed=True;continue
        output.append(row)
    path=tmp_path/'incomplete.log';path.write_text('\n'.join(output),encoding='utf-8')
    with pytest.raises(ValueError):accuracy.full_gate(path)
