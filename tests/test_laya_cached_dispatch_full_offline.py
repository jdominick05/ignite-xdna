# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full cached execution must cover every MLP exactly once and preserve gates."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_cached_dispatch_full as full


def profile():
    return dict(dispatches=66,stages=[dict(name=f'mlp_{i}_m128',family='fast_mlp') for i in range(28)])


def test_exactly_all_m128_mlp_stages():
    assert full.stage_gate(profile(),91)==28
    assert full.stage_gate(dict(dispatches=66,stages=[]),129)==0


@pytest.mark.parametrize('change',['missing','duplicate','extra_dispatch','wrong_bucket'])
def test_reject_incomplete_or_wrong_physical_execution(change):
    stats=profile();valid=91
    if change=='missing':stats['stages'].pop()
    elif change=='duplicate':stats['stages'][-1]=stats['stages'][0]
    elif change=='extra_dispatch':stats['dispatches']=67
    else:valid=512
    with pytest.raises(ValueError):full.stage_gate(stats,valid)


def test_actual_self_contained_layer_prerequisite():
    path=Path.cwd()/'results/laya/cached_dispatch_layer_v1_desktop2_20261017.log'
    if not path.exists():pytest.skip('self-contained layer evidence unavailable')
    assert full.layer_gate(path)==full.c.sha(path)
