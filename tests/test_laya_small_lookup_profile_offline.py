# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject partial, duplicated or impossible complete-model stage attribution."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_small_lookup_profile as profile


def example():
    names=['stem','carry','head_pre','head_attention','head_attention','head_post',
        'second_pre','second_post','scorer','action']
    names += [f'{family}_{index}_m128' for family in ('attention','mlp') for index in range(28)]
    return dict(stages=[dict(name=name,ms=1.) for name in names],dispatches=66,
        dispatch_ms=67.,prepare_ms=1.,readback_ms=1.,total_ms=69.)


def test_complete_stage_sum_keeps_both_head_attention_dispatches():
    value=profile.stage_groups(example())
    assert value['stage_sum_ms']==66.
    assert value['groups_ms']['head_attention']==2.
    assert value['groups_ms']['encoder_mlp']==28.
    assert value['groups_ms']['encoder_attention_projection_norm']==28.
    assert value['dispatch_loop_unattributed_ms']==1.


@pytest.mark.parametrize('kind',['missing','duplicated','nan','negative','overlap','count'])
def test_reject_unreliable_stage_attribution(kind):
    stats=example()
    if kind=='missing':stats['stages'].pop()
    elif kind=='duplicated':stats['stages'][-1]=stats['stages'][-2].copy()
    elif kind=='nan':stats['stages'][0]['ms']=float('nan')
    elif kind=='negative':stats['stages'][0]['ms']=-1.
    elif kind=='overlap':stats['dispatch_ms']=65.
    else:stats['dispatches']=65
    with pytest.raises(ValueError):profile.stage_groups(stats)
