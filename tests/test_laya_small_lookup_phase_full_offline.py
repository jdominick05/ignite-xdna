# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise the actual assignment/append pattern used by bucket controllers."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_small_lookup_phase_full as full


def owner(enabled=True):
    return SimpleNamespace(phase_enabled=enabled,phase_steps={f'mlp_{i}_m128':tuple(f'mlp_{i}_m128_{phase}' for phase in full.phases.PHASES) for i in range(28)})


def names(bucket):
    return ['stem','carry']+[f'{family}_{i}_m{bucket}' for i in range(28) for family in ('attention','mlp')]+[
        'head_pre','head_attention','head_post','second_pre','head_attention','second_post','scorer','action']


def test_bucket_controller_append_pattern_and_return_to_short_bucket():
    value=owner()
    for bucket in (128,256,384,512,128):
        steps=full.Steps(value)
        for name in names(bucket):steps.append(name)
        assert len(steps)==(150 if bucket==128 else 66)
        assert steps.count('head_attention')==2
        assert sum(name.endswith('_wi') for name in steps)==(28 if bucket==128 else 0)


def test_assignment_extend_iadd_do_not_double_expand():
    value=owner();steps=full.Steps(value,names(128))
    assert len(steps)==150 and len(full.Steps(value,steps))==150
    extra=full.Steps(value);extra.extend(names(128));assert extra==steps
    extra=full.Steps(value);extra+=names(128);assert extra==steps


def test_unsplit_control_and_constructor_steps_keep_original_dispatches():
    value=owner(False)
    assert full.Steps(value,names(128))==names(128)
    value.phase_enabled=True
    assert len(full.Steps(value,names(128)))==150


def test_completed_layer_prerequisite_rejects_failed_numeric_verdict(tmp_path):
    path=tmp_path/'failed.log'
    path.write_text('SMALL_PHASE_LAYER_RESULT {"cases":6,"all_arena_bits_exact":false}\nRESEARCH_PROCESS_RESULT {"exit_code":0,"stop":null}\n')
    with pytest.raises(ValueError):full.layer_gate(path)
