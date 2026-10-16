# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Physical-dispatch accounting must include every phase and both head attentions."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_small_lookup_phase_compare as compare


def stats():
    names=[f'mlp_{i}_m128_{phase}' for i in range(28) for phase in compare.candidate.phases.PHASES]
    names += [f'attention_{i}_m128' for i in range(28)]
    names += ['stem','carry','head_pre','head_attention','head_attention','head_post','second_pre','second_post','scorer','action']
    return dict(dispatches=150,dispatch_ms=151.,stages=[dict(name=name,ms=1.) for name in names])


def test_profile_counts_all_phases_and_head_invocations():
    assert compare.phase_totals(stats())=={phase:28. for phase in compare.candidate.phases.PHASES}


@pytest.mark.parametrize('mutation',['missing','duplicate','nan','duration','dispatches'])
def test_reject_unreliable_phase_attribution(mutation):
    values=stats()
    if mutation=='missing':values['stages'].pop()
    elif mutation=='duplicate':values['stages'][0]=values['stages'][1].copy()
    elif mutation=='nan':values['stages'][0]['ms']=float('nan')
    elif mutation=='duration':values['dispatch_ms']=149.
    else:values['dispatches']=66
    with pytest.raises(ValueError):compare.phase_totals(values)
