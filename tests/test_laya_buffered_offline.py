# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Arm selection must independently switch MLP and attention, retaining fallback."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import buffered_probe as bp


@pytest.mark.parametrize('options,mlp,attention',[
    ({},True,True),
    ({'candidate':False},False,False),
    ({'mlp_candidate':False},False,True),
    ({'attention_candidate':False},True,False),
    ({'baseline':True},False,True),
    ({'mlp_only':True},False,True),
])
def test_arms_switch_mlp_independently_and_forward_attention(monkeypatch,options,mlp,attention):
    # The attention parent's own baseline/mlp_only gate disables its candidate.
    session=object.__new__(bp.Session)
    original=object(); candidate=object(); untouched=object()
    session.original_mlp={'mlp_0_m128':original}
    session.buffered_mlp={'mlp_0_m128':candidate}
    session.instructions={'mlp_0_m128':original,'mlp_0_m256':untouched}
    seen=[]
    monkeypatch.setattr(bp.apb.Session,'run',lambda self,*args,**kwargs:seen.append(kwargs))
    session.run([],[],0,**options)
    assert session.instructions['mlp_0_m128'] is (candidate if mlp else original)
    assert session.instructions['mlp_0_m256'] is untouched
    assert seen[0]['candidate'] is attention
    assert seen[0]['baseline']==options.get('baseline',False)
    assert seen[0]['mlp_only']==options.get('mlp_only',False)
