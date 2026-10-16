# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-first integration must never select the candidate in other layers."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_pack_layer_probe as probe


@pytest.mark.parametrize('layers',[(),(0,),(1,)])
def test_only_requested_layer_selects_candidate_and_its_mask(monkeypatch,layers):
    session=object.__new__(probe.Session)
    keys=[f'attention_{i}_m128' for i in range(28)]
    candidate={key:object() for key in keys};masks={key:[100+i] for i,key in enumerate(keys)}
    session.candidate_attention=candidate;session.candidate_mask_fields=masks
    session.original_attention={key:object() for key in keys}
    session.original_mask_fields={key:[i] for i,key in enumerate(keys)}
    def parent(self,*args,**kwargs):
        for index,key in enumerate(keys):
            assert self.candidate_attention[key] is (candidate[key] if index in layers else self.original_attention[key])
            assert self.candidate_mask_fields[key]==(masks[key] if index in layers else [index])
        return 'checked'
    monkeypatch.setattr(probe.op.Session,'run',parent)
    assert session.run(pack_layers=layers)=='checked'
    assert session.candidate_attention is candidate and session.candidate_mask_fields is masks
