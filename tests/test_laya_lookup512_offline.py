# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-512 lookup selection must never substitute the short-row schedule."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_lookup512_full as full


def session():
    obj=object.__new__(full.Session)
    obj.baseline_steps=['stem']+[name for i in range(28) for name in (f'attention_{i}',f'mlp_{i}')]
    obj.full512_original={f'mlp_{i}':f'original{i}' for i in range(28)}
    obj.lookup512={f'mlp_{i}':f'lookup{i}' for i in range(28)}
    obj.full512_heads={'head_pre':'original_head'}
    obj.instructions={}
    obj.original_head_types=[100]
    obj.original_head_masks=[200]
    obj.original_masks={'head_attention':[999],'attention_0':[300]}
    obj.states=dict.fromkeys(range(5))
    return obj


@pytest.mark.parametrize('length',[1,91,128,129,512])
def test_actual_length_never_selects_short_schedule(monkeypatch,length):
    obj=session()
    def run(self,*args,**kwargs):
        assert self.steps==self.baseline_steps
        assert self.mask_patches=={'head_attention':[200],'attention_0':[300]}
        assert self.type_patches==[100]
        assert self.instructions['head_pre']=='original_head'
        self.last_stats=dict(dispatches=66)
        return 'sentinel'
    monkeypatch.setattr(full.n.NativeSession,'run',run)
    assert obj.run([1]*length,[0,0],0)=='sentinel'
    assert all(obj.instructions[f'mlp_{i}']==f'lookup{i}' for i in range(28))
    obj.run([1]*length,[0,0],0,candidate=False)
    assert all(obj.instructions[f'mlp_{i}']==f'original{i}' for i in range(28))


def test_invalid_request_rejected_before_dispatch(monkeypatch):
    obj=session()
    def forbidden(*args,**kwargs):
        pytest.fail('invalid request reached dispatcher')
    monkeypatch.setattr(full.n.NativeSession,'run',forbidden)
    with pytest.raises(ValueError):
        obj.run([1]*513,[0,0],0)


def test_changed_dispatch_policy_rejected(monkeypatch):
    obj=session()
    def run(self,*args,**kwargs):
        self.last_stats=dict(dispatches=65)
    monkeypatch.setattr(full.n.NativeSession,'run',run)
    with pytest.raises(ValueError,match='execution policy'):
        obj.run([1],[0,0],0)
