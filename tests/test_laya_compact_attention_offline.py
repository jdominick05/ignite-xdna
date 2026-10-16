# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dynamic KV-count image must retain full head attention semantics."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import compact_attention as ca


def test_compact_head_sets_sixteen_kv_blocks_and_restores_emitter(monkeypatch):
    calls=[]; original=ca.head.r.Sequence
    class RecordingSequence(original):
        def control(self,mode,*args,**kwargs):
            calls.append((mode,kwargs.copy()))
    # The plan only supplies symbolic offsets: no model, oracle or device.
    offsets={name:index*1000000 for index,name in enumerate(('projection','keys','head_identity','queries','kv','attention','packed_attention'))}
    monkeypatch.setattr(ca.head.r,'Sequence',RecordingSequence)
    ca.runtime(dict(words=10000000,offsets=offsets))
    assert ca.head.r.Sequence is RecordingSequence
    assert [kw for mode,kw in calls if mode==0]==[{'kblocks':16}]
    assert len(calls)==66  # four transforms * four heads * four row groups + attention + pack
    assert all('kblocks' not in kw for mode,kw in calls if mode!=0)
