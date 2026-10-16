# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen criteria preservation and safe session-adapter binding."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import lookup_head_accuracy as a


def test_lookup_registration_retains_original_gates_and_discloses_changed_math(tmp_path,monkeypatch):
    original=tmp_path/'original.json'; original.write_text('{}')
    monkeypatch.setattr(a.na,'ORIGINAL',original); monkeypatch.setattr(a.p,'ROOT',tmp_path)
    pre=dict(criteria=dict(agree_pooled_min=.97,dacc_pooled_min=-.005,dacc_task_min=-.015,dece1_pooled_max=.01,dece1_task_max=.02),
      criteria_text=['frozen'],items=dict(labelled=2,fixture_questions=3),reference=dict(files={}),decision=dict(temperature=1))
    before=copy.deepcopy(pre)
    monkeypatch.setattr(a.na,'inputs',lambda *args:(pre,[dict(ids=[1]*91),dict(ids=[1]*129)]))
    monkeypatch.setattr(a,'identity',lambda *args:(None,{'source/runner':'registered-source'}))
    monkeypatch.setattr(a.ba,'previous',lambda *args:dict(file_sha256='qualified-native-output',registration_id='qualified-native-arm'))
    args=Namespace(items=tmp_path/'items',reference=tmp_path/'ref',base=tmp_path,model=tmp_path/'model.onnx',
      mlp_folder=tmp_path,attention_folder=tmp_path,prereg=tmp_path/'new.json')
    a.register(args); stored=json.loads(args.prereg.read_text())
    for key in ('criteria','criteria_text','items','reference','decision'): assert stored[key]==before[key]
    assert pre==before and set(stored['arms'])=={a.NAME}
    assert stored['item_buckets']=={'128':1,'256':1}
    assert stored['development_disclosure']['labelled_final_scores_before_registration']==0
    assert 'not an independent held-out study' in stored['development_disclosure']['heldout']
    assert 'bit equivalence is not required' in stored['native_equivalence_reference_use']
    assert not stored['native_policy']['cpu_neural_fallback']
    with pytest.raises(FileExistsError): a.register(args)


def test_scoring_adapter_keeps_original_module_global_session_unchanged(monkeypatch):
    module=a.ba.buffered; original=module.Session; calls=[]
    def candidate(*args,**kwargs): calls.append((args,kwargs)); return 'candidate'
    monkeypatch.setattr(a.op,'Session',candidate)
    def score(args):
        assert a.ba.buffered is not module
        assert module.Session is original
        assert a.na.NAME==a.NAME and a.ba.NAME==a.NAME
        assert a.ba.buffered.Session('bundle','mlp','attention','mc','ac','compact',['fixture'])=='candidate'
    monkeypatch.setattr(a.ba,'score',score)
    argv=['runner','score']
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','fast-folder',
      'items','reference','prereg','out','native-reference','native-prereg'): argv+=['--'+name,name]
    monkeypatch.setattr(sys,'argv',argv)
    a.main()
    assert a.ba.buffered is module and module.Session is original
    assert calls[0][1]==dict(fast_folder=Path('fast-folder'),mlp_kind='lookup')
    assert calls[0][0][-1]==Path('head-folder')
