# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A new buffering arm must retain the original frozen decision criteria."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import buffered_accuracy as a


def test_registration_preserves_original_criteria_and_discloses_prior_results(tmp_path,monkeypatch):
    original=tmp_path/'original.json'; original.write_text('{}')
    monkeypatch.setattr(a.na,'ORIGINAL',original)
    # Registration stores a repo-relative path; use a temporary root for this fixture.
    monkeypatch.setattr(a.p,'ROOT',tmp_path)
    pre=dict(criteria=dict(agree_pooled_min=.97,dacc_pooled_min=-.005,dacc_task_min=-.015,dece1_pooled_max=.01,dece1_task_max=.02),
      criteria_text=['frozen'],items=dict(labelled=2,fixture_questions=3),reference=dict(files={}),decision=dict(temperature=1))
    untouched=copy.deepcopy(pre)
    monkeypatch.setattr(a.na,'inputs',lambda *args:(pre,[dict(ids=[1]*91),dict(ids=[1]*129)]))
    monkeypatch.setattr(a,'identity',lambda *args:(None,{'source/runner':'frozen-source'}))
    monkeypatch.setattr(a,'previous',lambda *args:dict(file_sha256='qualified-native-output',registration_id='qualified-native-arm'))
    args=Namespace(items=tmp_path/'items',reference=tmp_path/'ref',base=tmp_path,model=tmp_path/'model.onnx',
      mlp_folder=tmp_path,attention_folder=tmp_path,prereg=tmp_path/'new.json')
    a.register(args)
    stored=json.loads(args.prereg.read_text())
    for key in ('criteria','criteria_text','items','reference','decision'): assert stored[key]==pre[key]
    assert pre==untouched
    assert set(stored['arms'])=={a.NAME}
    assert stored['native_policy']['contexts']==5 and stored['native_policy']['dispatches']==66
    assert stored['item_buckets']=={'128':1,'256':1}
    assert stored['development_disclosure']['buffered_labelled_final_scores_before_registration']==0
    assert 'not an independent held-out study' in stored['development_disclosure']['previous_native_arm']
    assert stored['native_equivalence_reference']['file_sha256']=='qualified-native-output'
    with pytest.raises(FileExistsError): a.register(args)
