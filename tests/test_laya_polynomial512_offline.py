# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Original accuracy criteria and complete polynomial development are mandatory."""
import copy
import json
from pathlib import Path
import sys

import pytest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'kernels/laya_transformer'))
import polynomial512_accuracy as accuracy


def evidence():
    checks=[dict(case=str(i),exact=dict(logits=True,act_probs=True),contexts=5,
                 dispatches=66,polynomial_stages=28) for i in range(25)]
    result=dict(cases=25,all_output_bits_exact=True,cpu_neural_oracle_calls=0,
                idle_after_close=True,registered_files_unchanged=True)
    end=dict(exit_code=0,stop=None)
    return checks,result,end


@pytest.mark.parametrize('change',['none','missing_case','nonexact','fallback','missing_stage','killed'])
def test_accuracy_registration_requires_full_development(tmp_path,change):
    checks,result,end=evidence()
    if change=='missing_case':
        checks.pop()
    elif change=='nonexact':
        checks[0]['exact']['logits']=False
    elif change=='fallback':
        result['cpu_neural_oracle_calls']=1
    elif change=='missing_stage':
        checks[0]['polynomial_stages']=27
    elif change=='killed':
        end['exit_code']=1
    lines=['POLYNOMIAL512_FULL_CHECK '+json.dumps(row) for row in checks]
    lines+=['POLYNOMIAL512_FULL_RESULT '+json.dumps(result),'RESEARCH_PROCESS_RESULT '+json.dumps(end)]
    path=tmp_path/'evidence.log'
    path.write_text('\n'.join(lines),encoding='utf-8')
    if change=='none':
        assert accuracy.development_gate(path)==accuracy.c.sha(path)
    else:
        with pytest.raises(ValueError):
            accuracy.development_gate(path)


def test_original_frozen_acceptance_is_unchanged():
    original=json.loads(accuracy.na.ORIGINAL.read_text(encoding='utf-8'))
    before=copy.deepcopy(original)
    arm=accuracy.document(original,{'test':'source-map'},{'test':'provenance'})
    for key in ('criteria','criteria_text','items','reference','decision'):
        assert arm[key]==before[key]
    assert original==before
    assert arm['native_policy']['encoder_row_bucket'].startswith('fixed512 for EVERY item')
    assert arm['native_policy']['embedding_and_head_schedule']==512
    assert arm['native_policy']['contexts']==5
    assert arm['native_policy']['dispatches']==66
    assert arm['native_policy']['cpu_neural_fallback'] is False
