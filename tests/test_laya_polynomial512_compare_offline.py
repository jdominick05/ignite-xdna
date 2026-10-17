# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adoption and success require frozen accuracy, complete samples and quiet timing."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_polynomial512_compare as comparison


def rows(polynomial=800,hybrid=700):
    values=dict(control=1000,polynomial=polynomial,hybrid512=hybrid)
    return [dict(round=i,arm=arm,samples_ms=[values[arm]]*5,idle_after_close=True,
                 policy=comparison.protocol()['hybrid_policy' if arm=='hybrid512' else 'native_policy'])
            for i in range(3) for arm in comparison.ARMS]


def test_adoption_is_not_automatically_the_matched512_goal():
    result=comparison.decision(rows())
    assert result['adoption_from_samples']=='adopt'
    assert result['pooled_saved_ms']==200
    assert result['matched512_goal_from_samples'] is False
    assert comparison.decision(rows(hybrid=900))['matched512_goal_from_samples'] is True
    assert comparison.decision(rows(polynomial=801))['adoption_from_samples']=='kill'


def test_pooled_gain_cannot_hide_a_losing_round():
    values=rows(polynomial=700)
    values[4]['samples_ms']=[1100]*5
    result=comparison.decision(values)
    assert result['pooled_saved_ms']==300
    assert result['adoption_from_samples']=='kill'


def test_pooled_hybrid_win_cannot_hide_a_losing_round():
    values=rows(polynomial=600,hybrid=700)
    values[5]['samples_ms']=[500]*5
    result=comparison.decision(values)
    assert result['adoption_from_samples']=='adopt'
    assert result['matched512_goal_from_samples'] is False


@pytest.mark.parametrize('change',['missing','duplicate','nan','zero','short'])
def test_incomplete_or_invalid_measurement_rejected(change):
    values=rows()
    if change=='missing':values.pop()
    elif change=='duplicate':values[-1]=values[0]
    elif change=='nan':values[0]['samples_ms'][0]=float('nan')
    elif change=='zero':values[0]['samples_ms'][0]=0
    else:values[0]['samples_ms'].pop()
    with pytest.raises(ValueError):
        comparison.decision(values)


@pytest.mark.parametrize('change',['none','ineligible','contention','wrong_policy','summary_tamper','missing_start'])
def test_wrapper_and_execution_policy_are_binding(tmp_path,change):
    values=rows()
    result=dict(comparison.decision(values),cpu_neural_oracle_calls=0,idle_after_close=True,
                registered_files_unchanged=True)
    end=dict(exit_code=0,stop=None,timing_eligible=True,foreign_contention_observed=False)
    if change=='ineligible':end['timing_eligible']=False
    elif change=='contention':end['foreign_contention_observed']=True
    elif change=='wrong_policy':values[0]['policy']={'contexts':4}
    elif change=='summary_tamper':result['pooled_saved_ms']=999
    start=dict(prereg_sha256='0'*64,scope=comparison.protocol()['scope'],metric=comparison.protocol()['metric'])
    lines=[] if change=='missing_start' else ['POLYNOMIAL512_COMPARE_START '+json.dumps(start)]
    lines+=['POLYNOMIAL512_COMPARE_PROFILE '+json.dumps(row) for row in values]
    lines+=['POLYNOMIAL512_COMPARE_RESULT '+json.dumps(result),'RESEARCH_PROCESS_RESULT '+json.dumps(end)]
    path=tmp_path/'timing.log'
    path.write_text('\n'.join(lines),encoding='utf-8')
    if change=='none':
        assert comparison.timing_verdict(path)['adoption_from_samples']=='adopt'
    else:
        with pytest.raises(ValueError):comparison.timing_verdict(path)


@pytest.mark.parametrize('change',['none','partial','failed_gate','wrong_score','wrong_registration','failed_report'])
def test_frozen_report_must_cover_the_original_full_set(tmp_path,monkeypatch,change):
    prereg=tmp_path/'pre.json'
    prereg.write_text('{}',encoding='utf-8')
    score=tmp_path/'score.npz'
    score.write_bytes(b'score-identity-fixture')
    monkeypatch.setattr(comparison.c.na,'committed_registration',lambda *args:'fixture')
    verdict=dict(passed=True,criteria={f'C{i}':True for i in range(6)},labelled=2400,
                 registration_id=comparison.c.na.digest({}),output_sha256=comparison.c.sha(score))
    end=dict(exit_code=0,stop=None)
    if change=='partial':verdict['labelled']=500
    elif change=='failed_gate':verdict['criteria']['C5']=False
    elif change=='wrong_score':verdict['output_sha256']='wrong'
    elif change=='wrong_registration':verdict['registration_id']='wrong'
    elif change=='failed_report':end['exit_code']=1
    report=tmp_path/'report.log'
    report.write_text('NATIVE_ACCURACY_VERDICT '+json.dumps(verdict)+'\nRESEARCH_PROCESS_RESULT '+json.dumps(end),encoding='utf-8')
    if change=='none':
        assert comparison.frozen_gate(prereg,score,report)['output_sha256']==comparison.c.sha(score)
    else:
        with pytest.raises(ValueError):comparison.frozen_gate(prereg,score,report)


@pytest.mark.parametrize('wrong_registration',[False,True])
def test_log_must_match_the_committed_timing_registration(tmp_path,monkeypatch,wrong_registration):
    pre={'frozen':{'registration_id':'frozen-accuracy-identity'}}
    prereg=tmp_path/'pre.json'
    prereg.write_text(json.dumps(pre),encoding='utf-8')
    monkeypatch.setattr(comparison.c.na,'committed_registration',lambda *args:'fixture')
    values=rows()
    start=dict(prereg_sha256='wrong' if wrong_registration else comparison.c.sha(prereg),
               frozen=pre['frozen'],scope=comparison.protocol()['scope'],metric=comparison.protocol()['metric'])
    result=dict(comparison.decision(values),cpu_neural_oracle_calls=0,idle_after_close=True,
                registered_files_unchanged=True)
    end=dict(exit_code=0,stop=None,timing_eligible=True,foreign_contention_observed=False)
    lines=['POLYNOMIAL512_COMPARE_START '+json.dumps(start)]
    lines+=['POLYNOMIAL512_COMPARE_PROFILE '+json.dumps(row) for row in values]
    lines+=['POLYNOMIAL512_COMPARE_RESULT '+json.dumps(result),'RESEARCH_PROCESS_RESULT '+json.dumps(end)]
    path=tmp_path/'timing.log'
    path.write_text('\n'.join(lines),encoding='utf-8')
    if wrong_registration:
        with pytest.raises(ValueError,match='another registration'):
            comparison.timing_verdict(path,prereg)
    else:
        assert comparison.timing_verdict(path,prereg)['adoption_from_samples']=='adopt'
