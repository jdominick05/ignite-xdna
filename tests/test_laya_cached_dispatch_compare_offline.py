# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adoption requires the preset gain, consistent rounds and eligible completion."""
from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_cached_dispatch_compare as comparison


def rows(cache=80):
    return [dict(round=index,arm=arm,samples_ms=[100 if arm=='control' else cache]*5,idle_after_close=True)
        for index in range(3) for arm in comparison.ARMS]


def test_gain_and_kill_line():
    assert comparison.decision(rows())['adoption_from_samples']=='adopt'
    assert comparison.decision(rows(90))['adoption_from_samples']=='adopt'
    assert comparison.decision(rows(95))['adoption_from_samples']=='kill'


def test_pooled_gain_cannot_hide_losing_round():
    values=rows();values[3]['samples_ms']=[110]*5
    result=comparison.decision(values)
    assert result['pooled_saved_ms']==20
    assert result['adoption_from_samples']=='kill'


def test_incomplete_samples_rejected():
    values=rows();values.pop()
    with pytest.raises(ValueError):comparison.decision(values)
    values=rows();values[0]['samples_ms'][0]=float('nan')
    with pytest.raises(ValueError):comparison.decision(values)


@pytest.mark.parametrize('eligible',[True,False])
def test_wrapper_eligibility_controls_adoption(tmp_path,eligible):
    values=rows();result=dict(comparison.decision(values),cpu_neural_oracle_calls=0,
        idle_after_close=True,registered_files_unchanged=True)
    lines=['CACHED_COMPARE_PROFILE '+json.dumps(row) for row in values]
    lines+=['CACHED_COMPARE_RESULT '+json.dumps(result),'RESEARCH_PROCESS_RESULT '+json.dumps(dict(exit_code=0,stop=None,timing_eligible=eligible))]
    path=tmp_path/'comparison.log';path.write_text('\n'.join(lines),encoding='utf-8')
    if eligible:assert comparison.timing_verdict(path)['adoption_from_samples']=='adopt'
    else:
        with pytest.raises(ValueError):comparison.timing_verdict(path)


def test_actual_frozen_accuracy_allows_timing_registration():
    root=Path.cwd();report=root/'results/laya/cached_dispatch_accuracy_v1_report_desktop2_20261017.log'
    if not report.exists():pytest.skip('full frozen report unavailable')
    proof=comparison.frozen_gate(root/'results/laya/cached_dispatch_accuracy_v1_prereg_desktop2_20261017.json',
        root/'scratch/laya/native_accuracy/native_cached_wi_dispatch_bf16_v1.npz',report)
    assert proof['output_sha256']=='0187aa86da7e03585bbd5a2224def67d6bd496a3c47b64abf98ddee2d23a91ab'
