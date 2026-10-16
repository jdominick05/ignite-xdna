# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Comparison preregistration, unstable CPU baselines, and raw-summary provenance."""
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_native_hybrid_compare as c


def test_cpu_drift_is_detected_independently_for_thread_counts_and_schedules():
    rows=[dict(config=config,schedule=schedule,timing=dict(median_ms=value))
      for config,schedule,values in [('eight','actual91',[120.8,187.5]),('one','actual91',[150.,151.]),('eight','padded512',[800.,810.])]
      for value in values]
    result=c.stability(rows,1.10)
    assert not result['eight/actual91']['stable']
    assert result['one/actual91']['stable'] and result['eight/padded512']['stable']


@pytest.mark.parametrize('eligible,qualification,complete,accepted',[(True,False,True,True),(True,True,True,False),
  (False,False,True,False),(True,False,False,False)])
def test_summary_cannot_promote_checks_only_or_incomplete_runs(tmp_path,monkeypatch,capsys,eligible,qualification,complete,accepted):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    prereg=tmp_path/'pre.json'; prereg.write_text('{}')
    raw=tmp_path/'raw.log'
    lines=['COMPARE_START '+json.dumps(dict(qualification=qualification,prereg_sha256=c.sha(prereg)))]
    if complete: lines.append('COMPARE_RESULT {}')
    lines.append('RESEARCH_PROCESS_RESULT '+json.dumps(dict(exit_code=0,timing_eligible=eligible)))
    raw.write_text('\n'.join(lines))
    c.summarize(Namespace(prereg=prereg,raw=raw))
    records=capsys.readouterr().out.splitlines()
    status=json.loads(next(line.split(' ',1)[1] for line in records if line.startswith('COMPARE_SUMMARY_STATUS ')))
    assert status['accepted_timing']==accepted
    evidence=json.loads(records[0].split(' ',1)[1]); assert evidence['raw_sha256']==c.sha(raw)


def test_registration_preserves_required_alternatives_and_is_exclusive(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    fixture=dict(ids=[1]*91,markers=[1,2,3],qtype=0,task='fixture',gold=-1)
    monkeypatch.setattr(c,'identity',lambda:(None,None,{'source':'frozen'},[fixture],dict(seq=512,shorter_bucket_supported=False)))
    args=Namespace(prereg=tmp_path/'new.json',rounds=3,reps=5)
    c.register(args); pre=json.loads(args.prereg.read_text())
    expected={'hybrid512','native_full512','native_buffered','native_lookup'}
    assert all(set(order)==expected for order in pre['timing']['order'])
    assert {row['threads'] for row in pre['cpu']['configs']}=={1,8}
    assert pre['cpu']['schedules']==['actual91','padded512']
    assert pre['fixture']==fixture and pre['files']=={'source':'frozen'}
    with pytest.raises(FileExistsError): c.register(args)


def test_correctness_gate_rejects_nonfinite_outputs_and_changed_choice():
    reference=dict(logits=np.asarray([[1.,0.]],np.float32),act_probs=np.asarray([[1.,0.]],np.float32))
    for changed in (np.asarray([[np.nan,0.]],np.float32),np.asarray([[0.,1.]],np.float32)):
        with pytest.raises(ValueError): c.check(dict(reference,logits=changed),reference)
