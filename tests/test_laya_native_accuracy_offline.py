# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frozen gate boundaries and scored-file provenance; no device access."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import native_accuracy as a


def arm():
    return dict(items=dict(file_sha256='frozen-items'),arms={a.NAME:dict(files={'image':'registered-image'})},
      criteria=dict(agree_pooled_min=.97,dacc_pooled_min=-.005,dacc_task_min=-.015,dece1_pooled_max=.01,dece1_task_max=.02))


def toy(tasks=('ag_news',),per_task=200):
    items=[dict(task=task,k=2,gold=0) for task in tasks for _ in range(per_task)]
    z=np.full((len(items),a.la.KMAX),np.nan,np.float32); z[:,:2]=[4,0]
    return items,z


def test_incomplete_checkpoint_cannot_be_reported_as_complete(tmp_path):
    items,z=toy(per_task=3); probabilities=np.tile([1.,0.],(len(items),1))
    checkpoint=tmp_path/'run.checkpoint.npz'
    a.save_snapshot(checkpoint,z,probabilities,items,arm(),'registered-commit',2,False,0)
    with pytest.raises(ValueError,match='incomplete'): a.validate_scored(checkpoint,arm(),items)
    # Checkpoint replacement is atomic and still incomplete, even when all rows exist.
    a.save_snapshot(checkpoint,z,probabilities,items,arm(),'registered-commit',3,False,0)
    with pytest.raises(ValueError,match='incomplete'): a.validate_scored(checkpoint,arm(),items)
    final=tmp_path/'run.npz'
    a.save_snapshot(final,z,probabilities,items,arm(),'registered-commit',3,True,0)
    assert np.array_equal(a.validate_scored(final,arm(),items),z,equal_nan=True)
    with pytest.raises(FileExistsError): a.save_snapshot(final,z,probabilities,items,arm(),'registered-commit',3,True,0)


@pytest.mark.parametrize('change', ['provenance','cpu_call','nonfinite','padding'])
def test_reject_invalid_completed_output(tmp_path,change):
    items,z=toy(per_task=3); probabilities=np.tile([1.,0.],(len(items),1)); registered=arm()
    if change=='nonfinite': z[0,0]=np.nan
    if change=='padding': z[0,2]=0
    out=tmp_path/'run.npz'
    a.save_snapshot(out,z,probabilities,items,registered,'registered-commit',3,True,int(change=='cpu_call'))
    if change=='provenance': registered['arms'][a.NAME]['files']['image']='different-image'
    with pytest.raises(ValueError): a.validate_scored(out,registered,items)


def test_exact_original_accuracy_delta_boundary_passes():
    items,ref=toy(); observed=ref.copy(); observed[0,:2]=[0,4]
    result=a.gates(arm(),items,ref,observed)
    assert result['dacc']==pytest.approx(-.005)
    assert result['criteria']==dict(C0=True,C1=True,C2=True,C3=True,C4=True,C5=True)


def test_per_task_loss_cannot_hide_in_pooled_result():
    items,ref=toy(tasks=('ag_news','emotion','banking77','massive_en'),per_task=400)
    observed=ref.copy(); observed[:7,:2]=[0,4]
    result=a.gates(arm(),items,ref,observed)
    assert result['criteria']['C1'] and result['criteria']['C2']
    assert not result['criteria']['C3'] and not result['passed']
    assert result['tasks']['ag_news']['dacc']==pytest.approx(-.0175)


def test_agreement_gate_is_independent_of_accuracy():
    items,ref=toy(); observed=ref.copy(); observed[:7,:2]=[0,4]
    # Gold -1 rows are the actual fixtures and are excluded by their task name.
    items.append(dict(task='fixture',k=2,gold=-1))
    ref=np.concatenate([ref,ref[:1]]); observed=np.concatenate([observed,observed[:1]])
    result=a.gates(arm(),items,ref,observed)
    assert result['labelled']==200 and result['agreement']==pytest.approx(.965)
    assert not result['criteria']['C1']
