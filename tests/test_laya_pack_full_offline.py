# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The full-model protocol retains row boundaries and serial comparison controls."""
from argparse import Namespace
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_pack_full_probe as full


def fixtures():
    return [dict(ids=[1]*91,markers=[1,2],qtype=i,task='fixture',gold=-1) for i in range(3)]


def test_cases_cover_every_bucket_boundary_and_restore_short_input():
    cases=full.cases(fixtures())
    assert len(cases)==25
    assert {len(row['ids']) for row in cases}>={1,31,32,33,127,128,129,255,256,257,383,384,385,511,512}
    assert len(cases[21]['ids'])==512 and len(cases[22]['ids'])==91
    assert {row['qtype'] for row in cases}=={0,1,2}
    for row in cases:
        assert min(row['markers'])>=0 and max(row['markers'])<len(row['ids'])
    assert full.feed(cases[-1])['mask'].tolist()==[False,True]


def test_preregistration_is_exclusive_and_keeps_serial_native_controls(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'identity',lambda:(None,None,fixtures(),{'source':'hash'},{'proof':True}))
    args=Namespace(prereg=tmp_path/'pre.json')
    full.register(args);pre=json.loads(args.prereg.read_text())
    assert pre['order']==[['lookup','wide'],['wide','lookup'],['lookup','wide']]
    assert pre['reps']==5 and pre['warmups']==2
    assert pre['policy']==dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0)
    with pytest.raises(FileExistsError):full.register(args)
