# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host request contract; these checks open no hardware context."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import native


def test_padding_clipping_and_copied_option_mask():
    ids=np.array([5,6,7]); markers=np.array([-4,2]); mask=np.array([True,False])
    padded,positions,kind,flags,valid=native.request(ids,markers,np.int64(2),mask)
    assert valid==3 and kind==2
    assert padded.shape==(512,) and np.array_equal(padded[:3],ids)
    assert np.all(padded[3:]==50283) and positions.tolist()==[0,2]
    mask[:]=False
    assert flags.tolist()==[True,False]
    assert ids.tolist()==[5,6,7] and markers.tolist()==[-4,2]


@pytest.mark.parametrize('ids,markers,kind,mask',[
    ([],[0,0],0,None),([1]*513,[0,0],0,None),
    ([1.,2.],[0,1],0,None),([-1,2],[0,1],0,None),
    ([50368,2],[0,1],0,None),([1,2],[0,2],0,None),
    ([1,2],[0,1],True,None),([1,2],[0,1],3,None),
    ([1,2],[0,1],0,[1,1]),([1,2],[0,1],0,[True]*3),
    ([1,2],[0],0,None),([1,2],[0.,1.],0,None),
])
def test_reject_invalid_request_before_device_work(ids,markers,kind,mask):
    with pytest.raises(ValueError): native.request(ids,markers,kind,mask)


def test_markers_use_actual_length_even_though_device_padding_is_larger():
    for length in (1,31,32,33,127,128,129,511,512):
        result=native.request(np.ones(length,np.int64),[-1,length-1],0)
        assert result[-1]==length and result[1].tolist()==[0,length-1]
        with pytest.raises(ValueError): native.request(np.ones(length,np.int64),[0,length],0)
