# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent addresses expose ND wrapping and repeated object mistakes."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import transport_walk as walk


def test_repeated_unchanged_packet_wraps_iteration_to_zero():
    words=[4,0,0,0,0,0,0,0]
    assert np.array_equal(walk.descriptor_addresses(words,3,16),[16,17,18,19]*3)


def test_complete_gather_objects_advance_by_row_group_stride():
    words=[2112,0,0,66<<20,(2<<30)|(32<<20)|2079,2<<24,(3<<20)|66559,2<<24]
    actual=walk.descriptor_addresses(words,4,16).reshape(4,32,66)
    expected=16+np.arange(4)[:,None,None]*66560+np.arange(32)[None,:,None]*2080+np.arange(66)[None,None,:]
    assert np.array_equal(actual,expected)


def test_wrong_stride_changes_payload_addresses():
    words=[8,0,0,4<<20,2<<20|7,0,0,0]
    assert np.array_equal(walk.descriptor_addresses(words,1,0),[0,1,2,3,8,9,10,11])
    words[4]^=1
    assert not np.array_equal(walk.descriptor_addresses(words,1,0),[0,1,2,3,8,9,10,11])


def test_excessive_extent_is_rejected():
    with pytest.raises(ValueError):walk.descriptor_addresses([0]*8,1,0)
