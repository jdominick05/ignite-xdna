# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""All feature/column addresses must fit guarded, disjoint packing destinations."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import pack_sweep as sweep


def test_every_destination_is_disjoint_and_guarded():
    layout=sweep.plan(); regions=sweep.output_regions()
    assert [feature for _,feature in regions]==list(range(16))
    occupied=np.zeros(layout['words'],bool)
    for start,feature in regions:
        assert start==layout['offsets']['packed_norm']+feature*16*sweep.p.OUT
        assert start>16+128*sweep.p.OUT and start+4*sweep.p.OUT<layout['words']
        assert not occupied[start:start+4*sweep.p.OUT].any()
        occupied[start:start+4*sweep.p.OUT]=True
    assert occupied.sum()==64*sweep.p.OUT


@pytest.mark.parametrize('compact',[False,True])
def test_runtime_covers_all_columns_features_and_existing_waits(compact):
    text=sweep.runtime(compact)
    dmas=[line for line in text.splitlines() if 'dma_memcpy_nd' in line]
    assert len(dmas)==48
    assert text.count('dma_wait')==16
    for col in range(4):
        for kind in ('q','kv','out'):
            assert sum(f'@{kind}L2_{col}' in line for line in dmas)==4


@pytest.mark.parametrize('feature',range(16))
def test_reference_feature_selection_is_distinct_for_every_tile(feature):
    values=np.zeros((128,sweep.p.OUT),np.float32)
    identities=(np.arange(128*1024,dtype=np.uint32).reshape(128,1024)%128)+0x3f00
    identities+=np.arange(1024,dtype=np.uint32)[None,:]//64*128
    values[:,:1024]=(identities<<16).view(np.float32)
    selected=values.copy();selected[:,:64]=values[:,feature*64:feature*64+64]
    payload=sweep.expected(selected).view(np.uint16).reshape(4,2048)
    for core in range(4):
        for row in range(32):
            for channel in range(64):
                index=(row//4)*256+(channel//8)*32+(row%4)*8+channel%8
                assert payload[core,index]==identities[core*32+row,feature*64+channel]
