# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Short schedule ownership and boundary checks without a hardware context."""
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import bucket_mlp as bm


def test_bucket_boundaries_and_invalid_counts():
    for valid in range(1,513):
        m=bm.bucket(valid)
        assert m in (128,256,384,512) and m-128<valid<=m
    for valid in (0,-1,513,True,1.0):
        with pytest.raises(ValueError): bm.bucket(valid)


@pytest.mark.parametrize('m',[128,256,384])
def test_write_ownership_respects_active_rows_and_fixed_feature_pitches(m):
    class Capture:
        def __init__(self,words): self.records=[]
        def control(self,*args,**kwargs): pass
        def dma(self,col,kind,offset,dims,strides,output=False):
            if output: self.records.append((col,offset,dims,strides))
        def wait(self): pass
        def wait_columns(self,columns): pass
        def finish(self): return self.records
    pl=bm.b.plan(); off=pl['offsets']; p=bm.p
    with patch.object(bm.b,'MlpSequence',Capture): records=bm.runtime(pl,m)
    wi=[row for row in records if off['mlp_wi']<=row[1]<off['mlp_wi']+pl['sizes']['mlp_wi']]
    assert len(wi)==84*(m//128)
    # Every feature writes exactly the active prefix of M packets, and no pad packet.
    expected={(off['mlp_wi']+feature*16*p.OUT+packet*4*p.OUT) for feature in range(84) for packet in range(m//128)}
    assert {row[1] for row in wi}==expected
    geglu=[row for row in records if off['mlp_geglu']<=row[1]<off['mlp_geglu']+pl['sizes']['mlp_geglu']]
    assert len(geglu)==41
    assert {row[1] for row in geglu}=={off['mlp_geglu']+feature*16*p.OUT for feature in range(41)}
    assert all(row[2][2]*row[2][3]==(m//32)*p.OUT for row in geglu)
    norm=[row for row in records if off['mlp_norm']<=row[1]<off['mlp_norm']+pl['sizes']['mlp_norm']]
    assert len(norm)==4
    assert sorted((row[1]-off['mlp_norm'])//p.OUT for row in norm)==[i*m//4 for i in range(4)]
    assert sum(row[2][2]*row[2][3] for row in norm)==m*p.OUT


def test_full_bucket_preserves_baseline_stream():
    pl=bm.b.plan()
    assert bm.runtime(pl,512)==bm.b.mlp_runtime(pl)
    for m in (0,32,127,129,513):
        with pytest.raises(ValueError): bm.runtime(pl,m)
