# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent head/row ownership checks for shortened attention streams."""
from itertools import product
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import bucket_attention as ba


@pytest.mark.parametrize('m',[128,256,384,512])
def test_query_gaps_kv_replay_and_projection_ownership(m):
    class Capture:
        def __init__(self,words): self.records=[]; self.mode=None
        def control(self,mode,kblocks=1,**kwargs): self.mode=mode; self.kblocks=kblocks
        def dma(self,col,kind,offset,dims,strides,output=False,**kwargs):
            self.records.append((self.mode,self.kblocks,col,kind,offset,dims,strides,output))
        def wait(self): pass
        def finish(self): return self.records
    pl=ba.n.c.l.plan(28); off=pl['offsets']; p=ba.p
    with patch.object(ba.r,'Sequence',Capture): records=ba.runtime(pl,m)
    queries=set(); outputs=set(); kv=set()
    for mode,kblocks,col,kind,start,dims,strides,output in records:
        if mode!=0: continue
        assert kblocks==m//32
        starts={start+sum(i*s for i,s in zip(index,strides)) for index in product(*(range(d) for d in dims[:3]))}
        if kind=='q': queries.update(starts)
        elif output: outputs.update(starts)
        else:
            assert dims==[m//64,1,m//32,p.KVWORDS]
            assert strides==[0,0,p.OUT,1]
            kv.update(starts)
    assert queries=={off['queries']+(head*32+row)*p.OUT for head in range(16) for row in range(m//16)}
    assert outputs=={off['attention']+(head*32+row)*p.OUT for head in range(16) for row in range(m//16)}
    assert kv=={off['kv']+(head*16+block)*p.OUT+16 for head in range(16) for block in range(m//32)}
    # QKV outputs preserve original physical feature pitches and cover each active tile.
    projection=[row for row in records if row[0]==2 and row[-1]]
    assert len(projection)==48*(m//128)
    assert {row[4] for row in projection}=={off['projection']+(feature*16+group*4)*p.OUT for feature in range(48) for group in range(m//128)}
    norm=[row for row in records if row[0]==1 and row[-1]]
    assert len(norm)==4 and sum(row[5][2]*row[5][3] for row in norm)==m*p.OUT


def test_explicit_loop_count_and_bounds():
    prefix=ba.prefix()
    assert prefix.count('%attention_count = arith.index_cast %kblocks : i32 to index')==16
    assert '%count = arith.select %is_attention,%c16,' not in prefix
    assert prefix.count('stack_size = 4864 : i32')==16
    for m in (0,127,129,513):
        with pytest.raises(ValueError): ba.runtime(ba.n.c.l.plan(28),m)
