# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent all-word gather coverage and distinct live Shim BD slots."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import compact_pack as cp


class RecordingSequence:
    def __init__(self): self.events=[]
    def control(self,mode,**kwargs): self.mode=mode; self.position=kwargs['position']
    def dma(self,col,kind,offset,dims,strides,output=False,bid=None):
        self.events.append(dict(col=col,kind=kind,offset=offset,dims=dims,strides=strides,
          output=output,bid=({'q':0,'out':2,'kv':3}[kind] if bid is None else bid),feature=self.position[col]))
    def wait(self): self.events.append(None)


def addresses(event):
    return np.fromiter((event['offset']+sum(i*s for i,s in zip(indices,event['strides']))
      for indices in itertools.product(*(range(size) for size in event['dims']))),np.int64)


@pytest.mark.parametrize('m',[128,256,384,512])
def test_every_gathered_payload_and_filler_word_and_no_live_bd_collision(m):
    seq=RecordingSequence(); p=cp.p; source=16
    # Unique word identities make every row/feature/column mix-up observable.
    data=np.arange(source+512*p.OUT,dtype=np.uint32)
    cp.pack(seq,dict(offsets=dict(packed_norm=3000000)),source,m)
    group=[]; blocks=0
    for event in seq.events:
        if event is not None:
            group.append(event); continue
        for col in range(4):
            current=[item for item in group if item['col']==col]
            assert len({item['bid'] for item in current})==len(current)==10
            assert [item['bid'] for item in current if item['kind']=='kv']==list(range(3,11))
            kv=[item for item in current if item['kind']=='kv']
            for row in range(4):
                payload,filler=kv[row*2:row*2+2]
                words=np.concatenate([data[addresses(payload)],data[addresses(filler)]])
                assert words.size==p.KVWORDS==2064
                first=payload['offset']; feature=payload['feature']
                row_first=(first-source-feature*64)//p.OUT
                expected=data[source:source+512*p.OUT].reshape(512,p.OUT)[row_first:row_first+32,feature*64:(feature+1)*64].reshape(-1)
                assert np.array_equal(words[:2048],expected)
                assert np.array_equal(words[2048:],expected[:16])
                assert 0<=row_first<row_first+32<=m
            blocks+=1
        group=[]
    assert not group and blocks==16*(m//128)


def test_only_pack_function_and_loop_count_change_in_core_program():
    original=cp.af.prefix(2,2,2); candidate=cp.prefix()
    restored=candidate.replace('@norm_pack_compact_step(','@norm_pack_step(')
    restored=restored.replace('%c128 = arith.constant 4 : index','%c128 = arith.constant 128 : index')
    restored=restored.replace('%c127 = arith.constant 3 : index','%c127 = arith.constant 127 : index')
    assert restored==original
    assert candidate.count('func.call @norm_pack_compact_step(')==16


def test_query_and_output_dmas_are_identical_to_previous_pack():
    plan=dict(offsets=dict(packed_norm=3000000)); before=RecordingSequence(); after=RecordingSequence()
    cp.ba.pack(before,plan,16,128); cp.pack(after,plan,16,128)
    keep=lambda seq:[event for event in seq.events if event is None or event['kind']!='kv']
    assert keep(before)==keep(after)
