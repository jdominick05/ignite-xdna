# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wide packet gathers and unchanged live payloads for all other commands."""
from pathlib import Path
import itertools
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import compact_pack_wide as cw


class RecordingSequence:
    def __init__(self): self.events=[]
    def control(self,mode,**kwargs): self.position=kwargs['position']
    def dma(self,col,kind,offset,dims,strides,output=False,bid=None):
        self.events.append(dict(col=col,kind=kind,offset=offset,dims=dims,strides=strides,
          bid=({'q':0,'out':2,'kv':3}[kind] if bid is None else bid),feature=self.position[col]))
    def wait(self): self.events.append(None)


def addresses(event):
    return np.fromiter((event['offset']+sum(i*s for i,s in zip(indices,event['strides']))
      for indices in itertools.product(*(range(size) for size in event['dims']))),np.int64)


@pytest.mark.parametrize('m',[128,256,384,512])
def test_one_bd_per_kv_group_covers_each_desired_and_unused_word(m):
    seq=RecordingSequence(); p=cw.p; source=16
    data=np.arange(source+512*p.OUT,dtype=np.uint32)
    cw.pack(seq,dict(offsets=dict(packed_norm=3000000)),source,m)
    blocks=0; group=[]
    for event in seq.events:
        if event is not None: group.append(event); continue
        for col in range(4):
            events=[item for item in group if item['col']==col]
            assert len(events)==3 and len({item['bid'] for item in events})==3
            kv=next(item for item in events if item['kind']=='kv')
            words=data[addresses(kv)].reshape(4,32,66)
            assert words.size==4*cw.KVWORDS
            feature=kv['feature']; first=(kv['offset']-source-feature*64)//p.OUT
            expected=data[source:source+512*p.OUT].reshape(512,p.OUT)[first:first+128,feature*64:feature*64+66].reshape(4,32,66)
            assert np.array_equal(words,expected)
            assert np.array_equal(words[:,:,:64],expected[:,:,:64])
            assert first+128<=m
            blocks+=1
        group=[]
    assert blocks==16*(m//128) and not group


def test_prefix_widens_fifo_types_consistently_and_restores_globals():
    original=(cw.p.KVWORDS,cw.p.shared.KVWORDS)
    prefix=cw.prefix()
    for name in ('kvF','kvL2'):
        lines=[line for line in prefix.splitlines() if 'aie.objectfifo @'+name+'_' in line]
        assert len(lines)==4 and all('memref<2112xi32>' in line for line in lines)
    assert prefix.count('!aie.objectfifosubview<memref<2112xi32>>')>0
    assert (cw.p.KVWORDS,cw.p.shared.KVWORDS)==original==(2064,2064)


class RuntimeRecorder:
    def __init__(self,words): self.events=[]; self.mode=None
    def control(self,mode,*args,**kwargs): self.mode=mode
    def dma(self,col,kind,offset,dims,strides,output=False,bid=None):
        self.events.append(dict(mode=self.mode,col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output,bid=bid))
    def wait(self): pass
    def finish(self): return self.events


@pytest.mark.parametrize('m',[128,256,384,512])
def test_other_operations_retain_every_original_live_payload_start_and_pitch(monkeypatch,m):
    names=('mlp_norm','packed_norm','projection','input','queries','keys','kv','coeff','attention',
      'packed_attention','projection_weight','norm','affine')
    plan=dict(words=100000000,offsets={name:(index+1)*4000000 for index,name in enumerate(names)})
    monkeypatch.setattr(cw.ba.r,'Sequence',RuntimeRecorder)
    old=cw.ba.runtime(plan,m); new=cw.runtime(plan,m)
    assert cw.ba.r.Sequence is RuntimeRecorder
    # Pack Q/output schedules and every non-pack transfer keep old addresses.
    old_rest=[event for event in old if event['mode']!=10 or event['kind']!='kv']
    new_rest=[event for event in new if event['mode']!=10 or event['kind']!='kv']
    assert len(old_rest)==len(new_rest)
    for before,after in zip(old_rest,new_rest):
        if before['kind']=='kv':
            assert before['dims'][-1]==2064 and after['dims'][-1]==2112
            after=dict(after,dims=after['dims'][:-1]+[2064])
        assert before==after
    assert all(np.prod(event['dims'])%2112==0 for event in new if event['kind']=='kv')
