# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Short head emitters must preserve the existing full-bucket packet program."""
from pathlib import Path
import itertools
import sys

import pytest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import bucket_head as bh
import bucket_head_probe as hp


def plan():
    names=['mlp_norm','norm_input','head_type','head_affine','packed_norm','projection','head_qkv_weight','head_qkv_bias',
      'norm','head_out','head_out_weight','head_out_bias','head_norm2','head_wi_weight','head_wi_bias','head_relu',
      'head_wo_weight','head_wo_bias','head_next_norm','second_qkv_weight','second_qkv_bias','second_out_weight',
      'second_out_bias','second_norm2','second_wi_weight','second_wi_bias','second_wo_weight','second_wo_bias','packed_attention',
      'head_identity','queries','keys','kv','attention','mlp_wi']
    return dict(words=100000000,offsets={name:(index+1)*2000000 for index,name in enumerate(names)})


@pytest.mark.parametrize('new,old',[(bh.pre_runtime,bh.pre.runtime),(bh.post_runtime,bh.post.runtime),
  (bh.second_pre_runtime,bh.second.pre_runtime),(lambda pl,m:bh.post_runtime(pl,m,True),bh.second.post_runtime)])
def test_full_bucket_pointwise_stream_is_exact_original_text(new,old):
    assert new(plan(),512)==old(plan())


@pytest.mark.parametrize('m',[0,1,127,129,513,True])
def test_refuse_unsupported_head_bucket_before_emission(m):
    with pytest.raises(ValueError): bh.streams(plan(),m)


def test_full_head_attention_preserves_every_dma_word_and_control(monkeypatch):
    class RecordingSequence:
        def __init__(self,words): self.events=[]
        def control(self,mode,*args,**kwargs): self.events.append(('control',mode,args,kwargs))
        def dma(self,col,direction,offset,sizes,strides,output=False,**kwargs):
            # Canonical contiguous runs, independent of tensor rank factoring.
            # This checks every transferred word without allocating tensor data.
            runs=[]
            assert strides[-1]==1
            for indices in itertools.product(*(range(size) for size in sizes[:-1])):
                start=offset+sum(index*stride for index,stride in zip(indices,strides))
                if runs and runs[-1][0]+runs[-1][1]==start:
                    runs[-1]=(runs[-1][0],runs[-1][1]+sizes[-1])
                else: runs.append((start,sizes[-1]))
            self.events.append(('dma',col,direction,output,kwargs,runs))
        def wait(self): self.events.append(('wait',))
        def finish(self): return self.events
    monkeypatch.setattr(bh.ba.r,'Sequence',RecordingSequence)
    assert bh.attention_runtime(plan(),512)==bh.compact.runtime(plan())


@pytest.mark.parametrize('options,selected',[
    ({},True),({'candidate':False},False),({'head_candidate':False},False),
    ({'candidate':False,'head_candidate':True},True),
    ({'baseline':True,'head_candidate':True},False),
    ({'mlp_only':True,'head_candidate':True},False),
])
def test_each_frame_switches_head_instructions_and_both_metadata_maps(monkeypatch,options,selected):
    session=object.__new__(hp.Session)
    session.original_heads={'head_pre':object(),'head_attention':object()}
    session.original_head_types=[1]; session.original_head_masks=[2]
    session.short_heads={m:{name:object() for name in session.original_heads} for m in (128,256,384,512)}
    session.short_head_types={m:[m+1] for m in session.short_heads}
    session.short_head_masks={m:[m+2] for m in session.short_heads}
    session.instructions=dict(session.original_heads); session.original_masks={}
    seen=[]
    monkeypatch.setattr(hp.buffered.Session,'run',lambda self,*args,**kwargs:seen.append(kwargs))
    for length in (91,385,129,91):
        m=((length+127)//128)*128
        session.run(np.ones(length,np.int64),[0,length-1],length%3,**options)
        expected=session.short_heads[m] if selected else session.original_heads
        assert all(session.instructions[name] is inst for name,inst in expected.items())
        assert session.type_patches==(session.short_head_types[m] if selected else session.original_head_types)
        assert session.original_masks['head_attention']==(session.short_head_masks[m] if selected else session.original_head_masks)
    assert len(seen)==4


def test_invalid_marker_is_rejected_before_selecting_head_instructions(monkeypatch):
    session=object.__new__(hp.Session); sentinel=object()
    session.instructions={'head_pre':sentinel}
    monkeypatch.setattr(hp.buffered.Session,'run',lambda *args,**kwargs:pytest.fail('parent dispatch reached'))
    with pytest.raises(ValueError,match='marker beyond'):
        session.run(np.ones(91,np.int64),[0,91],0)
    assert session.instructions['head_pre'] is sentinel
