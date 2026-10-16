# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Marker-address safety, tail metadata and output-word ownership."""
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import scorer as s
import action as a
import tail as t
import tail_probe as tp
import embedding_carry as ec
p=s.p


def base():
    second=s.second; post=second.post; attn=post.a
    return s.plan(second.plan(post.plan(attn.plan(attn.h.plan(s.b.plan())))))


def ownership(text,pl,names):
    writes={key:np.zeros(pl['sizes'][key],np.uint8) for key in names}
    entries=re.findall(r'dma_memcpy_nd\(%([ao])\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]\) \{id = (\d+).*metadata = @(q|kv|out)L2_(\d+)',text)
    assert entries
    for var,offset,dimensions,strides,bid,kind,col in entries:
        start=int(offset); dims=list(map(int,dimensions.split(','))); strides=list(map(int,strides.split(',')))
        end=start+sum((n-1)*z for n,z in zip(dims,strides))
        assert 0<=start<=end<pl['words']
        if kind!='out': continue
        assert var=='o' and int(bid)==2
        key=next(k for k in names if pl['offsets'][k]<=start<pl['offsets'][k]+pl['sizes'][k])
        for index in np.ndindex(*dims[:3]):
            positions=start-pl['offsets'][key]+sum(i*z for i,z in zip(index,strides))+np.arange(dims[3])*strides[3]
            assert positions.max()<writes[key].size; writes[key][positions]+=1
    return writes


def test_scorer_owns_every_output_word_once_at_one_m_group():
    pl=base(); names=('scorer_input','scorer_norm','scorer_packed','scorer_hidden','scorer_activated','scorer_scores','scorer_biased')
    writes=ownership(s.runtime(pl),pl,names)
    assert all(np.all(value==1) for value in writes.values())


def test_action_owns_every_output_word_once_and_waits_for_its_single_pack_column():
    pl=a.plan(base()); names=('tail_features','action_input','action_hidden','action_activated','action_scores','action_biased','action_probs')
    text=a.runtime(pl); writes=ownership(text,pl,names)
    assert all(np.all(value==1) for value in writes.values())
    assert text.count('aiex.npu.dma_wait {symbol = @outL2_0}')==23  # collect +17 pack +4 product/bias +final softmax


def stream(reads):
    body=b''.join(struct.pack('<12I',129,48,0,0,0,0,0x1d004,0,arg,0,offset,0) for arg,offset in reads)
    return struct.pack('<4I',0x06030000,260,len(reads),len(body)+16)+body


def test_marker_patch_clips_negative_and_changes_only_selected_read_fields():
    pl=base(); start=pl['offsets']['mlp_norm']; order=[col*32+group*4+row for group in range(8) for col in range(4) for row in range(4)]
    reads=[(0,(start+i*p.OUT)*4) for i in order]+[(1,start*4),(0,32)]
    raw=stream(reads); markers=np.array([-9,511,17],np.int64); updated=s.prepare(raw,pl,markers)
    ops=s.v.s.txn.parse(updated); expected=s.markers(markers)
    assert [op['arg_offset'] for op in ops[:128]]==[(start+int(expected[i])*p.OUT)*4 for i in order]
    assert [op['arg_offset'] for op in ops[128:]]==[start*4,32]
    allowed={j for op in s.v.s.txn.parse(raw)[:128] for j in range(op['offset']+40,op['offset']+44)}
    assert all(x==y or i in allowed for i,(x,y) in enumerate(zip(raw,updated)))
    with pytest.raises(ValueError): s.prepare(stream(reads[1:]),pl,markers)


@pytest.mark.parametrize('value',[np.array([0],np.int64),np.zeros(129,np.int64),np.array([0,512],np.int64),np.zeros(2,np.float32),np.zeros((1,2),np.int64)])
def test_bad_marker_vectors_refused_before_instruction_patching(value):
    with pytest.raises(ValueError): s.markers(value)


@pytest.mark.parametrize('value',[np.zeros(1,bool),np.zeros(129,bool),np.zeros(3,np.int64),np.zeros((1,3),bool)])
def test_invalid_tail_metadata_refused(value):
    with pytest.raises(ValueError): a.metadata(value)


def test_real_masked_options_are_preserved_but_padding_is_excluded_from_reductions():
    mask=np.array([False,False,False]); raw=a.metadata(mask)
    assert raw[0]==3 and not raw[16:144].any()
    logits,prob,features,logs=tp.oracle(3,mask,np.zeros(128,np.float32))
    assert np.array_equal(logits,np.full(128,-10000,np.float32))
    assert np.allclose(prob,1/3) and len(prob)==3
    assert np.isclose(features[2],np.log(3)/np.log(2)) and np.isclose(features[3],2/255)


def test_tail_image_suppresses_unused_operator_families_and_bounds_probe_dma():
    text=t.prefix()
    assert text.count('%is_norm = arith.constant false')==16
    assert text.count('%pack = arith.constant false')==16
    assert text.count('func.call @tail_collect(')==16
    for mode in (0,1,3,9,10,11,12,13,17,18):
        with pytest.raises(ValueError): t.Sequence(128000).control(mode)
    entries=re.findall(r'dma_memcpy_nd\(%[ao]\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]\)',t.runtime())
    for offset,dimensions,strides in entries:
        maximum=int(offset)+sum((int(n)-1)*int(z) for n,z in zip(dimensions.split(','),strides.split(',')))
        assert maximum<128000


def test_embedding_carry_overwrites_only_complete_norm_packets():
    pl=a.plan(base()); writes=ownership(ec.runtime(pl),pl,('mlp_norm',))
    assert np.all(writes['mlp_norm']==1)
    text=s.v.prefix()
    assert text.count('func.call @vector_carry(')==16
    assert text.count('%head_norm = arith.ori %head_seed_or_sum,%carry : i1')==16
