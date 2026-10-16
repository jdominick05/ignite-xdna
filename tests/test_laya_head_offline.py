# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Head DMA ownership, bounds and strict nonempty-mask policy."""
from pathlib import Path
import re
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import head_attention as a
import head_post as post
import head_second as second
p=a.p


def ownership(text,pl,names):
    writes={name:np.zeros(pl['sizes'][name],np.uint8) for name in names}
    entries=re.findall(r'dma_memcpy_nd\(%([ao])\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]\) \{id = (\d+).*metadata = @(q|kv|out)L2_(\d+)',text)
    assert entries
    for var,offset,dimensions,strides,bid,kind,col in entries:
        start=int(offset); dims=list(map(int,dimensions.split(','))); strides=list(map(int,strides.split(',')))
        last=start+sum((n-1)*s for n,s in zip(dims,strides))
        assert 0<=start<=last<pl['words']
        if kind!='out': continue
        assert var=='o' and int(bid)==2
        name=next(n for n in names if pl['offsets'][n]<=start<pl['offsets'][n]+pl['sizes'][n])
        for index in np.ndindex(*dims[:3]):
            positions=start-pl['offsets'][name]+sum(i*s for i,s in zip(index,strides))+np.arange(dims[3])*strides[3]
            assert positions.max()<writes[name].size
            writes[name][positions]+=1
    return writes


def test_head_attention_writes_only_complete_query_key_value_attention_and_packed_banks():
    pl=a.plan(a.h.plan(a.b.plan()))
    names=('queries','keys','kv','attention','packed_attention')
    writes=ownership(a.runtime(pl),pl,names)
    assert all(np.all(value==1) for value in writes.values())
    # All formatting uses identity coefficients, never encoder rotary banks.
    assert pl['offsets']['head_identity']!=pl['offsets']['coeff']


@pytest.mark.parametrize('valid',[0,-1,513,1.5])
def test_head_attention_refuses_unqualified_or_invalid_mask(valid):
    with pytest.raises(ValueError): a.runtime(a.plan(a.h.plan(a.b.plan())),valid)


def test_head_post_scatter_does_not_overwrite_residual_before_normalization():
    pl=post.plan(a.plan(a.h.plan(a.b.plan())))
    names=('head_out','mlp_norm','norm','packed_norm','mlp_wi','head_relu','projection')
    writes=ownership(post.runtime(pl),pl,names)
    for name in ('mlp_norm','norm'):
        rows=writes[name].reshape(512,p.OUT)
        assert np.all(rows[:,:1040]==2) and np.all(rows[:,1040:]==1)
    for name in ('head_out','packed_norm','head_relu'): assert np.all(writes[name]==1)
    for name,packets in [('mlp_wi',64*16),('projection',16*16)]:
        assert np.all(writes[name][:packets*p.OUT]==1)
        assert np.all(writes[name][packets*p.OUT:]==0)


def test_head_identity_coefficients_are_exact_and_leave_padding_untouched():
    pl=a.plan(a.h.plan(a.b.plan())); fill=np.uint32(0x7fc12345)
    data=np.full(pl['words'],fill,np.uint32); original=data.copy(); a.initialize(data,pl)
    start=pl['offsets']['head_identity']; rows=data[start:start+16*p.KVWORDS].reshape(16,p.KVWORDS)
    assert np.all(rows[:,:1024]==np.float32(1).view(np.uint32)) and np.all(rows[:,1024:2048]==0)
    assert np.all(rows[:,2048:]==fill)
    assert np.array_equal(data[:start],original[:start]) and np.array_equal(data[start+16*p.KVWORDS:],original[start+16*p.KVWORDS:])


def test_second_qkv_pre_writes_only_packed_input_and_biased_projection():
    pl=second.plan(post.plan(a.plan(a.h.plan(a.b.plan()))))
    writes=ownership(second.pre_runtime(pl),pl,('packed_norm','projection'))
    assert np.all(writes['packed_norm']==1) and np.all(writes['projection']==2)


def test_second_post_uses_the_same_residual_safe_scatter_without_a_final_norm():
    pl=second.plan(post.plan(a.plan(a.h.plan(a.b.plan()))))
    names=('head_out','mlp_norm','norm','packed_norm','mlp_wi','head_relu','projection')
    writes=ownership(second.post_runtime(pl),pl,names)
    for name in ('mlp_norm','norm'):
        rows=writes[name].reshape(512,p.OUT)
        assert np.all(rows[:,:1040]==2) and np.all(rows[:,1040:]==1)
    text=second.post_runtime(pl)
    # Last operation must select exact raw sum rather than learned norm.
    assert 'arith.constant 18 : i32' in text
