# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent packet-address and controller gates for the small-query experiment."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import small_query as sq


class Recorder:
    def __init__(self,words): self.events=[]; self.mode=None
    def control(self,mode,*args,**kwargs): self.mode=mode; self.events.append(('control',mode,args,kwargs))
    def dma(self,col,kind,offset,dims,strides,output=False):
        self.events.append(dict(mode=self.mode,col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output))
    def wait(self): self.events.append(('wait',))
    def wait_columns(self,columns): self.events.append(('wait_columns',tuple(columns)))
    def finish(self): return self.events


def addresses(event):
    values=np.zeros(event['dims'],np.int64)
    for axis,(size,stride) in enumerate(zip(event['dims'],event['strides'])):
        shape=[1]*len(event['dims']); shape[axis]=size
        values+=np.arange(size,dtype=np.int64).reshape(shape)*stride
    return (values+event['offset']).reshape(-1)


def schedules(monkeypatch,m):
    names=('packed_norm','mlp_wi','mlp_wi_weight','mlp_geglu','norm','mlp_wo_weight','mlp_norm','mlp_affine')
    plan=dict(words=100000000,offsets={name:(index+1)*4000000 for index,name in enumerate(names)})
    monkeypatch.setattr(sq.bm.b,'MlpSequence',Recorder)
    before=sq.bm.runtime(plan,m); after=sq.runtime(plan,m)
    assert sq.bm.b.MlpSequence is Recorder
    assert sq.p.QWORDS==2064
    return before,after


@pytest.mark.parametrize('m',[128,256,384,512])
def test_gemm_drops_only_unused_words_and_preserves_all_other_streams(monkeypatch,m):
    old,new=schedules(monkeypatch,m)
    old_gemm=[event for event in old if isinstance(event,dict) and event['mode'] in (2,9)]
    new_gemm=[event for event in new if isinstance(event,dict) and event['mode'] in (2,9)]
    assert len(old_gemm)==len(new_gemm)
    for before,after in zip(old_gemm,new_gemm):
        if before['kind']=='q':
            original=addresses(before).reshape(-1,2064)
            observed=addresses(after).reshape(-1,1040)
            assert np.array_equal(observed,original[:,:1040])
            assert after['offset']==before['offset'] and after['strides']==before['strides']
        else: assert before==after
    for mode in (1,3):
        for col in range(4):
            for kind in ('kv','out'):
                before=[addresses(event) for event in old if isinstance(event,dict) and (event['mode'],event['col'],event['kind'])==(mode,col,kind)]
                after=[addresses(event) for event in new if isinstance(event,dict) and (event['mode'],event['col'],event['kind'])==(mode,col,kind)]
                assert np.array_equal(np.concatenate(before),np.concatenate(after))


@pytest.mark.parametrize('m',[128,256,384,512])
def test_two_query_halves_reach_same_core_and_preserve_every_live_word(monkeypatch,m):
    old,new=schedules(monkeypatch,m)
    for mode in (1,3):
        for col in range(4):
            def stream(events):
                return np.concatenate([addresses(event) for event in events if isinstance(event,dict) and
                  (event['mode'],event['col'],event['kind'])==(mode,col,'q')])
            original=stream(old).reshape(-1,4,2064)
            received=stream(new).reshape(-1,2,4,1040)
            # MemTile distributes each consecutive four packets to cores0..3.
            for core in range(4):
                joined=np.concatenate((received[:,0,core],received[:,1,core]),axis=1)
                assert np.array_equal(joined[:,:2064],original[:,core])
                assert np.array_equal(joined[:,2064:],original[:,core,0,None]+np.arange(2064,2080))
            assert len(received)==len(original)


def test_controller_waits_for_data_before_reading_rtp_and_balances_acquisitions():
    text=sq.prefix()
    assert text.count('memref<4160xi32>')==4
    for i in range(16):
        core=text.split(f'%core{i} = aie.core',1)[1].split(' {link_with',1)[0]
        assert core.index(f'acquire @qF_{i}(Consume,1)')<core.index('%override = memref.load')
        assert core.count(f'acquire @qF_{i}(Consume,1)')==1
        assert core.count(f'acquire @qF_{i}(Consume,2)')==1
        for count in (1,2): assert core.count(f'release @qF_{i}(Consume,{count})')==1
        assert core.count('@chain_gemm_step(%q,%kv,')==2
        assert core.count('@command_norm_split(%q,%q1,%kv,%o)')==2
        assert core.count('@chain_geglu_split(%q,%q1,%kv,%go,%fill)')==2
    assert text.count('aie.objectfifo @kvF_')==4
    assert all('memref<2064xi32>' in line for line in text.splitlines() if 'aie.objectfifo @kv' in line)


def test_kernel_address_changes_preserve_rounding_fence_and_matmul_payload():
    sources=sq.kernel_sources()
    assert 'reinterpret_cast<const float *>(second)' in sources['norm']
    assert 'volatile int32_t mode_fence=rounded.get(0);' in sources['dispatch']
    assert 'aie::set_rounding(aie::rounding_mode::floor);' in sources['dispatch']
    assert 'reinterpret_cast<const bfloat16 *>(q+16)' in sources['mlp']
    assert 'reinterpret_cast<const float *>(second)+(row-16)*64+col' in sources['mlp']
    assert 'for(int i=0;i<2080;i+=16)' in sources['mlp']
    # Numerical device equivalence is deliberately not inferred from this static gate.


@pytest.mark.parametrize('m',[128,256,384,512])
def test_each_split_group_completes_before_descriptor_reuse(monkeypatch,m):
    _,events=schedules(monkeypatch,m); pending=[]
    for event in events:
        if isinstance(event,dict):
            if event['mode'] in (1,3): pending.append(event)
        elif event[0]=='control':
            assert not pending
        elif event[0]=='wait_columns' and pending:
            assert set(event[1])=={row['col'] for row in pending}
            for col in event[1]:
                rows=[row for row in pending if row['col']==col]
                assert len(rows)==3 and {row['kind'] for row in rows}=={'q','kv','out'}
                query=next(row for row in rows if row['kind']=='q')
                assert np.prod(query['dims'])==8*1040
                output=next(row for row in rows if row['kind']=='out')
                assert np.prod(output['dims'])==4*2080
            pending=[]
    assert not pending


@pytest.mark.parametrize('m',[128,256,384,512])
def test_split_query_descriptor_uses_factored_wrap_extents(m):
    import re
    names=('packed_norm','mlp_wi','mlp_wi_weight','mlp_geglu','norm','mlp_wo_weight','mlp_norm','mlp_affine')
    plan=dict(words=100000000,offsets={name:(index+1)*4000000 for index,name in enumerate(names)})
    text=sq.runtime(plan,m)
    matches=re.findall(r'aiex.npu.dma_memcpy_nd\([^\n]*?\[0,0,0,\d+\]\[([\d,]+)\][^\n]*metadata = @qL2_',text)
    assert matches
    for value in matches:
        dims=list(map(int,value.split(',')))
        if dims[0]==2:
            assert dims==[2,4,65,16]
        else:
            # The GEMM descriptor keeps its existing outer-one factoring form.
            # Compiler stride normalization, not raw MLIR extents, determines
            # the hardware wrap fields; the successful build audits that step.
            assert dims[0]==1 and dims[-1]==1040
