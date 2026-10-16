# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Query reuse must retain each original product and the qualified lookup formula."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import window_lookup as candidate


def addresses(event):
    result=np.zeros(event['dims'],np.int64)
    for axis,(size,stride) in enumerate(zip(event['dims'],event['strides'])):
        shape=[1]*4;shape[axis]=size
        result+=np.arange(size,dtype=np.int64).reshape(shape)*stride
    return (result+event['offset']).reshape(-1)


@pytest.mark.parametrize('m',[128,256,384,512])
def test_every_output_retains_original_query_weights_and_k_order(m):
    parent=candidate.window.bm.b.MlpSequence
    class Trace(parent):
        def __init__(self,*args):super().__init__(*args);self.events=[]
        def dma(self,col,kind,offset,dims,strides,output=False):
            self.events.append(dict(mode=self.current[0],width=self.current.get(6,1),col=col,kind=kind,
                offset=offset,dims=list(dims),strides=list(strides),output=output))
        def wait(self):pass
        def wait_columns(self,columns):pass
        def finish(self):return self.events
    with patch.object(candidate.window.bm.b,'MlpSequence',Trace):
        plan=candidate.window.bm.b.plan()
        old=candidate.sq.runtime(plan,m);new=candidate.runtime(plan,m)
    assert candidate.window.WIDTH==3
    for mode,groups,kblocks in ((2,21,16),(9,4,41)):
        for col in range(4):
            before=[event for event in old if event['mode']==mode and event['col']==col]
            after=[event for event in new if event['mode']==mode and event['col']==col]
            assert len(after)==((groups+1)//2)*(m//128)*3
            seen=set()
            for index in range(0,len(after),3):
                items={event['kind']:event for event in after[index:index+3]}
                window=(index//3)//(m//128);mg=(index//3)%(m//128);first=window*2
                width=min(2,groups-first)
                assert {event['width'] for event in items.values()}=={width}
                queries=addresses(items['q']).reshape(kblocks,4,1040)
                weights=addresses(items['kv']).reshape(kblocks,width,2064)
                outputs=addresses(items['out']).reshape(width,4*2080)
                for slot in range(width):
                    group=first+slot;seen.add((group,mg))
                    reference={event['kind']:event for event in before[(group*(m//128)+mg)*3:(group*(m//128)+mg+1)*3]}
                    assert np.array_equal(queries,addresses(reference['q']).reshape(kblocks,4,1040))
                    assert np.array_equal(weights[:,slot],addresses(reference['kv']).reshape(kblocks,2064))
                    assert np.array_equal(outputs[slot],addresses(reference['out']))
            assert seen=={(group,mg) for group in range(groups) for mg in range(m//128)}
    for mode in (1,3):
        assert [event for event in old if event['mode']==mode]==[event for event in new if event['mode']==mode]


def test_lookup_math_is_unchanged_except_query_pointer_selection():
    source=candidate.lookup_source()
    restored=source.replace('void chain_geglu_split(const int32_t *q,const int32_t *second,const int32_t *v,int32_t *out,int poison)',
        'void chain_geglu(const int32_t *q,const int32_t *v,int32_t *out,int poison)')
    restored=restored.replace('aie::load_v<16>(row<16 ? gate+row*64+col : reinterpret_cast<const float *>(second)+(row-16)*64+col)',
        'aie::load_v<16>(gate+row*64+col)')
    assert restored==Path(candidate.lookup.__file__).with_suffix('.cc').read_text()


def test_two_states_and_reduced_core_kv_depth_are_explicit():
    text=candidate.prefix()
    assert text.count(': memref<4096xi32>')==16
    assert 'memref<6144xi32>' not in text
    core_kv=[line for line in text.splitlines() if 'aie.objectfifo @kvF_' in line]
    assert len(core_kv)==4 and all(',1 : i32)' in line for line in core_kv)
    mem_kv=[line for line in text.splitlines() if 'aie.objectfifo @kvL2_' in line]
    assert len(mem_kv)==4 and all(',2 : i32)' in line for line in mem_kv)
