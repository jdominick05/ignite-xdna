# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""All-word weight/output mapping and unchanged per-tile K order for N windows."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import window_mlp as w


def addresses(event):
    result=np.zeros(event['dims'],np.int64)
    for axis,(size,stride) in enumerate(zip(event['dims'],event['strides'])):
        shape=[1]*4; shape[axis]=size
        result+=np.arange(size,dtype=np.int64).reshape(shape)*stride
    return (result+event['offset']).reshape(-1)


def schedules(monkeypatch,m):
    original=w.bm.b.MlpSequence
    class Trace(original):
        def __init__(self,*args): super().__init__(*args); self.events=[]
        def dma(self,col,kind,offset,dims,strides,output=False):
            self.events.append(dict(mode=self.current[0],kblocks=self.current[1],width=self.current.get(6,1),
              col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output))
        def wait(self): pass
        def wait_columns(self,columns): pass
        def finish(self): return self.events
    monkeypatch.setattr(w.bm.b,'MlpSequence',Trace)
    plan=w.bm.b.plan()
    old=w.sq.runtime(plan,m); new=w.runtime(plan,m)
    assert w.bm.b.MlpSequence is Trace
    return old,new


@pytest.mark.parametrize('m',[128,256,384,512])
def test_all_window_weights_queries_outputs_match_original_tiles_in_k_order(monkeypatch,m):
    old,new=schedules(monkeypatch,m); mg_count=m//128
    for mode,groups,kblocks in ((2,21,16),(9,4,41)):
        for col in range(4):
            reference=[event for event in old if event['mode']==mode and event['col']==col]
            windows=[event for event in new if event['mode']==mode and event['col']==col]
            assert len(reference)==groups*mg_count*3
            assert len(windows)==((groups+2)//3)*mg_count*3
            seen=set()
            for index in range(0,len(windows),3):
                rows={event['kind']:event for event in windows[index:index+3]}
                window_index=(index//3)//mg_count; mg=(index//3)%mg_count; first=window_index*3
                width=min(3,groups-first)
                assert all(event['width']==width and event['kblocks']==kblocks for event in rows.values())
                query=addresses(rows['q']).reshape(kblocks,4,1040)
                weights=addresses(rows['kv']).reshape(kblocks,width,2064)
                outputs=addresses(rows['out']).reshape(width,4*2080)
                for slot in range(width):
                    group=first+slot; seen.add((group,mg))
                    before={event['kind']:event for event in reference[(group*mg_count+mg)*3:(group*mg_count+mg+1)*3]}
                    assert np.array_equal(query,addresses(before['q']).reshape(kblocks,4,1040))
                    # Equality by K index rules out an altered accumulation order.
                    assert np.array_equal(weights[:,slot],addresses(before['kv']).reshape(kblocks,2064))
                    assert np.array_equal(outputs[slot],addresses(before['out']))
            assert seen=={(group,mg) for group in range(groups) for mg in range(mg_count)}
    for mode in (1,3):
        assert [event for event in old if event['mode']==mode]==[event for event in new if event['mode']==mode]


def test_each_output_slot_has_independent_state_and_original_counter_reset():
    text=w.prefix()
    assert text.count('aie.buffer(')>=16
    assert text.count(': memref<6144xi32>')==16
    assert text.count('%width_i = memref.load')==32
    for i in range(16):
        core=text.split(f'%core{i} = aie.core',1)[1].split(' {link_with',1)[0]
        assert core.count('%reset_bool = arith.cmpi eq, %pos,%zero : i32')==2
        assert core.count('%next_pos = arith.select %final_k,%zero,%next : i32')==2
        assert core.count(f'@window_gemm_step(%q,%kv,%state{i},%reset,%slot)')==2
        assert core.count(f'@window_finish(%state{i},%o,%tiled,%slot)')==2
        assert core.count('scf.for %output_slot = %c0 to %width step %c1')==2
        assert core.count('%slot = arith.index_cast %output_slot : index to i32')==2
    helper=Path(w.__file__).with_suffix('.cc').read_text()
    assert helper.count('states+slot*2048')==2
