# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Width-only experiment must preserve operands and its parent's globals/image."""
import json
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import window_single as single


def test_single_output_preserves_every_small_query_operand_and_k_order():
    plan=single.parent.window.bm.b.plan();base=single.parent.window.bm.b.MlpSequence
    class Trace(base):
        def __init__(self,*args):super().__init__(*args);self.events=[]
        def dma(self,col,kind,offset,dims,strides,output=False):
            self.events.append(dict(col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output,mode=self.current[0]))
        def wait(self):pass
        def wait_columns(self,columns):pass
        def finish(self):return self.events
    with patch.object(single.parent.window.bm.b,'MlpSequence',Trace):
        old=single.parent.sq.runtime(plan,128)
    with patch.object(single.parent,'WIDTH',1):new=single.transport.events(plan)
    new=[event for event in new if event['kind']!='wait']
    assert len(old)==len(new)==519
    for before,after in zip(old,new):
        assert (before['col'],before['kind'],before['mode'],before['output'])==(after['col'],after['kind'],after['mode'],after['output'])
        assert after['width']==1
        assert np.array_equal(single.transport.walk.addresses(before),single.transport.walk.addresses(after))
    assert single.parent.WIDTH==2 and single.parent.window.WIDTH==3


def test_compiled_single_core_and_initialization_match_qualified_parent():
    root=Path.cwd()/'scratch/laya_transformer';folder=root/'window_single_v1_m128';parent=root/'window_lookup_v2_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled single-output image unavailable')
    meta=json.loads((folder/'build.json').read_text());old=json.loads((parent/'build.json').read_text())
    assert meta['audits']==old['audits'] and meta['cdo']==old['cdo']
    assert single.p.attention.sha(folder/'kernel.o')==single.p.attention.sha(parent/'kernel.o')
    proof=single.verify((folder/'insts.bin').read_bytes(),single.parent.window.bm.b.plan(),meta['audits'])
    assert proof['queues']==519 and proof['completion_waits']==173


def test_compiled_width_corruption_rejected_and_parent_width_restored():
    folder=Path.cwd()/'scratch/laya_transformer/window_single_v1_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled single-output image unavailable')
    meta=json.loads((folder/'build.json').read_text());raw=(folder/'insts.bin').read_bytes()
    audit=next(a for a in meta['audits'] if a['core']==[0,2]);base=next(a for a,b,name in audit['spans'] if name=='chain_control_0')
    op=next(o for o in single.parent.n.c.txn.parse(raw) if o['op']=='WRITE' and o['addr']==(2<<20)+base+24)
    changed=bytearray(raw);struct.pack_into('<I',changed,op['offset']+16,2)
    with pytest.raises(ValueError):single.verify(bytes(changed),single.parent.window.bm.b.plan(),meta['audits'])
    assert single.parent.WIDTH==2 and single.parent.window.WIDTH==3
