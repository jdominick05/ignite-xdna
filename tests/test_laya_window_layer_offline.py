# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compiled full-layer controls/packets/waits and intentional capture stops."""
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/name) for name in ('tools','kernels/laya_transformer')]
import window_mlp_transport as transport
import laya_window_layer_probe as probe


def compiled():
    folder=Path.cwd()/'scratch/laya_transformer/window_lookup_v2_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled MLP unavailable')
    return (folder/'insts.bin').read_bytes(),json.loads((folder/'build.json').read_text())['audits']


def test_complete_compiled_mlp_address_control_and_wait_coverage():
    raw,audits=compiled();result=transport.verify(raw,transport.candidate.window.bm.b.plan(),audits)
    assert result['queues']==375 and result['completion_waits']==125
    assert result['word_addresses_checked']==10872384


@pytest.mark.parametrize('mutation',['query_length','mode','width','wait_column'])
def test_corrupt_complete_mlp_stream_is_rejected(mutation):
    raw,audits=compiled();ops=transport.n.c.txn.parse(raw);changed=bytearray(raw)
    if mutation=='query_length':
        op=next(o for o in ops if o['op']=='BLOCKWRITE' and o['addr']==0x1d000);offset=op['offset']+16;bit=1
    elif mutation=='wait_column':
        op=next(o for o in ops if o['op']=='TCT');offset=op['offset']+8;bit=1<<16
    else:
        audit=next(a for a in audits if a['core']==[0,2]);base=next(a for a,b,name in audit['spans'] if name=='chain_control_0')
        address=(2<<20)+base+(24 if mutation=='width' else 0)
        op=next(o for o in ops if o['op']=='WRITE' and o['addr']==address);offset=op['offset']+16;bit=1
    struct.pack_into('<I',changed,offset,struct.unpack_from('<I',changed,offset)[0]^bit)
    with pytest.raises(ValueError):transport.verify(bytes(changed),transport.candidate.window.bm.b.plan(),audits)


@pytest.mark.parametrize('completed',[False,True])
def test_diagnostic_stop_only_marks_authoritatively_completed_dispatch_healthy(completed):
    # This tests the capture hook's stop semantics, not NPU execution.
    records=[];bo=object();done=7;status=done if completed else 0
    owner=SimpleNamespace(arena=SimpleNamespace(sync=lambda direction:None),sync=SimpleNamespace(XCL_BO_SYNC_BO_FROM_DEVICE=1),
        av=np.array([123],np.uint32),failed=True,xrt=SimpleNamespace(ert_cmd_state=SimpleNamespace(ERT_CMD_STATE_COMPLETED=done)))
    selected=SimpleNamespace(bo=bo)
    handle=probe.CaptureKernel(owner,lambda *args:SimpleNamespace(wait=lambda timeout:status),selected,records)(3,bo)
    if completed:
        with pytest.raises(probe.Captured):handle.wait(5000)
        assert not owner.failed and len(records)==1
        owner.av[0]=456
        assert records[0][0][0]==records[0][1][0]==123
    else:
        assert handle.wait(5000)==0
        assert owner.failed and not records
