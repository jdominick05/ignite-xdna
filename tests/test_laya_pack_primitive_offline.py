# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Packing transport gates reject framing, packet and lock corruption."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import pack_primitive as pp


def cdo(body):
    header=[4,0x004f4443,0x200,len(body)]
    header.append((0xffffffff-sum(header))&0xffffffff)
    return struct.pack('<%dI'%(len(header)+len(body)),*(header+body))


def test_cdo_scalar_masked_and_block_write_order():
    data=cdo([(3<<16)|0x108,0,0x21f000,2,(4<<16)|0x107,0,0x21f000,15,1,
        (4<<16)|0x105,0,0x21d000,7,8,0x111])
    assert pp.cdo_writes(data)==[(0x21f000,0xffffffff,2),(0x21f000,15,1),
        (0x21d000,0xffffffff,7),(0x21d004,0xffffffff,8)]


@pytest.mark.parametrize('data',[b'',b'abc',cdo([(3<<16)|0x108,0,1]),cdo([0x999]),cdo([0xff0105])])
def test_cdo_rejects_bad_framing_and_unknown_records(data):
    with pytest.raises(ValueError): pp.cdo_writes(data)


def test_expected_payload_uses_every_source_row_and_feature_once():
    values=np.zeros((128,pp.p.OUT),np.float32)
    # Exactly representable bfloat16 identities including each row/feature bit.
    identities=np.arange(128*64,dtype=np.uint32).reshape(128,64)+0x3e00
    values[:,:64]=(identities<<16).view(np.float32)
    payload=pp.expected(values).view(np.uint16).reshape(4,-1)
    for core in range(4):
        for row in range(32):
            for feature in range(64):
                index=(row//4)*256+(feature//8)*32+(row%4)*8+feature%8
                assert payload[core,index]==identities[core*32+row,feature]


def test_runtime_only_dispatches_first_group_column_and_wait():
    for compact in (False,True):
        text=pp.runtime(compact)
        dma=[line for line in text.splitlines() if 'dma_memcpy_nd' in line]
        assert len(dma)==3
        assert all('L2_0' in line for line in dma)
        assert text.count('dma_wait')==1 and 'symbol = @outL2_0' in text
    assert pp.topology_proof()['fifo_links_depths_and_acquire_release_structure_identical']


@pytest.mark.parametrize('mutation',['tlast','length','repeat'])
def test_compiled_shim_gate_rejects_wrong_packet_boundaries(mutation):
    root=Path.cwd()/'scratch/laya_transformer'
    if not (root/'pack_primitive_v1/compact/insts.bin').exists():
        pytest.skip('offline compiled primitive unavailable')
    before=(root/'pack_primitive_v1/original/insts.bin').read_bytes()
    after=bytearray((root/'pack_primitive_v1/compact/insts.bin').read_bytes())
    a=json.loads((root/'attention_fifo_v1_double_m128/build.json').read_text())['audits']
    b=json.loads((root/'compact_pack_wide_v2_m128/build.json').read_text())['audits']
    if mutation=='repeat':
        op=next(op for op in pp.n.c.txn.parse(after) if op['op']=='WRITE' and op['addr']==0x1d21c)
        offset=op['offset']+16; bit=1<<16
    else:
        op=next(op for op in pp.n.c.txn.parse(after) if op['op']=='BLOCKWRITE' and op['addr']==0x1d060)
        offset=op['offset']+16+(28 if mutation=='tlast' else 0); bit=1<<31 if mutation=='tlast' else 1
    struct.pack_into('<I',after,offset,struct.unpack_from('<I',after,offset)[0]^bit)
    with pytest.raises(ValueError): pp.shim_proof(before,bytes(after),a,b)


def test_initialization_gate_rejects_changed_lock(tmp_path):
    root=Path.cwd()/'scratch/laya_transformer'
    old=root/'attention_fifo_v1_double_m128'; new=root/'compact_pack_wide_v2_m128'
    file=new/'prj/cdo_main/main_aie_cdo_init.bin'
    if not file.exists(): pytest.skip('offline compiled initialization unavailable')
    a=json.loads((old/'build.json').read_text())['audits']
    b=json.loads((new/'build.json').read_text())['audits']
    records=pp.cdo_writes(file.read_bytes())
    index=next(i for i,(address,mask,value) in enumerate(records) if address==0x21f000)
    address,mask,value=records[index]; records[index]=(address,mask,value^1)
    body=[]
    for address,mask,value in records:
        body.extend([(4<<16)|0x107,address>>32,address&0xffffffff,mask,value])
    changed=tmp_path/'changed.bin';changed.write_bytes(cdo(body))
    with pytest.raises(ValueError):
        pp.init_proof(old/'prj/cdo_main/main_aie_cdo_init.bin',changed,a,b)
