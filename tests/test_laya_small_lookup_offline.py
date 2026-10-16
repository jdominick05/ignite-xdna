# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one-state KV2 candidate retains qualified math and rejects bad transport."""
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import small_query_lookup as candidate
import small_lookup_checks as checks


def compiled():
    root=Path.cwd()/'scratch/laya_transformer';folder=root/'small_lookup_v3_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled small lookup image unavailable')
    return root,folder,json.loads((folder/'build.json').read_text())


def test_complete_compiled_small_lookup_transport_and_capacities():
    root,folder,meta=compiled();plan=candidate.sq.bm.b.plan()
    proof=candidate.verify((folder/'insts.bin').read_bytes(),candidate.events(plan),meta['audits'],plan['words'])
    assert proof==meta['transport'] and proof['queues']==519
    relation=checks.relation(root/'window_lookup_primitive_v2/original0',folder)
    assert relation['physical_kv_objects']==2 and relation['physical_query_objects']==3
    assert relation['compiled_bd_lock_fields']==208
    checks.ledger(folder);checks.ledger(folder,True)


def test_generated_math_and_tables_match_qualified_split_lookup():
    root,folder,meta=compiled();old=json.loads((root/'window_lookup_v2_m128/build.json').read_text())
    assert all(sha==old['generated_sources'][name] for name,sha in meta['generated_sources'].items())
    assert meta['unchanged_objects']==old['unchanged_objects']


@pytest.mark.parametrize('mutation',['query_length','rtp_mode','wait_column'])
def test_corrupt_compiled_small_lookup_transaction_is_rejected(mutation):
    _,folder,meta=compiled();raw=(folder/'insts.bin').read_bytes();changed=bytearray(raw);ops=candidate.n.c.txn.parse(raw)
    if mutation=='query_length':
        op=next(op for op in ops if op['op']=='BLOCKWRITE');offset=op['offset']+16;bit=1
    elif mutation=='wait_column':
        op=next(op for op in ops if op['op']=='TCT');offset=op['offset']+8;bit=1<<16
    else:
        address=(2<<20)+15872
        op=next(op for op in ops if op['op']=='WRITE' and op['addr']==address);offset=op['offset']+16;bit=1
    struct.pack_into('<I',changed,offset,struct.unpack_from('<I',changed,offset)[0]^bit)
    plan=candidate.sq.bm.b.plan()
    with pytest.raises(ValueError):candidate.verify(bytes(changed),candidate.events(plan),meta['audits'],plan['words'])


def test_aliased_kv_ring_is_rejected(monkeypatch):
    _,folder,_=compiled();path=folder/'prj/input_with_addresses.mlir';original=Path.read_text
    def changed(self,*args,**kwargs):
        text=original(self,*args,**kwargs)
        return text.replace('%kvF_0_0_cons_buff_1','%kvF_0_0_cons_buff_0') if self==path else text
    monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(ValueError):checks.ledger(folder)
