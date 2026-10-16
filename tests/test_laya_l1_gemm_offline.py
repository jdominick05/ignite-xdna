# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The L1 oracle and actual compiled stream must fail closed before timing."""
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_l1_gemm as probe


def test_zero_calls_expects_explicit_positive_zero_not_product_sign():
    product=np.array([-13.,-1.,0.,1.,17.],np.float64)
    assert probe.expected_bits('generic',0,product).tolist()==[0,0,0,0,0]
    assert probe.expected_bits('specialized',0,product).tolist()==[0,0,0,0,0]
    assert probe.expected_bits('empty',131072,product).tolist()==[0,0,0,0,0]
    assert probe.expected_bits('generic',3,product).tolist()==np.array([-39.,-3.,0.,3.,51.],np.float32).view(np.uint32).tolist()


def compiled():
    folder=Path.cwd()/'scratch/laya_transformer/l1_gemm_v3'
    if not (folder/'build.json').exists():pytest.skip('compiled L1 diagnostic unavailable')
    return folder


def test_compiled_transport_and_core_memory_pass():
    folder=compiled()
    meta=probe.validate(folder)
    assert len(meta['variants'])==3
    assert len({row['artifacts']['insts.bin'] for row in meta['variants'].values()})==1


@pytest.mark.parametrize('mutation',['packet_length','wait_column'])
def test_compiled_descriptor_or_completion_corruption_is_rejected(mutation):
    raw=(compiled()/'generic/insts.bin').read_bytes();changed=bytearray(raw)
    ops=probe.walk.n.c.txn.parse(raw)
    op=next(op for op in ops if op['op']==('BLOCKWRITE' if mutation=='packet_length' else 'TCT'))
    offset=op['offset']+(16 if mutation=='packet_length' else 8)
    value=struct.unpack_from('<I',changed,offset)[0]
    struct.pack_into('<I',changed,offset,value^(1 if mutation=='packet_length' else 1<<16))
    with pytest.raises(ValueError):probe.stream_proof(bytes(changed))
