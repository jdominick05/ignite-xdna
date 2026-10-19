"""BF16 packet layout parity against the original scalar tile loop."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from kernels.transformer_bf16.state import ArenaCompiler,ArenaProgram,pack_matrix


def scalar_pack(weight,precise):
    n,k=weight.shape
    pn,pk=((n+15)//16)*16,((k+255)//256)*256
    padded=np.zeros((pn,pk),np.uint16);padded[:n,:k]=weight
    rows=[]
    for row in range(0,pn,16):
        for column in range(0,pk,256):
            tile=padded[row:row+16,column:column+256]
            packed=tile.T if precise else tile.reshape(4,4,32,8).transpose(2,0,3,1)
            rows.append(packed.reshape(-1).copy())
    return np.stack(rows).view(np.uint32)


@pytest.mark.parametrize('shape',[(1,1),(15,255),(16,256),(17,257),(32,512),(256,512),(512,256),(4096,256)])
@pytest.mark.parametrize('precise',[False,True])
def test_packed_words_equal_scalar(shape,precise):
    # Include every bit pattern, noncontiguous inputs, padding and multiple tiles.
    weight=np.random.default_rng(49).integers(0,65536,size=shape[::-1],dtype=np.uint16).T
    original=weight.copy()
    actual=pack_matrix(weight,precise)
    assert actual.flags.c_contiguous
    assert np.array_equal(actual,scalar_pack(weight,precise))
    assert np.array_equal(weight,original)


def test_prepared_patch_changes_only_matrix_payload(tmp_path):
    p=ArenaProgram(compiler=ArenaCompiler(exact_norm=True,compact=True,small_packets=True))
    p.upload(0,np.ones(257,np.uint16))
    p.projection(0,np.zeros((17,257),np.uint16),512,label='matrix')
    p.download('result',512,17)
    # Avoid compiling: the transport captures packets and returns guarded output.
    p.build=lambda cache:tmp_path
    prepared=p.prepare(tmp_path)
    original=prepared._packets.copy()
    class Transport:
        def execute(self,folder,packets,**kwargs):
            self.packets=packets.copy()
            return np.full((2,1056),p.poison,np.uint32),{}
    t=Transport()
    weight=np.arange(17*257,dtype=np.uint16).reshape(17,257)
    prepared.execute(t,matrices={'matrix':weight})
    expected=original.copy()
    expected[1:5,144:2192]=scalar_pack(weight,False)
    assert np.array_equal(t.packets,expected)
    assert np.array_equal(prepared._packets,original)
    prepared.execute(t)
    assert np.array_equal(t.packets,original)


@pytest.mark.parametrize('weight',[np.zeros((1,1),np.float32),np.zeros((0,256),np.uint16),np.zeros(4,np.uint16)])
def test_invalid_matrix_rejected(weight):
    with pytest.raises(ValueError):pack_matrix(weight)
