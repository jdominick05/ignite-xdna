# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Embedding transfer ownership, descriptor reuse and explicit address patching."""
from pathlib import Path
import re
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import stem as s
import embedded as e
p=s.p


@pytest.mark.parametrize('count',[16,512])
def test_every_embedding_output_word_and_packet_source(count):
    tokens=np.arange(count,dtype=np.int64); tokens[-1]=s.VOCAB-1
    words=512*p.OUT+32; text=s.runtime(tokens,words); writes=np.zeros(words,np.uint8); table_reads=[]
    entries=re.findall(r'dma_memcpy_nd\(%([ao])\[0,0,0,(\d+)\]\[([^]]+)\]\[([^]]+)\]\) \{id = (\d+).*metadata = @(q|kv|out)L2_(\d+)',text)
    for var,offset,dimensions,strides,bid,kind,col in entries:
        offset=int(offset); col=int(col); dims=list(map(int,dimensions.split(','))); stride=list(map(int,strides.split(',')))
        if kind=='out':
            assert var=='o' and int(bid)==2
            for outer in np.ndindex(*dims[:3]): writes[offset+sum(i*z for i,z in zip(outer,stride))+np.arange(dims[3])*stride[3]]+=1
        elif kind=='q' and var=='a':
            assert offset%1024==0 and offset+p.QWORDS<=s.AFFINE
            table_reads.append((col,offset//1024,int(bid)))
        elif kind=='q':
            assert var=='o' and offset==16+col*128*p.OUT
        elif kind=='kv': assert var=='a' and offset==s.AFFINE
    percol=count//4; expected=np.zeros(words,np.uint8)
    for col in range(4): expected[16+col*128*p.OUT:16+(col*128+percol)*p.OUT]=2
    assert np.array_equal(writes,expected)
    assert table_reads==[(col,int(tokens[col*percol+group*4+row]),bid) for group in range(percol//4) for col in range(4) for row,bid in enumerate((0,1,4,5))]


def test_stem_commands_suppress_the_generic_gemm_output():
    text=s.prefix()
    assert text.count('%not_norm = arith.cmpi eq, %is_norm,%not_norm_false : i1')==16
    assert '%not_norm = arith.cmpi ne, %command, %one' not in text
    assert text.count('func.call @stem_copy(')==text.count('func.call @stem_norm(')==16


def stream(reads):
    payload=[]
    for arg,offset in reads: payload.append(struct.pack('<12I',129,48,0,0,0,0,0x1d004,0,arg,0,offset,0))
    body=b''.join(payload)
    return struct.pack('<4I',0x06030000,260,len(payload),len(body)+16)+body


def test_prepare_respects_interleaved_group_column_order_and_only_patches_embedding_addresses():
    original=np.arange(512,dtype=np.int64); new=original[::-1].copy(); new[-1]=s.VOCAB-1
    order=[col*128+group*4+row for group in range(32) for col in range(4) for row in range(4)]
    reads=[(0,int(original[i])*4096) for i in order]+[(0,s.AFFINE*4),(1,64)]
    raw=stream(reads); prepared,offsets=s.prepare(raw,original,new); ops=s.txn.parse(prepared)
    assert [op['arg_offset'] for op in ops[:512]]==[int(new[i])*4096 for i in order]
    assert [op['arg_offset'] for op in ops[512:]]==[s.AFFINE*4,64]
    allowed={byte for start in offsets for byte in range(start,start+4)}
    assert all(a==z or i in allowed for i,(a,z) in enumerate(zip(raw,prepared)))


def test_embedding_output_relocation_preserves_readonly_table_addresses():
    raw=stream([(0,0),(0,s.AFFINE*4),(1,64),(1,(16+511*p.OUT)*4)])
    patched,changes=e.relocate_output(raw,16,4096,4096+512*p.OUT)
    ops=s.txn.parse(patched)
    assert [op['arg_offset'] for op in ops]==[0,s.AFFINE*4,4096*4,(4096+511*p.OUT)*4]
    assert len(changes)==2
    with pytest.raises(ValueError): e.relocate_output(raw,16,4096,4096+512*p.OUT-1)


@pytest.mark.parametrize('tokens',[np.zeros(15,np.int64),np.full(16,-1,np.int64),np.full(16,s.VOCAB,np.int64),np.zeros(16,np.float32),np.zeros((1,16),np.int64)])
def test_bad_token_vectors_are_refused(tokens):
    with pytest.raises(ValueError): s.validate_tokens(tokens)
