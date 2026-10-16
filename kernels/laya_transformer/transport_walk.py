# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline all-word Shim descriptor walk against the generated operand schedule."""
from unittest.mock import patch
import numpy as np
import pack_primitive as primitive
p=primitive.p; ba=primitive.ba; n=primitive.n


class Recorder:
    def __init__(self,words): self.events=[]; self.mode=None
    def control(self,mode,*args,**kwargs): self.mode=mode
    def dma(self,col,kind,offset,dims,strides,output=False,bid=None):
        self.events.append(dict(col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output,mode=self.mode))
    def wait(self): pass
    def finish(self): return self.events


def addresses(event):
    result=np.zeros(event['dims'],np.int64)
    for axis,(size,stride) in enumerate(zip(event['dims'],event['strides'])):
        shape=[1]*4;shape[axis]=size
        result+=np.arange(size,dtype=np.int64).reshape(shape)*stride
    return result.reshape(-1)+event['offset']


def descriptor_addresses(words,repeat,offset):
    """AIE2 Shim fields from local xaiemlgbl_params.h, steps encoded minus one."""
    length=words[0]; wrap0=(words[3]>>20)&1023;wrap1=(words[4]>>20)&1023
    if not 0<length<=4_000_000 or not 1<=repeat<=64: raise ValueError('descriptor extent bound')
    index=np.arange(length,dtype=np.int64)
    if wrap0:
        first=index%wrap0; rest=index//wrap0
        if wrap1:
            second=rest%wrap1;third=rest//wrap1
        else: second=rest;third=np.zeros_like(rest)
    else: first=index;second=third=np.zeros_like(index)
    one=first*((words[3]&0xfffff)+1)+second*((words[4]&0xfffff)+1)+third*((words[5]&0xfffff)+1)
    iteration_wrap=((words[6]>>20)&63)+1;current=(words[6]>>26)&63
    if current>=iteration_wrap: raise ValueError('descriptor iteration state exceeds wrap')
    iterations=(np.arange(repeat,dtype=np.int64)+current)%iteration_wrap
    return (one[None,:]+iterations[:,None]*((words[6]&0xfffff)+1)+offset).reshape(-1)


def verify(raw,plan,compact):
    with patch.object(ba.r,'Sequence',Recorder):
        expected=(primitive.wide.runtime if compact else ba.runtime)(plan,128)
    descriptors={};operands={};index=0;words_checked=0;packets=0
    for op in n.c.txn.parse(raw):
        addr=op.get('addr',0);col=addr>>25;reg=addr&0x1ffffff
        if op['op']=='BLOCKWRITE':
            if not 0x1d000<=reg<0x1d200 or len(op['words'])!=8: raise ValueError('unexpected transport BD')
            descriptors[col,(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':
            operands[col,(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='WRITE' and reg in (0x1d214,0x1d204,0x1d21c):
            if index>=len(expected): raise ValueError('extra queued transfer')
            event=expected[index];index+=1
            kind={0x1d214:'q',0x1d204:'out',0x1d21c:'kv'}[reg]
            bid=op['val']&15;repeat=((op['val']>>16)&255)+1
            if (event['col'],event['kind'])!=(col,kind): raise ValueError('transfer queue order differs')
            descriptor=descriptors[col,bid];arg,offset=operands[col,bid]
            if arg!=int(event['output']): raise ValueError('input/output operand differs')
            actual=descriptor_addresses(descriptor,repeat,offset);wanted=addresses(event)
            if not np.array_equal(actual,wanted): raise ValueError(f'compiled gather differs from schedule at transfer{index}, mode{event["mode"]}, {kind}')
            if actual.min()<0 or actual.max()>=plan['words']: raise ValueError('compiled transfer exceeds arena')
            if kind=='kv':
                packet=2112 if compact else 2064
                if actual.size%packet or descriptor[7]&0x84000000:
                    raise ValueError('partial KV object or unexpected TLAST/chaining')
                packets+=actual.size//packet
            words_checked+=actual.size
    if index!=len(expected): raise ValueError('missing queued transfers')
    return dict(transfers=index,words_checked=words_checked,kv_objects=packets,
        every_compiled_address_matches_schedule=True,all_transfers_within_arena=True)


def verify_pair(original,compact,plan):
    proofs=[verify(original,plan,False),verify(compact,plan,True)]
    with patch.object(ba.r,'Sequence',Recorder):
        before=ba.runtime(plan,128);after=primitive.wide.runtime(plan,128)
    if len(before)!=len(after): raise ValueError('transport event count differs')
    preserved=0
    for old,new in zip(before,after):
        if (old['col'],old['kind'],old['mode'])!=(new['col'],new['kind'],new['mode']):
            raise ValueError('transport command coverage differs')
        if old['mode']==10 and old['kind']=='kv': continue
        a=addresses(old);b=addresses(new)
        if old['kind']=='kv': b=b.reshape(-1,2112)[:,:2064].reshape(-1)
        if not np.array_equal(a,b): raise ValueError('original live operand addresses changed outside packing')
        preserved+=a.size
    return dict(original=proofs[0],compact=proofs[1],preserved_non_pack_live_operand_words=preserved,
        remaining_change='packing KV gathers, independently qualified by all-feature guarded primitive')
