# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""All-word compiled transport and completion checks for the complete M128 MLP."""
import struct
from unittest.mock import patch

import numpy as np
import window_lookup as candidate
import transport_walk as walk
p=candidate.p;n=candidate.n


def events(plan):
    parent=candidate.window.bm.b.MlpSequence
    class Trace(parent):
        def __init__(self,*args):super().__init__(*args);self.events=[]
        def dma(self,col,kind,offset,dims,strides,output=False):
            self.events.append(dict(col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output,
                mode=self.current[0],kblocks=self.current[1],width=self.current.get(6,1)))
        def wait(self):self.wait_columns(range(4))
        def wait_columns(self,columns):
            self.events.extend(dict(kind='wait',col=col) for col in columns)
        def finish(self):return self.events
    with patch.object(candidate.window.bm.b,'MlpSequence',Trace):return candidate.runtime(plan,128)


def verify(raw,plan,audits):
    wanted=events(plan);index=0;descriptors={};operands={};controls={};words=0;waits=0;queues=0
    bases={}
    for audit in audits:
        col,row=audit['core'];core=col*4+row-2
        starts=[start for start,end,name in audit['spans'] if name==f'chain_control_{core}']
        if len(starts)!=1:raise ValueError('MLP control allocation')
        bases[core]=(col<<25)|(row<<20)|starts[0]
    if len(bases)!=16:raise ValueError('MLP core coverage')
    for op in n.c.txn.parse(raw):
        address=op.get('addr',0);col=address>>25;reg=address&0x1ffffff;row=(address>>20)&31
        if op['op']=='WRITE' and row>=2:controls[address]=op['val'];continue
        if op['op']=='BLOCKWRITE':
            if reg not in (0x1d000,0x1d040,0x1d060) or len(op['words'])!=8:raise ValueError('unexpected MLP descriptor')
            descriptors[col,(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':operands[col,(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='TCT':
            if index>=len(wanted) or wanted[index]['kind']!='wait':raise ValueError('unexpected MLP completion wait')
            event=wanted[index];index+=1
            # Local TxnEncoding.h: word2=dir|row<<8|col<<16;
            # word3=nrow<<8|ncol<<16|channel<<24. outL2 is S2MM channel0.
            if struct.unpack_from('<2I',raw,op['offset']+8)!=(event['col']<<16,0x10100):
                raise ValueError('MLP completion waits on wrong column/channel/direction')
            waits+=1
        elif op['op']=='WRITE' and reg in (0x1d214,0x1d204,0x1d21c):
            if index>=len(wanted):raise ValueError('extra MLP queue')
            event=wanted[index];index+=1
            kind={0x1d214:'q',0x1d204:'out',0x1d21c:'kv'}[reg]
            if (event['kind'],event['col'])!=(kind,col):raise ValueError('MLP queue order')
            bid=op['val']&15;bd=descriptors[col,bid];arg,offset=operands[col,bid]
            actual=walk.descriptor_addresses(bd,((op['val']>>16)&255)+1,offset)
            if arg!=int(event['output']) or not np.array_equal(actual,walk.addresses(event)):raise ValueError('MLP compiled addresses differ')
            if actual.min()<0 or actual.max()>=plan['words']:raise ValueError('MLP compiled arena overrun')
            packet={'q':1040,'kv':2064,'out':2080}[kind]
            if bd[0]%packet or bd[7]&0x84000000:raise ValueError('MLP partial FIFO packet/TLAST/chaining')
            expected={}
            for core,base in bases.items():
                for field,value in ((0,event['mode']),(1,event['kblocks']),(2,candidate.window.bm.b.r.mlp.POISON),(3,core%4),(6,event['width'])):
                    expected[base+4*field]=value
            if controls!=expected:raise ValueError('MLP effective control values differ at queue')
            words+=len(actual);queues+=1
    if index!=len(wanted) or not waits or not queues:raise ValueError('incomplete compiled MLP schedule')
    return dict(queues=queues,completion_waits=waits,word_addresses_checked=words,all_addresses_controls_and_packets_exact=True,
        scope='complete M128 MLP compiled transport only; numerical layer/full-model gates separate')
