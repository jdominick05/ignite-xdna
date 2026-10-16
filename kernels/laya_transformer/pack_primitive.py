#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline qualification of a one-column packing primitive before device use."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import compact_pack_wide as wide
af=wide.cp.af; ba=wide.ba; p=wide.p; n=wide.n


def plan():
    output=128*p.OUT+64
    return dict(words=output+4*p.OUT+64,offsets=dict(mlp_norm=16,packed_norm=output))


class Sequence(ba.r.Sequence):
    def __init__(self,words):
        super().__init__(words); self.active=False; self.groups=0
    def control(self,*args,**kwargs):
        self.active=self.groups==0; self.groups+=1
        if self.active: super().control(*args,**kwargs)
    def dma(self,col,*args,**kwargs):
        if self.active and col==0: super().dma(col,*args,**kwargs)
    def wait(self):
        if self.active: self.lines.append('      aiex.npu.dma_wait {symbol = @outL2_0}')


def runtime(compact):
    layout=plan(); seq=Sequence(layout['words'])
    (wide.pack if compact else ba.pack)(seq,layout,layout['offsets']['mlp_norm'],128)
    return seq.finish()


def cdo_writes(data):
    """Decode the local bootgen writer's CDO v2 write records, bounded by input."""
    # Format source: third_party/bootgen/cdo-driver/cdo_driver.c.
    if len(data)%4 or len(data)<20: raise ValueError('CDO framing')
    words=struct.unpack('<%dI'%(len(data)//4),data)
    if words[0]!=4 or words[1]!=0x004f4443 or words[2]!=0x200:
        raise ValueError('unexpected CDO header')
    if (sum(words[:5])&0xffffffff)!=0xffffffff:
        raise ValueError('CDO header checksum')
    if words[3]!=len(words)-5: raise ValueError('CDO body length')
    index=5; result=[]
    while index<len(words):
        header=words[index]; index+=1
        if header>>24: raise ValueError('reserved CDO header bits')
        command=header&0xffff; size=(header>>16)&255
        if size==255:
            if index>=len(words): raise ValueError('CDO extended size')
            size=words[index]; index+=1
        if size>len(words)-index: raise ValueError('CDO truncated record')
        payload=words[index:index+size]; index+=size
        if command==0x111: continue
        if command in (0x108,0x107):
            if size!={0x108:3,0x107:4}[command]: raise ValueError('CDO scalar write size')
            address=(payload[0]<<32)|payload[1]
            result.append((address,0xffffffff if command==0x108 else payload[2],payload[-1]))
        elif command==0x105:
            if size<3: raise ValueError('empty CDO DMA write')
            address=(payload[0]<<32)|payload[1]
            result.extend((address+4*i,0xffffffff,value) for i,value in enumerate(payload[2:]))
        else: raise ValueError('unsupported CDO command '+hex(command))
    return result


def init_proof(original,compact,old_audits,new_audits):
    def span(audits,col,row,value):
        start=((value>>14)&0x3fff)*4; length=(value&0x3fff)*4
        audit=next(a for a in audits if a['core']==[col,row])
        matches=[(name,start-begin,end-begin) for begin,end,name in audit['spans'] if begin<=start and start+length<=end]
        if len(matches)!=1: raise ValueError('BD does not fit exactly one audited buffer')
        name,offset,size=matches[0]
        if offset or length!=size: raise ValueError('BD is not a complete FIFO object')
        return name,length
    def records(path,audits,compact):
        by_register=defaultdict(list); locks=[]; changes=[]
        for address,mask,value in cdo_writes(path.read_bytes()):
            col=address>>25; row=(address>>20)&31; reg=address&0xfffff
            key=(address,mask)
            if row>=2 and 0x1d000<=reg<0x1d200 and (reg-0x1d000)%32==0:
                if mask!=0xffffffff: raise ValueError('masked core BD address/length')
                name,length=span(audits,col,row,value)
                if name.startswith('kvF_'):
                    if length!=(2112 if compact else 2064)*4: raise ValueError('KV core packet length')
                    changes.append(('core_kv',col,row)); length=2064*4
                value=(name,length)
            elif row==1 and reg in (0xa0100,0xa0120,0xa0300,0xa0320):
                if mask!=0xffffffff or value!=(2112 if compact else 2064):
                    raise ValueError('KV MemTile packet length')
                changes.append(('mem_kv',col,reg)); value=2064
            if (row>=2 and 0x1f000<=reg<0x1f100) or (row==1 and 0xc0000<=reg<0xc0400):
                locks.append((address,mask,value))
            by_register[key].append(value)
        return dict(by_register),locks,changes
    before,old_locks,old_changes=records(original,old_audits,False)
    after,new_locks,new_changes=records(compact,new_audits,True)
    # Initialization can visit independent lock registers in a different order.
    # Preserve the ordered trace for EACH register (including every lock), rather
    # than requiring identical interleaving before the program is enabled.
    if before!=after or len(old_locks)!=len(new_locks) or old_changes!=new_changes:
        raise ValueError('initial descriptor/packet/lock state differs beyond checked KV size and buffer relocation')
    if not old_locks or len(old_changes)!=48:
        raise ValueError('incomplete CDO lock/descriptor coverage')
    return dict(register_traces=len(before),lock_writes=len(old_locks),normalized_kv_descriptors=len(old_changes),
        lock_register_traces_identical=True,other_bd_fields_including_tlast_identical=True,
        normalization='audited complete core buffers by name; KV object2064 to2112 words; all other register traces exact')


def expected(values):
    """Independent row/feature to GEMM operand layout, returned as payload words."""
    values=np.asarray(values,np.float32)
    if values.shape!=(128,p.OUT): raise ValueError('primitive source shape')
    bits=values[:,:64].view(np.uint32)
    rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
    tiles=rounded.reshape(4,8,4,8,8).transpose(0,1,3,2,4).copy()
    return tiles.reshape(4,2048).view(np.uint32)


def sources():
    return dict(wide.sources(),**{'kernels/laya_transformer/pack_primitive.py':p.attention.sha(Path(__file__))})


def topology_proof():
    restored=wide.prefix().replace('2112xi32','2064xi32')
    restored=restored.replace('@norm_pack_compact_step(','@norm_pack_step(')
    restored=restored.replace('%c128 = arith.constant 4 : index','%c128 = arith.constant 128 : index')
    restored=restored.replace('%c127 = arith.constant 3 : index','%c127 = arith.constant 127 : index')
    if restored!=af.prefix(2,2,2):
        raise ValueError('unaccounted FIFO topology or acquire/release change')
    return dict(fifo_links_depths_and_acquire_release_structure_identical=True,
        changes='KV packet type2112; packing consumes4 whole packets instead of128; helper consumes first64 of each66-float row')


def shim_proof(original,compact,old_audits,new_audits):
    def signature(data,audits,compact):
        controls={}
        for address,(core,field) in n.c.txn.registers(audits).items():
            index={'mode':0,'valid':5,'local':6}[field]
            controls.update({address+(i-index)*4:(core,i) for i in range(7)})
        state={}; operations=[]; descriptors=[]; patches=[]; queues=[]
        for op in n.c.txn.parse(data):
            if op.get('addr') in controls:
                if op['op']!='WRITE': raise ValueError('non-scalar primitive control')
                state[controls[op['addr']]]=op['val']; continue
            if op['op']=='BLOCKWRITE':
                if op['addr'] not in (0x1d000,0x1d040,0x1d060) or len(op['words'])!=8:
                    raise ValueError('unexpected primitive Shim BD')
                words=list(op['words']); descriptors.append(op['addr'])
                if op['addr']==0x1d060:
                    expected=([2112,0,0,66<<20,(2<<30)|(32<<20)|(p.OUT-1),2<<24,
                        (3<<20)|(32*p.OUT-1),2<<24] if compact else
                        [128*2064,0,0,16<<20,(2<<30)|(129<<20)|15,(2<<24)|(p.OUT-1),0,2<<24])
                    if words!=expected: raise ValueError('KV gather BD/packet/iteration fields differ')
                    # Normalize only after every dimension, repeat, TLAST and lock bit passes.
                    words=['checked_kv']
                operations.append(('BLOCKWRITE',op['addr'],words))
            elif op['op']=='WRITE' and op.get('addr')==0x1d21c:
                if op['val']!=(0x30003 if compact else 3): raise ValueError('KV queue repetition differs')
                queues.append(op['addr']); operations.append(('checked_kv_queue',))
            else:
                if op['op']=='DDR_PATCH': patches.append((op['addr'],op['arg_idx'],op['arg_offset']))
                if op['op']=='WRITE': queues.append(op['addr'])
                operations.append(data[op['offset']:op['offset']+op['size']])
        if descriptors!=[0x1d000,0x1d040,0x1d060] or queues!=[0x1d214,0x1d204,0x1d21c]:
            raise ValueError('unexpected descriptor/queue coverage')
        layout=plan()
        if patches!=[(0x1d004,0,64),(0x1d044,1,layout['offsets']['packed_norm']*4),(0x1d064,0,64)]:
            raise ValueError('primitive operand offset/bank differs')
        if len(state)!=112: raise ValueError('incomplete primitive RTP controls')
        return operations,state
    if signature(original,old_audits,False)!=signature(compact,new_audits,True):
        raise ValueError('primitive non-KV transport, controls or waits differ')
    return dict(query_and_output_bd_bytes_identical=True,queue_order_and_completion_waits_identical=True,
        kv_single_complete_bd=True,kv_packet_words=2112,kv_repetitions=4,tlast_suppress=False,
        iteration_stride_words=32*p.OUT,receiver_consumes_packets=4)


def build(original,compact,out):
    out=out.absolute(); frozen=sources(); out.mkdir(parents=True,exist_ok=False); layouts={}
    parents={}
    for name,folder,module in [('original',original,af),('compact',compact,wide)]:
        meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=module.sources(): raise ValueError('stale primitive parent '+name)
        for file,key in [('kernel.o','kernel_object_sha256'),('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('changed primitive parent '+name)
        parents[name]=meta
        dest=out/name; dest.mkdir(); shutil.copyfile(folder/'kernel.o',dest/'kernel.o')
        prefix=wide.prefix() if name=='compact' else af.prefix(2,2,2)
        (dest/'harness.mlir').write_text(prefix+runtime(name=='compact'),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',dest/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={dest/"prj"}'],dest)
        old=p.STACK
        try:
            p.STACK=4864; audits=json.loads(json.dumps(p.audit(dest,{'fp_exp_ab':2048,'fp_exp_cd':2048})))
        finally: p.STACK=old
        cdo={f.name:p.attention.sha(f) for f in (dest/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=meta['audits'] or cdo!=meta['cdo']: raise ValueError('primitive changes parent core/init '+name)
        layouts[name]=dict(audits=audits,cdo=cdo,instructions=p.attention.sha(dest/'insts.bin'),xclbin=p.attention.sha(dest/'final.xclbin'))
    topology=topology_proof()
    proof=init_proof(out/'original/prj/cdo_main/main_aie_cdo_init.bin',out/'compact/prj/cdo_main/main_aie_cdo_init.bin',
        layouts['original']['audits'],layouts['compact']['audits'])
    shim=shim_proof((out/'original/insts.bin').read_bytes(),(out/'compact/insts.bin').read_bytes(),
        layouts['original']['audits'],layouts['compact']['audits'])
    if frozen!=sources(): raise ValueError('primitive source changed')
    p.attention.dump(out/'build.json',dict(sources=frozen,layouts=layouts,plan=plan(),init_proof=proof,topology_proof=topology,shim_proof=shim,
        parents={name:p.attention.sha(folder/'build.json') for name,folder in [('original',original),('compact',compact)]},
        scope='one packing group, column0/four cores; offline transport checks pass; device qualification pending'))
    print('PACK_PRIMITIVE_BUILD '+json.dumps(dict(layouts=list(layouts),init_proof=proof,topology_proof=topology,
        shim_proof=shim,scope='offline only; no device qualification')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('original','compact','out'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args(); build(args.original,args.compact,args.out)
