#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-state smaller-query MLP with adopted lookup math and double-buffered KV."""
import argparse
import json
from pathlib import Path
import shutil
import struct
from unittest.mock import patch

import numpy as np
import window_lookup as reference
import transport_walk as walk
sq=reference.sq;p=reference.p;n=reference.n
STACK=3328


def prefix():
    text=sq.replace(sq.prefix(),'stack_size = 5376 : i32',f'stack_size = {STACK} : i32',16)
    # The default control allocation clips the table's final 64 bytes. Reserve
    # a free bank-zero gap; audit() checks every actual local/neighbor span.
    for core in range(16):
        text=sq.replace(text,f'{{sym_name = "chain_control_{core}"}}',
            f'{{address = 15872 : i32, sym_name = "chain_control_{core}"}}')
    return text


def sources():
    return dict(reference.sources(),**{'kernels/laya_transformer/small_query_lookup.py':p.attention.sha(Path(__file__)),
        'kernels/laya_transformer/transport_walk.py':p.attention.sha(Path(walk.__file__))})


def events(plan):
    parent=sq.bm.b.MlpSequence
    class Trace(parent):
        def __init__(self,*args):super().__init__(*args);self.events=[]
        def dma(self,col,kind,offset,dims,strides,output=False):
            self.events.append(dict(col=col,kind=kind,offset=offset,dims=list(dims),strides=list(strides),output=output,
                mode=self.current[0],kblocks=self.current[1]))
        def wait(self):self.wait_columns(range(4))
        def wait_columns(self,columns):self.events.extend(dict(kind='wait',col=col) for col in columns)
        def finish(self):return self.events
    with patch.object(sq.bm.b,'MlpSequence',Trace):return sq.runtime(plan,128)


def verify(raw,wanted,audits,arena_words):
    bases={}
    for audit in audits:
        col,row=audit['core'];core=col*4+row-2
        starts=[a for a,b,name in audit['spans'] if name==f'chain_control_{core}']
        if len(starts)!=1:raise ValueError('small lookup control allocation')
        bases[core]=(col<<25)|(row<<20)|starts[0]
    if len(bases)!=16:raise ValueError('small lookup core coverage')
    index=words=waits=queues=0;descriptors={};operands={};controls={}
    for op in n.c.txn.parse(raw):
        address=op.get('addr',0);col=address>>25;reg=address&0x1ffffff;row=(address>>20)&31
        if op['op']=='WRITE' and row>=2:controls[address]=op['val'];continue
        if op['op']=='BLOCKWRITE':
            if reg not in (0x1d000,0x1d040,0x1d060) or len(op['words'])!=8:raise ValueError('unexpected small lookup BD')
            descriptors[col,(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':operands[col,(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='TCT':
            if index>=len(wanted) or wanted[index]['kind']!='wait':raise ValueError('unexpected small lookup completion')
            event=wanted[index];index+=1
            if struct.unpack_from('<2I',raw,op['offset']+8)!=(event['col']<<16,0x10100):raise ValueError('small lookup completion column/channel')
            waits+=1
        elif op['op']=='WRITE' and reg in (0x1d214,0x1d204,0x1d21c):
            if index>=len(wanted):raise ValueError('extra small lookup queue')
            event=wanted[index];index+=1;kind={0x1d214:'q',0x1d204:'out',0x1d21c:'kv'}[reg]
            if (event['col'],event['kind'])!=(col,kind):raise ValueError('small lookup queue order')
            bid=op['val']&15;bd=descriptors[col,bid];arg,offset=operands[col,bid]
            actual=walk.descriptor_addresses(bd,((op['val']>>16)&255)+1,offset)
            if arg!=int(event['output']) or not np.array_equal(actual,walk.addresses(event)):raise ValueError('small lookup compiled addresses')
            if actual.min()<0 or actual.max()>=arena_words:raise ValueError('small lookup arena bounds')
            if bd[0]%{'q':1040,'kv':2064,'out':2080}[kind] or bd[7]&0x84000000:raise ValueError('small lookup partial object/TLAST/chaining')
            expected={}
            for core,base in bases.items():
                for field,value in ((0,event['mode']),(1,event['kblocks']),(2,sq.bm.b.r.mlp.POISON),(3,core%4)):
                    expected[base+4*field]=value
            if controls!=expected:raise ValueError('small lookup effective RTP values')
            words+=len(actual);queues+=1
    if index!=len(wanted) or not waits or not queues:raise ValueError('incomplete small lookup stream')
    return dict(queues=queues,completion_waits=waits,word_addresses_checked=words,all_addresses_controls_and_packets_exact=True)


def build(base,model,parent,out):
    parent=parent.absolute();out=out.absolute();frozen=sources();bundle=n.Bundle(base,model)
    old=json.loads((parent/'build.json').read_text())
    if old['sources']!=reference.lookup.sources() or old['base_identity']!=bundle.identity:raise ValueError('qualified lookup parent changed')
    if p.attention.sha(parent/'gelu_tables.cc')!=old['table_source_sha256']:raise ValueError('lookup table changed')
    out.mkdir(parents=True,exist_ok=False);clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-O3','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
        '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}',f'-I{p.ROOT/"kernels/laya_transformer"}',
        '-Wno-attributes','-DTRANSFORMER_NORM_TILED=1']
    split=sq.kernel_sources();texts=dict(norm=split['norm'],dispatch=split['dispatch'],lookup=reference.lookup_source(),
        tables=(parent/'gelu_tables.cc').read_text())
    objects=[];generated={};unchanged={}
    for name,text in texts.items():
        source=out/(name+'.cc');source.write_text(text,encoding='utf-8');generated[name]=p.attention.sha(source)
        obj=out/(name+'.o');p.attention.command([clang,*flags,'-c',source,'-o',obj],out);objects.append(obj)
    for name in ('block','gemm'):
        obj=out/(name+'.o');shutil.copyfile(bundle.folders['vector']/(name+'.o'),obj);objects.append(obj);unchanged[name]=p.attention.sha(obj)
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip():raise ValueError('unresolved small lookup helper')
    (out/'harness.mlir').write_text(prefix()+sq.runtime(bundle.plan,128),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',STACK):audits=json.loads(json.dumps(p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
    cdo={file.name:p.attention.sha(file) for file in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen:raise ValueError('missing initialization or changed small lookup source')
    proof=verify((out/'insts.bin').read_bytes(),events(bundle.plan),audits,bundle.plan['words'])
    meta=dict(sources=frozen,base_identity=bundle.identity,lookup_parent_sha256=p.attention.sha(parent/'build.json'),
        policy=dict(m=128,qwords=1040,kvwords=2064,q_depth=4,kv_depth=2,l2_q_depth=2,l2_kv_depth=2,state_words=2048,stack_bytes=STACK,control_address=15872),
        generated_sources=generated,unchanged_objects=unchanged,table_source_sha256=old['table_source_sha256'],audits=audits,cdo=cdo,transport=proof,
        instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),kernel_object_sha256=p.attention.sha(out/'kernel.o'),
        compiler_sha256=p.attention.sha(clang),scope='offline one-state smaller-query lookup MLP; lock/core/primitive/layer/full/speed gates pending')
    p.attention.dump(out/'build.json',meta)
    print('SMALL_LOOKUP_BUILD '+json.dumps(dict(policy=meta['policy'],code_bytes=sorted({a['code_bytes'] for a in audits}),
        conservative_stack_bound=max(sum(a['frames'])+256 for a in audits),transport=proof,build_sha256=p.attention.sha(out/'build.json'),
        sources=frozen,scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','parent','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.base,args.model,args.parent,args.out)
