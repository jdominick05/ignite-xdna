#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guarded normalization then GeGLU streams for the paired small-query ABI."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import window_lookup_primitive as gemm
import window_lock_check as locks
import window_core_check as control
p=gemm.p;n=gemm.n


def plan():
    sizes=dict(norm_query=4*2080,affine=2064,geglu_query=4*2080,value=4*2080,norm_output=4*2080,geglu_output=4*2080)
    offsets={};cursor=16
    for name,size in sizes.items():offsets[name]=cursor;cursor+=size+64
    return dict(words=cursor,offsets=offsets,sizes=sizes)


def events(window):
    off=plan()['offsets'];result=[]
    for mode,q,kv,out in [(1,'norm_query','affine','norm_output'),(3,'geglu_query','value','geglu_output')]:
        rows=[dict(kind='q',offset=off[q],dims=[2,4,65,16] if window else [1,1,4,2064],
                   strides=[1040,2080,16,1] if window else [0,0,2080,1],output=False),
              dict(kind='out',offset=off[out],dims=[1,1,520,16],strides=[0,0,16,1],output=True),
              dict(kind='kv',offset=off[kv],dims=[1,1,1 if mode==1 else 4,2064],strides=[0,0,2080,1],output=False)]
        result.extend(dict(col=0,mode=mode,**row) for row in rows)
    return result


def runtime(window):
    seq=gemm.candidate.window.bm.b.MlpSequence(plan()['words'])
    for mode in (1,3):
        seq.control(mode)
        if mode==1:
            width=seq.constant(1)
            for core in range(16):seq.lines.append(f'      "aiex.npu.rtp_write"({width}) {{buffer = @chain_control_{core}, index = 6 : ui32}} : (i32) -> ()')
        for row in events(window):
            if row['mode']==mode:seq.dma(row['col'],row['kind'],row['offset'],row['dims'],row['strides'],row['output'])
        seq.wait_columns([0])
    return seq.finish()


def verify_stream(raw,window,audits):
    descriptors={};operands={};controls={};index=0;writes=0;wanted=events(window)
    for op in n.c.txn.parse(raw):
        address=op.get('addr',0);reg=address&0xfffff;row=(address>>20)&31
        if op['op']=='WRITE' and row>=2:controls[address]=op['val'];writes+=1;continue
        if op['op']=='BLOCKWRITE':
            if address not in (0x1d000,0x1d040,0x1d060) or len(op['words'])!=8:raise ValueError('unexpected vector descriptor')
            descriptors[(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':operands[(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='WRITE' and reg in (0x1d214,0x1d204,0x1d21c):
            if index>=len(wanted):raise ValueError('extra vector queue')
            event=wanted[index];index+=1
            if event['kind']!={0x1d214:'q',0x1d204:'out',0x1d21c:'kv'}[reg]:raise ValueError('vector queue order')
            bid=op['val']&15;bd=descriptors[bid];arg,offset=operands[bid]
            actual=gemm.walk.descriptor_addresses(bd,((op['val']>>16)&255)+1,offset)
            if arg!=int(event['output']) or not np.array_equal(actual,gemm.walk.addresses(event)):raise ValueError('compiled vector addresses differ')
            if actual.min()<0 or actual.max()>=plan()['words']:raise ValueError('vector arena overrun')
            packet=(1040 if window else 2064) if event['kind']=='q' else 2064 if event['kind']=='kv' else 2080
            if bd[0]%packet or bd[7]&0x84000000:raise ValueError('partial vector FIFO object or changed TLAST/chaining')
            expected={}
            for audit in audits:
                col,r=audit['core'];core=col*4+r-2
                starts=[start for start,end,name in audit['spans'] if name==f'chain_control_{core}']
                if len(starts)!=1:raise ValueError('vector control allocation')
                base=(col<<25)|(r<<20)|starts[0]
                for field,value in ((0,event['mode']),(1,1),(2,gemm.candidate.window.bm.b.r.mlp.POISON),(3,core%4),(6,1)):
                    expected[base+field*4]=value
            if controls!=expected:raise ValueError('vector RTP state differs at queued transfer')
    if index!=6 or writes!=96:raise ValueError('incomplete vector queue/control stream')
    return dict(transfers=index,rtp_writes=writes,all_addresses_and_controls_exact=True,whole_fifo_objects=True,
        scope='norm then GeGLU one column/four cores; no numerical device claim')


def fixture(seed):
    rng=np.random.default_rng(seed);pl=plan();data=np.full(pl['words'],0x6badbeef,np.uint32);off=pl['offsets']
    q=np.full((4,2080),gemm.candidate.window.bm.b.r.mlp.POISON,np.uint32)
    x=rng.normal(0,0.7,(4,1024)).astype(np.float32);residual=rng.normal(0,0.3,(4,1024)).astype(np.float32)
    q[:,:1040].reshape(4,16,65)[:,:,:64]=x.view(np.uint32).reshape(4,16,64)
    q[:,1040:2064]=residual.view(np.uint32)
    data[off['norm_query']:off['norm_query']+q.size]=q.reshape(-1)
    affine=np.full(2064,0x7fc12345,np.uint32)
    affine[:1024]=rng.normal(1,0.1,1024).astype(np.float32).view(np.uint32)
    affine[1024:2048]=rng.normal(0,0.1,1024).astype(np.float32).view(np.uint32)
    data[off['affine']:off['affine']+2064]=affine
    gates=rng.normal(0,3,(4,32,64)).astype(np.float32)
    gates[:,:,0]=np.linspace(-17,17,32,dtype=np.float32)
    values=rng.normal(0,1,(4,32,64)).astype(np.float32)
    for name,payload in [('geglu_query',gates),('value',values)]:
        packets=np.full((4,2080),0x7fc12345,np.uint32);packets[:,16:2064]=payload.reshape(4,2048).view(np.uint32)
        data[off[name]:off[name]+packets.size]=packets.reshape(-1)
    return data


def sources():
    files=['kernels/laya_transformer/window_vector_primitive.py','kernels/laya_transformer/window_lock_check.py',
           'kernels/laya_transformer/window_core_check.py']
    return dict(gemm.sources(),**{file:p.attention.sha(Path(file)) for file in files})


def build(parent,out):
    out=out.absolute();frozen=sources();parent_meta=json.loads((parent/'build.json').read_text())
    if parent_meta['sources']!=gemm.sources():raise ValueError('stale qualified GEMM primitive source')
    out.mkdir(parents=True,exist_ok=False);layouts={}
    for name,old,window in [('original','original0',False),('window','window',True)]:
        folder=out/name;folder.mkdir();shutil.copyfile(parent/old/'kernel.o',folder/'kernel.o')
        prefix=gemm.candidate.prefix() if window else gemm.candidate.lookup.gf.prefix()
        (folder/'harness.mlir').write_text(prefix+runtime(window),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
            '--dump-intermediates',f'--tmpdir={folder/"prj"}'],folder)
        with patch.object(p,'STACK',gemm.candidate.window.STACK if window else 5376):
            audits=json.loads(json.dumps(p.audit(folder,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
        cdo={file.name:p.attention.sha(file) for file in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=parent_meta['layouts'][old]['audits'] or cdo!=parent_meta['layouts'][old]['cdo']:
            raise ValueError('vector primitive changed qualified parent core/init')
        transport=verify_stream((folder/'insts.bin').read_bytes(),window,audits)
        ledger=control.verify_vector((folder/'prj/input_with_addresses.mlir').read_text(),window)
        layouts[name]=dict(audits=audits,cdo=cdo,transport=transport,ledger=ledger,
            instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('WINDOW_VECTOR_LAYOUT '+json.dumps(dict(name=name,transport=transport,ledger=ledger,parent_core_init_identical=True)),flush=True)
    relation=locks.compare(out/'original',out/'window')
    if frozen!=sources():raise ValueError('vector primitive sources changed')
    p.attention.dump(out/'build.json',dict(sources=frozen,plan=plan(),layouts=layouts,lock_relation=relation,
        parent_sha256=p.attention.sha(parent/'build.json'),scope='offline paired-query vector primitive; hardware exactness pending'))
    print('WINDOW_VECTOR_BUILD '+json.dumps(dict(lock_relation=relation,scope='offline only; hardware pending')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('parent','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.parent,args.out)
