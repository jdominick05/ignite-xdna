#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline one-column/two-output GEMM primitive for the lookup window candidate."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import window_lookup as candidate
import transport_walk as walk
p=candidate.p;n=candidate.n
K=128


def plan():
    qwords=2*4*2064;weight_words=4*2*2*2064
    output=16+qwords+weight_words+64
    return dict(words=output+8*p.OUT+64,input=16,input_words=qwords+weight_words,
        weights=16+qwords,output=output,kblocks=2,slots=2)


def events(window,slot=0):
    layout=plan();width=2 if window else 1
    return [dict(col=0,kind='q',offset=16,dims=[1,2,4,1040 if window else 2064],
        strides=[0,4*2064,2064,1],output=False),
        dict(col=0,kind='out',offset=layout['output']+(0 if window else slot*4*p.OUT),
        dims=[width,4,p.OUT//16,16],strides=[4*p.OUT,p.OUT,16,1],output=True),
        dict(col=0,kind='kv',offset=layout['weights']+(0 if window else slot*2*2064),
        dims=[2,2,129,16] if window else [1,1,2,2064],
        strides=[2064,2*2064,16,1] if window else [0,0,2064,1],output=False)]


def runtime(window,slot=0):
    seq=candidate.window.bm.b.MlpSequence(plan()['words']);seq.control(2,2)
    width=seq.constant(2 if window else 1)
    for core in range(16):
        seq.lines.append(f'      "aiex.npu.rtp_write"({width}) {{buffer = @chain_control_{core}, index = 6 : ui32}} : (i32) -> ()')
    for row in events(window,slot):
        seq.dma(row['col'],row['kind'],row['offset'],row['dims'],row['strides'],row['output'])
    seq.wait_columns([0]);return seq.finish()


def verify_stream(raw,window,slot,audits):
    wanted=events(window,slot);descriptors={};operands={};index=0;controls={}
    for op in n.c.txn.parse(raw):
        address=op.get('addr',0);reg=address&0xfffff;row=(address>>20)&31
        if op['op']=='WRITE' and row>=2:
            controls[address]=op['val'];continue
        if op['op']=='BLOCKWRITE':
            if address not in (0x1d000,0x1d040,0x1d060) or len(op['words'])!=8:
                raise ValueError('unexpected primitive descriptor')
            descriptors[(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':operands[(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='WRITE' and reg in (0x1d214,0x1d204,0x1d21c):
            if index>=len(wanted):raise ValueError('extra primitive queue')
            event=wanted[index];index+=1
            if event['kind']!={0x1d214:'q',0x1d204:'out',0x1d21c:'kv'}[reg]:raise ValueError('primitive queue order')
            bid=op['val']&15;descriptor=descriptors[bid];arg,offset=operands[bid]
            actual=walk.descriptor_addresses(descriptor,((op['val']>>16)&255)+1,offset)
            if arg!=int(event['output']) or not np.array_equal(actual,walk.addresses(event)):
                raise ValueError('compiled primitive addresses differ')
            if actual.min()<0 or actual.max()>=plan()['words']:raise ValueError('primitive exceeds arena')
            packet={'q':1040 if window else 2064,'kv':2064,'out':2080}[event['kind']]
            if descriptor[0]%packet or descriptor[7]&0x84000000:
                raise ValueError('partial FIFO object or changed TLAST/chaining')
    expected_controls={}
    for audit in audits:
        col,row=audit['core'];core=col*4+row-2
        spans=[start for start,end,name in audit['spans'] if name==f'chain_control_{core}']
        if len(spans)!=1:raise ValueError('primitive control allocation')
        address=(col<<25)|(row<<20)|spans[0]
        for field,value in ((0,2),(1,2),(2,candidate.window.bm.b.r.mlp.POISON),(3,core%4),(6,2 if window else 1)):
            expected_controls[address+field*4]=value
    if index!=3 or len(controls)!=80 or controls!=expected_controls:
        raise ValueError('incomplete/incorrect mode, K count, row, poison or window width controls')
    return dict(transfers=index,rtp_writes=len(controls),all_compiled_addresses_exact=True,
        complete_query_and_kv_objects=True,output_packets=8 if window else 4,
        scope='address/packet checks; cross-image lock relation and device behavior still pending')


def fixture(seed):
    rng=np.random.default_rng(seed)
    a=rng.integers(-2,3,(128,128)).astype(np.float32)
    b=rng.integers(-2,3,(128,512)).astype(np.float32)
    packed=p.gemm_packet(a,b,candidate.window.bm.b.r.mlp.POISON)
    if packed.size!=plan()['input_words']:raise ValueError('primitive input geometry')
    golden=np.stack([(a@b[:,first:first+64]).reshape(4,2048) for first in (0,256)])
    return packed,golden


def sources():
    return dict(candidate.sources(),**{'kernels/laya_transformer/window_lookup_primitive.py':p.attention.sha(Path(__file__)),
        'kernels/laya_transformer/transport_walk.py':p.attention.sha(Path(walk.__file__))})


def build(original,window,out):
    out=out.absolute();frozen=sources();out.mkdir(parents=True,exist_ok=False);layouts={}
    for name,parent,module,windowed,slot in [('original0',original,candidate.lookup,False,0),
        ('original1',original,candidate.lookup,False,1),('window',window,candidate,True,0)]:
        meta=json.loads((parent/'build.json').read_text())
        if meta['sources']!=module.sources() or p.attention.sha(parent/'kernel.o')!=meta['kernel_object_sha256']:
            raise ValueError('primitive parent source/object changed')
        folder=out/name;folder.mkdir();shutil.copyfile(parent/'kernel.o',folder/'kernel.o')
        prefix=candidate.prefix() if windowed else candidate.lookup.gf.prefix()
        (folder/'harness.mlir').write_text(prefix+runtime(windowed,slot),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
            '--dump-intermediates',f'--tmpdir={folder/"prj"}'],folder)
        with patch.object(p,'STACK',candidate.window.STACK if windowed else 5376):
            audits=json.loads(json.dumps(p.audit(folder,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
        cdo={file.name:p.attention.sha(file) for file in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=meta['audits'] or cdo!=meta['cdo']:raise ValueError('primitive changes parent core/init')
        transport=verify_stream((folder/'insts.bin').read_bytes(),windowed,slot,audits)
        layouts[name]=dict(audits=audits,cdo=cdo,transport=transport,instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('WINDOW_PRIMITIVE_LAYOUT '+json.dumps(dict(name=name,transport=transport,parent_core_and_init_identical=True)),flush=True)
    if frozen!=sources():raise ValueError('primitive source changed')
    p.attention.dump(out/'build.json',dict(sources=frozen,plan=plan(),layouts=layouts,
        parents=dict(original=p.attention.sha(original/'build.json'),window=p.attention.sha(window/'build.json')),
        scope='offline primitive; cross-image lock proof, paired-query vector primitives and all hardware gates pending'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('original','window','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.original,args.window,args.out)
