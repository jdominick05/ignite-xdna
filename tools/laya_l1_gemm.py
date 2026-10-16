#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Laya's actual GEMM objects in a bounded, DMA-free repeated L1 call loop."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import l1_tile_bench as base
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'kernels/laya_transformer'))
import transport_walk as walk
import window_lock_check as locks

GEOMETRY=base.geometry('bf16',SimpleNamespace(m=32,k=64,n=64,mode='direct'))
WRAPPER='''#include <stdint.h>
extern "C" void transformer_matmul(const void *,const void *,void *,int,int,int);
extern "C" void laya_tile(const int32_t *a,const int32_t *b,int32_t *c) {
  transformer_matmul(a,b,c,0,8,16);
}
'''


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def emit(tag,row):print(tag+' '+json.dumps(row),flush=True)


def expected_bits(variant,count,product):
    # Zero calls returns the core's explicit +0 initialization. Multiplying a
    # negative reference product by zero would instead manufacture -0 bits.
    values=np.zeros_like(product) if variant=='empty' or count==0 else count*product
    return values.astype(np.float32).view(np.uint32)


def sources():
    files=(Path(__file__),Path(base.__file__),Path(walk.__file__),Path(locks.__file__),
        ROOT/'src/ignite_xdna/runtime/driver.py',ROOT/'kernels/laya_transformer/gemm.cc',
        ROOT/'kernels/l1_tile_bench/empty_call.s')
    return {str(file.absolute().relative_to(ROOT)).replace('\\','/'):sha(file) for file in files}


def stream_proof(raw):
    descriptors={};patches={};queues=[];waits=0
    expected={0:(0,GEOMETRY['a_w']),1:(1,GEOMETRY['bh_w']),2:(2,GEOMETRY['c_w'])}
    for op in walk.n.c.txn.parse(raw):
        address=op.get('addr',0);reg=address&0xfffff
        if op['op']=='BLOCKWRITE':
            if address>>20 or not 0x1d000<=reg<0x1d200 or len(op['words'])!=8:raise ValueError('unexpected L1 Shim BD')
            descriptors[(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':patches[(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='WRITE' and 0x1d204<=reg<=0x1d21c:
            bid=op['val']&15
            if bid not in expected:raise ValueError('unexpected L1 descriptor ID')
            arg,count=expected[bid];actual_arg,offset=patches[bid];bd=descriptors[bid]
            addresses=walk.descriptor_addresses(bd,((op['val']>>16)&255)+1,offset)
            if actual_arg!=arg or not np.array_equal(addresses,np.arange(count)) or bd[0]!=count or bd[7]&0x84000000:
                raise ValueError('L1 compiled address/packet mismatch')
            queues.append(bid)
        elif op['op']=='TCT':
            # One completion token for column0 Shim S2MM channel0, one row/column.
            import struct
            words=struct.unpack_from('<4I',raw,op['offset'])
            if words[2]!=0 or words[3]!=0x10100:raise ValueError('L1 completion token differs')
            waits+=1
    if sorted(queues)!=[0,1,2] or waits!=1:raise ValueError('incomplete L1 queues/wait')
    return dict(queues=queues,completion_waits=waits,all_word_addresses_exact=True,complete_fifo_objects=True)


def core_dma_proof(folder):
    text=locks.clean((folder/'prj/input_with_addresses.mlir').read_text())
    tiles={name:(int(c),int(r)) for name,c,r in re.findall(r'%(\w+) = aie.tile\((\d+), (\d+)\)',text)}
    declared={name:(*tiles[tile],int(index),int(initial)) for name,tile,index,initial in re.findall(
        r'%(\w+) = aie.lock\(%(\w+), (\d+)\) \{init = (-?\d+) : i32',text)}
    state={}
    for address,mask,value in locks.packing.cdo_writes((folder/'prj/cdo_main/main_aie_cdo_init.bin').read_bytes()):
        state[address]=(state.get(address,0)&(~mask&0xffffffff))|(value&mask)
    hardware=0
    for name,(col,row,index,initial) in declared.items():
        if row!=2:continue
        if initial not in (0,1) or state.get((col<<25)|(row<<20)|0x1f000|16*index)!=initial:
            raise ValueError('L1 lock initialization mismatch')
        hardware+=1
    constants={name:int(value) for name,value in re.findall(r'(%[\w-]+) = arith.constant (-?\d+) : i32',text)}
    owners={name:tiles[tile] for name,tile in re.findall(r'%(\w+) = aie.buffer\(%(\w+)\)',text)}
    sizes={'inA_cons_buff_0':GEOMETRY['a_w'],'inB_cons_buff_0':GEOMETRY['bh_w'],'outC_buff_0':GEOMETRY['c_w']}
    checked=[]
    for block in re.split(r'(?m)^\s*\^bb\d+',text):
        if 'aie.dma_bd(' not in block:continue
        bds=re.findall(r'aie.dma_bd\(%(\w+) : memref<(\d+)xi32> offset = (\d+) len = (\d+)\)',block)
        uses=re.findall(r'aie.use_lock\(%(\w+), (AcquireGreaterEqual|Release), (%[\w-]+)\)',block)
        ids=re.findall(r'\{bd_id = (\d+) : i32',block)
        if len(bds)!=1 or len(uses)!=2 or len(ids)!=1:raise ValueError('L1 DMA handshake shape')
        name,size,offset,length=bds[0]
        if name not in sizes or int(size)!=sizes[name] or int(offset)!=0 or int(length)!=sizes[name]:
            raise ValueError('L1 DMA packet geometry')
        if [u[1] for u in uses]!=['AcquireGreaterEqual','Release'] or any(constants[u[2]]!=1 for u in uses):
            raise ValueError('L1 DMA lock ordering/value')
        locks.verify_bd_lock_word(state,owners[name],int(ids[0]),declared[uses[0][0]],declared[uses[1][0]])
        checked.append(name)
    if hardware!=6 or sorted(checked)!=sorted(sizes):raise ValueError('incomplete L1 core DMA/locks')
    return dict(hardware_lock_initializations=hardware,compiled_dma_lock_fields=len(checked),complete_objects=sorted(checked))


def audit(folder):
    place=base.linker_placement(folder/'prj')
    sizes={'inA_cons_buff_0':GEOMETRY['a_w']*4,'inB_cons_buff_0':GEOMETRY['bh_w']*4,'outC_buff_0':GEOMETRY['c_w']*4}
    spans=sorted((place[name],place[name]+size,name) for name,size in sizes.items())
    if any(start<0 or end>65536 for start,end,name in spans) or any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError('L1 operand overlap/bounds')
    elf=folder/'prj/main_core_0_2.elf'
    elfs=list((folder/'prj').rglob('*.elf'))
    if len(elfs)!=1:raise ValueError('L1 must have exactly one core')
    elf=elfs[0]
    sizes_text=base.run_cmd([base.PEANO/'bin/llvm-size.exe','-A',elf],folder)
    code=sum(int(x) for x in re.findall(r'^\.text\S*\s+(\d+)\s+',sizes_text,re.M))
    report=base.run_cmd([base.PEANO/'bin/llvm-readobj.exe','--stack-sizes',elf],folder)
    frames=[int(x,16) for x in re.findall(r'Size: (0x[0-9a-fA-F]+)',report)]
    ld=(folder/'prj/ldScripts_main_core_0_2.ld.script').read_text()
    reserved=int(re.search(r'\. \+= (0x[0-9a-fA-F]+); /\* stack \*/',ld)[1],16)
    symbols=base.run_cmd([base.PEANO/'bin/llvm-nm.exe',elf],folder)
    if code>16384 or not frames or sum(frames)+256>reserved or walk.p.attention.software_float_helpers(symbols):
        raise ValueError('L1 program/stack/helper bounds')
    return dict(spans=spans,code_bytes=code,conservative_stack_bound=sum(frames)+256,reserved_stack_bytes=reserved,
        runtime=stream_proof((folder/'insts.bin').read_bytes()),core_dma=core_dma_proof(folder))


def build(out):
    out=out.absolute();out.mkdir(parents=True,exist_ok=False);frozen=sources()
    parent=ROOT/'scratch/laya_transformer/geglu_lut_v1_double_m128'
    specialized=ROOT/'scratch/laya_transformer/gemm_specialize_v1'
    old=json.loads((parent/'build.json').read_text());screen=json.loads((specialized/'screen.json').read_text())
    fixed=next(row for row in screen['objects'] if row['variant']=='fixed_reset')
    objects={'generic':parent/'gemm.o','specialized':specialized/'fixed_reset.o'}
    if sha(objects['generic'])!=old['unchanged_objects']['gemm'] or sha(objects['specialized'])!=fixed['object_sha256']:
        raise ValueError('Laya object identity mismatch')
    clang=base.PEANO/'bin/clang++.exe';wrapper=out/'wrapper.cc';wrapper.write_text(WRAPPER,encoding='utf-8')
    base.run_cmd([clang,'-Oz','-std=c++20','--target=aie2-none-unknown-elf','-fstack-size-section','-c',wrapper,'-o',out/'wrapper.o'],out)
    empty=out/'empty.s';empty.write_text((ROOT/'kernels/l1_tile_bench/empty_call.s').read_text().replace('empty_call','transformer_matmul'))
    base.run_cmd([clang,'--target=aie2-none-unknown-elf','-c',empty,'-o',out/'empty.o'],out);objects['empty']=out/'empty.o'
    rows={}
    for name in ('empty','generic','specialized'):
        folder=out/name;folder.mkdir()
        base.run_cmd([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',out/'wrapper.o',objects[name],'-o',folder/'kernel.o'],folder)
        if base.run_cmd([base.PEANO/'bin/llvm-nm.exe','-u',folder/'kernel.o'],folder).strip():raise ValueError('unresolved L1 helper')
        loop=base.CALL_LOOP['i32'].format(fn='laya_tile',**GEOMETRY)
        (folder/'harness.mlir').write_text(base.HARNESS_DIRECT.format(fn='laya_tile',obj='kernel.o',call_loop=loop,**GEOMETRY),encoding='utf-8')
        base.run_cmd([base.AIECC,folder/'harness.mlir',f'--peano={base.PEANO}','--get-npu-insts','--npu-insts-name=insts.bin',
            '--get-xclbin','--xclbin-name=final.xclbin','--dump-intermediates',f'--tmpdir={folder/"prj"}'],folder)
        proof=audit(folder)
        rows[name]=dict(proof=proof,object_sha256=sha(objects[name]),
            artifacts={file:sha(folder/file) for file in ('kernel.o','insts.bin','final.xclbin')},
            cdo={f.name:sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')})
        emit('LAYA_L1_BUILD',dict(variant=name,proof=proof))
    for key in ('spans','runtime','core_dma'):
        if any(row['proof'][key]!=rows['empty']['proof'][key] for row in rows.values()):raise ValueError('L1 layout/transport differs across objects')
    if len({row['artifacts']['insts.bin'] for row in rows.values()})!=1:raise ValueError('L1 transactions differ')
    if frozen!=sources():raise ValueError('L1 source changed during build')
    result=dict(sources=frozen,variants=rows,geometry=GEOMETRY,shape=[32,64,64],reset=0,
        compiler_sha256=sha(clang),scope='L1 resident accumulated GEMM only; no complete-model or streamed-memory speed claim')
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    emit('LAYA_L1_BUILD_RESULT',dict(build_sha256=sha(out/'build.json'),sources=frozen,scope=result['scope']))


def validate(folder):
    meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=sources() or meta['geometry']!=GEOMETRY:raise ValueError('stale L1 image')
    for name,row in meta['variants'].items():
        for file,digest in row['artifacts'].items():
            if sha(folder/name/file)!=digest:raise ValueError('changed L1 artifact')
        if json.loads(json.dumps(audit(folder/name)))!=row['proof']:raise ValueError('changed L1 offline proof')
    return meta


def run(folder,qualify_only):
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    import ignite_xdna
    if not Path(ignite_xdna.__file__).is_relative_to(ROOT/'src'):raise ValueError('engine import is outside worktree')
    folder=folder.absolute();meta=validate(folder)
    if base.partitions()!='idle':raise ValueError('device occupied')
    calls=[0,1,3] if qualify_only else [0,32768,65536,131072]
    reps=1 if qualify_only else 3
    # Every accumulated integer, including intermediate partial sums, stays below 2^24.
    rng=np.random.default_rng(20261016);a=rng.integers(-1,2,(32,64)).astype(np.float32);b=rng.integers(-1,2,(64,64)).astype(np.float32)
    pa,pb,_=base.pack('bf16',a,b,np.zeros((32,64),np.float32),'tiled')
    product=base.tile_rows(a.astype(np.float64)@b.astype(np.float64),4,4)
    order=['empty','generic','specialized'] if qualify_only else ['empty','generic','specialized','specialized','generic','empty']
    rows=[]
    emit('LAYA_L1_START',dict(build_sha256=sha(folder/'build.json'),calls=calls,reps=reps,order=order,
        reset=0,scope=meta['scope'],all_timed_outputs_checked=True,clock_conversion_used=False))
    for index,name in enumerate(order):
        if base.partitions()!='idle':raise ValueError('device occupied before L1 profile')
        h=XrtSiliconHarness();bo_instr=abo=bbo=cbo=handle=None;poisoned=False;observations=[]
        try:
            h.load_xclbin(str(folder/name/'final.xclbin'));bo_instr,n=h.create_instruction_bo(str(folder/name/'insts.bin'))
            abo=h.create_host_bo(len(pa)+64,3);bbo=h.create_host_bo(len(pb)+64+64,4);cbo=h.create_host_bo(GEOMETRY['c_w']*4+64,5)
            to=h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE;fr=h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            for count in calls:
                header=np.zeros(16,np.int32);header[0]=count
                data_a=pa+bytes([0x35])*64;data_b=pb+header.tobytes()+bytes([0x57])*64
                for rep in range(reps):
                    abo.write(data_a,0);bbo.write(data_b,0);cbo.write(bytes([0xab])*(GEOMETRY['c_w']*4+64),0)
                    abo.sync(to);bbo.sync(to);cbo.sync(to)
                    poisoned=True;t0=time.perf_counter();handle=h.kernel(3,bo_instr,n,abo,bbo,cbo)
                    state=handle.wait(5000);elapsed=(time.perf_counter()-t0)*1000
                    if state!=h.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:raise RuntimeError('L1 timeout; no retry')
                    poisoned=False;handle=None
                    abo.sync(fr);bbo.sync(fr);cbo.sync(fr)
                    output=bytes(cbo.read(GEOMETRY['c_w']*4+64,0));got=np.frombuffer(output[:GEOMETRY['c_w']*4],np.uint32)
                    expected=expected_bits(name,count,product)
                    checks=dict(output_bits_exact=bool(np.array_equal(got,expected)),
                        input_a_unchanged=bytes(abo.read(len(data_a),0))==data_a,input_b_unchanged=bytes(bbo.read(len(data_b),0))==data_b,
                        output_tail_unchanged=output[-64:]==bytes([0xab])*64)
                    if not all(checks.values()):
                        mismatch=np.flatnonzero(got!=expected)
                        capture=folder/f'failure_{index}_{count}_{rep}.npz'
                        with capture.open('xb') as file:np.savez(file,got=got,expected=expected,a=a,b=b,calls=count)
                        emit('LAYA_L1_FAILURE',dict(variant=name,calls=count,checks=checks,mismatch_words=len(mismatch),
                            first_indices=mismatch[:8].tolist(),actual_words=got[mismatch[:8]].tolist(),
                            expected_words=expected[mismatch[:8]].tolist(),capture_sha256=sha(capture)))
                        raise ValueError('L1 numerical/guard failure')
                    observations.append(dict(calls=count,ms=elapsed))
            emit('LAYA_L1_PROFILE',dict(index=index,variant=name,samples=observations,all_output_input_guard_checks_pass=True))
        finally:
            if poisoned:globals()['_quarantined']=(h,bo_instr,abo,bbo,cbo,handle)
            else:handle=bo_instr=abo=bbo=cbo=None;gc.collect();h.close();h=None;gc.collect()
        if base.partitions()!='idle':raise ValueError('L1 close did not release context')
        rows.append(dict(variant=name,samples=observations,idle_after_close=True))
    if validate(folder)!=meta:raise ValueError('L1 identity changed during run')
    fits={}
    if not qualify_only:
        for name in ('empty','generic','specialized'):
            samples=[sample for row in rows if row['variant']==name for sample in row['samples']]
            intercept,slope,r2=base.fit([s['calls'] for s in samples],[s['ms']*1e-3 for s in samples])
            if slope<=0:raise ValueError('nonpositive L1 slope')
            fits[name]=dict(intercept_ms=intercept*1000,slope_ns_per_call=slope*1e9,r2=r2)
        for name in ('generic','specialized'):
            fits[name]['over_empty_ns_per_call']=fits[name]['slope_ns_per_call']-fits['empty']['slope_ns_per_call']
    emit('LAYA_L1_RESULT',dict(fits=fits,profiles=len(rows),all_outputs_inputs_guards_exact=True,idle_after_close=True,
        registered_files_unchanged=True,scope=meta['scope']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('build','qualify','run'));ap.add_argument('--folder',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='build':build(args.folder)
    else:run(args.folder,args.command=='qualify')
