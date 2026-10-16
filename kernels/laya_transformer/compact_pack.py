#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline DMA-gather norm packing with unchanged operand and FIFO packet ABIs."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import attention_fifo as af
ba=af.ba; n=ba.n; p=ba.p


def sources():
    return dict(af.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file)
      for file in (Path(__file__),Path(__file__).with_suffix('.cc'))})


def prefix():
    text=af.prefix(2,2,2)
    replacements={'@norm_pack_step(':'@norm_pack_compact_step(',
      '%c128 = arith.constant 128 : index':'%c128 = arith.constant 4 : index',
      '%c127 = arith.constant 127 : index':'%c127 = arith.constant 3 : index'}
    for old,new in replacements.items():
        if text.count(old)!=(17 if old.startswith('@') else 16): raise ValueError('compact pack prefix coverage')
        text=text.replace(old,new)
    return text


def pack(seq,pl,source,m):
    if m not in (128,256,384,512): raise ValueError('compact pack bucket')
    for group in range(4):
        seq.control(10,position=tuple(group*4+col for col in range(4)))
        for mg in range(m//128):
            for col in range(4):
                start=source+mg*128*p.OUT; feature=group*4+col
                seq.dma(col,'q',start,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                seq.dma(col,'out',pl['offsets']['packed_norm']+(feature*16+mg*4)*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                # Two distinct Shim BDs per unchanged KV object. The payload
                # gathers32rows*64features; sixteen unused words complete2064.
                # Q uses BD0 and output BD2; KV owns distinct IDs3 through10.
                for row in range(4):
                    first=start+row*32*p.OUT+feature*64
                    seq.dma(col,'kv',first,[1,1,32,64],[0,0,p.OUT,1],bid=3+row*2)
                    seq.dma(col,'kv',first,[1,1,1,16],[0,0,0,1],bid=4+row*2)
            seq.wait()


def runtime(plan,m):
    with patch.object(ba,'pack',pack): return ba.runtime(plan,m)


def build(base,model,out,m):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False)
    clang=p.attention.PEANO/'bin/clang++.exe'
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.attention.INSTALL/"include"}','-Wno-attributes']
    p.attention.command([clang,*flags,'-c',Path(__file__).with_suffix('.cc'),'-o',out/'compact_pack.o'],out)
    original=bundle.folders['chain']/'repeat/kernel.o'
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',original,out/'compact_pack.o','-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip(): raise ValueError('unresolved compact pack helper')
    (out/'harness.mlir').write_text(prefix()+runtime(bundle.plan,m),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=4864; audits=p.audit(out,{'fp_exp_ab':2048,'fp_exp_cd':2048})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3 or sources()!=frozen: raise ValueError('missing init or changed compact source')
    meta=dict(sources=frozen,base_identity=bundle.identity,policy=dict(m=m,q_depth=2,kv_depth=2,l2_q_depth=2,
      packing='DMA gather FP32[32,64] plus16unused words per KV object; four multicast packets per query'),
      audits=audits,cdo=cdo,instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
      kernel_object_sha256=p.attention.sha(out/'kernel.o'),original_kernel_object_sha256=p.attention.sha(original),
      scope='offline compact norm-pack candidate; partial-object Shim DMA ordering, numerical parity and speed need hardware qualification')
    p.attention.dump(out/'build.json',meta)
    print('COMPACT_PACK_BUILD '+json.dumps(dict(policy=meta['policy'],code_bytes=sorted({row['code_bytes'] for row in audits}),
      stack_bound=max(sum(row['frames'])+256 for row in audits),scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m)
