#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unrotated head attention, consuming resident biased QKV from head_pre."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import chain as c
import head_pre as h
b=c.b; p=b.p; r=b.r


def sources():
    return dict(c.sources(),**h.sources(),**{str(Path(__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(__file__))})


def plan(base):
    pl=copy.deepcopy(base); pl['offsets']['head_identity']=pl['words']; pl['sizes']['head_identity']=16*p.KVWORDS
    pl['words']+=16*p.KVWORDS+16
    return pl


def initialize(data,pl):
    off=pl['offsets']['head_identity']; packets=data[off:off+16*p.KVWORDS].reshape(16,p.KVWORDS)
    packets[:,:1024]=np.float32(1).view(np.uint32); packets[:,1024:2048]=0


def runtime(pl,valid=512):
    if type(valid) is not int or not 1<=valid<=512: raise ValueError('head attention requires a nonempty mask')
    off=pl['offsets']; seq=r.Sequence(pl['words']); size=16*p.OUT
    for mode in (4,5,6,7):
        for which in range(4):
            for mg in range(4):
                seq.control(mode,position=mg*128,valid=valid,local=0)
                for col in range(4):
                    head=col*4+which; feature=head+(16 if mode==6 else 32 if mode==7 else 0)
                    proj=off['projection']+((feature%4)*12+feature//4)*size+mg*4*p.OUT
                    if mode==7: qoffset=off['keys']+(head*16+mg*4)*p.OUT; voffset=proj
                    else: qoffset=proj; voffset=off['head_identity']+mg*4*p.KVWORDS
                    seq.dma(col,'q',qoffset,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                    if mode<6:
                        target=off['queries']+(head*32+mg*8+(mode-4))*p.OUT
                        seq.dma(col,'out',target,[1,1,4,p.OUT],[0,0,2*p.OUT,1],True)
                    else:
                        target=off['keys' if mode==6 else 'kv']+(head*16+mg*4)*p.OUT
                        seq.dma(col,'out',target,[1,1,p.OUT//4,16],[0,0,16,1],True)
                    seq.dma(col,'kv',voffset,[1,1,4,p.KVWORDS],[0,0,p.OUT if mode==7 else p.KVWORDS,1])
                seq.wait()
    seq.control(0)
    for col in range(4):
        seq.dma(col,'q',off['queries']+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['attention']+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        for head in range(4):
            seq.dma(col,'kv',off['kv']+(col*4+head)*16*p.OUT+16,[8,1,16,p.KVWORDS],[0,0,p.OUT,1],bid=head+3)
    seq.wait(); seq.control(8)
    for which in range(4):
        for mg in range(4):
            for col in range(4):
                head=col*4+which; source=off['attention']+(head*32+mg*8)*p.OUT
                seq.dma(col,'q',source,[1,1,4,p.QWORDS],[0,0,2*p.OUT,1])
                target=off['packed_attention']+(head*16+mg*4)*p.OUT
                seq.dma(col,'out',target,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',source+p.OUT,[1,1,4,p.KVWORDS],[0,0,2*p.OUT,1])
            seq.wait()
    return seq.finish()


def verify(observed,pl,valid):
    if not 1<=valid<=512: raise ValueError('head mask semantics for empty input unqualified')
    off=pl['offsets']; sizes=pl['sizes']
    regions={name:observed[off[name]:off[name]+sizes[name]].reshape(-1,p.OUT) for name in ('projection','queries','keys','kv','attention','packed_attention')}
    projected=r.packet_matrix(regions['projection'],3072)
    expected=p.attention.rounded(projected.reshape(512,3,16,64).transpose(1,2,0,3))
    q=r.unpack_query(regions['queries']); k,value=r.unpack_kv(regions['kv'])
    result={name:dict(ok=bool(np.array_equal(a.view(np.uint32),z.view(np.uint32)))) for name,a,z in zip(('unrotated_q','unrotated_k','unrotated_v'),(q,k,value),expected)}
    # The exported head uses -inf; with at least one valid key this selects
    # the same global key set as the existing attention oracle. No empty-mask
    # equivalence is claimed. Device receives only valid/local RTP metadata.
    mask=np.zeros((512,512),np.int32); mask[:,:valid]=1
    reference=p.attention.oracle(*expected,mask)
    actual=regions['attention'][:,16:1040].copy().view(np.float32).reshape(16,512,64)
    result['head_attention']=p.attention.numeric(actual,reference)
    packed=r.unpack_activation(regions['packed_attention'])
    result['head_attention_pack']=dict(ok=bool(np.array_equal(packed,p.attention.rounded(actual.transpose(1,0,2).reshape(512,1024)))))
    guards=True
    for name,used in [('queries',512),('keys',1024),('kv',2064),('attention',1024),('packed_attention',1024)]:
        rows=regions[name]
        guards &= bool(np.all(rows[:,16+used:]==r.mlp.POISON))
        if name!='queries': guards &= bool(np.all(rows[:,:16]==r.mlp.POISON))
    headers=regions['queries'][:,:16]; expected_header=np.full((512,16),r.mlp.POISON,np.uint32)
    expected_header[:,0]=np.tile(np.arange(0,512,16),16); expected_header[:,1]=valid
    expected_header[:,2]=r.mlp.POISON; expected_header[:,3:5]=0
    result['head_attention_guards']=dict(ok=guards and bool(np.array_equal(headers,expected_header)))
    result['head_key_ramp']=dict(ok=bool(np.all(regions['kv'][:,2064:].copy().view(np.uint16)==np.arange(32,dtype=np.uint16))))
    return result


def build(chain,prelude,out):
    cm=json.loads((chain/'build.json').read_text()); hm=json.loads((prelude/'build.json').read_text())
    if cm['sources']!=c.sources() or hm['sources']!=h.sources() or hm['chain_build_sha256']!=p.attention.sha(chain/'build.json'): raise ValueError('stale head/encoder build')
    pl=plan(hm['plan']); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(chain/'repeat/kernel.o',out/'kernel.o')
    prefix=r.prefix().replace('stack_size = 4608 : i32','stack_size = 4864 : i32')
    (out/'harness.mlir').write_text(prefix+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=4864; audits=p.audit(out,{'fp_exp_ab':2048,'fp_exp_cd':2048})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    original=cm['artifacts']['first']
    if json.loads(json.dumps(audits))!=original['audits'] or cdo!=original['cdo']: raise ValueError('head attention changes resident core/init')
    p.attention.dump(out/'build.json',dict(sources=sources(),plan=pl,head_pre_build_sha256=p.attention.sha(prelude/'build.json'),
      chain_build_sha256=p.attention.sha(chain/'build.json'),audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('HEAD_ATTENTION_BUILD '+json.dumps(dict(same_core_and_init=True,code_bytes=sorted({a['code_bytes'] for a in audits}),scope='compile/audit only; head attention needs device qualification')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--chain',type=Path,required=True); ap.add_argument('--head-pre',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.chain.resolve(),args.head_pre.resolve(),args.out.resolve())
