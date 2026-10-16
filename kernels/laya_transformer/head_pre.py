#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device type embedding, first head norm and biased QKV; no host activations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import vector as v
b=v.b; p=v.p


def sources():
    return dict(v.sources(),**{str(Path(__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(__file__))})


def plan(base):
    pl=copy.deepcopy(base); cursor=pl['words']; bank={}
    for key,size in [('head_qkv_weight',48*16*p.KVWORDS),('head_qkv_bias',48*p.KVWORDS),('head_affine',p.KVWORDS),('head_type',3*p.KVWORDS)]:
        pl['offsets'][key]=cursor; pl['sizes'][key]=size; bank[key]=[cursor,size]; cursor+=size+16
    pl['head_banks']=bank; pl['words']=cursor
    return pl


def parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); nodes=graph.graph.node; init={t.name:t for t in graph.graph.initializer}
    names=['m.type_emb.weight','m.head.layers.0.norm1.weight','m.head.layers.0.norm1.bias','val_3097','m.head.layers.0.self_attn.in_proj_bias']
    if nodes[1621].op_type!='Gather' or list(nodes[1621].input)!=[names[0],'qtype']: raise ValueError('head type graph changed')
    if nodes[1628].op_type!='LayerNormalization' or list(nodes[1628].input[1:])!=names[1:3]: raise ValueError('head norm graph changed')
    attrs={a.name:onnx.helper.get_attribute_value(a) for a in nodes[1628].attribute}
    if attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5): raise ValueError('head norm semantics')
    if nodes[1630].op_type!='MatMul' or nodes[1630].input[1]!=names[3] or nodes[1631].op_type!='Add' or nodes[1631].input[1]!=names[4]: raise ValueError('head QKV graph changed')
    arrays=[onnx.numpy_helper.to_array(init[name],base_dir=str(model.parent)).copy() for name in names]
    for value,shape in zip(arrays,[(3,1024),(1024,),(1024,),(1024,3072),(3072,)]):
        if value.shape!=shape or value.dtype!=np.float32: raise ValueError('head parameter geometry')
    hashes={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in zip(names,arrays)}
    arrays[3]=p.attention.rounded(arrays[3])
    return arrays,hashes


def initialize(data,pl,values,fill):
    table,gamma,beta,weight,bias=values; off=pl['offsets']; sizes=pl['sizes']
    packet=b.weight_packet(weight,fill)
    if packet.size!=sizes['head_qkv_weight']: raise ValueError('head weight packet geometry')
    data[off['head_qkv_weight']:off['head_qkv_weight']+packet.size]=packet
    affine=data[off['head_affine']:off['head_affine']+p.KVWORDS]; affine.fill(0)
    affine[:1024]=gamma.view(np.uint32); affine[1024:2048]=beta.view(np.uint32)
    packets=data[off['head_type']:off['head_type']+sizes['head_type']].reshape(3,p.KVWORDS)
    packets[:,:1024]=table.view(np.uint32)
    packets=data[off['head_qkv_bias']:off['head_qkv_bias']+sizes['head_qkv_bias']].reshape(48,p.KVWORDS)
    packets[:,:64]=bias.reshape(48,64).view(np.uint32)


def runtime(pl):
    off=pl['offsets']; seq=v.Sequence(pl['words']); size=16*p.OUT
    seq.control(17)
    for col in range(4):
        seq.dma(col,'q',off['mlp_norm']+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['norm_input']+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['head_type'],[32,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait(); seq.control(1)
    for col in range(4):
        seq.dma(col,'q',off['norm_input']+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['mlp_norm']+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['head_affine'],[32,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait(); b.r.pack_norm(seq,pl,off['mlp_norm'])
    seq.control(2,16)
    for group in range(12):
        for mg in range(4):
            for col in range(4):
                seq.dma(col,'q',off['packed_norm']+mg*4*p.OUT,[1,16,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                seq.dma(col,'out',off['projection']+((col*12+group)*4+mg)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off['head_qkv_weight']+(col*12+group)*16*p.KVWORDS,[1,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    seq.control(15)
    for first in range(0,48,4):
        for col in range(4):
            feature=first+col; start=off['projection']+(col*12+feature//4)*size
            seq.dma(col,'q',start,[1,4,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            seq.dma(col,'out',start,[1,1,size//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',off['head_qkv_bias']+feature*p.KVWORDS,[16,1,1,p.KVWORDS],[0,0,0,1])
        seq.wait()
    return seq.finish()


def prepare(raw,pl,qtype):
    if not isinstance(qtype,(int,np.integer)) or not 0<=qtype<3: raise ValueError('qtype range')
    result=bytearray(raw); start=pl['offsets']['head_type']*4; count=0
    for op in v.s.txn.parse(raw):
        if op['op']=='DDR_PATCH' and op['arg_idx']==0 and op['arg_offset']==start:
            struct.pack_into('<I',result,op['offset']+40,start+int(qtype)*p.KVWORDS*4); count+=1
    if count!=4: raise ValueError('head type read coverage')
    return bytes(result)


def verify(observed,pl,values,qtype,encoder_y):
    table,gamma,beta,weight,bias=values; off=pl['offsets']; sizes=pl['sizes']
    seed=observed[off['norm_input']:off['norm_input']+sizes['norm_input']].reshape(512,p.OUT)
    seed_x=seed[:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
    seed_r=seed[:,1040:2064].copy().view(np.float32)
    final=observed[off['mlp_norm']:off['mlp_norm']+sizes['mlp_norm']].reshape(512,p.OUT)
    y=final[:,:1024].copy().view(np.float32); residual=final[:,1040:2064].copy().view(np.float32)
    ref_r,ref_y=p.norm.oracle(encoder_y,np.broadcast_to(table[qtype],encoder_y.shape),gamma,beta)
    packed=b.unpack_operand(observed[off['packed_norm']:off['packed_norm']+sizes['packed_norm']].reshape(-1,p.OUT),16)
    actual=b.r.packet_matrix(observed[off['projection']:off['projection']+sizes['projection']].reshape(-1,p.OUT),3072)
    direct=((packed.astype(np.float64)@weight.astype(np.float64)).astype(np.float32).astype(np.float64)+bias.astype(np.float64)).astype(np.float32)
    return dict(guards=dict(ok=bool(np.all(seed[:,:1040].reshape(512,16,65)[:,:,64]==b.r.mlp.POISON) and np.all(seed[:,2064:]==b.r.mlp.POISON) and np.all(final[:,1024:1040]==b.r.mlp.POISON) and np.all(final[:,2064:]==b.r.mlp.POISON))),
      seed_x=dict(ok=bool(np.array_equal(seed_x.view(np.uint32),encoder_y.view(np.uint32)))),
      seed_type=dict(ok=bool(np.array_equal(seed_r, np.broadcast_to(table[qtype],seed_r.shape)))),
      head_norm=b.r.norm_verdict(residual,y,ref_r,ref_y),packed_norm=dict(ok=bool(np.array_equal(packed,p.attention.rounded(y)))),
      biased_qkv=b.numeric(actual,direct,.0002,.0002,.00001))


def build(chain,source,out):
    meta=json.loads((chain/'build.json').read_text()); vm=json.loads((source/'build.json').read_text())
    v.validate_compatibility(source,vm,meta['artifacts']['mlp'])
    pl=plan(meta['plan']); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(source/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(v.prefix()+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('head prelude core/init differs')
    result=dict(sources=sources(),plan=pl,vector_build_sha256=p.attention.sha(source/'build.json'),chain_build_sha256=p.attention.sha(chain/'build.json'),
      audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin'))
    p.attention.dump(out/'build.json',result)
    print('HEAD_PRE_BUILD '+json.dumps(dict(words=pl['words'],same_core_and_init=True,scope='head type/norm/biased QKV only')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--chain',type=Path,required=True); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.chain.resolve(),args.source.resolve(),args.out.resolve())
