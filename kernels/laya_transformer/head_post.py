#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""First head output projection/ReLU MLP into the second head's norm."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import head_attention as a
h=a.h; v=h.v; b=a.b; p=a.p


def sources():
    return dict(a.sources(),**{str(Path(__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(__file__))})


def plan(base):
    pl=copy.deepcopy(base); cursor=pl['words']
    for key,size in [('head_out_weight',16*16*p.KVWORDS),('head_out_bias',16*p.KVWORDS),
      ('head_norm2',p.KVWORDS),('head_wi_weight',64*16*p.KVWORDS),('head_wi_bias',64*p.KVWORDS),
      ('head_wo_weight',16*64*p.KVWORDS),('head_wo_bias',16*p.KVWORDS),('head_next_norm',p.KVWORDS),
      ('head_out',16*16*p.OUT),('head_relu',64*16*p.OUT)]:
        pl['offsets'][key]=cursor; pl['sizes'][key]=size; cursor+=size+16
    pl['words']=cursor
    return pl


def parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); nodes=graph.graph.node; initial={t.name:t for t in graph.graph.initializer}
    names=['m.head.layers.0.self_attn.out_proj.weight','m.head.layers.0.self_attn.out_proj.bias',
      'm.head.layers.0.norm2.weight','m.head.layers.0.norm2.bias','val_3201','m.head.layers.0.linear1.bias',
      'val_3203','m.head.layers.0.linear2.bias','m.head.layers.1.norm1.weight','m.head.layers.1.norm1.bias']
    for index,kind,params in [(1677,'Gemm',names[:2]),(1682,'LayerNormalization',names[2:4]),
      (1683,'MatMul',[names[4]]),(1684,'Add',[names[5]]),(1686,'MatMul',[names[6]]),
      (1687,'Add',[names[7]]),(1689,'LayerNormalization',names[8:])]:
        node=nodes[index]
        if node.op_type!=kind or list(node.input[1:])!=params: raise ValueError('head projection/MLP structure')
        attrs={x.name:onnx.helper.get_attribute_value(x) for x in node.attribute}
        if kind=='Gemm' and (attrs.get('transB',0)!=1 or attrs.get('transA',0)!=0 or attrs.get('alpha',1)!=1 or attrs.get('beta',1)!=1): raise ValueError('head Gemm semantics')
        if kind=='LayerNormalization' and (attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5)): raise ValueError('head norm semantics')
    if nodes[1685].op_type!='Relu': raise ValueError('head activation changed')
    values=[onnx.numpy_helper.to_array(initial[name],base_dir=str(model.parent)).copy() for name in names]
    for value,shape in zip(values,[(1024,1024),(1024,),(1024,),(1024,),(1024,4096),(4096,),(4096,1024),(1024,),(1024,),(1024,)]):
        if value.dtype!=np.float32 or value.shape!=shape: raise ValueError('head post parameter geometry')
    hashes={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in zip(names,values)}
    for index in (0,4,6): values[index]=p.attention.rounded(values[index].T.copy() if index==0 else values[index])
    return values,hashes


def initialize(data,pl,values,fill):
    off=pl['offsets']; sizes=pl['sizes']
    for index,key in [(0,'head_out_weight'),(4,'head_wi_weight'),(6,'head_wo_weight')]:
        packet=b.weight_packet(values[index],fill)
        if packet.size!=sizes[key]: raise ValueError('head post weight packet geometry')
        data[off[key]:off[key]+packet.size]=packet
    for index,key in [(1,'head_out_bias'),(5,'head_wi_bias'),(7,'head_wo_bias')]:
        dest=data[off[key]:off[key]+sizes[key]].reshape(-1,p.KVWORDS)
        dest[:,:64]=values[index].reshape(-1,64).view(np.uint32)
    for index,key in [(2,'head_norm2'),(8,'head_next_norm')]:
        dest=data[off[key]:off[key]+p.KVWORDS]; dest.fill(0)
        dest[:1024]=values[index].view(np.uint32); dest[1024:2048]=values[index+1].view(np.uint32)


def gemm(seq,pl,source,target,weight,kblocks,features):
    off=pl['offsets']; groups=features//4; seq.control(2,kblocks)
    if features%4: raise ValueError('four-column feature extent')
    for group in range(groups):
        for mg in range(4):
            for col in range(4):
                seq.dma(col,'q',off[source]+mg*4*p.OUT,[1,kblocks,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                seq.dma(col,'out',off[target]+((col*groups+group)*4+mg)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off[weight]+(col*groups+group)*kblocks*p.KVWORDS,[1,1,kblocks,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()


def bias(seq,pl,source,target,bias,features,mode):
    off=pl['offsets']; groups=features//4; seq.control(mode); size=16*p.OUT
    for first in range(0,features,4):
        for col in range(4):
            feature=first+col; start=off[source]+(col*groups+feature//4)*size
            seq.dma(col,'q',start,[1,4,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            if mode==16:
                seq.dma(col,'out',off[target]+feature*65,[4,4,32,65],[128*p.OUT,32*p.OUT,p.OUT,1],True)
            else:
                seq.dma(col,'out',off[target]+feature*size,[1,1,size//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',off[bias]+feature*p.KVWORDS,[16,1,1,p.KVWORDS],[0,0,0,1])
        seq.wait()


def norm(seq,pl,source,target,affine):
    off=pl['offsets']; seq.control(1)
    for col in range(4):
        seq.dma(col,'q',off[source]+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off[target]+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off[affine],[32,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait()


def runtime(pl):
    seq=v.Sequence(pl['words'])
    gemm(seq,pl,'packed_attention','head_out','head_out_weight',16,16)
    bias(seq,pl,'head_out','mlp_norm','head_out_bias',16,16)
    norm(seq,pl,'mlp_norm','norm','head_norm2')
    b.r.pack_norm(seq,pl,pl['offsets']['norm'])
    gemm(seq,pl,'packed_norm','mlp_wi','head_wi_weight',16,64)
    bias(seq,pl,'mlp_wi','head_relu','head_wi_bias',64,13)
    gemm(seq,pl,'head_relu','projection','head_wo_weight',64,16)
    bias(seq,pl,'projection','norm','head_wo_bias',16,16)
    norm(seq,pl,'norm','mlp_norm','head_next_norm')
    return seq.finish()


def unpack_relu(raw):
    # Generalized feature-major physical operand layout, 512 x 4096.
    bits=raw[:,16:1040].copy().view(np.uint16).reshape(64,4,4,8,8,4,8)
    value=bits.transpose(0,1,2,3,5,4,6).reshape(64,4,4,32,64)
    u=value.transpose(1,2,3,0,4).reshape(512,4096).astype(np.uint32)<<16
    return u.view(np.float32)


def reference(encoder_y,pre,post,qtype,valid):
    table,g0,b0,qkv,bias=pre
    residual,y=p.norm.oracle(encoder_y,np.broadcast_to(table[qtype],encoder_y.shape),g0,b0)
    projected=((p.attention.rounded(y).astype(np.float64)@qkv.astype(np.float64)).astype(np.float32).astype(np.float64)+bias).astype(np.float32)
    heads=p.attention.rounded(projected.reshape(512,3,16,64).transpose(1,2,0,3))
    mask=np.zeros((512,512),np.int32); mask[:,:valid]=1
    attention=p.attention.oracle(*heads,mask).transpose(1,0,2).reshape(512,1024)
    out=((p.attention.rounded(attention).astype(np.float64)@post[0].astype(np.float64)).astype(np.float32).astype(np.float64)+post[1]).astype(np.float32)
    residual,y=p.norm.oracle(out,residual,post[2],post[3])
    hidden=((p.attention.rounded(y).astype(np.float64)@post[4].astype(np.float64)).astype(np.float32).astype(np.float64)+post[5]).astype(np.float32)
    hidden=p.attention.rounded(np.maximum(hidden,0))
    out=((hidden.astype(np.float64)@post[6].astype(np.float64)).astype(np.float32).astype(np.float64)+post[7]).astype(np.float32)
    return p.norm.oracle(out,residual,post[8],post[9])


def verify(observed,pl,values,encoder_y,type_embedding):
    off=pl['offsets']; sizes=pl['sizes']
    def region(key): return observed[off[key]:off[key]+sizes[key]].reshape(-1,p.OUT)
    attention=b.r.unpack_activation(region('packed_attention'))
    raw_out=b.r.packet_matrix(region('head_out'),1024)
    direct_out=attention.astype(np.float64)@values[0].astype(np.float64)
    added=(raw_out.astype(np.float64)+values[1]).astype(np.float32)
    head_r=(encoder_y.astype(np.float64)+type_embedding.astype(np.float64)).astype(np.float32)
    ref_r,ref_y=p.norm.oracle(added,head_r,values[2],values[3])
    norm_input=region('norm'); stored_r=norm_input[:,1040:2064].copy().view(np.float32)
    packed=b.r.unpack_activation(region('packed_norm'))
    wi=b.r.packet_matrix(region('mlp_wi')[:64*16],4096)
    direct_wi=packed.astype(np.float64)@values[4].astype(np.float64)
    relu=unpack_relu(region('head_relu'))
    direct_relu=p.attention.rounded(np.maximum((wi.astype(np.float64)+values[5]).astype(np.float32),0))
    wo=b.r.packet_matrix(region('projection')[:16*16],1024)
    direct_wo=relu.astype(np.float64)@values[6].astype(np.float64)
    actual_added=norm_input[:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
    expected_added=(wo.astype(np.float64)+values[7]).astype(np.float32)
    final=region('mlp_norm'); y=final[:,:1024].copy().view(np.float32); residual=final[:,1040:2064].copy().view(np.float32)
    next_r,next_y=p.norm.oracle(actual_added,stored_r,values[8],values[9])
    guards=bool(np.all(norm_input[:,:1040].reshape(512,16,65)[:,:,64]==b.r.mlp.POISON) and np.all(norm_input[:,2064:]==b.r.mlp.POISON) and np.all(final[:,1024:1040]==b.r.mlp.POISON) and np.all(final[:,2064:]==b.r.mlp.POISON))
    for key,count,used in [('head_out',256,2048),('projection',256,2048),('mlp_wi',1024,2048),('head_relu',1024,1024)]:
        rows=region(key)[:count]; guards &= bool(np.all(rows[:,:16]==b.r.mlp.POISON) and np.all(rows[:,16+used:]==b.r.mlp.POISON))
    return dict(head_post_guards=dict(ok=guards),head_out=b.numeric(raw_out,direct_out,.0002,.0002,.00001),
      head_norm2_residual=dict(ok=bool(np.array_equal(stored_r.view(np.uint32),ref_r.view(np.uint32)))),
      head_norm2_packed=b.numeric(packed,p.attention.rounded(ref_y),.000002,.0078125,.004),
      head_wi=b.numeric(wi,direct_wi,.0002,.0002,.00001),
      head_relu=dict(ok=bool(np.array_equal(relu.view(np.uint32),direct_relu.view(np.uint32)))),
      head_wo=b.numeric(wo,direct_wo,.0002,.0002,.00001),
      head_wo_bias=dict(ok=bool(np.array_equal(actual_added.view(np.uint32),expected_added.view(np.uint32)))),
      head_next_norm=b.r.norm_verdict(residual,y,next_r,next_y))


def build(attention,vector,out):
    am=json.loads((attention/'build.json').read_text()); vm=json.loads((vector/'build.json').read_text())
    if am['sources']!=a.sources() or vm['sources']!=v.sources(): raise ValueError('stale head attention/vector')
    pl=plan(am['plan']); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(vector/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(v.prefix()+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('head post changes resident vector core/init')
    p.attention.dump(out/'build.json',dict(sources=sources(),plan=pl,head_attention_build_sha256=p.attention.sha(attention/'build.json'),
      vector_build_sha256=p.attention.sha(vector/'build.json'),audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('HEAD_POST_BUILD '+json.dumps(dict(words=pl['words'],same_core_and_init=True,scope='compile/audit only; first head post operations require device qualification')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--head-attention',type=Path,required=True); ap.add_argument('--vector',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.head_attention.resolve(),args.vector.resolve(),args.out.resolve())
