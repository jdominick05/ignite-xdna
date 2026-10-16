#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Second head projections and ReLU MLP; attention reuses the qualified image."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import head_post as post
a=post.a; v=post.v; b=post.b; p=post.p


def sources():
    return dict(post.sources(),**{str(Path(__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(__file__))})


def plan(base):
    pl=copy.deepcopy(base); cursor=pl['words']
    for key,size in [('second_qkv_weight',48*16*p.KVWORDS),('second_qkv_bias',48*p.KVWORDS),
      ('second_out_weight',16*16*p.KVWORDS),('second_out_bias',16*p.KVWORDS),('second_norm2',p.KVWORDS),
      ('second_wi_weight',64*16*p.KVWORDS),('second_wi_bias',64*p.KVWORDS),
      ('second_wo_weight',16*64*p.KVWORDS),('second_wo_bias',16*p.KVWORDS)]:
        pl['offsets'][key]=cursor; pl['sizes'][key]=size; cursor+=size+16
    pl['words']=cursor
    return pl


def parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); nodes=graph.graph.node; init={t.name:t for t in graph.graph.initializer}
    names=['val_3210','m.head.layers.1.self_attn.in_proj_bias','m.head.layers.1.self_attn.out_proj.weight',
      'm.head.layers.1.self_attn.out_proj.bias','m.head.layers.1.norm2.weight','m.head.layers.1.norm2.bias',
      'val_3313','m.head.layers.1.linear1.bias','val_3315','m.head.layers.1.linear2.bias']
    for index,kind,params in [(1691,'MatMul',[names[0]]),(1692,'Add',[names[1]]),(1726,'Gemm',names[2:4]),
      (1730,'LayerNormalization',names[4:6]),(1731,'MatMul',[names[6]]),(1732,'Add',[names[7]]),
      (1734,'MatMul',[names[8]]),(1735,'Add',[names[9]])]:
        node=nodes[index]; attrs={x.name:onnx.helper.get_attribute_value(x) for x in node.attribute}
        if node.op_type!=kind or list(node.input[1:])!=params: raise ValueError('second head graph changed')
        if kind=='Gemm' and (attrs.get('transB',0)!=1 or attrs.get('transA',0)!=0 or attrs.get('alpha',1)!=1 or attrs.get('beta',1)!=1): raise ValueError('second head Gemm semantics')
        if kind=='LayerNormalization' and (attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5)): raise ValueError('second head norm semantics')
    if nodes[1733].op_type!='Relu' or nodes[1736].op_type!='Add': raise ValueError('second head activation/residual changed')
    values=[onnx.numpy_helper.to_array(init[name],base_dir=str(model.parent)).copy() for name in names]
    for value,shape in zip(values,[(1024,3072),(3072,),(1024,1024),(1024,),(1024,),(1024,),(1024,4096),(4096,),(4096,1024),(1024,)]):
        if value.dtype!=np.float32 or value.shape!=shape: raise ValueError('second head parameter geometry')
    hashes={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in zip(names,values)}
    for index in (0,2,6,8): values[index]=p.attention.rounded(values[index].T.copy() if index==2 else values[index])
    return values,hashes


def initialize(data,pl,values,fill):
    off=pl['offsets']; sizes=pl['sizes']
    for index,key in [(0,'second_qkv_weight'),(2,'second_out_weight'),(6,'second_wi_weight'),(8,'second_wo_weight')]:
        packet=b.weight_packet(values[index],fill)
        if packet.size!=sizes[key]: raise ValueError('second head weight packet geometry')
        data[off[key]:off[key]+packet.size]=packet
    for index,key in [(1,'second_qkv_bias'),(3,'second_out_bias'),(7,'second_wi_bias'),(9,'second_wo_bias')]:
        dest=data[off[key]:off[key]+sizes[key]].reshape(-1,p.KVWORDS); dest[:,:64]=values[index].reshape(-1,64).view(np.uint32)
    start=off['second_norm2']; data[start:start+p.KVWORDS]=0
    data[start:start+1024]=values[4].view(np.uint32); data[start+1024:start+2048]=values[5].view(np.uint32)


def pre_runtime(pl):
    seq=v.Sequence(pl['words']); off=pl['offsets']; b.r.pack_norm(seq,pl,off['mlp_norm'])
    post.gemm(seq,pl,'packed_norm','projection','second_qkv_weight',16,48)
    seq.control(15); size=16*p.OUT
    for first in range(0,48,4):
        for col in range(4):
            feature=first+col; start=off['projection']+(col*12+feature//4)*size
            seq.dma(col,'q',start,[1,4,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            seq.dma(col,'out',start,[1,1,size//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',off['second_qkv_bias']+feature*p.KVWORDS,[16,1,1,p.KVWORDS],[0,0,0,1])
        seq.wait()
    return seq.finish()


def post_runtime(pl):
    seq=v.Sequence(pl['words']); off=pl['offsets']
    post.gemm(seq,pl,'packed_attention','head_out','second_out_weight',16,16)
    post.bias(seq,pl,'head_out','mlp_norm','second_out_bias',16,16)
    post.norm(seq,pl,'mlp_norm','norm','second_norm2')
    b.r.pack_norm(seq,pl,off['norm'])
    post.gemm(seq,pl,'packed_norm','mlp_wi','second_wi_weight',16,64)
    post.bias(seq,pl,'mlp_wi','head_relu','second_wi_bias',64,13)
    post.gemm(seq,pl,'head_relu','projection','second_wo_weight',64,16)
    post.bias(seq,pl,'projection','norm','second_wo_bias',16,16)
    seq.control(18)
    for col in range(4):
        seq.dma(col,'q',off['norm']+col*128*p.OUT,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['mlp_norm']+col*128*p.OUT,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['second_norm2'],[32,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait(); return seq.finish()


def reference(initial,values,valid):
    residual,y=initial
    projected=((p.attention.rounded(y).astype(np.float64)@values[0].astype(np.float64)).astype(np.float32).astype(np.float64)+values[1]).astype(np.float32)
    heads=p.attention.rounded(projected.reshape(512,3,16,64).transpose(1,2,0,3)); mask=np.zeros((512,512),np.int32); mask[:,:valid]=1
    attention=p.attention.oracle(*heads,mask).transpose(1,0,2).reshape(512,1024)
    added=((p.attention.rounded(attention).astype(np.float64)@values[2].astype(np.float64)).astype(np.float32).astype(np.float64)+values[3]).astype(np.float32)
    residual,y=p.norm.oracle(added,residual,values[4],values[5])
    hidden=((p.attention.rounded(y).astype(np.float64)@values[6].astype(np.float64)).astype(np.float32).astype(np.float64)+values[7]).astype(np.float32)
    added=((p.attention.rounded(np.maximum(hidden,0)).astype(np.float64)@values[8].astype(np.float64)).astype(np.float32).astype(np.float64)+values[9]).astype(np.float32)
    summed=(added.astype(np.float64)+residual.astype(np.float64)).astype(np.float32)
    return summed,summed


def verify_pre(observed,pl,values,input_y):
    off=pl['offsets']; sizes=pl['sizes']
    packed=b.r.unpack_activation(observed[off['packed_norm']:off['packed_norm']+sizes['packed_norm']].reshape(-1,p.OUT))
    actual=b.r.packet_matrix(observed[off['projection']:off['projection']+sizes['projection']].reshape(-1,p.OUT),3072)
    expected=((packed.astype(np.float64)@values[0].astype(np.float64)).astype(np.float32).astype(np.float64)+values[1]).astype(np.float32)
    return dict(second_input_pack=dict(ok=bool(np.array_equal(packed,p.attention.rounded(input_y)))),second_biased_qkv=b.numeric(actual,expected,.0002,.0002,.00001))


def verify_post(observed,pl,values,input_residual):
    # Reuse the first head's direct checks with identity final norm, then
    # replace that norm verdict with the actual exact residual-sum verdict.
    equivalent=[*values[2:10],np.ones(1024,np.float32),np.zeros(1024,np.float32)]
    checks=post.verify(observed,pl,equivalent,input_residual,np.zeros(1024,np.float32))
    checks.pop('head_next_norm')
    off=pl['offsets']; rows=observed[off['norm']:off['norm']+512*p.OUT].reshape(512,p.OUT)
    x=rows[:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
    residual=rows[:,1040:2064].copy().view(np.float32)
    expected=(x.astype(np.float64)+residual.astype(np.float64)).astype(np.float32)
    final=observed[off['mlp_norm']:off['mlp_norm']+512*p.OUT].reshape(512,p.OUT)
    checks['second_final_sum']=dict(ok=bool(np.array_equal(final[:,:1024],expected.view(np.uint32)) and np.array_equal(final[:,1040:2064],expected.view(np.uint32))))
    return {'second_'+key:value for key,value in checks.items()}


def build(first_post,vector,out):
    pm=json.loads((first_post/'build.json').read_text()); vm=json.loads((vector/'build.json').read_text())
    if pm['sources']!=post.sources() or vm['sources']!=v.sources(): raise ValueError('stale head post/vector')
    pl=plan(pm['plan']); out.mkdir(parents=True,exist_ok=False); artifacts={}
    for name,runtime in [('pre',pre_runtime),('post',post_runtime)]:
        folder=out/name; folder.mkdir(); shutil.copyfile(vector/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(v.prefix()+runtime(pl),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        old=p.STACK
        try:
            p.STACK=5376; audits=p.audit(folder,{})
        finally: p.STACK=old
        cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if json.loads(json.dumps(audits))!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('second head changes vector core/init')
        artifacts[name]=dict(audits=audits,cdo=cdo,xclbin=p.attention.sha(folder/'final.xclbin'),instructions=p.attention.sha(folder/'insts.bin'))
    p.attention.dump(out/'build.json',dict(sources=sources(),plan=pl,head_post_build_sha256=p.attention.sha(first_post/'build.json'),
      vector_build_sha256=p.attention.sha(vector/'build.json'),artifacts=artifacts))
    print('HEAD_SECOND_BUILD '+json.dumps(dict(words=pl['words'],same_core_and_init=True,scope='compile/audit only; second head needs device qualification')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--head-post',type=Path,required=True); ap.add_argument('--vector',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.head_post.resolve(),args.vector.resolve(),args.out.resolve())
