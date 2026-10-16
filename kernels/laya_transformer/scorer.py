#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NPU marker-row gather, scorer norm and both biased scorer products."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import head_second as second
v=second.v; b=second.b; p=second.p


def sources():
    return dict(second.sources(),**{str(Path(__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(__file__))})


def markers(value):
    array=np.asarray(value)
    if array.ndim!=1 or array.dtype.kind not in 'iu' or not 2<=array.size<=128 or np.any(array>=512): raise ValueError('marker positions/count')
    result=np.zeros(128,np.int64); result[:array.size]=np.maximum(array,0)
    return result


def plan(base):
    pl=copy.deepcopy(base); cursor=pl['words']
    for key,size in [('scorer_affine',p.KVWORDS),('scorer_w1',16*16*p.KVWORDS),('scorer_b1',16*p.KVWORDS),
      ('scorer_w2',4*16*p.KVWORDS),('scorer_b2',4*p.KVWORDS),('scorer_input',128*p.OUT),
      ('scorer_norm',128*p.OUT),('scorer_packed',16*4*p.OUT),('scorer_hidden',16*4*p.OUT),
      ('scorer_activated',16*4*p.OUT),('scorer_scores',4*4*p.OUT),('scorer_biased',4*4*p.OUT)]:
        pl['offsets'][key]=cursor; pl['sizes'][key]=size; cursor+=size+16
    pl['words']=cursor
    return pl


def parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); nodes=graph.graph.node; initial={t.name:t for t in graph.graph.initializer}
    names=['m.scorer.0.weight','m.scorer.0.bias','val_3346','m.scorer.1.bias','val_3355','m.scorer.3.bias']
    for index,kind,params in [(1741,'LayerNormalization',names[:2]),(1742,'MatMul',[names[2]]),
      (1743,'Add',[names[3]]),(1749,'MatMul',[names[4]]),(1750,'Add',[names[5]])]:
        node=nodes[index]; attrs={x.name:onnx.helper.get_attribute_value(x) for x in node.attribute}
        if node.op_type!=kind or list(node.input[1:])!=params: raise ValueError('scorer graph changed')
        if kind=='LayerNormalization' and (attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5)): raise ValueError('scorer norm semantics')
    if nodes[1737].op_type!='Clip' or list(nodes[1737].input)!=['marker_pos','val_10'] or nodes[1740].op_type!='GatherElements': raise ValueError('marker gather graph changed')
    values=[onnx.numpy_helper.to_array(initial[name],base_dir=str(model.parent)).copy() for name in names]
    for value,shape in zip(values,[(1024,),(1024,),(1024,1024),(1024,),(1024,1),(1,)]):
        if value.dtype!=np.float32 or value.shape!=shape: raise ValueError('scorer parameter geometry')
    hashes={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in zip(names,values)}
    for index in (2,4): values[index]=p.attention.rounded(values[index])
    return values,hashes


def initialize(data,pl,values,fill):
    off=pl['offsets']; sizes=pl['sizes']
    start=off['scorer_affine']; data[start:start+p.KVWORDS]=0
    data[start:start+1024]=values[0].view(np.uint32); data[start+1024:start+2048]=values[1].view(np.uint32)
    for index,key in [(2,'scorer_w1'),(4,'scorer_w2')]:
        packet=b.weight_packet(values[index],fill)
        if packet.size!=sizes[key]: raise ValueError('scorer weight packet geometry')
        data[off[key]:off[key]+packet.size]=packet
    for index,key,features in [(3,'scorer_b1',16),(5,'scorer_b2',4)]:
        dest=data[off[key]:off[key]+sizes[key]].reshape(features,p.KVWORDS); dest[:,:64]=0
        padded=np.zeros(features*64,np.float32); padded[:values[index].size]=values[index]
        dest[:,:64]=padded.reshape(features,64).view(np.uint32)


def runtime(pl,positions=None):
    positions=markers(np.arange(128,dtype=np.int64) if positions is None else positions)
    off=pl['offsets']; seq=v.Sequence(pl['words']); seq.control(11)
    for group in range(8):
        for col in range(4):
            start=off['scorer_input']+(col*32+group*4)*p.OUT
            seq.dma(col,'out',start,[1,1,p.OUT//4,16],[0,0,16,1],True)
            for row,bid in enumerate((0,1,4,5)):
                selected=int(positions[col*32+group*4+row])
                seq.dma(col,'q',off['mlp_norm']+selected*p.OUT,[1,1,p.QWORDS//16,16],[0,0,16,1],bid=bid)
            seq.dma(col,'kv',off['scorer_affine'],[1,1,1,p.KVWORDS],[0,0,0,1])
        seq.wait()
    seq.control(12)
    for col in range(4):
        seq.dma(col,'q',off['scorer_input']+col*32*p.OUT,[1,1,32,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['scorer_norm']+col*32*p.OUT,[1,1,32*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['scorer_affine'],[8,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait()
    for group in range(4):
        seq.control(10,position=tuple(group*4+col for col in range(4)))
        for col in range(4):
            feature=group*4+col
            seq.dma(col,'q',off['scorer_norm'],[1,1,4,p.QWORDS],[0,0,p.OUT,1])
            seq.dma(col,'out',off['scorer_packed']+feature*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
            seq.dma(col,'kv',off['scorer_norm'],[1,1,128,p.KVWORDS],[0,0,p.OUT,1])
        seq.wait()
    for source,target,weight,bias,features,mode,biased in [
      ('scorer_packed','scorer_hidden','scorer_w1','scorer_b1',16,14,'scorer_activated'),
      ('scorer_activated','scorer_scores','scorer_w2','scorer_b2',4,15,'scorer_biased')]:
        groups=features//4; seq.control(2,16)
        for group in range(groups):
            for col in range(4):
                seq.dma(col,'q',off[source],[1,16,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
                seq.dma(col,'out',off[target]+(col*groups+group)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off[weight]+(col*groups+group)*16*p.KVWORDS,[1,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
        seq.control(mode)
        for first in range(0,features,4):
            for col in range(4):
                feature=first+col; start=off[target]+(col*groups+feature//4)*4*p.OUT
                seq.dma(col,'q',start,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                output=(col*groups+feature//4) if mode==15 else feature
                seq.dma(col,'out',off[biased]+output*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off[bias]+feature*p.KVWORDS,[4,1,1,p.KVWORDS],[0,0,0,1])
            seq.wait()
    return seq.finish()


def prepare(raw,pl,positions):
    positions=markers(positions); off=pl['offsets']['mlp_norm']; end=off+512*p.OUT
    order=[col*32+group*4+row for group in range(8) for col in range(4) for row in range(4)]
    output=bytearray(raw); count=0
    for op in v.s.txn.parse(raw):
        if op['op']!='DDR_PATCH' or op['arg_idx']!=0 or not off*4<=op['arg_offset']<end*4: continue
        if count>=128 or op['arg_offset']!=(off+order[count]*p.OUT)*4: raise ValueError('scorer gather template changed')
        struct.pack_into('<I',output,op['offset']+40,(off+int(positions[order[count]])*p.OUT)*4); count+=1
    if count!=128: raise ValueError('missing scorer marker reads')
    return bytes(output)


def unpack(raw,features):
    bits=raw[:,16:1040].copy().view(np.uint16).reshape(features,4,8,8,4,8)
    full=bits.transpose(0,1,2,4,3,5).reshape(features,128,64).transpose(1,0,2).reshape(128,features*64)
    return (full.astype(np.uint32)<<16).view(np.float32)


def matrix(raw,n):
    return p.gemm_output(raw[:,16:2064].copy().view(np.float32),n,m=128)


def reference(head_sum,positions,values):
    gathered=head_sum[markers(positions)]
    _,y=p.norm.oracle(np.zeros_like(gathered),gathered,values[0],values[1])
    hidden=((p.attention.rounded(y).astype(np.float64)@values[2].astype(np.float64)).astype(np.float32).astype(np.float64)+values[3]).astype(np.float32)
    activated=p.attention.rounded(b.gelu(hidden))
    return ((activated.astype(np.float64)@values[4].astype(np.float64)).astype(np.float32).astype(np.float64)+values[5]).astype(np.float32).ravel()


def verify(observed,pl,values,positions,head_sum):
    off=pl['offsets']; sizes=pl['sizes']
    def region(name): return observed[off[name]:off[name]+sizes[name]].reshape(-1,p.OUT)
    selected=head_sum[markers(positions)]; normalized=region('scorer_norm')
    y=normalized[:,:1024].copy().view(np.float32); residual=normalized[:,1040:2064].copy().view(np.float32)
    _,norm_ref=p.norm.oracle(np.zeros_like(selected),selected,values[0],values[1])
    packed=unpack(region('scorer_packed'),16); hidden=matrix(region('scorer_hidden'),1024)
    direct_hidden=packed.astype(np.float64)@values[2].astype(np.float64)
    activated=unpack(region('scorer_activated'),16)
    direct_activation=p.attention.rounded(b.gelu((hidden.astype(np.float64)+values[3]).astype(np.float32)))
    raw_scores=matrix(region('scorer_scores'),1).ravel(); biased=matrix(region('scorer_biased'),1).ravel()
    direct_scores=(activated.astype(np.float64)@values[4].astype(np.float64)).ravel()
    direct_bias=(raw_scores.astype(np.float64)+values[5][0]).astype(np.float32)
    guard=True
    for name,used in [('scorer_packed',1024),('scorer_activated',1024),('scorer_hidden',2048),('scorer_scores',2048),('scorer_biased',2048)]:
        rows=region(name); guard &= bool(np.all(rows[:,:16]==b.r.mlp.POISON) and np.all(rows[:,16+used:]==b.r.mlp.POISON))
    guard &= bool(np.all(normalized[:,1024:1040]==b.r.mlp.POISON) and np.all(normalized[:,2064:]==b.r.mlp.POISON))
    return dict(scorer_norm=b.r.norm_verdict(residual,y,selected,norm_ref),scorer_pack=dict(ok=bool(np.array_equal(packed,p.attention.rounded(y)))),
      scorer_hidden=b.numeric(hidden,direct_hidden,.0002,.0002,.00001),scorer_gelu=b.numeric(activated,direct_activation,.000002,.0078125,.004),
      scorer_product=b.numeric(raw_scores,direct_scores,.0002,.0002,.00001),scorer_bias=dict(ok=bool(np.array_equal(biased.view(np.uint32),direct_bias.view(np.uint32)))),
      scorer_guards=dict(ok=guard))


def build(head,vector,out):
    hm=json.loads((head/'build.json').read_text()); vm=json.loads((vector/'build.json').read_text())
    if hm['sources']!=second.sources() or vm['sources']!=v.sources(): raise ValueError('stale head/vector build')
    pl=plan(hm['plan']); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(vector/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(v.prefix()+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('scorer changes vector core/init')
    p.attention.dump(out/'build.json',dict(sources=sources(),plan=pl,head_second_build_sha256=p.attention.sha(head/'build.json'),
      vector_build_sha256=p.attention.sha(vector/'build.json'),audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('SCORER_BUILD '+json.dumps(dict(words=pl['words'],same_core_and_init=True,scope='marker gather/norm/products only; logits masking and action tail absent')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--head-second',type=Path,required=True); ap.add_argument('--vector',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.head_second.resolve(),args.vector.resolve(),args.out.resolve())
