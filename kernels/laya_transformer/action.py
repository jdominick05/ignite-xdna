#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NPU logits masking, action features and both action-head products."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import tail as t
import tail_probe as tp
s=t.scorer; v=t.v; b=t.b; p=t.p


def sources():
    return dict(t.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in (Path(__file__),Path(tp.__file__))})


def plan(base):
    pl=copy.deepcopy(base); cursor=pl['words']
    for key,size in [('tail_metadata',p.QWORDS),('tail_features',16*p.OUT),('action_w1',4*17*p.KVWORDS),
      ('action_b1',4*p.KVWORDS),('action_w2',4*4*p.KVWORDS),('action_b2',4*p.KVWORDS),
      ('action_input',17*4*p.OUT),('action_hidden',16*p.OUT),('action_activated',16*p.OUT),
      ('action_scores',16*p.OUT),('action_biased',16*p.OUT),('action_probs',16*p.OUT)]:
        pl['offsets'][key]=cursor; pl['sizes'][key]=size; cursor+=size+16
    pl['words']=cursor
    return pl


def parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); nodes=graph.graph.node; initial={x.name:x for x in graph.graph.initializer}
    names=['m.act_head.0.weight','m.act_head.0.bias','m.act_head.2.weight','m.act_head.2.bias']
    for index,params in [(1778,names[:2]),(1784,names[2:])]:
        node=nodes[index]; attrs={x.name:onnx.helper.get_attribute_value(x) for x in node.attribute}
        if node.op_type!='Gemm' or list(node.input[1:])!=params: raise ValueError('action products changed')
        if attrs.get('transB',0)!=1 or attrs.get('transA',0)!=0 or attrs.get('alpha',1)!=1 or attrs.get('beta',1)!=1: raise ValueError('action Gemm semantics')
    if nodes[1785].op_type!='Softmax' or nodes[1753].op_type!='Where': raise ValueError('action outputs changed')
    values=[onnx.numpy_helper.to_array(initial[name],base_dir=str(model.parent)).copy() for name in names]
    for value,shape in zip(values,[(256,1028),(256,),(2,256),(2,)]):
        if value.dtype!=np.float32 or value.shape!=shape: raise ValueError('action parameter geometry')
    hashes={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in zip(names,values)}
    for index in (0,2): values[index]=p.attention.rounded(values[index].T.copy())
    return values,hashes


def metadata(mask):
    mask=np.asarray(mask)
    if mask.ndim!=1 or mask.dtype.kind!='b' or not 2<=mask.size<=128: raise ValueError('option mask/count')
    output=np.zeros(p.QWORDS,np.uint32); output[0]=mask.size; output[16:16+mask.size]=mask.astype(np.uint32)
    return output


def initialize(data,pl,values,mask,fill):
    off=pl['offsets']; sizes=pl['sizes']; data[off['tail_metadata']:off['tail_metadata']+p.QWORDS]=metadata(mask)
    for index,key in [(0,'action_w1'),(2,'action_w2')]:
        packet=b.weight_packet(values[index],fill)
        if packet.size!=sizes[key]: raise ValueError('action weight geometry')
        data[off[key]:off[key]+packet.size]=packet
    for index,key in [(1,'action_b1'),(3,'action_b2')]:
        dest=data[off[key]:off[key]+sizes[key]].reshape(4,p.KVWORDS); dest[:,:64]=0
        padded=np.zeros(256,np.float32); padded[:values[index].size]=values[index]
        dest[:,:64]=padded.reshape(4,64).view(np.uint32)


def runtime(pl):
    off=pl['offsets']; seq=t.Sequence(pl['words']); seq.control(19)
    for col in range(4):
        seq.dma(col,'q',off['tail_metadata'],[1,1,4,p.QWORDS],[0,0,0,1])
        seq.dma(col,'out',off['tail_features']+col*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['scorer_biased'],[1,1,4,p.KVWORDS],[0,0,p.OUT,1])
    seq.wait()
    # One column creates the shared feature-major operands; core row zero
    # owns the sole action row. Other rows are zero before the biased products.
    for feature in range(17):
        seq.control(20,position=feature)
        seq.dma(0,'q',off['mlp_norm'],[1,1,4,p.QWORDS],[0,0,0,1])
        seq.dma(0,'out',off['action_input']+feature*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
        seq.dma(0,'kv',off['tail_features'],[4,1,1,p.KVWORDS],[0,0,0,1])
        seq.lines.append('      aiex.npu.dma_wait {symbol = @outL2_0}')
    for source,target,weight,bias,activated,kblocks,mode in [
      ('action_input','action_hidden','action_w1','action_b1','action_activated',17,14),
      ('action_activated','action_scores','action_w2','action_b2','action_biased',4,15)]:
        seq.control(2,kblocks)
        for col in range(4):
            seq.dma(col,'q',off[source],[1,kblocks,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            seq.dma(col,'out',off[target]+col*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
            seq.dma(col,'kv',off[weight]+col*kblocks*p.KVWORDS,[1,1,kblocks,p.KVWORDS],[0,0,p.KVWORDS,1])
        seq.wait(); seq.control(mode)
        for col in range(4):
            seq.dma(col,'q',off[target]+col*4*p.OUT,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
            seq.dma(col,'out',off[activated]+col*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
            seq.dma(col,'kv',off[bias]+col*p.KVWORDS,[4,1,1,p.KVWORDS],[0,0,0,1])
        seq.wait()
    seq.control(21)
    for col in range(4):
        seq.dma(col,'q',off['action_biased'],[1,1,4,p.QWORDS],[0,0,0,1])
        seq.dma(col,'out',off['action_probs']+col*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['tail_metadata'],[4,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait(); return seq.finish()


def reference(head_sum,scores,mask,values):
    logits,prob,features,logs=tp.oracle(len(mask),mask,scores)
    source=p.attention.rounded(np.concatenate((head_sum[0],features.astype(np.float32))))
    hidden=((source.astype(np.float64)@values[0].astype(np.float64)).astype(np.float32).astype(np.float64)+values[1]).astype(np.float32)
    activated=p.attention.rounded(b.gelu(hidden))
    raw=((activated.astype(np.float64)@values[2].astype(np.float64)).astype(np.float32).astype(np.float64)+values[3]).astype(np.float32)
    output=np.exp(raw.astype(np.float64)-float(raw.max())); output/=output.sum()
    return logits[:len(mask)],output


def outputs(observed,pl,n):
    off=pl['offsets']
    logits=observed[off['tail_features']+16:off['tail_features']+16+n].copy().view(np.float32)
    probs=observed[off['action_probs']+16:off['action_probs']+18].copy().view(np.float32)
    return logits,probs


def verify(observed,pl,values,mask):
    off=pl['offsets']; sizes=pl['sizes']; n=len(mask)
    def region(name): return observed[off[name]:off[name]+sizes[name]].reshape(-1,p.OUT)
    raw=s.matrix(region('scorer_biased'),1).ravel(); logits,prob,features,logs=tp.oracle(n,mask,raw)
    tail=region('tail_features'); floats=tail.copy().view(np.float32)
    source=s.unpack(region('action_input'),17); expected_source=np.zeros((128,1088),np.float32)
    cls=observed[off['mlp_norm']:off['mlp_norm']+1024].copy().view(np.float32)
    expected_source[0,:1028]=p.attention.rounded(np.concatenate((cls,floats[0,144:148])))
    hidden=s.matrix(region('action_hidden'),256); padded_weight=np.zeros((1088,256),np.float32); padded_weight[:1028]=values[0]
    direct_hidden=source.astype(np.float64)@padded_weight.astype(np.float64)
    activated=s.unpack(region('action_activated'),4)
    direct_activation=p.attention.rounded(b.gelu((hidden.astype(np.float64)+values[1]).astype(np.float32)))
    scores=s.matrix(region('action_scores'),2); biased=s.matrix(region('action_biased'),2)
    direct_scores=activated.astype(np.float64)@values[2].astype(np.float64)
    direct_biased=(scores.astype(np.float64)+values[3]).astype(np.float32)
    expected_prob=np.exp(biased[0].astype(np.float64)-float(biased[0].max())); expected_prob/=expected_prob.sum()
    result=region('action_probs'); actual=result[:,16:18].copy().view(np.float32)
    guards=np.ones(p.OUT,bool); guards[16:18]=False; guard=bool(np.all(result[:,guards]==b.r.mlp.POISON))
    tail_used=np.zeros(p.OUT,bool); tail_used[16:148]=True; tail_used[160:416]=True
    guard &= bool(np.all(tail[:,~tail_used]==b.r.mlp.POISON))
    for name,used in [('action_input',1024),('action_hidden',2048),('action_activated',1024),('action_scores',2048),('action_biased',2048)]:
        rows=region(name); guard &= bool(np.all(rows[:,:16]==b.r.mlp.POISON) and np.all(rows[:,16+used:]==b.r.mlp.POISON))
    return dict(masked_logits_bits=dict(ok=bool(np.all(tail[:,16:144]==logits.view(np.uint32)))),
      tail_probabilities=b.numeric(floats[:,160:160+n],np.broadcast_to(prob,(16,n)),*tp.GATES),
      tail_features=b.numeric(floats[:,144:148],np.broadcast_to(features,(16,4)),*tp.GATES),
      tail_logarithms=b.numeric(floats[:,288:288+n],np.broadcast_to(logs,(16,n)),2e-5,2e-6,2e-5),
      action_input_bits=dict(ok=bool(np.array_equal(source.view(np.uint32),expected_source.view(np.uint32)))),
      action_hidden=b.numeric(hidden,direct_hidden,.0002,.0002,.00001),action_gelu=b.numeric(activated,direct_activation,.000002,.0078125,.004),
      action_product=b.numeric(scores,direct_scores,.0002,.0002,.00001),action_bias_bits=dict(ok=bool(np.array_equal(biased.view(np.uint32),direct_biased.view(np.uint32)))),
      action_softmax=b.numeric(actual,np.broadcast_to(expected_prob,(16,2)),*tp.GATES),action_guards=dict(ok=guard))


def build(scorer,tail,out):
    sm=json.loads((scorer/'build.json').read_text()); tm=json.loads((tail/'build.json').read_text())
    if sm['sources']!=s.sources() or tm['sources']!=t.sources(): raise ValueError('stale scorer/tail')
    pl=plan(sm['plan']); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(tail/'kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(t.prefix()+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=tm['audits'] or cdo!=tm['cdo']: raise ValueError('action core/init differs from qualified tail')
    p.attention.dump(out/'build.json',dict(sources=sources(),plan=pl,scorer_build_sha256=p.attention.sha(scorer/'build.json'),tail_build_sha256=p.attention.sha(tail/'build.json'),
      audits=audits,cdo=cdo,xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('ACTION_BUILD '+json.dumps(dict(words=pl['words'],same_qualified_core_and_init=True,scope='compile/audit only; full-model splice needs device qualification')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--scorer',type=Path,required=True); ap.add_argument('--tail',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.scorer.resolve(),args.tail.resolve(),args.out.resolve())
