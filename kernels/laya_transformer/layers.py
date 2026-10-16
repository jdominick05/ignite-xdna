# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pinned encoder geometry, constant banks and device-resident layer views."""
import copy
import hashlib
import struct

import numpy as np
import block as b
import instructions as txn
p=b.p


def model_spec(model,count):
    import onnx
    if not 1<=count<=28: raise ValueError('encoder layer count')
    graph=onnx.load(str(model),load_external_data=False)
    initial={t.name:t for t in graph.graph.initializer}; nodes=list(graph.graph.node)
    products=[(i,n) for i,n in enumerate(nodes) if n.op_type=='MatMul' and n.input[1] in initial and len(initial[n.input[1]].dims)==2]
    if len(products)<112: raise ValueError('missing encoder products')
    by={out:node for node in nodes for out in node.output}; result=[]
    for layer in range(count):
        group=products[layer*4:layer*4+4]
        if [tuple(initial[n.input[1]].dims) for _,n in group]!=[(1024,3072),(1024,1024),(1024,5248),(2624,1024)]: raise ValueError('encoder geometry/order')
        trig=set(); visited=set(); boundary=set(group[0][1].output)
        def ancestors(name):
            if name in visited or name in boundary: return
            visited.add(name)
            if name in ('cos','sin','cos_1','sin_1'): trig.add(name); return
            if name in by:
                for parent in by[name].input: ancestors(parent)
        soft=[n for n in nodes[group[0][0]:group[1][0]] if n.op_type=='Softmax']
        if len(soft)!=1: raise ValueError('encoder attention softmax')
        ancestors(soft[0].input[0])
        if trig=={'cos','sin'}: mode='local'
        elif trig=={'cos_1','sin_1'}: mode='global'
        else: raise ValueError('unrecognized rotary coefficients')
        next_name=f'm.encoder.layers.{layer+1}.attn_norm.weight' if layer<27 else 'm.encoder.final_norm.weight'
        norms=[]
        for name in (f'm.encoder.layers.{layer}.mlp_norm.weight',next_name):
            matches=[n for n in nodes if n.op_type=='LayerNormalization' and n.input[1]==name]
            if len(matches)!=1: raise ValueError('missing encoder norm')
            node=matches[0]; attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5): raise ValueError('norm semantics')
            norms.append([name,node.input[2] if len(node.input)>2 else None])
        result.append(dict(layer=layer,mode=mode,weights=[n.input[1] for _,n in group],norms=norms))
    return graph,result


def constants(model,graph,spec):
    import onnx
    initial={t.name:t for t in graph.graph.initializer}; result=[]; hashes={}
    def load(name):
        value=onnx.numpy_helper.to_array(initial[name],base_dir=str(model.parent)).copy()
        hashes[name]=hashlib.sha256(value.tobytes()).hexdigest()
        return value
    for layer in spec:
        weights=[p.attention.rounded(load(name)) for name in layer['weights']]
        affine=[]
        for gamma,beta in layer['norms']:
            g=load(gamma); z=load(beta) if beta else np.zeros_like(g)
            if g.shape!=(1024,) or z.shape!=(1024,): raise ValueError('norm affine shape')
            affine.append((g,z))
        result.append(dict(weights=weights,affine=affine))
    return result,hashes


def plan(count):
    if not 1<=count<=28: raise ValueError('encoder layer count')
    pl=b.plan(); off=pl['offsets']; cursor=pl['words']
    pl['banks']=[dict(qkv=(off['input']+256*p.QWORDS,768*p.KVWORDS),
      projection=(off['projection_weight'],256*p.KVWORDS),wi=(off['mlp_wi_weight'],1344*p.KVWORDS),
      wo=(off['mlp_wo_weight'],656*p.KVWORDS),affine=(off['affine'],p.KVWORDS),next_affine=(off['mlp_affine'],p.KVWORDS))]
    pl['coefficients']={'global':(off['coeff'],16*p.KVWORDS),'local':(cursor,16*p.KVWORDS)}; cursor+=16*p.KVWORDS+16
    for layer in range(1,count):
        bank={}
        for name,(_,size) in pl['banks'][0].items(): bank[name]=(cursor,size); cursor+=size+16
        pl['banks'].append(bank)
    pl['words']=cursor
    return pl


def view(pl,layer,mode):
    result=copy.deepcopy(pl); off=result['offsets']; bank=pl['banks'][layer]
    off['input']=bank['qkv'][0]-256*p.QWORDS
    off['projection_weight']=bank['projection'][0]; off['mlp_wi_weight']=bank['wi'][0]; off['mlp_wo_weight']=bank['wo'][0]
    off['affine']=bank['affine'][0]; off['mlp_affine']=bank['next_affine'][0]; off['coeff']=pl['coefficients'][mode][0]
    if layer: off['norm_input']=pl['offsets']['mlp_norm']
    return result


def bindings(pl,layer,mode):
    return dict(pl['banks'][layer],coeff=pl['coefficients'][mode])


def relocate(data,source,target):
    ranges=sorted((start,start+size,name) for name,(start,size) in source.items())
    if any(a[1]>z[0] for a,z in zip(ranges,ranges[1:])): raise ValueError('overlapping constant banks')
    if source.keys()!=target.keys() or any(source[name][1]!=target[name][1] for name in source): raise ValueError('constant bank geometry')
    output=bytearray(data); changes=[]; used=set()
    for op in txn.parse(data):
        if op['op']!='DDR_PATCH' or op['arg_idx']!=0: continue
        if op['arg_offset']%4: raise ValueError('unaligned constant read')
        word=op['arg_offset']//4
        for first,last,name in ranges:
            if first<=word<last:
                patched=(target[name][0]+word-first)*4
                struct.pack_into('<I',output,op['offset']+40,patched)
                changes.append(dict(field=name,value_offset=op['offset']+40,before=op['arg_offset'],after=patched)); used.add(name)
                break
    return bytes(output),changes,used
