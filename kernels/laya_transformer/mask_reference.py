# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent exported-mask oracle for arbitrary contiguous valid-key lengths.

These CPU masks serve comparisons only. Device attention creates its own mask
from RTP valid/local controls, never from an uploaded CPU mask tensor.
"""
import numpy as np


def extract(model,valid):
    import onnx
    import onnxruntime as ort
    from onnx import helper,numpy_helper
    if type(valid) is not int or not 0<=valid<=512: raise ValueError('valid-key length')
    graph=onnx.load(str(model),load_external_data=False); by={out:node for node in graph.graph.node for out in node.output}
    initial={t.name:t for t in graph.graph.initializer}; masks=[]; scales=[]
    for node in [n for n in graph.graph.node if n.op_type=='Softmax'][:2]:
        add=by[node.input[0]]
        if add.op_type!='Add': raise ValueError('attention mask Add')
        where=next(by[name] for name in add.input if name in by and by[name].op_type=='Where'); masks.append(where.output[0])
        product=next(by[name] for name in add.input if name in by and by[name].op_type=='MatMul')
        scale=[]
        for name in product.input:
            mul=by[name]
            if mul.op_type!='Mul': raise ValueError('attention scaling')
            tensor=next(initial[x] for x in mul.input if x in initial)
            scale.append(float(numpy_helper.to_array(tensor)))
        scales.append(scale)
    if len(masks)!=2 or not all(abs(a*z-.125)<1e-7 for a,z in scales): raise ValueError('attention families/scales')
    needed=set(); names=set(masks)
    def visit(name):
        if name in by and name not in needed:
            needed.add(name)
            for parent in by[name].input: names.add(parent); visit(parent)
    for name in masks: visit(name)
    nodes=[n for n in graph.graph.node if any(out in needed for out in n.output)]
    inputs=[x for x in graph.graph.input if x.name in names]; tensors=[t for t in graph.graph.initializer if t.name in names]
    if any(t.data_location==onnx.TensorProto.EXTERNAL for t in tensors): raise ValueError('mask oracle unexpectedly needs learned weights')
    tiny=helper.make_model(helper.make_graph(nodes,'mask_reference',inputs,[helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None) for name in masks],tensors),
      opset_imports=graph.opset_import,ir_version=graph.ir_version)
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.add_session_config_entry('session.intra_op.allow_spinning','0')
    session=ort.InferenceSession(tiny.SerializeToString(),options,providers=['CPUExecutionProvider'])
    feeds={}
    for x in session.get_inputs():
        if x.name=='attention_mask': feeds[x.name]=np.array([[1]*valid+[0]*(512-valid)],np.int64)
        elif x.name=='qtype': feeds[x.name]=np.array([0],np.int64)
        else: raise ValueError('unexpected mask input')
    result={}
    for mode,value in zip(('global','local'),session.run(None,feeds)):
        allowed=(np.broadcast_to(value,(1,1,512,512))[0,0]==0).astype(np.uint8)
        allowed[~allowed.any(axis=1)]=2
        result[f'{mode}_{valid}']=allowed
    return result
