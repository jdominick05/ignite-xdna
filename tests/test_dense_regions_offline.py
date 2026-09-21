"""Named host boundaries, branch liveness and dense packing without a device."""
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import pytest

from ignite_xdna.compiler.dense_regions import lower_dense, boundary_metadata
from ignite_xdna.compiler.graph_ir import HostLayer
from ignite_xdna.compiler import engine_schedule as es, graph_reference as gr
from ignite_xdna.compiler.engine_compile import build_manifest, plan_segments, graph_task
from ignite_xdna.runtime.graph_session import HostStep, read_boundary, write_boundary


def fixture():
    initializers = [nh.from_array(np.array(.125, np.float32), 's'),
                    nh.from_array(np.array(128, np.uint8), 'z'),
                    nh.from_array(np.array(0, np.int8), 'wz'),
                    nh.from_array(np.ones((19, 3, 1, 1), np.int8), 'w')]
    nodes = [h.make_node('QuantizeLinear', ['x','s','z'], ['xq'], name='ingress'),
             h.make_node('DequantizeLinear', ['xq','s','z'], ['xd']),
             h.make_node('DequantizeLinear', ['w','s','wz'], ['wd']),
             h.make_node('Conv', ['xd','wd'], ['c'], name='conv', kernel_shape=[1,1]),
             h.make_node('QuantizeLinear', ['c','s','z'], ['cq']),
             h.make_node('DequantizeLinear', ['cq','s','z'], ['cd']),
             h.make_node('AveragePool', ['cd'], ['p'], kernel_shape=[2,2], strides=[2,2]),
             h.make_node('QuantizeLinear', ['p','s','z'], ['pq']),
             h.make_node('DequantizeLinear', ['pq','s','z'], ['pd']),
             h.make_node('Add', ['pd','pd'], ['a']),
             h.make_node('QuantizeLinear', ['a','s','z'], ['aq']),
             h.make_node('DequantizeLinear', ['aq','s','z'], ['out'])]
    model = h.make_model(h.make_graph(nodes, 'dense', [h.make_tensor_value_info('x',1,[1,3,32,32])],
                                    [h.make_tensor_value_info('out',1,[1,19,16,16])], initializers),
                         opset_imports=[h.make_opsetid('',17)])
    model.ir_version = 9
    return model


def test_dense_reference_small_maps_and_named_outputs():
    model = fixture()
    ir = lower_dense(model, 'bisenetv2')
    assert ir.task == 'segment'
    assert ir.tensors['out'].storage == 'host'
    assert 'pq' not in ir.tensors  # internal CPU values never cross a boundary
    x = np.random.default_rng(4).normal(size=(1,3,32,32)).astype(np.float32)
    direct = gr.run_direct(ir, x[0])
    reference = gr.host_session(model.SerializeToString()).run(None, {'x':x})[0]
    np.testing.assert_array_equal(direct['out'][None], reference)
    for reuse in (False, True):
        ws = es.plan_workspace(ir, reuse=reuse)
        a = ws.halo_fill()
        ws.write_tensor(a, ir.input, x[0])
        for layer in ir.layers:
            if isinstance(layer, HostLayer):
                es.emulate_host_layer(ir, ws, layer, a)
            else:
                ws.write_tensor(a, layer.output, direct[layer.output])
        np.testing.assert_array_equal(ws.read_tensor(a, 'out')[:19], reference[0])
        scheds, store = es.schedule_graph(ir, ws)
        manifest = build_manifest(ir, ws, scheds, store, 'fixture', 0, '', '', 0)
        assert manifest['dense_output']['shape'] == [1,19,16,16]
        assert manifest['dense_output']['dtype'] == 'float32'
        assert all(s['boundary_version'] == 2 for s in plan_segments(ir, scheds) if s['kind'] == 'host')


def test_single_spatial_native_output_is_not_classification():
    from ignite_xdna.compiler.graph_ir import lower_yolov8n
    model = onnx.shape_inference.infer_shapes(fixture())
    sub = onnx.utils.Extractor(model).extract_model(['xq'],['cd'])
    ir = lower_yolov8n(sub)
    with pytest.raises(ValueError,match='explicit'):
        graph_task(ir)
    ir.task = 'segment'
    ws = es.plan_workspace(ir)
    schedules,store = es.schedule_graph(ir,ws)
    manifest = build_manifest(ir,ws,schedules,store,'native',0,'','',0)
    assert manifest['dense_output']['dtype'] == 'uint8'
    assert manifest['dense_output']['shape'] == [1,19,32,32]


def test_thin_spatial_output_also_requires_explicit_task():
    model = fixture()
    pool = next(n for n in model.graph.node if n.op_type == 'AveragePool')
    del pool.attribute[:]
    pool.attribute.extend([h.make_attribute('kernel_shape',[32,2]),h.make_attribute('strides',[32,2])])
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 1
    ir = lower_dense(model,'bisenetv2')
    ir.task = ''
    with pytest.raises(ValueError,match='explicit'):
        graph_task(ir)


def test_named_host_runtime_uses_every_input_and_output():
    model = fixture()
    ir = lower_dense(model, 'bisenetv2')
    # Two independently live inputs and outputs, with float host storage.
    model = h.make_model(h.make_graph([
        h.make_node('Add', ['left','right'], ['sum']), h.make_node('Sub', ['left','right'], ['diff'])],
        'join', [h.make_tensor_value_info(n,1,[1,3,32,32]) for n in ('left','right')],
        [h.make_tensor_value_info(n,1,[1,3,32,32]) for n in ('sum','diff')]),
        opset_imports=[h.make_opsetid('',17)])
    model.ir_version = 9
    from dataclasses import replace
    for n in ('left','right','sum','diff'):
        ir.tensors[n] = replace(ir.tensors['x'], name=n, producer=n, storage='workspace')
    ir.input = 'left'
    ir.layers = [HostLayer('join',0,gr.Segment('left',0,1),'sum',model.SerializeToString(),
                          named_inputs={'left':'left','right':'right'}, named_outputs={'sum':'sum','diff':'diff'})]
    # right is produced before the join; use an independent slot for this plumbing fixture.
    ws = es.Workspace({}, 0, 'left')
    for n in ('left','right','sum','diff'):
        p = es.Placement(n, ws.nbytes, 0,32,32,1,1,dtype='float32')
        ws.placements[n] = p
        ws.nbytes += p.nbytes
    ge = {'placements':{n:dict(vars(p),channels=3) for n,p in ws.placements.items()}}
    backing = np.zeros(ws.nbytes, np.uint8)
    bo = SimpleNamespace(sync=lambda *args: None)
    reader = SimpleNamespace(get_blob_bytes=lambda name: model.SerializeToString())
    session = SimpleNamespace(ge=ge, _ws_map=backing, bo_ws=bo, path='fixture', _reader=reader,
        harness=SimpleNamespace(pyxrt=SimpleNamespace(xclBOSyncDirection=SimpleNamespace(
            XCL_BO_SYNC_BO_FROM_DEVICE=0,XCL_BO_SYNC_BO_TO_DEVICE=1))))
    bindings = {n:boundary_metadata(ir.tensors[n]) for n in ('left','right','sum','diff')}
    for n, v in [('left',3.),('right',1.)]:
        write_boundary(session,bindings[n],np.full((1,3,32,32),v,np.float32))
    step = HostStep(session,dict(blob='host.onnx', boundary_version=2,
        input_bindings=[bindings[n] for n in ('left','right')],
        output_bindings=[bindings[n] for n in ('sum','diff')]))
    step.run()
    assert np.all(read_boundary(session,bindings['sum']) == 4)
    assert np.all(read_boundary(session,bindings['diff']) == 2)
    with pytest.raises(ValueError, match='expected'):
        write_boundary(session,bindings['left'],np.zeros((1,3,32,32),np.uint8))


def test_host_only_boundaries_never_sync_and_are_frame_scoped():
    binding = dict(name='x',tensor='x',shape=[1,3,7,9],dtype='float32',storage='host')
    session = SimpleNamespace(_host_values={})  # intentionally no BO or placement
    x = np.ones(binding['shape'],np.float32)
    write_boundary(session,binding,x)
    np.testing.assert_array_equal(read_boundary(session,binding),x)
    session._host_values.clear()
    with pytest.raises(KeyError):
        read_boundary(session,binding)


@pytest.mark.parametrize('mapped',[False,True])
@pytest.mark.parametrize('channels',[3,8,11,16])
def test_quantized_boundary_roundtrip_keeps_halo_and_channel_padding(mapped,channels):
    blocks = (channels+7)//8
    p = dict(base=64,height=7,width=9,halo=1,blocks=blocks,planes=blocks,dtype='uint8')
    size = blocks*9*11*8
    backing = np.full(size+64,128,np.uint8)
    directions = []
    bo = SimpleNamespace(read=lambda n,b:backing[b:b+n].tobytes(),
        write=lambda a,b:backing.__setitem__(slice(b,b+len(a)),a),
        sync=lambda d,n,b:directions.append((d,n,b)))
    session = SimpleNamespace(ge={'placements':{'q':p}},bo_ws=bo,_ws_map=backing if mapped else None,
        harness=SimpleNamespace(pyxrt=SimpleNamespace(xclBOSyncDirection=SimpleNamespace(
            XCL_BO_SYNC_BO_FROM_DEVICE=0,XCL_BO_SYNC_BO_TO_DEVICE=1))))
    b = dict(name='q',tensor='q',shape=[1,channels,7,9],dtype='uint8',zero_point=128)
    q = np.random.default_rng(13).integers(0,256,size=b['shape'],dtype=np.uint8)
    write_boundary(session,b,q)
    result = read_boundary(session,b)
    np.testing.assert_array_equal(result,q)
    planes = backing[64:].reshape(blocks,9,11,8)
    assert np.all(planes[:,0] == 128) and np.all(planes[:,-1] == 128)
    if channels % 8:
        assert np.all(planes[-1,1:-1,1:-1,channels%8:] == 128)
    assert directions == [(1,size,64),(0,size,64)]
    backing[:] = 0  # a later region can now reuse the physical workspace slot
    np.testing.assert_array_equal(result,q)


def test_branch_lifetime_includes_secondary_host_inputs():
    from dataclasses import replace
    from ignite_xdna.compiler.graph_ir import GraphIR, Segment
    base = lower_dense(fixture(),'bisenetv2').tensors['x']
    tensors = {n:replace(base,name=n,storage='workspace') for n in ('x','a','b','c','d')}
    def host(index,inputs,outputs):
        return HostLayer(str(index),index,Segment(inputs[0],0,1),outputs[0],b'',
                         named_inputs={n:n for n in inputs},named_outputs={n:n for n in outputs})
    ir = GraphIR(tensors,[host(0,['x'],['a','b']),host(1,['b'],['c']),host(2,['c','a'],['d'])],
                 'x',[('d','d')])
    ws = es.plan_workspace(ir,reuse=True)
    assert ws.placements['a'].base != ws.placements['c'].base
    assert ws.placements['a'].base != ws.placements['b'].base


def test_segmentation_channel_scan_matches_argmax_with_ties_and_nan():
    from npu.bisenetv2 import postprocess_mask
    x = np.random.default_rng(12).normal(size=(19,11,9)).astype(np.float32)
    x[:,0,0] = 3
    x[4,0,1] = x[8,0,1] = np.nan
    x[0,0,2] = np.nan
    x[1,0,3] = np.inf
    x[:,0,4] = -np.inf
    for value in (x,x[:,:,::-1],x[None]):
        np.testing.assert_array_equal(postprocess_mask(value),np.argmax(np.squeeze(value),axis=0))
