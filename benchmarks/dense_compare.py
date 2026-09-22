"""Pinned dense-model comparison. Run device modes through research-lowlevel.sh.

Outputs are reference agreement on local unlabeled images, never task accuracy.
Every invocation checks the pinned artifacts and image list before running.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'Ignition' / 'src')]
import cv2
import numpy as np
import onnxruntime as ort

FAMILIES = {
    'bisenetv2': ('models/bisenetv2_fp32_xint8.onnx','models/bisenetv2_fp32.onnx','data/bisenetv2_val','segment'),
    'modnet_cut': ('models/modnet/modnet_cut_xint8_calibfix.onnx','models/modnet/modnet_cut_fp32.onnx','data/modnet_val','matte'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(label, **data):
    print(label + ' ' + json.dumps(data, allow_nan=False), flush=True)


def preprocess(family, image, shape):
    from npu import bisenetv2, modnet
    if len(shape) != 4 or shape[0:2] != [1,3] or shape[2] != shape[3]:
        raise ValueError(f'unsupported image shape {shape}')
    return (bisenetv2 if family == 'bisenetv2' else modnet).preprocess(image, target_size=shape[2])


def postprocess(family, value, shape):
    from npu import bisenetv2, modnet
    return (bisenetv2.postprocess_mask(value, shape) if family == 'bisenetv2'
            else modnet.postprocess_matte(value, shape))


def session(path, optimize=True):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


def diff(a, b):
    d = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(exact=bool(np.array_equal(a,b)), close=bool(np.allclose(a,b,rtol=1e-5,atol=1e-6)),
                max_abs=float(d.max()), mean_abs=float(d.mean()))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('mode', choices=['pin','inputs','baseline','compile','verify','bench'])
    ap.add_argument('--family', choices=FAMILIES, required=True)
    ap.add_argument('--pins', type=Path, required=True)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--backend', choices=['cpu','cpu_current','amd','ignite'], default='cpu')
    ap.add_argument('--container', type=Path)
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--frames', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0, help='Diagnostic subset only; zero is the complete pinned set')
    ap.add_argument('--model', help='repo-relative ONNX to use instead of the family default, so a '
                    'variant (a different quantization recipe) can be pinned, compiled and verified '
                    'without inventing a family for it. It is recorded in the pins, so a later run '
                    'against those pins cannot drift onto a different model.')
    ap.add_argument('--cache-key', help='compile-cache directory for --backend amd. REQUIRED with '
                    '--model: npu.paths.modnet_cache_key picks the key from a filename marker, so '
                    'every "*cut*" variant resolves to one key and a different graph would silently '
                    'reuse another model\'s compiled artifacts instead of being recompiled.')
    args = ap.parse_args()
    model, fp32, valdir, task = FAMILIES[args.family]
    if args.model:
        model = Path(args.model).as_posix()
    if args.backend == 'amd' and args.model and not args.cache_key:
        ap.error('--model with --backend amd needs --cache-key, or the run silently reuses the '
                 'family cache: modnet_cache_key matches on "cut", which every MODNet-Cut variant '
                 'contains, so a different graph would be served another model\'s compile')
    if args.cache_key and args.backend != 'amd':
        ap.error('--cache-key only applies to --backend amd')
    if args.mode == 'pin':
        paths = [model, fp32, f'npu/{"bisenetv2" if args.family == "bisenetv2" else "modnet"}.py']
        if args.family == 'modnet_cut':
            paths += ['models/modnet/modnet_fp32.onnx']
        images = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/valdir).iterdir()
                        if p.suffix.lower() in ('.jpg','.png','.jpeg'))
        if not images:
            raise ValueError('empty validation set')
        pins = dict(family=args.family, model=model, fp32=fp32, task=task, images=images,
                    hashes={p:sha(ROOT/p) for p in paths+images})
        args.pins.parent.mkdir(parents=True,exist_ok=True)
        with args.pins.open('x',encoding='utf-8') as f:
            json.dump(pins,f,indent=2)
        emit('PINS', **pins)
        return
    pins = json.loads(args.pins.read_text())
    assert pins['family'] == args.family and pins['model'] == model
    for p,want in pins['hashes'].items():
        if sha(ROOT/p) != want:
            raise ValueError(f'pinned artifact changed: {p}')
    images = pins['images'][:args.limit or None]
    emit('IDENTITY', family=args.family,pins_sha256=sha(args.pins),model=model,
         cache_key=args.cache_key,model_sha256=sha(ROOT/model),
         numpy=np.__version__,ort=ort.__version__,images=len(images),full_set=not args.limit,
         source_sha256={p:sha(ROOT/p) for p in ['benchmarks/dense_compare.py',
             'src/ignite_xdna/compiler/dense_regions.py','src/ignite_xdna/compiler/graph_ir.py',
             'src/ignite_xdna/compiler/engine_schedule.py','src/ignite_xdna/compiler/engine_compile.py',
             'src/ignite_xdna/compiler/graph_reference.py','src/ignite_xdna/runtime/session.py',
             'src/ignite_xdna/runtime/graph_session.py','src/ignite_xdna/runtime/dense_session.py',
             'Ignition/src/ignition/pipelines/dense.py','Ignition/src/ignition/pipelines/vision.py',
             'Ignition/src/ignition/live.py']})
    args.artifacts.mkdir(parents=True,exist_ok=True)
    if args.mode == 'inputs':
        import onnx
        shape = [d.dim_value for d in onnx.load(ROOT/model).graph.input[0].type.tensor_type.shape.dim]
        for p in images:
            x,_ = preprocess(args.family,cv2.imread(str(ROOT/p)),shape)
            emit('INPUT',image=p,shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.tobytes()).hexdigest())
        return
    if args.mode == 'compile':
        from ignite_xdna.compiler.engine_compile import compile_graph_container
        manifest = compile_graph_container(ROOT/model,args.container,dense_recipe=args.family,task=task,retire_batch=2)
        emit('COMPILED', container_sha256=sha(args.container),manifest=manifest)
        return
    if args.mode == 'verify':
        return verify(args,pins,images)
    cpu = session(ROOT/model, optimize=args.mode != 'baseline')
    shape = cpu.get_inputs()[0].shape
    name = cpu.get_inputs()[0].name
    ref = session(ROOT/fp32,optimize=False) if args.mode == 'baseline' else None
    optimized = session(ROOT/model) if args.mode == 'baseline' else None
    stock = session(ROOT/'models/modnet/modnet_fp32.onnx',optimize=False) if ref and args.family == 'modnet_cut' else None
    native = None
    pipeline = None
    target = cpu
    if args.backend == 'amd':
        from npu.session import build_session, clear_cache
        from npu.paths import BISENETV2_CACHE_KEY, modnet_cache_key
        key = args.cache_key or (BISENETV2_CACHE_KEY if args.family == 'bisenetv2'
                                 else modnet_cache_key(model))
        if args.fresh:
            clear_cache(key)
        target = build_session(ROOT/model,'npu',cache_key=key)
        report = ROOT/key/'vitisai_ep_report.json'
        emit('AMD_REPORT', path=report.relative_to(ROOT).as_posix(),sha256=sha(report),
             report=json.loads(report.read_text()))
    elif args.backend == 'ignite':
        from ignite_xdna.runtime.dense_session import DenseTensorSession
        if args.mode == 'bench':
            from ignition.pipelines.vision import create_pipeline
            _,pipeline = create_pipeline(args.container,task=task)
            native = pipeline.native
        else:
            native = DenseTensorSession(args.container)
        emit('CONTAINER',sha256=sha(args.container),segments=native.segments)
    try:
        if args.mode == 'baseline':
            for i,p in enumerate(images):
                image = cv2.imread(str(ROOT/p))
                x, original = preprocess(args.family,image,shape)
                y_cpu = cpu.run(None,{name:x})[0]
                y = native.run(x)[0] if native else target.run(None,{name:x})[0]
                y_ref = ref.run(None,{ref.get_inputs()[0].name:x})[0]
                y_optimized = optimized.run(None,{name:x})[0]
                saved = args.artifacts/f'{args.family}_{args.backend}_{i:04d}.npy'
                if saved.exists():
                    raise FileExistsError(saved)
                np.save(saved,y)
                for suffix,value in [('oracle',y_cpu),('fp32',y_ref),('optimized',y_optimized)]:
                    path = saved.with_stem(saved.stem+'_'+suffix)
                    if path.exists():
                        raise FileExistsError(path)
                    np.save(path,value)
                metrics = dict(image=p, output_sha256=sha(saved), cpu=diff(y,y_cpu),fp32=diff(y,y_ref))
                metrics['optimized_cpu_vs_oracle'] = diff(y_optimized,y_cpu)
                a,b = postprocess(args.family,y,original),postprocess(args.family,y_ref,original)
                metrics['postprocessed_fp32'] = diff(a,b)
                if args.family == 'bisenetv2':
                    metrics['fp32_pixel_agreement'] = float(np.mean(a == b))
                if stock:
                    st = stock.run(None,{stock.get_inputs()[0].name:x})[0]
                    metrics['cut_fp32_vs_stock_postprocessed'] = diff(b,postprocess(args.family,st,original))
                emit('AGREEMENT',**metrics)
            return
        image = cv2.imread(str(ROOT/images[0]))
        samples, stages = [], []
        for i in range(args.warmup+args.frames):
            if pipeline:
                result = pipeline.predict(image)
                if i >= args.warmup:
                    tm = result.timings_ms
                    samples.append(tm['total_ms'])
                    stages.append(dict(pre_ms=tm['preprocess_ms'],network_ms=tm['backbone_ms'],
                        post_ms=tm['postprocess_ms'],npu_ms=tm['dispatch_ms'],host_ms=tm['host_ms'],
                        transfer_ms=tm['transfer_ms']))
                continue
            t0 = time.perf_counter()
            x,original = preprocess(args.family,image,shape)
            t1 = time.perf_counter()
            if native:
                y,hw = native.run(x)
            else:
                y = target.run(None,{name:x})[0]
                hw = {}
            t2 = time.perf_counter()
            postprocess(args.family,y,original)
            t3 = time.perf_counter()
            if i >= args.warmup:
                samples.append((t3-t0)*1e3)
                stages.append(dict(pre_ms=(t1-t0)*1e3,network_ms=(t2-t1)*1e3,post_ms=(t3-t2)*1e3,**hw))
        emit('BENCHMARK',backend=args.backend,warmup=args.warmup,frames=args.frames,image=images[0],
             mean_ms=float(np.mean(samples)),p95_ms=float(np.percentile(samples,95)),
             stages={k:float(np.mean([s[k] for s in stages])) for k in stages[0]},samples_ms=samples)
        # Check the actual timed output after measurement; no oracle runs inside
        # a timed frame, and an AMD mismatch remains a measured negative result.
        oracle = session(ROOT/model,optimize=False)
        x,_ = preprocess(args.family,image,shape)
        expected = oracle.run(None,{name:x})[0]
        emit('BENCHMARK_OUTPUT',image=images[0],oracle=diff(result.tensor if pipeline else y,expected))
        del oracle
    finally:
        if native:
            native.close()
        if pipeline:
            pipeline.close()
        del target,cpu,ref,stock,optimized
        gc.collect()


def verify(args,pins,images):
    """Region-local oracle checks; silicon checks use every named region output."""
    import onnx
    from ignite_xdna.compiler.dense_regions import lower_dense, boundary_metadata
    from ignite_xdna.compiler import graph_reference as gr
    from ignite_xdna.compiler.graph_ir import HostLayer
    from ignite_xdna.runtime.graph_session import read_boundary,write_boundary
    from ignite_xdna.runtime.dense_session import DenseTensorSession
    ir = lower_dense(ROOT/pins['model'],args.family)
    model = onnx.shape_inference.infer_shapes(onnx.load(ROOT/pins['model']))
    infos = {v.name:v for v in [*model.graph.input,*model.graph.output,*model.graph.value_info]}
    existing = {v.name for v in model.graph.output}
    for n in ir.tensors:
        if n not in existing:
            model.graph.output.append(infos[n])
    oracle = gr.host_session(model.SerializeToString())
    names = [o.name for o in oracle.get_outputs()]
    from ignite_xdna import InferenceSession
    native = InferenceSession.from_file(args.container) if args.backend == 'ignite' else None
    if native is not None and not isinstance(native,DenseTensorSession):
        raise TypeError('public session factory did not select the dense task runtime')
    streams = {}
    if native:
        from ignite_xdna.compiler import engine_schedule as es
        from ignite_xdna.compiler.engine_sequence import split_instruction_stream,program_task_count
        if native.ignite_manifest.get('source_model_sha256') != sha(ROOT/pins['model']):
            raise ValueError('container source model hash does not match pinned model')
        ws = es.plan_workspace(ir,reuse=native.ge.get('workspace_reuse',True))
        if ws.nbytes != native.workspace_bytes:
            raise ValueError('container workspace does not match re-lowered pinned model')
        scheds,packets = es.schedule_graph(ir,ws)
        if native._reader.get_blob_bytes('wpackets.bin') != packets.blob().tobytes():
            raise ValueError('container packets differ from pinned integer lowering')
        for seg in native.segments:
            if seg['kind'] != 'npu':
                continue
            indices = list(range(*seg['layers']))
            counts = [sum(program_task_count(p) for p in scheds[i].programs) for i in indices]
            pieces = split_instruction_stream(native._reader.get_blob_bytes(seg['blob']),counts)
            streams.update(zip(indices,pieces))
    failed = 0
    try:
        for p in images:
            x,_ = preprocess(args.family,cv2.imread(str(ROOT/p)),list(ir.tensors[ir.input].shape))
            values = dict(zip(names,oracle.run(names,{ir.input:x})))
            tensors = {n:v[0] for n,v in values.items()}
            for layer in ir.layers:
                if isinstance(layer,HostLayer):
                    actual = gr.run_named_host(layer,tensors,ir)
                else:
                    inp = ir.tensors[layer.inputs[0].tensor]
                    actual = {layer.output:gr.conv_direct(layer,gr.gather_input(tensors,layer.inputs,inp.height,inp.width))}
                checks = {n:diff(v,values[n][0]) for n,v in actual.items()}
                ok = all(c['exact'] if ir.tensors[n].dtype == 'uint8' else c['close'] for n,c in checks.items())
                failed += not ok
                emit('REGION',image=p,layer=layer.index,name=layer.name,kind=type(layer).__name__,passed=ok,checks=checks)
                if native:
                    inputs = (list(layer.named_inputs.values()) if isinstance(layer,HostLayer)
                              else list(dict.fromkeys(s.tensor for s in layer.inputs)))
                    for n in inputs:
                        write_boundary(native,boundary_metadata(ir.tensors[n]),values[n])
                    if isinstance(layer,HostLayer):
                        step = next(s for s in native._host_steps if s.name == layer.name)
                        step.run()
                    else:
                        bo,count = native.harness.create_instruction_bo_from_bytes(streams[layer.index])
                        _,state = native.harness.dispatch_kernel(bo,count,native.bo_ws,native.bo_wp,timeout_ms=2000)
                        if str(state) != 'ert_cmd_state.ERT_CMD_STATE_COMPLETED':
                            raise RuntimeError(f'layer {layer.index}: {state}')
                        del bo
                    silicon = {n:diff(read_boundary(native,boundary_metadata(ir.tensors[n]))[0],v)
                               for n,v in actual.items()}
                    good = all(c['exact'] if ir.tensors[n].dtype == 'uint8' else c['close'] for n,c in silicon.items())
                    failed += not good
                    emit('SILICON_REGION',image=p,layer=layer.index,passed=good,checks=silicon)
            if native:
                from unittest.mock import patch
                original_run = ort.InferenceSession.run
                calls = []
                def counted_run(session,*args,**kwargs):
                    calls.append(session)
                    return original_run(session,*args,**kwargs)
                with patch.object(ort.InferenceSession,'run',counted_run):
                    y,hw = native.run(x)
                check = diff(y,values[ir.outputs[0][0]])
                declared_calls = len(native._host_steps)
                good = check['close'] and len(calls) == declared_calls
                failed += not good
                emit('SILICON_OUTPUT',image=p,passed=good,check=check,timings=hw,
                     ort_calls=len(calls),declared_host_regions=declared_calls)
        emit('VERDICT',passed=failed == 0,failures=failed,images=len(images),full_set=not args.limit)
        if failed:
            raise SystemExit(1)
    finally:
        if native:
            native.close()


if __name__ == '__main__':
    main()
