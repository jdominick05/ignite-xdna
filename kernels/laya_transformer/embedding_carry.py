#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seed the first encoder residual from normalized embedding Y, on device."""
import argparse
import json
from pathlib import Path
import shutil

import vector as v
p=v.p


def sources():
    return dict(v.sources(),**{str(Path(__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(__file__))})


def graph_contract(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); nodes=graph.graph.node
    norm=nodes[7]; addition=nodes[93]
    if norm.op_type!='LayerNormalization' or norm.input[1]!='m.encoder.embeddings.norm.weight': raise ValueError('embedding norm graph changed')
    if addition.op_type!='Add' or addition.input[0]!=norm.output[0] or addition.input[1]!=nodes[92].output[0]: raise ValueError('first encoder residual must be normalized embedding')
    return dict(norm_node=7,add_node=93,residual_tensor=norm.output[0],raw_embedding_tensor=nodes[6].output[0],residual_is_normalized=True)


def runtime(pl):
    seq=v.Sequence(pl['words']); seq.control(22); off=pl['offsets']
    for col in range(4):
        start=off['mlp_norm']+col*128*p.OUT
        seq.dma(col,'q',start,[1,1,128,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',start,[1,1,128*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['affine'],[32,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait(); return seq.finish()


def build(vector,plan_build,model,out):
    vm=json.loads((vector/'build.json').read_text()); meta=json.loads((plan_build/'build.json').read_text())
    if vm['sources']!=v.sources(): raise ValueError('stale vector build')
    contract=graph_contract(model); pl=meta['plan']; out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(vector/'kernel.o',out/'kernel.o'); (out/'harness.mlir').write_text(v.prefix()+runtime(pl),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    old=p.STACK
    try:
        p.STACK=5376; audits=p.audit(out,{})
    finally: p.STACK=old
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('carry core/init differs')
    p.attention.dump(out/'build.json',dict(sources=sources(),plan=pl,model_sha256=p.attention.sha(model),contract=contract,
      vector_build_sha256=p.attention.sha(vector/'build.json'),plan_build_sha256=p.attention.sha(plan_build/'build.json'),audits=audits,cdo=cdo,
      xclbin=p.attention.sha(out/'final.xclbin'),instructions=p.attention.sha(out/'insts.bin')))
    print('EMBEDDING_CARRY_BUILD '+json.dumps(dict(contract=contract,same_core_and_init=True)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--vector',type=Path,required=True); ap.add_argument('--plan-build',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); build(args.vector.resolve(),args.plan_build.resolve(),args.model.resolve(),args.out.resolve())
