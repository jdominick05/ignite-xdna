#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build repeated encoder layers with resident constants and two native images."""
import argparse
import json
from pathlib import Path
import shutil

import block as b
import instructions as txn
import layers as l
p=b.p


def sources():
    files=[Path(__file__),Path(l.__file__),Path(txn.__file__),p.ROOT/'src/ignite_xdna/compiler/scheduler.py']
    return dict(b.sources(),**{str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in files})


def build(source,out,model,count,reuse_repeat=None):
    source=source.resolve(); out=out.resolve(); model=model.resolve()
    frozen=sources(); base=json.loads((source/'build.json').read_text())
    if base['sources']!=b.sources() or base['model_sha256']!=p.attention.sha(model): raise ValueError('stale base/model')
    if not base.get('precise_probs') or not base.get('precise_exp'): raise ValueError('precise probability image required')
    _,spec=l.model_spec(model,count); pl=l.plan(count)
    if count<2: raise ValueError('repeat build requires at least two layers')
    out.mkdir(parents=True,exist_ok=False); artifacts={}
    for name,family in [('first','attention'),('mlp','mlp')]:
        folder=out/name; folder.mkdir(); layout=base['families'][family]['global_512']
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(source/family/'global_512'/file)!=layout[key]: raise ValueError('base artifact changed')
            shutil.copyfile(source/family/'global_512'/file,folder/file)
        artifacts[name]=layout
    if p.attention.sha(source/'coefficients.npz')!=base['coefficients_sha256']: raise ValueError('changed coefficients')
    shutil.copyfile(source/'coefficients.npz',out/'coefficients.npz')
    folder=out/'repeat'; folder.mkdir()
    shutil.copyfile(source/'attention/kernel.o',folder/'kernel.o')
    prefix=b.r.prefix().replace('stack_size = 4608 : i32','stack_size = 4864 : i32')
    repeat_view=l.view(pl,1,spec[1]['mode'])
    (folder/'harness.mlir').write_text(prefix+b.r.runtime(repeat_view,False,512,packed_input=True),encoding='utf-8')
    if reuse_repeat is None:
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
    else:
        reuse_repeat=reuse_repeat.resolve()
        if (reuse_repeat/'harness.mlir').read_bytes()!=(folder/'harness.mlir').read_bytes() or (reuse_repeat/'kernel.o').read_bytes()!=(folder/'kernel.o').read_bytes():
            raise ValueError('repeat cache source differs')
        for file in ('final.xclbin','insts.bin'): shutil.copyfile(reuse_repeat/file,folder/file)
        shutil.copytree(reuse_repeat/'prj',folder/'prj')
    before=p.STACK
    try:
        p.STACK=4864; audits=p.audit(folder,{'fp_exp_ab':2048,'fp_exp_cd':2048})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=artifacts['first']['audits'] or cdo!=artifacts['first']['cdo']: raise ValueError('repeated stream changes resident core or initialization image')
    artifacts['repeat']=dict(audits=audits,cdo=cdo,xclbin=p.attention.sha(folder/'final.xclbin'),instructions=p.attention.sha(folder/'insts.bin'))
    stages=[]
    for index,layer in enumerate(spec):
        for family in ('attention','mlp'):
            template='first' if family=='attention' and index==0 else 'repeat' if family=='attention' else 'mlp'
            source_layer=1 if template=='repeat' else 0
            source_mode=spec[source_layer]['mode'] if template=='repeat' else 'global'
            raw=(out/template/'insts.bin').read_bytes()
            relocated,changes,used=l.relocate(raw,l.bindings(pl,source_layer,source_mode),l.bindings(pl,index,layer['mode']))
            expected={'qkv','projection','affine','coeff'} if family=='attention' else {'wi','wo','next_affine'}
            if used!=expected: raise ValueError('constant operand coverage differs')
            name=f'layer_{index:02d}_{family}.bin'; (out/name).write_bytes(relocated)
            stages.append(dict(layer=index,family=family,template=template,instructions=name,sha256=p.attention.sha(out/name),constant_relocations=changes))
    if frozen!=sources(): raise ValueError('source changed during build')
    meta=dict(sources=frozen,base_build_sha256=p.attention.sha(source/'build.json'),model_sha256=p.attention.sha(model),
      coefficients_sha256=p.attention.sha(out/'coefficients.npz'),plan=pl,spec=spec,artifacts=artifacts,stages=stages,
      scope='encoder trunk starting from supplied normalized input/residual; embeddings and head absent')
    p.attention.dump(out/'build.json',meta)
    print('CHAIN_BUILD '+json.dumps(dict(layers=count,modes=[s['mode'] for s in spec],arena_bytes=pl['words']*4,
      code_bytes=sorted({a['code_bytes'] for a in audits}),same_core_and_init=True,stages=len(stages),sources=frozen)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--reuse-repeat',type=Path)
    ap.add_argument('--masks',type=Path); ap.add_argument('--valid',type=int,default=193); ap.add_argument('--seed',type=int,default=71)
    ap.add_argument('--collect-composition-failures',action='store_true'); args=ap.parse_args()
    if args.command=='build': build(args.source,args.out,args.model,args.layers,args.reuse_repeat)
    else:
        from chain_run import run
        run(args.out,args.model,args.masks,args.valid,args.seed,args.collect_composition_failures)
