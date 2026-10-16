#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Use the guarded specialized core with the working full MLP transaction."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import gemm_specialized_primitive as primitive
p=primitive.p


def sources():
    files=(Path(__file__),p.ROOT/'tools/laya_gemm_specialized_layer.py',p.ROOT/'tools/laya_window_layer_probe.py')
    return dict(primitive.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file) for file in files})


def build(parent,guarded,reference,out):
    parent=parent.absolute();guarded=guarded.absolute();reference=reference.absolute();out=out.absolute()
    frozen=sources();old=json.loads((parent/'build.json').read_text());pm=primitive.qualify(guarded,reference)
    if (old['sources']!=primitive.original.candidate.lookup.sources() or
        p.attention.sha(parent/'build.json')!=pm['parent_sha256']):raise ValueError('full MLP parent differs')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(guarded/'kernel.o',out/'kernel.o');shutil.copyfile(parent/'harness.mlir',out/'harness.mlir')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
        '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    with patch.object(p,'STACK',5376):audits=json.loads(json.dumps(p.audit(out,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
    cdo={file.name:p.attention.sha(file) for file in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if audits!=pm['layouts']['original0']['audits'] or cdo!=pm['layouts']['original0']['proof']['cdo']:
        raise ValueError('full MLP changes guarded core/init')
    if (out/'insts.bin').read_bytes()!=(parent/'insts.bin').read_bytes():raise ValueError('full MLP changes working runtime')
    if sources()!=frozen:raise ValueError('full MLP source changed')
    meta=dict(sources=frozen,base_identity=old['base_identity'],policy=old['policy'],audits=audits,cdo=cdo,
        kernel_object_sha256=pm['kernel_sha256'],instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin'),
        parent_sha256=p.attention.sha(parent/'build.json'),primitive_sha256=p.attention.sha(guarded/'build.json'),
        scope='full MLP with identical guarded core/init and working runtime; real-valued parity and speed unqualified')
    p.attention.dump(out/'build.json',meta)
    print('SPECIALIZED_MLP_BUILD '+json.dumps(dict(build_sha256=p.attention.sha(out/'build.json'),
        code_bytes=sorted({a['code_bytes'] for a in audits}),conservative_stack_bound=max(sum(a['frames'])+256 for a in audits),
        guarded_core_init_identical=True,working_runtime_bytes_identical=True,sources=frozen,scope=meta['scope'])),flush=True)


def validate(folder,parent,guarded,reference,bundle):
    meta=json.loads((folder/'build.json').read_text());pm=primitive.qualify(guarded,reference)
    if meta['sources']!=sources() or meta['base_identity']!=bundle.identity:raise ValueError('full MLP source/base changed')
    if meta['parent_sha256']!=p.attention.sha(parent/'build.json') or meta['primitive_sha256']!=p.attention.sha(guarded/'build.json'):
        raise ValueError('full MLP prerequisite changed')
    for file,key in [('kernel.o','kernel_object_sha256'),('insts.bin','instructions'),('final.xclbin','xclbin')]:
        if p.attention.sha(folder/file)!=meta[key]:raise ValueError('full MLP artifact changed')
    if (meta['kernel_object_sha256']!=pm['kernel_sha256'] or meta['audits']!=pm['layouts']['original0']['audits'] or
        meta['cdo']!=pm['layouts']['original0']['proof']['cdo'] or (folder/'insts.bin').read_bytes()!=(parent/'insts.bin').read_bytes()):
        raise ValueError('full MLP changes qualified core/init/transport')
    return meta


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('parent','guarded','reference','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.parent,args.guarded,args.reference,args.out)
