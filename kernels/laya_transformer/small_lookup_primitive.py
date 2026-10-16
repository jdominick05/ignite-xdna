#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-column GEMM and paired-query vector gates for one-state lookup/KV2."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import small_query_lookup as candidate
import small_lookup_checks as checks
import window_lookup_primitive as gemm
import window_vector_primitive as vector
p=candidate.p


def sources():
    files=(Path(__file__),Path(checks.__file__),Path(gemm.__file__),Path(vector.__file__),
        Path(checks.locks.__file__),Path(checks.control.__file__),
        p.ROOT/'tools/laya_small_lookup_primitive.py',p.ROOT/'tools/laya_window_primitive.py',p.ROOT/'tools/laya_window_vector_primitive.py')
    return dict(candidate.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file) for file in files})


def events(kind,slot=0):
    rows=vector.events(True) if kind=='vector' else gemm.events(False,slot)
    result=[]
    for mode in ((1,3) if kind=='vector' else (2,)):
        for row in rows:
            if kind=='vector' and row['mode']!=mode:continue
            row=dict(row,dims=list(row['dims']),strides=list(row['strides']),mode=mode,kblocks=1 if kind=='vector' else 2)
            if kind=='gemm' and row['kind']=='q':row['dims'][-1]=1040
            result.append(row)
        result.append(dict(kind='wait',col=0,mode=mode))
    return result


def runtime(kind,slot=0):
    plan=vector.plan() if kind=='vector' else gemm.plan();seq=candidate.sq.bm.b.MlpSequence(plan['words'])
    mode=None
    for row in events(kind,slot):
        if row['kind']=='wait':seq.wait_columns([0]);continue
        if mode!=row['mode']:mode=row['mode'];seq.control(mode,row['kblocks'])
        seq.dma(row['col'],row['kind'],row['offset'],row['dims'],row['strides'],row['output'])
    return seq.finish()


def build(parent,original,out):
    parent=parent.absolute();original=original.absolute();out=out.absolute();frozen=sources()
    small=json.loads((parent/'build.json').read_text());old=json.loads((original/'build.json').read_text())
    if small['sources']!=candidate.sources() or old['sources']!=candidate.reference.lookup.sources():raise ValueError('primitive parents changed')
    if small['lookup_parent_sha256']!=p.attention.sha(original/'build.json'):raise ValueError('primitive lookup parent differs')
    for folder,meta in ((parent,small),(original,old)):
        if p.attention.sha(folder/'kernel.o')!=meta['kernel_object_sha256']:raise ValueError('primitive parent core changed')
    out.mkdir(parents=True,exist_ok=False);layouts={}
    specs=[('original',original,old,'vector',0),('original0',parent,small,'gemm',0),
           ('original1',parent,small,'gemm',1),('window',parent,small,'vector',0)]
    for name,source,meta,kind,slot in specs:
        baseline=name=='original';folder=out/name;folder.mkdir();shutil.copyfile(source/'kernel.o',folder/'kernel.o')
        prefix=candidate.reference.lookup.gf.prefix() if baseline else candidate.prefix()
        body=vector.runtime(False) if baseline else runtime(kind,slot)
        (folder/'harness.mlir').write_text(prefix+body,encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
            '--dump-intermediates',f'--tmpdir={folder/"prj"}'],folder)
        with patch.object(p,'STACK',5376 if baseline else candidate.STACK):
            audits=json.loads(json.dumps(p.audit(folder,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
        cdo={file.name:p.attention.sha(file) for file in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if audits!=meta['audits'] or cdo!=meta['cdo']:raise ValueError('primitive changes full parent core/init')
        raw=(folder/'insts.bin').read_bytes()
        if baseline:
            proof=vector.verify_stream(raw,False,audits)
            ledger=checks.control.verify_vector((folder/'prj/input_with_addresses.mlir').read_text(),False)
        else:
            plan=vector.plan() if kind=='vector' else gemm.plan()
            proof=candidate.verify(raw,events(kind,slot),audits,plan['words'])
            ledger=checks.ledger(folder,kind=='vector')
        relation=None if baseline else checks.relation(out/'original',folder)
        layouts[name]=dict(audits=audits,cdo=cdo,transport=proof,ledger=ledger,relation=relation,
            instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'),
            kernel_sha256=p.attention.sha(folder/'kernel.o'))
        print('SMALL_LOOKUP_PRIMITIVE_LAYOUT '+json.dumps(dict(name=name,transport=proof,relation=relation,ledger=ledger,
            full_parent_core_init_identical=True)),flush=True)
    if sources()!=frozen:raise ValueError('primitive source changed during build')
    result=dict(sources=frozen,plans=dict(gemm=gemm.plan(),vector=vector.plan()),layouts=layouts,
        parents=dict(small=p.attention.sha(parent/'build.json'),original=p.attention.sha(original/'build.json')),
        scope='offline guarded primitive profiles; window profile means paired vector input, one output state with KV2')
    p.attention.dump(out/'build.json',result)
    print('SMALL_LOOKUP_PRIMITIVE_BUILD '+json.dumps(dict(build_sha256=p.attention.sha(out/'build.json'),sources=frozen,scope=result['scope'])),flush=True)


def qualify(folder,kind):
    meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=sources() or meta['plans']!=dict(gemm=gemm.plan(),vector=vector.plan()):raise ValueError('stale small lookup primitives')
    for name,row in meta['layouts'].items():
        path=folder/name
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_sha256')]:
            if p.attention.sha(path/file)!=row[key]:raise ValueError('small lookup primitive artifact changed')
        if {file.name:p.attention.sha(file) for file in (path/'prj').rglob('*_aie_cdo_*.bin')}!=row['cdo']:raise ValueError('small lookup CDO changed')
        if name=='original':
            proof=vector.verify_stream((path/'insts.bin').read_bytes(),False,row['audits'])
            ledger=checks.control.verify_vector((path/'prj/input_with_addresses.mlir').read_text(),False)
        else:
            variant='vector' if name=='window' else 'gemm';slot=0 if name=='window' else int(name[-1])
            proof=candidate.verify((path/'insts.bin').read_bytes(),events(variant,slot),row['audits'],meta['plans'][variant]['words'])
            ledger=checks.ledger(path,variant=='vector')
            if checks.relation(folder/'original',path)!=row['relation']:raise ValueError('small lookup lock proof changed')
        if proof!=row['transport'] or ledger!=row['ledger']:raise ValueError('small lookup transport/core proof changed')
    return dict(meta,plan=meta['plans'][kind])


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('parent','original','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.parent,args.original,args.out)
