#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Link a screened GEMM into the working lookup primitive's unchanged transport."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import gemm_specialize as specialize
import window_lookup_primitive as original
import window_lock_check as locks
import window_core_check as control
p=original.p


def sources():
    files=(Path(__file__),Path(specialize.__file__),Path(locks.__file__),Path(control.__file__),
           p.ROOT/'tools/laya_gemm_specialized_primitive.py',p.ROOT/'tools/laya_window_primitive.py')
    return dict(original.sources(),**{str(file.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(file) for file in files})


def check_layout(folder,reference,audits,slot):
    raw=(folder/'insts.bin').read_bytes()
    if raw!=(reference/'insts.bin').read_bytes():raise ValueError('specialization changed transaction stream')
    old=json.loads((reference.parent/'build.json').read_text())['layouts']['original'+str(slot)]
    if [a['spans'] for a in audits]!=[a['spans'] for a in old['audits']]:
        raise ValueError('specialization changed SRAM buffers')
    cdo={file.name:p.attention.sha(file) for file in (folder/'prj').rglob('*_aie_cdo_*.bin')}
    if len(cdo)!=3:raise ValueError('missing compiled initialization')
    # Program-load CDO changes; initialization and enable records must not.
    for name,digest in cdo.items():
        if not name.endswith('_aie_cdo_elfs.bin') and digest!=old['cdo'][name]:
            raise ValueError('specialization changed non-program initialization')
    a=locks.analyze(reference,False);b=locks.analyze(folder,False)
    for key in ('locks','roles','flows','hardware_locks_checked','dma_descriptors'):
        if a[key]!=b[key]:raise ValueError('specialization changed compiled lock/route graph')
    ledger=control.verify((folder/'prj/input_with_addresses.mlir').read_text(),False)
    proof=original.verify_stream(raw,False,slot,audits)
    return dict(transport=proof,cdo=cdo,core_ledger=ledger,
        physical_spans_identical=True,nonprogram_initialization_identical=True,
        compiled_locks_routes_identical=True,transaction_bytes_identical=True,
        checked_lock_initializations=b['hardware_locks_checked'],checked_dma_bds=b['dma_descriptors'])


def build(parent,reference,screen,out):
    parent=parent.absolute();reference=reference.absolute();screen=screen.absolute()
    frozen=sources();meta=json.loads((parent/'build.json').read_text())
    ref=json.loads((reference/'build.json').read_text());report=json.loads((screen/'screen.json').read_text())
    if meta['sources']!=original.candidate.lookup.sources() or ref['sources']!=original.sources():
        raise ValueError('lookup/primitive sources changed')
    if ref['parents']['original']!=p.attention.sha(parent/'build.json'):
        raise ValueError('primitive parent differs')
    if p.attention.sha(parent/'kernel.o')!=meta['kernel_object_sha256']:
        raise ValueError('lookup parent object changed')
    if any(p.attention.sha(p.ROOT/file)!=sha for file,sha in report['sources'].items()):
        raise ValueError('screen source changed')
    selected=next(row for row in report['objects'] if row['variant']=='fixed_reset')
    for suffix,key in [('cc','source_sha256'),('o','object_sha256')]:
        if p.attention.sha(screen/('fixed_reset.'+suffix))!=selected[key]:raise ValueError('screen artifact changed')
    if (screen/'fixed_reset.cc').read_text()!=specialize.generated((p.ROOT/'kernels/laya_transformer/gemm.cc').read_text(),True):
        raise ValueError('screen generation differs')
    clang=p.attention.PEANO/'bin/clang++.exe'
    if p.attention.sha(clang)!=report['compiler_sha256']:raise ValueError('compiler changed')
    out=out.absolute();out.mkdir(parents=True,exist_ok=False)
    names=('geglu_lut','tables','block','gemm','dispatch','norm')
    # Reconstruct the working object before accepting its loose object files.
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',
        *[parent/(name+'.o') for name in names],'-o',out/'original_kernel.o'],out)
    if p.attention.sha(out/'original_kernel.o')!=meta['kernel_object_sha256']:
        raise ValueError('loose parent objects do not reconstruct working kernel')
    objects={name:p.attention.sha(parent/(name+'.o')) for name in names if name!='gemm'}
    shutil.copyfile(screen/'fixed_reset.o',out/'gemm.o')
    p.attention.command([clang,'--target=aie2-none-unknown-elf','-r','-nostdlib',
        *[out/'gemm.o' if name=='gemm' else parent/(name+'.o') for name in names],'-o',out/'kernel.o'],out)
    if p.attention.command([p.attention.PEANO/'bin/llvm-nm.exe','-u',out/'kernel.o'],out).strip():
        raise ValueError('unresolved specialized helper')
    layouts={}
    for slot in (0,1):
        name='original'+str(slot);folder=out/name;folder.mkdir()
        shutil.copyfile(out/'kernel.o',folder/'kernel.o')
        shutil.copyfile(reference/name/'harness.mlir',folder/'harness.mlir')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
            '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
            '--dump-intermediates',f'--tmpdir={folder/"prj"}'],folder)
        with patch.object(p,'STACK',5376):
            audits=json.loads(json.dumps(p.audit(folder,{'gelu_tail_ab':4096,'gelu_tail_cd':4096})))
        proof=check_layout(folder,reference/name,audits,slot)
        layouts[name]=dict(audits=audits,proof=proof,instructions=p.attention.sha(folder/'insts.bin'),
            xclbin=p.attention.sha(folder/'final.xclbin'))
        print('SPECIALIZED_PRIMITIVE_LAYOUT '+json.dumps(dict(name=name,code_bytes=sorted({a['code_bytes'] for a in audits}),
            conservative_stack_bound=max(sum(a['frames'])+256 for a in audits),reserved_stack_bytes=5376,
            proof=proof)),flush=True)
    if frozen!=sources():raise ValueError('specialized primitive sources changed')
    result=dict(sources=frozen,plan=original.plan(),layouts=layouts,unchanged_objects=objects,
        kernel_sha256=p.attention.sha(out/'kernel.o'),gemm_sha256=selected['object_sha256'],
        parent_sha256=p.attention.sha(parent/'build.json'),reference_sha256=p.attention.sha(reference/'build.json'),
        screen_sha256=p.attention.sha(screen/'screen.json'),
        scope='offline linked primitive; integer fixtures and real-valued layer parity pending, no speed claim')
    p.attention.dump(out/'build.json',result)
    print('SPECIALIZED_PRIMITIVE_BUILD '+json.dumps(dict(build_sha256=p.attention.sha(out/'build.json'),
        sources=frozen,scope=result['scope'])),flush=True)


def qualify(folder,reference):
    meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=sources() or meta['plan']!=original.plan():raise ValueError('stale specialization sources/plan')
    if p.attention.sha(reference/'build.json')!=meta['reference_sha256']:raise ValueError('reference changed')
    for name,row in meta['layouts'].items():
        for filename,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(folder/name/filename)!=row[key]:raise ValueError('specialized artifact changed')
        if p.attention.sha(folder/name/'kernel.o')!=meta['kernel_sha256']:raise ValueError('specialized core changed')
        if check_layout(folder/name,reference/name,row['audits'],int(name[-1]))!=row['proof']:
            raise ValueError('specialized offline proof changed')
    return meta


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('parent','reference','screen','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.parent,args.reference,args.screen,args.out)
