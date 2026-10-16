#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind dynamic mask instructions to a qualified persistent block image."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import block as b
import instructions as txn
p=b.p


def sources():
    paths=[Path(__file__),Path(txn.__file__),p.ROOT/'src/ignite_xdna/compiler/scheduler.py']
    return {str(f.relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(f) for f in paths}


def derive(source,reference,out):
    original=json.loads((source/'build.json').read_text()); ref=json.loads((reference/'build.json').read_text())
    if original['sources']!=b.sources(): raise ValueError('stale base sources')
    base=original['families']['attention']['global_512']; audits=base['audits']
    data=(source/'attention/global_512/insts.bin').read_bytes()
    if p.attention.sha(source/'attention/global_512/insts.bin')!=base['instructions']: raise ValueError('changed base instructions')
    if p.attention.sha(source/'attention/global_512/final.xclbin')!=base['xclbin']: raise ValueError('changed base image')
    if p.attention.sha(reference/'global_512/insts.bin')!=base['instructions']: raise ValueError('reference uses a different base instruction stream')
    prepared={}; plans={}; reference_hashes={}
    for name,layout in ref['layouts'].items():
        if name not in {f'{mode}_{valid}' for mode in ('global','local') for valid in (512,193,1,0)}: raise ValueError('unexpected reference mask')
        if p.attention.sha(reference/name/'insts.bin')!=layout['instructions']: raise ValueError('changed reference instructions')
        if txn.registers(layout['audits'])!=txn.registers(audits): raise ValueError('reference control allocations differ')
        local=name.startswith('local_'); valid=int(name.split('_')[1])
        patched,changes=txn.patch(data,audits,valid,local)
        txn.compare_reference(patched,(reference/name/'insts.bin').read_bytes(),audits)
        prepared[name]=patched; plans[name]=changes; reference_hashes[name]=layout['instructions']
    if len(prepared)!=8: raise ValueError('incomplete compiled reference mask set')
    # Exercise every representable valid-key count before emitting any artifact.
    for local in (False,True):
        for valid in range(513): txn.patch(data,audits,valid,local)
    out.mkdir(parents=True,exist_ok=False)
    meta=copy.deepcopy(original); meta['families']['attention']={}
    for name,patched in prepared.items():
        folder=out/'attention'/name; folder.mkdir(parents=True)
        shutil.copyfile(source/'attention/global_512/final.xclbin',folder/'final.xclbin')
        (folder/'insts.bin').write_bytes(patched)
        layout=copy.deepcopy(base); layout.update(mode=name.split('_')[0],valid=int(name.split('_')[1]),instructions=p.attention.sha(folder/'insts.bin'))
        meta['families']['attention'][name]=layout
    folder=out/'mlp/global_512'; folder.mkdir(parents=True)
    for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
        if p.attention.sha(source/'mlp/global_512'/file)!=original['families']['mlp']['global_512'][key]: raise ValueError('changed MLP artifact')
        shutil.copyfile(source/'mlp/global_512'/file,folder/file)
    if p.attention.sha(source/'coefficients.npz')!=meta['coefficients_sha256']: raise ValueError('changed coefficients')
    shutil.copyfile(source/'coefficients.npz',out/'coefficients.npz')
    meta['instruction_derivation']=dict(sources=sources(),base_build_sha256=p.attention.sha(source/'build.json'),
      compiled_reference_build_sha256=p.attention.sha(reference/'build.json'),reference_instructions=reference_hashes,
      changed_fields=plans,reference_semantics='all other operation bytes and live query-header controls match',
      all_valid_counts_offline=1026,scope='instruction derivation only; hardware mask coverage requires a run')
    p.attention.dump(out/'build.json',meta)
    print('DYNAMIC_BUILD '+json.dumps(dict(base_instructions=base['instructions'],layouts=list(prepared),
      all_valid_counts_offline=1026,changes_per_layout=32,derivation=meta['instruction_derivation'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('derive','run')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path); ap.add_argument('--reference',type=Path); ap.add_argument('--model',type=Path)
    ap.add_argument('--masks',type=Path); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--collect-composition-failures',action='store_true')
    args=ap.parse_args()
    if args.command=='derive': derive(args.source,args.reference,args.out)
    else:
        meta=json.loads((args.out/'build.json').read_text())
        if meta['instruction_derivation']['sources']!=sources(): raise ValueError('stale instruction patcher')
        b.run(args.out,args.masks,args.model,args.smoke,args.collect_composition_failures)
