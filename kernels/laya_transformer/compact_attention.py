#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recompile head attention for the existing variable-KV encoder context."""
import argparse
import gc
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import bucket_attention as ba
import head_attention as head
n=ba.n; p=ba.p


def sources():
    return dict(ba.sources(),**head.sources(),**{'kernels/laya_transformer/compact_attention.py':p.attention.sha(Path(__file__))})


def runtime(plan):
    original=head.r.Sequence
    class HeadSequence(original):
        def control(self,mode,*args,**kwargs):
            if mode==0: kwargs['kblocks']=16
            return super().control(mode,*args,**kwargs)
    # Offline emitter only. The original full head image hardcodes 16 KV
    # blocks; the variable-count image must receive that count explicitly.
    with patch.object(head.r,'Sequence',HeadSequence): return head.runtime(plan)


def build(base,model,attention_folder,out):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    am=json.loads((attention_folder/'build.json').read_text())
    if am['sources']!=ba.sources() or am['base_identity']!=bundle.identity: raise ValueError('stale short attention')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(bundle.folders['chain']/'repeat/kernel.o',out/'kernel.o')
    (out/'harness.mlir').write_text(ba.prefix()+runtime(bundle.plan),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.attention.PEANO}',
      '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={out/"prj"}'],out)
    before=p.STACK
    try:
        p.STACK=4864; audits=p.audit(out,{'fp_exp_ab':2048,'fp_exp_cd':2048})
    finally: p.STACK=before
    cdo={f.name:p.attention.sha(f) for f in (out/'prj').rglob('*_aie_cdo_*.bin')}
    if json.loads(json.dumps(audits))!=am['layouts']['512']['audits'] or cdo!=am['layouts']['512']['cdo']: raise ValueError('head changes short attention core/init')
    if sources()!=frozen: raise ValueError('compact source changed')
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,
      attention_build_sha256=p.attention.sha(attention_folder/'build.json'),audits=audits,cdo=cdo,
      instructions=p.attention.sha(out/'insts.bin'),xclbin=p.attention.sha(out/'final.xclbin')))
    print('COMPACT_ATTENTION_BUILD '+json.dumps(dict(same_short_core_and_init=True,head_kvblocks=16,scope='offline audit only')),flush=True)


def consolidate(session,folder,attention_folder):
    """Move full encoder/head instructions to the already live short context."""
    folder=Path(folder); meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=sources() or meta['base_identity']!=session.bundle.identity: raise ValueError('stale compact head')
    if meta['attention_build_sha256']!=p.attention.sha(Path(attention_folder)/'build.json'): raise ValueError('compact parent changed')
    if meta['instructions']!=p.attention.sha(folder/'insts.bin'): raise ValueError('compact instructions changed')
    for index in range(28):
        name=f'attention_{index}'; source=name+'_m512'
        raw=bytes(session.instructions[source].raw)
        replacement=n.Instruction(session,'short_attention',raw)
        session.instructions[name].close(); session.instructions[name]=replacement
        session.original_masks[name]=list(session.short_masks[512][source])
    raw,changes=n.c.txn.patch((folder/'insts.bin').read_bytes(),meta['audits'],512,False)
    replacement=n.Instruction(session,'short_attention',raw)
    session.instructions['head_attention'].close(); session.instructions['head_attention']=replacement
    session.original_masks['head_attention']=[row['value_offset'] for row in changes if row['field']=='valid']
    if any(inst.family=='attention' for inst in session.instructions.values()): raise ValueError('old attention instruction remains')
    state=session.states.pop('attention')
    state['kernel']=None; state['ctx']=None; state['xb']=None
    gc.collect()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','attention-folder','out'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args(); build(args.base,args.model,args.attention_folder,args.out)
