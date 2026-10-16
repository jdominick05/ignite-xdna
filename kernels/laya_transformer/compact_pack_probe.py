#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepared complete-model qualification of DMA-gather norm packing."""
import argparse
import gc
import json
from pathlib import Path

import buffered_probe as buffered
import compact_pack as cp
bp=buffered.bp; n=buffered.n; p=buffered.p


class Session(buffered.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,pack_folder):
        folder=Path(pack_folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=cp.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale compact pack build')
        if meta['policy']['m']!=128: raise ValueError('compact pack timing bucket differs')
        if meta['original_kernel_object_sha256']!=p.attention.sha(bundle.folders['chain']/'repeat/kernel.o'): raise ValueError('compact pack base arithmetic differs')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('compact pack artifact changed')
        raw=(folder/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures)
        try:
            if any(inst.family=='candidate_attention' for inst in self.instructions.values()): raise ValueError('old candidate already selected')
            for inst in self.candidate_attention.values(): inst.close()
            self.candidate_attention={}; self.candidate_mask_fields={}
            state=self.states.pop('candidate_attention'); state['kernel']=None; state['ctx']=None; state['xb']=None
            gc.collect()
            xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['candidate_attention']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any((group&0xffff)!=(self.groups[0]&0xffff) for group in groups): raise ValueError('compact pack bank differs')
            for index,spec in enumerate(bundle.meta['chain']['spec']):
                relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                if used!={'qkv','projection','affine','coeff'}: raise ValueError('compact pack constant coverage')
                relocated,changes=n.c.txn.patch(relocated,meta['audits'],512,spec['mode']=='local')
                name=f'attention_{index}_m128'
                self.candidate_attention[name]=n.Instruction(self,'candidate_attention',relocated)
                self.candidate_mask_fields[name]=[row['value_offset'] for row in changes if row['field']=='valid']
        except BaseException:
            self.close(); raise


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','pack-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('unlabelled setup fixtures required')
    modules=(buffered,buffered.apb,buffered.compact,cp,bp)
    print('COMPACT_PACK_RUN '+json.dumps(dict(runner_sha256=p.attention.sha(Path(__file__)),
      sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__)) for module in modules},
      pack_build_sha256=p.attention.sha(args.pack_folder/'build.json'),
      mlp_candidate_sha256=p.attention.sha(args.mlp_candidate/'build.json'),attention_candidate_sha256=p.attention.sha(args.attention_candidate/'build.json'),
      compact_build_sha256=p.attention.sha(args.compact_folder/'build.json'),
      scope='M128 compact norm packing in encoder; original math, larger buckets and heads retained; partial-object Shim ordering needs device verification')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures,args.pack_folder)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),
      buffered_mlp=dict(attention_candidate=False),compact_attention=dict(mlp_candidate=False),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
