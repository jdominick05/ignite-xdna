#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same-session complete inference with an encoder attention FIFO candidate."""
import argparse
import json
from pathlib import Path

import bucket_attention_probe as bp
import attention_fifo as af
ba=bp.ba; n=ba.n; p=ba.p


class Session(ba.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,candidate_folder):
        folder=Path(candidate_folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=af.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale attention FIFO candidate')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('attention FIFO artifact changed')
        if meta['kernel_object_sha256']!=p.attention.sha(bundle.folders['chain']/'repeat/kernel.o'): raise ValueError('attention FIFO changes arithmetic object')
        raw=(folder/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder,attention_folder)
        try:
            xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['candidate_attention']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any((g&0xffff)!=(self.groups[0]&0xffff) for g in groups): raise ValueError('attention FIFO bank differs')
            self.original_attention={}; self.candidate_attention={}; self.candidate_mask_fields={}
            self.candidate_m=meta['policy']['m']; m=self.candidate_m
            self.original_mask_fields=dict(self.short_masks[m])
            for index,spec in enumerate(bundle.meta['chain']['spec']):
                relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                if used!={'qkv','projection','affine','coeff'}: raise ValueError('attention FIFO constant coverage')
                relocated,changes=n.c.txn.patch(relocated,meta['audits'],512,spec['mode']=='local')
                name=f'attention_{index}_m{m}'; self.original_attention[name]=self.instructions[name]
                self.candidate_attention[name]=n.Instruction(self,'candidate_attention',relocated)
                self.candidate_mask_fields[name]=[row['value_offset'] for row in changes if row['field']=='valid']
        except BaseException:
            self.close(); raise

    def run(self,*args,baseline=False,mlp_only=False,candidate=True,**kwargs):
        selected=candidate and not baseline and not mlp_only
        for name,inst in (self.candidate_attention if selected else self.original_attention).items(): self.instructions[name]=inst
        self.short_masks[self.candidate_m]=self.candidate_mask_fields if selected else self.original_mask_fields
        return super().run(*args,baseline=baseline,mlp_only=mlp_only,**kwargs)

    def close(self):
        if self.failed:
            super().close(); return
        for field in ('original_attention','candidate_attention'):
            for inst in getattr(self,field,{}).values(): inst.close()
            setattr(self,field,{})
        super().close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','candidate-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixture=json.loads(args.items.read_text().splitlines()[0]); meta=json.loads((args.candidate_folder/'build.json').read_text())
    if meta['policy']['m']!=ba.bm.bucket(len(fixture['ids'])): raise ValueError('attention candidate row bucket differs from timing fixture')
    print('ATTENTION_FIFO_RUN '+json.dumps(dict(build_sha256=p.attention.sha(args.candidate_folder/'build.json'),
      runner_sha256=p.attention.sha(Path(__file__)),comparison_runner_sha256=p.attention.sha(Path(bp.__file__)),
      scope='candidate encoder attention only in its row bucket; original MLP, other attention buckets and heads retained')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.candidate_folder)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
