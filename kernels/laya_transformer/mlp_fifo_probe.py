#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serialized complete-model qualification and comparison of MLP FIFO candidates."""
import argparse
import json
from pathlib import Path

import bucket_attention_probe as bp
import mlp_fifo as mf
ba=bp.ba; n=ba.n; p=ba.p


class Session(ba.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,candidate_folder):
        folder=Path(candidate_folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=mf.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale FIFO candidate')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('FIFO candidate artifact changed')
        if meta['kernel_object_sha256']!=p.attention.sha(bundle.folders['vector']/'kernel.o'): raise ValueError('FIFO candidate changes arithmetic object')
        raw=(folder/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder,attention_folder)
        try:
            xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['candidate_mlp']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any((g&0xffff)!=(self.groups[0]&0xffff) for g in groups): raise ValueError('FIFO candidate bank differs')
            self.original_mlp={}; self.candidate_mlp={}; m=meta['policy']['m']
            for index,spec in enumerate(bundle.meta['chain']['spec']):
                relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                if used!={'wi','wo','next_affine'}: raise ValueError('FIFO candidate constant coverage')
                name=f'mlp_{index}_m{m}'; self.original_mlp[name]=self.instructions[name]
                self.candidate_mlp[name]=n.Instruction(self,'candidate_mlp',relocated)
        except BaseException:
            self.close(); raise

    def run(self,*args,baseline=False,mlp_only=False,candidate=True,**kwargs):
        selected=self.candidate_mlp if candidate and not baseline and not mlp_only else self.original_mlp
        for name,inst in selected.items(): self.instructions[name]=inst
        return super().run(*args,baseline=baseline,mlp_only=mlp_only,**kwargs)

    def close(self):
        if self.failed:
            super().close(); return
        for field in ('original_mlp','candidate_mlp'):
            for inst in getattr(self,field,{}).values(): inst.close()
            setattr(self,field,{})
        super().close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','candidate-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixture=json.loads(args.items.read_text().splitlines()[0])
    meta=json.loads((args.candidate_folder/'build.json').read_text())
    if meta['policy']['m']!=ba.bm.bucket(len(fixture['ids'])): raise ValueError('candidate row bucket differs from timing fixture')
    print('MLP_FIFO_RUN '+json.dumps(dict(build_sha256=p.attention.sha(args.candidate_folder/'build.json'),
      runner_sha256=p.attention.sha(Path(__file__)),comparison_runner_sha256=p.attention.sha(Path(bp.__file__)),
      scope='FIFO candidate only for its compiled row bucket; other buckets retain qualified original image')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.candidate_folder)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
