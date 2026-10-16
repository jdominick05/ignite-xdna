#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Combine independently qualified attention/MLP buffering in a complete model."""
import argparse
import json
from pathlib import Path
import numpy as np

import attention_fifo_probe as apb
import mlp_fifo as mf
import compact_attention as compact
bp=apb.bp; n=apb.n; p=apb.p


class Session(apb.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures):
        folder=Path(mlp_candidate); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=mf.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale MLP buffering candidate')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('MLP buffering artifact changed')
        if meta['kernel_object_sha256']!=p.attention.sha(bundle.folders['vector']/'kernel.o'): raise ValueError('buffering changes MLP arithmetic')
        raw=(folder/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder,attention_folder,attention_candidate)
        try:
            # Prove consolidation against actual old-context outputs while CPU
            # neural/oracle calls are forbidden by the enclosing comparison.
            expected=[super(Session,self).run(item['ids'],item['markers'],item['qtype'],candidate=False) for item in fixtures]
            compact.consolidate(self,compact_folder,attention_folder)
            for index,(item,reference) in enumerate(zip(fixtures,expected)):
                observed=super().run(item['ids'],item['markers'],item['qtype'],candidate=False)
                exact={name:bool(np.array_equal(observed[name].view(np.uint32),reference[name].view(np.uint32))) for name in reference}
                print('COMPACT_ATTENTION_CHECK '+json.dumps(dict(fixture=index,exact=exact)),flush=True)
                if not all(exact.values()): raise ValueError('compact head changes original complete outputs')
            if meta['policy']['m']!=self.candidate_m: raise ValueError('buffering row buckets disagree')
            xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['buffered_mlp']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any((g&0xffff)!=(self.groups[0]&0xffff) for g in groups): raise ValueError('MLP buffering bank differs')
            self.original_mlp={}; self.buffered_mlp={}
            for index,spec in enumerate(bundle.meta['chain']['spec']):
                relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                if used!={'wi','wo','next_affine'}: raise ValueError('MLP buffering constant coverage')
                name=f'mlp_{index}_m{self.candidate_m}'; self.original_mlp[name]=self.instructions[name]
                self.buffered_mlp[name]=n.Instruction(self,'buffered_mlp',relocated)
        except BaseException:
            self.close(); raise

    def run(self,*args,baseline=False,mlp_only=False,candidate=True,mlp_candidate=None,attention_candidate=None,**kwargs):
        mlp=candidate if mlp_candidate is None else mlp_candidate
        attention=candidate if attention_candidate is None else attention_candidate
        selected=self.buffered_mlp if mlp and not baseline and not mlp_only else self.original_mlp
        for name,inst in selected.items(): self.instructions[name]=inst
        return super().run(*args,baseline=baseline,mlp_only=mlp_only,candidate=attention,**kwargs)

    def close(self):
        if self.failed:
            super().close(); return
        for field in ('original_mlp','buffered_mlp'):
            for inst in getattr(self,field,{}).values(): inst.close()
            setattr(self,field,{})
        super().close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]; fixture=fixtures[0]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('only unlabelled fixtures')
    for folder in (args.mlp_candidate,args.attention_candidate):
        if json.loads((folder/'build.json').read_text())['policy']['m']!=apb.ba.bm.bucket(len(fixture['ids'])): raise ValueError('candidate timing fixture bucket differs')
    print('BUFFERED_RUN '+json.dumps(dict(runner_sha256=p.attention.sha(Path(__file__)),
      attention_runner_sha256=p.attention.sha(Path(apb.__file__)),comparison_runner_sha256=p.attention.sha(Path(bp.__file__)),
      compact_source_sha256=p.attention.sha(Path(compact.__file__)),compact_build_sha256=p.attention.sha(args.compact_folder/'build.json'),
      mlp_build_sha256=p.attention.sha(args.mlp_candidate/'build.json'),attention_build_sha256=p.attention.sha(args.attention_candidate/'build.json'),
      scope='both encoder buffering candidates in one row bucket; original arithmetic, other buckets and heads retained')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),
      buffered_mlp=dict(attention_candidate=False),buffered_attention=dict(mlp_candidate=False),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
