#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepared complete-model comparison for GeGLU O3 plus encoder buffering."""
import argparse
import gc
import json
from pathlib import Path

import buffered_probe as buffered
import geglu_fifo as gf
bp=buffered.bp; n=buffered.n; p=buffered.p


class Session(buffered.Session):
    def validate_candidate(self,meta,bundle):
        if meta['sources']!=gf.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale combined GeGLU build')
        if meta['policy']!=dict(m=128,opt='O3',q_depth=2,kv_depth=2,l2_q_depth=2): raise ValueError('unexpected fast MLP policy')

    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,fast_folder):
        folder=Path(fast_folder); meta=json.loads((folder/'build.json').read_text())
        self.validate_candidate(meta,bundle)
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('fast MLP artifact changed')
        for name,sha in meta['unchanged_objects'].items():
            if p.attention.sha(bundle.folders['vector']/(name+'.o'))!=sha: raise ValueError('fast MLP changes another helper')
        raw=(folder/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures)
        try:
            if any(inst.family=='buffered_mlp' for inst in self.instructions.values()): raise ValueError('old buffered instructions already selected')
            for inst in self.buffered_mlp.values(): inst.close()
            self.buffered_mlp={}
            state=self.states.pop('buffered_mlp'); state['kernel']=None; state['ctx']=None; state['xb']=None
            gc.collect()
            xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['fast_mlp']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any((group&0xffff)!=(self.groups[0]&0xffff) for group in groups): raise ValueError('fast MLP bank differs')
            for index,spec in enumerate(bundle.meta['chain']['spec']):
                relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                if used!={'wi','wo','next_affine'}: raise ValueError('fast MLP constant coverage')
                self.buffered_mlp[f'mlp_{index}_m128']=n.Instruction(self,'fast_mlp',relocated)
        except BaseException:
            self.close(); raise


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','fast-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('unlabelled setup fixtures required')
    if any(buffered.apb.ba.bm.bucket(len(item['ids']))!=128 for item in fixtures): raise ValueError('compiled fast bucket is M128')
    na_sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__))
      for module in (buffered,buffered.apb,buffered.compact,bp,gf)}
    print('FAST_BUFFERED_RUN '+json.dumps(dict(runner_sha256=p.attention.sha(Path(__file__)),sources=na_sources,
      fast_build_sha256=p.attention.sha(args.fast_folder/'build.json'),
      mlp_build_sha256=p.attention.sha(args.mlp_candidate/'build.json'),attention_build_sha256=p.attention.sha(args.attention_candidate/'build.json'),
      compact_build_sha256=p.attention.sha(args.compact_folder/'build.json'),
      scope='prepared combination; M128 only; replaces the earlier buffered MLP context before dispatch; no comparative gain over the earlier buffered image assumed')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures,args.fast_folder)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),
      fast_mlp=dict(attention_candidate=False),buffered_attention=dict(mlp_candidate=False),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
