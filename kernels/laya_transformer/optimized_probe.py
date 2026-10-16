#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serialized combinations of independently versioned Laya research candidates."""
import argparse
import gc
import json
from pathlib import Path

import bucket_head_probe as hp
import fast_buffered_probe as fp
import geglu_lut_probe as lp
import compact_pack as cp
import compact_pack_wide as cw
bp=hp.bp; n=hp.n; p=hp.p


def artifacts(folder,meta):
    for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin'),('kernel.o','kernel_object_sha256')]:
        if p.attention.sha(folder/file)!=meta[key]: raise ValueError('optimized artifact changed')
    return (folder/'insts.bin').read_bytes()


class Session(hp.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
      fast_folder=None,pack_folder=None,mlp_kind='polynomial'):
        candidates=[]
        if fast_folder is not None:
            folder=Path(fast_folder); meta=json.loads((folder/'build.json').read_text())
            validator=fp.Session if mlp_kind=='polynomial' else lp.Session
            validator.validate_candidate(None,meta,bundle)
            for name,sha in meta['unchanged_objects'].items():
                if p.attention.sha(bundle.folders['vector']/(name+'.o'))!=sha: raise ValueError('optimized MLP changes another helper')
            candidates.append(('mlp',folder,meta,artifacts(folder,meta)))
        if pack_folder is not None:
            folder=Path(pack_folder); meta=json.loads((folder/'build.json').read_text())
            if meta['sources']!=cw.sources() or meta['base_identity']!=bundle.identity or meta['policy']['m']!=128 or meta['policy']['kvwords']!=2112: raise ValueError('stale optimized wide-pack candidate')
            if meta['original_kernel_object_sha256']!=p.attention.sha(bundle.folders['chain']/'repeat/kernel.o'): raise ValueError('optimized pack base changed')
            candidates.append(('attention',folder,meta,artifacts(folder,meta)))
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder)
        try:
            for family,folder,meta,raw in candidates: self.install(family,folder,meta,raw)
        except BaseException:
            self.close(); raise

    def install(self,family,folder,meta,raw):
        attention=family=='attention'
        cached='candidate_attention' if attention else 'buffered_mlp'
        old_state=cached; new_state='candidate_attention' if attention else 'fast_mlp'
        if any(inst.family==old_state for inst in self.instructions.values()): raise ValueError('candidate context already submitted/selected')
        for inst in getattr(self,cached).values(): inst.close()
        setattr(self,cached,{})
        if attention: self.candidate_mask_fields={}
        state=self.states.pop(old_state); state['kernel']=None; state['ctx']=None; state['xb']=None
        gc.collect()
        xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
        ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
        groups=[kernel.group_id(3),kernel.group_id(4)]
        self.states[new_state]=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
        if any((group&0xffff)!=(self.groups[0]&0xffff) for group in groups): raise ValueError('optimized bank differs')
        wanted={'qkv','projection','affine','coeff'} if attention else {'wi','wo','next_affine'}
        for index,spec in enumerate(self.bundle.meta['chain']['spec']):
            relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
            if used!=wanted: raise ValueError('optimized constant coverage')
            name=f'{family}_{index}_m128'
            if attention:
                relocated,changes=n.c.txn.patch(relocated,meta['audits'],512,spec['mode']=='local')
                self.candidate_mask_fields[name]=[row['value_offset'] for row in changes if row['field']=='valid']
            getattr(self,cached)[name]=n.Instruction(self,new_state,relocated)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--fast-folder',type=Path); ap.add_argument('--pack-folder',type=Path)
    ap.add_argument('--mlp-kind',choices=('polynomial','lookup'),default='polynomial')
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    if args.mlp_kind=='lookup' and args.fast_folder is None: raise ValueError('lookup profile needs its compiled image')
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('unlabelled setup fixtures required')
    modules=(hp,hp.buffered,hp.buffered.apb,hp.buffered.compact,fp,lp,cp,cw,bp)
    print('OPTIMIZED_PROFILE '+json.dumps(dict(mlp_kind=args.mlp_kind,
      runner_sha256=p.attention.sha(Path(__file__)),
      sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__)) for module in modules},
      head_build_sha256=p.attention.sha(args.head_folder/'build.json'),
      fast_build_sha256=p.attention.sha(args.fast_folder/'build.json') if args.fast_folder else None,
      pack_build_sha256=p.attention.sha(args.pack_folder/'build.json') if args.pack_folder else None,
      scope='shorter heads plus selected M128 encoder candidates; no labelled scoring; full frozen gates required before adoption')),flush=True)
    make=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.mlp_candidate,args.attention_candidate,
      args.compact_folder,fixtures,args.head_folder,args.fast_folder,args.pack_folder,args.mlp_kind)
    if args.mlp_kind=='lookup':
        factory=lambda bundle,mlp,attention,*unused:make(bundle,mlp,attention)
        lp.run(args,session_factory=factory)
    else:
        variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),
          new_encoder_full_heads=dict(head_candidate=False),short_heads_only=dict(candidate=False,head_candidate=True),short_encoder={})
        bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=make,variants=variants)
