#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepared complete-model qualification of shorter head row schedules."""
import argparse
import json
from pathlib import Path

import buffered_probe as buffered
import bucket_head as bh
bp=buffered.bp; n=buffered.n; p=buffered.p


class Session(buffered.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder):
        folder=Path(head_folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=bh.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale shorter head build')
        if meta['attention_build_sha256']!=p.attention.sha(Path(attention_folder)/'build.json'): raise ValueError('head attention parent differs')
        if set(meta['layouts'])!={'128','256','384','512'}: raise ValueError('missing head buckets')
        am=json.loads((Path(attention_folder)/'build.json').read_text()); raw={}
        for m,layouts in meta['layouts'].items():
            if set(layouts)!={'head_pre','head_attention','head_post','second_pre','second_post'}: raise ValueError('head stage missing')
            raw[int(m)]={}
            for name,layout in layouts.items():
                expected=am['layouts']['512'] if name=='head_attention' else bundle.meta['vector']
                if layout['audits']!=expected['audits'] or layout['cdo']!=expected['cdo']: raise ValueError('short head changes live core/init')
                for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
                    if p.attention.sha(folder/m/name/file)!=layout[key]: raise ValueError('short head artifact changed')
                raw[int(m)][name]=(folder/m/name/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures)
        try:
            self.original_heads={name:self.instructions[name] for name in raw[128]}
            self.original_head_types=list(self.type_patches)
            self.original_head_masks=list(self.original_masks['head_attention'])
            self.short_heads={}; self.short_head_types={}; self.short_head_masks={}
            for m,streams in raw.items():
                self.short_heads[m]={}
                for name,instructions in streams.items():
                    if name=='head_attention':
                        instructions,changes=n.c.txn.patch(instructions,meta['layouts'][str(m)][name]['audits'],512,False)
                        self.short_head_masks[m]=[row['value_offset'] for row in changes if row['field']=='valid']
                    self.short_heads[m][name]=n.Instruction(self,'short_attention' if name=='head_attention' else 'mlp',instructions)
                start=self.plan['offsets']['head_type']*4
                self.short_head_types[m]=[op['offset']+40 for op in n.c.txn.parse(streams['head_pre'])
                  if op['op']=='DDR_PATCH' and op['arg_idx']==0 and op['arg_offset']==start]
                if len(self.short_head_types[m])!=4: raise ValueError('short head type coverage')
        except BaseException:
            self.close(); raise

    def run(self,ids,markers,qtype,mask=None,*,baseline=False,mlp_only=False,candidate=True,head_candidate=None,**kwargs):
        valid=n.request(ids,markers,qtype,mask)[-1]; m=buffered.apb.ba.bm.bucket(valid)
        selected=(candidate if head_candidate is None else head_candidate) and not baseline and not mlp_only
        for name,inst in (self.short_heads[m] if selected else self.original_heads).items(): self.instructions[name]=inst
        self.type_patches=self.short_head_types[m] if selected else self.original_head_types
        self.original_masks['head_attention']=self.short_head_masks[m] if selected else self.original_head_masks
        return super().run(ids,markers,qtype,mask,baseline=baseline,mlp_only=mlp_only,candidate=candidate,**kwargs)

    def close(self):
        if self.failed:
            super().close(); return
        for inst in getattr(self,'original_heads',{}).values(): inst.close()
        for group in getattr(self,'short_heads',{}).values():
            for inst in group.values(): inst.close()
        self.original_heads={}; self.short_heads={}
        super().close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','items'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(item['gold']!=-1 or item['task']!='fixture' for item in fixtures): raise ValueError('unlabelled setup fixtures required')
    modules=(buffered,buffered.apb,buffered.compact,bh,bp)
    print('BUCKET_HEAD_RUN '+json.dumps(dict(runner_sha256=p.attention.sha(Path(__file__)),
      sources={str(Path(module.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(module.__file__)) for module in modules},
      head_build_sha256=p.attention.sha(args.head_folder/'build.json'),
      mlp_candidate_sha256=p.attention.sha(args.mlp_candidate/'build.json'),attention_candidate_sha256=p.attention.sha(args.attention_candidate/'build.json'),
      compact_build_sha256=p.attention.sha(args.compact_folder/'build.json'),
      scope='prepared shorter head schedules, reuse existing vector/short-attention contexts; no hardware gain assumed')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.mlp_candidate,args.attention_candidate,args.compact_folder,fixtures,args.head_folder)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),
      buffered_encoder=dict(head_candidate=False),short_heads=dict(candidate=False,head_candidate=True),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
