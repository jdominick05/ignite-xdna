#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same-session complete inference with only encoder MLP GEMM codegen changed."""
import argparse
import json
from pathlib import Path

import bucket_attention_probe as bp
import gemm_tune as gt
ba=bp.ba; n=ba.n; p=ba.p


class Session(ba.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,tuned_folder):
        folder=Path(tuned_folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=gt.sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale tuned GEMM build')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if p.attention.sha(folder/file)!=meta[key]: raise ValueError('tuned GEMM artifact changed')
        if meta['instructions']!=bundle.meta['vector']['instructions']: raise ValueError('tuned GEMM runtime ABI differs')
        if [a['spans'] for a in meta['audits']]!=[a['spans'] for a in bundle.meta['vector']['audits']]: raise ValueError('tuned GEMM buffer ABI differs')
        super().__init__(bundle,mlp_folder,attention_folder)
        try:
            xb=self.xrt.xclbin(str(folder/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['tuned_mlp']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any((g&0xffff)!=(self.groups[0]&0xffff) for g in groups): raise ValueError('tuned GEMM bank differs')
            self.old_mlp={}; self.tuned_mlp={}
            for m in self.bucket_raw:
                for index in range(28):
                    name=f'mlp_{index}_m{m}'; original=self.instructions[name]
                    self.old_mlp[name]=original; self.tuned_mlp[name]=n.Instruction(self,'tuned_mlp',original.raw)
        except BaseException:
            self.close(); raise

    def run(self,*args,baseline=False,mlp_only=False,tuned=True,**kwargs):
        for name,inst in (self.tuned_mlp if tuned and not baseline and not mlp_only else self.old_mlp).items():
            self.instructions[name]=inst
        return super().run(*args,baseline=baseline,mlp_only=mlp_only,**kwargs)

    def close(self):
        if self.failed:
            super().close(); return
        for field in ('old_mlp','tuned_mlp'):
            for inst in getattr(self,field,{}).values(): inst.close()
            setattr(self,field,{})
        super().close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--mlp-folder',type=Path,required=True); ap.add_argument('--attention-folder',type=Path,required=True)
    ap.add_argument('--tuned-folder',type=Path,required=True); ap.add_argument('--items',type=Path,required=True)
    ap.add_argument('--reps',type=int,default=5); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    print('GEMM_TUNE_RUN '+json.dumps(dict(build_sha256=p.attention.sha(args.tuned_folder/'build.json'),
      runner_sha256=p.attention.sha(Path(__file__)),comparison_runner_sha256=p.attention.sha(Path(bp.__file__)),
      scope='tuned encoder MLP GEMM only; original vector image retained for stem and heads')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.tuned_folder)
    variants=dict(baseline=dict(baseline=True),mlp_only=dict(mlp_only=True),short_encoder_original=dict(tuned=False),short_encoder={})
    bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
