#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepared exact complete-model gate for the independent N-window MLP image."""
import argparse
import json
from pathlib import Path

import optimized_probe as op
import window_mlp as wm
hp=op.hp; p=op.p


class Session(hp.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,window_folder):
        folder=Path(window_folder); meta=json.loads((folder/'build.json').read_text())
        expected=dict(m=128,width=3,qwords=1040,kvwords=2064,q_depth=2,kv_depth=2,l2_q_depth=2,stack_bytes=3328)
        if meta['sources']!=wm.sources() or meta['base_identity']!=bundle.identity or meta['policy']!=expected: raise ValueError('stale window MLP image')
        for name,sha in meta['unchanged_objects'].items():
            if p.attention.sha(bundle.folders['vector']/(name+'.o'))!=sha: raise ValueError('window GEMM arithmetic object changed')
        for name,sha in meta['generated_sources'].items():
            if p.attention.sha(folder/(name+'.cc'))!=sha: raise ValueError('window generated source changed')
        raw=op.artifacts(folder,meta)
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder)
        try: op.Session.install(self,'mlp',folder,meta,raw)
        except BaseException:
            self.close(); raise


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','mlp-folder','attention-folder','mlp-candidate','attention-candidate','compact-folder','head-folder','window-folder','items'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--reps',type=int,default=3); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    fixtures=[json.loads(row) for row in args.items.read_text().splitlines()[:3]]
    if len(fixtures)!=3 or any(row['gold']!=-1 or row['task']!='fixture' for row in fixtures): raise ValueError('unlabelled setup fixtures required')
    print('WINDOW_MLP_RUN '+json.dumps(dict(runner_sha256=p.attention.sha(Path(__file__)),
      window_build_sha256=p.attention.sha(args.window_folder/'build.json'),
      runtime_source_sha256=p.attention.sha(Path(op.__file__)),
      scope='prepared exact fixture/boundary gate; no labelled scoring or speed claim')),flush=True)
    factory=lambda bundle,mlp,attention:Session(bundle,mlp,attention,args.mlp_candidate,args.attention_candidate,args.compact_folder,
      fixtures,args.head_folder,args.window_folder)
    variants=dict(baseline=dict(baseline=True),short_encoder_original=dict(candidate=False),
      window_full_heads=dict(head_candidate=False),short_encoder={})
    op.bp.run(args.base,args.model,args.mlp_folder,args.attention_folder,args.items,args.reps,args.quick,session_factory=factory,variants=variants)
