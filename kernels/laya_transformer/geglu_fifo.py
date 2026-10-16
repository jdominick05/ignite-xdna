#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline combination of unchanged-source GeGLU O3 codegen and deeper FIFOs."""
import argparse
import json
from pathlib import Path
import re
from unittest.mock import patch

import geglu_tune as gt
p=gt.p
BASE_SOURCES=gt.sources
BASE_PREFIX=gt.prefix


def sources():
    return dict(BASE_SOURCES(),**{'kernels/laya_transformer/geglu_fifo.py':p.attention.sha(Path(__file__))})


def prefix():
    text=BASE_PREFIX()
    for family,expected in [('qF',16),('qL2',4),('kvF',4),('kvL2',4)]:
        text,count=re.subn(r'(aie.objectfifo @'+family+r'_\d+\([^\n]*),1 : i32\)',lambda match:match[1]+',2 : i32)',text)
        if count!=expected: raise ValueError('dedicated MLP FIFO coverage: '+family)
    return text


def build(base,model,out,m):
    # Reuse the isolated compiler experiment without modifying its source or
    # any registered buffered-arm file. Patches affect only this offline call.
    with patch.object(gt,'prefix',prefix),patch.object(gt,'sources',sources):
        gt.build(base,model,out,m,'O3')
    meta=json.loads((out/'build.json').read_text())
    meta['policy'].update(q_depth=2,kv_depth=2,l2_q_depth=2)
    meta['scope']='offline combined GeGLU O3 and FIFO candidate; no hardware qualification or speed claim'
    p.attention.dump(out/'build.json',meta)
    print('GEGLU_FIFO_BUILD '+json.dumps(dict(policy=meta['policy'],
      code_bytes=sorted({row['code_bytes'] for row in meta['audits']}),
      stack_bound=max(sum(row['frames'])+256 for row in meta['audits']),scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--m',type=int,choices=(128,256,384,512),default=128)
    args=ap.parse_args(); build(args.base,args.model,args.out,args.m)
