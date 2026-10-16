#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Further isolate preparation without changing the qualified transfer sequence."""
import argparse
from pathlib import Path
from unittest.mock import patch

import attention_phases as coarse
PHASES=('pack','qkv','rope','attend','finish')
original_sources=coarse.sources


class PhaseSequence(coarse.PhaseSequence):
    def control(self,mode,*args,**kwargs):
        if mode==0:
            selected='attend'; self.after_attention=True
        elif self.after_attention:
            selected='finish'
        else:
            selected={10:'pack',2:'qkv',4:'rope',5:'rope',6:'rope',7:'rope'}[mode]
        self.active=selected==self.phase
        if self.active:
            coarse.BaseSequence.control(self,mode,*args,**kwargs)


def runtime(plan,m,phase):
    if phase not in PHASES:
        raise ValueError('unknown fine attention phase')
    with patch.object(coarse.ba.r,'Sequence',lambda words:PhaseSequence(words,phase)):
        return coarse.ba.runtime(plan,m)


def sources():
    return dict(original_sources(),**{'kernels/laya_transformer/attention_fine_phases.py':coarse.p.attention.sha(Path(__file__))})


def verify_streams(candidate,out,audits):
    with patch.object(coarse,'PHASES',PHASES):
        return coarse.verify_streams(candidate,out,audits)


def build(base,model,candidate,out):
    # Reuse the same audited compiler path; restore every module attribute afterward.
    with patch.object(coarse,'PHASES',PHASES),patch.object(coarse,'runtime',runtime),patch.object(coarse,'sources',sources):
        coarse.build(base,model,candidate,out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','candidate','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args(); build(args.base,args.model,args.candidate,args.out)
