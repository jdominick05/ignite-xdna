#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Original frozen full-set gates for attention depth2 at fixed512 rows."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import polynomial512_accuracy as base
sys.path.insert(0,str(base.p.ROOT/'tools'))
import laya_attention512_full as candidate
c=candidate.c
NAME='native_attention_fifo_full512_bf16_v1'
BASE_DOCUMENT=base.document


def development_gate(path):
    lines=path.read_text(encoding='utf-8').splitlines()
    records=lambda tag:[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(tag+' ')]
    layers=records('ATTENTION512_LAYER_CHECK')
    rows=records('ATTENTION512_FULL_CHECK')
    result=records('ATTENTION512_FULL_RESULT')
    end=records('RESEARCH_PROCESS_RESULT')
    if (len(layers)!=6 or len(rows)!=25 or len(result)!=1 or len(end)!=1 or
        end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        not all(all(row['exact'].values()) for row in layers) or
        not all(all(row['exact'].values()) and row['contexts']==5 and row['dispatches']==66 and
            row['attention_stages']==28 for row in rows) or
        result[0]['cases']!=25 or not result[0]['all_output_bits_exact'] or
        result[0]['cpu_neural_oracle_calls'] or not result[0]['idle_after_close'] or
        not result[0]['registered_files_unchanged']):
        raise ValueError('attention512 complete development gate missing')
    return c.sha(path)


def identity(*unused):
    _,bundle,_,_,_,files=candidate.identity()
    files=dict(files)
    evidence=base.p.ROOT/'results/laya/attention_fifo_m512_v1_full_desktop2_20261017.log'
    files['attention512_full_development']=development_gate(evidence)
    files['source/kernels/laya_transformer/attention512_accuracy.py']=c.sha(Path(__file__))
    # The reused registration/scoring adapter itself is also frozen.
    files['source/kernels/laya_transformer/polynomial512_accuracy.py']=c.sha(Path(base.__file__))
    return bundle,files


def document(original,files,prior):
    arm=BASE_DOCUMENT(original,files,prior)
    arm.update(study='Full512 attention FIFO depth2 addendum; original frozen criteria unchanged',
        tool='kernels/laya_transformer/attention512_accuracy.py',
        frozen='before this full512 attention depth2 arm has labelled frozen outputs',
        arms={NAME:dict(files=files,what='Attention Q/KV/L2 Q depth2 at full512 over adopted polynomial512 MLP')})
    arm['native_policy'].update(attention_stages=28,
        attention='original arithmetic object; Q/KV/L2 Q depth2; full512 at EVERY logical length',
        codegen='adopted polynomial512 MLP unchanged; attention arithmetic object unchanged',
        approximation='no new approximation')
    arm['development_disclosure'].update(
        observed='six representative layer checks, 25 unlabelled complete cases; prior native results known',
        prior_kill='prior lookup512 development and cached Wi speed failures remain distinct negative results')
    for key in ('criteria','criteria_text','items','reference','decision'):
        if arm[key]!=original[key]:
            raise ValueError('attention512 changed original '+key)
    return arm


def main():
    proxy=SimpleNamespace(**vars(base.candidate))
    proxy.Session=lambda paths,bundle,fixtures,folder:candidate.Session(
        paths,bundle,fixtures,folder,paths.base/'attention_fifo_v1_double_m512')
    with patch.object(base,'NAME',NAME),patch.object(base,'identity',identity),\
        patch.object(base,'document',document),patch.object(base,'candidate',proxy):
        base.main()


if __name__=='__main__':
    main()
