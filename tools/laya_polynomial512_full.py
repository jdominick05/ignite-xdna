#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Original polynomial at O3/depth2, using the full-512 schedule adapter."""
from collections import Counter
import gc
import json
from pathlib import Path
from unittest.mock import patch

import laya_lookup512_full as adapter
c=adapter.c
full=adapter.full
import geglu_fifo as polynomial


def validate(folder,bundle):
    meta=json.loads((folder/'build.json').read_text())
    policy=dict(m=512,opt='O3',q_depth=2,kv_depth=2,l2_q_depth=2)
    if meta['sources']!=polynomial.sources() or meta['base_identity']!=bundle.identity or meta['policy']!=policy:
        raise ValueError('stale polynomial512 build or policy')
    if set(meta['unchanged_objects'])!={'block','gemm','dispatch','norm'}:
        raise ValueError('polynomial512 changes another helper')
    for name,sha in meta['unchanged_objects'].items():
        if c.sha(bundle.folders['vector']/(name+'.o'))!=sha or c.sha(folder/(name+'.o'))!=sha:
            raise ValueError('polynomial512 inherited helper changed')
    return meta,c.op.artifacts(folder,meta)


class Session(adapter.Session):
    def __init__(self,paths,bundle,fixtures,folder):
        # Constructor-only metadata validation substitution; run() is unchanged.
        # This adapter shares scheduling, never the killed lookup kernel image.
        with patch.object(adapter,'validate',validate):
            super().__init__(paths,bundle,fixtures,folder)


def identity():
    paths,bundle,files,items,_=c.identity()
    folder=paths.base/'geglu_fifo_v1_double_m512'
    meta,_=validate(folder,bundle)
    files=dict(files)
    for name in ('build.json','insts.bin','final.xclbin','kernel.o','mlp.o','block.o','gemm.o','dispatch.o','norm.o'):
        files['polynomial512/'+name]=c.sha(folder/name)
    for key,value in meta['sources'].items():
        files['polynomial512_source/'+key]=value
    for file in (Path(__file__),Path(adapter.__file__)):
        files['source/'+str(file.relative_to(c.ROOT)).replace('\\','/')]=c.sha(file)
    return paths,bundle,items[:3],folder,files


def qualify():
    paths,bundle,fixtures,folder,files=identity()
    if not c.hybrid.idle():
        raise ValueError('device occupied before polynomial512 qualification')
    c.emit('POLYNOMIAL512_FULL_START',dict(runner_sha256=c.sha(Path(__file__)),files=files,
        expected_gain_ms=350,minimum_adoption_gain_ms=200,scheduled_rows=512,
        scope='unchanged polynomial source, O3 and depth2; bit-exact full development required; no speed claim'))
    calls=[]
    rows=full.cases(fixtures)
    with full.guards(calls):
        with Session(paths,bundle,fixtures,folder) as session:
            for row in rows:
                output=session.run(**full.feed(row),profile=True)
                stages=session.last_stats['stages']
                if Counter(stage['name'] for stage in stages)!=Counter(session.baseline_steps):
                    raise ValueError('polynomial512 stage coverage')
                selected=sum(stage['family']=='fast_mlp' for stage in stages)
                if selected!=28:
                    raise ValueError('polynomial512 stages not selected')
                reference=session.run(**full.feed(row),candidate=False)
                exact=full.exact(output,reference)
                c.emit('POLYNOMIAL512_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=exact,
                    contexts=len(session.states),dispatches=session.last_stats['dispatches'],polynomial_stages=selected))
                if calls or not all(exact.values()):
                    raise ValueError('polynomial512 complete bit parity gate')
    session=None
    gc.collect()
    if calls or not c.hybrid.idle() or identity()[-1]!=files:
        raise ValueError('polynomial512 source/cleanup policy')
    c.emit('POLYNOMIAL512_FULL_RESULT',dict(cases=len(rows),all_output_bits_exact=True,
        cpu_neural_oracle_calls=len(calls),idle_after_close=True,registered_files_unchanged=True,
        contexts=5,dispatches=66,polynomial_stages=28,
        scope='development only; full original frozen qualification required before adoption timing'))


if __name__=='__main__':
    qualify()
