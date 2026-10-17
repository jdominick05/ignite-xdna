#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wall attribution of the adopted full512 arm; never an adoption measurement."""
from collections import Counter, defaultdict
import gc
from pathlib import Path
import time

import laya_native_hybrid_compare as c
import laya_polynomial512_full as candidate
from laya_stage_profile import group


def main():
    paths,bundle,fixtures,folder,files=candidate.identity()
    initial_runner=c.sha(Path(__file__))
    c.emit('POLYNOMIAL512_PROFILE_START',dict(runner_sha256=initial_runner,
        identity=c.na.digest(files),scheduled_rows=512,valid_tokens=len(fixtures[0]['ids']),
        expected_gain_ms=0,kill_line='reject on output mismatch, coverage or accounting failure',
        warmups=2,repetitions=3,
        scope='diagnostic stage wall includes dispatch, driver, waits and DMA; not core cycles or adoption timing'))
    if not c.hybrid.idle():
        raise ValueError('device occupied before profile')
    calls=[]
    rows=[]
    with candidate.full.guards(calls):
        with candidate.Session(paths,bundle,fixtures,folder) as session:
            # Compare complete outputs on all three development fixtures before profiling.
            for index,item in enumerate(fixtures):
                reference=session.run(**candidate.full.feed(item),candidate=False)
                output=session.run(**candidate.full.feed(item))
                exact=candidate.full.exact(output,reference)
                c.emit('POLYNOMIAL512_PROFILE_CHECK',dict(fixture=index,exact=exact))
                if not all(exact.values()):
                    raise ValueError('full512 profile candidate differs from qualified control')
            item=fixtures[0]
            golden=None
            for index in range(5):
                start=time.perf_counter()
                output=session.run(**candidate.full.feed(item),profile=index>=2)
                wall=(time.perf_counter()-start)*1000
                if golden is None:
                    golden={key:value.copy() for key,value in output.items()}
                if not all(candidate.full.exact(output,golden).values()):
                    raise ValueError('profile changes complete output bits')
                stats=session.last_stats
                if len(session.states)!=5 or stats['dispatches']!=66 or calls:
                    raise ValueError('profile execution policy')
                if index<2:
                    continue
                stages=stats['stages']
                if Counter(stage['name'] for stage in stages)!=Counter(session.baseline_steps):
                    raise ValueError('profile stage coverage')
                counts=Counter(group(stage['name']) for stage in stages)
                if counts['encoder_mlp']!=28 or counts['encoder_attention']!=28:
                    raise ValueError('encoder coverage')
                totals=defaultdict(float)
                for stage in stages:
                    totals[group(stage['name'])]+=stage['ms']
                accounted=sum(totals.values())+stats['prepare_ms']+stats['readback_ms']
                if accounted>wall+0.01:
                    raise ValueError('profile accounting exceeds external wall')
                row=dict(index=index-2,wall_ms=wall,groups_ms=dict(totals),counts=dict(counts),
                    outside_stage_prepare_read_ms=wall-accounted,stats=stats)
                rows.append(row)
                c.emit('POLYNOMIAL512_PROFILE_SAMPLE',row)
    session=None
    gc.collect()
    if calls or not c.hybrid.idle() or candidate.identity()[-1]!=files or c.sha(Path(__file__))!=initial_runner:
        raise ValueError('profile cleanup/source/CPU gate')
    c.emit('POLYNOMIAL512_PROFILE_RESULT',dict(wall=c.stats([row['wall_ms'] for row in rows]),
        groups={name:c.stats([row['groups_ms'][name] for row in rows]) for name in rows[0]['groups_ms']},
        fixtures_exact=3,contexts=5,dispatches=66,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,registered_files_unchanged=True,adoption_decision=False))


if __name__=='__main__':
    main()
