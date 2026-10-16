#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostic wall-time attribution for the frozen lookup serving configuration."""
from collections import defaultdict
from contextlib import ExitStack
import gc
from pathlib import Path
from unittest.mock import patch

import numpy as np
import laya_native_hybrid_compare as compare


def group(name):
    if name.startswith('mlp_'):
        return 'encoder_mlp'
    if name.startswith('attention_'):
        return 'encoder_attention'
    return name


def main():
    prereg=compare.ROOT/'results/laya/native_hybrid_compare_v1_prereg_desktop2_20261016.json'
    _,paths,bundle,fixtures=compare.load_registration(prereg)
    compare.emit('STAGE_PROFILE_IDENTITY',dict(runner_sha256=compare.sha(Path(__file__)),
        registration_sha256=compare.sha(prereg),configuration='qualified lookup/short heads',
        scope='diagnostic wall times include driver/wait; no core-cycle, bandwidth or speedup claim',
        repetitions=3,warmups=2))
    if not compare.hybrid.idle():
        raise RuntimeError('device occupied before diagnostic')
    op=compare.op; n=op.n; p=compare.p; calls=[]
    def forbidden(*args,**kwargs):
        calls.append(True)
        raise AssertionError('CPU neural/oracle call during NPU diagnostic')
    functions=[(compare.ort.InferenceSession,'run'),(n.cr,'references'),
        (n.embedded.mask_reference,'extract'),(p.norm,'oracle'),(p.attention,'oracle'),
        (n.action,'reference'),(n.action,'verify'),(n.action.s,'reference'),(n.action.s,'verify'),
        (n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    common=(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,
        paths.attention_candidate,paths.compact_folder,fixtures,paths.head_folder)
    runs=[]
    with ExitStack() as stack:
        for module,method in functions:
            stack.enter_context(patch.object(module,method,forbidden))
        with op.Session(*common,fast_folder=paths.fast_folder,mlp_kind='lookup') as session:
            item=fixtures[0]
            infer=lambda profile=False:session.run(item['ids'],item['markers'],item['qtype'],profile=profile)
            for _ in range(2):
                reference={key:value.copy() for key,value in infer().items()}
            for index in range(3):
                output=infer(True)
                if any(not np.array_equal(value.view(np.uint32),reference[key].view(np.uint32)) for key,value in output.items()):
                    raise ValueError('profiling changed complete outputs')
                stats=session.last_stats
                if stats['dispatches']!=66 or len(session.states)!=5 or calls:
                    raise ValueError('execution policy changed')
                totals=defaultdict(float)
                counts=defaultdict(int)
                for stage in stats['stages']:
                    name=group(stage['name']); totals[name]+=stage['ms']; counts[name]+=1
                if counts['encoder_mlp']!=28 or counts['encoder_attention']!=28:
                    raise ValueError('missing encoder stages')
                row=dict(index=index,stats=stats,group_ms=dict(totals),group_dispatches=dict(counts))
                runs.append(row); compare.emit('STAGE_PROFILE_SAMPLE',row)
    session=None; gc.collect()
    if calls or not compare.hybrid.idle():
        raise ValueError('CPU neural call or context cleanup failure')
    compare.load_registration(prereg)
    compare.emit('STAGE_PROFILE_RESULT',dict(groups={key:compare.stats([row['group_ms'][key] for row in runs])
        for key in runs[0]['group_ms']},cpu_neural_oracle_calls=len(calls),contexts=5,dispatches=66,
        exact_repeats=True,idle_after_close=True,registered_artifacts_unchanged=True))


if __name__=='__main__':
    main()
