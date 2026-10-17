#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split the original 512-row MLP at existing waits, for wall attribution only."""
from collections import Counter
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import laya_matched512_profile as diagnostic
c = diagnostic.c
import mlp_phases as phases


def install(session, folder):
    meta = json.loads((folder/'build.json').read_text())
    if meta['sources']!=phases.sources() or meta['base_identity']!=session.bundle.identity or meta['m']!=512:
        raise ValueError('stale or mismatched phase image')
    if set(meta['layouts'])!=set(phases.PHASES):
        raise ValueError('missing phases')
    n = c.op.n
    for phase, layout in meta['layouts'].items():
        if layout['audits']!=session.bundle.meta['vector']['audits'] or layout['cdo']!=session.bundle.meta['vector']['cdo']:
            raise ValueError('phase changes core/init image')
        for file,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
            if c.sha(folder/phase/file)!=layout[key]:
                raise ValueError('phase artifact changed')
        raw = (folder/phase/'insts.bin').read_bytes()
        for i,spec in enumerate(session.bundle.meta['chain']['spec']):
            relocated,_,used = n.c.l.relocate(raw,n.c.l.bindings(session.plan,0,'global'),
                                             n.c.l.bindings(session.plan,i,spec['mode']))
            if used!={'wi':{'wi'},'geglu':set(),'wo':{'wo'},'norm':{'next_affine'}}[phase]:
                raise ValueError('phase relocation coverage')
            session.instructions[f'mlp_{i}_{phase}'] = n.Instruction(session,'mlp',relocated)


def run_split(session, item, original_steps, profile):
    session.steps = [part for name in original_steps for part in
                    ([name+'_'+phase for phase in phases.PHASES] if name.startswith('mlp_') else [name])]
    if len(session.steps)!=150:
        raise ValueError('split dispatch coverage')
    return c.op.n.NativeSession.run(session,item['ids'],item['markers'],item['qtype'],profile=profile)


def main():
    prereg = c.ROOT/'results/laya/native_hybrid_compare_v1_prereg_desktop2_20261016.json'
    _,paths,bundle,fixtures = c.load_registration(prereg)
    folder = paths.base/'mlp_phases_v1_m512'
    initial_build = c.sha(folder/'build.json')
    c.emit('MATCHED512_MLP_START',dict(runner_sha256=c.sha(Path(__file__)),build_sha256=initial_build,
        registration_sha256=c.sha(prereg),scheduled_rows=512,expected_gain_ms=0,
        kill_line='reject on any complete-output mismatch or missing phase',
        scope='diagnostic only; four phases add84 dispatches; no adoption or core-cycle claim'))
    if not c.hybrid.idle():
        raise ValueError('device occupied before diagnostic')
    calls = []
    def forbidden(*args,**kwargs):
        calls.append(True)
        raise AssertionError('CPU neural/oracle call')
    n = c.op.n
    functions = [(c.ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
        (c.p.norm,'oracle'),(c.p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
        (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    common = (bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
              paths.compact_folder,fixtures,paths.head_folder)
    rows = []
    with ExitStack() as stack:
        for module,method in functions:
            stack.enter_context(patch.object(module,method,forbidden))
        with c.op.hp.Session(*common) as session:
            install(session,folder)
            for i,item in enumerate(fixtures):
                original = session.run(item['ids'],item['markers'],item['qtype'],baseline=True)
                original_steps = session.steps.copy()
                observed = run_split(session,item,original_steps,False)
                diagnostic.exact(observed,original)
                c.emit('MATCHED512_MLP_CHECK',dict(fixture=i,exact=True,contexts=len(session.states),
                    dispatches=session.last_stats['dispatches']))
            item = fixtures[0]
            for i in range(3):
                result = {}
                for split in ([False,True] if i%2==0 else [True,False]):
                    start = time.perf_counter()
                    if split:
                        output = run_split(session,item,original_steps,True)
                    else:
                        output = session.run(item['ids'],item['markers'],item['qtype'],baseline=True,profile=True)
                    wall = (time.perf_counter()-start)*1000
                    if not split:
                        reference = {key:value.copy() for key,value in output.items()}
                    else:
                        # The complete first fixture already established an unsplit reference below.
                        diagnostic.exact(output,reference)
                    if calls or len(session.states)!=5:
                        raise ValueError('phase execution policy')
                    stats = session.last_stats
                    if stats['dispatches']!=(150 if split else 66):
                        raise ValueError('phase dispatch count')
                    result['split' if split else 'unsplit'] = dict(wall_ms=wall,stats=stats)
                    if split:
                        expected = Counter(f'mlp_{layer}_{phase}' for layer in range(28) for phase in phases.PHASES)
                        actual = Counter(row['name'] for row in stats['stages'] if row['name'].startswith('mlp_'))
                        if actual!=expected:
                            raise ValueError('phase profile coverage')
                        result['phases_ms'] = {phase:sum(row['ms'] for row in stats['stages']
                            if row['name'].startswith('mlp_') and row['name'].endswith('_'+phase)) for phase in phases.PHASES}
                rows.append(result)
                c.emit('MATCHED512_MLP_SAMPLE',dict(index=i,**result))
    session = None
    gc.collect()
    if calls or not c.hybrid.idle() or c.sha(folder/'build.json')!=initial_build:
        raise ValueError('cleanup or source identity gate')
    c.load_registration(prereg)
    c.emit('MATCHED512_MLP_RESULT',dict(phases_ms={phase:c.stats([row['phases_ms'][phase] for row in rows])
        for phase in phases.PHASES},complete={arm:c.stats([row[arm]['wall_ms'] for row in rows])
        for arm in ('unsplit','split')},fixtures_exact=3,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,registered_artifacts_unchanged=True,adoption_decision=False))


if __name__=='__main__':
    main()
