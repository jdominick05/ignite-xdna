#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded layer-first qualification and diagnostics of unchanged attention transport."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import gc
import json
from pathlib import Path
from unittest.mock import patch

import laya_native_hybrid_compare as compare
import attention_phases as phases
op=compare.op; n=op.n; p=compare.p
NativeRun=n.NativeSession.run


class Session(op.Session):
    def __init__(self,*args,phase_folder,**kwargs):
        meta=json.loads((phase_folder/'build.json').read_text())
        candidate=args[4]
        parent=json.loads((candidate/'build.json').read_text())
        if meta['sources']!=phases.sources() or meta['candidate_build_sha256']!=compare.sha(candidate/'build.json'):
            raise ValueError('phase metadata stale')
        if meta['base_identity']!=args[0].identity or meta['policy']!=parent['policy']:
            raise ValueError('phase policy/base differs')
        if phases.verify_streams(candidate,phase_folder,parent['audits'])!=meta['equivalence']:
            raise ValueError('phase equivalence record differs')
        for phase,layout in meta['layouts'].items():
            if layout['audits']!=parent['audits'] or layout['cdo']!=parent['cdo']:
                raise ValueError('phase changes core or initialization')
            for name,key in [('insts.bin','instructions'),('final.xclbin','xclbin')]:
                if compare.sha(phase_folder/phase/name)!=layout[key]:
                    raise ValueError('phase artifact changed')
        super().__init__(*args,**kwargs)
        self.phase_masks={}
        try:
            for phase,layout in meta['layouts'].items():
                raw=(phase_folder/phase/'insts.bin').read_bytes()
                for index,spec in enumerate(self.bundle.meta['chain']['spec']):
                    relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                    expected={'prepare':{'qkv','coeff'},'pack':set(),'qkv':{'qkv'},'rope':{'coeff'},
                        'attend':set(),'finish':{'projection','affine'}}[phase]
                    if used!=expected:
                        raise ValueError('phase constant relocation coverage: '+phase+' '+str(used))
                    relocated,changes=n.c.txn.patch(relocated,layout['audits'],512,spec['mode']=='local')
                    name=f'attention_{index}_m128_{phase}'
                    self.instructions[name]=n.Instruction(self,'candidate_attention',relocated)
                    self.phase_masks[name]=[row['value_offset'] for row in changes if row['field']=='valid']
        except BaseException:
            self.close(); raise

    def run(self,*args,split_layers=(),**kwargs):
        selected={f'attention_{index}_m128' for index in split_layers}
        def dispatch(session,*args,**kwargs):
            session.steps=[child for name in session.steps for child in
                ([name+'_'+phase for phase in phases.PHASES] if name in selected else [name])]
            if selected:
                session.mask_patches=dict(session.mask_patches,**session.phase_masks)
            return NativeRun(session,*args,**kwargs)
        with patch.object(n.NativeSession,'run',dispatch):
            return super().run(*args,**kwargs)


def main():
    global phases
    ap=argparse.ArgumentParser(); ap.add_argument('--phase-folder',type=Path,required=True)
    ap.add_argument('--fine',action='store_true')
    ap.add_argument('--mode',choices=('layer','full'),required=True); args=ap.parse_args()
    if args.fine:
        import attention_fine_phases
        phases=attention_fine_phases
    prereg=compare.ROOT/'results/laya/native_hybrid_compare_v1_prereg_desktop2_20261016.json'
    _,paths,bundle,fixtures=compare.load_registration(prereg)
    compare.emit('ATTENTION_PHASE_PROFILE_IDENTITY',dict(runner_sha256=compare.sha(Path(__file__)),
        build_sha256=compare.sha(args.phase_folder/'build.json'),registration_sha256=compare.sha(prereg),
        mode=args.mode,scope='diagnostic, timing-ineligible; unchanged qualified attention descriptors and initialization'))
    if not compare.hybrid.idle(): raise ValueError('device occupied')
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(True); raise AssertionError('CPU neural/oracle during native diagnostic')
    functions=[(compare.ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
        (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
        (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    common=(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
        paths.compact_folder,fixtures,paths.head_folder)
    with ExitStack() as stack:
        for module,name in functions: stack.enter_context(patch.object(module,name,forbidden))
        with Session(*common,phase_folder=args.phase_folder,fast_folder=paths.fast_folder,mlp_kind='lookup') as session:
            # One layer first; each call remains a complete inference with other layers unchanged.
            representatives={spec['mode']:index for index,spec in reversed(list(enumerate(bundle.meta['chain']['spec'])))}
            selections=[(index,) for index in representatives.values()] if args.mode=='layer' else [tuple(range(28))]
            for selected in selections:
                for index,item in enumerate(fixtures):
                    feed=(item['ids'],item['markers'],item['qtype'])
                    expected=session.run(*feed)
                    observed=session.run(*feed,split_layers=selected)
                    exact={key:bool(compare.np.array_equal(value.view(compare.np.uint32),expected[key].view(compare.np.uint32))) for key,value in observed.items()}
                    compare.emit('ATTENTION_PHASE_CHECK',dict(layers=selected,fixture=index,exact=exact,dispatches=session.last_stats['dispatches']))
                    if not all(exact.values()): raise ValueError('phase split changed complete outputs')
            if args.mode=='full':
                item=fixtures[0]; samples=[]
                for index in range(3):
                    session.run(item['ids'],item['markers'],item['qtype'],split_layers=tuple(range(28)),profile=True)
                    totals=defaultdict(float); counts=defaultdict(int)
                    for stage in session.last_stats['stages']:
                        if stage['name'].startswith('attention_'):
                            phase=stage['name'].rsplit('_',1)[-1]
                            totals[phase]+=stage['ms']; counts[phase]+=1
                    if set(counts)!=set(phases.PHASES) or set(counts.values())!={28}:
                        raise ValueError('incomplete phase coverage')
                    samples.append(dict(totals))
                    compare.emit('ATTENTION_PHASE_SAMPLE',dict(index=index,phase_ms=dict(totals),stats=session.last_stats))
                compare.emit('ATTENTION_PHASE_RESULT',dict(groups={phase:compare.stats([row[phase] for row in samples]) for phase in phases.PHASES}))
            if len(session.states)!=5 or calls: raise ValueError('execution policy changed')
    session=None; gc.collect()
    if calls or not compare.hybrid.idle(): raise ValueError('cleanup or CPU neural policy failed')
    compare.load_registration(prereg)
    compare.emit('ATTENTION_PHASE_IDLE',dict(idle=True,cpu_neural_oracle_calls=len(calls),contexts=5,registered_artifacts_unchanged=True))


if __name__=='__main__':
    main()
