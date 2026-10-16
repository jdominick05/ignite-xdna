#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete inference with current qualified MLP phases in the same five contexts."""
import gc
import json
from pathlib import Path

import laya_small_lookup_phase_layer as layer
comparison=layer.comparison
full=layer.full
c=layer.c
n=layer.n
phases=layer.phases


class Steps(list):
    """Bucket controllers both assign lists and append their selected stage names."""
    def __init__(self,owner,values=()):
        super().__init__();self.owner=owner;self.extend(values)

    def append(self,name):
        selected=self.owner.phase_steps.get(name) if self.owner.phase_enabled else None
        super().extend(selected if selected is not None else (name,))

    def extend(self,values):
        for value in tuple(values):self.append(value)

    def __iadd__(self,values):self.extend(values);return self


class Session(comparison.candidate.Session):
    @property
    def steps(self):return self._steps

    @steps.setter
    def steps(self,values):self._steps=Steps(self,values)

    def __init__(self,paths,bundle,fixtures,parent,folder):
        self.phase_enabled=False;self.phase_steps={}
        meta=json.loads((folder/'build.json').read_text());reference=json.loads((parent/'build.json').read_text())
        if meta['parent_build_sha256']!=c.sha(parent/'build.json') or meta['base_identity']!=bundle.identity:
            raise ValueError('full phase parent identity differs')
        phases.validate(folder,meta,bundle.plan,reference)
        super().__init__(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
            paths.compact_folder,fixtures,paths.head_folder,parent,paths.base/'compact_pack_wide_v2_m128')
        try:
            for index,spec in enumerate(bundle.meta['chain']['spec']):
                name=f'mlp_{index}_m128';selected=[]
                for phase in phases.PHASES:
                    raw=(folder/phase/'insts.bin').read_bytes()
                    raw,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                    if used!={'wi':{'wi'},'geglu':set(),'wo':{'wo'},'norm':{'next_affine'}}[phase]:
                        raise ValueError('full phase constant coverage')
                    key=name+'_'+phase;selected.append(key)
                    self.instructions[key]=n.Instruction(self,'fast_mlp',raw)
                self.phase_steps[name]=tuple(selected)
            if len(self.phase_steps)!=28 or len(self.states)!=5:raise ValueError('phase/context coverage')
            self.phase_enabled=True
        except BaseException:self.close();raise

    def run(self,ids,markers,qtype,mask=None,*,split=True,profile=False):
        valid=n.request(ids,markers,qtype,mask)[-1]
        previous=self.phase_enabled;self.phase_enabled=split
        try:
            output=super().run(ids,markers,qtype,mask,profile=profile)
            expected=150 if split and phases.parent.sq.bm.bucket(valid)==128 else 66
            if self.last_stats['dispatches']!=expected or len(self.states)!=5:raise ValueError('physical phase execution policy')
            return output
        finally:self.phase_enabled=previous


def layer_gate(path):
    lines=path.read_text().splitlines()
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(tag+' ')]
    verdict=get('SMALL_PHASE_LAYER_RESULT');checks=get('SMALL_PHASE_LAYER_CHECK');end=get('RESEARCH_PROCESS_RESULT')
    if (len(verdict)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        verdict[0]['cases']!=6 or len(checks)!=6 or not verdict[0]['all_arena_bits_exact'] or
        verdict[0]['cpu_neural_oracle_calls'] or not verdict[0]['idle_after_close'] or not verdict[0]['registered_files_unchanged'] or
        not all(row['all_arena_bits_exact'] and row['mismatch_words']==0 for row in checks)):
        raise ValueError('complete phase layer prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,parent,folder,files,proof=layer.identity()
    evidence=c.ROOT/'results/laya/small_lookup_phase_layer_v1_desktop2_20261016.log'
    files=dict(files);files['complete_phase_layer_prerequisite']=layer_gate(evidence)
    files['full_phase_source']=c.sha(Path(__file__))
    return paths,bundle,fixtures,parent,folder,files,proof


def qualify():
    paths,bundle,fixtures,parent,folder,files,proof=identity();requests=full.cases(fixtures);references=[];calls=[]
    c.emit('SMALL_PHASE_FULL_START',dict(files_count=len(files),files_digest=c.na.digest(files),proof=proof,cases=len(requests)))
    if not c.hybrid.idle():raise ValueError('device occupied before full phase qualification')
    with full.guards(calls):
        with comparison.candidate.make(paths,bundle,fixtures,parent,True) as session:
            for row in requests:references.append({key:value.copy() for key,value in session.run(**full.feed(row)).items()})
        session=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('full reference cleanup')
        with Session(paths,bundle,fixtures,parent,folder) as session:
            for row,reference in zip(requests,references):
                output=session.run(**full.feed(row));exact=full.exact(output,reference)
                expected=150 if len(row['ids'])<=128 else 66
                c.emit('SMALL_PHASE_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=exact,
                    contexts=len(session.states),dispatches=session.last_stats['dispatches']))
                if not all(exact.values()) or session.last_stats['dispatches']!=expected:raise ValueError('full phase output gate')
            for index,item in enumerate(fixtures):
                output=session.run(**full.feed(item),split=False)
                if not all(full.exact(output,references[index]).values()):raise ValueError('unsplit control changed')
                c.emit('SMALL_PHASE_UNSPLIT_CHECK',dict(fixture=index,dispatches=session.last_stats['dispatches'],exact=True))
        session=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[5]!=files:raise ValueError('full phase CPU/source/cleanup gate')
    c.emit('SMALL_PHASE_FULL_RESULT',dict(cases=len(requests),unsplit_controls=3,all_output_bits_exact=True,
        cpu_neural_oracle_calls=len(calls),idle_after_close=True,registered_files_unchanged=True,
        scope='diagnostic full parity; no phase timing or new serving adoption'))


if __name__=='__main__':qualify()
