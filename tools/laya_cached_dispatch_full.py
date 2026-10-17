#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete Laya with the same self-contained cached MLP stream every stage."""
from collections import Counter
import gc
import json
from pathlib import Path
import sys

import laya_cached_dispatch_layer as layer
c=layer.c
n=layer.n
full=layer.previous.reference.full
op=full.op
dispatch=layer.dispatch


class Session(full.Session):
    def __init__(self,paths,bundle,fixtures,parent,folder):
        dispatch.verify(parent,folder)
        meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
        super().__init__(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
            paths.compact_folder,fixtures,paths.head_folder,fast_folder=None,
            pack_folder=paths.base/'compact_pack_wide_v2_m128',mlp_kind='lookup')
        try:
            op.Session.install(self,'mlp',folder,meta,(folder/'insts.bin').read_bytes())
            if len(self.states)!=5 or len(self.buffered_mlp)!=28:raise ValueError('self-contained model stage/context coverage')
        except BaseException:self.close();raise

    def run(self,*args,**kwargs):
        output=super().run(*args,**kwargs)
        if len(self.states)!=5 or self.last_stats['dispatches']!=66:
            raise ValueError('self-contained full-model execution policy')
        return output


def layer_gate(path):
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in path.read_text(encoding='utf-8').splitlines() if line.startswith(tag+' ')]
    rows=get('CACHED_DISPATCH_LAYER_CHECK');verdict=get('CACHED_DISPATCH_LAYER_RESULT');end=get('RESEARCH_PROCESS_RESULT')
    if (len(rows)!=6 or len(verdict)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        verdict[0]['cases']!=6 or not verdict[0]['all_arena_bits_exact'] or verdict[0]['cpu_neural_oracle_calls'] or
        not verdict[0]['idle_after_close'] or not all(row['all_arena_bits_exact'] and row['mismatch_words']==0 for row in rows)):
        raise ValueError('self-contained complete layer prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,parent,folder,_,proof=layer.identity()
    evidence=c.ROOT/'results/laya/cached_dispatch_layer_v1_desktop2_20261017.log'
    files=dict(cache_build=c.sha(folder/'build.json'),cached_layer=layer_gate(evidence))
    for module in (layer,dispatch,sys.modules[__name__]):
        path=Path(module.__file__);files['source/'+str(path.relative_to(c.ROOT)).replace('\\','/')]=c.sha(path)
    return paths,bundle,fixtures,parent,folder,files,proof


def stage_gate(stats,valid):
    actual=Counter(row['name'] for row in stats['stages'] if row['family']=='fast_mlp')
    expected=Counter(f'mlp_{i}_m128' for i in range(28)) if valid<=128 else Counter()
    if actual!=expected or stats['dispatches']!=66:raise ValueError('self-contained cached stage profile coverage')
    return sum(actual.values())


def qualify():
    paths,bundle,fixtures,parent,folder,files,proof=identity()
    requests=full.cases(fixtures);references=[];calls=[]
    c.emit('CACHED_DISPATCH_FULL_START',dict(files=files,files_digest=c.na.digest(files),proof=proof,cases=len(requests)))
    if not c.hybrid.idle():raise ValueError('device occupied before self-contained full-model gate')
    with full.guards(calls):
        with layer.previous.reference.comparison.candidate.make(paths,bundle,fixtures,parent,True) as session:
            for row in requests:references.append({key:value.copy() for key,value in session.run(**full.feed(row)).items()})
        session=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('self-contained full reference cleanup')
        with Session(paths,bundle,fixtures,parent,folder) as session:
            for row,reference in zip(requests,references):
                output=session.run(**full.feed(row),profile=True);exact=full.exact(output,reference)
                stages=stage_gate(session.last_stats,len(row['ids']))
                c.emit('CACHED_DISPATCH_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=exact,
                    contexts=len(session.states),dispatches=session.last_stats['dispatches'],cached_mlp_stages=stages))
                if not all(exact.values()):raise ValueError('self-contained complete model output gate')
        session=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[5]!=files:raise ValueError('self-contained full source/CPU/cleanup gate')
    c.emit('CACHED_DISPATCH_FULL_RESULT',dict(cases=len(requests),all_output_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,scope='full development parity only; original frozen full-set gates and adoption timing pending'))


if __name__=='__main__':qualify()
