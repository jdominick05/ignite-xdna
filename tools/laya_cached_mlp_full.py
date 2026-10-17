#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Failed context-residency integration probe; full execution disabled by default."""
import gc
import json
from pathlib import Path

import laya_cached_mlp_layer as layer

c=layer.c
n=layer.n
full=layer.reference.full
op=full.op


class CachedKernel:
    def __init__(self,delegate,cold,first_warm_bo,completed):
        self.delegate=delegate;self.cold=cold;self.first_warm_bo=first_warm_bo
        self.completed=completed;self.started=False;self.pending=False;self.cold_calls=0;self.warm_calls=0

    def __getattr__(self,name):return getattr(self.delegate,name)

    def __call__(self,opcode,bo,length,source,target):
        if self.pending:raise RuntimeError('cached MLP task still pending; no retry')
        cold=not self.started
        if cold:
            if bo is not self.first_warm_bo:raise ValueError('first cached stage must be layer0')
            bo,length=self.cold.bo,self.cold.length
        self.pending=True;handle=self.delegate(opcode,bo,length,source,target);owner=self
        class Wait:
            consumed=False
            def wait(self,timeout):
                if self.consumed:raise RuntimeError('cached MLP completion already consumed')
                state=handle.wait(timeout)
                if state==owner.completed:
                    self.consumed=True;owner.started=True;owner.pending=False
                    if cold:owner.cold_calls+=1
                    else:owner.warm_calls+=1
                return state
        return Wait()


class Session(full.Session):
    def __init__(self,paths,bundle,fixtures,parent,folder,warm):
        self.cached_cold=None;self.cached_kernel=None
        layer.qualify(parent,folder,warm)
        meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
        super().__init__(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
            paths.compact_folder,fixtures,paths.head_folder,fast_folder=None,
            pack_folder=paths.base/'compact_pack_wide_v2_m128',mlp_kind='lookup')
        try:
            op.Session.install(self,'mlp',folder,meta,(warm/'insts.bin').read_bytes())
            raw=(folder/'insts.bin').read_bytes();spec=bundle.meta['chain']['spec'][0]
            raw,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,0,spec['mode']))
            if used!={'wi','wo','next_affine'}:raise ValueError('cold cached MLP binding coverage')
            self.cached_cold=n.Instruction(self,'fast_mlp',raw)
            self.cached_kernel=CachedKernel(self.states['fast_mlp']['kernel'],self.cached_cold,
                self.buffered_mlp['mlp_0_m128'].bo,self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED)
            self.states['fast_mlp']['kernel']=self.cached_kernel
            if len(self.states)!=5:raise ValueError('cached complete model context count')
        except BaseException:self.close();raise

    def run(self,*args,**kwargs):
        before=self.cached_kernel.cold_calls+self.cached_kernel.warm_calls
        output=super().run(*args,**kwargs)
        after=self.cached_kernel.cold_calls+self.cached_kernel.warm_calls
        if after-before not in (0,28) or len(self.states)!=5 or self.last_stats['dispatches']!=66:
            raise ValueError('cached complete model stage/context coverage')
        return output

    def close(self):
        if not getattr(self,'failed',False):
            if self.cached_kernel is not None:
                self.cached_kernel.delegate=self.cached_kernel.cold=self.cached_kernel.first_warm_bo=None
                self.cached_kernel=None
            if self.cached_cold is not None:self.cached_cold.close();self.cached_cold=None
        super().close()


def layer_gate(path):
    get=lambda tag:[json.loads(line.split(' ',1)[1]) for line in path.read_text(encoding='utf-8').splitlines() if line.startswith(tag+' ')]
    rows=get('CACHED_MLP_LAYER_CHECK');verdict=get('CACHED_MLP_LAYER_RESULT');end=get('RESEARCH_PROCESS_RESULT')
    if (len(rows)!=6 or len(verdict)!=1 or len(end)!=1 or end[0]['exit_code']!=0 or end[0]['stop'] is not None or
        verdict[0]['cases']!=6 or not verdict[0]['all_arena_bits_exact'] or verdict[0]['cpu_neural_oracle_calls'] or
        not verdict[0]['idle_after_close'] or not all(row['all_arena_bits_exact'] and row['mismatch_words']==0 for row in rows)):
        raise ValueError('cached complete MLP layer prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,parent,_,files,_=layer.reference.identity()
    folder=paths.base/'cached_mlp_v3_m128';warm=paths.base/'cached_mlp_warm_v1_m128'
    proof=layer.qualify(parent,folder,warm)
    evidence=c.ROOT/'results/laya/cached_mlp_v3_layer_desktop2_20261017.log'
    files=dict(files,cache_build=c.sha(folder/'build.json'),warm_cache_build=c.sha(warm/'build.json'),cached_layer=layer_gate(evidence))
    for module in (layer,layer.checks,layer.checks.c,__import__(__name__)):
        path=Path(module.__file__);files['cache_source/'+str(path.relative_to(c.ROOT)).replace('\\','/')]=c.sha(path)
    return paths,bundle,fixtures,parent,folder,warm,files,proof


def qualify(*,allow_unqualified_context_probe=False):
    if not allow_unqualified_context_probe:
        raise RuntimeError('resident cache state across context switches is unqualified; use the self-contained dispatch image after its layer gate')
    paths,bundle,fixtures,parent,folder,warm,files,proof=identity()
    requests=full.cases(fixtures);references=[];calls=[]
    c.emit('CACHED_MLP_FULL_START',dict(files_digest=c.na.digest(files),proof=proof,cases=len(requests)))
    if not c.hybrid.idle():raise ValueError('device occupied before complete cached model gate')
    with full.guards(calls):
        with layer.reference.comparison.candidate.make(paths,bundle,fixtures,parent,True) as session:
            for row in requests:references.append({key:value.copy() for key,value in session.run(**full.feed(row)).items()})
        session=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('cached full reference cleanup')
        with Session(paths,bundle,fixtures,parent,folder,warm) as session:
            for row,reference in zip(requests,references):
                before=session.cached_kernel.cold_calls+session.cached_kernel.warm_calls
                output=session.run(**full.feed(row));exact=full.exact(output,reference)
                stages=session.cached_kernel.cold_calls+session.cached_kernel.warm_calls-before
                expected=28 if len(row['ids'])<=128 else 0
                c.emit('CACHED_MLP_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=exact,
                    contexts=len(session.states),dispatches=session.last_stats['dispatches'],cached_mlp_stages=stages))
                if not all(exact.values()) or stages!=expected:raise ValueError('complete cached model output/stage gate')
            c.emit('CACHED_MLP_FULL_COUNTS',dict(cold=session.cached_kernel.cold_calls,warm=session.cached_kernel.warm_calls))
            if session.cached_kernel.cold_calls!=1:raise ValueError('cache cold setup must occur exactly once')
        session=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[6]!=files:raise ValueError('cached full source/CPU/cleanup gate')
    c.emit('CACHED_MLP_FULL_RESULT',dict(cases=len(requests),all_output_bits_exact=True,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,scope='complete model development parity; original full frozen accuracy and speed unqualified'))


if __name__=='__main__':qualify()
