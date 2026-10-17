#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Development gates for lookup GeGLU with the original full-512 schedule."""
from collections import Counter
import gc
import json
from pathlib import Path

import laya_pack_full_probe as full
c=full.c
n=c.op.n
import geglu_lut as lookup


def validate(folder,bundle):
    meta=json.loads((folder/'build.json').read_text())
    policy=dict(m=512,q_depth=2,kv_depth=2,l2_q_depth=2,
        approximation='512 FP32 Gaussian tail-CDF entries at step1/32; Q16 fractional indexing; linear interpolation; asymptote at abs15.9375')
    if meta['sources']!=lookup.sources() or meta['base_identity']!=bundle.identity or meta['policy']!=policy:
        raise ValueError('stale lookup512 build or policy')
    if set(meta['unchanged_objects'])!={'block','gemm','dispatch','norm'}:
        raise ValueError('lookup512 changes another helper')
    for name,sha in meta['unchanged_objects'].items():
        if c.sha(bundle.folders['vector']/(name+'.o'))!=sha or c.sha(folder/(name+'.o'))!=sha:
            raise ValueError('lookup512 inherited helper changed')
    if c.sha(folder/'gelu_tables.cc')!=meta['table_source_sha256']:
        raise ValueError('lookup table changed')
    raw=c.op.artifacts(folder,meta)
    return meta,raw


class Session(c.op.hp.Session):
    def __init__(self,paths,bundle,fixtures,folder):
        meta,raw=validate(folder,bundle)
        super().__init__(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,
                         paths.attention_candidate,paths.compact_folder,fixtures,paths.head_folder)
        try:
            self.full512_original={name:self.instructions[name] for name in self.baseline_steps if name.startswith('mlp_')}
            # Reuse the otherwise unused buffered-MLP slot; keep five contexts.
            c.op.Session.install(self,'mlp',folder,meta,raw)
            self.lookup512={name.removesuffix('_m128'):inst for name,inst in self.buffered_mlp.items()}
            if set(self.lookup512)!=set(self.full512_original):
                raise ValueError('lookup512 complete encoder coverage')
            self.full512_heads=dict(self.original_heads)
        except BaseException:
            self.close()
            raise

    def run(self,ids,markers,qtype,mask=None,*,candidate=True,**kwargs):
        # Select full-512 scheduling explicitly, independently of logical length.
        # No bucket, attention or head optimization is smuggled into this arm.
        n.request(ids,markers,qtype,mask)
        self.instructions.update(self.lookup512 if candidate else self.full512_original)
        self.instructions.update(self.full512_heads)
        self.type_patches=self.original_head_types
        self.original_masks['head_attention']=self.original_head_masks
        self.mask_patches=dict(self.original_masks)
        self.steps=self.baseline_steps.copy()
        result=n.NativeSession.run(self,ids,markers,qtype,mask,**kwargs)
        if len(self.states)!=5 or self.last_stats['dispatches']!=66:
            raise ValueError('lookup512 execution policy changed')
        return result

    def close(self):
        if not self.failed:
            for field in ('full512_original','lookup512'):
                for inst in getattr(self,field,{}).values():
                    inst.close()
                setattr(self,field,{})
        super().close()


def identity():
    paths,bundle,files,items,_=c.identity()
    folder=paths.base/'geglu_lut_v1_double_m512'
    meta,_=validate(folder,bundle)
    files=dict(files)
    for name in ('build.json','insts.bin','final.xclbin','kernel.o','gelu_tables.cc'):
        files['lookup512/'+name]=c.sha(folder/name)
    for name in ('block','gemm','dispatch','norm','geglu_lut','tables'):
        files['lookup512/'+name+'.o']=c.sha(folder/(name+'.o'))
    for key,value in meta['sources'].items():
        files['lookup512_source/'+key]=value
    files['source/tools/laya_lookup512_full.py']=c.sha(Path(__file__))
    return paths,bundle,items[:3],folder,files


def qualify():
    paths,bundle,fixtures,folder,files=identity()
    if not c.hybrid.idle():
        raise ValueError('device occupied before lookup512 qualification')
    c.emit('LOOKUP512_FULL_START',dict(runner_sha256=c.sha(Path(__file__)),files=files,
        expected_gain_ms=600,minimum_adoption_gain_ms=300,scheduled_rows=512,
        scope='development numeric parity, not frozen accuracy or timing eligibility'))
    calls=[]
    rows=full.cases(fixtures)
    with full.guards(calls):
        with Session(paths,bundle,fixtures,folder) as session:
            for row in rows:
                output=session.run(**full.feed(row),profile=True)
                expected=Counter(self_name for self_name in session.baseline_steps)
                if Counter(stage['name'] for stage in session.last_stats['stages'])!=expected:
                    raise ValueError('lookup512 stage coverage')
                selected=sum(stage['family']=='fast_mlp' for stage in session.last_stats['stages'])
                if selected!=28:
                    raise ValueError('lookup512 dispatches not selected')
                reference=session.run(**full.feed(row),candidate=False)
                comparisons={key:c.p.attention.numeric(output[key],reference[key]) for key in reference}
                choice=bool(c.np.argmax(output['logits'])==c.np.argmax(reference['logits']))
                c.emit('LOOKUP512_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),
                    numeric=comparisons,same_choice=choice,contexts=len(session.states),
                    dispatches=session.last_stats['dispatches'],lookup_stages=selected))
                if calls or not choice or not all(value['ok'] for value in comparisons.values()):
                    raise ValueError('lookup512 complete development numerical gate')
    session=None
    gc.collect()
    if calls or not c.hybrid.idle() or identity()[-1]!=files:
        raise ValueError('lookup512 source/cleanup policy')
    c.emit('LOOKUP512_FULL_RESULT',dict(cases=len(rows),all_numeric_gates=True,all_choices_agree=True,
        cpu_neural_oracle_calls=len(calls),idle_after_close=True,registered_files_unchanged=True,
        contexts=5,dispatches=66,lookup_stages=28,
        scope='unlabelled development only; original full frozen accuracy still required before adoption timing'))


if __name__=='__main__':
    qualify()
