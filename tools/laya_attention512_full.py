#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full512 attention buffering over the adopted unchanged-polynomial MLP."""
from collections import Counter
import gc
import json
from pathlib import Path

import laya_polynomial512_full as polynomial
c=polynomial.c
n=c.op.n
import attention_fifo as attention


def validate(folder,bundle):
    meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    if (meta['sources']!=attention.sources() or meta['base_identity']!=bundle.identity or
        meta['policy']!=dict(m=512,q_depth=2,kv_depth=2,l2_q_depth=2)):
        raise ValueError('attention512 source/base/policy changed')
    if meta['kernel_object_sha256']!=c.sha(bundle.folders['chain']/'repeat/kernel.o'):
        raise ValueError('attention512 changes arithmetic object')
    return meta,c.op.artifacts(folder,meta)


class Session(polynomial.Session):
    def __init__(self,paths,bundle,fixtures,folder,attention_folder):
        meta,raw=validate(attention_folder,bundle)
        super().__init__(paths,bundle,fixtures,folder)
        try:
            self.full512_attention_original={name:self.instructions[name]
                for name in self.baseline_steps if name.startswith('attention_')}
            self.full512_attention_masks=dict(self.original_masks)
            c.op.Session.install(self,'attention',attention_folder,meta,raw)
            self.full512_attention={name.removesuffix('_m128'):inst
                for name,inst in self.candidate_attention.items()}
            self.full512_candidate_masks={name.removesuffix('_m128'):fields
                for name,fields in self.candidate_mask_fields.items()}
            if (set(self.full512_attention)!=set(self.full512_attention_original) or
                set(self.full512_candidate_masks)!=set(self.full512_attention_original)):
                raise ValueError('attention512 full encoder coverage')
        except BaseException:
            self.close()
            raise

    def run(self,*args,attention_layers=tuple(range(28)),**kwargs):
        if (len(set(attention_layers))!=len(attention_layers) or
            any(type(i) is not int or not 0<=i<28 for i in attention_layers)):
            raise ValueError('attention512 layer selection')
        # Keep the adopted full512 polynomial MLP and heads. Only attention changes.
        selected={f'attention_{i}' for i in attention_layers}
        self.instructions.update(self.full512_attention_original)
        self.instructions.update({name:self.full512_attention[name] for name in selected})
        self.original_masks=dict(self.full512_attention_masks)
        self.original_masks.update({name:self.full512_candidate_masks[name] for name in selected})
        return super().run(*args,**kwargs)

    def close(self):
        if not self.failed:
            for inst in getattr(self,'full512_attention_original',{}).values():
                inst.close()
            self.full512_attention_original={}
        super().close()


def identity():
    paths,bundle,fixtures,folder,files=polynomial.identity()
    attention_folder=paths.base/'attention_fifo_v1_double_m512'
    meta,_=validate(attention_folder,bundle)
    files=dict(files)
    for name in ('build.json','insts.bin','final.xclbin','kernel.o'):
        files['attention512/'+name]=c.sha(attention_folder/name)
    files.update({'attention512_source/'+key:value for key,value in meta['sources'].items()})
    files['source/tools/laya_attention512_full.py']=c.sha(Path(__file__))
    return paths,bundle,fixtures,folder,attention_folder,files


def qualify():
    paths,bundle,fixtures,folder,attention_folder,files=identity()
    c.emit('ATTENTION512_FULL_START',dict(files=files,expected_gain_ms=250,minimum_adoption_gain_ms=100,
        kill_line='any development output-bit mismatch or execution-policy failure',scheduled_rows=512,
        scope='development only; full original frozen scoring and one adoption timing still required'))
    if not c.hybrid.idle():
        raise ValueError('device occupied before attention512 development')
    calls=[]
    with polynomial.full.guards(calls):
        with Session(paths,bundle,fixtures,folder,attention_folder) as session:
            # Local/global representatives first; each call still executes the full network.
            representatives=sorted({next(i for i,spec in enumerate(bundle.meta['chain']['spec'])
                if spec['mode']==mode) for mode in ('local','global')})
            for layer in representatives:
                for index,item in enumerate(fixtures):
                    reference=session.run(**polynomial.full.feed(item),attention_layers=())
                    output=session.run(**polynomial.full.feed(item),attention_layers=(layer,))
                    exact=polynomial.full.exact(output,reference)
                    c.emit('ATTENTION512_LAYER_CHECK',dict(layer=layer,fixture=index,exact=exact))
                    if not all(exact.values()):
                        raise ValueError('attention512 representative layer bit mismatch')
            rows=polynomial.full.cases(fixtures)
            for row in rows:
                reference=session.run(**polynomial.full.feed(row),attention_layers=())
                output=session.run(**polynomial.full.feed(row),profile=True)
                exact=polynomial.full.exact(output,reference)
                stages=session.last_stats['stages']
                attention_stages=[stage for stage in stages if stage['name'].startswith('attention_')]
                if (Counter(stage['name'] for stage in stages)!=Counter(session.baseline_steps) or
                    len(attention_stages)!=28 or any(stage['family']!='candidate_attention' for stage in attention_stages) or
                    len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls):
                    raise ValueError('attention512 execution policy')
                c.emit('ATTENTION512_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=exact,
                    contexts=len(session.states),dispatches=session.last_stats['dispatches'],attention_stages=28))
                if not all(exact.values()):
                    raise ValueError('attention512 complete development bit mismatch')
    session=None
    gc.collect()
    if calls or not c.hybrid.idle() or identity()[-1]!=files:
        raise ValueError('attention512 cleanup/source/CPU gate')
    c.emit('ATTENTION512_FULL_RESULT',dict(cases=len(rows),representative_checks=6,all_output_bits_exact=True,
        contexts=5,dispatches=66,attention_stages=28,cpu_neural_oracle_calls=len(calls),
        idle_after_close=True,registered_files_unchanged=True,
        scope='development only; no frozen accuracy or eligible timing claimed'))


if __name__=='__main__':
    qualify()
