#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Separate query reuse from the rest of the window image's costs."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from types import FunctionType

import laya_window_full_probe as reference
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import window_single as single
c=reference.c;full=reference.full;op=reference.op


class Session(full.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                 single_folder,pack_folder):
        meta=single.validate(single_folder,bundle);raw=op.artifacts(single_folder,meta)
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                         fast_folder=None,pack_folder=pack_folder,mlp_kind='lookup')
        try:op.Session.install(self,'mlp',single_folder,meta,raw)
        except BaseException:self.close();raise


def details():
    paths,bundle,fixtures,old_folder,files,proof=reference.identity()
    folder=paths.base/'window_single_v1_m128';meta=single.validate(folder,bundle)
    parent=json.loads((old_folder/'build.json').read_text())
    if (meta['parent_sha256']!=c.sha(old_folder/'build.json') or meta['kernel_object_sha256']!=parent['kernel_object_sha256'] or
        meta['audits']!=parent['audits'] or meta['cdo']!=parent['cdo']):raise ValueError('single-output changes window parent core/init')
    for file in ('build.json','kernel.o','insts.bin','final.xclbin'):files['single/'+file]=c.sha(folder/file)
    for module in (single,sys.modules[__name__]):
        file=Path(module.__file__);files['single_source/'+str(file.relative_to(c.ROOT)).replace('\\','/')]=c.sha(file)
    return paths,bundle,fixtures,folder,meta,(folder/'insts.bin').read_bytes(),files,dict(proof,single_transport=meta['transport'])


def identity():
    paths,bundle,fixtures,folder,meta,raw,files,proof=details()
    return paths,bundle,fixtures,folder,files,proof


def make(paths,bundle,fixtures,folder,window):
    if not window:return full.make(paths,bundle,fixtures,True)
    return Session(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
                   paths.compact_folder,fixtures,paths.head_folder,folder,paths.base/'compact_pack_wide_v2_m128')


def make_arm(paths,bundle,fixtures,folder,name):
    if name=='packing':return make(paths,bundle,fixtures,folder,False)
    if name=='single':return make(paths,bundle,fixtures,folder,True)
    if name=='window':return reference.make(paths,bundle,fixtures,paths.base/'window_lookup_v2_m128',True)
    raise ValueError('unknown comparison arm')


def gates():
    result={}
    for file,tag,count in [('window_single_v1_layer_desktop2_20261016.log','WINDOW_LAYER_RESULT',6),
                           ('window_single_v1_full_desktop2_20261016.log','WINDOW_FULL_RESULT',25)]:
        path=c.ROOT/'results/laya'/file;lines=path.read_text().splitlines()
        rows=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(tag+' ')]
        terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
        if (len(rows)!=1 or rows[0]['cases']!=count or rows[0]['cpu_neural_oracle_calls'] or not rows[0]['idle_after_close'] or
            not rows[0]['all_arena_bits_exact' if count==6 else 'all_output_bits_exact'] or len(terminal)!=1 or terminal[0]['exit_code']!=0):
            raise ValueError('single-output qualification incomplete')
        result[file]=c.sha(path)
    return result


def protocol():
    return dict(order=[['packing','window','single'],['single','packing','window'],['window','single','packing']],
        warmups=2,reps=5,policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        scope='same91-token/M128 complete inference; separate fresh sessions; setup excluded; window and single share core/init',
        disclosure='unlabelled development outcomes observed; full frozen single/window accuracy unqualified; no CPU/hybrid ratio')


def register(args):
    _,_,fixtures,_,files,proof=identity()
    pre=dict(version=1,files=files,proof=proof,fixture=fixtures[0],qualifications=gates(),**protocol())
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(pre,indent=2)+'\n')
    c.emit('SINGLE_COMPARE_REGISTER',dict(sha256=c.sha(args.prereg),identity=c.na.digest(pre),files=len(files)))


def benchmark(args):
    pre=json.loads(args.prereg.read_text());c.na.committed_registration(args.prereg,pre)
    paths,bundle,fixtures,folder,files,proof=identity()
    if (pre['files']!=files or pre['proof']!=proof or pre['fixture']!=fixtures[0] or pre['qualifications']!=gates() or
        any(pre[key]!=value for key,value in protocol().items())):raise ValueError('single-output comparison changed')
    calls=[];golden=None;rows=[]
    c.emit('SINGLE_COMPARE_START',dict(prereg_sha256=c.sha(args.prereg),scope=pre['scope']))
    with full.guards(calls):
        for index,order in enumerate(pre['order']):
            for name in order:
                if not c.hybrid.idle():raise ValueError('device occupied before profile')
                with make_arm(paths,bundle,fixtures,folder,name) as session:
                    for _ in range(pre['warmups']):output=session.run(**full.feed(fixtures[0]))
                    if golden is None:golden={key:value.copy() for key,value in output.items()}
                    samples=[]
                    for _ in range(pre['reps']):
                        start=time.perf_counter();output=session.run(**full.feed(fixtures[0]));samples.append((time.perf_counter()-start)*1000)
                        if not all(full.exact(output,golden).values()):raise ValueError('timed output changed')
                    if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:raise ValueError('execution policy changed')
                session=None;gc.collect()
                if not c.hybrid.idle():raise ValueError('profile context cleanup')
                row=dict(round=index,arm=name,**c.stats(samples),idle_after_close=True);rows.append(row);c.emit('SINGLE_COMPARE_PROFILE',row)
    if calls or identity()[4]!=files or gates()!=pre['qualifications']:raise ValueError('comparison CPU/source gate')
    c.emit('SINGLE_COMPARE_RESULT',dict(arms={name:c.stats([value for row in rows if row['arm']==name for value in row['samples_ms']])
        for name in ('packing','window','single')},cpu_neural_oracle_calls=len(calls),idle_after_close=c.hybrid.idle(),registered_files_unchanged=True,
        scope='incremental native comparison; no new frozen accuracy or CPU/hybrid ratio'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('layer','qualify','register','benchmark'))
    ap.add_argument('--prereg',type=Path,default=c.ROOT/'results/laya/window_single_compare_v1_prereg_desktop2_20261016.json');args=ap.parse_args()
    if args.command=='layer':
        # A private globals dictionary reuses the already-qualified capture and
        # cleanup path without changing the registered layer module's globals.
        namespace=dict(reference.layer.__dict__,identity=details,__file__=__file__)
        FunctionType(reference.layer.main.__code__,namespace)()
    elif args.command=='qualify':
        namespace=dict(reference.__dict__,identity=identity,make=make,__file__=__file__)
        FunctionType(reference.qualify.__code__,namespace)()
    elif args.command=='register':register(args)
    else:benchmark(args)
