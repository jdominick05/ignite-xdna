#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""All-layer packing qualification and preregistered complete native comparison."""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import time
from unittest.mock import patch

import laya_pack_layer_probe as layer
c=layer.compare;op=layer.op;n=layer.n;p=layer.p
LAYERS=tuple(range(28))


class Session(layer.Session):
    def run(self,*args,**kwargs):
        return super().run(*args,pack_layers=LAYERS,**kwargs)


def cases(fixtures):
    result=[dict(name=f'fixture{i}',ids=row['ids'],markers=row['markers'],qtype=row['qtype']) for i,row in enumerate(fixtures)]
    for index,length in enumerate((1,31,32,33,63,64,65,91,127,128,129,255,256,257,383,384,385,511,512,91)):
        ids=[50283]*length;keep=min(length,len(fixtures[0]['ids']));ids[:keep]=fixtures[0]['ids'][:keep]
        result.append(dict(name=f'length{length}_step{index}',ids=ids,markers=[0,length-1],qtype=length%3))
    for mask in ([True,False],[False,True]):
        result.append(dict(name='option_mask'+str(int(mask[0])),ids=fixtures[0]['ids'],markers=[0,len(fixtures[0]['ids'])-1],qtype=2,mask=mask))
    return result


def feed(row):
    return dict(ids=row['ids'],markers=row['markers'],qtype=row['qtype'],**({'mask':c.np.array(row['mask'],bool)} if 'mask' in row else {}))


def identity():
    pre=c.ROOT/'results/laya/native_hybrid_compare_v1_prereg_desktop2_20261016.json'
    _,paths,bundle,fixtures=c.load_registration(pre)
    evidence=c.ROOT/'results/laya/pack_expanded_v1_device_desktop2_20261016.log'
    rows=[line for line in evidence.read_text().splitlines() if line.startswith('PACK_LAYER_CHECK ')]
    verdicts=[json.loads(line.split(' ',1)[1]) for line in evidence.read_text().splitlines() if line.startswith('PACK_LAYER_RESULT ')]
    if len(rows)!=9 or len(verdicts)!=1 or not verdicts[0]['all_fixture_bits_exact'] or verdicts[0]['cpu_neural_oracle_calls'] or not verdicts[0]['idle_after_close']:
        raise ValueError('layer prerequisite missing')
    for line in rows:
        row=json.loads(line.split(' ',1)[1])
        if not all(row['exact'].values()) or row['contexts']!=5 or row['dispatches']!=66: raise ValueError('layer prerequisite failed')
    folder=paths.base/'compact_pack_wide_v2_m128';meta=json.loads((folder/'build.json').read_text())
    parent=json.loads((paths.attention_candidate/'build.json').read_text())
    if meta['sources']!=op.cw.sources():raise ValueError('wide source differs')
    files=dict(c.identity()[2])
    for name in ('build.json','kernel.o','insts.bin','final.xclbin'):
        files['wide/'+name]=c.sha(folder/name)
    for module in (layer,layer.transport_walk,layer.pack_sweep,layer.pack_sweep.base):
        file=Path(module.__file__);files['new_source/'+str(file.relative_to(c.ROOT)).replace('\\','/')]=c.sha(file)
    files['new_source/tools/laya_pack_full_probe.py']=c.sha(Path(__file__))
    files['layer_prerequisite']=c.sha(evidence)
    walk=layer.transport_walk.verify_pair((paths.attention_candidate/'insts.bin').read_bytes(),(folder/'insts.bin').read_bytes(),bundle.plan)
    init=layer.pack_sweep.init_proof(paths.attention_candidate/'prj/cdo_main/main_aie_cdo_init.bin',
        folder/'prj/cdo_main/main_aie_cdo_init.bin',parent['audits'],meta['audits'])
    return paths,bundle,fixtures,files,dict(transport=walk,initialization=init)


def guards(calls):
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('CPU neural/oracle call')
    methods=[(c.ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
        (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
        (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    stack=ExitStack()
    for module,name in methods:stack.enter_context(patch.object(module,name,forbidden))
    return stack


def make(paths,bundle,fixtures,wide):
    common=(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
        paths.compact_folder,fixtures,paths.head_folder)
    factory=Session if wide else op.Session
    extra=dict(pack_folder=paths.base/'compact_pack_wide_v2_m128') if wide else {}
    return factory(*common,fast_folder=paths.fast_folder,mlp_kind='lookup',**extra)


def exact(output,reference):
    return {key:bool(c.np.array_equal(value.view(c.np.uint32),reference[key].view(c.np.uint32))) for key,value in output.items()}


def qualify():
    paths,bundle,fixtures,files,proof=identity();requests=cases(fixtures);refs=[];calls=[]
    if not c.hybrid.idle():raise ValueError('device occupied')
    c.emit('PACK_FULL_IDENTITY',dict(files=files,proof=proof,cases=len(requests),scope='all-layer unlabelled/synthetic correctness, no performance or full accuracy claim'))
    with guards(calls):
        with make(paths,bundle,fixtures,False) as session:
            for row in requests:refs.append({key:value.copy() for key,value in session.run(**feed(row)).items()})
        session=None;gc.collect()
        if not c.hybrid.idle():raise ValueError('reference context cleanup')
        with make(paths,bundle,fixtures,True) as session:
            for row,reference in zip(requests,refs):
                output=session.run(**feed(row));check=exact(output,reference)
                c.emit('PACK_FULL_CHECK',dict(case=row['name'],valid=len(row['ids']),exact=check,contexts=len(session.states),dispatches=session.last_stats['dispatches']))
                if not all(check.values()) or len(session.states)!=5 or session.last_stats['dispatches']!=66:
                    raise ValueError('all-layer fixture/boundary gate')
        session=None;gc.collect()
    if calls or not c.hybrid.idle() or identity()[3]!=files:raise ValueError('identity/CPU/cleanup gate')
    c.emit('PACK_FULL_RESULT',dict(cases=len(requests),all_output_bits_exact=True,cpu_neural_oracle_calls=len(calls),idle_after_close=True,
        scope='development qualification only; new frozen accuracy arm still required'))


def register(args):
    _,_,fixtures,files,proof=identity()
    pre=dict(version=1,files=files,proof=proof,fixture=fixtures[0],
        order=[['lookup','wide'],['wide','lookup'],['lookup','wide']],warmups=2,reps=5,
        policy=dict(contexts=5,dispatches=66,cpu_neural_oracle_calls=0),
        scope='same-sitting native serving configurations at M128 for91-token request; setup excluded; no CPU/hybrid ratio or held-out accuracy claim',
        disclosure='candidate development outputs already observed; full frozen candidate accuracy still unqualified')
    with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(pre,indent=2)+'\n')
    c.emit('PACK_COMPARE_REGISTER',dict(sha256=c.sha(args.prereg),identity=c.na.digest(pre)))


def benchmark(args):
    pre=json.loads(args.prereg.read_text());c.na.committed_registration(args.prereg,pre)
    paths,bundle,fixtures,files,proof=identity()
    if pre['files']!=files or pre['fixture']!=fixtures[0] or pre['proof']!=proof:raise ValueError('comparison identity changed')
    if pre['order']!=[['lookup','wide'],['wide','lookup'],['lookup','wide']] or pre['warmups']!=2 or pre['reps']!=5:
        raise ValueError('comparison protocol changed')
    calls=[];reference=None;rows=[]
    c.emit('PACK_COMPARE_START',dict(prereg_sha256=c.sha(args.prereg),scope=pre['scope']))
    with guards(calls):
        for round_index,order in enumerate(pre['order']):
            for name in order:
                if not c.hybrid.idle():raise ValueError('device occupied before profile')
                with make(paths,bundle,fixtures,name=='wide') as session:
                    for _ in range(pre['warmups']):output=session.run(**feed(fixtures[0]))
                    if reference is None:reference={key:value.copy() for key,value in output.items()}
                    samples=[]
                    for _ in range(pre['reps']):
                        start=time.perf_counter();output=session.run(**feed(fixtures[0]));samples.append((time.perf_counter()-start)*1000)
                        if not all(exact(output,reference).values()):raise ValueError('timed output changed')
                    if len(session.states)!=5 or session.last_stats['dispatches']!=66 or calls:raise ValueError('execution policy changed')
                session=None;gc.collect()
                if not c.hybrid.idle():raise ValueError('profile context cleanup')
                row=dict(round=round_index,arm=name,**c.stats(samples),idle_after_close=True);rows.append(row);c.emit('PACK_COMPARE_PROFILE',row)
    if identity()[3]!=files or calls:raise ValueError('comparison source/CPU policy changed')
    c.emit('PACK_COMPARE_RESULT',dict(arms={name:c.stats([value for row in rows if row['arm']==name for value in row['samples_ms']]) for name in ('lookup','wide')},
        cpu_neural_oracle_calls=len(calls),idle_after_close=c.hybrid.idle(),registered_files_unchanged=True,
        scope='development performance only; full candidate frozen accuracy still required'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('qualify','register','benchmark'))
    parser.add_argument('--prereg',type=Path,default=c.ROOT/'results/laya/pack_compare_v1_prereg_desktop2_20261016.json')
    args=parser.parse_args()
    if args.command=='qualify':qualify()
    elif args.command=='register':register(args)
    else:benchmark(args)
