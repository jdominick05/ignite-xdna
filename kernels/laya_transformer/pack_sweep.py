#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expand the guarded packing primitive to every feature and column."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pack_primitive as base
p=base.p; n=base.n; ba=base.ba
original_sources=base.sources
expected=base.expected
topology_proof=base.topology_proof
init_proof=base.init_proof


def plan():
    output=128*p.OUT+64
    return dict(words=output+16*16*p.OUT+64,offsets=dict(mlp_norm=16,packed_norm=output))


def output_regions():
    return [(plan()['offsets']['packed_norm']+feature*16*p.OUT,feature) for feature in range(16)]


def runtime(compact):
    layout=plan(); seq=ba.r.Sequence(layout['words'])
    (base.wide.pack if compact else ba.pack)(seq,layout,16,128)
    return seq.finish()


def sources():
    return dict(original_sources(),**{'kernels/laya_transformer/pack_sweep.py':p.attention.sha(Path(__file__))})


def shim_proof(original,compact,old_audits,new_audits):
    def signature(data,audits,compact):
        controls={}
        for addr,(core,field) in n.c.txn.registers(audits).items():
            index={'mode':0,'valid':5,'local':6}[field]
            controls.update({addr+(i-index)*4:(core,i) for i in range(7)})
        state={}; groups={col:0 for col in range(4)}; result=[]; descriptors=0; queues=0; patches=0
        for op in n.c.txn.parse(data):
            if op.get('addr') in controls:
                if op['op']!='WRITE': raise ValueError('sweep control encoding')
                state[controls[op['addr']]]=op['val'];continue
            address=op.get('addr',0); col=address>>25; reg=address&0x1ffffff
            if op['op']=='BLOCKWRITE':
                if col not in groups or reg not in (0x1d000,0x1d040,0x1d060) or len(op['words'])!=8:
                    raise ValueError('unexpected sweep descriptor')
                words=list(op['words']); descriptors+=1
                if reg==0x1d060:
                    wanted=([2112,0,0,66<<20,(2<<30)|(32<<20)|(p.OUT-1),2<<24,
                        (3<<20)|(32*p.OUT-1),2<<24] if compact else
                        [128*2064,0,0,16<<20,(2<<30)|(129<<20)|15,(2<<24)|(p.OUT-1),0,2<<24])
                    if words!=wanted: raise ValueError('sweep KV packet dimensions/TLAST/locks')
                    words=['checked_whole_kv_packet']
                result.append(('BD',address,words))
            elif op['op']=='DDR_PATCH':
                if col not in groups or reg not in (0x1d004,0x1d044,0x1d064) or groups[col]>=4:
                    raise ValueError('unexpected sweep operand patch')
                feature=groups[col]*4+col
                arg=1 if reg==0x1d044 else 0
                offset=(plan()['offsets']['packed_norm']+feature*16*p.OUT)*4 if arg else 64
                if compact and reg==0x1d064: offset+=feature*64*4
                if (op['arg_idx'],op['arg_offset'])!=(arg,offset):
                    raise ValueError('sweep operand offset/feature differs')
                if len(state)!=112: raise ValueError('sweep used uninitialized controls')
                result.append(('PATCH',address,arg,feature,tuple(sorted(state.items()))));patches+=1
                if reg==0x1d064: groups[col]+=1
            elif op['op']=='WRITE' and reg==0x1d21c and col in groups:
                if op['val']!=(0x30003 if compact else 3): raise ValueError('sweep KV repeat count')
                result.append(('checked_KV_QUEUE',address));queues+=1
            else:
                result.append(data[op['offset']:op['offset']+op['size']])
        if descriptors!=48 or patches!=48 or queues!=16 or set(groups.values())!={4}:
            raise ValueError('incomplete sweep transfer coverage')
        return result
    if signature(original,old_audits,False)!=signature(compact,new_audits,True):
        raise ValueError('sweep non-KV transport or effective controls differ')
    return dict(features=16,columns=4,cores=16,query_output_descriptors_identical=True,
        effective_controls_identical=True,whole_kv_object_words=2112,kv_iterations_per_group=4,
        descriptor_and_patch_coverage=48,kv_queue_coverage=16)


def build(original,compact,out):
    with patch.object(base,'plan',plan),patch.object(base,'runtime',runtime),patch.object(base,'sources',sources),patch.object(base,'shim_proof',shim_proof):
        base.build(original,compact,out)
    path=out/'build.json'; meta=json.loads(path.read_text())
    meta['scope']='all16 packing features and4columns, M128; primitive only, complete attention unqualified'
    meta['output_regions']=output_regions()
    p.attention.dump(path,meta)
    print('PACK_SWEEP_BUILD '+json.dumps(dict(plan=plan(),coverage=meta['shim_proof'],scope=meta['scope'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('original','compact','out'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();build(args.original,args.compact,args.out)
