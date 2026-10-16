#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Count actual M128 lookup MLP Shim words; no device or bandwidth inference."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'kernels/laya_transformer'))
import geglu_lut as lookup
import transport_walk as walk


def inventory(folder):
    folder=folder.absolute();meta=json.loads((folder/'build.json').read_text())
    if meta['sources']!=lookup.sources():raise ValueError('lookup source changed')
    raw=(folder/'insts.bin').read_bytes()
    if walk.p.attention.sha(folder/'insts.bin')!=meta['instructions']:raise ValueError('lookup transaction changed')
    bm=lookup.gf.gt.bm;original=bm.b.MlpSequence
    class Recorder(original):
        def __init__(self,*args):super().__init__(*args);self.events=[];self.mode=None
        def control(self,mode,*args,**kwargs):self.mode=mode;super().control(mode,*args,**kwargs)
        def dma(self,col,kind,offset,dims,strides,output=False):
            self.events.append(dict(col=col,kind=kind,offset=offset,dims=dims,strides=strides,output=output,mode=self.mode))
        def wait(self):pass
        def wait_columns(self,columns):pass
        def finish(self):return self.events
    with patch.object(bm.b,'MlpSequence',Recorder):events=bm.runtime(bm.b.plan(),128)
    audit=next(row for row in meta['audits'] if row['core']==[0,2])
    control=(2<<20)+next(start for start,end,name in audit['spans'] if name=='chain_control_0')
    descriptors={};patches={};index=0;mode=None;totals={};calls=[0]*4;resets=[0]*4
    for op in walk.n.c.txn.parse(raw):
        address=op.get('addr',0);reg=address&0xfffff;col=address>>25
        if op['op']=='WRITE' and address==control:mode=op['val']
        if (address>>20)&31:continue
        if op['op']=='BLOCKWRITE':descriptors[col,(reg-0x1d000)//32]=op['words']
        elif op['op']=='DDR_PATCH':patches[col,(reg-0x1d004)//32]=(op['arg_idx'],op['arg_offset']//4)
        elif op['op']=='WRITE' and reg in (0x1d214,0x1d204,0x1d21c):
            event=events[index];index+=1;bid=op['val']&15;arg,offset=patches[col,bid]
            actual=walk.descriptor_addresses(descriptors[col,bid],((op['val']>>16)&255)+1,offset)
            kind={0x1d214:'q',0x1d204:'out',0x1d21c:'kv'}[reg]
            if (event['col'],event['kind'],event['mode'],event['output'])!=(col,kind,mode,bool(arg)) or not np.array_equal(actual,walk.addresses(event)):
                raise ValueError('compiled MLP schedule differs')
            row=totals.setdefault(str(mode),dict(q=0,kv=0,out=0));row[kind]+=len(actual)*4
            if kind=='q' and mode in (2,9):
                if len(actual)%(4*walk.p.QWORDS):raise ValueError('unequal four-core query distribution')
                calls[col]+=len(actual)//(4*walk.p.QWORDS);resets[col]+=1
    if index!=len(events) or len(set(calls))!=1 or len(set(resets))!=1:raise ValueError('incomplete/unbalanced MLP schedule')
    return dict(tag='DERIVED from compiled descriptors; not physical DDR traffic or measured bandwidth',
        queues=index,scheduled_shim_bytes_by_mode=totals,
        scheduled_shim_bytes={kind:sum(row[kind] for row in totals.values()) for kind in ('q','kv','out')},
        gemm_calls_per_core=calls[0],gemm_calls_all_cores=sum(calls)*4,reset_calls_per_core=resets[0],
        scope='one M128 complete lookup MLP; counts repeated scheduled words and broadcast at Shim; excludes core/MemTile traffic',
        evidence=dict(build_sha256=walk.p.attention.sha(folder/'build.json'),instructions_sha256=walk.p.attention.sha(folder/'insts.bin'),
            tool_sha256=walk.p.attention.sha(Path(__file__)),sources=lookup.sources()))


if __name__=='__main__':
    print('LAYA_MLP_TRAFFIC '+json.dumps(inventory(ROOT/'scratch/laya_transformer/geglu_lut_v1_double_m128')),flush=True)
