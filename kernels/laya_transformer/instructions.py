# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Patch explicit RTP value fields; never search instruction words by value."""
from pathlib import Path
import struct

from ignite_xdna.compiler import scheduler

ROOT=Path(__file__).resolve().parents[2]
if Path(scheduler.__file__).resolve()!=ROOT/'src/ignite_xdna/compiler/scheduler.py':
    raise ImportError('transaction decoder must come from this checkout')


def parse(data):
    if len(data)<16 or len(data)%4: raise ValueError('transaction framing')
    _,_,count,size=struct.unpack_from('<4I',data)
    if size!=len(data) or not 0<count<=(len(data)-16)//16: raise ValueError('transaction header')
    ops=scheduler.parse_transaction_stream(data)
    end=ops[-1]['offset']+ops[-1]['size']; tail=data[end:]
    if end>len(data) or len(tail)>=64 or any(tail): raise ValueError('transaction tail')
    return ops


def registers(audits):
    result={}; cores=set()
    for audit in audits:
        col,row=audit['core']; core=col*4+row-2
        if not 0<=col<4 or not 2<=row<6 or (col,row) in cores: raise ValueError('core coordinates')
        cores.add((col,row))
        own=[(start,end) for start,end,name in audit['spans'] if name==f'chain_control_{core}']
        if len(own)!=1: raise ValueError('missing own RTP allocation')
        start,end=own[0]
        if not 0<=start<end<=65536 or end-start!=64 or start%4: raise ValueError('RTP allocation bounds')
        base=(col<<25)|(row<<20)|start
        for field,index in [('mode',0),('valid',5),('local',6)]:
            result[base+4*index]=(core,field)
    if len(cores)!=16 or len(result)!=48: raise ValueError('incomplete RTP allocation map')
    return result


def patch(data,audits,valid,local):
    if type(valid) is not int or not 0<=valid<=512 or type(local) is not bool: raise ValueError('mask controls')
    mapping=registers(audits); expected={'valid':512,'local':0}; wanted={'valid':valid,'local':int(local)}
    output=bytearray(data); seen=set(); changes=[]
    for op in parse(data):
        target=mapping.get(op.get('addr'))
        if target is None or target[1]=='mode': continue
        if op['op']!='WRITE' or op['size']!=24 or op['val_low']!=0: raise ValueError('unexpected RTP write encoding')
        core,field=target
        if target in seen or op['val']!=expected[field]: raise ValueError('base controls must have one initial write per field/core')
        seen.add(target); offset=op['offset']+16
        if struct.unpack_from('<I',data,offset)[0]!=op['val']: raise ValueError('decoder/value-field disagreement')
        struct.pack_into('<I',output,offset,wanted[field])
        changes.append(dict(core=core,field=field,address=op['addr'],value_offset=offset,before=op['val'],after=wanted[field]))
    if len(seen)!=32: raise ValueError('missing initial mask controls')
    parse(output)
    return bytes(output),changes


def execution_signature(data,mapping):
    """Ignore unused RTP state, compare it when query-header commands consume it.

    Commands 4/5 read valid/local while forming Q. Other commands read neither;
    attention consumes the resulting Q header. DDR patch boundaries follow each
    RTP update batch and precede its operand transfer/queue operations.
    """
    state={}; operations=[]; consumed=[]
    for op in parse(data):
        target=mapping.get(op.get('addr'))
        if target is not None:
            if op['op']!='WRITE' or op['size']!=24 or op['val_low']!=0: raise ValueError('unexpected RTP encoding')
            state[target]=op['val']
            if target[1] in ('valid','local'): continue
        operations.append(data[op['offset']:op['offset']+op['size']])
        if op['op']=='DDR_PATCH':
            snapshot=[]
            for core in range(16):
                if state.get((core,'mode')) in (4,5):
                    if (core,'valid') not in state or (core,'local') not in state: raise ValueError('uninitialized mask control')
                    snapshot.append((core,state[(core,'valid')],state[(core,'local')]))
            if snapshot: consumed.append((len(operations),tuple(snapshot)))
    if not consumed: raise ValueError('no query-header control consumers')
    return operations,consumed


def compare_reference(candidate,reference,audits):
    mapping=registers(audits)
    if execution_signature(candidate,mapping)!=execution_signature(reference,mapping):
        raise ValueError('patched instruction semantics differ from compiled reference')
