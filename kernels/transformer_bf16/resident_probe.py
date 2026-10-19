"""Qualify page updates against full-upload NPU command execution.

Checks repeated/changed/restored inputs, complete BO readback, zero-byte reuse,
and full initialization after context eviction. No performance claim.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha
from kernels.transformer_bf16.input_residency import changed_ranges
from kernels.transformer_bf16.patch_probe import data,schedule
from kernels.transformer_bf16.state import ArenaCompiler
from kernels.transformer_bf16.persistent import PacketSession


def offline():
    before=np.zeros(3073,np.uint32)
    current=before.copy();current[[1023,1024,3072]]=[1,2,3]
    spans=changed_ranges(current,before)
    if spans!=[(0,8192),(12288,12292)]:raise ValueError('Boundary/tail page ranges differ')
    if changed_ranges(before,before)!=[]:raise ValueError('Unchanged input has dirty ranges')
    if changed_ranges(before,current)!=spans:raise ValueError('Restoring input lost changed pages')
    rejected=[]
    for a,b in ((current.astype(np.float32),before),(current[:-1],before),(current[::-1],before)):
        try:changed_ranges(a,b)
        except ValueError:rejected.append(True)
        else:raise ValueError('Invalid residency array accepted')
    return dict(boundary_ranges=spans,invalid_arrays_rejected=len(rejected),passed=True)


def run(cache,output,offline_only=False):
    if output.exists():raise FileExistsError(output)
    checks=offline()
    if offline_only:
        save(output,dict(scope=__doc__,offline=checks,source_sha256=sha(__file__)));print(str(output),flush=True);return
    a,b=data(11),data(29)
    control=ArenaCompiler(exact_norm=True)
    candidate=ArenaCompiler(exact_norm=True,compact=True)
    prepared=schedule(a,candidate,False).prepare(cache)
    controls=[schedule(v,control,False) for v in (a,b)]
    for p in controls:p.build(cache)
    rows=[];expected_a=None
    with PacketSession(share_program=True,resident_inputs=True) as session:
        for number,index in enumerate((0,0,1,0,0)):
            expected,baseline=controls[index].execute(cache,session)
            if index==0:expected_a=expected
            values=(a,b)[index]
            patches={} if index==0 else dict(uploads={'q':values['q']},
                matrices={'key':values['key'],'value':values['value'].T},rotations={'position':values['factors']})
            actual,evidence=prepared.execute(session,**patches)
            equal={key:bool(np.array_equal(actual[key],value)) for key,value in expected.items()}
            row=dict(index=number,case=index,passed=all(equal.values()),bitwise_equal=equal,control=baseline,candidate=evidence)
            rows.append(row);print(json.dumps(row),flush=True)
            if not row['passed']:raise ValueError('Resident input differs from full upload')
        if rows[0]['candidate']['input_update_mode']!='full':raise ValueError('Fresh input was not fully initialized')
        if any(rows[i]['candidate']['input_write_bytes']!=0 for i in (1,4)):
            raise ValueError('Unchanged resident input was uploaded')
        if any(rows[i]['candidate']['input_update_mode']!='partial' for i in (2,3)):
            raise ValueError('Changed/restored input did not use page updates')
    evicted=[]
    with PacketSession(max_contexts=1,share_program=True,resident_inputs=True) as eviction_session:
        for repeat in range(2):
            actual,evidence=prepared.execute(eviction_session)
            equal=all(np.array_equal(actual[k],v) for k,v in expected_a.items())
            if not equal or evidence['input_update_mode']!='full':raise ValueError('Evicted input was not reinitialized')
            evicted.append(dict(passed=equal,candidate=evidence))
            if repeat==0:controls[0].execute(cache,eviction_session)
    save(output,dict(passed=True,scope=__doc__,source_sha256=sha(__file__),offline=checks,rows=rows,
                     eviction_checks=evicted,evictions=eviction_session.evictions,
                     contexts_released=not session.active and not session.images and not eviction_session.active and not eviction_session.images))
    print(str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offline',action='store_true')
    a=p.parse_args();run(a.cache,a.output,a.offline)
