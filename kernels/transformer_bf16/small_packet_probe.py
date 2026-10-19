"""Compare smaller arena packets and resident GELU LUT with the wide control."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha,bf16_values
from kernels.transformer_bf16.state import ArenaCompiler,ArenaProgram
from kernels.transformer_bf16.state_probe import program
from kernels.transformer_bf16.persistent import PacketSession


def run(fixtures,cache,output,build_only=False):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((fixtures/'manifest.json').read_text())
    cases=[]
    for row in manifest['cases'][:2]:
        path=fixtures/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Fixture changed')
        with np.load(path,allow_pickle=False) as f:cases.append({k:f[k].copy() for k in f.files})
    wide=ArenaCompiler(exact_norm=True)
    small=ArenaCompiler(exact_norm=True,compact=True,small_packets=True)
    programs=[(program(a,wide),program(a,small)) for a in cases]
    rejected=[]
    p=program(cases[0],small)
    changed=cases[0]['table'].copy();changed[0]^=1
    for operation in (lambda:p.upload(8192,np.zeros(16,np.uint16)),
                      lambda:p.upload(0,np.zeros(4353,np.uint16)),
                      lambda:p.geglu(0,2048,changed,4096)):
        try:operation()
        except ValueError:rejected.append(True)
        else:raise ValueError('Invalid resident table/upload accepted')
    bad=ArenaProgram(compiler=small)
    bad.upload(0,cases[0]['geglu.gate'])
    bad.upload(2048,cases[0]['geglu.up'])
    bad.geglu(0,2048,cases[0]['table'],4096)
    bad.download('geglu',4096,2048)
    prepared=bad.prepare(cache)
    for pair in programs:
        for p in pair:p.build(cache)
    registration=dict(scope=__doc__,source_sha256=sha(__file__),fixtures_sha256=sha(fixtures/'manifest.json'),
                      builds=[str(p.build(cache)/'build.json') for pair in programs for p in pair]+[str(prepared.folder/'build.json')])
    if build_only:
        save(output,dict(build_only=True,registration=registration));print(str(output),flush=True);return
    rows=[]
    with PacketSession(share_program=True,resident_inputs=True) as session:
        for index in (0,1,0):
            control,candidate=programs[index]
            expected,wide_evidence=control.execute(cache,session)
            actual,evidence=candidate.execute(cache,session)
            equal={k:bool(np.array_equal(v,expected[k])) for k,v in actual.items()}
            x=bf16_values(actual['geglu']).astype(np.float64)
            y=bf16_values(cases[index]['geglu.expected']).astype(np.float64)
            gates=manifest['gates'];error=np.abs(x-y)
            relative=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
            reference_ok=bool(np.isfinite(x).all() and np.all(error<=gates['atol']+gates['rtol']*np.abs(y)) and relative<=gates['relative_l2'])
            reference=dict(passed=reference_ok,gates=gates,relative_l2=relative,max_abs=float(error.max()))
            rows.append(dict(case=index,passed=all(equal.values()) and reference_ok,bitwise_equal=equal,
                             frozen_geglu=reference,control=wide_evidence,candidate=evidence))
            if not rows[-1]['passed']:raise ValueError('Small packet outputs or frozen GELU criteria failed')
        # Corrupt only a bounded, non-DMA routing field. The arena kernel must
        # reject the table before accessing it; the aggregate exposes failure.
        prepared.execute(session)
        packets=prepared._packets.copy()
        gelu=np.flatnonzero(packets[:,15]==11)
        if gelu.size!=1:raise ValueError('Expected one resident GELU command')
        packets[gelu[0],12]=16384
        result,evidence=session.execute(prepared.folder,packets)
        kernel_rejected=bool(np.any(result[:,:16]!=prepared.poison))
        if not kernel_rejected:raise ValueError('Invalid LUT arena offset was not rejected')
        # Upload count is checked against the smaller packet's physical payload.
        packets=prepared._packets.copy();packets[0,8]=4353
        result,upload_evidence=session.execute(prepared.folder,packets)
        upload_rejected=bool(np.any(result[:,:16]!=prepared.poison))
        if not upload_rejected:raise ValueError('Overlong device upload was not rejected')
    save(output,dict(passed=True,registration=registration,rows=rows,host_rejections=rejected,
                     invalid_table_rejected=kernel_rejected,invalid_upload_rejected=upload_rejected,
                     invalid_table_evidence=evidence,invalid_upload_evidence=upload_evidence,
                     contexts_released=not session.active and not session.images))
    print(str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-only',action='store_true')
    a=p.parse_args();run(a.fixtures,a.cache,a.output,a.build_only)
