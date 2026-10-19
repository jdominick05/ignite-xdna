"""Compare export-only transport with full-frame arena execution on Phoenix.

All requested outputs must match bitwise; a bad non-exported command must fail
the aggregate guard frame. No numerical tolerances or performance claim.
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
from kernels.transformer_bf16.state import ArenaProgram,ArenaCompiler
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
    full=ArenaCompiler(exact_norm=True)
    compact=ArenaCompiler(exact_norm=True,compact=True)
    programs=[(program(a,full),program(a,compact)) for a in cases]
    for pair in programs:
        for p in pair:p.build(cache)
    # The final command is intentionally non-exported. With an invalid
    # destination, the kernel must reject it and the aggregate must expose it.
    bad=ArenaProgram(compiler=compact)
    bad.upload(0,np.zeros(256,np.uint16))
    bad.download('before_bad_command',0,256)
    bad.upload(0,np.zeros(256,np.uint16))
    bad.packets[-1][7]=16384
    bad_prepared=bad.prepare(cache)
    patch=ArenaProgram(compiler=compact)
    patch.upload(0,cases[0]['norm.x'],label='x')
    patch.norm(0,cases[0]['norm.w'],2048,rows=4)
    patch.download('norm',2048,2048)
    prepared=patch.prepare(cache)
    registration=dict(scope=__doc__,source_sha256=sha(__file__),fixture_sha256=sha(fixtures/'manifest.json'),
                      builds=[str(p.build(cache)/'build.json') for pair in programs for p in pair],
                      compact_builds=[str(bad_prepared.folder/'build.json'),str(prepared.folder/'build.json')])
    if build_only:
        save(output,dict(build_only=True,registration=registration));print(str(output),flush=True);return
    rows=[]
    with PacketSession(share_program=True) as session:
        for index in (0,1,0):
            original,candidate=programs[index]
            expected,control=original.execute(cache,session)
            actual,evidence=candidate.execute(cache,session)
            equal={k:bool(np.array_equal(v,expected[k])) for k,v in actual.items()}
            row=dict(case=index,passed=all(equal.values()),bitwise_equal=equal,control=control,candidate=evidence)
            rows.append(row);print(json.dumps(row),flush=True)
            if not row['passed']:raise ValueError('Compact outputs differ from full frames')
        patched=[]
        for index in (0,1,0):
            p=ArenaProgram(compiler=full)
            p.upload(0,cases[index]['norm.x'])
            p.norm(0,cases[0]['norm.w'],2048,rows=4)
            p.download('norm',2048,2048)
            expected,_=p.execute(cache,session)
            actual,evidence=prepared.execute(session,uploads={'x':cases[index]['norm.x']})
            matched=bool(np.array_equal(actual['norm'],expected['norm']))
            patched.append(dict(case=index,passed=matched,evidence=evidence))
            if not matched:raise ValueError('Compact prepared upload differs')
        rejected=False
        try:bad_prepared.execute(session)
        except ValueError as error:
            if 'output prefix guard' not in str(error):raise
            rejected=True
        if not rejected:raise ValueError('Non-exported command failure was hidden')
        recovered,_=prepared.execute(session,uploads={'x':cases[0]['norm.x']})
        recovery=bool(np.array_equal(recovered['norm'],expected['norm']))
        if not recovery:raise ValueError('Aggregate failure carried into the next stream')
        routing_rejected=[]
        for index,value in ((0,4),(-1,0)):
            changed=prepared._packets.copy()
            changed[index,11]=value
            try:session.execute(prepared.folder,changed)
            except ValueError as error:
                if 'output routing' not in str(error):raise
                routing_rejected.append(True)
            else:raise ValueError('Changed compact routing reached device execution')
    report=dict(passed=all(r['passed'] for r in rows+patched) and rejected,
                registration=registration,rows=rows,patched=patched,invalid_nonexport_rejected=rejected,
                recovery=recovery,routing_rejected=routing_rejected,
                contexts_released=not session.active and not session.images)
    save(output,report);print(str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-only',action='store_true')
    a=p.parse_args();run(a.fixtures,a.cache,a.output,a.build_only)
