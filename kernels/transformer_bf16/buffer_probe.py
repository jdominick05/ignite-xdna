"""Check double-buffered packet DMA against the single-buffer arena control."""
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
from kernels.transformer_bf16.state import ArenaCompiler,ArenaProgram
from kernels.transformer_bf16.state_probe import program
from kernels.transformer_bf16.persistent import PacketSession


def copy_boundaries(compiler):
    p=ArenaProgram(compiler=compiler);expected={}
    spans=[(0,n) for n in (1,31,32,33,63,64,2048)]+[(1,65),(31,33),(32,65),
          (16351,33),(16352,32),(16383,1),(8192,4352),(12032,4352)]
    for index,(slot,count) in enumerate(spans):
        start=max(0,slot-1);end=min(16384,slot+count+1)
        values=(np.arange(end-start,dtype=np.uint32)*65521+index*131).astype(np.uint16)
        for offset in range(0,values.size,compiler.upload_limit):
            p.upload(start+offset,values[offset:offset+compiler.upload_limit])
        bits=(np.arange(count,dtype=np.uint32)*251+index*997).astype(np.uint16)
        p.upload(slot,bits);values[slot-start:slot-start+count]=bits
        for offset in range(0,end-start,2048):
            label=f'copy_{index}_{offset}'
            size=min(2048,end-start-offset)
            p.download(label,start+offset,size)
            expected[label]=values[offset:offset+size].copy()
    return p,expected


def run(fixtures,cache,output,build_only=False,vector_copy=False):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((fixtures/'manifest.json').read_text())
    cases=[]
    for row in manifest['cases'][:2]:
        path=fixtures/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Fixture changed')
        with np.load(path,allow_pickle=False) as f:cases.append({k:f[k].copy() for k in f.files})
    compilers=[ArenaCompiler(exact_norm=True,compact=True,small_packets=True,input_depth=depth,vector_copy=copy)
               for depth,copy in (((2,False),(2,True)) if vector_copy else ((1,False),(2,False)))]
    programs=[[program(case,compiler).prepare(cache) for compiler in compilers] for case in cases]
    boundaries=[]
    if vector_copy:
        for compiler in compilers:
            p,expected=copy_boundaries(compiler)
            boundaries.append((p.prepare(cache),expected))
    registration=dict(scope=__doc__,source_sha256=sha(__file__),fixtures_sha256=sha(fixtures/'manifest.json'),
                      builds=[str(p.folder/'build.json') for pair in programs for p in pair]+
                             [str(p.folder/'build.json') for p,_ in boundaries],vector_copy=vector_copy)
    if build_only:
        save(output,dict(build_only=True,registration=registration));print(str(output),flush=True);return
    rows=[]
    with PacketSession(resident_inputs=True) as session:
        for index in (0,1,0):
            control,candidate=programs[index]
            expected,a=control.execute(session)
            actual,b=candidate.execute(session)
            equal={k:bool(np.array_equal(v,expected[k])) for k,v in actual.items()}
            if not all(equal.values()):raise ValueError('Candidate changed output')
            rows.append(dict(case=index,bitwise_equal=equal,control=a,candidate=b))
        boundary_rows=[]
        for p,expected in boundaries:
            actual,evidence=p.execute(session)
            if any(not np.array_equal(actual[k],v) for k,v in expected.items()):
                raise ValueError('Copy crossed a boundary, changed bits or corrupted a guard')
            boundary_rows.append(dict(exports=len(expected),passed=True,evidence=evidence))
        # Invalid non-export commands must still propagate to the aggregate
        # guard, and the next complete invocation must recover.
        candidate=programs[0][1]
        packets=candidate._packets.copy()
        commands=np.flatnonzero((packets[:,15]==8)&(packets[:,11]==0))
        if not commands.size:raise ValueError('Expected an unexported projection')
        packets[commands[0],5]=16384
        result,bad=session.execute(candidate.folder,packets)
        rejected=bool(np.any(result[:,:16]!=candidate.poison))
        if not rejected:raise ValueError('Invalid input was not rejected')
        recovered,evidence=candidate.execute(session)
        expected,_=programs[0][0].execute(session)
        if any(not np.array_equal(recovered[k],expected[k]) for k in expected):raise ValueError('Recovery changed output')
    report=dict(passed=True,registration=registration,cases=rows,invalid_rejected=rejected,
                invalid_evidence=bad,recovery=evidence,boundary_checks=boundary_rows,
                npu_released=not session.active and not session.images)
    save(output,report);print(str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--vector-copy',action='store_true',help='Compare vector/scalar copies at input depth two, including boundary guards')
    args=parser.parse_args();run(args.fixtures,args.cache,args.output,args.build_only,args.vector_copy)
