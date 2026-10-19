"""Check true-length softmax patches in a fixed zero-padded attention schedule."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import bf16_bits,save,sha
from kernels.transformer_bf16.state import ArenaCompiler,ArenaProgram
from kernels.transformer_bf16.persistent import PacketSession


def run(cache,output,capacity=64):
    if output.exists():raise FileExistsError(output)
    rng=np.random.default_rng(83)
    query=bf16_bits(rng.normal(size=512))
    key=bf16_bits(rng.normal(size=(capacity,512))*.1)
    value=bf16_bits(rng.normal(size=(capacity,32)))
    def matrices(length):
        k,v=np.zeros_like(key),np.zeros_like(value)
        k[:length]=key[:length];v[:length]=value[:length]
        return k,v
    full=ArenaCompiler(exact_norm=True)
    compact=ArenaCompiler(exact_norm=True,compact=True)
    def program(length,padded=False):
        p=ArenaProgram(compiler=compact if padded else full)
        k,v=matrices(length) if padded else (key[:length],value[:length])
        p.upload(0,query)
        p.projection(0,k,1024,label='key')
        p.softmax(1024,length,capacity=capacity if padded else None,label='attention' if padded else None)
        p.projection(1024,v.T,4096,label='value')
        p.download('context',4096,32)
        return p
    prepared=program(17,True).prepare(cache)
    lengths=tuple(dict.fromkeys([1,17,*[n for n in (257,511,512,513,767,768,1023,1024,2047,2048) if n<=capacity],capacity]))+(17,)
    controls={n:program(n) for n in lengths}
    for p in controls.values():p.build(cache)
    rows=[];rejected=[]
    with PacketSession(share_program=True,resident_inputs=True) as session:
        for length in lengths:
            expected,control=controls[length].execute(cache,session)
            k,v=matrices(length)
            actual,evidence=prepared.execute(session,matrices={'key':k,'value':v.T},lengths={'attention':length})
            equal=bool(np.array_equal(actual['context'],expected['context']))
            rows.append(dict(length=length,passed=equal,control=control,candidate=evidence))
            if not equal:raise ValueError('Padded true-length attention differs from exact-length execution')
        for patch in ({'attention':0},{'attention':capacity+1},{'missing':1},{'attention':1.0}):
            try:prepared.execute(session,lengths=patch)
            except ValueError:rejected.append(True)
            else:raise ValueError('Invalid softmax length reached execution')
    save(output,dict(passed=True,capacity=capacity,rows=rows,invalid_lengths_rejected=len(rejected),source_sha256=sha(__file__),
                     scope=__doc__,contexts_released=not session.active and not session.images))
    print(str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capacity',type=int,choices=(64,512,768,1024,2048,2560),default=64)
    a=p.parse_args();run(a.cache,a.output,a.capacity)
