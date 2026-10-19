"""Check exact tile layout and diagnostic host packing cost; no model inference."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import sha,save
from kernels.transformer_bf16.state import pack_matrix


def scalar(weight,precise):
    # Frozen old per-tile layout, including matrix padding.
    n,k=weight.shape;pn,pk=((n+15)//16)*16,((k+255)//256)*256
    padded=np.zeros((pn,pk),np.uint16);padded[:n,:k]=weight
    result=np.empty((pn//16*(pk//256),2048),np.uint32);index=0
    for row in range(0,pn,16):
        for column in range(0,pk,256):
            block=padded[row:row+16,column:column+256]
            packed=block.T if precise else block.reshape(4,4,32,8).transpose(2,0,3,1)
            result[index].view(np.uint16)[:]=packed.reshape(-1);index+=1
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    sources={str(path):sha(path) for path in (Path(__file__).resolve(),ROOT/'kernels/transformer_bf16/state.py')}
    rows=[];rng=np.random.default_rng(20261018)
    for shape in ((17,257),(256,512),(512,256),(4096,256)):
        weight=rng.integers(0,65536,size=shape[::-1],dtype=np.uint16).T
        original=weight.copy()
        for precise in (False,True):
            expected=scalar(weight,precise);actual=pack_matrix(weight,precise)
            assert np.array_equal(actual,expected)
            samples={'scalar':[],'batched':[]}
            for i in range(20):
                order=(('scalar',scalar),('batched',pack_matrix))
                for name,fn in (order if i%2 else order[::-1]):
                    start=time.perf_counter_ns();result=fn(weight,precise)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    assert np.array_equal(result,expected)
                    if i>=2:samples[name].append(elapsed)
            assert np.array_equal(original,weight)
            rows.append(dict(shape=shape,precise=precise,bitwise_equal=True,unchanged=True,
                             samples_ms=samples,median_ms={k:statistics.median(v) for k,v in samples.items()}))
    assert all(sha(path)==expected for path,expected in sources.items())
    save(a.out,dict(passed=True,scope=__doc__,timing_eligible=False,source_sha256=sources,cases=rows))
    print(json.dumps(dict(passed=True,cases=[dict(shape=r['shape'],precise=r['precise'],median_ms=r['median_ms']) for r in rows])),flush=True)


if __name__=='__main__':main()
