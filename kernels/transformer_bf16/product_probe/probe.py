"""Check BF16-aware RN-even FP32 products across normal finite operands/results.

Sweeps every normal finite BF16 encoding at three fixed FP32 scales and adds
seeded random pairs. Subnormal, zero, infinity, NaN and overflow fallback paths
are outside this qualification. Not a model-quality or performance experiment.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16 import stream
from kernels.transformer_bf16.common import sha,save
from kernels.transformer_bf16.persistent import PacketSession


def run(folder,output):
    if output.exists():raise FileExistsError(output)
    rng=np.random.default_rng(20261018)
    encodings=np.arange(65536,dtype=np.uint32)
    normal=encodings[((encodings>>7)&255)>0]
    normal=normal[((normal>>7)&255)<255]<<16
    a=np.concatenate((rng.integers(0,65536,65536,dtype=np.uint32)<<16,np.tile(normal,3)))
    b=np.concatenate((rng.integers(0,2**32,65536,dtype=np.uint32),
                      np.repeat(np.array([0x3f800001,0x3f7fffff,0xbf800000],np.uint32),normal.size)))
    with np.errstate(all='ignore'):expected=(a.view(np.float32)*b.view(np.float32)).view(np.uint32)
    def finite_normal(bits):
        exponent=(bits>>23)&255
        return (exponent>0)&(exponent<255)
    eligible=finite_normal(a)&finite_normal(b)&finite_normal(expected)
    a,b,expected=a[eligible],b[eligible],expected[eligible]
    count=int(a.size);blocks=(count+15)//16;poison=0x7fc12345
    packets=np.zeros((blocks,48),np.uint32);packets[:,0]=poison
    packets[:,16:32]=0x3f800000;packets[:,32:48]=0x3f800000
    # Strided column slices must be filled via flat temporary arrays.
    left=np.full(blocks*16,0x3f800000,np.uint32);right=left.copy()
    left[:count]=a;right[:count]=b
    packets[:,16:32]=left.reshape(-1,16);packets[:,32:48]=right.reshape(-1,16)
    here=Path(__file__).resolve().parent
    sources={str(p):sha(p) for p in (Path(__file__),here/'kernel.cc',here.parent/'bf16_product.hpp',
                                    here.parent/'stream.py',here.parent/'common.py')}
    contract=dict(scope=__doc__,seed=20261018,eligible=count,bitwise_required=True)
    stream.build(folder,here/'kernel.cc','bf16_product_probe',48,32,blocks,sources,contract)
    session=PacketSession()
    with session:raw,evidence=session.execute(folder,packets)
    actual=raw[:,16:].reshape(-1)[:count]
    wrong=np.flatnonzero(actual!=expected)
    guards=bool(np.all(raw[:,:16]==poison))
    report=dict(passed=guards and wrong.size==0,scope=__doc__,cases=count,different=int(wrong.size),
                guards=guards,source_sha256=sha(__file__),build_sha256=sha(folder/'build.json'),
                contexts_released=not session.active and not session.images,dispatch=evidence)
    if wrong.size:
        i=int(wrong[0]);report['first_mismatch']=dict(index=i,a_bits=int(a[i]),b_bits=int(b[i]),
                                                                  expected=int(expected[i]),actual=int(actual[i]))
    save(output,report);print(json.dumps(report),flush=True)
    if not report['passed']:raise ValueError('BF16-aware product bitwise gate failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.folder,a.output)
