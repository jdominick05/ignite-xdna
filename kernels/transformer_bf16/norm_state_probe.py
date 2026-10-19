"""Read actual NPU RMSNorm variance and inverse bits on a saved native input.

Correctness diagnostic only, not model qualification or a timing experiment.
CPU diagnostic values are compared after execution; never used as NPU operands.
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
from kernels.transformer_bf16 import stream
from kernels.transformer_bf16.common import sha,save,bf16_values,bf16_bits
from kernels.transformer_bf16.persistent import PacketSession


def run(trace,weights,diagnostic,layer,folder,output,exact=False):
    if output.exists():raise FileExistsError(output)
    prefix=f'model.layers.{layer}.attention.'
    with np.load(trace,allow_pickle=False) as archive:
        x=archive[prefix+'query_projection'].copy()
        expected=archive[prefix+'query_norm'].copy()
    with np.load(weights,allow_pickle=False) as archive:
        w=archive[f'model.layers.{layer}.self_attn.q_norm.weight'].copy()
    if x.dtype!=np.uint16 or w.dtype!=np.uint16 or x.shape!=(4,w.size) or w.size not in (256,512):
        raise ValueError('Expected four BF16 query heads')
    cpu=json.loads(diagnostic.read_text())
    if cpu['trace_sha256']!=sha(trace) or cpu['weights_sha256']!=sha(weights) or cpu['layer']!=layer:
        raise ValueError('CPU diagnostic input identity mismatch')
    packet=np.zeros((1,1296),np.uint32);poison=0x7fc12345
    packet[0,:3]=[poison,w.size,4];packet[0,3:4].view(np.float32)[0]=1.0/w.size
    packet[0,16:1040].view(np.uint16)[:x.size]=x.reshape(-1)
    packet[0,1040:].view(np.uint16)[:w.size]=w.reshape(-1)
    here=Path(__file__).resolve().parent
    source=here/('norm_exact_probe.cc' if exact else 'norm_state_probe.cc')
    sources={str(p):sha(p) for p in (here/'norm_state_probe.py',source,
        here/'rmsnorm_refined.cc',here/'bf16_product.hpp',here/'stream.py',here/'common.py')}
    contract=dict(scope=__doc__,trace_sha256=sha(trace),weights_sha256=sha(weights),layer=layer,exact_product=exact)
    stream.build(folder,source,'draft_norm_trace',1296,1056,1,sources,contract)
    session=PacketSession()
    with session:raw,evidence=session.execute(folder,packet)
    states=raw[0,:16].reshape(4,4)
    normalized=raw[0,16:16+x.size//2].copy().view(np.uint16).reshape(x.shape)
    suffix=bool(np.all(raw[0,16+x.size//2:1040]==poison))
    rows=[dict(head=i,variance_bits=int(row[0]),initial_inverse_bits=int(row[1]),
               refined_inverse_bits=int(row[2]),residual_bits=int(row[3]),
               torch_variance_bits=cpu['torch_variance_bits'][i],
               torch_inverse_bits=cpu['torch_inverse_bits'][i]) for i,row in enumerate(states)]
    for row,products in zip(rows,raw[0,1040:1056].reshape(4,4)):
        values=products.view(np.float32)
        inverse=np.array(row['refined_inverse_bits'],np.uint32).view(np.float32)
        first=np.float32(values[0]*inverse)
        last=np.float32(values[2]*values[1])
        row['element_129']=dict(x_bits=int(products[0]),weight_bits=int(products[1]),
            first_product_bits=int(products[2]),last_product_bits=int(products[3]),
            ieee_first_product_bits=int(first.view(np.uint32)),
            ieee_last_from_native_first_bits=int(last.view(np.uint32)))
    cpu_inverse=np.array(cpu['torch_inverse_bits'],np.uint32).view(np.float32)[:,None]
    cpu_expected=bf16_bits((bf16_values(x)*cpu_inverse)*bf16_values(w))
    report=dict(scope=__doc__,contract=contract,source_sha256=sha(__file__),
                build_sha256=sha(folder/'build.json'),diagnostic_sha256=sha(diagnostic),rows=rows,
                normalized_matches_saved_native=bool(np.array_equal(normalized,expected)),suffix_guard=suffix,
                different_from_cpu=int(np.count_nonzero(normalized!=cpu_expected)),
                contexts_released=not session.active and not session.images,dispatch=evidence)
    save(output,report);print(json.dumps(report),flush=True)
    if not suffix:raise ValueError('Norm state probe suffix guard changed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--layer',type=int,required=True)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exact',action='store_true')
    a=p.parse_args();run(a.trace,a.weights,a.diagnostic,a.layer,a.folder,a.output,a.exact)
