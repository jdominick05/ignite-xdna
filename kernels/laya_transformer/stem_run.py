# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent embedding/norm qualification; no host activation feeds the NPU."""
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import stem as s
b=s.b; p=s.p


def parameters(model):
    import onnx
    graph=onnx.load(str(model),load_external_data=False); initial={t.name:t for t in graph.graph.initializer}
    first=graph.graph.node[6:8]
    if [n.op_type for n in first]!=['Gather','LayerNormalization'] or first[0].input[0]!='m.encoder.embeddings.tok_embeddings.weight': raise ValueError('embedding graph structure')
    if first[1].input[0]!=first[0].output[0] or len(first[1].input)!=2: raise ValueError('embedding norm inputs')
    attrs={a.name:onnx.helper.get_attribute_value(a) for a in first[1].attribute}
    if attrs.get('axis',-1)!=-1 or not np.isclose(attrs.get('epsilon',1e-5),1e-5): raise ValueError('embedding norm semantics')
    table=onnx.numpy_helper.to_array(initial[first[0].input[0]],base_dir=str(model.parent)).copy()
    gamma=onnx.numpy_helper.to_array(initial[first[1].input[1]],base_dir=str(model.parent)).copy()
    if table.dtype!=np.float32 or table.shape!=(s.VOCAB,1024) or gamma.shape!=(1024,): raise ValueError('embedding parameter geometry')
    return table,gamma


def constant_packet(table,gamma,fill):
    values=np.full(s.INPUT_WORDS,fill,np.uint32)
    values[:s.TABLE]=table.ravel().view(np.uint32)
    values[s.AFFINE:s.AFFINE+p.KVWORDS]=0
    values[s.AFFINE:s.AFFINE+1024]=gamma.view(np.uint32)
    return values


def run(out,model,tokens_file):
    out=out.resolve(); model=model.resolve(); meta=json.loads((out/'build.json').read_text())
    if meta['sources']!=s.sources(): raise ValueError('stale embedding build')
    if meta['tokens_file_sha256']!=p.attention.sha(tokens_file): raise ValueError('changed token fixtures')
    if meta['xclbin']!=p.attention.sha(out/'final.xclbin') or meta['instructions']!=p.attention.sha(out/'insts.bin'): raise ValueError('changed embedding artifact')
    table,gamma=parameters(model); fixtures=np.load(tokens_file)
    cases={key:s.validate_tokens(fixtures[key]) for key in fixtures.files if fixtures[key].ndim==1 and fixtures[key].size==len(meta['tokens'])}
    raw=(out/'insts.bin').read_bytes(); prepared={key:s.prepare(raw,meta['tokens'],tokens)[0] for key,tokens in cases.items()}
    refs={key:p.norm.oracle(np.zeros((len(tokens),1024),np.float32),table[tokens],gamma,np.zeros(1024,np.float32))[1] for key,tokens in cases.items()}
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); xb=xrt.xclbin(str(out/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    ibs={}; bi=None; bo=None; iv=None; ov=None; hrun=None; failed=False; sync=xrt.xclBOSyncDirection
    try:
        for key,data in prepared.items():
            ib=xrt.bo(dev,len(data),xrt.bo.cacheable,kernel.group_id(1)); ib.write(data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE); ibs[key]=(ib,len(data))
        bi=xrt.bo(dev,s.INPUT_WORDS*4,xrt.bo.host_only,kernel.group_id(3)); bo=xrt.bo(dev,meta['output_words']*4,xrt.bo.host_only,kernel.group_id(4))
        iv=np.frombuffer(bi.map(),np.uint32); ov=np.frombuffer(bo.map(),np.uint32)
        print('STEM_RUN '+json.dumps(dict(build=meta,model_sha256=p.attention.sha(model),runner_sha256=p.attention.sha(Path(__file__)),
          table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),gamma_sha256=hashlib.sha256(gamma.tobytes()).hexdigest(),
          cases=list(cases),contexts_created=1,host_work='token-address scheduling; CPU gathers are independent references only',
          gates=dict(y=[p.norm.ATOL,p.norm.RTOL,p.norm.L2],residual='embedding FP32 bits exact'),scope='embedding and initial norm only')),flush=True)
        checks=0
        for key,tokens in cases.items():
            percol=len(tokens)//4; rows=np.concatenate([np.arange(col*128,col*128+percol) for col in range(4)])
            previous=None
            for fill in (0x7fc12345,0x7fc54321):
                original=constant_packet(table,gamma,fill); iv[:]=original; bi.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                ov.fill(fill); bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE); ib,nb=ibs[key]
                failed=True; hrun=kernel(3,ib,nb,bi,bo)
                if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('embedding dispatch failed; no retry')
                failed=False; hrun=None; bo.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE); observed=ov.copy()
                output=observed[meta['target']:meta['target']+512*p.OUT].reshape(512,p.OUT)[rows]
                y=output[:,:1024].copy().view(np.float32); residual=output[:,1040:2064].copy().view(np.float32)
                residual_ok=bool(np.array_equal(residual.view(np.uint32),table[tokens].view(np.uint32)))
                diff=np.abs(y-refs[key]); rel=float(np.linalg.norm(diff)/max(np.linalg.norm(refs[key]),1e-30))
                good=bool(np.isfinite(y).all() and np.all(diff<=p.norm.ATOL+p.norm.RTOL*np.abs(refs[key])) and rel<=p.norm.L2)
                np.savez(out/f'observed_{key}_{fill:x}.npz',observed=observed,rows=rows,y=y,residual=residual,reference=refs[key],expected_residual=table[tokens])
                print('STEM_DIAGNOSTIC '+json.dumps(dict(case=key,fill=fill,y_ok=good,relative_l2=rel,residual_bits_exact=residual_ok,
                  guard_values=[int(v) for v in np.unique(output[:,1024:1040])],tail_values=[int(v) for v in np.unique(output[:,2064:])])),flush=True)
                writable=np.zeros(meta['output_words'],bool)
                for row in rows: writable[meta['target']+row*p.OUT:meta['target']+(row+1)*p.OUT]=True
                if not np.all(observed[~writable]==fill): raise ValueError('embedding write outside owned rows')
                if not np.all(output[:,1024:1040]==b.r.mlp.POISON) or not np.all(output[:,2064:]==b.r.mlp.POISON): raise ValueError('embedding norm guards')
                if previous is not None and not np.array_equal(previous,output): raise ValueError('embedding poison dependence')
                bi.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                if not np.array_equal(iv,original): raise ValueError('embedding constant table modified')
                print('STEM_CHECK '+json.dumps(dict(case=key,fill=fill,tokens=len(tokens),ok=good and residual_ok,
                  max_abs=float(diff.max()),relative_l2=rel,residual_bits_exact=residual_ok)),flush=True)
                if not good or not residual_ok: raise ValueError('embedding/norm gate')
                previous=output.copy(); checks+=1
        print('STEM_RESULT '+json.dumps(dict(checks=checks,contexts_created=1,host_activation_gather=False,
          interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,scope='embedding/norm; encoder and full decisions not tested here')),flush=True)
    finally:
        if failed: globals()['_quarantined']=(dev,xb,ctx,kernel,ibs,bi,bo,iv,ov,hrun)
        else:
            hrun=None; ib=None; iv=None; ov=None; bi=None; bo=None; ibs.clear(); kernel=None; ctx=None; xb=None; dev=None; gc.collect()
            print('STEM_IDLE '+str(idle()),flush=True)
