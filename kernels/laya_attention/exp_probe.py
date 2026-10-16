# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolate the experimental FP32 exponential and physical lookup-bank layout."""
from pathlib import Path
import argparse
import gc
import json
import sys
import numpy as np
root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/'kernels/laya_attention'))
import probe as p
ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('build','run')); ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--raw-table',action='store_true'); args=ap.parse_args()
folder=args.out.resolve(); folder.mkdir(parents=True,exist_ok=True)
source_files=[Path(__file__),root/'kernels/laya_attention/exp_fp32.h',root/'kernels/laya_attention/probe.py',root/'kernels/laya_transformer/block.py']
sources={str(f.relative_to(root)).replace('\\','/'):p.sha(f) for f in source_files}
if args.command=='build':
    source='''#include <aie_api/aie.hpp>
#include <stdint.h>
using F=aie::vector<float,16>;
static inline F full(float x) { return aie::broadcast<float,16>(x); }
static inline F mul(F a,F b) { return aie::mul(a,b).to_vector<float>(); }
#include "exp_fp32.h"
extern "C" void attention_block(const int32_t *q,float *state,int32_t *) {
 for(int i=0;i<1024;i+=16) aie::store_v(state+i,exp16(aie::load_v<16>((const float *)(q+16+i))));
}
extern "C" void attention_finish(const float *state,float *out) {
 for(int i=0;i<1024;i+=16) aie::store_v(out+16+i,aie::load_v<16>(state+i));
}
'''
    (folder/'probe.cc').write_text(source)
    flags=['-Oz','-std=c++20','--target=aie2-none-unknown-elf','-DNDEBUG','-fstack-size-section',
      '-D__AIE_API_AIE_ADF_HPP__',f'-I{p.INSTALL/"include"}',f'-I{root/"kernels/laya_attention"}','-Wno-attributes']
    table=root/'scratch/laya_transformer/block_exp_v2/attention/exp_tables.cc'
    if not args.raw_table:
        sys.path.insert(0,str(root/'kernels/laya_transformer')); import block as b
        table=folder/'tables.cc'; values=','.join(float(x).hex()+'f' for x in b.exp_table_values())
        table.write_text('extern "C" {alignas(32) float fp_exp_ab[512]={'+values+'};alignas(32) float fp_exp_cd[512]={'+values+'};}')
    if args.raw_table:
        table=folder/'tables.cc'; values=','.join(str(i)+'.0f' for i in range(512))
        table.write_text('extern "C" {alignas(32) float fp_exp_ab[512]={'+values+'};alignas(32) float fp_exp_cd[512]={'+values+'};}')
    objects=[]
    for name,path in [('probe',folder/'probe.cc'),('tables',table)]:
        obj=folder/f'{name}.o'; objects.append(obj)
        p.command([p.PEANO/'bin/clang++.exe',*flags,'-c',path,'-o',obj],folder)
    p.command([p.PEANO/'bin/clang++.exe','--target=aie2-none-unknown-elf','-r','-nostdlib',*objects,'-o',folder/'kernel.o'],folder)
    assert not p.command([p.PEANO/'bin/llvm-nm.exe','-u',folder/'kernel.o'],folder).strip()
    (folder/'harness.mlir').write_text(p.harness(1,1))
    p.command([p.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.PEANO}','--get-npu-insts','--npu-insts-name=insts.bin',
      '--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
    p.dump(folder/'build.json',dict(sources=sources,kernel_sha=p.sha(folder/'kernel.o'),raw_table=args.raw_table,
      instructions_sha=p.sha(folder/'insts.bin'),xclbin_sha=p.sha(folder/'final.xclbin')))
    print('EXP_PROBE_BUILD_COMPLETE',flush=True)
else:
    meta=json.loads((folder/'build.json').read_text())
    assert meta['sources']==sources and meta['raw_table']==args.raw_table
    assert meta['kernel_sha']==p.sha(folder/'kernel.o')
    assert meta['instructions_sha']==p.sha(folder/'insts.bin') and meta['xclbin_sha']==p.sha(folder/'final.xclbin')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    sys.path.insert(0,str(root/'tools')); from laya_hybrid import idle
    assert idle()
    dev=xrt.device(0); xb=xrt.xclbin(str(folder/'final.xclbin')); dev.register_xclbin(xb)
    ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
    code=np.frombuffer((folder/'insts.bin').read_bytes(),np.uint32)
    ib=xrt.bo(dev,code.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(code.tobytes(),0)
    bi=xrt.bo(dev,p.PACKET*4,xrt.bo.host_only,kernel.group_id(3)); bo=xrt.bo(dev,p.OUT*4,xrt.bo.host_only,kernel.group_id(4))
    sync=xrt.xclBOSyncDirection; ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
    run=None; failed=False
    try:
        cases=[np.concatenate([np.linspace(-80,0,768,dtype=np.float32),np.linspace(-1,0,256,dtype=np.float32)]),np.zeros(1024,np.float32)]
        grids=[np.repeat((np.arange(first,first+64,dtype=np.float32)+.25)*np.float32(np.log(2)/256)-np.float32(np.log(2)),16) for first in range(0,256,64)]
        cases=grids[:1] if args.raw_table else cases+grids
        print('EXP_PROBE_RUN '+json.dumps(dict(build=meta,atol=0,rtol=.00002,cases=len(cases),poisons=2,scope='isolated exponential, correctness only')),flush=True)
        for case,x in enumerate(cases):
            previous=None
            for fill in (0x7fc12345,0x7fc54321):
                data=np.zeros(p.PACKET,np.uint32); data[1:4]=[1,fill,1]; data[16:1040]=x.view(np.uint32)
                bi.write(data.tobytes(),0); bi.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                bo.write(np.full(p.OUT,fill^0x10000,np.uint32).tobytes(),0); bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                failed=True; run=kernel(3,ib,code.size,bi,bo)
                if run.wait(30000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('dispatch failed; no retry')
                failed=False; bo.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                raw=np.frombuffer(bo.read(p.OUT*4,0),np.uint32).copy(); got=raw[16:1040].view(np.float32)
                assert np.all(raw[:16]==fill) and np.all(raw[1040:]==fill)
                assert previous is None or np.array_equal(got,previous)
                previous=got.copy(); expected=np.exp(x.astype(np.float64)); rel=np.abs(got-expected)/expected
                print('EXP_PROBE_RESULT '+json.dumps(dict(case=case,fill=fill,max_relative=float(rel.max()),
                  relative_l2=float(np.linalg.norm(got-expected)/np.linalg.norm(expected)))),flush=True)
                if args.raw_table:
                    scaled=np.floor(x*np.float32(np.log2(np.e))*256).astype(np.int32)
                    remainder=x-scaled.astype(np.float32)*np.float32(np.log(2)/256)
                    recovered=np.rint(got/np.exp2((scaled>>8).astype(np.float32))/(1+remainder)).astype(np.int32).reshape(64,16)
                    print('LOOKUP_PHYSICAL_INDICES '+json.dumps(recovered[:16,:2].tolist()),flush=True)
                else: assert np.isfinite(got).all() and rel.max()<.00002
    finally:
        if failed: globals()['_quarantined']=(run,bo,bi,ib,kernel,ctx,xb,dev)
        else:
            run=None; bo=None; bi=None; ib=None; kernel=None; ctx=None; xb=None; dev=None; gc.collect()
            print('EXP_PROBE_IDLE '+str(idle()),flush=True)
