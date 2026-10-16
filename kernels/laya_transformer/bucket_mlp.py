#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shorter encoder MLP streams; preserve the resident vector image and pitches."""
import argparse
import json
from pathlib import Path
import shutil

import native
b=native.c.b
p=native.p


def bucket(valid):
    if isinstance(valid,bool) or not isinstance(valid,int) or not 1<=valid<=512:
        raise ValueError('valid token count')
    return ((valid+127)//128)*128


def runtime(pl,m):
    if m not in (128,256,384,512): raise ValueError('MLP row bucket')
    if m==512: return b.mlp_runtime(pl)
    mg_count=m//128; off=pl['offsets']; seq=b.MlpSequence(pl['words']); pitch=16*p.OUT
    seq.control(2,16)
    for group in range(21):
        for mg in range(mg_count):
            for col in range(4):
                seq.dma(col,'q',off['packed_norm']+mg*4*p.OUT,[1,16,4,p.QWORDS],[0,pitch,p.OUT,1])
                seq.dma(col,'out',off['mlp_wi']+((col*21+group)*4+mg)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off['mlp_wi_weight']+(col*21+group)*16*p.KVWORDS,[1,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    seq.control(3)
    for first in range(0,41,4):
        columns=[]
        for feature in range(first,min(first+4,41)):
            col=feature%4; columns.append(col); value=feature+41
            qoff=off['mlp_wi']+((feature%4)*21+feature//4)*pitch
            voff=off['mlp_wi']+((value%4)*21+value//4)*pitch
            seq.dma(col,'q',qoff,[1,mg_count,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            seq.dma(col,'out',off['mlp_geglu']+feature*pitch,[1,1,mg_count*4*p.OUT//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',voff,[1,mg_count,4,p.KVWORDS],[0,4*p.OUT,p.OUT,1])
        seq.wait_columns(columns)
    seq.control(9,41)
    for group in range(4):
        for mg in range(mg_count):
            for col in range(4):
                seq.dma(col,'q',off['mlp_geglu']+mg*4*p.OUT,[1,41,4,p.QWORDS],[0,pitch,p.OUT,1])
                seq.dma(col,'out',off['norm']+mg*128*p.OUT+(group*4+col)*65,[1,4,32,65],[0,32*p.OUT,p.OUT,1],True)
                seq.dma(col,'kv',off['mlp_wo_weight']+(col*4+group)*41*p.KVWORDS,[1,1,41,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    seq.control(1)
    rows=m//4
    for col in range(4):
        seq.dma(col,'q',off['norm']+col*rows*p.OUT,[1,1,rows,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['mlp_norm']+col*rows*p.OUT,[1,1,rows*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['mlp_affine'],[rows//4,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait()
    return seq.finish()


def sources():
    return dict(native.vector.sources(),**{'kernels/laya_transformer/bucket_mlp.py':p.attention.sha(Path(__file__))})


def build(base,model,out,counts):
    out=out.resolve()
    frozen=sources(); bundle=native.Bundle(base,model); vm=bundle.meta['vector']
    out.mkdir(parents=True,exist_ok=False); layouts={}
    for m in counts:
        folder=out/str(m); folder.mkdir()
        shutil.copyfile(bundle.folders['vector']/'kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(native.vector.prefix()+runtime(bundle.plan,m),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        before=p.STACK
        try:
            p.STACK=5376; audits=p.audit(folder,{})
        finally: p.STACK=before
        cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        if json.loads(json.dumps(audits))!=vm['audits'] or cdo!=vm['cdo']: raise ValueError('short MLP changes resident core/init')
        layouts[str(m)]=dict(audits=audits,cdo=cdo,instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('BUCKET_MLP_LAYOUT '+json.dumps(dict(m=m,core_and_init_identical=True,instructions=layouts[str(m)]['instructions'])),flush=True)
    if sources()!=frozen: raise ValueError('source changed during build')
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,layouts=layouts,scope='encoder MLP schedule only; attention and heads remain512'))


class Session(native.NativeSession):
    def __init__(self,bundle,folder):
        self.bucket_folder=Path(folder); meta=json.loads((self.bucket_folder/'build.json').read_text())
        if meta['sources']!=sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale MLP bucket bundle')
        self.bucket_meta=meta; self.bucket_raw={}
        for m,layout in meta['layouts'].items():
            raw=(self.bucket_folder/m/'insts.bin').read_bytes()
            if p.attention.sha(self.bucket_folder/m/'insts.bin')!=layout['instructions']: raise ValueError('MLP bucket instructions changed')
            if layout['audits']!=bundle.meta['vector']['audits'] or layout['cdo']!=bundle.meta['vector']['cdo']: raise ValueError('MLP bucket image differs')
            self.bucket_raw[int(m)]=raw
        super().__init__(bundle)

    def _instructions(self):
        super()._instructions(); self.baseline_steps=self.steps.copy()
        for m,raw in self.bucket_raw.items():
            for index,spec in enumerate(self.bundle.meta['chain']['spec']):
                relocated,_,used=native.c.l.relocate(raw,native.c.l.bindings(self.plan,0,'global'),native.c.l.bindings(self.plan,index,spec['mode']))
                if used!={'wi','wo','next_affine'}: raise ValueError('short MLP relocation coverage')
                name=f'mlp_{index}_m{m}'; self.instructions[name]=native.Instruction(self,'mlp',relocated)

    def run(self,ids,markers,qtype,mask=None,*,baseline=False,**kwargs):
        # Validate before selecting a schedule, allocation, or device submission.
        valid=native.request(ids,markers,qtype,mask)[-1]; m=bucket(valid)
        if not baseline and m not in self.bucket_raw: raise ValueError('MLP bucket not built')
        self.steps=[name if baseline or not name.startswith('mlp_') else name+f'_m{m}' for name in self.baseline_steps]
        return super().run(ids,markers,qtype,mask,**kwargs)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--counts',nargs='+',type=int,default=[128,256,384,512])
    args=ap.parse_args(); build(args.base,args.model,args.out,args.counts)
