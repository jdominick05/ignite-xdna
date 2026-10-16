#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Short encoder attention/projection streams with an explicit KV loop count."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import bucket_mlp as bm
n=bm.native; b=bm.b; r=b.r; p=bm.p


def sources():
    return dict(bm.sources(),**{'kernels/laya_transformer/bucket_attention.py':p.attention.sha(Path(__file__))})


def prefix():
    text=r.prefix().replace('stack_size = 4608 : i32','stack_size = 4864 : i32')
    old='%count = arith.select %is_attention,%c16,%non_attention_count : index'
    if text.count(old)!=16: raise ValueError('attention loop source changed')
    return text.replace(old,'%attention_count = arith.index_cast %kblocks : i32 to index\n        %count = arith.select %is_attention,%attention_count,%non_attention_count : index')


def pack(seq,pl,source,m):
    for group in range(4):
        seq.control(10,position=tuple(group*4+col for col in range(4)))
        for mg in range(m//128):
            for col in range(4):
                start=source+mg*128*p.OUT; feature=group*4+col
                seq.dma(col,'q',start,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                seq.dma(col,'out',pl['offsets']['packed_norm']+(feature*16+mg*4)*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',start,[1,1,128,p.KVWORDS],[0,0,p.OUT,1])
            seq.wait()


def runtime(pl,m):
    if m not in (128,256,384,512): raise ValueError('attention row bucket')
    seq=r.Sequence(pl['words']); off=pl['offsets']; size=16*p.OUT; groups=m//128
    pack(seq,pl,off['mlp_norm'],m)
    seq.control(2,16)
    for group in range(12):
        for mg in range(groups):
            for col in range(4):
                seq.dma(col,'q',off['packed_norm']+mg*4*p.OUT,[1,16,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                seq.dma(col,'out',off['projection']+((col*12+group)*4+mg)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off['input']+256*p.QWORDS+(col*12+group)*16*p.KVWORDS,[1,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    for mode in (4,5,6,7):
        for which in range(4):
            for mg in range(groups):
                seq.control(mode,position=mg*128,valid=512,local=0)
                for col in range(4):
                    head=col*4+which; feature=head+(16 if mode==6 else 32 if mode==7 else 0)
                    proj=off['projection']+((feature%4)*12+feature//4)*size+mg*4*p.OUT
                    qoffset=off['keys']+(head*16+mg*4)*p.OUT if mode==7 else proj
                    voffset=proj if mode==7 else off['coeff']+mg*4*p.KVWORDS
                    seq.dma(col,'q',qoffset,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                    if mode<6:
                        target=off['queries']+(head*32+mg*8+(mode-4))*p.OUT
                        seq.dma(col,'out',target,[1,1,4,p.OUT],[0,0,2*p.OUT,1],True)
                    else:
                        target=off['keys' if mode==6 else 'kv']+(head*16+mg*4)*p.OUT
                        seq.dma(col,'out',target,[1,1,p.OUT//4,16],[0,0,16,1],True)
                    seq.dma(col,'kv',voffset,[1,1,4,p.KVWORDS],[0,0,p.OUT if mode==7 else p.KVWORDS,1])
                seq.wait()
    # Every consumed KV block is produced above, including masked padding within M.
    # The original core loop used a hardcoded16; this image reads RTP kblocks.
    seq.control(0,kblocks=m//32)
    for col in range(4):
        seq.dma(col,'q',off['queries']+col*128*p.OUT,[4,1,m//16,p.QWORDS],[32*p.OUT,0,p.OUT,1])
        seq.dma(col,'out',off['attention']+col*128*p.OUT,[4,1,m//16,p.OUT],[32*p.OUT,0,p.OUT,1],True)
        for head in range(4):
            seq.dma(col,'kv',off['kv']+(col*4+head)*16*p.OUT+16,[m//64,1,m//32,p.KVWORDS],[0,0,p.OUT,1],bid=head+3)
    seq.wait(); seq.control(8)
    for which in range(4):
        for mg in range(groups):
            for col in range(4):
                head=col*4+which; source=off['attention']+(head*32+mg*8)*p.OUT
                seq.dma(col,'q',source,[1,1,4,p.QWORDS],[0,0,2*p.OUT,1])
                seq.dma(col,'out',off['packed_attention']+(head*16+mg*4)*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',source+p.OUT,[1,1,4,p.KVWORDS],[0,0,2*p.OUT,1])
            seq.wait()
    seq.control(9,16)
    for group in range(4):
        for mg in range(groups):
            for col in range(4):
                seq.dma(col,'q',off['packed_attention']+mg*4*p.OUT,[1,16,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                seq.dma(col,'out',off['mlp_norm']+mg*128*p.OUT+(group*4+col)*65,[1,4,32,65],[0,32*p.OUT,p.OUT,1],True)
                seq.dma(col,'kv',off['projection_weight']+(col*4+group)*16*p.KVWORDS,[1,1,16,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()
    seq.control(1); rows=m//4
    for col in range(4):
        seq.dma(col,'q',off['mlp_norm']+col*rows*p.OUT,[1,1,rows,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off['norm']+col*rows*p.OUT,[1,1,rows*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off['affine'],[rows//4,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait(); pack(seq,pl,off['norm'],m)
    return seq.finish()


def build(base,model,out,counts):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    out.mkdir(parents=True,exist_ok=False); layouts={}; identity=None
    for m in counts:
        folder=out/str(m); folder.mkdir()
        shutil.copyfile(bundle.folders['chain']/'repeat/kernel.o',folder/'kernel.o')
        (folder/'harness.mlir').write_text(prefix()+runtime(bundle.plan,m),encoding='utf-8')
        p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
          '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
        before=p.STACK
        try:
            p.STACK=4864; audits=p.audit(folder,{'fp_exp_ab':2048,'fp_exp_cd':2048})
        finally: p.STACK=before
        cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
        this_identity=json.loads(json.dumps(dict(audits=audits,cdo=cdo)))
        if len(cdo)!=3: raise ValueError('missing attention init')
        if identity is None: identity=this_identity
        if this_identity!=identity: raise ValueError('short attention layout changes resident image')
        layouts[str(m)]=dict(**this_identity,instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'))
        print('BUCKET_ATTENTION_LAYOUT '+json.dumps(dict(m=m,kvblocks=m//32,code_bytes=sorted({a['code_bytes'] for a in audits}),stack_bound=max(sum(a['frames'])+256 for a in audits))),flush=True)
    if sources()!=frozen: raise ValueError('source changed during build')
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,layouts=layouts))


class Session(bm.Session):
    def __init__(self,bundle,mlp_folder,folder):
        folder=Path(folder); meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=sources() or meta['base_identity']!=bundle.identity: raise ValueError('stale short attention build')
        raw_by_m={}
        for m,layout in meta['layouts'].items():
            if p.attention.sha(folder/m/'insts.bin')!=layout['instructions'] or p.attention.sha(folder/m/'final.xclbin')!=layout['xclbin']: raise ValueError('attention artifact changed')
            raw_by_m[int(m)]=(folder/m/'insts.bin').read_bytes()
        super().__init__(bundle,mlp_folder)
        try:
            image=next(iter(meta['layouts'])); xb=self.xrt.xclbin(str(folder/image/'final.xclbin')); self.device.register_xclbin(xb)
            ctx=self.xrt.hw_context(self.device,xb.get_uuid()); kernel=self.xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            groups=[kernel.group_id(3),kernel.group_id(4)]
            self.states['short_attention']=dict(xb=xb,ctx=ctx,kernel=kernel,groups=groups)
            if any(g&0xffff!=self.groups[0]&0xffff for g in groups): raise ValueError('short attention bank differs')
            self.original_masks=copy.deepcopy(self.mask_patches); self.short_masks={}; self.attention_counts=set(raw_by_m)
            for m,raw in raw_by_m.items():
                self.short_masks[m]={}
                for index,spec in enumerate(bundle.meta['chain']['spec']):
                    relocated,_,used=n.c.l.relocate(raw,n.c.l.bindings(self.plan,0,'global'),n.c.l.bindings(self.plan,index,spec['mode']))
                    if used!={'qkv','projection','affine','coeff'}: raise ValueError('short attention constant coverage')
                    relocated,changes=n.c.txn.patch(relocated,meta['layouts'][str(m)]['audits'],512,spec['mode']=='local')
                    name=f'attention_{index}_m{m}'; self.instructions[name]=n.Instruction(self,'short_attention',relocated)
                    self.short_masks[m][name]=[row['value_offset'] for row in changes if row['field']=='valid']
        except BaseException:
            self.close(); raise

    def run(self,ids,markers,qtype,mask=None,*,baseline=False,mlp_only=False,**kwargs):
        valid=n.request(ids,markers,qtype,mask)[-1]; m=bm.bucket(valid)
        if not baseline and (m not in self.bucket_raw or (not mlp_only and m not in self.attention_counts)): raise ValueError('row bucket missing')
        self.steps=[]
        for name in self.baseline_steps:
            shortened=not baseline and (name.startswith('mlp_') or (not mlp_only and name.startswith('attention_')))
            self.steps.append(name+f'_m{m}' if shortened else name)
        self.mask_patches=self.original_masks if baseline or mlp_only else dict(self.short_masks[m],head_attention=self.original_masks['head_attention'])
        return n.NativeSession.run(self,ids,markers,qtype,mask,**kwargs)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--counts',nargs='+',type=int,default=[128,256,384,512])
    args=ap.parse_args(); build(args.base,args.model,args.out,args.counts)
