#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline shorter head streams; retain full physical pitches and neural operations."""
import argparse
import json
from pathlib import Path
import shutil

import bucket_attention as ba
import compact_attention as compact
n=ba.n; p=ba.p; v=n.vector
second=n.action.s.second; post=second.post; head=post.a; pre=head.h


def sources():
    return dict(compact.sources(),**second.sources(),**{'kernels/laya_transformer/bucket_head.py':p.attention.sha(Path(__file__))})


def check(m):
    if type(m) is not int or m not in (128,256,384,512): raise ValueError('head row bucket')


def row_op(seq,pl,source,target,affine,mode,m):
    seq.control(mode); off=pl['offsets']; rows=m//4
    for col in range(4):
        seq.dma(col,'q',off[source]+col*rows*p.OUT,[1,1,rows,p.QWORDS],[0,0,p.OUT,1])
        seq.dma(col,'out',off[target]+col*rows*p.OUT,[1,1,rows*p.OUT//16,16],[0,0,16,1],True)
        seq.dma(col,'kv',off[affine],[rows//4,1,1,p.KVWORDS],[0,0,0,1])
    seq.wait()


def gemm(seq,pl,source,target,weight,kblocks,features,m):
    off=pl['offsets']; groups=features//4; seq.control(2,kblocks)
    if features%4: raise ValueError('feature extent')
    for group in range(groups):
        for mg in range(m//128):
            for col in range(4):
                seq.dma(col,'q',off[source]+mg*4*p.OUT,[1,kblocks,4,p.QWORDS],[0,16*p.OUT,p.OUT,1])
                seq.dma(col,'out',off[target]+((col*groups+group)*4+mg)*4*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',off[weight]+(col*groups+group)*kblocks*p.KVWORDS,[1,1,kblocks,p.KVWORDS],[0,0,p.KVWORDS,1])
            seq.wait()


def bias(seq,pl,source,target,affine,features,mode,m):
    if mode not in (13,15,16) or features%4: raise ValueError('head bias mode/extent')
    off=pl['offsets']; groups=features//4; seq.control(mode); size=16*p.OUT; mg=m//128
    for first in range(0,features,4):
        for col in range(4):
            feature=first+col; start=off[source]+(col*groups+feature//4)*size
            seq.dma(col,'q',start,[1,mg,4,p.QWORDS],[0,4*p.OUT,p.OUT,1])
            if mode==16:
                seq.dma(col,'out',off[target]+feature*65,[mg,4,32,65],[128*p.OUT,32*p.OUT,p.OUT,1],True)
            else:
                destination=start if mode==15 else off[target]+feature*size
                seq.dma(col,'out',destination,[1,1,mg*4*p.OUT//16,16],[0,0,16,1],True)
            seq.dma(col,'kv',off[affine]+feature*p.KVWORDS,[mg*4,1,1,p.KVWORDS],[0,0,0,1])
        seq.wait()


def pre_runtime(pl,m):
    check(m); seq=v.Sequence(pl['words']); off=pl['offsets']
    row_op(seq,pl,'mlp_norm','norm_input','head_type',17,m)
    row_op(seq,pl,'norm_input','mlp_norm','head_affine',1,m)
    ba.pack(seq,pl,off['mlp_norm'],m)
    gemm(seq,pl,'packed_norm','projection','head_qkv_weight',16,48,m)
    bias(seq,pl,'projection','projection','head_qkv_bias',48,15,m)
    return seq.finish()


def second_pre_runtime(pl,m):
    check(m); seq=v.Sequence(pl['words']); off=pl['offsets']
    ba.pack(seq,pl,off['mlp_norm'],m)
    gemm(seq,pl,'packed_norm','projection','second_qkv_weight',16,48,m)
    bias(seq,pl,'projection','projection','second_qkv_bias',48,15,m)
    return seq.finish()


def post_runtime(pl,m,second_head=False):
    check(m); seq=v.Sequence(pl['words']); off=pl['offsets']; prefix='second' if second_head else 'head'
    gemm(seq,pl,'packed_attention','head_out',prefix+'_out_weight',16,16,m)
    bias(seq,pl,'head_out','mlp_norm',prefix+'_out_bias',16,16,m)
    row_op(seq,pl,'mlp_norm','norm',prefix+'_norm2',1,m)
    ba.pack(seq,pl,off['norm'],m)
    gemm(seq,pl,'packed_norm','mlp_wi',prefix+'_wi_weight',16,64,m)
    bias(seq,pl,'mlp_wi','head_relu',prefix+'_wi_bias',64,13,m)
    gemm(seq,pl,'head_relu','projection',prefix+'_wo_weight',64,16,m)
    bias(seq,pl,'projection','norm',prefix+'_wo_bias',16,16,m)
    row_op(seq,pl,'norm','mlp_norm','second_norm2' if second_head else 'head_next_norm',18 if second_head else 1,m)
    return seq.finish()


def attention_runtime(pl,m):
    check(m); seq=ba.r.Sequence(pl['words']); off=pl['offsets']; size=16*p.OUT
    for mode in (4,5,6,7):
        for which in range(4):
            for mg in range(m//128):
                seq.control(mode,position=mg*128,valid=512,local=0)
                for col in range(4):
                    h=col*4+which; feature=h+(16 if mode==6 else 32 if mode==7 else 0)
                    projected=off['projection']+((feature%4)*12+feature//4)*size+mg*4*p.OUT
                    query=off['keys']+(h*16+mg*4)*p.OUT if mode==7 else projected
                    value=projected if mode==7 else off['head_identity']+mg*4*p.KVWORDS
                    seq.dma(col,'q',query,[1,1,4,p.QWORDS],[0,0,p.OUT,1])
                    if mode<6:
                        target=off['queries']+(h*32+mg*8+(mode-4))*p.OUT
                        seq.dma(col,'out',target,[1,1,4,p.OUT],[0,0,2*p.OUT,1],True)
                    else:
                        target=off['keys' if mode==6 else 'kv']+(h*16+mg*4)*p.OUT
                        seq.dma(col,'out',target,[1,1,p.OUT//4,16],[0,0,16,1],True)
                    seq.dma(col,'kv',value,[1,1,4,p.KVWORDS],[0,0,p.OUT if mode==7 else p.KVWORDS,1])
                seq.wait()
    seq.control(0,kblocks=m//32)
    for col in range(4):
        seq.dma(col,'q',off['queries']+col*128*p.OUT,[4,1,m//16,p.QWORDS],[32*p.OUT,0,p.OUT,1])
        seq.dma(col,'out',off['attention']+col*128*p.OUT,[4,1,m//16,p.OUT],[32*p.OUT,0,p.OUT,1],True)
        for h in range(4):
            seq.dma(col,'kv',off['kv']+(col*4+h)*16*p.OUT+16,[m//64,1,m//32,p.KVWORDS],[0,0,p.OUT,1],bid=h+3)
    seq.wait(); seq.control(8)
    for which in range(4):
        for mg in range(m//128):
            for col in range(4):
                h=col*4+which; source=off['attention']+(h*32+mg*8)*p.OUT
                seq.dma(col,'q',source,[1,1,4,p.QWORDS],[0,0,2*p.OUT,1])
                seq.dma(col,'out',off['packed_attention']+(h*16+mg*4)*p.OUT,[1,1,p.OUT//4,16],[0,0,16,1],True)
                seq.dma(col,'kv',source+p.OUT,[1,1,4,p.KVWORDS],[0,0,2*p.OUT,1])
            seq.wait()
    return seq.finish()


def streams(pl,m):
    return dict(head_pre=pre_runtime(pl,m),head_attention=attention_runtime(pl,m),head_post=post_runtime(pl,m),
      second_pre=second_pre_runtime(pl,m),second_post=post_runtime(pl,m,True))


def build(base,model,attention_folder,out,counts,reuse=None):
    out=out.resolve(); frozen=sources(); bundle=n.Bundle(base,model)
    am=json.loads((attention_folder/'build.json').read_text())
    if am['sources']!=ba.sources() or am['base_identity']!=bundle.identity: raise ValueError('stale head parent')
    out.mkdir(parents=True,exist_ok=False); layouts={}
    for m in counts:
        layouts[str(m)]={}
        for name,runtime in streams(bundle.plan,m).items():
            attention=name=='head_attention'; folder=out/str(m)/name
            original=bundle.folders['chain']/'repeat/kernel.o' if attention else bundle.folders['vector']/'kernel.o'
            harness=(ba.prefix() if attention else v.prefix())+runtime
            previous=Path(reuse)/str(m)/name if reuse is not None else None
            reused=previous is not None and all((previous/file).is_file() for file in ('harness.mlir','kernel.o','final.xclbin','insts.bin'))
            if reused:
                if (previous/'harness.mlir').read_text(encoding='utf-8')!=harness or p.attention.sha(previous/'kernel.o')!=p.attention.sha(original): raise ValueError('reuse input differs')
                folder.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(previous,folder)
            else:
                folder.mkdir(parents=True); shutil.copyfile(original,folder/'kernel.o')
                (folder/'harness.mlir').write_text(harness,encoding='utf-8')
                p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',folder/'harness.mlir',f'--peano={p.attention.PEANO}',
                  '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',f'--tmpdir={folder/"prj"}'],folder)
            before=p.STACK
            try:
                p.STACK=4864 if attention else 5376
                audits=p.audit(folder,{'fp_exp_ab':2048,'fp_exp_cd':2048} if attention else {})
            finally: p.STACK=before
            cdo={f.name:p.attention.sha(f) for f in (folder/'prj').rglob('*_aie_cdo_*.bin')}
            expected=am['layouts']['512'] if attention else bundle.meta['vector']
            if json.loads(json.dumps(audits))!=expected['audits'] or cdo!=expected['cdo']: raise ValueError('short head changes resident core/init')
            layouts[str(m)][name]=dict(audits=audits,cdo=cdo,instructions=p.attention.sha(folder/'insts.bin'),xclbin=p.attention.sha(folder/'final.xclbin'),reused_completed_layout=reused)
            print('BUCKET_HEAD_LAYOUT '+json.dumps(dict(m=m,stage=name,resident_core_and_init_identical=True,reused_completed_layout=reused,scope='offline only')),flush=True)
    if sources()!=frozen: raise ValueError('head bucket source changed')
    p.attention.dump(out/'build.json',dict(sources=frozen,base_identity=bundle.identity,
      attention_build_sha256=p.attention.sha(attention_folder/'build.json'),layouts=layouts,
      scope='offline head query/key/pointwise row reduction; physical pitches retained, no hardware accuracy/speed claim'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('base','model','attention-folder','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--counts',type=int,nargs='+',default=[128,256,384,512])
    ap.add_argument('--reuse',type=Path)
    args=ap.parse_args(); build(args.base,args.model,args.attention_folder,args.out,args.counts,args.reuse)
