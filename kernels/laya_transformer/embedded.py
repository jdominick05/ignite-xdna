# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real token inputs through embedding/norm and the complete native encoder."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import chain as c
import chain_run as cr
import mask_reference
import stem as s
import stem_run as sr
b=c.b; p=b.p


def relocate_output(raw,source,target,words):
    import struct
    if target<0 or target+512*p.OUT>words: raise ValueError('embedding destination bounds')
    output=bytearray(raw); changed=[]
    for op in c.txn.parse(raw):
        if op['op']!='DDR_PATCH' or op['arg_idx']!=1: continue
        if op['arg_offset']%4 or not source<=op['arg_offset']//4<source+512*p.OUT: raise ValueError('embedding output reference outside owned rows')
        value=op['arg_offset']+(target-source)*4
        struct.pack_into('<I',output,op['offset']+40,value)
        changed.append(dict(value_offset=op['offset']+40,before=op['arg_offset'],after=value))
    if not changed: raise ValueError('missing embedding output references')
    return bytes(output),changed


def run(out,stem,model,tokens_file,token_key,collect,head_folder=None,attention_folder=None,post_folder=None,second_folder=None):
    out=out.resolve(); model=model.resolve(); meta=json.loads((out/'build.json').read_text())
    stem=stem.resolve(); sm=json.loads((stem/'build.json').read_text())
    vector_image=sm.get('kind')=='vector-without-attention'
    if not vector_image and sm['sources']!=s.sources(): raise ValueError('stale embedding build')
    if len(sm['tokens'])!=512: raise ValueError('partial embedding build')
    for file,field in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
        if p.attention.sha(stem/file)!=sm[field]: raise ValueError('changed embedding artifact')
    mlp=meta['artifacts']['mlp']
    if vector_image:
        import vector
        vector.validate_compatibility(stem,sm,mlp)
    else:
        if [a['spans'] for a in sm['audits']]!=[a['spans'] for a in mlp['audits']]: raise ValueError('embedding/MLP physical ABI differs')
        if any(sm['cdo'][name]!=mlp['cdo'][name] for name in ('main_aie_cdo_init.bin','main_aie_cdo_enable.bin')): raise ValueError('embedding/MLP initialization differs')
    if meta['sources']!=c.sources() or meta['model_sha256']!=p.attention.sha(model): raise ValueError('stale chain build/model')
    if meta['coefficients_sha256']!=p.attention.sha(out/'coefficients.npz'): raise ValueError('changed coefficient table')
    for name,artifact in meta['artifacts'].items():
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(out/name/file)!=artifact[key]: raise ValueError('chain artifact changed')
    for stage in meta['stages']:
        if p.attention.sha(out/stage['instructions'])!=stage['sha256']: raise ValueError('changed stage instructions')
    graph,spec=c.l.model_spec(model,len(meta['spec']))
    if spec!=meta['spec']: raise ValueError('changed model layer structure')
    parameters,weight_hashes=c.l.constants(model,graph,spec)
    # numpy_helper materializes external FP32 raw_data in this graph. The
    # independent rounded constants own their storage; drop the original
    # graph before allocating the device arena and verification snapshots.
    graph=None; gc.collect()
    tables=np.load(out/'coefficients.npz'); fixtures=np.load(tokens_file); tokens=s.validate_tokens(fixtures[token_key],512)
    valid=int(fixtures[token_key+'_valid']) if token_key+'_valid' in fixtures else 512
    mask_values=mask_reference.extract(model,valid)
    embedding_table,embedding_gamma=sr.parameters(model)
    embedding_reference=embedding_table[tokens].copy()
    _,stem_reference=p.norm.oracle(np.zeros_like(embedding_reference),embedding_reference,embedding_gamma,np.zeros(1024,np.float32))
    refs=cr.references(stem_reference,parameters,spec,tables,mask_values,valid,initial_residual=embedding_reference)
    pl=meta['plan']; head=None; head_values=None; head_meta=None
    if head_folder is not None:
        import head_pre as head
        if not vector_image: raise ValueError('head prelude needs the vector image')
        head_folder=head_folder.resolve(); head_meta=json.loads((head_folder/'build.json').read_text())
        if head_meta['sources']!=head.sources() or head_meta['vector_build_sha256']!=p.attention.sha(stem/'build.json') or head_meta['chain_build_sha256']!=p.attention.sha(out/'build.json'): raise ValueError('stale head prelude')
        if head_meta['cdo']!=sm['cdo'] or head_meta['audits']!=sm['audits']: raise ValueError('head prelude image differs')
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(head_folder/file)!=head_meta[key]: raise ValueError('changed head artifact')
        pl=head_meta['plan']; head_values,head_hashes=head.parameters(model)
        if token_key+'_qtype' not in fixtures: raise ValueError('missing qtype')
        qtype=int(fixtures[token_key+'_qtype'])
    head_attention=None; attention_meta=None
    if attention_folder is not None:
        if head is None: raise ValueError('head attention requires the head prelude')
        import head_attention
        attention_folder=attention_folder.resolve(); attention_meta=json.loads((attention_folder/'build.json').read_text())
        if not 1<=valid<=512: raise ValueError('head attention requires nonempty valid keys')
        if attention_meta['sources']!=head_attention.sources() or attention_meta['head_pre_build_sha256']!=p.attention.sha(head_folder/'build.json') or attention_meta['chain_build_sha256']!=p.attention.sha(out/'build.json'): raise ValueError('stale head attention')
        if attention_meta['audits']!=meta['artifacts']['first']['audits'] or attention_meta['cdo']!=meta['artifacts']['first']['cdo']: raise ValueError('head attention core/init differs')
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(attention_folder/file)!=attention_meta[key]: raise ValueError('changed head attention artifact')
        pl=attention_meta['plan']
    head_post=None; post_meta=None
    if post_folder is not None:
        if head_attention is None: raise ValueError('head post requires head attention')
        import head_post
        post_folder=post_folder.resolve(); post_meta=json.loads((post_folder/'build.json').read_text())
        if post_meta['sources']!=head_post.sources() or post_meta['head_attention_build_sha256']!=p.attention.sha(attention_folder/'build.json') or post_meta['vector_build_sha256']!=p.attention.sha(stem/'build.json'): raise ValueError('stale head post')
        if post_meta['audits']!=sm['audits'] or post_meta['cdo']!=sm['cdo']: raise ValueError('head post core/init differs')
        for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
            if p.attention.sha(post_folder/file)!=post_meta[key]: raise ValueError('changed head post artifact')
        pl=post_meta['plan']; post_values,post_hashes=head_post.parameters(model)
        post_ref=head_post.reference(refs[-1][1],head_values,post_values,qtype,valid)
    head_second=None; second_meta=None
    if second_folder is not None:
        if head_post is None: raise ValueError('second head requires the first head post')
        import head_second
        second_folder=second_folder.resolve(); second_meta=json.loads((second_folder/'build.json').read_text())
        if second_meta['sources']!=head_second.sources() or second_meta['head_post_build_sha256']!=p.attention.sha(post_folder/'build.json') or second_meta['vector_build_sha256']!=p.attention.sha(stem/'build.json'): raise ValueError('stale second head')
        for name,artifact in second_meta['artifacts'].items():
            if artifact['audits']!=sm['audits'] or artifact['cdo']!=sm['cdo']: raise ValueError('second head core/init differs')
            for file,key in [('final.xclbin','xclbin'),('insts.bin','instructions')]:
                if p.attention.sha(second_folder/name/file)!=artifact[key]: raise ValueError('changed second head artifact')
        pl=second_meta['plan']; second_values,second_hashes=head_second.parameters(model)
        second_ref=head_second.reference(post_ref,second_values,valid)
    off=pl['offsets']; sizes=pl['sizes']; count=len(spec)
    sys.path.insert(0,str(p.ROOT/'tools')); from laya_hybrid import idle
    if not idle(): raise RuntimeError('foreign context')
    sys.path.insert(0,'C:/Xilinx/XRT/xrt_sdk/xrt/python'); import pyxrt as xrt
    dev=xrt.device(0); states={}; arena=None; av=None; embedding_bo=None; embedding_view=None; hrun=None; failed=False
    sync=xrt.xclBOSyncDirection; failures=[]; strict_failures=[]; snapshots={}; residual_snapshots={}; checks=0
    try:
        for family,name in [('attention','first'),('mlp','mlp')]:
            xb=xrt.xclbin(str((stem if family=='mlp' else out/name)/'final.xclbin')); dev.register_xclbin(xb)
            ctx=xrt.hw_context(dev,xb.get_uuid()); kernel=xrt.kernel(ctx,xb.get_kernels()[0].get_name())
            states[family]=dict(xb=xb,ctx=ctx,kernel=kernel,instructions={},groups=[kernel.group_id(3),kernel.group_id(4)])
        groups=[g for state in states.values() for g in state['groups']]
        if len({g&0xffff for g in groups})!=1: raise ValueError('incompatible activation arena banks')
        prepared=[]
        for stage in meta['stages']:
            family=stage['family']; data=(out/stage['instructions']).read_bytes(); changes=[]
            if family=='attention' and stage['layer']==0:
                data=(out/'repeat/insts.bin').read_bytes()
                data,relocations,used=c.l.relocate(data,c.l.bindings(pl,1,spec[1]['mode']),c.l.bindings(pl,0,spec[0]['mode']))
                if used!={'qkv','projection','affine','coeff'}: raise ValueError('first packed QKV constant coverage')
            if family=='attention': data,changes=c.txn.patch(data,meta['artifacts']['first']['audits'],valid,spec[stage['layer']]['mode']=='local')
            words=np.frombuffer(data,np.uint32); kernel=states[family]['kernel']
            ib=xrt.bo(dev,words.nbytes,xrt.bo.cacheable,kernel.group_id(1)); ib.write(data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
            key=(stage['layer'],family); states[family]['instructions'][key]=(ib,words.nbytes)
            prepared.append(dict(layer=stage['layer'],family=family,sha256=hashlib.sha256(data).hexdigest(),control_changes=changes))
        stem_data,token_offsets=s.prepare((stem/'insts.bin').read_bytes(),sm['tokens'],tokens)
        stem_data,stem_relocations=relocate_output(stem_data,sm['target'],off['mlp_norm'],pl['words'])
        ib=xrt.bo(dev,len(stem_data),xrt.bo.cacheable,states['mlp']['kernel'].group_id(1)); ib.write(stem_data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
        states['mlp']['instructions']['stem']=(ib,len(stem_data))
        if head is not None:
            head_data=head.prepare((head_folder/'insts.bin').read_bytes(),pl,qtype)
            ib=xrt.bo(dev,len(head_data),xrt.bo.cacheable,states['mlp']['kernel'].group_id(1)); ib.write(head_data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
            states['mlp']['instructions']['head_pre']=(ib,len(head_data))
        if head_attention is not None:
            attention_data,attention_changes=c.txn.patch((attention_folder/'insts.bin').read_bytes(),attention_meta['audits'],valid,False)
            ib=xrt.bo(dev,len(attention_data),xrt.bo.cacheable,states['attention']['kernel'].group_id(1)); ib.write(attention_data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
            states['attention']['instructions']['head_attention']=(ib,len(attention_data))
        if head_post is not None:
            post_data=(post_folder/'insts.bin').read_bytes()
            ib=xrt.bo(dev,len(post_data),xrt.bo.cacheable,states['mlp']['kernel'].group_id(1)); ib.write(post_data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
            states['mlp']['instructions']['head_post']=(ib,len(post_data))
        if head_second is not None:
            for name in ('pre','post'):
                second_data=(second_folder/name/'insts.bin').read_bytes()
                ib=xrt.bo(dev,len(second_data),xrt.bo.cacheable,states['mlp']['kernel'].group_id(1)); ib.write(second_data,0); ib.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                states['mlp']['instructions']['second_'+name]=(ib,len(second_data))
        del xb,ctx,kernel,ib
        arena=xrt.bo(dev,pl['words']*4,xrt.bo.host_only,groups[0]); av=np.frombuffer(arena.map(),np.uint32)
        embedding_bo=xrt.bo(dev,s.INPUT_WORDS*4,xrt.bo.host_only,states['mlp']['kernel'].group_id(3)); embedding_view=np.frombuffer(embedding_bo.map(),np.uint32)
        print('EMBEDDED_RUN '+json.dumps(dict(build=meta,stem_build=sm,runner_sha256=p.attention.sha(Path(__file__)),weights=weight_hashes,
          helpers={str(Path(x.__file__).relative_to(p.ROOT)).replace('\\','/'):p.attention.sha(Path(x.__file__)) for x in (cr,sr,mask_reference)},
          table_sha256=hashlib.sha256(embedding_table.tobytes()).hexdigest(),gamma_sha256=hashlib.sha256(embedding_gamma.tobytes()).hexdigest(),
          key=token_key,valid=valid,tokens_file_sha256=p.attention.sha(tokens_file),tokens_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),prepared=prepared,stem_output_relocations=stem_relocations,
          allocation_group=groups[0],argument_groups=groups,contexts_created=2,
          interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
          head_build=head_meta,head_attention_build=attention_meta,head_post_build=post_meta,head_post_weights=post_hashes if head_post is not None else None,
          second_head_build=second_meta,second_head_weights=second_hashes if head_second is not None else None,
          head_weights=head_hashes if head is not None else None,qtype=qtype if head is not None else None,
          host_work='token/type-address DMA scheduling, no embedding activation gather',scope='real embeddings and encoder; optional first head prelude; full decisions absent; correctness only')),flush=True)
        prefixes=sorted({0,1,2,count})
        if head is not None: prefixes.append(count+1)
        if head_attention is not None: prefixes.append(count+2)
        if head_post is not None: prefixes.append(count+3)
        if head_second is not None: prefixes.extend((count+4,count+5,count+6))
        for prefix in prefixes:
            previous=None; previous_regions=None
            for fill in (0x7fc12345,0x7fc54321):
                data=np.full(pl['words'],fill,np.uint32); cr.initialize_constants(data,pl,parameters,tables,fill)
                if head is not None: head.initialize(data,pl,head_values,fill)
                if head_attention is not None: head_attention.initialize(data,pl)
                if head_post is not None: head_post.initialize(data,pl,post_values,fill)
                if head_second is not None: head_second.initialize(data,pl,second_values,fill)
                av[:]=data; arena.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                embedding_packet=sr.constant_packet(embedding_table,embedding_gamma,fill)
                embedding_view[:]=embedding_packet; embedding_bo.sync(sync.XCL_BO_SYNC_BO_TO_DEVICE)
                # Every constant and instruction is resident before this loop.
                # No host tensor calculation or activation transfer occurs here.
                ib,nb=states['mlp']['instructions']['stem']
                failed=True; hrun=states['mlp']['kernel'](3,ib,nb,embedding_bo,arena)
                if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('stem dispatch failed; no retry')
                failed=False; hrun=None
                for stage in meta['stages'][:prefix*2]:
                    family=stage['family']; key=(stage['layer'],family)
                    ib,nb=states[family]['instructions'][key]
                    failed=True; hrun=states[family]['kernel'](3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('chain dispatch failed; no retry')
                    failed=False; hrun=None
                if prefix>count:
                    ib,nb=states['mlp']['instructions']['head_pre']
                    failed=True; hrun=states['mlp']['kernel'](3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('head prelude dispatch failed; no retry')
                    failed=False; hrun=None
                if prefix>count+1:
                    ib,nb=states['attention']['instructions']['head_attention']
                    failed=True; hrun=states['attention']['kernel'](3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('head attention dispatch failed; no retry')
                    failed=False; hrun=None
                if prefix>count+2:
                    ib,nb=states['mlp']['instructions']['head_post']
                    failed=True; hrun=states['mlp']['kernel'](3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('head post dispatch failed; no retry')
                    failed=False; hrun=None
                for threshold,family,key in ((count+3,'mlp','second_pre'),(count+4,'attention','head_attention'),(count+5,'mlp','second_post')):
                    if prefix<=threshold: continue
                    ib,nb=states[family]['instructions'][key]
                    failed=True; hrun=states[family]['kernel'](3,ib,nb,arena,arena)
                    if hrun.wait(15000)!=xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED: raise RuntimeError('second head dispatch failed; no retry')
                    failed=False; hrun=None
                arena.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE); observed=av.copy()
                final=observed[off['mlp_norm']:off['mlp_norm']+sizes['mlp_norm']].reshape(512,p.OUT)
                if not np.all(final[:,1024:1040]==b.r.mlp.POISON) or not np.all(final[:,2064:]==b.r.mlp.POISON): raise ValueError('final norm guards')
                y=final[:,:1024].copy().view(np.float32); residual=final[:,1040:2064].copy().view(np.float32)
                if previous is not None and not np.array_equal(final,previous): raise ValueError('chain poison dependence')
                previous=final.copy()
                if prefix>count+5:
                    verdicts=head_second.verify_post(observed,pl,second_values,residual_snapshots[count+3])
                    verdicts['chain_y']=p.attention.numeric(y,second_ref[1]); verdicts['chain_residual']=p.attention.numeric(residual,second_ref[0])
                elif prefix>count+3:
                    verdicts=head_second.verify_pre(observed,pl,second_values,snapshots[count+3])
                    if prefix>count+4: verdicts.update(head_attention.verify(observed,pl,valid))
                    verdicts['chain_y']=p.attention.numeric(y,post_ref[1]); verdicts['chain_residual']=p.attention.numeric(residual,post_ref[0])
                elif prefix>count+2:
                    verdicts=head_post.verify(observed,pl,post_values,snapshots[count],head_values[0][qtype])
                    verdicts['chain_y']=p.attention.numeric(y,post_ref[1])
                    verdicts['chain_residual']=p.attention.numeric(residual,post_ref[0])
                elif prefix>count:
                    verdicts=head.verify(observed,pl,head_values,qtype,snapshots[count])
                    composed_r,composed_y=p.norm.oracle(refs[-1][1],np.broadcast_to(head_values[0][qtype],refs[-1][1].shape),head_values[1],head_values[2])
                    verdicts['chain_y']=p.attention.numeric(y,composed_y)
                    verdicts['chain_residual']=p.attention.numeric(residual,composed_r)
                    if prefix>count+1: verdicts.update(head_attention.verify(observed,pl,valid))
                elif prefix:
                    norm_input=observed[off['norm']:off['norm']+sizes['norm']].reshape(512,p.OUT)
                    wo=norm_input[:,:1040].reshape(512,16,65)[:,:,:64].copy().view(np.float32).reshape(512,1024)
                    prior=norm_input[:,1040:2064].copy().view(np.float32)
                    gamma,beta=parameters[prefix-1]['affine'][1]; direct_r,direct_y=p.norm.oracle(wo,prior,gamma,beta)
                    geglu=b.unpack_operand(observed[off['mlp_geglu']:off['mlp_geglu']+sizes['mlp_geglu']].reshape(-1,p.OUT),41)
                    mlp_input=b.unpack_operand(observed[off['packed_norm']:off['packed_norm']+sizes['packed_norm']].reshape(-1,p.OUT),16)
                    wi=b.r.packet_matrix(observed[off['mlp_wi']:off['mlp_wi']+sizes['mlp_wi']].reshape(-1,p.OUT),5248)
                    direct_wi=mlp_input.astype(np.float64)@parameters[prefix-1]['weights'][2].astype(np.float64)
                    direct_geglu=b.gelu(direct_wi[:,:2624])*direct_wi[:,2624:]
                    direct_wo=geglu.astype(np.float64)@parameters[prefix-1]['weights'][3].astype(np.float64)
                    strict_wo=p.attention.rounded(direct_geglu).astype(np.float64)@parameters[prefix-1]['weights'][3].astype(np.float64)
                    verdicts=dict(final_norm=b.r.norm_verdict(residual,y,direct_r,direct_y),
                      final_wi=b.numeric(wi,direct_wi,.0002,.0002,.00001),final_geglu=b.numeric(geglu,direct_geglu,.000002,.0078125,.004),
                      final_wo=b.numeric(wo,direct_wo,.0002,.0002,.00001),retained_mlp_strict=b.numeric(wo,strict_wo,.0002,.002,.002),
                      chain_y=p.attention.numeric(y,refs[prefix-1][1]),chain_residual=p.attention.numeric(residual,refs[prefix-1][0]))
                else:
                    exact=bool(np.array_equal(residual.view(np.uint32),embedding_reference.view(np.uint32)))
                    verdicts=dict(stem_y=b.numeric(y,stem_reference,p.norm.ATOL,p.norm.RTOL,p.norm.L2),stem_residual=dict(ok=exact),
                      chain_y=b.numeric(y,stem_reference,p.norm.ATOL,p.norm.RTOL,p.norm.L2),chain_residual=dict(ok=exact))
                if prefix in (1,2):
                    projected=b.r.packet_matrix(observed[off['projection']:off['projection']+sizes['projection']].reshape(-1,p.OUT),3072)
                    independent=p.attention.rounded(snapshots[prefix-1]).astype(np.float64)@parameters[prefix-1]['weights'][0].astype(np.float64)
                    verdicts['handoff_qkv']=b.numeric(projected,independent,.0002,.0002,.00001)
                mutable=np.zeros(pl['words'],bool)
                fields=('projection','queries','keys','kv','attention','packed_attention','norm','packed_norm','mlp_wi','mlp_geglu','mlp_norm') if prefix else ('mlp_norm',)
                if prefix>count: fields=fields+('norm_input',)
                if prefix>count+2: fields=fields+('head_out','head_relu')
                for field in fields:
                    start=off[field]; mutable[start:start+sizes[field]]=True
                current_regions={field:hashlib.sha256(observed[off[field]:off[field]+sizes[field]]).hexdigest() for field in fields}
                if previous_regions is not None and current_regions!=previous_regions: raise ValueError('intermediate poison dependence')
                previous_regions=current_regions
                if not np.array_equal(observed[~mutable],data[~mutable]): raise ValueError('constant, input or arena border overwritten')
                embedding_bo.sync(sync.XCL_BO_SYNC_BO_FROM_DEVICE)
                if not np.array_equal(embedding_view,embedding_packet): raise ValueError('embedding constants modified')
                if not all(v['ok'] for k,v in verdicts.items() if k not in ('chain_y','chain_residual','retained_mlp_strict')): raise ValueError('direct operator/handoff gate')
                print('EMBEDDED_CHECK '+json.dumps(dict(prefix=prefix,fill=fill,dispatches=min(prefix,count)*2+1+max(prefix-count,0),verdicts=verdicts)),flush=True)
                if 0<prefix<=count and not verdicts['retained_mlp_strict']['ok']: strict_failures.append(dict(prefix=prefix,fill=fill,**verdicts['retained_mlp_strict']))
                if not verdicts['chain_y']['ok'] or not verdicts['chain_residual']['ok']:
                    failures.append(dict(prefix=prefix,fill=fill,verdicts=verdicts))
                    if not collect: raise ValueError('composed encoder-prefix gate')
                snapshots[prefix]=y.copy(); residual_snapshots[prefix]=residual.copy(); checks+=1
        print('EMBEDDED_RESULT '+json.dumps(dict(checks=checks,layers=count,prefixes=prefixes,composition_failures=failures,strict_mlp_failures=strict_failures,
          contexts_created=2,interstage_host_reads=0,interstage_host_writes=0,interstage_host_syncs=0,
          scope='embedding, complete encoder and both head attention/MLP blocks' if head_second is not None else 'embedding and encoder with optional head prefixes',
          remaining='scorer/action tail, both outputs, frozen accuracy and eligible performance unverified')),flush=True)
        if failures or strict_failures: raise ValueError('retained prefix gates failed; thresholds unchanged')
    finally:
        if failed: globals()['_quarantined']=(dev,states,arena,av,embedding_bo,embedding_view,hrun)
        else:
            hrun=None; ib=None; av=None; arena=None; embedding_view=None; embedding_bo=None; state=None; kernel=None; ctx=None; xb=None
            for family in reversed(tuple(states)): states[family]['instructions'].clear()
            for family in reversed(tuple(states)): states[family]['kernel']=None
            for family in reversed(tuple(states)): states[family]['ctx']=None
            for family in reversed(tuple(states)): states[family]['xb']=None
            states.clear(); dev=None; gc.collect(); print('EMBEDDED_IDLE '+str(idle()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--stem',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True); ap.add_argument('--tokens',type=Path,required=True); ap.add_argument('--key',default='fixture')
    ap.add_argument('--head-pre',type=Path)
    ap.add_argument('--head-attention',type=Path)
    ap.add_argument('--head-post',type=Path)
    ap.add_argument('--head-second',type=Path)
    ap.add_argument('--collect-composition-failures',action='store_true'); args=ap.parse_args()
    run(args.out,args.stem,args.model,args.tokens,args.key,args.collect_composition_failures,args.head_pre,args.head_attention,args.head_post,args.head_second)
