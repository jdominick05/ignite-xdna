# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compiled lock and one-state control checks for smaller queries with KV depth two."""
from collections import Counter
import inspect
import json
import re

import window_lock_check as locks
import window_core_check as control


def counts(folder):
    text=locks.clean((folder/'prj/input_with_addresses.mlir').read_text())
    buffers=Counter(name.rsplit('_buff_',1)[0] for name in re.findall(r'%(\w+_buff_\d+) = aie.buffer',text))
    q=[buffers[f'qF_{core}_cons'] for core in range(16)]
    kv=[buffers[f'kvF_{core//4}_{core%4}_cons'] for core in range(16)]
    if len(set(q))!=1 or q[0]!=3 or kv!=[2]*16:raise ValueError('small lookup physical query/KV capacities')
    states=re.findall(r'%attn_state_\d+ = aie.buffer[^\n]*memref<(\d+)xi32>',text)
    if states!=['2048']*16:raise ValueError('small lookup must have one output state')
    return text,q,kv


def analyze(folder):
    text,q,kv=counts(folder)
    # Reuse the exact compiled-register decoder. Its old terminal BD census was
    # fixed to two Q/two KV objects; account explicitly for the added Q objects.
    expected_bds=192+sum(depth-2 for depth in q)+sum(depth-2 for depth in kv)
    source=inspect.getsource(locks.analyze)
    if source.count('if bd_count!=192:')!=1:raise ValueError('lock decoder contract changed')
    namespace=dict(locks.analyze.__globals__,expected_bds=expected_bds)
    exec(compile(source.replace('if bd_count!=192:','if bd_count!=expected_bds:'),str(locks.__file__),'exec'),namespace)
    row=namespace['analyze'](folder,True)
    return row


def relation(original,small):
    a=locks.analyze(original,False);b=analyze(small)
    if a['roles']!=b['roles'] or a['flows']!=b['flows'] or set(a['locks'])!=set(b['locks']):
        raise ValueError('small lookup DMA roles/routes changed')
    changed=0
    for name,old in a['locks'].items():
        new=b['locks'][name]
        expected=3 if re.fullmatch(r'qF_\d+_cons_prod_lock_0',name) else old[3]
        if old[:3]!=new[:3] or old[4]!=new[4] or new[3]!=expected:raise ValueError('unaccounted small lookup lock change')
        if expected!=old[3]:changed+=1
    if changed!=16:raise ValueError('small lookup query capacity coverage')
    return dict(hardware_locks_checked=b['hardware_locks_checked'],compiled_bd_lock_fields=b['dma_descriptors'],
        physical_query_objects=3,physical_kv_objects=2,query_capacity_changes=changed,
        original_ir_sha256=a['ir_sha256'],small_ir_sha256=b['ir_sha256'],
        per_object_dma_roles_and_routes_identical=True,original_kv_capacity_preserved=True)


def ledger(folder,vector=False,cycles=6):
    text,q,kv=counts(folder);rows=[]
    for core in range(4):
        row=control.trace(text,core,vector,cycles,modes=[1,3] if vector else None)
        if vector:
            weight_index=0
            if len(row['calls'])!=2*cycles:raise ValueError('small vector helper count')
            for index,(name,args) in enumerate(row['calls']):
                mode=1 if index%2==0 else 3;selected=weight_index+(core if mode==3 else 0);ring=index*2%3
                wanted=[f'%qF_{core}_cons_buff_{ring}',f'%qF_{core}_cons_buff_{(ring+1)%3}',
                    f'%kvF_0_{core}_cons_buff_{selected%2}',f'%outF_{core}_buff_0']
                if mode==3:wanted.append(2143363909)
                if name!=('command_norm_split' if mode==1 else 'chain_geglu_split') or args!=wanted or row['call_tokens'][index]['kv']-1!=selected:
                    raise ValueError('small vector paired query/weight operand order')
                weight_index+=4 if mode==3 else 1
        else:
            gemm=[args for name,args in row['calls'] if name=='chain_gemm_step']
            finish=[args for name,args in row['calls'] if name=='chain_finish']
            if len(row['calls'])!=3*cycles or [args[-1] for args in gemm]!=[1,0]*cycles or [args[-1] for args in finish]!=[0]*cycles:
                raise ValueError('small GEMM reset/helper/finish sequence')
            for index,args in enumerate(gemm):
                wanted=[f'%qF_{core}_cons_buff_{index%3}',f'%kvF_0_{core}_cons_buff_{index%2}',f'%attn_state_{core}']
                if args[:3]!=wanted:raise ValueError('small GEMM physical query/KV/state ring')
        rows.append(dict(core=[0,core+2],tokens=row['tokens'],helper_calls=len(row['calls']),parked_on_next_query=row['parked_on_next_query']))
    return dict(vector=vector,cycles=cycles,cores=rows,
        scope='lowered control/token ledger; complete-object DMA delivery/drain assumed; numerical/global-deadlock proof excluded')
