# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline compiled FIFO/lock relations for the bounded GEMM window primitive."""
from collections import Counter,defaultdict
import hashlib
import re

import pack_primitive as packing


def clean(text):return re.sub(r' loc\([^)]*\)','',text)


def verify_bd_lock_word(state,owner,bd_id,acquire,release):
    """Decode AIE-ML core/MemTile BD fields, not a text-only lock census.

    Field definitions: aie-rt driver xaiemlgbl_params.h DMA_BD0_{5,7}.
    AIERT.cpp localizes MemTile DMA IDs with getLockLocalBaseIndex;
    AIETargetModel.cpp selects the internal bank (64 locks) for local locks.
    AcquireGreaterEqual(1) is the signed seven-bit value -1.
    """
    col,row=owner
    if row<1 or acquire[:2]!=owner or release[:2]!=owner:
        raise ValueError('unsupported nonlocal DMA lock ownership')
    if row==1:
        address=(col<<25)|(row<<20)|(0xa0000+32*bd_id+28)
        shifts=(0,8,15,16,24,31);id_mask=255;local_base=64
    else:
        address=(col<<25)|(row<<20)|(0x1d000+32*bd_id+20)
        shifts=(0,5,12,13,18,25);id_mask=15;local_base=0
    if address not in state:raise ValueError('missing compiled BD lock register')
    word=state[address]
    actual=tuple((word>>shift)&mask for shift,mask in zip(shifts,(id_mask,127,1,id_mask,127,1)))
    wanted=(acquire[2]+local_base,127,1,release[2]+local_base,1,1)
    if actual!=wanted:raise ValueError('compiled BD lock fields differ from lowered handshake')


def analyze(folder,window):
    ir=folder/'prj/input_with_addresses.mlir';text=clean(ir.read_text())
    tiles={name:(int(col),int(row)) for name,col,row in re.findall(r'%(\w+) = aie.tile\((\d+), (\d+)\)',text)}
    locks={name:(*tiles[tile],int(index),int(initial),symbol) for name,tile,index,initial,symbol in re.findall(
        r'%(\w+) = aie.lock\(%(\w+), (\d+)\) \{init = (-?\d+) : i32, sym_name = "([^"]+)"',text)}
    if len(locks)!=192:raise ValueError('incomplete primitive lock declarations')
    buffers=Counter()
    owners={name:tiles[tile] for name,tile in re.findall(r'%(\w+) = aie.buffer\(%(\w+)\)',text)}
    for name in re.findall(r'%(\w+_buff_\d+) = aie.buffer',text):buffers[name.rsplit('_buff_',1)[0]]+=1
    state={}
    cdo=folder/'prj/cdo_main/main_aie_cdo_init.bin'
    for address,mask,value in packing.cdo_writes(cdo.read_bytes()):
        state[address]=(state.get(address,0)&(~mask&0xffffffff))|(value&mask)
    checked=0
    for name,(col,row,index,initial,symbol) in locks.items():
        if name!=symbol:raise ValueError('lock SSA/symbol differs')
        if row==0:
            if initial:raise ValueError('unexpected shim virtual lock state')
            continue
        address=(col<<25)|(row<<20)|(0xc0000 if row==1 else 0x1f000)+16*index
        if state.get(address)!=initial:raise ValueError('CDO lock initialization differs from lowered declaration: '+name)
        if '_prod_lock_' in name:
            base=name.rsplit('_prod_lock_',1)[0]
            if initial!=buffers[base] or initial<=0:raise ValueError('free lock capacity differs from physical FIFO buffers: '+name)
        elif '_cons_lock_' in name and initial!=0:raise ValueError('nonempty FIFO at initialization')
        checked+=1
    constants=defaultdict(set)
    for name,value in re.findall(r'(%[\w-]+) = arith.constant (-?\d+) : i32',text):constants[name].add(int(value))
    roles=set();bd_count=0
    for block in re.split(r'(?m)^\s*\^bb\d+',text):
        if 'aie.dma_bd(' not in block:continue
        bds=re.findall(r'aie.dma_bd\(%(\w+) : memref<(\d+)xi32> offset = (\d+) len = (\d+)\)',block)
        # Basic-block splitting can include a following region's entry operations;
        # every actual BD block must still contain one complete handshake.
        uses=re.findall(r'aie.use_lock\(%(\w+), (AcquireGreaterEqual|Release), (%[\w-]+)\)',block)
        if len(bds)!=1 or len(uses)!=2 or [row[1] for row in uses]!=['AcquireGreaterEqual','Release']:
            raise ValueError('DMA descriptor lacks one acquire/transfer/release handshake')
        name,size,offset,length=bds[0];size=int(size);offset=int(offset);length=int(length)
        packet=(1040 if window else 2064) if name.startswith(('qF_','qL2_')) else 2064 if name.startswith(('kvF_','kvL2_')) else 2080
        if length!=packet or offset%packet or offset+length>size:raise ValueError('DMA cuts a FIFO object')
        for lock,operation,value in uses:
            if lock not in locks or constants[value]!={1}:raise ValueError('DMA lock reference/value differs')
        if block.index('aie.use_lock')>block.index('aie.dma_bd') or block.rindex('aie.use_lock')<block.index('aie.dma_bd'):
            raise ValueError('DMA handshake ordering differs')
        ids=re.findall(r'\{bd_id = (\d+) : i32',block)
        if len(ids)!=1:raise ValueError('missing/ambiguous compiled DMA descriptor ID')
        verify_bd_lock_word(state,owners[name],int(ids[0]),locks[uses[0][0]],locks[uses[1][0]])
        roles.add((name.rsplit('_buff_',1)[0],offset//packet,uses[0][0],uses[1][0]));bd_count+=1
    if bd_count!=192:raise ValueError('incomplete DMA descriptor graph')
    flows=set(re.findall(r'aie.flow\(([^\n]+?)\)',text))
    return dict(locks=locks,roles=roles,flows=flows,hardware_locks_checked=checked,dma_descriptors=bd_count,
        ir_sha256=hashlib.sha256(ir.read_bytes()).hexdigest(),cdo_sha256=hashlib.sha256(cdo.read_bytes()).hexdigest())


def compare(original,window):
    a=analyze(original,False);b=analyze(window,True)
    if a['roles']!=b['roles'] or a['flows']!=b['flows'] or set(a['locks'])!=set(b['locks']):
        raise ValueError('per-object lock roles or physical routes differ')
    capacities=Counter()
    for name,old in a['locks'].items():
        new=b['locks'][name]
        if old[:3]!=new[:3] or old[4]!=new[4]:raise ValueError('lock owner/identity changed')
        expected=old[3]
        if re.fullmatch(r'qF_\d+_cons_prod_lock_0',name):expected=3;capacities['query']+=1
        if re.fullmatch(r'kvF_\d+_\d+_cons_prod_lock_0',name):expected=1;capacities['kv']+=1
        if new[3]!=expected:raise ValueError('unaccounted lock capacity change')
    if capacities!={'query':16,'kv':16}:raise ValueError('capacity coverage differs')
    return dict(original_ir_sha256=a['ir_sha256'],window_ir_sha256=b['ir_sha256'],
        original_cdo_sha256=a['cdo_sha256'],window_cdo_sha256=b['cdo_sha256'],
        hardware_lock_initializations_per_image=a['hardware_locks_checked'],dma_descriptors_per_image=a['dma_descriptors'],
        compiled_bd_lock_fields_checked_per_image=a['dma_descriptors'],
        per_object_dma_handshake_roles_identical=True,physical_routes_identical=True,
        changed_free_capacities=dict(query=3,kv=1),
        scope='DMA and initialization relation; core GEMM control-path/token conservation checked separately')
