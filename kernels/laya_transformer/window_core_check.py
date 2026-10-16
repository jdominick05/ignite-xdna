# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed interpreter of lowered GEMM control flow, without numerical kernels.

This checks token conservation for the supplied primitive input, assuming the DMA
can deliver/drain each complete object. It is not a global deadlock or silicon proof.
"""
from collections import Counter
import re

from window_lock_check import clean


SSA=r'%[\w-]+'


def trace(text,core,window,cycles=3,modes=None):
    text=clean(text)
    match=re.search(rf'%core_0_{core+2} = aie.core\([^\n]+\) \{{\n(.*?)\n    \}}',text,re.S)
    if not match:raise ValueError('missing lowered active core')
    blocks={'entry':([],[])};current='entry';switch=[]
    for line in match[1].splitlines():
        line=line.split('//')[0].strip()
        if not line:continue
        label=re.match(r'\^(bb\d+)(?:\((.*)\))?:',line)
        if label:
            current=label[1];blocks[current]=(re.findall(r'(%[\w-]+):',label[2] or ''),[])
        elif switch:
            switch.append(line)
            if line==']':blocks[current][1].append(' '.join(switch));switch=[]
        elif line.startswith('cf.switch'):switch=[line]
        else:blocks[current][1].append(line)
    if switch:raise ValueError('unterminated switch')
    env={};memory={};events=[];calls=[];call_tokens=[];held=Counter();counts=Counter()
    vector=modes is not None
    if vector and (not modes or any(mode not in (1,3) for mode in modes)):
        raise ValueError('unsupported vector mode schedule')
    schedule=list(modes)*cycles if vector else [2,2]*cycles
    completed_commands=0
    supplied=({'q':len(schedule)*(2 if window else 1),'kv':sum(4 if mode==3 else 1 for mode in schedule),'out':len(schedule)}
              if vector else {'q':2*cycles,'kv':(4 if window else 2)*cycles,'out':(2 if window else 1)*cycles})
    def val(name):
        if name in env:return env[name]
        if re.fullmatch(r'%(?:qF_|kvF_|attn_state_|outF_)\w+',name):return name
        raise ValueError('undefined or unsupported SSA operand '+name)
    def jump(target):
        name=re.match(r'\^(bb\d+)',target)[1]
        args=re.findall(SSA,target.split(':',1)[0])
        values=[val(arg) for arg in args]
        if len(values)!=len(blocks[name][0]):raise ValueError('branch arity differs')
        env.update(zip(blocks[name][0],values));return name
    block='entry';steps=0
    while steps<200000:
        branched=False
        for line in blocks[block][1]:
            steps+=1
            if steps>=200000:raise ValueError('lowered control path exceeds finite bound')
            assignment=re.match(r'('+SSA+r') = (.*)',line)
            if assignment:
                dst,op=assignment.groups();args=re.findall(SSA,op)
                if op.startswith('arith.constant'):value=int(re.search(r'constant (-?\d+)',op)[1])
                elif op.startswith(('arith.index_cast ','arith.extui ')):value=val(args[0])
                elif op.startswith('arith.select '):value=val(args[1] if val(args[0]) else args[2])
                elif op.startswith(('arith.addi ','arith.andi ','arith.ori ')):
                    a,b=map(val,args);value=a+b if op.startswith('arith.addi') else a&b if op.startswith('arith.andi') else a|b
                elif op.startswith('arith.cmpi '):
                    pred=re.search(r'cmpi (\w+)',op)[1];a,b=map(val,args)
                    if pred not in ('eq','ne','slt','sge'):raise ValueError('unsupported comparison')
                    value=int({'eq':a==b,'ne':a!=b,'slt':a<b,'sge':a>=b}[pred])
                elif op.startswith('memref.load %chain_control_'):
                    field=val(args[1]);value=memory.get(field,{0:schedule[completed_commands],1:1 if vector else 2,
                        2:2143363909,3:core,6:1 if vector else 2 if window else 1}.get(field))
                    if value is None:raise ValueError('unknown RTP field')
                else:raise ValueError('unsupported executed SSA operation '+op)
                env[dst]=value
            elif line.startswith('memref.store '):
                args=re.findall(SSA,line)
                if args[1].startswith('%chain_control_'):memory[val(args[2])]=val(args[0])
                elif args[1].startswith('%outF_'):val(args[0]);val(args[2])
                else:raise ValueError('unsupported executed store')
            elif line.startswith('aie.use_lock'):
                lock,operation,amount=re.search(r'aie.use_lock\(('+SSA+r'), (\w+), ('+SSA+r')\)',line).groups()
                amount=val(amount)
                kind='q' if lock.startswith('%qF_') else 'kv' if lock.startswith('%kvF_') else 'out' if lock.startswith('%outF_') else None
                if kind is None:raise ValueError('unexpected core lock')
                acquire=operation=='AcquireGreaterEqual'
                allowed=2 if vector and window and kind=='q' and operation=='Release' else 1
                if amount!=allowed:raise ValueError('core lock amount differs')
                wanted_suffix='_prod_lock_0' if (kind=='out')==acquire else '_cons_lock_0'
                if not lock.endswith(wanted_suffix):raise ValueError('core uses wrong lock role')
                if acquire:
                    if counts[kind]>=supplied[kind]:
                        if kind!='q' or any(held.values()):raise ValueError('core exhausts nonquery token or retains object')
                        if counts!=Counter(supplied):raise ValueError('core parks before consuming/producing expected objects')
                        if memory.get(15)!=0:raise ValueError('K counter does not reset')
                        return dict(tokens=dict(counts),calls=calls,call_tokens=call_tokens,events=events,steps=steps,parked_on_next_query=True)
                    limit=2 if vector and window and kind=='q' else 1
                    if held[kind]>=limit:raise ValueError('core acquires another held object')
                    held[kind]+=1;counts[kind]+=1
                else:
                    if operation!='Release' or held[kind]!=amount:raise ValueError('core releases unheld object')
                    held[kind]-=amount
                    if kind=='q':completed_commands+=1
                events.append((kind,'acquire' if acquire else 'release'))
            elif line.startswith('func.call '):
                name,args=re.search(r'@([\w]+)\((.*?)\)',line).groups();args=[val(x) for x in re.findall(SSA,args)]
                allowed_helpers=('command_norm','command_norm_split','chain_geglu','chain_geglu_split') if vector else (
                    'chain_gemm_step','window_gemm_step','chain_finish','window_finish')
                if name not in allowed_helpers:
                    raise ValueError('unexpected numerical helper on GEMM path')
                if (held['q']!=(2 if vector and window else 1) or
                    (vector and (held['kv']!=1 or held['out']!=1)) or
                    not held['kv' if name.endswith('gemm_step') else 'out']):
                    raise ValueError('helper called without acquired operand/output')
                calls.append((name,args))
                call_tokens.append(dict(counts))
            elif line.startswith('cf.br '):block=jump(line[6:]);branched=True;break
            elif line.startswith('cf.cond_br '):
                cond,left,right=re.match(r'cf.cond_br ('+SSA+r'), (.*?), (.*)',line).groups()
                block=jump(left if val(cond) else right);branched=True;break
            elif line.startswith('cf.switch '):
                operand=re.search(SSA,line)[0]
                routes=dict((key,target) for key,target in re.findall(r'(default|-?\d+): (\^bb\d+(?:\([^)]*\))?)',line))
                block=jump(routes.get(str(val(operand)),routes['default']));branched=True;break
            else:raise ValueError('unsupported executed control operation '+line)
        if not branched:raise ValueError('control block lacks supported terminator')
    raise ValueError('control path exceeds finite bound')


def verify_vector(text,window,cycles=3):
    rows=[]
    for core in range(4):
        row=trace(text,core,window,cycles,modes=[1,3]);weight_index=0
        if len(row['calls'])!=2*cycles:raise ValueError('vector helper count differs')
        for index,(name,args) in enumerate(row['calls']):
            mode=1 if index%2==0 else 3
            expected_name=('command_norm' if mode==1 else 'chain_geglu')+('_split' if window else '')
            ring=index*(2 if window else 1)%(3 if window else 2)
            query=[f'%qF_{core}_cons_buff_{ring}']
            if window:query.append(f'%qF_{core}_cons_buff_{(ring+1)%3}')
            selected=weight_index+(core if mode==3 else 0)
            wanted=query+[f'%kvF_0_{core}_cons_buff_{0 if window else selected%2}',f'%outF_{core}_buff_0']
            if mode==3:wanted.append(2143363909)
            if (name!=expected_name or args!=wanted or row['call_tokens'][index]['kv']-1!=selected):
                raise ValueError('vector helper/paired-query ring/selected value operand differs')
            weight_index+=4 if mode==3 else 1
        rows.append(dict(core=[0,core+2],tokens=row['tokens'],helper_calls=len(row['calls']),parked_on_next_query=True))
    return dict(cores=rows,repeated_norm_then_geglu_cycles=cycles,paired_query_ring_and_selected_weight_checked=True,
        scope='lowered vector control only; DMA delivery/drain assumed; numerical helpers and global deadlock proof excluded')


def verify(text,window,cycles=3):
    rows=[]
    for core in range(4):
        row=trace(text,core,window,cycles)
        gemm=[args for name,args in row['calls'] if name.endswith('gemm_step')]
        finish=[args for name,args in row['calls'] if name.endswith('finish')]
        resets=[args[-2] if window else args[-1] for args in gemm]
        slots=[args[-1] for args in gemm] if window else [0]*len(gemm)
        if resets!=([1,1,0,0] if window else [1,0])*cycles or slots!=([0,1,0,1] if window else [0,0])*cycles:
            raise ValueError('GEMM reset or slot sequence differs')
        if [args[-1] for args in finish]!=( [0,1] if window else [0])*cycles:
            raise ValueError('finish slot/activation differs')
        if window and any(args[-2]!=0 for args in finish):
            raise ValueError('unexpected activation')
        width=2 if window else 1
        for index,args in enumerate(gemm):
            query=f'%qF_{core}_cons_buff_{(index//width)%(3 if window else 2)}'
            weight=f'%kvF_0_{core}_cons_buff_{0 if window else index%2}'
            if args[:3]!=[query,weight,f'%attn_state_{core}']:
                raise ValueError('query retention, physical buffer ring or state owner differs')
        expected_names=Counter({'window_gemm_step':4*cycles,'window_finish':2*cycles} if window else
                               {'chain_gemm_step':2*cycles,'chain_finish':cycles})
        if Counter(name for name,args in row['calls'])!=expected_names:raise ValueError('helper identity differs')
        rows.append(dict(core=[0,core+2],tokens=row['tokens'],gemm_reset_sequence=resets,gemm_slot_sequence=slots,
            finish_count=len(finish),parked_on_next_query=row['parked_on_next_query']))
    return dict(cores=rows,repeated_stream_cycles=cycles,query_and_kv_physical_rings_checked=True,
        scope='lowered core control/token ledger only; complete-object DMA delivery/drain assumed; numerical and global deadlock proof excluded')
