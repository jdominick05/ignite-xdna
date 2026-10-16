# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sixteen independent heads, four per column, with MemTile split/join."""
import argparse
import json
from pathlib import Path
import re
import shutil

import probe as p


def harness(packets):
    template=p.harness(packets,32)
    body=template[template.index('    %state ='):template.index('    aie.runtime_sequence')]
    declarations='\n'.join(line for line in body.splitlines() if 'func.func private' in line)
    body='\n'.join(line for line in body.splitlines() if 'func.func private' not in line)
    parts=['module {\n  aie.device(npu1) {',declarations]
    cores=[]
    for col in range(4):
        parts += [f'    %s{col} = aie.tile({col}, 0)',f'    %m{col} = aie.tile({col}, 1)']
        for row in range(4):
            i=col*4+row
            parts.append(f'    %t_{i} = aie.tile({col}, {row+2})')
            core=re.sub(r'%(\w+)',lambda m:f'%{m[1]}_{i}',body)
            core=core.replace(f'%s_{i}',f'%m{col}')
            core=core.replace('@inF',f'@inF_{i}').replace('@outF',f'@outF_{i}')
            core=core.replace('"attn_state"',f'"attn_state_{i}"').replace('"attn_scratch"',f'"attn_scratch_{i}"')
            before,worker=core.split(f'    %core_{i} =',1)
            parts.append(before)
            cores.append(f'    %core_{i} ='+worker)
        parts += [
            f'    aie.objectfifo @inL2_{col}(%s{col}, {{%m{col}}}, 1 : i32) : !aie.objectfifo<memref<{4*p.PACKET}xi32>>',
            f'    aie.objectfifo @outL2_{col}(%m{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{4*p.OUT}xi32>>',
            f'    aie.objectfifo.link [@inL2_{col}] -> ['+', '.join(f'@inF_{col*4+r}' for r in range(4))+']([] ['+', '.join(str(r*p.PACKET) for r in range(4))+'])',
            '    aie.objectfifo.link ['+', '.join(f'@outF_{col*4+r}' for r in range(4))+f'] -> [@outL2_{col}](['+', '.join(str(r*p.OUT) for r in range(4))+'] [])',
        ]
    parts.extend(cores)
    ni=16*packets*p.PACKET; no=16*32*p.OUT
    parts.append(f'    aie.runtime_sequence(%a: memref<{ni}xi32>, %o: memref<{no}xi32>) {{')
    for col in range(4):
        n=4*packets*p.PACKET; m=4*32*p.OUT
        parts += [
            f'      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, {col*n}][1, 1, {n//16}, 16][0, 0, 16, 1]) {{id = 0 : i64, metadata = @inL2_{col}}} : memref<{ni}xi32>',
            f'      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, {col*m}][1, 1, {m//16}, 16][0, 0, 16, 1]) {{id = 1 : i64, metadata = @outL2_{col}, issue_token = true}} : memref<{no}xi32>',
        ]
    for col in range(4):
        parts.append(f'      aiex.npu.dma_wait {{symbol = @outL2_{col}}}')
    return '\n'.join(parts+['    }','  }','}'])+'\n'


def build(out,masks,mode,control,valid=512):
    base=json.loads((control/'build.json').read_text())
    source=p.HERE/base['kernel_source']
    if base['kernel_sha256']!=p.sha(source) or base['implementation']!='lut':
        raise ValueError('need current LUT one-core control')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(control/'kernel.o',out/'kernel.o')
    mask=p.np.load(masks)[f'{mode}_{valid}']
    packets,outputs=p.packet_geometry(mask,16)
    (out/'harness.mlir').write_text(harness(packets//16),encoding='utf-8')
    p.command([p.IRON/'Scripts/aiecc.exe',out/'harness.mlir',f'--peano={p.PEANO}',
               '--get-npu-insts','--npu-insts-name=insts.bin','--get-xclbin','--xclbin-name=final.xclbin',
               f'--tmpdir={out/"prj"}'],out)
    audits=[]
    elfs=sorted((out/'prj').rglob('*.elf'))
    if len(elfs)!=16: raise ValueError(f'expected 16 cores, got {len(elfs)}')
    for elf in elfs:
        col,row=map(int,re.search(r'core_(\d+)_(\d+)\.elf$',elf.name).groups())
        i=col*4+row-2
        symbols=p.command([p.PEANO/'bin/llvm-nm.exe',elf],out)
        if p.software_float_helpers(symbols): raise ValueError('software float in array')
        names={f'attn_state_{i}':p.STATE*4,f'attn_scratch_{i}':p.SCRATCH*4,
               f'inF_{i}_cons_buff_0':p.PACKET*4,f'outF_{i}_buff_0':p.OUT*4}
        names.update({n:1024 for n in ('exp_ilut_ab','exp_ilut_cd','exp_flut_ab','exp_flut_cd')})
        spans=[]
        for name,size in names.items():
            match=re.search(r'^([0-9a-fA-F]+) [A-Za-z] '+name+r'$',symbols,re.M)
            if not match: raise ValueError(f'missing {name}')
            lo=int(match[1],16)-0x70000
            spans.append((lo,lo+size,name))
        spans.sort()
        if any(lo<p.STACK or hi>65536 for lo,hi,_ in spans) or any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError(f'invalid core memory {i}: {spans}')
        sections=p.command([p.PEANO/'bin/llvm-size.exe','-A',elf],out)
        code=sum(int(m[1]) for m in re.findall(r'^(\.(?:text|text\.[^\s]+))\s+(\d+)\s+',sections,re.M))
        if not 0<code<=16384: raise ValueError(f'program size {code}')
        stack=p.command([p.PEANO/'bin/llvm-readobj.exe','--stack-sizes',elf],out)
        frames=[int(x,16) for x in re.findall(r'Size: (0x[0-9a-fA-f]+)',stack)]
        if not frames or sum(frames)+256>p.STACK: raise ValueError('stack too large')
        ld=(out/f'prj/ldScripts_main_core_{col}_{row}.ld.script').read_text()
        if f'. += 0x{p.STACK:X}; /* stack */'.lower() not in ld.lower(): raise ValueError('stack reservation missing')
        audits.append(dict(core=[col,row],code_bytes=code,stack_sizes=frames,placement=spans))
    meta=dict(base,heads=16,cores=16,mode=mode,valid=valid,packets=packets,outputs=outputs,
              core_audits=audits,xclbin_sha256=p.sha(out/'final.xclbin'),
              instructions_sha256=p.sha(out/'insts.bin'),masks_sha256=p.sha(masks))
    p.dump(out/'build.json',meta)
    print(json.dumps(meta),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--masks',type=Path,required=True)
    ap.add_argument('--mode',choices=('global','local'),required=True)
    ap.add_argument('--valid',type=int,choices=(512,193,1,0),default=512)
    ap.add_argument('--control',type=Path,required=True)
    a=ap.parse_args()
    build(a.out.resolve(),a.masks,a.mode,a.control.resolve(),a.valid)
