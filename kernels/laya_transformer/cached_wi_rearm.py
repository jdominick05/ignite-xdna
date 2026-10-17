# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compile and prove five finite-task restarts, preserving the qualified image."""
import argparse
import json
from pathlib import Path

import cached_wi_checks as checks
import instructions as decoder

cache=checks.cache
p=cache.p
RESTARTS=((0xa0604,0),(0xa0644,0x14000a),(0xa064c,0x14001e),
          (0xa0654,0x14000d),(0xa065c,0x140021))


def provenance(parent,folder):
    meta=json.loads((folder/'build.json').read_text(encoding='utf-8'))
    previous=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    for name,wanted in previous['cdo'].items():
        if p.attention.sha(parent/'prj/cdo_main'/name)!=wanted:
            raise ValueError('qualified parent CDO changed')
    for field in ('resident_elfs','inherited_linker_layouts'):
        for name,wanted in meta[field].items():
            if p.attention.sha(folder/'prj'/name)!=wanted:
                raise ValueError('qualified resident provenance changed')
    return checks.verify(parent,folder)


def harness(text):
    marker='    aie.runtime_sequence('
    start=text.index(marker);left=text.index('{',start)+1
    writes=[]
    for index,(address,value) in enumerate(RESTARTS):
        writes.extend((f'      %cache_rearm_a{index} = arith.constant {address} : i32',
            f'      %cache_rearm_v{index} = arith.constant {value} : i32',
            f'      aiex.npu.write32(%cache_rearm_a{index}, %cache_rearm_v{index}) '
            '{column = 0 : i32, row = 1 : i32} : i32, i32'))
    return text[:left]+'\n'+'\n'.join(writes)+text[left:]


def instruction_proof(cold,warm):
    old=decoder.parse(cold);new=decoder.parse(warm)
    expected=[(0x100000+address,value) for address,value in RESTARTS]
    if len(new)!=len(old)+len(expected):raise ValueError('restart operation census')
    for op,(address,value) in zip(new,expected):
        if (op['op'],op['size'],op.get('addr'),op.get('val'),op.get('val_low'))!=('WRITE',24,address,value,0):
            raise ValueError('unauthorized restart write or order')
    def records(data,ops):return [data[op['offset']:op['offset']+op['size']] for op in ops]
    if records(cold,old)!=records(warm,new[len(expected):]):
        raise ValueError('warm instructions change cold semantics')
    # Header differs only in record count and total byte length; all trailing
    # padding has already been validated by the authoritative decoder.
    if cold[:8]!=warm[:8]:raise ValueError('restart target header changed')
    return dict(restart_writes=len(expected),unchanged_cold_operations=len(old),
        all_other_instruction_records_bit_identical=True,
        scope='finite MemTile tasks only; cold first, warm only after completed wait; no core FIFO reset')


def build(parent,folder,out):
    proof=provenance(parent,folder)
    out.mkdir(parents=True,exist_ok=False)
    (out/'harness.mlir').write_text(harness((folder/'harness.mlir').read_text(encoding='utf-8')),encoding='utf-8')
    p.attention.command([p.attention.IRON/'Scripts/aiecc.exe',out/'harness.mlir',
        f'--peano={p.attention.PEANO}','--get-npu-insts','--npu-insts-name=insts.bin',
        '--dump-intermediates',f'--tmpdir={out/"prj"}'],out)
    delta=instruction_proof((folder/'insts.bin').read_bytes(),(out/'insts.bin').read_bytes())
    result=dict(parent_build_sha256=p.attention.sha(parent/'build.json'),
        cache_build_sha256=p.attention.sha(folder/'build.json'),
        warm_instructions_sha256=p.attention.sha(out/'insts.bin'),
        source_sha256=p.attention.sha(Path(__file__)),cache_proof=proof,instruction_proof=delta,
        xclbin_sha256=p.attention.sha(folder/'final.xclbin'))
    p.attention.dump(out/'build.json',result)
    print('CACHED_WI_REARM_BUILD '+json.dumps(result),flush=True)


def verify(parent,folder,out):
    proof=provenance(parent,folder)
    meta=json.loads((out/'build.json').read_text(encoding='utf-8'))
    if meta['source_sha256']!=p.attention.sha(Path(__file__)):raise ValueError('restart source changed')
    for path,key in ((parent/'build.json','parent_build_sha256'),(folder/'build.json','cache_build_sha256'),
        (folder/'final.xclbin','xclbin_sha256'),(out/'insts.bin','warm_instructions_sha256')):
        if meta[key]!=p.attention.sha(path):raise ValueError('restart artifact identity')
    if meta['cache_proof']!=proof:raise ValueError('restart cache proof changed')
    return instruction_proof((folder/'insts.bin').read_bytes(),(out/'insts.bin').read_bytes())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','folder','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args.parent.absolute(),args.folder.absolute(),args.out.absolute())
