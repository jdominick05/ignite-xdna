"""Audit standalone BF16 image identities across runtime packet counts.

Offline evidence only. Matching core/program bytes does not itself authorize
cross-image dispatch; runtime argument connectivity and device checks are separate.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha


def main(cache,output):
    if output.exists():raise FileExistsError('Refusing to replace compatibility evidence')
    groups=defaultdict(list)
    for path in sorted(cache.glob('*/build.json')):
        meta=json.loads(path.read_text())
        folder=path.parent
        if any(sha(folder/name)!=expected for name,expected in meta['artifacts'].items()):
            raise ValueError(f'Changed build artifact: {folder.name}')
        files=dict(pdi=folder/'prj/main.pdi',topology=folder/'prj/memTopology_main.json',
                   kernels=folder/'prj/kernels_main.json')
        elfs=list((folder/'prj').rglob('*.elf'))
        if len(elfs)!=1 or not all(p.is_file() for p in files.values()):
            raise ValueError(f'Incomplete compiler output: {folder.name}')
        files['elf']=elfs[0]
        key=(meta['function'],meta['packet_words'],meta['output_words'],meta.get('state_words',0),meta['artifacts']['kernel.o'])
        groups[key].append(dict(image=folder.name,blocks=meta['blocks'],
            identities={name:sha(file) for name,file in files.items()},
            instructions_sha256=meta['artifacts']['insts.bin'],xclbin_sha256=meta['artifacts']['final.xclbin']))
    records=[]
    for key,rows in groups.items():
        uniqueness={name:len(set(row['identities'][name] for row in rows)) for name in rows[0]['identities']}
        record=dict(function=key[0],packet_words=key[1],output_words=key[2],state_words=key[3],
            kernel_object_sha256=key[4],images=len(rows),packet_counts=sorted(set(row['blocks'] for row in rows)),
            distinct=uniqueness,records=rows)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='records'}),flush=True)
    save(output,dict(scope=__doc__,source_sha256=sha(__file__),groups=records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();main(args.cache,args.out)
