"""Create public exact-length chat prompts and check extended rotary prefixes."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from npu.gemma4.tokenizer import ChatTokenizer
from kernels.transformer_bf16.common import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    a=args.artifacts/'npu/rope-e2b-live-512-20261018.npz'
    b=args.artifacts/'npu/rope-e2b-live-2560-20261018.npz'
    with np.load(a,allow_pickle=False) as small,np.load(b,allow_pickle=False) as large:
        assert small.files==large.files
        assert all(np.array_equal(small[k],large[k][:512]) for k in small.files)
    prepared=[]
    with ChatTokenizer(args.python,args.snapshot,max_prompt_tokens=2048) as tokenizer:
        for target in (511,513,1024,2048):
            low,high=1,2048;found=None
            while low<=high:
                repeats=(low+high)//2
                text='Background notes:'+(' blue'*repeats)+'\nWrite a short paragraph about the sky.'
                try:ids=tokenizer.encode([dict(role='user',content=text)])
                except ValueError as error:
                    if '2048-token' not in str(error):raise
                    high=repeats-1;continue
                if len(ids)==target:found=(text,ids);break
                if len(ids)<target:low=repeats+1
                else:high=repeats-1
            if found is None:raise ValueError(f'Could not create exactly {target} tokens')
            prepared.append((target,*found))
        identity=tokenizer.identity
    args.out.mkdir()
    for target,text,ids in prepared:
        path=args.out/f'public-{target}.i32'
        path.write_bytes(np.asarray(ids,np.int32).tobytes())
        path.with_suffix('.txt').write_text(text,encoding='utf-8')
        path.with_suffix('.json').write_text(json.dumps(dict(prompt=text,tokens=target,sha256=sha(path),
            tokenizer=identity,source_sha256=sha(__file__),rotary_prefix_equal=True,
            rotary_sha256=sha(b)),indent=2),encoding='utf-8')
    print(json.dumps(dict(folder=str(args.out),lengths=[p[0] for p in prepared],rotary_prefix_equal=True)),flush=True)


if __name__=='__main__':main()
