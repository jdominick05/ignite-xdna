"""Check local tokenizer isolation and equality with frozen public prompt IDs."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from npu.gemma4.tokenizer import ChatTokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out and args.out.exists():raise FileExistsError(args.out)
    with ChatTokenizer(args.python,args.snapshot) as tokenizer:
        pid=tokenizer.process.pid
        for name in ('dictionary','ok','unicode'):
            path=args.artifacts/('npu/live-public-'+name+'-20261018.i32')
            meta=json.loads(path.with_suffix('.json').read_text())
            ids=tokenizer.encode([dict(role='user',content=meta['prompt'])])
            expected=np.fromfile(path,dtype=np.int32).tolist()
            assert ids==expected,(name,ids,expected)
            assert tokenizer.process.pid==pid
        try:
            tokenizer.encode([dict(role='user',content='repeated text '*1000)])
        except ValueError as error:
            assert '256-token' in str(error)
        else:
            raise AssertionError('Oversized conversation accepted')
        # A rejected request must not poison the worker for later requests.
        assert tokenizer.encode([dict(role='user',content='Hello')])
        tools=[dict(type='function',function=dict(name='calculate',description='Calculate.',parameters=dict(
            type='object',properties=dict(expression=dict(type='string')),required=['expression'])))]
        messages=[dict(role='user',content='Calculate 1+2.'),dict(role='assistant',content='',tool_calls=[dict(
            id='c1',type='function',function=dict(name='calculate',arguments={'expression':'1+2'}))]),
            dict(role='tool',name='calculate',tool_call_id='c1',content=json.dumps({'result':'3'}))]
        ids=tokenizer.encode(messages,tools=tools);raw=tokenizer.decode(ids)
        assert '<|tool_call>call:calculate' in raw and '<|tool_response>response:calculate' in raw
        assert raw.endswith('<tool_response|>'),raw
        messages[-1]['content']={'result':'3'}
        try:tokenizer.encode(messages,tools=tools)
        except ValueError:pass
        else:raise AssertionError('Mapping tool content accepted by the string-content protocol')
        assert tokenizer.encode([dict(role='user',content='Hello')])
        assert tokenizer.process.pid==pid
        worker=tokenizer.process
        result=dict(passed=True,cases=['dictionary','ok','unicode','oversize','recovery','native_tool_response','tool_content_rejection','decode'],
                    scope='tokenization only; no model inference or hardware performance claim',
                    identity=tokenizer.identity,
                    source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):
                                   hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'npu/gemma4/tokenizer.py')})
    assert worker.poll() is not None
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
