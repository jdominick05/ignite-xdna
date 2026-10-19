"""Observe one bounded native tool turn through the GPU/NPU assistant."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Ignition/src'))
from ignition.experimental import HybridAssistant
from npu.gemma4.tokenizer import ChatTokenizer

TOOLS=[dict(type='function',function=dict(name='calculate',description='Evaluate an arithmetic expression exactly.',
       parameters=dict(type='object',properties=dict(expression=dict(type='string',description='Arithmetic using numbers, +, -, *, / and parentheses.')),
                       required=['expression'])))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    messages=[dict(role='user',content='Use calculate to evaluate (137 + 29) * 7. Do not calculate it yourself.')]
    with HybridAssistant(study_root=ROOT,artifacts=args.artifacts,llama_runtime=args.runtime,
                         context_tokens=2560,input_depth=2,vector_copy=True,collect_trace=True) as assistant, ChatTokenizer(
            args.python,args.snapshot,max_prompt_tokens=assistant.max_prompt_tokens) as tokenizer:
        ids=tokenizer.encode(messages,tools=TOOLS)
        for _ in assistant.stream_tokens(ids,max_new_tokens=64):pass
        result=dict(assistant.last_result);raw=tokenizer.decode(result['tokens'])
        report=dict(scope=__doc__,prompt_tokens=len(ids),raw=raw,result=result)
        print(json.dumps(dict(prompt_tokens=len(ids),raw=raw)),flush=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
