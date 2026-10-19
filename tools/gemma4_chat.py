"""Bounded terminal chat through Ignition's persistent GPU/NPU assistant."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'Ignition/src'))
from ignition.experimental import HybridAssistant
from npu.gemma4.runtime import HardwareBusy
from npu.gemma4.tokenizer import ChatTokenizer
from kernels.transformer_bf16.common import sha
from tools.gemma4_request import stream_with_admission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--llama-runtime',type=Path,required=True)
    parser.add_argument('--tokenizer-python',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,choices=range(1,65),default=64)
    parser.add_argument('--depth',type=int,choices=(1,2,3,4),default=2)
    parser.add_argument('--admission-wait',type=int,choices=range(61),default=0,
                        help='Retry pre-output admission rejections for up to this many seconds')
    parser.add_argument('--input-depth',type=int,choices=(1,2),default=1)
    parser.add_argument('--vector-copy',action='store_true')
    parser.add_argument('--prefix-binding',action='store_true',help='Use owned NPU prefix binding; requires input depth 2 and vector copies')
    parser.add_argument('--gpu-activity-helper',type=Path,help='Opt into bounded GPU activity; energy benefit unmeasured')
    parser.add_argument('--context-tokens',type=int,choices=(512,2560),default=512)
    one_shot=parser.add_mutually_exclusive_group()
    one_shot.add_argument('--prompt',help='Answer one prompt and exit')
    one_shot.add_argument('--prompt-file',type=Path,help='Read one UTF-8 prompt from a file, then exit')
    parser.add_argument('--report',type=Path,help='Write local diagnostics for a one-shot request; may contain private output')
    args=parser.parse_args()
    assistant_type=HybridAssistant
    if args.prefix_binding:
        if args.input_depth!=2 or not args.vector_copy:
            parser.error('--prefix-binding requires --input-depth 2 --vector-copy')
        from ignition.experimental.prefix_bound import PrefixBoundAssistant
        assistant_type=PrefixBoundAssistant
    single=args.prompt is not None or args.prompt_file is not None
    text_once=args.prompt
    if args.prompt_file is not None:
        if args.prompt_file.stat().st_size>65536:parser.error('Prompt file exceeds 64 KiB')
        text_once=args.prompt_file.read_text(encoding='utf-8-sig')
    if single and not text_once.strip():parser.error('Prompt must contain text')
    if args.report is not None and (not single or args.report.exists()):
        parser.error('--report requires a one-shot prompt and a new output path')
    sources={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'npu/gemma4/tokenizer.py',
                                   ROOT/'Ignition/src/ignition/experimental/hybrid.py',ROOT/'tools/gemma4_request.py')}
    messages=[]
    result=None;worker=None;exit_code=0;rejections=[];request_error=None
    with assistant_type(
            study_root=ROOT,artifacts=args.artifacts,llama_runtime=args.llama_runtime,
            input_depth=args.input_depth,vector_copy=args.vector_copy,context_tokens=args.context_tokens,
            collect_trace=args.report is not None,gpu_activity_helper=args.gpu_activity_helper) as assistant, ChatTokenizer(
            args.tokenizer_python,args.snapshot,max_prompt_tokens=assistant.max_prompt_tokens) as tokenizer:
        while True:
            try:
                text=text_once if single else input('\nYou (/reset, /quit): ')
            except (EOFError,KeyboardInterrupt):
                break
            if not single and text=='/quit':break
            if not single and text=='/reset':messages.clear();continue
            if not text.strip():
                if single:break
                continue
            candidate=[*messages,dict(role='user',content=text)]
            stream=None;parts=[];complete=False
            try:
                tokens=tokenizer.encode(candidate)
                stream=stream_with_admission(assistant,tokens,max_new_tokens=args.max_new_tokens,
                    depth=args.depth,wait_seconds=args.admission_wait,
                    on_rejection=rejections.append if args.report is not None else None)
                for chunk in stream:
                    print(chunk.text,end='',flush=True)
                    parts.append(chunk.text)
                    complete=chunk.finished
            except KeyboardInterrupt:
                assistant.cancel()
                print('\n[Cancelled]',file=sys.stderr)
            except (HardwareBusy,ValueError) as error:
                print(str(error),file=sys.stderr)
                if single:
                    exit_code=2
                    request_error=str(error)
            finally:
                if stream is not None:stream.close()
            if complete:
                messages=[*candidate,dict(role='assistant',content=''.join(parts))]
            print(flush=True)
            if single:
                result=dict(assistant.last_result) if assistant.last_result is not None else dict(cancelled=request_error is None,completed=False)
                worker=assistant._backend.gpu.process if assistant._backend.gpu is not None else None
                break
        identity=tokenizer.identity
    if args.report is not None:
        if any(sha(Path(p))!=value for p,value in sources.items()):raise ValueError('Chat frontend source changed')
        args.report.write_text(json.dumps(dict(scope=__doc__,source_sha256=sources,tokenizer=identity,
            admission_wait_seconds=args.admission_wait,admission_rejections=rejections,request_error=request_error,
            worker_released=worker.poll() is not None if worker is not None else None,result=result),indent=2),encoding='utf-8')
    return exit_code


if __name__=='__main__':raise SystemExit(main())
