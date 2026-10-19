"""Local bounded workspace tasks through Ignition's native Gemma tool loop."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Ignition/src'))
from ignition.experimental import HybridAgent
from kernels.transformer_bf16.common import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--tokenizer-python',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--gpu-activity-helper',type=Path,help='Opt into bounded GPU activity; energy benefit unmeasured')
    task=parser.add_mutually_exclusive_group()
    task.add_argument('--task')
    task.add_argument('--task-file',type=Path)
    parser.add_argument('--report',type=Path,help='One-shot local diagnostic output; may contain private workspace text')
    args=parser.parse_args()
    single=args.task is not None or args.task_file is not None
    text_once=args.task
    if args.task_file is not None:
        if args.task_file.stat().st_size>16384:parser.error('Task file exceeds 16 KiB')
        text_once=args.task_file.read_text(encoding='utf-8-sig')
    if single and not text_once.strip():parser.error('Task requires text')
    if args.report is not None and (not single or args.report.exists()):parser.error('Use a new report path with a one-shot task')
    sources={str(path.resolve()):sha(path) for path in (Path(__file__).resolve(),
             ROOT/'Ignition/src/ignition/experimental/agent.py',ROOT/'Ignition/src/ignition/experimental/hybrid.py')}
    final=None;exit_code=0
    with HybridAgent(study_root=ROOT,artifacts=args.artifacts,llama_runtime=args.runtime,
                     tokenizer_python=args.tokenizer_python,snapshot=args.snapshot,workspace=args.workspace,
                     input_depth=2,vector_copy=True,collect_trace=args.report is not None,
                     gpu_activity_helper=args.gpu_activity_helper) as agent:
        def show(event):
            if event['kind']=='answer':print(event['text'],flush=True)
            elif event['kind']=='waiting':print('[Waiting for inference hardware]',file=sys.stderr,flush=True)
            elif event['kind']=='tool':
                label={'read_text':'Reading a file','list_files':'Listing files','calculate':'Calculating'}.get(event['name'],'Using a tool')
                print('['+label+']',file=sys.stderr,flush=True)
        while True:
            try:
                text=text_once if single else input('\nTask (/reset, /quit): ')
                if not single and text=='/quit':break
                if not single and text=='/reset':agent.reset();continue
                if not text.strip():continue
                agent.run(text,on_event=show)
            except (EOFError,KeyboardInterrupt):
                agent.cancel();print('\n[Stopped]',file=sys.stderr)
                final=agent.last_result
                break
            except (RuntimeError,ValueError) as error:
                print(str(error),file=sys.stderr)
                if single:
                    final=agent.last_result or dict(completed=False,error=str(error))
                    exit_code=2;break
            if single:
                final=agent.last_result;break
    if args.report is not None:
        if not all(sha(path)==expected for path,expected in sources.items()):raise ValueError('Agent frontend source changed')
        if final is None:final=dict(completed=False,cancelled=True)
        final['source_sha256']={**final.get('source_sha256',{}),**sources}
        args.report.write_text(json.dumps(final,indent=2),encoding='utf-8')
    if exit_code:raise SystemExit(exit_code)


if __name__=='__main__':main()
