"""Run a public file-to-calculator task through the native hybrid tool loop."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Ignition/src'))
from ignition.experimental import HybridAgent
from kernels.transformer_bf16.common import sha,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    progress=args.out.with_suffix('.progress.json')
    if args.out.exists() or progress.exists() or args.workspace.exists():raise FileExistsError('Use new report and fixture paths')
    args.workspace.mkdir();fixture=args.workspace/'brief.txt'
    fixture.write_text('The calibration batch contains 137 samples. The retention factor is 7.\n',encoding='utf-8')
    paths=[Path(__file__).resolve(),ROOT/'Ignition/src/ignition/experimental/agent.py',fixture]
    sources={str(p):sha(p) for p in paths}
    with HybridAgent(study_root=ROOT,artifacts=args.artifacts,llama_runtime=args.runtime,
                     tokenizer_python=args.python,snapshot=args.snapshot,workspace=args.workspace,
                     input_depth=2,vector_copy=True,collect_trace=True) as agent:
        try:
            answer=agent.run('Read brief.txt with read_text, then use calculate to multiply the sample count by the retention factor. Reply only with the result.',
                             on_event=lambda event:print(json.dumps(event),flush=True))
        finally:
            save(progress,dict(scope='Unqualified task snapshot before GPU control checks; not a passing report',
                               passed=False,source_sha256=sources,result=agent.last_result))
        result=agent.last_result
        assert result['completed'] and answer.strip()=='959',answer
        names=[call['name'] for call in result['tool_calls']]
        assert 'read_text' in names and 'calculate' in names,names
        assert any(call['name']=='calculate' and call['response']=={'result':'959'} for call in result['tool_calls'])
        backend=agent.assistant._backend;worker=backend.gpu.process
        assert backend.model_loads==1
        # Test-only GPU replays; agent serving does not run these controls.
        for row in result['rounds']:
            preflight=backend._preflight()
            expected=row['model']['tokens'];current=backend.gpu.prepare(row['prompt_ids']);actual=[]
            for index in range(len(expected)):
                token=current['predictions'][-1];actual.append(token)
                if index+1<len(expected):current=backend.gpu.decode([token])
            assert actual==expected,(actual,expected)
            row['gpu_control_equal']=True;row['control_preflight']=preflight
        identity=agent.tokenizer.identity
    assert worker.poll() is not None and sha(fixture)==sources[str(fixture)]
    assert all(sha(Path(p))==expected for p,expected in sources.items())
    args.out.write_text(json.dumps(dict(passed=True,scope=__doc__,answer=answer,source_sha256=sources,
                                       tokenizer=identity,worker_released=True,result=result),indent=2),encoding='utf-8')
    print('PASS: native read, calculate, final answer, GPU control equality and cleanup',flush=True)


if __name__=='__main__':main()
