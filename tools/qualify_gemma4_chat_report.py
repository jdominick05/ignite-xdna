"""Verify a terminal chat record used the exact qualified public prompt."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--prompt',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--case',default='2048')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    report=json.loads(args.report.read_text(encoding='utf-8'))
    result=report['result'];prompt_sha=sha(args.prompt)
    reference=json.loads(args.reference.read_text())
    if 'reference' in reference and 'warmup' in reference:
        # This profiler records a separate greedy GPU generation, followed by
        # complete NPU input/output equality and repeated request checks.
        assert reference['passed'] and reference['packet_inputs_and_outputs_bitwise_equal']
        baseline=reference['warmup']['result']
        case=dict(prompt_sha256=baseline['prompt_sha256'],prompt_tokens=baseline['prompt_tokens'],
                  tokens=reference['reference'])
        assert all(row['result']['tokens']==case['tokens'] for row in reference['cases'])
        assert all(result['source_sha256'].get(path)==expected
                   for path,expected in baseline['source_sha256'].items())
    else:
        case=next(row for row in reference['cases'] if row['case']==args.case and row['completed'])
    for identities in (report['source_sha256'],result['source_sha256']):
        assert all(sha(Path(p))==expected for p,expected in identities.items())
    count=int(np.fromfile(args.prompt,np.int32).size)
    assert result['prompt_sha256']==prompt_sha==case['prompt_sha256']
    assert result['prompt_tokens']==count==case['prompt_tokens']
    assert result['tokens'][:len(case['tokens'])]==case['tokens']
    assert result['completed'] and result['npu_released'] and result['npu_idle'] and report['worker_released']
    save(args.out,dict(passed=True,scope=__doc__,prompt_tokens=count,control_prefix_equal=True,
                      prompt_sha256=prompt_sha,reference_sha256=sha(args.reference),
                      report_sha256=sha(args.report),source_sha256={str(Path(__file__).resolve()):sha(__file__)},
                      frontend=report))
    print(json.dumps(dict(passed=True,prompt_tokens=count,control_prefix_tokens=len(case['tokens']),
                          generated_tokens=len(result['tokens']))),flush=True)


if __name__=='__main__':main()
