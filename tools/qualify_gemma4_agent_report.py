"""Check a native agent CLI report against previously GPU-qualified turn tokens.

Reuses the registered reference; does not perform a new GPU control replay.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    current=json.loads(a.report.read_text());prior=json.loads(a.reference.read_text())
    assert prior['passed'] and prior['result']['completed'] and current['completed']
    assert current['answer']==prior['result']['answer']=='959'
    assert [c['name'] for c in current['tool_calls']]==['read_text','calculate']
    assert current['tool_calls']==prior['result']['tool_calls']
    assert len(current['rounds'])==len(prior['result']['rounds'])==3
    identities=dict(current['source_sha256'])
    for actual,expected in zip(current['rounds'],prior['result']['rounds'],strict=True):
        model=actual['model'];control=expected['model']
        assert expected['gpu_control_equal']
        assert model['tokens']==control['tokens'] and model['prompt_sha256']==control['prompt_sha256']
        assert model['npu_idle'] and model['npu_released'] and model['finish_reason']=='eos' and model['model_loads']==1
        activity=model['gpu_activity']
        assert activity['enabled'] and activity['idle'] and activity['identity']['hip_luid_matched']
        assert activity['windows'] and all(w['gpu_validated'] for w in activity['windows'])
        for path,digest in control['source_sha256'].items():
            if Path(path).suffix in ('.exe','.dll','.gguf','.npz','.json'):
                assert model['source_sha256'][path]==digest
        identities.update(model['source_sha256'])
    assert all(sha(path)==digest for path,digest in identities.items())
    sources={str(path.resolve()):sha(path) for path in (Path(__file__).resolve(),a.report,a.reference)}
    save(a.out,dict(passed=True,scope=__doc__,source_sha256=sources,answer=current['answer'],
                    prior_control_turns_equal=True,timing_eligible=False,result=current))
    print('PASS: native CLI tools, prior GPU control tokens, activity checks and sources',flush=True)


if __name__=='__main__':main()
