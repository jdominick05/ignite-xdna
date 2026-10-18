"""Summarize diagnostic dispatch costs from a saved packet-composition result.

These checks-only observations are not a controlled serving benchmark.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main(path):
    result=json.loads(path.read_text())
    offset=0
    for case in result['results']:
        count=case['dispatches'];dispatches=result['dispatches'][offset:offset+count];offset+=count
        costs=defaultdict(lambda:dict(calls=0,dispatch_ms=0.,wall_ms=0.))
        for item in dispatches:
            row=costs[item['function']];row['calls']+=1
            row['dispatch_ms']+=item['dispatch_ms'];row['wall_ms']+=item['wall_ms']
        print(json.dumps(dict(model=case['model'],fixture=case['fixture'],total_wall_seconds=case['wall_seconds'],
            dispatch_wall_ms=sum(r['wall_ms'] for r in dispatches),dispatch_ms=sum(r['dispatch_ms'] for r in dispatches),
            operations=dict(costs))),flush=True)
    if offset!=len(result['dispatches']):raise ValueError('Dispatch/case counts disagree')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    args=parser.parse_args();main(args.result)
