"""Archive qualification reports with verified source/build provenance.

Refuses replacement, verifies referenced source/program artifacts and scrubs the
current local account prefix. Does not turn a negative report into a pass.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import save,sha


def run(result,output,sources_only=False):
    if output.exists():raise FileExistsError(output)
    report=json.loads(result.read_text())
    paths=set()
    def discover(value):
        if isinstance(value,dict):
            if ('prompt_tokens' in value or 'rounds' in value) and isinstance(value.get('source_sha256'),dict):
                for source,expected in value['source_sha256'].items():
                    if sha(Path(source))!=expected:raise ValueError('Assistant source or artifact changed')
            if 'build' in value and 'commands' in value:
                if 'COMPLETED' not in value['status'] or not value['input_unchanged']:
                    raise ValueError('Incomplete or mutated command stream')
                path=Path(value['build'])
                meta=json.loads(path.read_text())
                if meta['blocks']!=value['commands']:raise ValueError('Command count changed')
                paths.add(path)
                for source,expected in value.get('transport_sources_sha256',{}).items():
                    if sha(Path(source))!=expected:raise ValueError('Packet transport source changed')
            for key,item in value.items():
                if key.endswith('builds') and isinstance(item,list):
                    paths.update(Path(p) for p in item)
                else:discover(item)
        elif isinstance(value,list):
            for item in value:discover(item)
    discover(report)
    if sources_only:
        identities=report.get('source_sha256')
        if not isinstance(identities,dict) or not identities:raise ValueError('No source identities found')
        for source,expected in identities.items():
            if sha(Path(source))!=expected:raise ValueError('Qualification source or artifact changed')
    elif not paths:raise ValueError('No build provenance found')
    builds={}
    for path in sorted(paths):
        meta=json.loads(path.read_text())
        for collection in ('sources','artifacts','program_artifacts'):
            for name,expected in meta[collection].items():
                artifact=Path(name) if collection=='sources' else path.parent/name
                if sha(artifact)!=expected:raise ValueError('Build source or artifact changed')
        builds[str(path)]=dict(manifest_sha256=sha(path),**meta)
    home=str(Path.home())
    def scrub(value):
        if isinstance(value,str):return value.replace(home,'C:\\Users\\<user>').replace(home.replace('\\','/'),'C:/Users/<user>')
        if isinstance(value,dict):return {scrub(k):scrub(v) for k,v in value.items()}
        if isinstance(value,list):return [scrub(v) for v in value]
        return value
    save(output,scrub(dict(scope=__doc__,source_sha256=sha(__file__),result_sha256=sha(result),report=report,builds=builds)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources-only',action='store_true',help='Allow non-packet checks with an explicit verified source map')
    a=p.parse_args();run(a.result,a.output,a.sources_only)
