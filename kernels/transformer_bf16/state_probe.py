"""Check activation-arena commands against qualified packet operations.

Runs A/B/A under one owned session. This is correctness evidence, not a
controlled timing result or a complete draft-model qualification.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kernels.transformer_bf16 import Bf16Ops
from kernels.transformer_bf16.common import save, sha
from kernels.transformer_bf16.persistent import PacketSession
from kernels.transformer_bf16.state import ArenaProgram, ArenaCompiler


def program(a,compiler=None):
    p = ArenaProgram(compiler=compiler)
    p.upload(0, a['projection.x'])
    p.projection(0, a['projection.w'], 1024)
    p.download('projection', 1024, 384)
    p.upload(0, a['norm.x'])
    p.norm(0, a['norm.w'], 2048, rows=4)
    p.download('norm', 2048, 2048)
    p.upload(0, a['rope.q'])
    p.rope(0, a['rope.cosine'], a['rope.sine'], 512)
    p.download('rope', 0, 2048)
    # An odd length tests padding, two-pass statistics, and in-place writes.
    p.upload(0, a['softmax.x'][0])
    p.softmax(0, 513)
    p.download('softmax', 0, 513)
    p.upload(0, a['geglu.gate'])
    p.upload(2048, a['geglu.up'])
    p.geglu(0, 2048, a['table'], 4096)
    p.download('geglu', 4096, 2048)
    # Consume the preceding operation's output without a host round trip.
    p.residual(4096, 2048, 6144, count=2048, scale=np.uint16(0x3f60))
    p.download('chain', 6144, 2048)
    p.upload(0, a['topk.x'])
    p.topk('topk', 0, 2048, 32)
    return p


def rejected_reads():
    checks = []
    initialized = ArenaProgram()
    initialized.upload(0, np.zeros(256, np.uint16))
    for operation in (lambda p: p.download('unwritten', 0, 1),
                      lambda p: p.upload(16384, np.zeros(1, np.uint16)),
                      lambda p: p.norm(0, np.zeros(256, np.uint16), 256),
                      lambda p: initialized.projection(0, np.zeros((16, 256), np.uint16), 0)):
        try:
            operation(ArenaProgram())
        except ValueError:
            checks.append(True)
        else:
            checks.append(False)
    if not all(checks):
        raise ValueError('Invalid arena access was admitted')
    return checks


def run(fixtures, cache, output,exact_norm=False):
    if output.exists():
        raise FileExistsError(output)
    rejection_checks = rejected_reads()
    manifest = json.loads((fixtures/'manifest.json').read_text())
    cases = []
    for row in manifest['cases'][:2]:
        path = fixtures/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Fixture changed')
        with np.load(path, allow_pickle=False) as archive:
            cases.append({k: archive[k].copy() for k in archive.files})
    if len(cases) != 2:
        raise ValueError('Two fixtures required')
    rows = []
    first = None
    session = PacketSession(max_contexts=2, share_program=True)
    compiler = ArenaCompiler(exact_norm=exact_norm)
    if exact_norm:
        forbidden=ArenaProgram(compiler=compiler)
        forbidden.upload(0,np.zeros(256,np.uint16))
        try:forbidden.projection(0,np.zeros((16,256),np.uint16),512,precise=True)
        except ValueError:rejection_checks.append(True)
        else:raise ValueError('Unsupported compensated projection admitted')
    dynamic = ArenaProgram(compiler=compiler)
    dynamic.upload(0, cases[0]['norm.x'], label='x')
    dynamic.norm(0, cases[0]['norm.w'], 2048, rows=4)
    dynamic.download('norm', 2048, 2048)
    prepared = dynamic.prepare(cache/'arena')
    with session:
        ops = Bf16Ops(cache/'baseline', transport=session, single_image=True,exact_norm=exact_norm)
        for index, a in enumerate(cases+[cases[0]]):
            expected = dict(
                projection=ops.projection(a['projection.x'], a['projection.w']),
                norm=ops.norm(a['norm.x'], a['norm.w']).reshape(-1),
                rope=ops.rope(a['rope.q'], a['rope.cosine'], a['rope.sine']).reshape(-1),
                softmax=ops.softmax(a['softmax.x'][:1]).reshape(-1),
                geglu=ops.geglu(a['geglu.gate'], a['geglu.up'], a['table']),
                topk=ops.topk(a['topk.x'], 32))
            expected['chain'] = ops.residual(expected['geglu'], a['geglu.up'], np.uint16(0x3f60))
            values, evidence = program(a,compiler).execute(cache/'arena', session)
            equal = {k: bool(np.array_equal(values[k], v)) for k, v in expected.items()}
            replay = index == 2
            replay_equal = not replay or all(np.array_equal(values[k], first[k]) for k in values)
            if first is None:
                first = values
            row = dict(index=index, passed=all(equal.values()) and replay_equal,
                       bitwise_equal=equal, replay=replay, replay_equal=replay_equal, **evidence)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if not row['passed']:
                save(output, dict(passed=False, rows=rows))
                raise ValueError('Activation arena differs from packet baseline')
        prepared_rows = []
        for index, a in enumerate(cases+[cases[0]]):
            expected = ops.norm(a['norm.x'], cases[0]['norm.w']).reshape(-1)
            values, evidence = prepared.execute(session, uploads={'x': a['norm.x']})
            row = dict(index=index, passed=bool(np.array_equal(values['norm'], expected)), **evidence)
            prepared_rows.append(row)
            print(json.dumps(dict(prepared_upload=row)), flush=True)
        # Invalid patches must be rejected before device submission.
        for patch in ({'missing': cases[0]['norm.x']}, {'x': np.zeros(1, np.uint16)}):
            try:
                prepared.execute(session, uploads=patch)
            except ValueError:
                rejection_checks.append(True)
            else:
                raise ValueError('Invalid prepared upload accepted')
    report = dict(passed=all(r['passed'] for r in rows+prepared_rows),exact_norm=exact_norm, rows=rows, prepared_rows=prepared_rows,
                  rejected_access=rejection_checks, contexts_released=not session.active and not session.images,
                  fixture_manifest_sha256=sha(fixtures/'manifest.json'), scope=__doc__)
    save(output, report)
    if not report['passed']:
        raise ValueError('Prepared upload differs from packet baseline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exact-norm',action='store_true')
    args = parser.parse_args()
    run(args.fixtures, args.cache, args.output,args.exact_norm)
