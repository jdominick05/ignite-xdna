"""Check changed Q/K/V and rotary factors through one prepared NPU schedule.

Synthetic transport qualification against native packet operations, not model
accuracy, acceptance or serving timing. Every activation operation uses the NPU.
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
from kernels.transformer_bf16.common import bf16_bits, save, sha
from kernels.transformer_bf16.persistent import PacketSession
from kernels.transformer_bf16.state import ArenaProgram, ArenaCompiler


def data(seed):
    rng = np.random.default_rng(seed)
    return dict(q=bf16_bits(rng.normal(size=(4, 512))),
                key=bf16_bits(rng.normal(size=(37, 512))*.1),
                value=bf16_bits(rng.normal(size=(37, 512))),
                factors=(bf16_bits(rng.uniform(-1, 1, size=512)),
                         bf16_bits(rng.uniform(-1, 1, size=512))))


def schedule(a, compiler, precise):
    p = ArenaProgram(compiler=compiler)
    p.upload(0, a['q'], label='q')
    p.rope(0, *a['factors'], 512, label='position')
    p.download('rotary', 0, 2048)
    p.projection(0, a['key'], 2048, precise=precise, label='key')
    p.softmax(2048, 37)
    p.download('probabilities', 2048, 37)
    p.projection(2048, a['value'].T, 4096, precise=precise, label='value')
    p.download('context', 4096, 512)
    return p


def run(cache, output):
    if output.exists():
        raise FileExistsError(output)
    a, b = data(11), data(29)
    compiler = ArenaCompiler()
    programs = {precise: schedule(a, compiler, precise).prepare(cache/'arena') for precise in (False, True)}
    rows, rejections = [], []
    session = PacketSession(share_program=True)
    with session:
        for precise, prepared in programs.items():
            ops = Bf16Ops(cache/'baseline', precise_projection=precise, transport=session, single_image=True)
            for index, values in enumerate((a, b, a)):
                q = ops.rope(values['q'], *values['factors'])
                scores = ops.projection(q[0], values['key'])
                probabilities = ops.softmax(scores[None]).reshape(-1)
                context = ops.projection(probabilities, values['value'].T)
                expected = dict(rotary=q.reshape(-1), probabilities=probabilities, context=context)
                # Final execution without patches proves that patching B did
                # not alter the owned A template or leak B's state into A.
                patches = {} if index == 2 else dict(uploads={'q': values['q']},
                    matrices={'key': values['key'], 'value': values['value'].T},
                    rotations={'position': values['factors']})
                actual, evidence = prepared.execute(session, **patches)
                equal = {key: bool(np.array_equal(value, expected[key])) for key, value in actual.items()}
                row = dict(precise=precise, index=index, passed=all(equal.values()), bitwise_equal=equal, **evidence)
                rows.append(row)
                print(json.dumps(row), flush=True)
            bad = (dict(matrices={'missing': b['key']}),
                   dict(matrices={'key': b['key'][:-1]}),
                   dict(matrices={'key': b['key'].astype(np.float32)}),
                   dict(rotations={'missing': b['factors']}),
                   dict(rotations={'position': (b['factors'][0][:-1], b['factors'][1])}))
            for patch in bad:
                try:
                    prepared.execute(session, **patch)
                except ValueError:
                    rejections.append(True)
                else:
                    raise ValueError('Invalid prepared patch admitted')
    report = dict(passed=all(row['passed'] for row in rows), rows=rows,
                  rejections=rejections, contexts_released=not session.active and not session.images,
                  source_sha256=sha(__file__), scope=__doc__)
    save(output, report)
    if not report['passed']:
        raise ValueError('Patched attention differs from native packet baseline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)
