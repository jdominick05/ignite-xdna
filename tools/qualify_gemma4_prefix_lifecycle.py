"""Run the existing lifecycle contract with the isolated prefix-binding candidate.

Cancellation is deferred briefly until an owned bound-upload dispatch is
entered. The candidate must release hardware and resume in the same workers.
Serving sources and the pending energy protocol remain unchanged.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'Ignition/src'))
from ignition.experimental.prefix_bound import PrefixBoundAssistant
from kernels.transformer_bf16.common import save, sha
from kernels.transformer_bf16.prefix_binding import BoundUploadsArena
from tools import qualify_gemma4_activity as contract


class BoundAssistant(PrefixBoundAssistant):
    current = None
    extra_sources = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._backend.sources.update(self.extra_sources)
        self.binding_entered = threading.Event()
        self.binding_active = threading.Event()
        self.cancel_witnesses = []
        type(self).current = self

    def stream_tokens(self, *args, **kwargs):
        self.binding_entered.clear()
        return super().stream_tokens(*args, **kwargs)

    def cancel(self):
        entered = self.binding_entered.wait(1)
        active = self.binding_active.is_set()
        helper = self._backend._activity
        self.cancel_witnesses.append(dict(bound_dispatch_entered=entered,
            bound_dispatch_active=active, activity_active=helper is not None and helper.active))
        super().cancel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--helper', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = a.out.with_suffix('.raw.json')
    if a.out.exists() or raw.exists() or raw.with_suffix('.progress.json').exists():
        raise FileExistsError(a.out)
    paths = [Path(__file__).resolve(), Path(contract.__file__),
             ROOT/'Ignition/src/ignition/experimental/prefix_bound.py',
             ROOT/'tools/gemma4_prefix_candidate.py', ROOT/'kernels/transformer_bf16/prefix_cache.py',
             ROOT/'kernels/transformer_bf16/prefix_binding.py']
    sources = {str(path.resolve()): sha(path) for path in paths}
    BoundAssistant.extra_sources = sources
    report = dict(passed=False, scope=__doc__, source_sha256=sources,
                  timing_eligible=False, energy_eligible=False, candidate_kind='prefix_binding')
    original_assistant, original_execute, original_argv = contract.HybridAssistant, BoundUploadsArena.execute, sys.argv

    def observed_execute(owner, *args, **kwargs):
        assistant = BoundAssistant.current
        assistant.binding_active.set()
        assistant.binding_entered.set()
        try:
            return original_execute(owner, *args, **kwargs)
        finally:
            assistant.binding_active.clear()

    contract.HybridAssistant = BoundAssistant
    BoundUploadsArena.execute = observed_execute
    sys.argv = [str(Path(contract.__file__)), '--artifacts', str(a.artifacts),
                '--runtime', str(a.runtime), '--helper', str(a.helper), '--out', str(raw)]
    try:
        contract.main()
        report['contract'] = json.loads(raw.read_text())
        report['raw_report_sha256'] = sha(raw)
        report['cancel_witnesses'] = BoundAssistant.current.cancel_witnesses
        assert report['contract']['passed']
        assert report['cancel_witnesses'] and all(all(row.values()) for row in report['cancel_witnesses'])
        assert all(sha(Path(path)) == expected for path, expected in sources.items())
        report['passed'] = True
        print('PASS: cancellation during bound upload, cleanup and same-worker recovery', flush=True)
    finally:
        contract.HybridAssistant = original_assistant
        BoundUploadsArena.execute = original_execute
        sys.argv = original_argv
        if BoundAssistant.current is not None:
            report['cancel_witnesses'] = BoundAssistant.current.cancel_witnesses
        save(a.out, report)


if __name__ == '__main__':
    main()
