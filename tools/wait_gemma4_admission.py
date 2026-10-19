"""Bounded read-only wait for the existing assistant admission policy.

Records rejected snapshots; does not stop processes, reserve devices, or prove
performance isolation. Callers must check admission again before model work.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.transformer_bf16.common import save, sha
from npu.gemma4.runtime import HardwareBusy, admit_hardware


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, choices=range(1, 61), default=30)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    paths = [Path(__file__).resolve(), ROOT/'npu/contention.py', ROOT/'npu/gemma4/runtime.py']
    report = dict(admitted=False, scope=__doc__, snapshots=[],
                  source_sha256={str(path.resolve()): sha(path) for path in paths})
    deadline = time.monotonic()+a.seconds
    try:
        while True:
            try:
                witness = admit_hardware()
                report.update(admitted=True, accepted_snapshot=witness)
                print('ADMISSION PASS: existing policy accepted the fresh snapshot', flush=True)
                break
            except HardwareBusy as error:
                report['snapshots'].append(error.report)
                save(a.out, report)
                print(f'ADMISSION WAIT: {len(report["snapshots"])} rejected snapshots retained', flush=True)
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(1, max(0, deadline-time.monotonic())))
        return 0 if report['admitted'] else 2
    finally:
        save(a.out, report)


if __name__ == '__main__':
    raise SystemExit(main())
