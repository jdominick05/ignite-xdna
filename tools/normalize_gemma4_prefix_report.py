"""Preserve a raw prefix audit while disambiguating its packet-count field.

Audit hashes are not transport result records. The archive reader identifies
transport records by build+commands, so audit counts use packet_count instead.
The raw report and this transformation are pinned in the derived source map.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernels.transformer_bf16.common import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    report = json.loads(a.result.read_text())
    for audit in report['audits'].values():
        for row in audit['packets']:
            if 'commands' in row:
                row['packet_count'] = row.pop('commands')
    report['source_sha256'].update({str(path.resolve()): sha(path) for path in (a.result, Path(__file__))})
    save(a.out, report)


if __name__ == '__main__':
    main()
