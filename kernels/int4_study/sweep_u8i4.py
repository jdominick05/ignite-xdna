# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run u8i4_probe.py over (kernel, mode, K), one fresh process per row, into one JSONL.

This is kernels/w4a8_probe/sweep.py -- the same rotation of arms and K, the same xrt-smi
partition record at start and end -- with its probe pointed at u8i4_probe.py. The W4A8
sweep is imported rather than copied, and left byte-identical, because its 2026-09-10 logs
describe it.

Usage (ironenv, Desktop 2):
    python kernels/int4_study/sweep_u8i4.py --out results/aie/<raw>.jsonl --tag <tag>
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "w4a8_probe"))

import sweep  # noqa: E402

DEFAULT_ARMS = "u8i8:default,u8i8:unroll2,u8native:default,u8native:unroll2"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--arms" or a.startswith("--arms=") for a in argv):
        argv += ["--arms", DEFAULT_ARMS]
    sweep._PROBE = os.path.join(_HERE, "u8i4_probe.py")
    return sweep.main(argv)


if __name__ == "__main__":
    sys.exit(main())
