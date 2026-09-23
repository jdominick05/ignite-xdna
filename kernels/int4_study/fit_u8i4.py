# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit u8i4_probe rows exactly as kernels/w4a8_probe/fit.py fits the W4A8 probe's.

fit.py is imported, not copied (it is kept byte-identical because its 2026-09-10 logs
describe it); only its MACs-per-vmac table learns the two uint8 kernels.

Usage:
    python kernels/int4_study/fit_u8i4.py results/aie/<raw>.jsonl [--tag u8i4] [--control u8i8:default]
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "w4a8_probe"))

import fit  # noqa: E402

fit.MACS_PER_VMAC.update({"u8i8": 256, "u8native": 512})


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--control" or a.startswith("--control=") for a in argv):
        argv += ["--control", "u8i8:default"]
    return fit.main(argv)


if __name__ == "__main__":
    sys.exit(main())
