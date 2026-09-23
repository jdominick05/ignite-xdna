# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find and fix licence headers that claim AMD authorship of code written in this repo.

This repository is GNU AGPL-3.0-or-later as a whole (LICENSE; confirmed by the owner on
2026-09-23). Many files written here were started from mlir-aie's file template and carry
`Copyright (C) 2026 Advanced Micro Devices, Inc.` with `SPDX-License-Identifier: Apache-2.0
WITH LLVM-exception`. That notice is false for original work. The files that genuinely are
copies of upstream mlir-aie code (UPSTREAM below) must keep AMD's notice and licence, which is
compatible with AGPLv3, and gain a line saying where they came from.

The classification is an allowlist, not a guess: every file that is not in UPSTREAM is treated
as written here. Each UPSTREAM entry is checked against the mlir-aie checkout (difflib ratio
against the named upstream path) so a wrong mapping shows up in the report.

Only the header block is touched (the first HEADER_LINES lines). `results/*.log` are evidence
and are never rewritten; an AMD mention elsewhere in a file is reported, not changed.

    python tools/license_header_audit.py            # report only (the default)
    python tools/license_header_audit.py --apply    # rewrite headers in place
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLIR_AIE = Path.home() / "mlir-aie"
HOLDER = "The ignite-xdna contributors"
SPDX_NEW = "AGPL-3.0-or-later"
HEADER_LINES = 15

# Local copies of upstream mlir-aie files (v1.4.2 checkout at MLIR_AIE), and where they came from.
UPSTREAM = {
    "kernels/conv_accum/aie2/conv2dk1.cc": "aie_kernels/aie2/conv2dk1.cc",
    "kernels/conv_accum/aie2/conv2dk1_skip.cc": "aie_kernels/aie2/conv2dk1_skip.cc",
    "kernels/gemm_reblock/aie2/mm_2x2.cc": "aie_kernels/aie2/mm.cc",
    "kernels/gemm_reblock/aie2/zero.cc": "aie_kernels/aie2/zero.cc",
    "kernels/conv_accum/aie_kernel_utils.h": "aie_kernels/aie_kernel_utils.h",
    "kernels/gemm_reblock/aie_kernel_utils.h": "aie_kernels/aie_kernel_utils.h",
    "kernels/bank_placement/whole_array_bankpad.py": "programming_examples/basic/matrix_multiplication/whole_array/whole_array.py",
}

AMD_RE = re.compile(r"Copyright\s*(\(C\)|\(c\)|©)?\s*[0-9, -]*\s*(Advanced Micro Devices|Xilinx)[^\n]*", re.I)
SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([^\n*]+?)\s*(\*/)?\s*$")
APACHE_BOILER = re.compile(r"(licensed under the Apache License v2\.0 with LLVM Exceptions"
                           r"|See https://llvm\.org/LICENSE\.txt for license information)", re.I)


def tracked_with_amd() -> list[str]:
    """Tracked files naming AMD/Xilinx, or carrying an Apache SPDX line (with or without AMD's name)."""
    out = subprocess.run(["git", "grep", "-l", "-I", "-E",
                          "Advanced Micro Devices|Copyright.*Xilinx|SPDX-License-Identifier: *Apache",
                          "--", ".", ":!Ignition"], cwd=ROOT, capture_output=True, text=True).stdout
    return sorted(p for p in out.splitlines() if p)


def upstream_ratio(local: str) -> str:
    up = MLIR_AIE / UPSTREAM[local]
    if not up.exists():
        return f"upstream {UPSTREAM[local]} NOT FOUND in {MLIR_AIE.name}"
    a = (ROOT / local).read_text(encoding="utf-8", errors="replace").splitlines()
    b = up.read_text(encoding="utf-8", errors="replace").splitlines()
    return f"similarity {difflib.SequenceMatcher(None, a, b, autojunk=False).ratio():.3f} to mlir-aie {UPSTREAM[local]}"


def rewrite_header(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return (new_lines, notes) for a file authored here."""
    notes = []
    new = list(lines)
    n = min(HEADER_LINES, len(new))
    for i in range(n):
        line = new[i]
        if AMD_RE.search(line):
            new[i] = AMD_RE.sub(f"Copyright (C) 2026 {HOLDER}", line)
            notes.append(f"L{i + 1} copyright -> {HOLDER}")
        m = SPDX_RE.search(new[i])
        if m and "AGPL" not in m.group(1):
            new[i] = new[i].replace(m.group(1), SPDX_NEW, 1)
            notes.append(f"L{i + 1} SPDX {m.group(1).strip()} -> {SPDX_NEW}")
    # Drop mlir-aie's Apache boilerplate sentences inside the header.
    kept = []
    for i, line in enumerate(new):
        if i < n and APACHE_BOILER.search(line):
            notes.append(f"L{i + 1} removed Apache boilerplate")
            continue
        kept.append(line)
    return kept, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="rewrite headers (default: report only)")
    args = ap.parse_args(argv)

    files = tracked_with_amd()
    print(f"tracked files naming AMD/Xilinx (excluding Ignition/): {len(files)}")
    changed = skipped_logs = 0
    for rel in files:
        p = ROOT / rel
        if rel.startswith("results/") and rel.endswith(".log"):
            print(f"  LOG, never rewritten     {rel}")
            skipped_logs += 1
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        in_header = any(AMD_RE.search(l) or re.search(r"SPDX-License-Identifier:\s*Apache", l)
                        for l in lines[:HEADER_LINES])
        if rel in UPSTREAM:
            print(f"  UPSTREAM copy, keep AMD  {rel}  ({upstream_ratio(rel)})")
            if args.apply and "Adapted from mlir-aie" not in text:
                idx = next(i for i, l in enumerate(lines[:HEADER_LINES]) if AMD_RE.search(l))
                prefix = re.match(r"^\s*(#|//|;|\*|/\*)?\s*", lines[idx]).group(0) or "// "
                lines.insert(idx + 1, f"{prefix}Adapted from mlir-aie v1.4.2 {UPSTREAM[rel]}; local changes are marked inline.")
                p.write_text("\n".join(lines), encoding="utf-8", newline="")
                changed += 1
            continue
        if not in_header:
            print(f"  mention outside header   {rel}  (reported, not changed)")
            continue
        new, notes = rewrite_header(lines)
        print(f"  AUTHORED here, rewrite   {rel}: " + "; ".join(notes))
        if args.apply and new != lines:
            p.write_text("\n".join(new), encoding="utf-8", newline="")
            changed += 1
    print(f"{'changed' if args.apply else 'would change'}: {changed if args.apply else 'see above'}; logs skipped: {skipped_logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
