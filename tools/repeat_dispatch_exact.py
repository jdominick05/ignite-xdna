#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dispatch graph-engine detect containers many times on one staged frame and require identical heads.

    bash scripts/research-iron.sh tools/repeat_dispatch_exact.py --iters 300 build/a.ignite build/b.ignite ...

Every dispatch's heads must equal that container's first dispatch byte for byte, and every container's first heads
must equal the first container's. ``tools/verify_engine_container.py`` checks one dispatch layer by layer; this
checks that a container stays exact across the hundreds of dispatches a timing sitting runs, which matters for any
transport that reconfigures MemTile state inside the frame (box fills reset five channels per layer). Exit 0 on
PASS. It times nothing. Written and run on Desktop 2 (Phoenix), 2026-10-15, for the box-fills sitting; first run
as scratch/b42/repeat_exact.py, the same code but for the repository-root depth.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ignite_xdna.runtime.graph_session import GraphSession  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("containers", nargs="+")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
a = ap.parse_args()
img = cv2.cvtColor(cv2.resize(cv2.imread(a.image), (640, 640)), cv2.COLOR_BGR2RGB)
x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
ref0 = None
ok = True
for c in a.containers:
    s = GraphSession(c)
    try:
        def heads():
            h = s.run_yolo_monolithic(x)
            leaves = []

            def walk(v):
                if isinstance(v, dict):
                    for k in sorted(v):
                        walk(v[k])
                elif isinstance(v, (list, tuple)):
                    for e in v:
                        walk(e)
                else:
                    leaves.append(np.asarray(v).tobytes())
            walk(h)
            return leaves
        first = heads()
        bad = 0
        for _ in range(a.iters - 1):
            if any(not p == q for p, q in zip(first, heads())):
                bad += 1
        same = ref0 is None or all(p == q for p, q in zip(ref0, first))
        ref0 = ref0 if ref0 is not None else first
        print(f"{Path(c).name}: {a.iters} dispatches, {bad} differ from the first; "
              f"first equals {Path(a.containers[0]).name}'s: {same}; head bytes {sum(len(t) for t in first)}")
        ok &= bad == 0 and same
    finally:
        s.close()
print("REPEAT EXACT PASS" if ok else "REPEAT EXACT FAIL")
sys.exit(0 if ok else 1)
