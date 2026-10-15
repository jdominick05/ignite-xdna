#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run-length decoding of an AIE Event-Time trace byte stream, for traces too long to expand cycle by cycle.

``kernels/clock_probe/clock_probe.decode_event_time`` returns one (slot, t) hit per cycle a level event is true,
which is exact and what every count in this repo has used, but a 16 ms graph-engine frame expands to tens of millions
of tuples (5.1 GB for one YOLOv8s shim trace, 2026-10-15). ``decode_runs`` parses the same frames the same way and
keeps a Repeat as one run: (event bits, first t, count, stride). ``hits_in`` counts a slot's hits in [lo, hi) from the
runs. ``python tools/trace_runs.py --selftest <bytes.bin>`` checks both against ``decode_event_time`` on a stream.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

SYNC_CYCLES = 1 << 18   # as clock_probe.SYNC_CYCLES: one Event_Sync frame = the 18-bit delta counter wrapped once

Run = Tuple[int, int, int, int]   # (event bits, t of the first frame, frames, stride between them)


def decode_runs(bs) -> List[Run]:
    """The frames of ``clock_probe.decode_event_time``, as runs. Timestamps follow its "+1 per event frame"."""
    t = 0
    cursor = 0
    out: List[Run] = []
    prev = None
    n = len(bs)
    while cursor < n:
        b = bs[cursor]
        if (b & 0xFB) == 0xF0:
            cursor += 8
            prev = None
            continue
        if (b & 0xFC) == 0xDC:
            cursor += 4
            continue
        if b == 0xFE:
            cursor += 1
            continue
        if b == 0xFF:
            t += SYNC_CYCLES
            prev = ("sync",)
            cursor += 1
            continue
        if (b & 0x80) == 0:
            ev, cyc, w = 1 << ((b >> 4) & 7), b & 0xF, 1
        elif (b & 0xE0) == 0x80:
            ev, cyc, w = 1 << ((b >> 2) & 7), ((b & 3) << 8) | bs[cursor + 1], 2
        elif (b & 0xE0) == 0xA0:
            ev, cyc, w = 1 << ((b >> 2) & 7), ((b & 3) << 16) | (bs[cursor + 1] << 8) | bs[cursor + 2], 3
        elif (b & 0xF0) == 0xC0:
            ev, cyc, w = ((b & 0xF) << 4) | (bs[cursor + 1] >> 4), bs[cursor + 1] & 0xF, 2
        elif (b & 0xFC) == 0xD0:
            ev = ((b & 3) << 6) | (bs[cursor + 1] >> 2)
            cyc, w = ((bs[cursor + 1] & 3) << 8) | bs[cursor + 2], 3
        elif (b & 0xFC) == 0xD4:
            ev = ((b & 3) << 6) | (bs[cursor + 1] >> 2)
            cyc, w = ((bs[cursor + 1] & 3) << 16) | (bs[cursor + 2] << 8) | bs[cursor + 3], 4
        elif (b & 0xF0) == 0xE0 or (b & 0xFC) == 0xD8:
            if (b & 0xF0) == 0xE0:
                r, w = b & 0xF, 1
            else:
                r, w = ((b & 3) << 8) | bs[cursor + 1], 2
            if prev is None:
                raise ValueError(f"Repeat frame at byte {cursor} with nothing to repeat")
            if prev[0] == "sync":
                t += r * SYNC_CYCLES
            elif r:
                stride = prev[2] + 1
                out.append((prev[1], t + stride, r, stride))
                t += r * stride
            cursor += w
            continue
        else:
            raise ValueError(f"unknown trace frame byte 0x{b:02x} at {cursor}")
        t += cyc + 1
        out.append((ev, t, 1, cyc + 1))
        prev = ("event", ev, cyc)
        cursor += w
    return out


def slot_runs(runs: List[Run], slot: int) -> np.ndarray:
    """[(first t, count, stride)] of the runs carrying ``slot``, as an int64 array."""
    bit = 1 << slot
    return np.array([(t0, k, s) for ev, t0, k, s in runs if ev & bit], dtype=np.int64).reshape(-1, 3)


def hits_in(sr: np.ndarray, lo: int, hi: int) -> int:
    """Hits of one slot's runs ``sr`` (from ``slot_runs``) with lo <= t < hi."""
    if not len(sr):
        return 0
    t0, k, s = sr[:, 0], sr[:, 1], sr[:, 2]
    # index of the first frame >= lo and the first frame >= hi, clipped to [0, k]
    a = np.clip(-((t0 - lo) // s), 0, k)
    b = np.clip(-((t0 - hi) // s), 0, k)
    return int((b - a).sum())


def first_hit(sr: np.ndarray, lo: int):
    """The first hit time >= lo of one slot's runs, or None."""
    if not len(sr):
        return None
    t0, k, s = sr[:, 0], sr[:, 1], sr[:, 2]
    a = np.maximum(-((t0 - lo) // s), 0)
    ok = a < k
    return int((t0[ok] + a[ok] * s[ok]).min()) if ok.any() else None


def times(sr: np.ndarray) -> np.ndarray:
    """Every hit time of a pulse slot (expand; only for sparse slots such as a marker)."""
    if not len(sr):
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([t0 + s * np.arange(k) for t0, k, s in sr])


def _selftest(path: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels" / "clock_probe"))
    import clock_probe as cp
    bs = list(path.read_bytes())
    ref = cp.decode_event_time(bs)
    runs = decode_runs(bs)
    bad = 0
    for slot in range(8):
        want = np.array(sorted(t for s, t in ref if s == slot), dtype=np.int64)
        sr = slot_runs(runs, slot)
        got = np.sort(times(sr))
        same = np.array_equal(want, got)
        lo, hi = (int(want[len(want) // 3]), int(want[2 * len(want) // 3])) if len(want) > 3 else (0, 1)
        win = hits_in(sr, lo, hi) == int(((want >= lo) & (want < hi)).sum())
        print(f"slot {slot}: {len(want)} hits; expanded runs equal: {same}; window count equal: {win}")
        bad += (not same) + (not win)
    print("SELFTEST PASS" if not bad else "SELFTEST FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", type=Path, required=True, help="a raw trace byte stream to check against")
    sys.exit(_selftest(ap.parse_args().selftest))
