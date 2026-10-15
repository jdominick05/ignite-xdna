#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-layer shim DMA activity of a graph-engine container built with ``--trace-shims``, from the shim trace unit.

    bash scripts/research-iron.sh tools/engine_shim_trace.py build/yolov8s_tr_ctl.ignite [--runs 3] [--json out]

The container's shims trace eight events (``design.TRACE_SHIM_EVENTS``, read back from the manifest) for the whole
frame, and the emitter fires USER_EVENT_1 (the start event, harmless once running) on them before every layer. Each run dispatches the frame once on a staged
image, reads the trace buffer back, decodes it with ``tools/trace_runs.py`` (``clock_probe.decode_event_time``'s
frame parser kept as runs, since a whole frame expanded cycle by cycle took 5.1 GB; ``--dump-bytes`` saves the stream
so ``trace_runs.py --selftest`` can check the two agree on it) and splits it at the markers: layer k is the cycles from marker k to
marker k+1, the last layer ends at the first flush marker. The core tiles cannot be traced in this design (no free
southbound switch port; design.py, the shim-trace note), so this sees the transport side only:

- ``DMA_MM2S_n_STREAM_BACKPRESSURE``: shim channel n has data and the consumer will not take it (n = 0 activations
  into the MemTile, n = 1 weights into the cores): that input is AHEAD of its consumer.
- ``DMA_MM2S_n_MEMORY_STARVATION``: the channel waits on DDR reads.
- ``DMA_S2MM_0_STREAM_STARVATION``: the output channel has a task and no data: the cores have not produced yet.
- ``DMA_MM2S_n_FINISHED_TASK``: one count per completed shim task.

ONE TRACED DISPATCH PER HARDWARE CONTEXT. The trace's shim S2MM 1 task is one buffer descriptor as long as the
buffer, which a frame never fills, so it is still running when the dispatch completes; the next dispatch's trace
queues behind it and lands mid-buffer (measured 2026-10-15: the second dispatch in one context read 121 of 130 markers
and a de-interleave error, once a 45 ms dispatch). So each run is its own process with its own context, and a run is
that context's first, cold dispatch: compare traced runs with traced runs, never with warm untraced timings.

Level events are counted in cycles (the trace unit emits one hit per cycle an event is true, Repeat-compressed; see
``results/aie/pmu_probe_npu.log``); pulse events in occurrences. Refuses a buffer whose last word is non-zero (it may
have overflowed) and a trace whose marker gaps do not split as the layers then the flush group.
Written on Desktop 2 (Phoenix), 2026-10-15; what has been run is in its logs.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.runtime.graph_session import GraphSession  # noqa: E402
from trace_runs import decode_runs, first_hit, hits_in, slot_runs, times  # noqa: E402

# docs/SILICON.md's MEASURED core clock. That the shim's trace timer runs at it is ASSUMED, not checked: every ms this
# tool prints is DERIVED from it. Counts and ratios in cycles do not depend on it. (Per-layer windows track
# engine_layer_timing's per-layer times at r = 0.997 on YOLOv8s, 2026-10-15, which fits 1.695 GHz, but those times
# include each separate dispatch's ramp, so that is a consistency check, not a clock measurement.)
CLOCK_GHZ = 1.80
MARKER = "USER_EVENT_1"
FLUSH_GAP = 2000   # cycles: flush markers fire ~110 apart, the shortest layer measured ~36,000
LEVEL = ("STREAM_BACKPRESSURE", "MEMORY_STARVATION", "STREAM_STARVATION", "STALLED_LOCK")


def shim_streams(words: np.ndarray):
    """{(row, col): byte stream} of the shim-tile trace packets in a raw trace buffer."""
    from aie.utils.trace.events import PacketType
    from aie.utils.trace.utils import convert_to_byte_stream, trace_pkts_de_interleave
    hexw = [f"{int(w):08x}" for w in words]
    streams = convert_to_byte_stream(trace_pkts_de_interleave(hexw))
    return {tuple(int(v) for v in loc.split(",")): bs for loc, bs in streams[int(PacketType.SHIMTILE)].items()}


def per_layer(runs, events, n_layers):
    """Split decoded runs at the marker slot; returns (rows, frame_cycles, n_markers)."""
    marks = np.sort(times(slot_runs(runs, events.index(MARKER))))
    # Measured on four single-dispatch traces (2026-10-15): the trace does not record its own start event, every
    # layer marker arrives, and only the flush tail loses markers (56-63 of its 64), the tail's last partial packet
    # going unsent. So layer k = [marks[k], marks[k+1]) from the front, the first flush marker ending the last layer;
    # checked here rather than assumed: every layer gap is wide and the marker after the last bound starts the
    # tight flush group (writes ~110 cycles apart).
    if len(marks) < n_layers + 2:
        raise SystemExit(f"[trace] {len(marks)} markers for {n_layers} layers: the trace is truncated or the build "
                         f"is not a --trace-shims build")
    bounds = marks[:n_layers + 1]
    gaps = np.diff(marks[:n_layers + 2])
    if (gaps[:n_layers] < FLUSH_GAP).any() or gaps[n_layers] >= FLUSH_GAP:
        raise SystemExit(f"[trace] marker gaps do not split as {n_layers} layers then the flush group: "
                         f"{gaps.tolist()}")
    srs = {s: slot_runs(runs, s) for s in range(len(events)) if events[s] != MARKER}
    rows = []
    for k in range(n_layers):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        row = {"layer": k, "cycles": hi - lo}
        firsts = []
        for s, sr in srs.items():
            row[events[s]] = hits_in(sr, lo, hi)
            f = first_hit(sr, lo)
            if f is not None and f < hi:
                firsts.append(f)
        # Marker to the first DMA event of any traced kind: the layer's lead-in before its traffic shows, which holds
        # a box layer's MemTile configure.
        row["lead"] = (min(firsts) - lo) if firsts else hi - lo
        rows.append(row)
    return rows, int(bounds[-1] - bounds[0]), len(marks)


def trace_once(container: str, image: str, dump: Path = None) -> dict:
    """Open a context, dispatch once with the trace on, decode it; the context's first dispatch only."""
    s = GraphSession(container)
    try:
        trace = s.ge.get("trace")
        if not trace:
            raise SystemExit(f"{container} was not built with --trace-shims")
        events = list(trace["events"])
        names = [L["name"] for L in s.ge["layers"]]
        img = cv2.cvtColor(cv2.resize(cv2.imread(image), (640, 640)), cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        pyxrt = s.harness.pyxrt
        s.bo_trace.write(np.zeros(s.trace_bytes, dtype=np.uint8), 0)
        s.bo_trace.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        ms = s.run_yolo_monolithic(x, return_timestamps=True)[1]["npu_ms"]
        time.sleep(0.02)   # trace packets drain on their own DMA queue, after the output completes
        s.bo_trace.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        words = np.frombuffer(s.bo_trace.read(s.trace_bytes, 0), dtype=np.uint32)
    finally:
        s.close()
    if words[-1] != 0:
        raise SystemExit("[trace] the buffer's last word is non-zero; it may have overflowed")
    used = int(np.flatnonzero(words)[-1] + 1) * 4 if words.any() else 0
    streams = shim_streams(words[:used // 4])
    if not streams:
        head = " ".join(f"{int(w):08x}" for w in words[:16])
        raise SystemExit(f"[trace] no shim-tile packets in {used:,} used bytes; first words: {head}")
    per_shim = {}
    for (row, col), bs in sorted(streams.items()):
        if dump:
            dump.mkdir(parents=True, exist_ok=True)
            (dump / f"shim_r{row}c{col}_{int(time.time() * 1000)}.bin").write_bytes(bytes(bs))
        rows, frame_cycles, n_marks = per_layer(decode_runs(bs), events, len(names))
        per_shim[col] = {"layers": rows, "frame_cycles": frame_cycles, "markers": n_marks}
    return {"dispatch_ms": ms, "trace_bytes_used": used, "shims": per_shim, "events": events, "layers": names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("container")
    ap.add_argument("--runs", type=int, default=3, help="traced dispatches, each in its own process and context")
    ap.add_argument("--image", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-bytes", type=Path, default=None, help="save each run's shim byte streams in this folder")
    ap.add_argument("--child-out", type=Path, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child_out:
        a.child_out.write_text(json.dumps(trace_once(a.container, a.image, a.dump_bytes)), encoding="utf-8")
        return 0

    runs = []
    with tempfile.TemporaryDirectory() as td:
        for r in range(a.runs):
            out = Path(td) / f"run{r}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), a.container, "--image", a.image,
                   "--child-out", str(out)] + (["--dump-bytes", str(a.dump_bytes)] if a.dump_bytes else [])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"[trace] run {r} failed; stopping - do not retry in a loop")
            run = json.loads(out.read_text(encoding="utf-8"))
            runs.append(run)
            f0 = next(iter(run["shims"].values()))
            print(f"[trace] run {r}: dispatch {run['dispatch_ms']:.3f} ms (first in its context); buffer "
                  f"{run['trace_bytes_used']:,} B; shims {sorted(run['shims'])} (physical columns); marker span "
                  f"{f0['frame_cycles']:,} cycles = {f0['frame_cycles'] / CLOCK_GHZ / 1e6:.3f} ms at {CLOCK_GHZ} GHz (DERIVED); "
                  f"{f0['markers']} markers")
    events, names = runs[0]["events"], runs[0]["layers"]

    # Per layer, the mean over runs of the first traced shim's counts.
    col = sorted(runs[0]["shims"])[0]
    keys = ["cycles", "lead"] + [e for e in events if e != MARKER]
    print(f"\n[trace] shim physical column {col}, mean of {len(runs)} runs (level events in cycles, pulses in counts)")
    print("  k  " + " ".join(f"{k[:22]:>22}" for k in keys) + "  layer")
    for k, name in enumerate(names):
        vals = [np.mean([r["shims"][col]["layers"][k][key] for r in runs]) for key in keys]
        print(f"{k:3d}  " + " ".join(f"{v:22.0f}" for v in vals) + f"  {name}")
    if a.json:
        Path(a.json).write_text(json.dumps({"container": str(a.container), "events": events, "layers": names,
                                            "clock_ghz": CLOCK_GHZ, "runs": runs}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
