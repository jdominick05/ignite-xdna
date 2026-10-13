#!/usr/bin/env python3
"""Per-layer device timestamps from AMD's own DPU profiler (Ryzen AI 1.7.1, X1), and the difference between arms.

    python -I tools/amd_dpu_layer_profile.py --run ctl=DIR [--run cmt=DIR ...] [--layers ctl=census.txt ...]
        [--skip-frames 2]

How the files come about (Desktop 2, 2026-10-13): compile with the xcompiler attribute `profile` set to
uint 1 (tools/amd_xcompiler_config.py --set profile=uint:1, which adds about 720 B to YOLOv8s's mc_code),
then run the session with DEEPHI_PROFILING=1 (set together with DEBUG_LEVEL1_DPU_PROFILE,
XLNX_ONNX_EP_DL_ANALYZER_PROFILING and ENABLE_FLEXMLRT_PROFILING; which of the four is required is not
isolated). XRT then writes, into the working directory, `dpu_timestamp_info.json` (event id -> layer name,
start/end) and `record_timer_ts.json` (id, cycle). Each frame is bracketed by ids 10485760 (open) and
11534336 (close); one recorded layer is a start id followed by its end id.

The timer's unit is NOT established. The record header declares clock_freq_MHz 1000; this tool converts with
that declaration and labels it so. The profiled program is not the production program (the timestamp writes
are extra instructions), so absolute times describe the instrumented build; compare arms only when both
were compiled with profile=1, and treat the per-layer differences as the usable signal.

Output: one LAYER line per recorded layer with the median cycles over the kept frames per arm, the frame span
per arm, and SUMMARY lines. `--layers` (amd_xmodel_census.py --layers output) adds each layer's declared
bytes so time can be set beside traffic.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

FRAME_OPEN, FRAME_CLOSE = 10485760, 11534336


def load_run(d: Path, skip: int) -> dict:
    info = json.loads((d / "dpu_timestamp_info.json").read_text(encoding="utf-8"))["events"]
    hdr = json.loads((d / "record_timer_ts.json").read_text(encoding="utf-8"))
    ts = hdr["record_timer_ts"]
    ev = {e["id"]: e for e in info}
    frames, cur = [], None
    for r in ts:
        if r["id"] == FRAME_OPEN:
            cur = {"open": r["cycle"], "starts": {}, "layers": {}}
        elif r["id"] == FRAME_CLOSE and cur is not None:
            cur["close"] = r["cycle"]
            frames.append(cur)
            cur = None
        elif cur is not None and r["id"] in ev:
            e = ev[r["id"]]
            if e["type"] != "layer":
                continue
            if e["start"]:
                cur["starts"][e["name"]] = r["cycle"]
            elif e["name"] in cur["starts"]:
                cur["layers"][e["name"]] = r["cycle"] - cur["starts"].pop(e["name"])
    kept = frames[skip:]
    order = [e["name"] for e in info if e["type"] == "layer" and e["start"]]
    types = {e["name"]: e["op_type"] for e in info if e["type"] == "layer"}
    per = {}
    for n in order:
        v = [f["layers"][n] for f in kept if n in f["layers"]]
        if v:
            per[n] = statistics.median(v)
    spans = [f["close"] - f["open"] for f in kept]
    return {"frames": len(frames), "kept": len(kept), "order": order, "types": types, "per": per,
            "span_median": statistics.median(spans), "span_range": [min(spans), max(spans)],
            "mhz": hdr["header"].get("clock_freq_MHz")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--layers", action="append", default=[], metavar="NAME=CENSUS")
    ap.add_argument("--skip-frames", type=int, default=2, help="warm-up frames to drop (default 2)")
    a = ap.parse_args()
    runs = [(s.partition("=")[0], load_run(Path(s.partition("=")[2]), a.skip_frames)) for s in a.run]
    cen = {}
    for s in a.layers:
        n, _, p = s.partition("=")
        cen[n] = {j["layer"]: j for j in (json.loads(ln[10:]) for ln in Path(p).read_text(encoding="utf-8").splitlines()
                                          if ln.startswith("AMD_LAYER "))}
    base = runs[0][1]
    for n, r in runs:
        mhz = r["mhz"] or 0
        print("RUN " + json.dumps({"arm": n, "frames": r["frames"], "kept": r["kept"], "layers_recorded": len(r["per"]),
                                   "span_cycles_median": r["span_median"], "span_cycles_range": r["span_range"],
                                   "header_clock_mhz": mhz,
                                   "span_ms_at_header_clock": round(r["span_median"] / (mhz * 1e3), 4) if mhz else None,
                                   "sum_layer_cycles": sum(r["per"].values())}), flush=True)
    for name in base["order"]:
        rec = {"layer": name, "type": base["types"].get(name)}
        for n, r in runs:
            rec[n] = r["per"].get(name)
            c = cen.get(n, {}).get(name)
            if c:
                rec[n + "_bytes"] = sum(c.get(k, 0) for k in ("const_data_load_size", "input_feature_load_size",
                                                              "output_feature_save_size"))
        print("LAYER " + json.dumps(rec), flush=True)
    b0 = runs[0][0]
    for n, r in runs[1:]:
        common = [x for x in base["order"] if x in base["per"] and x in r["per"]]
        d = sorted(((r["per"][x] - base["per"][x], x) for x in common))
        print("SUMMARY " + json.dumps({
            "arm": n, "vs": b0, "layers_compared": len(common),
            "span_delta_cycles": r["span_median"] - base["span_median"],
            "sum_layer_delta_cycles": sum(v for v, _ in d),
            "faster_layers": sum(v < 0 for v, _ in d), "slower_layers": sum(v > 0 for v, _ in d),
            "largest_gains": [[x, v] for v, x in d[:8]], "largest_losses": [[x, v] for v, x in d[::-1][:5]]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
