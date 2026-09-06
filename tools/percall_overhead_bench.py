"""Where does the missing ~60% of the measured ~39-40% per-column TOPS
ceiling actually go (RESEARCH.md finding 5's open question)?

Two different things could produce the same wall-clock ceiling:
  (a) the NPU compute node itself just runs slower than an ideal 4-TOPS
      column would (a compiler/scheduling limit inside the DPU op), or
  (b) the compute node runs close to that ideal, and the shortfall is
      dispatch overhead OUTSIDE it -- host-side sync, tensor alloc,
      EP/ORT call overhead that doesn't shrink as a fraction of a longer
      compute-bound call the way it should.

ORT's own profiler (session_options.enable_profiling) reports a duration
for every node, including the single fused `vitis_ai_ep_*_kernel_time` node
that is the entire on-NPU subgraph, AND a `model_run` duration for the whole
sess.run() call. The gap between them is (b); comparing the fused node's own
duration to GMACs/inference at a 4-TOPS (16 TOPS / 4 columns) ideal tests (a).
Both read from the same trace, on the same held-open single-column session --
not two separate measurements that could drift.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/percall_overhead_bench.py --model models/yolov8n_cut_xint8.onnx \\
        --cache-key yolocut1x4cachekey \\
        --xclbin "C:\\Program Files\\RyzenAI\\1.7.1\\voe-4.0-win_amd64\\xclbins\\phoenix\\1x4.xclbin" \\
        --calls 200

Writes nothing to results/ itself -- pipe through scripts/lib.sh's run_logged
(or plain `tee`) if the run is worth keeping as a logged claim per
CONTRIBUTING.md. Name the machine (Phoenix / Desktop 2 or Hawk Point /
laptop -- column count and clock differ).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx_tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import npu.yolo as yc
from npu.paths import ASSETS, MODELS
from npu.session import build_session

COLUMN_TOPS = 16.0 / 4  # linear 4-way split of the 16 TOPS nameplate -- same
                         # assumption tools/estimate_tops.py's "% of 16" already uses


def macs_per_inference(model_path):
    m = onnx_tool.Model(str(model_path))
    m.graph.shape_infer()
    m.graph.profile()
    return m.graph.macs[0]


def group_by_call(events):
    """[{'wall_us':.., 'vitis_ep_us':.., 'other_node_us':.., 'gap_us':..}, ...]
    one dict per sess.run() call, built from ORT's chrome-trace events.
    `model_run` gives the call's own wall time; every Node event whose window
    falls inside it is attributed to that call; the fused VitisAI op is
    pulled out by name, everything else Node-cat (Quantize/DequantizeLinear
    at the graph boundary) is summed separately."""
    runs = sorted((e for e in events if e.get("name") == "model_run"),
                  key=lambda e: e["ts"])
    nodes = sorted((e for e in events if e.get("cat") == "Node"),
                   key=lambda e: e["ts"])
    calls = []
    ni = 0
    for run in runs:
        lo, hi = run["ts"], run["ts"] + run["dur"]
        vitis_us = 0
        other_us = 0
        seen = 0
        while ni < len(nodes) and nodes[ni]["ts"] < hi:
            n = nodes[ni]
            if n["ts"] >= lo:
                seen += 1
                if "vitis_ai_ep" in n["name"]:
                    vitis_us += n["dur"]
                else:
                    other_us += n["dur"]
            ni += 1
        calls.append({"wall_us": run["dur"], "vitis_ep_us": vitis_us,
                       "other_node_us": other_us,
                       "gap_us": run["dur"] - vitis_us - other_us,
                       "n_nodes": seen})
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-key", required=True)
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--img", default=str(ASSETS / "test_image.jpg"))
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    macs = macs_per_inference(Path(args.model))
    ideal_us = macs * 2 / (COLUMN_TOPS * 1e12) * 1e6
    print(f"model: {args.model}\nMACs/inference: {macs:.4g}  "
          f"ideal time at a 4-TOPS column: {ideal_us / 1000:.3f} ms")

    sess = build_session(args.model, "npu", args.cache_key, args.xclbin,
                          log_severity=2, enable_profiling=True)
    inp = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, args.model)
    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"could not read {args.img}")
    x, _pad, _scale = yc.letterbox(img, imgsz)

    for _ in range(args.warmup):
        sess.run(None, {inp: x})

    wall_py = []
    t_all0 = time.perf_counter()
    for _ in range(args.calls):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        wall_py.append(time.perf_counter() - t0)
    wall_total = time.perf_counter() - t_all0

    prof_path = sess.end_profiling()
    events = json.load(open(prof_path))
    calls = group_by_call(events)[-args.calls:]  # drop warmup calls' model_run entries
    if len(calls) < args.calls:
        print(f"WARNING: only recovered {len(calls)}/{args.calls} calls from the trace")

    def avg(key):
        return sum(c[key] for c in calls) / len(calls) / 1000  # -> ms

    wall_ms = avg("wall_us")
    vitis_ms = avg("vitis_ep_us")
    other_ms = avg("other_node_us")
    gap_ms = avg("gap_us")
    py_ms = 1000 * sum(wall_py) / len(wall_py)

    print(f"\n{len(calls)} calls, ORT trace vs. this process's own perf_counter "
          f"({py_ms:.3f} ms/call, {args.calls / wall_total:.1f} fps measured "
          "independently of the trace):")
    print(f"  model_run (ORT wall/call):        {wall_ms:8.3f} ms  100.0%")
    print(f"  vitis_ai_ep node (NPU compute):    {vitis_ms:8.3f} ms  "
          f"{100 * vitis_ms / wall_ms:5.1f}%")
    print(f"  other Node ops (Quant/Dequant):    {other_ms:8.3f} ms  "
          f"{100 * other_ms / wall_ms:5.1f}%")
    print(f"  unaccounted gap (dispatch/sync):   {gap_ms:8.3f} ms  "
          f"{100 * gap_ms / wall_ms:5.1f}%")
    print(f"\n  ideal 4-TOPS-column compute time:  {ideal_us / 1000:8.3f} ms")
    print(f"  vitis_ep node / ideal (node-level efficiency): "
          f"{100 * ideal_us / 1000 / vitis_ms:5.1f}%")
    print(f"  ideal / model_run (this call's overall %-of-column-TOPS): "
          f"{100 * ideal_us / 1000 / wall_ms:5.1f}%")


if __name__ == "__main__":
    main()
