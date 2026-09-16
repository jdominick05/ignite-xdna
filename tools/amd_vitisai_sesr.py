"""AMD's arm of a same-sitting comparison on SESR M7: ONNX Runtime + Vitis AI EP (Ryzen AI 1.7.1) on the XINT8 model.

Does per frame what Ignition's SuperResolutionPipeline does on its ONNX path: Ignition's own sr_preprocess, one
session.run and sr_postprocess, timed from the BGR frame to the upscaled BGR image, 50 warm-up and 500 timed frames by
default. Provider options, cache handling, the EP placement report, the ``[run] frame N`` progress line every 100
timed frames and ``--max-fps`` follow amd_vitisai_yolo.py, so ``tools/energy_sitting.py`` can window it the same way.
This is the arm of results/aie/yolov8s_sesr_vs_amd_phoenix_20260915T2146Z.log and
latency_balanced_default_phoenix_20260916T1745Z.log, which ran it before the progress line and --max-fps existed.

Run in resnet_env17 with RYZEN_AI_INSTALLATION_PATH set.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import psutil

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--image", type=Path, required=True)
ap.add_argument("--ignition-src", type=Path, required=True)
ap.add_argument("--cache-dir", type=Path, required=True)
ap.add_argument("--cache-key", required=True)
ap.add_argument("--xclbin", type=Path, required=True)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--iterations", type=int, default=500)
ap.add_argument("--max-fps", type=float, default=0.0,
                help="pace timed frames to at most this rate, like a camera (0 = as fast as possible); frames start on "
                     "a fixed schedule and the wait is outside each frame's G2G time")
ap.add_argument("--json", type=Path, default=None)
args = ap.parse_args()

sys.path.insert(0, str(args.ignition_src))
import onnxruntime as ort  # noqa: E402

from ignition.pipelines.vision import sr_postprocess, sr_preprocess  # noqa: E402

proc = psutil.Process()
opts = {"cacheDir": str(args.cache_dir), "cacheKey": args.cache_key, "enable_cache_file_io_in_mem": "0",
        "target": "X1", "xlnx_enable_py3_round": "0", "xclbin": str(args.xclbin)}
t0 = time.perf_counter()
sess = ort.InferenceSession(str(args.model), providers=["VitisAIExecutionProvider"], provider_options=[opts])
load_s = time.perf_counter() - t0
if "VitisAIExecutionProvider" not in sess.get_providers():
    raise SystemExit(f"VitisAIExecutionProvider failed to initialize: {sess.get_providers()}")
report_path = args.cache_dir / args.cache_key / "vitisai_ep_report.json"
placement = {}
if report_path.exists():
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    placement = dict(Counter(str(n.get("device", "?")) for n in rep.get("nodeStat", [])))
print(f"[amd] ort {ort.__version__} | model {args.model.name} | session load {load_s:.2f} s | EP report node devices "
      f"{placement or 'report not found'}", flush=True)

inp = sess.get_inputs()[0]
input_hw = (int(inp.shape[2]), int(inp.shape[3]))
img = cv2.imread(str(args.image))
if img is None:
    raise SystemExit(f"cannot read {args.image}")
for _ in range(args.warmup):
    sr_postprocess(sess.run(None, {inp.name: sr_preprocess(img, input_hw)})[0])
total, pre, run, post = [], [], [], []
out = None
period = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
t_start = time.perf_counter()
for i in range(1, args.iterations + 1):
    if period:
        # Absolute schedule: a late frame does not make the next one early, and the rate averages exactly max_fps.
        delay = t_start + (i - 1) * period - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    a = time.perf_counter()
    x = sr_preprocess(img, input_hw)
    b = time.perf_counter()
    y = sess.run(None, {inp.name: x})[0]
    c = time.perf_counter()
    out = sr_postprocess(y)
    d = time.perf_counter()
    pre.append((b - a) * 1e3)
    run.append((c - b) * 1e3)
    post.append((d - c) * 1e3)
    total.append((d - a) * 1e3)
    if i % 100 == 0:
        print(f"[run] frame {i} | G2G {total[-1]:.2f} ms | output {out.shape[1]}x{out.shape[0]}", flush=True)
v = np.asarray(total)
p50, p95, p99 = np.percentile(v, [50, 95, 99])
rss = proc.memory_info().rss / (1024 * 1024)
print(f"[amd] G2G mean {v.mean():.3f} ms | P50 {p50:.3f} | P95 {p95:.3f} | P99 {p99:.3f} | max {v.max():.3f} | "
      f"over {v.size} timed frames after {args.warmup} warm-up", flush=True)
print(f"[amd] stage means (ms): preprocess {np.mean(pre):.3f} | session.run {np.mean(run):.3f} | postprocess "
      f"{np.mean(post):.3f}", flush=True)
print(f"[amd] output {tuple(out.shape)} from input {input_hw} | RSS {rss:.1f} MB", flush=True)
if args.json:
    args.json.write_text(json.dumps({
        "engine": "ONNX Runtime + Vitis AI EP (Ryzen AI 1.7.1)", "model": args.model.name, "cache_key": args.cache_key,
        "session_load_s": load_s, "ep_node_devices": placement, "warmup": args.warmup, "iterations": args.iterations,
        "g2g_ms": {"mean": float(v.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99),
                   "min": float(v.min()), "max": float(v.max())},
        "stages_ms": {"preprocess": float(np.mean(pre)), "session_run": float(np.mean(run)),
                      "postprocess": float(np.mean(post))},
        "rss_mb": rss, "output_shape": list(out.shape), "input_hw": list(input_hw),
    }, indent=1), encoding="utf-8")
