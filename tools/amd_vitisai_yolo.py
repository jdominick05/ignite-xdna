"""AMD's arm of a same-sitting comparison: ONNX Runtime + Vitis AI EP (Ryzen AI 1.7.1) on a QDQ YOLO cut model.

Mirrors Ignition's benchmarks/benchmark_yolo_vitisai.py ``run_vitisai_benchmark``: the same letterbox, head decode and
per-class NMS from Ignition's src, the same provider options, 50 warm-up and 500 timed frames by default. Differences:
its own ``--cache-dir``/``--cache-key`` (the Ignition script hard-codes yolocutcachekey, so another model would load
YOLOv8n's compiled cache), the EP placement report is read back, the summary uses live.py's percentile lines, and a
``[run] frame N`` line every 100 timed frames in the same shape Ignition's headless loop prints, so
``tools/energy_sitting.py`` can window both arms the same way.

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
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--iou", type=float, default=0.45)
ap.add_argument("--json", type=Path, default=None)
args = ap.parse_args()

sys.path.insert(0, str(args.ignition_src))
import onnxruntime as ort  # noqa: E402

from ignition.pipelines.yolo import decode_heads, letterbox, postprocess_detections  # noqa: E402

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
    devices = Counter(str(n.get("device", "?")) for n in rep.get("nodeStat", []))
    placement = dict(devices)
print(f"[amd] ort {ort.__version__} | model {args.model.name} | session load {load_s:.2f} s | EP report node devices "
      f"{placement or 'report not found'}", flush=True)

inp = sess.get_inputs()[0].name
img = cv2.imread(str(args.image))
if img is None:
    raise SystemExit(f"cannot read {args.image}")
for _ in range(args.warmup):
    blob, pad, scale = letterbox(img, 640)
    outs = sess.run(None, {inp: blob})
    postprocess_detections(decode_heads(outs, imgsz=640, conf_thres=args.conf), pad=pad, scale=scale,
                           conf_thres=args.conf, iou_thres=args.iou)
total, pre, run, post = [], [], [], []
dets = []
for i in range(1, args.iterations + 1):
    a = time.perf_counter()
    blob, pad, scale = letterbox(img, 640)
    b = time.perf_counter()
    outs = sess.run(None, {inp: blob})
    c = time.perf_counter()
    dets = postprocess_detections(decode_heads(outs, imgsz=640, conf_thres=args.conf), pad=pad, scale=scale,
                                  conf_thres=args.conf, iou_thres=args.iou)
    d = time.perf_counter()
    pre.append((b - a) * 1e3)
    run.append((c - b) * 1e3)
    post.append((d - c) * 1e3)
    total.append((d - a) * 1e3)
    if i % 100 == 0:
        print(f"[run] frame {i} | G2G {total[-1]:.2f} ms | {len(dets)} objects", flush=True)
v = np.asarray(total)
p50, p95, p99 = np.percentile(v, [50, 95, 99])
rss = proc.memory_info().rss / (1024 * 1024)
names = Counter(dd.class_name for dd in dets)
print(f"[amd] G2G mean {v.mean():.3f} ms | P50 {p50:.3f} | P95 {p95:.3f} | P99 {p99:.3f} | max {v.max():.3f} | "
      f"over {v.size} timed frames after {args.warmup} warm-up", flush=True)
print(f"[amd] stage means (ms): letterbox {np.mean(pre):.3f} | session.run {np.mean(run):.3f} | decode+NMS "
      f"{np.mean(post):.3f}", flush=True)
print(f"[amd] detections on the last frame: {len(dets)} {dict(names)} | RSS {rss:.1f} MB", flush=True)
if args.json:
    args.json.write_text(json.dumps({
        "engine": "ONNX Runtime + Vitis AI EP (Ryzen AI 1.7.1)", "model": args.model.name, "cache_key": args.cache_key,
        "session_load_s": load_s, "ep_node_devices": placement, "warmup": args.warmup, "iterations": args.iterations,
        "g2g_ms": {"mean": float(v.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99),
                   "min": float(v.min()), "max": float(v.max())},
        "stages_ms": {"letterbox": float(np.mean(pre)), "session_run": float(np.mean(run)),
                      "decode_nms": float(np.mean(post))},
        "rss_mb": rss,
        "detections": [{"class_name": dd.class_name, "class_id": dd.class_id, "score": float(dd.score),
                        "xyxy": [float(x) for x in dd.xyxy]} for dd in dets],
    }, indent=1), encoding="utf-8")
