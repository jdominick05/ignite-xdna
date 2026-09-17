"""
YOLO-World v2 step 4b: glass-to-glass latency on one image, the same way for every stack.

Why:
  5_eval_map.py times the network call only, so it leaves out what each stack pays around it: turning a frame into the
  model input and the contrastive decode, which grows with the vocabulary. This times a frame to its detections.

  --ep cpu | dml | npu (ONNX Runtime on a head-cut model; npu is AMD's Vitis AI EP)
      preprocess   npu.yolow.letterbox: cv2 resize, pad, RGB, /255, NCHW float32
      network      session.run
      decode       YoloWorldDecoder.decode on the float heads + postprocess (per-class NMS)
  --ep ignite (a graph-engine container through ignite_xdna.pipelines.yolow_pipeline.YoloWorldPipeline)
      preprocess   native letterbox and quantization straight into the NPU input plane
      network      NPU segments, attention host steps and head readback
      decode       the int8 decode (bit-exact with the float one) + postprocess

  Every stack decodes with the same class embeddings and contrastive constants, from the text bundle of
  6_text_encoder.py. --classes other than COCO's names rewrites the ONNX models' text guides (a temporary copy, which
  AMD's EP compiles again under its own cache key) and sets the container's host constants.

    conda activate resnet_env17        # cpu, dml, npu (with RYZEN_AI_INSTALLATION_PATH set)
    python pipelines/yolow/4b_g2g.py --model models/yolov8s-worldv2_cut_xint8_fp32cv2.onnx --ep npu --cache-key yolow_fp32D
    conda activate mlir-aie-iron       # ignite
    python pipelines/yolow/4b_g2g.py --model build/yolow.ignite --ep ignite

Prints per-stage mean, median, p95 and p99 over --frames timed frames after --warmup, the detections per frame, and
with --json writes the record. --max-fps paces the timed frames on an absolute schedule like a camera (the wait is not
part of a frame's time), and every 100th timed frame prints a "[run] frame N" line, which tools/energy_sitting.py uses
as its measurement window.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import npu.yolow as yw
from npu.paths import MODELS, yolow_cache_key


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="head-cut ONNX model, or a graph-engine .ignite container for --ep ignite")
    ap.add_argument("--ep", choices=["cpu", "dml", "npu", "ignite"], required=True)
    ap.add_argument("--source", default=str(ROOT / "assets" / "bus.jpg"))
    ap.add_argument("--classes", default=None, help="comma-separated class names (default: COCO's 80)")
    ap.add_argument("--text-encoder", default=str(MODELS / "yolow_text_encoder.onnx"))
    ap.add_argument("--text-bundle", default=str(MODELS / "yolow_text.npz"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--max-fps", type=float, default=0.0, help="pace the timed frames to this rate (0: flat out)")
    ap.add_argument("--cache-key", default=None)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--log", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    names = [c.strip() for c in args.classes.split(",")] if args.classes else list(yw.COCO_CLASSES)
    img = cv2.imread(args.source)
    if img is None:
        raise SystemExit(f"could not read {args.source}")
    rows, dets = [], []
    period = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    schedule = {}

    def before_frame(k):
        """Absolute schedule from the first timed frame: a late frame does not make the next one early."""
        if k < args.warmup or not period:
            return
        start = schedule.setdefault("t0", time.perf_counter())
        delay = start + (k - args.warmup) * period - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def after_frame(k, g2g_ms, n):
        i = k - args.warmup + 1
        if i > 0 and i % 100 == 0:
            print(f"[run] frame {i} | G2G {g2g_ms:.2f} ms | {n} detections", flush=True)

    if args.ep == "ignite":
        from ignite_xdna.pipelines.yolow_pipeline import YoloWorldPipeline
        with YoloWorldPipeline(args.model, args.text_encoder, args.text_bundle, names, conf_thres=args.conf,
                               iou_thres=args.iou) as world:
            desc = f"graph-engine container ({len(world.session.segments)} segments), native ingress, int8 decode"
            for k in range(args.warmup + args.frames):
                before_frame(k)
                d, t = world.predict_sync(img)
                if k >= args.warmup:
                    rows.append((t.preprocess_ms, t.npu_forward_ms, t.postprocess_ms, t.glass_to_glass_ms))
                    dets.append(len(d))
                after_frame(k, t.glass_to_glass_ms, len(d))
    else:
        from ignite_xdna.pipelines.yolow_pipeline import YoloWorldDecoder
        from ignite_xdna.pipelines.yolow_text import YoloWorldText
        from npu.session import build_session
        text = YoloWorldText(args.text_encoder, args.text_bundle)
        embeddings, guides = text.vocabulary(names)
        model_path = args.model
        cache_key = args.cache_key or yolow_cache_key(args.model)
        if args.classes:
            import onnx
            from onnx import numpy_helper
            m = onnx.load(args.model)
            for t in m.graph.initializer:
                if t.name in guides:
                    t.CopyFrom(numpy_helper.from_array(guides[t.name], t.name))
            model_path = os.path.join(tempfile.mkdtemp(prefix="yolow_g2g_"), Path(args.model).stem + "_vocab.onnx")
            onnx.save(m, model_path)
            cache_key = cache_key + "_vocab"
        sess = build_session(model_path, args.ep, cache_key, args.xclbin, log_severity=args.log)
        imgsz = yw.input_size(sess.get_inputs()[0].shape, args.model)
        order = yw.head_order(sess, imgsz)
        inp = sess.get_inputs()[0].name
        dec = YoloWorldDecoder(embeddings, names, text.contrastive_scales, text.contrastive_biases, imgsz=imgsz,
                               conf_thres=args.conf, iou_thres=args.iou)
        desc = f"ONNX Runtime {args.ep} ({sess.get_providers()[0]}), numpy letterbox, float decode"
        for k in range(args.warmup + args.frames):
            before_frame(k)
            t0 = time.perf_counter()
            x, pad, scale = yw.letterbox(img, imgsz)
            t1 = time.perf_counter()
            raw = sess.run(None, {inp: x})
            t2 = time.perf_counter()
            d = dec.postprocess(dec.decode([raw[i] for i in order], None, args.conf), pad, scale)
            t3 = time.perf_counter()
            if k >= args.warmup:
                rows.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3))
                dets.append(len(d))
            after_frame(k, (t3 - t0) * 1e3, len(d))

    a = np.asarray(rows)
    record = {"ep": args.ep, "model": Path(args.model).name, "source": Path(args.source).name, "classes": len(names),
              "warmup": args.warmup, "frames": args.frames, "description": desc,
              "preprocess_ms": stats(a[:, 0]), "network_ms": stats(a[:, 1]), "decode_ms": stats(a[:, 2]),
              "glass_to_glass_ms": stats(a[:, 3]), "detections_per_frame": float(np.mean(dets))}
    print(f"{desc}; {len(names)} classes; {args.frames} frames of {Path(args.source).name} after {args.warmup} warm-up")
    for key in ("preprocess_ms", "network_ms", "decode_ms", "glass_to_glass_ms"):
        s = record[key]
        print(f"  {key:<18} mean {s['mean']:7.3f}  median {s['median']:7.3f}  p95 {s['p95']:7.3f}  p99 {s['p99']:7.3f}")
    print(f"  detections per frame {record['detections_per_frame']:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
