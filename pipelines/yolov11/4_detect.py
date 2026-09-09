"""
YOLOv11 step 4: run detection on an image, video or webcam, on CPU, DML or NPU.

Handles both graph shapes:
  * full graph  (1 output, `output0`)  - the ONNX decode tail runs in the model
  * head-cut    (6 outputs, HEAD_OUTS) - the tail runs in numpy via
    npu.yolov11.decode_heads

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov11/4_detect.py --model models/yolo11n.onnx --ep cpu --source assets/test_image.jpg
    python pipelines/yolov11/4_detect.py --model models/yolo11n_cut_xint8.onnx --ep npu --source assets/test_image.jpg --fresh --log 1
    python pipelines/yolov11/4_detect.py --model models/yolo11n_cut_xint8.onnx --ep dml --source assets/test_image.jpg
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import npu.yolov11 as y11
from npu.paths import RESULTS, yolo11_cache_key
from npu.session import build_session, clear_cache

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep", choices=["cpu", "dml", "npu"], default="cpu")
    ap.add_argument("--source", default="0", help="image path, video path, or camera index")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--cache-key", default=None)
    ap.add_argument("--fresh", action="store_true", help="delete compile cache first")
    ap.add_argument("--out", default=None, help="output image path (image source only)")
    ap.add_argument("--runs", type=int, default=20, help="timed runs, image source only")
    ap.add_argument("--log", type=int, default=1, help="0=verbose 1=info 2=warning")
    args = ap.parse_args()

    cache_key = args.cache_key or yolo11_cache_key(args.model)
    if args.fresh:
        clear_cache(cache_key)

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    inp = sess.get_inputs()[0].name
    imgsz = y11.input_size(sess.get_inputs()[0].shape, args.model)
    n_out = len(sess.get_outputs())

    if n_out == len(y11.HEAD_OUTS):
        order = y11.head_order(sess, imgsz)
        print(f"model: head-cut ({n_out} outputs), decode tail runs in numpy")

        def run_raw(x):
            return sess.run(None, {inp: x})

        def decode_raw(r):
            return y11.decode_heads([r[i] for i in order], imgsz=imgsz,
                                    conf_thres=args.conf)
    elif n_out == 1:
        print("model: full graph (1 output), decode tail runs in ONNX")

        def run_raw(x):
            return sess.run(None, {inp: x})

        def decode_raw(r):
            return r[0]
    else:
        raise SystemExit(
            f"model has {n_out} outputs; expected 1 (full graph) or "
            f"{len(y11.HEAD_OUTS)} (head-cut)"
        )

    # Warmup
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    for _ in range(3):
        decode_raw(run_raw(dummy))

    src = args.source
    is_img = src.lower().endswith(IMAGE_EXTS)
    is_cam = src.isdigit()

    if is_img:
        orig = cv2.imread(src)
        if orig is None:
            raise SystemExit(f"could not read image: {src}")

        x, pad, scale = y11.letterbox(orig, imgsz)
        # Warmup
        for _ in range(3):
            decode_raw(run_raw(x))
        # Timed runs
        latencies = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            raw = run_raw(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        # Timed decode / postprocess
        t_post0 = time.perf_counter()
        preds = decode_raw(raw)
        dets = y11.postprocess(preds, pad, scale,
                               conf_thres=args.conf, iou_thres=args.iou)
        t_post = (time.perf_counter() - t_post0) * 1000.0

        ts = np.array(latencies)
        print(f"\n=== {args.ep.upper()} | {args.model} | {imgsz}px | {args.source} ===")
        print(f"infer   mean {ts.mean():.2f} ms   median {np.median(ts):.2f} ms   "
              f"p95 {np.percentile(ts, 95):.2f} ms   ({1000 / ts.mean():.1f} fps)")
        print(f"post    mean {t_post:.2f} ms")
        print(f"{len(dets)} detections:")
        for x0, y0, w, h, s, c in sorted(dets, key=lambda d: -d[4])[:10]:
            cls_name = y11.COCO_CLASSES[c] if c < len(y11.COCO_CLASSES) else str(c)
            print(f"  {cls_name:15s} {s:5.2f}   [{x0:.1f}, {y0:.1f}, {w:.1f}, {h:.1f}]")

        if args.out:
            y11.draw(orig, dets)
            cv2.imwrite(args.out, orig)
            print(f"saved visualization to {args.out}")
        return

    # Video or camera stream
    cap = cv2.VideoCapture(int(src) if is_cam else src)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source: {src}")

    print("Running stream detection. Press 'q' in window to exit.")
    fps_smooth = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, pad, scale = y11.letterbox(frame, imgsz)
        t0 = time.perf_counter()
        raw = run_raw(x)
        preds = decode_raw(raw)
        dets = y11.postprocess(preds, pad, scale,
                               conf_thres=args.conf, iou_thres=args.iou)
        dt = time.perf_counter() - t0
        cur_fps = 1.0 / max(dt, 1e-6)
        fps_smooth = cur_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * cur_fps)

        y11.draw(frame, dets)
        cv2.putText(frame, f"FPS: {fps_smooth:4.1f} ({args.ep})", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("YOLOv11 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
