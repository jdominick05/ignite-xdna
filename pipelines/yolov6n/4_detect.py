"""
YOLOv6n step 4: run detection on an image, video or webcam, on CPU or NPU.

Handles both graph shapes:
  * full graph  (1 output, `outputs`)   - the ONNX decode tail runs in the model
  * head-cut    (6 outputs, HEAD_OUTS)  - the tail runs in numpy via
    npu.yolov6_decode, see pipelines/yolov6n/1b_cut_head.py
The mode is picked from the session's output count, same as yolov8n's 4_detect.py.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov6n/4_detect.py --model models/yolov6n.onnx \
        --ep cpu --source assets/test_image.jpg
    python pipelines/yolov6n/4_detect.py --model models/yolov6n_cut_xint8.onnx \
        --ep npu --source assets/test_image.jpg --fresh --log 1

Cache key: npu.paths.YOLOV6_CUT_CACHE_KEY, its own key so a yolov6n compile
can never be served from (or collide with) a yolov8-family cache.
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

import npu.yolov6 as yc
from npu.paths import RESULTS, YOLOV6_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolov6_decode import decode_heads, head_order

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

    stem = os.path.splitext(os.path.basename(args.model))[0]
    cache_key = args.cache_key or YOLOV6_CUT_CACHE_KEY
    if args.fresh:
        clear_cache(cache_key)

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    inp = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, args.model)
    n_out = len(sess.get_outputs())
    if n_out == len(yc.HEAD_OUTS):
        order = head_order(sess, imgsz)
        print(f"model: head-cut ({n_out} outputs), decode tail runs in numpy")

        def run_raw(x):
            return sess.run(None, {inp: x})

        def decode_raw(r):
            return decode_heads([r[i] for i in order], imgsz=imgsz,
                                conf_thres=args.conf)
    elif n_out == 1:
        print("model: full graph (1 output), decode tail runs in the ONNX model")

        def run_raw(x):
            return sess.run(None, {inp: x})

        def decode_raw(r):
            # (1,8400,85) -> (1,84,8400): drop the constant-1.0 objectness
            # column so this matches decode_heads' (1, 4+nc, N) contract and
            # feeds npu.yolo.postprocess unchanged.
            out = r[0][0]
            return np.concatenate([out[:, :4], out[:, 5:]], axis=1).T[None]
    else:
        raise SystemExit(f"don't know what to do with {n_out} outputs: "
                         f"{[o.name for o in sess.get_outputs()]}")
    print(f"input: {imgsz}x{imgsz}")

    def infer(img):
        x, pad, scale = yc.letterbox(img, imgsz)
        t0 = time.perf_counter()
        raw = run_raw(x)
        t1 = time.perf_counter()
        out = decode_raw(raw)
        dets = yc.postprocess(out, pad, scale, conf_thres=args.conf, iou_thres=args.iou)
        t2 = time.perf_counter()
        return dets, (t1 - t0) * 1000, (t2 - t1) * 1000

    is_cam = args.source.isdigit()
    ext = os.path.splitext(args.source)[1].lower()

    if not is_cam and ext in IMAGE_EXTS:
        img = cv2.imread(args.source)
        if img is None:
            raise SystemExit(f"could not read {args.source}")
        infer(img)  # warm-up / compile, not timed
        ts, ps = [], []
        for _ in range(max(1, args.runs)):
            dets, ti, tp = infer(img)
            ts.append(ti)
            ps.append(tp)
        ts = np.array(ts)
        print(f"\n=== {args.ep.upper()} | {args.model} | {imgsz}px | {args.source} ===")
        print(f"infer   mean {ts.mean():.2f} ms   median {np.median(ts):.2f} ms   "
              f"p95 {np.percentile(ts, 95):.2f} ms   ({1000 / ts.mean():.1f} fps)")
        print(f"post    mean {np.mean(ps):.2f} ms")
        print(f"{len(dets)} detections:")
        for x0, y0, w, h, s, c in sorted(dets, key=lambda d: -d[4]):
            print(f"   {yc.COCO_CLASSES[c]:16s} {s:.2f}  "
                  f"({int(x0)},{int(y0)},{int(w)},{int(h)})")
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = args.out or str(RESULTS / f"out_{stem}_{args.ep}.jpg")
        cv2.imwrite(out, yc.draw(img, dets))
        print("wrote", out)
        return

    if not is_cam and ext not in VIDEO_EXTS:
        raise SystemExit(f"unrecognised source {args.source!r}: expected an image "
                         f"{IMAGE_EXTS}, a video {VIDEO_EXTS}, or a camera index")

    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {args.source}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no frames")
    infer(frame)
    print("running - press q to quit")

    times = []
    while True:
        t_all = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        dets, ti, tp = infer(frame)
        yc.draw(frame, dets)
        total = (time.perf_counter() - t_all) * 1000
        times.append(ti)
        if len(times) > 60:
            times.pop(0)
        cv2.putText(frame,
                    f"{args.ep.upper()}  infer {np.mean(times):5.1f} ms  "
                    f"post {tp:4.1f} ms  loop {total:5.1f} ms  {1000 / total:4.1f} fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("yolov6n " + args.ep, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
