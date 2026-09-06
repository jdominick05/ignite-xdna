"""
YOLO-pose step 4: run pose estimation on an image, video or webcam, on CPU or NPU.

Head-cut only (9 outputs, HEAD_OUTS) - the tail runs in numpy via
npu.yolo_pose_decode, see pipelines/yolov8n-pose/1b_cut_head.py. The full
graph is not wired up here: the detect pipeline found the VitisAI EP takes
zero nodes from the equivalent full yolov8 graph, and the pose tail ends in
the same op types, so there is no reason to expect a different result.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov8n-pose/4_pose.py --model models/yolov8n-pose_cut.onnx \
        --ep cpu --source assets/test_image.jpg
    python pipelines/yolov8n-pose/4_pose.py --model models/yolov8n-pose_cut_xint8.onnx \
        --ep npu --source assets/test_image.jpg --fresh --log 1
    python pipelines/yolov8n-pose/4_pose.py --model models/yolov8n-pose_cut_xint8.onnx \
        --ep npu --source 0        # webcam, q quits
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import npu.yolo_pose as yp
from npu.paths import RESULTS, YOLO_POSE_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolo import input_size
from npu.yolo_pose_decode import decode_heads, head_order

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep", choices=["cpu", "npu"], default="cpu")
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
    cache_key = args.cache_key or YOLO_POSE_CUT_CACHE_KEY
    if args.fresh:
        clear_cache(cache_key)

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    inp = sess.get_inputs()[0].name
    imgsz = input_size(sess.get_inputs()[0].shape, args.model)
    n_out = len(sess.get_outputs())
    if n_out != len(yp.HEAD_OUTS):
        raise SystemExit(f"expected {len(yp.HEAD_OUTS)} head outputs, got {n_out}: "
                         f"{[o.name for o in sess.get_outputs()]}. Full-graph pose "
                         "models are not supported by this script.")
    order = head_order(sess, imgsz)
    print(f"model: head-cut ({n_out} outputs), decode tail runs in numpy")
    print(f"input: {imgsz}x{imgsz}")

    def infer(img):
        """-> dets, infer_ms, post_ms"""
        x, pad, scale = yp.letterbox(img, imgsz)
        t0 = time.perf_counter()
        r = sess.run(None, {inp: x})
        out = decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=args.conf)
        t1 = time.perf_counter()
        dets = yp.postprocess(out, pad, scale, conf_thres=args.conf, iou_thres=args.iou)
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
        print(f"{len(dets)} people:")
        for x0, y0, w, h, s, kpts in sorted(dets, key=lambda d: -d[4]):
            n_vis = int((kpts[:, 2] >= 0.5).sum())
            print(f"   person {s:.2f}  ({int(x0)},{int(y0)},{int(w)},{int(h)})  "
                  f"{n_vis}/{yp.NUM_KPTS} keypoints visible")
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = args.out or str(RESULTS / f"out_{stem}_{args.ep}.jpg")
        cv2.imwrite(out, yp.draw(img, dets))
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
    infer(frame)  # warm-up so the compile isn't counted in the HUD
    print("running - press q to quit")

    times = []
    while True:
        t_all = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        dets, ti, tp = infer(frame)
        yp.draw(frame, dets)
        total = (time.perf_counter() - t_all) * 1000
        times.append(ti)
        if len(times) > 60:
            times.pop(0)
        cv2.putText(frame,
                    f"{args.ep.upper()}  infer {np.mean(times):5.1f} ms  "
                    f"post {tp:4.1f} ms  loop {total:5.1f} ms  {1000 / total:4.1f} fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("yolov8n-pose " + args.ep, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
