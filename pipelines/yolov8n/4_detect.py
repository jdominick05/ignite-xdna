"""
YOLO step 4: run detection on an image, video or webcam, on CPU or NPU.

Handles both graph shapes:
  * full graph  (1 output, `output0`)  - the ONNX decode tail runs in the model
  * head-cut    (6 outputs, HEAD_OUTS) - the tail runs in numpy via
    npu.yolo_decode, see pipelines/yolov8n/1b_cut_head.py
The mode is picked from the session's output count, so the same command works
for either model.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'

    python pipelines/yolov8n/4_detect.py --model models/yolov8n.onnx \
        --ep cpu --source assets/test_image.jpg
    python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx \
        --ep npu --source assets/test_image.jpg --fresh --log 1
    python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx \
        --ep npu --source 0        # webcam, q quits
    python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx \
        --ep npu --source 0 --max-seconds 15   # logged webcam run: auto-quits,
        # prints a per-frame summary over the whole run, and drops annotated
        # snapshots in outputs/webcam/ so the run leaves evidence behind

A camera run needs a person in front of the camera to mean anything: the summary
says so explicitly when it ends with zero detections.

Cache key defaults by model name: anything with "cut" in it uses
yolocutcachekey, everything else yolocachekey, so a cut compile can never be
served from a full-graph cache. Override with --cache-key.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Must precede every cv2 import, the transitive one through npu.yolo included:
# OpenCV reads this at videoio init, and setting it later is a measured silent
# no-op (tools/cam_probe.py --case msmf_late: still ~90s, with the variable
# reading back as "0"). Without it, MSMF -- the default Windows backend -- takes a
# fixed ~90s to open a webcam. Irrelevant to --source on a file.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

import npu.yolo as yc
from npu.paths import ROOT, RESULTS, YOLO_CACHE_KEY, YOLO_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolo_decode import decode_heads, head_order

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
    # Camera/video only. The webcam path used to leave nothing behind: cv2.imshow
    # and a 60-frame rolling HUD mean, both gone the moment the window closed, so
    # an attended run could not be evidence for anything. --max-seconds matches
    # demos/webcam_multipartition_demo.py's flag of the same name (15s windows
    # there), and the summary below is what makes the run quotable.
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="camera/video only: auto-quit after N seconds, for logged runs")
    ap.add_argument("--snapshot-dir", default=None,
                    help="camera/video only: where annotated snapshots go "
                         "(default outputs/webcam/). NEVER results/: these are "
                         "camera frames of whoever is at the machine, and "
                         "results/ is tracked")
    ap.add_argument("--log", type=int, default=1, help="0=verbose 1=info 2=warning")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.model))[0]
    cache_key = args.cache_key or (
        YOLO_CUT_CACHE_KEY if "cut" in stem.lower() else YOLO_CACHE_KEY)
    # Print it rather than let a caller re-derive it. Which cache a run used is
    # exactly the staleness question this repo keeps getting bitten by, and a
    # wrapper that needs the key for npu_verdict can read it back from the log
    # instead of hand-rolling a second copy of the rule above.
    print(f"cache key: {cache_key}")
    if args.fresh:
        clear_cache(cache_key)

    sess = build_session(args.model, args.ep, cache_key, args.xclbin,
                         log_severity=args.log)
    inp = sess.get_inputs()[0].name
    # Read the letterbox size off the model instead of assuming 640. This is the
    # same invariant that keeps preprocessing byte-identical between calibration
    # and inference: both sides read the number out of the graph, so a 512 model
    # cannot be fed 640 pixels by a caller who forgot a flag.
    imgsz = yc.input_size(sess.get_inputs()[0].shape, args.model)
    n_out = len(sess.get_outputs())
    # run_raw is timed alone as "infer" -- pure sess.run, comparable across EPs
    # and graph shapes. decode_raw (DFL/anchor decode for a cut model) is
    # folded into "post" instead: it runs in numpy regardless of EP, its cost
    # depends on --conf (more candidates survive the prefilter at low conf),
    # and lumping it into "infer" makes a cut NPU model's number include work
    # a full-graph model does inside the EP for free. Confirmed this matters:
    # the cut model's mean infer time moved from 8.94ms to 15.01ms between
    # --conf 0.25 and --conf 0.001 with nothing else changed.
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
            return r[0]
    else:
        raise SystemExit(f"don't know what to do with {n_out} outputs: "
                         f"{[o.name for o in sess.get_outputs()]}")
    print(f"input: {imgsz}x{imgsz}")

    def infer(img):
        """-> dets, infer_ms, post_ms. infer_ms is sess.run only; post_ms is
        numpy decode (cut models) + NMS, so infer_ms means the same thing for
        every EP and graph shape."""
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

    # Report what the camera actually delivered, not what was asked for above. The
    # multipartition demo's finding was that capture rate, not the NPU, capped the
    # on-screen number -- so if this run's fps differs from that one's 30.0, the
    # requested 1280x720 (which that demo never asked for) is a named confound
    # rather than a silent one.
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"source: {src_w}x{src_h} @ {src_fps:.1f} fps capture rate "
          f"-- that, not the NPU, may be the ceiling here", flush=True)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no frames")
    infer(frame)  # warm-up so the compile isn't counted in the HUD
    print("running - press q to quit", flush=True)

    times = []                      # rolling 60, for the on-screen HUD only
    all_infer, all_post, all_loop = [], [], []   # every frame, for the summary
    class_counts = {}
    best_n, best_frame, last_frame = -1, None, None
    run_start = time.perf_counter()
    last_log = run_start
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
        all_infer.append(ti)
        all_post.append(tp)
        all_loop.append(total)
        for *_rest, c in dets:
            name = yc.COCO_CLASSES[c]
            class_counts[name] = class_counts.get(name, 0) + 1
        # Keep the frame that best shows the thing this run exists to confirm:
        # real boxes drawn on a real subject. yc.draw annotates in place, so the
        # copy has to happen after it.
        last_frame = frame
        if len(dets) > best_n:
            best_n, best_frame = len(dets), frame.copy()
        cv2.putText(frame,
                    f"{args.ep.upper()}  infer {np.mean(times):5.1f} ms  "
                    f"post {tp:4.1f} ms  loop {total:5.1f} ms  {1000 / total:4.1f} fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("yolov8n " + args.ep, frame)

        now = time.perf_counter()
        if now - last_log >= 1.0:
            print(f"t={now - run_start:5.1f}s  infer {np.mean(times):5.1f} ms  "
                  f"loop {total:5.1f} ms  {1000 / total:4.1f} fps  "
                  f"frames={len(all_infer)}  dets={len(dets)}", flush=True)
            last_log = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if args.max_seconds is not None and now - run_start >= args.max_seconds:
            print(f"reached --max-seconds {args.max_seconds:.0f}s, quitting", flush=True)
            break
    elapsed = time.perf_counter() - run_start
    cap.release()
    cv2.destroyAllWindows()

    if not all_infer:
        raise SystemExit("no frames were processed -- nothing to report")

    # The summary is the evidence. The HUD above is a 60-frame rolling mean that
    # vanishes with the window; these are every frame of the run.
    ai, al = np.array(all_infer), np.array(all_loop)
    print(f"\n=== {args.ep.upper()} | {args.model} | {imgsz}px | source {args.source} "
          f"({src_w}x{src_h} @ {src_fps:.1f} fps) ===")
    print(f"frames  {len(all_infer)} in {elapsed:.1f}s "
          f"({len(all_infer) / elapsed:.1f} fps end to end)")
    print(f"infer   mean {ai.mean():.2f} ms   median {np.median(ai):.2f} ms   "
          f"p95 {np.percentile(ai, 95):.2f} ms   ({1000 / ai.mean():.1f} fps if infer-bound)")
    print(f"post    mean {np.mean(all_post):.2f} ms")
    print(f"loop    mean {al.mean():.2f} ms   median {np.median(al):.2f} ms   "
          f"p95 {np.percentile(al, 95):.2f} ms")
    if class_counts:
        tot = sum(class_counts.values())
        print(f"detections {tot} across {len(all_infer)} frames "
              f"({tot / len(all_infer):.2f}/frame):")
        for name, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
            print(f"   {name:16s} {n}")
    else:
        print("detections 0 -- nothing was in front of the camera, or the model "
              "found nothing. A run with no detections does not verify the demo.")

    snap_dir = Path(args.snapshot_dir) if args.snapshot_dir else ROOT / "outputs" / "webcam"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for tag, img in (("best", best_frame), ("last", last_frame)):
        if img is not None:
            p = snap_dir / f"webcam_{stem}_{args.ep}_{tag}.jpg"
            cv2.imwrite(str(p), img)
            print(f"wrote {p}" + (f"  ({best_n} detections)" if tag == "best" else ""))


if __name__ == "__main__":
    main()
