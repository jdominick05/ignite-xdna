"""
AdaRound diff demo: run plain XINT8 and XINT8+AdaRound on the same image and
visualise exactly what quantization noise costs and what AdaRound recovers.

Boxes are colour-coded by match status:
  GREEN  — shared detection (IoU >= iou_match with both models above conf)
  ORANGE — XINT8-only (AdaRound missed or downgraded below conf)
  BLUE   — AdaRound-only (XINT8 missed or dropped below conf)

Saves a side-by-side comparison JPEG and a text diff to results/.

    conda activate resnet_env17
    python tools/adaround_diff_demo.py
    python tools/adaround_diff_demo.py --variant s --conf 0.25
    python tools/adaround_diff_demo.py --source assets/test_image.jpg --runs 40
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import npu.yolo as yc
from npu.session import build_session, clear_cache
from npu.yolo_decode import decode_heads, head_order

# Colours in BGR
COL_SHARED   = (0,   200,  0)    # green  — both agree
COL_XINT8    = (0,   140, 255)   # orange — XINT8-only
COL_ADAROUND = (255, 100,   0)   # blue   — AdaRound-only
COL_LEGEND   = (230, 230, 230)


def _run(model_path, img, runs, ep, log_sev):
    """Compile, warm up, time, decode. Returns (dets, mean_ms)."""
    cache_key = "yolocutcachekey"
    clear_cache(cache_key)
    sess = build_session(str(model_path), ep, cache_key, log_severity=log_sev)
    inp  = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, str(model_path))
    order = head_order(sess, imgsz)

    x, pad, scale = yc.letterbox(img, imgsz)

    def infer():
        raw = sess.run(None, {inp: x})
        out = decode_heads([raw[i] for i in order], imgsz=imgsz, conf_thres=0.001)
        return yc.postprocess(out, pad, scale, conf_thres=args_conf,
                              iou_thres=0.5, max_det=300)

    for _ in range(3):          # warm-up
        infer()

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        ts.append((time.perf_counter() - t0) * 1000)

    dets = infer()
    return dets, float(np.mean(ts))


args_conf = 0.25   # module-level so _run can see it (set properly in main)


def _iou(b1, b2):
    """IoU of two (x,y,w,h) boxes."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
    inter  = max(0, x2-x1) * max(0, y2-y1)
    union  = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def _match(dets_a, dets_b, iou_thresh=0.45):
    """
    Greedy IoU match between two detection lists.
    Returns (shared_a, shared_b, only_a, only_b).
    shared_* are paired lists (same index = same detection).
    """
    used_b = set()
    shared_a, shared_b = [], []
    only_a = []
    for da in sorted(dets_a, key=lambda d: -d[4]):
        best_iou, best_j = 0.0, -1
        for j, db in enumerate(dets_b):
            if j in used_b or db[5] != da[5]:   # must be same class
                continue
            iou = _iou(da, db)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            shared_a.append(da)
            shared_b.append(dets_b[best_j])
            used_b.add(best_j)
        else:
            only_a.append(da)
    only_b = [db for j, db in enumerate(dets_b) if j not in used_b]
    return shared_a, shared_b, only_a, only_b


def _draw_boxes(img, dets, colour, label_prefix=""):
    for x, y, w, h, s, c in dets:
        p1, p2 = (int(x), int(y)), (int(x+w), int(y+h))
        cv2.rectangle(img, p1, p2, colour, 2)
        label = f"{label_prefix}{yc.COCO_CLASSES[c]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (p1[0], max(p1[1]-th-4, 0)),
                      (p1[0]+tw+2, max(p1[1], th+4)), colour, -1)
        cv2.putText(img, label, (p1[0]+1, max(p1[1]-3, th+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)


def _legend(img, x0, y0):
    items = [
        (COL_SHARED,   "shared (both agree)"),
        (COL_XINT8,    "XINT8 only (AdaRound missed)"),
        (COL_ADAROUND, "AdaRound only (XINT8 missed)"),
    ]
    for i, (col, text) in enumerate(items):
        y = y0 + i * 22
        cv2.rectangle(img, (x0, y), (x0+16, y+14), col, -1)
        cv2.putText(img, text, (x0+22, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_LEGEND, 1, cv2.LINE_AA)


def main():
    global args_conf
    ap = argparse.ArgumentParser(
        description="Visualise what AdaRound recovers vs plain XINT8 on one image.")
    ap.add_argument("--variant", choices=["n", "s"], default="n",
                    help="yolov8 variant with an AdaRound sibling (n or s, default n)")
    ap.add_argument("--source", default="assets/test_image.jpg")
    ap.add_argument("--ep", choices=["npu", "cpu"], default="npu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.45,
                    help="IoU threshold for declaring two boxes the same detection")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    args_conf = args.conf   # propagate to module level for _run

    src = ROOT / args.source if not os.path.isabs(args.source) else Path(args.source)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"cannot read image: {src}")

    v = args.variant
    xint8_path    = ROOT / "models" / f"yolov8{v}_cut_xint8.onnx"
    adaround_path = ROOT / "models" / f"yolov8{v}_cut_xint8_adaround.onnx"
    for p in (xint8_path, adaround_path):
        if not p.exists():
            raise SystemExit(f"model not found: {p}")

    print(f"\nAdaRound diff demo — yolov8{v} — {args.ep.upper()} — conf {args.conf}")
    print(f"Image : {src.name}  ({img.shape[1]}×{img.shape[0]})")
    print()

    print(f"  [1/2] plain XINT8 ({xint8_path.name}) …")
    dets_x, ms_x = _run(xint8_path, img, args.runs, args.ep, args.log)
    print(f"        {ms_x:.2f} ms mean   {len(dets_x)} detections")

    print(f"  [2/2] XINT8 + AdaRound ({adaround_path.name}) …")
    dets_a, ms_a = _run(adaround_path, img, args.runs, args.ep, args.log)
    print(f"        {ms_a:.2f} ms mean   {len(dets_a)} detections")

    # ------------------------------------------------------------------ #
    # Match boxes                                                         #
    # ------------------------------------------------------------------ #
    shared_x, shared_a, only_x, only_a = _match(dets_x, dets_a, args.iou_match)

    # ------------------------------------------------------------------ #
    # Text diff                                                           #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 64)
    print(f"  XINT8       {len(dets_x):>3} dets   {ms_x:>7.2f} ms  ({1000/ms_x:.1f} fps)")
    print(f"  AdaRound    {len(dets_a):>3} dets   {ms_a:>7.2f} ms  ({1000/ms_a:.1f} fps)")
    lat_delta = ms_a - ms_x
    print(f"  Δ latency  {lat_delta:>+.2f} ms  "
          f"({'AdaRound faster' if lat_delta < 0 else 'XINT8 faster'})")
    print()
    print(f"  Shared (IoU ≥ {args.iou_match}):  {len(shared_x)}")
    print(f"  XINT8-only  (orange):  {len(only_x)}")
    print(f"  AdaRound-only (blue):  {len(only_a)}")
    print("=" * 64)

    if only_x:
        print("\n  XINT8 found, AdaRound missed:")
        for x0,y0,w,h,s,c in sorted(only_x, key=lambda d: -d[4]):
            print(f"    – {yc.COCO_CLASSES[c]:16s}  conf {s:.3f}")
    if only_a:
        print("\n  AdaRound found, XINT8 missed:")
        for x0,y0,w,h,s,c in sorted(only_a, key=lambda d: -d[4]):
            print(f"    + {yc.COCO_CLASSES[c]:16s}  conf {s:.3f}")

    # Shared confidence delta
    if shared_x:
        deltas = [a[4] - x[4] for x, a in zip(shared_x, shared_a)]
        print(f"\n  Shared boxes: AdaRound conf delta  "
              f"mean {np.mean(deltas):+.3f}  "
              f"median {np.median(deltas):+.3f}  "
              f"({sum(d>0 for d in deltas)}/{len(deltas)} higher)")

    # ------------------------------------------------------------------ #
    # Render side-by-side image                                           #
    # ------------------------------------------------------------------ #
    left  = img.copy()
    right = img.copy()

    _draw_boxes(left,  shared_x, COL_SHARED)
    _draw_boxes(left,  only_x,   COL_XINT8)
    _draw_boxes(right, shared_a, COL_SHARED)
    _draw_boxes(right, only_a,   COL_ADAROUND)

    # Header bars
    hdr_h = 36
    H, W  = img.shape[:2]
    left_hdr  = np.zeros((hdr_h, W, 3), dtype=np.uint8)
    right_hdr = np.zeros((hdr_h, W, 3), dtype=np.uint8)
    cv2.putText(left_hdr,  f"XINT8  {ms_x:.1f}ms  {len(dets_x)} dets",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,200), 2)
    cv2.putText(right_hdr, f"AdaRound  {ms_a:.1f}ms  {len(dets_a)} dets",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,200), 2)

    left_panel  = np.vstack([left_hdr,  left])
    right_panel = np.vstack([right_hdr, right])
    side_by_side = np.hstack([left_panel, right_panel])

    # Legend bar at bottom
    leg_h = 80
    leg = np.zeros((leg_h, side_by_side.shape[1], 3), dtype=np.uint8)
    _legend(leg, 16, 12)
    cv2.putText(leg, f"yolov8{v}  {args.ep.upper()}  conf≥{args.conf}  "
                     f"IoU-match≥{args.iou_match}  image: {src.name}",
                (16, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
    composite = np.vstack([side_by_side, leg])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adaround_diff_yolov8{v}_{args.ep}.jpg"
    cv2.imwrite(str(out_path), composite)
    print(f"\n  Saved → {out_path}")
    print()


if __name__ == "__main__":
    main()
