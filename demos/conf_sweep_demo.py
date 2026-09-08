"""
Confidence threshold sweep: run one model at conf 0.001 (mAP eval) through
conf 0.90, show which detections survive at each level, and quantify what
'demo conf 0.25 deletes the tail of the precision-recall curve that AP
integrates' (CLAUDE.md invariant) looks like on a real image.

Saves a vertically-stacked composite JPEG — one annotated strip per threshold.

    conda activate resnet_env17
    python tools/conf_sweep_demo.py
    python tools/conf_sweep_demo.py --variant m --source assets/test_image.jpg
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

THRESHOLDS = [0.001, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]

# Colour goes from blue (low conf = cold) to red (high conf = hot), BGR
def _conf_colour(score):
    t = score  # 0..1
    r = int(255 * t)
    b = int(255 * (1 - t))
    return (b, 80, r)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep confidence thresholds and visualise what each level keeps.")
    ap.add_argument("--variant", choices=["n", "s", "m"], default="m")
    ap.add_argument("--source", default="assets/test_image.jpg")
    ap.add_argument("--ep", choices=["npu", "cpu"], default="npu")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    src = ROOT / args.source if not os.path.isabs(args.source) else Path(args.source)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"cannot read: {src}")

    v = args.variant
    # Prefer calib-200 for a better-calibrated model; fall back to plain.
    candidates = [
        ROOT / "models" / f"yolov8{v}_cut_xint8_c200.onnx",
        ROOT / "models" / f"yolov8{v}_cut_xint8.onnx",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise SystemExit(f"no yolov8{v} cut XINT8 model found in models/")

    print(f"\nConf threshold sweep — yolov8{v} — {args.ep.upper()}")
    print(f"Model : {model_path.name}")
    print(f"Image : {src.name}  ({img.shape[1]}×{img.shape[0]})")
    print()

    # Build session once, reuse for all thresholds.
    clear_cache("yolocutcachekey")
    sess = build_session(str(model_path), args.ep, "yolocutcachekey",
                         log_severity=args.log)
    inp   = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, str(model_path))
    order = head_order(sess, imgsz)

    x, pad, scale = yc.letterbox(img, imgsz)

    # Single inference pass at rock-bottom conf; filter in Python per threshold.
    print("  Compiling / running …", end="", flush=True)
    sess.run(None, {inp: x})   # warm-up / compile
    t0  = time.perf_counter()
    raw = sess.run(None, {inp: x})
    ms  = (time.perf_counter() - t0) * 1000
    print(f"  {ms:.1f} ms")
    print()

    out_decoded = decode_heads([raw[i] for i in order], imgsz=imgsz, conf_thres=0.001)
    # All detections at conf >= 0.001, no NMS yet — apply NMS per threshold below.
    all_dets_at_001 = yc.postprocess(out_decoded, pad, scale,
                                     conf_thres=0.001, iou_thres=0.7,
                                     max_det=300)

    # ------------------------------------------------------------------ #
    # Per-threshold table                                                 #
    # ------------------------------------------------------------------ #
    max_dets = len(all_dets_at_001)
    strips   = []

    print(f"  {'conf':>6}  {'dets':>4}  bar                       classes found")
    print(f"  {'':─>6}  {'':─>4}  {'':─>24}  {'':─>30}")

    for thr in THRESHOLDS:
        # Re-filter the already-decoded detections (avoids re-running the model).
        dets = yc.postprocess(out_decoded, pad, scale,
                              conf_thres=thr, iou_thres=0.7, max_det=300)
        n    = len(dets)
        frac = n / max_dets if max_dets > 0 else 0
        bar  = "█" * int(round(frac * 22)) + "░" * (22 - int(round(frac * 22)))

        tags = []
        for _, _, _, _, _, c in sorted(dets, key=lambda d: d[5]):
            name = yc.COCO_CLASSES[c]
            if name not in tags:
                tags.append(name)

        marker = "  ← mAP eval" if thr == 0.001 else \
                 "  ← demo default" if thr == 0.25 else ""
        print(f"  {thr:>6.3f}  {n:>4}  {bar}  {', '.join(tags[:5])}"
              f"{'…' if len(tags)>5 else ''}{marker}")

        # Build annotated strip for the composite image.
        strip_img = img.copy()
        for x0, y0, w, h, s, c in sorted(dets, key=lambda d: d[4]):
            col = _conf_colour(s)
            cv2.rectangle(strip_img, (int(x0), int(y0)),
                          (int(x0+w), int(y0+h)), col, 2)

        # Label bar at top of strip.
        hdr = np.zeros((32, img.shape[1], 3), dtype=np.uint8)
        label = (f"conf≥{thr:.3f}  {n} dets"
                 + ("  ← mAP eval (0.001)" if thr == 0.001 else
                    "  ← demo default (0.25)" if thr == 0.25 else ""))
        cv2.putText(hdr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        strips.append(np.vstack([hdr, strip_img]))

    print()
    lost = max_dets - len(yc.postprocess(out_decoded, pad, scale,
                                         conf_thres=0.25, iou_thres=0.7, max_det=300))
    print(f"  Boxes present at 0.001 but gone by 0.25: {lost} / {max_dets}")
    print(f"  These are the detections mAP integrates over that a demo window never shows.")
    print(f"  Colour key: blue = low confidence → red = high confidence")

    # ------------------------------------------------------------------ #
    # Save composite                                                      #
    # ------------------------------------------------------------------ #
    composite = np.vstack(strips)
    # Add footer
    footer = np.zeros((28, composite.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer,
                f"yolov8{v}  {args.ep.upper()}  {ms:.1f}ms/inference  {src.name}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1)
    composite = np.vstack([composite, footer])

    out_dir  = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"conf_sweep_yolov8{v}_{args.ep}.jpg"
    cv2.imwrite(str(out_path), composite)
    print(f"\n  Saved → {out_path}")
    print()


if __name__ == "__main__":
    main()
