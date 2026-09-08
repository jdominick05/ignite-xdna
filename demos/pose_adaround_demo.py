"""
YOLOv8n-Pose AdaRound Keypoint Stabilization Demo on AMD XDNA1 NPU.

Demonstrates the key finding recorded in docs/BENCHMARKS.md:
"YOLOv8n-pose: AdaRound recovers +1.68 OKS mAP on full 5000 images at 9.35 ms on NPU (1015/1025 nodes)"

Evaluates:
  1. Plain XINT8 (yolov8n-pose_cut_xint8.onnx) on Phoenix NPU
  2. XINT8 + AdaRound (yolov8n-pose_cut_xint8_adaround.onnx) on Phoenix NPU

Quantization noise in pose estimation manifests as keypoint coordinate jitter
(displacement of wrists, ankles, facial landmarks) and degraded joint visibility.
This demo matches detected persons across both models, measures per-joint Euclidean
drift in pixels, and renders a side-by-side comparison with skeleton overlays.

Usage:
    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/pose_adaround_demo.py
    python tools/pose_adaround_demo.py --source assets/test_image.jpg --runs 40
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

import npu.yolo_pose as yp
from npu.paths import YOLO_POSE_CUT_CACHE_KEY
from npu.session import build_session, clear_cache
from npu.yolo import input_size
from npu.yolo_pose_decode import decode_heads, head_order

COL_SKELETON_XINT8 = (0, 140, 255)    # Orange for XINT8
COL_SKELETON_ADAROUND = (0, 220, 0)   # Green for AdaRound
COL_JOINT_HOT = (0, 0, 255)          # Red for high confidence joints
COL_DISPLACEMENT = (255, 0, 255)     # Magenta for displacement vectors


def _run_pose(model_path, source_img, runs, ep, conf, log_sev):
    """Run one pose model on NPU and return (dets, mean_ms, median_ms)."""
    cache_key = YOLO_POSE_CUT_CACHE_KEY
    clear_cache(cache_key)

    sess = build_session(str(model_path), ep, cache_key, log_severity=log_sev)
    inp = sess.get_inputs()[0].name
    imgsz = input_size(sess.get_inputs()[0].shape, str(model_path))
    order = head_order(sess, imgsz)

    x, pad, scale = yp.letterbox(source_img, imgsz)

    def run_raw():
        return sess.run(None, {inp: x})

    # Warm-up (3 passes)
    for _ in range(3):
        run_raw()

    # Timed inference (sess.run strictly alone per CLAUDE.md invariant)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        raw = run_raw()
        ts.append((time.perf_counter() - t0) * 1000.0)

    # Postprocess
    out = decode_heads([raw[i] for i in order], imgsz=imgsz, conf_thres=conf)
    dets = yp.postprocess(out, pad, scale, conf_thres=conf, iou_thres=0.5)

    return dets, float(np.mean(ts)), float(np.median(ts))


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def draw_custom_pose(img, dets, skeleton_col, label_text, kpt_conf_thres=0.5):
    canvas = img.copy()
    for x, y, w, h, s, kpts in dets:
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(canvas, p1, p2, skeleton_col, 2)
        cv2.putText(canvas, f"{label_text}: {s:.2f}", (p1[0], max(p1[1] - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, skeleton_col, 2, cv2.LINE_AA)

        visible = kpts[:, 2] >= kpt_conf_thres
        # Draw skeleton bones
        for a, b in yp.SKELETON:
            if visible[a] and visible[b]:
                pa = (int(kpts[a, 0]), int(kpts[a, 1]))
                pb = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(canvas, pa, pb, skeleton_col, 2, cv2.LINE_AA)

        # Draw joints
        for i, (px, py, pv) in enumerate(kpts):
            if visible[i]:
                cv2.circle(canvas, (int(px), int(py)), 4, COL_JOINT_HOT, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(px), int(py)), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(
        description="YOLOv8n-pose AdaRound keypoint stabilization and jitter analysis on NPU."
    )
    ap.add_argument("--source", default="assets/test_image.jpg")
    ap.add_argument("--ep", choices=["npu", "cpu"], default="npu")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kpt-conf", type=float, default=0.4)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    src = ROOT / args.source if not os.path.isabs(args.source) else Path(args.source)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"Cannot read image: {src}")

    m_xint8 = ROOT / "models" / "yolov8n-pose_cut_xint8.onnx"
    m_adaround = ROOT / "models" / "yolov8n-pose_cut_xint8_adaround.onnx"

    for p in (m_xint8, m_adaround):
        if not p.exists():
            raise SystemExit(f"Model not found: {p}")

    print(f"\n{'='*75}")
    print(f"  YOLOv8n-Pose Keypoint Jitter & AdaRound Recovery Demo on {args.ep.upper()}")
    print(f"  Image: {src.name} ({img.shape[1]}x{img.shape[0]}) | Runs: {args.runs} | Conf: {args.conf}")
    print(f"{'='*75}\n")

    print(f"  [1/2] Plain XINT8 ({m_xint8.name})...", end="", flush=True)
    dets_x, mean_x, med_x = _run_pose(m_xint8, img, args.runs, args.ep, args.conf, args.log)
    print(f"  {mean_x:6.2f} ms (median {med_x:6.2f} ms, {1000/mean_x:.1f} fps) | {len(dets_x)} persons")

    print(f"  [2/2] XINT8 + AdaRound ({m_adaround.name})...", end="", flush=True)
    dets_a, mean_a, med_a = _run_pose(m_adaround, img, args.runs, args.ep, args.conf, args.log)
    print(f"  {mean_a:6.2f} ms (median {med_a:6.2f} ms, {1000/mean_a:.1f} fps) | {len(dets_a)} persons")

    # Match persons between XINT8 and AdaRound
    matched_pairs = []
    used_a = set()
    for dx in dets_x:
        best_iou, best_j = 0.0, -1
        for j, da in enumerate(dets_a):
            if j in used_a:
                continue
            iou = _iou(dx[:4], da[:4])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.4:
            matched_pairs.append((dx, dets_a[best_j]))
            used_a.add(best_j)

    print("\n" + "=" * 75)
    print(f"  {'Metric':<28} {'Plain XINT8':>18} {'XINT8 + AdaRound':>22}")
    print("-" * 75)
    print(f"  {'NPU Latency (mean)':<28} {mean_x:>16.2f} ms {mean_a:>20.2f} ms")
    print(f"  {'NPU Throughput':<28} {1000/mean_x:>16.1f} fps {1000/mean_a:>20.1f} fps")
    print(f"  {'Published COCO OKS mAP':<28} {'32.64 mAP':>18} {'34.32 mAP (+1.68)':>22}")
    print(f"  {'Persons Detected (conf>=0.25)':<28} {len(dets_x):>18} {len(dets_a):>22}")
    print(f"  {'Matched Persons':<28} {len(matched_pairs):>18} {len(matched_pairs):>22}")
    print("=" * 75)

    # Keypoint displacement analysis
    if matched_pairs:
        print("\n  KEYPOINT JITTER & DISPLACEMENT ANALYSIS (17 COCO Joints):")
        print(f"  {'Joint':<16} {'XINT8 Vis':>10} {'AdaRound Vis':>13} {'Drift (px)':>12}  Shift Bar")
        print("  " + "-" * 62)

        # Aggregate across all matched persons
        joint_drifts = []
        joint_vis_x = []
        joint_vis_a = []

        for k in range(yp.NUM_KPTS):
            drifts_k = []
            vis_xk = []
            vis_ak = []
            for dx, da in matched_pairs:
                kx = dx[5][k]  # x, y, vis
                ka = da[5][k]
                drift = np.sqrt((ka[0] - kx[0]) ** 2 + (ka[1] - kx[1]) ** 2)
                drifts_k.append(drift)
                vis_xk.append(kx[2])
                vis_ak.append(ka[2])

            mean_drift = float(np.mean(drifts_k))
            mean_vx = float(np.mean(vis_xk))
            mean_va = float(np.mean(vis_ak))

            joint_drifts.append(mean_drift)
            joint_vis_x.append(mean_vx)
            joint_vis_a.append(mean_va)

        max_drift = max(joint_drifts) if joint_drifts and max(joint_drifts) > 0 else 1.0
        for k in range(yp.NUM_KPTS):
            bar_len = int(round((joint_drifts[k] / max_drift) * 14))
            bar = "█" * bar_len + "░" * (14 - bar_len)
            print(f"  {yp.COCO_KEYPOINTS[k]:<16} {joint_vis_x[k]:>9.2f} {joint_vis_a[k]:>12.2f} {joint_drifts[k]:>10.2f}px  {bar}")

        mean_all_drift = float(np.mean(joint_drifts))
        print(f"\n  Average Keypoint Coordinate Drift: {mean_all_drift:.2f} pixels")
        print("  Takeaway: Plain XINT8 introduces coordinate regression jitter on non-convex limbs;")
        print("  AdaRound's weight rounding optimization recovers +1.68 OKS mAP without latency penalty.")

    # ------------------------------------------------------------------ #
    # Render Side-by-Side Visual Comparison                              #
    # ------------------------------------------------------------------ #
    left_img = draw_custom_pose(img, dets_x, COL_SKELETON_XINT8, "XINT8", args.kpt_conf)
    right_img = draw_custom_pose(img, dets_a, COL_SKELETON_ADAROUND, "AdaRound", args.kpt_conf)

    # Optional: Draw displacement quiver arrows on right image
    for dx, da in matched_pairs:
        for k in range(yp.NUM_KPTS):
            kx = dx[5][k]
            ka = da[5][k]
            if kx[2] >= args.kpt_conf or ka[2] >= args.kpt_conf:
                p_from = (int(round(kx[0])), int(round(kx[1])))
                p_to = (int(round(ka[0])), int(round(ka[1])))
                dist = np.sqrt((p_to[0] - p_from[0]) ** 2 + (p_to[1] - p_from[1]) ** 2)
                if 2.0 <= dist <= 50.0:
                    cv2.arrowedLine(right_img, p_from, p_to, COL_DISPLACEMENT, 2, tipLength=0.3)

    # Add header bars
    hdr_h = 42
    w = img.shape[1]
    hdr_left = np.zeros((hdr_h, w, 3), dtype=np.uint8)
    hdr_right = np.zeros((hdr_h, w, 3), dtype=np.uint8)

    cv2.putText(hdr_left, f"Plain XINT8 ({mean_x:.1f} ms, {len(dets_x)} persons, 32.64 OKS mAP)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(hdr_right, f"XINT8 + AdaRound ({mean_a:.1f} ms, {len(dets_a)} persons, 34.32 OKS mAP)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

    panel_left = np.vstack([hdr_left, left_img])
    panel_right = np.vstack([hdr_right, right_img])
    side_by_side = np.hstack([panel_left, panel_right])

    # Footer banner
    foot_h = 36
    footer = np.zeros((foot_h, side_by_side.shape[1], 3), dtype=np.uint8)
    foot_text = "YOLOv8n-Pose Keypoint Regression: Orange=XINT8, Green=AdaRound, Magenta Arrows=Keypoint Shift Vector"
    cv2.putText(footer, foot_text, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    composite = np.vstack([side_by_side, footer])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pose_adaround_diff_npu.jpg"
    cv2.imwrite(str(out_file), composite)
    print(f"\n  Saved skeleton comparison composite -> {out_file}\n")


if __name__ == "__main__":
    main()
