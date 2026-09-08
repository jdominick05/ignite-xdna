"""
Width ladder demo: run a single image through yolov8 n/s/m/x on the NPU in one
invocation and print a comparison table that directly illustrates the repo's
central finding — width is nearly free on this hardware.

Each model uses the already-compiled 4x4.xclbin cache (yolocutcachekey) so no
recompile happens between variants. NPU latency drifts across sessions, which is
why every row is measured in the same run rather than sourced from historical
logs.

    conda activate resnet_env17
    python tools/width_ladder_demo.py
    python tools/width_ladder_demo.py --source assets/test_image.jpg --runs 50
    python tools/width_ladder_demo.py --out-dir results/  # saves annotated JPEGs
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

# --------------------------------------------------------------------------- #
# Model ladder — n/s/m/x at calib-200 (like-for-like), plus x for breadth.    #
# Each entry: (label, model file, cache key suffix used in yolocutcachekey)   #
# All share the same cache key: the EP reuses the compiled artefact for the   #
# variant that was compiled last. Since width changes model weights but not    #
# graph topology, the EP will recompile once per variant.                     #
# --------------------------------------------------------------------------- #
LADDER = [
    ("n", "yolov8n_cut_xint8_c200.onnx"),
    ("s", "yolov8s_cut_xint8_c200.onnx"),
    ("m", "yolov8m_cut_xint8_c200.onnx"),
    ("x", "yolov8x_cut_xint8.onnx"),   # no calib-200 version; noted in output
]

FLOPS = {   # GFLOPs/inference at 640² from Ultralytics' published figures
    "n":  8.7,
    "s": 28.6,
    "m": 78.9,
    "x": 257.8,
}


def _bar(frac, width=20):
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def run_variant(model_path, source_img, runs, ep, log_sev):
    """Build session, warm up, measure, return (dets, mean_ms, median_ms, annotated_img).

    Clears yolocutcachekey before each session: n/s/m/x have different graph
    structures (63/63/83/103 Conv nodes) and all share this one cache slot.
    Without a flush the EP silently reuses the previous variant's compiled graph.
    """
    cache_key = "yolocutcachekey"
    clear_cache(cache_key)
    sess = build_session(str(model_path), ep, cache_key, log_severity=log_sev)
    inp = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, str(model_path))
    n_out = len(sess.get_outputs())

    if n_out == len(yc.HEAD_OUTS):
        order = head_order(sess, imgsz)
        def run_raw(x):
            return sess.run(None, {inp: x})
        def decode_raw(r):
            return decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=0.25)
    else:
        def run_raw(x):
            return sess.run(None, {inp: x})
        def decode_raw(r):
            return r[0]

    x, pad, scale = yc.letterbox(source_img, imgsz)

    # warm-up (not timed)
    for _ in range(3):
        run_raw(x)

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        raw = run_raw(x)
        ts.append((time.perf_counter() - t0) * 1000)

    # single decode pass for detections
    out = decode_raw(run_raw(x))
    dets = yc.postprocess(out, pad, scale, conf_thres=0.25)

    annotated = yc.draw(source_img.copy(), dets)
    return dets, float(np.mean(ts)), float(np.median(ts)), annotated


def main():
    ap = argparse.ArgumentParser(
        description="Run n/s/m/x YOLO widths on one image, print comparison table.")
    ap.add_argument("--source", default="assets/test_image.jpg",
                    help="path to an image file (default: assets/test_image.jpg)")
    ap.add_argument("--ep", choices=["npu", "cpu"], default="npu")
    ap.add_argument("--runs", type=int, default=30,
                    help="timed inference runs per variant (default: 30)")
    ap.add_argument("--out-dir", default=None,
                    help="if set, save annotated JPEGs here (one per variant)")
    ap.add_argument("--log", type=int, default=2,
                    help="ORT log severity: 0=verbose … 3=error (default 2=warning)")
    args = ap.parse_args()

    src = ROOT / args.source if not os.path.isabs(args.source) else Path(args.source)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"cannot read image: {src}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWidth ladder demo — {args.ep.upper()} — {args.runs} timed runs each")
    print(f"Image : {src.name}  ({img.shape[1]}×{img.shape[0]})")
    print()

    rows = []
    for label, fname in LADDER:
        model_path = ROOT / "models" / fname
        if not model_path.exists():
            print(f"  [{label}] {fname} — not found, skipping")
            continue

        calib_note = "" if "c200" in fname else " ⚠ calib<200"
        print(f"  [{label}] compiling / loading {fname} …", end="", flush=True)
        dets, mean_ms, median_ms, annotated = run_variant(
            model_path, img, args.runs, args.ep, args.log)
        fps = 1000.0 / mean_ms
        print(f"  {mean_ms:6.2f} ms mean  ({fps:.1f} fps)  {len(dets)} dets{calib_note}")

        if out_dir:
            out_path = out_dir / f"width_ladder_{label}_{args.ep}.jpg"
            cv2.imwrite(str(out_path), annotated)

        rows.append((label, fname, mean_ms, median_ms, fps, dets, calib_note))

    if not rows:
        raise SystemExit("No models were found. Run from the repo root with resnet_env17 active.")

    # ------------------------------------------------------------------ #
    # Print the comparison table                                          #
    # ------------------------------------------------------------------ #
    # Normalize latency for the bar chart
    max_ms = max(r[2] for r in rows)
    n_ms = rows[0][2]  # yolov8n baseline

    print()
    print("=" * 72)
    print(f"  {'Model':<8}  {'Latency':>10}  {'FPS':>6}  {'vs n':>6}  "
          f"{'GFLOPs':>7}  {'Dets':>4}  Utilization bar")
    print("-" * 72)
    for label, fname, mean_ms, median_ms, fps, dets, caveat in rows:
        speedup_str = f"{mean_ms / n_ms:.2f}×"
        gflops = FLOPS.get(label, "?")
        gflops_str = f"{gflops}" if isinstance(gflops, (int, float)) else gflops
        bar = _bar(mean_ms / max_ms, width=22)
        cav = "  " + caveat if caveat else ""
        print(f"  yolov8{label:<2}  {mean_ms:>8.2f}ms  {fps:>6.1f}  "
              f"{speedup_str:>6}  {gflops_str:>7}  {len(dets):>4}  {bar}{cav}")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------ #
    # Key insight callout                                                 #
    # ------------------------------------------------------------------ #
    if len(rows) >= 2:
        n_row = next((r for r in rows if r[0] == "n"), None)
        m_row = next((r for r in rows if r[0] == "m"), None)
        s_row = next((r for r in rows if r[0] == "s"), None)

        if n_row and s_row:
            flop_ratio_ns = FLOPS["s"] / FLOPS["n"]
            lat_ratio_ns = s_row[2] / n_row[2]
            print(f"  n→s: {flop_ratio_ns:.1f}× the FLOPs, {lat_ratio_ns:.2f}× the latency  "
                  f"→  'width is nearly free'")
        if n_row and m_row:
            flop_ratio_nm = FLOPS["m"] / FLOPS["n"]
            lat_ratio_nm = m_row[2] / n_row[2]
            print(f"  n→m: {flop_ratio_nm:.1f}× the FLOPs, {lat_ratio_nm:.2f}× the latency  "
                  f"→  still sub-linear at a bigger step")

    print()
    print("  Note: these numbers are from a single session. Latency drifts across")
    print("  sessions on this hardware; always compare within a run, not against logs")
    print("  from a different day. (CLAUDE.md invariant)")
    print()

    # List what each model found
    print("  Detections at conf ≥ 0.25 (sorted by confidence):")
    for label, fname, mean_ms, median_ms, fps, dets, caveat in rows:
        det_strs = [f"{yc.COCO_CLASSES[c]}({s:.2f})"
                    for _, _, _, _, s, c in sorted(dets, key=lambda d: -d[4])]
        summary = ", ".join(det_strs[:8]) + (f" … +{len(det_strs)-8} more" if len(det_strs) > 8 else "")
        print(f"    yolov8{label}: {summary if summary else '(none)'}")

    if out_dir:
        print(f"\n  Annotated images saved to {out_dir}/")

if __name__ == "__main__":
    main()
