#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rescore saved COCO detections on the val2017 images that calibration never saw.

Every quantized COCO model here was calibrated from data/coco_calib, which
pipelines/yolov8n/2_fetch_coco.py:65-70 fills with every 16th val2017 file, the first 300.
Those 300 are also among the 5,000 images every mAP is scored on. This scores each saved
detection file twice, with the pipelines' own call (COCOeval over sorted image ids,
pipelines/yolov8n/5_eval_map.py:245-247):
  - on all 5,000, which must reproduce the figure its backing log printed, at 2 decimals;
  - on H, the 4,700 val2017 images whose file is not in data/coco_calib.
A row whose 5,000 does not reproduce is marked MISMATCH and left out of the summary: that
detection file is not the one the claim was scored from.

    python tools/coco_heldout_rescore.py --root <checkout holding data/ and results/dets_*.json>

CPU only; no model runs. The detection files are git-ignored, so each is identified by its
byte size and sha256. The yolov8n and yolov8s plain-XINT8 files are left out on purpose: their
held-out scores are inputs to a pending pre-registered int4 verdict, which produces them.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# model, arm, iouType, detections (under --root), backing log (in this repo)
ROWS = [
    ("yolov6n", "fp32", "bbox", "results/dets_yolov6n_cpu.json", "results/map_yolov6n_fp32_cpu.log"),
    ("yolov6n", "xint8", "bbox", "results/dets_yolov6n_cut_xint8_npu.json", "results/map_yolov6n_cut_xint8_npu.log"),
    ("yolov6n", "adaround", "bbox", "results/dets_yolov6n_cut_xint8_adaround_npu.json",
     "results/map_yolov6n_cut_xint8_adaround_npu.log"),
    ("yolov8n-pose", "fp32", "keypoints", "results/dets_kpts_yolov8n-pose_cut_cpu.json",
     "results/map_kpts_yolov8n-pose_cut_full5000_cpu.log"),
    ("yolov8n-pose", "xint8", "keypoints", "results/dets_kpts_yolov8n-pose_cut_xint8_npu.json",
     "results/map_kpts_yolov8n-pose_cut_xint8_full5000_npu.log"),
    ("yolov8n-pose", "adaround", "keypoints", "results/dets_kpts_yolov8n-pose_cut_xint8_adaround_npu.json",
     "results/map_kpts_yolov8n-pose_cut_xint8_adaround_npu.log"),
    ("yolov8m", "fp32", "bbox", "results/dets_yolov8m_cpu.json", "results/bench/map_yolov8m_cpu.log"),
    ("yolov8m", "xint8_c64", "bbox", "results/dets_yolov8m_cut_xint8_npu.json", "results/wide/map_yolov8m_npu.log"),
    ("yolov8m", "xint8_c200", "bbox", "results/dets_yolov8m_cut_xint8_c200_npu.json",
     "results/bench/map_yolov8m_cut_xint8_c200_npu.log"),
    ("yolov8m", "adaround", "bbox", "results/dets_yolov8m_cut_xint8_adaround_npu.json",
     "results/map_yolov8m_cut_xint8_adaround_npu.log"),
    ("yolov8s", "fp32", "bbox", "results/dets_yolov8s_cpu.json", "results/bench/map_yolov8s_cpu.log"),
    ("yolov8s", "adaround", "bbox", "results/dets_yolov8s_cut_xint8_adaround_npu.json",
     "results/map_yolov8s_cut_xint8_adaround_npu.log"),
    ("yolov8n", "fp32", "bbox", "results/dets_yolov8n_cpu.json", "results/bench/map_yolov8n_cpu.log"),
    ("yolov8n", "adaround", "bbox", "results/dets_yolov8n_cut_xint8_adaround_npu.json",
     "results/bench/map_yolov8n_cut_xint8_adaround_npu.log"),
]
# (model, plain arm, AdaRound arm): the recovery each published claim states
RECOVERY = [("yolov6n", "xint8", "adaround"), ("yolov8n-pose", "xint8", "adaround"),
            ("yolov8m", "xint8_c64", "adaround")]
ANN = {"bbox": "data/coco/annotations/instances_val2017.json",
       "keypoints": "data/coco/annotations/person_keypoints_val2017.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def published(log):
    text = (REPO / log).read_text(encoding="utf-8", errors="replace")
    a = re.findall(r"^(?:OKS )?mAP@50-95\s+([0-9.]+)", text, re.M)
    b = re.findall(r"^(?:OKS )?mAP@50\s+([0-9.]+)", text, re.M)
    if not a or not b:
        raise SystemExit(f"{log}: no mAP@50-95 / mAP@50 line")
    return float(a[0]), float(b[0]), len(a)


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def score(coco_eval_cls, gt, dt, iou, ids):
    ev = coco_eval_cls(gt, dt, iou)
    ev.params.imgIds = ids
    quiet(ev.evaluate)
    quiet(ev.accumulate)
    quiet(ev.summarize)
    return ev.stats[0] * 100, ev.stats[1] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="checkout that holds the git-ignored data/ and results/dets_*.json")
    args = ap.parse_args()
    root = Path(args.root)
    if os.name == "nt":  # stay out of the way of anything timed on this machine
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS

    from importlib.metadata import version
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    tool = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    print("$ python tools/coco_heldout_rescore.py --root <root>   (<root> = the checkout holding data/ and dets)")
    print(f"tool LF sha256 {tool}")
    print(f"date {time.strftime('%Y-%m-%d %H:%M')}  env {Path(sys.prefix).name}  python {platform.python_version()}  "
          f"numpy {version('numpy')}  pycocotools {version('pycocotools')}  process priority below normal")
    calib = sorted(int(p.stem) for p in (root / "data" / "coco_calib").glob("*.jpg"))
    print(f"data/coco_calib: {len(calib)} jpg files, ids {calib[0]}..{calib[-1]}")

    gts, held = {}, {}
    for iou, ann in ANN.items():
        gt = quiet(COCO, str(root / ann))
        ids = sorted(gt.getImgIds())
        outside = sorted(set(calib) - set(ids))
        h = sorted(set(ids) - set(calib))
        print(f"{ann}: sha256 {sha256(root / ann)}  {len(ids)} images; calibration ids not in it: "
              f"{len(outside)}; H = {len(h)} images")
        gts[iou], held[iou] = (gt, ids), h

    print("\nper file: published (backing log) | rescored on all 5,000 | on H; mAP@50-95 / mAP@50")
    res = {}
    for model, arm, iou, dets, log in ROWS:
        t0 = time.time()
        p95, p50, nlines = published(log)
        path = root / dets
        gt, ids = gts[iou]
        dt = quiet(gt.loadRes, str(path))
        a95, a50 = score(COCOeval, gt, dt, iou, ids)
        h95, h50 = score(COCOeval, gt, dt, iou, held[iou])
        ok = f"{a95:.2f}" == f"{p95:.2f}" and f"{a50:.2f}" == f"{p50:.2f}"
        print(f"{model} {arm} ({iou})\n"
              f"  dets {dets}  {path.stat().st_size} B  sha256 {sha256(path)}  {len(dt.getAnnIds())} detections\n"
              f"  log  {log} ({nlines} mAP@50-95 line{'s' if nlines > 1 else ''}; the first is used)\n"
              f"  published {p95:.2f} / {p50:.2f} | all {a95:.2f} / {a50:.2f} "
              f"{'MATCH' if ok else 'MISMATCH'} | H {h95:.2f} / {h50:.2f}  ({time.time() - t0:.0f} s)")
        if ok:
            res[model, arm] = (a95, h95)

    print("\nsummary, mAP@50-95 points, reproduced rows only")
    print("  all-H: the arm's score on all 5,000 minus on H. excess: the arm's all-H minus FP32's all-H,")
    print("  i.e. how much more the arm gains from the 300 calibration images than the float model does.")
    for model, arm, *_ in ROWS:
        if (model, arm) not in res:
            continue
        a, h = res[model, arm]
        line = f"  {model:13s} {arm:11s} all {a:6.2f}  H {h:6.2f}  all-H {a - h:+.2f}"
        if arm != "fp32" and (model, "fp32") in res:
            fa, fh = res[model, "fp32"]
            line += f"  excess {(a - h) - (fa - fh):+.2f}  drop from FP32: all {fa - a:.2f}, H {fh - h:.2f}"
        print(line)
    print("\nrecovery the claims state (AdaRound minus plain XINT8), on all 5,000 and on H")
    for model, plain, ada in RECOVERY:
        if (model, plain) in res and (model, ada) in res:
            ra = res[model, ada][0] - res[model, plain][0]
            rh = res[model, ada][1] - res[model, plain][1]
            print(f"  {model:13s} {plain} -> {ada}: all {ra:+.2f}  H {rh:+.2f}  difference {ra - rh:+.2f}")
        else:
            print(f"  {model:13s} {plain} -> {ada}: not computed (a row did not reproduce)")
    bad = [f"{m} {a}" for m, a, *_ in ROWS if (m, a) not in res]
    print(f"\nrows reproduced: {len(res)} of {len(ROWS)}" + (f"; MISMATCH: {', '.join(bad)}" if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
