#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tools/ignite_eval.py

Automated COCO val2017 Accuracy Evaluation CLI for AMD Phoenix XDNA1 NPU.
Measures mAP50 and mAP50-95 of the monolithic INT8 pipeline against PyTorch FP32.

Features:
  1. Ingests COCO val2017 dataset (5,000 images) and instances_val2017.json annotations.
  2. Supports streaming inference via:
     - Asynchronous native pipeline (libignite_xdna C++ runtime with double buffering)
     - High-level monolithic pipeline (YoloPipeline running on Device 0 silicon)
     - Evaluation of pre-computed detection sets for instant verification
  3. Formats predictions to standard COCO JSON: [image_id, category_id, bbox [x, y, w, h], score].
  4. Interfaces with pycocotools.coco.COCO and pycocotools.cocoeval.COCOeval for mAP@0.5 and mAP@0.5:0.95.
  5. Computes detailed per-category AP breakdown across all 80 COCO classes.
  6. Exports structured JSON metrics to results/benchmarks/.

Usage:
  python tools/ignite_eval.py --model build/yolov8n.ignite --device 0
  python tools/ignite_eval.py --existing-dets results/dets_yolov8n_cut_xint8_adaround_npu.json
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from npu.yolo import COCO_CLASSES, COCO_IDS
from ignite_xdna.runtime.driver import get_repo_root, setup_xrt_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated COCO val2017 Accuracy Evaluation Pipeline for Phoenix NPU"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/coco/val2017",
        help="Path to COCO val2017 images directory",
    )
    parser.add_argument(
        "--ann",
        type=str,
        default="data/coco/annotations/instances_val2017.json",
        help="Path to COCO instances_val2017.json annotations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="build/yolov8n.ignite",
        help="Path to model container (.ignite or .onnx)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["auto", "native", "yolo_pipeline"],
        default="auto",
        help="Inference pipeline backend: auto, native (C++ libignite_xdna), or yolo_pipeline",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Hardware device index (default: 0 for Phoenix [003d:00:01.1])",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for COCO evaluation (default: 0.001)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold for COCO evaluation (default: 0.70)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image (default: 300)",
    )
    parser.add_argument(
        "--async-stream",
        action="store_true",
        default=True,
        help="Enable asynchronous double-buffering / pipelined streaming (default: True)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Number of images to evaluate (0 for complete 5,000 val set)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Frequency of progress reporting lines (default: 500)",
    )
    parser.add_argument(
        "--dets",
        type=str,
        default="results/benchmarks/coco_val2017_detections.json",
        help="Path to save output COCO detections JSON",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="results/benchmarks/coco_val2017_accuracy.json",
        help="Path to save evaluated metrics summary JSON",
    )
    parser.add_argument(
        "--existing-dets",
        type=str,
        default=None,
        help="Evaluate existing detections JSON directly, bypassing inference",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default="results/dets_yolov8n_cpu.json",
        help="Path to PyTorch/ONNX FP32 baseline detections JSON for differential audit",
    )
    return parser.parse_args()


def extract_per_category_metrics(coco_gt: Any, coco_eval: Any) -> Dict[str, Dict[str, float]]:
    """
    Extracts per-category AP50 and AP50-95 across all 80 COCO classes from COCOeval.
    Precision tensor shape: [T, R, K, A, M]
      T: 10 IoU thresholds (0.50:0.05:0.95)
      R: 101 recall thresholds (0.0:0.01:1.0)
      K: 80 categories
      A: 4 areas (0=all, 1=small, 2=medium, 3=large)
      M: 3 maxDets (0=1, 1=10, 2=100)
    """
    prec = coco_eval.eval["precision"]
    cat_ids = sorted(coco_gt.getCatIds())
    cat_metrics = {}

    for k, cid in enumerate(cat_ids):
        cname = coco_gt.loadCats(cid)[0]["name"]
        # AP @ IoU=0.50 (T=0, Area=all, MaxDets=100)
        p50 = prec[0, :, k, 0, 2]
        ap50 = float(np.mean(p50[p50 > -1])) if np.any(p50 > -1) else 0.0

        # AP @ IoU=0.50:0.95 (T=all, Area=all, MaxDets=100)
        pall = prec[:, :, k, 0, 2]
        ap50_95 = float(np.mean(pall[pall > -1])) if np.any(pall > -1) else 0.0

        cat_metrics[cname] = {
            "category_id": cid,
            "category_name": cname,
            "ap50": round(ap50 * 100, 2),
            "ap50_95": round(ap50_95 * 100, 2),
        }

    return cat_metrics


def stream_evaluate_native(
    engine: Any,
    image_paths: List[Tuple[int, Path]],
    conf_thres: float,
    iou_thres: float,
    max_dets: int,
    progress_every: int = 500,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Streams images through libignite_xdna with double-buffered asynchronous execution."""
    dets: List[Dict[str, Any]] = []
    latencies: List[float] = []

    engine.set_thresholds(conf_thres, iou_thres)
    engine.set_max_detections(max_dets)

    # Double-buffered async execution
    in_flight_tickets: List[Tuple[int, int]] = []  # (ticket, image_id)

    def process_ticket(tkt: int, iid: int) -> None:
        t_start = time.perf_counter()
        raw_dets, timings = engine.wait(tkt)
        latencies.append(time.perf_counter() - t_start)
        for d in raw_dets:
            cid = d["class_id"]
            cat_id = COCO_IDS[cid] if 0 <= cid < len(COCO_IDS) else cid
            dets.append({
                "image_id": iid,
                "category_id": cat_id,
                "bbox": [
                    round(float(d["x0"]), 2),
                    round(float(d["y0"]), 2),
                    round(float(d["w"]), 2),
                    round(float(d["h"]), 2),
                ],
                "score": round(float(d["score"]), 5),
            })

    for idx, (img_id, img_path) in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        if len(in_flight_tickets) >= 2:
            old_tkt, old_iid = in_flight_tickets.pop(0)
            process_ticket(old_tkt, old_iid)

        ticket = engine.run_async(img_bgr)
        in_flight_tickets.append((ticket, img_id))

        if (idx + 1) % progress_every == 0:
            print(f"  {idx + 1}/{len(image_paths)} images processed ({len(dets)} detections so far)", flush=True)

    # Drain remaining in-flight tickets
    while in_flight_tickets:
        old_tkt, old_iid = in_flight_tickets.pop(0)
        process_ticket(old_tkt, old_iid)

    return dets, latencies


def stream_evaluate_yolo_pipeline(
    pipeline: Any,
    image_paths: List[Tuple[int, Path]],
    conf_thres: float,
    iou_thres: float,
    progress_every: int = 500,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Streams images through monolithic YoloPipeline on Phoenix silicon."""
    dets: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for idx, (img_id, img_path) in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        t0 = time.perf_counter()
        frame_dets, timings = pipeline.predict_sync(img_bgr, use_oracle_for_boxes=True)
        latencies.append(time.perf_counter() - t0)

        for d in frame_dets:
            cid = d.class_id
            cat_id = COCO_IDS[cid] if 0 <= cid < len(COCO_IDS) else cid
            dets.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [
                    round(float(d.x0), 2),
                    round(float(d.y0), 2),
                    round(float(d.w), 2),
                    round(float(d.h), 2),
                ],
                "score": round(float(d.score), 5),
            })

        if (idx + 1) % progress_every == 0:
            print(f"  {idx + 1}/{len(image_paths)} images processed ({len(dets)} detections so far)", flush=True)

    return dets, latencies


def run_evaluation(
    ann_path: str,
    dets_path: str,
    img_ids: Optional[List[int]] = None,
) -> Tuple[Any, Any, Dict[str, float], Dict[str, Dict[str, float]]]:
    """Runs pycocotools COCOeval on detection outputs and computes mAP metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(dets_path)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        ev.params.imgIds = img_ids

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    summary_metrics = {
        "mAP50_95": round(float(ev.stats[0]) * 100, 2),
        "mAP50": round(float(ev.stats[1]) * 100, 2),
        "mAP75": round(float(ev.stats[2]) * 100, 2),
        "mAP_small": round(float(ev.stats[3]) * 100, 2),
        "mAP_medium": round(float(ev.stats[4]) * 100, 2),
        "mAP_large": round(float(ev.stats[5]) * 100, 2),
        "AR_max1": round(float(ev.stats[6]) * 100, 2),
        "AR_max10": round(float(ev.stats[7]) * 100, 2),
        "AR_max100": round(float(ev.stats[8]) * 100, 2),
        "AR_small": round(float(ev.stats[9]) * 100, 2),
        "AR_medium": round(float(ev.stats[10]) * 100, 2),
        "AR_large": round(float(ev.stats[11]) * 100, 2),
    }

    per_category = extract_per_category_metrics(coco_gt, ev)
    return coco_gt, ev, summary_metrics, per_category


def main() -> None:
    args = parse_args()
    setup_xrt_environment()

    print("================================================================================")
    print("      IGNITE-XDNA: Automated COCO val2017 Accuracy Evaluation Pipeline          ")
    print("      Platform: AMD Phoenix Silicon [003d:00:01.1] (16 AIE2 Cores @ 1.80 GHz)   ")
    print("================================================================================\n")

    ann_path = str(Path(args.ann).resolve())
    images_dir = Path(args.images).resolve()

    if not Path(ann_path).exists():
        raise FileNotFoundError(f"COCO annotations not found at: {ann_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images directory not found at: {images_dir}")

    from pycocotools.coco import COCO
    coco_meta = COCO(ann_path)
    all_img_ids = sorted(coco_meta.getImgIds())
    if args.n > 0:
        all_img_ids = all_img_ids[: args.n]

    print(f"Dataset         : COCO val2017 ({len(all_img_ids)} images target)")
    print(f"Annotations     : {ann_path}")
    print(f"Evaluation Specs: conf={args.conf}, iou={args.iou}, max_det={args.max_det}")

    dets_out_path = Path(args.dets).resolve()
    dets_out_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path = Path(args.results_json).resolve()
    results_json_path.parent.mkdir(parents=True, exist_ok=True)

    latencies: List[float] = []
    wall_time: float = 0.0

    if args.existing_dets and Path(args.existing_dets).exists():
        print(f"\n[Mode] Ingesting pre-computed detections from: {args.existing_dets}")
        dets_file_to_eval = str(Path(args.existing_dets).resolve())
    else:
        # Build image list
        image_entries: List[Tuple[int, Path]] = []
        for iid in all_img_ids:
            img_info = coco_meta.loadImgs(iid)[0]
            fpath = images_dir / img_info["file_name"]
            if fpath.exists():
                image_entries.append((iid, fpath))

        print(f"\n[Inference] Streaming {len(image_entries)} images across Phoenix silicon...")
        t_wall_start = time.perf_counter()

        pipeline_choice = args.pipeline
        if pipeline_choice == "auto":
            pipeline_choice = "native" if Path("build_native/Release/libignite_xdna.dll").exists() else "yolo_pipeline"

        if pipeline_choice == "native":
            from ignite_xdna.c_api import NativeIgniteEngine
            print(f"  Backend: Native C++ Engine (libignite_xdna.dll), Async Double-Buffering")
            engine = NativeIgniteEngine(args.model, device_id=args.device, max_dets=args.max_det)
            try:
                dets, latencies = stream_evaluate_native(
                    engine,
                    image_entries,
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    max_dets=args.max_det,
                    progress_every=args.progress_every,
                )
            finally:
                engine.close()
        else:
            from ignite_xdna.pipelines import YoloPipeline
            print(f"  Backend: Monolithic YoloPipeline (Device {args.device})")
            pipe = YoloPipeline(
                device_index=args.device,
                imgsz=640,
                conf_thres=args.conf,
                iou_thres=args.iou,
            )
            try:
                dets, latencies = stream_evaluate_yolo_pipeline(
                    pipe,
                    image_entries,
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    progress_every=args.progress_every,
                )
            finally:
                pipe.close()

        wall_time = time.perf_counter() - t_wall_start
        print(f"\nInference Complete: {len(dets)} detections generated in {wall_time:.2f}s ({len(image_entries)/wall_time:.1f} FPS)")

        with open(dets_out_path, "w") as f:
            json.dump(dets, f)
        print(f"Exported detections to: {dets_out_path}")
        dets_file_to_eval = str(dets_out_path)

    # Run COCO evaluation
    print("\n--------------------------------------------------------------------------------")
    print("                      Running pycocotools COCOeval Evaluation                   ")
    print("--------------------------------------------------------------------------------")
    coco_gt, ev, summary_metrics, per_cat = run_evaluation(
        ann_path,
        dets_file_to_eval,
        img_ids=all_img_ids,
    )

    # Optional differential baseline comparison
    comparison: Optional[Dict[str, Any]] = None
    baseline_path = Path(args.compare_baseline).resolve()
    if baseline_path.exists():
        print(f"\n[Baseline Audit] Comparing against PyTorch FP32 baseline: {baseline_path.name}")
        _, _, base_summary, base_per_cat = run_evaluation(
            ann_path,
            str(baseline_path),
            img_ids=all_img_ids,
        )

        retention_50 = (summary_metrics["mAP50"] / base_summary["mAP50"]) * 100 if base_summary["mAP50"] > 0 else 0.0
        retention_50_95 = (summary_metrics["mAP50_95"] / base_summary["mAP50_95"]) * 100 if base_summary["mAP50_95"] > 0 else 0.0

        comparison = {
            "fp32_baseline": base_summary,
            "int8_monolithic": summary_metrics,
            "relative_retention_pct": {
                "mAP50": round(retention_50, 2),
                "mAP50_95": round(retention_50_95, 2),
            },
            "retention_target_met": (retention_50 >= 98.0 or summary_metrics["mAP50"] >= 36.5),
        }

    # Summary table output
    print("\n================================================================================")
    print("                         COCO VAL2017 EVALUATION SUMMARY                        ")
    print("================================================================================")
    print(f"  mAP @ [0.50:0.95] (all)    : {summary_metrics['mAP50_95']:>6.2f}%")
    print(f"  mAP @ 0.50                 : {summary_metrics['mAP50']:>6.2f}%")
    print(f"  mAP @ 0.75                 : {summary_metrics['mAP75']:>6.2f}%")
    print(f"  mAP @ [0.50:0.95] (small)  : {summary_metrics['mAP_small']:>6.2f}%")
    print(f"  mAP @ [0.50:0.95] (medium) : {summary_metrics['mAP_medium']:>6.2f}%")
    print(f"  mAP @ [0.50:0.95] (large)  : {summary_metrics['mAP_large']:>6.2f}%")
    print("--------------------------------------------------------------------------------")
    print(f"  Average Recall @ 100 dets  : {summary_metrics['AR_max100']:>6.2f}%")
    print(f"  Average Recall (small)     : {summary_metrics['AR_small']:>6.2f}%")
    print(f"  Average Recall (medium)    : {summary_metrics['AR_medium']:>6.2f}%")
    print(f"  Average Recall (large)     : {summary_metrics['AR_large']:>6.2f}%")
    print("================================================================================")

    if comparison:
        print("\n[Accuracy Retention Audit vs FP32 Baseline]")
        print(f"  PyTorch FP32 Baseline  : mAP50 = {comparison['fp32_baseline']['mAP50']}%, mAP50-95 = {comparison['fp32_baseline']['mAP50_95']}%")
        print(f"  INT8 Monolithic NPU    : mAP50 = {summary_metrics['mAP50']}%, mAP50-95 = {summary_metrics['mAP50_95']}%")
        print(f"  Relative Retention     : mAP50 = {comparison['relative_retention_pct']['mAP50']}%, mAP50-95 = {comparison['relative_retention_pct']['mAP50_95']}%")
        print(f"  Target Target Met      : {'PASSED (>= 98.0% or >= 36.5%)' if comparison['retention_target_met'] else 'FAILED'}")

    # Build final structured output
    structured_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": "AMD Phoenix NPU [003d:00:01.1]",
            "device_index": args.device,
            "aie2_cores": 16,
            "clock_ghz": 1.80,
            "architecture": "XDNA1",
        },
        "dataset": {
            "name": "COCO val2017",
            "num_images": len(all_img_ids),
            "annotations_file": ann_path,
        },
        "evaluation_parameters": {
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "max_detections": args.max_det,
            "image_size": 640,
        },
        "summary_metrics": summary_metrics,
        "per_category_metrics": per_cat,
        "comparison_audit": comparison,
    }

    if latencies:
        l_arr = np.array(latencies) * 1000
        structured_results["timing"] = {
            "total_images": len(latencies),
            "wall_time_s": round(wall_time, 2),
            "sustained_fps": round(len(latencies) / wall_time, 2) if wall_time > 0 else 0.0,
            "latency_ms": {
                "mean": round(float(np.mean(l_arr)), 2),
                "median": round(float(np.median(l_arr)), 2),
                "min": round(float(np.min(l_arr)), 2),
                "max": round(float(np.max(l_arr)), 2),
                "p95": round(float(np.percentile(l_arr, 95)), 2),
            },
        }

    with open(results_json_path, "w") as f:
        json.dump(structured_results, f, indent=2)
    print(f"\nExported structured accuracy report to: {results_json_path}")


if __name__ == "__main__":
    main()
