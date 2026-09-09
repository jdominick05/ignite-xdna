"""Run one model on one input and write a usable result. A CLI, not a library.

    python -m tui.task matte --input photo.jpg --ep npu

The TUI spawns this exactly as it spawns a demo, and that is deliberate. An
in-process ORT session inside the UI would mean a DPU timeout kills the whole
launcher, a lingering session holds a hardware context on single-tenant silicon and
blocks the next run, and npu/session.py's own five lines of stdout land in the
middle of whatever is being rendered. Spawning gives one launcher code path, a
context that is always released on exit, and structured results through the sidecar
JSON rather than by parsing stdout -- which this repo has already been burned by.

Preprocessing and postprocessing come from npu/<family>. Nothing is reimplemented
here: MODNet once had its transform copied five times and two of them disagreed.
"""
import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from npu.paths import ROOT
from npu.session import build_session, clear_cache

from tui import guards, registry


def _out_dir(entry_key) -> Path:
    d = ROOT / "outputs" / "tasks" / entry_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_input(entry, given):
    """A file, a directory (first image in it), a camera index, or the entry's sample."""
    value = given or entry.sample
    if value in ("", None):
        raise SystemExit(f"{entry.key} needs --input")
    if str(value).isdigit():
        return int(value)
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        # Flat first, then recursive: data/sesr_val holds Set5_LR_x2/ and friends
        # rather than images, so a non-recursive glob finds nothing there.
        for globber in (p.glob, p.rglob):
            hits = sorted(h for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPEG")
                          for h in globber(pattern))
            if hits:
                return hits[0]
        raise SystemExit(f"no images under {p}")
    if not p.is_file():
        raise SystemExit(f"input not found: {p}")
    return p


# --------------------------------------------------------------------------
# Per-family adapters.
#
# These are not one loop with a switch. The npu/ modules look similar but do not
# share a contract: postprocess is postprocess_matte / postprocess_depth /
# postprocess_mask / postprocess depending on the family, yolo's takes (output,
# pad, scale, conf, iou), and npu/yolov6.py has no preprocess or postprocess at
# all -- it borrows yolo.letterbox and its own decode module. So each adapter is
# explicit glue, which is also what this repo's style asks for.
# --------------------------------------------------------------------------

def _run_matte(sess, img, args):
    from npu import modnet
    # Every npu/ preprocess returns (tensor, (orig_h, orig_w)) -- the source shape
    # comes back because the result is resized to it, never to the square the
    # network ran at.
    blob, _ = modnet.preprocess(img)
    t0 = time.perf_counter()
    raw = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    ms = (time.perf_counter() - t0) * 1000
    alpha = modnet.postprocess_matte(raw, img.shape[:2])
    a3 = cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    if args.background == "blur":
        bg = cv2.GaussianBlur(img, (0, 0), 12)
    elif args.background == "green":
        bg = np.zeros_like(img); bg[:, :] = (0, 177, 64)
    else:
        bg = np.zeros_like(img)
    comp = (img * a3 + bg * (1 - a3)).astype(np.uint8)
    return comp, ms, {"alpha_mean": float(alpha.mean())}


def _run_depth(sess, img, args, family):
    mod = __import__(f"npu.{family}", fromlist=["x"])
    blob, orig = mod.preprocess(img)
    t0 = time.perf_counter()
    raw = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    ms = (time.perf_counter() - t0) * 1000
    depth = mod.postprocess_depth(raw, orig)
    return mod.colorize_depth(depth), ms, {}


def _run_segment(sess, img, args):
    from npu import bisenetv2
    blob, orig = bisenetv2.preprocess(img)
    t0 = time.perf_counter()
    raw = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    ms = (time.perf_counter() - t0) * 1000
    mask = bisenetv2.postprocess_mask(raw, orig)
    return bisenetv2.overlay_segmentation(img, mask), ms, {}


def _decode_split(sess, img, args, dec, n_heads):
    """The shared half of detect and pose: letterbox, run, reorder, decode.

    head_order() returns INDICES into the session's outputs, not names -- the heads
    come back in whatever order the graph declares them, and decode_heads needs
    HEAD_OUTS order. pipelines/yolov8n/4_detect.py:91-99 is the reference.

    The timed span is sess.run alone. Decode is numpy on the host and its cost moves
    with --conf, not with the hardware: one measured "infer" went 8.94 -> 15.01 ms
    purely by lowering the threshold. Letting that leak in would make the number
    describe the threshold rather than the NPU.
    """
    from npu import yolo
    size = yolo.input_size(sess.get_inputs()[0].shape)
    lb, pad, scale = yolo.letterbox(img, size)
    inp = sess.get_inputs()[0].name
    n_out = len(sess.get_outputs())
    order = dec.head_order(sess, size) if n_out == n_heads else None

    t0 = time.perf_counter()
    raw = sess.run(None, {inp: lb})
    ms = (time.perf_counter() - t0) * 1000

    if order is not None:
        decoded = dec.decode_heads([raw[i] for i in order], imgsz=size,
                                   conf_thres=args.conf)
    elif n_out == 1:
        decoded = raw[0]          # full graph: the decode tail is inside the model
    else:
        raise SystemExit(f"unexpected output count {n_out}; expected {n_heads} "
                         f"(head-cut) or 1 (full graph)")
    return decoded, pad, scale, ms


def _run_detect(sess, img, args, family):
    from npu import yolo
    if family == "yolov6":
        from npu import yolov6 as head, yolov6_decode as dec
    else:
        from npu import yolo as head, yolo_decode as dec
    decoded, pad, scale, ms = _decode_split(sess, img, args, dec, len(head.HEAD_OUTS))
    dets = yolo.postprocess(decoded, pad, scale, conf_thres=args.conf,
                            iou_thres=args.iou)
    return yolo.draw(img.copy(), dets), ms, {"detections": len(dets)}


def _run_pose(sess, img, args):
    from npu import yolo_pose, yolo_pose_decode as dec
    decoded, pad, scale, ms = _decode_split(sess, img, args, dec,
                                            len(yolo_pose.HEAD_OUTS))
    dets = yolo_pose.postprocess(decoded, pad, scale, conf_thres=args.conf,
                                 iou_thres=args.iou)
    return yolo_pose.draw(img.copy(), dets), ms, {"people": len(dets)}


def _run_upscale(sess, img, args, family):
    """Tile a full-size photo through a fixed 64x64 or 256x256 network.

    split_into_tiles returns four values -- tiles, the original (h, w), the padded
    (h, w) and the (rows, cols) grid -- and merge_tiles needs all three of the
    latter to put the picture back together.
    """
    mod = __import__(f"npu.{family}", fromlist=["x"])
    size = mod.input_size(sess.get_inputs()[0].shape)
    tiles, orig_hw, padded_hw, grid_hw = mod.split_into_tiles(img, (size, size),
                                                             args.overlap)
    name = sess.get_inputs()[0].name
    out_tiles, total = [], 0.0
    for tile in tiles:
        blob, _ = mod.preprocess(tile, size)
        t0 = time.perf_counter()
        raw = sess.run(None, {name: blob})[0]
        total += (time.perf_counter() - t0) * 1000
        out_tiles.append(mod.postprocess(raw))
    sr = mod.merge_tiles(out_tiles, orig_hw, padded_hw, grid_hw,
                         scale=mod.SCALE, overlap=args.overlap)
    # Per-tile mean is the comparable number; the total is what the wall clock felt.
    return sr, total / max(len(tiles), 1), {"tiles": len(tiles),
                                            "total_ms": round(total, 2)}


DISPATCH = {
    "modnet": lambda s, i, a, f: _run_matte(s, i, a),
    "fastdepth": lambda s, i, a, f: _run_depth(s, i, a, "fastdepth"),
    "midas": lambda s, i, a, f: _run_depth(s, i, a, "midas"),
    "bisenetv2": lambda s, i, a, f: _run_segment(s, i, a),
    "yolov8": lambda s, i, a, f: _run_detect(s, i, a, "yolov8"),
    "yolov6": lambda s, i, a, f: _run_detect(s, i, a, "yolov6"),
    "yolo_pose": lambda s, i, a, f: _run_pose(s, i, a),
    "sesr": lambda s, i, a, f: _run_upscale(s, i, a, "sesr"),
    "realesrgan": lambda s, i, a, f: _run_upscale(s, i, a, "realesrgan"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m tui.task",
        description="Run one model on one input. Results go to outputs/tasks/, never results/.")
    ap.add_argument("entry", help="registry key, e.g. matte / detect / depth / segment")
    ap.add_argument("--input", default=None, help="image file, directory, or camera index")
    ap.add_argument("--model", default=None, help="models/-relative path; default is the entry's first")
    ap.add_argument("--ep", default=None, choices=["npu", "dml", "cpu"])
    ap.add_argument("--fresh", action="store_true", help="clear the compile cache first")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--background", choices=["blur", "green", "black"], default="blur")
    ap.add_argument("--sidecar", default=None, help="where to write the provenance JSON")
    args = ap.parse_args(argv)

    entry = registry.by_key(args.entry)
    if not entry.is_task:
        raise SystemExit(f"{entry.key} is a demo -- launch it with the TUI or run demos/ directly")

    choice = entry.models[0]
    if args.model:
        matches = [m for m in entry.models if m.relpath == args.model or m.label == args.model]
        if not matches:
            raise SystemExit(f"{args.model!r} is not one of "
                             + ", ".join(m.relpath for m in entry.models))
        choice = matches[0]
    if not choice.path.is_file():
        raise SystemExit(f"model not built on this machine: {choice.path}")

    ep = args.ep or entry.default_ep
    if ep not in entry.eps:
        raise SystemExit(f"{entry.key} supports {entry.eps}, not {ep!r}")
    cache_key = registry.cache_key_for(entry, choice)

    # The cache is keyed by name, so a different model in the same family silently
    # reuses the previous compile. context.json records what that compile was
    # actually built from, which makes this exact rather than a guess.
    stale = ep == "npu" and guards.is_stale(cache_key, choice.path)
    fresh = args.fresh or stale
    if stale and not args.fresh:
        print(f"[tui] {guards.cache_status(cache_key, choice.path).detail}")
    if fresh and ep == "npu":
        clear_cache(cache_key)

    src = _resolve_input(entry, args.input)
    img = cv2.imread(str(src)) if not isinstance(src, int) else None
    if img is None and not isinstance(src, int):
        raise SystemExit(f"could not read image: {src}")
    if isinstance(src, int):
        raise SystemExit("webcam input is not wired into the task lane yet; "
                         "use the portrait-matting demo for a live camera")

    sess = build_session(str(choice.path), ep, cache_key, log_severity=2)
    handler = DISPATCH.get(entry.family)
    if handler is None:
        raise SystemExit(f"no adapter for family {entry.family!r}")

    # One untimed call first. Without it the number below is compile tail, not
    # inference -- pipelines/resnet50/4_run.py does the same for the same reason.
    handler(sess, img, args, entry.family)
    result, ms, extra = handler(sess, img, args, entry.family)

    stem = Path(str(src)).stem
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = _out_dir(entry.key) / f"{ts}_{stem}_{ep}.png"
    cv2.imwrite(str(out_path), result)

    verdict = guards.ep_verdict(cache_key) if ep == "npu" else None
    sidecar = {
        "entry": entry.key, "family": entry.family,
        "model": f"models/{choice.relpath}",
        "cache_key": cache_key, "cache_was_fresh": bool(fresh),
        "ep_requested": ep,
        "ep_engaged": ("npu" if verdict and verdict.status == "ok" else
                       ep if ep != "npu" else "NOT npu"),
        "nodes_npu": verdict.npu if verdict else None,
        "nodes_total": verdict.total if verdict else None,
        "ep_report_sha256": verdict.sha256 if verdict else None,
        "ep_report_note": (str(verdict) if verdict else
                           f"{ep} has no EP report -- only VitisAI writes one"),
        "indicative_ms": round(ms, 3),
        "quotable": False,
        "quotable_note": "sess.run only, one image, one sitting. Not a measurement: "
                         "see docs/BENCHMARKS.md for numbers with method and caveats.",
        "input": str(src), "output": str(out_path),
        "machine": guards.check_machine().detail,
        "timestamp": ts,
        **extra,
    }
    side_path = Path(args.sidecar) if args.sidecar else out_path.with_suffix(".json")
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"[tui] wrote {out_path}")
    print(f"[tui] {sidecar['ep_report_note']}")
    print(f"[tui] {ms:.2f} ms (indicative, sess.run only -- not doc-quotable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
