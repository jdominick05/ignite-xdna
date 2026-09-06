"""
Does one NPU serve two independent camera streams, and does it keep their
results straight while doing it?

Two questions, since the batch>1 finding (RESEARCH.md: batch 2 on resnet50
drops the EP's partition to 80/395 nodes and silently returns garbage for
slot 1) means "does it work at all" and "is it fast" are separate questions
here, not one:

  1. THROUGHPUT: build two independent onnxruntime InferenceSessions against
     the SAME compiled cache (as two camera-handling threads plausibly would),
     and measure whether round-robin / concurrent-thread inference gets any
     more throughput than one stream alone, or whether the single NPU array
     just serializes both queues.
  2. CORRECTNESS: feed stream A an image with a known-positive detection and
     stream B a known-negative one, hammer both concurrently, and count any
     iteration where a result crosses streams. This is the same class of bug
     the batch-2 finding was -- shared hardware state leaking between
     logically independent requests -- so it is checked directly, not assumed
     away because two sessions are "obviously" independent.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/dual_stream_bench.py --model models/yolov8n-pose_cut_xint8.onnx \
        --cache-key yoloposecutcachekey --seconds 10

Writes nothing to results/ itself -- pipe through scripts/lib.sh's run_logged
(or plain `tee`) if the run is worth keeping as a logged claim per
CONTRIBUTING.md.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from npu.paths import ASSETS, DATA
from npu.session import build_session, clear_cache


def make_forward(model, cache_key, xclbin, log_severity):
    """Build one session and a closure that runs the model + reports whether
    the head-cut decode path or the full-graph path applies, using whichever
    of npu.yolo / npu.yolo_pose matches the output count."""
    sess = build_session(model, "npu", cache_key, xclbin, log_severity=log_severity)
    inp = sess.get_inputs()[0].name
    n_out = len(sess.get_outputs())

    import npu.yolo as yc
    import npu.yolo_pose as yp
    from npu.yolo_decode import decode_heads as decode_det, head_order as order_det
    from npu.yolo_pose_decode import decode_heads as decode_pose, head_order as order_pose

    if n_out == len(yp.HEAD_OUTS):
        imgsz = yc.input_size(sess.get_inputs()[0].shape, model)
        order = order_pose(sess, imgsz)
        letterbox, postprocess, kind = yp.letterbox, yp.postprocess, "pose"

        def infer(img):
            x, pad, scale = letterbox(img, imgsz)
            r = sess.run(None, {inp: x})
            out = decode_pose([r[i] for i in order], imgsz=imgsz, conf_thres=0.25)
            dets = postprocess(out, pad, scale, conf_thres=0.25, iou_thres=0.5)
            return len(dets) > 0
    elif n_out == len(yc.HEAD_OUTS):
        imgsz = yc.input_size(sess.get_inputs()[0].shape, model)
        order = order_det(sess, imgsz)
        letterbox, postprocess, kind = yc.letterbox, yc.postprocess, "detect"

        def infer(img):
            x, pad, scale = letterbox(img, imgsz)
            r = sess.run(None, {inp: x})
            out = decode_det([r[i] for i in order], imgsz=imgsz, conf_thres=0.25)
            dets = postprocess(out, pad, scale, conf_thres=0.25, iou_thres=0.5)
            # "person" specifically, not "any of 80 classes" -- an 80-class
            # detector finds SOMETHING in almost any real photo at conf 0.25,
            # so len(dets) > 0 is not a usable ground truth here the way it is
            # for the single-class pose head.
            return any(c == 0 for *_, c in dets)
    else:
        raise SystemExit(f"{model}: {n_out} outputs, don't know how to decode")

    print(f"  session ready: {kind}, {imgsz}x{imgsz}, {n_out} outputs")
    return infer


class Stream:
    """One simulated camera: a fixed image, an inference closure, an expected
    detect/no-detect ground truth, and running counters."""

    def __init__(self, name, img_path, infer, expect_positive):
        self.name = name
        self.img = cv2.imread(str(img_path))
        if self.img is None:
            raise SystemExit(f"could not read {img_path}")
        self.infer = infer
        self.expect_positive = expect_positive
        self.calls = 0
        self.mismatches = 0
        self.times = []

    def step(self):
        t0 = time.perf_counter()
        found = self.infer(self.img)
        self.times.append(time.perf_counter() - t0)
        self.calls += 1
        if found != self.expect_positive:
            self.mismatches += 1

    def run_for(self, seconds, stop_event=None):
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline and (stop_event is None or not stop_event.is_set()):
            self.step()

    def report(self, label):
        arr = np.array(self.times) * 1000
        fps = self.calls / arr.sum() * 1000 if arr.sum() else 0.0
        want = "positive" if self.expect_positive else "negative"
        print(f"  [{label}] {self.name}: {self.calls} calls, "
              f"mean {arr.mean():.2f} ms, {fps:.1f} fps, "
              f"expected {want}, {self.mismatches} mismatches"
              f"{'  *** CROSS-TALK ***' if self.mismatches else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-key", required=True)
    ap.add_argument("--img-a", default=str(ASSETS / "test_image.jpg"),
                    help="known-positive image (a person should be detected)")
    ap.add_argument("--img-b", default=str(DATA / "coco_calib" / "000000001532.jpg"),
                    help="known-negative image (no person)")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="duration of each of the two timed phases")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    print("building session A...")
    infer_a = make_forward(args.model, args.cache_key, args.xclbin, args.log)
    print("building session B (same cache -- should be fast, no recompile)...")
    infer_b = make_forward(args.model, args.cache_key, args.xclbin, args.log)

    a = Stream("camera A", args.img_a, infer_a, expect_positive=True)
    b = Stream("camera B", args.img_b, infer_b, expect_positive=False)

    # Sanity: each stream alone, single-threaded, before anything shares the NPU.
    print(f"\n=== solo baseline, {args.seconds:.0f}s each ===")
    a.run_for(args.seconds)
    a.report("solo")
    b.run_for(args.seconds)
    b.report("solo")

    # Phase 1: round-robin from one thread -- alternate A/B calls. Tests
    # whether interleaving two independent sessions costs anything beyond
    # each call's own latency.
    a2 = Stream("camera A", args.img_a, infer_a, expect_positive=True)
    b2 = Stream("camera B", args.img_b, infer_b, expect_positive=False)
    print(f"\n=== round-robin, single thread, {args.seconds:.0f}s ===")
    deadline = time.perf_counter() + args.seconds
    while time.perf_counter() < deadline:
        a2.step()
        b2.step()
    a2.report("round-robin")
    b2.report("round-robin")
    combined_fps = (a2.calls + b2.calls) / (sum(a2.times) + sum(b2.times))
    print(f"  combined: {combined_fps:.1f} fps across both streams")

    # Phase 2: true concurrency -- two threads, each hammering its own
    # session as fast as it can, at the same time. If the NPU array
    # serializes underneath, aggregate throughput here should land close to
    # phase 1's combined number, not double it.
    a3 = Stream("camera A", args.img_a, infer_a, expect_positive=True)
    b3 = Stream("camera B", args.img_b, infer_b, expect_positive=False)
    print(f"\n=== concurrent threads, {args.seconds:.0f}s ===")
    ta = threading.Thread(target=a3.run_for, args=(args.seconds,))
    tb = threading.Thread(target=b3.run_for, args=(args.seconds,))
    t0 = time.perf_counter()
    ta.start(); tb.start()
    ta.join(); tb.join()
    wall = time.perf_counter() - t0
    a3.report("concurrent")
    b3.report("concurrent")
    print(f"  combined: {(a3.calls + b3.calls) / wall:.1f} fps across both streams "
          f"(wall clock, {wall:.1f}s)")

    print("\n=== verdict ===")
    total_mismatches = a2.mismatches + b2.mismatches + a3.mismatches + b3.mismatches
    if total_mismatches:
        print(f"  *** {total_mismatches} cross-talk mismatches -- do not trust concurrent "
              "sessions on this backend without per-call verification ***")
    else:
        print("  no cross-talk: each stream's detections stayed correct throughout.")
    speedup = ((a3.calls + b3.calls) / wall) / combined_fps
    print(f"  concurrent-thread throughput vs round-robin: {speedup:.2f}x "
          f"({'genuine parallelism' if speedup > 1.3 else 'serialized, as expected for one NPU array'})")


if __name__ == "__main__":
    main()
