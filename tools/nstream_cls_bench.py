"""
Does the concurrent-stream saturation shape found for detection
(tools/nstream_bench.py: yolov8n saturates at 2.1x by 3 streams, yolov8m at
1.29x by 2) generalize to a classification model, and does accuracy itself
survive N-way contention, not just "found something or not"?

tools/dual_stream_bench.py and nstream_bench.py only ever checked a binary
found/not-found ground truth. That is enough to catch gross cross-talk (a
result landing on the wrong stream) but not enough to catch a subtler
failure: outputs that are still "roughly right" but numerically corrupted by
contention (wrong top-1 on borderline images, a few points of accuracy lost
under load). This tool checks the real thing -- each stream classifies its
OWN disjoint slice of the labeled eval set and top-1 accuracy is compared
solo vs. under N-way concurrency, not just presence/absence on one image.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/nstream_cls_bench.py --model models/resnet50_xint8_c64.onnx \
        --streams 1 2 3 4 6 8 12 16 --per-stream 100

Writes nothing to results/ itself -- pipe through scripts/lib.sh's run_logged
(or plain `tee`) if the run is worth keeping as a logged claim per
CONTRIBUTING.md.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import npu.preprocess as q
from npu.paths import DATA, MODELS
from npu.session import build_session, clear_cache

DEFAULT_IMAGES = DATA / "eval"
DEFAULT_CFG = MODELS / "preprocess_config.json"


class ClsStream:
    """One simulated camera classifying labeled images on its own session,
    tallying top-1 accuracy, latency, and per-image argmax predictions (so a
    caller can diff those predictions against a solo reference -- an exact
    per-image correctness check, not a top-1 delta across different samples,
    which sample noise alone can produce even with zero contention effect)."""

    def __init__(self, name, session, input_name, transform, files, labels):
        self.name = name
        self.session = session
        self.input_name = input_name
        self.transform = transform
        self.files = files
        self.labels = labels
        self.calls = 0
        self.correct = 0
        self.times = []
        self.preds = {}  # basename -> predicted class

    def step(self, path):
        x = self.transform(path)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0][0]
        self.times.append(time.perf_counter() - t0)
        self.calls += 1
        pred = int(np.argmax(out))
        self.preds[os.path.basename(path)] = pred
        y = self.labels.get(os.path.basename(path))
        if y is not None and pred == y:
            self.correct += 1

    def run_all(self):
        for p in self.files:
            self.step(p)

    def top1(self):
        return 100.0 * self.correct / self.calls if self.calls else 0.0

    def fps(self):
        arr = np.array(self.times)
        return self.calls / arr.sum() if arr.sum() else 0.0


def build_stream(i, model, cache_key, xclbin, log_severity, transform, files, labels):
    session = build_session(model, "npu", cache_key, xclbin, log_severity=log_severity)
    input_name = session.get_inputs()[0].name
    return ClsStream(f"cam{i}", session, input_name, transform, files, labels)


def run_n(n, model, cache_key, xclbin, log_severity, transform, shared_files, labels, ref_preds):
    """Build n independent sessions and hand EVERY one of them the SAME
    `shared_files` slice, run all n concurrently, return (combined_fps,
    mean_top1, per_stream_top1_list, worst_mismatch_rate).

    Every stream sees identical input on purpose: it makes top-1 directly
    comparable to the solo reference (no per-slice sample-size noise to
    explain away), and it lets us diff each stream's per-image predictions
    against `ref_preds` for an EXACT correctness check -- the classification
    equivalent of the found/not-found cross-talk check in
    dual_stream_bench.py / nstream_bench.py, but catching a subtler failure:
    an output that is still "plausible" but wrong under contention, not just
    a result landing on the wrong stream entirely."""
    streams = []
    for i in range(n):
        streams.append(build_stream(i, model, cache_key, xclbin, log_severity,
                                     transform, shared_files, labels))

    threads = [threading.Thread(target=s.run_all) for s in streams]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total_calls = sum(s.calls for s in streams)
    per_top1 = [s.top1() for s in streams]
    mean_top1 = sum(s.correct for s in streams) / total_calls * 100.0
    mismatch_rates = []
    for s in streams:
        n_diff = sum(1 for k, v in s.preds.items() if ref_preds.get(k) != v)
        mismatch_rates.append(100.0 * n_diff / len(s.preds) if s.preds else 0.0)
    return total_calls / wall, mean_top1, per_top1, max(mismatch_rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-key", default="modelcachekey")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--cfg-path", default=str(DEFAULT_CFG))
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--per-stream", type=int, default=60,
                     help="labeled images each stream classifies per run")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    with open(args.cfg_path) as f:
        cfg = json.load(f)
    transform = q.build_transform(cfg)

    files = []
    for ext in q.IMG_EXTS:
        files.extend(glob.glob(os.path.join(args.images, "**", ext), recursive=True))
    files = sorted(files)
    labels = json.load(open(os.path.join(args.images, "labels.json")))
    if not files or not labels:
        raise SystemExit(f"no labeled images under {args.images}")

    print(f"model: {args.model}")
    print(f"pool: {len(files)} labeled images, {args.per_stream} per stream")
    print(f"sweep: {args.streams} streams\n")

    shared_files = files[:args.per_stream]

    # Solo reference: one session, one uncontended pass over the SAME slice
    # every concurrent stream below will also see. Same slice both ways --
    # it makes top-1 directly comparable to the concurrent runs (no
    # per-slice sample-size noise to explain away), and gives per-image
    # predictions (ref.preds) to diff every concurrent stream's predictions
    # against for an exact correctness check.
    print("=== solo reference (1 session, no contention) ===")
    ref = build_stream(0, args.model, args.cache_key, args.xclbin, args.log,
                        transform, shared_files, labels)
    ref.run_all()
    print(f"  top-1 {ref.top1():.2f}%  ({ref.calls} images)  {ref.fps():.1f} fps\n")

    rows = []
    baseline_fps = None
    for n in sorted(set(args.streams)):
        print(f"=== {n} concurrent stream(s), all on the same {len(shared_files)} images ===")
        fps, mean_top1, per_top1, worst_mismatch = run_n(
            n, args.model, args.cache_key, args.xclbin, args.log,
            transform, shared_files, labels, ref.preds)
        if baseline_fps is None:
            baseline_fps = fps
        speedup = fps / baseline_fps
        top1_str = ", ".join(f"{v:.1f}" for v in per_top1)
        flag = "  *** PREDICTION MISMATCH VS SOLO ***" if worst_mismatch > 0 else ""
        print(f"  combined {fps:.1f} fps  ({speedup:.2f}x vs 1-stream)  "
              f"mean top-1 {mean_top1:.2f}%  per-stream top-1: [{top1_str}]  "
              f"worst mismatch-vs-solo {worst_mismatch:.2f}%{flag}\n")
        rows.append((n, fps, speedup, mean_top1, worst_mismatch))

    print("=== summary ===")
    print(f"{'streams':>7}  {'combined fps':>12}  {'speedup':>8}  {'mean top-1':>10}  {'mismatch':>8}")
    for n, fps, speedup, mean_top1, worst_mismatch in rows:
        print(f"{n:>7}  {fps:>12.1f}  {speedup:>7.2f}x  {mean_top1:>9.2f}%  {worst_mismatch:>7.2f}%")

    total_mismatch = sum(r[4] for r in rows)
    if total_mismatch > 0:
        print(f"\n*** predictions changed vs the solo reference under contention on at "
              "least one stream -- do not trust concurrent classification sessions on "
              "this backend without per-call verification ***")
    else:
        print(f"\nexact match: every concurrent stream at every N produced the identical "
              f"per-image predictions as the solo reference ({ref.top1():.2f}% top-1). "
              "Accuracy is unaffected by contention on this backend.")


if __name__ == "__main__":
    main()
