"""
How many independent camera streams before this NPU stops giving back more
throughput per stream added?

dual_stream_bench.py answered "does 2 beat 1" (yes, 1.8-1.9x, zero cross-talk).
This tool answers the two things RESEARCH.md left open on that finding:

  1. Does the multiplier keep climbing at 3, 4, 6, 8 concurrent streams, or
     does it saturate once the array's idle headroom (the gap between
     yolov8n's ~1.1/16 TOPS solo and the array's ceiling) is used up?
  2. Does a wider model -- one that already uses more of the array per call,
     leaving less idle headroom for other streams to fill -- get the same
     multiplier, a smaller one, or none at all?

One stream in every sweep is a known-negative image, checked every run for
cross-talk the same way dual_stream_bench.py does -- more concurrent sessions
sharing one NPU is exactly the situation where that check matters most, not
less.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/nstream_bench.py --model models/yolov8n_cut_xint8.onnx \
        --cache-key yolocutcachekey --streams 1 2 3 4 6 8 --seconds 8

Writes nothing to results/ itself -- pipe through scripts/lib.sh's run_logged
(or plain `tee`) if the run is worth keeping as a logged claim per
CONTRIBUTING.md.
"""
import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from npu.paths import ASSETS, DATA
from npu.session import clear_cache

# Import the sibling script by path, not `tools.dual_stream_bench` -- some
# conda envs (resnet_env17 included) have an unrelated "tools" package in
# site-packages that shadows this repo's tools/ directory as a regular
# package import.
_dsb_path = Path(__file__).resolve().parent / "dual_stream_bench.py"
_spec = importlib.util.spec_from_file_location("dual_stream_bench", _dsb_path)
_dsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dsb)
Stream, make_forward = _dsb.Stream, _dsb.make_forward


def run_n(n, model, cache_key, xclbin, log_severity, img_a, img_b, seconds):
    """Build n independent sessions (1 known-negative, n-1 known-positive),
    hammer all of them concurrently for `seconds`, return (combined_fps,
    mismatches, per_stream_fps_list)."""
    streams = []
    for i in range(n):
        infer = make_forward(model, cache_key, xclbin, log_severity)
        if i == 0:
            streams.append(Stream(f"cam{i} (neg)", img_b, infer, expect_positive=False))
        else:
            streams.append(Stream(f"cam{i} (pos)", img_a, infer, expect_positive=True))

    threads = [threading.Thread(target=s.run_for, args=(seconds,)) for s in streams]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total_calls = sum(s.calls for s in streams)
    mismatches = sum(s.mismatches for s in streams)
    per_stream_fps = [s.calls / wall for s in streams]
    return total_calls / wall, mismatches, per_stream_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-key", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--img-a", default=str(ASSETS / "test_image.jpg"))
    ap.add_argument("--img-b", default=str(DATA / "coco_calib" / "000000001532.jpg"))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    print(f"model: {args.model}")
    print(f"sweep: {args.streams} streams, {args.seconds:.0f}s each\n")

    rows = []
    baseline_fps = None
    for n in sorted(set(args.streams)):
        print(f"=== {n} concurrent stream(s) ===")
        fps, mism, per = run_n(n, args.model, args.cache_key, args.xclbin, args.log,
                                args.img_a, args.img_b, args.seconds)
        if baseline_fps is None:
            baseline_fps = fps
        speedup = fps / baseline_fps
        per_str = ", ".join(f"{v:.1f}" for v in per)
        flag = "  *** CROSS-TALK ***" if mism else ""
        print(f"  combined {fps:.1f} fps  ({speedup:.2f}x vs 1-stream)  "
              f"per-stream: [{per_str}]  mismatches: {mism}{flag}\n")
        rows.append((n, fps, speedup, mism))

    print("=== summary ===")
    print(f"{'streams':>7}  {'combined fps':>12}  {'speedup':>8}  {'mismatches':>10}")
    for n, fps, speedup, mism in rows:
        print(f"{n:>7}  {fps:>12.1f}  {speedup:>7.2f}x  {mism:>10}")

    total_mismatches = sum(r[3] for r in rows)
    if total_mismatches:
        print(f"\n*** {total_mismatches} total cross-talk mismatches -- "
              "do not trust this backend under N concurrent sessions without "
              "per-call verification ***")
    else:
        print("\nno cross-talk at any stream count.")


if __name__ == "__main__":
    main()
