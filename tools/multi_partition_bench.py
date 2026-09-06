"""
Does splitting the array into independent per-column partitions beat the
single monolithic partition every other concurrency tool in this repo has
been measuring?

Unplanned discovery while re-checking RESEARCH.md's "compute headroom"
explanation for the stream-count throughput ceiling: `xrt-smi examine -r
aie-partitions`'s full report (previously only its GOPS/memory columns were
parsed -- see session_hold.py) shows that under the `4x4.xclbin` overlay,
every session in this repo has ever built -- any number of threads, any
number of separate OS processes -- lands on ONE `Partition Index: 0,
Columns: [1, 2, 3, 4]`. Two independent processes on that partition were
measured to exactly halve each other's throughput (~149/s solo -> ~76/s
each concurrent, combined ~152/s): one shared physical resource being
time-sliced, not "compute headroom" running out in the abstract.

`1x4.xclbin` (bundled with the SDK's Phoenix xclbins, sitting unused next to
4x4.xclbin -- not a rejected approach, just unexplored territory; see
docs/DECISIONS.md #1, which is about driver-dropped xclbins under the 1.8
EP, a different thing) claims only ONE column per context. Two separate OS
processes against it were observed getting two *separate* Partition Index
entries on two different columns -- genuine spatial parallelism the 4x4
overlay structurally cannot offer, since it always claims the whole array
as one partition regardless of context count.

This tool answers what that observation leaves open:
  1. Does combined throughput across N independent 1x4 partitions actually
     scale with N, all the way to N=4 (one context per column on this
     4-column Phoenix chip), or does a shared resource (DMA/memory bus)
     cap it short of 4x the single-partition rate? Measured with all N
     processes held in a common, confirmed-active window (every process
     started, xrt-smi polled until N contexts report Active, THEN the
     clock starts) -- not reconstructed from staggered runs with
     uncontrolled overlap, which is what the exploratory version of this
     check did.
  2. Does this new allocation mechanism stay correct under concurrency?
     Same class of check dual_stream_bench.py/nstream_bench.py already
     apply to thread-based concurrency, now applied to process-level,
     separate-partition concurrency: one column gets a known-negative
     image, the rest get a known-positive one, and any crossed result is
     reported, not assumed away.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/multi_partition_bench.py --model models/yolov8n_cut_xint8.onnx \\
        --cache-key yolocut1x4cachekey \\
        --xclbin "C:\\Program Files\\RyzenAI\\1.7.1\\voe-4.0-win_amd64\\xclbins\\phoenix\\1x4.xclbin" \\
        --procs 1 2 3 4 --seconds 12

Writes nothing to results/ itself -- pipe through scripts/lib.sh's run_logged
(or plain `tee`) if the run is worth keeping as a logged claim per
CONTRIBUTING.md. Name the machine (this only means anything as a Phoenix /
Desktop 2 or Hawk Point / laptop number, never generic).
"""
import argparse
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from npu.paths import ASSETS, DATA
from npu.session import build_session, clear_cache

XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"


def raw_partitions():
    try:
        return subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                               capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return str(e)


def active_count(raw):
    return raw.count("|Active")


def partitions_summary(raw):
    """[(partition_index, [columns]), ...] -- for the diagnostic record, not
    the throughput measurement."""
    out = []
    for idx, cols in re.findall(r"Partition Index\s*:\s*(\d+)\s*\n\s*Columns:\s*\[([^\]]*)\]", raw):
        out.append((idx, [c.strip() for c in cols.split(",") if c.strip()]))
    return out


def worker(rank, model, cache_key, xclbin, img_path, expect_positive,
           call_count, mismatch_count, ready_flag, stop_event):
    """One independent process: build its own session (own OS process ->
    own HW context, per the discovery above), then hammer it until told to
    stop. Imports done inside the function -- this runs in a fresh
    interpreter under multiprocessing's spawn start method (Windows
    default), which does not inherit the parent's already-imported modules."""
    import cv2
    import npu.yolo as yc
    from npu.yolo_decode import decode_heads, head_order

    sess = build_session(model, "npu", cache_key, xclbin, log_severity=3)
    inp = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, model)
    order = head_order(sess, imgsz)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"rank {rank}: could not read {img_path}")
    x, pad, scale = yc.letterbox(img, imgsz)

    ready_flag.value = 1
    while not stop_event.is_set():
        r = sess.run(None, {inp: x})
        out = decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=0.25)
        dets = yc.postprocess(out, pad, scale, conf_thres=0.25, iou_thres=0.5)
        found = any(c == 0 for *_, c in dets)
        with call_count.get_lock():
            call_count.value += 1
        if found != expect_positive:
            with mismatch_count.get_lock():
                mismatch_count.value += 1


def run_n(n, model, cache_key, xclbin, img_a, img_b, seconds, ready_timeout):
    """Spawn n worker processes, wait until xrt-smi confirms all n HW
    contexts are Active, measure combined throughput over a clean common
    window, then stop. Returns (combined_fps, per_worker_fps, mismatches,
    partitions_seen_during_window)."""
    ctx = mp.get_context("spawn")
    call_counts = [ctx.Value("i", 0) for _ in range(n)]
    mismatch_counts = [ctx.Value("i", 0) for _ in range(n)]
    ready_flags = [ctx.Value("i", 0) for _ in range(n)]
    stop_event = ctx.Event()

    procs = []
    for i in range(n):
        is_neg = (i == 0)
        img = img_b if is_neg else img_a
        p = ctx.Process(target=worker, args=(
            i, model, cache_key, xclbin, img, not is_neg,
            call_counts[i], mismatch_counts[i], ready_flags[i], stop_event))
        p.start()
        procs.append(p)

    # Wait for every worker's own "session built" flag AND for xrt-smi to
    # report n Active contexts -- both, because a process can be built and
    # still be the (n+1)th one queued behind a column limit, which is
    # exactly the case this tool needs to be able to detect rather than
    # silently measure a smaller n than requested.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ready_timeout:
        if all(f.value for f in ready_flags) and active_count(raw_partitions()) >= n:
            break
        time.sleep(0.5)
    else:
        for p in procs:
            p.terminate()
        raise SystemExit(f"n={n}: only reached "
                          f"{sum(f.value for f in ready_flags)}/{n} ready workers / "
                          f"{active_count(raw_partitions())} active HW contexts after "
                          f"{ready_timeout:.0f}s -- a column limit or a slow compile. "
                          "Re-run with --fresh if this followed a cache-key change.")

    time.sleep(1.0)  # settle past the first post-ready sample noise
    window_raw = raw_partitions()  # diagnostic snapshot taken inside the measured window
    start = [c.value for c in call_counts]
    t_start = time.perf_counter()
    time.sleep(seconds)
    end = [c.value for c in call_counts]
    t_end = time.perf_counter()

    stop_event.set()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    wall = t_end - t_start
    per_worker_fps = [(e - s) / wall for s, e in zip(start, end)]
    combined_fps = sum(per_worker_fps)
    mismatches = sum(m.value for m in mismatch_counts)
    return combined_fps, per_worker_fps, mismatches, partitions_summary(window_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-key", required=True)
    ap.add_argument("--xclbin", required=True,
                     help="e.g. the SDK's phoenix/1x4.xclbin -- required, not "
                          "defaulted, since running this against 4x4.xclbin would "
                          "just reproduce the single-partition sharing this tool "
                          "exists to get past")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--img-a", default=str(ASSETS / "test_image.jpg"))
    ap.add_argument("--img-b", default=str(DATA / "coco_calib" / "000000001532.jpg"))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--ready-timeout", type=float, default=60.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    print(f"model: {args.model}\nxclbin: {args.xclbin}")
    print("pre-building one session to force the compile before any worker forks "
          "-- avoids N processes racing the same cache directory on first run.")
    build_session(args.model, "npu", args.cache_key, args.xclbin, log_severity=2)
    print("compile cache warm.\n")

    rows = []
    for n in sorted(set(args.procs)):
        print(f"=== {n} independent process(es), separate partitions if the "
              f"overlay allows it ===")
        combined, per, mism, parts = run_n(n, args.model, args.cache_key, args.xclbin,
                                            args.img_a, args.img_b, args.seconds,
                                            args.ready_timeout)
        per_str = ", ".join(f"{v:.1f}" for v in per)
        print(f"  partitions during window: {parts}")
        print(f"  combined {combined:.1f} fps  per-process: [{per_str}]  "
              f"mismatches: {mism}" + ("  *** CROSS-TALK ***" if mism else ""))
        rows.append((n, combined, mism, parts))
        print()

    print("=== summary ===")
    baseline = rows[0][1] if rows else 1.0
    print(f"{'procs':>6}  {'combined fps':>12}  {'vs 1-proc':>9}  "
          f"{'partitions':>12}  {'mismatches':>10}")
    for n, combined, mism, parts in rows:
        print(f"{n:>6}  {combined:>12.1f}  {combined/baseline:>8.2f}x  "
              f"{len(parts):>12}  {mism:>10}")

    total_mismatches = sum(r[2] for r in rows)
    if total_mismatches:
        print(f"\n*** {total_mismatches} total cross-talk mismatches -- do not trust "
              "multi-partition concurrency on this backend without per-call "
              "verification ***")
    else:
        print("\nno cross-talk at any process count.")


if __name__ == "__main__":
    main()
