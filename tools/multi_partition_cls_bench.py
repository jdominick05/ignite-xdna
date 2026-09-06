"""Classification counterpart to multi_partition_bench.py: does splitting the
array into independent per-column 1x4.xclbin partitions scale for a
classification model the same way it does for YOLO, and does a wider
classifier push achieved ops/s (tools/estimate_tops.py) past whatever the
current best is?

Same methodology as multi_partition_bench.py -- N independent OS processes
(spawn, so each gets its own HW context/partition), held in a common
confirmed-active window before the clock starts, not reconstructed from
staggered runs. The correctness check is adapted for classification instead
of detection, and label-free the same way probe_batch_slot_write.py is: two
random-noise images verified (not assumed) to get DIFFERENT top-1
predictions from a solo session before any workers start, alternating across
ranks -- a process silently receiving another partition's output would flip
which of the two classes it reports. No labeled eval set needed.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/multi_partition_cls_bench.py --model models/wide_resnet101_2_xint8_c64.onnx \\
        --cache-key yolocut1x4cachekey \\
        --xclbin "C:\\Program Files\\RyzenAI\\1.7.1\\voe-4.0-win_amd64\\xclbins\\phoenix\\1x4.xclbin" \\
        --cfg-path models/preprocess_config_wide101.json \\
        --procs 1 2 3 4 --seconds 12 --fresh

Needs no labeled eval set -- the cross-talk check uses random noise (see below).

Writes nothing to results/ itself -- pipe through scripts/lib.sh's run_logged
(or plain `tee`) if the run is worth keeping as a logged claim per
CONTRIBUTING.md. Name the machine (Desktop 2 / Phoenix or laptop / Hawk
Point) in whatever log this produces.
"""
import argparse
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

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
    out = []
    for idx, cols in re.findall(r"Partition Index\s*:\s*(\d+)\s*\n\s*Columns:\s*\[([^\]]*)\]", raw):
        out.append((idx, [c.strip() for c in cols.split(",") if c.strip()]))
    return out


def _shape_from_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    return tuple(cfg["input_size"])  # (C, H, W)


def find_two_distinct_seeds(model, cache_key, xclbin, chw, max_tries=20):
    """Solo-session check (not assumed): try random-noise seeds until two
    give DIFFERENT top-1 predictions, so the correctness check below has a
    real known-a/known-b pair rather than a guessed one. Same rationale as
    probe_batch_slot_write.py: no labeled data needed, and two independent
    draws of continuous random floats landing on the same class by luck
    rather than a real difference is not a real concern here since we only
    need *some* seed pair that differs, not a representative sample."""
    import numpy as np

    sess = build_session(model, "npu", cache_key, xclbin, log_severity=2)
    inp = sess.get_inputs()[0].name
    first_seed, first_pred = None, None
    for seed in range(max_tries):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((1, *chw), dtype=np.float32)
        pred = int(np.argmax(sess.run(None, {inp: x})[0][0]))
        if first_seed is None:
            first_seed, first_pred = seed, pred
        elif pred != first_pred:
            return (first_seed, first_pred), (seed, pred)
    raise SystemExit(f"every one of {max_tries} random seeds predicted the same "
                      "class -- unusual, try raising --max-seed-tries")


def worker(rank, model, cache_key, xclbin, chw, seed, expect_pred,
           call_count, mismatch_count, ready_flag, stop_event):
    """One independent process -- own OS process -> own HW context. Imports
    done inside the function: this runs in a fresh interpreter under
    multiprocessing's spawn start method (Windows default)."""
    import numpy as np

    sess = build_session(model, "npu", cache_key, xclbin, log_severity=3)
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, *chw), dtype=np.float32)

    ready_flag.value = 1
    while not stop_event.is_set():
        pred = int(np.argmax(sess.run(None, {inp: x})[0][0]))
        with call_count.get_lock():
            call_count.value += 1
        if pred != expect_pred:
            with mismatch_count.get_lock():
                mismatch_count.value += 1


def run_n(n, model, cache_key, xclbin, chw, seed_a, seed_b, pred_a, pred_b,
          seconds, ready_timeout):
    ctx = mp.get_context("spawn")
    call_counts = [ctx.Value("i", 0) for _ in range(n)]
    mismatch_counts = [ctx.Value("i", 0) for _ in range(n)]
    ready_flags = [ctx.Value("i", 0) for _ in range(n)]
    stop_event = ctx.Event()

    procs = []
    for i in range(n):
        use_a = (i % 2 == 0)
        seed, expect = (seed_a, pred_a) if use_a else (seed_b, pred_b)
        p = ctx.Process(target=worker, args=(
            i, model, cache_key, xclbin, chw, seed, expect,
            call_counts[i], mismatch_counts[i], ready_flags[i], stop_event))
        p.start()
        procs.append(p)

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

    time.sleep(1.0)
    window_raw = raw_partitions()
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
                     help="e.g. the SDK's phoenix/1x4.xclbin -- required, not defaulted, "
                          "since 4x4.xclbin would just reproduce the single shared partition")
    ap.add_argument("--cfg-path", required=True,
                     help="preprocess config matching --model, e.g. "
                          "models/preprocess_config_wide101.json -- only its "
                          "input_size is used, to size the random-noise input")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--ready-timeout", type=float, default=60.0)
    ap.add_argument("--max-seed-tries", type=int, default=20)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    chw = _shape_from_cfg(args.cfg_path)
    print(f"model: {args.model}\nxclbin: {args.xclbin}")
    print("finding two random-noise seeds with distinct top-1 predictions "
          "for the cross-talk check...")
    (seed_a, pred_a), (seed_b, pred_b) = find_two_distinct_seeds(
        args.model, args.cache_key, args.xclbin, chw, args.max_seed_tries)
    print(f"  a: seed {seed_a} -> class {pred_a}")
    print(f"  b: seed {seed_b} -> class {pred_b}\n")

    rows = []
    for n in sorted(set(args.procs)):
        print(f"=== {n} independent process(es), separate partitions if the "
              f"overlay allows it ===")
        combined, per, mism, parts = run_n(n, args.model, args.cache_key, args.xclbin,
                                            chw, seed_a, seed_b, pred_a, pred_b,
                                            args.seconds, args.ready_timeout)
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
