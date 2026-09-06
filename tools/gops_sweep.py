"""
Turn xrt-smi's per-context GOPS reading into a utilization-vs-16-TOPS number,
across model sizes -- the "still untried" half of the direct NPU utilization
question in RESEARCH.md (Windows' GPU Engine/GPU Adapter Memory counters
can't see this device at all; xrt-smi's `aie-partitions` report is the tool
that can, and it carries a GOPS column nothing else in this repo has read).

For each --model, builds one NPU session, holds it busy in a background
thread, and polls `xrt-smi examine -r aie-partitions` a few times, taking the
GOPS value once it stabilizes (it reads as a flat integer per model, not a
per-poll fluctuating rate, in every case measured so far).

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/gops_sweep.py \
        --pairs models/yolov8n_cut_xint8.onnx:yolocutcachekey \
                models/yolov8s_cut_xint8.onnx:yolocutcachekey \
        --hold 15 --fresh

Writes nothing to results/ itself -- pipe through scripts/lib.sh's
run_logged (or plain `tee`) if the run is worth keeping as a logged claim.
"""
import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from npu.session import build_session, clear_cache

XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
NPU_PEAK_GOPS = 16000  # 16 TOPS nameplate, in giga-ops


def sample_gops():
    """One `xrt-smi examine -r aie-partitions` call. Returns (gops, mb) for
    this process's own context, or (None, None) if no context is up yet."""
    try:
        out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None, None
    mb_m = re.search(r"Total Memory Usage:\s*(\d+)\s*MB", out)
    # The GOPS value sits on the "python.exe |Active |<completions> |... |<GOPS>" row.
    gops_m = re.search(r"python\.exe\s+\|Active\s+\|\d+\s+\|\d+\s+\|[^|]*\|\s*(\d+)", out)
    mb = int(mb_m.group(1)) if mb_m else None
    gops = int(gops_m.group(1)) if gops_m else None
    return gops, mb


def hold(sess, inp_name, x, stop_event):
    while not stop_event.is_set():
        sess.run(None, {inp_name: x})


def sweep_one(model, cache_key, hold_s, fresh):
    if fresh:
        clear_cache(cache_key)
    print(f"\n== {model}  (cache: {cache_key}) ==")
    sess = build_session(model, "npu", cache_key, log_severity=2)
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.zeros(shape, dtype=np.float32)

    stop_event = threading.Event()
    t = threading.Thread(target=hold, args=(sess, inp.name, x, stop_event))
    t.start()
    time.sleep(1.5)  # let the hold loop actually start submitting

    samples = []
    deadline = time.perf_counter() + hold_s
    while time.perf_counter() < deadline:
        gops, mb = sample_gops()
        if gops is not None:
            samples.append((gops, mb))
        time.sleep(2.0)

    stop_event.set()
    t.join()

    if not samples:
        print(f"  no samples captured for {model}")
        return None
    gops_vals = [g for g, _ in samples]
    mb_vals = [m for _, m in samples if m is not None]
    gops = max(gops_vals)  # flat per model in practice; max avoids a slow first poll
    mb = max(mb_vals) if mb_vals else None
    pct = 100.0 * gops / NPU_PEAK_GOPS
    print(f"  samples: {samples}")
    print(f"  GOPS={gops}  ({pct:.3f}% of {NPU_PEAK_GOPS} GOPS / 16 TOPS nameplate)"
          f"  memory={mb} MB")
    return model, gops, pct, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                     help="model:cache_key entries, e.g. models/x.onnx:yolocutcachekey")
    ap.add_argument("--hold", type=float, default=15.0, help="seconds to hold each session busy")
    ap.add_argument("--fresh", action="store_true", help="clear the compile cache before each model")
    args = ap.parse_args()

    rows = []
    for pair in args.pairs:
        model, cache_key = pair.rsplit(":", 1)
        result = sweep_one(model, cache_key, args.hold, args.fresh)
        if result:
            rows.append(result)

    print("\n================ RESULT ================")
    print(f"{'model':<45} {'GOPS':>6} {'% of 16 TOPS':>13} {'mem MB':>8}")
    for model, gops, pct, mb in rows:
        print(f"{Path(model).name:<45} {gops:>6} {pct:>12.3f}% {mb!s:>8}")


if __name__ == "__main__":
    main()
