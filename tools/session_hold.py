"""
Hold N independent NPU sessions open and busy for a fixed duration, so an
external tool (xrt-smi examine -r aie-partitions) can be polled against a
known, stable stream count -- built to test a specific alternative
explanation for the nstream_bench.py / nstream_cls_bench.py saturation
curves: is throughput capped by compute headroom running out (the
explanation those tools were built to support), or by NPU memory filling up
as more concurrent sessions/contexts pile on?

Also parses xrt-smi's per-context GOPS column (previously unused -- only
memory was parsed). GOPS is xrt-smi's own reported compute rate for the HW
context, independent of anything this process measures itself, so it's a
second, corroborating axis on the same saturation question: if the
nstream_bench.py fps curve flattens at N streams while GOPS keeps climbing
past N, throughput is capped by something other than raw compute throughput
on this context; if GOPS flattens at the same N, that's direct evidence the
array itself is the saturating resource.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/session_hold.py --model models/yolov8n_cut_xint8.onnx \
        --cache-key yolocutcachekey --streams 4 --hold 20

While this runs, in another shell:
    xrt-smi examine -r aie-partitions

Writes nothing to results/ itself.
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


_CTX_ROW_RE = re.compile(r"\|\s*\S+\.exe\s*\|Active\s*\|\d+\s*\|\d+\s*\|\s*\|(\d+)\s*\|")


def sample_memory():
    """One `xrt-smi examine -r aie-partitions` call, parsed down to
    (total_mb, total_gops, raw_output). total_gops sums the GOPS column
    across every Active HW-context row (there can be more than one row even
    from a single process holding multiple sessions, and in practice we've
    also seen all sessions from one process collapse onto a single shared
    context -- summing handles both cases). 0/0 if no context is running.

    Polled from inside this process, not by a separate shell loop racing
    this script's own timing -- a prior attempt at external bash-side
    polling missed the whole window because build_session on a warm cache
    returns fast enough that a 5s poll interval started from a separate
    process can miss the entire hold."""
    try:
        out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return None, None, str(e)
    m = re.search(r"Total Memory Usage:\s*(\d+)\s*MB", out)
    gops_rows = [int(g) for g in _CTX_ROW_RE.findall(out)]
    total_gops = sum(gops_rows) if gops_rows else 0
    return (int(m.group(1)) if m else 0), total_gops, out


def hold(sess, inp_name, x, stop_event, counter):
    while not stop_event.is_set():
        sess.run(None, {inp_name: x})
        counter[0] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-key", required=True)
    ap.add_argument("--streams", type=int, required=True)
    ap.add_argument("--hold", type=float, default=20.0, help="seconds to hold sessions busy")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log", type=int, default=2)
    args = ap.parse_args()

    if args.fresh:
        clear_cache(args.cache_key)

    print(f"building {args.streams} session(s) against {args.model} ...")
    sessions = []
    for i in range(args.streams):
        sess = build_session(args.model, "npu", args.cache_key, args.xclbin, log_severity=args.log)
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        x = np.zeros(shape, dtype=np.float32)
        sessions.append((sess, inp.name, x))

    print(f"{args.streams} session(s) ready. Holding busy for {args.hold:.0f}s, "
          f"sampling xrt-smi every 3s from inside this process ...")
    stop_event = threading.Event()
    counters = [[0] for _ in sessions]  # one own list per thread -- no shared-write race
    threads = [threading.Thread(target=hold, args=(sess, name, x, stop_event, counters[i]))
               for i, (sess, name, x) in enumerate(sessions)]
    for t in threads:
        t.start()

    mem_samples = []
    gops_samples = []
    fps_samples = []
    deadline = time.perf_counter() + args.hold
    time.sleep(1.0)  # let the hold threads actually start submitting first
    last_total = sum(c[0] for c in counters)
    last_t = time.perf_counter()
    while time.perf_counter() < deadline:
        mb, gops, _ = sample_memory()
        mem_samples.append(mb)
        gops_samples.append(gops)
        now = time.perf_counter()
        total = sum(c[0] for c in counters)
        fps = (total - last_total) / (now - last_t)
        fps_samples.append(fps)
        last_total, last_t = total, now
        print(f"  streams={args.streams}  npu memory: {mb} MB  gops: {gops}  "
              f"measured completions: {fps:.1f}/s")
        time.sleep(3.0)

    stop_event.set()
    for t in threads:
        t.join()
    ok_mem = [s for s in mem_samples if s is not None]
    ok_gops = [s for s in gops_samples if s is not None]
    ok_fps = fps_samples
    if ok_mem:
        print(f"streams={args.streams}  mem_samples={ok_mem}  "
              f"mem_mean={sum(ok_mem)/len(ok_mem):.0f} MB  mem_max={max(ok_mem)} MB")
    if ok_gops:
        print(f"streams={args.streams}  gops_samples={ok_gops}  "
              f"gops_mean={sum(ok_gops)/len(ok_gops):.1f}  gops_max={max(ok_gops)}")
    if ok_fps:
        print(f"streams={args.streams}  fps_samples={[round(f,1) for f in ok_fps]}  "
              f"fps_mean={sum(ok_fps)/len(ok_fps):.1f}")
    print("done.")


if __name__ == "__main__":
    main()
