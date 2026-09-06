"""
Hold N independent NPU sessions open and busy for a fixed duration, so an
external tool (xrt-smi examine -r aie-partitions) can be polled against a
known, stable stream count -- built to test a specific alternative
explanation for the nstream_bench.py / nstream_cls_bench.py saturation
curves: is throughput capped by compute headroom running out (the
explanation those tools were built to support), or by NPU memory filling up
as more concurrent sessions/contexts pile on?

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


def sample_memory():
    """One `xrt-smi examine -r aie-partitions` call, parsed down to total MB
    (0 if no context is running). Polled from inside this process, not by a
    separate shell loop racing this script's own timing -- a prior attempt at
    external bash-side polling missed the whole window because build_session
    on a warm cache returns fast enough that a 5s poll interval started from
    a separate process can miss the entire hold."""
    try:
        out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return None, str(e)
    m = re.search(r"Total Memory Usage:\s*(\d+)\s*MB", out)
    return (int(m.group(1)) if m else 0), out


def hold(sess, inp_name, x, stop_event):
    while not stop_event.is_set():
        sess.run(None, {inp_name: x})


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
    threads = [threading.Thread(target=hold, args=(sess, name, x, stop_event))
               for sess, name, x in sessions]
    for t in threads:
        t.start()

    samples = []
    deadline = time.perf_counter() + args.hold
    time.sleep(1.0)  # let the hold threads actually start submitting first
    while time.perf_counter() < deadline:
        mb, _ = sample_memory()
        samples.append(mb)
        print(f"  streams={args.streams}  npu memory: {mb} MB")
        time.sleep(3.0)

    stop_event.set()
    for t in threads:
        t.join()
    ok = [s for s in samples if s is not None]
    if ok:
        print(f"streams={args.streams}  samples={ok}  "
              f"mean={sum(ok)/len(ok):.0f} MB  max={max(ok)} MB")
    print("done.")


if __name__ == "__main__":
    main()
