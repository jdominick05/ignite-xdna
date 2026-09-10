"""Run w4a8_probe.py over (kernel, mode, K), one fresh process per row, into one JSONL.

Arms rotate within each K and K rotates across repetitions, so no arm always runs first
or last. xrt-smi's partition report is recorded at the start and at the end: a foreign
hardware context on the device while this runs is contention, not a finding. The trace
cycle count of a call has come back identical on every call of a process so far, so a
second repetition checks reproducibility across processes rather than averaging noise.

Usage (ironenv, Desktop 2):
    python kernels/w4a8_probe/sweep.py --out results/aie/w4a8_probe_raw.jsonl --tag main
    python kernels/w4a8_probe/sweep.py --arms native:unroll2,i8i8:default --K 64,128
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_PROBE = os.path.join(_HERE, "w4a8_probe.py")


def _rel(arg: str) -> str:
    """Repo-relative, so a committed row carries no local profile or worktree path."""
    try:
        return os.path.relpath(arg, _ROOT) if os.path.isabs(arg) or os.path.exists(arg) else arg
    except ValueError:
        return arg
_XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"

DEFAULT_ARMS = ",".join(
    [
        "i8i8:default",  # upstream's kernel re-typed, as IRON builds it -- the control
        "i8i8:unroll2",  # the same with the unroll native needs, to separate the two
        "unpack:default",
        "unpack:no-unroll",
        "unpack:unroll2",
        "native:default",
        "native:unroll2",
        "i8i8_2x2:default",
        "native_2x2:unroll2",
    ]
)


def xrt_contexts() -> str:
    try:
        out = subprocess.run(
            [_XRT_SMI, "examine", "-r", "aie-partitions"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as e:  # noqa: BLE001
        return f"xrt-smi failed: {e}"
    lines = [ln.strip() for ln in out.splitlines() if "context" in ln.lower()]
    return " | ".join(lines) or out.strip()[-200:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="JSONL to append rows to")
    ap.add_argument("--arms", default=DEFAULT_ARMS, help="comma-separated kernel:mode")
    ap.add_argument("--K", default="64,128,256")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    arms = [tuple(a.split(":")) for a in args.arms.split(",") if a]
    ks = [int(k) for k in args.K.split(",") if k]

    def log(row):
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    log({"kind": "start", "host": socket.gethostname(), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
         "argv": [_rel(a) for a in sys.argv], "xrt_smi": xrt_contexts(), "tag": args.tag})
    failures = []
    for rep in range(args.reps):
        k_order = ks[rep % len(ks):] + ks[: rep % len(ks)]
        for ki, K in enumerate(k_order):
            shift = (rep + ki) % len(arms)
            for kernel, mode in arms[shift:] + arms[:shift]:
                cmd = [sys.executable, _PROBE, "--kernel", kernel, "--mode", mode,
                       "--M", str(args.M), "--K", str(K), "--N", str(args.N),
                       "--iters", str(args.iters), "--warmup", str(args.warmup),
                       "--tag", f"{args.tag} rep{rep}", "--json-out", args.out]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                dt = time.perf_counter() - t0
                tail = [ln for ln in proc.stdout.splitlines() if ln.startswith(("verify", "cycles", "build"))]
                print(f"rep{rep} K={K:<4} {kernel}:{mode:<10} exit {proc.returncode} {dt:5.1f}s  "
                      + " || ".join(tail), flush=True)
                if proc.returncode != 0:
                    failures.append((rep, K, kernel, mode, proc.returncode))
                    print(proc.stdout[-1500:], proc.stderr[-1500:], sep="\n", flush=True)
    log({"kind": "end", "host": socket.gethostname(), "t": time.strftime("%Y-%m-%d %H:%M:%S"),
         "xrt_smi": xrt_contexts(), "tag": args.tag, "failures": failures})
    print(f"done; {len(failures)} failed rows: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
