#!/usr/bin/env python3
r"""Sample AMD RAPL package and per-core power while a workload runs.

This is the first thing in this repo that measures a watt. It exists because the NPU
exposes no power telemetry of its own -- `xrt-smi` reports `Estimated Power: N/A` and the
electrical query fails at the driver escape -- so the only way to ask "does the NPU
actually save energy" is to watch the package it shares with the CPU.

WHERE THE NUMBERS COME FROM
Windows exposes AMD's RAPL counters through PDH, with no driver, no elevation and no
hardware context:

    \Energy Meter(RAPL_Package0_PKG)\Power        package
    \Energy Meter(RAPL_Package0_CoreN_CORE)\Power per core, N = 0..7
    \Energy Meter(_Total)\Power                   reads 0 on this machine, ignored

`Power` is in MILLIWATTS. That is MEASURED here, not read off a spec: a one-thread busy
loop steps the package +9.59 W over a 41.3 W idle, and sixteen threads step it +42.1 W to
83.5 W, which is where a 65 W-class Zen 4 desktop part belongs. The same counter set also
exposes `Energy` and `Time`; `Time` is ~1007 units/s (milliseconds), while `Energy`'s
absolute scale does NOT resolve to picojoules, microjoules or millijoules against the
Power reading. Energy tracks Power faithfully -- their ratio is constant at 0.0036 across
a 2x power range -- but its unit is UNRESOLVED, so this tool integrates the Power column
and does not use Energy.

WHAT THIS CANNOT DO, AND IT MATTERS
The NPU is on the same package as the CPU, so `RAPL_Package0_PKG` includes it while the
per-core meters do not. There is no NPU power domain. Any NPU figure is therefore a
RESIDUAL -- package minus the sum of cores minus whatever idle draws -- and it is only
meaningful against an idle baseline AND a matched CPU-only control run in the same
sitting. Never quote a bare wattage from this tool; quote a delta against a stated
baseline. That is docs/SILICON.md objective S4's method and it is not optional.

    python tools/power_probe.py --label idle --duration 30
    python tools/power_probe.py --label cpu-gemm --duration 30 --command "python bench.py"

Sampling shells out to Windows' built-in `typeperf` (1 s minimum interval), the same way
this repo already shells out to `xrt-smi`.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PKG = r"\Energy Meter(RAPL_Package0_PKG)\Power"
CORES = [rf"\Energy Meter(RAPL_Package0_Core{i}_CORE)\Power" for i in range(8)]


def sample_counters(duration, interval, extra_paths=None):
    """Run typeperf for `duration` seconds; return {path: [values]}."""
    paths = [PKG] + CORES + list(extra_paths or [])
    out = Path(tempfile.gettempdir()) / f"power_probe_{int(time.time()*1000)}.csv"
    # typeperf's -si takes [[hh:]mm:]ss -- an integer number of seconds, never a float.
    si = max(1, int(round(interval)))
    n = max(1, int(round(duration / si)))
    # -y answers "overwrite?" without prompting. Without it typeperf blocks forever on an
    # existing -o file, with no output at all -- which is exactly what a temp file that
    # was created before being handed over looks like.
    cmd = ["typeperf", *paths, "-si", str(si), "-sc", str(n), "-y", "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"typeperf produced nothing (rc={proc.returncode})\n{proc.stdout}\n{proc.stderr}")

    series = {p: [] for p in paths}
    with out.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        sys.exit("typeperf CSV empty")
    header = rows[0]
    # typeperf writes fully-qualified \\HOST\Energy Meter(...)\Power headers
    idx = {}
    for i, h in enumerate(header):
        for p in paths:
            if h.lower().endswith(p.lower().lstrip("\\")):
                idx[p] = i
    for row in rows[1:]:
        if not row or len(row) < 2:
            continue
        for p, i in idx.items():
            try:
                series[p].append(float(row[i]))
            except (ValueError, IndexError):
                pass
    out.unlink(missing_ok=True)
    return series


def summarize(series):
    def stat(vals):
        if not vals:
            return None
        return dict(
            mean_mw=statistics.mean(vals),
            median_mw=statistics.median(vals),
            min_mw=min(vals),
            max_mw=max(vals),
            n=len(vals),
        )

    pkg = stat(series.get(PKG, []))
    core_series = [series.get(c, []) for c in CORES]
    ncore = min((len(s) for s in core_series if s), default=0)
    core_sum = [sum(s[i] for s in core_series if len(s) > i) for i in range(ncore)]
    return dict(package=pkg, core_sum=stat(core_sum),
                per_core={f"core{i}": stat(s) for i, s in enumerate(core_series)})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True, help="phase name, e.g. idle / cpu-gemm / npu-gemm")
    p.add_argument("--duration", type=float, default=30.0, help="seconds to sample")
    p.add_argument("--interval", type=float, default=1.0, help="seconds per sample (typeperf min 1)")
    p.add_argument("--command", default=None,
                   help="workload to run in a subprocess while sampling; omit for an idle baseline")
    p.add_argument("--json", default=None, help="append the summary as one JSON line to this file")
    args = p.parse_args()

    proc = None
    if args.command:
        # shell=True is deliberate: --command is a full command line typed by whoever runs
        # this probe (it needs pipes, env prefixes and quoting), never untrusted input.
        # This is a local measurement tool, not a service.
        proc = subprocess.Popen(args.command, shell=True,  # noqa: S602
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # let the workload reach steady state before sampling

    t0 = time.time()
    series = sample_counters(args.duration, args.interval)
    wall = time.time() - t0

    if proc is not None:
        # proc.terminate() is NOT enough here: shell=True on Windows makes proc a cmd.exe
        # wrapper, and TerminateProcess on it does not touch its child (the actual
        # workload). That child was measured surviving 20+ seconds past this function
        # returning, bleeding into the NEXT phase's sampling window -- this is what
        # silently contaminated four throughput readings in an earlier ResNet50 energy study
        # and, worse, the POWER phases of a BiSeNetV2 study (both run on side branches and
        # never committed; the BiSeNetV2 figures were retracted there). `taskkill /T` kills
        # the whole process tree, not just the tracked PID.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass

    s = summarize(series)
    s.update(label=args.label, wall_s=wall, command=args.command,
             sampled_s=args.duration, interval_s=args.interval)

    pkg, csum = s["package"], s["core_sum"]
    print(f"=== {args.label} ===")
    if pkg:
        print(f"  package   mean {pkg['mean_mw']/1000:7.3f} W   "
              f"min {pkg['min_mw']/1000:7.3f}   max {pkg['max_mw']/1000:7.3f}   n={pkg['n']}")
    if csum:
        print(f"  cores sum mean {csum['mean_mw']/1000:7.3f} W   "
              f"min {csum['min_mw']/1000:7.3f}   max {csum['max_mw']/1000:7.3f}")
    if pkg and csum:
        print(f"  package - cores  {(pkg['mean_mw'] - csum['mean_mw'])/1000:7.3f} W"
              "   (uncore + SoC + NPU; NOT an NPU figure on its own)")

    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(s) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
