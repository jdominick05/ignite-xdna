"""Run every built design, forward then reverse, each in its own process; echo RESULT lines with the order,
then the per-arm least-squares slope (ns per written word) of median dispatch ms over N, per order."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = r"C:\Users\Ignis\mlir-aie\ironenv\Scripts\python.exe"
SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
ARMS, NS = ["w32", "ctrlr1", "ctrlr4"], [64, 256, 1024, 4096, 16384]
designs = [HERE / "build" / f"{a}_{n}" for a in ARMS for n in NS]
env = dict(os.environ, PYTHONPATH=r"C:\Xilinx\XRT\xrt_sdk\xrt\python")
rows = []
for order, seq in (("forward", designs), ("reverse", designs[::-1])):
    for d in seq:
        s = subprocess.run([SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True).stdout
        if "No hardware contexts running" not in s:
            sys.exit(f"device not idle before {d.name}")
        r = subprocess.run([PY, str(HERE / "run.py"), "--dir", str(d), "--iters", "200", "--warmup", "20"],
                           capture_output=True, text=True, env=env, timeout=300)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            sys.exit(f"{d.name}: no result (exit {r.returncode}): {r.stdout[-400:]} {r.stderr[-400:]}")
        row = {"order": order, **json.loads(line[7:])}
        rows.append(row)
        print("SWEEP " + json.dumps(row), flush=True)


def slope(pts):
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


for order in ("forward", "reverse"):
    for arm in ARMS:
        pts = [(r["n"], r["median_ms"]) for r in rows if r["order"] == order and r["arm"] == arm]
        print("SLOPE " + json.dumps({"order": order, "arm": arm, "ns_per_word": round(slope(pts) * 1e6, 2),
                                     "exact": all(r["mismatch_first"] == 0 and r["mismatch_last"] == 0
                                                  for r in rows if r["order"] == order and r["arm"] == arm)}),
              flush=True)
