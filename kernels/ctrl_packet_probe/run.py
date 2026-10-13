"""Run one built (arm, N) design on Device 0: exactness of the written buffer, then timed dispatches.

python run.py --dir build/<arm>_<n> [--iters 200 --warmup 20]
Prints one RESULT json line: arm, n, insts bytes, ctrl bytes, exact (first and last run), median/p10/p90 ms.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pyxrt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--warmup", type=int, default=20)
a = ap.parse_args()
d = a.dir
arm, n = d.name.split("_")
insts = np.frombuffer((d / "insts.bin").read_bytes(), dtype=np.uint32)
ctrl = np.frombuffer((d / "ctrl.bin").read_bytes(), dtype=np.uint32)
exp = np.frombuffer((d / "expected.bin").read_bytes(), dtype=np.uint32)
S = pyxrt.xclBOSyncDirection

dev = pyxrt.device(0)
xb = pyxrt.xclbin(str(d / "aie.xclbin"))
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
ib = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
ib.write(insts.tobytes(), 0)
ib.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
out = pyxrt.bo(dev, exp.nbytes, pyxrt.bo.host_only, kern.group_id(3))
cb = pyxrt.bo(dev, max(ctrl.nbytes, 8), pyxrt.bo.host_only, kern.group_id(4))
cb.write(ctrl.tobytes(), 0)
cb.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
poison = np.full(exp.size, 0xDEADBEEF, dtype=np.uint32).tobytes()


def once(check):
    if check:
        out.write(poison, 0)
        out.sync(S.XCL_BO_SYNC_BO_TO_DEVICE)
    t0 = time.perf_counter()
    h = kern(3, ib, insts.nbytes, out, cb)
    st = h.wait(10000)
    t = (time.perf_counter() - t0) * 1e3
    if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise SystemExit(f"{arm} {n}: {st}")
    if not check:
        return t, None
    out.sync(S.XCL_BO_SYNC_BO_FROM_DEVICE)
    got = np.frombuffer(out.read(exp.nbytes, 0), dtype=np.uint32)
    return t, int((got != exp).sum())


_, bad_first = once(True)
for _ in range(a.warmup):
    once(False)
ts = [once(False)[0] for _ in range(a.iters)]
_, bad_last = once(True)
q = statistics.quantiles(ts, n=10)
print("RESULT " + json.dumps({
    "arm": arm, "n": int(n), "insts_bytes": int(insts.nbytes), "ctrl_bytes": int(ctrl.nbytes) if arm != "w32" else 0,
    "mismatch_first": bad_first, "mismatch_last": bad_last, "iters": a.iters,
    "median_ms": round(statistics.median(ts), 4), "p10_ms": round(q[0], 4), "p90_ms": round(q[-1], 4)}), flush=True)
del kern, ctx
