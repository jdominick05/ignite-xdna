"""Fit w4a8_probe rows: cycles per call = a + b*K per (kernel, mode), against the static loop.

The trace cycle count of one kernel call splits into a per-K part (the k loop: every
extra unit of K is M*N more MACs) and a per-call part (C tile loads and stores, the
outer loops, prologue/epilogue). A fit over K separates them: the slope b is cycles per
unit of K, so M*N / b is the kernel's MARGINAL MAC rate -- the figure that decides
whether int4 weights buy compute -- and the intercept a is what a call costs before any
k-step runs.

The static prediction for the slope assumes every vmac issues inside the steady-state
loop w4a8_probe's build check read out of the IRON object: (M*N / MACs-per-vmac) vmacs
per unit K, one loop execution per `loop_vmacs` of them, `loop_bundles` cycles each.
Measured above predicted is stall or work outside that loop; measured equal to it means
the silicon ran the loop exactly as scheduled.

Usage:
    python kernels/w4a8_probe/fit.py results/aie/w4a8_probe_raw.jsonl [--tag main]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

MACS_PER_VMAC = {"i8i8": 256, "unpack": 256, "native": 512,
                 "i8i8_2x2": 256, "unpack_2x2": 256, "native_2x2": 512}


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl")
    ap.add_argument("--tag", default=None, help="only rows whose tag starts with this")
    ap.add_argument("--control", default="i8i8:default", help="kernel:mode the ratios divide by")
    args = ap.parse_args(argv)

    rows = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "run":
                continue
            if args.tag and not str(r.get("tag", "")).startswith(args.tag):
                continue
            rows.append(r)
    if not rows:
        print("no run rows")
        return 1

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["kernel"], r["mode"])].append(r)

    print("Per call, per K: trace cycles (every call of every process), verify, build check")
    print(f"  {'arm':<22} {'K':>4} {'cycles':>14} {'procs':>5} {'verify':>10} {'build':>10} "
          f"{'loop':>9} {'frame':>6}")
    bad = []
    for arm in sorted(by_arm):
        for K in sorted({r["K"] for r in by_arm[arm]}):
            rs = [r for r in by_arm[arm] if r["K"] == K]
            cyc = sorted({c for r in rs for c in r["cycles"] if c is not None})
            ver = "ok" if all(r["verify"]["mismatches"] == 0 and r["verify_last"]["mismatches"] == 0
                              for r in rs) else "FAIL"
            reading = {r["verify"]["matches_reading"] for r in rs}
            bc = {r["build_check"]["status"] for r in rs}
            b0 = rs[0]["build_check"]
            loop = f"{b0.get('loop_bundles')}b/{b0.get('loop_vmacs')}v"
            cs = str(cyc[0]) if len(cyc) == 1 else f"{cyc[0]}..{cyc[-1]}"
            print(f"  {arm[0] + ':' + arm[1]:<22} {K:>4} {cs:>14} {len(rs):>5} {ver:>10} "
                  f"{'/'.join(sorted(bc)):>10} {loop:>9} {b0.get('frame') or '-':>6}")
            if ver != "ok" or bc != {"identical"}:
                bad.append((arm, K, ver, bc, reading))
    print()

    ctl = tuple(args.control.split(":"))
    fits = {}
    print("Fit over K: cycles = a + b*K (medians per K)")
    print(f"  {'arm':<22} {'a (cyc)':>9} {'b (cyc/K)':>10} {'r^2':>9} {'MAC/cyc':>8} "
          f"{'b static':>9} {'b/static':>9} {'vs ctl':>7}")
    for arm in sorted(by_arm):
        pts = defaultdict(list)
        for r in by_arm[arm]:
            pts[r["K"]].extend(c for c in r["cycles"] if c is not None)
        ks = sorted(pts)
        if len(ks) < 2:
            continue
        med = [sorted(pts[k])[len(pts[k]) // 2] for k in ks]
        a, b, r2 = fit(ks, med)
        r0 = by_arm[arm][0]
        MN = r0["M"] * r0["N"]
        bc = r0["build_check"]
        per_vmac = MACS_PER_VMAC[arm[0]]
        b_static = None
        if bc.get("loop_bundles") and bc.get("loop_vmacs"):
            b_static = (MN / per_vmac) / bc["loop_vmacs"] * bc["loop_bundles"]
        fits[arm] = (a, b, r2, MN / b, b_static)
    for arm, (a, b, r2, mpc, bs) in sorted(fits.items()):
        ratio = (fits[ctl][1] / b) if ctl in fits else float("nan")
        bs_s = f"{bs:.1f}" if bs else "-"
        br_s = f"{b / bs:.3f}" if bs else "-"
        print(f"  {arm[0] + ':' + arm[1]:<22} {a:>9.1f} {b:>10.3f} {r2:>9.6f} {mpc:>8.1f} "
              f"{bs_s:>9} {br_s:>9} {ratio:>6.2f}x")
    print()
    print(f"MAC/cyc = M*N / b, the marginal rate of the k loop; 'vs ctl' = the control's "
          f"slope over this arm's ({args.control}). b static here is the K={min(ks)} "
          f"build's loop only; a schedule that changes with K (native:unroll2 is 18 bundles "
          f"at K=64 and 20 above) is handled by the per-build table below.")
    print()

    # Per build: the static steady-state loop's total for THIS K's object, and what the
    # measured call spends beyond it. Fitting that residual over K splits it into cycles
    # per unit K the schedule does not contain (stalls, e.g. same-bank paired loads) and
    # a per-call overhead (C in/out, outer loops, prologue).
    print("Per build: measured vs the IRON object's own steady-state loop")
    print(f"  {'arm':<22} {'K':>4} {'measured':>9} {'loop':>9} {'static':>8} {'beyond':>8} "
          f"{'per loop exec':>13}")
    resid_fits = {}
    for arm in sorted(by_arm):
        xs, rs_ = [], []
        for K in sorted({r["K"] for r in by_arm[arm]}):
            rs = [r for r in by_arm[arm] if r["K"] == K]
            cyc = sorted(c for r in rs for c in r["cycles"] if c is not None)
            med = cyc[len(cyc) // 2]
            bc = rs[0]["build_check"]
            lb, lv = bc.get("loop_bundles"), bc.get("loop_vmacs")
            vm = rs[0]["vmacs"]
            execs = vm / lv
            static = execs * lb
            beyond = med - static
            print(f"  {arm[0] + ':' + arm[1]:<22} {K:>4} {med:>9} {f'{lb}b/{lv}v':>9} "
                  f"{static:>8.0f} {beyond:>8.0f} {'x'.join([f'{execs:.0f}']):>13}")
            xs.append(K)
            rs_.append(beyond)
        if len(xs) >= 2:
            resid_fits[arm] = fit(xs, rs_)
    print()
    print("Residual (measured - static loop) = c + d*K")
    print(f"  {'arm':<22} {'c overhead':>10} {'d stall/K':>10} {'r^2':>9}")
    for arm, (c0, d, r2) in sorted(resid_fits.items()):
        print(f"  {arm[0] + ':' + arm[1]:<22} {c0:>10.1f} {d:>10.3f} {r2:>9.6f}")
    print("d near 0 = the k loop ran exactly its static schedule; d > 0 = stall cycles per")
    print("unit K that no bundle accounts for. A schedule that changes with K puts its own")
    print("difference into d too, so read d next to the per-build table, not alone.")
    if bad:
        print("\nROWS THAT DID NOT VERIFY OR WHOSE BUILD DIFFERED:")
        for b_ in bad:
            print("  ", b_)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
