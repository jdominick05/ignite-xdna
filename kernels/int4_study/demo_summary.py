"""Gate C summary: today's int4 silicon runs against the 2026-09-10 W4A8 record. No hardware.

Reads the raw JSONL the three sweeps wrote (the one-core W4A8 probe repeated, the uint8 x int4
arm, the whole-array best tile repeated) plus the hwinfo_npu_bridge witness, and prints:

  A. one-core W4A8 probe: every (kernel, mode, K) today against 2026-09-10 -- mismatches and
     the median trace cycles per call, and whether they are the same number;
  B. uint8 x int4: mismatches, the reading the output matched, cycles per call, beside
     today's int8 x int4 and int8 x int8 arms at the same mode and K;
  C. whole array, 2048^3 at 64/128/64: per-run GOPS, native int4 over upstream int8 today
     and on 2026-09-10, and whether the output hash is the one every 2026-09-10 arm returned;
  W. the witness: samples, and the most hardware contexts it ever saw.

Usage:
    python kernels/int4_study/demo_summary.py --probe-new ... --probe-old ... --u8 ... \
        --array-new ... --array-old ... --witness ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def rows(path, tag_prefix=None, kind="run"):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") != kind:
                continue
            if tag_prefix and not str(r.get("tag", "")).startswith(tag_prefix):
                continue
            out.append(r)
    return out


def med_cycles(rs):
    cyc = sorted(c for r in rs for c in r["cycles"] if c is not None)
    return cyc[len(cyc) // 2] if cyc else None


def probe_table(rs):
    by = defaultdict(list)
    for r in rs:
        by[(r["kernel"], r["mode"], r["K"])].append(r)
    return by


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe-new", required=True)
    ap.add_argument("--probe-old", required=True)
    ap.add_argument("--probe-old-tag", default="main")
    ap.add_argument("--u8", required=True)
    ap.add_argument("--array-new", required=True)
    ap.add_argument("--array-old", required=True)
    ap.add_argument("--array-old-tag", default="main")
    ap.add_argument("--witness", required=True)
    args = ap.parse_args(argv)

    new = probe_table(rows(args.probe_new))
    old = probe_table(rows(args.probe_old, args.probe_old_tag))
    print("A. ONE-CORE W4A8 PROBE, today vs 2026-09-10 (median trace cycles per call, all calls of all processes)")
    print(f"  {'kernel:mode':<22} {'K':>4} {'procs':>5} {'mismatch':>8} {'today':>7} {'09-10':>7} {'same':>5}")
    n_same = n_all = n_bad = 0
    for key in sorted(new):
        k, m, K = key
        rn = new[key]
        mis = sum(r["verify"]["mismatches"] + r["verify_last"]["mismatches"] for r in rn)
        n_bad += mis
        ct, co = med_cycles(rn), med_cycles(old.get(key, []))
        same = "yes" if ct == co else ("-" if co is None else "NO")
        n_all += 1
        n_same += ct == co
        print(f"  {k + ':' + m:<22} {K:>4} {len(rn):>5} {mis:>8} {ct!s:>7} {co!s:>7} {same:>5}")
    print(f"  {n_same} of {n_all} (kernel, mode, K) medians identical to 2026-09-10; total mismatches {n_bad}")
    print()

    u8 = probe_table(rows(args.u8))
    print("B. uint8 x int4 (the engine's operand pair), one core, beside today's int8 arms")
    print(f"  {'kernel:mode':<22} {'K':>4} {'procs':>5} {'mismatch':>8} {'A>=128':>7} {'cycles':>7}"
          f" {'int8 twin':<18} {'twin cyc':>8}  output matched")
    twin = {"u8i8": "i8i8", "u8native": "native"}
    for key in sorted(u8):
        k, m, K = key
        rs = u8[key]
        mis = sum(r["verify"]["mismatches"] + r["verify_last"]["mismatches"] for r in rs)
        frac = statistics.mean(r["verify"].get("a_ge_128_fraction", float("nan")) for r in rs)
        reading = sorted({str(r["verify"]["matches_reading"]) for r in rs})
        tk = (twin[k], m, K)
        tc = med_cycles(new.get(tk, []))
        print(f"  {k + ':' + m:<22} {K:>4} {len(rs):>5} {mis:>8} {frac:>7.1%} {med_cycles(rs)!s:>7}"
              f" {tk[0] + ':' + m:<18} {tc!s:>8}  {'; '.join(reading)}")
    print()

    def array_stats(rs):
        by = defaultdict(list)
        for r in rs:
            by[(r["arm"], r["mode"], r["c_single_buffer"], r["m"], r["k"], r["n"])].append(r)
        return by

    an = array_stats(rows(args.array_new))
    ao = array_stats(rows(args.array_old, args.array_old_tag))
    print("C. WHOLE ARRAY, 2048^3 (GOPS per run; ratio = mean native / mean upstream at the same tile)")
    for label, st in (("today", an), ("2026-09-10", ao)):
        up = [r["gops"] for key, rs in st.items() if key[0] == "upstream" and key[3:] == (64, 128, 64)
              for r in rs]
        nat = [r["gops"] for key, rs in st.items() if key[:3] == ("native", "unroll2", 1)
               and key[3:] == (64, 128, 64) for r in rs]
        hashes = sorted({r.get("c_sha256") for rs in st.values() for r in rs
                         if r["m"] == 64 and r["k"] == 128 and r["n"] == 64})
        mism = sum(r.get("mismatches", 0) for rs in st.values() for r in rs)
        if up and nat:
            print(f"  {label:<11} upstream int8 {', '.join(f'{g:,.2f}' for g in up)} | native int4 "
                  f"{', '.join(f'{g:,.2f}' for g in nat)} | ratio {statistics.mean(nat) / statistics.mean(up):.3f}x"
                  f" | mismatches {mism} | C hash {', '.join(hashes)}")
            print(f"  {'':<11} ranges disjoint: {min(nat) > max(up)}")
    print()

    n = mx = multi = multi_pid = 0
    pids, procs, statuses = set(), set(), defaultdict(int)
    with open(args.witness, encoding="utf-8") as f:
        for line in f:
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            c = (w.get("xrt_smi") or {}).get("active_contexts") or 0
            mx = max(mx, c)
            multi += c > 1
            ctxs = w.get("contexts") or []
            here = {ctx.get("pid") for ctx in ctxs}
            multi_pid += len(here) > 1
            for ctx in ctxs:
                pids.add(ctx.get("pid"))
                procs.add(ctx.get("process"))
                statuses[ctx.get("status")] += 1
    dev = {}
    with open(args.witness, encoding="utf-8") as f:
        first = f.readline()
        if first.strip():
            dev = json.loads(first).get("device", {})
    print(f"W. WITNESS: {n} samples at 1 s; most hardware contexts at once {mx}; samples with more than "
          f"one context {multi}")
    print(f"  device {dev.get('name')} {dev.get('bdf')}, NPU driver {dev.get('npu_driver')}, XRT {dev.get('xrt')}, "
          f"firmware {dev.get('firmware')}, power mode {dev.get('power_mode')}")
    n_rows = sum(len(rows(p)) for p in (args.probe_new, args.u8, args.array_new))
    print(f"  contexts came from {len(pids)} distinct processes ({', '.join(sorted(map(str, procs)))}); the "
          f"sweeps ran {n_rows} fresh processes, and a sub-second one can fall between 1 s samples")
    print(f"  samples with two different processes on the NPU at once (xrt-smi contexts or engine-only "
          f"use): {multi_pid}")
    print(f"  context status counts over all samples: {dict(sorted(statuses.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
