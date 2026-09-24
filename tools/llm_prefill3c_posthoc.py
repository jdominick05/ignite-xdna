#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study stage 3c, POST-HOC and REPORT-ONLY: not pre-registered, it decides nothing and changes no rule,
outcome or score. Asked by the gate after the verdict (7608d0a) for the write-up.

It reads the committed sitting logs (A2 and B) through tools/llm_prefill3c.py's frozen verdict functions,
which it imports and never edits (the printed VERDICT_CODE_SHA256 must be the frozen 6e459e56...), and prints:
- each rule's binding rival and margin, per NPU arm, metric and chip, with the at-least-as-accurate set
  (MISSING members are named; their single-pass values are not printed);
- energy with the idle included, E_gross = window package W / (M x layers per second), beside the
  pre-registered E, which charges idle to no one;
- every window's idle package power, and the report-only witnesses per window;
- the NPU arms' per-dispatch medians, and 3b's gate/up concatenation timed beside them.

    python tools/llm_prefill3c_posthoc.py SITTING_LOG [SITTING_LOG ...]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import llm_prefill3c as p3c  # noqa: E402  (the frozen verdict functions; imported, never edited)

LABEL = "POST-HOC, REPORT-ONLY, not pre-registered: it decides nothing and changes no rule, outcome or score"
FROZEN = "6e459e560c838909295e554994c950ad3ce32cf0b15a150123ab5481d17dc549"
CHIPS = (("C", "CPU"), ("D", "DirectML"))
METRIC = {"T": "speed", "E": "energy"}


def f(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def mj(x):
    return f(None if x is None else x * 1e3, ".4f")


def windows_taken(rm: list, arm: str) -> dict:
    """The frozen selection: per pass, taken()'s window if state_of() calls it OK, else None."""
    out = {}
    for p in (1, 2):
        w = p3c.taken([r for r in rm if r["arm"] == arm and r["pass"] == p])
        out[p] = w if w and p3c.state_of(w) == "OK" else None
    return out


def rate(w: dict) -> float:
    """Prompt tokens per second through one layer: M x fractional layers / the window's seconds."""
    a, b = w["window"]
    return w["M"] * w["layers"]["fractional"] / (b - a)


def energy(w: dict) -> dict:
    pw = w["power"]
    wp, ip = pw["window_pkg_w"], pw["idle_pkg_w"]
    r = rate(w)
    return {"gross": wp / r, "net_recomputed": (wp - ip) / r, "net_logged": w["E"], "window_w": wp, "idle_w": ip}


def arm_energy(rm: list, arm: str, rec: dict):
    """Two-pass means for a COMPLETE arm (the rules' own form); None for a MISSING or BROKEN arm."""
    if rec["state"] != "COMPLETE":
        return None
    es = [energy(w) for w in windows_taken(rm, arm).values()]
    return {k: statistics.fmean(e[k] for e in es) for k in es[0]}


def members(arms: dict, ref, chip: str) -> tuple:
    """The frozen rule's membership: rel-L2 <= ACC_TIE x ref; unknown error counts as possibly in."""
    inset, out = [], []
    for k, v in arms.items():
        if k[0] != chip or k[:2] not in ("C-", "D-"):
            continue
        m = None if v["err"] is None or ref is None else v["err"] <= p3c.ACC_TIE * ref
        (out if m is False else inset).append(k)
    return inset, out


def binding(arms: dict, inset: list, metric: str, npu_val: float, key=None) -> dict:
    """The COMPLETE member with the lowest value (the one the NPU arm must beat by MARGIN), its ratio, and
    the MISSING or BROKEN members named."""
    val = key or (lambda k: arms[k]["values"][metric][0])
    comp = [k for k in inset if arms[k]["state"] == "COMPLETE"]
    other = [f"{k} ({arms[k]['state']})" for k in inset if arms[k]["state"] != "COMPLETE"]
    if not comp:
        return {"member": None, "value": None, "ratio": None, "other": other}
    k = min(comp, key=val)
    return {"member": k, "value": val(k), "ratio": val(k) / npu_val, "other": other}


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        if p3c.sha_lf(p) in p3c.PARTIAL_LOGS:
            print(f"REFUSE {p.name}: {p3c.PARTIAL_LOGS[p3c.sha_lf(p)]}")
            return 3
    print("STAGE 3c POST-HOC: the binding rivals, energy with the idle included, and every window's idle")
    print(LABEL)
    sha = p3c.verdict_code_sha()
    print(f"VERDICT_CODE_SHA256 {sha} ({'the frozen value' if sha == FROZEN else 'NOT THE FROZEN VALUE'};"
          " this script imports the verdict functions and edits none)")
    if sha != FROZEN:
        return 4
    for p in paths:
        rel = p.resolve().relative_to(ROOT).as_posix()
        print(f"INPUT {rel} sha256_lf {p3c.sha_lf(p)}")
    recs, _, _ = p3c.parse(paths)
    ev = p3c.evaluate(recs)
    print(f"MARGIN {p3c.MARGIN}  ACC_TIE {p3c.ACC_TIE}  (the frozen constants)")
    print("E and E_gross are J per prompt token for one layer's seven weight GEMMs (blk.16), printed in mJ;"
          " package counters only, DRAM outside")
    summary = {}
    worst_net = 0.0
    for M in p3c.MS:
        rm = [r for r in recs if r["M"] == M]
        arms, res = ev[M]["arms"], ev[M]
        summary[M] = {"role": res["role"], "rules": {}, "gross": {}}
        print(f"\n==================== M = {M}  (the frozen role: {res['role']})")

        print("\n-- the arms: the frozen arm_record; a COMPLETE arm's two taken passes averaged")
        print(f"  {'arm':<10} {'state':<9} {'T ms':>9} {'E mJ':>8} {'window W':>9} {'idle W':>7} {'above W':>8}"
              f" {'E_gross mJ':>10} {'rel-L2':>9}")
        eng = {}
        for arm, rec in arms.items():
            e = arm_energy(rm, arm, rec)
            eng[arm] = e
            if e is None:
                print(f"  {arm:<10} {rec['state']:<9} {'(one valid pass; not quoted)':>52} {f(rec['err'], '.2e'):>9}")
                continue
            for w in windows_taken(rm, arm).values():
                x = energy(w)
                worst_net = max(worst_net, abs(x["net_recomputed"] / x["net_logged"] - 1))
            summary[M]["gross"][arm] = e["gross"]
            print(f"  {arm:<10} {rec['state']:<9} {f(rec['values']['T'][0]):>9} {mj(e['net_logged']):>8}"
                  f" {f(e['window_w']):>9} {f(e['idle_w']):>7} {f(e['window_w'] - e['idle_w']):>8}"
                  f" {mj(e['gross']):>10} {f(rec['err'], '.2e'):>9}")

        print("\n-- the rules: per NPU arm, metric and chip, the at-least-as-accurate set's binding member (the"
              " COMPLETE member with the lowest value) and its ratio to the NPU arm; beaten needs >= MARGIN")
        nb4 = [arms[k]["err"] for k in ("C-nb4@8", "C-nb4@16") if k in arms and arms[k]["err"] is not None]
        for npu in (*p3c.NPU_ARMS, "N-w4"):
            if npu not in arms or arms[npu]["state"] != "COMPLETE":
                print(f"  {npu}: {arms.get(npu, {}).get('state', 'no windows')}")
                continue
            ref = arms[npu]["err"] if npu != "N-w4" else max(nb4)
            ref_note = "its own" if npu != "N-w4" else "C-nb4's (N-w4 has no float accuracy; U4)"
            for metric in ("T", "E"):
                if npu == "N-w4":
                    frozen = res["w4"][metric]["reading"]
                else:
                    frozen = res["rules"][npu][METRIC[metric]]["outcome"]
                nv = arms[npu]["values"][metric][0]
                print(f"  {npu} {METRIC[metric]} ({metric}; the frozen {'reading' if npu == 'N-w4' else 'outcome'}:"
                      f" {frozen}); ref rel-L2 {f(ref, '.3e')}, {ref_note}; the set: rel-L2 <="
                      f" {f(p3c.ACC_TIE * ref, '.3e')}; the NPU arm {f(nv) if metric == 'T' else mj(nv)}")
                row = {"frozen": frozen}
                for chip, name in CHIPS:
                    inset, out = members(arms, ref, chip)
                    bd = binding(arms, inset, metric, nv)
                    v = "-" if bd["value"] is None else (f(bd["value"]) if metric == "T" else mj(bd["value"]))
                    beaten = None if bd["ratio"] is None else bd["ratio"] >= p3c.MARGIN
                    print(f"    {name:<8} binding {bd['member'] or '-'} {v} -> {f(bd['ratio'], '.3f')}x"
                          f" ({'beaten' if beaten else 'not beaten' if beaten is False else '-'} by the COMPLETE members);"
                          f" MISSING/BROKEN in or possibly in the set: {', '.join(bd['other']) or 'none'};"
                          f" out (less accurate): {', '.join(out) or 'none'}")
                    row[name] = {"member": bd["member"], "ratio": bd["ratio"], "other": bd["other"]}
                summary[M]["rules"][f"{npu} {METRIC[metric]}"] = row

        print("\n-- energy with the idle included (POST-HOC): each NPU arm's E_gross against the lowest E_gross"
              " among the COMPLETE members of its energy set, per chip; the same >= MARGIN line, for reading only")
        for npu in (*p3c.NPU_ARMS, "N-w4"):
            if eng.get(npu) is None:
                continue
            ref = arms[npu]["err"] if npu != "N-w4" else max(nb4)
            ng = eng[npu]["gross"]
            line = [f"  {npu}: E_gross {mj(ng)} (net {mj(eng[npu]['net_logged'])})"]
            for chip, name in CHIPS:
                inset, _ = members(arms, ref, chip)
                bd = binding(arms, inset, "E", ng, key=lambda k: eng[k]["gross"] if eng.get(k) else float("inf"))
                line.append(f"{name} {bd['member'] or '-'} {mj(bd['value'])} -> {f(bd['ratio'], '.3f')}x"
                            + (f" (MISSING/BROKEN: {', '.join(bd['other'])})" if bd["other"] else ""))
                summary[M]["rules"].setdefault(f"{npu} energy", {})[f"{name} gross"] = {
                    "member": bd["member"], "ratio": bd["ratio"]}
            print("; ".join(line))

        print("\n-- every window's idle (all windows, VOID and re-runs included; W, package counter)")
        print(f"  {'pos':>3} {'pass':>4} {'arm':<10} {'rerun':<5} {'state':<5} {'idle W':>7} {'idle SD':>7} {'rows':>4}")
        idle = {}
        for w in sorted(rm, key=lambda r: r["position"]):
            pw = w.get("power") or {}
            idle[(w["arm"], w["pass"], w["position"])] = pw.get("idle_pkg_w")
            print(f"  {w['position']:>3} {w['pass']:>4} {w['arm']:<10} {str(w.get('rerun')):<5} {w['state']:<5}"
                  f" {f(pw.get('idle_pkg_w')):>7} {f(pw.get('idle_pkg_sd_w')):>7} {f(pw.get('idle_rows'), 'd'):>4}")
        vals = [v for v in idle.values() if v is not None]
        print(f"  idle range {f(min(vals))} to {f(max(vals))} W over {len(vals)} windows")
        for npu in (*p3c.NPU_ARMS, "N-w4"):
            diffs = []
            for p in (1, 2):
                wn, wd = windows_taken(rm, npu)[p], windows_taken(rm, "D-nb16")[p]
                if wn and wd:
                    diffs.append(wn["power"]["idle_pkg_w"] - wd["power"]["idle_pkg_w"])
            print(f"  {npu} idle minus D-nb16's idle, same pass (p1, p2): {', '.join(f(d, '+.2f') for d in diffs)} W")

        print("\n-- the witnesses per window (report-only; rows after the 1 s trims, row gaps in s, CPU s in the"
              " window, the reader's CPU % of one logical CPU, the 780M's summed busy %, other adapters' busy %)")
        print(f"  {'pos':>3} {'p':>1} {'arm':<10} {'state':<5} {'idle':>4} {'win':>4} {'gap med':>7} {'gap max':>7}"
              f" {'typeperf s':>10} {'suite s':>7} {'reader %':>8} {'780M %':>7} {'other %':>7}")
        for w in sorted(rm, key=lambda r: r["position"]):
            pw, cw, cs, g = w.get("power") or {}, w.get("cadence_window") or {}, w.get("cpu_s_in_window") or {}, w.get("gpu") or {}
            print(f"  {w['position']:>3} {w['pass']:>1} {w['arm']:<10} {w['state']:<5} {f(pw.get('idle_rows'), 'd'):>4}"
                  f" {f(pw.get('window_rows'), 'd'):>4} {f(cw.get('gap_median_s'), '.3f'):>7} {f(cw.get('gap_max_s'), '.3f'):>7}"
                  f" {f(cs.get('typeperf'), '.3f'):>10} {f(cs.get('suite'), '.3f'):>7} {f(w.get('process_cpu_pct'), '.1f'):>8}"
                  f" {f(g.get('gpu780_all'), '.1f'):>7} {f(g.get('other_adapters'), '.1f'):>7}")

        print("\n-- the NPU arms, per taken window: the sum of the 16 dispatch medians and the down sums' median,"
              " beside T, and 3b's gate/up concatenation timed once outside the loop (report-only, ms)")
        for npu in (*p3c.NPU_ARMS, "N-w4"):
            for p, w in windows_taken(rm, npu).items():
                if not w:
                    continue
                d = (w.get("reader") or {}).get("dispatch_ms_median") or {}
                pieces = sum(v for k, v in d.items() if k != "down_sum")
                print(f"  {npu} p{p}: T {f(w['T'])}; 16 dispatches {f(pieces)} + down sums {f(d.get('down_sum'))};"
                      f" 3b's concatenation {f((w.get('reader') or {}).get('concat_3b_ms'))}")
    print(f"\nCHECK the net E recomputed from the logged powers and window edges against the logged E: max"
          f" |ratio - 1| = {worst_net:.1e} (the window edges are logged to 3 decimals)")
    print("POSTHOC_JSON " + json.dumps({str(M): v for M, v in summary.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
