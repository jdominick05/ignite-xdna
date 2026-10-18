# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""F2-0: the ISA gate for F2's epilogue (integerized 8-bit scales) on AIE2, compile only (the F2 plan v2).

Compiles kernels/f2_epilogue/f2_epilogue.cc four ways with the Peano command IRON runs, disassembles each object with
Peano's llvm-objdump, and reads each case's hardware loop with U6-0's reader (tools/hybrid_u6_0.py, committed at
a513ac0; imported and pinned by its LF sha256, never edited). On AIE2 the bundle count of a zero-overhead hardware
loop body is its cycle count (tools/aie_disasm.py:7-12). E_core is F2_CORE's loop minus CONTROL's, in cycles per 32-lane
block-tile; FLUSH is F2_FLUSH's loop, in cycles per tile and superblock; E_F2 = E_core + FLUSH / 64 decides. No NPU,
no hardware context, no xclbin: nothing runs on a core.

    python tools/hybrid_f2_0.py selftest   # U6-0's fixture selftest, then F2's checks on synthetic text; no compiler
    python tools/hybrid_f2_0.py run        # the pins, the selftest, four compiles, E_core, FLUSH, E_F2 and the tier

The logged run is ./scripts/hybrid-stack.sh f2-0. The F2 plan (scratch/llm/hybrid_f2_plan_draft.md, git-ignored), the
source and the U6-0 tool are pinned by their LF sha256 below; any difference is a STOP.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aie_disasm  # noqa: E402
import hybrid_u6_0 as u60  # noqa: E402

U60_TOOL = ROOT / "tools/hybrid_u6_0.py"
U60_TOOL_SHA = "6f47d8616b364502a3da9739bee524f99c27f2fea7a12afe86987b0f56915e46"   # committed at a513ac0 (LF)
PLAN_F2_SHA = "b98787f451c11fbc9e847a4cb973b8f12b4988e3dce7729435d1fd34334a07c5"   # the F2 plan v2, final (LF)
PLAN_F2 = ROOT / "scratch/llm/hybrid_f2_plan_draft.md"                              # git-ignored
SOURCE = ROOT / "kernels/f2_epilogue/f2_epilogue.cc"
SOURCE_SHA = "76af35b24916f74c31474c32ea6afca5211f9cd8b3f535c6940a8cc1f52cd4ad"     # the source's LF sha256
U60_LOG = "results/llm/hybrid_u6_0_desktop2_20260925.log"                            # CONTROL 7, E (2-term) 68

# (case, defines, function, role, loop rule). CONTROL and F2_CORE decide E_core; F2_FLUSH decides through E_F2.
CASES = [
    ("CONTROL", ["-DF2_CONTROL"], "f2_block", "BASELINE", "two_macs"),
    ("F2_CORE", ["-DF2_CORE"], "f2_block", "DECIDING", "one_loop"),
    ("F2_FLUSH", ["-DF2_FLUSH"], "f2_flush", "DECIDING (through E_F2)", "one_loop"),
    ("F2_CORE_I32", ["-DF2_CORE_I32"], "f2_block", "REPORT-ONLY", "one_loop"),
]
LANES = 32                  # one 4 x 8 output tile per iteration
KILL_LINE = 24              # the gate's ruling (2026-09-25), in the plan's form; the plan, section 0
ROW_BLOCKS = 64             # S = ROW at the smallest block count (K = 2,048); the plan, section 2
S_SET = (("ROW", ROW_BLOCKS), ("16", 16), ("8", 8))
U60_CONTROL = 7             # U6-0's CONTROL loop bundles (U60_LOG); a different count here is a NOTE
U60_E2 = 68                 # U6-0's 2-term E (U60_LOG), for F2-P3
# The U6 plan's section 3 model and the rivals (the plan, section 0; BENCHMARKS at 554989e; 3c post-hoc log).
MAC_MS = 18.02              # the model's MACs per layer at M = 2,048 (DERIVED)
MS_PER_CYCLE = 6.55         # one cycle of E per layer (DERIVED)
NW4_MS = 83.91              # N-w4, the same int8 x int4 GEMM with no epilogue (MEASURED)
POWER_W = 32.78             # N-w4's with-idle power, the plan's assumption E2 (MEASURED)
M_TOKENS = 2048
D_NB16_MJ = 2.8569          # D-nb16's with-idle mJ per prompt token per layer (MEASURED; post-hoc:24)
N_BF16_MJ = 2.7417          # N-bf16's (MEASURED; post-hoc:25), F3's floor
BREAK_EVEN_MS = D_NB16_MJ * M_TOKENS / POWER_W   # 178.49 ms

LOAD_RE = re.compile(r"^(lda|ldb|vlda|vldb)\b")
STORE_RE = re.compile(r"^(st|vst)\b")
VCONV_RE = re.compile(r"^vconv\b")

EXACT = {"i_max": 32 * 127 * 8, "p_max": 255 * 127, "int16_max": 2 ** 15 - 1,
         "product_max": 32 * 127 * 8 * 255 * 127, "row_blocks_max": 10240 // 32}
TIERS = ("E_core > 24: F2-0 KILL (no placement of the flush saves it; F2-E and F2-A are not proposed); "
         "E_core <= 24 < E_F2: F2-0 STOP (where the flush runs, the core or the host, comes to the gate; a host flush "
         "is priced, its CPU time and the int64 output bytes); E_F2 <= 24: F2-0 PASS (necessary, not sufficient)")
E_CONTAINS = ("E_core = F2_CORE's loop bundles minus CONTROL's. CONTROL holds the A and B loads, the two integer MACs "
              "and the store of i as int32 (U6-0's CONTROL). So E_core holds the qx and qw loads, the qx broadcasts, "
              "qw's widening and repeat, P's multiply and srs, i's srs to int16, the acc64 load and store of I (the "
              "L1 round-trip), the two acc64 MACs and any spills. FLUSH is F2_FLUSH's whole loop: the acc64 load, "
              "fp32(I), the Sx and Dw loads and broadcasts, the 3-term splits, P, the y terms and y's load and store")
CONSERVATIVE = ("I is round-tripped through L1 every block (U6-0's form); qx and qw are loaded, broadcast and widened "
                "per tile, where a real kernel can widen once per row and column; FLUSH / 64 is S = ROW at the "
                "smallest block count")
PREDICTIONS = {
    "F2-P1": "every case compiles",
    "F2-P2": "F2_CORE's loop holds no vconv and no float MAC: no float work runs per block",
    "F2-P3": f"E_core < {U60_E2}, U6-0's 2-term E",
    "F2-P4": f"E_F2 <= {KILL_LINE}",
}

say = u60.say


def stop(msg: str) -> None:
    print(f"STOP: {msg}", flush=True)
    sys.exit(2)


# ---------------------------------------------------------------- the arithmetic of the outcome

def tier(e_core, e_f2) -> str:
    """The pre-registered outcome (the plan, section 2). e_f2 is None when F2_FLUSH is VOID."""
    if e_core > KILL_LINE:
        return "KILL"
    if e_f2 is None or e_f2 > KILL_LINE:
        return "STOP"
    return "PASS"


def allowed_s(e_core, flush) -> list:
    """The superblock sizes the budget allows: E_core + FLUSH / S <= 24."""
    return [name for name, s in S_SET if e_core + flush / s <= KILL_LINE]


def mj(ms: float) -> float:
    """With-idle mJ per prompt token per layer at N-w4's power (the plan's assumption E2)."""
    return ms * POWER_W / M_TOKENS


def energy(e_f2: float, control: int) -> dict:
    plan_ms = MAC_MS + MS_PER_CYCLE * e_f2
    serial_ms = NW4_MS + MS_PER_CYCLE * e_f2
    loop_ms = MS_PER_CYCLE * (control + e_f2)
    return {"e_f2": e_f2,
            "plan_form": {"ms": round(plan_ms, 2), "mj_with_idle": round(mj(plan_ms), 4)},
            "serial_form": {"ms": round(serial_ms, 2), "mj_with_idle": round(mj(serial_ms), 4)},
            "loop_form": {"ms": round(loop_ms, 2), "mj_with_idle": round(mj(loop_ms), 4)},
            "d_nb16_mj": D_NB16_MJ, "n_bf16_mj": N_BF16_MJ, "power_w": POWER_W,
            "basis": "compute only, at N-w4's power (the plan's assumption E2); not a silicon measurement"}


def lines_for(control: int) -> dict:
    """The kill line and its two sensitivities, as the largest whole E under the break-even (DERIVED)."""
    return {"break_even_ms": round(BREAK_EVEN_MS, 2),
            "plan_form": {"rule": "18.02 + 6.55 x E < 178.5",
                          "max_e": math.floor((BREAK_EVEN_MS - MAC_MS) / MS_PER_CYCLE),
                          "role": "DECIDING, fixed as E_F2 <= 24"},
            "loop_form": {"rule": f"6.55 x ({control} + E) < 178.5",
                          "max_e": math.floor(BREAK_EVEN_MS / MS_PER_CYCLE - control), "role": "sensitivity"},
            "serial_form": {"rule": "83.91 + 6.55 x E < 178.5",
                            "max_e": math.floor((BREAK_EVEN_MS - NW4_MS) / MS_PER_CYCLE), "role": "sensitivity"}}


# ---------------------------------------------------------------- reading the loops

def ldst(rec: dict) -> dict:
    """Loads and stores in the loop E reads, from its slot census (sp-relative ones are also in stack_refs)."""
    mn = (rec.get("slot_census") or {}).get("mnemonics") or {}
    loads = sum(n for h, n in mn.items() if LOAD_RE.match(h))
    stores = sum(n for h, n in mn.items() if STORE_RE.match(h))
    return {"loads": loads, "stores": stores,
            "vector_loads": sum(n for h, n in mn.items() if re.match(r"^(vlda|vldb)\b", h)),
            "vector_stores": sum(n for h, n in mn.items() if re.match(r"^vst\b", h))}


def min_macs(fn: str, rule: str) -> int:
    """f2_block's cases read under one_loop must still hold the mmul's two integer MACs; the flush holds none."""
    return 2 if fn == "f2_block" and rule == "one_loop" else 0


def read_case(sections, fn: str, rule: str, need: int) -> dict:
    """U6-0's reader, with F2's rule: a loop with fewer than `need` integer MACs is VOID."""
    rec = u60.read_function(sections, fn, rule)
    why = u60.validity(rec)
    i = rec.get("loop_index")
    lp = rec["loops"][i] if i is not None else {}
    if why is None and lp.get("int_macs", 0) < need:
        why = f"the loop holds {lp.get('int_macs')} integer MACs, fewer than the mmul's {need}"
    rec["void"] = why
    return rec


def no_float_work(rec: dict) -> bool:
    mn = (rec.get("slot_census") or {}).get("mnemonics") or {}
    i = rec.get("loop_index")
    lp = rec["loops"][i] if i is not None else {}
    return lp.get("float_macs", 1) == 0 and not any(VCONV_RE.match(h) for h in mn)


# ---------------------------------------------------------------- the source

def source_text(text: str | None = None) -> str:
    return SOURCE.read_text(encoding="utf-8") if text is None else text


def flush_order(text: str | None = None) -> dict:
    """The P_FIRST / P_TERM / Y_TERM calls inside the F2_FLUSH branch, in source order."""
    out = {"first": [], "P": [], "Y": []}
    inside = False
    for ln in source_text(text).splitlines():
        s = ln.strip()
        if s == "#if defined(F2_FLUSH)":
            inside = True
        elif inside and s == "#else":
            break
        elif inside:
            for kind, a, b in u60.TERM_RE.findall(s):
                out["first" if kind == "P_FIRST" else ("P" if kind == "P_TERM" else "Y")].append((int(a), int(b)))
    return {"P": out["first"] + out["P"], "Y": out["Y"], "n_first": len(out["first"])}


def source_lines(text: str | None = None) -> dict:
    """Where the branches, set_rounding and the two tile loops sit (1-based)."""
    lines = source_text(text).splitlines()
    find = lambda pat: [i + 1 for i, ln in enumerate(lines) if re.search(pat, ln)]  # noqa: E731
    return {"flush_branch": find(r"^#if defined\(F2_FLUSH\)"), "else": find(r"^#else$"),
            "set_rounding": find(r"aie::set_rounding\("), "flush_loop": find(r"for \(int32_t t = 0; t < ntiles"),
            "f2_flush": find(r"void f2_flush\("), "f2_block": find(r"void f2_block\(")}


def loop_head(text: str) -> list:
    """The stripped lines of a tile loop from 'MMUL C;' through 'B += 128;' (the loads and the two MACs)."""
    lines = [ln.strip() for ln in text.splitlines()]
    try:
        a = lines.index("MMUL C;")
        b = lines.index("B += 128;", a)
    except ValueError:
        return []
    return lines[a:b + 1]


# ---------------------------------------------------------------- synthetic text for F2's own rules

def _fn(name: str, rows: list) -> str:
    out = [f"Disassembly of section .text.{name}:", "", f"00000000 <{name}>:"]
    for addr, ops, label in rows:
        if label:
            out += ["", f"{addr:08x} <{label}>:"]
        out.append(f"{addr:8x}:      \t" + ";\t\t".join(ops))
    return "\n".join(out) + "\n"


SYN_F2 = "\n".join([
    # one loop, the mmul's two MACs plus P's multiply and two acc64 MACs: five integer MACs, no float work
    _fn("syn_f2core", [(0x0, ["add.nc\tlc, r0, #0x0"], None),
                       (0x10, ["vlda\twl0, [p0], #0x40", "vmul\tcm0, x0, x2, r5"], ".LBB1_1"),
                       (0x16, ["vmac\tcm0, cm0, x1, x3, r5"], None),
                       (0x1a, ["lda\tr1, [p2], #0x4", "vmul\tcm1, x4, x5, r6"], None),
                       (0x20, ["vlda\twl2, [p3], #0x40", "vmac\tcm2, cm2, x6, x7, r7"], None),
                       (0x26, ["vmac\tcm3, cm3, x8, x9, r7"], None),
                       (0x30, ["vst\twl2, [p3], #0x40", "vmov\tx10, x11"], ".L_LEnd1"),
                       (0x36, ["ret\tlr"], None)]),
    # the same with a vconv in the loop: float work, which F2-P2 must see
    _fn("syn_f2conv", [(0x0, ["add.nc\tlc, r0, #0x0"], None),
                       (0x10, ["vlda\twl0, [p0], #0x40", "vmul\tcm0, x0, x2, r5"], ".LBB2_1"),
                       (0x16, ["vmac\tcm0, cm0, x1, x3, r5"], None),
                       (0x20, ["vconv.fp32.bf16\tbml0, wl1"], ".L_LEnd2"),
                       (0x26, ["ret\tlr"], None)]),
    # one loop with a single integer MAC: an f2_block case below the mmul's two is VOID
    _fn("syn_f2one", [(0x0, ["add.nc\tlc, r0, #0x0"], None),
                      (0x10, ["vlda\twl0, [p0], #0x40", "vmul\tcm0, x0, x2, r5"], ".LBB3_1"),
                      (0x20, ["vst\twl0, [p1], #0x40"], ".L_LEnd3"),
                      (0x26, ["ret\tlr"], None)]),
])


def selftest() -> bool:
    results = []

    def check(name, ok, **kw):
        results.append(bool(ok))
        say("SELFTEST_JSON", {"check": name, "ok": bool(ok), **kw})

    u60_sha = u60.lf_sha(U60_TOOL)
    check("the U6-0 tool is the one committed at a513ac0 (its reader is imported)", u60_sha == U60_TOOL_SHA,
          lf_sha=u60_sha)
    print("U6-0's fixture selftest follows (its SELFTEST_JSON lines and its verdict):", flush=True)
    check("U6-0's fixture selftest passes (the reader, the MAC and vector-op classes, the three fixture parts)",
          u60.selftest())

    # F2's arithmetic (the plan, section 1): exactness in int16, int32 and a 64-bit accumulator.
    e = EXACT
    check("i and P are exact in int16 (|i| <= 32,512, |P| <= 32,385)",
          e["i_max"] == 32512 and e["p_max"] == 32385 and max(e["i_max"], e["p_max"]) <= e["int16_max"], **e)
    check("one product fits int32 (< 2^30); two blocks fit int32, four do not; 320 blocks fit 39 bits",
          e["product_max"] < 2 ** 30 and 2 * e["product_max"] < 2 ** 31 and 4 * e["product_max"] >= 2 ** 31
          and e["row_blocks_max"] * e["product_max"] < 2 ** 39, product_max=e["product_max"])

    # The outcome's arithmetic (the plan, sections 0 and 2), on synthetic numbers.
    check("the three tiers: KILL above 24 on E_core alone; STOP between; PASS at or under 24",
          tier(25, 25.5) == "KILL" and tier(25, None) == "KILL" and tier(20, 20 + 400 / 64) == "STOP"
          and tier(20, None) == "STOP" and tier(20, 20 + 100 / 64) == "PASS" and tier(24, 24.0) == "PASS"
          and tier(24, 24.02) == "STOP")
    check("the allowed S set: E_core + FLUSH / S <= 24", allowed_s(20, 100) == ["ROW"]
          and allowed_s(10, 100) == ["ROW", "16", "8"] and allowed_s(20, 64) == ["ROW", "16"]
          and allowed_s(25, 0) == [], examples={"20,100": allowed_s(20, 100), "10,100": allowed_s(10, 100),
                                                "20,64": allowed_s(20, 64)})
    lf = lines_for(U60_CONTROL)
    check("the lines: the plan's form 24, the loop form 20 (CONTROL 7), the serial form 14; break-even 178.49 ms",
          lf["plan_form"]["max_e"] == 24 and lf["loop_form"]["max_e"] == 20 and lf["serial_form"]["max_e"] == 14
          and abs(BREAK_EVEN_MS - 178.49) < 0.01, lines=lf)
    en = energy(24.0, U60_CONTROL)
    check("the energy print at E = 24: 175.22 ms and 2.8045 mJ in the plan's form, 241.11 ms and 3.8592 in the "
          "serial form", en["plan_form"] == {"ms": 175.22, "mj_with_idle": 2.8045}
          and en["serial_form"] == {"ms": 241.11, "mj_with_idle": 3.8592}, energy=en)

    # The source: the CONTROL branch is U6-0's, the flush's order is U6-E's, set_rounding sits in the flush only.
    ours, theirs = loop_head(source_text()), loop_head(u60.SOURCE.read_text(encoding="utf-8"))
    check("the tile loop's loads and two MACs are U6-0's, line for line", ours and ours == theirs, lines=ours)
    u6e_sha = u60.lf_sha(u60.U6E_TOOL)
    want = u60.u6e_order()[3]["P"] if u6e_sha == u60.U6E_TOOL_SHA else None
    got = flush_order()
    check("the flush's P and Y terms both follow U6-E's frozen ORDER[3]['P']",
          want is not None and got["n_first"] == 1 and got["P"] == [tuple(x) for x in want]
          and got["Y"] == [tuple(x) for x in want], flush=got, u6e=want)
    text = source_text()
    swapped = text.replace("Y_TERM(2, 2);", "@@").replace("Y_TERM(1, 3);", "Y_TERM(2, 2);").replace("@@",
                                                                                                    "Y_TERM(1, 3);")
    check("a copy with two Y terms swapped fails the order check",
          swapped != text and flush_order(swapped)["Y"] != [tuple(x) for x in (want or [])])
    w = source_lines()
    check("set_rounding sits once, inside the F2_FLUSH branch, before f2_flush's tile loop; f2_block has none",
          len(w["set_rounding"]) == 1 and len(w["flush_branch"]) == 1 and w["else"]
          and w["flush_branch"][0] < w["f2_flush"][0] < w["set_rounding"][0] < w["flush_loop"][0] < w["else"][0]
          < w["f2_block"][0], lines=w)

    # The loop reader on synthetic text: U6-0's fixture (loads and stores) and F2's own functions.
    sec = aie_disasm.parse(u60.fixture_parts().get("synthetic", ""))
    got = {fn: ldst(u60.read_function(sec, fn, rule)) for fn, rule in
           (("syn_full", "two_macs"), ("syn_ctrl", "two_macs"), ("syn_tofloat", "one_loop"))}
    check("loads and stores: syn_full 1 and 2 (one sp-relative), syn_ctrl 1 and 1, syn_tofloat 2 and 1",
          [(got[f]["loads"], got[f]["stores"]) for f in ("syn_full", "syn_ctrl", "syn_tofloat")]
          == [(1, 2), (1, 1), (2, 1)], counts=got)
    sec = aie_disasm.parse(SYN_F2)
    core = read_case(sec, "syn_f2core", "one_loop", 2)
    li = core.get("loop_index")
    lc = core["loops"][li] if li is not None else {}
    check("syn_f2core, one loop: read under F2's rule with 5 integer MACs, 6 bundles, 3 loads and 1 store",
          core["void"] is None and lc.get("int_macs") == 5 and lc.get("bundles") == 6
          and (ldst(core)["loads"], ldst(core)["stores"]) == (3, 1), loop=lc, ldst=ldst(core))
    two = u60.read_function(sec, "syn_f2core", "two_macs")
    check("syn_f2core under U6-0's two-MAC rule is a STOP (why F2_CORE reads the one loop instead)",
          u60.validity(two) is not None and "exactly 2" in (two.get("reason") or ""), reason=two.get("reason"))
    check("F2-P2's reading: syn_f2core has no float work, syn_f2conv has a vconv",
          no_float_work(core) and not no_float_work(read_case(sec, "syn_f2conv", "one_loop", 2)))
    one = read_case(sec, "syn_f2one", "one_loop", 2)
    check("the run's minimum: 2 integer MACs for f2_block under one_loop, 0 for the flush and for U6-0's rule",
          [min_macs(c[2], c[4]) for c in CASES] == [0, 2, 0, 2],
          minimums={c[0]: min_macs(c[2], c[4]) for c in CASES})
    check("an f2_block loop with fewer than the mmul's 2 integer MACs is VOID",
          one["void"] is not None and "fewer than the mmul's 2" in one["void"], reason=one["void"])

    ok = all(results)
    print("F2-0 SELFTEST OK" if ok else f"F2-0 SELFTEST FAILED ({results.count(False)} of {len(results)})", flush=True)
    return ok


# ---------------------------------------------------------------- the run

def compile_obj(defines: list, obj: str):
    cmd = [str(u60.CLANG), str(SOURCE), "-c", "-o", obj, f"-I{u60.INCLUDE}", *u60.IRON_FLAGS, *defines]
    p = subprocess.run(cmd, capture_output=True, text=True)
    err = (p.stderr or "").replace(str(Path.home()), "~").strip()
    return p.returncode == 0 and os.path.exists(obj), err[-4000:]


def protocol(where: dict) -> dict:
    return {"stage": "hybrid F2-0", "scope": "compile only: no NPU, no hardware context, no xclbin",
            "plan_f2_sha": PLAN_F2_SHA, "source": "kernels/f2_epilogue/f2_epilogue.cc", "source_sha": SOURCE_SHA,
            "reader": {"tool": "tools/hybrid_u6_0.py", "lf_sha256": U60_TOOL_SHA,
                       "use": "imported: its disassembly reader, loop statistics, validity, fixture selftest, "
                              "toolchain pin, flags and sizes; never edited"},
            "cases": [{"case": c, "defines": d, "function": f, "role": r, "loop_rule": lr}
                      for c, d, f, r, lr in CASES],
            "loop_rule": "CONTROL: U6-0's rule (the one loop with exactly 2 integer MACs). F2_CORE and F2_CORE_I32: "
                         "the one hardware loop in f2_block, which must hold at least the mmul's 2 integer MACs (P's "
                         "multiply and the acc64 MACs are integer MACs too). F2_FLUSH: the one hardware loop in "
                         "f2_flush. No loop, or more than one, is a STOP for a deciding case and a VOID otherwise",
            "e_definition": "cycles per 32-lane block-tile, DERIVED from bundles: a zero-overhead hardware loop "
                            "issues one bundle per cycle (tools/aie_disasm.py:7-12)",
            "e_contains": E_CONTAINS, "conservative": CONSERVATIVE, "excludes": u60.EXCLUDES,
            "deciding": "E_F2 = E_core + FLUSH / 64 (S = ROW at the smallest block count)", "tiers": TIERS,
            "kill_line": KILL_LINE, "s_set": dict(S_SET), "lines": lines_for(U60_CONTROL),
            "model": {"mac_ms": MAC_MS, "ms_per_cycle": MS_PER_CYCLE, "nw4_ms": NW4_MS, "power_w": POWER_W,
                      "m": M_TOKENS, "d_nb16_mj": D_NB16_MJ, "n_bf16_mj": N_BF16_MJ,
                      "sources": "the F2 plan, section 0; docs/BENCHMARKS.md at 554989e; "
                                 "results/llm/llm_prefill3c_posthoc_desktop2_20260924.log:24,25,27"},
            "rounding": {"where": "aie::set_rounding(conv_even) once, in f2_flush before its tile loop; f2_block "
                                  "sets none: its srs to int16 (i and P) are at shift 0, exact, so neither the "
                                  "rounding nor the saturation mode changes a value", "source_lines": where},
            "lowering": "i: C.to_vector<int16>(0); P: aie::mul of int16 broadcasts (qx) and the widened, repeated qw, "
                        "then to_vector<int16>(0); I: aie::mac on accum<acc64, 16> halves; the L1 round-trip "
                        "through aie::vector_cast<acc64> / <int32>, bit for bit; fp32(I) in the flush: "
                        "aie::to_float<float> on the acc64 halves (the F2 plan, section 2a)",
            "u6_0_cross_check": {"log": U60_LOG, "control_bundles": U60_CONTROL,
                                 "rule": "this run's CONTROL should read the same; a difference is a NOTE, and "
                                         "E_core subtracts this run's CONTROL"},
            "flags": u60.IRON_FLAGS, "clang": u60.home(u60.CLANG),
            "objdump": [u60.home(u60.OBJDUMP), "-d", "--no-show-raw-insn"],
            "size": [u60.home(u60.LLVM_SIZE), "-A", "summing .text*"], "text_limit": u60.TEXT_LIMIT,
            "predictions": PREDICTIONS}


def run() -> int:
    print(f"F2-0 run {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    print(f"COMMIT {commit}", flush=True)
    say("TOOL_SHA_JSON", {"file": "tools/hybrid_f2_0.py", "lf_sha256": u60.lf_sha(Path(__file__))})
    src = u60.lf_sha(SOURCE)
    say("SOURCE_SHA_JSON", {"file": "kernels/f2_epilogue/f2_epilogue.cc", "pinned": SOURCE_SHA, "lf_sha256": src})
    plan = u60.lf_sha(PLAN_F2) if PLAN_F2.exists() else None
    say("PLAN_JSON", {"f2": {"pinned": PLAN_F2_SHA, "file_lf_sha": plan}})
    reader = u60.lf_sha(U60_TOOL)
    say("READER_JSON", {"file": "tools/hybrid_u6_0.py", "pinned": U60_TOOL_SHA, "lf_sha256": reader})
    tc = u60.toolchain() if u60.SITE.exists() else None
    say("TOOLCHAIN_JSON", {"pinned": u60.TOOLCHAIN_PIN, "read": tc})
    bad = [n for n, ok in (("the F2 plan", plan == PLAN_F2_SHA), ("the source", src == SOURCE_SHA),
                           ("the U6-0 reader", reader == U60_TOOL_SHA), ("the toolchain", tc == u60.TOOLCHAIN_PIN))
           if not ok]
    if bad:
        stop(f"pin mismatch: {', '.join(bad)}")
    for p in (u60.CLANG, u60.OBJDUMP, u60.LLVM_SIZE):
        if not p.exists():
            stop(f"{u60.home(p)} not found")
    if not selftest():
        stop("the selftest failed")
    where = source_lines()
    say("PROTOCOL_JSON", protocol(where))

    recs = {}
    with tempfile.TemporaryDirectory(prefix="f2_0_") as tmp:
        for case, defines, fn, role, rule in CASES:
            obj = os.path.join(tmp, case + ".o")
            ok, err = compile_obj(defines, obj)
            rec = {"case": case, "role": role, "defines": defines, "function": fn, "loop_rule": rule, "compiled": ok,
                   "error": err or None}
            if ok:
                sections = aie_disasm.parse(aie_disasm.disassemble(obj, str(u60.OBJDUMP)))
                fr = read_case(sections, fn, rule, min_macs(fn, rule))
                say("LOOPS_JSON", {"case": case, "function": fn, "loops": fr.get("loops", [])})
                i = fr.get("loop_index")
                lp = fr["loops"][i] if i is not None else {}
                if fr.get("body"):
                    print(f"LOOP_BODY {case} {lp.get('name')}: {len(fr['body'])} bundles", flush=True)
                    for ln in fr["body"]:
                        print(f"    {ln}", flush=True)
                rec.update({"loop_found": i is not None, "loop": lp.get("name"), "loop_bundles": lp.get("bundles"),
                            "int_macs_in_loop": lp.get("int_macs"), "float_macs_in_loop": lp.get("float_macs"),
                            "vector_ops_in_loop": lp.get("vector_ops"), "stack_refs_in_loop": lp.get("stack_refs"),
                            "loads_stores_in_loop": ldst(fr) if i is not None else None,
                            "no_float_work": no_float_work(fr) if i is not None else None,
                            "function_stack_refs": fr.get("function_stack_refs"), "frame_bytes": fr.get("frame_bytes"),
                            "pipelined": fr.get("pipelined"), "outside": fr.get("outside"),
                            "cr_writes": fr.get("cr_writes"), "slot_census": fr.get("slot_census"),
                            "text_bytes": u60.text_bytes(obj), "void": fr["void"]})
            else:
                rec.update({"loop_found": False, "void": "did not compile"})
            say("CASE_JSON", rec)
            recs[case] = rec
            if case in ("CONTROL", "F2_CORE") and rec["void"]:
                stop(f"{case}: {rec['void']} (back to the gate for a fix, not a kill)"
                     + (f"; error: {err}" if not ok else ""))
            if ok and fn == "f2_flush" and not rec.get("cr_writes"):
                at = ", ".join(map(str, where["set_rounding"])) or "?"
                print(f"NOTE: no control-register write located in {case}, though the source calls set_rounding at "
                      f"line {at}", flush=True)

    control = recs["CONTROL"]["loop_bundles"]
    e_core = recs["F2_CORE"]["loop_bundles"] - control
    fl = recs["F2_FLUSH"]
    flush = None if fl["void"] else fl["loop_bundles"]
    e_f2 = None if flush is None else e_core + flush / ROW_BLOCKS
    i32 = recs["F2_CORE_I32"]
    e_i32 = None if i32["void"] else i32["loop_bundles"] - control
    outcome = tier(e_core, e_f2)
    allowed = allowed_s(e_core, flush) if flush is not None else None
    say("OUTPUT_JSON", {
        "control_bundles": control, "E_core": e_core, "FLUSH": flush, "E_F2": e_f2, "tier": outcome,
        "allowed_s": allowed, "E_core_i32": e_i32, "kill_line": KILL_LINE, "lines": lines_for(control),
        "pipelined": {c: r.get("pipelined") for c, r in recs.items()},
        "loads_stores": {c: r.get("loads_stores_in_loop") for c, r in recs.items()},
        "stack_refs_in_loop": {c: r.get("stack_refs_in_loop") for c, r in recs.items()},
        "voids": {c: r["void"] for c, r in recs.items() if r["void"]},
        "text_bytes": {c: r.get("text_bytes") for c, r in recs.items()}, "text_limit": u60.TEXT_LIMIT})
    if e_f2 is not None:
        say("ENERGY_JSON", energy(e_f2, control))

    def score(ok):
        return "VOID" if ok is None else ("HOLDS" if ok else "FAILS")

    say("PRED_JSON", {
        "F2-P1": score(all(r["compiled"] for r in recs.values())),
        "F2-P2": score(recs["F2_CORE"].get("no_float_work")),
        "F2-P3": score(e_core < U60_E2),
        "F2-P4": score(None if e_f2 is None else e_f2 <= KILL_LINE)})
    say("MEM_JSON", {"measured": False, "what": "L1 bytes per block-tile, DERIVED from the source's pointer strides; "
                                                "not process memory",
                     "f2_block": {"a": 128, "b": 128, "qx": 4, "qw_record": 16, "I_in": 256, "I_out": 256},
                     "control": {"a": 128, "b": 128, "y_out": 128},
                     "f2_flush_per_tile": {"I_in": 256, "sx": 16, "dw": 32, "y_in": 128, "y_out": 128}})
    if control != U60_CONTROL:
        print(f"NOTE: CONTROL reads {control} bundles, U6-0's read {U60_CONTROL}; E_core subtracts this run's",
              flush=True)
    pc, pf = recs["CONTROL"]["pipelined"], recs["F2_CORE"]["pipelined"]
    if pc != pf:
        print(f"NOTE: CONTROL reads PIPELINED: {pc} and F2_CORE {pf}, so E_core subtracts an initiation interval from "
              f"a body length (or the reverse)", flush=True)
    for c in ("F2_CORE", "F2_FLUSH"):
        if recs[c].get("stack_refs_in_loop"):
            print(f"NOTE: {c}'s loop holds {recs[c]['stack_refs_in_loop']} stack references (a spill); they are "
                  f"counted", flush=True)

    print(f"F2-0 E_core: {e_core} cycles per 32-lane block-tile (DERIVED from bundles, static)", flush=True)
    print(f"F2-0 FLUSH: {flush if flush is not None else 'VOID (' + str(fl['void']) + ')'} cycles per tile and "
          f"superblock", flush=True)
    print(f"F2-0 E_F2: {'%.2f' % e_f2 if e_f2 is not None else 'UNKNOWN'} (E_core + FLUSH / {ROW_BLOCKS})", flush=True)
    print(f"F2-0 ALLOWED S: {allowed if allowed is not None else 'UNKNOWN'}", flush=True)
    if e_f2 is not None:
        en = energy(e_f2, control)
        print(f"F2-0 ENERGY (compute only, at N-w4's power): plan form {en['plan_form']['mj_with_idle']} mJ, serial "
              f"form {en['serial_form']['mj_with_idle']} mJ, with idle per prompt token per layer; D-nb16 "
              f"{D_NB16_MJ}, N-bf16 {N_BF16_MJ}", flush=True)
    shown = f"{e_i32} cycles per 32-lane block-tile" if e_i32 is not None else f"VOID ({i32['void']})"
    print(f"F2-0 E_core (i kept as int32, REPORT-ONLY): {shown}", flush=True)
    words = {"KILL": "E_core > 24; F2-E and F2-A are not proposed",
             "STOP": "E_core <= 24 < E_F2 (or FLUSH VOID); where the flush runs comes to the gate",
             "PASS": "E_F2 <= 24; necessary, not sufficient"}
    print(f"F2-0 {outcome}: {words[outcome]}", flush=True)
    return 0


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or [""])[0]
    if mode == "selftest":
        print(f"F2-0 selftest {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
        say("TOOL_SHA_JSON", {"file": "tools/hybrid_f2_0.py", "lf_sha256": u60.lf_sha(Path(__file__))})
        return 0 if selftest() else 1
    if mode == "run":
        return run()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
