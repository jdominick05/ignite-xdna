# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""U6-0: the ISA gate for U6's epilogue on AIE2, compile only (the U6-0 plan v2).

Compiles kernels/u6_epilogue/u6_epilogue.cc five ways with the Peano command IRON runs, disassembles each object
with Peano's llvm-objdump, and reads the tile loop's bundles. On AIE2 the bundle count of a zero-overhead hardware
loop body is its cycle count (tools/aie_disasm.py:7-12), so E, the epilogue's cost in cycles per 32-lane block-tile,
is FULL3's loop minus CONTROL's. No NPU, no hardware context, no xclbin: nothing runs on a core.

    python tools/hybrid_u6_0.py selftest   # the counters on the committed fixture and the source's ORDER; no compiler
    python tools/hybrid_u6_0.py run        # the pins, the selftest, the five compiles, E and the report-only readings

The logged run is ./scripts/hybrid-stack.sh u6-0. The plan (scratch/llm/hybrid_u6_0_plan_draft.md, git-ignored) and
the U6 plan are pinned by their LF sha256 below, and so is the source; any difference is a STOP.
"""

from __future__ import annotations

import ast
import collections
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aie_disasm  # noqa: E402

PLAN_U6_SHA = "830eb8b9cd5d3512c1ffb20b5a1cee57e67280f7dd1d8333ffae6ea478e9508d"   # U6 v2 + E1/E2, as approved (LF)
PLAN_U6 = ROOT / "scratch/llm/hybrid_u6_plan_draft.md"                              # git-ignored
PLAN_U60_SHA = "352a30d818b93fc55f3da6169bd2a396c9e709cc3f5f25e20f15c570c61c29f5"   # U6-0 plan v2 (LF)
PLAN_U60 = ROOT / "scratch/llm/hybrid_u6_0_plan_draft.md"                           # git-ignored
SOURCE = ROOT / "kernels/u6_epilogue/u6_epilogue.cc"
SOURCE_SHA = "7130b078a9ef51530b8c485183900939c79ad23b861a7bcf4bb6afcbe7498cc9"     # the source's LF sha256
FIXTURE = ROOT / "kernels/u6_epilogue/u6_0_disasm_fixture.txt"
U6E_TOOL = ROOT / "tools/hybrid_u6e.py"
U6E_TOOL_SHA = "b88d23d30c6c2ec3ffa55407dd4969957acf98e788fca47f1a703feec146e892"   # frozen at 2505530; its ORDER

TOOLCHAIN_PIN = "llvm_aie-22.0.0.2026090201+a36c62b9, mlir_aie-1.4.2"
SITE = Path.home() / "mlir-aie/ironenv/Lib/site-packages"
BIN = SITE / "llvm-aie/bin"
CLANG = BIN / "clang++.exe"
OBJDUMP = BIN / "llvm-objdump.exe"
LLVM_SIZE = BIN / "llvm-size.exe"
INCLUDE = SITE / "mlir_aie/include"
# kernels/w4a8_probe/static_probe.py:63-73, the Peano command IRON builds (mlir-aie v1.4.2) minus -MD/-MF. A copy, so
# this tool's frozen text carries it; the selftest checks it equals static_probe's.
IRON_FLAGS = [
    "-std=c++20",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
    "-O2",
    "-DNDEBUG",
    "-D__AIE_API_AIE_ADF_HPP__",
    "--target=aie2-none-unknown-elf",
]

# (case, defines, function, role). CONTROL and FULL3 decide E; a STOP in either ends the run.
CASES = [
    ("CONTROL", ["-DU6_CONTROL"], "u6_block", "BASELINE"),
    ("FULL3", ["-DU6_SPLIT=3"], "u6_block", "DECIDING"),
    ("FULL2", ["-DU6_SPLIT=2"], "u6_block", "REPORT-ONLY"),
    ("FULL3_I16", ["-DU6_SPLIT=3", "-DU6_I16"], "u6_block", "REPORT-ONLY"),
    ("TOFLOAT", ["-DU6_TOFLOAT"], "u6_tofloat", "REPORT-ONLY"),
]
DECIDING = ("CONTROL", "FULL3")
V1_CASES = ("CONTROL", "FULL3", "FULL2", "TOFLOAT")
LANES = 32                  # one 4 x 8 output tile per iteration
TEXT_LIMIT = 16384          # program memory per core, bytes (SPEC)
E_LINE = 12                 # the U6 plan's line, read in cycles here

INT_MACS = ("vmul", "vmac", "vmsc", "vnegmul", "vnegmac", "vnegmsc", "vaddmac", "vaddmsc", "vsubmac", "vsubmsc")
FLOAT_MACS = tuple(m + ".f" for m in INT_MACS)
LDST_RE = re.compile(r"^(st|lda|ldb|vst|vlda|vldb)\b")      # static_probe.py:101
SP_RE = re.compile(r"\[sp\b")                                # static_probe.py:97-100: sp-relative only
CR_WRITE_RE = re.compile(r"^\S+\s+(cr[A-Z][A-Za-z0-9]*)\b")  # a control register as the destination
TERM_RE = re.compile(r"\b(P_FIRST|P_TERM|Y_TERM)\((\d),\s*(\d)\)\s*;")

UNIT_CHANGE = ("the U6 plan's section 3 E counted vector-slot operations per 32-lane block-tile (an ESTIMATE, 12-16, "
               "INFERRED from SPEC rates); U6-0's E is cycles per 32-lane block-tile, DERIVED from bundles. The "
               "E >= 12 line is read in cycles. F2 is already triggered by U6-E's 3-term split (the user's "
               "decision), so the line decides nothing about F2; E feeds the 6.55 ms x E per layer time ESTIMATE only")
LOOP_RULE = ("every hardware loop in the case's function is printed with its bundles; E reads the ONE loop that holds "
             "exactly 2 integer MACs (the mmul's mul and mac); no such loop, or integer MACs in more than one loop, is "
             "a STOP for FULL3 or CONTROL and a VOID for FULL2 or FULL3_I16; TOFLOAT reads the one hardware loop in "
             "u6_tofloat (none, or more than one, is its VOID); there is no fallback to whole-function counts")
PIPELINED = ("with unrolling disabled, the body's bundles are the cycles per iteration whether or not the loop is "
             "modulo-scheduled (then they are the initiation interval). What the text can show is a pipelined "
             "prologue or epilogue: integer MACs, or vector stores not relative to sp, in the function outside the "
             "loop body. Either nonzero prints PIPELINED: SHOWN, else NOT SHOWN; a reading of the text (INFERRED), "
             "not a compiler report")
E_CONTAINS = ("E = FULL3's loop bundles minus CONTROL's. CONTROL holds the A and B loads, the two integer MACs and the "
              "store of i as int32 (to_vector<int32>, shift 0). So E holds the y load, the s_x and d_w loads and "
              "broadcasts, whatever FULL3 does to bring i into vector registers for the conversion, the int32-to-fp32 "
              "conversion, the h, s_x, d_w and P splits, P, the Y MACs and any spills: not the arithmetic alone")
CONSERVATIVE = ("s_x and d_w are loaded, broadcast and split per tile; the real kernel can split s_x once per row and "
                "d_w once per column and reuse the pieces, so for that part E is an upper bound")
EXCLUDES = ["bank conflicts (+1 cycle per iteration when two accesses share a 16 KB bank; SILICON.md:85-86), which a "
            "static count cannot see", "lock, stream or DMA waits (BENCHMARKS.md:1526-1529)", "code outside the loop"]
PREDICTIONS = {
    "U0-P1": "the four v1 cases compile (CONTROL, FULL3, FULL2, TOFLOAT); FULL3_I16 carries no prediction",
    "U0-P2": "FULL2's E is within the U6 plan's 12-16 (section 3, INFERRED, counted there in operations)",
    "U0-P3": "FULL3's E exceeds FULL2's",
    "U0-P4": "TOFLOAT's loop holds at least 4 vector ops per 16 lanes, so no single conversion instruction",
}


def say(tag: str, obj) -> None:
    print(f"{tag} {json.dumps(obj)}", flush=True)


def stop(msg: str) -> None:
    print(f"STOP: {msg}", flush=True)
    sys.exit(2)


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def home(p: Path) -> str:
    """A path under the profile, written from ~ so no log carries the profile name."""
    try:
        return "~/" + p.relative_to(Path.home()).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------- reading the disassembly

def head(field: str) -> str:
    return field.split()[0] if field.split() else field


def is_int_mac(h: str) -> bool:
    return h in INT_MACS


def is_float_mac(h: str) -> bool:
    return h in FLOAT_MACS


def is_vector_op(h: str) -> bool:
    """A live op starting with 'v', vector loads and stores (and their suffixed forms) excluded."""
    return h.startswith("v") and not re.match(r"^(vlda|vldb|vst)(\.|$)", h)


def is_stack_ref(field: str) -> bool:
    return bool(LDST_RE.match(head(field)) and SP_RE.search(field))


def section_for(sections, fn: str):
    for s in sections:
        if s.name == ".text." + fn or s.name == fn:
            return s
    return None


def loop_stats(lp) -> dict:
    heads = [head(f) for b in lp.bundles for f in b.live]
    fields = [f for b in lp.bundles for f in b.live]
    return {"name": lp.name, "start": f"0x{lp.start:x}", "end": f"0x{lp.end:x}", "bundles": lp.n_bundles,
            "int_macs": sum(map(is_int_mac, heads)), "float_macs": sum(map(is_float_mac, heads)),
            "vector_ops": sum(map(is_vector_op, heads)), "stack_refs": sum(map(is_stack_ref, fields)),
            "live_ops": len(heads)}


def pick_loop(stats: list, rule: str):
    """(index, None) for the loop E reads, or (None, reason)."""
    if rule == "one_loop":
        if len(stats) != 1:
            return None, f"{len(stats)} hardware loops, not exactly one"
        return 0, None
    with_macs = [i for i, s in enumerate(stats) if s["int_macs"]]
    if not with_macs:
        return None, "no hardware loop holds an integer MAC"
    if len(with_macs) > 1:
        return None, f"integer MACs in {len(with_macs)} loops ({[stats[i]['int_macs'] for i in with_macs]})"
    i = with_macs[0]
    if stats[i]["int_macs"] != 2:
        return None, f"no loop holds exactly 2 integer MACs (the one with MACs holds {stats[i]['int_macs']})"
    return i, None


def read_function(sections, fn: str, rule: str) -> dict:
    """Every hardware loop in fn, the loop E reads (or why none), and what sits outside it."""
    s = section_for(sections, fn)
    if s is None:
        return {"found": False, "reason": f"no section for {fn}"}
    stats = [loop_stats(lp) for lp in s.loops]
    i, reason = pick_loop(stats, rule)
    rec = {"found": True, "loops": stats, "loop_index": i, "reason": reason, "frame_bytes": s.frame_bytes,
           "function_stack_refs": sum(is_stack_ref(f) for b in s.bundles for f in b.live)}
    lp = s.loops[i] if i is not None else None
    cr = []
    for b in s.bundles:
        for f in b.live:
            m = CR_WRITE_RE.match(f)
            if m:
                inside = lp is not None and lp.start <= b.addr <= lp.end
                cr.append({"addr": f"0x{b.addr:x}", "op": " ".join(f.split()), "inside_loop": inside})
    rec["cr_writes"] = cr
    if lp is None:
        return rec
    out = {"int_macs_before": 0, "int_macs_after": 0, "vst_before": 0, "vst_after": 0}
    for b in s.bundles:
        if lp.start <= b.addr <= lp.end:
            continue
        side = "before" if b.addr < lp.start else "after"
        for f in b.live:
            if is_int_mac(head(f)):
                out[f"int_macs_{side}"] += 1
            if re.match(r"^vst(\.|$)", head(f)) and not SP_RE.search(f):
                out[f"vst_{side}"] += 1
    rec["outside"] = out
    rec["pipelined"] = "SHOWN" if any(out.values()) else "NOT SHOWN"
    census = collections.Counter(head(f) for b in lp.bundles for f in b.live)
    rec["slot_census"] = {"mnemonics": dict(sorted(census.items())),
                          "full_width_slots": aie_disasm.slot_histogram(lp.bundles),
                          "full_width_bundles": sum(b.is_full_width for b in lp.bundles),
                          "compressed_bundles": sum(not b.is_full_width for b in lp.bundles)}
    rec["body"] = [f"0x{b.addr:04x}  {' | '.join(' '.join(f.split()) for f in b.live) or '(all nop)'}"
                   for b in lp.bundles]
    return rec


def validity(rec: dict) -> str | None:
    """Why a function's reading cannot be used, or None."""
    if not rec.get("found"):
        return rec.get("reason")
    if rec["reason"]:
        return rec["reason"]
    inside = [c for c in rec["cr_writes"] if c["inside_loop"]]
    if inside:
        return f"a control-register write inside the loop: {inside}"
    return None


# ---------------------------------------------------------------- the fixture and the ORDER

def fixture_parts() -> dict:
    parts, name, buf = {}, None, []
    for raw in FIXTURE.read_text(encoding="utf-8").splitlines():
        if raw.startswith("#=== PART "):
            if name:
                parts[name] = "\n".join(buf)
            name, buf = raw.split()[2], []
        elif name:
            buf.append(raw)
    if name:
        parts[name] = "\n".join(buf)
    return parts


def u6e_order() -> dict:
    """ORDER from the frozen U6-E tool's text (ast; the tool is not imported)."""
    tree = ast.parse(U6E_TOOL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ORDER" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("no ORDER in the U6-E tool")


def source_order(text: str | None = None) -> dict:
    """The P_FIRST / P_TERM / Y_TERM calls of each U6_SPLIT branch, in source order."""
    lines = (SOURCE.read_text(encoding="utf-8") if text is None else text).splitlines()
    out, t = {}, None
    for ln in lines:
        s = ln.strip()
        if s == "#if U6_SPLIT == 3":
            t = 3
        elif s == "#else" and t == 3:
            t = 2
        elif s == "#endif" and t == 2:
            t = None
        elif t is not None:
            for kind, a, b in TERM_RE.findall(s):
                d = out.setdefault(t, {"P": [], "Y": [], "first": []})
                d["first" if kind == "P_FIRST" else ("P" if kind == "P_TERM" else "Y")].append((int(a), int(b)))
    return {t: {"P": d["first"] + d["P"], "Y": d["Y"], "n_first": len(d["first"])} for t, d in out.items()}


def source_lines() -> dict:
    """Where set_rounding and the tile loop sit in the source (1-based)."""
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    rnd = [i + 1 for i, ln in enumerate(lines) if "aie::set_rounding(" in ln]
    loop = [i + 1 for i, ln in enumerate(lines) if re.search(r"for \(int32_t t = 0; t < ntiles", ln)]
    return {"set_rounding": rnd, "tile_loop": loop}


def selftest() -> bool:
    results = []

    def check(name, ok, **kw):
        results.append(bool(ok))
        say("SELFTEST_JSON", {"check": name, "ok": bool(ok), **kw})

    import importlib.util
    spec = importlib.util.spec_from_file_location("static_probe", ROOT / "kernels/w4a8_probe/static_probe.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    check("the flags, clang and include equal static_probe.py's", IRON_FLAGS == sp.IRON_FLAGS
          and Path(sp.CLANG) == CLANG and Path(sp.INCLUDE) == INCLUDE, flags=IRON_FLAGS)

    u6e_sha = lf_sha(U6E_TOOL)
    check("the U6-E tool is the frozen one (its ORDER is read next)", u6e_sha == U6E_TOOL_SHA, lf_sha=u6e_sha)
    want = u6e_order()
    got = source_order()
    for t in (3, 2):
        g = got.get(t, {})
        ok = (g.get("n_first") == 1 and [tuple(x) for x in want[t]["P"]] == g.get("P")
              and [tuple(x) for x in want[t]["Y"]] == g.get("Y"))
        check(f"the source's {t}-term P_FIRST/P_TERM/Y_TERM order equals U6-E's ORDER[{t}]", ok,
              source=g, u6e=want[t])
    # The negative control: the same text with the 3-term branch's first two P_TERM calls swapped must not match.
    text = SOURCE.read_text(encoding="utf-8")
    swapped = (text.replace("P_TERM(2, 2);", "@@").replace("P_TERM(1, 3);", "P_TERM(2, 2);")
               .replace("@@", "P_TERM(1, 3);"))
    bad = source_order(swapped).get(3, {})
    check("a copy with two P terms swapped fails the ORDER check", swapped != text
          and bad.get("P") != [tuple(x) for x in want[3]["P"]], swapped_p=bad.get("P"))
    where = source_lines()
    check("set_rounding sits once in the source, before the tile loop",
          len(where["set_rounding"]) == 1 and len(where["tile_loop"]) == 1
          and where["set_rounding"][0] < where["tile_loop"][0], lines=where)

    heads = ["vmac", "vmul", "vmac.f", "vmul.f", "vnegmul", "vmsc", "vaddmac", "vmov", "vadd.f", "mov"]
    check("integer MACs are the bare family; .f MACs are apart",
          [h for h in heads if is_int_mac(h)] == ["vmac", "vmul", "vnegmul", "vmsc", "vaddmac"]
          and [h for h in heads if is_float_mac(h)] == ["vmac.f", "vmul.f"])
    vheads = ["vlda", "vldb", "vst", "vst.conv", "vlda.ups", "vshuffle", "vadd.f", "vmac.f", "vmov", "vsrs", "mov",
              "lda", "vldb.unpack"]
    check("vector ops exclude vector loads and stores",
          [h for h in vheads if is_vector_op(h)] == ["vshuffle", "vadd.f", "vmac.f", "vmov", "vsrs"])

    parts = fixture_parts()
    check("the fixture has its three parts", sorted(parts) == ["clock", "synthetic", "w4a8"], parts=sorted(parts))

    # clock_kernels.o: results/aie/aie2_isa_static.log:53-76 (loops of 2 and 9 bundles; 2.000 and 9.000 on silicon)
    sec = aie_disasm.parse(parts.get("clock", ""))
    s = section_for(sec, "clock_probe")
    loops = {lp.name: loop_stats(lp) for lp in s.loops} if s else {}
    check("clock_probe: four loops; .L_LEnd3 is 2 bundles and .L_LEnd2 is 9; no integer MAC",
          len(loops) == 4 and loops.get(".L_LEnd3", {}).get("bundles") == 2
          and loops.get(".L_LEnd2", {}).get("bundles") == 9 and all(v["int_macs"] == 0 for v in loops.values()),
          loops={k: v["bundles"] for k, v in loops.items()})

    # w4a8_native_default_64x256x64.o: results/aie/w4a8_probe_npu.log:257 (16 bundles, 8 vmacs, loop stack 0,
    # function stack 20, frame 64); BENCHMARKS.md:2117 (Static 16.0, the IRON object's own loop). The eight
    # integer MACs after the loop (0x290 to 0x316) are counted by hand from the fixture text.
    sec = aie_disasm.parse(parts.get("w4a8", ""))
    rec = read_function(sec, "mm_i8i4_native", "two_macs")
    mac_loops = [x for x in rec.get("loops", []) if x["int_macs"]]
    check("mm_i8i4_native: one MAC loop of 16 bundles and 8 integer MACs, loop stack 0, function stack 20, frame 64",
          len(mac_loops) == 1 and mac_loops[0]["bundles"] == 16 and mac_loops[0]["int_macs"] == 8
          and mac_loops[0]["stack_refs"] == 0 and rec.get("function_stack_refs") == 20
          and rec.get("frame_bytes") == 64, mac_loops=mac_loops, function_stack_refs=rec.get("function_stack_refs"),
          frame_bytes=rec.get("frame_bytes"))
    check("mm_i8i4_native: the rule finds no loop with exactly 2 integer MACs (a STOP)",
          validity(rec) is not None and "exactly 2" in (rec.get("reason") or ""), reason=rec.get("reason"))
    idx = next((i for i, x in enumerate(rec.get("loops", [])) if x["int_macs"]), None)
    s = section_for(sec, "mm_i8i4_native")
    before = after = None
    if s is not None and idx is not None:
        lp = s.loops[idx]
        after = sum(is_int_mac(head(f)) for b in s.bundles if b.addr > lp.end for f in b.live)
        before = sum(is_int_mac(head(f)) for b in s.bundles if b.addr < lp.start for f in b.live)
    check("mm_i8i4_native: 0 integer MACs before the loop and 8 after it", before == 0 and after == 8,
          before=before, after=after)

    sec = aie_disasm.parse(parts.get("synthetic", ""))
    full = read_function(sec, "syn_full", "two_macs")
    ctrl = read_function(sec, "syn_ctrl", "two_macs")
    lf = full["loops"][full["loop_index"]] if full.get("loop_index") is not None else {}
    lc = ctrl["loops"][ctrl["loop_index"]] if ctrl.get("loop_index") is not None else {}
    check("synthetic full: 6 bundles, 2 integer and 3 .f MACs, 1 stack ref",
          validity(full) is None and lf.get("bundles") == 6 and lf.get("int_macs") == 2
          and lf.get("float_macs") == 3 and lf.get("stack_refs") == 1, loop=lf)
    check("synthetic full: an integer MAC before the loop reads PIPELINED: SHOWN",
          full.get("pipelined") == "SHOWN" and full.get("outside", {}).get("int_macs_before") == 1,
          outside=full.get("outside"))
    check("synthetic full: the crRnd write before the loop is located, outside it",
          full.get("cr_writes") == [{"addr": "0x0", "op": "mov crRnd, #0xc", "inside_loop": False}],
          cr_writes=full.get("cr_writes"))
    check("synthetic control: 3 bundles, 2 integer MACs, PIPELINED: NOT SHOWN, no control-register write",
          validity(ctrl) is None and lc.get("bundles") == 3 and lc.get("int_macs") == 2
          and ctrl.get("pipelined") == "NOT SHOWN" and ctrl.get("cr_writes") == [], loop=lc)
    check("synthetic E = 6 - 3 = 3", lf.get("bundles", 0) - lc.get("bundles", 0) == 3)
    split = read_function(sec, "syn_split", "two_macs")
    check("synthetic split: integer MACs in two loops is a STOP",
          validity(split) is not None and "in 2 loops" in (split.get("reason") or ""), reason=split.get("reason"))
    crin = read_function(sec, "syn_crin", "two_macs")
    v = validity(crin)
    check("synthetic crin: a crRnd write inside the loop is a STOP", v is not None and "control-register" in v,
          reason=v)
    tof = read_function(sec, "syn_tofloat", "one_loop")
    lt = tof["loops"][0] if tof.get("loops") else {}
    check("synthetic tofloat: one loop of 7 bundles with 5 vector ops (loads and stores excluded)",
          validity(tof) is None and lt.get("bundles") == 7 and lt.get("vector_ops") == 5, loop=lt)

    ok = all(results)
    print("SELFTEST OK" if ok else f"SELFTEST FAILED ({results.count(False)} of {len(results)})", flush=True)
    return ok


# ---------------------------------------------------------------- the run

def toolchain() -> str:
    dists = sorted(d for d in os.listdir(SITE) if d.endswith(".dist-info")
                   and (d.startswith("mlir_aie") or d.startswith("llvm_aie")))
    return ", ".join(d[:-10] for d in dists)                  # isa_gate.py:265-267


def compile_obj(defines: list, obj: str):
    cmd = [str(CLANG), str(SOURCE), "-c", "-o", obj, f"-I{INCLUDE}", *IRON_FLAGS, *defines]
    p = subprocess.run(cmd, capture_output=True, text=True)
    err = (p.stderr or "").replace(str(Path.home()), "~").strip()
    return p.returncode == 0 and os.path.exists(obj), err[-4000:]


def text_bytes(obj: str) -> int:
    out = subprocess.run([str(LLVM_SIZE), "-A", obj], capture_output=True, text=True).stdout
    return sum(int(m.group(1)) for m in re.finditer(r"^\.text\S*\s+(\d+)", out, re.M))  # engine_epilogue_variants:113


def protocol(where: dict) -> dict:
    return {"stage": "hybrid U6-0", "scope": "compile only: no NPU, no hardware context, no xclbin",
            "plan_u6_sha": PLAN_U6_SHA, "plan_u60_sha": PLAN_U60_SHA, "source": "kernels/u6_epilogue/u6_epilogue.cc",
            "source_sha": SOURCE_SHA,
            "cases": [{"case": c, "defines": d, "function": f, "role": r} for c, d, f, r in CASES],
            "e_definition": "cycles per 32-lane block-tile, DERIVED from bundles: a zero-overhead hardware loop "
                            "issues one bundle per cycle (tools/aie_disasm.py:7-12; calibrated at "
                            "BENCHMARKS.md:1512-1519 and :2112-2118)",
            "e_contains": E_CONTAINS, "unit_change": UNIT_CHANGE, "loop_rule": LOOP_RULE,
            "int_macs": list(INT_MACS), "float_macs": "the same names with a .f suffix, counted apart",
            "pipelined_reading": PIPELINED,
            "rounding": {"where": "aie::set_rounding(conv_even) once, before the tile loop, never inside it",
                         "source_lines": where,
                         "check": "every control-register write (cr plus a capital) is located; one inside the E "
                                  "loop is a STOP for FULL3 or CONTROL and a VOID otherwise; any srs of i (to int32, "
                                  "or to int16 in FULL3_I16) is at shift 0, so crRnd does not change its value and "
                                  "matters for the bf16 conversions",
                         "printed_form": "mov crRnd, #0xc, as Peano printed set_rounding(conv_even) in a throwaway "
                                         "probe; a u6_block case with no located write prints a NOTE"},
            "to_float": "FULL3 and FULL2: aie::to_float<float>(C.to_vector<int32>()), the int32-vector overload's "
                        "32-bit branch (aie.hpp:7581-7605, elementary.hpp:66-79); FULL3_I16: C.to_vector<int16>(), "
                        "then its 16-bit branch (elementary.hpp:80-87); TOFLOAT: FULL's call alone; the accumulator "
                        "overload (aie.hpp:7615-7624) is not used",
            "broadcast": "s_x: four scalar loads, aie::broadcast<float, 8> of each, aie::concat of the four (lanes "
                         "r*8 + 0..7); d_w: aie::load_v<8>, then aie::concat(d, d, d, d)",
            "b_offset": "B holds (c - 8) as signed int4, repacked on the host (U6 plan section 1, exact), so the "
                        "epilogue has no -8 x sum(q) term",
            "conservative": CONSERVATIVE, "excludes": EXCLUDES, "static_count": "a lower bound on time on silicon",
            "order": {str(t): v for t, v in u6e_order().items()},
            "flags": IRON_FLAGS, "clang": home(CLANG), "objdump": [home(OBJDUMP), "-d", "--no-show-raw-insn"],
            "size": [home(LLVM_SIZE), "-A", "summing .text*"], "text_limit": TEXT_LIMIT,
            "no_fast_math": "the flags carry no fast-math option, so the fp32 add chain stays in the order written",
            "predictions": PREDICTIONS}


def run() -> int:
    print(f"U6-0 run {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    print(f"COMMIT {commit}", flush=True)
    say("TOOL_SHA_JSON", {"file": "tools/hybrid_u6_0.py", "lf_sha256": lf_sha(Path(__file__))})
    src = lf_sha(SOURCE)
    say("SOURCE_SHA_JSON", {"file": "kernels/u6_epilogue/u6_epilogue.cc", "pinned": SOURCE_SHA, "lf_sha256": src})
    p6 = lf_sha(PLAN_U6) if PLAN_U6.exists() else None
    p60 = lf_sha(PLAN_U60) if PLAN_U60.exists() else None
    say("PLAN_JSON", {"u6": {"pinned": PLAN_U6_SHA, "file_lf_sha": p6},
                      "u6_0": {"pinned": PLAN_U60_SHA, "file_lf_sha": p60}})
    tc = toolchain() if SITE.exists() else None
    say("TOOLCHAIN_JSON", {"pinned": TOOLCHAIN_PIN, "read": tc})
    bad = [n for n, ok in (("the U6 plan", p6 == PLAN_U6_SHA), ("the U6-0 plan", p60 == PLAN_U60_SHA and p60),
                           ("the source", src == SOURCE_SHA), ("the toolchain", tc == TOOLCHAIN_PIN)) if not ok]
    if bad:
        stop(f"pin mismatch: {', '.join(bad)}")
    for p in (CLANG, OBJDUMP, LLVM_SIZE):
        if not p.exists():
            stop(f"{home(p)} not found")
    if not selftest():
        stop("the fixture selftest failed")
    where = source_lines()
    say("PROTOCOL_JSON", protocol(where))

    recs = {}
    with tempfile.TemporaryDirectory(prefix="u6_0_") as tmp:
        for case, defines, fn, role in CASES:
            obj = os.path.join(tmp, case + ".o")
            ok, err = compile_obj(defines, obj)
            rec = {"case": case, "role": role, "defines": defines, "function": fn, "compiled": ok,
                   "error": err or None}
            if ok:
                sections = aie_disasm.parse(aie_disasm.disassemble(obj, str(OBJDUMP)))
                fr = read_function(sections, fn, "one_loop" if case == "TOFLOAT" else "two_macs")
                say("LOOPS_JSON", {"case": case, "function": fn, "loops": fr.get("loops", [])})
                why = validity(fr)
                i = fr.get("loop_index")
                lp = fr["loops"][i] if i is not None else {}
                if fr.get("body"):
                    print(f"LOOP_BODY {case} {lp.get('name')}: {len(fr['body'])} bundles", flush=True)
                    for ln in fr["body"]:
                        print(f"    {ln}", flush=True)
                rec.update({"loop_found": i is not None, "loop": lp.get("name"), "loop_bundles": lp.get("bundles"),
                            "int_macs_in_loop": lp.get("int_macs"), "float_macs_in_loop": lp.get("float_macs"),
                            "vector_ops_in_loop": lp.get("vector_ops"), "stack_refs_in_loop": lp.get("stack_refs"),
                            "function_stack_refs": fr.get("function_stack_refs"), "frame_bytes": fr.get("frame_bytes"),
                            "pipelined": fr.get("pipelined"), "outside": fr.get("outside"),
                            "cr_writes": fr.get("cr_writes"), "slot_census": fr.get("slot_census"),
                            "text_bytes": text_bytes(obj), "void": why})
            else:
                rec.update({"loop_found": False, "void": "did not compile"})
            say("CASE_JSON", rec)
            recs[case] = rec
            if ok and fn == "u6_block" and not rec.get("cr_writes"):
                at = ", ".join(map(str, where["set_rounding"])) or "?"
                print(f"NOTE: no control-register write located in {case}, though the source calls set_rounding at "
                      f"line {at}", flush=True)
            if case in DECIDING and rec["void"]:
                stop(f"{case}: {rec['void']}" + (f"; error: {err}" if not ok else ""))

    def e(case):
        r = recs.get(case, {})
        return None if r.get("void") else r["loop_bundles"] - recs["CONTROL"]["loop_bundles"]

    e3, e2, e3i = e("FULL3"), e("FULL2"), e("FULL3_I16")
    tof = recs["TOFLOAT"]
    per16 = None if tof.get("void") else tof["vector_ops_in_loop"] * 16 / LANES
    say("OUTPUT_JSON", {
        "E3": e3, "E2": e2, "E3_I16": e3i, "control_bundles": recs["CONTROL"]["loop_bundles"],
        "cycles_per_block_tile_3": recs["FULL3"]["loop_bundles"], "e3_ge_12": e3 >= E_LINE,
        "pipelined": {c: recs[c].get("pipelined") for c in recs},
        "full3_stack_refs_in_loop": recs["FULL3"]["stack_refs_in_loop"],
        "voids": {c: r["void"] for c, r in recs.items() if r["void"]},
        "text_bytes": {c: r.get("text_bytes") for c, r in recs.items()}, "text_limit": TEXT_LIMIT,
        "tofloat": {"vector_ops_in_loop": tof.get("vector_ops_in_loop"), "vector_ops_per_16_lanes": per16,
                    "mnemonics": (tof.get("slot_census") or {}).get("mnemonics")}})

    def score(ok):
        return "VOID" if ok is None else ("HOLDS" if ok else "FAILS")

    say("PRED_JSON", {
        "U0-P1": score(all(recs[c]["compiled"] for c in V1_CASES)),
        "U0-P2": score(None if e2 is None else 12 <= e2 <= 16),
        "U0-P3": score(None if e2 is None else e3 > e2),
        "U0-P4": score(None if per16 is None else per16 >= 4)})
    say("MEM_JSON", {"measured": False, "what": "L1 bytes per block-tile, DERIVED from the source's pointer strides; "
                                                "not process memory",
                     "a": 128, "b": 128, "sx": 16, "dw": 32, "y_in": 128, "y_out": 128})
    pc, pf = recs["CONTROL"]["pipelined"], recs["FULL3"]["pipelined"]
    if pc != pf:
        print(f"NOTE: CONTROL reads PIPELINED: {pc} and FULL3 {pf}, so E subtracts an initiation interval from a "
              f"body length (or the reverse)", flush=True)
    if recs["FULL3"]["stack_refs_in_loop"]:
        print(f"NOTE: FULL3's loop holds {recs['FULL3']['stack_refs_in_loop']} stack references (a spill); they are "
              f"counted in E", flush=True)
    print(f"U6-0 E: {e3} cycles per 32-lane block-tile (3-term, round-trip; DERIVED from bundles)", flush=True)
    print(f"U6-0 E >= 12: {'YES' if e3 >= E_LINE else 'NO'}", flush=True)
    for name, val, case in (("2-term", e2, "FULL2"), ("3-term, int16 conversion", e3i, "FULL3_I16")):
        shown = f"{val} cycles per 32-lane block-tile" if val is not None else f"VOID ({recs[case]['void']})"
        print(f"U6-0 E ({name}, REPORT-ONLY): {shown}", flush=True)
    return 0


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or [""])[0]
    if mode == "selftest":
        print(f"U6-0 selftest {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
        say("TOOL_SHA_JSON", {"file": "tools/hybrid_u6_0.py", "lf_sha256": lf_sha(Path(__file__))})
        return 0 if selftest() else 1
    if mode == "run":
        return run()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
