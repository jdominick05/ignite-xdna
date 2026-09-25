# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""F2-0b: F2's flush with no srs on an acc64, and the acc64 confirmation, compile only (the F2-0b addendum).

F2-0 (tools/hybrid_f2_0.py) read E_core = 22 and a VOID flush: aie::to_float on an acc64 sets the saturation and
rounding modes per srs, inside the loop. F2-0b does three things, each read with U6-0's reader (tools/hybrid_u6_0.py)
and F2-0's helpers (tools/hybrid_f2_0.py), both imported and pinned by their LF sha256, never edited:
  1. recompiles F2_CORE from the pinned kernels/f2_epilogue/f2_epilogue.cc; it must be F2-0's loop, bit for bit;
  2. finds the loop's MACs on I (the MACs whose accumulator the loop stores, all four quarters) and decodes their
     config register's pre-loop value against llvm-aie's aiev2_compute_control (aiev2/aiev2_vmult.h:15-25): amode,
     bits 1-2, is the accumulator width (0 acc32, 1 acc64, 2 accfloat, re-tabulated from the header);
  3. compiles kernels/f2_epilogue/f2_flush_b.cc's two flush forms, FLUSH_B3 and FLUSH_B2, and reads each one's loop.
Per form: E_F2 = 22 + FLUSH / 64; PASS at E_F2 <= 24, MARGINAL at <= 24.49, else FAIL; both FAIL is KILL. No NPU, no
hardware context, no xclbin: nothing runs on a core.

    python tools/hybrid_f2_0b.py selftest   # F2-0's and U6-0's selftests, then F2-0b's checks; no compiler
    python tools/hybrid_f2_0b.py run        # the pins, the selftest, the header, three compiles, the decode, the tiers

The logged run is ./scripts/hybrid-stack.sh f2-0b. The F2 plan and the F2-0b addendum (git-ignored), both sources,
both imported tools, F2-0's log and the header are pinned by their LF sha256 below; any difference is a STOP.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aie_disasm  # noqa: E402
import hybrid_f2_0 as f20  # noqa: E402
import hybrid_u6_0 as u60  # noqa: E402

U60_TOOL = ROOT / "tools/hybrid_u6_0.py"
U60_TOOL_SHA = "6f47d8616b364502a3da9739bee524f99c27f2fea7a12afe86987b0f56915e46"   # committed at a513ac0 (LF)
F20_TOOL = ROOT / "tools/hybrid_f2_0.py"
F20_TOOL_SHA = "1d6960e974de2aeab69a9140cf12886363b0c2763ddd0be3ae6fbac5e456a109"   # committed at a47e631 (LF)
F20_LOG = ROOT / "results/llm/hybrid_f2_0_desktop2_20260925.log"
F20_LOG_SHA = "2bf5e1afc60d9ed9e1194335fbe7e95ce0eeee250c5250b88ac872b2811b5140"    # committed at ca2f0de (LF)
CORE_SOURCE = ROOT / "kernels/f2_epilogue/f2_epilogue.cc"
CORE_SOURCE_SHA = "76af35b24916f74c31474c32ea6afca5211f9cd8b3f535c6940a8cc1f52cd4ad"   # F2-0's source (LF)
SOURCE = ROOT / "kernels/f2_epilogue/f2_flush_b.cc"
SOURCE_SHA = "caa62ce055f6e1a3e8de350349ac76bef3f8d9dd521416c9fb8db0c64a2e18d4"        # F2-0b's source (LF)
PLAN_F2 = ROOT / "scratch/llm/hybrid_f2_plan_draft.md"                                  # git-ignored
PLAN_F2_SHA = "b98787f451c11fbc9e847a4cb973b8f12b4988e3dce7729435d1fd34334a07c5"       # the F2 plan v2, final (LF)
ADDENDUM = ROOT / "scratch/llm/hybrid_f2_0b_addendum.md"                                # git-ignored
ADDENDUM_SHA = "81dfe8da781aedd8513945c95b556739545326f80cda2b66fab0ba2504b3c26c"      # after the gate's fix (LF)
HEADER_REL = "llvm-aie/lib/clang/22/include/aiev2/aiev2_vmult.h"                       # under the pinned site-packages
HEADER = u60.SITE / HEADER_REL
HEADER_SHA = "685cf91053f20e7d051550e65cb718836a299115b2bf982178c5eac8a2fe5c6e"        # llvm_aie 22.0.0.2026090201

# (case, source, defines, function, role, loop rule, the minimum integer MACs in the loop)
CASES = [
    ("F2_CORE", CORE_SOURCE, ["-DF2_CORE"], "f2_block", "PRECONDITION (a re-read of F2-0's)", "one_loop", 2),
    ("FLUSH_B3", SOURCE, ["-DF2_FLUSH_B3"], "f2_flush_b", "DECIDING (per form)", "one_loop", 0),
    ("FLUSH_B2", SOURCE, ["-DF2_FLUSH_B2"], "f2_flush_b", "DECIDING (per form)", "one_loop", 0),
]
FORMS = ("FLUSH_B3", "FLUSH_B2")
E_CORE = 22                 # F2-0's E_core (F2-0 log:271, 274), a pinned input; the log is re-read and must agree
F20_CORE_BUNDLES = 29       # F2-0's F2_CORE loop (F2-0 log:62)
F20_CORE_TEXT = 496         # F2-0's F2_CORE .text bytes (F2-0 log:92)
F20_CONTROL = 7             # F2-0's CONTROL (F2-0 log:271), for the loop form's energy line
ROW_BLOCKS = f20.ROW_BLOCKS  # 64: S = ROW at the smallest block count (the plan, section 2)
PASS_LINE = 24              # E_F2 <= 24: PASS, necessary and not sufficient (the plan, section 0)
MARGINAL_LINE = 24.49       # the plan's break-even E, (178.49 - 18.02) / 6.5536 = 24.486 (the gate's 24.49)
S_SMALL = (("16", 16), ("8", 8))  # allowed while E_CORE + FLUSH / S <= 24: FLUSH <= 32 and <= 16
F20_FLUSH_TAIL = (67, 94)   # f2_epilogue.cc's flush after fp32(I); FLUSH_B3's branch copies it verbatim
F2_I_MAX = 320 * 1052901120  # |I| <= 320 blocks x 32,512 x 32,385 = 336,928,358,400 < 2^39 (the plan, section 1)
AMODE_FAMILY = {"0": ("acc32", {"acc32"}), "1": ("acc64", {"acc64", "cacc64"}),
                "2": ("accfloat", {"accfloat", "caccfloat"})}
CONF_EXPECTED = 0x35A       # sgn_x 1, sgn_y 1, amode 1, bmode 3, variant 2 (aiev2_vmult.h:16189-16195; DERIVED)

# The source's piece map, which the emulation below mirrors (the addendum, section 1).
SHUFFLE_LINES = [
    "aie::vector<uint16, 32> la = aie::filter_even(wa).template cast_to<uint16>();",
    "aie::vector<uint16, 32> lb = aie::filter_even(wb).template cast_to<uint16>();",
    "aie::vector<int16, 32> ha = aie::filter_odd(wa).template cast_to<int16>();",
    "aie::vector<int16, 32> hb = aie::filter_odd(wb).template cast_to<int16>();",
    "aie::vector<uint16, 32> H0 = aie::concat(aie::filter_even(la), aie::filter_even(lb));",
    "aie::vector<uint16, 32> H1 = aie::concat(aie::filter_odd(la), aie::filter_odd(lb));",
    "aie::vector<int16, 32> H2 = aie::concat(aie::filter_even(ha), aie::filter_even(hb));",
]
TO_FLOAT_RE = re.compile(r"aie::to_float<float>\((H[012]), (-?\d+)\)")

TIERS = ("per flush form, E_F2 = 22 + FLUSH / 64: FLUSH <= 128 (E_F2 <= 24) PASS, necessary and not sufficient; "
         "128 < FLUSH <= 159 (E_F2 <= 24.48) MARGINAL: F2-E may run on the gate's go, and the energy verdict is the "
         "user's, with the serial form (E <= 14) in hand; FLUSH > 159 or VOID: that form FAILS; both forms FAIL: "
         "F2-0b KILL, the core flush is killed and F2 ends. Allowed S per form: ROW with a PASS or MARGINAL; 16 at "
         "FLUSH <= 32; 8 at FLUSH <= 16")
PRECONDITION = ("the acc64 confirmation, a precondition of any PASS or MARGINAL: F2_CORE recompiled from the pinned "
                "source must read 29 bundles with F2-0's loop text and .text size (anything else is a STOP); the "
                "MACs on I (the loop's integer MACs whose accumulator the loop stores as all four quarters) must be "
                "exactly two sharing one config register; that register's writes must be pre-loop immediate moves "
                "of one value (else 'cannot be decoded', a STOP); its amode (aiev2_vmult.h:15-25, bits 1-2) reads "
                "1 (acc64): CONFIRMED; 0 (acc32): 'ACC64 VOID: E_core = 22 is void; F2_CORE returns to design', "
                "each form's FLUSH and allowed S REPORT-ONLY, no PASS or MARGINAL, KILL still printed if both FAIL; "
                "anything else a STOP. The loop's other integer MACs' config registers are decoded the same way and "
                "must read amode 0 where decodable, else a STOP (the decoder is not trusted)")
PREDICTIONS = {
    "F2b-P1": "both flush cases compile, each to one hardware loop with no control-register write inside it",
    "F2b-P2": "neither flush loop holds a vsrs",
    "F2b-P3": "FLUSH_B2 < FLUSH_B3",
    "F2b-P4": "the F2_CORE recompile is bit-identical to F2-0's loop",
    "F2b-P5": "the config register of the MACs on I decodes to amode 1",
}

say = u60.say


def stop(msg: str) -> None:
    print(f"STOP: {msg}", flush=True)
    sys.exit(2)


# ---------------------------------------------------------------- the tiers

def form_tier(flush):
    """(tier, E_F2) for one flush form; flush None is VOID."""
    if flush is None:
        return "FAIL", None
    e = E_CORE + flush / ROW_BLOCKS
    if e <= PASS_LINE:
        return "PASS", e
    if e <= MARGINAL_LINE:
        return "MARGINAL", e
    return "FAIL", e


def allowed_s(flush, t: str) -> list:
    """ROW with a PASS or MARGINAL; 16 and 8 while E_CORE + FLUSH / S <= 24."""
    if t not in ("PASS", "MARGINAL"):
        return []
    return ["ROW"] + [name for name, s in S_SMALL if E_CORE + flush / s <= PASS_LINE]


def verdict(acc: str, tiers: dict) -> list:
    """The headline readings, in order. acc is CONFIRMED or VOID (a STOP never reaches here)."""
    out = []
    kill = all(t == "FAIL" for t in tiers.values())
    if acc == "VOID":
        out.append("ACC64 VOID")
    if kill:
        out.append("KILL")
    if acc == "CONFIRMED" and not kill:
        out.append("PASS" if "PASS" in tiers.values() else "MARGINAL")
    return out


# ---------------------------------------------------------------- the header: the control word and amode

CTRL_DEF_RE = re.compile(r"^static inline int aiev2_compute_control\(")
SHIFT_RE = re.compile(r"\(unsigned\)(\w+)\s*<<\s*(\d+)")
INTR_RE = re.compile(r"INTRINSIC\((\w+)\)\s*(\w+)\s*\(([^)]*)\)\s*\{(.*?)\n\}", re.S)
CTL_ARGS_RE = re.compile(r"aiev2_compute_control\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),")
OVERLOAD_HEAD = "mac_elem_16_2_conf(v32int16 a, int sgn_x, v32int16 b, int sgn_y, v16acc64 acc1,"


def header_formula(text: str) -> dict | None:
    """aiev2_compute_control's line span, each argument's shift, and the line that shifts amode."""
    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if CTRL_DEF_RE.match(ln)), None)
    if first is None:
        return None
    last = next((j for j in range(first, len(lines)) if lines[j].strip() == "}"), None)
    if last is None:
        return None
    body = "\n".join(lines[first:last + 1])
    return {"lines": [first + 1, last + 1], "shifts": {n: int(s) for n, s in SHIFT_RE.findall(body)},
            "amode_line": next((i + 1 for i in range(first, last + 1) if "(unsigned)amode <<" in lines[i]), None)}


def widths(shifts: dict) -> dict:
    """Each field's width, to the next field's shift (the last field to bit 32)."""
    order = sorted(shifts.items(), key=lambda kv: kv[1])
    return {n: (order[i + 1][1] if i + 1 < len(order) else 32) - s for i, (n, s) in enumerate(order)}


def decode(word: int, shifts: dict) -> dict:
    w = widths(shifts)
    return {n: (word >> s) & ((1 << w[n]) - 1) for n, s in sorted(shifts.items(), key=lambda kv: kv[1])}


def compute_control(shifts: dict, **fields) -> int:
    """The header's formula, rebuilt from the parsed shifts (for the selftest's known words)."""
    return sum(int(fields.get(n, 0)) << s for n, s in shifts.items())


def amode_partition(text: str) -> dict:
    """Every INTRINSIC whose body builds a control word: the amode it passes against the accumulator it returns."""
    table, n = {}, 0
    for m in INTR_RE.finditer(text):
        ret, _name, _args, body = m.groups()
        c = CTL_ARGS_RE.search(body)
        if not c:
            continue
        acc = re.sub(r"^v\d+", "", ret)
        row = table.setdefault(c.group(3).strip(), {})
        row[acc] = row.get(acc, 0) + 1
        n += 1
    ok = bool(table) and all(a in AMODE_FAMILY and set(r) <= AMODE_FAMILY[a][1] for a, r in table.items())
    return {"intrinsics": n, "table": {a: dict(sorted(r.items())) for a, r in sorted(table.items())},
            "partition": ok}


def overload(text: str) -> dict | None:
    """The v32int16 mac_elem_16_2_conf that aie::mac on an acc64 calls (aie_api detail/aie2/mul_acc64.hpp:130-139)."""
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith(OVERLOAD_HEAD)), None)
    if i is None:
        return None
    j = next((k for k in range(i, len(lines)) if lines[k].strip() == "}"), i)
    c = CTL_ARGS_RE.search("\n".join(lines[i:j + 1]))
    # 1-based: the INTRINSIC(v16acc64) line (the one before the definition, so index i) through the closing brace
    return {"lines": [i, j + 1], "args": [g.strip() for g in c.groups()] if c else None,
            "amode": c.group(3).strip() if c else None}


def header_record(text: str) -> dict:
    f = header_formula(text)
    rec = {"formula": f, "widths": widths(f["shifts"]) if f else None, "amode": amode_partition(text),
           "overload": overload(text)}
    why = []
    if not f or f["shifts"].get("amode") != 1 or widths(f["shifts"]).get("amode") != 2:
        why.append("the formula does not put amode at bits 1-2")
    if not rec["amode"]["partition"]:
        why.append("amode does not partition the accumulator types")
    if not rec["overload"] or rec["overload"]["amode"] != "1":
        why.append("the v32int16 mac_elem_16_2_conf does not pass amode 1")
    rec["why"] = why or None
    return rec


# ---------------------------------------------------------------- the disassembly: the MACs on I and their setup

QUARTER_RE = re.compile(r"^am(ll|lh|hl|hh)(\d+)$")
CM_RE = re.compile(r"^cm(\d+)$")
REG_RE = re.compile(r"^r\d+$")
IMM_MOVE_RE = re.compile(r"^(mov|movx|mova|movxm|movs) (r\d+), #(-?(?:0x[0-9a-fA-F]+|\d+))$")
NOT_A_WRITE_RE = re.compile(r"^(st|vst|j|ret|nop)")


def norm(f: str) -> str:
    return " ".join(f.split())


def operands(f: str) -> list:
    parts = norm(f).split(" ", 1)
    return [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []


def macs_on_i(lp) -> dict:
    """The loop's integer MACs, split into those whose destination cmN the loop stores as all four quarters (the
    MACs on I) and the rest."""
    stored = {}
    for b in lp.bundles:
        for f in b.live:
            if re.match(r"^vst(\.|$)", u60.head(f)):
                ops = operands(f)
                m = QUARTER_RE.match(ops[0]) if ops else None
                if m:
                    stored.setdefault(m.group(2), set()).add(m.group(1))
    full = {n for n, q in stored.items() if q == {"ll", "lh", "hl", "hh"}}
    on_i, others = [], []
    for b in lp.bundles:
        for f in b.live:
            if not u60.is_int_mac(u60.head(f)):
                continue
            ops = operands(f)
            m = CM_RE.match(ops[0]) if ops else None
            rec = {"addr": f"0x{b.addr:x}", "op": norm(f), "dest": ops[0] if ops else None,
                   "config": ops[-1] if ops else None}
            (on_i if m and m.group(1) in full else others).append(rec)
    return {"on_i": on_i, "others": others, "stored_quarters": {n: sorted(q) for n, q in sorted(stored.items())}}


def writes_to(section, reg: str, lp) -> list:
    """Every op in the function whose first operand is reg (stores, jumps and returns excluded)."""
    out = []
    for b in section.bundles:
        for f in b.live:
            n = norm(f)
            if NOT_A_WRITE_RE.match(u60.head(n)):
                continue
            ops = operands(n)
            if not ops or ops[0] != reg:
                continue
            where = "inside" if lp.start <= b.addr <= lp.end else ("before" if b.addr < lp.start else "after")
            m = IMM_MOVE_RE.match(n)
            out.append({"addr": f"0x{b.addr:x}", "op": n, "where": where,
                        "imm": (int(m.group(3), 0) & 0xFFFFFFFF) if m else None})
    return out


def setup_value(writes: list):
    """(value, None), or (None, why): decodable iff no write is inside the loop, at least one is before it, and every
    write before it is an immediate move, all of one value."""
    inside = [w for w in writes if w["where"] == "inside"]
    before = [w for w in writes if w["where"] == "before"]
    if inside:
        return None, f"written inside the loop: {[w['op'] for w in inside]}"
    if not before:
        return None, "no write before the loop"
    if any(w["imm"] is None for w in before):
        return None, f"a write before the loop is not an immediate move: {[w['op'] for w in before]}"
    vals = sorted({w["imm"] for w in before})
    if len(vals) != 1:
        return None, f"the writes before the loop carry {len(vals)} values: {[hex(v) for v in vals]}"
    return vals[0], None


def decode_register(section, lp, reg: str, shifts: dict, table: dict) -> dict:
    writes = writes_to(section, reg, lp)
    value, why = setup_value(writes)
    rec = {"register": reg, "writes": writes, "value": None if value is None else hex(value), "why": why}
    if value is not None:
        fields = decode(value, shifts)
        a = str(fields.get("amode"))
        rec.update({"fields": fields, "amode": fields.get("amode"),
                    "accumulator": AMODE_FAMILY[a][0] if a in AMODE_FAMILY and a in table else None})
    return rec


def acc64_reading(section, lp, shifts: dict, table: dict) -> dict:
    """The precondition's decode: the MACs on I, their config register and its value; the controls."""
    m = macs_on_i(lp)
    rec = {"macs": m, "reading": None, "why": None, "controls": []}
    regs = sorted({x["config"] for x in m["on_i"]})
    if len(m["on_i"]) != 2 or len(regs) != 1 or not REG_RE.match(regs[0] or ""):
        rec.update({"reading": "STOP", "why": f"not exactly two MACs on I sharing one config register: "
                                              f"{len(m['on_i'])} MACs, registers {regs}"})
        return rec
    r = decode_register(section, lp, regs[0], shifts, table)
    rec["config"] = r
    if r["why"]:
        rec.update({"reading": "STOP", "why": f"cannot be decoded: {r['why']}"})
    elif r["accumulator"] == "acc64":
        rec["reading"] = "CONFIRMED"
    elif r["accumulator"] == "acc32":
        rec["reading"] = "VOID"
    else:
        rec.update({"reading": "STOP", "why": f"amode {r['amode']} is neither acc64 nor acc32"})
    for reg in sorted({x["config"] for x in m["others"]} - set(regs)):
        c = decode_register(section, lp, reg, shifts, table) if REG_RE.match(reg or "") else \
            {"register": reg, "why": "not a scalar register"}
        rec["controls"].append(c)
        if rec["reading"] != "STOP" and not c.get("why") and c.get("accumulator") != "acc32":
            rec.update({"reading": "STOP", "why": f"a control config ({reg}) decodes to {c.get('accumulator')}, not "
                                                  f"acc32: the decoder is not trusted"})
    return rec


# ---------------------------------------------------------------- F2-0's log and the sources

def f20_log() -> dict:
    """F2-0's F2_CORE loop (name, bundles, body), its .text, E_core and CONTROL, as the log printed them."""
    lines = F20_LOG.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("LOOP_BODY F2_CORE "))
    m = re.match(r"^LOOP_BODY F2_CORE (\S+): (\d+) bundles$", lines[i])
    n = int(m.group(2))
    body = [ln[4:] for ln in lines[i + 1:i + 1 + n]]
    case = next(json.loads(ln[len("CASE_JSON "):]) for ln in lines
                if ln.startswith("CASE_JSON ") and '"case": "F2_CORE"' in ln)
    out = next(json.loads(ln[len("OUTPUT_JSON "):]) for ln in lines if ln.startswith("OUTPUT_JSON "))
    return {"loop": m.group(1), "bundles": n, "body": body, "body_lines": [i + 1, i + 1 + n],
            "text_bytes": case["text_bytes"], "e_core": out["E_core"], "control": out["control_bundles"]}


def source_text(text: str | None = None) -> str:
    """f2_flush_b.cc with LF line ends (the swapped-order copies below match on LF)."""
    return (SOURCE.read_text(encoding="utf-8") if text is None else text).replace("\r\n", "\n")


def branch_lines(text: str, which: str) -> list:
    """The stripped lines of the FLUSH_B3 branch ("B3") or the FLUSH_B2 branch ("B2")."""
    out, state = [], None
    for ln in text.splitlines():
        s = ln.strip()
        if s == "#if defined(F2_FLUSH_B3)":
            state = "B3"
            continue
        if s == "#else" and state == "B3":
            state = "B2"
            continue
        if s == "#endif" and state == "B2":
            break
        if state == which:
            out.append(s)
    return out


def branch_order(text: str, which: str) -> dict:
    out = {"first": [], "P": [], "Y": []}
    for s in branch_lines(text, which):
        for kind, a, b in u60.TERM_RE.findall(s):
            out["first" if kind == "P_FIRST" else ("P" if kind == "P_TERM" else "Y")].append((int(a), int(b)))
    return {"P": out["first"] + out["P"], "Y": out["Y"], "n_first": len(out["first"])}


def f20_tail() -> list:
    lines = CORE_SOURCE.read_text(encoding="utf-8").splitlines()
    a, b = F20_FLUSH_TAIL
    return [ln.strip() for ln in lines[a - 1:b]]


def code_lines(text: str) -> list:
    return [re.sub(r"//.*", "", ln).strip() for ln in text.splitlines()]


def source_facts(text: str | None = None) -> dict:
    t = source_text(text)
    code = code_lines(t)
    lines = t.splitlines()
    find = lambda pat: [i + 1 for i, ln in enumerate(lines) if re.search(pat, ln)]  # noqa: E731
    return {"acc64_in_code": sum("acc64" in c for c in code),
            "vector_cast_in_code": sum("vector_cast" in c for c in code),
            "srs_in_code": sum(bool(re.search(r"\bsrs", c)) for c in code),
            "to_float": [(h, int(s)) for c in code for h, s in TO_FLOAT_RE.findall(c)],
            "shuffles_present": all(ln in [x.strip() for x in lines] for ln in SHUFFLE_LINES),
            "set_rounding": find(r"aie::set_rounding\("), "loop": find(r"for \(int32_t t = 0; t < ntiles"),
            "f2_flush_b": find(r"void f2_flush_b\(")}


# ---------------------------------------------------------------- the conversion, emulated

def fp32_of_i(values, swap: bool = False) -> np.ndarray:
    """fp32(I) as f2_flush_b.cc takes it: the 16-bit pieces of each lane's little-endian words (h0, h1 of the low
    word, h2 of the high word, signed), each exact in fp32, summed high to low in float32. swap reads the two words
    the other way round: a negative control, which shows only that the emulation depends on the order."""
    w = np.ascontiguousarray(np.asarray(values, dtype="<i8")).view("<u4").reshape(-1, 2)
    if swap:
        w = w[:, ::-1]
    h = np.ascontiguousarray(w).view("<u2").reshape(-1, 4)
    h0 = h[:, 0].astype(np.float32)
    h1 = h[:, 1].astype(np.float32)
    h2 = h[:, 2].copy().view(np.int16).astype(np.float32)
    f2 = h2 * np.float32(2.0 ** 32)
    f1 = h1 * np.float32(2.0 ** 16)
    return (f2 + f1) + h0


def rne(values) -> np.ndarray:
    """RNE(I) in fp32: I is exact in float64 below 2^53, and float64 to float32 rounds to nearest even."""
    return np.asarray(values, dtype=np.int64).astype(np.float64).astype(np.float32)


def conv_values() -> list:
    v = {0, 1, -1, F2_I_MAX, -F2_I_MAX, 2 ** 39 - 1, -(2 ** 39 - 1)}
    for k in range(39):
        for x in (2 ** k, 2 ** k - 1, 2 ** k + 1):
            v.update({x, -x})
    for k in range(24, 39):                  # fp32's ulp on [2^k, 2^(k+1)) is 2^(k-23): ties and their neighbours
        u = 2 ** (k - 23)
        for x in (2 ** k + u // 2, 2 ** k + 3 * u // 2, 2 ** k + u // 2 + 1, 2 ** k + u // 2 - 1):
            v.update({x, -x})
    return sorted(v)


# ---------------------------------------------------------------- synthetic text for the decoder

def _syn(name: str, pre: list, loop_ops: list, stores: list) -> str:
    rows = [(0x0, ["add.nc\tlc, r0, #0x0"], None)]
    rows += [(0x2 + 2 * i, [op], None) for i, op in enumerate(pre)]
    addr = 0x20
    ops = loop_ops + stores
    for i, op in enumerate(ops):
        label = ".LBB9_1" if i == 0 else (".L_LEnd9" if i == len(ops) - 1 else None)
        rows.append((addr, [op], label))
        addr += 0x8
    rows.append((addr, ["ret\tlr"], None))
    return f20._fn(name, rows)


QUARTERS = ["vst\tam%s%d, [p4, #0x%x]" % (q, n, 0x20 * i + (0x80 if n == 6 else 0))
            for n in (4, 6) for i, q in enumerate(("ll", "lh", "hl", "hh"))]
MACS = ["vmul\tcm0, x1, x5, r1", "vmac\tcm2, cm0, x9, x2, r1", "vmul\tcm1, x11, x4, r3",
        "vmac\tcm4, cm3, x8, x1, r4", "vmac\tcm6, cm5, x4, x2, r4"]
SYN_DEC = "\n".join([
    _syn("syn_dec_ok", ["movx\tr4, #0x35a", "movx\tr1, #0x8", "mova\tr3, #0x218"], MACS, QUARTERS),
    _syn("syn_dec_acc32", ["movx\tr4, #0x358", "movx\tr1, #0x8", "mova\tr3, #0x218"], MACS, QUARTERS),
    _syn("syn_dec_in", ["movx\tr4, #0x35a"], MACS + ["movx\tr4, #0x35a"], QUARTERS),
    _syn("syn_dec_reg", ["mov\tr4, r5"], MACS, QUARTERS),
    _syn("syn_dec_two", ["movx\tr4, #0x35a", "movx\tr4, #0x358"], MACS, QUARTERS),
    _syn("syn_dec_part", ["movx\tr4, #0x35a"], MACS, QUARTERS[:7]),
    _syn("syn_dec_badctl", ["movx\tr4, #0x35a", "movx\tr1, #0x2", "mova\tr3, #0x218"], MACS, QUARTERS),
])


def syn_read(sections, fn: str, shifts: dict, table: dict) -> dict:
    s = u60.section_for(sections, fn)
    rec = u60.read_function(sections, fn, "one_loop")
    lp = s.loops[rec["loop_index"]]
    return acc64_reading(s, lp, shifts, table)


# ---------------------------------------------------------------- the selftest

def selftest() -> bool:
    results = []

    def check(name, ok, **kw):
        results.append(bool(ok))
        say("SELFTEST_JSON", {"check": name, "ok": bool(ok), **kw})

    shas = {"u6_0": u60.lf_sha(U60_TOOL), "f2_0": u60.lf_sha(F20_TOOL)}
    check("the imported tools are U6-0's (a513ac0) and F2-0's (a47e631)",
          shas == {"u6_0": U60_TOOL_SHA, "f2_0": F20_TOOL_SHA}, lf_sha=shas)
    print("F2-0's selftest follows (it runs U6-0's fixture selftest first):", flush=True)
    check("F2-0's selftest passes (U6-0's fixture, F2's arithmetic, tiers, lines, energy, source and reader checks)",
          f20.selftest())

    # The tiers and the allowed S (the addendum, section 4).
    got = {f: form_tier(f)[0] for f in (0, 128, 129, 159, 160, 400)}
    check("the tiers: FLUSH 128 PASS; 129 and 159 MARGINAL; 160 FAIL; VOID FAIL",
          got == {0: "PASS", 128: "PASS", 129: "MARGINAL", 159: "MARGINAL", 160: "FAIL", 400: "FAIL"}
          and form_tier(None) == ("FAIL", None), tiers=got)
    check("E_F2 at the edges: 128 gives 24.00, 159 gives 24.48 (24.484), 160 gives 24.50",
          "%.2f" % form_tier(128)[1] == "24.00" and "%.2f" % form_tier(159)[1] == "24.48"
          and "%.2f" % form_tier(160)[1] == "24.50" and form_tier(159)[1] <= MARGINAL_LINE < form_tier(160)[1])
    sets = {f: allowed_s(f, form_tier(f)[0]) for f in (16, 17, 32, 33, 128, 159, 160)}
    check("the allowed S: 16 gives ROW, 16, 8; 17 and 32 give ROW, 16; 33 to 159 give ROW; 160 gives none",
          sets == {16: ["ROW", "16", "8"], 17: ["ROW", "16"], 32: ["ROW", "16"], 33: ["ROW"], 128: ["ROW"],
                   159: ["ROW"], 160: []}, sets=sets)
    check("the readings: two FAILs are KILL; ACC64 VOID with a PASS is VOID alone; VOID with two FAILs is both; "
          "CONFIRMED gives PASS if any form passes, else MARGINAL",
          verdict("CONFIRMED", {"B3": "FAIL", "B2": "FAIL"}) == ["KILL"]
          and verdict("VOID", {"B3": "PASS", "B2": "FAIL"}) == ["ACC64 VOID"]
          and verdict("VOID", {"B3": "FAIL", "B2": "FAIL"}) == ["ACC64 VOID", "KILL"]
          and verdict("CONFIRMED", {"B3": "MARGINAL", "B2": "PASS"}) == ["PASS"]
          and verdict("CONFIRMED", {"B3": "MARGINAL", "B2": "FAIL"}) == ["MARGINAL"])

    # The conversion (the addendum, section 1), bit for bit against RNE(I).
    vals = conv_values()
    a = np.array(vals, dtype=np.int64)
    hw = np.ascontiguousarray(a.astype("<i8")).view("<u2").reshape(-1, 4).tolist()
    ok_id = all(x == (r[2] - 65536 * (r[2] >> 15)) * 2 ** 32 + r[1] * 2 ** 16 + r[0] for x, r in zip(vals, hw))
    check("the identity I = h2 x 2^32 + h1 x 2^16 + h0 (h2 signed) holds on every listed value", ok_id, n=len(vals))
    w = np.ascontiguousarray(a.astype("<i8")).view("<u2").reshape(-1, 4)
    f21 = (w[:, 2].copy().view(np.int16).astype(np.float32) * np.float32(2.0 ** 32)
           + w[:, 1].astype(np.float32) * np.float32(2.0 ** 16))
    exact = (w[:, 2].copy().view(np.int16).astype(np.float64) * 2.0 ** 32 + w[:, 1].astype(np.float64) * 2.0 ** 16)
    check("f2 + f1 is exact in fp32 on every listed value (|I / 2^16| < 2^23)",
          bool(np.all(f21.astype(np.float64) == exact)) and F2_I_MAX // 2 ** 16 < 2 ** 23,
          a_max=F2_I_MAX // 2 ** 16)
    same = fp32_of_i(a).view(np.uint32) == rne(a).view(np.uint32)
    check("fp32(I) equals RNE(I) bit for bit on 0, +-1, the bound 336,928,358,400, +-(2^39 - 1), 2^k and 2^k +- 1 "
          "(k <= 38) and fp32 ties", bool(np.all(same)), n=len(vals), mismatches=int((~same).sum()),
          ties=[x for x in (2 ** 25 + 2, 2 ** 24 + 1) if x in vals])
    rng = np.random.default_rng(20260925)
    r = rng.integers(-F2_I_MAX, F2_I_MAX + 1, size=10 ** 6, dtype=np.int64)
    same = fp32_of_i(r).view(np.uint32) == rne(r).view(np.uint32)
    check("fp32(I) equals RNE(I) bit for bit on 10^6 seeded random I with |I| <= 336,928,358,400",
          bool(np.all(same)), n=int(r.size), seed=20260925, mismatches=int((~same).sum()))
    sw = fp32_of_i(a, swap=True).view(np.uint32) != rne(a).view(np.uint32)
    check("a word-swapped reading fails (the emulation depends on the order; the order rests on the IR derivation)",
          bool(np.any(sw)), mismatches=int(sw.sum()), n=len(vals))

    # The header (read, not compiled): the formula, amode's partition, the overload aie::mac calls.
    htext = HEADER.read_text(encoding="utf-8") if HEADER.exists() else ""
    hr = header_record(htext) if htext else {"why": ["the header is missing"]}
    check("the header: aiev2_compute_control at lines 15-25 shifts amode by 1 (line 21), 2 bits wide; amode "
          "partitions the accumulators (0 acc32, 1 acc64/cacc64, 2 accfloat/caccfloat); the v32int16 "
          "mac_elem_16_2_conf at lines 16189-16195 passes amode 1",
          hr.get("why") is None and hr["formula"]["lines"] == [15, 25] and hr["formula"]["amode_line"] == 21
          and hr["overload"]["lines"] == [16189, 16195],
          formula=hr.get("formula"), amode=hr.get("amode"), overload=hr.get("overload"), why=hr.get("why"))
    shifts = hr["formula"]["shifts"] if hr.get("formula") else {}
    table = hr["amode"]["table"] if hr.get("amode") else {}
    k1, k2 = decode(0x35A, shifts), decode(0x352, shifts)
    check("known words: 0x35a is amode 1, bmode 3, variant 2, sgn_x 1, sgn_y 1, zero_acc 0; 0x352 is amode 1, "
          "bmode 2, variant 2; the formula rebuilds 0x35a",
          (k1.get("amode"), k1.get("bmode"), k1.get("variant"), k1.get("sgn_x"), k1.get("sgn_y"), k1.get("zero_acc"))
          == (1, 3, 2, 1, 1, 0) and (k2.get("amode"), k2.get("bmode"), k2.get("variant")) == (1, 2, 2)
          and compute_control(shifts, sgn_x=1, sgn_y=1, amode=1, bmode=3, variant=2) == CONF_EXPECTED,
          d_35a=k1, d_352=k2)

    # The decoder on synthetic text.
    sec = aie_disasm.parse(SYN_DEC)
    rd = {fn: syn_read(sec, fn, shifts, table) for fn in
          ("syn_dec_ok", "syn_dec_acc32", "syn_dec_in", "syn_dec_reg", "syn_dec_two", "syn_dec_part",
           "syn_dec_badctl")}
    ok = rd["syn_dec_ok"]
    check("syn_dec_ok: two MACs on I (cm4, cm6) on r4 = 0x35a: CONFIRMED; controls r1 (0x8) and r3 (0x218) read "
          "acc32", ok["reading"] == "CONFIRMED" and [m["dest"] for m in ok["macs"]["on_i"]] == ["cm4", "cm6"]
          and ok["config"]["value"] == "0x35a" and [c.get("accumulator") for c in ok["controls"]] == ["acc32", "acc32"],
          reading=ok["reading"], config=ok.get("config", {}).get("value"))
    check("syn_dec_acc32: r4 = 0x358 (amode 0) reads VOID", rd["syn_dec_acc32"]["reading"] == "VOID")
    check("a write to r4 inside the loop, a register copy, or two values is 'cannot be decoded' (a STOP)",
          all(rd[f]["reading"] == "STOP" and "cannot be decoded" in rd[f]["why"]
              for f in ("syn_dec_in", "syn_dec_reg", "syn_dec_two")),
          why={f: rd[f]["why"] for f in ("syn_dec_in", "syn_dec_reg", "syn_dec_two")})
    check("a MAC whose cm quarters are not all stored is not a MAC on I (one MAC on I: a STOP)",
          rd["syn_dec_part"]["reading"] == "STOP" and len(rd["syn_dec_part"]["macs"]["on_i"]) == 1,
          why=rd["syn_dec_part"]["why"])
    check("a control config that decodes to other than acc32 (r1 = 0x2, amode 1) is a STOP",
          rd["syn_dec_badctl"]["reading"] == "STOP" and "not trusted" in (rd["syn_dec_badctl"]["why"] or ""),
          why=rd["syn_dec_badctl"]["why"])

    # The source (the addendum, section 2).
    text = source_text()
    check("FLUSH_B3's branch is f2_epilogue.cc's lines 67-94, text for text",
          branch_lines(text, "B3") == f20_tail() and f20_tail()[0].startswith("// Sx[r] to lanes")
          and f20_tail()[-1] == "y += 32;", n=len(f20_tail()))
    u6e = u60.u6e_order() if u60.lf_sha(u60.U6E_TOOL) == u60.U6E_TOOL_SHA else None
    b3, b2 = branch_order(text, "B3"), branch_order(text, "B2")
    want3 = [tuple(x) for x in u6e[3]["P"]] if u6e else None
    check("FLUSH_B3's P and Y follow U6-E's frozen ORDER[3]['P']",
          want3 is not None and b3["n_first"] == 1 and b3["P"] == want3 and b3["Y"] == want3, b3=b3)
    check("FLUSH_B2's P follows ORDER[2]['P'] and its Y ORDER[2]['Y']",
          u6e is not None and b2["n_first"] == 1 and b2["P"] == [tuple(x) for x in u6e[2]["P"]]
          and b2["Y"] == [tuple(x) for x in u6e[2]["Y"]], b2=b2)
    sw3 = text.replace("    Y_TERM(2, 2);\n    Y_TERM(1, 3);\n", "    Y_TERM(1, 3);\n    Y_TERM(2, 2);\n", 1)
    sw2 = text.replace("    P_TERM(1, 2);\n    P_TERM(1, 1);\n    split2", "    P_TERM(1, 1);\n    P_TERM(1, 2);\n"
                       "    split2", 1)
    check("copies with two terms swapped fail the order checks (B3's Y, B2's P)",
          sw3 != text and sw2 != text and branch_order(sw3, "B3")["Y"] != want3
          and branch_order(sw2, "B2")["P"] != [tuple(x) for x in (u6e[2]["P"] if u6e else [])])
    sf = source_facts()
    check("the source has no acc64, vector_cast or srs in code; three to_float calls, H2 at -32, H1 at -16, H0 at 0, "
          "in that order; the shuffles the emulation mirrors", sf["acc64_in_code"] == 0
          and sf["vector_cast_in_code"] == 0 and sf["srs_in_code"] == 0
          and sf["to_float"] == [("H2", -32), ("H1", -16), ("H0", 0)] and sf["shuffles_present"], facts=sf)
    check("set_rounding sits once, in f2_flush_b before its tile loop",
          len(sf["set_rounding"]) == 1 and len(sf["loop"]) == 1 and len(sf["f2_flush_b"]) == 1
          and sf["f2_flush_b"][0] < sf["set_rounding"][0] < sf["loop"][0], lines=sf)

    ok = all(results)
    print("F2-0b SELFTEST OK" if ok else f"F2-0b SELFTEST FAILED ({results.count(False)} of {len(results)})",
          flush=True)
    return ok


# ---------------------------------------------------------------- the run

def compile_obj(source: Path, defines: list, obj: str):
    cmd = [str(u60.CLANG), str(source), "-c", "-o", obj, f"-I{u60.INCLUDE}", *u60.IRON_FLAGS, *defines]
    p = subprocess.run(cmd, capture_output=True, text=True)
    err = (p.stderr or "").replace(str(Path.home()), "~").strip()
    return p.returncode == 0 and os.path.exists(obj), err[-4000:]


def protocol() -> dict:
    return {"stage": "hybrid F2-0b", "scope": "compile only: no NPU, no hardware context, no xclbin",
            "plan_f2_sha": PLAN_F2_SHA, "addendum_sha": ADDENDUM_SHA,
            "sources": {"kernels/f2_epilogue/f2_epilogue.cc": CORE_SOURCE_SHA,
                        "kernels/f2_epilogue/f2_flush_b.cc": SOURCE_SHA},
            "imports": {"tools/hybrid_u6_0.py": U60_TOOL_SHA, "tools/hybrid_f2_0.py": F20_TOOL_SHA,
                        "use": "U6-0's reader, validity, flags and sizes; F2-0's case reader, load and store census, "
                               "lines and energy; neither is edited, and F2-0's tier() is not used"},
            "inputs": {"f2_0_log": {"file": "results/llm/hybrid_f2_0_desktop2_20260925.log", "lf_sha256": F20_LOG_SHA,
                                    "E_core": E_CORE, "F2_CORE_bundles": F20_CORE_BUNDLES,
                                    "F2_CORE_text_bytes": F20_CORE_TEXT, "control": F20_CONTROL}},
            "header": {"file": HEADER_REL, "lf_sha256": HEADER_SHA,
                       "cited": "aiev2_compute_control, lines 15-25 (amode << 1 at line 21); the v32int16 "
                                "mac_elem_16_2_conf, lines 16189-16195 (amode 1), which aie::mac on an acc64 calls "
                                "(aie_api detail/aie2/mul_acc64.hpp:130-139)"},
            "cases": [{"case": c, "source": s.relative_to(ROOT).as_posix(), "defines": d, "function": f, "role": r,
                       "loop_rule": lr, "min_int_macs": n} for c, s, d, f, r, lr, n in CASES],
            "precondition": PRECONDITION, "tiers": TIERS, "e_core": E_CORE, "row_blocks": ROW_BLOCKS,
            "pass_line": PASS_LINE, "marginal_line": MARGINAL_LINE,
            "void": "a control-register write inside the loop, or not exactly one hardware loop (U6-0's rule)",
            "conversion": "fp32(I) = (h2 x 2^32 + h1 x 2^16) + h0 from the 16-bit pieces of each lane's words, by "
                          "shuffles; each piece exact by aie::to_float's 16-bit path; one rounding, at the last add "
                          "(the addendum, section 1)",
            "excludes": u60.EXCLUDES, "flags": u60.IRON_FLAGS, "clang": u60.home(u60.CLANG),
            "objdump": [u60.home(u60.OBJDUMP), "-d", "--no-show-raw-insn"],
            "size": [u60.home(u60.LLVM_SIZE), "-A", "summing .text*"], "text_limit": u60.TEXT_LIMIT,
            "predictions": PREDICTIONS}


def read_obj(obj: str, fn: str, rule: str, need: int):
    sections = aie_disasm.parse(aie_disasm.disassemble(obj, str(u60.OBJDUMP)))
    return sections, f20.read_case(sections, fn, rule, need)


def print_body(case: str, fr: dict) -> None:
    i = fr.get("loop_index")
    lp = fr["loops"][i] if i is not None else {}
    if fr.get("body"):
        print(f"LOOP_BODY {case} {lp.get('name')}: {len(fr['body'])} bundles", flush=True)
        for ln in fr["body"]:
            print(f"    {ln}", flush=True)


def case_record(case, source, defines, fn, role, rule, ok, err, fr, obj) -> dict:
    rec = {"case": case, "source": source.relative_to(ROOT).as_posix(), "role": role, "defines": defines,
           "function": fn, "loop_rule": rule, "compiled": ok, "error": err or None}
    if not ok:
        rec.update({"loop_found": False, "void": "did not compile"})
        return rec
    i = fr.get("loop_index")
    lp = fr["loops"][i] if i is not None else {}
    mn = (fr.get("slot_census") or {}).get("mnemonics") or {}
    rec.update({"loop_found": i is not None, "loop": lp.get("name"), "loop_bundles": lp.get("bundles"),
                "int_macs_in_loop": lp.get("int_macs"), "float_macs_in_loop": lp.get("float_macs"),
                "vector_ops_in_loop": lp.get("vector_ops"), "stack_refs_in_loop": lp.get("stack_refs"),
                "loads_stores_in_loop": f20.ldst(fr) if i is not None else None,
                "vsrs_in_loop": sum(n for h, n in mn.items() if h.startswith("vsrs")),
                "function_stack_refs": fr.get("function_stack_refs"), "frame_bytes": fr.get("frame_bytes"),
                "pipelined": fr.get("pipelined"), "outside": fr.get("outside"), "cr_writes": fr.get("cr_writes"),
                "slot_census": fr.get("slot_census"), "text_bytes": u60.text_bytes(obj), "void": fr["void"]})
    return rec


def run() -> int:
    print(f"F2-0b run {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    print(f"COMMIT {commit}", flush=True)
    say("TOOL_SHA_JSON", {"file": "tools/hybrid_f2_0b.py", "lf_sha256": u60.lf_sha(Path(__file__))})
    pins = [("the U6-0 reader", U60_TOOL, U60_TOOL_SHA), ("the F2-0 tool", F20_TOOL, F20_TOOL_SHA),
            ("F2-0's log", F20_LOG, F20_LOG_SHA), ("f2_epilogue.cc", CORE_SOURCE, CORE_SOURCE_SHA),
            ("f2_flush_b.cc", SOURCE, SOURCE_SHA), ("the F2 plan", PLAN_F2, PLAN_F2_SHA),
            ("the F2-0b addendum", ADDENDUM, ADDENDUM_SHA), ("aiev2_vmult.h", HEADER, HEADER_SHA)]
    read = {name: (u60.lf_sha(p) if p.exists() else None) for name, p, _ in pins}
    say("PIN_JSON", [{"what": name, "file": (p.relative_to(ROOT).as_posix() if name != "aiev2_vmult.h"
                                             else HEADER_REL), "pinned": sha, "lf_sha256": read[name]}
                     for name, p, sha in pins])
    tc = u60.toolchain() if u60.SITE.exists() else None
    say("TOOLCHAIN_JSON", {"pinned": u60.TOOLCHAIN_PIN, "read": tc})
    bad = [name for name, _, sha in pins if read[name] != sha] + ([] if tc == u60.TOOLCHAIN_PIN else ["the toolchain"])
    if bad:
        stop(f"pin mismatch: {', '.join(bad)}")
    for p in (u60.CLANG, u60.OBJDUMP, u60.LLVM_SIZE):
        if not p.exists():
            stop(f"{u60.home(p)} not found")
    if not selftest():
        stop("the selftest failed")
    say("PROTOCOL_JSON", protocol())

    hr = header_record(HEADER.read_text(encoding="utf-8"))
    say("HEADER_JSON", {"file": HEADER_REL, **hr})
    if hr["why"]:
        stop(f"the header: {hr['why']}")
    shifts, table = hr["formula"]["shifts"], hr["amode"]["table"]

    log = f20_log()
    say("F20_LOG_JSON", {k: v for k, v in log.items() if k != "body"})
    if (log["e_core"], log["bundles"], log["text_bytes"], log["control"]) != (E_CORE, F20_CORE_BUNDLES, F20_CORE_TEXT,
                                                                             F20_CONTROL):
        stop("F2-0's log does not read E_core 22, F2_CORE 29, .text 496 and CONTROL 7")

    recs = {}
    with tempfile.TemporaryDirectory(prefix="f2_0b_") as tmp:
        # 1-2. The precondition: F2_CORE recompiled, then the decode on that object.
        case, source, defines, fn, role, rule, need = CASES[0]
        obj = os.path.join(tmp, case + ".o")
        ok, err = compile_obj(source, defines, obj)
        fr = {}
        if ok:
            sections, fr = read_obj(obj, fn, rule, need)
            say("LOOPS_JSON", {"case": case, "function": fn, "loops": fr.get("loops", [])})
            print_body(case, fr)
        rec = case_record(case, source, defines, fn, role, rule, ok, err, fr, obj)
        say("CASE_JSON", rec)
        recs[case] = rec
        if rec["void"]:
            stop(f"{case}: {rec['void']}" + (f"; error: {err}" if not ok else ""))
        same_body = fr.get("body") == log["body"]
        re_read = rec["loop_bundles"] == F20_CORE_BUNDLES and same_body and rec["text_bytes"] == F20_CORE_TEXT
        say("RECOMPILE_JSON", {"bundles": rec["loop_bundles"], "f2_0_bundles": F20_CORE_BUNDLES,
                               "body_identical": same_body, "text_bytes": rec["text_bytes"],
                               "f2_0_text_bytes": F20_CORE_TEXT, "reading": "RE-READ" if re_read else "STOP"})
        if not re_read:
            stop(f"F2_CORE recompiled reads {rec['loop_bundles']} bundles, body identical {same_body}, .text "
                 f"{rec['text_bytes']}: not F2-0's object (the gate's ruling 1)")
        s = u60.section_for(sections, fn)
        lp = s.loops[fr["loop_index"]]
        pre = [b for b in s.bundles if b.addr < lp.start]
        print(f"PRELOOP {fn}: {len(pre)} bundles before {lp.name} (0x{lp.start:x})", flush=True)
        for b in pre:
            print(f"    0x{b.addr:04x}  {' | '.join(norm(f) for f in b.live) or '(all nop)'}", flush=True)
        acc = acc64_reading(s, lp, shifts, table)
        say("ACC64_JSON", acc)
        if acc["reading"] == "STOP":
            stop(f"the acc64 confirmation: {acc['why']}")

        # 3. The two flush forms.
        for case, source, defines, fn, role, rule, need in CASES[1:]:
            obj = os.path.join(tmp, case + ".o")
            ok, err = compile_obj(source, defines, obj)
            fr = {}
            if ok:
                _, fr = read_obj(obj, fn, rule, need)
                say("LOOPS_JSON", {"case": case, "function": fn, "loops": fr.get("loops", [])})
                print_body(case, fr)
            rec = case_record(case, source, defines, fn, role, rule, ok, err, fr, obj)
            say("CASE_JSON", rec)
            recs[case] = rec
            if ok and not rec.get("cr_writes"):
                print(f"NOTE: no control-register write located in {case}, though the source calls set_rounding "
                      f"before the loop", flush=True)
            if ok and rec.get("stack_refs_in_loop"):
                print(f"NOTE: {case}'s loop holds {rec['stack_refs_in_loop']} stack references (a spill); they are "
                      f"counted", flush=True)

    reading = acc["reading"]
    forms = {}
    for f in FORMS:
        r = recs[f]
        flush = None if r["void"] else r["loop_bundles"]
        t, e = form_tier(flush)
        forms[f] = {"FLUSH": flush, "E_F2": None if e is None else round(e, 4), "tier": t,
                    "allowed_s": allowed_s(flush, t), "void": r["void"]}
    heads = verdict(reading, {f: v["tier"] for f, v in forms.items()})
    say("OUTPUT_JSON", {"E_core": E_CORE, "acc64": reading, "config": acc["config"]["value"],
                        "forms": forms, "readings": heads, "row_blocks": ROW_BLOCKS,
                        "pass_line": PASS_LINE, "marginal_line": MARGINAL_LINE,
                        "loads_stores": {c: r.get("loads_stores_in_loop") for c, r in recs.items()},
                        "text_bytes": {c: r.get("text_bytes") for c, r in recs.items()},
                        "text_limit": u60.TEXT_LIMIT})
    for f, v in forms.items():
        if v["E_F2"] is not None:
            say("ENERGY_JSON", {"form": f, **f20.energy(v["E_F2"], F20_CONTROL)})

    def score(ok):
        return "VOID" if ok is None else ("HOLDS" if ok else "FAILS")

    b3, b2 = forms["FLUSH_B3"]["FLUSH"], forms["FLUSH_B2"]["FLUSH"]
    say("PRED_JSON", {
        "F2b-P1": score(all(recs[f]["compiled"] and recs[f]["void"] is None for f in FORMS)),
        "F2b-P2": score(None if any(not recs[f]["compiled"] for f in FORMS)
                        else all(recs[f]["vsrs_in_loop"] == 0 for f in FORMS)),
        "F2b-P3": score(None if b3 is None or b2 is None else b2 < b3),
        "F2b-P4": score(True),
        "F2b-P5": score(acc["config"]["amode"] == 1)})

    cfg = acc["config"]
    print(f"F2-0b F2_CORE: {recs['F2_CORE']['loop_bundles']} bundles, RE-READ (bit-identical to F2-0's loop and "
          f".text)", flush=True)
    print(f"F2-0b ACC64: {'CONFIRMED' if reading == 'CONFIRMED' else 'VOID'} (the MACs on I use {cfg['register']} = "
          f"{cfg['value']}: amode {cfg['amode']}, {cfg['accumulator']}; aiev2_vmult.h:15-25)", flush=True)
    for f, v in forms.items():
        shown = f"{v['FLUSH']}" if v["FLUSH"] is not None else f"VOID ({v['void']})"
        e = "%.2f" % v["E_F2"] if v["E_F2"] is not None else "UNKNOWN"
        t = v["tier"] if reading == "CONFIRMED" or v["tier"] == "FAIL" else "REPORT-ONLY"
        print(f"F2-0b {f}: FLUSH {shown} cycles per tile and superblock; E_F2 {e} (22 + FLUSH / {ROW_BLOCKS}); "
              f"{t}; allowed S {v['allowed_s'] if t != 'FAIL' else []}", flush=True)
        if v["E_F2"] is not None:
            en = f20.energy(v["E_F2"], F20_CONTROL)
            print(f"F2-0b ENERGY {f} (compute only, at N-w4's power): plan form {en['plan_form']['mj_with_idle']} "
                  f"mJ, serial form {en['serial_form']['mj_with_idle']} mJ, with idle per prompt token per layer; "
                  f"D-nb16 {f20.D_NB16_MJ}, N-bf16 {f20.N_BF16_MJ}", flush=True)
    words = {"ACC64 VOID": "E_core = 22 is void; F2_CORE returns to design",
             "KILL": "both flush forms FAIL; the core flush is killed, and F2 ends",
             "PASS": "a form passes (E_F2 <= 24); necessary, not sufficient; F2-E may be proposed, on the gate's go",
             "MARGINAL": "a form is MARGINAL (E_F2 <= 24.49); F2-E may run on the gate's go; the energy verdict is "
                         "the user's, with the serial form (E <= 14) in hand"}
    for h in heads:
        print(f"F2-0b {h}: {words[h]}" if h != "ACC64 VOID" else f"ACC64 VOID: {words[h]}", flush=True)
    return 0


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or [""])[0]
    if mode == "selftest":
        print(f"F2-0b selftest {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
        say("TOOL_SHA_JSON", {"file": "tools/hybrid_f2_0b.py", "lf_sha256": u60.lf_sha(Path(__file__))})
        return 0 if selftest() else 1
    if mode == "run":
        return run()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
