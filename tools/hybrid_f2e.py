#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid stack, F2-E: the precision of F2 (integerized 8-bit scales) at S = ROW, CPU only (numpy, and one small ORT
session per linear). No NPU and no GPU.

F2 (the F2 plan, v2 final, LF b98787f4, section 1) replaces route (i)'s per-block fp32 scales with 8-bit integers
under one fp32 scale per superblock of S blocks:
    I[m,n,g] = sum over b in g of i[m,n,b] x qx[m,b] x qw[n,b], exact in int64;
    y[m,n]   = the flush of fp32(I) x Sx[m,g] x Dw[n,g], once per superblock, in increasing K.
F2-0b (its log at 5a5e15a) read both flush forms PASS at S = ROW only. This tool emulates the compiled flush
(kernels/f2_epilogue/f2_flush_b.cc, B3 and B2) on the weights of layers 0, 16 and 33 and S1's captured layer-16
inputs, and reads each arm against MatMulNBits accuracy level 0 (R0's arithmetic) by the rule below, fixed before
any real input is read. The F2-E addendum (git-ignored, LF 1ad61601) amends the plan's sections 1 and 3; this tool
follows it.

    python tools/hybrid_f2e.py selftest   # synthetic only: the grid, the round-up, the flush's order and floor, rules
    python tools/hybrid_f2e.py pins       # the pins' current values: hashes and sizes only, nothing computed
    python tools/hybrid_f2e.py run        # the pins, E_F2, the order, the anchors, then the 12 arms and the pick
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma_compress as gc  # noqa: E402
import gemma_decode as gd  # noqa: E402
import hybrid_f2_0b as f20b  # noqa: E402
import hybrid_s1 as s1  # noqa: E402
import hybrid_u6e as u6e  # noqa: E402
import llm_prefill3c as p3c  # noqa: E402

# ---------------------------------------------------------------- the pins (LF sha256)

PLAN_FILE = ROOT / "scratch/llm/hybrid_f2_plan_draft.md"                                 # git-ignored
PLAN_SHA = "b98787f451c11fbc9e847a4cb973b8f12b4988e3dce7729435d1fd34334a07c5"          # v2 final
ADD0B_FILE = ROOT / "scratch/llm/hybrid_f2_0b_addendum.md"                               # git-ignored
ADD0B_SHA = "81dfe8da781aedd8513945c95b556739545326f80cda2b66fab0ba2504b3c26c"
ADDENDUM = ROOT / "scratch/llm/hybrid_f2e_addendum.md"                                   # git-ignored
ADDENDUM_SHA = "1ad61601f743525b6c08c2de38a86bad6f32f742e756a25333d5b257f2f165ec"       # accepted by the gate
F20B_LOG = ROOT / "results/llm/hybrid_f2_0b_desktop2_20260925.log"                       # 5a5e15a
F20B_LOG_SHA = "dfd4849f5a7cbe8fe16ca39b69c9221a33333c682bd05b8588490eb9b371c4bf"
U6E_LOG = ROOT / "results/llm/hybrid_u6e_run_desktop2_20260925.log"                      # 8faa9db
U6E_LOG_SHA = "adaa8fa8a1f4db89a85c30f49b3b48487a3fed28e494313efe4cecf5bbd07e4e"
SOURCE = f20b.SOURCE                                                                      # f2_flush_b.cc
SOURCE_SHA = "caa62ce055f6e1a3e8de350349ac76bef3f8d9dd521416c9fb8db0c64a2e18d4"
U6E_TOOL = ROOT / "tools/hybrid_u6e.py"                                                   # frozen at 2505530
U6E_TOOL_SHA = "b88d23d30c6c2ec3ffa55407dd4969957acf98e788fca47f1a703feec146e892"
F20B_TOOL = ROOT / "tools/hybrid_f2_0b.py"                                                # a72726e
F20B_TOOL_SHA = "7a953146eaccc4b143bb4cf3072a84f10cad10f4f54bab1ec956d3588fcda354"
PINS = [("the F2 plan", PLAN_FILE, PLAN_SHA), ("the F2-0b addendum", ADD0B_FILE, ADD0B_SHA),
        ("the F2-E addendum", ADDENDUM, ADDENDUM_SHA), ("F2-0b's log", F20B_LOG, F20B_LOG_SHA),
        ("U6-E's run log", U6E_LOG, U6E_LOG_SHA), ("f2_flush_b.cc", SOURCE, SOURCE_SHA),
        ("the U6-E tool", U6E_TOOL, U6E_TOOL_SHA), ("the F2-0b tool", F20B_TOOL, F20B_TOOL_SHA)]

# ---------------------------------------------------------------- the protocol (fixed before any real input)

LAYERS = u6e.LAYERS                                      # 0, 16, 33
ROWS = u6e.ROWS                                          # rows 0, 8, ..., 2040 of S1's sequence 0 (256)
ACT_FORMS = ("B", "A")                                   # B deciding (host requantizes); A report-only (L4's codes)
FLUSH_FORMS = ("B2", "B3")
S_SET = ("ROW", 16, 8)
ARMS = [(f, fl, s) for f in ACT_FORMS for fl in FLUSH_FORMS for s in S_SET]
DECIDING = [("B", "B2", "ROW"), ("B", "B3", "ROW")]     # the pick order: the lower E_F2 first
F2_FACTOR = 2.0                                          # the gate's ruling (2026-09-25), deciding
FACTORS = (1.12, 1.41, 2.0)                              # the printed sensitivity
A2_FACTOR = 2.0
ANCHOR_TOL = 1e-6                                        # U6-E's l4_vs_l0, relative (the gate's ruling 2)
FLUSH_PIN = {"B2": 82, "B3": 116}                        # F2-0b's loop bundles (its CASE_JSON)
E_CORE_PIN, ROW_PIN = 22, 64                             # F2-0b's OUTPUT_JSON E_core and row_blocks
K_FLOOR = 8                                              # B3's per-sample floor, in units of 2^-24 (fix F2)
FLOOR = K_FLOOR * 2.0 ** -24
I_BOUND = 2 ** 39
QX_MAX, QW_MAX, CODE_MAX = 255, 127, 127

# The flush forms as f2_flush_b.cc compiles them (the F2-0b addendum, section 2): B3's y terms reuse ORDER[3]['P'].
FLUSH_ORDERS = {"B3": {"t": 3, "P": list(u6e.ORDER[3]["P"]), "Y": list(u6e.ORDER[3]["P"])},
                "B2": {"t": 2, "P": list(u6e.ORDER[2]["P"]), "Y": list(u6e.ORDER[2]["Y"])}}

RULE = ("an arm (activation form, flush form, S) is ACCEPTABLE iff, for every case c (layers 0, 16 and 33 x the "
        "seven linears), r_F2(c) <= F2_FACTOR x r_R(c), with r_F2 = rel_l2(emulation, L0) and r_R = rel_l2(L4, L0) "
        "on the same 256 rows; F2_FACTOR = 2.0, deciding. The ACCEPTABLE set is printed at 1.12, 1.41 and 2.0")
PICK_RULE = ("the pick is form B at S = ROW: FLUSH_B2 if ACCEPTABLE (E_F2 23.28125), else FLUSH_B3 (23.8125), else "
             "F2 SCREEN: NONE. A report-only arm (form A, or S = 16 or 8) is never picked; NONE ends F2 (plan "
             "section 6), and a pick opens F2-A only on the user's go")
A2_RULE = ("A2-F2's threshold = up2(A2_FACTOR x the maximum over the 21 cases of rel_l2(emulation, F64_F2)) for the "
           "picked arm; the integer I must match exactly. NOT SET under NONE")
GRID = ("Sx = the smallest fp32 with 255 x Sx >= max s_x over the superblock, Dw the same with 127 x Dw >= max "
        "abs(d_w): the float64 quotient's fp32 round-up, then the exact float64 product check, stepping one ulp "
        "up where it fails. Form B: qx = max(1, c), c the exact ceiling of s_x / Sx (c = 256 is a STOP), codes "
        "RNE(x / (qx Sx)) in float64. Form A: L4's codes, qx = max(1, RNE(s_x / Sx)). qw = RNE(d_w / Dw). Any "
        "code or qw clip on real data is a STOP (the addendum, sections 1, 2.2 and 2.3)")
FLUSH_TEXT = ("fp32(I) = RNE(I); per superblock in increasing K, y (fp32, from 0) takes one flush: split T of "
              "fp32(I), Sx and Dw (U6-E's split), P by ORDER P (a multiply, then fp32 adds of exact bf16 products), "
              "split T of P, then the ORDER Y terms added onto y. B3: T = 3, P and Y by ORDER[3]['P']; B2: T = 2, "
              "ORDER[2]['P'] and ORDER[2]['Y']")
SCOPE = ("layer 16's activations only: x16_* on S1's sequence 0, rows 0, 8, ..., 2040, for all 21 cases; layers 0 "
         "and 33 use layer 16's activations (U6-E's assumption, hybrid_u6e.py:123)")
NOTE_N2 = ("F2-E cannot change the N2 pick: F2's best modelled energy is 1.67x N-i8's measured energy (DECISIONS "
           "at ca75870); F2-E answers only whether F2's arithmetic is acceptable at S = ROW")
PREDICTIONS = {
    "F2E-P1": "for each activation form and S, abs(r_F2(B2) - r_F2(B3)) <= 0.01 x r_R(c) in every case",
    "F2E-P2": "at ROW, form B's r_F2 <= form A's in every case, for both flush forms",
    "F2E-P3": "under form B, r_F2 does not increase as S shrinks (ROW, then 16, then 8), in every case, both flushes",
    "F2E-P4": "in each layer, down has the largest ROW qw = 0 fraction of the seven linears",
}
ASSUMPTIONS = u6e.ASSUMPTIONS[:1] + [
    "the fp32 add at fp32(I)'s last step rounds to nearest even (INFERRED; the F2-0b addendum, section 1)",
    "a host reproduces the round-up of Sx and Dw and form B's float64 codes; a host rounding to nearest is not "
    "modelled"]


class Stop(Exception):
    """A pipeline check failed: the run ends with rc 2, and the reason goes to the gate."""


def protocol() -> dict:
    return {"stage": "hybrid F2-E", "scope": "CPU only: numpy and ORT's CPU EP; no NPU, no GPU",
            "pins": {what: {"file": p.relative_to(ROOT).as_posix(), "lf_sha256": sha} for what, p, sha in PINS},
            "layers": LAYERS,
            "rows": {"count": len(ROWS), "first": ROWS[0], "last": ROWS[-1], "step": ROWS[1] - ROWS[0]},
            "arms": ["/".join(map(str, a)) for a in ARMS], "deciding": ["/".join(map(str, a)) for a in DECIDING],
            "rule": RULE, "f2_factor": F2_FACTOR, "factors": FACTORS, "pick": PICK_RULE, "a2": A2_RULE,
            "a2_factor": A2_FACTOR, "grid": GRID, "flush": FLUSH_TEXT,
            "orders": {k: {"P": v["P"], "Y": v["Y"]} for k, v in FLUSH_ORDERS.items()},
            "e_f2": {"flush_pin": FLUSH_PIN, "e_core": E_CORE_PIN, "row_blocks": ROW_PIN,
                     "rule": "E_F2 = E_core + FLUSH / row_blocks, exact; the log's 4-dp print must agree"},
            "anchors": {"c2": "L4 on layer 16 equals S1's saved C2 output bit for bit, and reproduces C2's logged "
                              "values (U6-E's anchor)",
                        "u6e": f"rel_l2(L4, L0) reproduces U6-E's logged l4_vs_l0 within {ANCHOR_TOL:g} relative"},
            "b3_floor": f"selftest: abs(B3 - F64) <= {K_FLOOR} x 2^-24 x abs(F64) per sample (DERIVED worst ~7.1)",
            "scope_note": SCOPE, "note_n2": NOTE_N2, "predictions": PREDICTIONS, "assumptions": ASSUMPTIONS,
            "memory": u6e.MEMORY, "ort_version": u6e.ORT_VERSION}


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def header(title: str) -> None:
    print(f"{title} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    s1.say("TOOL_SHA_JSON", {"file": "tools/hybrid_f2e.py", "lf_sha256": lf_sha(Path(__file__))})
    s1.say("PROTOCOL_JSON", protocol())


def arm_key(a) -> str:
    return "/".join(map(str, a))


# ---------------------------------------------------------------- the grid (the addendum, sections 2.2 and 2.3)

F32_UP = np.float32(np.inf)


def fp32_up(q) -> np.ndarray:
    """float64 (finite) -> the smallest fp32 >= it."""
    q = np.asarray(q, dtype=np.float64)
    f = q.astype(np.float32)
    with np.errstate(under="ignore"):                              # nextafter(0) is a denormal, never selected
        return np.where(f.astype(np.float64) < q, np.nextafter(f, F32_UP), f).astype(np.float32)


def step_up(S, mx, den: int) -> tuple:
    """The exact product check: den x S >= mx in float64 (den <= 255: 8 x 24 bits, exact). Where it fails, S steps
    one fp32 ulp up. Returns (S, the number of steps)."""
    S = np.array(S, dtype=np.float32)
    bad = den * S.astype(np.float64) < np.asarray(mx, dtype=np.float32).astype(np.float64)
    with np.errstate(under="ignore"):
        S[bad] = np.nextafter(S[bad], F32_UP)
    return S, int(np.count_nonzero(bad))


def scale_up(mx, den: int) -> tuple:
    """Sx (den 255) or Dw (den 127): the float64 quotient's fp32 round-up, then the product check and its step."""
    mx = np.asarray(mx, dtype=np.float32)
    return step_up(fp32_up(mx.astype(np.float64) / den), mx, den)


def scale_faults(S, mx, den: int) -> int:
    """The hard check (section 6): superblocks where den x S < mx, or where a smaller fp32 would also do."""
    S = np.asarray(S, dtype=np.float32)
    mx64 = np.asarray(mx, dtype=np.float32).astype(np.float64)
    below = den * S.astype(np.float64) < mx64
    with np.errstate(under="ignore"):
        smaller = (S > 0) & (den * np.nextafter(S, np.float32(0)).astype(np.float64) >= mx64)
    return int(np.count_nonzero(below | smaller))


def groups(kb: int, s) -> tuple:
    """(the number of superblocks, their size in blocks) for S in S_SET; ROW is all of K."""
    size = kb if s == "ROW" else int(s)
    if kb % size:
        raise ValueError(f"S = {s} does not divide {kb} blocks")
    return kb // size, size


def ceil_exact(num64, den64, ratio) -> np.ndarray:
    """The exact integer ceiling of num / den from the float64 quotient, corrected by the exact products so that
    c x den >= num > (c - 1) x den. c <= 255 has 8 bits (256, a STOP, is a power of two) and den 24, so both
    products are exact in float64. den == 0 gives 0."""
    c = np.ceil(ratio)
    pos = den64 > 0
    up = pos & (c * den64 < num64)
    c[up] += 1
    down = pos & (c >= 1) & ((c - 1) * den64 >= num64)
    c[down] -= 1
    c[~pos] = 0
    return c.astype(np.int64)


def quotient(num64, den64) -> np.ndarray:
    pos = den64 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pos, num64 / np.where(pos, den64, 1.0), 0.0)


def codes_b(xb, qx, Sb) -> tuple:
    """Form B's codes: RNE(x / (qx Sx)) in float64; the clip to +-127 is a guard, counted (a real clip is a STOP)."""
    eff = (qx.astype(np.float64) * Sb)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.rint(np.where(eff > 0, xb.astype(np.float64) / np.where(eff > 0, eff, 1.0), 0.0))
    clip = int(np.count_nonzero(np.abs(c) > CODE_MAX))
    return np.clip(c, -CODE_MAX, CODE_MAX), clip


def qw_of(dw, Db) -> tuple:
    """qw = RNE(d_w / Dw) in float64; Dw = 0 gives 0; the clip to +-127 is a guard, counted (a real clip is a STOP)."""
    q = np.rint(quotient(np.asarray(dw, dtype=np.float32).astype(np.float64), Db))
    clip = int(np.count_nonzero(np.abs(q) > QW_MAX))
    return np.clip(q, -QW_MAX, QW_MAX).astype(np.int64), clip


def act_grid(xb, q, sx, s, form: str) -> dict:
    """xb [M, kb, 32] fp32 inputs, q [M, kb, 32] L4's codes, sx [M, kb] L4's fp32 scales (S1's q8_ort)."""
    M, kb, _ = xb.shape
    G, size = groups(kb, s)
    mx = sx.reshape(M, G, size).max(axis=2)
    Sx, steps = scale_up(mx, QX_MAX)
    Sb = np.repeat(Sx, size, axis=1).astype(np.float64)
    sx64 = sx.astype(np.float64)
    ratio = quotient(sx64, Sb)
    out = {"Sx": Sx, "G": G, "size": size, "steps": steps, "scale_faults": scale_faults(Sx, mx, QX_MAX)}
    if form == "B":
        c = ceil_exact(sx64, Sb, ratio)
        pos = Sb > 0
        out["ceil_faults"] = int(np.count_nonzero(pos & ~((c * Sb >= sx64) & ((c - 1) * Sb < sx64))))
        out["c_over"] = int(np.count_nonzero(c > QX_MAX))
        qx = np.maximum(1, np.minimum(c, QX_MAX))
        out["cover_faults"] = int(np.count_nonzero(qx * Sb < sx64))
        codes, out["clip"] = codes_b(xb, qx, Sb)
        live = sx > 0
        zero = live & np.all(codes == 0, axis=2)
        out["zero_code_blocks"] = int(zero.sum())
        out["live_blocks"] = int(live.sum())
    else:
        a = np.rint(ratio).astype(np.int64)
        out["c_over"] = int(np.count_nonzero(a > QX_MAX))
        out.update(ceil_faults=0, cover_faults=0, clip=0)
        qx = np.maximum(1, np.minimum(a, QX_MAX))
        codes = q
    out["qx"] = qx.astype(np.int64)
    out["codes"] = codes
    out["qx1_blocks"] = int(np.count_nonzero(qx == 1))
    out["blocks"] = int(qx.size)
    out["min_qx"] = int(qx.min())
    return out


def w_grid(dw, s) -> dict:
    """dw [N, kb] fp32 (from the GGUF's fp16, exact)."""
    N, kb = dw.shape
    G, size = groups(kb, s)
    mx = np.abs(dw).reshape(N, G, size).max(axis=2)
    Dw, steps = scale_up(mx, QW_MAX)
    qw, clip = qw_of(dw, np.repeat(Dw, size, axis=1).astype(np.float64))
    nz = dw != 0
    zeroed = nz & (qw == 0)
    live = np.abs(qw[qw != 0])
    return {"Dw": Dw, "qw": qw, "G": G, "size": size, "steps": steps, "scale_faults": scale_faults(Dw, mx, QW_MAX),
            "clip": clip, "qw0_blocks": int(zeroed.sum()), "nonzero_blocks": int(nz.sum()),
            "qw0_columns": int(zeroed.any(axis=1).sum()), "min_nonzero_abs_qw": int(live.min()) if live.size else None}


def act_stops(g: dict) -> list:
    out = []
    for k, what in (("scale_faults", "255 x Sx >= max s_x (the smallest such fp32) fails"),
                    ("ceil_faults", "c is not the exact ceiling"), ("c_over", "c or qx above 255"),
                    ("cover_faults", "qx x Sx < s_x"), ("clip", "a code clips")):
        if g.get(k):
            out.append(f"{what} on {g[k]}")
    return out


def w_stops(g: dict) -> list:
    out = []
    if g["scale_faults"]:
        out.append(f"127 x Dw >= max abs(d_w) (the smallest such fp32) fails on {g['scale_faults']}")
    if g["clip"]:
        out.append(f"a qw clips on {g['clip']}")
    return out


def act_mech(g: dict, form: str) -> dict:
    m = {"qx1_blocks": g["qx1_blocks"], "qx1_fraction": g["qx1_blocks"] / g["blocks"], "min_qx": g["min_qx"],
         "sx_steps": g["steps"]}
    if form == "B":
        m["zero_code_blocks"] = g["zero_code_blocks"]
        m["zero_code_fraction"] = g["zero_code_blocks"] / g["live_blocks"] if g["live_blocks"] else None
    return m


def w_mech(g: dict) -> dict:
    return {"qw0_blocks": g["qw0_blocks"], "nonzero_dw_blocks": g["nonzero_blocks"],
            "qw0_fraction": g["qw0_blocks"] / g["nonzero_blocks"] if g["nonzero_blocks"] else None,
            "qw0_columns": g["qw0_columns"], "min_nonzero_abs_qw": g["min_nonzero_abs_qw"], "dw_steps": g["steps"]}


# ---------------------------------------------------------------- the integer sum and the flush (sections 2.4, 2.5)

def int_sums(codes_act, codes_w, qx: dict, qw: dict, specs: dict) -> dict:
    """I [G, M, N] int64 per S in specs ({s: (G, size)}): one exact int_dot per block, shared by every S."""
    M, kb, _ = codes_act.shape
    N = codes_w.shape[0]
    I = {s: np.zeros((G, M, N), dtype=np.int64) for s, (G, _) in specs.items()}
    for b in range(kb):
        i = u6e.int_dot(codes_act[:, b, :], codes_w[:, b, :])
        for s, (_, size) in specs.items():
            I[s][b // size] += i * (qx[s][:, b, None] * qw[s][None, :, b])
    return I


def flush(I, Sx, Dw, form: str, order=None) -> np.ndarray:
    """I [G, M, N] int64, Sx [M, G] and Dw [N, G] fp32. y starts at 0 in fp32 and takes one flush per superblock,
    in increasing K, as f2_flush_b.cc compiles it. order defaults to FLUSH_ORDERS[form]; the selftest passes another."""
    t = FLUSH_ORDERS[form]["t"]
    o = order or FLUSH_ORDERS[form]
    G, M, N = I.shape
    y = np.zeros((M, N), dtype=np.float32)
    for g in range(G):
        h = u6e.split(f20b.rne(I[g]), t)
        a = [p[:, None] for p in u6e.split(Sx[:, g], t)]
        w = [p[None, :] for p in u6e.split(Dw[:, g], t)]
        (j, k), rest = o["P"][0], o["P"][1:]
        P = (a[j - 1] * w[k - 1]).astype(np.float32)
        for j, k in rest:
            P = (P + a[j - 1] * w[k - 1]).astype(np.float32)
        p = u6e.split(P, t)
        for u, v in o["Y"]:
            y = (y + h[u - 1] * p[v - 1]).astype(np.float32)
    return y


def flush_scalar(I, Sx, Dw, form: str, order: dict) -> np.ndarray:
    """An independent form of flush(), for the selftest only: Python loops over (m, n, g), numpy float32 scalars
    (each op rounds to fp32), bf16 RNE through ml_dtypes, and the order read from the source, term by term."""
    import ml_dtypes
    f32 = np.float32
    t = FLUSH_ORDERS[form]["t"]

    def b16(v):
        return f32(np.float32(v).astype(ml_dtypes.bfloat16).astype(np.float32))

    def pieces(v, n):
        out, r = [], f32(v)
        for _ in range(n):
            p = b16(r)
            out.append(p)
            r = f32(r - p)
        return out
    G, M, N = I.shape
    y = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            acc = f32(0.0)
            for g in range(G):
                h = pieces(f32(float(int(I[g, m, n]))), t)
                a, w = pieces(Sx[m, g], t), pieces(Dw[n, g], t)
                (j, k), rest = order["P"][0], order["P"][1:]
                P = f32(a[j - 1] * w[k - 1])
                for j, k in rest:
                    P = f32(P + f32(a[j - 1] * w[k - 1]))
                p = pieces(P, t)
                for u, v in order["Y"]:
                    acc = f32(acc + f32(h[u - 1] * p[v - 1]))
            y[m, n] = acc
    return y


def f64_form(I, Sx, Dw) -> np.ndarray:
    """F64_F2: the grid's value in float64, with no flush rounding (section 2.6)."""
    G, M, N = I.shape
    y = np.zeros((M, N))
    for g in range(G):
        y += I[g].astype(np.float64) * (Sx[:, g].astype(np.float64)[:, None] * Dw[:, g].astype(np.float64)[None, :])
    return y


def conv_mismatches(I) -> int:
    """F2-0b's model of the compiled conversion against RNE(I), bit for bit (section 2.5)."""
    v = np.asarray(I, dtype=np.int64).ravel()
    return int(np.count_nonzero(f20b.fp32_of_i(v).view(np.uint32) != f20b.rne(v).view(np.uint32)))


def rel_l2(y, ref) -> float:
    return u6e.rel_l2(y, ref)


# ---------------------------------------------------------------- the rules

def acceptable(r_f2: dict, r_r: dict, factor: float) -> bool:
    return all(r_f2[c] <= factor * r_r[c] for c in r_r)


def pick(accept: dict):
    """accept: arm -> ACCEPTABLE at F2_FACTOR. The first deciding arm, in DECIDING's order, else None (NONE)."""
    return next((a for a in DECIDING if accept.get(a)), None)


def score_predictions(r: dict, r_r: dict, qw0_row: dict) -> dict:
    """r: arm -> {case: r_F2}; r_r: case -> r_R; qw0_row: case -> the ROW qw = 0 fraction. Cases are 'L.name'."""
    d1 = max(abs(r[(f, "B2", s)][c] - r[(f, "B3", s)][c]) / r_r[c] for f in ACT_FORMS for s in S_SET for c in r_r)
    p2 = [(fl, c) for fl in FLUSH_FORMS for c in r_r if not r[("B", fl, "ROW")][c] <= r[("A", fl, "ROW")][c]]
    p3 = [(fl, c) for fl in FLUSH_FORMS for c in r_r
          if not r[("B", fl, "ROW")][c] >= r[("B", fl, 16)][c] >= r[("B", fl, 8)][c]]
    layers = sorted({c.split(".")[0] for c in qw0_row}, key=int)
    top = {L: max((qw0_row[c], c.split(".")[1]) for c in qw0_row if c.split(".")[0] == L) for L in layers}
    p4 = {L: {"down": qw0_row[f"{L}.down"], "largest": top[L][1], "largest_fraction": top[L][0]} for L in layers}
    held4 = all(qw0_row[f"{L}.down"] >= top[L][0] for L in layers)
    return {"F2E-P1": {"held": d1 <= 0.01, "worst_abs_diff_over_r_R": d1},
            "F2E-P2": {"held": not p2, "violations": len(p2), "first": p2[:3]},
            "F2E-P3": {"held": not p3, "violations": len(p3), "first": p3[:3]},
            "F2E-P4": {"held": held4, "per_layer": p4}}


def ef2_check(lines) -> dict:
    """Fix F1: F2-0b's integer loop bundles and E_core from its log, E_F2 exact, and the log's 4-dp print."""
    bundles, out = {}, None
    for ln in lines:
        tag, _, rest = ln.partition(" ")
        if tag == "CASE_JSON":
            c = json.loads(rest)
            if c.get("case") in ("FLUSH_B2", "FLUSH_B3"):
                bundles[c["case"][6:]] = c.get("loop_bundles")
        elif tag == "OUTPUT_JSON":
            out = json.loads(rest)
    why = []
    if out is None:
        return {"ok": False, "why": ["no OUTPUT_JSON"]}
    e_core, row = out.get("E_core"), out.get("row_blocks")
    if type(e_core) is not int or e_core != E_CORE_PIN:
        why.append(f"E_core {e_core!r} is not {E_CORE_PIN}")
    if type(row) is not int or row != ROW_PIN:
        why.append(f"row_blocks {row!r} is not {ROW_PIN}")
    exact, printed = {}, {}
    for fl, pin in FLUSH_PIN.items():
        form = (out.get("forms") or {}).get(f"FLUSH_{fl}") or {}
        lb, fx = bundles.get(fl), form.get("FLUSH")
        if type(lb) is not int or lb != pin:
            why.append(f"FLUSH_{fl} loop_bundles {lb!r} is not {pin}")
        if type(fx) is not int or fx != pin:
            why.append(f"FLUSH_{fl} FLUSH {fx!r} is not {pin}")
        e = Fraction(E_CORE_PIN) + Fraction(pin, ROW_PIN)
        exact[fl] = str(e)
        printed[fl] = form.get("E_F2")
        if printed[fl] != round(float(e), 4):
            why.append(f"FLUSH_{fl} printed E_F2 {printed[fl]!r} is not the exact {float(e)} at 4 dp")
        if form.get("allowed_s") != ["ROW"]:
            why.append(f"FLUSH_{fl} allowed_s {form.get('allowed_s')!r} is not ['ROW']")
    return {"ok": not why, "why": why, "exact": exact, "exact_float": {k: float(Fraction(v)) for k, v in exact.items()},
            "printed": printed, "bundles": bundles, "E_core": e_core, "row_blocks": row}


def u6e_logged(lines) -> dict:
    out = {}
    for ln in lines:
        tag, _, rest = ln.partition(" ")
        if tag == "CASE_JSON":
            c = json.loads(rest)
            out[f"{c['layer']}.{c['name']}"] = c["l4_vs_l0"]
    return out


def anchor_rel(ours: float, logged: float) -> float:
    return abs(ours - logged) / abs(logged)


def order_check(text=None) -> dict:
    t = f20b.source_text(text)
    got = {fl: f20b.branch_order(t, fl) for fl in ("B3", "B2")}
    ok = all(got[fl]["n_first"] == 1 and got[fl]["P"] == FLUSH_ORDERS[fl]["P"] and got[fl]["Y"] == FLUSH_ORDERS[fl]["Y"]
             for fl in got)
    return {"ok": ok, "source": {fl: {"P": g["P"], "Y": g["Y"]} for fl, g in got.items()},
            "emulation": {fl: {"P": v["P"], "Y": v["Y"]} for fl, v in FLUSH_ORDERS.items()}}


def pin_rows() -> list:
    rows = []
    for what, p, sha in PINS:
        cur = lf_sha(p) if p.exists() else None
        rows.append({"what": what, "file": p.relative_to(ROOT).as_posix(), "pinned": sha, "lf_sha256": cur,
                     "equal": cur == sha})
    return rows


def up2(x: float) -> float:
    return u6e.up2(x)


# ---------------------------------------------------------------- the selftest (synthetic only)

def synth(rng, M, kb, N, outlier=True):
    """Synthetic inputs as U6-E's selftest makes them: an outlier row, Q4_0-like codes and fp16 d_w."""
    X = (rng.standard_normal((M, kb * 32)) * np.exp(rng.standard_normal((M, 1)))).astype(np.float32)
    if outlier:
        X[0, :64] *= 300.0
    codes = rng.integers(0, 16, (N, kb, 32)).astype(np.uint8)
    d = (np.abs(rng.standard_normal(N * kb)) * 0.01 + 1e-4).astype(np.float16).view(np.uint16)
    q, sx = s1.q8_ort(X)
    return X, s1.blocks(X), q, sx, codes, gc.f16_to_f32(d).reshape(N, kb)


def emulate(xb, q, sx, codes, dw, form, s, fl) -> tuple:
    """The pipeline for one arm on small inputs (the selftest's), from the same functions run() uses."""
    ag, wg = act_grid(xb, q, sx, s, form), w_grid(dw, s)
    I = int_sums(ag["codes"], codes, {s: ag["qx"]}, {s: wg["qw"]}, {s: (ag["G"], ag["size"])})[s]
    return flush(I, ag["Sx"], wg["Dw"], fl), I, ag, wg


def selftest() -> int:
    header("F2-E SELFTEST (synthetic only; no model, no real input, no ORT, no chip).")
    rng = np.random.default_rng(s1.SEED)
    ok = True

    def check(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        s1.say("SELFTEST_JSON", {"check": name, "ok": bool(cond), **info})

    # 1. the imported tools are the pinned ones, and the source's orders are the emulation's
    shas = {"u6e": lf_sha(U6E_TOOL), "f2_0b": lf_sha(F20B_TOOL), "source": lf_sha(SOURCE)}
    check("the imported tools and the source are the pinned ones (U6-E b88d23d3, F2-0b 7a953146, source caa62ce0)",
          shas == {"u6e": U6E_TOOL_SHA, "f2_0b": F20B_TOOL_SHA, "source": SOURCE_SHA}, lf_sha=shas)
    oc = order_check()
    check("the source's P and Y orders for B3 and B2 equal the emulation's", oc["ok"], source=oc["source"])
    text = f20b.source_text()
    tail = "    Y_TERM(2, 1);\n    Y_TERM(1, 2);\n    Y_TERM(1, 1);\n    aie::store_v"
    swap = "    Y_TERM(1, 2);\n    Y_TERM(2, 1);\n    Y_TERM(1, 1);\n    aie::store_v"
    at = text.rfind(tail)                                          # the last occurrence: B2's branch
    bad = text[:at] + swap + text[at + len(tail):] if at >= 0 else text
    check("a source copy with two of B2's Y terms swapped fails the order check",
          bad != text and not order_check(bad)["ok"]
          and order_check(bad)["source"]["B3"]["Y"] == FLUSH_ORDERS["B3"]["Y"])

    # 2. fp32_up is the smallest fp32 at or above a float64 value (exact rationals on a sample)
    qv = np.concatenate([np.exp(rng.uniform(-40, 10, 20_000)), np.array([0.0, 1.0, 2.0 ** -20, 3.0])])
    up = fp32_up(qv)
    exact_up = all(Fraction(float(u)) >= Fraction(float(v))
                   and (u == 0 or Fraction(float(np.nextafter(u, np.float32(0)))) < Fraction(float(v)))
                   for u, v in zip(up[:3000], qv[:3000]))
    check("fp32_up is the smallest fp32 >= its float64 input", exact_up and bool(np.all(up.astype(np.float64) >= qv)),
          n=int(qv.size))

    # 3. the round-up: maxima whose nearest fp32 quotient rounds down read the round-up scale, exactly
    for den, name in ((QX_MAX, "Sx"), (QW_MAX, "Dw")):
        m = np.exp(rng.uniform(-20, 5, 50_000)).astype(np.float32)
        near = (m.astype(np.float64) / den).astype(np.float32)
        down = np.array([Fraction(float(n)) * den < Fraction(float(v)) for n, v in zip(near, m)])
        mm = m[down]
        S, steps = scale_up(mm, den)
        exact = all(Fraction(float(a)) * den >= Fraction(float(v))
                    and Fraction(float(np.nextafter(a, np.float32(0)))) * den < Fraction(float(v))
                    for a, v in zip(S, mm))
        check(f"{name}: where the nearest fp32 of max / {den} rounds down, the round-up reads the smallest fp32 with "
              f"{den} x {name} >= max, exactly", mm.size > 1000 and exact and scale_faults(S, mm, den) == 0,
              rounded_down=int(mm.size), of=int(m.size), steps=steps)

    # 4. the step: planted candidates one ulp under step up, and are counted
    mx = np.exp(rng.uniform(-10, 3, 1000)).astype(np.float32)
    good, _ = scale_up(mx, QX_MAX)
    cand = good.copy()
    plant = rng.choice(1000, 37, replace=False)
    cand[plant] = np.nextafter(cand[plant], np.float32(0))
    stepped, n = step_up(cand, mx, QX_MAX)
    check("the step: 37 planted candidates one ulp under are stepped up and counted", n == 37
          and np.array_equal(stepped.view(np.uint32), good.view(np.uint32)), steps=n)

    # 5. the round-up check's STOP: an Sx or a Dw one ulp under the true quotient is a fault
    fx = scale_faults(np.nextafter(good, np.float32(0)), mx, QX_MAX)
    gw, _ = scale_up(mx, QW_MAX)
    fw = scale_faults(np.nextafter(gw, np.float32(0)), mx, QW_MAX)
    check("an Sx or a Dw one ulp under the true quotient trips the round-up check", fx == 1000 and fw == 1000
          and scale_faults(good, mx, QX_MAX) == 0 and scale_faults(gw, mx, QW_MAX) == 0, sx_faults=fx, dw_faults=fw)

    # 6. the exact ceiling on planted edge quotients: integer quotients, and quotients a hair above an integer
    base = (np.float32(2.0 ** -10) * (1 + rng.integers(0, 64, 400) / np.float32(64))).astype(np.float32)
    k = rng.integers(1, 256, 400)
    on = (base.astype(np.float64) * k).astype(np.float32)          # k x base is exact in fp32 (<= 15 bits)
    above = np.nextafter(on, F32_UP)
    B = base.astype(np.float64)
    c_on = ceil_exact(on.astype(np.float64), B, quotient(on.astype(np.float64), B))
    c_ab = ceil_exact(above.astype(np.float64), B, quotient(above.astype(np.float64), B))
    check("the exact ceiling: k on integer quotients, k + 1 a hair above", np.array_equal(c_on, k)
          and np.array_equal(c_ab, k + 1))

    # 7. the superblocks partition each K
    part = all(groups(kb, s)[0] * groups(kb, s)[1] == kb
               and sorted(b // groups(kb, s)[1] for b in range(kb)) == [g for g in range(groups(kb, s)[0])
                                                                        for _ in range(groups(kb, s)[1])]
               for kb in (64, 80, 320) for s in S_SET)
    check("the superblocks partition K = 64, 80 and 320 blocks at ROW, 16 and 8", part)

    # the synthetic case for checks 8-14
    M, kb, N = 12, 32, 20
    X, xb, q, sx, codes, dw = synth(rng, M, kb, N)

    # 8. the grid bounds, and form B's cover and codes
    grids = {(f, s): act_grid(xb, q, sx, s, f) for f in ACT_FORMS for s in S_SET}
    wgs = {s: w_grid(dw, s) for s in S_SET}
    bounds = all(1 <= g["qx"].min() and g["qx"].max() <= QX_MAX for g in grids.values()) \
        and all(np.abs(g["qw"]).max() <= QW_MAX for g in wgs.values()) \
        and all(int(grids[(f, s)]["qx"].max()) * int(np.abs(wgs[s]["qw"]).max()) <= 32767
                for f in ACT_FORMS for s in S_SET)
    check("the grid bounds: qx in [1, 255], abs(qw) <= 127, qx x qw fits int16", bounds)
    fb = [grids[("B", s)] for s in S_SET]
    check("form B: codes within +-127, qx x Sx >= s_x on every block, c the exact ceiling, no clip, no STOP",
          all(np.abs(g["codes"]).max() <= CODE_MAX and not act_stops(g) for g in fb)
          and all(not w_stops(g) for g in wgs.values()), stops=[act_stops(g) for g in fb])

    # 9. equal block scales in form A: qx = 255, and 0 <= 255 x Sx - s < 255 x ulp(Sx) (the round-up's bound;
    #    to nearest, it was 1 ulp of s)
    #    The top 1/256 of a binade (s in [255 x 2^e, 256 x 2^e)) is where the gap can pass 1 ulp of s: planted.
    s_eq = np.concatenate([np.exp(rng.uniform(-12, 2, 48)),
                           255.5 * 2.0 ** rng.integers(-20, 2, 16)]).astype(np.float32)
    sxe = np.repeat(s_eq[:, None], 16, axis=1)
    xe = np.zeros((64, 16, 32), dtype=np.float32)
    ge = act_grid(xe, np.zeros_like(xe), sxe, "ROW", "A")
    Sx64 = ge["Sx"][:, 0].astype(np.float64)
    ulp = np.spacing(ge["Sx"][:, 0]).astype(np.float64)
    gap = 255 * Sx64 - s_eq.astype(np.float64)
    check("equal block scales, form A: qx = 255 and 0 <= 255 x Sx - s < 255 x ulp(Sx), under 2 ulp of s (the "
          "round-up's bound; the addendum's 1 ulp was the to-nearest bound, a NOTE by the gate's ruling)",
          bool(np.all(ge["qx"] == 255)) and bool(np.all(gap >= 0)) and bool(np.all(gap < 255 * ulp)),
          worst_gap_in_ulps_of_s=float(np.max(gap / np.spacing(s_eq).astype(np.float64))))

    # 10. a zero superblock gives zero, with no division by zero (a division or an invalid op raises here)
    with np.errstate(divide="raise", invalid="raise"):
        xz = xb.copy()
        xz[:, :16, :] = 0.0
        qz, sxz = s1.q8_ort(xz.reshape(M, kb * 32))
        dz = dw.copy()
        dz[:, 16:] = 0.0
        yz, Iz, agz, wgz = emulate(xz, qz, sxz, codes, dz, "B", 16, "B3")
    check("a zero superblock (all s_x zero, or all d_w zero) gives zero, with no division by zero",
          bool(np.all(Iz == 0)) and bool(np.all(yz == 0)) and bool(np.all(agz["Sx"][:, 0] == 0))
          and bool(np.all(agz["qx"][:, :16] == 1)) and bool(np.all(agz["codes"][:, :16] == 0))
          and bool(np.all(wgz["Dw"][:, 1] == 0)) and bool(np.all(wgz["qw"][:, 16:] == 0)))

    # 11. I exact against Python integers
    ga, gw16 = grids[("B", 16)], wgs[16]
    I16 = int_sums(ga["codes"], codes, {16: ga["qx"]}, {16: gw16["qw"]}, {16: (ga["G"], ga["size"])})[16]
    ref = [[[sum(sum(int(ga["codes"][m, b, e]) * (int(codes[n, b, e]) - 8) for e in range(32))
                 * int(ga["qx"][m, b]) * int(gw16["qw"][n, b]) for b in range(g * 16, g * 16 + 16))
             for n in range(4)] for m in range(3)] for g in range(2)]
    check("I exact against Python integers", np.array_equal(I16[:, :3, :4], np.array(ref, dtype=np.int64)))

    # 12. a planted wrong qw is seen, in its column only
    qw_bad = gw16["qw"].copy()
    qw_bad[7, 5] += 1 if qw_bad[7, 5] < QW_MAX else -1
    Ib = int_sums(ga["codes"], codes, {16: ga["qx"]}, {16: qw_bad}, {16: (ga["G"], ga["size"])})[16]
    yb = flush(Ib, ga["Sx"], gw16["Dw"], "B3")
    y0 = flush(I16, ga["Sx"], gw16["Dw"], "B3")
    cols = sorted(set(np.nonzero(np.any(yb != y0, axis=0))[0].tolist()))
    check("a planted wrong qw is seen, in its column only", cols == [7], columns=cols)

    # 13. the flush's order: flush() equals a scalar walk of the source's order, bit for bit; reversed changes bits
    src = {fl: {"P": f20b.branch_order(text, fl)["P"], "Y": f20b.branch_order(text, fl)["Y"]} for fl in FLUSH_FORMS}
    Ism = I16[:, :3, :5]
    same = {fl: bool(np.array_equal(flush(Ism, ga["Sx"][:3], gw16["Dw"][:5], fl).view(np.uint32),
                                    flush_scalar(Ism, ga["Sx"][:3], gw16["Dw"][:5], fl, src[fl]).view(np.uint32)))
            for fl in FLUSH_FORMS}
    check("flush equals the scalar walk of the source's order, bit for bit", all(same.values()), by_form=same)
    Iw = (rng.integers(-2 ** 38, 2 ** 38, (3, 40, 50))).astype(np.int64)
    Sxw = np.exp(rng.uniform(-9, -3, (40, 3))).astype(np.float32)
    Dww = np.exp(rng.uniform(-8, -3, (50, 3))).astype(np.float32)
    differ = {fl: int(np.count_nonzero(flush(Iw, Sxw, Dww, fl).view(np.uint32) != flush(
        Iw, Sxw, Dww, fl, {"P": FLUSH_ORDERS[fl]["P"][::-1], "Y": FLUSH_ORDERS[fl]["Y"][::-1]}).view(np.uint32)))
        for fl in FLUSH_FORMS}
    check("the reversed order changes some bits (the order check has teeth)", all(v > 0 for v in differ.values()),
          elements_differing=differ, of=40 * 50)

    # 14. the flush's error (fix F2): B3 within 8 x 2^-24 per sample; B2's worst sample above it
    n1 = 600
    mag = np.floor(2.0 ** rng.uniform(0, 39, (n1, n1)))
    Ie = (mag * rng.choice([-1, 1], (n1, n1))).astype(np.int64)[None]
    Sxe = np.exp(rng.uniform(np.log(1e-8), np.log(1e-1), (n1, 1))).astype(np.float32)
    Dwe = np.exp(rng.uniform(np.log(1e-6), np.log(1e-1), (n1, 1))).astype(np.float32)
    F = f64_form(Ie, Sxe, Dwe)
    e3 = np.abs(flush(Ie, Sxe, Dwe, "B3").astype(np.float64) - F)
    e2 = np.abs(flush(Ie, Sxe, Dwe, "B2").astype(np.float64) - F)
    w3 = float(np.max(e3 / np.abs(F)))
    w2 = float(np.max(e2 / np.abs(F)))
    check(f"B3 within {K_FLOOR} x 2^-24 of float64 on every sample; B2's worst sample above it (in aggregate)",
          bool(np.all(e3 <= FLOOR * np.abs(F))) and w2 > FLOOR, samples=n1 * n1,
          b3_worst_in_2_24=w3 / 2.0 ** -24, b2_worst_in_2_24=w2 / 2.0 ** -24)

    # 15. the conversion: fp32(I) = RNE(I), and F2-0b's model of the compiled conversion agrees
    vals = np.concatenate([np.array(f20b.conv_values(), dtype=np.int64),
                           rng.integers(-f20b.F2_I_MAX, f20b.F2_I_MAX + 1, 100_000)])
    rn = np.array([np.float32(float(int(v))) for v in vals[:2000]], dtype=np.float32)
    check("fp32(I) = RNE(I), and fp32_of_i agrees bit for bit", conv_mismatches(vals) == 0
          and np.array_equal(rn.view(np.uint32), f20b.rne(vals[:2000]).view(np.uint32)), n=int(vals.size))

    # 16. the STOPs trip: a code clip, a qw clip
    mxb = np.float32(3.0)
    sx_bad = np.full((1, 1), mxb * np.float32(0.99) / np.float32(127), dtype=np.float32)
    xc = np.full((1, 1, 32), mxb, dtype=np.float32)
    gc_ = act_grid(xc, np.zeros_like(xc), sx_bad, "ROW", "B")
    _, qclip = qw_of(np.array([[3.0]], dtype=np.float32), np.array([[0.99 * 3.0 / 127]]))
    check("a planted s_x of 0.99 x amax / 127 trips the code-clip STOP; a Dw of 0.99 x max / 127 the qw-clip STOP",
          gc_["clip"] > 0 and any("clips" in s for s in act_stops(gc_)) and qclip > 0,
          code_clips=gc_["clip"], qw_clips=qclip)

    # 17. the counters on a planted grid: known qw = 0, qx = 1 and all-zero-code blocks
    Mp, kbp, Np = 6, 16, 5
    xp = (rng.uniform(0.5, 1.0, (Mp, kbp, 32)) * rng.choice([-1, 1], (Mp, kbp, 32))).astype(np.float32)
    xp[:, :, 0] = 1.0                                             # every block's amax is 1
    tiny = [(0, 3), (2, 9), (5, 14)]
    for m, b in tiny:
        xp[m, b] *= np.float32(1e-6)                              # s_x far under Sx / 254: qx = 1, codes all 0
    qp, sxp = s1.q8_ort(xp.reshape(Mp, kbp * 32))
    dwp = rng.uniform(0.5, 1.0, (Np, kbp)).astype(np.float32)
    dwp[:, 0] = 1.0
    zeroed = [(1, 4), (3, 11)]
    for n, b in zeroed:
        dwp[n, b] = np.float32(1.0 / 127 / 4)                     # |d_w| / Dw = 0.25: qw = 0
    gp, wp = act_grid(s1.blocks(xp.reshape(Mp, kbp * 32)), qp, sxp, "ROW", "B"), w_grid(dwp, "ROW")
    mb, mw = act_mech(gp, "B"), w_mech(wp)
    check("the counters on a planted grid: 3 qx = 1 and 3 all-zero-code blocks, 2 qw = 0 blocks in 2 columns",
          mb["qx1_blocks"] == 3 and mb["zero_code_blocks"] == 3 and mw["qw0_blocks"] == 2 and mw["qw0_columns"] == 2,
          act=mb, w=mw)

    # 18. the rules on synthetic numbers: the factor, the pick order, report-only arms never picked
    rr = {"a": 1.0, "b": 2.0}
    check("the rule", acceptable({"a": 2.0, "b": 4.0}, rr, 2.0) and not acceptable({"a": 2.01, "b": 1.0}, rr, 2.0)
          and acceptable({"a": 1.1, "b": 2.2}, rr, 1.12) and not acceptable({"a": 1.2, "b": 2.2}, rr, 1.12))
    yes = {a: True for a in ARMS}
    check("the pick: B2 first, B3 second, NONE, and a report-only arm is never picked",
          pick(yes) == ("B", "B2", "ROW") and pick({**yes, ("B", "B2", "ROW"): False}) == ("B", "B3", "ROW")
          and pick({a: a not in DECIDING for a in ARMS}) is None)
    check("up2", all(math.isclose(up2(v), want, rel_tol=1e-12) for v, want in ((3.44e-7, 3.5e-7), (1.2e-6, 1.2e-6),
                                                                              (9.91e-6, 1e-5), (1.001e-6, 1.1e-6))))

    # 19. the predictions' scoring: HELD and NOT HELD both read
    cases = [f"{L}.{n}" for L in LAYERS for n in ("q", "down")]
    r_r = {c: 1e-2 for c in cases}
    base_r = {a: {c: 1e-2 * (1.5 if a[0] == "A" else 1.0) * (1.0 if a[2] == "ROW" else 0.9 if a[2] == 16 else 0.8)
                  for c in cases} for a in ARMS}
    qw0 = {c: (0.3 if c.endswith("down") else 0.1) for c in cases}
    good_p = score_predictions(base_r, r_r, qw0)
    bad_r = {a: dict(v) for a, v in base_r.items()}
    bad_r[("B", "B3", "ROW")]["0.q"] *= 2.0                       # P1 (B2 against B3) and P2 (B above A)
    bad_r[("B", "B2", 8)]["16.down"] = 1.0
    bad_p = score_predictions(bad_r, r_r, {**qw0, "33.q": 0.9})
    check("the predictions' scoring: all HELD on one set, all NOT HELD on another",
          all(v["held"] for v in good_p.values()) and not any(v["held"] for v in bad_p.values()),
          held=[k for k, v in good_p.items() if v["held"]], not_held=[k for k, v in bad_p.items() if not v["held"]])

    # 20. fix F1: E_F2 from F2-0b's integers, exact, and the printed 4-dp values
    def lines(b2=82, b3=116, p2=23.2812, s=None):
        return ['CASE_JSON {"case": "F2_CORE", "loop_bundles": 29}',
                f'CASE_JSON {{"case": "FLUSH_B3", "loop_bundles": {b3}}}',
                f'CASE_JSON {{"case": "FLUSH_B2", "loop_bundles": {b2}}}',
                "OUTPUT_JSON " + json.dumps({"E_core": 22, "row_blocks": 64, "forms": {
                    "FLUSH_B3": {"FLUSH": b3, "E_F2": 23.8125, "allowed_s": s or ["ROW"]},
                    "FLUSH_B2": {"FLUSH": b2, "E_F2": p2, "allowed_s": ["ROW"]}}})]
    e_ok = ef2_check(lines())
    check("E_F2: exact from the integers (23.28125, 23.8125), the 4-dp print agrees; a wrong FLUSH, a wrong print "
          "or a wrong allowed S is a STOP", e_ok["ok"] and e_ok["exact"] == {"B2": "745/32", "B3": "381/16"}
          and not ef2_check(lines(b2=83))["ok"] and not ef2_check(lines(p2=23.2813))["ok"]
          and not ef2_check(lines(s=["ROW", "16"]))["ok"], exact=e_ok["exact"])

    # 21. U6-E's anchor: the parser and the tolerance
    lg = u6e_logged(['CASE_JSON {"layer": 0, "name": "q", "l4_vs_l0": 0.005366717576963061}'])
    check("U6-E's anchor: parsed at full precision; 1e-7 relative passes, 2e-6 fails",
          lg == {"0.q": 0.005366717576963061} and anchor_rel(0.005366717576963061 * (1 + 1e-7), lg["0.q"]) <= ANCHOR_TOL
          and anchor_rel(0.005366717576963061 * (1 + 2e-6), lg["0.q"]) > ANCHOR_TOL)

    # 22. the gate's fix: a STOP planted in the last case prints no CASE_JSON; progress lines carry no F2 number
    jobs = [(L, n) for L in LAYERS for n in ("q", "k", "v", "o", "gate", "up", "down")]

    def fake(job, stop_at=None):
        if job == stop_at:
            raise Stop("planted in the last case")
        return {"layer": job[0], "name": job[1], "arms": {"B/B2/ROW": {"vs_l0": 1e-2}}, "_arms": {}}
    got, got_ok, stopped = [], [], False
    try:
        hold_cases(jobs, lambda j: fake(j, jobs[-1]), lambda k, p: got.append((k, p)))
    except Stop:
        stopped = True
    hold_cases(jobs, fake, lambda k, p: got_ok.append((k, p)))
    prog = re.compile(r"^F2-E CASE \d+ of 21 done: layer \d+ \w+ \(\d+\.\d s\)$")
    check("a STOP planted in the last case prints no CASE_JSON, only 20 progress lines with no F2 number; with no "
          "STOP, 21 progress lines, then 21 CASE_JSON in case order",
          stopped and [k for k, _ in got] == ["PROGRESS"] * 20 and all(prog.match(p) for _, p in got)
          and [k for k, _ in got_ok] == ["PROGRESS"] * 21 + ["CASE_JSON"] * 21
          and [(p["layer"], p["name"]) for k, p in got_ok if k == "CASE_JSON"] == jobs
          and all("_arms" not in p for k, p in got_ok if k == "CASE_JSON"),
          emitted_on_stop=[k for k, _ in got].count("CASE_JSON"))

    # 23. determinism
    y1, *_ = emulate(xb, q, sx, codes, dw, "B", "ROW", "B2")
    y2, *_ = emulate(xb, q, sx, codes, dw, "B", "ROW", "B2")
    check("deterministic", np.array_equal(y1.view(np.uint32), y2.view(np.uint32)))

    s1.say("MEM_JSON", {"child": "F2-E selftest", **s1.own_memory()})
    print("SELFTEST OK" if ok else "SELFTEST FAIL", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- pins and the run

def pins() -> int:
    header("F2-E PINS (hashes and sizes only; nothing is computed).")
    s1.say("PIN_JSON", pin_rows())
    mm, by = s1.open_gguf(hash_check=False)
    g = gc.GGUF_PIN
    s1.say("GGUF_JSON", {"file": g["file"], "size_pinned": g["size"], "sha256_pinned": g["sha256"],
                         "size_equal": True, "sha256_checked": False,
                         "note": "open_gguf checked the size; run mode hashes the whole file"})
    xs, ws = u6e.current_pins(mm, by)
    s1.say("X16_SHA_JSON", {"current": xs, "equal_u6e_pins": xs == u6e.X16_SHA})
    s1.say("W_SHA_JSON", {"current": ws, "equal_u6e_pins": ws == u6e.W_SHA})
    s1.say("ORT_JSON", {"version": u6e.ort_version(), "pinned": u6e.ORT_VERSION})
    s1.say("MEM_JSON", {"child": "F2-E pins", **s1.own_memory()})
    print("PINS DONE", flush=True)
    return 0


def run_case(L, name, K, N, X, codes, d, y0, y4, r_r) -> dict:
    """One case: the grids, the integer sums (form A once for every S, form B once per S), the 12 arms."""
    t0 = time.time()
    rows = np.asarray(ROWS)
    Xr = np.ascontiguousarray(X[rows])
    xb = s1.blocks(Xr)
    q, sx = s1.q8_ort(Xr)
    dw = gc.f16_to_f32(d).reshape(N, K // 32)
    kb = K // 32
    f64_l4 = s1.q4_dot(q, sx, codes, d)
    ql, sl = s1.q8_llama(Xr)
    yl = s1.q4_dot(ql, sl, codes, d)
    specs = {s: groups(kb, s) for s in S_SET}
    wg = {s: w_grid(dw, s) for s in S_SET}
    ag = {(f, s): act_grid(xb, q, sx, s, f) for f in ACT_FORMS for s in S_SET}
    stops = {f"w/{s}": w_stops(g) for s, g in wg.items()} | {f"x/{f}/{s}": act_stops(g) for (f, s), g in ag.items()}
    stops = {k: v for k, v in stops.items() if v}
    if stops:
        raise Stop(f"layer {L} {name}: {stops}")
    arms, grid = {}, {}

    def arms_of(form, s, I):
        if np.abs(I).max(initial=0) >= I_BOUND:
            raise Stop(f"layer {L} {name} {form}/{s}: abs(I) reaches 2^39")
        mis = conv_mismatches(I)
        if mis:
            raise Stop(f"layer {L} {name} {form}/{s}: fp32_of_i differs from RNE(I) on {mis} values")
        Sx, Dw = ag[(form, s)]["Sx"], wg[s]["Dw"]
        F = f64_form(I, Sx, Dw)
        grid[f"{form}/{s}"] = {"f64_vs_l0": rel_l2(F, y0), "f64_vs_l4": rel_l2(F, y4)}
        for fl in FLUSH_FORMS:
            y = flush(I, Sx, Dw, fl)
            if not np.isfinite(y).all():
                raise Stop(f"layer {L} {name} {form}/{fl}/{s}: a non-finite output")
            v0 = rel_l2(y, y0)
            arms[(form, fl, s)] = {"vs_l0": v0, "vs_l4": rel_l2(y, y4), "vs_f64": rel_l2(y, F), "ratio": v0 / r_r}

    IA = int_sums(q, codes, {s: ag[("A", s)]["qx"] for s in S_SET}, {s: wg[s]["qw"] for s in S_SET}, specs)
    for s in S_SET:
        arms_of("A", s, IA[s])
    del IA
    for s in S_SET:
        IB = int_sums(ag[("B", s)]["codes"], codes, {s: ag[("B", s)]["qx"]}, {s: wg[s]["qw"]}, {s: specs[s]})[s]
        arms_of("B", s, IB)
        del IB
    return {"layer": L, "name": name, "K": K, "N": N, "blocks": kb, "rows": len(ROWS), "r_R": r_r,
            "l4_vs_f64": rel_l2(y4, f64_l4), "llama_vs_l4": rel_l2(yl, y4), "llama_vs_l0": rel_l2(yl, y0),
            "arms": {arm_key(a): v for a, v in arms.items()}, "grid": grid,
            "mech_w": {str(s): w_mech(g) for s, g in wg.items()},
            "mech_x": {f"{f}/{s}": act_mech(g, f) for (f, s), g in ag.items()},
            "seconds": round(time.time() - t0, 1), "_arms": arms}


def emit(kind: str, payload) -> None:
    """PROGRESS lines carry no F2 number; CASE_JSON lines carry the arms."""
    if kind == "PROGRESS":
        print(payload, flush=True)
    else:
        s1.say(kind, payload)


def hold_cases(jobs, do_case, out) -> list:
    """Pass 2's order (the gate's fix; the addendum, section 6: no F2 number before every check has passed). Each
    job runs its checks (a Stop propagates); meanwhile only a progress line with no F2 number goes out. Every
    CASE_JSON goes out, in case order, once all the jobs have passed."""
    held = []
    for k, job in enumerate(jobs, 1):
        t0 = time.time()
        held.append(do_case(job))
        out("PROGRESS", f"F2-E CASE {k} of {len(jobs)} done: layer {job[0]} {job[1]} ({time.time() - t0:.1f} s)")
    for case in held:
        out("CASE_JSON", {k: v for k, v in case.items() if not k.startswith("_")})
    return held


def run() -> int:
    header("F2-E RUN (hybrid F2), CPU only.")
    try:
        return run_body()
    except Stop as e:
        print(f"F2-E STOP: {e}", flush=True)
        return 2


def run_body() -> int:
    rows_p = pin_rows()
    s1.say("PIN_JSON", rows_p)
    if not all(r["equal"] for r in rows_p):
        raise Stop("a pinned file is missing or differs: " + ", ".join(r["what"] for r in rows_p if not r["equal"]))
    ef = ef2_check(F20B_LOG.read_text(encoding="utf-8").splitlines())
    s1.say("EF2_JSON", ef)
    if not ef["ok"]:
        raise Stop(f"F2-0b's E_F2 inputs: {ef['why']}")
    oc = order_check()
    s1.say("ORDER_JSON", oc)
    if not oc["ok"]:
        raise Stop("the source's orders differ from the emulation's")
    s1.start_gate("F2-E")
    if u6e.ort_version() != u6e.ORT_VERSION:
        raise Stop(f"onnxruntime {u6e.ort_version()} is not {u6e.ORT_VERSION}")
    mm, by = s1.open_gguf(hash_check=True)
    xs, ws = u6e.current_pins(mm, by)
    if xs != u6e.X16_SHA or ws != u6e.W_SHA:
        s1.say("PIN_DIFF_JSON", {"x16": {k: v for k, v in xs.items() if u6e.X16_SHA.get(k) != v},
                                 "w": {k: v for k, v in ws.items() if u6e.W_SHA.get(k) != v}})
        raise Stop("an input differs from U6-E's pin")
    s1.say("INPUT_PINS_JSON", {"gguf": "size and sha256 equal the pin", "x16": "equal", "w": "equal",
                               "ort": u6e.ORT_VERSION})
    X16 = {x: np.load(s1.CHECK / f"x16_{x}.npy") for x in p3c.INPUTS}
    rows = np.asarray(ROWS)
    logged = u6e_logged(U6E_LOG.read_text(encoding="utf-8").splitlines())

    # pass 1: L0 and L4, C2's anchor and U6-E's anchor, before any F2 number
    ref, c2_ok, worst_rel = {}, True, 0.0
    for L in LAYERS:
        for name, gg, K, N, x in p3c.LINEARS:
            codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
            y4f = u6e.nb_run(X16[x], codes, d, K, N, 4)
            if L == 16:
                qf, sf = s1.q8_ort(X16[x])
                v = rel_l2(y4f, s1.q4_dot(qf, sf, codes, d))
                prior = s1.CHECK / f"c2_nb4_{name}.npy"
                bits = bool(np.array_equal(np.load(prior).view(np.uint32), y4f.view(np.uint32))) \
                    if prior.exists() else False
                good = bits and abs(v - u6e.C2_LOGGED[name]) <= u6e.C2_REL_TOL * u6e.C2_LOGGED[name]
                c2_ok &= good
                s1.say("C2_ANCHOR_JSON", {"name": name, "nb4_vs_ort_form": v, "logged": u6e.C2_LOGGED[name],
                                          "l4_bits_equal_s1_c2_output": bits, "ok": good})
            y4 = y4f[rows]
            y0 = u6e.nb_run(X16[x], codes, d, K, N, 0)[rows]
            key = f"{L}.{name}"
            r_r = rel_l2(y4, y0)
            rd = anchor_rel(r_r, logged[key]) if key in logged else float("inf")
            worst_rel = max(worst_rel, rd)
            ref[key] = (y0, y4, r_r)
            s1.say("REF_JSON", {"case": key, "l4_vs_l0": r_r, "u6e_logged": logged.get(key), "rel_diff": rd,
                                "ok": rd <= ANCHOR_TOL and r_r > 0.0})
    n_ok = sum(1 for k, (_, _, r) in ref.items() if k in logged and anchor_rel(r, logged[k]) <= ANCHOR_TOL and r > 0)
    s1.say("ANCHORS_JSON", {"c2_ok": c2_ok, "u6e_reproduced": n_ok, "of": len(ref), "worst_rel_diff": worst_rel,
                            "tol": ANCHOR_TOL})
    if not c2_ok:
        raise Stop("C2's anchor did not reproduce")
    if n_ok != len(ref) or len(ref) != 21:
        raise Stop(f"U6-E's l4_vs_l0 reproduced on {n_ok} of {len(ref)} cases (worst relative difference "
                   f"{worst_rel:.3e})")
    print(f"ANCHORS OK: C2 bit-equal and reproduced on all seven linears; U6-E's l4_vs_l0 reproduced on 21 of 21 "
          f"(worst relative difference {worst_rel:.3e})", flush=True)

    # pass 2: the arms. Every CASE_JSON is held until all 21 cases have passed their checks (the gate's fix)
    jobs = [(L, name, gg, K, N, x) for L in LAYERS for name, gg, K, N, x in p3c.LINEARS]

    def do_case(job):
        L, name, gg, K, N, x = job
        codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
        y0, y4, rr = ref[f"{L}.{name}"]
        return run_case(L, name, K, N, X16[x], codes, d, y0, y4, rr)
    held = hold_cases(jobs, do_case, emit)
    r_f2 = {a: {} for a in ARMS}
    vs_f64 = {a: {} for a in ARMS}
    r_r, qw0_row, qw0_line = {}, {}, {}
    for case in held:
        key = f"{case['layer']}.{case['name']}"
        r_r[key] = case["r_R"]
        for a, v in case["_arms"].items():
            r_f2[a][key] = v["vs_l0"]
            vs_f64[a][key] = v["vs_f64"]
        m = case["mech_w"]["ROW"]
        qw0_row[key] = m["qw0_fraction"] or 0.0
        qw0_line[key] = m
    accept = {a: {f: acceptable(r_f2[a], r_r, f) for f in FACTORS} for a in ARMS}
    for a in ARMS:
        worst_c = max(r_r, key=lambda c: r_f2[a][c] / r_r[c])
        s1.say("ARM_JSON", {"arm": arm_key(a), "role": "DECIDING" if a in DECIDING else "report-only",
                            "worst_ratio": r_f2[a][worst_c] / r_r[worst_c], "worst_case": worst_c,
                            "acceptable": {str(f): accept[a][f] for f in FACTORS},
                            "worst_vs_f64": max(vs_f64[a].values())})
    chosen = pick({a: accept[a][F2_FACTOR] for a in ARMS})
    thr = up2(A2_FACTOR * max(vs_f64[chosen].values())) if chosen else None
    preds = score_predictions(r_f2, r_r, qw0_row)
    s1.say("PRED_JSON", preds)
    s1.say("OUTPUT_JSON", {"pick": arm_key(chosen) if chosen else None,
                           "e_f2": ef["exact"][chosen[1]] if chosen else None,
                           "e_f2_float": ef["exact_float"][chosen[1]] if chosen else None, "a2_f2_threshold": thr,
                           "f2_factor": F2_FACTOR, "cases": len(r_r),
                           "acceptable_sets": {str(f): [arm_key(a) for a in ARMS if accept[a][f]] for f in FACTORS}})
    s1.say("MEM_JSON", {"child": "F2-E", **s1.own_memory()})

    order = sorted(qw0_line, key=lambda c: (c.split(".")[1] != "down", int(c.split(".")[0]), c))
    for c in order:
        m = qw0_line[c]
        print(f"F2-E QW0 ROW {c}: {m['qw0_blocks']} of {m['nonzero_dw_blocks']} nonzero-d_w blocks round to qw = 0 "
              f"({m['qw0_fraction']:.4%}), in {m['qw0_columns']} columns", flush=True)
    for a in ARMS:
        role = "DECIDING" if a in DECIDING else "report-only"
        worst = max(r_f2[a][c] / r_r[c] for c in r_r)
        sets = ", ".join(f"{f:g} {'yes' if accept[a][f] else 'no'}" for f in FACTORS)
        print(f"F2-E ARM {arm_key(a)} ({role}): worst r_F2 / r_R {worst:.4f}; ACCEPTABLE at {sets}", flush=True)
    for k, v in preds.items():
        print(f"F2-E {k}: {'HELD' if v['held'] else 'NOT HELD'} ({PREDICTIONS[k]})", flush=True)
    if chosen:
        print(f"F2-E PICK: form B, FLUSH_{chosen[1]}, S = ROW, E_F2 {ef['exact_float'][chosen[1]]} (every case's "
              f"r_F2 <= {F2_FACTOR:g} x r_R)", flush=True)
        print(f"F2-E A2-F2 THRESHOLD: {thr:.1e} (up2({A2_FACTOR:g} x {max(vs_f64[chosen].values()):.4e}))",
              flush=True)
    else:
        print(f"F2-E SCREEN: NONE (neither deciding arm has every case's r_F2 <= {F2_FACTOR:g} x r_R); F2-A is not "
              "proposed, and F2 ends (plan section 6)", flush=True)
        print("F2-E A2-F2 THRESHOLD: NOT SET (NONE)", flush=True)
    print(f"F2-E NOTE: {NOTE_N2}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "pins", "run"))
    a = ap.parse_args()
    return {"selftest": selftest, "pins": pins, "run": run}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
