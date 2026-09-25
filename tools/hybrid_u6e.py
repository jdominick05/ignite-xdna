#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid stack, U6-E: the precision of U6's planned epilogue, CPU only (numpy, and one small ORT session per
linear). No NPU and no GPU.

U6's kernel (the plan, v2 + E1/E2, section 1) forms y[m,n] = sum over 32-blocks b of float(i[m,n,b]) * P[m,n,b],
with P = s_x[m,b] * d_w[n,b], in fp32. AIE2 has no vector fp32 multiply, so the core builds fp32-grade products
from bf16 pieces: a bf16 x bf16 product is exact in fp32, and only the fp32 adds round. This tool emulates that
epilogue in numpy, faithful to bf16 RNE and to fp32 adds in a fixed order (ORDER below), with 2- and 3-term
splits. It reads each split against MatMulNBits accuracy level 4 (R's kernel), built as S1's C2 built it, per
linear, on the weights of layers 0, 16 and 33 and S1's captured layer-16 inputs (x16_*, sequence 0).

Its two outputs are the split U6 needs and A2's threshold. Both follow from the run's numbers by SPLIT_RULE and
A2_RULE, fixed here before any real input is computed; U6's prereg then freezes them.

    python tools/hybrid_u6e.py selftest   # synthetic only: bf16 RNE (ties), the splits, the add order; no model
    python tools/hybrid_u6e.py pins       # the input pins' current values: hashes and sizes only, nothing computed
    python tools/hybrid_u6e.py run        # the pins, C2's anchor, then the emulation and the two outputs
"""
import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma_compress as gc  # noqa: E402
import gemma_decode as gd  # noqa: E402
import hybrid_s1 as s1  # noqa: E402
import llm_prefill3c as p3c  # noqa: E402

CHECK = s1.CHECK                                         # S1's check outputs, read-only
PLAN_SHA = "830eb8b9cd5d3512c1ffb20b5a1cee57e67280f7dd1d8333ffae6ea478e9508d"   # U6 v2 + E1/E2, as approved (LF)
PLAN_FILE = ROOT / "scratch/llm/hybrid_u6_plan_draft.md"                        # git-ignored

# ---------------------------------------------------------------- the protocol (fixed before any real input)

LAYERS = (0, 16, 33)                                     # A2's weights (the plan, section 4)
ROWS = tuple(range(0, s1.SEQ_LEN, 8))                    # 256 rows of S1's sequence 0: 0, 8, ..., 2040 (BOS row 0 in)
ROWS_TEXT = "range(0, 2048, 8): rows 0, 8, ..., 2040 of each x16_* input, the same 256 rows for every linear"
SPLITS = (2, 3)

# The rule that turns the numbers into the two outputs. L4 is MatMulNBits at accuracy level 4 and F64 the float64
# value of its own form (MLAS CompInt8's codes and scales times the Q4_0 weights, S1's q4_dot), both on ROWS.
SPLIT_FACTOR = 2.0                                       # proposed margin (the gate rules on it before any run)
SPLIT_RULE = ("a T-term split is ENOUGH iff, for every case c (layers 0, 16 and 33 x the seven linears), "
              "rel_l2(emu_T, L4) <= SPLIT_FACTOR x rel_l2(L4, F64). The split is the smallest ENOUGH T of (2, 3), "
              "and NONE if neither is.")
SPLIT_BOUND = ("the bound is L4's own departure from its exact form, per case, on the same rows. On layer 16 it is "
               "anchored to S1's C2 (3.02e-7 to 7.57e-7 over 2,048 rows): the run STOPs unless L4 reproduces C2's "
               "output bit for bit. On layers 0 and 33 it is this run's own measurement. SPLIT_FACTOR 2.0 is a "
               "proposed margin with no measured source: the emulation may sit up to twice as far from L4 as L4 sits "
               "from exact arithmetic (an error uncorrelated with L4's own may reach about sqrt(3) = 1.73 x the floor).")
A2_FACTOR = 2.0                                          # proposed margin (the gate rules on it before any run)
A2_RULE = ("A2's threshold = up2(A2_FACTOR x the maximum over the 21 cases of rel_l2(emu_T, L4)) for the chosen T, "
           "rounded up at 2 significant figures. It is NOT SET if the split is NONE.")
A2_BOUND = ("A2_FACTOR 2.0 is a proposed margin with no measured source: headroom for what the emulation does not "
            "model (the kernel's lane order within a tile, and accfloat's rounding; ASSUMPTIONS).")
NONE_READING = ("if the split is NONE, even 3 pieces (which carry s_x, d_w and P exactly) do not bring the "
                "emulation within the rule, so the limiter is the fp32 add order, not the split; each case's "
                "vs_f64 is then the reading to look at, and A2 is not set.")

ORT_VERSION = "1.23.3.dev20260320"                       # S1's ORT (tools/hybrid_s1.py:841)
C2_LOGGED = {"q": 3.4442778290698247e-07, "k": 3.1888439057385116e-07, "v": 5.645603249807371e-07,
             "o": 3.0176442337857616e-07, "gate": 3.33442604459366e-07, "up": 5.047437104722956e-07,
             "down": 7.572597852043961e-07}              # S1's C2_JSON nb4_vs_ort_form (hybrid_s1_check log)
C2_REL_TOL = 1e-6                                        # the anchor: |ours - logged| <= 1e-6 x logged
X16_SHA = {}                                             # filled from `pins` before the frozen commit
W_SHA = {}                                               # "L.gguf_name": sha256 of the tensor's raw Q4_0 bytes

# The fixed fp32 add order. Pieces are numbered from 1 (the largest). A term (j, k) is piece j times piece k; its
# order is j + k - 2, and a split of T pieces keeps the terms of order <= T - 1. Terms are added smallest first.
ORDER = {
    2: {"P": [(2, 1), (1, 2), (1, 1)],                   # s_x piece j x d_w piece k
        "Y": [(2, 1), (1, 2), (1, 1)]},                  # float(i) piece u x P piece v, added into y
    3: {"P": [(3, 1), (2, 2), (1, 3), (2, 1), (1, 2), (1, 1)],
        "Y": [(2, 2), (1, 3), (2, 1), (1, 2), (1, 1)]},  # float(i) has 2 pieces only: it is exact in 2
}
FORM = ("per block b in increasing k: i = the exact int32 dot; float(i) = h1 + h2 (exact, |i| <= 32,512); s_x and "
        "d_w split into T bf16 pieces each (RNE of the running fp32 residual); P = the fp32 sum of ORDER[T]['P']'s "
        "exact products, the first term a multiply, the rest adds; P split into T bf16 pieces; then each "
        "ORDER[T]['Y'] product is added into y, the fp32 running sum over blocks, one fp32 rounding per add")
ASSUMPTIONS = [
    "AIE2's accfloat add rounds each MAC to fp32 by RNE, as numpy's float32 add does; SILICON does not record "
    "its rounding mode or denormal handling, and the scale products sit far above the denormal range",
    "the inputs of layers 0 and 33 are layer 16's (x16_*); other layers' inputs are an assumption, as in A2",
    "d_w goes to fp32 on the host (exact from fp16); s_x is MLAS CompInt8's fp32 amax / 127 (S1's q8_ort)",
]
MEMORY = "S1's start_gate: refuse below 15 GB available, and a watchdog ends the process below 5 GB; MEM_JSON last"


def protocol() -> dict:
    return {"stage": "hybrid U6-E", "plan_sha": PLAN_SHA, "layers": LAYERS,
            "rows": {"text": ROWS_TEXT, "count": len(ROWS), "first": ROWS[0], "last": ROWS[-1],
                     "step": ROWS[1] - ROWS[0]},
            "splits": SPLITS, "split_factor": SPLIT_FACTOR, "split_rule": SPLIT_RULE, "split_bound": SPLIT_BOUND,
            "a2_factor": A2_FACTOR, "a2_rule": A2_RULE, "a2_bound": A2_BOUND, "none_reading": NONE_READING,
            "order": {str(t): v for t, v in ORDER.items()}, "form": FORM, "ort_version": ORT_VERSION,
            "c2_logged": C2_LOGGED, "c2_rel_tol": C2_REL_TOL, "x16_sha": X16_SHA, "w_sha": W_SHA,
            "gguf_pin": {k: gc.GGUF_PIN[k] for k in ("repo", "file", "revision", "size", "sha256")},
            "l4": "MatMulNBits (com.microsoft), bits 4, block_size 32, accuracy_level 4, fp32 scales from the fp16 "
                  "d_w, opsets 21 / com.microsoft 1, ir 10: the graph of S1's compare_child "
                  "(tools/hybrid_s1.py:1597-1610); S1's session_options(): ORT_DISABLE_ALL, PRIORITY_BASED, 8 "
                  "intra-op threads, CPU EP",
            "memory": MEMORY, "assumptions": ASSUMPTIONS}


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def header(title: str) -> None:
    print(f"{title} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    s1.say("TOOL_SHA_JSON", {"file": "tools/hybrid_u6e.py", "lf_sha256": lf_sha(Path(__file__))})
    s1.say("PROTOCOL_JSON", protocol())


# ---------------------------------------------------------------- the arithmetic

def bf16(x: np.ndarray) -> np.ndarray:
    """fp32 -> the bf16 RNE value, held in fp32 (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def split(x: np.ndarray, t: int) -> list:
    """T bf16 pieces of fp32 x: each is RNE of the running residual; the residual's subtraction is exact."""
    out, r = [], np.asarray(x, dtype=np.float32)
    for _ in range(t):
        p = bf16(r)
        out.append(p)
        r = (r - p).astype(np.float32)
    return out


def int_dot(qb: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """One 32-block's exact dot: q codes [M, 32] (float, -127..127) x (c - 8) [N, 32] -> int64 [M, N]."""
    y = qb.astype(np.float64) @ (cb.astype(np.float64) - 8.0).T
    i = np.rint(y).astype(np.int64)
    assert np.array_equal(i.astype(np.float64), y) and np.abs(i).max(initial=0) <= 32 * 127 * 8
    return i


def emulate(q: np.ndarray, sx: np.ndarray, codes: np.ndarray, dw: np.ndarray, t: int, order=None) -> np.ndarray:
    """The epilogue with T-piece splits. q [M, kb, 32], sx [M, kb] fp32, codes [N, kb, 32], dw [N, kb] fp32.
    order defaults to ORDER[t]; the selftest passes another to show that the order is what runs."""
    M, kb, _ = q.shape
    N = codes.shape[0]
    y = np.zeros((M, N), dtype=np.float32)
    order = order or ORDER[t]
    for b in range(kb):
        fi = int_dot(q[:, b, :], codes[:, b, :]).astype(np.float32)
        h1 = bf16(fi)
        h = [h1, (fi - h1).astype(np.float32)]
        a = [p[:, None] for p in split(sx[:, b], t)]
        w = [p[None, :] for p in split(dw[:, b], t)]
        (j, k), rest = order["P"][0], order["P"][1:]
        P = (a[j - 1] * w[k - 1]).astype(np.float32)
        for j, k in rest:
            P = (P + a[j - 1] * w[k - 1]).astype(np.float32)
        p = split(P, t)
        for u, v in order["Y"]:
            y = (y + h[u - 1] * p[v - 1]).astype(np.float32)
    return y


def emulate_scalar(q: np.ndarray, sx: np.ndarray, codes: np.ndarray, dw: np.ndarray, t: int) -> np.ndarray:
    """An independent form of emulate(), for the selftest only: Python loops over (m, n, b), numpy float32
    scalars (each op rounds to fp32), bf16 RNE through ml_dtypes, and ORDER[t] walked term by term."""
    import ml_dtypes
    f32 = np.float32

    def b16(v):
        return f32(np.float32(v).astype(ml_dtypes.bfloat16).astype(np.float32))

    def pieces(v, n):
        out, r = [], f32(v)
        for _ in range(n):
            p = b16(r)
            out.append(p)
            r = f32(r - p)
        return out
    M, kb, _ = q.shape
    N = codes.shape[0]
    y = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            acc = f32(0.0)
            for b in range(kb):
                i = sum(int(q[m, b, e]) * (int(codes[n, b, e]) - 8) for e in range(32))
                fi = f32(i)
                h = [b16(fi)]
                h.append(f32(fi - h[0]))
                a, w = pieces(sx[m, b], t), pieces(dw[n, b], t)
                (j, k), rest = ORDER[t]["P"][0], ORDER[t]["P"][1:]
                P = f32(a[j - 1] * w[k - 1])
                for j, k in rest:
                    P = f32(P + f32(a[j - 1] * w[k - 1]))
                p = pieces(P, t)
                for u, v in ORDER[t]["Y"]:
                    acc = f32(acc + f32(h[u - 1] * p[v - 1]))
            y[m, n] = acc
    return y


def up2(x: float) -> float:
    """x rounded up at 2 significant figures."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - 1
    return float(f"{math.ceil(x / 10.0 ** e - 1e-9) * 10.0 ** e:.1e}")


def rel_l2(y, ref) -> float:
    return s1.rel_l2(y, ref)


def max_rel(y, ref) -> float:
    y, ref = np.asarray(y, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.abs(y - ref).max() / np.abs(ref).max())


# ---------------------------------------------------------------- ORT and the inputs

def nb_run(X: np.ndarray, codes: np.ndarray, d: np.ndarray, K: int, N: int, level: int) -> np.ndarray:
    """MatMulNBits as S1's C2 built it (compare_child, tools/hybrid_s1.py:1597-1610), at accuracy_level `level`."""
    import onnxruntime as ort
    from onnx import TensorProto as P, helper
    node = helper.make_node("MatMulNBits", ["x", "b", "s"], ["y"], domain="com.microsoft", K=K, N=N, bits=4,
                            block_size=32, accuracy_level=level)
    packed = gd.pack_ort(codes)
    g = helper.make_graph([node], "u6e", [helper.make_tensor_value_info("x", P.FLOAT, [len(X), K])],
                          [helper.make_tensor_value_info("y", P.FLOAT, [len(X), N])],
                          [helper.make_tensor("b", P.UINT8, list(packed.shape), packed.tobytes(), raw=True),
                           helper.make_tensor("s", P.FLOAT, [N * (K // 32)], gc.f16_to_f32(d).astype(np.float32).tobytes(),
                                              raw=True)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                          ir_version=10)
    sess = ort.InferenceSession(m.SerializeToString(), s1.session_options(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": X})[0]


def tensor_bytes(mm, t) -> bytes:
    return bytes(mm[t["start"]:t["start"] + (t["n"] // 32) * 18])


def current_pins(mm, by) -> tuple:
    xs = {x: s1.sha_file(CHECK / f"x16_{x}.npy") for x in p3c.INPUTS}
    ws = {f"{L}.{gg}": s1.sha_bytes(tensor_bytes(mm, by[f"blk.{L}.{gg}.weight"]))
          for L in LAYERS for _, gg, _, _, _ in p3c.LINEARS}
    return xs, ws


def ort_version() -> str:
    import onnxruntime as ort
    return ort.__version__


# ---------------------------------------------------------------- modes

def selftest() -> int:
    import ml_dtypes
    header("U6-E SELFTEST (synthetic only; no model, no real input, no ORT, no chip).")
    rng = np.random.default_rng(s1.SEED)
    ok = True

    def check(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        s1.say("SELFTEST_JSON", {"check": name, "ok": bool(cond), **info})

    # 1. bf16() is RNE on known ties. bf16 keeps 7 fraction bits, so its spacing is 2^-7 on [1, 2) and 2 on [256, 512).
    #    1 + 2^-8 is halfway between 1 and 1 + 2^-7: RNE keeps the even one, 1. 1 + 3 x 2^-8 is halfway between
    #    1 + 2^-7 (odd last bit) and 1 + 2^-6 (even): up. 257 is halfway between 256 (even) and 258 (odd): 256.
    #    259 is halfway between 258 (odd) and 260 (even): 260. The signs mirror.
    ties = [(1.00390625, 1.0), (1.01171875, 1.015625), (257.0, 256.0), (259.0, 260.0)]
    ties += [(-a, -b) for a, b in ties]
    got = bf16(np.array([a for a, _ in ties], dtype=np.float32))
    check("bf16 RNE on known ties", np.array_equal(got, np.array([b for _, b in ties], dtype=np.float32)),
          cases=[[a, b, float(g)] for (a, b), g in zip(ties, got)])

    # 2. bf16() equals ml_dtypes' RNE on random values and on every tie pattern (low half 0x8000; denormals included)
    x = np.concatenate([rng.standard_normal(1_000_000).astype(np.float32) * np.float32(10.0) ** rng.integers(-6, 6, 1_000_000),
                        (np.arange(1, 65536, dtype=np.uint32) << np.uint32(16) | np.uint32(0x8000)).view(np.float32)[:30000],
                        np.array([1.0, -1.0, 0.0, 3.0e-3, 1.5e-5], dtype=np.float32)]).astype(np.float32)
    x = x[np.isfinite(x)]
    ref = x.astype(ml_dtypes.bfloat16).astype(np.float32)
    check("bf16 equals ml_dtypes RNE", np.array_equal(bf16(x).view(np.uint32), ref.view(np.uint32)), n=int(x.size))

    # 3. float(i) is exact in 2 bf16 pieces over the whole range of |i| <= 32,512
    i = np.arange(-32512, 32513, dtype=np.int64)
    fi = i.astype(np.float32)
    h1 = bf16(fi)
    h2 = (fi - h1).astype(np.float32)
    check("float(i) = h1 + h2 exactly, h2 bf16-exact", np.array_equal(h1.astype(np.float64) + h2.astype(np.float64), i.astype(np.float64))
          and np.array_equal(bf16(h2), h2), n=int(i.size))

    # 4. the splits: each residual subtraction is exact; 3 pieces (8 + 8 + 8 >= 24 bits) carry a normal fp32 exactly
    s = (np.abs(rng.standard_normal(200_000)) * 10.0 ** rng.uniform(-5, 1, 200_000)).astype(np.float32)
    worst = {}
    exact = True
    for t in SPLITS + (1,):
        r, acc = s.copy(), np.zeros(s.shape)
        for p in split(s, t):
            exact &= np.array_equal((r.astype(np.float64) - p.astype(np.float64)), (r - p).astype(np.float32).astype(np.float64))
            acc += p.astype(np.float64)
            r = (r - p).astype(np.float32)
        worst[t] = float(np.max(np.abs(acc - s.astype(np.float64)) / s.astype(np.float64)))
    check("split residuals exact", exact)
    check("split error shrinks with T, and T=3 is exact", worst[1] > worst[2] > worst[3] and worst[3] == 0.0,
          worst_rel=worst)

    # 5. every product the order uses is exact in fp32
    a, w = split(s[:1000], 3), split(s[1000:2000], 3)
    prod_exact = all(np.array_equal((a[j] * w[k]).astype(np.float64), a[j].astype(np.float64) * w[k].astype(np.float64))
                     for j in range(3) for k in range(3))
    hp = split(s[:1000], 3)
    prod_exact &= all(np.array_equal((hh * pp).astype(np.float64), hh.astype(np.float64) * pp.astype(np.float64))
                      for hh in (h1[:1000], h2[:1000]) for pp in hp)
    check("bf16 x bf16 products exact in fp32", prod_exact)

    # the synthetic blocks for checks 6-10 (an outlier row, as BOS is)
    M, N, kb = 16, 48, 40
    X = (rng.standard_normal((M, kb * 32)) * np.exp(rng.standard_normal((M, 1)))).astype(np.float32)
    X[0, :64] *= 300.0
    codes = rng.integers(0, 16, (N, kb, 32)).astype(np.uint8)
    d = (np.abs(rng.standard_normal(N * kb)) * 0.01 + 1e-4).astype(np.float16).view(np.uint16)
    q, sx = s1.q8_ort(X)
    dw = gc.f16_to_f32(d).reshape(N, kb)

    # 6. the fp32 add order is what runs: emulate() equals a scalar walk of ORDER term by term, bit for bit
    #    (a sub-case: 3 rows, 5 columns, 6 blocks), and the same terms added in reverse order change some bits
    same = {t: bool(np.array_equal(emulate(q[:3, :6], sx[:3, :6], codes[:5, :6], dw[:5, :6], t).view(np.uint32),
                                   emulate_scalar(q[:3, :6], sx[:3, :6], codes[:5, :6], dw[:5, :6], t).view(np.uint32)))
            for t in SPLITS}
    check("emulate equals the scalar walk of ORDER, bit for bit", all(same.values()), by_t=same)
    rev = {t: {"P": ORDER[t]["P"][::-1], "Y": ORDER[t]["Y"][::-1]} for t in SPLITS}
    differ = {t: int(np.count_nonzero(emulate(q, sx, codes, dw, t).view(np.uint32)
                                      != emulate(q, sx, codes, dw, t, rev[t]).view(np.uint32))) for t in SPLITS}
    check("the reversed order changes some bits (the order check has teeth)", all(v > 0 for v in differ.values()),
          elements_differing=differ, of=M * N)

    # 7. the emulation against S1's float64 q4_dot (the ORT form's exact value)
    exact_y = s1.q4_dot(q, sx, codes, d)
    e = {t: rel_l2(emulate(q, sx, codes, dw, t), exact_y) for t in SPLITS}
    check("T=3 within the fp32 floor, T=2 above it and below 1e-4", e[3] <= 1e-6 and e[3] < e[2] <= 1e-4, rel_l2=e)

    # 8. the block-to-scale mapping: a planted swap of two blocks' d_w shows at >= 1e-3
    bad = dw.copy()
    bad[:, [3, 4]] = bad[:, [4, 3]]
    planted = rel_l2(emulate(q, sx, codes, bad, 3), exact_y)
    check("a planted d_w block swap is seen", planted >= 1e-3, rel_l2=planted)

    # 9. determinism: the same inputs give the same bits
    check("deterministic", np.array_equal(emulate(q, sx, codes, dw, 2).view(np.uint32), emulate(q, sx, codes, dw, 2).view(np.uint32)))

    # 10. the rules' arithmetic: up2 rounds up at 2 figures, and the split rule picks the smallest ENOUGH T
    check("up2", all(math.isclose(up2(v), want, rel_tol=1e-12) for v, want in ((3.44e-7, 3.5e-7), (1.2e-6, 1.2e-6),
                                                                              (9.91e-6, 1e-5), (1.001e-6, 1.1e-6))))
    check("the split rule", pick_split({2: 3.1, 3: 1.2}) == 3 and pick_split({2: 1.9, 3: 1.1}) == 2
          and pick_split({2: 5.0, 3: 2.5}) is None and pick_split({2: 2.0, 3: 2.0}) == 2)
    s1.say("MEM_JSON", {"child": "U6-E selftest", **s1.own_memory()})
    print("SELFTEST OK" if ok else "SELFTEST FAIL", flush=True)
    return 0 if ok else 1


def pick_split(worst_ratio: dict):
    """SPLIT_RULE: the smallest T whose worst ratio over the cases is <= SPLIT_FACTOR, else None (NONE)."""
    return next((t for t in SPLITS if worst_ratio[t] <= SPLIT_FACTOR), None)


def pins() -> int:
    header("U6-E PINS (hashes and sizes only; nothing is computed).")
    mm, by = s1.open_gguf(hash_check=False)
    g = gc.GGUF_PIN
    s1.say("GGUF_JSON", {"file": g["file"], "size_pinned": g["size"], "sha256_pinned": g["sha256"],
                         "size_equal": True, "sha256_checked": False,
                         "note": "open_gguf checked the size; run mode hashes the whole file"})
    xs, ws = current_pins(mm, by)
    s1.say("X16_SHA_JSON", xs)
    s1.say("X16_BYTES_JSON", {x: (CHECK / f"x16_{x}.npy").stat().st_size for x in p3c.INPUTS})
    s1.say("W_SHA_JSON", ws)
    s1.say("W_BYTES_JSON", {f"{L}.{gg}": (by[f"blk.{L}.{gg}.weight"]["n"] // 32) * 18
                            for L in LAYERS for _, gg, _, _, _ in p3c.LINEARS})
    s1.say("ORT_JSON", {"version": ort_version(), "pinned": ORT_VERSION})
    s1.say("MEM_JSON", {"child": "U6-E pins", **s1.own_memory()})
    print("PINS DONE", flush=True)
    return 0


def run() -> int:
    header("U6-E RUN (hybrid U6), CPU only.")
    plan = lf_sha(PLAN_FILE) if PLAN_FILE.exists() else None
    s1.say("PLAN_JSON", {"pinned": PLAN_SHA, "file_lf_sha": plan})
    if plan is not None and plan != PLAN_SHA:
        print("U6-E STOP: the plan file differs from the approved plan", flush=True)
        return 2
    s1.start_gate("U6-E")
    if not X16_SHA or not W_SHA:
        print("U6-E STOP: the pins are empty", flush=True)
        return 2
    if ort_version() != ORT_VERSION:
        print(f"U6-E STOP: onnxruntime {ort_version()} is not {ORT_VERSION}", flush=True)
        return 2
    mm, by = s1.open_gguf(hash_check=True)
    xs, ws = current_pins(mm, by)
    if xs != X16_SHA or ws != W_SHA:
        s1.say("PIN_DIFF_JSON", {"x16": {k: v for k, v in xs.items() if X16_SHA.get(k) != v},
                                 "w": {k: v for k, v in ws.items() if W_SHA.get(k) != v}})
        print("U6-E STOP: an input differs from its pin", flush=True)
        return 2
    s1.say("PINS_JSON", {"gguf": "size and sha256 equal the pin", "x16": "equal", "w": "equal", "ort": ORT_VERSION})

    X16 = {x: np.load(CHECK / f"x16_{x}.npy") for x in p3c.INPUTS}
    rows = np.asarray(ROWS)

    # C2's anchor: layer 16, all 2,048 rows. L4 must equal S1's saved C2 output bit for bit, and the float64 ORT
    # form against L4 must reproduce S1's logged values.
    anchor_ok, l4_16 = True, {}
    for name, gg, K, N, x in p3c.LINEARS:
        codes, d = gd.gguf_linear(mm, by[f"blk.16.{gg}.weight"])
        y4 = nb_run(X16[x], codes, d, K, N, 4)
        q, sx = s1.q8_ort(X16[x])
        v = rel_l2(y4, s1.q4_dot(q, sx, codes, d))
        prior = CHECK / f"c2_nb4_{name}.npy"
        bits = bool(np.array_equal(np.load(prior).view(np.uint32), y4.view(np.uint32))) if prior.exists() else False
        good = bits and abs(v - C2_LOGGED[name]) <= C2_REL_TOL * C2_LOGGED[name]
        anchor_ok &= good
        s1.say("ANCHOR_JSON", {"name": name, "nb4_vs_ort_form": v, "logged": C2_LOGGED[name],
                               "l4_bits_equal_s1_c2_output": bits, "ok": good})
        l4_16[name] = y4[rows]
    if not anchor_ok:
        print("U6-E STOP: C2's anchor did not reproduce", flush=True)
        return 2
    print("ANCHOR OK: L4 equals S1's C2 output bit for bit, and reproduces C2's values, on all seven linears",
          flush=True)

    worst = {t: 0.0 for t in SPLITS}
    worst_ratio = {t: 0.0 for t in SPLITS}
    for L in LAYERS:
        for name, gg, K, N, x in p3c.LINEARS:
            t0 = time.time()
            codes, d = gd.gguf_linear(mm, by[f"blk.{L}.{gg}.weight"])
            Xr = np.ascontiguousarray(X16[x][rows])
            y4 = l4_16[name] if L == 16 else nb_run(X16[x], codes, d, K, N, 4)[rows]
            y0 = nb_run(X16[x], codes, d, K, N, 0)[rows]
            q, sx = s1.q8_ort(Xr)
            f64 = s1.q4_dot(q, sx, codes, d)
            dw = gc.f16_to_f32(d).reshape(N, K // 32)
            floor = rel_l2(y4, f64)
            case = {"layer": L, "name": name, "rows": len(rows), "l4_vs_f64": floor, "l4_vs_l0": rel_l2(y4, y0)}
            for t in SPLITS:
                e = emulate(q, sx, codes, dw, t)
                assert np.isfinite(e).all()
                r4 = rel_l2(e, y4)
                case[f"T{t}"] = {"vs_l4": r4, "vs_f64": rel_l2(e, f64), "max_rel_vs_l4": max_rel(e, y4),
                                 "ratio_to_floor": r4 / floor}
                worst[t] = max(worst[t], r4)
                worst_ratio[t] = max(worst_ratio[t], r4 / floor)
            case["seconds"] = round(time.time() - t0, 1)
            s1.say("CASE_JSON", case)
    split_t = pick_split(worst_ratio)
    thr = up2(A2_FACTOR * worst[split_t]) if split_t else None
    s1.say("SPLITS_JSON", {str(t): {"worst_vs_l4": worst[t], "worst_ratio_to_floor": worst_ratio[t],
                                    "enough": worst_ratio[t] <= SPLIT_FACTOR} for t in SPLITS})
    s1.say("OUTPUT_JSON", {"split": split_t, "a2_threshold": thr, "split_factor": SPLIT_FACTOR,
                           "a2_factor": A2_FACTOR, "cases": len(LAYERS) * len(p3c.LINEARS)})
    s1.say("MEM_JSON", {"child": "U6-E", **s1.own_memory()})
    if split_t:
        print(f"U6-E SPLIT: {split_t}-term (the smallest T with every case's ratio <= {SPLIT_FACTOR:g})", flush=True)
        print(f"U6-E A2 THRESHOLD: {thr:.1e} (up2({A2_FACTOR:g} x {worst[split_t]:.4e}))", flush=True)
    else:
        print(f"U6-E SPLIT: NONE (neither split has every case's ratio <= {SPLIT_FACTOR:g}; see none_reading)",
              flush=True)
        print("U6-E A2 THRESHOLD: NOT SET (the split is NONE)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "pins", "run"))
    a = ap.parse_args()
    return {"selftest": selftest, "pins": pins, "run": run}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
