"""Probe the bf16 core's multiply-accumulate at the positions where a network frame and the emulator part.

WHAT IT IS FOR. On SESR-M7's AdaRound weights the device departs from the emulator's accumulate model
at single values next to bf16 rounding ties (BENCHMARKS, "The AdaRound mismatch starts in body.1").
Replayed offline from the device's own inputs, each such value is one output lane's chain of
multiply-accumulates: a bias, then one ``vmac.f`` per (ky, kx, input block), each adding eight
products. A candidate model that reproduces the device's final bits has not been tested by those
bits, because it was chosen by them. This tool measures the instructions themselves on one core, and
prints every candidate's prediction first, from the same packets, so a run can be read against
predictions committed before it.

THREE ARMS, all in one dispatch. The sitting runs pass 1 once, then pass 2 twice, so determinism is
checked inside it, and every accumulator must read the same in both passes:

  A  The network's own packet, replayed. For each position, every tile that covers it is rebuilt
     exactly as the compiler packs it (``chunk_packet``) and fed the DEVICE's input window. The whole
     tile, 1,600 values, is compared with the tile the network wrote.
  B  Each position's chain, from its exact accumulator. The chain starts at the first instruction
     where any candidate model leaves the exact running sum. The accumulator entering it is exact
     and every candidate agrees on it. The chain runs to the last instruction, one packet per
     instruction, and the fp32 accumulator is read back after every one.
  C  Fresh operands, built to separate the candidates. Each is one instruction from a stated
     accumulator. None was seen by any fit.

READING AN fp32 ACCUMULATOR BACK. The core stores bf16, so one instruction's fp32 result is read in
three packets that load psum, emit, and leave psum as it was:
  - direct: emits bf16(acc);
  - coarse: adds (-t1) x 1, where t1 is a reference truncated to 8 bits;
  - fine: adds (-t1) x 1 and (-t2) x 1, where t2 is the rest of the reference rounded to 8 bits.
When acc is within 256 of its own units of t1 + t2, acc - t1 - t2 has at most 8 significant bits, so
bf16 holds it exactly and acc = t1 + t2 + that. A fine residual under 256 units, with acc at t1's
exponent, proves that case: rounding cannot carry a larger value below 256. The readout adds only
operands on acc's own 24-bit grid, so every candidate model computes it exactly.

The candidates' chains drift apart by up to thousands of units once a sum cancels, so no fixed
reference serves them all. The read therefore takes TWO PASSES of the same packets:
  - pass 1 uses the exact running sum, rounded to fp32, as the reference;
  - pass 2 uses t1 from pass 1 and, as t2, pass 1's coarse residual, which is bf16 already.
The pass-2 fine residual is then acc less its own 8-bit rounding, at most 128 units, so any
accumulator reads exactly, predicted or not. Offline, this is checked for every model: its pass 1
feeds its pass 2, and the pass-2 decode must equal its own psum after every instruction.

An exact start is assembled the same way. One packet adds hi, mid and lo, each x 1, to a zero bias.
They are v's significand in three truncated 8-bit pieces, all on v's grid, so every model sums them
to v exactly. The readout after that packet checks the assembly on silicon.

LAYOUT. Experiment i owns pixel i and output lane (i // 4, i % 4) of a k = 1 packet: its activations
sit at pixel i, and its weights on its lane. Every other (pixel, lane) pair computes a product of one
experiment's activations with another's weights. Those values are real arithmetic but nobody's
experiment, and they are never read. A group holds at most 16 experiments, one per lane.

THE HARNESS IS NOT THE NETWORK. Arms B and C run on kernels/bf16_conv/engine_bf16.py's one-core
design, which links its own build of engine_bf16.cc. The SESR container links another. A ``vmac.f``
is one hardware instruction in either. Arm A is what checks that the harness reproduces the
network's values.

PRE-REGISTERED OUTCOMES (committed with the offline predictions, before any sitting):
  A1  One core equals the network's tile at every value. The harness reproduces the network for
      these tiles, so what B and C find carries to the network.
  A2  At a miss, one core equals the emulator's in-force model, not the network. The network's value
      came from something the single packet does not reproduce: operand delivery, state left by an
      earlier packet, or the container's own kernel build. The accumulate rule is then not the cause
      at that position, and B and C cannot explain it.
  A3  One core matches neither, or its two dispatches differ. Read nothing else from the sitting
      until that is explained.
  B1  Every readout equals one model's prediction, at every instruction of every chain. That
      model's rule reproduces the network's values as single instructions.
  B2  At a chain's first instruction the core returns the in-force model's value (it drops what the
      network kept). Under A1, the network's in-register chain differs from the same instructions
      chained through psum: the accumulator carries more than fp32 between instructions in a packet,
      which the 2026-09-21 inter-instruction probe did not show. Under A2, see A2.
  B3  A readout matches no model. That instruction's operands, all logged, are a counterexample to
      every candidate, and the next model has to fit them.
  C1  Every fresh vector equals one model. Its rule holds on operands it was not fitted to.
  C2  The vectors that separate "exp_sum" from "aligned" come back as "aligned". "exp_sum" is refuted
      on fresh operands. If B1 held for it anyway, the rule depends on something these vectors do
      not share with the network's operands, and that is unexplained.
  C3  The vectors below "exp_sum"'s grid come back as "wide". The grid is wider than "exp_sum"
      places it, and "exp_sum" approximates the rule without being it.
  C4  A readout where every model agrees comes back different. Every assembled start is one, and so
      is every chain instruction before its first step. That is a harness or readout fault, and the
      sitting is discarded.
A model is put in force only on A1 + B1 + C1 for the same model. The layer replay over every value
of the frame (bf16_sr_silicon_check.py --from-dump --isolate) must also stay exact under it. That
replay is the data the positions came from, so it checks that a model breaks nothing else; it does
not test the model. Anything else is recorded, and the in-force model stays.

    python tools/bf16_mac_position_probe.py --dump DIR --isolated-log LOG --qdq Q --container C
        offline: prints the plan, the operands and every model's predictions (no device)
    bash scripts/research-lowlevel.sh --log results/aie/<name>.log --checks-only --npu -- \\
        bash scripts/research-iron.sh tools/bf16_mac_position_probe.py ... --npu
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule_bf16 as eb  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler import graph_reference_bf16 as gr16  # noqa: E402
from ignite_xdna.compiler.engine_schedule import layer_chunks, tile_origins  # noqa: E402
from tools import bf16_sr_silicon_check as chk  # noqa: E402

F32 = np.float32
LANES = em.NCO * 4                                   # 16 experiments per group, one per output lane
K1_ROWS, K1_COLS = em.TILE_ROWS, em.TILE_COLS        # a k = 1, stride 1 packet reads the tile itself
K1_PLANE = K1_ROWS * K1_COLS * 8


def emit(tag: str, obj) -> None:
    print(f"BF16_MAC_PROBE_{tag} " + json.dumps(obj, sort_keys=True), flush=True)


def h32(v) -> str:
    return f"{int(np.array([v], F32).view(np.uint32)[0]):08x}"


# --------------------------------------------------------------------------- exact pieces on the bits
def trunc8(v: float) -> float:
    """v truncated toward zero to bf16's 8 significant bits: its top 16 bits as a float32."""
    return float((np.array([v], F32).view(np.uint32) & np.uint32(0xFFFF0000)).view(F32)[0])


def split3(v: float) -> tuple:
    """Three bf16 values whose exact sum is the float32 v, all on v's own 24-bit grid."""
    v = F32(v)
    hi = trunc8(v)
    r = F32(v - F32(hi))                              # exact: same sign, same grid, smaller
    mid = trunc8(r)
    lo = F32(r - F32(mid))
    assert float(em.to_bf16(np.array([lo], F32))[0]) == float(lo), "the last piece must be 8 bits"
    assert math.fsum([hi, mid, float(lo)]) == float(v)
    return hi, mid, float(lo)


def readout_terms(ref: float) -> tuple:
    """(t1, t2): t1 is ref truncated to 8 bits and keeps ref's exponent; t2 is the remainder to 8 bits."""
    t1 = trunc8(ref)
    t2 = float(em.to_bf16(np.array([F32(ref) - F32(t1)], F32))[0])
    return t1, t2


def bf16_value(bits: int) -> float:
    return float(em.from_bf16_bits(np.array([bits], np.uint16))[0])


def decode(t1: float, t2: float, direct: int, coarse: int, fine: int) -> dict:
    """The accumulator from its three readouts. Exact when the fine residual is under 256 units of t1's
    exponent: a residual that needed more than 8 bits would round to at least 256 units, never under.
    Otherwise t1 + the coarse residual, good to bf16's 8 bits of acc - t1, and marked inexact."""
    r = bf16_value(fine)
    v = math.fsum([t1, t2, r])
    if t1 == 0.0:
        exact = r == 0.0 and bf16_value(coarse) == 0.0 and direct in (0x0000, 0x8000)
    else:
        unit = 2.0 ** (math.frexp(t1)[1] - 24)
        # acc must sit at t1's exponent, or the readout's grid is not acc's own
        exact = (abs(r) < 256 * unit and float(F32(v)) == v and v != 0.0
                 and math.frexp(v)[1] == math.frexp(t1)[1])
    if not exact:
        v = math.fsum([t1, bf16_value(coarse)])
    ok = int(em.bf16_bits(em.to_bf16(np.array([v], F32)))[0]) == direct
    return {"acc": v, "exact": exact, "consistent": ok}


# --------------------------------------------------------------------------- k = 1 packets
def out_index(pixel: int, lane: int) -> int:
    """Where (pixel, output lane) lands in an emitted tile: 8 channels a block, (b % 2) * 4 + n."""
    b, n = divmod(lane, 4)
    r, x = divmod(pixel, K1_COLS)
    return (b // 2) * em.OUT_BLOCK8_ELEMS + (r * K1_COLS + x) * 8 + (b % 2) * 4 + n


def psum_index(pixel: int, lane: int) -> int:
    b, n = divmod(lane, 4)
    r, x = divmod(pixel, K1_COLS)
    return ((b * K1_ROWS + r) * K1_COLS + x) * 4 + n


def k1_packet(flags: int, acts: dict, lane_w: dict, all_lanes_w=None):
    """A k = 1, one-block packet. acts: pixel -> 8 activations; lane_w: lane -> 8 weights, or
    all_lanes_w: the same 8 weights on every lane. The bias is zero."""
    header = em.make_header(k=1, stride=1, ncin=1, flags=flags, rows_in=K1_ROWS, cols_in=K1_COLS,
                            plane_elems=K1_PLANE)
    act = np.zeros(K1_PLANE, F32)
    for p, a in acts.items():
        act[p * 8:p * 8 + 8] = a
    w = np.zeros((em.NCO, 8, 4), F32)
    if all_lanes_w is not None:
        w[:, :, :] = np.asarray(all_lanes_w, F32)[None, :, None]
    for lane, ws in lane_w.items():
        b, n = divmod(lane, 4)
        w[b, :, n] = ws
    for arr in (act, w):
        assert np.array_equal(em.to_bf16(arr), arr), "every operand must be exact in bf16"
    return header, act, w.reshape(-1), np.zeros(em.NCO * 16, F32)


def chain_packets(exps: list, terms_at: dict | None = None) -> tuple:
    """One group's packets: assemble each start and read it, then per instruction a step and a read.
    A read is three packets (direct, coarse, fine). Returns (packets, readouts) where readouts[j] is
    (direct, coarse, fine packet indices, {experiment: (t1, t2, ref)}), and j = 0 reads the start.
    A group whose chains differ in length pads the short ones with all-zero steps, which leave any
    accumulator unchanged under every model. terms_at {(i, j): (t1, t2, ref)} replaces readout j's
    terms for experiment i; that is how pass 2 is built from pass 1."""
    assert len(exps) <= LANES
    pkts, readouts = [], []
    pieces = {i: split3(e["start"]) for i, e in enumerate(exps)}
    pkts.append(k1_packet(0, {i: list(pieces[i]) + [0.0] * 5 for i in pieces}, {},
                          all_lanes_w=[1, 1, 1, 0, 0, 0, 0, 0]))
    exact = {i: [float(e["start"])] for i, e in enumerate(exps)}

    def read():
        j = len(readouts)
        refs = {i: float(F32(math.fsum(exact[i]))) for i in exact}
        terms = {i: (terms_at or {}).get((i, j)) or (readout_terms(refs[i]) + (refs[i],)) for i in refs}
        pkts.append(k1_packet(em.F_LOAD_PSUM | em.F_EMIT, {}, {}))
        pkts.append(k1_packet(em.F_LOAD_PSUM | em.F_EMIT, {i: [-terms[i][0]] + [0.0] * 7 for i in terms},
                              {}, all_lanes_w=[1, 0, 0, 0, 0, 0, 0, 0]))
        pkts.append(k1_packet(em.F_LOAD_PSUM | em.F_EMIT, {i: [-terms[i][0], -terms[i][1]] + [0.0] * 6 for i in terms},
                              {}, all_lanes_w=[1, 1, 0, 0, 0, 0, 0, 0]))
        readouts.append((len(pkts) - 3, len(pkts) - 2, len(pkts) - 1, terms))

    read()
    for j in range(max(len(e["steps"]) for e in exps)):
        acts, lanes = {}, {}
        for i, e in enumerate(exps):
            if j < len(e["steps"]):
                a, w = e["steps"][j]
                acts[i], lanes[i] = a, w
                exact[i] += [float(x) * float(y) for x, y in zip(a, w)]
        pkts.append(k1_packet(em.F_LOAD_PSUM, acts, lanes))
        read()
    return pkts, readouts


def pass2_terms(rows: list) -> dict:
    """Pass 2's readout terms from pass 1's reads: pass 1's t1, and its coarse residual as t2."""
    out = {}
    for i, seq in enumerate(rows):
        for j, row in enumerate(seq):
            out[(i, j)] = (row["t1"], row["coarse"], row["acc"])
    return out


def run_models(pkts: list, models) -> dict:
    """Each model's emitted bits per packet and its psum after every packet: the emulator run the way
    the harness runs a dispatch (one psum, one hold buffer, packets in order)."""
    res = {}
    for m in models:
        psum, scratch = np.zeros(em.PSUM_FLOATS, F32), np.zeros(em.SCRATCH_ELEMS, F32)
        outs, psums = [], []
        for header, act, wts, bias in pkts:
            out = np.zeros(em.OUT_ELEMS, F32)
            em.run_packet(header, act, wts, bias, psum, out, scratch, mac_model=m)
            outs.append(em.bf16_bits(out))
            psums.append(psum.copy())
        res[m] = (np.stack(outs), np.stack(psums))
    return res


def read_chain(bits: np.ndarray, exps: list, readouts: list) -> list:
    """Per experiment, per readout: the decoded fp32 accumulator, whether it is exact, and whether it
    rounds to the direct bf16 readout."""
    rows = []
    for i, _ in enumerate(exps):
        seq = []
        for d_ix, c_ix, f_ix, terms in readouts:
            t1, t2, _ = terms[i]
            at = out_index(i, i)
            direct, coarse = int(bits[d_ix][at]), int(bits[c_ix][at])
            row = decode(t1, t2, direct, coarse, int(bits[f_ix][at]))
            seq.append({**row, "acc_hex": h32(row["acc"]), "direct": f"{direct:04x}",
                        "t1": t1, "coarse": bf16_value(coarse)})
        rows.append(seq)
    return rows


# --------------------------------------------------------------------------- the positions
def load(args):
    info = json.loads((args.dump / "dump.json").read_text())
    facts = {"container_sha256": chk.container_facts(args.container)["sha256"], "qdq_sha256": chk.sha16(args.qdq)}
    frames = {f["label"]: f for f in chk.load_frames(args.dump, facts)}
    ir = graph_ir.lower_yolov8n(str(args.qdq))
    layers = {L.name: L for L in ir.layers}
    picks = []
    for line in open(args.isolated_log, encoding="utf-8"):
        if not line.startswith("BF16_SR_ISOLATED "):
            continue
        r = json.loads(line.split(" ", 1)[1])
        for L in r["layers"]:
            for p in L.get("positions", []):
                if args.miss_model not in p:
                    raise SystemExit(f"{args.isolated_log} has no {args.miss_model!r} column; replay it with that model")
                if p[args.miss_model] != p["device"]:
                    picks.append((r["label"], L["layer"], p))
    return info, facts, frames, ir, layers, picks


def position_steps(ir, L, values: dict, c: int, y: int, x: int) -> tuple:
    """The lane's bias and its instructions, in the core's walk: chunk by chunk, then ky, kx, block."""
    if L.residual is not None or len(L.inputs) != 1:
        raise SystemExit(f"{L.name}: a residual or multi-input layer is not supported by this probe")
    seg = L.inputs[0]
    xin = values[seg.tensor][seg.block_offset * 8:(seg.block_offset + seg.blocks) * 8]
    w = eb.dequantized_weights(L).astype(F32)
    assert np.array_equal(em.to_bf16(w), w)
    steps = []
    for ch in layer_chunks(ir, L):
        if ch.kind == "res":
            continue
        # the blocks the packet header walks, which is what the core multiplies
        ncin = int(eb.unpack_w(eb.chunk_packet(L, c // eb.GROUP_CHANNELS, ch))[0][em.H_NCIN])
        for ky in range(L.k):
            for kx in range(L.k):
                yi, xi = L.stride * y - L.pad + ky, L.stride * x - L.pad + kx
                inside = 0 <= yi < xin.shape[1] and 0 <= xi < xin.shape[2]
                for cb in range(ch.block_start, ch.block_start + ncin):
                    a = xin[cb * 8:(cb + 1) * 8, yi, xi] if inside else np.zeros(8, F32)
                    ws = np.zeros(8, F32)
                    have = w[c, cb * 8:(cb + 1) * 8, ky, kx]
                    ws[:have.size] = have
                    steps.append((np.asarray(a, F32), ws, {"ky": ky, "kx": kx, "block": cb}))
    bias = float(em.to_bf16(eb.dequantized_bias(L)[c:c + 1])[0])
    return bias, steps


def scalar_chain(bias: float, steps: list, model: str) -> list:
    """One lane's fp32 accumulator after every instruction, through the emulator's own mac()."""
    acc, out = F32(bias), []
    for a, w, _ in steps:
        A = np.zeros((1, 1, 8), F32); A[0, 0] = a
        W = np.zeros((em.NCO, 8, 4), F32); W[0, :, 0] = w
        C = np.zeros((em.NCO, 1, 1, 4), F32); C[0, 0, 0, 0] = acc
        acc = F32(em.mac(C, A, W, model)[0, 0, 0, 0])
        out.append(float(acc))
    return out


def chain_experiment(label: str, bias: float, steps: list, models) -> dict:
    """Arm B: start at the first instruction where any model leaves the exact running sum."""
    exact = [bias]
    running = []
    for a, w, _ in steps:
        exact += [float(p) * float(q) for p, q in zip(a, w)]
        running.append(math.fsum(exact))
    per = {m: scalar_chain(bias, steps, m) for m in models}
    first = next((s for s in range(len(steps)) if any(per[m][s] != running[s] for m in models)), None)
    if first is None:
        raise SystemExit(f"{label}: no model leaves the exact sum, nothing to probe")
    start = bias if first == 0 else running[first - 1]
    assert all((bias if first == 0 else per[m][first - 1]) == start for m in models)
    assert float(F32(start)) == start
    return {"label": label, "start": start, "first_step": first,
            "steps": [(a, w) for a, w, _ in steps[first:]], "where": [s[2] for s in steps[first:]]}


# --------------------------------------------------------------------------- arm C
def discriminators() -> list:
    """(label, start, [(a, w), ...], what it separates). Every value is exact in bf16."""
    e = 2.0 ** -23
    lead2 = (1.5, 1.5)                                # 2.25: leads at 2^1, placed at 2^0 by ea + ew
    cancel = (-1.5, 1.0)
    V = [
        ("acc bit under a [2,4) product", 1 + e, [lead2, cancel], "aligned drops the accumulator's 2^-23; exp_sum and wide keep it"),
        ("acc 3 units under a [2,4) product", 1 + 3 * e, [lead2, cancel], "aligned rounds the tie up to 2^-21"),
        ("acc bit under a [2,4) product, negated", -(1 + e), [(-1.5, 1.5), (1.5, 1.0)], "sign symmetry of the first"),
        ("product bit under a [2,4) product", 0.0, [lead2, cancel, (2.0 ** -12, 2.0 ** -11)], "the 2^-23 is a product, not the accumulator"),
        ("half a step under exp_sum's grid", 0.0, [lead2, cancel, (2.0 ** -12, 2.0 ** -12)], "exp_sum ties to even and drops 2^-24; wide keeps it"),
        ("three quarters of a step", 0.0, [lead2, cancel, (2.0 ** -12, 1.5 * 2.0 ** -12)], "exp_sum rounds up to 2^-23; aligned drops it; wide keeps 1.5 x 2^-24"),
        ("near-4 significand product", 1 + e, [(255 / 128, 255 / 128), cancel, cancel], "a product just under 4 x its place"),
        ("control: significands under 2", 1 + e, [(1.25, 1.5), cancel],
         "1.875 leads at its place, so aligned and exp_sum agree (sequential, refuted 2026-09-21, differs)"),
        ("control: the accumulator leads", 2 + 2 * e, [lead2, cancel],
         "the accumulator sets the grid, so aligned and exp_sum agree (sequential differs)"),
        ("control: a larger place elsewhere", 1 + e, [lead2, (-1.0, 2.0)], "aligned and exp_sum agree (the -2 is placed at 2^1); wide differs"),
        ("acc bit under two [2,4) products", 1 + e, [lead2, lead2, cancel, cancel, cancel], "two products share the top place"),
        ("acc bit, [2,4) product in the last lane", 1 + e, [cancel] + [(0.0, 0.0)] * 6 + [lead2], "the product in slot 7 rather than slot 0"),
    ]
    for label, start, pairs, _ in V:
        vals = np.array([start] + [x for pr in pairs for x in pr], F32)
        assert float(F32(start)) == start, label
        assert np.array_equal(em.to_bf16(vals[1:]), vals[1:]), label
        assert len(pairs) <= 8, label
    return V


def discriminator_experiments() -> list:
    out = []
    for label, start, pairs, note in discriminators():
        a = np.zeros(8, F32); w = np.zeros(8, F32)
        for k, (x, y) in enumerate(pairs):
            a[k], w[k] = x, y
        out.append({"label": label, "start": float(start), "steps": [(a, w)], "note": note})
    return out


# --------------------------------------------------------------------------- arm A
def replay_packets(ir, L, values: dict, y0: int, x0: int, c: int) -> list:
    """The network's packets for the tile at (y0, x0) of c's output group, on the device's input."""
    t = ir.tensors[L.output]
    g = c // eb.GROUP_CHANNELS
    pkts = []
    for ch in layer_chunks(ir, L):
        if ch.kind == "res":
            raise SystemExit(f"{L.name}: arm A does not replay residual chunks")
        seg = L.inputs[ch.seg_index]
        src = values[seg.tensor][seg.block_offset * 8:(seg.block_offset + seg.blocks) * 8]
        xp = np.zeros((seg.blocks * 8, src.shape[1] + 2 * gr16.MARGIN, src.shape[2] + 2 * gr16.MARGIN), F32)
        xp[:src.shape[0], gr16.MARGIN:gr16.MARGIN + src.shape[1], gr16.MARGIN:gr16.MARGIN + src.shape[2]] = src
        hdr, bias, wts = eb.unpack_w(eb.chunk_packet(L, g, ch))
        act = gr16._window(xp, ch.block_start, int(hdr[em.H_NCIN]), L.stride * y0 - L.pad, L.stride * x0 - L.pad,
                           ch.rows_in, ch.cols_in)
        pkts.append((hdr, act, wts, bias))
    assert t.blocks * 8 >= c
    return pkts


def network_tile(dev_bits: np.ndarray, y0: int, x0: int, c: int) -> np.ndarray:
    """The tile the network wrote, in the emitted layout, as 16-bit patterns (+0 past the channels)."""
    g0 = (c // eb.GROUP_CHANNELS) * eb.GROUP_CHANNELS
    tile = np.zeros((eb.GROUP_CHANNELS, em.TILE_ROWS, em.TILE_COLS), np.uint16)
    part = dev_bits[g0:g0 + eb.GROUP_CHANNELS, y0:y0 + em.TILE_ROWS, x0:x0 + em.TILE_COLS]
    tile[:part.shape[0]] = part
    return tile.reshape(eb.OUT_BLOCKS_8, 8, em.TILE_ROWS, em.TILE_COLS).transpose(0, 2, 3, 1).reshape(-1)


def tile_slot(y: int, x: int, y0: int, x0: int, c: int) -> int:
    ch = c % eb.GROUP_CHANNELS
    return (ch // 8) * em.OUT_BLOCK8_ELEMS + ((y - y0) * em.TILE_COLS + (x - x0)) * 8 + ch % 8


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True, help="a --dump-frames directory of bf16_sr_silicon_check")
    ap.add_argument("--isolated-log", type=Path, required=True, help="a log with BF16_SR_ISOLATED lines for that dump")
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--container", type=Path, required=True, help="the container the dump came from (hash-checked)")
    ap.add_argument("--miss-model", default=em.MAC_MODEL, choices=em.MAC_MODELS,
                    help="probe the positions where this model's bits differ from the device's")
    ap.add_argument("--npu", action="store_true", help="dispatch on one core; without it, predictions only")
    ap.add_argument("--dispatches", type=int, default=2, help="--npu: identical pass-2 dispatches, for determinism")
    args = ap.parse_args()

    models = em.MAC_MODELS
    info, facts, frames, ir, layers, picks = load(args)
    emit("SETUP", {"host": platform.node(), "dump": chk.rel(args.dump), "isolated_log": chk.rel(args.isolated_log),
                   "qdq": chk.rel(args.qdq), "container": chk.rel(args.container), **facts,
                   "models": list(models), "in_force": em.MAC_MODEL, "miss_model": args.miss_model,
                   "positions": len(picks), "npu": args.npu})
    if not picks:
        raise SystemExit("no positions: the miss model matches the device everywhere in that log")
    if len(picks) > LANES:
        raise SystemExit(f"{len(picks)} positions; one group holds {LANES}")

    # --- arm B
    chains, replays = [], []
    for label, lname, p in picks:
        L = layers[lname]
        f = frames[label]
        values = {n: chk._values(b, ir.tensors[n].blocks) for n, b in f["tensors"].items()}
        bias, steps = position_steps(ir, L, values, p["c"], p["y"], p["x"])
        name = f"{label} {lname} c{p['c']} ({p['y']},{p['x']})"
        exp = chain_experiment(name, bias, steps, models)
        exp.update({"input": label, "layer": lname, "c": p["c"], "y": p["y"], "x": p["x"],
                    "device": p["device"], "final_by_model": {m: p.get(m) for m in models}})
        chains.append(exp)
        # --- arm A: every tile covering the position
        t = ir.tensors[L.output]
        for y0 in [o for o in tile_origins(t.height, em.TILE_ROWS) if o <= p["y"] < o + em.TILE_ROWS]:
            for x0 in [o for o in tile_origins(t.width, em.TILE_COLS) if o <= p["x"] < o + em.TILE_COLS]:
                replays.append({"name": name, "tile": [y0, x0], "slot": tile_slot(p["y"], p["x"], y0, x0, p["c"]),
                                "packets": replay_packets(ir, L, values, y0, x0, p["c"]),
                                "network": network_tile(f["tensors"][L.output], y0, x0, p["c"])})
    fresh = discriminator_experiments()

    def build(tb=None, tc=None):
        b_pkts, b_reads = chain_packets(chains, tb)
        c_pkts, c_reads = chain_packets(fresh, tc)
        off = len(b_pkts)
        c_reads = [(d + off, c + off, f + off, t) for d, c, f, t in c_reads]
        return (b_pkts + c_pkts + [pk for r in replays for pk in r["packets"]], b_reads, c_reads,
                len(b_pkts) + len(c_pkts), len(b_pkts))

    pkts, b_reads, c_reads, a_off, nb = build()
    emit("PLAN", {"packets": len(pkts), "arm_b_packets": nb, "arm_c_packets": a_off - nb, "arm_a_packets": len(pkts) - a_off,
                  "arm_b_experiments": len(chains), "arm_c_experiments": len(fresh), "arm_a_tiles": len(replays),
                  "passes": "pass 1 once, pass 2 --dispatches times"})

    # --- the operands, so the log stands alone
    for i, e in enumerate(chains):
        emit("OPERANDS", {"arm": "B", "i": i, "label": e["label"], "pixel": i, "lane": i, "first_step": e["first_step"],
                          "start": e["start"], "start_hex": h32(e["start"]), "device_final": e["device"],
                          "steps": [{**wh, "a": [float(v) for v in a], "w": [float(v) for v in w]}
                                    for (a, w), wh in zip(e["steps"], e["where"])]})
    for i, e in enumerate(fresh):
        a, w = e["steps"][0]
        emit("OPERANDS", {"arm": "C", "i": i, "label": e["label"], "pixel": i, "lane": i, "start": e["start"],
                          "start_hex": h32(e["start"]), "a": [float(v) for v in a], "w": [float(v) for v in w],
                          "separates": e["note"]})

    # --- predictions: every model's psum at every readout, from the pass-1 packets
    arms = (("B", chains), ("C", fresh))
    pred = run_models(pkts, models)
    predicted = {}
    for arm, exps in arms:
        reads = {"B": b_reads, "C": c_reads}[arm]
        for m in models:
            for i in range(len(exps)):
                for j, (d_ix, _, _, _) in enumerate(reads):
                    predicted[(arm, i, j, m)] = float(pred[m][1][d_ix - 1][psum_index(i, i)])
        for i, e in enumerate(exps):
            for j in range(len(reads)):
                if j == 0:
                    after = "assembled start"
                elif j > len(e["steps"]):
                    after = "padding (no instruction)"
                else:
                    after = f"step {e['first_step'] + j - 1}" if arm == "B" else "the instruction"
                emit("PREDICT", {"arm": arm, "i": i, "label": e["label"], "readout": j, "after": after,
                                 "models": {m: h32(predicted[(arm, i, j, m)]) for m in models},
                                 "values": {m: predicted[(arm, i, j, m)] for m in models}})

    # --- the two-pass readout, checked under every model on that model's own trajectory
    check = {}
    for m in models:
        bits1 = pred[m][0]
        rows1 = {arm: read_chain(bits1, exps, {"B": b_reads, "C": c_reads}[arm]) for arm, exps in arms}
        pkts2, b2, c2, _, _ = build(pass2_terms(rows1["B"]), pass2_terms(rows1["C"]))
        bits2 = run_models(pkts2, [m])[m][0]
        good = pass1_exact = n = 0
        for arm, exps in arms:
            rows2 = read_chain(bits2, exps, {"B": b2, "C": c2}[arm])
            for i, seq in enumerate(rows2):
                for j, row in enumerate(seq):
                    n += 1
                    pass1_exact += rows1[arm][i][j]["exact"]
                    good += row["exact"] and row["consistent"] and row["acc"] == predicted[(arm, i, j, m)]
        check[m] = {"readouts": n, "pass1_exact": pass1_exact, "pass2_equals_psum": good}
    ok = all(c["pass2_equals_psum"] == c["readouts"] for c in check.values())
    emit("SELFCHECK", {"by_model": check, "ok": ok})
    if not ok:
        raise SystemExit("the two-pass readout does not recover every model's accumulator; fix the probe first")

    k = a_off
    for r in replays:
        n = len(r["packets"])
        last = k + n - 1
        emit("REPLAY_PREDICT", {"name": r["name"], "tile": r["tile"], "network_at_position": f"{int(r['network'][r['slot']]):04x}",
                                "models_at_position": {m: f"{int(pred[m][0][last][r['slot']]):04x}" for m in models},
                                "tile_values_equal_network": {m: int(np.sum(pred[m][0][last] == r["network"])) for m in models},
                                "values": int(r["network"].size)})
        r["last"] = last
        k += n

    if not args.npu:
        emit("RESULT", {"npu": False, "note": "predictions only; no device was opened"})
        return 0

    # --- silicon
    sys.path.insert(0, str(ROOT / "kernels" / "bf16_conv"))
    import engine_bf16 as harness  # noqa: PLC0415 - IRON and ml_dtypes, only when a device is wanted

    bits1 = harness.dispatch(pkts)
    rows1 = {arm: read_chain(bits1, exps, {"B": b_reads, "C": c_reads}[arm]) for arm, exps in arms}
    pkts2, b2, c2, _, _ = build(pass2_terms(rows1["B"]), pass2_terms(rows1["C"]))
    runs = [harness.dispatch(pkts2) for _ in range(args.dispatches)]
    same = all(np.array_equal(runs[0], r) for r in runs[1:])
    bits = runs[0]
    score = {arm: {m: 0 for m in models} for arm in ("B", "C")}
    counts = {"B": 0, "C": 0}
    unread = 0
    for arm, exps in arms:
        rows = read_chain(bits, exps, {"B": b2, "C": c2}[arm])
        for i, seq in enumerate(rows):
            for j, row in enumerate(seq):
                again = rows1[arm][i][j]
                # the same accumulator in both passes: its direct bf16 must not move
                stable = again["direct"] == row["direct"]
                match = {m: row["exact"] and row["acc"] == predicted[(arm, i, j, m)] for m in models}
                counts[arm] += 1
                unread += not row["exact"]
                for m in models:
                    score[arm][m] += match[m]
                emit("SILICON", {"arm": arm, "i": i, "label": exps[i]["label"], "readout": j,
                                 "acc": row["acc"], "acc_hex": row["acc_hex"], "exact": row["exact"],
                                 "consistent": row["consistent"], "direct": row["direct"], "stable_across_passes": stable,
                                 "pass1_exact": again["exact"], "matches": [m for m in models if match[m]]})
    a_rows = []
    for r in replays:
        got = bits[r["last"]]
        row = {"name": r["name"], "tile": r["tile"], "one_core_at_position": f"{int(got[r['slot']]):04x}",
               "network_at_position": f"{int(r['network'][r['slot']]):04x}",
               "one_core_equals_network": int(np.sum(got == r["network"])),
               "one_core_equals_model": {m: int(np.sum(got == pred[m][0][r["last"]])) for m in models},
               "same_in_pass_1": bool(np.array_equal(got, bits1[r["last"]])), "values": int(got.size)}
        a_rows.append(row)
        emit("REPLAY", row)
    emit("RESULT", {"npu": True, "pass2_dispatches": args.dispatches, "pass2_identical": same,
                    "readouts_not_exact": unread,
                    "arm_b_readouts": counts["B"], "arm_b_matches_by_model": score["B"],
                    "arm_c_readouts": counts["C"], "arm_c_matches_by_model": score["C"],
                    "arm_a_tiles": len(a_rows),
                    "arm_a_tiles_equal_network": sum(r["one_core_equals_network"] == r["values"] for r in a_rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
