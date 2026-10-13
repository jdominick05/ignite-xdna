#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode what AMD's compiled VitisAI DPU program configures per layer: shim and MemTile DMA descriptors.

    python -I tools/amd_code_decode.py --xmodel <compiled.xmodel> [--layer <name substring>] [--column C]
    python -I tools/amd_code_decode.py --xmodel <compiled.xmodel> --totals [--json out.json]

What the bytes are, as decoded (2026-10-13, Desktop 2, YOLOv8s default and cost-model arms):
- The xmodel's `reg_id_to_parameter_value` holds REG_0 (CONST, the weights) and REG_4 (CODE), each with a
  protobuf length prefix. `mc_code` is a list of records, each starting with a `0x0B` word. Most are 48 bytes:
  word 0 is `0x0B` | column << 16 | a low byte 0x5c..0x5f; word 2 is a byte offset into REG_4; the last word
  carries (len_units - 1) << 16 | an id. Each names a chunk of REG_4 of len_units x 32 bytes. Records whose
  bytes include a long ASCII tensor name start a new layer segment. That name is the layer's INPUT tensor: the
  segment named after model.6's output is model.7 (checked by bytes against AMD's declared per-layer loads).
  Only the 48-byte 0x5c..0x5f records are decoded. The cost-model build's other record types (0x5a, 0x37..0x3e)
  are not, and 9 of its 90 segments do not reconcile with the declared bytes.
- A REG_4 chunk is a run of 32-byte AIE2 control packets: a stream header whose low bits are the packet id
  (1 reaches the shim's control port and 2 the MemTile's, as the overlay PDI routes them:
  tools/overlay_dataflow_census.py), a control header (address 19:0, beats - 1 at 21:20, operation 23:22), then
  the data words. Each one is a register write.
- Shim (NOC) and MemTile DMA buffer descriptors, and their channels' start queues, are then decoded with
  aie-rt's xaiemlgbl_params.h field layouts (SPEC).

What is inferred rather than read: the packet-id-to-tile mapping (from the PDI routing, one overlay), and that a
shim channel's bytes per push are its chain's buffer lengths times (repeat + 1). MemTile chains cycle under
locks, so their totals are not computed. Cores program their own DMAs (`st.tm`, overlay_dataflow_census), so
no core descriptor appears here.

--totals prints, per layer segment, the shim's DDR bytes per direction (reads also split by DDR region tag,
address bits 47:32: 0x3000 activations, 0x1000 constants), and whether the pushing columns read the same regions.
Column 3's shim descriptors are written but no start-queue write for it appears, so push-derived totals cover
columns 0-2: 3/4 of AMD's declared per-layer loads, exactly, on the segments that join (who starts column 3's
transfers is unexplained).
Reads AMD's compiled artifact; writes nothing of it out. Run on YOLOv8s's two arms only.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import amd_xmodel_census as cz  # noqa: E402

PKT_SHIM, PKT_MEMTILE = 1, 2
MT_BD0, MT_BD_END, MT_QUEUE0 = 0xA0000, 0xA0600, 0xA0600
SHIM_BD0, SHIM_BD_END, SHIM_QUEUE0 = 0x1D000, 0x1D200, 0x1D200


def bits(v, lo, n):
    return (v >> lo) & ((1 << n) - 1)


def xmodel_parts(path: Path):
    """(mc_code bytes, REG_4 bytes) of the largest mc_code subgraph."""
    b = path.read_bytes()
    top = [v for f, wt, v in cz.fields(b, 0, len(b)) if f == 10 and wt == 2]
    mc, code = None, None
    for _, _, attrs in cz.subgraphs(b, top[0]):
        if "mc_code" in attrs and mc is None:
            s, e = cz.attr_value(b, attrs["mc_code"])["bytes"]
            mc = b[s:e]
        if "reg_id_to_parameter_value" in attrs and code is None:
            outer = cz.fields(b, *attrs["reg_id_to_parameter_value"])[0][2]
            for _, _, ev in cz.fields(b, *outer):
                kv = {kf: kv_ for kf, _, kv_ in cz.fields(b, *ev)}
                if cz.text(b, kv[1]) == "REG_4":
                    s, e = kv[2]
                    blob = b[s:e]
                    # The value is a bytes message: a one-byte tag (0x0a) and a varint length, then the CODE.
                    assert blob[0] == 0x0A, hex(blob[0])
                    n, i = cz.varint(blob, 1)
                    code = blob[i:i + n]
                    assert len(code) == n, (len(code), n)
    if mc is None or code is None:
        raise SystemExit(f"{path}: no mc_code or no REG_4 found")
    return mc, code


def segments(mc: bytes):
    words = [int.from_bytes(mc[i:i + 4], "little") for i in range(0, len(mc) - 3, 4)]
    starts = [i for i, w in enumerate(words) if w >> 24 == 0x0B]
    out, cur = [], ["<prologue>", []]
    for a, b in zip(starts, starts[1:] + [len(words)]):
        rec = words[a:b]
        raw = b"".join(w.to_bytes(4, "little") for w in rec)
        names = re.findall(rb"[A-Za-z0-9_./]{12,}", raw)
        if names:
            out.append(cur)
            cur = [names[0].decode(), []]
        if len(rec) == 12:
            cur[1].append(rec)
    out.append(cur)
    return out


def writes(code: bytes, rec):
    """(packet id, address, value) for every register write in the REG_4 chunk an mc_code record names."""
    off, n = rec[2], ((rec[11] >> 16) + 1) * 32
    blob = code[off:off + n]
    ws = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob) - 3, 4)]
    for i in range(0, len(ws) - 1, 8):
        pkt, ctl = ws[i] & 0x1F, ws[i + 1]
        if bits(ctl, 22, 2) != 0:
            continue                                  # not a write
        addr, beats = bits(ctl, 0, 20), bits(ctl, 20, 2) + 1
        for k, v in enumerate(ws[i + 2:i + 2 + beats]):
            yield pkt, addr + 4 * k, v


def shim_bd(w):
    return {"bytes": w[0] * 4, "addr": (bits(w[2], 0, 16) << 32) | (w[1] & 0xFFFFFFFC),
            "pkt": (bits(w[2], 30, 1), bits(w[2], 19, 5)),
            "d0": (bits(w[3], 20, 10), bits(w[3], 0, 20) + 1), "d1": (bits(w[4], 20, 10), bits(w[4], 0, 20) + 1),
            "d2_step": bits(w[5], 0, 20) + 1, "iter": (bits(w[6], 20, 6) + 1, bits(w[6], 0, 20) + 1),
            "next": bits(w[7], 27, 4) if bits(w[7], 26, 1) else None, "valid": bits(w[7], 25, 1)}


def memtile_bd(w):
    return {"bytes": bits(w[0], 0, 17) * 4, "base": bits(w[1], 0, 19) * 4,
            "pkt": (bits(w[0], 31, 1), bits(w[0], 23, 5)),
            "next": bits(w[1], 20, 6) if bits(w[1], 19, 1) else None,
            "d0": (bits(w[2], 17, 10), bits(w[2], 0, 17) + 1), "d1": (bits(w[3], 17, 10), bits(w[3], 0, 17) + 1),
            "d2": (bits(w[4], 17, 10), bits(w[4], 0, 17) + 1), "d3_step": bits(w[5], 0, 17) + 1,
            "iter": (bits(w[6], 17, 6) + 1, bits(w[6], 0, 17) + 1),
            "acq": (bits(w[7], 0, 8), bits(w[7], 8, 7) - (128 if bits(w[7], 14, 1) else 0), bits(w[7], 15, 1)),
            "rel": (bits(w[7], 16, 8), bits(w[7], 24, 7) - (128 if bits(w[7], 30, 1) else 0)),
            "valid": bits(w[7], 31, 1)}


SHIM_QUEUES = {0x1D204: "S2MM_0", 0x1D20C: "S2MM_1", 0x1D214: "MM2S_0", 0x1D21C: "MM2S_1"}
MT_QUEUES = {0xA0604 + 8 * k: f"S2MM_{k}" for k in range(6)}
MT_QUEUES.update({0xA0634 + 8 * k: f"MM2S_{k}" for k in range(6)})


def decode_column(code, recs, col):
    """Per column of one segment: shim BDs, MemTile BDs, and queue pushes in program order."""
    shim, mt = defaultdict(dict), defaultdict(dict)
    pushes = []                                      # (tile, channel, start bd, repeat, token)
    for rec in recs:
        if bits(rec[0], 16, 8) != col:
            continue
        for pkt, addr, v in writes(code, rec):
            if pkt == PKT_SHIM and SHIM_BD0 <= addr < SHIM_BD_END:
                shim[(addr - SHIM_BD0) // 0x20][(addr - SHIM_BD0) % 0x20 // 4] = v
            elif pkt == PKT_SHIM and addr in SHIM_QUEUES:
                pushes.append(("shim", SHIM_QUEUES[addr], bits(v, 0, 4), bits(v, 16, 8), bits(v, 31, 1),
                               {k: dict(w) for k, w in shim.items()}))
            elif pkt == PKT_MEMTILE and MT_BD0 <= addr < MT_BD_END:
                mt[(addr - MT_BD0) // 0x20][(addr - MT_BD0) % 0x20 // 4] = v
            elif pkt == PKT_MEMTILE and addr in MT_QUEUES:
                pushes.append(("memtile", MT_QUEUES[addr], bits(v, 0, 6), bits(v, 16, 8), bits(v, 31, 1), None))
    return shim, mt, pushes


def chain_bytes(table, start, decoder):
    """Bytes one pass of a descriptor chain moves, and the chain; None for bytes if the chain cycles."""
    seen, total, bd = [], 0, start
    while bd is not None:
        if bd in seen:
            return None, seen + [bd]
        seen.append(bd)
        w = table.get(bd)
        if w is None:
            return None, seen
        d = decoder([w.get(k, 0) for k in range(8)])
        total += d["bytes"]
        bd = d["next"]
    return total, seen


def shim_bd_episodes(code, recs, col):
    """Every completed shim BD programming in one column: the BD's state each time its last word (7) is written."""
    shim, out = defaultdict(dict), []
    for rec in recs:
        if bits(rec[0], 16, 8) != col:
            continue
        for pkt, addr, v in writes(code, rec):
            if pkt == PKT_SHIM and SHIM_BD0 <= addr < SHIM_BD_END:
                k, j = (addr - SHIM_BD0) // 0x20, (addr - SHIM_BD0) % 0x20 // 4
                shim[k][j] = v
                if j == 7:
                    out.append((k, shim_bd([shim[k].get(i, 0) for i in range(8)])))
    return out


def segment_totals(code, recs, cols=4):
    """Per segment: push-derived shim bytes (by direction, and reads by DDR region tag = address bits 47:32),
    per column; and the sum of every programmed shim BD's length, per column, as the push-free cross-check."""
    t = {"shim_mm2s": 0, "shim_s2mm": 0, "unresolved": 0, "mm2s_by_tag": defaultdict(int),
         "push_bytes_by_col": [0] * cols, "bd_bytes_by_col": [0] * cols, "repeats": 0}
    for c in range(cols):
        _, _, pushes = decode_column(code, recs, c)
        for tile, ch, bd, rep, _, snapshot in pushes:
            if tile != "shim":
                continue
            n, chain = chain_bytes(snapshot, bd, shim_bd)
            if n is None:
                t["unresolved"] += 1
                continue
            t["repeats"] += rep
            t["shim_mm2s" if ch.startswith("MM2S") else "shim_s2mm"] += n * (rep + 1)
            t["push_bytes_by_col"][c] += n * (rep + 1)
            if ch.startswith("MM2S"):
                for k in chain:
                    d = shim_bd([snapshot[k].get(j, 0) for j in range(8)])
                    t["mm2s_by_tag"][f"{d['addr'] >> 32:#x}"] += d["bytes"] * (rep + 1)
        t["bd_bytes_by_col"][c] = sum(d["bytes"] for _, d in shim_bd_episodes(code, recs, c))
    t["mm2s_by_tag"] = dict(t["mm2s_by_tag"])
    # Column split: do the pushing columns read the same DDR regions (by tag) or different ones?
    regions = [defaultdict(set) for _ in range(cols)]
    for c in range(cols):
        for tile, ch, bd, rep, _, snapshot in decode_column(code, recs, c)[2]:
            if tile == "shim" and ch.startswith("MM2S"):
                n, chain = chain_bytes(snapshot, bd, shim_bd)
                for k in chain if n is not None else []:
                    d = shim_bd([snapshot[k].get(j, 0) for j in range(8)])
                    regions[c][f"{d['addr'] >> 32:#x}"].add((d["addr"], d["bytes"], d["iter"]))
    pushing = [c for c in range(cols) if regions[c]]
    t["same_across_columns"] = {tag: all(regions[c][tag] == regions[pushing[0]][tag] for c in pushing)
                                for tag in (regions[pushing[0]] if pushing else {})}
    return t


def show_layer(code, name, recs, col):
    print(f"##### {name}  column {col}")
    shim, mt, pushes = decode_column(code, recs, col)
    for k in sorted(shim):
        d = shim_bd([shim[k].get(j, 0) for j in range(8)])
        print(f"  shim BD{k:2d} {d['bytes']:8d} B addr {d['addr']:#11x} d0(wrap,step) {d['d0']} d1 {d['d1']} "
              f"d2step {d['d2_step']} iter(wrap,step) {d['iter']} next {d['next']} pkt {d['pkt']}")
    for k in sorted(mt):
        d = memtile_bd([mt[k].get(j, 0) for j in range(8)])
        print(f"  mt BD{k:2d} {d['bytes']:7d} B base {d['base']:#8x} d0 {d['d0']} d1 {d['d1']} d2 {d['d2']} "
              f"d3step {d['d3_step']} iter {d['iter']} next {d['next']} pkt {d['pkt']} acq {d['acq']} rel {d['rel']}")
    for tile, ch, bd, rep, tok, snapshot in pushes:
        extra = ""
        if tile == "shim":
            n, chain = chain_bytes(snapshot, bd, shim_bd)
            extra = f" chain {chain} = {n} B x {rep + 1}" if n is not None else f" chain {chain} (cycle or unknown)"
            if n is not None:
                d = shim_bd([snapshot[chain[0]].get(j, 0) for j in range(8)])
                extra += (f" | tag {d['addr'] >> 32:#x} off {d['addr'] & 0xFFFFFFFF:#x} d0 {d['d0']} d1 {d['d1']}"
                          f" d2step {d['d2_step']} iter {d['iter']}")
        print(f"  push {tile:7s} {ch:7s} BD{bd:2d} repeat {rep} token {tok}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xmodel", type=Path, required=True)
    ap.add_argument("--layer", default=None, help="substring of a segment name: dump that segment")
    ap.add_argument("--column", type=int, default=None, help="with --layer: one column (default all four)")
    ap.add_argument("--totals", action="store_true", help="per segment shim DDR bytes, all columns")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    mc, code = xmodel_parts(a.xmodel)
    segs = segments(mc)
    print(f"[decode] {a.xmodel}: mc_code {len(mc)} B, CODE {len(code)} B, {len(segs)} segments")
    if a.layer:
        for name, recs in segs:
            if a.layer in name:
                for c in ([a.column] if a.column is not None else range(4)):
                    show_layer(code, name, recs, c)
                break
        else:
            raise SystemExit(f"no segment matches {a.layer!r}")
    if a.totals:
        rows = []
        for i, (name, recs) in enumerate(segs):
            t = segment_totals(code, recs)
            rows.append({"segment": i, "name": name, **t})
            print("SEGMENT " + json.dumps(rows[-1]), flush=True)
        if a.json:
            a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
