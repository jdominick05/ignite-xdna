#!/usr/bin/env python3
"""What AMD's compiled model declares it does per frame, beside what an engine container does.

    python -I tools/amd_xmodel_census.py --xmodel yolov8s=<cache>/compiled.<fp>.xmodel \\
        [--container yolov8s=build/yolov8s.ignite]

Reads the VitisAI EP's compiled .xmodel (an XIR protobuf) with no vendor library: every DPU
subgraph's attributes - the compiler's own per-frame byte counts (input_feature_load_size,
output_feature_save_size, const_data_load_size), its workload, column count, the register table
(reg_id_to_size / reg_id_to_context_type) - and the mc_code blob the DPU runner executes.

The protobuf layout used here was read off the file, not from a schema: the root graph's field 10
is the subgraph tree (children again field 10, name field 1, attributes field 5 as
{1: key, 2: AttrValue}); integer attributes sit in AttrValue fields 4/5/8, strings in field 11,
bytes in field 12 -> 1, and string maps in field 26 / int maps in field 22 as repeated
{1: key, 2: value}. Anything that does not parse that way is printed as raw sizes, never guessed.

mc_code is summarised, not decoded: its length, the count of 32-bit words whose high byte is
0x0B (on YOLOv8s these start 48-byte records, e.g. 0b00005f ... 0b00005e, 12 words apart), and
the share of words that are printable ASCII. No field meaning is claimed for any record; see
tools/dpu_transaction_disasm.py's docstring on the earlier heuristic decoder and why its output is
not a decode.

--container (needs ignite_xdna importable, e.g. the mlir-aie-iron env) prints the engine
container's manifest counts beside it: activation packets (6,400 B each), weight-packet bytes,
instruction-stream bytes and rounds. Both sides are static declarations; neither is a timing.
Run so far on YOLOv8n and YOLOv8s only (Desktop 2, 2026-10-13).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def fields(b, lo, hi):
    """[(field, wiretype, int | (start, end))]; raises ValueError when b[lo:hi] is not a message."""
    out, i = [], lo
    while i < hi:
        key, i = varint(b, i)
        f, wt = key >> 3, key & 7
        if f == 0:
            raise ValueError("field 0")
        if wt == 0:
            v, i = varint(b, i)
            out.append((f, wt, v))
        elif wt == 2:
            n, i = varint(b, i)
            if i + n > hi:
                raise ValueError("overrun")
            out.append((f, wt, (i, i + n)))
            i += n
        elif wt in (1, 5):
            w = 8 if wt == 1 else 4
            out.append((f, wt, int.from_bytes(b[i:i + w], "little")))
            i += w
        else:
            raise ValueError(f"wiretype {wt}")
    if i != hi:
        raise ValueError("trailing bytes")
    return out


def text(b, span):
    return b[span[0]:span[1]].decode("utf-8", "replace")


def attr_value(b, span):
    """Decode one AttrValue into a python value, or a {'raw_field': n, 'bytes': len} stub."""
    fs = fields(b, *span)
    if len(fs) != 1:
        return {"raw_fields": [f for f, _, _ in fs]}
    f, wt, v = fs[0]
    if wt == 0:
        return v
    if f == 11:
        return text(b, v)
    if f == 12:
        inner = fields(b, *v)
        if len(inner) == 1 and inner[0][0] == 1 and inner[0][1] == 2:
            return {"bytes": inner[0][2]}
    if f in (22, 26):
        out = {}
        for ef, ewt, ev in fields(b, *v):
            if ewt != 2:
                continue
            kv = {kf: kv_ for kf, _, kv_ in fields(b, *ev)}
            k = text(b, kv[1])
            out[k] = text(b, kv[2]) if f == 26 else kv[2]
        return out
    return {"raw_field": f, "bytes": v[1] - v[0] if isinstance(v, tuple) else None}


def subgraphs(b, span, depth=0):
    """Yield (depth, name, {attr: value}) for every subgraph under span, depth-first."""
    name, attrs, children = None, {}, []
    for f, wt, v in fields(b, *span):
        if f == 1 and wt == 2:
            name = text(b, v)
        elif f == 5 and wt == 2:
            kv = {kf: kv_ for kf, _, kv_ in fields(b, *v)}
            attrs[text(b, kv[1])] = kv.get(2)
        elif f == 10 and wt == 2:
            children.append(v)
    yield depth, name, attrs
    for c in children:
        yield from subgraphs(b, c, depth + 1)


def mc_summary(blob: bytes) -> dict:
    n = len(blob) // 4
    words = [int.from_bytes(blob[4 * k:4 * k + 4], "little") for k in range(n)]
    printable = sum(1 for w in words if all(32 <= (w >> s) & 0xFF < 127 for s in (0, 8, 16, 24)))
    return {"bytes": len(blob), "words": n, "words_hi_0x0B": sum(1 for w in words if w >> 24 == 0x0B),
            "printable_ascii_words": printable, "sha256": hashlib.sha256(blob).hexdigest()[:16]}


SCALARS = ("device", "dpu_name", "column_number", "workload", "workload_on_arch",
           "input_feature_load_size", "output_feature_save_size", "const_data_load_size",
           "num_of_pdi_subgraphs", "enable_pad_control_packet", "enable_halo_mt_copy", "debug_mode")


def census_xmodel(path: Path) -> list[dict]:
    b = path.read_bytes()
    top = [v for f, wt, v in fields(b, 0, len(b)) if f == 10 and wt == 2]
    if len(top) != 1:
        raise SystemExit(f"{path}: expected one subgraph root, found {len(top)}")
    rows = []
    for depth, name, attrs in subgraphs(b, top[0]):
        vals = {}
        for k, span in attrs.items():
            if k in SCALARS or k.startswith("reg_id_to_") and k != "reg_id_to_parameter_value":
                try:
                    vals[k] = attr_value(b, span)
                except (ValueError, KeyError):
                    vals[k] = "unparsed"
        if vals.get("device") != "DPU" and "mc_code" not in attrs:
            continue
        row = {"subgraph": name, "depth": depth, **vals}
        if "mc_code" in attrs:
            mc = attr_value(b, attrs["mc_code"])
            if isinstance(mc, dict) and "bytes" in mc and isinstance(mc["bytes"], tuple):
                row["mc_code"] = mc_summary(b[mc["bytes"][0]:mc["bytes"][1]])
        if "reg_id_to_parameter_value" in attrs:
            row["reg_id_to_parameter_value_bytes"] = attrs["reg_id_to_parameter_value"][1] - attrs["reg_id_to_parameter_value"][0]
        rows.append(row)
    return rows


def census_container(path: Path) -> dict:
    from ignite_xdna.compiler.serializer import IgniteModelReader
    m = IgniteModelReader(str(path)).manifest
    ge = m.get("graph_engine", m)
    out = {k: ge.get(k) for k in ("activation_packets", "wpackets_bytes", "insts_bytes", "rounds",
                                  "workspace_bytes", "weight_fills", "silu")}
    out["activation_bytes"] = int(ge["activation_packets"]) * 6400
    return out


def pairs(specs):
    out = []
    for s in specs or ():
        name, _, p = s.partition("=")
        out.append((name, Path(p)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xmodel", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--container", action="append", default=None, metavar="NAME=PATH")
    a = ap.parse_args()
    for name, p in pairs(a.xmodel):
        print(f"AMD_XMODEL {name} {p.name} {p.stat().st_size} B sha256 "
              f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}", flush=True)
        for row in census_xmodel(p):
            print("AMD_SUBGRAPH " + json.dumps({"model": name, **row}, sort_keys=True), flush=True)
    for name, p in pairs(a.container):
        print("ENGINE_CONTAINER " + json.dumps({"model": name, "container": p.name, **census_container(p)},
                                               sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
