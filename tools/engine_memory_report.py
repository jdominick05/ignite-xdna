#!/usr/bin/env python3
"""Per-tile SRAM use of a built convolution-engine design, against the Phoenix tile capacities.

    python tools/engine_memory_report.py build/conv_engine/yolov8s [--json out.json]

Reads the placed design (``design.prj/input_with_addresses.mlir``): every ``aie.buffer`` with its
tile, byte size, address and bank, and every core's stack size. A compute tile has 64 KB of data
memory (4 banks of 16 KB) and a MemTile 512 KB (docs/SILICON.md, 1.2 and 1.3). The placer already
refuses a design that does not fit; this report says how close each tile is, and which buffers use it.
Weight packets, activation packets and output objects have fixed sizes in the engine, so a larger
model changes how many packets move, not what a tile holds.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CORE_BYTES = 64 * 1024
MEMTILE_BYTES = 512 * 1024
ELEMENT_BYTES = {"i8": 1, "ui8": 1, "i16": 2, "i32": 4, "ui32": 4, "i64": 8, "f32": 4}

TILE_RE = re.compile(r"%(\w+) = aie\.tile\((\d+), (\d+)\)")
BUFFER_RE = re.compile(r"aie\.buffer\(%(\w+)\)\s*\{([^}]*)\}\s*:\s*memref<([^>]+)>")
CORE_RE = re.compile(r"aie\.core\(%(\w+)\)")
STACK_RE = re.compile(r"stack_size\s*=\s*(\d+)")


def memref_bytes(spec: str) -> int:
    parts = spec.split("x")
    elem = parts[-1].strip()
    n = 1
    for d in parts[:-1]:
        n *= int(d)
    return n * ELEMENT_BYTES[elem]


def report(design_dir: Path) -> dict:
    text = (design_dir / "design.prj" / "input_with_addresses.mlir").read_text(encoding="utf-8")
    tiles = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in TILE_RE.finditer(text)}
    usage = defaultdict(lambda: {"buffers": [], "bytes": 0})
    for m in BUFFER_RE.finditer(text):
        tile = tiles[m.group(1)]
        attrs = m.group(2)
        name = re.search(r'sym_name = "([^"]+)"', attrs)
        bank = re.search(r"mem_bank = (\d+)", attrs)
        size = memref_bytes(m.group(3))
        entry = usage[tile]
        entry["buffers"].append({"name": name.group(1) if name else "?", "bytes": size,
                                 "bank": int(bank.group(1)) if bank else None})
        entry["bytes"] += size
    # A core's attribute dictionary (with stack_size) follows its region, before the next core.
    cores = list(CORE_RE.finditer(text))
    for i, cm in enumerate(cores):
        end = cores[i + 1].start() if i + 1 < len(cores) else len(text)
        sm = STACK_RE.search(text, cm.end(), end)
        if sm:
            usage[tiles[cm.group(1)]]["stack_bytes"] = int(sm.group(1))
    rows = []
    for (col, row), entry in sorted(usage.items()):
        kind = "shim" if row == 0 else "memtile" if row == 1 else "core"
        capacity = {"core": CORE_BYTES, "memtile": MEMTILE_BYTES}.get(kind)
        used = entry["bytes"] + entry.get("stack_bytes", 0)
        rows.append({"tile": [col, row], "kind": kind, "buffer_bytes": entry["bytes"],
                     "stack_bytes": entry.get("stack_bytes", 0), "used_bytes": used, "capacity_bytes": capacity,
                     "headroom_bytes": None if capacity is None else capacity - used, "buffers": entry["buffers"]})
    worst = {k: max((r for r in rows if r["kind"] == k), key=lambda r: r["used_bytes"], default=None)
             for k in ("core", "memtile")}
    return {"design": str(design_dir), "tiles": rows,
            "max_core_used_bytes": worst["core"]["used_bytes"] if worst["core"] else 0,
            "max_memtile_used_bytes": worst["memtile"]["used_bytes"] if worst["memtile"] else 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rep = report(Path(args.design_dir))
    for r in rep["tiles"]:
        if r["kind"] == "shim":
            continue
        big = sorted(r["buffers"], key=lambda b: -b["bytes"])[:4]
        cap = r["capacity_bytes"]
        print(f"tile {tuple(r['tile'])} {r['kind']:7s} used {r['used_bytes']:7,d} B of {cap:7,d} "
              f"({100.0 * r['used_bytes'] / cap:5.1f}%, headroom {r['headroom_bytes']:7,d}) stack {r['stack_bytes']:5,d} | "
              + ", ".join(f"{b['name']} {b['bytes']:,}" for b in big))
    print(f"max core tile {rep['max_core_used_bytes']:,} B of {CORE_BYTES:,}; "
          f"max MemTile {rep['max_memtile_used_bytes']:,} B of {MEMTILE_BYTES:,}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
