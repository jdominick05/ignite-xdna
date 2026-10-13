#!/usr/bin/env python3
"""How AMD's VitisAI DPU overlay moves data and control: a census of its PDIs and core programs.

    python -I tools/overlay_dataflow_census.py --pdi-dir <dir of *.pdi> [--route <pdi> --column 0]
    python -I tools/overlay_dataflow_census.py --disasm-dir <dir of Peano .s files>

Inputs are AMD binaries extracted by tools/xclbin_overlay_extract.py (branch dpu-isa-re), kept
under git-ignored scratch/; this tool prints counts and decoded register values only, never code.

--pdi-dir   per PDI: its sha256, every MASTER_CONFIG_TILE_CTRL value by tile kind, the shim's
            packet slots on SOUTH_3, the shim mux, and how many DMA buffer-descriptor registers
            the PDI writes on each tile kind.
--route     one PDI, one column: every stream-switch slot and master decoded into routes.
            A packet slot (id, msel, arbiter) reaches each packet master whose arbiter matches and
            whose msel-enable mask has bit msel (aie-rt: master configuration = msel_enable << 3 |
            arbiter, drop-header bit 7). A circuit master's configuration is a slave-port index,
            resolved by the order of that tile kind's SLAVE_CONFIG registers in the header.
--disasm-dir  across Peano disassemblies of the core images: counts of st.tm, lock acquire and
            release, and every immediate that resolves to a named tile register as 0x80000 +
            offset (Peano's aiev2intrin.h: write_tm(..., TMAddrSpaceStart = 0x80000)). The bare
            base 0x80000 is excluded.

Register names come from aie-rt's xaiemlgbl_params.h (--header, default: the mlir-aie checkout).
The PDI record walk is copied from tools/xclbin_overlay_extract.py; its layout is inferred from
bytes, as that tool's docstring says. Run on the Phoenix 4x4 overlay only (Desktop 2, 2026-10-13).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import re
import struct
from pathlib import Path

DEFAULT_HEADER = Path.home() / "mlir-aie" / "third_party" / "aie-rt" / "driver" / "src" / "global" / "xaiemlgbl_params.h"
CDO_MAGIC = 0x004F4443
TM_BASE = 0x80000
PREFIX = {"shim": ("NOC_MODULE", "PL_MODULE"), "mem": ("MEM_TILE_MODULE",), "core": ("CORE_MODULE", "MEMORY_MODULE")}


def load_regs(header: Path):
    names = {}
    for m in re.finditer(r"#define\s+XAIEMLGBL_(\w+)\s+0x([0-9A-Fa-f]+)U?\b", header.read_text(errors="replace")):
        n, v = m.group(1), int(m.group(2), 16)
        if n.endswith(("_MASK", "_LSB", "_WIDTH", "_DEFVAL")):
            continue
        for kind, pres in PREFIX.items():
            if n.startswith(pres):
                names.setdefault((kind, v), n)
    return names


def kind_of(row):
    return "shim" if row == 0 else "mem" if row == 1 else "core"


def pdi_writes(blob: bytes):
    """(col, row, offset, value) for every type-3 write and type-2 masked write."""
    o = blob.find(struct.pack("<I", CDO_MAGIC))
    if o < 4:
        raise SystemExit("no CDO header")
    o += 16
    n = (len(blob) - o) // 4
    w = struct.unpack_from("<%dI" % n, blob, o)
    i, out = 0, []
    while i < n:
        t, c = w[i], w[i + 1]
        if t == 0 and c == 0xFFFFFFFF:
            return out
        if t == 3:
            pairs = [(w[i + 2 + 2 * k], w[i + 3 + 2 * k]) for k in range(c)]
            i += 2 + 2 * c
        elif t == 2:
            pairs = [(w[i + 2 + 3 * k], w[i + 4 + 3 * k]) for k in range(c)]
            i += 2 + 3 * c
        elif t == 5:
            pairs = []
            i += 2 + 4 * c
        else:
            raise SystemExit(f"unknown CDO record type {t:#x} at word {i}")
        out += [(a >> 25, (a >> 20) & 0x1F, a & 0xFFFFF, v) for a, v in pairs]
    raise SystemExit("no CDO terminator")


def pdi_summary(p: Path, names) -> str:
    ctrl, bds = collections.Counter(), collections.Counter()
    slots, mux = 0, set()
    for col, row, off, val in pdi_writes(p.read_bytes()):
        k = kind_of(row)
        n = names.get((k, off), "")
        if n.endswith("MASTER_CONFIG_TILE_CTRL"):
            ctrl[f"{k}={val:#010x}"] += 1
        if k == "shim" and "SOUTH_3_SLOT" in n and val & 0x100:
            slots += 1
        if n == "NOC_MODULE_MUX_CONFIG":
            mux.add(f"{val:#x}")
        if "_DMA_BD" in n:
            bds[k] += 1
    return (f"PDI {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]} "
            f"tile_ctrl_masters {dict(sorted(ctrl.items()))} shim_south3_slots {slots} "
            f"shim_mux {sorted(mux)} dma_bd_writes {dict(bds) or 0}")


def routes(p: Path, column: int, names):
    slave_order = {}
    for (k, off), n in sorted(names.items(), key=lambda kv: kv[0][1]):
        if "STREAM_SWITCH_SLAVE_CONFIG_" in n:
            slave_order.setdefault(k, []).append(n.split("SLAVE_CONFIG_")[1])
    tiles = collections.defaultdict(dict)
    for col, row, off, val in pdi_writes(p.read_bytes()):
        if col == column:
            n = names.get((kind_of(row), off))
            if n and "STREAM_SWITCH" in n:
                tiles[row][n.split("STREAM_SWITCH_")[1]] = val
    for row in sorted(tiles):
        k, regs = kind_of(row), tiles[row]
        masters = {n.split("MASTER_CONFIG_")[1]: v for n, v in regs.items() if n.startswith("MASTER_CONFIG_") and v >> 31}
        for n, v in sorted(regs.items()):
            m = re.match(r"SLAVE_(\w+?)_SLOT(\d)$", n)
            if m and v & 0x100:
                pid, msel, arb = (v >> 24) & 0x1F, (v >> 4) & 3, v & 7
                dest = [mn for mn, mv in masters.items() if (mv >> 30) & 1 and (mv & 7) == arb
                        and ((mv >> 3) & 0xF) >> msel & 1]
                drop = [mn for mn in dest if masters[mn] & 0x80]
                yield (f"ROUTE c{column}r{row} {k} {m.group(1)} packet id {pid} -> {','.join(dest) or '?'}"
                       + (f" (header dropped at {','.join(drop)})" if drop else ""))
        for mn, mv in sorted(masters.items()):
            if not (mv >> 30) & 1:
                idx = mv & 0x7F
                src = slave_order[k][idx] if idx < len(slave_order[k]) else f"index {idx}"
                yield f"ROUTE c{column}r{row} {k} circuit {src} -> {mn}"


def disasm_census(d: Path, names) -> list[str]:
    imm_re = re.compile(r"\b(?:movxm|mova|movx|mov)\s+\w+,\s*#(-?0x[0-9a-f]+|-?\d+)")
    line_re = re.compile(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2} )+\s*(.*)$")
    ops, files = collections.Counter(), 0
    regs = collections.Counter()
    for f in sorted(d.glob("*.s")):
        files += 1
        for ln in f.read_text(errors="replace").splitlines():
            m = line_re.match(ln)
            for slot in (m.group(1).split(";") if m else ()):
                tok = slot.split()
                if tok and tok[0] in ("st.tm", "acq", "rel", "acq.cond", "rel.cond"):
                    ops[tok[0]] += 1
            for m in imm_re.finditer(ln):
                v = int(m.group(1), 0) & 0xFFFFFFFF
                if TM_BASE < v < 2 * TM_BASE and ("core", v - TM_BASE) in names:
                    regs[names[("core", v - TM_BASE)]] += 1
    out = [f"DISASM files {files} " + " ".join(f"{k}={v}" for k, v in sorted(ops.items()))]
    out += [f"TM_REGISTER {c} {n}" for n, c in regs.most_common()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    ap.add_argument("--pdi-dir", type=Path)
    ap.add_argument("--route", type=Path, help="one PDI to decode routes for")
    ap.add_argument("--column", type=int, default=0)
    ap.add_argument("--disasm-dir", type=Path)
    a = ap.parse_args()
    names = load_regs(a.header)
    print(f"HEADER {a.header.name} sha256 {hashlib.sha256(a.header.read_bytes()).hexdigest()[:16]} "
          f"{len(names)} registers", flush=True)
    if a.pdi_dir:
        for p in sorted(a.pdi_dir.glob("*.pdi")):
            print(pdi_summary(p, names), flush=True)
    if a.route:
        for line in routes(a.route, a.column, names):
            print(line, flush=True)
    if a.disasm_dir:
        for line in disasm_census(a.disasm_dir, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
