#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode an aiecc NPU instruction stream (insts.bin) op by op, including the reconfiguration ops.

tools/disasm_txn.py stops at opcodes it does not know. This one follows the layouts in the installed aie-rt header
(mlir_aie/runtime_lib/x86_64/xaiengine/include/xaiengine/xaie_txn.h): XAie_TxnHeader (16 B), then per op:
    0 WRITE        24 B   RegOff u64 @8, Value @16
    1 BLOCKWRITE   16 B + payload; RegOff u32 @8, Size (whole op, bytes) @12
    3 MASKWRITE    28 B   RegOff u64 @8, Value @16, Mask @20
    4/7 MASKPOLL   28 B   as MASKWRITE
    5 NOOP, 6 PREEMPT  4 B
    8 LOADPDI      16 B   PdiId u16 @2, PdiSize u32 @4, PdiAddress u64 @8
    >= 0x80 custom ops: Size (whole op) @4
A register offset is col << 25 | row << 20 | tile offset (AIE2). ``--ops`` prints every op; the default is a summary
per (op, row, tile-offset region).

    python tools/txn_decode.py STREAM.bin [--ops] [--strip-loadpdi OUT.bin]

``--strip-loadpdi`` writes the same stream without its LOADPDI ops (header op count and size fixed up), for a probe
that keeps everything else. Written on Desktop 2, 2026-10-19.
"""
import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

NAMES = {0: "WRITE", 1: "BLOCKWRITE", 2: "BLOCKSET", 3: "MASKWRITE", 4: "MASKPOLL", 5: "NOOP", 6: "PREEMPT",
         7: "MASKPOLL_BUSY", 8: "LOADPDI", 9: "LOAD_PM_START", 0x80: "TCT", 0x81: "DDR_PATCH", 0x82: "READ_REGS",
         0x83: "RECORD_TIMER", 0x84: "MERGE_SYNC"}


def where(off):
    return off >> 25 & 0x7F, off >> 20 & 0x1F, off & 0xFFFFF


def ops(data):
    """-> header dict, list of (byte offset, size, opcode, fields)."""
    major, minor, gen, rows, cols, mrows, _, nops, size = struct.unpack_from("<6BHII", data, 0)
    hdr = dict(major=major, minor=minor, devgen=gen, rows=rows, cols=cols, memtile_rows=mrows, nops=nops, size=size)
    out, p = [], 16
    for _ in range(nops):
        op = data[p]
        if op == 0:
            off, val = struct.unpack_from("<QI", data, p + 8)
            n, f = 24, dict(reg=off, value=val)
        elif op == 1:
            off, n = struct.unpack_from("<II", data, p + 8)
            f = dict(reg=off, words=(n - 16) // 4)
        elif op in (3, 4, 7):
            off, val, mask = struct.unpack_from("<QII", data, p + 8)
            n, f = 28, dict(reg=off, value=val, mask=mask)
        elif op in (5, 6):
            n, f = 4, dict(level=data[p + 1])
        elif op == 8:
            pid, psz, paddr = struct.unpack_from("<HIQ", data, p + 2)
            n, f = 16, dict(pdi_id=pid, pdi_size=psz, pdi_address=paddr)
        elif op >= 0x80:
            n = struct.unpack_from("<I", data, p + 4)[0]
            f = {}
        else:
            raise ValueError(f"opcode {op} at byte {p}: layout not modelled")
        out.append((p, n, op, f))
        p += n
    if p != size or p != len(data):
        raise ValueError(f"ops end at {p}, header size {size}, file {len(data)}")
    return hdr, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", type=Path)
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--strip-loadpdi", type=Path)
    a = ap.parse_args()
    data = a.stream.read_bytes()
    hdr, lst = ops(data)
    print(f"header {hdr}")
    summary = Counter()
    for p, n, op, f in lst:
        name = NAMES.get(op, hex(op))
        if "reg" in f:
            col, row, t = where(f["reg"])
            summary[(name, row, t >> 12 << 12)] += 1
            if a.ops:
                extra = f" value 0x{f['value']:08x}" if "value" in f else f" {f['words']} words"
                print(f"  @{p:6d} {name:13s} col {col} row {row} +0x{t:05x}{extra}")
        else:
            summary[(name, None, None)] += 1
            if a.ops:
                print(f"  @{p:6d} {name:13s} {f}")
    for (name, row, t), c in sorted(summary.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2] or 0)):
        print(f"{name:13s} {'' if row is None else f'row {row} +0x{t:05x}'}: {c}")
    if a.strip_loadpdi:
        keep = [(p, n) for p, n, op, _ in lst if op != 8]
        body = b"".join(data[p:p + n] for p, n in keep)
        head = bytearray(data[:16])
        struct.pack_into("<II", head, 8, len(keep), 16 + len(body))
        a.strip_loadpdi.write_bytes(bytes(head) + body)
        print(f"wrote {a.strip_loadpdi}: {len(lst) - len(keep)} LOADPDI ops removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
