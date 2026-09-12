#!/usr/bin/env python3
"""
tools/disasm_txn.py - XDNA1 NPU Instruction Transaction Binary Disassembler & Auditor

Disassembles AMD XDNA1 instruction stream transaction binaries (.bin), decoding:
  - Header: version, platform flags, num_ops, total_size_bytes
  - Opcode 0: WRITE (24 bytes)
  - Opcode 1: BLOCKWRITE (variable size with 32-bit payload words)
  - Opcode 3: MASKWRITE (28 bytes)
  - Opcode 0x80: TCT (16 bytes wait token)
  - Opcode 0x81: DDR_PATCH (48 bytes relocation token)

Provides disassemble_transaction() for pipeline tooling and CLI audit capabilities.
"""

import argparse
import os
import struct
import sys
from typing import Any, Dict, List


def disassemble_transaction(data: bytes) -> List[Dict[str, Any]]:
    """
    Parse an XDNA1 transaction binary into a list of structured opcode dictionaries.
    """
    if len(data) < 16:
        raise ValueError(f"Transaction data too small: {len(data)} bytes (expected >= 16)")

    major, minor, num_ops, total_size = struct.unpack('<4I', data[:16])
    p = 16
    ops = []

    while p < len(data):
        op_code = struct.unpack('<I', data[p:p+4])[0]
        col_row = struct.unpack('<I', data[p+4:p+8])[0] if p + 8 <= len(data) else 0
        col = col_row & 0xFF
        row = (col_row >> 8) & 0xFF

        if op_code == 0:  # WRITE (24 bytes)
            o, cr, addr, vl, vh, sz = struct.unpack('<6I', data[p:p+24])
            ops.append({
                'op': 'WRITE',
                'opcode': 0,
                'col': col,
                'row': row,
                'col_row': cr,
                'addr': addr,
                'val': vh,
                'val_low': vl,
                'size': sz if sz > 0 else 24,
                'offset': p
            })
            p += (sz if sz > 0 else 24)

        elif op_code == 3:  # MASKWRITE (28 bytes)
            o, cr, addr, pad, mask, val, sz = struct.unpack('<7I', data[p:p+28])
            ops.append({
                'op': 'MASKWRITE',
                'opcode': 3,
                'col': col,
                'row': row,
                'col_row': cr,
                'addr': addr,
                'val': val,
                'mask': mask,
                'size': sz if sz > 0 else 28,
                'offset': p
            })
            p += (sz if sz > 0 else 28)

        elif op_code == 1:  # BLOCKWRITE
            o, cr, addr, sz = struct.unpack('<4I', data[p:p+16])
            num_words = (sz - 16) // 4
            words = struct.unpack(f'<{num_words}I', data[p+16:p+sz])
            ops.append({
                'op': 'BLOCKWRITE',
                'opcode': 1,
                'col': col,
                'row': row,
                'col_row': cr,
                'addr': addr,
                'size': sz,
                'words': words,
                'offset': p
            })
            p += sz

        elif op_code == 0x81:  # DDR_PATCH (48 bytes)
            w = struct.unpack('<12I', data[p:p+48])
            ops.append({
                'op': 'DDR_PATCH',
                'opcode': 0x81,
                'size': 48,
                'addr': w[6],
                'arg_idx': w[8],
                'offset': p
            })
            p += 48

        elif op_code == 0x80:  # TCT (16 bytes)
            ops.append({
                'op': 'TCT',
                'opcode': 0x80,
                'size': 16,
                'offset': p
            })
            p += 16

        else:
            raise ValueError(f"Unknown opcode 0x{op_code:X} at offset {p} (0x{p:X})")

    return ops


def audit_transaction(bin_path: str):
    """
    Perform a structural audit of the transaction binary.
    """
    if not os.path.exists(bin_path):
        print(f"Error: file not found: {bin_path}")
        sys.exit(1)

    with open(bin_path, 'rb') as f:
        data = f.read()

    major, minor, num_ops, total_size = struct.unpack('<4I', data[:16])
    ops = disassemble_transaction(data)

    print("=" * 80)
    print(f"TRANSACTION BINARY AUDIT: {bin_path}")
    print(f"Header: num_ops={num_ops}, total_size={total_size} bytes (Actual={len(data)} B)")
    print(f"Decoded Operations: {len(ops)}")
    print("=" * 80)

    # 1. MMIO Weight & Bias Injections
    blockwrites = [o for o in ops if o['op'] == 'BLOCKWRITE']
    print(f"\n[1] BLOCKWRITE INJECTIONS ({len(blockwrites)} operations):")
    for bw in blockwrites:
        target = f"Tile({bw['col']},{bw['row']})"
        addr_hex = f"0x{bw['addr']:08X}"
        payload_bytes = len(bw['words']) * 4
        print(f"  - {target:<12} Addr={addr_hex} Size={bw['size']:>5} B (Payload={payload_bytes:>5} B)")

    # 2. Core Control / Un-reset Registers (0x32000)
    core_ctrls = [o for o in ops if (o.get('addr', 0) & 0xFFFFF) == 0x32000]
    print(f"\n[2] CORE CONTROL REGISTERS (0x32000) ({len(core_ctrls)} writes):")
    for cc in core_ctrls:
        row = (cc['addr'] >> 20) & 0xF
        col = (cc['addr'] >> 25) & 0x1F
        print(f"  - Tile({col},{row}) Addr=0x{cc['addr']:08X} Val={cc.get('val', 0)} (Op={cc['op']})")

    # 3. MemTile Lock Initializations
    lock_writes = [o for o in ops if (o.get('addr', 0) & 0xFFFFF) in (0x1C0020, 0x1C0024)]
    print(f"\n[3] MEMTILE LOCK INJECTIONS ({len(lock_writes)} writes):")
    for lw in lock_writes:
        row = (lw['addr'] >> 20) & 0xF
        col = (lw['addr'] >> 25) & 0x1F
        print(f"  - Tile({col},{row}) Addr=0x{lw['addr']:08X} Val={lw.get('val', 0)}")

    # 4. DDR Patches
    ddr_patches = [o for o in ops if o['op'] == 'DDR_PATCH']
    print(f"\n[4] DDR RELOCATION PATCHES ({len(ddr_patches)} patches):")
    for dp in ddr_patches:
        print(f"  - Target Addr=0x{dp['addr']:08X} -> Kernel Arg #{dp['arg_idx']}")

    # 5. Tail Token
    tct = [o for o in ops if o['op'] == 'TCT']
    print(f"\n[5] TAIL EXECUTION TOKEN: {'PRESENT (TCT opcode 0x80)' if tct else 'MISSING'}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Disassemble and audit XDNA1 transaction binaries.")
    parser.add_argument("binary", type=str, help="Path to transaction binary (.bin)")
    args = parser.parse_args()
    audit_transaction(args.binary)
