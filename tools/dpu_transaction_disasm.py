#!/usr/bin/env python3
"""Heuristic byte scan of the mc_code fields in a compiled .xmodel (AMD XDNA1).

NOT a disassembler, despite the filename. Read this before citing its output.

What it does: finds the mc_code bytefield by a literal ASCII b"mc_code" search, then
walks the archive one byte at a time looking for any 32-bit word whose high byte is
0x0B, and consumes 48 bytes from each hit. Alignment is never validated. OPCODE_MAP
below is a six-entry hand-written guess; no AIE-ML or DPU ISA document backs it, and
none is cited anywhere in this repo.

What its output supports: that the archive contains a dense ~48-byte-strided record
stream, and that the assumed opcode position is strongly bimodal (on FastDepth, 49.38%
"3" and 37.35% "6"), which is CONSISTENT WITH a mostly pointwise-and-depthwise graph.

What it does not support: any semantic decode. On FastDepth 13.27% of the reported
opcodes are ASCII metadata strings read as instruction words -- 0x74746F62 is "bott",
0x6E617274 is "tran", and 0x6F630A0A is "oc" followed by two newline bytes, which no
instruction opcode contains. So an unquantified share of the "packets" are not
instructions and the record boundary is not established. Every packet in the listing
also carries the identical Inst Ptr, the signature of a misaligned constant-offset read.

For real AIE2 core machine code, use tools/aie_disasm.py (Peano's own llvm-objdump).

Usage:
  python tools/dpu_transaction_disasm.py --cache fastdepthcachekey
  python tools/dpu_transaction_disasm.py --cache bisenetv2cachekey
"""

import argparse
import glob
import os
import struct
import sys
from pathlib import Path


OPCODE_MAP = {
    1: "POOL2D / MAXPOOL",
    2: "AVGPOOL / ELEMWISE_ADD",
    3: "CONV2D / 1x1_DENSE",
    6: "DWCONV2D / DEPTHWISE",
    7: "RESIZE / NEAREST_UPSAMPLE",
    8: "ELEMWISE_MUL / HARDSIGMOID",
}


def parse_packets(data: bytes, start_pos: int, max_packets: int = 2000):
    packets = []
    p = start_pos
    while p < len(data) - 48 and len(packets) < max_packets:
        hdr = struct.unpack("<I", data[p:p+4])[0]
        len_w = (hdr >> 24) & 0xFF
        tag = hdr & 0xFF
        if len_w == 0x0B:
            words = struct.unpack("<12I", data[p:p+48])
            packets.append((p, tag, words))
            p += 48
        else:
            p += 1
    return packets


def disassemble_xmodel(xmodel_path: Path):
    print("=" * 80)
    print(f"XDNA1 DPU Microcode Disassembler: {xmodel_path.name}")
    print("=" * 80)

    data = xmodel_path.read_bytes()
    print(f"Total Model Archive Size: {len(data):,} bytes")

    # Find all mc_code sections
    mc_indices = []
    pos = 0
    while True:
        idx = data.find(b"mc_code", pos)
        if idx == -1:
            break
        mc_indices.append(idx)
        pos = idx + 7

    print(f"Found {len(mc_indices)} DPU microcode segment(s) at offsets: {mc_indices}")

    all_packets = []
    for seg_idx, mc_idx in enumerate(mc_indices):
        print(f"\n--- Disassembling Microcode Segment #{seg_idx + 1} (Offset 0x{mc_idx:08X}) ---")
        # Search for first packet header (starts with len_w == 0x0B within 128 bytes of mc_code)
        p_start = mc_idx
        while p_start < mc_idx + 256 and p_start < len(data) - 4:
            hdr = struct.unpack("<I", data[p_start:p_start+4])[0]
            if ((hdr >> 24) & 0xFF) == 0x0B:
                break
            p_start += 1

        packets = parse_packets(data, p_start)
        print(f"Segment #{seg_idx + 1}: Disassembled {len(packets)} instruction packets")
        all_packets.extend(packets)

    if not all_packets:
        print("No valid DPU instruction packets found in xmodel.")
        return

    # Opcode histogram
    op_counts = {}
    for _, _, words in all_packets:
        op = words[1]
        op_counts[op] = op_counts.get(op, 0) + 1

    print("\n" + "=" * 80)
    print("DPU Microcode Opcode Distribution:")
    print(f"{'Opcode':>8} | {'Description':<28} | {'Count':>7} | {'Percentage':>10}")
    print("-" * 80)
    total_insts = len(all_packets)
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        desc = OPCODE_MAP.get(op, f"SPECIAL_OP_0x{op:X}" if op > 10 else f"OP_{op}")
        pct = (cnt / total_insts) * 100.0
        print(f"{op:>8} | {desc:<28} | {cnt:>7} | {pct:>9.2f}%")
    print("-" * 80)
    print(f"{'TOTAL':>8} | {'':<28} | {total_insts:>7} | 100.00%")

    # Print first 20 instructions disassembly listing with decoded field semantics
    print("\n" + "=" * 80)
    print("DPU Instruction Disassembly Listing (First 20 Packets):")
    print(f"{'Idx':>4} | {'Step':>4} | {'Seq':>3} | {'Col':>4} | {'Opcode / Name':<20} | {'L1 Buffer':>10} | {'Dim Reg':>10} | {'Inst Ptr':>10}")
    print("-" * 80)
    for i, (offset, tag, words) in enumerate(all_packets[:20]):
        op = words[1]
        desc = OPCODE_MAP.get(op, f"OP_{op}")
        step = (words[0] >> 16) & 0xFF
        seq = words[0] & 0xFF
        col_id = words[11] & 0x0F
        col_name = f"Col{col_id - 12}" if 12 <= col_id <= 15 else f"0x{col_id:X}"
        l1_offset = f"0x{words[2]:06X}"
        dim_reg = f"0x{words[7]:08X}"
        inst_ptr = f"0x{words[10]:08X}"
        print(f"{i:04d} | {step:>4d} | {seq:>3d} | {col_name:>4} | {desc:<20} | {l1_offset:>10} | {dim_reg:>10} | {inst_ptr:>10}")

    print("\nDISASSEMBLY COMPLETE.")


def main():
    parser = argparse.ArgumentParser(description="Disassemble DPU microcode from compiled xmodel")
    parser.add_argument("--cache", type=str, default="fastdepthcachekey", help="Cache directory containing .xmodel")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    xmodels = list(cache_dir.glob("*.xmodel"))
    if not xmodels:
        print(f"Error: No .xmodel found in {cache_dir}")
        sys.exit(1)

    for xm in xmodels:
        disassemble_xmodel(xm)


if __name__ == "__main__":
    main()
