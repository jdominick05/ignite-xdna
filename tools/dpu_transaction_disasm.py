#!/usr/bin/env python3
"""DPU Microcode & Transaction Disassembly Engine for AMD XDNA1.

Directly disassembles compiled .xmodel artifacts produced by the VitisAI EP
into low-level DPU micro-instruction packets, revealing:
1. Low-level DPU opcode taxonomy (Op 3 = Conv/Dense, Op 6 = Depthwise Conv, etc.).
2. Packet framing (48 bytes / 12 words per microcode instruction).
3. Memory bank address offsets, activation dimensions, and hardware shift scaling.

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

    # Print first 20 instructions disassembly listing
    print("\n" + "=" * 80)
    print("DPU Instruction Disassembly Listing (First 20 Packets):")
    print(f"{'Idx':>4} | {'Tag':>4} | {'Opcode / Name':<20} | {'Input BO':>10} | {'Output BO':>10} | {'Spatial/Dim':>12} | {'Scale Reg':>10}")
    print("-" * 80)
    for i, (offset, tag, words) in enumerate(all_packets[:20]):
        op = words[1]
        desc = OPCODE_MAP.get(op, f"OP_{op}")
        in_bo = f"0x{words[2]:08X}"
        out_bo = f"0x{words[7]:08X}"
        spatial = f"0x{words[11]:08X}"
        scale = f"0x{words[9]:08X}"
        print(f"{i:04d} | 0x{tag:02X} | {desc:<20} | {in_bo:>10} | {out_bo:>10} | {spatial:>12} | {scale:>10}")

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
