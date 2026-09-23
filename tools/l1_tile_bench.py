# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compiler vs hand schedule for one GEMM tile held in L1, on one compute tile.

Hello XDNA! (tnzr.org/xdna/xdna1_kernel.html) measured a hand-scheduled bf16 32x32x32 tile at
398 GFLOPS (reproduced here at 397.5: results/aie/tnzr_bf16_32x32x32_repro_desktop2_20260923T0518Z.log).
This repo's own bf16 and int8 numbers all include data movement, so none of them says how much of
the gap is *schedule* and how much is *movement*. This tool removes movement: the kernel runs
against operands already in L1, called back to back from a core loop, with no DMA in the timed
region, and the same harness times every kernel in one sitting.

Kernels (`--kernels`, comma-separated):
    empty         the control, kernels/l1_tile_bench/empty_call.s: `ret lr` and its five delay slots.
                  Its cycles per call are the harness's own cost, reported so every other kernel
                  can be read with and without it ("over empty")
    tnzr_bf16     the reference's hand-written .s (from `tools/tnzr_repro.py fetch`, in scratch/;
                  their repository has no licence, so it is never copied into the tracked tree).
                  A fixed 32x32x32 kernel
    mm_bf16_f32   mlir-aie's aie_kernels/aie2/mm.cc, bf16 in / f32 out, aie::mmul<4,8,4>, compiled
                  in place from the mlir-aie checkout with upstream's Peano flags at DIM_M/K/N
    mm_i8_i32     the same file, int8 in / int32 out, aie::mmul<4,8,8>
    chain_<bf16|i8>_d<d>   probe P2, generated: 8 dependent MACs on one accumulator, d bundles
                  apart; reports how many landed (see CHAIN below)

Shape: `--m/--k/--n` (default 32 each). mm.cc re-loads and re-stores its 16 (bf16) or 8 (int8) C
accumulators once per output block whatever K is, so at K=32 its 4-iteration inner loop is short
beside that fixed cost; a K sweep separates the two.

Two harness modes, both written here (not taken from the reference):
    copy    (default) three buffers at fixed addresses (`--a-addr/--b-addr/--c-addr`, default the
            reference's: A 0x400 = bank 0 after the stack, B 0x4000 = bank 1, C 0x8000 = bank 2),
            so bank placement is identical for every kernel. One ObjectFifo brings a header word
            (the call count), A, B and an initial C0; the core copies them in, calls the kernel
            `calls` times, and copies C out. Fits up to about 40 KB of operands.
    direct  the kernel runs on the FIFO buffers themselves: A and B on their own channels (the
            call count rides in 16 words after B), C in the output FIFO, zeroed by the core. Half
            the memory of copy mode, so a 64x64x64 bf16 tile (32 KB) fits; placement is aiecc's
            default allocator, and the log reports which bank each buffer landed in.
The build parses the core's linker script and refuses on an overlap (and, in copy mode, if a
buffer is not where it was asked to be).

Because the call count is read at run time, one xclbin serves both
    correctness  calls = 1 and 3, output compared with C0 + calls * A@B (C0 = 0 in direct mode)
                 in mm.cc's tile layout; small integers, so bf16 and int8 are both exact, and
    timing       calls = 0 / 250k / 500k / 1M by default, time_ms = a + b * calls. The slope b is
                 the kernel alone (copies, DMA and dispatch land in the intercept), and
                 b * 1.80 GHz is cycles per call. The reference's own figure (FLOP / wall time at
                 the largest call count) is printed beside it.
The reference kernel's operand layout is not documented, so its output is compared against the
candidate layouts tried below, and a miss is reported as such rather than as an error.

The harness's calling loop is not free, and it is not the reference's. Their core loop has a
constant trip count, which Peano unrolls 4x with the call's delay slots filled (~9 cycles per
call). This one reads its trip count at run time, so it cannot unroll: `--loop index` (64-bit
counter) costs ~25 cycles per call, `--loop i32` (default) ~19. The empty control measures it.

    python tools/l1_tile_bench.py build --kernels empty,tnzr_bf16,mm_bf16_f32,mm_i8_i32
    python tools/l1_tile_bench.py run --kernels empty,tnzr_bf16,mm_bf16_f32,mm_i8_i32 --out results/aie/<name>.log
    python tools/l1_tile_bench.py build --mode direct --m 64 --k 64 --n 64 --kernels empty,mm_bf16_f32

Run through `bash scripts/research-iron.sh tools/l1_tile_bench.py ...` (aiecc needs the ironenv's
xclbinutil on PATH). Before `run`, `xrt-smi examine -r aie-partitions` must show no hardware
contexts; the tool checks that itself and refuses otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

IRON = Path.home() / "mlir-aie" / "ironenv"
PEANO = IRON / "Lib" / "site-packages" / "llvm-aie"
AIE_INCLUDE = IRON / "Lib" / "site-packages" / "mlir_aie" / "include"
AIECC = IRON / "Scripts" / "aiecc.exe"
MLIR_AIE_SRC = Path.home() / "mlir-aie"
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
CLOCK_GHZ = 1.80  # MEASURED, docs/SILICON.md
HDR_WORDS = 16
BANK = 16 * 1024

# Operand geometry per data type: the aie::mmul shape mm.cc uses, the input element size,
# and the per-core peak in MACs per cycle (docs/SILICON.md: 128 bf16, 256 int8).
KINDS = {
    "bf16": dict(r=4, s=8, t=4, in_bytes=2, mac_per_cycle=128, unit="GFLOPS"),
    "i8": dict(r=4, s=8, t=8, in_bytes=1, mac_per_cycle=256, unit="GOPS"),
}

MM_FLAGS = ["-O2", "-std=c++20", "--target=aie2-none-unknown-elf", "-DNDEBUG",
            "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body",
            "-D__AIE_API_AIE_ADF_HPP__", "-DVECTORIZED_ONLY"]

KERNELS = {
    "empty": dict(fn="empty_call", kind="bf16", src="asm", path=ROOT / "kernels" / "l1_tile_bench" / "empty_call.s"),
    "tnzr_bf16": dict(fn="tensor_kernel_32x32x32_bf16_bf16_fp32", kind="bf16", src="tnzr"),
    "mm_bf16_f32": dict(fn="matmul_bf16_f32", kind="bf16", src="mm", define="bf16_f32_ONLY"),
    "mm_i8_i32": dict(fn="matmul_i8_i32", kind="i8", src="mm", define="i8_i32_ONLY"),
}

# Probe P2, the VMAC forwarding distance: `chain_<bf16|i8>_d<d>` is generated, not stored. It loads
# the first mmul block of A, B and C, issues CHAIN_MACS dependent MACs on ONE accumulator with d-1
# nop bundles between consecutive MACs, waits well past the MAC latency, stores C back, and
# returns. The core has no interlocks (tnzr isa.html; Peano models none), so a MAC issued before
# its predecessor's result can be read uses a stale accumulator and that predecessor's product is
# lost: the stored block is C0 + m * (A_blk @ B_blk), and m, read back exactly, counts the MACs that
# landed. Peano spaces dependent MACs 4 bundles apart for bf16 and 2 for int8
# (results/aie/peano_aie2_machine_model_a36c62b9.log section 4), so it predicts m = CHAIN_MACS from
# those distances up and fewer below them. If the hardware stalled instead, m would be CHAIN_MACS at every
# d and the cycles per call would exceed the static bundle count; the run reports both.
CHAIN_MACS = 8
CHAIN = {
    # kind: (A loads, B loads, C loads, config, mac, C stores)
    "bf16": (["vlda wl0, [p0, #0]", "vlda wh0, [p0, #32]"],
             ["vldb wl2, [p1, #0]", "vldb wh2, [p1, #32]"],
             ["vlda amhl0, [p2, #0]", "vlda amhh0, [p2, #32]"],
             "mova r0, #28", "vmac.f bmh0, bmh0, x0, x2, r0",
             ["vst amhl0, [p2, #0]", "vst amhh0, [p2, #32]"]),
    "i8": (["vlda wl0, [p0, #0]"],
           ["vldb wl2, [p1, #0]", "vldb wh2, [p1, #32]"],
           ["vlda amll0, [p2, #0]", "vlda amlh0, [p2, #32]", "vlda amhl0, [p2, #64]", "vlda amhh0, [p2, #96]"],
           "mova r0, #776", "vmac cm0, cm0, x0, x2, r0",
           ["vst amll0, [p2, #0]", "vst amlh0, [p2, #32]", "vst amhl0, [p2, #64]", "vst amhh0, [p2, #96]"]),
}


def chain_asm(kind: str, d: int) -> str:
    a, b, c, cfg, mac, st = CHAIN[kind]
    body = [*a, *b, *c, cfg] + ["nop"] * 8  # every load 7 cycles; 8 spare bundles before the first MAC
    for i in range(CHAIN_MACS):
        body.append(mac)
        if i < CHAIN_MACS - 1:
            body += ["nop"] * (d - 1)
    body += ["nop"] * 8 + st + ["ret lr"] + ["nop"] * 5  # past the 5-6 cycle MAC latency; 5 delay slots
    lines = ["// Generated by tools/l1_tile_bench.py (probe P2); not a tracked source.",
             "\t.text", "\t.globl\tvmac_chain", "\t.p2align\t4", "\t.type\tvmac_chain,@function", "vmac_chain:"]
    lines += [f"\t{op}" for op in body]
    lines += [".Lfunc_end0:", "\t.size\tvmac_chain, .Lfunc_end0-vmac_chain", ""]
    return "\n".join(lines)


def kernel_spec(name: str) -> dict:
    if name in KERNELS:
        return KERNELS[name]
    m = re.fullmatch(r"chain_(bf16|i8)_d(\d+)", name)
    if m:
        return dict(fn="vmac_chain", kind=m.group(1), src="gen", d=int(m.group(2)))
    raise SystemExit(f"unknown kernel {name}")


# --------------------------------------------------------------------------- harness text

HARNESS_COPY = """\
module {{
  aie.device(npu1) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %bufA = aie.buffer(%t02) {{address = {a_addr} : i32, sym_name = "tileA"}} : memref<{a_w}xi32>
    %bufB = aie.buffer(%t02) {{address = {b_addr} : i32, sym_name = "tileB"}} : memref<{b_w}xi32>
    %bufC = aie.buffer(%t02) {{address = {c_addr} : i32, sym_name = "tileC"}} : memref<{c_w}xi32>
    func.func private @{fn}{sig}
    aie.objectfifo @inF(%t00, {{%t02}}, 1 : i32) : !aie.objectfifo<memref<{in_w}xi32>>
    aie.objectfifo @outF(%t02, {{%t00}}, 1 : i32) : !aie.objectfifo<memref<{c_w}xi32>>
    %core = aie.core(%t02) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 4294967295 : index
      %nA = arith.constant {a_w} : index
      %nB = arith.constant {b_w} : index
      %nC = arith.constant {c_w} : index
      %oA = arith.constant {o_a} : index
      %oB = arith.constant {o_b} : index
      %oC = arith.constant {o_c} : index
      scf.for %it = %c0 to %cmax step %c1 {{
        %sin = aie.objectfifo.acquire @inF(Consume, 1) : !aie.objectfifosubview<memref<{in_w}xi32>>
        %in = aie.objectfifo.subview.access %sin[0] : !aie.objectfifosubview<memref<{in_w}xi32>> -> memref<{in_w}xi32>
        %sout = aie.objectfifo.acquire @outF(Produce, 1) : !aie.objectfifosubview<memref<{c_w}xi32>>
        %out = aie.objectfifo.subview.access %sout[0] : !aie.objectfifosubview<memref<{c_w}xi32>> -> memref<{c_w}xi32>
        scf.for %i = %c0 to %nA step %c1 {{
          %j = arith.addi %i, %oA : index
          %v = memref.load %in[%j] : memref<{in_w}xi32>
          memref.store %v, %bufA[%i] : memref<{a_w}xi32>
        }}
        scf.for %i = %c0 to %nB step %c1 {{
          %j = arith.addi %i, %oB : index
          %v = memref.load %in[%j] : memref<{in_w}xi32>
          memref.store %v, %bufB[%i] : memref<{b_w}xi32>
        }}
        scf.for %i = %c0 to %nC step %c1 {{
          %j = arith.addi %i, %oC : index
          %v = memref.load %in[%j] : memref<{in_w}xi32>
          memref.store %v, %bufC[%i] : memref<{c_w}xi32>
        }}
        %n32 = memref.load %in[%c0] : memref<{in_w}xi32>
{call_loop}
        scf.for %i = %c0 to %nC step %c1 {{
          %v = memref.load %bufC[%i] : memref<{c_w}xi32>
          memref.store %v, %out[%i] : memref<{c_w}xi32>
        }}
        aie.objectfifo.release @outF(Produce, 1)
        aie.objectfifo.release @inF(Consume, 1)
      }}
      aie.end
    }} {{link_with = "{obj}"}}
    aie.runtime_sequence(%a: memref<{in_w}xi32>, %c: memref<{c_w}xi32>) {{
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, {in_rows}, 16][0, 0, 16, 1]) {{id = 0 : i64, metadata = @inF}} : memref<{in_w}xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, {c_rows}, 16][0, 0, 16, 1]) {{id = 1 : i64, metadata = @outF, issue_token = true}} : memref<{c_w}xi32>
      aiex.npu.dma_wait {{symbol = @outF}}
    }}
  }}
}}
"""

HARNESS_DIRECT = """\
module {{
  aie.device(npu1) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    func.func private @{fn}{sig}
    aie.objectfifo @inA(%t00, {{%t02}}, 1 : i32) : !aie.objectfifo<memref<{a_w}xi32>>
    aie.objectfifo @inB(%t00, {{%t02}}, 1 : i32) : !aie.objectfifo<memref<{bh_w}xi32>>
    aie.objectfifo @outC(%t02, {{%t00}}, 1 : i32) : !aie.objectfifo<memref<{c_w}xi32>>
    %core = aie.core(%t02) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 4294967295 : index
      %nC = arith.constant {c_w} : index
      %oN = arith.constant {b_w} : index
      %zero = arith.constant 0 : i32
      scf.for %it = %c0 to %cmax step %c1 {{
        %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<{a_w}xi32>>
        %bufA = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<{a_w}xi32>> -> memref<{a_w}xi32>
        %sb = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<{bh_w}xi32>>
        %bufB = aie.objectfifo.subview.access %sb[0] : !aie.objectfifosubview<memref<{bh_w}xi32>> -> memref<{bh_w}xi32>
        %sc = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<{c_w}xi32>>
        %bufC = aie.objectfifo.subview.access %sc[0] : !aie.objectfifosubview<memref<{c_w}xi32>> -> memref<{c_w}xi32>
        scf.for %i = %c0 to %nC step %c1 {{
          memref.store %zero, %bufC[%i] : memref<{c_w}xi32>
        }}
        %n32 = memref.load %bufB[%oN] : memref<{bh_w}xi32>
{call_loop}
        aie.objectfifo.release @outC(Produce, 1)
        aie.objectfifo.release @inB(Consume, 1)
        aie.objectfifo.release @inA(Consume, 1)
      }}
      aie.end
    }} {{link_with = "{obj}"}}
    aie.runtime_sequence(%a: memref<{a_w}xi32>, %b: memref<{bh_w}xi32>, %c: memref<{c_w}xi32>) {{
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, {a_rows}, 16][0, 0, 16, 1]) {{id = 0 : i64, metadata = @inA}} : memref<{a_w}xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, {bh_rows}, 16][0, 0, 16, 1]) {{id = 1 : i64, metadata = @inB}} : memref<{bh_w}xi32>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, {c_rows}, 16][0, 0, 16, 1]) {{id = 2 : i64, metadata = @outC, issue_token = true}} : memref<{c_w}xi32>
      aiex.npu.dma_wait {{symbol = @outC}}
    }}
  }}
}}
"""

# The timed loop. `index` is 64-bit on this target, so its trip test is a two-word compare with
# unfilled branch delay slots; `i32` is the cheaper default. Neither is unrolled, because the trip
# count is read at run time.
CALL_LOOP = {
    "index": """\
        %n = arith.index_cast %n32 : i32 to index
        scf.for %k = %c0 to %n step %c1 {{
          func.call @{fn}(%bufA, %bufB, %bufC) : {sig}
        }}""",
    "i32": """\
        %z32 = arith.constant 0 : i32
        %u32 = arith.constant 1 : i32
        scf.for %k = %z32 to %n32 step %u32 : i32 {{
          func.call @{fn}(%bufA, %bufB, %bufC) : {sig}
        }}""",
}


def geometry(kind: str, args) -> dict:
    """Word counts for A (MxK), B (KxN), C (MxN, 4-byte) and each FIFO element."""
    k = KINDS[kind]
    M, K, N = args.m, args.k, args.n
    a_w = M * K * k["in_bytes"] // 4
    b_w = K * N * k["in_bytes"] // 4
    c_w = M * N
    in_w = HDR_WORDS + a_w + b_w + c_w
    bh_w = b_w + HDR_WORDS
    for w in (a_w, b_w, c_w):
        assert w % 16 == 0, "operand word counts must be multiples of 16 (the DMA row)"
    b_t = bh_w if args.mode == "direct" else b_w
    sig = f"(memref<{a_w}xi32>, memref<{b_t}xi32>, memref<{c_w}xi32>) -> ()"
    return dict(a_w=a_w, b_w=b_w, c_w=c_w, in_w=in_w, bh_w=bh_w, o_a=HDR_WORDS, o_b=HDR_WORDS + a_w,
                o_c=HDR_WORDS + a_w + b_w, in_rows=in_w // 16, a_rows=a_w // 16, bh_rows=bh_w // 16,
                c_rows=c_w // 16, sig=sig)


def build_dir(name: str, args) -> Path:
    place = f"a{args.a_addr:x}_b{args.b_addr:x}_c{args.c_addr:x}" if args.mode == "copy" else "alloc"
    return (ROOT / "scratch" / "l1_tile_bench"
            / f"{name}_{args.mode}_m{args.m}k{args.k}n{args.n}_{args.loop}_{place}")


def check_shape(name: str, args) -> None:
    spec = kernel_spec(name)
    k = KINDS[spec["kind"]]
    if spec["src"] == "tnzr" and (args.m, args.k, args.n) != (32, 32, 32):
        raise SystemExit("tnzr_bf16 is a fixed 32x32x32 kernel")
    if spec["src"] == "mm" and (args.m % (4 * k["r"]) or args.k % k["s"] or args.n % (2 * k["t"] if spec["kind"] == "i8" else 4 * k["t"])):
        raise SystemExit(f"{name}: shape {args.m}x{args.k}x{args.n} violates mm.cc's static_asserts")


def tnzr_source() -> Path:
    base = ROOT / "scratch" / "tnzr-xdna"
    commit = (base / "LATEST").read_text().strip()
    p = base / commit / f"{KERNELS['tnzr_bf16']['fn']}.s"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `python tools/tnzr_repro.py fetch` first")
    return p


def run_cmd(cmd: list, cwd: Path) -> str:
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(str(c) for c in cmd)}")
    return p.stdout


def compile_kernel(spec: dict, out: Path, args) -> list[str]:
    clang = PEANO / "bin" / "clang++.exe"
    if spec["src"] == "gen":
        src = out.with_suffix(".s")
        src.write_text(chain_asm(spec["kind"], spec["d"]))
        cmd = [clang, "--target=aie2-none-unknown-elf", "-c", src, "-o", out]
    elif spec["src"] in ("tnzr", "asm"):
        src = tnzr_source() if spec["src"] == "tnzr" else spec["path"]
        cmd = [clang, "--target=aie2-none-unknown-elf", "-c", src, "-o", out]
    else:
        src = MLIR_AIE_SRC / "aie_kernels" / "aie2" / "mm.cc"
        cmd = [clang, *MM_FLAGS, f"-DDIM_M={args.m}", f"-DDIM_K={args.k}", f"-DDIM_N={args.n}",
               f"-D{spec['define']}", f"-I{AIE_INCLUDE}", "-c", src, "-o", out]
    run_cmd(cmd, cwd=out.parent)
    return [str(c) for c in cmd]


def linker_placement(prj: Path) -> dict[str, int]:
    """Symbol -> tile-local byte offset, read from the core's generated linker script."""
    text = (prj / "ldScripts_main_core_0_2.ld.script").read_text()
    out = {}
    for m in re.finditer(r"\. = (0x[0-9A-Fa-f]+);\s*\n\s*([A-Za-z_]\w*) = \.;", text):
        out[m.group(2)] = int(m.group(1), 16) - 0x70000
    return out


def static_count(obj: Path, fn: str) -> dict:
    """Bundles and vmac-issuing bundles in `fn`, plus each hardware loop's body."""
    import aie_disasm
    objdump = aie_disasm.find_objdump(None)
    secs = aie_disasm.parse(aie_disasm.disassemble(str(obj), objdump))
    bundles = []
    for s in secs:
        inside = False
        for b in s.bundles:
            if b.label and not b.label.startswith("."):
                inside = b.label == fn
            if inside:
                bundles.append(b)
    is_vmac = lambda b: any(f.split()[0].startswith(("vmac", "vmul")) for f in b.fields)  # noqa: E731
    loops = []
    for s in secs:
        for lp in s.loops:
            if any(lp.bundles[0] is b for b in bundles) or any(lp.bundles[-1] is b for b in bundles):
                loops.append(dict(name=lp.name, bundles=lp.n_bundles, vmac=sum(map(is_vmac, lp.bundles))))
    names = [b.label or "" for b in bundles]
    s_i = next((i for i, n in enumerate(names) if n.endswith("l_start")), None)
    e_i = next((i for i, n in enumerate(names) if n.endswith("l_end")), None)
    if s_i is not None and e_i is not None:
        body = bundles[s_i:e_i + 1]
        loops.append(dict(name=".l_start..l_end", bundles=len(body), vmac=sum(map(is_vmac, body))))
    return dict(bundles=len(bundles), vmac_bundles=sum(map(is_vmac, bundles)), loops=loops)


def cmd_build(args) -> int:
    for name in args.kernels.split(","):
        spec = kernel_spec(name)
        check_shape(name, args)
        g = geometry(spec["kind"], args)
        d = build_dir(name, args)
        d.mkdir(parents=True, exist_ok=True)
        obj = d / "kernel.o"
        cc = compile_kernel(spec, obj, args)
        loop = CALL_LOOP[args.loop].format(fn=spec["fn"], **g)
        tmpl = HARNESS_COPY if args.mode == "copy" else HARNESS_DIRECT
        (d / "harness.mlir").write_text(tmpl.format(fn=spec["fn"], obj=obj.name, a_addr=args.a_addr,
                                                    b_addr=args.b_addr, c_addr=args.c_addr,
                                                    call_loop=loop, **g))
        alloc = ["--alloc-scheme=basic-sequential"] if args.mode == "copy" else []
        run_cmd([AIECC, d / "harness.mlir", *alloc, f"--peano={PEANO}",
                 "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-xclbin", "--xclbin-name=final.xclbin",
                 f"--tmpdir={d / 'prj'}"], cwd=d)
        place = linker_placement(d / "prj")
        if args.mode == "copy":
            want = {"tileA": (args.a_addr, g["a_w"] * 4), "tileB": (args.b_addr, g["b_w"] * 4),
                    "tileC": (args.c_addr, g["c_w"] * 4)}
            for sym, (addr, _) in want.items():
                if place.get(sym) != addr:
                    raise SystemExit(f"{name}: {sym} placed at {place.get(sym)}, wanted {addr}")
            sizes = {**{k: v[1] for k, v in want.items()}, "inF_cons_buff_0": g["in_w"] * 4,
                     "outF_buff_0": g["c_w"] * 4}
        else:
            sizes = {"inA_cons_buff_0": g["a_w"] * 4, "inB_cons_buff_0": g["bh_w"] * 4, "outC_buff_0": g["c_w"] * 4}
        missing = [s for s in sizes if s not in place]
        if missing:
            raise SystemExit(f"{name}: linker script has no {missing}; found {sorted(place)}")
        spans = sorted((place[s], place[s] + n, s) for s, n in sizes.items())
        for (a0, a1, sa), (b0, b1, sb) in zip(spans, spans[1:]):
            if b0 < a1:
                raise SystemExit(f"{name}: {sa} [{a0},{a1}) overlaps {sb} [{b0},{b1})")
        sc = static_count(obj, spec["fn"])
        meta = dict(kernel=name, fn=spec["fn"], kind=spec["kind"], compile=cc, mode=args.mode,
                    placement={s: dict(offset=o, end=e, bank=o // BANK, bank_end=(e - 1) // BANK)
                               for o, e, s in spans}, static=sc)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"built {name} ({args.mode} {args.m}x{args.k}x{args.n}): placement "
              + ", ".join(f"{s} {v['offset']:#x} b{v['bank']}-{v['bank_end']}" for s, v in meta["placement"].items())
              + f"; static {sc}")
    return 0


# --------------------------------------------------------------------------- data + oracle

def tile_rows(x: np.ndarray, r: int, c: int) -> np.ndarray:
    """(R, C) matrix -> [R/r][C/c][r][c] blocks, flattened: the layout mm.cc indexes."""
    R, C = x.shape
    return x.reshape(R // r, r, C // c, c).transpose(0, 2, 1, 3).reshape(-1)


def untile(v: np.ndarray, R: int, C: int, r: int, c: int) -> np.ndarray:
    return v.reshape(R // r, C // c, r, c).transpose(0, 2, 1, 3).reshape(R, C)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def make_case(kind: str, args):
    """A (MxK), B (KxN), C0 (MxN) of small integers, so every result is exact in fp32/int32."""
    rng = np.random.default_rng(args.seed)
    M, K, N = args.m, args.k, args.n
    if kind == "bf16":
        A = rng.integers(-4, 5, (M, K)).astype(np.float32)
        B = rng.integers(-4, 5, (K, N)).astype(np.float32)
        C0 = rng.integers(-8, 9, (M, N)).astype(np.float32)
    else:
        A = rng.integers(-128, 128, (M, K)).astype(np.int64)
        B = rng.integers(-128, 128, (K, N)).astype(np.int64)
        C0 = rng.integers(-1000, 1001, (M, N)).astype(np.int64)
    if args.mode == "direct":
        C0 = np.zeros_like(C0)
    return A, B, C0


def pack(kind: str, A, B, C0, layout: str) -> tuple[bytes, bytes, bytes]:
    k = KINDS[kind]
    r, s, t = k["r"], k["s"], k["t"]
    if layout == "tiled":
        a, b, c = tile_rows(A, r, s), tile_rows(B, s, t), tile_rows(C0, r, t)
    else:
        a, b, c = A.reshape(-1), B.reshape(-1), C0.reshape(-1)
    if kind == "bf16":
        return bf16_bits(a).tobytes(), bf16_bits(b).tobytes(), c.astype(np.float32).tobytes()
    return a.astype(np.int8).tobytes(), b.astype(np.int8).tobytes(), c.astype(np.int32).tobytes()


def unpack_c(kind: str, raw: bytes, layout: str, M: int, N: int) -> np.ndarray:
    k = KINDS[kind]
    v = np.frombuffer(raw, dtype=np.float32 if kind == "bf16" else np.int32).astype(np.float64)
    return untile(v, M, N, k["r"], k["t"]) if layout == "tiled" else v.reshape(M, N)


# --------------------------------------------------------------------------- run

def xrt_versions() -> list[str]:
    out = subprocess.run([str(XRT_SMI), "examine"], capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.splitlines()
            if re.search(r"^\s*(Version|NPU Driver Version|NPU Firmware Version)\s*:", ln)]


def partitions() -> str:
    out = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True).stdout
    return "idle" if "No hardware contexts running" in out else "BUSY:\n" + out


def fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    b, a = np.polyfit(xs, ys, 1)
    res = ys - (a + b * xs)
    r2 = 1 - (res ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return a, b, r2


def bench_one(name: str, args, say) -> dict:
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    spec = kernel_spec(name)
    kind = spec["kind"]
    k = KINDS[kind]
    check_shape(name, args)
    g = geometry(kind, args)
    d = build_dir(name, args)
    meta = json.loads((d / "meta.json").read_text())
    say(f"\n## {name}  (`{spec['fn']}`, {kind}, M{args.m} x K{args.k} x N{args.n}, {args.mode} mode)")
    say(f"compile: {' '.join(Path(c).name if os.path.isabs(c) else c for c in meta['compile'])}")
    say(f"placement (tile-local offset, bank): " + ", ".join(
        f"{s} {v['offset']:#06x}-{v['end']:#06x} b{v['bank']}" + (f"-{v['bank_end']}" if v.get("bank_end", v["bank"]) != v["bank"] else "")
        for s, v in meta["placement"].items()))
    st = meta["static"]
    say(f"static: {st['bundles']} bundles in `{spec['fn']}`, {st['vmac_bundles']} issue a vmac; loops "
        + (", ".join(f"{lp['name']} {lp['vmac']}/{lp['bundles']}" for lp in st["loops"]) or "none"))
    macs = args.m * args.k * args.n
    ideal = macs / k["mac_per_cycle"]
    result = dict(kernel=name)
    with XrtSiliconHarness() as h:
        h.load_xclbin(str(d / "final.xclbin"), kernel_name="MLIR_AIE")
        bo_instr, n = h.create_instruction_bo(str(d / "insts.bin"))
        to_dev = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        from_dev = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        if args.mode == "copy":
            bos = [h.create_host_bo(g["in_w"] * 4, 3), h.create_host_bo(g["c_w"] * 4, 4)]
        else:
            bos = [h.create_host_bo(g["a_w"] * 4, 3), h.create_host_bo(g["bh_w"] * 4, 4),
                   h.create_host_bo(g["c_w"] * 4, 5)]
        bo_out = bos[-1]

        def load(calls: int, a: bytes, b: bytes, c: bytes) -> None:
            hdr = np.zeros(HDR_WORDS, np.int32)
            hdr[0] = calls
            if args.mode == "copy":
                parts = [hdr.tobytes() + a + b + c]
            else:
                parts = [a, b + hdr.tobytes()]
            for bo, buf in zip(bos, parts):
                bo.write(buf, 0)
                bo.sync(to_dev)

        def dispatch() -> float:
            t0 = time.perf_counter()
            run, state = h.dispatch_kernel(bo_instr, n, *bos, timeout_ms=args.timeout_ms)
            t1 = time.perf_counter()
            if "COMPLETED" not in str(state):
                raise RuntimeError(f"dispatch state {state}")
            return t1 - t0

        def read_out() -> bytes:
            bo_out.sync(from_dev)
            return bytes(bo_out.read(g["c_w"] * 4, 0))

        # Correctness: C0 + calls * A@B, exact for these small integers.
        A, B, C0 = make_case(kind, args)
        layouts = ["tiled", "rowmajor"] if spec["src"] == "tnzr" else ["tiled"]
        verdicts = []
        if spec["src"] == "gen":
            # P2: only the first mmul block is touched. Its value is C0 + m * (A_blk @ B_blk); m is
            # the number of chained MACs that landed, and the rest of C must come back unchanged.
            r, s, t = k["r"], k["s"], k["t"]
            a_t, b_t, c_t = tile_rows(A, r, s), tile_rows(B, s, t), tile_rows(C0, r, t)
            prod = (a_t[:r * s].reshape(r, s) @ b_t[:s * t].reshape(s, t)).reshape(-1).astype(np.float64)
            dtype = np.float32 if kind == "bf16" else np.int32
            ms = []
            for calls in (1, 3):
                load(calls, *pack(kind, A, B, C0, "tiled"))
                dispatch()
                out = np.frombuffer(read_out(), dtype=dtype).astype(np.float64)
                rest_ok = np.array_equal(out[r * t:], c_t[r * t:].astype(np.float64))
                nz = prod != 0
                ratio = (out[:r * t] - c_t[:r * t])[nz] / prod[nz]
                m = float(ratio[0]) if ratio.size else float("nan")
                consistent = bool(ratio.size) and np.all(ratio == m) and m == round(m)
                ms.append(m if consistent else None)
                say(f"chain d={spec['d']} calls={calls}: block = C0 + m*(A_blk@B_blk) with m = "
                    + (f"{m:g} ({m / calls:g} of {CHAIN_MACS} MACs per call)" if consistent
                       else f"NOT a single integer (ratios {np.unique(ratio)[:6]})")
                    + ("" if rest_ok else "; REST OF C CHANGED"))
            m1 = ms[0]
            result["correct"] = (f"m={m1:g}/{CHAIN_MACS}" if m1 is not None and ms[1] == 3 * m1
                                 else "inconsistent")
            layouts = []
        for layout in layouts:
            pa, pb, pc = pack(kind, A, B, C0, layout)
            for calls in (1, 3):
                load(calls, pa, pb, pc)
                dispatch()
                got = unpack_c(kind, read_out(), layout, args.m, args.n)
                acc = C0 if spec["src"] == "asm" else C0 + calls * (A @ B)  # the control leaves C alone
                ovw = A @ B
                ok_acc, ok_ovw = np.array_equal(got, acc), np.array_equal(got, ovw)
                verdicts.append((layout, calls, ok_acc, ok_ovw, float(np.abs(got - acc).max())))
        for layout, calls, ok_acc, ok_ovw, err in verdicts:
            say(f"correctness layout={layout} calls={calls}: "
                + ("EXACT (C unchanged)" if ok_acc and spec["src"] == "asm" else "EXACT (C0 + calls*A@B)" if ok_acc
                   else "EXACT (A@B, overwrites C)" if ok_ovw
                   else f"mismatch (max |err| vs accumulate {err:g})"))
        if layouts:
            exact = [lay for lay in layouts if all(v[2] for v in verdicts if v[0] == lay)]
            result["correct"] = f"exact ({exact[0]})" if exact else "NOT VERIFIED"

        # Timing on the same data.
        pa, pb, pc = pack(kind, A, B, C0, "tiled")
        top = max(args.calls)
        load(top, pa, pb, pc)
        for _ in range(args.warmups):
            dispatch()
        xs, ys, raw_top = [], [], []
        for calls in args.calls:
            load(calls, pa, pb, pc)
            ts = [dispatch() for _ in range(args.reps)]
            xs += [calls] * len(ts)
            ys += ts
            if calls == top:
                raw_top = ts
            say(f"calls={calls:>8}: mean {np.mean(ts) * 1e3:9.3f} ms  (min {min(ts) * 1e3:.3f}, max {max(ts) * 1e3:.3f}, n={len(ts)})")
        del bos, bo_out, bo_instr
    a, b, r2 = fit(xs, ys)
    cyc = b * CLOCK_GHZ * 1e9
    rate_slope = 2 * macs / b / 1e9
    rate_raw = 2 * macs * top / np.mean(raw_top) / 1e9
    peak = 2 * k["mac_per_cycle"] * CLOCK_GHZ
    say(f"fit: time = {a * 1e3:.3f} ms + {b * 1e9:.2f} ns x calls   (R^2 {r2:.6f})")
    if spec["src"] == "asm":
        say(f"per call: {cyc:.1f} cycles at {CLOCK_GHZ} GHz -- the harness's own cost per call, no MACs")
    elif spec["src"] == "gen":
        say(f"per call: {cyc:.1f} cycles at {CLOCK_GHZ} GHz against {st['bundles']} static bundles in the probe "
            f"(a stall would show as cycles over empty exceeding the bundles the probe executes)")
    else:
        say(f"per call: {cyc:.1f} cycles at {CLOCK_GHZ} GHz; ideal {ideal:.0f} ({args.m}*{args.k}*{args.n} MACs / "
            f"{k['mac_per_cycle']} per cycle) -> {100 * ideal / cyc:.1f}% issue efficiency")
        say(f"rate: {rate_slope:.1f} {k['unit']} from the slope ({100 * rate_slope / peak:.1f}% of {peak:.1f}); "
            f"{rate_raw:.1f} by the reference's method (2*MACs*{top} / mean wall at {top} calls)")
    result.update(cycles_per_call=cyc, rate_slope=rate_slope, rate_raw=rate_raw, peak=peak, intercept_ms=a * 1e3,
                  r2=r2, static=st, unit=k["unit"], ideal=ideal)
    return result


def cmd_run(args) -> int:
    L = []
    say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
    say("L1-resident GEMM tile: compiler vs hand schedule (tools/l1_tile_bench.py)")
    say(f"date: {dt.datetime.now().isoformat(timespec='seconds')}  host: {os.environ.get('COMPUTERNAME')}")
    say(f"peano: {subprocess.run([str(PEANO / 'bin' / 'clang.exe'), '--version'], capture_output=True, text=True).stdout.splitlines()[0]}")
    say(f"aiecc: mlir-aie ironenv v1.4.2; mm.cc from the mlir-aie checkout (not vendored)")
    for v in xrt_versions():
        say(f"xrt-smi: {v}")
    say(f"pmode: {args.pmode_note}")
    say(f"shape: M{args.m} x K{args.k} x N{args.n}; mode {args.mode}; loop counter {args.loop}")
    if args.mode == "copy":
        say(f"buffers: A {args.a_addr:#x}, B {args.b_addr:#x}, C {args.c_addr:#x} (fixed)")
    say(f"seed {args.seed}; calls {args.calls}, {args.reps} reps each, {args.warmups} warm-ups")
    before = partitions()
    say(f"partitions before: {before}")
    if before != "idle" and not args.force:
        say("REFUSED: device not idle")
        return 2
    results = []
    for name in args.kernels.split(","):
        try:
            results.append(bench_one(name, args, say))
        except Exception as e:  # noqa: BLE001 - report and stop; never retry into a wedged device
            say(f"!! {name}: {type(e).__name__}: {e} -- stopping")
            break
    say(f"\npartitions after close: {partitions()}")
    # "Over empty" subtracts the control's whole call, including its own `ret lr` and five delay
    # slots, which a real kernel also executes (though it may fill them with work). It is the
    # kernel's cost as if the harness were free; the plain column is what the harness measured.
    empty = next((r["cycles_per_call"] for r in results if r["kernel"] == "empty"), None)
    say(f"\nharness loop: {args.loop}; empty control: "
        + (f"{empty:.1f} cycles per call" if empty is not None else "not run"))
    say("\n| kernel | output | static vmac/bundles | cycles/call | over empty | ideal | eff. (over empty) | rate (slope) | % peak | reference method |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    probes = [r for r in results if r["kernel"].startswith("chain_")]
    if probes:
        say("\n| probe | MACs landed per call | static bundles | cycles/call | over empty |")
        say("|---|---|---|---|---|")
        for r in probes:
            net = r["cycles_per_call"] - empty if empty is not None else float("nan")
            say(f"| {r['kernel']} | {r['correct']} | {r['static']['bundles']} | {r['cycles_per_call']:.1f} | {net:.1f} |")
    for r in results:
        if r["kernel"] == "empty" or r["kernel"].startswith("chain_"):
            continue
        net = r["cycles_per_call"] - empty if empty is not None else float("nan")
        say(f"| {r['kernel']} | {r['correct']} | {r['static']['vmac_bundles']}/{r['static']['bundles']} | "
            f"{r['cycles_per_call']:.1f} | {net:.1f} | {r['ideal']:.0f} | {100 * r['ideal'] / net:.1f}% | "
            f"{r['rate_slope']:.1f} {r['unit']} | {100 * r['rate_slope'] / r['peak']:.1f}% | {r['rate_raw']:.1f} |")
    if args.out:
        text = "\n".join(L) + "\n"
        text = text.replace(str(Path.home()), "C:\\Users\\<user>")  # results/ logs carry no profile path
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run"):
        p = sub.add_parser(name)
        p.add_argument("--kernels", default=",".join(KERNELS))
        p.add_argument("--mode", choices=("copy", "direct"), default="copy")
        p.add_argument("--m", type=int, default=32)
        p.add_argument("--k", type=int, default=32)
        p.add_argument("--n", type=int, default=32)
        p.add_argument("--a-addr", type=lambda s: int(s, 0), default=0x400, help="copy mode only")
        p.add_argument("--b-addr", type=lambda s: int(s, 0), default=0x4000, help="copy mode only")
        p.add_argument("--c-addr", type=lambda s: int(s, 0), default=0x8000, help="copy mode only")
        p.add_argument("--loop", choices=sorted(CALL_LOOP), default="i32", help="type of the timed loop's counter")
    r = sub.choices["run"]
    r.add_argument("--calls", type=lambda s: [int(x) for x in s.split(",")], default=[0, 250_000, 500_000, 1_000_000])
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--warmups", type=int, default=3)
    r.add_argument("--seed", type=int, default=20260923)
    r.add_argument("--timeout-ms", type=int, default=20000)
    r.add_argument("--pmode-note", default="default (not changed by this run; docs/SILICON.md measured 1.80 GHz in default)")
    r.add_argument("--force", action="store_true", help="run even if xrt-smi reports contexts")
    r.add_argument("--out")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
