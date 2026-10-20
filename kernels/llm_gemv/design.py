"""The whole-array M=1 GEMV: one xclbin for every shape, one instruction stream per program of calls.

Data movement copies ``tools/npu_read_bw_probe.py``'s 8-channel read (4 columns x 2 shim MM2S), the layout that read
DDR at 47.62 GB/s, and puts compute cores behind it in the 4 x 4 split/join shape of ``kernels/laya_norm/probe.py``:
per column, shim MM2S channel h -> MemTile -> split to cores 2h and 2h+1; the four cores' outputs join in the MemTile
and leave through one shim S2MM. x is not a separate stream (each column has only two MM2S channels and both carry
weights): each call's first object on every core is the header-and-x object (``pack.x_slot``), read from the x
buffer before that channel's weights.

The core program is shape-free. It takes the counts from each call's header, so the xclbin is the same for every
program and only the runtime sequence changes: one call is ``(w_offset, x_offset, out_offset, N, K)`` in words of the
three runtime arguments (weights, x, out). A program of several calls waits for each call's outputs before the next
reuses the shim BDs.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kernels.llm_gemv import pack
from kernels.transformer_bf16.common import INSTALL, IRON, PEANO, command, save, sha

HERE = Path(__file__).resolve().parent
STACK = 1024
FIFO_DEPTH = 2


@dataclass(frozen=True)
class Call:
    w_offset: int      # words into the weights argument
    x_offset: int      # words into the x argument: a pack.x_slot (X_SLOT_WORDS), or with x_out a pack.tail
    out_offset: int    # words into the out argument; N/2 are written
    n: int
    k: int
    x_out: Optional[int] = None   # chained: x is the K/2 words at this offset of the out argument (an earlier
                                  # call's output), and x_offset names the call's constant tail (pack.tail)
    flags: int = 0                # pack.F_INIT, F_NORM, F_ROPE, F_SILU
    h_out: Optional[int] = None   # chained with F_INIT: the residual h, N/2 words at this offset of the out argument
    rope_end: int = 0             # with F_ROPE: output positions below this are rotated
    out_map: Optional[Tuple[Tuple[int, int, int], ...]] = None
    # out_map: (first output, outputs, out-argument word offset) pieces covering [0, N) in order; the output goes to
    # out_offset + position/2 without one. A column's S2MM takes one BD per piece it overlaps (the KV append: k and v
    # land in the cache rows of the position, q in its own region).

    def __post_init__(self):
        pack.check_call(self.n, self.k, self.flags, self.rope_end)
        if (self.h_out is not None) != (self.x_out is not None and bool(self.flags & pack.F_INIT)):
            raise ValueError("h_out is given exactly for a chained call with F_INIT")
        if self.out_map is not None:
            at = 0
            for first, count, _ in self.out_map:
                if first != at or count <= 0 or first % 2 or count % 2:
                    raise ValueError("out_map pieces must tile [0, N) in order, in whole words")
                at += count
            if at != self.n:
                raise ValueError("out_map pieces must cover N outputs")

    def header(self, cs=None):
        """The tail this call's x region must hold: cos/sin with F_ROPE, pad and header, from the same (n, k, flags)
        as the streams (zero cos/sin unless given: its size is what the design needs)."""
        if self.flags & pack.F_ROPE and cs is None:
            cs = np.zeros(pack.CS_ELEMS, np.uint16)
        return pack.tail(self.n, self.k, self.flags, self.rope_end, cs)

    def weight_words(self) -> int:
        return pack.weight_words(self.n, self.k, self.flags)

    def out_pieces(self, c: int) -> list:
        """(out-argument word offset, words) for column c's quarter of the output, in stream order."""
        lo, hi = c * self.n // pack.COLS, (c + 1) * self.n // pack.COLS
        pieces = self.out_map or ((0, self.n, self.out_offset),)
        out = []
        for first, count, dest in pieces:
            a, b = max(lo, first), min(hi, first + count)
            if a < b:
                out.append((dest + (a - first) // 2, (b - a) // 2))
        return out

    def x_words_used(self) -> int:
        return pack.X_SLOT_WORDS if self.x_out is None else self.header().size

    def lead(self) -> list:
        """(argument, word offset, words) for each piece of one first object, in stream order."""
        if self.x_out is None:
            return [("x", self.x_offset, pack.X_SLOT_WORDS)]
        pieces = [("o", self.x_out, self.k // 2)]
        if self.flags & pack.F_INIT:
            pieces.append(("o", self.h_out, self.n // 2))
        return pieces + [("x", self.x_offset, self.header().size)]


@dataclass(frozen=True)
class Attn:
    """One attention phase over CAP cached positions, on columns 0..KV_COLS-1 (the last column gets no transfer).
    F_SCORE: x_out is the q region (9 heads x 64, rope_perm order), cache the K cache, out the scores. F_PV: x_out
    is F_SCORE's out, cache the V cache, out the attention output in pack.attn_out_perm order. Offsets in words of
    the out argument, except x_offset (the call's pack.attn_tail, in the x argument)."""
    flags: int
    cap: int
    x_out: int
    x_offset: int
    cache: int
    out_offset: int

    def __post_init__(self):
        pack.attn_tail(self.flags, self.cap)

    def lead_words(self) -> int:
        return pack.attn_lead_words(self.flags, self.cap)

    def out_words_per_column(self) -> int:
        return self.cap * pack.GQA // 2 if self.flags == pack.F_SCORE else pack.GQA * pack.HEAD_DIM // 2

    def stream(self, c: int, h: int) -> Tuple[int, list, list]:
        """(offset, sizes, strides) in the out argument of channel h of column c: two cores' objects per step."""
        row, step = pack.CACHE_WORDS, pack.PV_ROWS * pack.CACHE_WORDS
        if self.flags == pack.F_SCORE:      # objects 4g+2h and 4g+2h+1: 72 contiguous positions, 32 words of head c
            return (self.cache + 2 * h * pack.SCORE_ROWS * row + c * pack.HEAD_DIM // 2,
                    [self.cap // (pack.SCORE_ROWS * pack.ROWS), 1, 2 * pack.SCORE_ROWS, pack.HEAD_DIM // 2],
                    [step, 0, row, 1])
        dims = pack.HEAD_DIM // pack.ROWS // 2     # a core's 16 dims: 8 words
        return (self.cache + c * pack.HEAD_DIM // 2 + 2 * h * dims,
                [self.cap // pack.PV_ROWS, 2, pack.PV_ROWS, dims], [step, dims, row, 1])

    def check(self, x_words: int, out_words: int) -> None:
        """Every core consumes exactly what its channel sends and the join emits exactly what its S2MM takes, from
        the header the core reads; every region inside its argument."""
        hdr = pack.attn_tail(self.flags, self.cap)[-pack.HDR_WORDS:]
        objects = 1 + int(hdr[pack.H_G]) * int(hdr[pack.H_Q])
        for h in range(2):
            _, sizes, strides = self.stream(0, h)
            if 2 * pack.OBJ_WORDS + int(np.prod(sizes)) != 2 * objects * pack.OBJ_WORDS:
                raise ValueError(f"{self}: channel {h} sends {np.prod(sizes)} stream words for {objects} objects")
        if int(hdr[pack.H_G]) * int(hdr[pack.H_E]) * 6 * pack.ROWS != self.out_words_per_column():
            raise ValueError(f"{self}: the join emits a different count than the S2MM takes")
        last = max(self.stream(c, h)[0] + sum((s - 1) * t for s, t in zip(*self.stream(c, h)[1:])) + 1
                   for c in range(pack.KV_COLS) for h in range(2))
        if (last > out_words or self.x_out + pack.KV_COLS * self.lead_words() > out_words
                or self.out_offset + pack.KV_COLS * self.out_words_per_column() > out_words
                or self.x_offset + pack.OBJ_WORDS - self.lead_words() > x_words):
            raise ValueError(f"{self}: a region runs past its argument")

    def sequence(self, x_words: int, out_words: int, ssa=None) -> List[str]:
        L, lead, tail = [], self.lead_words(), pack.OBJ_WORDS - self.lead_words()
        for c in range(pack.KV_COLS):
            for h in range(2):
                fifo = f"@wL{c}_{h}"
                pieces = [("o", self.x_out + c * lead, lead, out_words), ("x", self.x_offset, tail, x_words)]
                if pack.LEAD_COPIES == 1:
                    L += lead_task(c, h, [(a, o, n) for a, o, n, _ in pieces], ssa)
                else:
                    for j, (arg, off, length, size) in enumerate(pieces * 2):
                        L.append(f"      aiex.npu.dma_memcpy_nd(%{arg}[0,0,0,{off}][1,1,1,{length}][0,0,0,1]) "
                                 f"{{id = {7 * h + j} : i64, metadata = {fifo}}} : memref<{size}xi32>")
                off, sizes, strides = self.stream(c, h)
                L.append(f"      aiex.npu.dma_memcpy_nd(%o[0,0,0,{off}][{','.join(map(str, sizes))}]"
                         f"[{','.join(map(str, strides))}]) {{id = {7 * h + 6} : i64, metadata = {fifo}}} "
                         f": memref<{out_words}xi32>")
            n = self.out_words_per_column()
            L.append(f"      aiex.npu.dma_memcpy_nd(%o[0,0,0,{self.out_offset + c * n}][1,1,1,{n}][0,0,0,1]) "
                     f"{{id = 15 : i64, metadata = @oL{c}, issue_token = true}} : memref<{out_words}xi32>")
        L += [f"      aiex.npu.dma_wait {{symbol = @oL{c}}}" for c in range(pack.KV_COLS)]
        return L


ACC = 96          # two accumulator sets of 3 x 16 fp32 (gate and up with F_SILU)


ARG = {"w": 0, "x": 1, "o": 2}
SHIM_BD_BASE, SHIM_BD_BYTES = 0x1D000, 0x20


def lead_task(c: int, h: int, pieces: Sequence[Tuple[str, int, int]], ssa) -> List[str]:
    """One chained BD task on shim column c, MM2S channel h (BDs 7h..): each (argument, word offset, words) piece a
    BD whose address is patched to its argument, chained in order, pushed once with repeat 1, so the pair's two
    cores each receive the whole lead (pack.LEAD_COPIES). The lowering's dma_memcpy_nd would write one BD and one
    push per piece per core: two to three times the control operations, and those set a call's fixed cost."""
    L = []
    for j, (arg, off, length) in enumerate(pieces):
        bd, last = 7 * h + j, j == len(pieces) - 1
        L.append(f"      aiex.npu.writebd {{bd_id = {bd} : i32, buffer_length = {length} : i32, buffer_offset = 0 : i32, "
                 f"column = {c} : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, "
                 "d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, "
                 "d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, "
                 "d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, "
                 "iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, "
                 "lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
                 f"next_bd = {0 if last else bd + 1} : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, "
                 f"packet_type = 0 : i32, row = 0 : i32, use_next_bd = {0 if last else 1} : i32, valid_bd = 1 : i32}}")
        v = next(ssa)
        L.append(f"      %lp{v} = arith.constant {off * 4} : i32")
        L.append(f"      aiex.npu.address_patch(%lp{v} : i32) {{addr = {(c << 25) | (SHIM_BD_BASE + bd * SHIM_BD_BYTES + 4)}"
                 f" : ui32, arg_idx = {ARG[arg]} : i32}}")
    v = next(ssa)
    L += [f"      %lb{v} = arith.constant {7 * h} : i32", f"      %lr{v} = arith.constant 1 : i32",
          f"      aiex.npu.push_queue({c}, 0, MM2S : {h}) bd_id %lb{v} repeat %lr{v} {{issue_token = false}} : i32, i32"]
    return L


def _core(i: int) -> str:
    O = pack.OBJ_WORDS
    view = f"!aie.objectfifosubview<memref<{O}xi32>>"
    oview = "!aie.objectfifosubview<memref<6xi32>>"
    return f'''    %core{i} = aie.core(%t{i}) {{
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %seven = arith.constant 7 : index
      %forever = arith.constant 4294967295 : index
      %core = arith.constant {i} : i32
      scf.for %it = %z to %forever step %one {{
        %hs = aie.objectfifo.acquire @wF{i}(Consume, 1) : {view}
        %h = aie.objectfifo.subview.access %hs[0] : {view} -> memref<{O}xi32>
        func.call @gemv_begin(%h, %xl{i}, %ctl{i}, %core) : (memref<{O}xi32>, memref<{pack.MAX_K // 2}xi32>, memref<16xi32>, i32) -> ()
        aie.objectfifo.release @wF{i}(Consume, 1)
        %g32 = memref.load %ctl{i}[%z] : memref<16xi32>
        %q32 = memref.load %ctl{i}[%one] : memref<16xi32>
        %G = arith.index_cast %g32 : i32 to index
        %Q = arith.index_cast %q32 : i32 to index
        %e32 = memref.load %ctl{i}[%seven] : memref<16xi32>
        %E = arith.index_cast %e32 : i32 to index
        scf.for %g = %z to %G step %one {{
          %gi = arith.index_cast %g : index to i32
          scf.for %q = %z to %Q step %one {{
            %ws = aie.objectfifo.acquire @wF{i}(Consume, 1) : {view}
            %w = aie.objectfifo.subview.access %ws[0] : {view} -> memref<{O}xi32>
            %qi = arith.index_cast %q : index to i32
            func.call @gemv_chunk(%w, %xl{i}, %acc{i}, %qi, %ctl{i}, %core, %gi) : (memref<{O}xi32>, memref<{pack.MAX_K // 2}xi32>, memref<{ACC}xf32>, i32, memref<16xi32>, i32, i32) -> ()
            aie.objectfifo.release @wF{i}(Consume, 1)
          }}
          scf.for %e = %z to %E step %one {{
            %ei = arith.index_cast %e : index to i32
            %os = aie.objectfifo.acquire @oF{i}(Produce, 1) : {oview}
            %o = aie.objectfifo.subview.access %os[0] : {oview} -> memref<6xi32>
            func.call @gemv_emit(%acc{i}, %o, %ctl{i}, %xl{i}, %core, %gi, %ei) : (memref<{ACC}xf32>, memref<6xi32>, memref<16xi32>, memref<{pack.MAX_K // 2}xi32>, i32, i32, i32) -> ()
            aie.objectfifo.release @oF{i}(Produce, 1)
          }}
        }}
      }}
      aie.end
    }} {{link_with = "gemv.o", stack_size = {STACK} : i32}}'''


def harness(program: Sequence, w_words: int, x_words: int, out_words: int) -> str:
    O, XK = pack.OBJ_WORDS, pack.MAX_K // 2
    for call in program:
        if isinstance(call, Attn):
            call.check(x_words, out_words)
            continue
        if call.w_offset + call.weight_words() > w_words:
            raise ValueError(f"{call}: weights run past the {w_words}-word argument")
        if call.x_offset + call.x_words_used() > x_words:
            raise ValueError(f"{call}: x runs past its argument")
        if any(off + n > out_words for c in range(pack.COLS) for off, n in call.out_pieces(c)):
            raise ValueError(f"{call}: out runs past its argument")
        if call.x_out is not None and call.x_out + call.k // 2 > out_words:
            raise ValueError(f"{call}: chained x runs past the out argument")
        if call.h_out is not None and call.h_out + call.n // 2 > out_words:
            raise ValueError(f"{call}: chained h runs past the out argument")
    L = ["module {", "  aie.device(npu1) {",
         f"    func.func private @gemv_begin(memref<{O}xi32>, memref<{XK}xi32>, memref<16xi32>, i32)",
         f"    func.func private @gemv_chunk(memref<{O}xi32>, memref<{XK}xi32>, memref<{ACC}xf32>, i32, memref<16xi32>, "
         "i32, i32)",
         f"    func.func private @gemv_emit(memref<{ACC}xf32>, memref<6xi32>, memref<16xi32>, memref<{XK}xi32>, i32, i32, i32)"]
    cores = []
    for c in range(pack.COLS):
        L += [f"    %s{c} = aie.tile({c}, 0)", f"    %m{c} = aie.tile({c}, 1)"]
        for r in range(pack.ROWS):
            i = c * pack.ROWS + r
            L += [f"    %t{i} = aie.tile({c}, {r + 2})",
                  f'    %xl{i} = aie.buffer(%t{i}) {{sym_name = "xl{i}"}} : memref<{XK}xi32>',
                  f'    %acc{i} = aie.buffer(%t{i}) {{sym_name = "acc{i}"}} : memref<{ACC}xf32>',
                  f'    %ctl{i} = aie.buffer(%t{i}) {{sym_name = "ctl{i}"}} : memref<16xi32>']
        for h in range(2):
            L.append(f"    aie.objectfifo @wL{c}_{h}(%s{c}, {{%m{c}}}, {FIFO_DEPTH} : i32) : "
                     f"!aie.objectfifo<memref<{2 * O}xi32>>")
            for e in range(2):
                i = c * pack.ROWS + 2 * h + e
                L.append(f"    aie.objectfifo @wF{i}(%m{c}, {{%t{i}}}, {FIFO_DEPTH} : i32) : "
                         f"!aie.objectfifo<memref<{O}xi32>>")
            a, b = c * pack.ROWS + 2 * h, c * pack.ROWS + 2 * h + 1
            L.append(f"    aie.objectfifo.link [@wL{c}_{h}] -> [@wF{a}, @wF{b}]([] [0, {O}])")
        L.append(f"    aie.objectfifo @oL{c}(%m{c}, {{%s{c}}}, {FIFO_DEPTH} : i32) : "
                 f"!aie.objectfifo<memref<{6 * pack.ROWS}xi32>>")
        for r in range(pack.ROWS):
            i = c * pack.ROWS + r
            L.append(f"    aie.objectfifo @oF{i}(%t{i}, {{%m{c}}}, {FIFO_DEPTH} : i32) : !aie.objectfifo<memref<6xi32>>")
            cores.append(_core(i))
        L.append("    aie.objectfifo.link [" + ", ".join(f"@oF{c * pack.ROWS + r}" for r in range(pack.ROWS))
                 + f"] -> [@oL{c}]([" + ", ".join(str(6 * r) for r in range(pack.ROWS)) + "] [])")
    L += cores
    L.append(f"    aie.runtime_sequence(%w: memref<{w_words}xi32>, %x: memref<{x_words}xi32>, "
             f"%o: memref<{out_words}xi32>) {{")
    ssa = iter(range(1 << 30))
    for call in program:
        if isinstance(call, Attn):
            L += call.sequence(x_words, out_words, ssa)
            continue
        sw = pack.stream_elems(call.n, call.k, call.flags) // 2
        # the host-built pair is one transfer; a chained first object is x (and h) from earlier outputs, then the
        # constant tail, once per core of the pair. BD ids per shim tile: channel h uses 7h..7h+5 for these and
        # 7h+6 for its weights, the output 14 (16 per tile).
        chained = call.x_out is not None and pack.LEAD_COPIES == 1
        xs = call.lead() if call.x_out is None else call.lead() * pack.LEAD_COPIES
        sizes = {"x": x_words, "o": out_words}
        for c in range(pack.COLS):
            for h in range(2):
                ch, fifo = c * 2 + h, f"@wL{c}_{h}"
                if chained:
                    L += lead_task(c, h, xs, ssa)
                else:
                    for j, (arg, off, length) in enumerate(xs):
                        L.append(f"      aiex.npu.dma_memcpy_nd(%{arg}[0,0,0,{off}][1,1,1,{length}][0,0,0,1]) "
                                 f"{{id = {7 * h + j} : i64, metadata = {fifo}}} : memref<{sizes[arg]}xi32>")
                L.append(f"      aiex.npu.dma_memcpy_nd(%w[0,0,0,{call.w_offset + ch * sw}][1,1,1,{sw}][0,0,0,1]) "
                         f"{{id = {7 * h + 6} : i64, metadata = {fifo}}} : memref<{w_words}xi32>")
            outs = call.out_pieces(c)
            if len(xs) * 2 + 2 + len(outs) > 16:
                raise ValueError(f"{call}: column {c} needs more than the shim tile's 16 BDs")
            # S2MM BDs from 15 down. Every S2MM push issues a completion token (the lowering sets the bit whatever
            # the attribute says), so the column waits once per piece: an unconsumed token makes a later call's wait
            # return a call early, and was left behind at teardown by the KV-append split (token_balance checks it)
            for j, (off, length) in enumerate(outs):
                L.append(f"      aiex.npu.dma_memcpy_nd(%o[0,0,0,{off}][1,1,1,{length}][0,0,0,1]) "
                         f"{{id = {15 - j} : i64, metadata = @oL{c}, issue_token = true}} "
                         f": memref<{out_words}xi32>")
        for c in range(pack.COLS):
            L += [f"      aiex.npu.dma_wait {{symbol = @oL{c}}}"] * len(call.out_pieces(c))
    L += ["    }", "  }", "}"]
    return "\n".join(L) + "\n"


def sources() -> dict:
    root = HERE.parents[1]
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p)
            for p in (HERE / "gemv.cc", HERE / "design.py", HERE / "pack.py", HERE / "reference.py")}


def tables_header() -> str:
    """tables.h: the core's range-reduced tables, the same arrays reference.py's device functions index."""
    from kernels.llm_gemv import reference as R

    def array(name, values):
        rows = [", ".join(f"0x{int(v):04x}" for v in values[i:i + 16]) for i in range(0, len(values), 16)]
        return f"alignas(32) static const uint16_t {name}[{len(values)}] = {{\n    " + ",\n    ".join(rows) + "\n};\n"

    return ("// generated by design.py from kernels/llm_gemv/reference.py; do not edit\n"
            "#pragma once\n#include <stdint.h>\n"
            f"constexpr uint32_t RSQ_BASE = 0x{R.RSQ_BASE:04x};\nconstexpr int RSQ_SIZE = {R.RSQ_SIZE};\n"
            f"constexpr int RSQ_STORED = {R.RSQ_STORED};\n"
            + array("RSQ", R.RSQ[:R.RSQ_STORED * 128]) + array("SILU_T", R.SILU_T) + array("EXP_T", R.EXP_T) + array("RECIP_T", R.RECIP_T))


def compile_kernel(folder: Path) -> Path:
    obj = folder / "gemv.o"
    (folder / "tables.h").write_text(tables_header(), encoding="utf-8")
    command([PEANO / "bin/clang++.exe", "-O2", "-std=c++20", "--target=aie2-none-unknown-elf", "-DNDEBUG",
             "-fstack-size-section", "-D__AIE_API_AIE_ADF_HPP__", f"-I{INSTALL / 'include'}", f"-I{folder}",
             "-Wno-attributes",
             "-c", HERE / "gemv.cc", "-o", obj], folder)
    if command([PEANO / "bin/llvm-nm.exe", "-u", obj], folder).strip():
        raise ValueError("unresolved helpers in gemv.o")
    return obj


SHIM_QUEUES = {0x1D204: ("s2mm", 0), 0x1D20C: ("s2mm", 1), 0x1D214: ("mm2s", 0), 0x1D21C: ("mm2s", 1)}


def token_balance(insts: bytes) -> dict:
    """Walk the binary instruction stream: per shim (column, direction, channel), the task pushes that issue a
    completion token and the syncs that consume one. Refused unless every pair is equal."""
    w = np.frombuffer(insts, np.uint32)
    pos, ops = 4, int(w[2])
    pushed, waited = {}, {}
    for _ in range(ops):
        op = int(w[pos]) & 0xFF
        if op == 0:                                   # write32: [op, 0, addr, 0, value, bytes]
            addr, value, size = int(w[pos + 2]), int(w[pos + 4]), int(w[pos + 5])
            reg = SHIM_QUEUES.get(addr & 0x1FFFFFF) if (addr >> 20) & 0x1F == 0 else None
            if reg is not None and value >> 31:
                key = ((addr >> 25) & 0x7F,) + reg
                pushed[key] = pushed.get(key, 0) + 1
        elif op == 1:                                 # blockwrite: [op, 0, addr, bytes, data...]
            size = int(w[pos + 3])
        elif op == 3:                                 # maskwrite: [op, 0, addr, 0, value, mask, bytes]
            size = int(w[pos + 6])
        elif op == 0x80:     # sync: [op, bytes, col<<16 | row<<8 | dir, channel<<24 | cols<<16 | rows<<8]
            size, a, b = int(w[pos + 1]), int(w[pos + 2]), int(w[pos + 3])
            if (b >> 8) & 0xFF != 1 or (b >> 16) & 0xFF != 1 or b & 0xFF or (a >> 8) & 0xFF:
                raise ValueError("a sync over more than one shim channel")
            key = ((a >> 16) & 0xFF, ("s2mm", "mm2s")[a & 1], (b >> 24) & 0xFF)
            waited[key] = waited.get(key, 0) + 1
        elif op >= 0x80:                              # address patch and other custom ops: [op, bytes, ...]
            size = int(w[pos + 1])
        else:
            raise ValueError(f"unknown instruction opcode {op:#x} at word {pos}")
        if size <= 0 or size % 4:
            raise ValueError(f"bad instruction size {size} at word {pos}")
        pos += size // 4
    if pos != w.size:
        raise ValueError(f"the stream is {w.size} words, its instructions {pos}")
    if pushed != waited:
        raise ValueError(f"completion tokens issued {pushed} but waited for {waited}")
    return {f"{c}/{d}{ch}": n for (c, d, ch), n in sorted(pushed.items())}


def audit(folder: Path) -> List[dict]:
    """Every core ELF: code <= 16 KiB, stack within STACK, no software float helper."""
    out = []
    for elf in sorted((folder / "prj").rglob("*core_*.elf")):
        sections = command([PEANO / "bin/llvm-size.exe", "-A", elf], folder)
        code = sum(int(s) for s in re.findall(r"^\.text\S*\s+(\d+)\s+", sections, re.M))
        frames = [int(s, 16) for s in re.findall(r"Size: (0x[0-9a-fA-F]+)",
                                                  command([PEANO / "bin/llvm-readobj.exe", "--stack-sizes", elf], folder))]
        symbols = command([PEANO / "bin/llvm-nm.exe", elf], folder)
        if re.search(r"\b__(?:add|sub|mul|div|eq|ne|lt|le|gt|ge|cmp)sf[23]\b", symbols):
            raise ValueError(f"{elf.name}: software float helper")
        if not 0 < code <= 16384 or not frames or sum(frames) + 256 > STACK:
            raise ValueError(f"{elf.name}: code {code} B or stack {frames} out of bounds")
        data = sum(int(s) for s in re.findall(r"^\.(?:data|rodata|bss)\S*\s+(\d+)\s+", sections, re.M))
        out.append(dict(elf=elf.name, code_bytes=code, data_bytes=data, stack_frames=frames))
    if len(out) != pack.CORES:
        raise ValueError(f"expected {pack.CORES} core images, found {len(out)}")
    return out


def build(folder: Path, program: Sequence[Call], w_words: int, x_words: int, out_words: int, label: str) -> dict:
    """Compile one program into ``folder`` (refused if it exists): gemv.o, harness.mlir, final.xclbin, insts.bin."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    before = sources()
    compile_kernel(folder)
    (folder / "harness.mlir").write_text(harness(program, w_words, x_words, out_words), encoding="utf-8")
    command([IRON / "Scripts/aiecc.exe", folder / "harness.mlir", f"--peano={PEANO}", "--get-npu-insts",
             "--npu-insts-name=insts.bin", "--get-xclbin", "--xclbin-name=final.xclbin", f"--tmpdir={folder / 'prj'}"],
            folder)
    if sources() != before:
        raise ValueError("a source changed during compilation")
    meta = dict(label=label, sources=before, program=[asdict(c) for c in program], w_words=w_words, x_words=x_words,
                out_words=out_words, cores=audit(folder),
                tokens=token_balance((folder / "insts.bin").read_bytes()),
                artifacts={n: sha(folder / n) for n in ("gemv.o", "harness.mlir", "final.xclbin", "insts.bin")})
    save(folder / "build.json", meta)
    print("LLM_GEMV_BUILD " + json.dumps(dict(label=label, folder=str(folder), calls=len(program),
                                                insts_sha256=meta["artifacts"]["insts.bin"],
                                                xclbin_sha256=meta["artifacts"]["final.xclbin"])), flush=True)
    return meta
