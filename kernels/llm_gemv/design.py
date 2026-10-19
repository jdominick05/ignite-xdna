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
from typing import List, Optional, Sequence

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
                                  # call's output), and x_offset names the shape's constant tail

    def __post_init__(self):
        pack.check_shape(self.n, self.k)

    def x_words_used(self) -> int:
        return pack.X_SLOT_WORDS if self.x_out is None else pack.OBJ_WORDS - self.k // 2


def _core(i: int) -> str:
    O = pack.OBJ_WORDS
    view = f"!aie.objectfifosubview<memref<{O}xi32>>"
    oview = "!aie.objectfifosubview<memref<6xi32>>"
    return f'''    %core{i} = aie.core(%t{i}) {{
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %forever = arith.constant 4294967295 : index
      scf.for %it = %z to %forever step %one {{
        %hs = aie.objectfifo.acquire @wF{i}(Consume, 1) : {view}
        %h = aie.objectfifo.subview.access %hs[0] : {view} -> memref<{O}xi32>
        func.call @gemv_begin(%h, %xl{i}, %ctl{i}) : (memref<{O}xi32>, memref<{pack.MAX_K // 2}xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @wF{i}(Consume, 1)
        %g32 = memref.load %ctl{i}[%z] : memref<16xi32>
        %q32 = memref.load %ctl{i}[%one] : memref<16xi32>
        %G = arith.index_cast %g32 : i32 to index
        %Q = arith.index_cast %q32 : i32 to index
        scf.for %g = %z to %G step %one {{
          scf.for %q = %z to %Q step %one {{
            %ws = aie.objectfifo.acquire @wF{i}(Consume, 1) : {view}
            %w = aie.objectfifo.subview.access %ws[0] : {view} -> memref<{O}xi32>
            %qi = arith.index_cast %q : index to i32
            func.call @gemv_chunk(%w, %xl{i}, %acc{i}, %qi) : (memref<{O}xi32>, memref<{pack.MAX_K // 2}xi32>, memref<48xf32>, i32) -> ()
            aie.objectfifo.release @wF{i}(Consume, 1)
          }}
          %os = aie.objectfifo.acquire @oF{i}(Produce, 1) : {oview}
          %o = aie.objectfifo.subview.access %os[0] : {oview} -> memref<6xi32>
          func.call @gemv_emit(%acc{i}, %o) : (memref<48xf32>, memref<6xi32>) -> ()
          aie.objectfifo.release @oF{i}(Produce, 1)
        }}
      }}
      aie.end
    }} {{link_with = "gemv.o", stack_size = {STACK} : i32}}'''


def harness(program: Sequence[Call], w_words: int, x_words: int, out_words: int) -> str:
    O, XK = pack.OBJ_WORDS, pack.MAX_K // 2
    for call in program:
        if call.w_offset + pack.block_words(call.n, call.k) > w_words:
            raise ValueError(f"{call}: weights run past the {w_words}-word argument")
        if call.x_offset + call.x_words_used() > x_words or call.out_offset + call.n // 2 > out_words:
            raise ValueError(f"{call}: x or out runs past its argument")
        if call.x_out is not None and call.x_out + call.k // 2 > out_words:
            raise ValueError(f"{call}: chained x runs past the out argument")
    L = ["module {", "  aie.device(npu1) {",
         f"    func.func private @gemv_begin(memref<{O}xi32>, memref<{XK}xi32>, memref<16xi32>)",
         f"    func.func private @gemv_chunk(memref<{O}xi32>, memref<{XK}xi32>, memref<48xf32>, i32)",
         "    func.func private @gemv_emit(memref<48xf32>, memref<6xi32>)"]
    cores = []
    for c in range(pack.COLS):
        L += [f"    %s{c} = aie.tile({c}, 0)", f"    %m{c} = aie.tile({c}, 1)"]
        for r in range(pack.ROWS):
            i = c * pack.ROWS + r
            L += [f"    %t{i} = aie.tile({c}, {r + 2})",
                  f'    %xl{i} = aie.buffer(%t{i}) {{sym_name = "xl{i}"}} : memref<{XK}xi32>',
                  f'    %acc{i} = aie.buffer(%t{i}) {{sym_name = "acc{i}"}} : memref<48xf32>',
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
    for call in program:
        sw = pack.stream_elems(call.n, call.k) // 2
        ow = call.n // 2 // pack.COLS
        xk, tw = call.k // 2, pack.OBJ_WORDS - call.k // 2
        for c in range(pack.COLS):
            for h in range(2):
                ch, fifo = c * 2 + h, f"@wL{c}_{h}"
                if call.x_out is None:     # the host-built pair, one transfer (BD ids 0-1, 4-5 per shim tile)
                    xs = [("x", call.x_offset, pack.X_SLOT_WORDS, x_words)]
                else:                      # x from an earlier output, then the constant tail; twice, one per core
                    xs = [("o", call.x_out, xk, out_words), ("x", call.x_offset, tw, x_words)] * 2
                for j, (arg, off, length, size) in enumerate(xs):
                    L.append(f"      aiex.npu.dma_memcpy_nd(%{arg}[0,0,0,{off}][1,1,1,{length}][0,0,0,1]) "
                             f"{{id = {5 * h + j} : i64, metadata = {fifo}}} : memref<{size}xi32>")
                L.append(f"      aiex.npu.dma_memcpy_nd(%w[0,0,0,{call.w_offset + ch * sw}][1,1,1,{sw}][0,0,0,1]) "
                         f"{{id = {5 * h + 4} : i64, metadata = {fifo}}} : memref<{w_words}xi32>")
            L.append(f"      aiex.npu.dma_memcpy_nd(%o[0,0,0,{call.out_offset + c * ow}][1,1,1,{ow}][0,0,0,1]) "
                     f"{{id = 10 : i64, metadata = @oL{c}, issue_token = true}} : memref<{out_words}xi32>")
        for c in range(pack.COLS):
            L.append(f"      aiex.npu.dma_wait {{symbol = @oL{c}}}")
    L += ["    }", "  }", "}"]
    return "\n".join(L) + "\n"


def sources() -> dict:
    root = HERE.parents[1]
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p)
            for p in (HERE / "gemv.cc", HERE / "design.py", HERE / "pack.py")}


def compile_kernel(folder: Path) -> Path:
    obj = folder / "gemv.o"
    command([PEANO / "bin/clang++.exe", "-O2", "-std=c++20", "--target=aie2-none-unknown-elf", "-DNDEBUG",
             "-fstack-size-section", "-D__AIE_API_AIE_ADF_HPP__", f"-I{INSTALL / 'include'}", "-Wno-attributes",
             "-c", HERE / "gemv.cc", "-o", obj], folder)
    if command([PEANO / "bin/llvm-nm.exe", "-u", obj], folder).strip():
        raise ValueError("unresolved helpers in gemv.o")
    return obj


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
        out.append(dict(elf=elf.name, code_bytes=code, stack_frames=frames))
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
                artifacts={n: sha(folder / n) for n in ("gemv.o", "harness.mlir", "final.xclbin", "insts.bin")})
    save(folder / "build.json", meta)
    print("LLM_GEMV_BUILD " + json.dumps(dict(label=label, folder=str(folder), calls=len(program),
                                                insts_sha256=meta["artifacts"]["insts.bin"],
                                                xclbin_sha256=meta["artifacts"]["final.xclbin"])), flush=True)
    return meta
