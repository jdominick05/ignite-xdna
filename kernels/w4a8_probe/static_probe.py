"""W4A8 on AIE2, read off the machine code: what int4 weights cost in a core.

Compile only -- no NPU, no hardware context, no IRON. Compiles `w4a8_kernels.cc` and
upstream's own `aie_kernels/aie2/mm.cc` with the exact Peano command IRON runs, then
reads every kernel's inner loop back with `tools/aie_disasm.py`. On AIE2 the bundle count
of a hardware loop body is its cycle count (the core is statically scheduled; see that
tool's docstring), so multiply-accumulates per cycle in a loop's steady state is
`vmacs x MACs-per-vmac / bundles`.

Why it exists. TODO 3.2's W4A8 item and docs/SILICON.md both record int8 x int4 MACs as
AIE2p (Strix) only, from the MAC table in OGOAT's `device.yaml` -- a cost model, the same
source that once had int16 x int16 missing and was wrong about it. aie_api's AIE2 headers
define `aie::mmul<M,K,N,int8,int4>` (detail/aie2/mmul_8_4.hpp), lowered by Peano to
`__builtin_aiev2_I512_I512_ACC1024_acc32_mac_conf` -- the builtin int8 x int8 uses -- with
the B-mode field of the MAC configuration word cleared. This measures what the compiler
actually emits for it, and for the alternative the item proposed (store int4, unpack to
int8, run the stock int8 mmul).

Kernels (all in w4a8_kernels.cc except matmul_i8_i32):
  copy_i8 / unpack_i4      standalone loops, 64 elements per iteration
  matmul_i8_i32            upstream mm.cc, compiled with IRON's -Di8_i32_ONLY
  mm_i8i8_local            the same kernel re-typed locally (the control for the two below)
  mm_i8i4_unpack           B stored as int4, aie::unpack'd to int8 inside the k loop
  mm_i8i4_native           B stored as int4, consumed by mmul<4,16,8,int8,int4>
  *_2x2                    the same three with A expanded twice instead of four times

What this does NOT establish: that the hardware executes an int8 x int4 vmac in one cycle,
or that the result is right. Both need a run on the NPU (`w4a8_probe.py`).

Usage:
    python kernels/w4a8_probe/static_probe.py
    python kernels/w4a8_probe/static_probe.py --dims 64x64x64,64x128x64 --verbose
    python kernels/w4a8_probe/static_probe.py --match-cache      # recipe == IRON's?
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import aie_disasm  # noqa: E402

SITE = os.path.expanduser("~/mlir-aie/ironenv/Lib/site-packages")
CLANG = os.path.join(SITE, "llvm-aie", "bin", "clang++.exe")
INCLUDE = os.path.join(SITE, "mlir_aie", "include")
UPSTREAM_MM = os.path.join(INCLUDE, "aie_kernels", "aie2", "mm.cc")
SOURCE = os.path.join(_HERE, "w4a8_kernels.cc")

# The Peano command IRON builds in aie/utils/compile/utils.py (mlir-aie v1.4.2), minus
# the -MD/-MF dependency-file pair, which does not touch code generation.
IRON_FLAGS = [
    "-std=c++20",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
    "-O2",
    "-DNDEBUG",
    "-D__AIE_API_AIE_ADF_HPP__",
    "--target=aie2-none-unknown-elf",
]

# MACs one vmac performs, per kernel: 4x8x8 = 256 for int8 x int8, 4x16x8 = 512 for
# int8 x int4.
GEMM = {
    "matmul_i8_i32": 256,
    "mm_i8i8_local": 256,
    "mm_i8i4_unpack": 256,
    "mm_i8i4_native": 512,
    "mm_i8i8_local_2x2": 256,
    "mm_i8i4_unpack_2x2": 256,
    "mm_i8i4_native_2x2": 512,
}
STANDALONE = ("copy_i8", "unpack_i4")

# k-loop handling, as compile defines (see the INNER_PRAGMA block in the .cc).
MODES = {
    "default": [],
    "no-unroll": ["-DINNER_NO_UNROLL"],
    "unroll2": ["-DINNER_UNROLL2"],
}

MAC_RE = re.compile(r"^(vmac|vmul|vnegmac|vnegmul|vmsc|vaddmac|vsubmac)\b")
UNPACK_RE = re.compile(r"\bunpack\b|\.unpack\.")
# Stack-pointer-relative only. aie_disasm's function-level count also takes p7 as a
# frame pointer, but these kernels use p7 as a plain data pointer, so inside a loop
# that rule counts ordinary operand loads.
SP_RE = re.compile(r"\[sp\b")
LDST_RE = re.compile(r"^(st|lda|ldb|vst|vlda|vldb)\b")


def compile_obj(src: str, defines: list[str], out: str) -> None:
    cmd = [CLANG, src, "-c", "-o", out, f"-I{INCLUDE}", *IRON_FLAGS, *defines]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"compile failed: {' '.join(defines)}\n{proc.stderr.strip()}")


def ops(bundle: aie_disasm.Bundle) -> list[str]:
    return [f.split()[0] if f.split() else f for f in bundle.live]


def loop_stats(loop: aie_disasm.Loop) -> dict:
    n_mac = n_unpack = n_stack = 0
    for b in loop.bundles:
        for f in b.live:
            head = f.split()[0]
            if MAC_RE.match(head):
                n_mac += 1
            if UNPACK_RE.search(head):
                n_unpack += 1
            if LDST_RE.match(head) and SP_RE.search(f):
                n_stack += 1
    return {"bundles": loop.n_bundles, "macs": n_mac, "unpack": n_unpack, "stack": n_stack}


def section_for(sections, fn: str):
    for s in sections:
        if s.name.endswith("." + fn) or s.name == fn:
            return s
    return None


def analyse(obj: str, objdump: str, fns) -> dict:
    sections = aie_disasm.parse(aie_disasm.disassemble(obj, objdump))
    out = {}
    for fn in fns:
        s = section_for(sections, fn)
        if s is None:
            continue
        loops = [loop_stats(lp) for lp in s.loops]
        sp_refs = sum(
            1
            for b in s.bundles
            for f in b.live
            if LDST_RE.match(f.split()[0]) and SP_RE.search(f)
        )
        out[fn] = {
            "section": s,
            "bundles": len(s.bundles),
            "frame": s.frame_bytes,
            "stack": sp_refs,
            "loops": loops,
        }
    return out


def fmt_gemm_row(dims, mode, fn, rec, macs_per_vmac) -> str:
    mac_loops = [lp for lp in rec["loops"] if lp["macs"]]
    if not mac_loops:
        return f"  {dims:>10}  {mode:>9}  {fn:<20} no hardware loop holds a vmac"
    # The steady-state loop is the one that issues the most vmacs.
    lp = max(mac_loops, key=lambda d: d["macs"])
    per_cycle = lp["macs"] * macs_per_vmac / lp["bundles"]
    vmac_rate = lp["macs"] / lp["bundles"]
    return (
        f"  {dims:>10}  {mode:>9}  {fn:<20} {lp['bundles']:>7} {lp['macs']:>5} "
        f"{vmac_rate:>9.3f} {per_cycle:>9.1f} {lp['unpack']:>7} {lp['stack']:>6} "
        f"{rec['stack']:>6} {rec['frame'] if rec['frame'] is not None else '-':>6}"
    )


LOAD_RE = re.compile(
    r"^(vlda|vldb|lda|ldb)(\.\S+)?\s+\S+,\s*\[(p\d+)(?:,\s*#?(0x[0-9a-f]+|\d+))?\]"
    r"(?:,\s*(#?\S+))?"
)

# What one A pointer and one B pointer advance per k-step, per kernel: A by one A tile
# (4 x s int8 = bytes), B by one tile row of B -- colB = N/8 tiles of the stored B tile
# size, so the B step depends on N. The unpack kernels advance B through a modifier
# register (`m0`) instead of an immediate, marked "m".
POINTER_STEP = {  # fn -> (A tile bytes, stored B tile bytes or "m")
    "matmul_i8_i32": (32, 64),  # upstream mm.cc, row-major B: the same 4x8x8 tiles
    "mm_i8i8_local": (32, 64), "mm_i8i4_unpack": (32, "m"), "mm_i8i4_native": (64, 64),
    "mm_i8i8_local_2x2": (32, 64), "mm_i8i4_unpack_2x2": (32, "m"),
    "mm_i8i4_native_2x2": (64, 64),
}


def paired_loads(loop, fn, n_dim: int) -> dict:
    """Two-load bundles in a loop, each load's pointer given a role from its increment.

    A pointer that post-increments by the A step anywhere in the loop is taken as an A
    pointer, likewise B; one only ever used with an offset, or seen with both steps
    (the compiler juggles some through `mov`), stays '?'. Same-role pairs are two
    loads from ONE buffer, so from one bank -- which a buffer-level bank check cannot
    flag. This is a heuristic reading of pointer roles, not a resolved address.
    """
    a_step, b_tile = POINTER_STEP[fn]
    b_step = b_tile if b_tile == "m" else b_tile * (n_dim // 8)
    if b_step == a_step:
        raise SystemExit(f"{fn} at N={n_dim}: A and B pointers step by the same {a_step} "
                         f"bytes, so their roles cannot be told apart by increment")
    role: dict[str, set] = {}
    for b in loop.bundles:
        for f in b.live:
            m = LOAD_RE.match(f)
            if not m:
                continue
            reg, inc = m.group(3), m.group(5)
            if inc is None:
                continue
            if inc.startswith("m"):
                r = "B" if b_step == "m" else "?"
            else:
                v = int(inc.lstrip("#"), 0)
                r = "A" if v == a_step else ("B" if v == b_step else "?")
            role.setdefault(reg, set()).add(r)
    resolved = {k: (v.pop() if len(v) == 1 else "?") for k, v in role.items()}
    pairs, counts = [], {"same": 0, "cross": 0, "unknown": 0}
    for b in loop.bundles:
        loads = [LOAD_RE.match(f) for f in b.live]
        loads = [m for m in loads if m]
        if len(loads) < 2:
            continue
        rs = [resolved.get(m.group(3), "?") for m in loads[:2]]
        kind = "unknown" if "?" in rs else ("same" if rs[0] == rs[1] else "cross")
        counts[kind] += 1
        pairs.append((" & ".join(f"{m.group(1)}{m.group(2) or ''} [{m.group(3)}]={r}"
                                 for m, r in zip(loads[:2], rs)), kind))
    return {"pairs": pairs, **counts}


def print_loop(rec, fn):
    s = rec["section"]
    for lp in s.loops:
        st = loop_stats(lp)
        if not st["macs"] and not st["unpack"]:
            continue
        print(f"    {fn} loop {lp.name}: {lp.n_bundles} bundles")
        for b in lp.bundles:
            print(f"        0x{b.addr:04x}  {' | '.join(b.live) or '(all nop)'}")


def match_cache(objdump: str, outdir: str) -> int:
    """Compile upstream mm.cc over a dims grid and look for byte-identical code in the
    IRON-built matmul_i8_i32 objects left in ~/.npu/cache. A match means this script's
    compile command reproduces IRON's, not just approximates it."""
    cached = sorted(glob.glob(os.path.expanduser("~/.npu/cache/*/matmul_i8_i32_*.o")))
    if not cached:
        print("no cached matmul_i8_i32 objects under ~/.npu/cache")
        return 1

    def text_of(obj):
        s = section_for(aie_disasm.parse(aie_disasm.disassemble(obj, objdump)), "matmul_i8_i32")
        return None if s is None else "\n".join(" ; ".join(b.fields) for b in s.bundles)

    want = {}
    for c in cached:
        t = text_of(c)
        if t is not None:
            want.setdefault(t, []).append(c)
    print(f"{len(cached)} cached objects, {len(want)} distinct matmul_i8_i32 bodies")

    grid = [16, 32, 64, 128, 256]
    found = {}
    for m, k, n in itertools.product(grid, grid, grid):
        if m % 16 or n % 16 or k % 8:
            continue
        obj = os.path.join(outdir, f"up_{m}x{k}x{n}.o")
        try:
            compile_obj(UPSTREAM_MM, [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
                                      "-Di8_i32_ONLY"], obj)
        except SystemExit:
            continue
        t = text_of(obj)
        if t in want and t not in found:
            found[t] = (m, k, n)
            if len(found) == len(want):
                break
    for t, objs in want.items():
        dims = found.get(t)
        tag = f"m/k/n {dims[0]}/{dims[1]}/{dims[2]}" if dims else "NO MATCH in grid"
        names = sorted({os.path.basename(o) for o in objs})
        print(f"  {tag:<22} identical to IRON's {', '.join(names)} "
              f"({len(objs)} cache dir{'s' if len(objs) != 1 else ''})")
    print(f"matched {len(found)} of {len(want)} distinct cached bodies bundle for bundle")
    return 0 if found else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dims", default="64x64x64,64x128x64,64x256x64",
                    help="comma-separated MxKxN kernel tile sizes")
    ap.add_argument("--modes", default="default,no-unroll,unroll2",
                    help=f"inner k-loop handling, comma-separated: {', '.join(MODES)}")
    ap.add_argument("--objdump", help="llvm-objdump that knows elf32-aie")
    ap.add_argument("--keep", help="directory to keep the objects in")
    ap.add_argument("--verbose", action="store_true", help="print each vmac loop body")
    ap.add_argument("--paired", action="store_true",
                    help="list each vmac loop's two-load bundles with pointer roles")
    ap.add_argument("--match-cache", action="store_true",
                    help="check the compile command against IRON-built objects and stop")
    args = ap.parse_args(argv)

    if not os.path.exists(CLANG):
        raise SystemExit(f"Peano clang++ not found at {CLANG}")
    objdump = aie_disasm.find_objdump(args.objdump)
    outdir = args.keep or tempfile.mkdtemp(prefix="w4a8_static_")
    os.makedirs(outdir, exist_ok=True)

    print("W4A8 static probe (Peano, IRON's compile command)")
    print(f"  clang    {CLANG}")
    print(f"  flags    {' '.join(IRON_FLAGS)}")
    print(f"  source   {SOURCE}")
    print(f"  upstream {UPSTREAM_MM}")
    print()

    if args.match_cache:
        return match_cache(objdump, outdir)

    # Standalone loops: one compile, the dims do not reach them.
    obj = os.path.join(outdir, "standalone.o")
    compile_obj(SOURCE, [], obj)
    rec = analyse(obj, objdump, STANDALONE)
    print("Standalone loops, 64 elements per iteration")
    for fn in STANDALONE:
        r = rec[fn]
        for lp, st in zip(r["section"].loops, r["loops"]):
            print(f"  {fn:<10} {st['bundles']} bundles/iteration = "
                  f"{64 / st['bundles']:.0f} elements/cycle, unpack ops {st['unpack']}, "
                  f"stack refs {st['stack']}")
            for b in lp.bundles:
                print(f"        0x{b.addr:04x}  {' | '.join(b.live)}")
    print()

    header = (f"  {'MxKxN':>10}  {'k loop':>9}  {'kernel':<20} {'bundles':>7} {'vmacs':>5} "
              f"{'vmac/cyc':>9} {'MAC/cyc':>9} {'unpacks':>7} {'lp stk':>6} "
              f"{'fn stk':>6} {'frame':>6}")
    print("GEMM kernels: the hardware loop that issues the most vmacs, per kernel")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dims in args.dims.split(","):
        m, k, n = (int(x) for x in dims.lower().split("x"))
        dflags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]
        up = os.path.join(outdir, f"upstream_{dims}.o")
        compile_obj(UPSTREAM_MM, dflags + ["-Di8_i32_ONLY"], up)
        up_rec = analyse(up, objdump, ["matmul_i8_i32"])
        for mode in args.modes.split(","):
            extra = MODES[mode]
            obj = os.path.join(outdir, f"w4a8_{dims}_{mode}.o")
            compile_obj(SOURCE, dflags + extra, obj)
            rec = analyse(obj, objdump, [f for f in GEMM if f != "matmul_i8_i32"])
            rows = [(fn, rec[fn], per, mode) for fn, per in GEMM.items() if fn in rec]
            if mode == "default":
                rows.insert(0, ("matmul_i8_i32", up_rec["matmul_i8_i32"], 256, "upstream"))
            for fn, r, per, label in rows:
                print(fmt_gemm_row(dims, label, fn, r, per))
                if args.verbose:
                    print_loop(r, fn)
                if args.paired:
                    mac = [lp for lp in r["section"].loops if loop_stats(lp)["macs"]]
                    if mac:
                        lp = max(mac, key=lambda L: loop_stats(L)["macs"])
                        pl = paired_loads(lp, fn, n)
                        print(f"      two-load bundles: {len(pl['pairs'])} "
                              f"(same buffer {pl['same']}, A&B {pl['cross']}, "
                              f"unresolved {pl['unknown']})")
                        for text, kind in pl["pairs"]:
                            print(f"        {kind:<7} {text}")
        print()
    print("vmac/cyc is the loop's issue rate against a ceiling of 1.000; MAC/cyc multiplies")
    print("it by the MACs each vmac performs (256 int8 x int8, 512 int8 x int4). Both are the")
    print("steady state of ONE loop, not a whole-kernel figure: C loads and stores, and the")
    print("loop's prologue and epilogue, sit outside it. lp stk / fn stk count loads and")
    print("stores through the stack pointer inside that loop and in the whole function.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
