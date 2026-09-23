# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which kernel loop does each W4A8 array arm run? Compile-only; no NPU.

For every (arm, mode) this compiles the kernel the array design links -- upstream mm.cc
(-Di8_i32_ONLY) for `upstream`, kernels/w4a8_array/w4a8_array_kernels.cc for the rest --
with IRON's exact Peano command (static_probe.IRON_FLAGS, which --match-cache showed
reproduces IRON's objects), reads the arm's function back with tools/aie_disasm.py and
reports its steady-state k loop (bundles, vmacs, MAC per cycle) and frame. Two
comparisons make that table mean something:

  probe   the same function compiled from the one-core probe's w4a8_kernels.cc with the
          same defines: "identical" means the array core runs, instruction for instruction,
          the loop results/aie/w4a8_probe_npu.log timed (not applicable to `upstream`).
  iron    the newest object of the arm's name under ~/.npu/cache (what an array run
          actually linked): "identical" means the static table describes what ran.
          "not built" until the design has been JIT-compiled.

Usage:
    python kernels/w4a8_array/object_check.py                      # 64/128/64, all arms
    python kernels/w4a8_array/object_check.py --tile 64x128x64 --arms native:unroll2,upstream
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "w4a8_probe"))

import static_probe  # noqa: E402  (also puts tools/ on sys.path for aie_disasm)

aie_disasm = static_probe.aie_disasm

ARRAY_SRC = os.path.join(_HERE, "w4a8_array_kernels.cc")
PROBE_DIR = os.path.join(os.path.dirname(_HERE), "w4a8_probe")
PROBE_SRC = os.path.join(PROBE_DIR, "w4a8_kernels.cc")

# arm -> (symbol, MACs per vmac); must agree with whole_array_w4a8.ARMS
ARMS = {
    "upstream": ("matmul_i8_i32", 256),
    "i8": ("mm_i8i8_local", 256),
    "unpack": ("mm_i8i4_unpack", 256),
    "native": ("mm_i8i4_native", 512),
}
DEFAULT_ARMS = ("upstream:default,i8:default,i8:unroll2,unpack:no-unroll,unpack:unroll2,"
                "native:default,native:unroll2")


def compile_to(src, defines, out, extra_includes=()):
    import subprocess
    cmd = [static_probe.CLANG, src, "-c", "-o", out, f"-I{static_probe.INCLUDE}",
           *[f"-I{d}" for d in extra_includes], *static_probe.IRON_FLAGS, *defines]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"compile failed ({os.path.basename(src)} {' '.join(defines)}):\n{p.stderr}")


def fn_body(obj, objdump, fn):
    s = static_probe.section_for(aie_disasm.parse(aie_disasm.disassemble(obj, objdump)), fn)
    return None if s is None else [" ; ".join(b.fields) for b in s.bundles]


def newest_cache_obj(name):
    c = glob.glob(os.path.expanduser(f"~/.npu/cache/*/{name}"))
    return max(c, key=os.path.getmtime) if c else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default="64x128x64", help="core tile m x k x n")
    ap.add_argument("--arms", default=DEFAULT_ARMS, help="comma list of arm:mode")
    ap.add_argument("--objdump", help="llvm-objdump that knows elf32-aie")
    args = ap.parse_args(argv)

    m, k, n = (int(x) for x in args.tile.lower().split("x"))
    objdump = aie_disasm.find_objdump(args.objdump)
    tmp = tempfile.mkdtemp(prefix="w4a8a_oc_")
    print(f"W4A8 array object check, tile {m}x{k}x{n} (Peano, IRON's compile command)")
    print(f"  flags {' '.join(static_probe.IRON_FLAGS)}")
    print(f"  {'arm':<8} {'mode':<9} {'function':<16} {'bundles':>7} {'vmacs':>5} "
          f"{'MAC/cyc':>8} {'frame':>5}  {'= probe':<10} {'= IRON cache object'}")

    rc = 0
    for tok in args.arms.split(","):
        arm, mode = tok.strip().split(":")
        fn, macs = ARMS[arm]
        dims = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]
        static = os.path.join(tmp, f"{arm}_{mode}.o")
        if arm == "upstream":
            if mode != "default":
                raise SystemExit("upstream takes mode default only")
            compile_to(static_probe.UPSTREAM_MM, dims + ["-Di8_i32_ONLY"], static)
            probe_cmp = "n/a"
            cache_cands = glob.glob(os.path.expanduser("~/.npu/cache/*/matmul_i8_i32*.o"))
        else:
            defines = dims + static_probe.MODES[mode]
            compile_to(ARRAY_SRC, defines, static, extra_includes=[PROBE_DIR])
            probe_obj = os.path.join(tmp, f"{arm}_{mode}_probe.o")
            compile_to(PROBE_SRC, defines, probe_obj)
            same = fn_body(static, objdump, fn) == fn_body(probe_obj, objdump, fn)
            probe_cmp = "identical" if same else "DIFFERENT"
            rc |= not same
            name = f"w4a8a_{arm}_{mode}_{m}x{k}x{n}.o"
            cache_cands = [p for p in [newest_cache_obj(name)] if p]

        rec = static_probe.analyse(static, objdump, [fn]).get(fn)
        if rec is None:
            print(f"  {arm:<8} {mode:<9} {fn:<16} function not found")
            rc = 1
            continue
        mac_loops = [lp for lp in rec["loops"] if lp["macs"]]
        lp = max(mac_loops, key=lambda d: d["macs"]) if mac_loops else None
        body = fn_body(static, objdump, fn)
        if not cache_cands:
            iron_cmp = "not built"
        else:
            hits = [p for p in cache_cands if fn_body(p, objdump, fn) == body]
            iron_cmp = (f"identical ({os.path.basename(os.path.dirname(max(hits, key=os.path.getmtime)))})"
                        if hits else f"DIFFERENT (newest {os.path.basename(os.path.dirname(max(cache_cands, key=os.path.getmtime)))})")
        if lp:
            print(f"  {arm:<8} {mode:<9} {fn:<16} {lp['bundles']:>7} {lp['macs']:>5} "
                  f"{lp['macs'] * macs / lp['bundles']:>8.1f} {rec['frame'] if rec['frame'] is not None else '-':>5}  "
                  f"{probe_cmp:<10} {iron_cmp}")
        else:
            print(f"  {arm:<8} {mode:<9} {fn:<16} no hardware loop holds a vmac")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
