#!/usr/bin/env python3
"""How many bits of the shift register does the AIE2 shift-round-saturate (vsrs) read?

WHY THIS EXISTS
---------------
The VitisAI EP only places a Conv on the NPU for shift_cut in [0, 16] (sigma 14..30,
docs/BENCHMARKS.md "Theorem 1's sigma window"), and the repo has carried a "5-bit shifter"
framing that nothing measured (TODO.md, TO VERIFY). The DPU overlay's requant handlers
(phoenix 4x4.xclbin, results/aie/dpu_overlay_extract_*.log) do `mov sN, rX; vsrs ..., sN`,
so whatever the s register's shifter width is, it bounds the DPU too. The toolchain does
not state it: llvm-aie gives s0-s3 a 32-bit scalar class with a 2-bit encoding, and the
aiev2_srs.h intrinsics take a plain `int shft`. This asks the silicon.

WHAT IT RUNS
------------
One Worker on Tile(0, 2) runs kernels/srs_probe/srs_kernels.cc once per (mode, shift):
  mode 0  v32acc32 -> v32int16 (lsrs)       32 lanes of known int32
  mode 1  v32acc32 -> v32int8  (ssrs)       the DPU's int8 output path
  mode 2  v16acc64 -> v16int32 (lsrs)       16 lanes of (int32 << up), 64-bit accumulators
  mode 3  v32int8 -> v32acc32 (ups)          the DPU's bias path: 32 fixed int8 lanes upshifted by the
                                             runtime scalar; reports every effective k that fits
                                             (results/aie/ups_shift_probe_desktop2_20261012T202624Z.log)
Rounding is forced to floor and saturation to saturate, and both are read back.
Every shift in --shifts is passed unmodified through `mov s0, rX`.

THE ORACLE
----------
For each shift s, every lane is compared EXACTLY (no tolerance) against
sat_out(floor(acc / 2**k)) for each candidate k(s):
  as_is   k = s (only for s >= 0)     and31   k = s & 31     and63   k = s & 63
  and127  k = s & 127                 clamp31 k = min(max(s, 0), 31)
  clamp63 k = min(max(s, 0), 63)
  f6_m4_59 k = (s & 63) if (s & 63) < 60 else (s & 63) - 64; a negative k shifts LEFT
           (added after the third sitting showed codes 60..63 multiplying, see the log)
A candidate survives only if it matches every lane at every shift in a mode. The verdict
names the survivors; if none survive, the per-shift table says where each one first fails.
Shifts are run in a deterministic order and the value set is fixed (seeded), so a re-run
is the same experiment.

USAGE (mlir-aie ironenv, NOT resnet_env17 -- see kernels/README.md)
    export PATH=/c/Users/<user>/mlir-aie/ironenv/Scripts:$PATH
    export PYTHONPATH=/c/Xilinx/XRT/xrt_sdk/xrt/python
    python3 kernels/srs_probe/srs_probe.py --shifts=-8:72
Check `xrt-smi examine -r aie-partitions` reads "No hardware contexts running" first.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config

N_IO = 64
LANE = 32  # must match srs_kernels.cc: lanes at a 128 B-aligned word offset
_KERNEL_SRC = Path(__file__).resolve().parent / "srs_kernels.cc"
_XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
MODES = {0: ("acc32->int16", 16, 32), 1: ("acc32->int8", 8, 32), 2: ("acc64->int32", 32, 16),
         3: ("int8-ups->acc32", 32, 32)}
LANES8 = [1, -1, 127, -128, 2, -2, 3, -3, 5, -5, 64, -64, 100, -100, 17, -17,
          0, 7, -7, 33, -33, 99, -99, 126, -127, 11, -11, 85, -85, 42, -42, 120]


def ups_fit(got):
    """Every effective shift k (negative = right) and 32-bit treatment that reproduces all mode-3 lanes."""
    fits = []
    if all(g == 0 for g in got):
        fits.append("zero")
    for k in range(-40, 64):
        for kind in ("wrap", "sat", "rnd_half_up"):
            exp = []
            for x in LANES8:
                if k >= 0:
                    v = x << k
                elif kind == "rnd_half_up":
                    v = (x + (1 << (-k - 1))) >> -k
                else:
                    v = x >> -k
                if kind == "sat":
                    v = sat(v, 32)
                else:
                    v = (v + (1 << 31)) % (1 << 32) - (1 << 31)
                exp.append(v)
            if exp == got:
                fits.append(f"k={k}/{kind}")
    return fits


@iron.jit
def srs_probe(a_in: In, c_out: Out):
    io_ty = np.ndarray[(N_IO,), np.dtype[np.int32]]
    fn = ExternalFunction(
        "srs_probe",
        source_file=str(_KERNEL_SRC),
        object_file_name=f"srs_kernels_lane{LANE}_m3.o",
        arg_types=[io_ty, io_ty],
        include_dirs=[config.cxx_header_path()],
    )
    of_in = ObjectFifo(io_ty, name="in")
    of_out = ObjectFifo(io_ty, name="out")

    def core_fn(of_in, of_out, k):
        e = of_in.acquire(1)
        o = of_out.acquire(1)
        k(e, o)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod(), fn], tile=Tile(0, 2))

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(sequence, [io_ty, io_ty, of_in.prod(tile=Tile(0, 0)), of_out.cons(tile=Tile(0, 0))])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def lanes_32(seed):
    edges = [0, 1, -1, 2, -2, 3, -3, 0x7FFFFFFF, -0x80000000, 0x40000000, -0x40000000,
             0x12345678, -0x12345678, 0x55555555, -0x55555556, 1 << 16, -(1 << 16), (1 << 31) - 2]
    rng = np.random.default_rng(seed)
    rest = rng.integers(-(1 << 31), 1 << 31, size=32 - len(edges), dtype=np.int64).tolist()
    return [int(x) for x in edges + rest]


def lanes_16(seed):
    edges = [0, 1, -1, 0x7FFFFFFF, -0x80000000, 0x12345678, -0x12345678, 3]
    rng = np.random.default_rng(seed + 1)
    rest = rng.integers(-(1 << 31), 1 << 31, size=16 - len(edges), dtype=np.int64).tolist()
    return [int(x) for x in edges + rest]


def candidates(s):
    c = {"and31": s & 31, "and63": s & 63, "and127": s & 127,
         "clamp31": min(max(s, 0), 31), "clamp63": min(max(s, 0), 63)}
    if s >= 0:
        c["as_is"] = s
    # 6-bit code, 0..59 shift right, 60..63 shift LEFT by 64 - code (i.e. range -4..59).
    # Added after run 3, which showed codes 60..63 multiplying by 16/8/4/2.
    code = s & 63
    c["f6_m4_59"] = code if code < 60 else code - 64
    return c


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo if v < lo else hi if v > hi else v


def run_point(a, c, mode, shift, up, vals):
    words = [0] * N_IO
    words[0], words[1], words[2] = mode, shift, up
    for i, v in enumerate(vals):
        words[LANE + i] = int(np.int64(v).astype(np.int32))
    for i, w in enumerate(words):
        a[i] = w
    srs_probe(a, c)
    out = c.numpy().astype(np.int64)
    return out


def platform_idle():
    try:
        r = subprocess.run([_XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"(xrt-smi not captured: {e})"


def parse_range(s):
    lo, hi = s.split(":")
    return list(range(int(lo), int(hi) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shifts", default="-8:72", help="inclusive range lo:hi of runtime shifts")
    ap.add_argument("--modes", default="0,1,2")
    ap.add_argument("--ups", default="0,31", help="mode 2 upshifts (acc64 lanes = int32 << up)")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--lanes16", help="explicit comma-separated int32 lanes for mode 2 (replaces the seeded set)")
    ap.add_argument("--require-idle", action="store_true", default=True)
    args = ap.parse_args()

    pre = platform_idle()
    print("== xrt-smi examine -r aie-partitions (before)\n" + pre.strip())
    if args.require_idle and "No hardware contexts running" not in pre:
        print("NPU not idle: refusing to measure (a foreign context is contention, not a finding)")
        return 2

    shifts = parse_range(args.shifts)
    a = iron.zeros(N_IO, dtype=np.int32, device="npu")
    c = iron.zeros_like(a)
    verdict_ok = True
    for mode in (int(m) for m in args.modes.split(",")):
        name, obits, nl = MODES[mode]
        if mode == 3:
            print(f"\n== mode 3 {name}: int8 lanes {LANES8}, {len(shifts)} shift scalars")
            for s in shifts:
                out = run_point(a, c, 3, s, 0, LANES8)
                if out[0] != 3 or out[1] != s:
                    print(f"  shift {s}: ECHO MISMATCH {out[:4].tolist()}")
                    verdict_ok = False
                    continue
                got = out[LANE:LANE + 32].tolist()
                fits = ups_fit(got)
                print(f"  shift {s:4d} (0x{s & 0xffffffff:08x}): fits {', '.join(fits[:6]) or 'NONE'}"
                      + (f" (+{len(fits) - 6})" if len(fits) > 6 else ""))
                if not fits:
                    print(f"     lanes: {got}")
            continue
        ups = [int(u) for u in args.ups.split(",")] if mode == 2 else [0]
        vals = lanes_16(args.seed) if mode == 2 else lanes_32(args.seed)
        if mode == 2 and args.lanes16:
            vals = [int(x) for x in args.lanes16.split(",")]
            assert len(vals) == 16, "--lanes16 needs exactly 16 values"
        survivors = None
        first_fail = {}
        print(f"\n== mode {mode} {name}, ups {ups}, {len(shifts)} shifts, {nl} lanes, floor + saturate")
        for up in ups:
            accs = [v << up for v in vals]
            for s in shifts:
                t0 = time.perf_counter()
                out = run_point(a, c, mode, s, up, vals)
                ms = (time.perf_counter() - t0) * 1e3
                if out[0] != mode or out[1] != s:
                    print(f"  up {up} shift {s}: ECHO MISMATCH {out[:4].tolist()}")
                    verdict_ok = False
                    continue
                got = out[LANE:LANE + nl].tolist()
                match = []
                for cname, k in candidates(s).items():
                    exp = [sat(acc >> k if k >= 0 else acc << -k, obits) for acc in accs]
                    if exp == got:
                        match.append(cname)
                    elif cname not in first_fail:
                        bad = next(i for i in range(nl) if exp[i] != got[i])
                        first_fail[cname] = (up, s, bad, accs[bad], exp[bad], got[bad])
                ms_txt = f"{ms:.2f} ms"
                print(f"  up {up:2d} shift {s:3d}: rnd {out[2]} sat {out[3]}  matches {','.join(match) or 'NONE'}  ({ms_txt})")
                if not match:
                    print(f"     lanes: {got}")
                    print(f"     accs:  {accs}")
                survivors = set(match) if survivors is None else survivors & set(match)
        print(f"-- mode {mode} {name}: candidates matching EVERY lane at EVERY shift: {sorted(survivors) or 'NONE'}")
        for cname, (up, s, i, acc, e, g) in sorted(first_fail.items()):
            print(f"     {cname:8s} first fails at up {up} shift {s} lane {i}: acc {acc} expected {e} got {g}")
    post = platform_idle()
    print("\n== xrt-smi examine -r aie-partitions (after)\n" + post.strip())
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
