"""W4A8 on one AIE2 core: does the int8 x int4 vmac compute right, and what does a call cost?

One Worker on Tile(0, 2) runs one GEMM-tile kernel from w4a8_kernels.cc per call. A
(M x K int8) and B (K x N: int8, or int4 packed two per byte) arrive by ObjectFifo, C
(M x N int32) leaves by one. The kernel wrapper zeroes C, then brackets exactly one
kernel call with event0/event1; the tile's trace unit stamps both, so a call's cost is
read straight off the tile timer in cycles -- clock_probe's method and its corrected
frame decoder -- with no DMA or lock inside the bracket and no dependence on the clock.

One (kernel, mode, M, K, N) per process. Each CompileTime set is its own xclbin, and two
designs in one process have returned wrong data on this machine before
(docs/DECISIONS.md); sweep.py runs the matrix, one fresh process per row.

Kernels (C functions in w4a8_kernels.cc):
  i8i8     mm_i8i8_local   upstream mm.cc's int8 loop re-typed locally -- the control.
                           Upstream's own matmul_i8_i32 fires event0/event1 itself, so it
                           cannot sit inside this wrapper's bracket.
  unpack   mm_i8i4_unpack  B stored as int4, widened to int8 on load, stock 4x8x8 mmul
  native   mm_i8i4_native  B stored as int4, consumed by mmul<4,16,8,int8,int4>
  *_2x2    the same with A expanded twice instead of four times
Modes: default (the k loop as upstream writes it, as IRON builds it), no-unroll, unroll2.

Verification. B is drawn in [-8, 7] for every kernel -- the int8 control included -- so
one int64 reference serves all of them. If an int4 kernel does not match, the output is
also compared against the other readings of the same packed bytes (nibbles swapped, B
unsigned, both); whichever matches is the finding, and none matching is one too.

Build check. After the first call the IRON-built kernel object is pulled from
~/.npu/cache and its kernel function compared, bundle for bundle, with static_probe's
compile of the same source and defines -- otherwise the static table does not describe
what ran.

Usage (ironenv, Desktop 2):
    python kernels/w4a8_probe/w4a8_probe.py --kernel native --mode unroll2 --K 64
    python kernels/w4a8_probe/w4a8_probe.py --kernel i8i8 --K 128 --json-out rows.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "clock_probe"))
sys.path.insert(0, str(_HERE))

import static_probe  # noqa: E402
from clock_probe import TraceReader  # noqa: E402

_SRC = _HERE / "w4a8_kernels.cc"

# name -> (C function, r, s, t, B packed as int4, MACs per vmac)
KERNELS = {
    "i8i8": ("mm_i8i8_local", 4, 8, 8, False, 256),
    "unpack": ("mm_i8i4_unpack", 4, 8, 8, True, 256),
    "native": ("mm_i8i4_native", 4, 16, 8, True, 512),
    "i8i8_2x2": ("mm_i8i8_local_2x2", 4, 8, 8, False, 256),
    "unpack_2x2": ("mm_i8i4_unpack_2x2", 4, 8, 8, True, 256),
    "native_2x2": ("mm_i8i4_native_2x2", 4, 16, 8, True, 512),
}


def source_rev() -> str:
    """@iron.jit keys its cache on the generator's bytecode and CompileTime args only, so
    an edit to the .cc alone would silently reuse a stale xclbin; this goes in as one."""
    h = hashlib.sha256()
    for p in (Path(__file__), _SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def kernel_defines(kernel, mode, M, K, N) -> list[str]:
    return [
        f"-DDIM_M={M}",
        f"-DDIM_K={K}",
        f"-DDIM_N={N}",
        f"-DRUN_KERNEL={KERNELS[kernel][0]}",
        *static_probe.MODES[mode],
    ]


def obj_name(kernel, mode, M, K, N) -> str:
    return f"w4a8_{kernel}_{mode}_{M}x{K}x{N}.o"


@iron.jit
def w4a8_probe(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    kernel: CompileTime[str],
    mode: CompileTime[str],
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    rev: CompileTime[str],
    stack: CompileTime[int] = 4096,
    trace_size: CompileTime[int] = 0,
):
    """One Worker on Tile(0, 2): A, B in, C out, every fifo single-buffered."""
    fn, r, s, t, packed, _ = KERNELS[kernel]
    a_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    b_ty = np.ndarray[((K * N) // 2 if packed else K * N,), np.dtype[np.int8]]
    c_ty = np.ndarray[(M * N,), np.dtype[np.int32]]
    run = ExternalFunction(
        "w4a8_run",
        source_file=str(_SRC),
        object_file_name=obj_name(kernel, mode, M, K, N),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=kernel_defines(kernel, mode, M, K, N),
    )
    # depth=1 throughout: at K=256 A+B+C is 48 KB single-buffered and would not fit
    # double-buffered next to the stack.
    of_a = ObjectFifo(a_ty, depth=1, name="a")
    of_b = ObjectFifo(b_ty, depth=1, name="b")
    of_c = ObjectFifo(c_ty, depth=1, name="c")

    def core_fn(of_a, of_b, of_c, run):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        run(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_a.cons(), of_b.cons(), of_c.prod(), run],
        tile=Tile(0, 2),
        stack_size=stack,
        trace=1 if trace_size > 0 else None,
    )

    def sequence(a, b, c, a_h, b_h, c_h):
        a_h.fill(a)
        b_h.fill(b)
        c_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            a_ty,
            b_ty,
            c_ty,
            of_a.prod(tile=Tile(0, 0)),
            of_b.prod(tile=Tile(0, 0)),
            of_c.cons(tile=Tile(0, 0)),
        ],
    )
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_size > 0:
        prog.enable_trace(
            trace_size=trace_size,
            workers=[worker],
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
        )
    return prog.resolve_program()


# ------------------------------------------------------------------- host layout


def tile(x: np.ndarray, r: int, c: int) -> np.ndarray:
    """(R x C) -> r x c tiles, each row-major, tiles in row-major order: mm.cc's layout."""
    R, C = x.shape
    return x.reshape(R // r, r, C // c, c).transpose(0, 2, 1, 3).reshape(-1)


def untile(flat: np.ndarray, R: int, C: int, r: int, c: int) -> np.ndarray:
    return flat.reshape(R // r, C // c, r, c).transpose(0, 2, 1, 3).reshape(R, C)


def pack_int4(v: np.ndarray) -> np.ndarray:
    """Two int4 per byte, element 2i in the LOW nibble -- the convention under test."""
    u = v.astype(np.int16) & 0xF
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8).view(np.int8)


def read_int4(b: np.ndarray, swap: bool, signed: bool) -> np.ndarray:
    """The values a reader of packed bytes `b` would see under one convention."""
    u = b.view(np.uint8).astype(np.int16)
    lo, hi = u & 0xF, u >> 4
    if swap:
        lo, hi = hi, lo
    out = np.empty(2 * u.size, dtype=np.int16)
    out[0::2], out[1::2] = lo, hi
    if signed:
        out = np.where(out >= 8, out - 16, out)
    return out


READINGS = {
    "low nibble first, signed": (False, True),
    "high nibble first, signed": (True, True),
    "low nibble first, unsigned": (False, False),
    "high nibble first, unsigned": (True, False),
}


def make_inputs(kernel, M, K, N, seed):
    fn, r, s, t, packed, _ = KERNELS[kernel]
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int64)
    B = rng.integers(-8, 8, size=(K, N), dtype=np.int64)
    # Pin both extremes of each range into every run.
    A[0, 0], A[0, 1], B[0, 0], B[0, 1] = -128, 127, -8, 7
    a_flat = tile(A, r, s).astype(np.int8)
    b_tiled = tile(B, s, t)
    b_flat = pack_int4(b_tiled) if packed else b_tiled.astype(np.int8)
    return A, B, a_flat, b_flat


def verify(kernel, M, K, N, A, B, b_flat, c_flat):
    fn, r, s, t, packed, _ = KERNELS[kernel]
    got = untile(np.asarray(c_flat, dtype=np.int64), M, N, r, t)
    ref = A @ B
    n_bad = int(np.count_nonzero(got != ref))
    res = {"n": int(ref.size), "mismatches": n_bad, "matches_reading": None}
    if n_bad == 0:
        res["matches_reading"] = "low nibble first, signed" if packed else "int8"
        return res
    if packed:
        for name, (swap, signed) in READINGS.items():
            b_alt = untile(read_int4(b_flat, swap, signed).astype(np.int64), K, N, s, t)
            if np.array_equal(got, A @ b_alt):
                res["matches_reading"] = name
                break
    res["first_bad"] = [
        [int(i), int(j), int(got[i, j]), int(ref[i, j])]
        for i, j in zip(*np.nonzero(got != ref))
    ][:5]
    return res


# ------------------------------------------------------------------- build check


def build_check(kernel, mode, M, K, N, since: float):
    """Compare the kernel function in IRON's built object with static_probe's compile."""
    name = obj_name(kernel, mode, M, K, N)
    cands = [
        p
        for p in glob.glob(os.path.expanduser(f"~/.npu/cache/*/{name}"))
        if os.path.getmtime(p) >= since - 3600
    ] or glob.glob(os.path.expanduser(f"~/.npu/cache/*/{name}"))
    if not cands:
        return {"status": "no IRON object found", "cache_dir": None}
    built = max(cands, key=os.path.getmtime)
    fn = KERNELS[kernel][0]
    objdump = static_probe.aie_disasm.find_objdump(None)
    tmp = os.path.join(tempfile.mkdtemp(prefix="w4a8_bc_"), "static.o")
    static_probe.compile_obj(str(_SRC), kernel_defines(kernel, mode, M, K, N), tmp)

    def body(obj):
        secs = static_probe.aie_disasm.parse(static_probe.aie_disasm.disassemble(obj, objdump))
        s = static_probe.section_for(secs, fn)
        return None if s is None else [" ; ".join(b.fields) for b in s.bundles]

    a, b = body(built), body(tmp)
    rec = static_probe.analyse(tmp, objdump, [fn])[fn]
    mac_loops = [lp for lp in rec["loops"] if lp["macs"]]
    lp = max(mac_loops, key=lambda d: d["macs"]) if mac_loops else None
    return {
        "status": "identical" if a is not None and a == b else "DIFFERENT",
        "cache_dir": os.path.basename(os.path.dirname(built)),
        "bundles": len(b) if b else None,
        "loop_bundles": lp["bundles"] if lp else None,
        "loop_vmacs": lp["macs"] if lp else None,
        "fn_stack_refs": rec["stack"],
        "frame": rec["frame"],
    }


# ------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    ap.add_argument("--mode", default="default", choices=sorted(static_probe.MODES))
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20, help="timed calls")
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls after the first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stack", type=int, default=4096, help="core stack bytes (constant across arms)")
    ap.add_argument("--trace-size", type=int, default=65536)
    ap.add_argument("--tag", default="", help="free text carried into the JSON row")
    ap.add_argument("--json-out", help="append one JSON row here")
    args = ap.parse_args(argv)

    kernel, mode, M, K, N = args.kernel, args.mode, args.M, args.K, args.N
    fn, r, s, t, packed, macs_per_vmac = KERNELS[kernel]
    rev = source_rev()
    print(f"w4a8_probe {kernel} ({fn}) mode={mode} M/K/N={M}/{K}/{N} rev={rev} "
          f"stack={args.stack} host={socket.gethostname()}")

    A, B, a_flat, b_flat = make_inputs(kernel, M, K, N, args.seed)
    A_t = iron.tensor(a_flat, dtype=np.int8, device="npu")
    B_t = iron.tensor(b_flat, dtype=np.int8, device="npu")
    C_t = iron.zeros(M * N, dtype=np.int32, device="npu")

    trace_file = os.path.join(tempfile.gettempdir(), f"w4a8_trace_{os.getpid()}.txt")
    tc = TraceConfig(trace_size=args.trace_size, trace_file=trace_file)
    w4a8_probe.trace_config = tc
    reader = TraceReader(tc)

    def call():
        t0 = time.perf_counter()
        res = w4a8_probe(
            A_t, B_t, C_t, kernel=kernel, mode=mode, M=M, K=K, N=N, rev=rev,
            stack=args.stack, trace_size=args.trace_size,
        )
        wall_ms = (time.perf_counter() - t0) * 1e3
        t_e0, t_e1, pairs, n_ev = reader.stamps()
        cyc = (t_e1 - t_e0) if t_e0 is not None else None
        npu_ns = getattr(res, "npu_time", None)
        return cyc, wall_ms, (npu_ns / 1e3 if isinstance(npu_ns, (int, float)) else None)

    t_first = time.time()
    cyc0, wall0, _ = call()
    ver = verify(kernel, M, K, N, A, B, b_flat, C_t.numpy())
    print(f"first call {wall0 / 1e3:.2f} s (compile or cache), cycles {cyc0}, "
          f"traced tile row {reader.tile[0] if reader.tile else '?'} col "
          f"{reader.tile[1] if reader.tile else '?'}")
    print(f"verify: {ver['mismatches']} of {ver['n']} mismatches; "
          f"output matches reading: {ver['matches_reading']}")
    if ver["mismatches"]:
        print(f"  first mismatches [i, j, got, ref]: {ver.get('first_bad')}")

    bc = build_check(kernel, mode, M, K, N, t_first)
    print(f"build check: IRON object vs static compile of {fn}: {bc['status']} "
          f"(cache {bc['cache_dir']}; loop {bc.get('loop_bundles')} bundles, "
          f"{bc.get('loop_vmacs')} vmacs)")

    for _ in range(args.warmup):
        call()
    cycles, walls, npus = [], [], []
    for _ in range(args.iters):
        c, w, h = call()
        cycles.append(c)
        walls.append(w)
        npus.append(h)
    # Re-verify on the last timed call's output, so a timed call is also a checked one.
    ver_last = verify(kernel, M, K, N, A, B, b_flat, C_t.numpy())

    good = [c for c in cycles if c is not None]
    macs = M * K * N
    vmacs = macs // macs_per_vmac
    med = statistics.median(good) if good else None
    print(f"cycles per call over {len(good)} of {len(cycles)} traced calls: "
          f"median {med}, min {min(good) if good else None}, max {max(good) if good else None}")
    if med:
        print(f"  {macs} MACs, {vmacs} vmacs per call -> {macs / med:.1f} MAC/cycle, "
              f"{vmacs / med:.3f} vmac/cycle (whole call: loop, C loads/stores, prologue)")
        lb, lv = bc.get("loop_bundles"), bc.get("loop_vmacs")
        if lb and lv:
            pred = vmacs / lv * lb
            print(f"  static steady-state loop alone would take {pred:.0f} cycles "
                  f"({vmacs / lv:.0f} executions x {lb} bundles); measured - that = {med - pred:.0f}")
    print(f"last timed call re-verified: {ver_last['mismatches']} mismatches")

    row = {
        "kind": "run",
        "host": socket.gethostname(),
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "kernel": kernel,
        "fn": fn,
        "mode": mode,
        "M": M, "K": K, "N": N,
        "rev": rev,
        "stack": args.stack,
        "seed": args.seed,
        "macs": macs,
        "vmacs": vmacs,
        "verify": ver,
        "verify_last": {"mismatches": ver_last["mismatches"]},
        "build_check": bc,
        "tile": list(reader.tile) if reader.tile else None,
        "cycles": cycles,
        "cycles_median": med,
        "wall_ms": walls,
        "npu_us": npus,
    }
    if args.json_out:
        with open(args.json_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    ok = ver["mismatches"] == 0 and ver_last["mismatches"] == 0 and bool(good)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
