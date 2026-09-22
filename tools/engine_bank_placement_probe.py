"""Can the int8 engine's core buffers be steered out of a shared bank without touching the kernel?

tools/aie_bank_check.py found that every core of every engine build places the second activation
buffer in the bank that holds the first weight buffer, and the stride-1 dual loop pairs a weight
load with an activation load four times per iteration - the shape the measured same-bank penalty
applies to. This asks whether IRON's two placement levers move it: the allocation scheme
(`Worker(allocation_scheme=...)`, bank-aware by default) and a pinned address on the raw
`psum`/`scratch` buffers (`Buffer(address=...)`). design.py is not edited; the levers are applied
by wrapping `Worker` and `Buffer` for the duration of one synthetic-design compile
(tests/test_conv_engine.py::compile_engine), and the placed MLIR is read back.

Compile only: it builds an xclbin it never runs, so it establishes a placement and no latency.

The readback half - `read_placement` / `report_placement` - is a pure function of a placed
MLIR module and knows nothing about which design produced it, so `--placed` reports any
build made through `compile_mlir_module`, including the bf16 engine's.

    bash scripts/research-iron.sh tools/engine_bank_placement_probe.py --scheme bank-aware
    bash scripts/research-iron.sh tools/engine_bank_placement_probe.py --scheme basic-sequential
    bash scripts/research-iron.sh tools/engine_bank_placement_probe.py --scheme bank-aware --psum-address 49152
    python tools/engine_bank_placement_probe.py --placed scratch/bf16_build/design.prj/input_with_addresses.mlir
"""
from __future__ import annotations

import argparse
import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

BANK = 16 * 1024
TILE_L1 = 64 * 1024      # one core tile's data memory
BUF = re.compile(r'aie\.buffer\(%tile_(\d)_(\d)\) \{address = (\d+) : i32(?:, mem_bank = (\d+) : i32)?, sym_name = "([^"]+)"\} : memref<(\d+)x(\w+)>')
WIDTH = {"i8": 1, "ui8": 1, "i32": 4, "f32": 4, "bf16": 2, "i16": 2}


def read_placement(placed_mlir: Path, core: str):
    """Placed L1 buffers of one core, as sorted (lo, hi, name, size, bank_lo, bank_hi) rows.

    Reads `design.prj/input_with_addresses.mlir`, which is written by any compile through
    `compile_mlir_module` - so this serves the int8 design, the bf16 design and anything
    else built the same way. It is a pure function of the file: no design is imported and
    nothing is compiled, which is what makes it shareable.
    """
    text = Path(placed_mlir).read_text(encoding="utf-8")
    rows = []
    for col, row, addr, _bank, name, n, ty in BUF.findall(text):
        if f"{col}_{row}" != core:
            continue
        size = int(n) * WIDTH.get(ty, 1)
        lo = int(addr)
        hi = lo + size - 1
        rows.append((lo, hi, name, size, lo // BANK, hi // BANK))
    rows.sort()
    return rows


def report_placement(rows) -> None:
    """Print one core's placement, then the line the L1 budget is actually decided by.

    The per-buffer sum is NOT the number that decides whether a design fits. The bank-aware
    allocator leaves holes between buffers, so what has to clear 65,536 B is the placed
    EXTENT, and on the int8 engine the two differ by several kilobytes - enough to turn a
    budget that looks clear into a build that does not place.

    The span below the first buffer is reported separately because it is the STACK, not a
    hole: the linker puts it at address 0 and `Worker(stack_size=...)` sizes it. Counting it
    as fragmentation would overstate the holes by exactly the stack on every design, and
    would make two designs with different stack sizes look differently fragmented when they
    are not.
    """
    for lo, hi, name, size, b0, b1 in rows:
        span = f"bank {b0}" if b0 == b1 else f"banks {b0}-{b1}"
        print(f"  {lo:6d}-{hi:6d}  {size:6d} B  {span:9s}  {name}")
    if not rows:
        print("  (no buffers placed on this core)")
        return
    total = sum(r[3] for r in rows)
    extent = max(r[1] for r in rows) + 1
    stack = min(r[0] for r in rows)
    print(f"EXTENT {extent} B placed of {TILE_L1} B "
          f"(buffers {total} B, stack {stack} B, holes {extent - total - stack} B, "
          f"{TILE_L1 - extent} B free)", flush=True)


TRIALS = [  # (scheme, psum pin, scratch pin)
    ("bank-aware", None, None),          # the shipped placement
    ("basic-sequential", None, None),
    ("bank-aware", 49152, None),         # psum into bank 3
    ("bank-aware", 32768, None),         # psum into bank 2
    ("bank-aware", 49152, 25856),        # and scratch beside the second weight buffer, so bank 1 is too full for a1
]


def trial(scheme, psum_address, scratch_address, core, build_dir=None) -> int:
    from kernels.aie2.conv_engine import design as eng
    from tests import test_conv_engine as tce

    real_worker, real_buffer = eng.Worker, eng.Buffer

    def worker(*a, **k):
        k.setdefault("allocation_scheme", scheme)
        return real_worker(*a, **k)

    def buffer(*a, **k):
        name = k.get("name", "")
        if psum_address is not None and name.startswith("psum"):
            k["address"] = psum_address
        if scratch_address is not None and name.startswith("scratch"):
            k["address"] = scratch_address
        return real_buffer(*a, **k)

    eng.Worker, eng.Buffer = worker, buffer
    tag = scheme + (f"_psum{psum_address}" if psum_address is not None else "") \
        + (f"_scratch{scratch_address}" if scratch_address is not None else "")
    build = Path(build_dir) if build_dir else ROOT / "scratch" / "bank_placement" / tag
    print(f"PLACEMENT scheme={scheme} core={core} psum_pin={psum_address} scratch_pin={scratch_address}", flush=True)
    try:
        tce.compile_engine(build, seed=0)
    except RuntimeError as e:
        first = next((ln for ln in str(e).splitlines() if "error" in ln.lower() or "Failed" in ln), str(e)[:200])
        print(f"  COMPILE FAILED: {first.split('design.prj')[-1].strip()}")
        print("WEIGHT_ACTIVATION_SHARED_BANKS not placed")
        return 1
    finally:
        eng.Worker, eng.Buffer = real_worker, real_buffer

    rows = read_placement(build / "design.prj" / "input_with_addresses.mlir", core)
    report_placement(rows)
    shared_banks(rows)
    return 0


def shared_banks(rows) -> None:
    by_bank: dict[int, set[str]] = {}
    for lo, hi, name, size, b0, b1 in rows:
        base = re.sub(r"_buff_\d+$", "", name)
        for b in range(b0, b1 + 1):
            by_bank.setdefault(b, set()).add(base)
    shared = {b: sorted(s) for b, s in by_bank.items() if any(n.startswith("w") for n in s) and any(n.startswith("a") for n in s)}
    print(f"WEIGHT_ACTIVATION_SHARED_BANKS {shared if shared else 'none'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=["bank-aware", "basic-sequential"], default="bank-aware")
    ap.add_argument("--psum-address", type=int, default=None, help="pin the psum buffer here (bytes)")
    ap.add_argument("--scratch-address", type=int, default=None, help="pin the scratch buffer here (bytes)")
    ap.add_argument("--all", action="store_true", help="run every trial in TRIALS instead of one configuration")
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--core", default="0_2")
    ap.add_argument("--placed", default=None,
                    help="report an existing design.prj/input_with_addresses.mlir instead of "
                         "compiling; the reader is design-agnostic, so this serves the bf16 "
                         "engine and anything else built through compile_mlir_module")
    args = ap.parse_args()

    if args.placed:
        # No IRON import and no compile: reading a placed module is a pure file operation.
        rows = read_placement(Path(args.placed), args.core)
        print(f"PLACEMENT core={args.core} from {args.placed}", flush=True)
        report_placement(rows)
        shared_banks(rows)
        return 0

    import aie.iron as iron  # noqa: F401 - the design needs the IRON runtime importable

    if args.all:
        # One process per trial: IRON registers the kernel once per process and reuses its object,
        # while compile_engine wipes the work directory, so a second compile in one process finds
        # no engine.o. Every trial's outcome is a finding; the exit code is not a verdict.
        import subprocess
        for scheme, psum_address, scratch_address in TRIALS:
            cmd = [sys.executable, __file__, "--scheme", scheme, "--core", args.core]
            if psum_address is not None:
                cmd += ["--psum-address", str(psum_address)]
            if scratch_address is not None:
                cmd += ["--scratch-address", str(scratch_address)]
            out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            for line in out.stdout.splitlines():
                if line.startswith(("PLACEMENT", "  ", "WEIGHT_ACTIVATION")):
                    print(line, flush=True)
        return 0
    return trial(args.scheme, args.psum_address, args.scratch_address, args.core, args.build_dir)


if __name__ == "__main__":
    raise SystemExit(main())
