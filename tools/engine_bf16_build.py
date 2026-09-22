"""Build the sixteen-core bf16 engine xclbin, and report what it cost in L1.

This is the first time a bf16 design has been built at all. Every bf16 number in the repo
until now came from one core of sixteen (kernels/bf16_conv/engine_bf16.py) or from a CPU
cast, so the question this answers has never been asked of the hardware: do sixteen cores'
worth of bf16 buffers place inside 65,536 B of tile data memory?

The arithmetic says yes with room:

    weights   9,472 x2 = 18,944      (unchanged from int8 - the packet is the same size)
    acts     12,800 x2 = 25,600      (bf16 doubles it)
    outputs   3,200 x2 =  6,400
    psum               =  6,400      (int8 carries 16,000, including a held tile in its tail)
    scratch            =  3,200      (the held tile, now its own buffer, which is what fits)
    stack              =  2,048
                          ------
                          62,592 of 65,536

but the sum is not what decides it. The bank-aware allocator leaves holes, and on the int8
engine it places out to 65,023 against a 59,392 B sum - 5.6 KB of fragmentation. So the
compile is the measurement and the sum is only a prediction, which is why this prints the
placed extent for every core rather than asserting on the total.

No dispatch is emitted by default. L1 placement is a function of the core programs and
their buffers, not of what the shim sends, so an allocation probe does not need a schedule -
and the bf16 schedule does not exist yet. `--sequence` is left as the hook for when it does.

    bash scripts/research-iron.sh tools/engine_bf16_build.py
    bash scripts/research-iron.sh tools/engine_bf16_build.py --build-dir scratch/bf16_build
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from tools.engine_bank_placement_probe import read_placement, report_placement  # noqa: E402


def empty_sequence(ws, wp):
    """No shim transfers.

    An xclbin built this way holds the same sixteen core programs and the same L1 allocation
    as one built with a full instruction stream; only the runtime sequence differs, and the
    sequence lives in insts.bin rather than in the placement. It is a real build, not a mock,
    and it cannot be run to produce a number - which is the point: it establishes a placement
    and claims nothing about latency.
    """
    return None


def build(build_dir: Path, w_depth: int = 2) -> dict:
    import aie.iron as iron
    from aie.iron.device import NPU1
    from aie.utils.compile.utils import compile_mlir_module

    from kernels.bf16_conv import design as eng

    t0 = time.perf_counter()
    iron.set_current_device(NPU1())
    program = eng.build_program(iron.get_current_device(), empty_sequence, w_depth=w_depth)
    module = program.resolve_program()

    build_dir.mkdir(parents=True, exist_ok=True)
    work = build_dir / "design.prj"
    # IRON reuses an existing engine_bf16.o; a stale one would silently ship an old kernel.
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (build_dir / "design.mlir").write_text(str(module), encoding="utf-8")

    xclbin_path = build_dir / "engine_bf16.xclbin"
    insts_path = build_dir / "insts.bin"
    compile_mlir_module(module, insts_path=insts_path, xclbin_path=xclbin_path,
                        work_dir=work, device=iron.get_current_device())

    obj = work / "engine_bf16.o"
    return {
        "seconds": time.perf_counter() - t0,
        "xclbin": xclbin_path,
        "xclbin_sha256": hashlib.sha256(xclbin_path.read_bytes()).hexdigest(),
        "insts_bytes": insts_path.stat().st_size,
        "kernel_source_sha256": hashlib.sha256(eng.KERNEL_SOURCE.read_bytes()).hexdigest(),
        "kernel_object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest() if obj.exists() else None,
        "kernel_object_bytes": obj.stat().st_size if obj.exists() else None,
        "placed_mlir": work / "input_with_addresses.mlir",
        "budget_sum": (eng.W_BYTES * w_depth + eng.A_BYTES * 2 + eng.O_BYTES * 2
                       + eng.PSUM_BYTES + eng.O_BYTES + 2048),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(ROOT / "scratch" / "bf16_build"))
    ap.add_argument("--w-depth", type=int, default=2,
                    help="weight fifo depth; 1 frees 9,472 B of L1 and costs weight "
                         "double-buffering on every layer, so it is a latency decision")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    print(f"BUILD bf16 16-core design -> {build_dir}  w_depth={args.w_depth}", flush=True)
    try:
        info = build(build_dir, w_depth=args.w_depth)
    except Exception as e:                       # noqa: BLE001 - the failure IS the result
        first = next((ln for ln in str(e).splitlines()
                      if "error" in ln.lower() or "Failed" in ln), str(e)[:400])
        print(f"BUILD FAILED: {first.strip()}")
        print("PLACEMENT not established")
        return 1

    print(f"BUILD ok in {info['seconds']:.1f} s")
    print(f"  xclbin        {info['xclbin'].name}  sha256 {info['xclbin_sha256'][:16]}")
    print(f"  insts.bin     {info['insts_bytes']} B")
    print(f"  kernel source sha256 {info['kernel_source_sha256'][:16]}")
    print(f"  kernel object sha256 {str(info['kernel_object_sha256'])[:16]}  "
          f"{info['kernel_object_bytes']} B")
    print(f"  predicted per-core sum {info['budget_sum']} B", flush=True)

    worst = 0
    for col in range(4):
        for row in range(2, 6):
            core = f"{col}_{row}"
            rows = read_placement(info["placed_mlir"], core)
            print(f"PLACEMENT core={core}")
            report_placement(rows)
            if rows:
                worst = max(worst, max(r[1] for r in rows) + 1)
    print(f"WORST_CORE_EXTENT {worst} B of 65536", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
