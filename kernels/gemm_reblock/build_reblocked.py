"""Build the int8 GEMM against a reblocked mm.cc, without touching the shared toolchain.

H11, proposed twice in this repo and never run. `mm.cc` dispatches int8->int32 to
`matmul_vectorized_4x2_mmul`, which holds EIGHT live 4x8x8 accumulators. Five is this
machine's measured spill-free ceiling, and the resulting object spills twelve vector
values into a 416-byte frame — while the bf16 path, at the same total accumulator
width, spills nothing (`results/aie/accumulator_width_vs_count.log`). The 2x2 template
already in `mm.cc` holds four. This builds the int8 design against that instead.

WHY IT IS DONE THIS WAY. `kernels.mm()` takes no source override: the path comes from
`_default_source_path("mm.cc")`, which resolves into the *installed* toolchain at
`ironenv/Lib/site-packages/mlir_aie/include/aie_kernels/aie2/mm.cc`. That file is shared
with every other session on this machine, so it is not edited. Instead this script
redirects the one lookup, in this process only, to the local copy in this directory.
Nothing outside this repo is modified.

Usage (ironenv, XRT SDK on PATH and pyxrt on PYTHONPATH):
    python kernels/gemm_reblock/build_reblocked.py                # build + verify + time
    python kernels/gemm_reblock/build_reblocked.py --stock        # same, upstream blocking
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_MM = HERE / "aie2" / "mm_2x2.cc"
DESIGN = HERE.parent / "bank_placement" / "whole_array_bankpad.py"


def redirect_mm_source() -> None:
    """Point `mm.cc` lookups at the local reblocked copy, in this process only."""
    from aie.iron.kernels import _common, linalg

    original = _common._default_source_path

    def patched(filename, subdir=None):
        if filename == "mm.cc":
            return LOCAL_MM
        return original(filename, subdir)

    _common._default_source_path = patched
    linalg._default_source_path = patched
    print(f"mm.cc redirected -> {LOCAL_MM}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stock", action="store_true",
                    help="do NOT redirect; build against the installed mm.cc")
    ap.add_argument("--cache-home", required=True,
                    help="private NPU_CACHE_HOME for this arm. REQUIRED, and the two arms "
                         "must differ: @iron.jit keys its cache on the design and its "
                         "compile-time arguments, NOT on the kernel source, so a reblocked "
                         "build with the same arguments silently reuses the stock arm's "
                         "whole cached xclbin and never recompiles the kernel. That is not "
                         "hypothetical -- it happened on the first attempt here, and the "
                         "only symptom was that no new .o appeared on disk.")
    ap.add_argument("--rest", nargs=argparse.REMAINDER,
                    help="arguments forwarded to the whole_array design")
    args = ap.parse_args()

    # Must be set before anything imports aie.utils.compile, which reads it at import.
    os.environ["NPU_CACHE_HOME"] = str(Path(args.cache_home).resolve())
    print(f"NPU_CACHE_HOME = {os.environ['NPU_CACHE_HOME']}")

    if not args.stock:
        if not LOCAL_MM.exists():
            raise SystemExit(f"missing {LOCAL_MM}")
        redirect_mm_source()
    else:
        print("building against the INSTALLED mm.cc (stock 4x2 blocking)")

    forwarded = args.rest or []
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        forwarded = [
            "--dev", "npu", "-M", "4096", "-K", "2048", "-N", "2048",
            "-m", "64", "-k", "64", "-n", "64", "--n-aie-cols", "4",
            "--dtype_in", "i8", "--dtype_out", "i32",
            "--warmup", "3", "--iters", "10",
        ]
    sys.argv = [str(DESIGN)] + forwarded
    print("design:", DESIGN)
    print("argv:  ", " ".join(forwarded))
    print()
    runpy.run_path(str(DESIGN), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
