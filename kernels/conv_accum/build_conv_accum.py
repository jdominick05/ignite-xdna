"""Build the int8 bottleneck against a conv2dk1.cc whose accumulators are register-resident.

THE DEFECT
----------
`results/aie/conv_issue_rate_decomposed.log` located the open int8 conv's 11.3x gap to the
vendor DPU: it is issue rate, and the 1x1's hot loop is the worst of it at **0.045 MACs per
cycle** against the int8 GEMM's 0.889. The cause is one line of C++.
`conv2dk1_i8_vector` declares

    MMUL4x8x8 acc_tmp[4];
    for (int x = 0; x < n; x++) acc_tmp[x].mac(in_a, in_b);

where `n` is a RUNTIME value. Registers cannot be dynamically addressed, so the array is
forced to memory and every `.mac()` becomes load-four-quarters / mac / store-four-quarters.
That is the 22-bundle loop that log disassembled: one `vmac`, eight quarter-accesses around
it, six idle bundles.

THE CHANGE
----------
`aie2/conv2dk1.cc` here peels the `n == 4` case into four NAMED accumulators -- the pattern
`mm.cc`'s `matmul_vectorized_2x2_mmul` already uses (C00, C01, C10, C11). The array loop is
kept as the tail, and is genuinely reached for widths whose `total_chunks = iw/4` is not a
multiple of 4 (iw=36 gives 9 chunks, iw=40 gives 10).

THE PREDICTION, AND IT IS FALSIFIABLE
-------------------------------------
H11 improved the int8 GEMM's kernel by 12.5% -- every spill gone, a full MAC every cycle --
and the wall clock did not move, because that design is bound by a per-buffer delivery floor
(`results/aie/gemm_reblock_h11_npu.log`). The conv is at 4.5% of peak rather than 41.9%, so
it should be genuinely issue-bound and should actually speed up. If it does NOT, the conv is
delivery-bound too, and the op class stays closed for a more interesting reason than anyone
has recorded.

WHY IT IS BUILT THIS WAY
------------------------
`kernels.conv2dk1()` takes no source override: the path comes from
`_default_source_path("conv2dk1.cc")`, which resolves into the *installed* toolchain, shared
with every other session on this machine. So that file is not edited -- this script redirects
the one lookup, in this process only.

Usage (ironenv, XRT SDK on PATH and pyxrt on PYTHONPATH):
    python kernels/conv_accum/build_conv_accum.py --cache-home <dir> -- <sweep args>
    python kernels/conv_accum/build_conv_accum.py --stock --cache-home <other dir> -- ...
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_DIR = HERE / "aie2"
# Both 1x1 stages of the bottleneck carry the same defect. conv2dk1.cc is stage 1 and
# conv2dk1_skip.cc is stage 3; the design is a core-to-core pipeline, so its rate is set
# by the slowest stage and patching only one of them measures Amdahl, not the fix.
LOCAL_SOURCES = {"conv2dk1.cc", "conv2dk1_skip.cc"}
SWEEP = HERE.parent / "bottleneck_sweep" / "sweep.py"


def redirect_conv_source() -> None:
    """Point the 1x1 kernel lookups at the local copies, in this process only."""
    from aie.iron.kernels import _common, conv

    original = _common._default_source_path

    def patched(filename, subdir=None):
        if filename in LOCAL_SOURCES:
            local = LOCAL_DIR / filename
            if not local.exists():
                raise SystemExit(f"missing {local}")
            return local
        return original(filename, subdir)

    _common._default_source_path = patched
    conv._default_source_path = patched
    print(f"redirected -> {LOCAL_DIR}: {', '.join(sorted(LOCAL_SOURCES))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stock", action="store_true",
                    help="do NOT redirect; build against the installed conv2dk1.cc")
    ap.add_argument("--cache-home", required=True,
                    help="private NPU_CACHE_HOME for this arm. REQUIRED, and the two arms "
                         "must differ: @iron.jit keys its cache on the design and its "
                         "compile-time arguments, NOT on the kernel source, so the patched "
                         "build with the same arguments silently reuses the stock arm's "
                         "whole cached xclbin and never recompiles the kernel. That is not "
                         "hypothetical -- it happened to the reblocked GEMM in this repo, "
                         "and the only symptom was that no new .o appeared on disk.")
    ap.add_argument("--rest", nargs=argparse.REMAINDER,
                    help="arguments forwarded to bottleneck_sweep/sweep.py")
    args = ap.parse_args()

    # Must be set before anything imports aie.utils.compile, which reads it at import.
    os.environ["NPU_CACHE_HOME"] = str(Path(args.cache_home).resolve())
    print(f"NPU_CACHE_HOME = {os.environ['NPU_CACHE_HOME']}")

    if not args.stock:
        redirect_conv_source()
    else:
        print("building against the INSTALLED 1x1 kernels (stock, array accumulators)")

    forwarded = args.rest or []
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    sys.argv = [str(SWEEP)] + forwarded
    print("design:", SWEEP)
    print("argv:  ", " ".join(forwarded))
    print()
    runpy.run_path(str(SWEEP), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
