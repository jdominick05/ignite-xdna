# kernels/

Hand-written AIE kernels for this machine's XDNA1 NPU, built with the open-source
`Xilinx/mlir-aie` (IRON + Peano) toolchain rather than Quark/VitisAI EP. They exist to
reach data types the standard pipeline structurally cannot (bf16, int16 -- the X1
backend is XINT8 or nothing, see `docs/DECISIONS.md`), for the specific ops VitisAI EP
leaves on the CPU. Nothing here changes how `resnet50`/`yolov8n`/`yolov8n-pose` are
quantized or run; each kernel is a standalone, measured artifact with its own log
under [`results/aie/`](../results/aie/README.md).

These do **not** run in `resnet_env17`. They need the mlir-aie `ironenv` described in
`RESEARCH.md` ("Custom C++ XRT / hand-written AIE kernels"), activated from
PowerShell:

```powershell
. C:\Users\<user>\mlir-aie\iron_env.ps1
python kernels/groupnorm_bf16/groupnorm.py -d npu --L 301056
```

`aiecc`'s final link needs `xclbinutil`, which ships with the XRT SDK and is **not** in
`ironenv/Scripts`. Put that directory on PATH as well, or a build that has already done
all its real work dies at step 36/37 with `tool 'xclbinutil' not found`:

```powershell
$env:PATH = "C:\Xilinx\XRT\xrt_sdk\xrt;$env:PATH"
```

The first run of a given shape JIT-compiles (Peano for the C++ kernel, aiecc for the
array program) into `~/.npu/cache/<hash>/`; later runs of the same shape hit the cache.

| Kernel | Op it replaces | Verdict |
|---|---|---|
| `bf16_matmul_sweep/` | Nothing — the bf16 GEMM shape sweep | **Wins.** NPU 1.18×–1.78× over CPU bf16 once M/N ≥ 1024 |
| `int8_matmul_sweep/` | Nothing — the int8 GEMM sweep, with the CPU int8 GEMM baseline | **Loses at the default tile; wins 1.10×–1.83× at M ≥ 512, N ≥ 2048 with `n=64`**, a tile bf16 can't fit |
| `groupnorm_bf16/` | `InstanceNormalization` / `GroupNorm(32)` in `resnetv2_50x3_xint8.onnx` | Wins on 33/49 nodes standalone; **0/49 once the handoff is counted** |
| `attention_bf16/` | Multi-head attention in `mobilevit_xxs` | **Loses 71×–240×.** Numerically correct, badly written |
| `conv2x_baseline/` | Nothing — the CPU baseline for `ml/resnet/layers_conv2_x` | **CPU wins 6.3×–8.5×** like-for-like int8 |
| `bottleneck_sweep/` | Nothing — the spatial sweep conv2x asked for | **CPU wins 5.7×–12.75×.** The int8 conv op class is closed |
| `conv2dk3_widthfix/` | Nothing — an upstream mlir-aie bug fix | **Resolved.** Bit-exact at every width tested |
| `dispatch_floor/` | Nothing — measures the per-dispatch fixed cost itself | Hardware floor **169.8 µs**, wall floor through IRON **617.0 µs** |

Each kernel's own findings, warnings and retractions follow. They are prose rather than
table cells because several of them are corrections to what an earlier version of this
file said, and a correction buried in a cell is a correction nobody reads.

## `bf16_matmul_sweep/`

The **first genuine NPU win** found in this project, after conv (12.75×) and
mobile-vision attention (71–240×) both closed against the NPU.

`cpu_matmul_sweep.py` (torch bf16/fp32 on this machine's Zen4 cores) is the CPU bf16 GEMM
baseline this repo never had; swept against `whole_array.py` (4-column bf16, upstream,
unmodified). **At the single shape previously measured (512³), the NPU's 895 GFLOPS
actually loses to CPU bf16 (1100.6 GFLOPS)** on this machine — but NPU throughput climbs
with M/N (895 → ~1800–2070 GFLOPS) while CPU bf16 stays close to flat (1100–1360 GFLOPS).
Crossover around N=1024 at M=K=512; from there **NPU wins 1.18×–1.78×**, every number a
verified PASS against numpy.

**Diagnosed (2026-09-07): "`K ≥ 3072` fails, undiagnosed" was the wrong framing.** It
isn't a threshold — error starts continuously around K/k≈23 reduction steps and grows
smoothly to near-total by K/k=96, always a systematic ~10–13% undercount, never
NaN/garbage. **Root cause: the K-reduction loop accumulates into a buffer typed
`dtype_out`, not fp32** — with `--dtype_out bf16`, the running sum round-trips through
bf16's 8-bit mantissa every reduction step and swamps small increments once the sum's
magnitude grows. `aie::mmul` does accumulate one MAC to fp32, but that result is
rounded back to bf16 before the next step adds onto it. **Fix, and it's free:
`--dtype_out f32`** — clean PASS at K=2880 and K=4096 on `whole_array`, 1741.7 and
1830.2 GFLOPS, same range as the bf16-output numbers above. K was never the ceiling;
the output dtype was. See `results/aie/bf16_matmul_k_limit_diagnosed_npu.log`. This is
the "LLM-scale, not mobile-vision" shape `attention_bf16/README.md`'s own math
predicted would be needed — a real fused attention block still needs its own check for
the same accumulate-in-`dtype_out` pattern, unchecked so far.

`results/aie/bf16_matmul_niche_npu.log`, `docs/DECISIONS.md`.

## `int8_matmul_sweep/`

The **second NPU win**, and the sharper test: on this machine the CPU is relatively
strongest (AVX-512 VNNI) at exactly the dtype the NPU is sold on. `npu_matmul_sweep.py`
drives the same upstream `whole_array.py` in int8 (`--dtype_in i8 --dtype_out i32` — i32
because the K reduction accumulates in `dtype_out`, the bf16 lesson above) and bf16→f32 in
the same sitting; `cpu_int8_matmul_sweep.py` (resnet_env) times **two** CPU int8 kernels,
torch `_int_mm` and ORT `MatMulInteger` u8s8 (MLAS), plus torch bf16/fp32. torch's is
1.05–1.68× faster and is the verdict line. Every int8 NPU row is bit-exact.

**At the default tile (m=64/k=64/n=32) the NPU's headline dtype loses** to the CPU's own
int8 kernel almost everywhere (0.7–1.0× at the 2048/4096-class shapes, one 1.21× win at
1024³) and runs only 1.1–1.5× the bf16 rate — the default tile is bound by dtype-blind
costs (A/B re-streaming from DDR, per-tile handshakes, a 4-byte output tile either way).
**int8's half-size tiles leave the L1 headroom bf16 doesn't have**: `n=64` fits at 52 KB
and doubles the rate, bit-exact — 4448–4607 GOPS at the 2048-class shapes, 2.5× bf16's,
the highest single-dispatch rate in this project. bf16 at `n=64` misses the 64 KB tile by
exactly the 3,328 B stack. **Verdict (mean-based, this repo's convention): NPU int8 at
`n=64` wins 1.10×–1.83× at every shape with M ≥ 512 and N ≥ 2048**; loses at 512³ and at
short prefill (M ≤ 256). Thin at the largest shapes — the CPU kernel's best-case (min) time
takes back the K=N=4096 rows. The small-M loss is a tile artifact: throughput tracks `m`
(forced to M/8 by the design), not token count, at both dtypes.

Not done: bf16 at `n=64` by single-buffering the C FIFO (`whole_array.py` was in use by
another live session). `results/aie/int8_matmul_sweep_npu.log`, `docs/DECISIONS.md`.

## `groupnorm_bf16/`

Replaces the `InstanceNormalization` (really `GroupNorm(32)`) in
`resnetv2_50x3_xint8.onnx` — the op that falls to CPU on that model.

The kernel alone beats CPU on 33 of 49 nodes
(`results/aie/groupnorm_bf16_kernel_npu.log`). **But the measured two-process handoff
floor erases the win at every shape — 0/49 once spliced**
(`results/aie/groupnorm_bf16_handoff_floor_npu.log`). The splice needs two OS processes
because the XRT Python binding is built against Python 3.13 and the EP's env is 3.12, a
hard ABI wall; the floor is 789 µs–23.6 ms per call depending on shape, and ~90% of it at
the largest shape is the fp32/bf16 conversion, not the shared-memory transfer.

`extract_golden.py` and `measure_handoff_floor.py --role ep` are the two scripts here that
run in `resnet_env17`, not ironenv: the former pulls a real node's input, params and ORT's
own CPU output out of the model into `data/golden/` (git-ignored) so the kernel is checked
against the actual tensors, not random data; the latter is one side of the two-process
handoff-floor measurement above (the other side, `--role kernel`, runs in ironenv like
everything else here).

## `attention_bf16/`

Replaces multi-head attention (`MatMul` + `Softmax` + `MatMul`) in `mobilevit_xxs`, whose
58 partition-thrashing EP subgraphs it was meant to eliminate.

Numerically correct (<1% rel L2, 0 NaN) on 8 AIE2 cores across all 3 MobileViT stages
(Stage 2: 57.6 ms, Stage 3: 4.57 ms, Stage 4: 0.86 ms at 8 heads) — and a **negative
result**: it loses to Zen4 CPU by 71×–240×.

**Its recorded diagnosis was wrong.** The loss was blamed on per-dispatch cost, which is
~1% of Stage 2's time now that the floor is measured (see `dispatch_floor/` below). The
real cause is that `attention_kernels.cc` uses `aie::mmul` **zero** times, hand-rolling dot
products with a horizontal `reduce_add` per output element for 0.61 GFLOPS on hardware
measured at 895.

**Do not reuse its inner loop as a template.** See its own
[README](attention_bf16/README.md) and `results/aie/attention_bf16_kernel_npu.log`.

## `conv2x_baseline/`

Not an operator — the **CPU baseline** for mlir-aie's `ml/resnet/layers_conv2_x`, the
chained int8 design that had run here since 2026-09-06 with its CPU side never measured.

3 bottlenecks, 436.21 MFLOP, all rows one sitting: NPU int8 **1869.6 µs** hw / **2497.8 µs**
end-to-end vs CPU ORT QDQ int8 **295 µs** — **CPU wins 6.3–8.5×** like-for-like. torch fp32
(1856 µs) sits within 1% of the NPU, so benchmarking against torch alone would have read as
parity. `results/aie/conv2x_int8_cpu_baseline.log`.

**Retracted:** its host-cost claim, that "628.2 µs reproduces 617.0 µs to ~2%". Those are
different brackets — 617.0 µs is the passthrough's *wall* intercept, and the like-for-like
host floor is 447.3 µs, so 628.2 is 40% over it rather than 2% under. Nothing in the
verdict depends on it.

**Superseded:** its scope note that the op class is *not* closed. It closed *this design at
this shape*; `bottleneck_sweep/` has since closed the op class.

## `bottleneck_sweep/`

Not an operator — the **spatial sweep** the conv2x log asked for: does NPU GOPS scale with
problem size, or is conv simply slower here? `sweep.py` runs one standalone `ml/bottleneck`
on the NPU; `cpu_sweep.py` runs the same arithmetic through ORT QDQ int8; same shapes, one
sitting.

`tensor_h` is the free axis (every L1 buffer is `tensor_w`×channels). NPU hw throughput
rises **116.7 → 143.7 GOPS** over 16× more work, marginal **146.1** vs the CPU's **819.0**
— the CPU wins **7.6×** at 32×32 and **5.7×** at 512×32, its own worst point. Utilization
was worth 23% against a 570% gap, so **the op class is closed.** That conclusion stands;
two things recorded alongside it did not.

**Corrected (2026-09-07): the `tensor_w`=32 ceiling.** The real `aiecc`/Tile(0,4) L1
ceiling for this channel config is `tensor_w`=**44**, not 32 — one of Tile(0,4)'s five
buffers is channel-sized rather than width-sized, so the original "5×w×256 B" estimate
overstated it. And 56×56, which this sweep recorded as impossible to compile at all, has
since been reached: single-buffering the final output `ObjectFifo` (`depth=1`, the one
buffer of Tile(0,4)'s five safe to shrink — `skip_buf`'s depth is load-bearing for the skip
connection's timing) frees 3584 B at w=56, just enough. Both 32×56 and 56×56 verify against
the torch golden. **At ResNet50's real conv2_x shape the verdict got worse, not better:**
NPU hw 4.2435 ms vs CPU 0.3327 ms, **CPU wins 12.75×** (marginal rate 111.1 vs 1678.8 GOPS,
15.1×), against the 5.7–11.4× found at every compile-limited 32-wide shape.

**Closed: the kernel-quality open item.** `conv2dk1.cc`/`conv2dk3.cc` were read
(2026-09-07): both vectorize correctly with `aie::mmul` — but both hardcoded a
width-32-only restriction, now fixed (below). What remains open is narrower: **column
count** (both measurements use 1–3 columns of a 4×5 array).

`results/aie/bottleneck_spatial_sweep_npu.log`, `results/aie/bottleneck_widthfix_npu.log`,
`results/aie/bottleneck_w56_npu.log`, `docs/DECISIONS.md`.

## `conv2dk3_widthfix/`, and the `conv2dk1`/`conv2dk1_skip` fix

Not operators — two upstream mlir-aie bugs, both a width-32 hardcode in the kernels
`bottleneck.py` depends on.

**`conv2dk3` — RESOLVED, 100% bit-exact across all widths** (32x32, 32x36, 32x40, 32x48,
32x64 all PASS with 0.000 max |diff|). Upstream attempted to unroll 8 chunks concurrently
using 8 `MMUL4x8x8` accumulators on hardware that only has 6 accumulator registers. Peano
LLVM spilled vector accumulators to stack (`0x580`), misaligning vectors and clobbering
channel-block 0. Upstream's aligned block (`iw_32 > 0`) also had compounding pointer bugs
(line reset off by 32, line buffer never advanced across `iw_32c` steps, weights never
reset per step, output stride off by 224 bytes). Replaced both blocks with a unified
`run_middle_block<ActType, N>` with `N <= 4` (blocks of 4 chunks, then remainder).
Accumulators stay within hardware registers (max 4 live anywhere), stack drops to 288 B
(`0x120`), and pointer invariants are completely restored. Verified in both
`conv2dk3_ui8_vector` and `conv2dk3_i8_vector` across wheel and git clone copies, and
independently reproduced in a fresh session (`results/aie/conv2dk3_widthfix_npu.log`).

**`conv2dk1`/`conv2dk1_skip` — RESOLVED.** A second, separate width-32 bug in the same
design's other two kernels, found while re-testing `conv2dk3`'s fix.
`conv2dk1_i8_vector`/`conv2dk1_ui8_vector` hardcoded `iw_32_rem = 0` (a real value, never
computed from `input_width` — dead code, not corrupted code); `conv2dk1_skip_i8_vector`'s
remainder path was never implemented at all (a commented-out stub). Both guarded only by an
`assert` compiled out in release, so any width not a multiple of 32 silently dropped its
remainder columns. Rewrote all three to walk `input_width` in blocks of N≤4 chunks addressed
directly from base pointers (no incremental pointer state), same accumulator-count fix as
`conv2dk3`. **`conv2dk1_skip_ui8_vector` (uint8-skip twin, unused by `bottleneck.py`) is
left unfixed and flagged.** Verified end-to-end through the real `bottleneck.py` design via
`kernels/bottleneck_sweep/sweep.py`'s torch-golden gate: 32×32/36/40/44 all pass
(previously failed at 36/40/44 even with `conv2dk3` alone fixed). Real ceiling for this
channel config is `tensor_w`=44 (45 fails the VMAC 4-pixel granularity, not this fix; 46/48
exceed Tile(0,4)'s 64 KB) *before* the single-buffering change described above.
`results/aie/bottleneck_widthfix_npu.log`, `docs/DECISIONS.md`.

## `dispatch_floor/`

Not an operator — measures the **per-dispatch fixed cost itself**, the constant every
isolated-op verdict in this repo rests on.

No compute tile at all (shim→memtile→shim), payloads swept 8 KB–32 MB, output verified per
payload. **Hardware floor 169.8 µs** — confirming the "~185–200 µs" constant — but **617.0
µs wall** through the `@iron.jit` path, 3.6× more, which is what the two bf16 kernels above
were actually charged while their write-ups reasoned with 185 µs.

**Go/no-go for any future kernel: the op's CPU time must exceed ~617 µs (IRON) or ~170 µs
(zero-overhead best case). Run this before writing a kernel.**
`results/aie/dispatch_floor_npu.log`.
