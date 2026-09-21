# bf16 convolution on AIE2

A convolution that runs on XDNA1's native bfloat16 matrix unit. Milestone 1 of a half-precision
path: **correctness only**. Nothing here is timed, and nothing here runs a model.

## What it establishes

Four shapes on Phoenix silicon, Desktop 2, NPU idle witnessed before and after, no foreign
contention ([`results/aie/bf16_conv_npu_20260921.log`](../../results/aie/bf16_conv_npu_20260921.log)):

| k | out rows x cols | in ch | out ch | MACs | result |
|---:|---|---:|---:|---:|---|
| 3 | 4 x 16 | 8 | 4 | 18,432 | 256/256 bit-exact |
| 3 | 8 x 32 | 16 | 16 | 589,824 | 4096/4096 bit-exact |
| 1 | 8 x 32 | 32 | 16 | 131,072 | 4096/4096 bit-exact |
| 3 | 8 x 32 | 32 | 8 | 589,824 | 2048/2048 bit-exact |

Every element matches a CPU reference that accumulates in fp32 from bf16-rounded inputs, which is
what the core does. `max_rel` and `rel_l2` are exactly 0.0 on all four. The repo's standing bar for
bf16 kernels is a tolerance, not bit-exactness, so this clears it by a margin rather than meeting it.

The reference is itself checked against a naive quadruple loop on three shapes before any of this
touched hardware, because the reference encodes the memory layouts and a wrong layout would have
been blamed on the kernel.

## Speed: one AIE core against one Zen 4 thread

[`results/aie/bf16_conv_bench_large_npu_20260921.log`](../../results/aie/bf16_conv_bench_large_npu_20260921.log),
`timing_eligible: true`. 3x3, 8 in / 32 out channels over an 8x32 tile, 589,824 MACs per pass,
torch 2.14 bf16 on the same machine as the baseline (the 8700G is Zen 4, so its CPU path has
AVX512-BF16 and is not a strawman):

| | per pass | GFLOPS |
|---|---:|---:|
| One AIE core, dispatch amortised | **12.81 us** | **92.10** |
| CPU torch bf16, 1 thread | 35.34 us | 33.38 |
| CPU torch bf16, 16 threads | 116.30 us | 10.14 |
| One AIE core, single dispatch | 577.90 us | 2.04 |

**2.76x one CPU thread, at 20.0% of this core's 460.8 GFLOPS bf16 ceiling.**

Three things that number needs attached to it, or it misleads:

* **The 16-thread row is degenerate and is not a win to quote.** At this tile torch's threading
  overhead exceeds the work, so 16 threads are slower than 1. The honest comparison is per core.
* **`single dispatch` is the dispatch floor, not the kernel.** 577.9 us for a tile that is ~13 us of
  arithmetic reproduces this repo's documented ~617 us one-shot IRON floor. Any real use has to put
  many passes in one dispatch, which is exactly what the graph engine already does for 66 layers.
* **This is one core of sixteen**, and the harness drives it directly. It is not a device number and
  not a model number.

### Accumulators in flight is the whole optimisation so far

[`results/aie/bf16_conv_bench_npu_20260921.log`](../../results/aie/bf16_conv_bench_npu_20260921.log).
Identical work in every row - 147,456 MACs, same tile, same kernel - with only the number of live
accumulators changing, because `ncout` is that number:

| accumulators | GFLOPS | us per pass |
|---:|---:|---:|
| 1 | 24.44 | 12.07 |
| 2 | 31.13 | 9.47 |
| 4 | 36.64 | 8.05 |
| 8 | **45.35** | 6.50 |

A single accumulator serialises on the vector MAC's latency: every `mac` waits for the one before
it. Eight independent chains is 1.86x one, from nothing but reordering. The int8 engine keeps four
for the same reason. *(Corrected 2026-09-21: it keeps eight - four output blocks times two column
groups, all in `cm0`-`cm7` with no spill, read off its object in
[`engine_census_widened_desktop2_20260921.log`](../../results/aie/engine_census_widened_desktop2_20260921.log).
The bf16 engine core holds eight as well, so this lever is spent in both and what remains is how
densely the hot loop issues them: see `docs/BENCHMARKS.md`, "Static readings of both engine cores".)*
The CPU-relative speedups in that log are inflated by torch's per-call overhead
on such a small tile and should not be quoted; the large-tile table above is the fair one.

Remaining headroom is large and the next levers are known: this is one core, the loop is not
software-pipelined, and 20% of ceiling is well short of what the bf16 GEMM reaches.

## Milestone 2: the engine core, `engine_bf16.cc`

One program whose kernel size, stride, input block count, geometry and epilogue arrive in the weight
packet's header. Measured on one core
([`engine_bf16_mac_model_probe_npu_20260921.log`](../../results/aie/engine_bf16_mac_model_probe_npu_20260921.log),
[`engine_bf16_bench_npu_20260921.log`](../../results/aie/engine_bf16_bench_npu_20260921.log); full working in
`docs/BENCHMARKS.md`, "bf16 convolution on one core"):

* **Byte-exact on every path it has**, compared as 16-bit patterns: k = 1, 3, 5; stride 1 and 2; 1 to 8
  input blocks; ReLU and ReLU6; and a two-packet `F_LOAD_PSUM` chain inside one dispatch. `F_HOLD` is
  not observable until `OP_RESIDUAL` exists.
* **The reference is only as good as its model of the multiply-accumulate**, and that model is now
  measured: the instruction aligns the accumulator and its eight products to the largest exponent and
  rounds each to a 24-bit grid, ties to even. It is not IEEE addition. A random sweep cannot see this -
  four different models all pass it - so `--probe` exists to break a wrong one.
* **10.10 us per pass dispatch-free, 91.3 GFLOPS, 19.8% of ceiling**, against 8.96 us and 102.9 GFLOPS for
  `conv_bf16.cc` on identical work in the same sitting: 12.8% slower, unattributed.

    bash scripts/research-lowlevel.sh --log results/aie/engine_bf16_npu_<date>.log --checks-only --npu \
        -- bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16.py --sweep --probe
    bash scripts/research-lowlevel.sh --log results/aie/engine_bf16_bench_npu_<date>.log --npu \
        -- bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16.py --bench --kdim 3 --ncin 4

## Why it did not exist before

There is no bf16 convolution in mlir-aie's `aie_kernels` tree (`conv2dk1.cc` and `conv2dk3.cc` are
int8/uint8), none in `programming_examples`, and none in this repo - `kernels/dispatch_floor/splice_conv.cc`
is a broadcast scale-and-bias, not a convolution. The bf16 GEMM, GroupNorm and eltwise kernels that
have run here are all non-convolutional. So this was written from the arithmetic, not adapted.

## Shape and layout

AIE2's bf16 matrix intrinsic is `mmul<4, 8, 4>`: 4 pixels by 8 input channels by 4 output channels,
fp32 accumulate. The N of 4 is the one structural difference from the int8 engine's `mmul<4, 8, 8>`,
and it is why the output blocks by 4 where the input still blocks by 8.

    act  [cin_block][row][col][8]                bf16, halo included in row/col
    wts  [cout_block][ky][kx][cin_block][32]     bf16, 32 = 8 in by 4 out, in walk order
         then [cout_block][16]                   bf16 bias, replicated to mmul's 4x4 C
    out  [cout_block][row][col][4]               bf16

Weights are laid out to match the kernel's walk exactly, so the core reads one contiguous stream and
never computes a weight address.

## Two hardware constraints this ran into

Both were found by the toolchain refusing, and both are worth not rediscovering:

* **A core tile has two input and two output DMA channels.** A third input fifo is refused at
  placement: `tile (0, 2) requires 3 input/1 output DMA channels, but only 2 input/2 output
  available`. Activations and weights take the two, so the bias rides at the end of the weight
  buffer. Rounding the bias to bf16 costs nothing measurable here - it is one addend against
  `ncin * 8 * k * k` products and the accumulation stays fp32 - but the reference rounds it too, so
  the comparison stays honest.
* **bf16 has no 4-element load.** `aie::load_v<4>` does not compile for `bfloat16`; 16 is the
  smallest bf16 vector the core loads. The bias is therefore stored pre-replicated to the
  accumulator's 16-element shape, which removes a `grow_replicate` from the inner loop as well.

## What it does NOT establish

* **No speed claim.** Nothing here is timed. A bf16 conv that is bit-exact and slow is still a
  negative result, and this repo has one already: `kernels/attention_bf16` is numerically correct and
  71x to 240x slower than the CPU.
* **No model runs on it.** This is one kernel on one core, driven by a test harness. The graph engine
  is int8 end to end and cannot host this: its core program uses 15,280 of 16,384 bytes of program
  RAM, so a second conv pass does not fit, and its packet geometry assumes one byte per element.
  *(Superseded 2026-09-21: 15,280 B is the kernel object. The linked core is 16,160 B, so the
  int8 engine has 224 B free, not 1,104 -
  [`engine_linked_program_size_desktop2_20260921.log`](../../results/aie/engine_linked_program_size_desktop2_20260921.log).
  The conclusion stands and is stronger.)*
* **No comparison against AMD's stack** is made here. That the Vitis AI EP places no float of any
  width on the NPU is measured separately, by `tools/amd_float_precision_probe.py`.

## Running it

Needs the mlir-aie IRON environment and an idle device:

    bash scripts/research-lowlevel.sh --log results/aie/bf16_conv_npu_<date>.log --checks-only --npu \
        -- bash scripts/research-iron.sh kernels/bf16_conv/conv_bf16.py --sweep

`--sweep` runs the whole shape family in one process; without it, `--kdim/--rows-out/--cols-out/--ncin/--ncout`
take a single shape. `--ncin` and `--ncout` count channel BLOCKS, of 8 and 4 respectively.
