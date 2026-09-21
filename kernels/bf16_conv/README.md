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
* **No comparison against AMD's stack** is made here. That the Vitis AI EP places no float of any
  width on the NPU is measured separately, by `tools/amd_float_precision_probe.py`.

## Running it

Needs the mlir-aie IRON environment and an idle device:

    bash scripts/research-lowlevel.sh --log results/aie/bf16_conv_npu_<date>.log --checks-only --npu \
        -- bash scripts/research-iron.sh kernels/bf16_conv/conv_bf16.py --sweep

`--sweep` runs the whole shape family in one process; without it, `--kdim/--rows-out/--cols-out/--ncin/--ncout`
take a single shape. `--ncin` and `--ncout` count channel BLOCKS, of 8 and 4 respectively.
