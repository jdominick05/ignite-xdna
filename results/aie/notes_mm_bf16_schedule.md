# Upstream `mm.cc` bf16 64x64x64: where the cycles go (2026-10-19, Desktop 2)

**Source.** `C:\Users\<user>\mlir-aie\aie_kernels\aie2\mm.cc` (checkout 760932a4abf), `matmul_vectorized_4x4`
(bf16 to f32). It was built with `tools/l1_tile_bench.py`'s `MM_FLAGS` (Peano `-O2`) and
`-DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -Dbf16_f32_ONLY`, then disassembled with Peano's `llvm-objdump -d`.

**Measured whole-tile figure.** 5427.4 cycles per call, 37.7% of 128 MAC/cycle
(`l1_tile_mm_direct_64x64x64_desktop2_20260923T0538Z.log`).

## The structure

The tile is 16 output blocks, each a 4 x 4 grid of `mmul<4,8,4>` accumulators (16 x 512-bit, in `bml`/`bmh`).
Per block:

| Phase | Bundles | What binds it |
|---|---:|---|
| prologue (`.LBB0_2`) | 51 | 32 accumulator-half loads (`vlda am..`), one per bundle on load unit A, plus pointer setup |
| K loop (`.LBB0_3` to `.L_LEnd0`), 7 iterations | 33 each | **30 `mov`s on the single move slot**; only 16 `vmac.f` |
| peeled last K-step + store (to the `jnz`) | 45 | 32 accumulator-half stores, one per bundle, the last 16 `vmac.f` under them |

- **Sum per block:** 51 + 7 x 33 + 45 = 327 bundles. Over 16 blocks, plus the call and outer-loop overhead, that is
  about 5,300. This matches the measured 5,427 (DERIVED; the fit is not exact).
- **The loop is pointer-bound, not load-bound or MAC-bound.** The source keeps eight post-incremented pointers
  (`pA1`-`pA4`, `pB1`-`pB4`). Peano moves them between `p` and `r` registers every iteration, about 30 moves through
  the one `mv` slot.
- **Loads have slack.** Each K-step needs 16 x 256-bit loads, A and B, which is 8 bundles on two units. The loop
  issues them across 33 bundles.
- **The C traffic is a fixed cost per block.** The 64-wide K tile gives only 8 K-steps, which is 128 MAC bundles at
  best. Each block still loads and stores 16 accumulators, 64 half-transfers. The rule is one accumulator load or
  store per bundle (SILICON.md).

## Targets for an owned kernel
1. **K loop: 16 `vmac.f` in 16 bundles.**
   - One A base pointer and one B base pointer, with immediate or modifier offsets, instead of eight pointers.
   - A and B loads spread over VLDA and VLDB, issued one K-step ahead.
2. **C traffic overlapped with MACs across blocks.**
   - Accumulator j's last MAC in a block sits 16 bundles before its first MAC in the next.
   - Store (6 bundles after `vmac.f`), reload, and the reload-to-`vmac.f` 5 bundles all fit inside that gap.
   - The 64 half-transfers per block fit in its 128 MAC bundles at one per bundle.
   - Only a hand schedule can do this (stage 2); Peano does not pipeline across the block loop.
3. **Ceilings, DERIVED:**
   - fixing (1) alone: about 51 + 8 x 16 + 16 = 195 bundles per block, about 3,200 cycles, about 64%;
   - (1) and (2): 128 bundles per block plus fill and drain, about 2,100 cycles, about 95% (tnzr's 32^3 tile
     measured 86%).
