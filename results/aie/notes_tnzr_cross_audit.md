# Cross-audit against "Hello XDNA!" (tnzr.org): what this repo got wrong, what it never knew, what is new

**Date:** 2026-09-23 (Desktop 2, `DESKTOP-CBL5NUA`, Ryzen 7 8700G, the same part the reference measured on)
**Reference:** T. Steinert and A. Breuer (Uni Jena), *Hello XDNA!*, <https://tnzr.org/xdna/>,
source <https://github.com/scalable-analyses/xdna>. Its initial commit (`3d86c8a1`) is authored 2025-12-18.
The GitHub repository was created 2026-01-21, the day of the Jena group's announcement, and was last pushed 2026-08-31
(GitHub API, fetched 2026-09-23). **Public since 2026-01-21.**
The pages audited are [Instruction Set Architecture](https://tnzr.org/xdna/isa.html) and
[XDNA1 Kernel](https://tnzr.org/xdna/xdna1_kernel.html). The reference repository carries **no licence**,
so it is cited here for its facts and was never vendored into this tree.
**Status:** living audit record. Phase 1 (static) rows are filled in; phase 2 (silicon) rows say
PENDING until a log under `results/aie/` backs them.

## How to read this

A disagreement between this repo and the reference is settled by a **third** source, never by picking
a side. The sources, in order:

1. **Peano's machine model:** `Xilinx/llvm-aie`, `llvm/lib/Target/AIE/aie2/*.td`. This is what the
   compiler that builds every kernel here believes about the core. Tag: SPEC(Peano).
2. **Silicon:** a log under `results/aie/`. Tag: MEASURED.
3. **This repo's own evidence:** logs and object code already in the tree.

The reference is not ground truth either. Its "sixteen available accumulator registers" understates
Peano's register file (row A1).

**Verdicts:**
- CONFIRMED: the repo claim holds.
- CONTRADICTED: the claim is false.
- REFINED: true, but narrower or broader than stated.
- UNBACKED ON MAIN: the evidence exists only on an unmerged branch.
- STALE: superseded, but not marked as such.
- RETRACT: the number must be withdrawn.
- NEW-FACT: the repo was silent and the reference supplies it.

A ✓ means this audit re-read the claim at the cited line on 2026-09-23. Rows without ✓ come from the
exploration sweep and are re-read when their fix lands.

## What the reference establishes for XDNA1

| # | Fact | Reference |
|---|---|---|
| R1 | XDNA1 is AMD's consumer AIE-ML (AIE2); register files as AIE-ML | isa.html |
| R2 | 6-slot VLIW, in-order, no interlocks for data/control/structural hazards | index, isa.html |
| R3 | The only native float matmul is BF16 4×8×4 `VMAC.F` (config register value 28), 256 FLOP/cycle/core. bf16 8×8×4, 4×16×4 and 8×8×8 are emulated or need data manipulation; fp32 matmul is emulated on bf16 | isa.html Table 1 |
| R4 | `VMAC.F` latency 6; the accumulator operand is read in cycle 3 (late forwarding) | isa.html Table 2; Peano `II_VMACf` |
| R5 | LDA/VLDA/VLDB 7 cycles; ST/VST 1; VSHUFFLE 2; MUL 2; J/JZ/RET 6 (5 delay slots) | isa.html Table 2 |
| R6 | Vector loads are 32 B; units A and B; only A loads accumulators; ≤1 accumulator load/store per instruction; loads and stores go in different instructions to avoid bank conflicts | xdna1_kernel.html |
| R7 | Kernel uses 16 × 512-bit accumulators `BML0–7`/`BMH0–7` ("eight out of sixteen available") | xdna1_kernel.html |
| R8 | Hardware loop (`MOVXM LS/LE/LC`): set up ≥64 B before the loop start; first and last loop instructions 16-B aligned; the last instruction 16 B wide; NOP = 2 B, full bundle = 16 B | xdna1_kernel.html |
| R9 | 1.8 GHz, with `xrt-smi configure --pmode turbo` to reach it | xdna1_kernel.html |
| R10 | Hand-scheduled BF16 32×32×32 tile (M32×K32×N32, output-stationary, 2×4 register blocking, double-buffered accumulators, one hardware loop): 256 of 288 instructions carry a `VMAC.F` (89%). **398 GFLOPS measured on one compute tile of a Ryzen 7 8700G**, against 410 theoretical. L1-resident, no DMA in the timed loop, 10⁶ calls per dispatch, host wall clock | xdna1_kernel.html; `src/XDNA1.mlir`, `src/driver_kernel.cpp` |

## A. Where this repo is wrong, contradicts itself, or over-claims

| # | Claim in this repo | Where | Verdict | Settled by |
|---|---|---|---|---|
| A1 | The accumulator file is stated four ways: 9×1024-bit `cm0–cm8` (SILICON); 8 × `cm0–cm7` (two notes); **"only 6 accumulator registers"** (kernels/README, `conv2dk1.cc`), not marked superseded. The "5 live is the spill-free ceiling" is contradicted by the repo's own engine kernels holding 8 spill-free | ✓ `docs/SILICON.md:77`; ✓ `kernels/README.md:290`; `results/aie/notes_im2col_kernel_vliw_audit.md:280`; `notes_accumulator_cascade_gemm_model.md:73`; `kernels/conv_accum/aie2/conv2dk1.cc:146` | **9 is CONFIRMED**, in both the compiler and the hardware. "6" and "8" are **STALE**. "5 = ceiling" is **REFINED**: it is the probe's vector-register pressure (fresh A and B vectors per MAC), not the accumulator file. Precisely, the engine keeps its 8 accumulators with **no accumulator spill**: 0 of the 169 stack references in `engine.o` are accumulator loads or stores. It does spill **vector** registers (31 references), which fits the same vector-pressure reading | SPEC(Peano): `results/aie/peano_aie2_machine_model_a36c62b9.log` §1 gives `cm0`–`cm8`, each `[bml_i, bmh_i]`, plus 9 each of `amll/amlh/amhl/amhh`. By use: `results/aie/engine_core_issue_census_desktop2_20260923.log` shows all four engine ELF versions naming the ninth accumulator 26 times, and the engine runs bit-exact on silicon, so `cm8` exists in hardware. A direct probe (P1) would only restate this |
| A2 | "`vshift`/`vmov` spend slot `[v]`… cannot co-issue with `vmac`" | ✓ `docs/SILICON.md:264,305-309` | **CONTRADICTED** by the repo's own disassembly and by Peano | ✓ `results/aie/conv_issue_rate_decomposed.log`: `0x072a vmov wh10, wl5 \| vmac cm1…`; `0x075c vshift x8… \| vmac cm3…`; `0x0780` carries `vldb`, `vlda`, `add`, `vshift` and `vmac` in one bundle. SPEC(Peano), `peano_aie2_machine_model_a36c62b9.log` §3: `vshift`, `vshift.align`, `vshuffle`, `vbcst` and register `vmov` issue in the **mv** slot; `vmac`/`vmul` in **vec**. What realignment costs is the mv slot plus 2-cycle latency, and nop bundles when it sits on the critical path, not the vector slot |
| A3 | int8×int4 is a native `vmac` at 512 MAC/cycle; `vldb.unpack.s8.s4`; "a widespread assumption that native sub-byte execution was exclusive to AIE2P" | `results/aie/notes_w4a8_aie2_roofline.md:8,54`; `TODO.md:530-534`; against `docs/SILICON.md:66,1023` ("AIE2p only") | **UNBACKED ON MAIN**, not unbacked. ✓ `w4a8_probe_npu.log` and `w4a8_array_npu.log` exist in commits `56b4e88` and `2fe7624` (`8410fc5` and `18622ab` after the branch's rebase onto `35d58d5`) on the unmerged branch `worktree-int4-study` (2026-09-10). Main's SILICON still says "AIE2p only", and that is **CONTRADICTED by SPEC**: AIE-API 2024.1 lists AIE-ML `8b x 4b: 4x16x8` as native, and Riallto states 512 int4×int8 MAC/cycle per core (D11, both fetched). The "widespread assumption" framing is retracted: AMD documented it. The branch's silicon probe is a confirmation, not a discovery | Branch logs; Peano `mmul_8_4.hpp` (cited by the note itself); AIE-API 2024.1; Riallto |
| A4 | **"YOLOv8n Complete End-to-End Monolithic Forward Pass … Demolishing AMD's 6.61 ms Baseline", 1.732 ms, 577.3 FPS, 3.82×, Layers 0–22, 0 CPU partitions** | ✓ `benchmarks/full_forward_latency_results.md:1-37`; ✓ `results/benchmarks/yolov8n_full_forward_silicon.log`; commit `c2e6e46` (2026-09-13) | **RETRACT.** Four independent defects, all read from the benchmark's own source: (1) ✓ `benchmarks/benchmark_yolov8n_full_forward.py:178-180` **hardcodes** Backbone = 0.709 ms and Neck = 0.407 ms and sets Detect Heads to the remainder (the logged percentages sum to 104.97%). (2) ✓ Lines 189-225: the "parity" 0.9965 compares the **CPU** ONNX cut model's heads, decoded, with the float ONNX model; the NPU output `raw_heads_hw` is only tested for being non-empty. (3) ✓ Lines 43, 45: the AMD "baseline" 6.61 ms and "unoptimized" 23.42 ms are constants, not measured in the run. (4) ✓ `docs/BENCHMARKS.md:7685` (same day): the shipped `build/yolov8n.ignite` has a 4,096-byte egress, no `head_layout`, zero-filled heads. The measured whole-network result is the later graph-engine container (`BENCHMARKS.md:7754`) | The benchmark's own source and log |
| A5 | `aie2/` row: "20-core full-array engine … M=2 … 1.000 vmac/cycle, **100% vector ALU saturation**"; "18.43 TOPS … compiler-verified" | ✓ `kernels/README.md:52`; `results/aie/notes_im2col_20core_array_synthesis.md:18,190` | **REFINED/over-claimed.** 1.000 vmac/cycle is a *static* inner-loop count; 18.43 TOPS is peak arithmetic. On silicon ✓ `results/aie/hardware_im2col_execution.log:11-14`: 20 cores ERT_CMD_STATE_TIMEOUT; 16 cores 0.0270 TOPS, logged as "0.37%" density; 1 core 0.0037 TOPS, logged as "0.80%". Those densities use the 128-MAC/cycle int8 peak of A7. Against the SPEC 256 at 1.80 GHz they are **0.18%** (0.0270 of 14.75 TOPS) and **0.40%** (0.0037 of 0.92) | Repo's own hardware log |
| A6 | "0 BYTES (100% L2 MemTile SRAM)" at `0x40000`/`0x60000` via Locks 4/5 | `docs/BENCHMARKS.md:7467-7497`; `TODO.md:562-572` | **Unreconciled** with `docs/LOW_LEVEL_AUDIT.md:29-32` (the Lock 4/5 init branch "could never fire") and `:146-153` (the shipped streams never touch those addresses). Needs a pointer in place | LOW_LEVEL_AUDIT |
| A7 | Per-core int8 peak computed at 128 MAC/cycle (460.8 GOPS/core) | `results/aie/notes_im2col_hardware_bringup.md:136-138` | **CONTRADICTED** by SPEC 256 MAC/cycle int8 (`docs/SILICON.md:64`). Every %-of-peak in that note is 2× too high | SILICON SPEC row |
| A8 | Stale text still printed: slot b = "branch"; "nine register names are a lower bound"; "nominal tile clock of 1.0 GHz"; the H11 2×2 reblock under "Whole-Call Rate" as if achieved; 6 vs 8 `vshift` count for one loop | `docs/BENCHMARKS.md:1500,1553`; `notes_a16w8_feasibility_audit.md:75`; `notes_memtile_4d_im2col_specification.md:278`; `docs/SILICON.md:278,282` | **STALE** | Corrections already in-tree (`SILICON.md:73`; `aie_disasm.py:21-22`; `conv_issue_rate_decomposed.log:105-123`) |
| A9 | "ERT timeout on Column 4 … Columns 0–3 complete physically addressable grid" | `results/aie/README.md:1109` | **Numbering clash** with SILICON's logical/physical split (logical 0 = physical 1) | `docs/SILICON.md:51-55` |
| A10 | "Peano ignores the pipelining macros" (exploration) vs "the `AIE_LOOP_*` hints are live" (branch) | `aie_kernel_utils.h` (both copies); commit `56b4e88` (rebased: `8410fc5`) on `worktree-int4-study` | **REFINED.** ✓ Peano predefines `__AIECC__`, so `aie_kernel_utils.h:32-57` applies. `AIE_LOOP_UNROLL`, `AIE_LOOP_MIN_ITERATION_COUNT`, `AIE_TRY_INITIATION_INTERVAL` and `AIE_PREPARE_FOR_POSTPIPELINING` are live clang pragmas. `AIE_PREPARE_FOR_PIPELINING` and `AIE_LOOP_FLATTEN` expand to nothing, so every use of those two in this tree is a no-op | The header itself; `clang++ -dM -E` in `56b4e88` |
| A11 | Program memory: 13,424 B used | `docs/DECISIONS.md:2054` | **STALE.** The newest engine ELF's `.text` is 16,160 of 16,384 B (224 B headroom); one variant sits at 16,368 B (16 B free) | `engine_core_issue_census_desktop2_20260923.log` |
| A13 | "One fixed core program … byte-identical across every model" | exploration; engine docs | **REFINED.** It holds within an engine version. `build/conv_engine/` holds **four** distinct core ELFs across 68 builds. 56 current builds share `45aeec4edc4f` (2026-09-18..21). **`yolov8n_full`**, the build behind `build/yolov8n_full.ignite`, is in the older `eaddc8c66b88` group (2026-09-17, `.text` 10,704 B), together with `yolov8n_opt`, `yolov8s_opt` and `resnet50_head`. Its conv loops issue at the same rates (0.242–0.348), so the issue-rate finding covers every container | `engine_core_issue_census_desktop2_20260923.log` |
| A12 | Licence: `src/` is "Apache-2.0 WITH LLVM-exception", and 86 tracked files written here carry `Copyright (C) 2026 Advanced Micro Devices, Inc.` | `pyproject.toml:11`; `README.md:249-250`; file headers | **CONTRADICTED.** The whole repository is **GNU AGPL-3.0-or-later** (the user, 2026-09-23). The AMD line was pasted from mlir-aie's template onto original work. Genuine upstream copies (`conv2dk1*.cc`, `mm_2x2.cc`, `zero.cc`, `aie_kernel_utils.h`, `whole_array_bankpad.py`) keep AMD's notice | `LICENSE`; git authorship |

## B. What this repo never knew (NEW-FACT unless marked)

| # | Fact | Status here before this audit | Settled by |
|---|---|---|---|
| B1 | `VMAC.F` 6-cycle latency with the accumulator read in cycle 3. **Refinement neither source stated:** int8 `VMAC` is **5** cycles, its accumulator operand is bypassed (`VEC_Bypass`), and Peano issues a dependent int8 `vmac` **2 bundles** after its producer, against **4** for bf16 `vmac.f`. One accumulator therefore sustains a MAC every 2 cycles in int8 and every 4 in bf16; 1 vmac/cycle needs ≥2 (int8) or ≥4 (bf16) independent accumulators. The engine holds 8, so its loops are not latency-bound | Silent. Nothing reasons about VMAC latency or forwarding | SPEC(Peano): `peano_aie2_machine_model_a36c62b9.log` §3 (`II_VMAC [5,3,1,1,1]` with VEC_Bypass, `II_VMACf [6,3,1,1,1]`) and §4 (Peano's own spacing: `bf16_mac2` 4 bundles, `i8_mac2` 2). **MEASURED P2** (`vmac_chain_forwarding_desktop2_20260923T0602Z.log`, Desktop 2): eight dependent MACs on one accumulator, `d` bundles apart, land **2, 4, 4, 8, 8, 8** of 8 for bf16 at d = 1…6 and **4, 8, 8, 8** for int8 at d = 1…4. The minimum correct distance is **4 (bf16) and 2 (int8)**, which is Peano's spacing exactly. The shortfall fits the no-interlock model to the MAC: each MAC reads the newest sum from a MAC at least 4 (bf16) or 2 (int8) bundles back, and a closer one is silently overwritten. Counting the bundles that execute, each variant costs exactly those over the empty control, to within 0.3 cycles. The static count carries one alignment bundle after the return's delay slots in every variant except bf16 d = 2. So a too-close chain gives a **wrong answer, not a stall**. That is R2 ("no interlocks") seen directly on the accumulator path. Scope: the spacing is MEASURED. Its split into latency and read cycle (6/3 bf16, 5 + bypass int8) stays SPEC(Peano) |
| B2 | Latencies: VLDA/VLDB 7 (the repo had scalar LDA 7 only), ST/VST 1, VSHUFFLE/VSHIFT/VMOV 2, MUL 2, `ret lr` with 5 delay slots. Accumulator load → first `vmac.f` 5 bundles; `vmac.f` → store 6; int8 `vmac` → store 5 | Scalar load CONFIRMED (`SILICON.md:76`); the rest silent. `asm_core_id.s` puts five NOPs after `ret lr` without saying why | SPEC(Peano) §3–§4. Peano's schedule of `bf16_mac` (§4) is bundle-for-bundle tnzr's Listing 2, so this Peano and theirs agree |
| B3 | Only load unit A writes accumulators; ≤1 accumulator load/store per bundle | Consistent with every disassembly here, never stated | SPEC(Peano) §3: every accumulator `vlda` (`II_VLDA_AM`, `II_VLDA_CONV`) reserves `LOAD_UNIT_A` and issues in slot lda; `vldb` has no accumulator form; accumulator stores are slot st |
| B4 | A load and a store to the same bank in one bundle conflict; the reference keeps them in separate instructions | The repo measured only load+load (+1 cycle, `SILICON.md:74`) | **MEASURED: +1 cycle per same-bank load+store bundle, as for two loads.** A natural experiment settles it. Upstream `mm.cc` int8 spills accumulators to the stack (bank 0, a 416-B frame), and its per-block epilogue has exactly **12** bundles pairing a stack reload (`vlda am.., [sp, ..]`) with a C store (`vst am.., [p3/p4/p5]`). At 32×64×32 (4 blocks per call) that predicts **+48** cycles when C shares bank 0 with the stack. Measured: **+48.5** (697.1 → 745.6 cycles, the same kernel and harness, only A/C swapped between banks 0 and 2: `l1_tile_mm_i8_k64_c_bank0_desktop2_20260923T0545Z.log` vs `l1_tile_mm_k64_…0533Z.log`). Direct mode's identical placement read 745.4. The control is the bf16 kernel: it has **0** such bundles, and its time does not move with placement (1418.5 vs 1418.6) |
| B5 | Hardware-loop rules (setup distance, 16-B alignment, 16-B last bundle) | Silent. Across every loop of all four engine ELFs, start and end are 16-B aligned and the last bundle is 16 B (consistent with the reference). But the `movxm ls/le` and `add.nc lc` writes sit **16–58 B before loop start**, and the code runs bit-exact. So "≥64 B before the start" is not what the hardware needs. Every loop's setup is **≥112 B before the loop end** (the minimum is exactly 112 in many loops), which suggests an end-anchored rule | Static: `engine_core_issue_census_desktop2_20260923.log`. MEASURED P3: PENDING (run last) |
| B6 | `VMAC.F` config value 28 = bf16 4×8×4; which bf16 shapes are emulated. **Also:** int8×int8 4×8×8 with acc32 uses config value **776** | Silent on both. The repo uses only native 4×8×4 bf16, so nothing here is wrong | Reference Table 1; `peano_aie2_machine_model_a36c62b9.log` §4 (`mova r0, #28` / `mova r0, #776`) |
| B7 | **A hand-scheduled assembly kernel runs on XDNA1 and reaches 86% of peak** | `SILICON.md:79`: "no hand-written kernel has been run on the NPU". The repo's only `.s` (`kernels/asm_probe/asm_core_id.s`) was assembled and linked, never executed | **MEASURED, reproduced here:** `tnzr_bf16_32x32x32_repro_desktop2_20260923T0518Z.log`. The reference's `.s`, assembled by this repo's Peano (a36c62b9) and linked by mlir-aie v1.4.2 aiecc on Windows, ran through `XrtSiliconHarness`: **397.5 GFLOPS** mean over 10 dispatches (396.6–397.9), 86.3% of 460.8 and 99.9% of the reference's 398. That was in **default** pmode, where the reference used turbo. Implied 296.7 cycles per call at 1.80 GHz, against 288 static kernel bundles (the other ~9 are the calling loop and call/return). Partitions idle before and after. The path this repo never tried (hand `.s` → `link_with` → aiecc → PyXRT) works end to end |

## C. The performance gap the reference exposes

Per-core peaks at the measured 1.80 GHz: bf16 460.8 GFLOPS, int8 921.6 GOPS (`docs/SILICON.md:64,159,200`).

| Kernel | Result | % of per-core peak | Data movement in the number? |
|---|---|---|---|
| Reference, hand-asm bf16 32×32×32, 1 tile | 398 GFLOPS (theirs, turbo); **397.5 reproduced here** (default pmode) | 86.3% | none (L1-resident): `tnzr_bf16_32x32x32_repro_desktop2_20260923T0518Z.log` |
| Upstream bf16 `mm.cc` hot loop, static | 16 `vmac.f` / 32 bundles | 50% cap | none (static) |
| Repo best single-tile bf16 conv (`chunk`), dispatch-free slope | 108.9 GFLOPS | 23.6% | yes (branch `worktree-bf16-engine`) |
| Repo bf16 array GEMM, per core | 168.8 GFLOPS | 36.6% | yes |
| Upstream int8 `mm.cc` loop, static / whole call / array per core | — | 88.9% / 41.9% / 31–33% | none / none / yes |
| **Shipped engine core, int8 conv inner loops, static** | 0.348 / 0.250 / 0.235 / 0.167 `vmac`/cycle (fused stage 2: 0.444) | 35 / 25 / 24 / 17% (44%) | none (static): `engine_core_issue_census_desktop2_20260923.log` |

The same census's bank check of the newest ELF reports a **HAZARD**. There are 12 paired-load (`vlda`+`vldb`)
bundles in loop bodies, 4 of the 23 in each of the two 0.348 loops, and `a0_0_cons_buff_1` shares bank 2
with `w0_0_cons_buff_0`. If those pairs read the shared bank, each such loop pays up to +4 cycles per
iteration (23 → 27, 0.348 → 0.296). This is static and depends on pointer resolution, so it is unmeasured.

`SILICON.md:418-425` says the bf16 array GEMM's "remaining two thirds" (at 64×64, against a 100%
input-bound ceiling) lie in "dispatch, the K-loop's C read-modify-write in f32, and the 3.2 kernel".
It names three candidates and measures none, and it never compares them with the 50% cap of the loop it runs. Its own production core issues a `vmac` on at most
~1 bundle in 3: loads, then `load_unaligned_v` `vshift` realignment, then idle, then 8 `vmac`s, with
no overlap between iterations. The reference shows the same silicon sustaining a `VMAC.F` on 256 of
288 bundles.

### C.2 Schedule without movement: the same tile, the same harness, the same banks (MEASURED, phase 2.2)

`tools/l1_tile_bench.py`, written here: operands in L1, the kernel called 10⁵–10⁶ times per dispatch,
time fitted against the call count so dispatch and DMA drop into the intercept (R² ≥ 0.9992 on
every fit). Every kernel's output is checked exactly against numpy (C0 + calls·A@B, small
integers). An empty control (`kernels/l1_tile_bench/empty_call.s`: `ret lr` + 5 delay slots) prices
the harness at 19.1–20.4 cycles per call with an `i32` counter, and at 25.4 with a 64-bit `index`
counter.

| Kernel, shape, mode | cycles/call | over empty | % of per-core peak | Log |
|---|---|---|---|---|
| Reference hand `.s`, bf16 32³, copy | 301.3 | **282.0** | 85.0% (391.5 GFLOPS) | `l1_tile_compiler_vs_hand_i32loop_desktop2_20260923T0530Z.log` |
| Upstream `mm.cc` bf16, 32³, copy | 898.6 | 879.3 | 28.5% (131.3) | same |
| Upstream `mm.cc` int8, 32³, copy | 533.4 | 514.1 | 24.0% (221.2 GOPS) | same |
| `mm.cc` bf16, 32×64×32 / 32×128×32, copy | 1418.5 / 2482.4 | — | 36.1% / 41.3% | `l1_tile_mm_k64_…0533Z.log`, `l1_tile_mm_k128_…0533Z.log` |
| `mm.cc` int8, 32×64×32 / 32×128×32, copy | 697.1 / 1009.4 | — | 36.7% / 50.7% | same two |
| **`mm.cc` bf16, 64³ (the array GEMM's tile), direct** | 5427.4 | 5407.0 | **37.7% (173.9 GFLOPS)** | `l1_tile_mm_direct_64x64x64_desktop2_20260923T0538Z.log` |
| `mm.cc` int8, 64³, direct | 2604.8 | 2584.4 | 39.3% (362.3 GOPS) | same |

What this settles:

- **The hand schedule runs exactly as written.** 282.0 cycles over the empty control is the
  kernel's 288 dynamic bundles less the control's own 6-bundle body. That leaves no memory stall and no
  hidden interlock, and the reference kernel is **exact**: it accumulates into C in `mm.cc`'s tiled
  layout (C0 + calls·A@B). The reference never checked its output. The 397.5-vs-391.5 difference between
  their harness and this one is the calling loop: their constant trip count lets Peano unroll the
  call 4× and fill its delay slots (~8.7 cycles/call), where this one reads the count at run time.
- **On an identical tile, harness and bank placement, the hand schedule is 3.0× the compiled
  upstream kernel** (282 vs 879 cycles, bf16 32³).
- **The compiled loops run at their static schedule plus one bank stall per iteration.** Adding
  K costs `mm.cc` bf16 32.5–33.3 cycles per inner iteration, against 32 static bundles, and int8 9.8–10.2
  against 9. Each loop body has exactly one paired load whose two operands come from the same
  buffer (int8: `vlda [p4]` + `vldb [p3]`, both A-pointers stepping `#0x20`; bf16: two B loads at `0x2a8`), so they hit one bank.
  The repo's own measured rule (+1 cycle for a same-bank pair, `SILICON.md:269`) predicts exactly
  +1. With no interlocks and no DMA in the timed loop, a bank conflict is the only stall source
  left. `tools/aie_bank_check.py` finds exactly one paired-load bundle per hardware-loop iteration in each
  (int8 `0x01f0`, bf16 `0x02a8`). Attribution: consistent and the only candidate standing, but not
  isolated. Both loads of the pair read one buffer, so no placement can separate them.
- **The bf16 array GEMM's "remaining two thirds" is the kernel.** At the array's own tile, with no
  movement at all, the kernel reaches 37.7% of peak. The array's measured 64/64/64 figures at 2048³ and
  above run from 2477.23 GFLOPS (`bf16_matmul_n64_single_buffer_npu.log`) to 2653.05
  (`gemm_tile_sweep_c_single_buffer_npu.log`), 33.6–36.0% of the 16-core 7,372.8. They are therefore 89–95% of what its kernel can do in L1. Input bandwidth and dispatch together account for at most a
  few points. The "K-loop's C read-modify-write" is inside the kernel (16 accumulator loads and
  stores per output block), not a separate cost. The int8 array (31.2% at 1.80 GHz, `SILICON.md`)
  reaches ~79% of its kernel's 39.3%, so int8 is more movement-bound.
- **Bank placement moves a whole compiled call by 7%, and why is now attributed.** `mm.cc` int8 at 32×64×32
  costs 697.1 cycles with A/B/C in banks 0/1/2 (copy mode) and 745.4 with C in bank 0 beside the stack
  (direct mode, `l1_tile_mm_direct_32x64x32_desktop2_20260923T0538Z.log`). Copy mode with direct mode's
  placement reproduces it (745.6), so the harness mode is not the cause. The kernel spills its
  accumulators to the stack, and 12 epilogue bundles per block pair a stack reload with a C store:
  4 blocks × 12 × 1 cycle = 48, against 48.5 measured. bf16 has no such bundle and does not move. Of the other
  pairings the placement change could affect, none exists in the kernel: its only other paired loads
  are A+A inside the hardware loop, which sit in one bank whatever the placement. See B4.
  `tools/aie_bank_check.py` cannot flag this hazard, for two reasons: it drops the stack from its bank map
  (`n != STACK_SYM`, `aie_bank_check.py:169`), and it looks only for load+load pairs (`PORT_A` +
  `PORT_B`), never for a load paired with a store. Both are tool gaps. The engine's own core has the
  same shape. In `modnet_cut_dense_20260921`'s `engine.o`, 27 bundles pair a stack access with another
  memory op, including vector spill reloads beside accumulator stores (`0x624`, `0x1d44`:
  `vlda wl8, [sp, #-0x160]` | `vst …, [p0]`). The stack is bank 0 of the core's own window (0x70000),
  and several of its buffers sit in a neighbour's memory (0x6xxxx), which cannot conflict with it. Whether any
  pair conflicts depends on where `p0` points at run time: **unchecked**, a follow-up.

A first sitting (`l1_tile_compiler_vs_hand_desktop2_20260923T0527Z.log`, `index` counter, no
control) measured the same kernels 5.5–6.1 cycles/call slower each. That shift is exactly the
`index`-vs-`i32` control difference (25.4 − 19.3, `l1_tile_empty_indexloop_desktop2_20260923T0530Z.log`).

## D. Novelty

A prior-art sweep (2026-09-23) searched for each candidate. Rows marked **✓fetched** rest on
a page this audit fetched and quoted on 2026-09-23. Rows marked *listing* rest on the sweep's search result or fetch
only and should be re-fetched before they are quoted elsewhere. "Contemporaneous" means dated 2026-06 or later:
independent corroboration, not prior art.

**Verdicts:**
- NEW: no prior art found.
- NEW-FOR-OPEN-XDNA1: done elsewhere, but not on Phoenix/Hawk Point with an open toolchain.
- PRIOR-ART: cite it.
- NEGATIVE-RESULT: a documented failure.

| # | Candidate | Verdict | Closest prior art |
|---|---|---|---|
| D1 | A whole CNN (YOLOv8n/s, pose, SESR) on XDNA1 through an open toolchain with **zero CPU partitions and zero ORT calls per frame** | **NEW-FOR-OPEN-XDNA1** | mlir-aie's `programming_examples/ml/resnet` offloads only conv2_x; `magika` runs two of three groups (*listing*). Riallto's ML inference goes through the VitisAI EP (*listing*). AIE4ML (arXiv 2512.15946) is whole-network and bit-exact, but on Versal and MLPs only (*listing*). open-xdna, xdna-engine and rlx-xdna are contemporaneous or XDNA2 (*listing*). AMD's own EP runs whole CNNs, but closed |
| D2 | One fixed core program driven by per-layer op headers and generated DMA/transaction streams | **PRIOR-ART** (the concept) | ✓fetched Rösti & Franz, arXiv 2504.03083 (2025-04-03, Phoenix 7940HS, IRON): "By using the same tile size m, k, and n for all variations, we completely eliminate the need to reconfigure the compute (L1) and memory (L2) cores … Only the shim cores and two runtime parameters in each core require reconfiguration." AMD's VitisAI EP is itself a fixed overlay driven by instruction streams (*listing*). The 128-byte op header of a *conv* core is an implementation, not a new idea |
| D3 | Bit-exact integer oracle for the engine, validated layer by layer on silicon | NEW-FOR-OPEN-XDNA1 (weak) | Bit-exact AIE verification is standard practice: AMD's x86 functional simulator, and AIE4ML's flow (*listing*). No silicon-validated per-layer NumPy oracle for a conv engine on Phoenix was found |
| D4 | 5 physical / 4 reachable columns | **PRIOR-ART** | ✓fetched Riallto: "column zero in the array (left most) does not contain an interface tile", with 4 interface tiles and 20 compute tiles. ✓fetched arXiv 2512.13282: "Due to the absence of a ShimTile in the last column of XDNA, we choose to map GEMM across 4 rows and 4 columns". The Linux `npu1_regs.c` sets `first_col = 1` (*listing*) |
| D5 | Core clock **1.80 GHz** | PRIOR-ART as a value; the per-pmode measurement and the method are **NEW** | ✓fetched tnzr's kernel page benchmarks at the maximum clock under `--pmode turbo`, and the Jena post (2026-01-21) gives 398 GFLOPS as "86% of peak", i.e. 1.8 GHz. Earlier public sources assume 1 GHz: ✓fetched Riallto ("clocked at 1 GHz") and ✓fetched 2512.13282 ("1 GHz"). This repo's sweep (0.80 / 1.03 / 1.80 GHz by pmode, from trace-unit timestamps) has no public equivalent found. It also shows that default pmode already runs at 1.80 GHz, and the reproduction confirms that: 397.5 GFLOPS in default versus the reference's 398 in turbo |
| D6 | Dispatch floor: ~75 µs amortized ring vs 531–617 µs one-shot | **NEW-FOR-OPEN-XDNA1** | 2512.13282 gives only whole-design reconfiguration: ✓fetched "3.4 ms" on XDNA. mlir-aie #3793 (npu2, ~400–450 µs one-shot, 64 µs persistent) was opened 2026-09-23 and is contemporaneous (*listing*) |
| D7 | AIE-ML microarchitecture facts: 6 slots, latencies, accumulator file, hardware loops | **PRIOR-ART**, and older than tnzr | llvm-aie, public since ✓fetched 2024-04-22, defines the slots (`AIE2Slots.td`) and the itineraries (`AIE2Schedule.td`, `peano_aie2_machine_model_a36c62b9.log`). ✓fetched UG1603 (2026.1, released 2026-07-08) documents `am0–am8` / `bm0–bm8` / `cm0–cm8`. tnzr (public 2026-01-21) states the slots, the latencies and the loop rules |
| D8 | Same-bank paired **load+load** costs +1 cycle; same-bank **load+store** costs +1 cycle | **NEW** as numbers | tnzr says "loads and stores in separate instructions to avoid bank conflicts" without quantifying (✓fetched kernel page). No public number was found (*listing*: llvm-aie `AIE2Subtarget.cpp`, AM020 queries). This repo measured both: `bank_conflict_survey.log`, and B4 here |
| D9 | GEMM efficiency on XDNA1 | **PRIOR-ART, and the published results beat this repo's** | ✓fetched Taka, Rösti, Melber, Vasireddy, Denolf, Marculescu, arXiv 2512.13282 (2025-12-15): XDNA int8 up to 6.76 TOPS, bf16 up to 3.14 TOPS at an assumed 1 GHz. Single core: **233.0 int8 and 112.6 bf16 MACs/cycle** (91% and 88% of peak). This repo's best bf16 array figure is 2,700.44 GFLOPS, and its upstream `mm.cc` kernel reaches 64 bf16 MACs/cycle in its loop (50% cap) and 48.3 whole-call at 64³ (37.7% of 128). The published single-core figure is 2.3× that. (Theirs is a reported per-core rate; the shape and harness differ from `l1_tile_bench`'s.) tnzr's single-tile 398 GFLOPS is the hand-scheduled equivalent. GAMA (Versal AIE-ML, 85–86%) and MaxEVA (AIE1) are *listing* |
| D10 | bf16 convolution on XDNA1 through an open toolchain (branch `worktree-bf16-engine`: "This is a first") | **PRIOR-ART** for "first open bf16 conv on XDNA1" | iree-amd-aie registers `conv_2d_nhwc_hwcf` bf16→f32 for target `npu1_4col` since 2024-09-24 (*listing*: the test definition, not a confirmed CI run). "First through mlir-aie/IRON on npu1" might survive; mlir-aie's `aie_kernels/aie2` has no bf16 conv2d |
| D11 | Native int8×int4 on AIE-ML at 512 MAC/cycle (branch `worktree-int4-study`) | **PRIOR-ART: retract the "widespread assumption" framing** | ✓fetched AIE-API 2024.1: AIE-ML "8b x 4b: 4x16x8, 8x16x8a, 4x32x8ab", legend "a - Emulated … b - Require additional data manipulation", so 4×16×8 is native. ✓fetched Riallto: "an AIE can achieve 512 MAC/cycle when operating using int4 x int8". The silicon probe on the branch is a confirmation, not a discovery. Main's `SILICON.md` "AIE2p only" is therefore **CONTRADICTED** by SPEC |
| D12 | Byte-for-byte reproduction of Quark XINT8 and AdaRound | **NEW** | None found (*listing*). Bounded: Quark is open source, so the reference implementation is readable |
| D13 | Native BO splicing across separately compiled designs, preserving `DDR_PATCH` | NEW (narrow) | XRT's `xrt::runlist` chains runs with shared buffers at the API level (*listing*). mlir-aie #3653 hit the same group-ID trap and is contemporaneous (*listing*) |
| D14 | A hand-scheduled kernel runs exactly as scheduled (282 = 288 − 6 cycles), and on an identical tile it is 3.0× upstream `mm.cc` | **NEW** as a measurement | tnzr reports GFLOPS, not a cycle-exact match, and checks no output. No public like-for-like L1 comparison of hand vs compiled on XDNA1 was found. It confirms tnzr; it is not independent of it |

Chronology settles the specific claim. `RESEARCH.md:1854` says the six issue slots per bundle were "stated
nowhere before". llvm-aie's `AIE2Slots.td` has been public since 2024-04-22, and tnzr since 2026-01-21 (its index page:
"VLIW instruction set supporting up to six operations per cycle"). This repo's ISA work is dated 2026-09.
The claim is **CONTRADICTED** as written. "Which no document in this repo stated" (`results/aie/README.md:284`)
remains true.

## E. Provenance of the code that runs on the tiles

- **One fixed core program:** `kernels/aie2/conv_engine/engine.cc`. AIE-API C++,
  `aie::mmul<4,8,8,uint8,int8>`, compiled by Peano `-O2` through IRON's `ExternalFunction` and aiecc.
  It is written in this repo, and its ELF is byte-identical across models. Per model, only the weight packets change (a
  128-byte op header interpreted by `run()`) and the DMA runtime sequence (`insts.bin`).
- **Written here:** all of `src/ignite_xdna` (lowering, workspace and packet scheduling, MemTile BD
  words, the `.ignite` container, the ring scheduler, native BO splicing, the bit-exact emulator,
  the AVX2 host paths).
- **Taken from mlir-aie/IRON/aiecc:** placement, routing, core compile/link, and transaction encoding.
- **Not used anywhere:** Chess, prebuilt AMD or Riallto binaries, hand assembly on any shipped path.
