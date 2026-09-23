# Cross-audit against "Hello XDNA!" (tnzr.org): what this repo got wrong, what it never knew, what is new

**Date:** 2026-09-23 (Desktop 2, `DESKTOP-CBL5NUA`, Ryzen 7 8700G, the same part the reference measured on)
**Reference:** T. Steinert and A. Breuer (Uni Jena), *Hello XDNA!*, <https://tnzr.org/xdna/>,
source <https://github.com/scalable-analyses/xdna> (first release 2025-12-18, revised 2026-08-31).
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
| A1 | The accumulator file is stated four ways: 9×1024-bit `cm0–cm8` (SILICON); 8 × `cm0–cm7` (two notes); **"only 6 accumulator registers"** (kernels/README, `conv2dk1.cc`), not marked superseded. The "5 live is the spill-free ceiling" is contradicted by the repo's own engine kernels holding 8 spill-free | ✓ `docs/SILICON.md:77`; ✓ `kernels/README.md:290`; `results/aie/notes_im2col_kernel_vliw_audit.md:280`; `notes_accumulator_cascade_gemm_model.md:73`; `kernels/conv_accum/aie2/conv2dk1.cc:146` | **9 is CONFIRMED**, in both the compiler and the hardware. "6" and "8" are **STALE**. "5 = ceiling" is **REFINED**: it is the probe's vector-register pressure (fresh A and B vectors per MAC), not the accumulator file | SPEC(Peano): `results/aie/peano_aie2_machine_model_a36c62b9.log` §1 gives `cm0`–`cm8`, each `[bml_i, bmh_i]`, plus 9 each of `amll/amlh/amhl/amhh`. By use: `results/aie/engine_core_issue_census_desktop2_20260923.log` shows all four engine ELF versions naming the ninth accumulator 26 times, and the engine runs bit-exact on silicon, so `cm8` exists in hardware. A direct probe (P1) would only restate this |
| A2 | "`vshift`/`vmov` spend slot `[v]`… cannot co-issue with `vmac`" | ✓ `docs/SILICON.md:264,305-309` | **CONTRADICTED** by the repo's own disassembly and by Peano | ✓ `results/aie/conv_issue_rate_decomposed.log`: `0x072a vmov wh10, wl5 \| vmac cm1…`; `0x075c vshift x8… \| vmac cm3…`; `0x0780` carries `vldb`, `vlda`, `add`, `vshift` and `vmac` in one bundle. SPEC(Peano), `peano_aie2_machine_model_a36c62b9.log` §3: `vshift`, `vshift.align`, `vshuffle`, `vbcst` and register `vmov` issue in the **mv** slot; `vmac`/`vmul` in **vec**. What realignment costs is the mv slot plus 2-cycle latency, and nop bundles when it sits on the critical path, not the vector slot |
| A3 | int8×int4 is a native `vmac` at 512 MAC/cycle; `vldb.unpack.s8.s4`; "a widespread assumption that native sub-byte execution was exclusive to AIE2P" | `results/aie/notes_w4a8_aie2_roofline.md:8,54`; `TODO.md:530-534`; against `docs/SILICON.md:66,1023` ("AIE2p only") | **UNBACKED ON MAIN**, not unbacked. ✓ `w4a8_probe_npu.log` and `w4a8_array_npu.log` exist in commits `56b4e88` and `2fe7624` on the unmerged branch `worktree-int4-study` (2026-09-10). Main's SILICON still says "AIE2p only". The "widespread assumption" framing needs the prior-art check (D) | Branch logs; Peano `mmul_8_4.hpp` (cited by the note itself) |
| A4 | **"YOLOv8n Complete End-to-End Monolithic Forward Pass … Demolishing AMD's 6.61 ms Baseline", 1.732 ms, 577.3 FPS, 3.82×, Layers 0–22, 0 CPU partitions** | ✓ `benchmarks/full_forward_latency_results.md:1-37`; ✓ `results/benchmarks/yolov8n_full_forward_silicon.log`; commit `c2e6e46` (2026-09-13) | **RETRACT.** Four independent defects, all read from the benchmark's own source: (1) ✓ `benchmarks/benchmark_yolov8n_full_forward.py:178-180` **hardcodes** Backbone = 0.709 ms and Neck = 0.407 ms and sets Detect Heads to the remainder (the logged percentages sum to 104.97%). (2) ✓ Lines 189-225: the "parity" 0.9965 compares the **CPU** ONNX cut model's heads, decoded, with the float ONNX model; the NPU output `raw_heads_hw` is only tested for being non-empty. (3) ✓ Lines 43, 45: the AMD "baseline" 6.61 ms and "unoptimized" 23.42 ms are constants, not measured in the run. (4) ✓ `docs/BENCHMARKS.md:7685` (same day): the shipped `build/yolov8n.ignite` has a 4,096-byte egress, no `head_layout`, zero-filled heads. The measured whole-network result is the later graph-engine container (`BENCHMARKS.md:7754`) | The benchmark's own source and log |
| A5 | `aie2/` row: "20-core full-array engine … M=2 … 1.000 vmac/cycle, **100% vector ALU saturation**"; "18.43 TOPS … compiler-verified" | ✓ `kernels/README.md:52`; `results/aie/notes_im2col_20core_array_synthesis.md:18,190` | **REFINED/over-claimed.** 1.000 vmac/cycle is a *static* inner-loop count; 18.43 TOPS is peak arithmetic. On silicon ✓ `results/aie/hardware_im2col_execution.log:11-14`: 20 cores ERT_CMD_STATE_TIMEOUT; 16 cores 0.0270 TOPS at **0.37%** density; 1 core 0.80% | Repo's own hardware log |
| A6 | "0 BYTES (100% L2 MemTile SRAM)" at `0x40000`/`0x60000` via Locks 4/5 | `docs/BENCHMARKS.md:7467-7497`; `TODO.md:562-572` | **Unreconciled** with `docs/LOW_LEVEL_AUDIT.md:29-32` (the Lock 4/5 init branch "could never fire") and `:146-153` (the shipped streams never touch those addresses). Needs a pointer in place | LOW_LEVEL_AUDIT |
| A7 | Per-core int8 peak computed at 128 MAC/cycle (460.8 GOPS/core) | `results/aie/notes_im2col_hardware_bringup.md:136-138` | **CONTRADICTED** by SPEC 256 MAC/cycle int8 (`docs/SILICON.md:64`). Every %-of-peak in that note is 2× too high | SILICON SPEC row |
| A8 | Stale text still printed: slot b = "branch"; "nine register names are a lower bound"; "nominal tile clock of 1.0 GHz"; the H11 2×2 reblock under "Whole-Call Rate" as if achieved; 6 vs 8 `vshift` count for one loop | `docs/BENCHMARKS.md:1500,1553`; `notes_a16w8_feasibility_audit.md:75`; `notes_memtile_4d_im2col_specification.md:278`; `docs/SILICON.md:278,282` | **STALE** | Corrections already in-tree (`SILICON.md:73`; `aie_disasm.py:21-22`; `conv_issue_rate_decomposed.log:105-123`) |
| A9 | "ERT timeout on Column 4 … Columns 0–3 complete physically addressable grid" | `results/aie/README.md:1109` | **Numbering clash** with SILICON's logical/physical split (logical 0 = physical 1) | `docs/SILICON.md:51-55` |
| A10 | "Peano ignores the pipelining macros" (exploration) vs "the `AIE_LOOP_*` hints are live" (branch) | `aie_kernel_utils.h` (both copies); commit `56b4e88` on `worktree-int4-study` | **REFINED.** ✓ Peano predefines `__AIECC__`, so `aie_kernel_utils.h:32-57` applies. `AIE_LOOP_UNROLL`, `AIE_LOOP_MIN_ITERATION_COUNT`, `AIE_TRY_INITIATION_INTERVAL` and `AIE_PREPARE_FOR_POSTPIPELINING` are live clang pragmas. `AIE_PREPARE_FOR_PIPELINING` and `AIE_LOOP_FLATTEN` expand to nothing, so every use of those two in this tree is a no-op | The header itself; `clang++ -dM -E` in `56b4e88` |
| A11 | Program memory: 13,424 B used | `docs/DECISIONS.md:2054` | **STALE.** The newest engine ELF's `.text` is 16,160 of 16,384 B (224 B headroom); one variant sits at 16,368 B (16 B free) | `engine_core_issue_census_desktop2_20260923.log` |
| A13 | "One fixed core program … byte-identical across every model" | exploration; engine docs | **REFINED.** It holds within an engine version. `build/conv_engine/` holds **four** distinct core ELFs across 68 builds. 56 current builds share `45aeec4edc4f` (2026-09-18..21). **`yolov8n_full`**, the build behind `build/yolov8n_full.ignite`, is in the older `eaddc8c66b88` group (2026-09-17, `.text` 10,704 B), together with `yolov8n_opt`, `yolov8s_opt` and `resnet50_head`. Its conv loops issue at the same rates (0.242–0.348), so the issue-rate finding covers every container | `engine_core_issue_census_desktop2_20260923.log` |
| A12 | Licence: `src/` is "Apache-2.0 WITH LLVM-exception", and 86 tracked files written here carry `Copyright (C) 2026 Advanced Micro Devices, Inc.` | `pyproject.toml:11`; `README.md:249-250`; file headers | **CONTRADICTED.** The whole repository is **GNU AGPL-3.0-or-later** (the user, 2026-09-23). The AMD line was pasted from mlir-aie's template onto original work. Genuine upstream copies (`conv2dk1*.cc`, `mm_2x2.cc`, `zero.cc`, `aie_kernel_utils.h`, `whole_array_bankpad.py`) keep AMD's notice | `LICENSE`; git authorship |

## B. What this repo never knew (NEW-FACT unless marked)

| # | Fact | Status here before this audit | Settled by |
|---|---|---|---|
| B1 | `VMAC.F` 6-cycle latency with the accumulator read in cycle 3. **Refinement neither source stated:** int8 `VMAC` is **5** cycles, its accumulator operand is bypassed (`VEC_Bypass`), and Peano issues a dependent int8 `vmac` **2 bundles** after its producer, against **4** for bf16 `vmac.f`. One accumulator therefore sustains a MAC every 2 cycles in int8 and every 4 in bf16; 1 vmac/cycle needs ≥2 (int8) or ≥4 (bf16) independent accumulators. The engine holds 8, so its loops are not latency-bound | Silent. Nothing reasons about VMAC latency or forwarding | SPEC(Peano): `peano_aie2_machine_model_a36c62b9.log` §3 (`II_VMAC [5,3,1,1,1]` with VEC_Bypass, `II_VMACf [6,3,1,1,1]`) and §4 (Peano's own spacing: `bf16_mac2` 4 bundles, `i8_mac2` 2). MEASURED P2: PENDING |
| B2 | Latencies: VLDA/VLDB 7 (the repo had scalar LDA 7 only), ST/VST 1, VSHUFFLE/VSHIFT/VMOV 2, MUL 2, `ret lr` with 5 delay slots. Accumulator load → first `vmac.f` 5 bundles; `vmac.f` → store 6; int8 `vmac` → store 5 | Scalar load CONFIRMED (`SILICON.md:76`); the rest silent. `asm_core_id.s` puts five NOPs after `ret lr` without saying why | SPEC(Peano) §3–§4. Peano's schedule of `bf16_mac` (§4) is bundle-for-bundle tnzr's Listing 2, so this Peano and theirs agree |
| B3 | Only load unit A writes accumulators; ≤1 accumulator load/store per bundle | Consistent with every disassembly here, never stated | SPEC(Peano) §3: every accumulator `vlda` (`II_VLDA_AM`, `II_VLDA_CONV`) reserves `LOAD_UNIT_A` and issues in slot lda; `vldb` has no accumulator form; accumulator stores are slot st |
| B4 | A load and a store to the same bank in one bundle conflict; the reference keeps them in separate instructions | The repo measured only load+load (+1 cycle, `SILICON.md:74`) | MEASURED P4: PENDING |
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

The repo attributes its bf16 GEMM's "remaining two thirds" to nothing (`SILICON.md:418-425`) and never
compared it with the 50% cap of the loop it runs. Its own production core issues a `vmac` on at most
~1 bundle in 3: loads, then `load_unaligned_v` `vshift` realignment, then idle, then 8 `vmac`s, with
no overlap between iterations. The reference shows the same silicon sustaining a `VMAC.F` on 256 of
288 bundles. The like-for-like split of *schedule* from *data movement* is phase 2.2: PENDING.

## D. Novelty

Filled in from the prior-art sweep (phase 1.3). Chronology already settles one claim.
`RESEARCH.md:1854` says the six issue slots per bundle were "stated nowhere before". The reference
published a six-slot VLIW description on 2025-12-18, and this repo's ISA work is dated 2026-09. That
claim is **CONTRADICTED** as written. "Which no document in this repo stated"
(`results/aie/README.md:284`) remains true.

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
