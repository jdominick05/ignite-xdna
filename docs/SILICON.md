# SILICON — what the XDNA1 array physically is, and what that makes possible

> A new file class, added 2026-09-07. `docs/BENCHMARKS.md` records what was measured,
> `docs/DECISIONS.md` records what was decided, `RESEARCH.md` carries the thesis and the
> open questions. This file is the layer underneath all three: an inventory of what the
> Phoenix AIE array physically contains, the ceilings that follow from it, where every
> number this repo has measured sits against those ceilings, and the objectives that are
> physically reachable on this silicon — whether or not the tooling to reach them exists
> today. Where it doesn't, building the tooling is the work item, not the reason to stop.

**Provenance.** Desktop 2 (Ryzen 7 8700G, Phoenix XDNA1; the laptop's Hawk Point is
untested for everything below). Repo state: `main` at `ce3e773`, `docs-restructure` at
`80770d3`. Toolchains: Ryzen AI 1.7.1 (VitisAI EP + `phoenix\4x4.xclbin` / `1x4.xclbin`),
`Xilinx/mlir-aie` v1.4.2 (IRON + Peano) in the ironenv, XRT SDK 2.21.75, NPU driver
32.0.20101.3760. If any of those move, re-check the SPEC rows before trusting a DERIVED one.

## How to read this

Every figure carries one of four tags, and the tag is the claim:

- **MEASURED** — a log under `results/` on this machine, path given. The repo's standing
  evidence rule applies unchanged.
- **SPEC** — read from a vendor or toolchain source file already on this machine, path
  given: AMD's `device.yaml` (excerpted verbatim in
  `results/aie/notes_aie2_device_dtypes.log`), or mlir-aie v1.4.2's target model
  (`include/aie/Dialect/AIE/IR/AIETargetModel.h`, `lib/Dialect/AIE/IR/AIETargetModel.cpp`)
  and kernel library. A SPEC row is what the toolchain *believes* about the chip; it has
  been right every time this repo has tested one, but it is not a measurement.
- **DERIVED** — arithmetic shown inline from MEASURED and SPEC inputs, assumptions named.
- **TO VERIFY** — physically checkable on this machine, and nobody has.

**The clock was not measured when this file was written; it is now.** Every per-second
ceiling below depends on the AIE core clock, and on 2026-09-06 nothing in this repo
measured it: `RESEARCH.md` cited 1.6 GHz for the 8700G from a web search,
`results/aie/bottleneck_spatial_sweep_npu.log` assumed 1 GHz and said so, and `xrt-smi
examine -r platform` prints no clock at all. So per-cycle figures are the primary SPEC
numbers and every per-second figure below is given at both 1.0 and 1.6 GHz, as originally
derived. Objective S0 then measured it (2026-09-07, section 1.7): **1.80 GHz** in the
`default`, `performance` and `turbo` power modes, **1.03 GHz** in `balanced`, **0.80 GHz**
in `powersaver` (`results/aie/clock_probe_npu.log`). The 1.0/1.6 columns are kept as
written and the 1.80 GHz value is added beside them wherever it changes a reading, with
the arithmetic; the conversion from a 1.6 GHz figure is the ratio 1.6 ÷ 1.8 = 0.889.
New numbers use 1.80 GHz and say which power mode they were taken in.

## 1. The physical array

### 1.1 Geometry

| Fact | Value | Tag and evidence |
|---|---|---|
| Physical columns | **5** | MEASURED: `xrt-smi examine -r platform` reports `Total Columns: 5` (re-run 2026-09-07, `results/aie/xrt_smi_platform_pmode.log`; first recorded in `docs/DECISIONS.md`). AMD's own `aiecompiler` derives `aie2_5x4_device` for the part string `xc10AIE24x5-die-1LP-e-S-es1` (`results/aie/aiecompiler_hostlib_fixed.log`, `results/aie/notes_aiecompiler_part_db.log`). |
| Rows per column | 6: one shim tile, one mem tile, four core tiles | SPEC: `BaseNPU1TargetModel::rows()` returns 6 ("1 Shim row, 1 memtile row, and 4 Core rows"); `device.yaml` `phoenix:` block has `num_rows: 4`, `memtile_rows: 1`. |
| Core tiles | 20 physical, 16 reachable today | DERIVED: 5 × 4 and 4 × 4. |
| Columns any path on this machine can drive | 4 | MEASURED: `4x4.xclbin` always lands on `Partition Index: 0, Columns: [1, 2, 3, 4]` (`results/gops_yolov8n.log`, `tools/session_hold.py`); `1x4.xclbin` exposes at most 4 partitions and a 5th process time-slices column 4 (`results/multi_partition_yolov8n_5col.log`); the driver's own `5x4_*.xclbin` overlays build, run, match CPU output and place 0 nodes on the NPU — fingerprint mismatch, `docs/DECISIONS.md`. SPEC: mlir-aie v1.4.2 models NPU1 as at most 4 columns (`python/iron/device/__init__.py:42`, `_MAX_COLS = {"NPU1": 4}`; `AIEAttrs.td` defines `npu1` as the 4-column "whole array" plus `npu1_1col..3col`). |
| Which column is the unreachable one | column 0, by elimination | DERIVED from the `[1, 2, 3, 4]` partition report. MEASURED 2026-09-07 that the numbering is physical: a Worker placed on IRON's logical `Tile(0, 2)` reports `get_coreid()` row 2, column 1, and its trace packets carry the same header — logical column 0 is physical column 1 (`results/aie/clock_probe_npu.log`). TO VERIFY: whether column 0's shim tile has a NoC DMA at all, or is compute-only (mlir-aie's "NPU1 has no ShimPL tiles" comment covers only the four columns it models). |
| `device.yaml`'s own `phoenix:` block | `num_columns: 4` | SPEC — AMD's cost-model config describes the 4-column overlay, not the die. The two AMD sources disagree with each other; `xrt-smi` and `aiecompiler` are the ones that talk to hardware. |

### 1.2 One core tile

| Resource | Value | Tag and evidence |
|---|---|---|
| Data memory | 64 KB, 4 banks | SPEC: `device.yaml` `core_data_memory: 64`, `core_num_banks: 4`; target model `getLocalMemorySize() = 0x10000`. MEASURED as a wall: every `bottleneck.py` width past 44 and every bf16 GEMM tile past `m=64,n=32`/f32 dies with `allocated buffers exceeded available memory` (`results/aie/bottleneck_spatial_sweep_npu.log`, `results/aie/bf16_matmul_ffn_shape_variants_npu.log`). The bf16 `n=64` tile needs 68,864 B — over by exactly the 3,328 B stack — while int8's fits at 52,480 B (`results/aie/int8_matmul_sweep_npu.log`). |
| Program memory | 16 KB | SPEC: `device.yaml` `core_program_memory: 16`. |
| MACs per cycle | int8×int8 **256**; bf16×bf16 **128**; int16×int8 **128** | SPEC: `device.yaml` AIE2 `macs_per_cycle`. |
| Adds per cycle | int8, int4: 64; bf16, int16: 32 | SPEC: `device.yaml` AIE2 `adds_per_cycle`. |
| Absent from the table | int16×int16, int8×int4, int16×int4, bfp16×bfp16 | SPEC: those appear only in the AIE2p (Strix) block. No vector fp32 multiply path is listed either; fp32 *accumulation* is native (`accfloat`), and `kernels/groupnorm_bf16/groupnorm_kernels.cc` gets fp32-grade products by splitting a coefficient into a bf16 hi part and a bf16 residual — two MACs, not one. |
| Vector load/store bus | 256 bits | SPEC: `getComputeTileLoadStoreBusWidth() = 256`. |
| Accumulator cascade to a neighbour | 512 bits | SPEC: `getAccumulatorCascadeSize() = 512`. Exercised on this chip by `02_vector_reduce_max`'s 4-core cascade (`results/aie/mlir_aie_examples_npu.log`). |
| DMA | 2 S2MM + 2 MM2S channels; 16 BDs; 16 locks | SPEC: `AIE2TargetModel::getNum{Dest,Source}SwitchboxConnections` (DMA bundle = 2 each way); `getNumBDs` = 16, `getNumLocks` = 16 for non-mem tiles. |
| BD fields | length ≤ 2¹⁴−1 words (65,532 B); 3-D addressing; 8-bit wrap; 13-bit step; 6-bit iteration wrap | SPEC: `getDmaBdMaxLen`, `getBDMaxDims`, `getDmaBdWrapBits`, `getDmaBdStepBits`, `getDmaBdIterBits`. MEASURED as a wall: a 64×688 bf16 B-tile (22,016 words) is refused with "exceeds the maximum of 16383 words supported by this tile type" (`results/aie/bf16_matmul_ffn_real_shape_npu.log`). |
| Neighbour memories a core can address directly | its own, west, north, and south unless the south tile is the mem tile | SPEC: `AIE2TargetModel::isLegalMemAffinity` (`AIETargetModel.cpp:809-823`). DERIVED: a core in the lowest core row sees 3 × 64 KB, the other three rows see 4 × 64 KB. |
| Stack | Peano defaults to 1024 B and grows *upward into tile buffers* | MEASURED: silent corruption until `Worker(..., stack_size=2048)` (`kernels/attention_bf16/README.md`). |
| Issue width | 6 slots per bundle: **`b` and `a` are the two load units**, `s` store, `x` scalar and control flow, `m` move/broadcast, `v` vector | MEASURED 2026-09-09 (`results/aie/aie2_isa_static.log`, corrected by `results/aie/bank_conflict_survey.log`). **Supersedes this row's earlier "`b` branch"**, which was inferred from the nop mnemonic alone. Tabulating every operation appearing in each slot of 226 *strictly six-field* bundles — the only encoding whose slot identity is unambiguous — puts `vldb` and `paddb` in slot b, `vlda`/`lda`/`mova` in slot a, and `ret` in the scalar slot x. So the core has **two load units**, which is what makes a same-bank paired load possible at all. `nopxm` is the fused encoding when x and m are both idle, so a five-field bundle still occupies six slots; bundles using few slots are emitted compressed and shorter than 16 B but still issue in one cycle. |
| Local memory banks | 64 KB of core-tile data memory in **4 banks of 16 KB**. Two loads issued in one bundle cost **one extra cycle** when both address the same bank | SPEC: `getLocalMemorySize()` 0x10000 and `getNumBanks()` 4 for a core tile in mlir-aie's `AIETargetModel.h`. MEASURED 2026-09-09 on branch `research/windows-lowlevel` (`results/aie/memory_desktop2_20260909_m01_*.log`), holding the compiled function bytes identical and changing only operand addresses: two loads same bank 12.0 cycles/iteration, separate banks 11.0, one load same bank 11.0, r² 1.0. Operand offsets of 64/128/256 B within a bank make no difference, so the granularity is the bank. Buffer addresses are not in the cached `aie.mlir`, which is pre-allocation; they are in the core ELF's symbol table (`tools/aie_bank_check.py`). |
| Cycles per hardware-loop iteration | equal to the loop body's bundle count | MEASURED 2026-09-09: the core is a statically scheduled VLIW with an exposed pipeline, so Peano covers operand latency with explicit nop bundles rather than an interlock. S0's two loops measured 9.000 and 2.000 cycles/iteration and disassemble to 9 and 2 bundles (`results/aie/aie2_isa_static.log`). An inner loop's cost is therefore readable before the kernel runs; a loop that waits on a lock, stream or DMA is the exception and needs the trace unit. **A fourth exception, added 2026-09-09:** a bundle issuing two loads whose addresses share a 16 KB bank costs one extra cycle, so the law holds for a loop whose simultaneous loads are in *different* banks (`results/aie/bank_conflict_survey.log`). The production int8 GEMM violates this and its 9-bundle loop should cost 10. |
| Scalar load-to-use | result available to the 7th bundle after the load issues | MEASURED 2026-09-09: six all-nop bundles separate the `lda` from the `add` consuming it in S0's scalar loop, which is why that loop costs 9 cycles for one add (`results/aie/aie2_isa_static.log`). |
| Live accumulators | **5** concurrently-live 4×8×8 int8 `aie::mmul` accumulators compile with no stack traffic; 6 is the first count that spills. The allocator names 9 accumulator registers, `cm0`–`cm8` | MEASURED 2026-09-09 by sweeping the count and reading the object code (`kernels/acc_spill_probe/`, `results/aie/aie2_isa_static.log`). Supersedes this row's earlier "≤4 stays in registers" and `docs/DECISIONS.md`'s "only 6 hardware accumulator registers"; both were inferred from the one `conv2dk3` kernel that spilled at 8. One shape and one optimisation level. The production int8 GEMM demonstrates the ceiling — it holds 8 live `mmul` accumulators, uses all 9 names, and spills, with a 416-byte frame and 33 stack references (`results/aie/gemm_cost_model.log`). **Refined 2026-09-09 (`results/aie/accumulator_width_vs_count.log`):** the file is **9 registers addressable at three granularities** — full 1024-bit `cm0`–`cm8`, 512-bit halves `bml`/`bmh`, 256-bit quarters `amll`…`amhh` — so "9 names" is the file, not a lower bound on it. And **the ceiling is not a width budget**, which this row previously implied by saying a wider accumulator fits fewer: bf16's `matmul_vectorized_4x4` holds 16×512-bit for the *same* 8192-bit total across the *same* 8 of 9 registers, and spills **nothing** — a 64-byte frame whose 15 stack references are all scalar, against int8's 12 vector spills. int8's 4×2 blocking is the defect, not the width. |
| Cycle counter | `aie::tile::current().cycles()` → `get_cycles()` — **not reachable from a Peano kernel**; the trace unit reads the same timer | SPEC: `ironenv/Lib/site-packages/mlir_aie/include/aie_api/tile.hpp`. MEASURED 2026-09-07: Peano (llvm-aie 22) declares `get_cycles()` and never defines it (`ld.lld: undefined symbol`), does not lower `__builtin_readcyclecounter`, and rejects inline asm; S0 read the timer through trace-unit event stamps instead (`results/aie/clock_probe_npu.log`). Re-checked 2026-09-09 from the machine-code side and the conclusion holds by a fourth route: the assembler accepts `CORE_ID` as a `mov` source and no timer name at all, and the register database puts the timer at memory-mapped `0x340F8`/`0x340FC` in the tile's configuration space rather than in the core's data space (`results/aie/aie2_isa_static.log`). |
| Hand-written assembly | **Assembles and links.** A standalone `.s` never enters instruction selection, so it reaches the integrated assembler intact | MEASURED 2026-09-09 (`kernels/asm_probe/`, `results/aie/aie2_isa_static.log`). Only statement-level inline asm inside a C++ function fails, in the IRTranslator, which is the wall S0 hit. Hand-scheduling an inner loop is therefore available where the compiler's schedule is the binding constraint. Toolchain result: the object assembles, disassembles and links, but no hand-written kernel has been run on the NPU. |
| `aie::mmul` shapes in mlir-aie's GEMM library | bf16 4×8×4; int8 4×8×8; int16 4×4×4 | SPEC: `aie_kernels/aie2/mm.cc`, and the compiler's own dispatch table `_MM_MAC_DIMS` in `python/iron/kernels/linalg.py`. One AIE2 int8 `vmac` therefore retires 4·8·8 = 256 MACs, which is exactly the 256 MACs/cycle nameplate. The same table gives AIE2P (Strix) 8×8×8 for int8, i.e. 512 per `vmac` — the generational difference is the MAC shape, not the issue rate. |
| MAC issue rate, production int8 GEMM | **107.3 MACs/cycle** over a whole kernel call, 41.9% of the 256 nameplate; 88.9% inside the hardware loop alone | MEASURED 2026-09-09 by walking the compiled object's control flow and dividing the tile's required MACs by the cycles the nest costs (`tools/gemm_cost_model.py`, `results/aie/gemm_cost_model_nest.log`). Supersedes the same day's first reading of **198.1 / 77.4%**, which modelled `matmul_i8_i32` as one hardware loop with straight-line setup; it is a nest, two software loops around the hardware loop, re-running the accumulator load/store body once per group. The gap between 88.9% and 41.9% is those 87 non-loop bundles per group — accumulator load, store and stack spill — and it follows directly from the row above: this kernel holds 8 live accumulators where 5 is the spill-free ceiling. A computed issue rate, not a hardware-counter reading; the branch targets are unresolved relocations in the object, so the evidence for the nest reading is that its trip counts reconcile exactly against the required `vmac` count. Charging the same-bank paired load this kernel is now known to carry lowers it further to **103.3 MACs/cycle, 40.3%** (`results/aie/bank_conflict_survey.log`). |
| Numerics | `aie::set_rounding(conv_even)` needed to match host round-to-nearest-even; Peano's AIE libc has no float `sqrtf` | MEASURED: `results/aie/groupnorm_bf16_kernel_npu.log`. |

### 1.3 One mem tile

| Resource | Value | Tag and evidence |
|---|---|---|
| Memory | 512 KB, 8 banks | SPEC: `device.yaml` `memtile_capacity: 512`, `memtile_num_banks: 8`; `getMemTileSize() = 0x80000`. |
| DMA | 6 S2MM + 6 MM2S channels; 48 BDs (even channels use BDs 0–23, odd 24–47); 64 locks | SPEC: switchbox DMA bundle = 6 each way; `getNumBDs(MemTile) = 48`; `isBdChannelAccessible`; `getNumLocks(MemTile) = 64`. |
| BD fields | length ≤ 2¹⁷−1 words (the whole tile); **4-D** addressing; 10-bit wrap; 17-bit step; 6-bit iteration wrap | SPEC: same accessors. The 4-D BD is the one address generator on the chip that can do an im2col or a transpose in flight without a core touching the data. |
| Neighbour access | east and west mem tiles are addressable | SPEC: `isLegalMemAffinity`'s mem-tile branch. TO VERIFY on NPU1 hardware. |

### 1.4 One shim tile

| Resource | Value | Tag and evidence |
|---|---|---|
| NoC DMA | 2 S2MM + 2 MM2S channels; 16 BDs; 16 locks | SPEC: switchbox DMA bundle = 2 each way; `getNumBDs` = 16. MEASURED as a design constraint: `kernels/groupnorm_bf16/groupnorm.py` puts exactly two workers per column so each of the 8 MM2S and 8 S2MM channels on the 4-column device carries one stream, and had to ride the per-core parameters on the data fifo because no third channel exists. |
| BD fields | 32-bit length; 3-D addressing; 10-bit wrap; **20-bit step**; 6-bit iteration wrap; repeat count ≤ 255 | SPEC: same accessors plus `getMaxRepeatCount() = 255`. Address granularity is 32-bit words (`getAddressGenGranularity() = 32`). |
| Shim type | all four modelled columns are NoC shims ("NPU1 has no ShimPL tiles") | SPEC: `VirtualizedNPU1TargetModel::getTileType`. Column 0 is outside the model — see 1.1. |

### 1.5 Streams and the switch

| Fact | Value | Tag and evidence |
|---|---|---|
| Stream width | 32-bit streams, one word per cycle per stream | SPEC-adjacent: `device.yaml` `max_stream_bw: 4` carries no unit in the excerpt; it is consistent with 4 bytes per cycle per stream and nothing else obvious. **TO VERIFY** with the single-channel microbenchmark in S1 — see the 9% discrepancy in 1.6. |
| Switch features used or provable here | circuit-switched routes, packet-switched routes (packet id ≤ 31), broadcast from one master to several slaves, trace ports on core, mem and shim tiles | SPEC: `getMaxPacketId() = 31`; `WireBundle::Trace` slave ports exist on all three tile types (`AIETargetModel.cpp`, port-index tables). Broadcast is what lets `whole_array.py` feed one A-tile to a whole row of cores and one B-tile to a whole column. |
| Switch ports per tile (masters out / slaves in) | core tile: DMA 2/2, core 1/1, FIFO 1/1, north 6/4, south 4/6, east and west 4/4 (0 at the array edge); mem tile: DMA 6/6, north 6/4, south 4/6; shim: north 6/4, south (the NoC side) 6/8, east and west 4/4, FIFO 1/1 | SPEC: `AIE2TargetModel::getNumDestSwitchboxConnections` and `getNumSourceSwitchboxConnections`, read with their `case` labels. Consistent with the model's own validation rule that a tile's north masters equal the tile above's south slaves. |

### 1.6 Off-chip bandwidth

Three independent measurements, none of them a clean single-direction read test:

| Design | Channels per direction | What moved | NPU time | Rate | Tag and evidence |
|---|---|---|---|---|---|
| `00_memcpy` (passthrough, 64 MiB round trip) | 4 ("2 columns × 2 channels") | 64 MiB in and 64 MiB out | 2388.5 µs avg | 56.19 GB/s "effective" = read+write bytes ÷ time, i.e. **28.1 GB/s per direction** | MEASURED `results/aie/mlir_aie_examples_npu.log`; the per-direction split is DERIVED (2 × 67,108,864 B ÷ 2388.5 µs = 56.19 GB/s reproduces the log's own figure). |
| `groupnorm_bf16`, L=301056 | 8 | 19.27 MB read twice, written once | 1487.3–1546.6 µs | 37.4–38.9 GB/s combined; the read stream alone is **25.9 GB/s** (2 × 19.27 MB ÷ 1487.3 µs) with 13.0 GB/s of writes running concurrently | MEASURED `results/aie/groupnorm_bf16_kernel_npu.log`; split DERIVED. |
| `dispatch_floor` passthrough (shim→memtile→shim, one fifo in, one forwarded out) | 1 | 8 KB–32 MB, each payload in and back out | slope 0.0726 ns per round-trip byte | 13.78 GB/s of read+write bytes, i.e. **6.9 GB/s per direction on one channel** | MEASURED `results/aie/dispatch_floor_npu.log`; the script counts `n * 4 * 2` bytes per call (`kernels/dispatch_floor/measure_floor.py`), so the split is DERIVED from its own accounting. |

What those three say together (DERIVED). Two independent designs agree on the per-channel
rate: memcpy at 7.0 GB/s per shim channel per direction across four channels, the
passthrough at 6.9 GB/s on one. Eight channels did not get eight times that — the
GroupNorm design's reads total 25.9 GB/s, 3.2 GB/s per channel — so above roughly four
channels a shared cap near 26–28 GB/s per direction binds, DRAM or NoC side, not the
channels. `device.yaml`'s `dram_read_bandwidth: 16` / `dram_write_bandwidth: 16` carry no
unit; 16 bytes per cycle at 1.6 GHz is 25.6 GB/s, within 10% of both shared-cap
measurements, which is suggestive and nothing more. The per-channel figure is the
uncomfortable one: 7.0 GB/s at 4 bytes per cycle needs a 1.75 GHz clock, 9% above the
cited 1.6 GHz. So either the clock is higher than cited, or a shim DMA channel is wider
than one 32-bit stream, or `max_stream_bw: 4` is not bytes per cycle at all. S0 and S1
between them settle which.

S0 has settled its half (1.7): the clock is 1.80 GHz, so 7.0 GB/s is 3.9 bytes per cycle
— one 32-bit stream word per cycle, and `max_stream_bw: 4` reads as bytes per cycle after
all. The DRAM figure at 1.8 GHz is 28.8 GB/s, still within 10% of the 26–28 GB/s shared
cap. What S1 still owns is whether that cap is DRAM, NoC, or channel count.

### 1.7 Clock and power mode

| Fact | Value | Tag and evidence |
|---|---|---|
| Core clock | **1.80 GHz** in `default`, `performance` and `turbo`; 1.03 GHz in `balanced`; 0.80 GHz in `powersaver` (1.7983 / 1.8002 / 1.7998 / 1.0274 / 0.7985 GHz by the scalar-loop fit; 1.7990 re-measured in `default` after the sweep) | MEASURED `results/aie/clock_probe_npu.log` (2026-09-07, `kernels/clock_probe/`): `event0()`/`event1()` around a DMA-free loop, stamped by the tile's trace unit; the runtime's submit+wait time fitted against the stamped cycles across 2^18–2^25 iterations so the dispatch cost cancels (R² ≥ 0.999995). Two loops with different costs — 9.000 and 2.000 cycles per iteration, exactly constant at every length — agree within 0.33%. Assumes the trace timer ticks at the core clock. Evidence: a timer at k times the clock would need both 9/k and 2/k to be whole numbers, which only k = 1 satisfies; a timer at a fraction of it would put the 7.0 GB/s shim stream of 1.6 at under half of `device.yaml`'s 4 bytes per cycle, and it lands at 3.9. Supersedes: 1.6 GHz from a web search (RESEARCH.md, 2026-09-06) and the 1 GHz `bottleneck_spatial_sweep_npu.log` assumed. One core tile (logical (0,2)) measured; the concurrent-VitisAI-EP leg of S0 was not run. |
| Clock readback, live | **800 MHz idle in every power mode → the mode's clock while a hardware context is active: 1800 `default`/`performance`/`turbo`, 1028 `balanced`, 800 `powersaver`** | MEASURED: XRT's `xrt::device::get_info<max_clock_frequency_mhz>` (also `pyxrt`), sampled idle, every 3 s across a 30 s IRON GEMM run, and idle again, ~0.05 ms per read (`results/aie/xrt_api_live_clock_and_pdh_npu.log`). A readback, not a nameplate: the value follows context activity. `tools/hwinfo_npu_bridge.exe` shows it live and publishes it to HWiNFO. All five modes then sampled under load and idle in one sitting (`results/aie/pmode_clock_readback_npu.log`, 2026-09-08): busy, the readback matches the trace-unit clock of the row above to the MHz; idle, 800 in every mode — the clock-probe run's "800 in every mode" was an idle reading, and the two agree. See docs/DECISIONS.md, "RESOLVED 2026-09-08". |
| Live utilization | Windows' GPU-engine statistics see the NPU: 84–88 % across an IRON GEMM run, 0 % idle, adapter memory = xrt-smi's | MEASURED: `\GPU Engine(pid_*_luid_0x00000000_0x0000d6bf_*_engtype_compute)\Utilization Percentage` via PDH, and DXCore lists the adapter as "NPU Compute Accelerator Device" under `DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU` (same log). xrt-smi's GOPS/FPS/latency read `N/A` for the same context. Not yet confirmed under a VitisAI EP session. |
| Power mode | `Default`; `xrt-smi configure --pmode` accepts `default, powersaver, balanced, performance, turbo` | MEASURED: `results/aie/xrt_smi_platform_pmode.log` (a verbatim `--batch` capture of `xrt-smi examine -r platform` and `configure --help`, 2026-09-07; nothing on the device was changed). Exercised 2026-09-07 (`results/aie/clock_probe_npu.log`): all five modes switch from an unelevated shell; `powersaver`, `balanced`, `performance` cleanly, `turbo` with `[xrt-smi] ERROR: Failed to escape (0xc0000001)` printed on entry and on leaving, the report reading `Turbo` regardless and the clock matching `performance`. pyxrt's `device.get_info(max_clock_frequency_mhz)` reads 800 in every mode — the `powersaver` clock, not the live one. |
| A symptom nobody has attributed | the same model, same cache, measured 12.7 ms alone and 6.8–6.9 ms minutes later in a sweep | MEASURED `docs/DECISIONS.md` ("NPU single-instance latency drifts session to session"). Background CPU load was the suspect; an NPU clock or power state that changes with activity is the other candidate, and S0 tests it. TESTED at the 5 s scale (`results/aie/clock_probe_npu.log`): three calls each after 5 s of idle read 1.77–1.79 GHz, the same as back-to-back calls, so idle is not it at that scale. Power mode is a 2.25× clock lever, and no log in this repo records the mode a number was taken under; a mode change between sessions would produce exactly this symptom. |

### 1.8 On-chip storage, totalled

DERIVED from 1.2–1.3: 16 core tiles × 64 KB + 4 mem tiles × 512 KB = **3.0 MB** reachable
today; 20 + 5 gives **3.75 MB** if column 0 is reached. Program memory is 16 KB per core,
256 KB across the reachable array. For scale: yolov8n's head-cut int8 graph is on the order
of 3 MB of weights (TO VERIFY the exact byte count with `onnx-tool`'s `graph.params`, which
`tools/estimate_tops.py` already reads), so a fully weight-resident small CNN is at the
edge of what the reachable array can hold and comfortably inside the physical one.

## 2. Ceilings, and where every measured number sits

### 2.1 The nameplate, reproduced

DERIVED: 20 cores × 256 int8 MAC/cycle × 2 ops/MAC × 1.6 GHz = **16.38 TOPS**. That is the
"16 TOPS" AMD prints, and it only reproduces with all 20 cores at 1.6 GHz. Sixteen cores at
the same clock give 13.1 TOPS; at 1.0 GHz the two figures are 10.24 and 8.19 TOPS.
Consequence: **the 4x4 overlay's physical ceiling is 82% of nameplate**, so the repo's best
achieved figure — yolov8l on four independent `1x4` columns, 6.29 TOPS, 39.3% of nameplate
(`results/multi_partition_yolov8l.log`, `tools/estimate_tops.py`) — is 48% of what the
16 cores it ran on can physically do at 1.6 GHz.

At the MEASURED clock (1.7, `default` mode, 1.80 GHz): 20 × 256 × 2 × 1.8 = **18.4 TOPS**
for the full array and **14.7 TOPS** for the 16 reachable cores. The nameplate is what the
20-core array does at 1.6 GHz, and this part runs 12.5% faster than that in `default`; the
4x4 overlay's physical ceiling is then **92% of nameplate**, not 82%, and yolov8l's 6.29
TOPS is 43% of what its 16 cores can do. In `powersaver` every figure here is 4/9 of it.

### 2.2 The per-column denominator is a finding, not a convention

`results/percall_overhead_yolov8_1x4.log` divides by "an ideal 4-TOPS column" (16 ÷ 4). A
four-core column is 4 × 256 × 2 × f:

| Clock | int8 per column | bf16 per column | int16×int8 per column | int8 per core |
|---|---|---|---|---|
| 1.0 GHz | 2.05 TOPS | 1.02 TFLOPS | 1.02 TOPS | 512 GOPS |
| 1.6 GHz | 3.28 TOPS | 1.64 TFLOPS | 1.64 TOPS | 819 GOPS |
| **1.8 GHz (MEASURED, `default`)** | 3.69 TOPS | 1.84 TFLOPS | 1.84 TOPS | 922 GOPS |

The vendor DPU's yolov8l run on one column is 8.433 × 10¹⁰ MACs in 102.190 ms of node time
(MEASURED, that log) = **1.650 TOPS achieved per column**. Against the three denominators:
41.3% of "4 TOPS", **50.3% of a 3.28-TOPS column, 80.5% of a 2.05-TOPS column.** Which of
the last two is true is exactly the clock question — and the answer is neither: at the
measured 1.80 GHz the column is 3.69 TOPS and the DPU's 1.650 is **44.8%** of it. Either
way, 1.65 TOPS per column is a
measured, clock-independent bar that this silicon demonstrably sustains on int8 conv, and
it is the bar every open kernel in section 4 is held to.

### 2.3 int8 conv: the vendor kernel vs the open one, same silicon

| Kernel | Marginal rate, one column | Per core | Share of 819 GOPS/core (1.6 GHz) | Tag and evidence |
|---|---|---|---|---|
| VitisAI DPU, yolov8l, `1x4.xclbin` | 1650 GOPS (node time, not a fit) | 412 GOPS | 50.3% | MEASURED `results/percall_overhead_yolov8_1x4.log`; DERIVED split |
| mlir-aie `ml/bottleneck`, **accumulators in registers** | **348.4–350.3 GOPS** (two fits, r² ≥ 0.9987) | 87.6 GOPS | 10.7% | MEASURED `results/aie/conv_accum_residency_npu.log` |
| mlir-aie `ml/bottleneck`, h-axis sweep | 146.1 GOPS (fit, r² = 0.99999) | 36.5 GOPS | 4.5% | MEASURED `results/aie/bottleneck_spatial_sweep_npu.log` |
| mlir-aie `ml/bottleneck`, stock, re-measured 2026-09-09 | 115.6–117.1 GOPS (two fits) | 29.0 GOPS | 3.5% | MEASURED `results/aie/conv_accum_residency_npu.log` |
| mlir-aie `ml/bottleneck`, 56×56 | 111.1 GOPS (2-point fit) | 27.8 GOPS | 3.4% | MEASURED `results/aie/bottleneck_w56_npu.log` |
| CPU, ORT QDQ int8 on 8 Zen4 cores, same shapes | 819.0 (sweep) / 1678.8 (56×56) GOPS | — | — | MEASURED, same two logs |

At the measured 1.80 GHz the per-core denominator is 922 GOPS and the three shares read
44.8%, 4.0% and 3.0% (`results/aie/clock_probe_npu.log`).

The vendor-to-open gap, **11.3×**, is clock-independent because both ran on the same
column. That gap, not the CPU, is the real statement about the open int8 conv kernels: the
op class was closed against the CPU (12.75× at 56×56), and it was closed by a kernel
running at one-eleventh of what the same column does under AMD's compiler. At the DPU's
rate one column matches the CPU's 1678.8 GOPS and four columns are ~4× ahead of it.

**Most of that 11.3× was one line of C++, and removing it did not reopen the op class.**
Both 1×1 kernels held their four accumulators in a **runtime-indexed array**, which cannot
live in registers, so every `.mac()` was load-four-quarters / mac / store-four-quarters.
Peeling the `n == 4` case into named accumulators takes the hot loop from 22 bundles with
one `vmac` to 14 with four — **0.045 → 0.286 MACs/cycle** — and marginal throughput from
115.6–117.1 to **348.4–350.3 GOPS**, a **2.99–3.01×** gain reproduced across two series with
every shape verifying. The vendor gap narrows to **~4.7×**. But the CPU, measured in the same
sitting on the same shapes (ORT CPU EP, QDQ int8, VNNI), still runs at 823.7–839.5 GOPS, so
**the CPU still wins by 2.4× and the op class stays closed** — for a new reason. It was "the
kernel uses 5% of its issue slots"; it is now "even with the slots used, one column does not
reach a VNNI-equipped Zen4." MEASURED `results/aie/conv_accum_residency_npu.log`.

Two caveats travel with that. **The stock arm re-measured at 115.6–117.1, not 146.1**: the
published figure predates the 2026-09-07 width fix, which rewrote the same loop, so the
kernel is not the same code — the A/B above is like-for-like within one sitting, but 350 is
2.4× the last *published* figure rather than 3× it. And **conv2dk3, the 3×3 middle stage, was
not touched** and is the likeliest remaining rate-limiter.
Whether an open kernel can reach the DPU's rate is objective K1; nothing physical says it
can't. **And 2026-09-09 located where the 11.3× goes: MAC issue density.** The open kernels'
hardware loops issue 0.045–0.333 `vmac` per cycle against the int8 GEMM's 0.889 on the same
silicon — a 4×–20× shortfall that more than covers the throughput gap, with no appeal to data
movement (`results/aie/conv_issue_rate_decomposed.log`). The 1×1 keeps its accumulator in
memory rather than in registers; the 3×3 spends half its issue slots on `vshift` window
alignment. Both are kernel defects, not silicon limits.

### 2.4 bf16 GEMM

| Quantity | Value | Tag and evidence |
|---|---|---|
| 16-core bf16 peak | 4.10 TFLOPS at 1.0 GHz; 6.55 TFLOPS at 1.6 GHz; **7.37 TFLOPS at the measured 1.80 GHz** | DERIVED: 16 × 128 × 2 × f; clock MEASURED `results/aie/clock_probe_npu.log` |
| Best measured | 2072.54 GFLOPS (1024³, bf16 out) | MEASURED `results/aie/bf16_matmul_niche_npu.log` |
| Typical at production shapes | 1776.68 (2048×4096×4096, f32 out), 1800.86 (down-projection K=11008), 1846.96 (4096×2048×2048) GFLOPS | MEASURED `results/aie/bf16_matmul_attention_scale_npu.log`, `..._ffn_real_shape_npu.log`, `..._niche_npu.log` |
| Share of peak | 50.6% (1.0 GHz) / 31.6% (1.6 GHz) / **28.1% (measured 1.80 GHz)** at the best point; 43–45% / 27–28% / 24–25% at production shapes | DERIVED |
| Per core at the best point | 2072.54 ÷ 16 = 129.5 GFLOPS = 64.8 GMAC/s = 40.5 MAC/cycle of 128 at 1.6 GHz, **36.0 at the measured 1.80 GHz** | DERIVED |
| CPU bar (torch bf16, 8 Zen4 cores) | 1100.6–1362.8 GFLOPS across every shape measured | MEASURED, the same logs |

The NPU's measured edge here is 1.13–1.78× over the CPU, and it comes from running the
array at roughly a third to a half of its bf16 peak. Section 3.1 says why, and how much of
the rest is physically recoverable.

### 2.5 Dispatch

| Path | Fixed cost per call | Tag and evidence |
|---|---|---|
| IRON `@iron.jit`, wall | **617.0 µs** | MEASURED `results/aie/dispatch_floor_npu.log` (R² 0.9990) |
| IRON, host-side share | 447.3 µs, flat in payload, mostly XRT outside the submit bracket | MEASURED, same log |
| IRON, hardware bracket (submit + wait) | **169.8 µs** | MEASURED, same log (R² 1.0000) |
| VitisAI EP, host-side gap outside every ORT node (its Q/DQ nodes are a further 0.44 ms of real CPU work, also outside the compute node) | **0.089–0.091 ms** per call, every model size, one column | MEASURED `results/percall_overhead_yolov8_1x4.log` ("unaccounted gap (dispatch/sync)") |

What the last row does and does not show. It shows that the host-side term is software:
the vendor path spends about 90 µs per call outside its nodes where IRON spends 447 µs, on
the same driver. It does **not** show what the hardware floor is — the vendor's own
submit-to-completion latency sits *inside* its compute node's 12–102 ms, mixed with the
DPU's execution, and that log cannot separate it. Whether IRON's 169.8 µs hardware bracket
is silicon, or IRON's per-call runtime sequence (instruction buffer reload, BD
reprogramming, context bookkeeping), is therefore **open**, and objectives D1–D3 are the
measurements that close it. Until they do, the repo's go/no-go rule stands as measured: an
op must cost more than ~617 µs on the CPU to win through IRON today, ~170 µs on a bare
resubmit.

### 2.6 The three FFN DMA limits are field widths, not compiler bugs

`results/aie/bf16_matmul_ffn_real_shape_npu.log` found three limits chasing Llama-2-7B's
`d_ff=11008`. Each one lines up with a BD field in the target model:

| Limit as measured | Field in the target model | Match | Consequence |
|---|---|---|---|
| C-output stride cap "[1:1048576]" words, i.e. a fixed 4 MiB regardless of dtype | shim BD **20-bit step** at 32-bit granularity → 2²⁰ words = 4 MiB | exact | Silicon. Cannot be lifted; must be decomposed around (more BDs, a different layout, or staging through the mem tile). |
| B-tile "exceeds the maximum of 16383 words supported by this tile type" | core BD **length ≤ 2¹⁴−1** | exact | Silicon. `n` at `k=64` bf16 is capped at 511 on any core-tile BD. |
| "Too many simultaneously active buffer descriptors on tile (1,0), which supports up to 16", after the compiler split a repeat count of 86 into 64 + 22 | shim tile **16 BDs**; **6-bit iteration wrap** (max 64) | exact on both numbers | The 16-BD budget and the 64 wrap are silicon. How the compiler chains a longer repeat is the tooling half — the chunking strategy is what overflowed, and it is replaceable. |

The workaround space is therefore about *decomposition*, and the mem tile is the tool the
silicon provides for it: 48 BDs, 4-D addressing, and a 17-bit step of its own. Routing the
C output shim→mem tile→DDR, or reshaping the A reload so its repeat stays under 64, are
designs, not compiler patches. `--c-col-maj` was tried and traded this limit for the 64 KB
L1 wall at a worse operating point (`results/aie/bf16_matmul_ffn_shape_variants_npu.log`);
Mistral's `d_ff=14336` showed that the surviving `n`-tile size, not "real-shape-ness",
decides the verdict (1.10× CPU at 11008, 1.13× NPU at 14336).

## 3. Rooflines per op class

### 3.1 GEMM: the tile decides whether the core is fed

A core computing an m×n output tile over a k-step needs (m+n)·k·2 bytes of bf16 operands
for m·n·k MACs, i.e. **2(m+n)/(m·n) bytes per MAC**. Its two S2MM channels deliver at most
8 bytes per cycle if a stream is 4 B/cycle (1.5, TO VERIFY). At 128 bf16 MACs per cycle
the core can absorb 0.0625 B/MAC. DERIVED:

| Tile m×n (k=64) | B/MAC | Input-bound MAC/cycle | Ceiling as % of bf16 peak | Where it was measured |
|---|---|---|---|---|
| 64×32 (`whole_array.py` default) | 0.0938 | 85.3 | **67%** | every 1776–2072 GFLOPS row above |
| 32×32 | 0.125 | 64 | 50% | — |
| 16×64 (forced at N=11008) | 0.156 | 51.2 | 40% | 909.97 GFLOPS — 0.51× the default tile's 1793 at the same K |
| 16×128 (Mistral N=14336) | 0.141 | 56.9 | 44% | 1454.37 GFLOPS |
| 64×64 | 0.0625 | 128 | **100%** | **Reached in bf16 once the C tile is single-buffered** (`--c-single-buffer 1`, a local `whole_array.py` patch kept as `kernels/gemm_tile_sweep/whole_array_c_single_buffer.patch`): 2477.23 GFLOPS at 2048³ and 2641.41 at 2048×4096×4096 against the default tile's 1775.65 / 1801.18, the CPU-bf16 margin widening to 1.29×–1.89× (`results/aie/bf16_matmul_n64_single_buffer_npu.log`). Before that it was out of reach: the double-buffered f32 C tile alone is 2 × 16 KB, the whole set (A 8 KB, B 8 KB, C 16 KB, ×2) is exactly 64 KB with no stack, the `n=64` attempt failed on L1 (`..._ffn_shape_variants_npu.log`) and a second attempt measured the miss at exactly the 3,328 B stack (`results/aie/int8_matmul_sweep_npu.log`). **Reached in int8** without the patch, its operands being half the bytes (52,480 B with stack): 4447.97–4607.05 GOPS at the 2048-class shapes, 1.9× the 64×32 int8 tile, bit-exact. Clean re-run of the bf16 tile 2026-09-08: 2501.71 at 2048³, **33.9% of the 7.37 TFLOPS peak against this row's 100% ceiling** (`results/aie/gemm_tile_sweep_c_single_buffer_npu.log`) |
| 128×32 | 0.0781 | 102.4 | 80% | 2136.09 GFLOPS at 2048³ single-C, 29.0% of peak — while 32×128, the same B/MAC, reads 2494.61: the model has no term for B's DMA run length (`results/aie/gemm_tile_sweep_c_single_buffer_npu.log`, Table 3: with B column-major the two swap order) |
| 32×128 | 0.0781 | 102.4 | 80% | 2494.61 at 2048³ and **2700.44 at 2048×4096×4096, the best bf16 figure in this repo, 36.6% of peak**, tracking 64×64 within 2% at every shape (same log) |
| 128×64, 64×128 at k=32 | 0.0469 | 128 | 100% | 2070.87 / 2146.76 at 2048³ — below 64×64 at k=64 despite the better B/MAC: k is a term the model lacks (same log). At k=64 both miss L1 by 19,712 B even single-buffered, MEASURED in the allocator |

MEASURED against the model, 2026-09-08 (`results/aie/gemm_tile_sweep_c_single_buffer_npu.log`): the L1
arithmetic is exact (all 28 compiles) and 64×64 is the best tile, as it predicts, but two
terms are missing — B's
contiguous DMA run length (n elements row-major; `--b-col-maj 1` moves 128×32 +4.3% and
32×128 −12.6%) and k (tiles with a better B/MAC at k=32 land 14–17% below 64/64/64 at k=64; doubling k at the default tile's B/MAC gains 5.5%). And at
64×64 the array reaches 33.9% of peak against a 100% input-bound ceiling, so the remaining
two thirds are not input bandwidth: dispatch, the K-loop's C read-modify-write in f32, and
the 3.2 kernel are where K2 goes next.

The int8 column of the same table is the bf16 one at half the bytes per MAC (0.0469 at
64×32, 0.0313 at 64×64) against the same 8 B/cycle and 256 MAC/cycle, so 64×64 is exactly
input-balanced for int8 — and MEASURED it goes 2387.01 → 4607.05 GOPS at 4096×2048×2048,
18% → 35% of the 16-core int8 peak at 1.6 GHz, 16% → 31% at the measured 1.80 GHz
(`results/aie/int8_matmul_sweep_npu.log`, `results/aie/clock_probe_npu.log`).
The tile decides whether the core is fed, at both dtypes.

Two readings. First, the default tile is input-bound at two thirds of peak *before* any
other inefficiency, and the measured 27–32% (1.6 GHz; 24–28% at the measured 1.80 GHz)
sits at well under half of that bound — so
roughly half of the remaining gap is data movement and half is inside the core (C tile
read-modify-write over the 256-bit bus every k-step, loop overhead, pipeline fill). The
trace in S2 apportions those. Second, the levers the silicon offers are specific: keep the
running sum out of L1 (accumulate across k in registers or across cores over the 512-bit
cascade, so C is written once), share A or B between adjacent cores through neighbour
memory so one DMA stream feeds two consumers (halves the per-core input demand without
touching the switch), and pick tiles by the B/MAC column above rather than by what fits
the generic design's DMA pattern. The CPU bar is flat at 1100–1360 GFLOPS; the input-bound
ceiling for a 64×64 tile at 1.6 GHz is 6.55 TFLOPS, at 1.0 GHz 4.10, at the measured
1.80 GHz 7.37.

### 3.2 GroupNorm and InstanceNorm live at the DRAM cap

`groupnorm_bf16` reads its 19.27 MB tensor twice and writes it once in 1.487–1.547 ms
(MEASURED, `results/aie/groupnorm_bf16_kernel_npu.log`), a read rate of 25.9 GB/s that sits
within 8% of memcpy's independently measured 28.1 GB/s per direction (1.6) — at twice the
channel count, with 13 GB/s of writes running alongside. The kernel is at the shared
off-chip cap, and the compute — about six ops per element on a 128-MAC core — is
invisible. DERIVED: the op is ~1 op per byte moved, so on this array
it can never run faster than the off-chip cap allows, and the only two levers are physical:
a one-pass statistics scheme (shifted sums or Welford in fp32, 1 read + 1 write) gives at
most 1.5× at the same cap; fusing the normalisation into the epilogue of the conv that
produces the tensor gives all of it, because the tensor then never leaves the array between
the two ops. That is why the standing verdict — 33/49 nodes win in isolation and 0/49
survive a two-process handoff — is not a kernel problem. It is a graph-boundary problem,
and it is addressed by D4 and A2, not by another norm kernel.

### 3.3 Conv is compute-bound if the schedule lets it be

DERIVED for the block the repo measured (3×3, 64→64 channels, 56×56): 56·56·64·64·9 =
115.6 MMAC over 200 KB in, 36 KB of weights and 200 KB out — about 265 MACs per byte of
traffic, against the 16 MAC/B the core's input channels demand at 128 MACs/cycle (and 32
MAC/B at the int8 rate of 256). Conv has ~10× more reuse than the array needs, the whole
layer's weights fit in one core's L1 and the block's fit in a mem tile, so the 146 GOPS is
a scheduling and vectorisation result, not a bandwidth one — which is what the width-fix
work already found from the other direction (`aie::mmul` used correctly, but width-32
strides hard-coded and accumulators spilled). The DPU at 1.65 TOPS per column is the
existence proof that this silicon schedules conv at least eleven times better than the
open kernel does today.

### 3.4 Small ops and the floor

Below the dispatch floor nothing wins, however good the kernel: attention stages 3 and 4
(34 µs and 12 µs on the CPU), GroupNorm at L ≤ 18816 (233 µs), the whole of MobileNetV2
(1.72 ms on the CPU, 2.68 ms through the EP), MobileViT's stage-2 attention (240 µs).
MEASURED in `results/aie/dispatch_floor_npu.log`, `results/mobilenet/`, `docs/BENCHMARKS.md`.
The vendor path's 90 µs host-side gap (2.5) shows that the 447 µs host half of the floor
is software; whether the 169.8 µs hardware half is silicon is what D1–D3 measure, and every
small-op verdict in this repo is conditional on where it lands.

**MEASURED 2026-09-09, and it lands low: ~36 µs for batchable work.** Batched `pyxrt.runlist`
submission amortises the same passthrough to **36.3 µs** per dispatch — 17× below the 617 µs
IRON floor and a *quarter* of the 169.8 µs bracket, so the hardware half is not silicon
either (`results/aie/dispatch_runlist_npu.log`). (**Sitting label**, because two figures for this same measurement class are both live: 36.3 µs is the original 2026-09-09 pyxrt run; the same-sitting rerun in `results/aie/dispatch_cpp_runlist_npu.log` reads **35.9 µs**, and the C++ figure is compared against *that*, not against 36.3 — see this repo's rule that every config being compared must be captured together.) Of the six ops listed above, four now clear
the floor: MobileNetV2 by 48×, MobileViT stage-2 attention and bf16 attention stage 2 by 6.7×,
GroupNorm at L ≤ 18816 by 6.5×. Attention stages 3 and 4 (34 µs, 12 µs) remain under it.
**The caveat is the shape of the result:** 36 µs is a throughput figure that holds with 64
dispatches in flight, so it reopens batchable work only, and a one-shot latency-critical call
still pays ~140 µs raw or 617 µs through IRON. Those four verdicts are not overturned — the
floor has simply stopped being the reason they lose, which makes kernel quality the deciding
question for the first time.

**SCOPED the same day, and it matters for all four: ~36 µs is a raw-pyxrt figure.** Batching
was then wired into IRON's own host path and measured there
(`results/aie/iron_batch_npu.log`). The device half reproduces from inside IRON — 37.5–37.9 µs
per dispatch at N=64 — but IRON's per-call host work is a near-constant ~500 µs that batching
does not touch, so a batched `@iron.jit` call still costs **~531 µs**, a 1.26–1.37× gain
rather than 17×. Against that floor none of the four reopens survive: MobileViT stage-2
attention (240 µs), bf16 attention stage 2 (240 µs) and GroupNorm at L ≤ 18816 (233 µs) are
all still under it, and MobileNetV2's 1720 µs is a whole-model CPU time being compared against
a per-dispatch floor, which needs per-layer arithmetic nobody has done. **The four reopen only
for a caller willing to write a raw-pyxrt driver and give up IRON's argument handling.** The
same run also closed this section's other caveat: a real 8-core bf16 kernel (GroupNorm,
L=150528) batches to 823.8–838.3 µs per dispatch, which is its own compute — the independently
measured 835.8 µs — so a real kernel's configuration cost does *not* swamp the passthrough's
floor.

**CLOSED 2026-09-09 by a C++ host: the 36 µs floor is the driver's, not the binding's, and a
deployable runner reaches it.** The scoping above rested on "a caller willing to write a
raw-pyxrt driver", which left open whether the residual cost was pybind or the driver.
`kernels/dispatch_floor/dispatch_runner.cpp` — a standalone C++ XRT host, no Python — drives the
same cache entry to **36.7 µs** at N=64 against pyxrt's 35.9 µs in the same sitting, the two
agreeing within ~2% from N=4 up (`results/aie/dispatch_cpp_runlist_npu.log`). So removing the
binding does not move the batched floor: **36 µs is a property of the driver and the device**,
and no host-side rewrite goes below it. Against the same design in the same sitting the stack is
671.5 µs (IRON unbatched) · 498.5 µs (IRON batched 64) · ~108 µs (C++, one call) · **36.7 µs**
(C++ batched 64) — so IRON's ~500 µs host share is work a cached-handle host pays *once* at
startup, not per call. Two limits survive intact: the four reopens still need ≥ 8 independent
dispatches in flight, and they still need a caller outside IRON — that caller is now known to be
writable in C++ rather than hypothetical. A single dispatch costs ~108 µs even in C++, of which
only ~20–30 µs was ever the binding.

## 4. Objectives

Ordered by dependency, and weighted toward where the NPU has *measured* edge — large bf16
GEMM, independent-stream throughput, wide graphs — with the closed int8-conv result kept
as the bar an open kernel would have to clear rather than a reason not to try. Each entry
names its physical basis, the tooling that has to exist, the measurement that decides it,
and what in this repo it reuses. None of them is a claim of a result.

### Tier 0 — instrumentation (cheap, and everything downstream needs it)

**S0. Measure the core clock, per power mode. — DONE 2026-09-07, two of its three legs.**
Result: **1.80 GHz** in `default`, `performance` and `turbo`; **1.03 GHz** in `balanced`;
**0.80 GHz** in `powersaver` (`results/aie/clock_probe_npu.log`, `kernels/clock_probe/`).
The tooling as planned did not survive contact: Peano never defines `get_cycles()`
(undefined symbol at link), does not lower `__builtin_readcyclecounter`, and rejects
inline asm, so the counter was read through the trace unit instead — `event0()` /
`event1()` around the loop, stamped by the tile timer, and the runtime's submit+wait time
fitted against the stamped cycles across 2^18–2^25 iterations so the dispatch cost
cancels (R² ≥ 0.999995; two loops with 9.000 and 2.000 cycles per iteration agree within
0.33%). Decided: every per-second column in 2–3 now carries the measured-clock value; the
5 s idle probe shows no clock penalty, so the drift in 1.7 is not an idle state at that
scale (a power-mode change would be); `turbo` is accepted and reported, errors on entry,
and clocks the same as `performance`. Not run: the concurrent-VitisAI-EP leg (the
worktree that ran this had no `models/`), and every tile other than logical (0,2). Still
an assumption: that the trace timer is the core clock — the integer, length-independent
cycles per iteration of two different loops is the evidence for it.

The plan this replaces also recorded a cheap side-channel worth keeping: XRT's
`max_clock_frequency_mhz` was read as a live readback on this driver — 800 MHz idle,
1800 MHz while a context is active (`results/aie/xrt_api_live_clock_and_pdh_npu.log`),
~0.05 ms per read via `tools/hwinfo_npu_bridge.exe --json`. The run above read the same
query as a flat 800 in every power mode — an idle reading, as `results/aie/pmode_clock_readback_npu.log`
(2026-09-08) showed by varying load and mode together: busy, the readback is the mode's
clock to the MHz (1800 / 1028 / 800); idle, 800 in every mode. The trace unit is the
measurement of the clock; the readback is its live indicator.

**S1. Pin the data-movement constants. — Bounded from the demand side 2026-09-09; the port
measurement itself is unstarted.** `tools/gemm_cost_model.py` computes what the int8 GEMM's
cores would need if never starved (**3.35** B/cycle into one core's L1 at n=64) against what the
measured time says they get (2.50 B/cycle), and finds the measured cycles per call almost
unchanged, 3,274 vs 3,160, when the work per buffer is halved — a per-buffer floor set by
delivery rather than by the work (`results/aie/gemm_cost_model_nest.log`). The demand figure
supersedes the 6.19 B/cycle of the same day's first reading, which mis-modelled the kernel's
loop nest; the two measured rates are unaffected. That brackets the answer but does not measure
a port. **The route to measuring one is `PORT_RUNNING` /
`PORT_STALLED` / `PORT_IDLE` on shim and mem-tile DMA ports, and none of that reader exists
yet**: those events need their own wrapper classes and `shimtile_events=` / `memtile_events=`
parameters, and `kernels/pmu_probe/`'s reader takes `streams[0]`, the CORE packet type, which
is empty in a no-compute passthrough design. The frame encoding for the shim and mem-tile
packet types is also unconfirmed against the core format.
Physical basis: 1.5–1.6 hold three mutually inconsistent inferences. Tooling: extend the
dispatch-floor passthrough with `--direction {read,write,both}` and `--channels 1..8`,
plus two more variants — mem tile → four cores by broadcast, and core → adjacent core by
shared memory versus by DMA. Measurement: bytes per cycle per stream (with S0's clock), the
per-direction off-chip cap and whether it moves with channel count, mem-tile fan-out
bandwidth, and the neighbour-memory read rate. Decides: the 8 B/cycle assumption behind
3.1, the DRAM cap behind 3.2, and whether column count (A1) would raise off-chip bandwidth
at all. Reuses: the same script; `results/aie/mlir_aie_examples_npu.log`'s memcpy as the
cross-check.

**S2. Hardware trace on npu1, end to end. — Instrument built and calibrated 2026-09-09; the
applied measurement is what remains.** `kernels/pmu_probe/` routes the stall taxonomy,
occupancy and instruction mix instead of S0's two instruction events, and gates on
reproducing S0's own loops before any of it is believed: cycles per iteration come back at
2.0003 and 9.0001 against the measured 2.000 and 9.000. A level event emits one frame per
cycle, compressed into Repeat frames by the hardware, so a 142,730-cycle window fits in 1,344
bytes. The decomposition `cycles alive = issuing + memory + stream + lock + cascade stalls`
closes to a constant 190/198-cycle prologue across a 16× range of work. First finding:
`LOCK_STALL` alone accounts for 8,500–12,700 cycles per dispatch, flat in the work done, and
68% of the shortest run's cycles (`results/aie/pmu_probe_npu.log`). Still open from S2: three
of the four stall categories have never been non-zero here, so they are unexercised rather
than verified; and the applied Perfetto-style timelines below.
S0 ran `Program.enable_trace` on this machine end to end: `input_with_addresses.mlir` was
present in the design cache, packets arrived, and the stamps decoded. Two things it
learned that every later trace here inherits: a core that emits fewer events than fill a
32-byte packet never gets them to host memory (filler events fix it), and mlir-aie
v1.4.2's `aie.utils.trace.parse` mis-times any gap longer than 2^18 cycles (146 µs at
1.8 GHz) by treating the `0xff` sync frame as a timer no-op —
`kernels/clock_probe/clock_probe.py` carries a corrected decoder, cross-checked against
upstream on a sync-free run (`results/aie/clock_probe_npu.log`). The rest of S2 stands as written.
Physical basis: every core, mem and shim tile has a trace unit that emits cycle-stamped
event packets (8 selectable events per tile, or program-counter samples) over the stream
switch to a shim DMA and into a host buffer; mlir-aie's `Program.enable_trace` configures
all of it (SPEC: `programming_guide/section-4/section-4b/README.md`). Tooling: the only
known gap is the post-step — `magika`'s `trace_py` produced an NPU PASS and then failed
because `aiecc` did not leave `input_with_addresses.mlir` on disk on this machine
(`results/aie/mlir_aie_magika_mobilenet_npu.log`); fix or bypass that parser. Measurement:
Perfetto timelines for `ml/bottleneck` at 56×56 and `whole_array.py` at 2048×4096×4096
showing per-core MAC busy time, DMA stalls and lock waits. Decides: which half of the
3.1 gap is data movement and which is in-core; where conv's 11.3× goes. This is the
"instruction/tile-level profiling this repo's toolchain does not expose" that closed two
threads in `RESEARCH.md` — the silicon exposes it, the post-step didn't.

**S3. In-kernel cycle accounting without trace. — Answered statically for one kernel
2026-09-09; the hardware-counter route is unstarted.** MACs per cycle per core no longer has to
be inferred from throughput. `tools/gemm_cost_model.py` reads it off the object code: the int8
GEMM's hardware loop issues 8 `vmac` per 9-bundle iteration, 88.9% of one per cycle, but that
loop is only 54 of the 141 cycles an accumulator group costs, so across the whole call the
kernel manages **107.3 MACs per cycle, 41.9% of the 256 nameplate**
(`results/aie/gemm_cost_model_nest.log`, superseding the same day's 198.1 / 77.4%). The core
issues for 74.6% of the dispatch, so the **larger loss is the schedule, not delivery** — and it
is accumulator spill, since the kernel holds 8 live accumulators where 5 is the measured
spill-free ceiling. **Unstarted:** the decimating hardware counter — subclass
`GenericEvent`, override `get_register_writes()` to program `Performance_Control0` / `Control2`
— which would measure the issue rate rather than compute it, and would cover kernels whose
control flow the static route cannot walk. Whether a counter may reset on the event it
generated itself is unverified; the fallback is two chained counters.
Physical basis: the same counter as S0, read inside the kernel around the `mmul` loop and
around each fifo acquire — except that S0 found Peano cannot read it (1.2), so "without
trace" now means the S0 mechanism itself: `event0()`/`event1()` brackets, two events per
region, decoded from the trace stream. S0's own numbers are the first S3 data: a
dependent 16-lane int32 `aie::add` chain costs exactly 2 cycles per iteration, a volatile
scalar load-add-store loop 9 (`results/aie/clock_probe_npu.log`). Tooling: a `-DPROFILE` build of `mm.cc`'s `matmul_vectorized_*`
and of `conv2dk3` that writes cycle deltas into a side buffer. Measurement: MACs per cycle
per core directly, independent of host timing and of S2. Decides: the same questions as
S2 at lower fidelity and zero toolchain risk; use whichever lands first.

**S4. Package power under load, and the NPU's share of it.**
Physical basis: nothing in this repo has ever measured a watt. 1.7 has the clock and the
GPU-engine utilization percentage, and every "is it worth it" verdict in
`docs/BENCHMARKS.md` — the iGPU comparison, MobileNetV2 being too cheap to accelerate,
`resnetv2_50x3_bit` losing by 25% — is a latency verdict, while the part's stated reason
to exist is work per watt. There is no per-NPU rail to read, and this repo has established
that rather than assumed it: `Estimated Power : N/A` in every `xrt-smi examine -r platform`
capture taken here (MEASURED `results/aie/xrt_smi_platform_pmode.log`,
`results/aie/clock_probe_npu.log`), and the electrical query itself failing at the driver
escape (MEASURED `results/aie/xrt_api_live_clock_and_pdh_npu.log`; `docs/SETUP.md` lists
power among what the monitor cannot show). So the measurable quantity is a package-power
delta with an attribution control — never an isolated NPU wattage, which on this silicon
can only be inferred.

Tooling: a fourth source in `tools/hwinfo_npu_bridge.exe`, which today publishes to
HWiNFO's custom-sensor registry interface and reads nothing back. Package and core power
are platform SMU sensors, so the candidate read paths are HWiNFO's own shared-memory
export — a separate documented interface from the registry one the bridge writes — or a
direct SMU read; which of them is available here is the first thing S4 has to settle, and
it may be neither. The sampling loop the bridge already runs (0.1 s PDH and XRT readback)
then gains a power column, an idle baseline before and an idle tail after the workload,
and a marker for the workload window.

Measurement: mean package power and mean core power, idle versus loaded, across a matched
pair taken in one sitting — the NPU run, and a CPU-only run of the same arithmetic — with
the machine and the power mode named. The attribution is only as good as the core column:
package rising while the cores stay flat puts the draw outside the cores; both rising says
nothing about the NPU. Report the delta and its CPU-only control, never a bare wattage.

Decides: whether the NPU's edge is work per watt where it is not work per second. Every
comparison the NPU loses or barely wins becomes a different question if the package draw
differs — MobileNetV2 at 1.72 ms CPU against 2.68 ms NPU, `resnetv2_50x3_bit` at 470.79
against 588.58 ms CPU-to-NPU, and yolov8n at 9.9–10.5 ms on the iGPU against 6.8–6.9 ms on
the NPU, a margin the demos re-run reversed outright (DML FP16 6.08 ms against the NPU's
6.59, `results/demos/demo_tri_hardware_showdown.log`). That last one is the case for S4 in
miniature: the latency verdict is already unstable between sittings, so watts would be the
tiebreaker rather than another way to restate it. It also decides whether
the 2.25× power-mode clock lever (1.7) is efficiency-neutral, and gives 1.7's
unattributed session-to-session latency drift a second signature to test.

Reuses: the bridge and its PDH/XRT sampling loop; the load-and-mode-in-one-sitting shape
of `results/aie/pmode_clock_readback_npu.log`; section 6's first rule (name the CPU
implementation) for the matched-arithmetic control.

Prior art, third-party, and not evidence for this repo: `open-xdna`
(`Scottcjn/open-xdna`) runs this measurement on Linux on a Ryzen 7 8845HS — package RAPL
against the x86-core domain, with the NPU's DPM level from the staging driver's debugfs as
a corroborating signal — and its `docs/POWER_INSTRUMENTATION.md` is a working model for
the method, including the discipline of reporting `package − core` as an upper bound
rather than an NPU wattage. Its figures are another machine, OS, driver and workload, and
its README headline reports +6.6 W from a script not in the published tree against the
+2.88 W its documented method measures, so none of its numbers travels into this file. What corroborates the paragraph
above is the direction: it reports the firmware telemetry buffer unpopulated on production
firmware (amd/xdna-driver#1447), the Linux-side counterpart of the failed electrical query
here.

### Tier 1 — the dispatch path

**D1. `xrt::runlist` batching. — ANSWERED 2026-09-09: 36.3 µs per dispatch, a 17× drop.**
The sweep this objective asked for was run at N = 1, 2, 4, 8, 16, 32, 64
(`results/aie/dispatch_runlist_npu.log`). Amortised cost per dispatch falls from 147 µs at
N=1 to **36.3 µs** at N=64 and is asymptotic there, so a larger batch buys little. (**Sitting label**, because two figures for this same measurement class are both live: 36.3 µs is the original 2026-09-09 pyxrt run; the same-sitting rerun in `results/aie/dispatch_cpp_runlist_npu.log` reads **35.9 µs**, and the C++ figure is compared against *that*, not against 36.3 — see this repo's rule that every config being compared must be captured together.) Raw pyxrt
single-dispatch is ~140 µs, already below the 169.8 µs "hardware" bracket, and the batched
figure is a quarter of it — the hardware half of the floor is not silicon. The prediction in
this entry's physical basis was correct: most of the host term is software and a list of N
runs costs one host round trip. **Remaining:** this is raw pyxrt; `@iron.jit` does not use
runlists, and nothing under `aie/utils/hostruntime/` references one, so a real design still
pays the old floor until that host path is written. That, not the measurement, is what D1
now blocks on.
**Remaining: CLOSED the same day, and the answer moves the problem to D2.** That host path
was written — `kernels/dispatch_floor/iron_batch.py` patches IRON's own transaction submit to
queue unstarted runs into a `pyxrt.runlist`, with nothing outside this repo modified
(`results/aie/iron_batch_npu.log`). From inside IRON the device cost per dispatch does fall to
**37.5–37.9 µs**, reproducing the raw figure. But the wall clock only improves **1.26–1.37×**,
because IRON's per-call host work is a near-constant **~500 µs** that batching never touches,
leaving a batched `@iron.jit` call at ~531 µs. So D1 is answered in full: batching removes the
dispatch half of the floor and nothing else. The ~500 µs residue is the same term
`dispatch_floor_npu.log` measured as 447.3 µs, now confirmed independent of how the submit is
done, and at 37.5 µs of device against ~500 µs of host **it is the larger term by more than an
order of magnitude**. Whatever removes it is D2's question, not D1's.
**And that residue is removable — measured 2026-09-09 in C++.** A standalone C++ XRT host with
no Python in it (`kernels/dispatch_floor/dispatch_runner.cpp`) drives the same cache entry at
**36.7 µs** per dispatch at N=64, against pyxrt's 35.9 µs in the same sitting; the two agree
within ~2% from N=4 up, so the batched floor is the *driver's* and the binding was never in it
(`results/aie/dispatch_cpp_runlist_npu.log`). The same-sitting stack is 671.5 µs (IRON
unbatched) · 498.5 µs (IRON batched 64) · ~108 µs (C++, one call) · **36.7 µs** (C++ batched 64):
**13.6× better than batched IRON.** So IRON's ~500 µs is not a property of the device or the
driver but of IRON's per-call work, and a cached-handle host pays it once at startup (33–60 ms)
instead. Two limits stand: it needs ≥ 8 dispatches in flight, and one dispatch still costs
~108 µs even in C++ — only ~20–30 µs of the single-dispatch path was ever the binding.
Physical basis: the vendor path's host-side cost per call is about 90 µs against IRON's
447 µs on the same driver (2.5), so most of the host term is software, and a list of N
runs costs one host round trip. Tooling: none new — `xrt::runlist` is in this
XRT's experimental header and `pyxrt.runlist` exposes `add`/`execute`/`wait` (MEASURED as
present, `results/aie/dispatch_floor_npu.log`, follow-up section). Measurement: the
passthrough sweep at N = 1, 8, 64 runs per list; report host µs per run and hardware µs per
run. Decides: how much of the 447.3 µs and of the 169.8 µs amortises. Reuses:
`measure_floor.py`.

**D2. A C++ host with cached handles.**
Physical basis: the host cost is "mostly XRT outside the submit bracket" (that log's own
cProfile), not interpreter time. Tooling: a small C++ program against the XRT SDK
(headers and libs are at `C:\Xilinx\XRT\xrt_sdk\xrt`) that loads a `final.xclbin` from
`~/.npu/cache/<hash>/` once, holds the `hw_context`, kernel handle, instruction BO and
data BOs, and resubmits. Measurement: wall per call versus the 617.0/169.8 µs brackets on
the identical xclbin. Decides: the true per-call floor of this driver for a non-vendor
design; sets the number that replaces 617 µs in every go/no-go. Reuses: the CMake and XRT
package setup already proven for `vision/*` (`results/aie/mlir_aie_vision_examples_npu.log`).

**D3. Persistent kernels: pay the floor once per session, not per op.**
Physical basis: a core is a processor running an ELF; nothing in the silicon requires it to
stop between operations. Cores can loop forever on ObjectFifo acquires, and the host's
runtime sequence is just BD programming plus a sync — the same mechanism the vendor DPU
uses to run a whole graph as one node. Tooling: an IRON design whose `Runtime` sequence
is re-issued per op with new buffer offsets while the `hw_context` and core programs stay
resident, and (through D2) a host path that issues only the DMA part. Measurement: per-op
cost for a 96 KB GroupNorm or a stage-3 attention block, against their 233 µs / 34 µs CPU
times. Decides: whether the small-op class (3.4) is reachable at all. This is the single
objective with the widest blast radius: every verdict in this repo that says "loses on the
dispatch floor" was measured against a floor D3 removes.

**D4. In-process splice: kill the two-process handoff.**
Physical basis: the EP and an IRON xclbin already hold NPU contexts concurrently with zero
contention (`results/bit/profile_instancenorm_splice_feasibility.log`), and in-process
handoff on this machine costs +0.13 ms, not the 789 µs–23.6 ms that two processes cost
(`results/mobilevit/splice_wall_clock_npu.log` vs
`results/aie/groupnorm_bf16_handoff_floor_npu.log`; the unmerged branch
`groupnorm-floor-v2` cut the two-process figure to 5.7 ms at L=301056 — still a loss —
and measured the protocol alone, without the fp32/bf16 conversion, at 1.19 ms, which
would fit inside that shape's 1.94 ms kernel margin). The wall was never the silicon; it
is that `pyxrt.pyd` links `python313.dll` and `resnet_env17` is Python 3.12.
Tooling: an ONNX Runtime custom-op DLL in C++ that owns an XRT context for an IRON xclbin
and is registered into the same `resnet_env17` session as the VitisAI EP — no Python
binding in the loop. Measurement: `resnetv2_50x3_xint8.onnx` end to end with the 33
winning `InstanceNormalization` nodes routed to `kernels/groupnorm_bf16/`, against the
measured 588.58 ms NPU / 470.79 ms CPU. Decides: whether the 33/49 per-node wins
(19.4 ms of 42.37 ms) survive contact with the real graph once the handoff is in-process.
Reuses: the kernel, `extract_golden.py`, the handoff harnesses on both branches.

### Tier 2 — kernel quality, held to measured bars

**K1. int8 conv at DPU-class efficiency. — The 11.3× is ISSUE RATE, answered 2026-09-09
without a trace.** This entry's own go/no-go said "the first trace will say whether it is data
movement or issue rate". It is issue rate, and the instruction schedule says so on its own
(`results/aie/conv_issue_rate_decomposed.log`). The int8 GEMM issues **0.889 vmac/cycle**; the
best conv loop in the design manages **0.333**, the 3×3's main loop **0.222**, and the 1×1's
hot loop **0.045**, with two of its three loops issuing no MAC at all. A 4×–20× shortfall in
MAC issue density against an 11.3× throughput gap — the schedule alone more than accounts for
it. **Two different causes, and the fixes differ.** The 1×1 never keeps an accumulator in a
register: its 22-bundle loop loads all four quarters from memory, issues one `vmac`, stores
four quarters back, and idles six bundles on load-to-use latency, while naming only 3 of the
file's 9 accumulators. The 3×3 *does* keep `cm1`–`cm4` live and still reaches 0.222, because
six of its eighteen bundles are `vshift` and four more `vmov` — sliding-window realignment
spent in issue slots, which is precisely what this entry's tooling section proposes moving to
the mem tile's 4-D BDs. Reaching 1 TOPS needs 6.8×; the 1×1's defect alone is worth up to 20×
on that kernel. **What this does not establish** is that a rewrite would get there — the 20×
and 4× are ceilings on what the schedule leaves unused, not predictions, and the per-loop
densities are unweighted by trip count.
**The tooling half was gated on hardware 2026-09-09, and the proposed mechanism does not work
through IRON.** This entry proposes moving the ten `vshift`/`vmov` bundles "to the mem tile's
4-D BDs", and 2.6 calls that BD "the one address generator on the chip that can do an im2col
or a transpose in flight". `kernels/im2col_bd/im2col_probe.py` tests exactly that and nothing
else — compute-free, shim → mem tile → shim, the 4-D pattern on the mem tile's outbound stream,
verified byte-for-byte against a host im2col (`results/aie/im2col_bd_probe_npu.log`).
**Non-overlapping patterns pass and every overlapping one hangs the device:** k=1 returns all
256 elements correct at 16×16 and all 64 at 8×8, while k=2 (3.52× expansion) and k=3 (6.89×)
both return `ERT_CMD_STATE_TIMEOUT` — as does k=3 with the forwarded fifo given the expanded
object type, so it is not a simple length mismatch. Since k=2 is the smallest overlap a square
window can ask for, the boundary is not a large expansion factor: any re-reading pattern tried
here times out. This does **not** refute 2.6's claim about the silicon — the BD field widths
fit with room to spare — it refutes the *route*: do not write a conv kernel against
`ObjectFifo.forward(dims_to_stream=...)`. A raw buffer descriptor outside the ObjectFifo
abstraction, where length and access pattern are set independently, is the next thing to try.
**Bandwidth is not what would kill the design, though** — DERIVED, gate 1 of the same run: an
im2col conv's expansion cancels, giving **1/C_out bytes per MAC**, so against 3.1's int8 ceiling
of 0.03125 B/MAC it is stream-bound only below C_out = 32 and has 2× headroom at C_out = 64.
(That ceiling inherits 1.5's TO VERIFY on the 4 B/cycle stream rate.)
Bar: 1.65 TOPS per column, clock-independent (2.2); today 146 GOPS (2.3). Physical
basis: 3.3 — conv has ten times the reuse the core needs, the weights fit on-chip, and the
same column sustains the bar under AMD's compiler. Tooling: a conv design of this repo's
own rather than `ml/bottleneck`: weight-stationary tiles in the mem tile, im2col or row
shifting done by the mem tile's 4-D BDs rather than by core code, all four cores of a column
on one layer with the K-reduction over the 512-bit cascade, then four columns; S2/S3 driving
every iteration. Measurement: the `sweep.py`/`cpu_sweep.py` fit at 56×56 and 512×32, one
sitting, against ORT int8 (1678.8 GOPS marginal) — the CPU implementation named, per the
standing rule. Go/no-go: nothing below ~1 TOPS per column is worth a second week; the
gap is 11.3× and the first trace will say whether it is data movement or issue rate.
Reuses: `kernels/bottleneck_sweep/`, the local `conv2dk1`/`conv2dk3` width fixes,
`kernels/conv2dk3_widthfix/`.

**K2. bf16 GEMM to its input-bound roofline.**
Bar: 27–32% of peak today (2.4) against a 67% bound at the default tile and 100% at 64×64
(3.1). The 64×64 lever is measured on both dtypes with the generic `whole_array.py` CLI and no
design work: 1.9× over 64×32 in int8, whose half-size operands fit it in L1
(`results/aie/int8_matmul_sweep_npu.log`), and 1.39–1.47× in bf16 once the C output tile is
single-buffered to free the 16 KB it was short by — 2477.23 GFLOPS at 2048³, the CPU-bf16
margin widening to 1.29×–1.89× (`results/aie/bf16_matmul_n64_single_buffer_npu.log`). That is
where the CLI stops: 128×64 and 64×128 at k=64 miss L1 by 19,712 B even single-buffered —
2A+2B+C+stack = 32,768 + 16,384 + 32,768 + 3,328 = 85,248 B against the 65,536 B bank,
`l1_estimate()` in `kernels/int8_matmul_sweep/npu_matmul_sweep.py`, and MEASURED: both die
in the allocator exactly as predicted while 25 of the 26 predicted-fit tiles compile (the
26th hits 2.6's stride cap) — `results/aie/gemm_tile_sweep_c_single_buffer_npu.log`, Table 1.
That sweep also settles what the freed 16 KB buys: 64×64 and 32×128 at 2501.71 / 2494.61
GFLOPS at 2048³ and 2700.44 at 2048×4096×4096, 1.46× the default tile and 33.9–36.6% of
peak against 3.1's 100% ceiling, so the next two thirds are not input bandwidth (3.1's
measured-against-the-model paragraph names the two terms the model lacks).
Physical basis: 3.1's three levers — C accumulated in registers or across the
cascade instead of read-modify-written in L1 every k-step; A or B shared between adjacent
cores through neighbour memory so one stream feeds two; and shape-specific DMA
decomposition through the mem tile for widths like 11008 where the shim BD's 20-bit step
forces the generic design down to `m=16` (2.6). Tooling: a repo-owned GEMM design (not the
generic `whole_array.py` CLI) with the tile chosen from the B/MAC table, f32 accumulation
kept as the K-limit diagnosis requires (`results/aie/bf16_matmul_k_limit_diagnosed_npu.log`).
Measurement: `kernels/bf16_matmul_sweep/cpu_matmul_sweep.py` on the same shapes, one
sitting; report the fit, not per-point GFLOPS. Go/no-go: the CPU is flat at 1100–1360
GFLOPS; every doubling of the NPU's 1.13–1.78× edge is a doubling of the case for the
transformer capstone C2.

**K3. int16×int8 (A16W8) kernels — the precision the EP can't reach.**
Physical basis: the tile does int16×int8 at 128 MACs per cycle, the same rate as bf16
with half the weight bytes (1.2); Quark's `A16W8` places 0/394 nodes because opset-17 Q/DQ
can't carry 16-bit types (`results/a16w8/diag_resnet50_a16w8_npu.log`) — an EP limit, not
a silicon one. Tooling: `mmul` for mixed int16×int8 (TO VERIFY which shapes the AIE API
offers on aie2; `mm.cc` ships only int16×int16 4×4×4), then a GEMM and a 1×1 conv on the
K2 template. Measurement: accuracy on a model XINT8 breaks — yolov8n-pose's 17.8-point OKS
loss, or MobileViT's 0.00% — with the K2 CPU harness for latency. Decides: whether an
int16-activation path recovers accuracy at a cost the array can pay.

**K4. Per-channel and non-power-of-two requantisation.**
Physical basis: nothing in the tile constrains scales; `XINT8`'s power-of-two per-tensor
grid is a DPU/Quark contract. MobileViT-XXS collapses to 0.00% because its depthwise
weight-scale grid reaches Δ=1.0 where MobileNetV2's stays at 0.25
(`results/mobilevit/quant_grid_audit.log`), and AdaRound cannot move Δ. Tooling: a
depthwise 3×3 int8 kernel with a per-channel multiplier-and-shift epilogue. Constraint:
those ops are tiny, so this is only worth building after D3, or inside a monolithic graph
(C1) — on its own it lands under the floor (3.4).

**K5. Fused attention at LLM scale, with `mmul` this time.**
Physical basis: the array wins bf16 GEMM from N ≈ 1024 upward, and the old K ceiling was
an accumulator dtype, not hardware (`results/aie/bf16_matmul_niche_npu.log`,
`..._k_limit_diagnosed_npu.log`). Budget: a 64-row Q block at head_dim 128 is 16 KB of
bf16; double-buffered K and V tiles of the same size are 64 KB, so the per-core block is
32 rows, running max and sum in fp32 in L1, scores never materialised (the 128 KB N×N
matrix that overflowed at N=256 is the thing to avoid, and row-wise streaming is right —
what killed `attention_bf16` was doing it without `mmul`, 0.61 GFLOPS on hardware
measured at 895). Tooling: K2's GEMM tile plus the already-validated bf16 softmax pieces
(`results/aie/mlir_aie_ml_examples_npu.log`). Measurement: seq_len 2048 and 4096, head_dim
128, against torch bf16 attention on this CPU measured first, in the same sitting.
Go/no-go: CPU time must clear the floor that D1–D3 leave.

### Tier 3 — the array

**A1. Reach the fifth column.**
Known: five physical columns (1.1); AMD's compiler derives a 5×4 device for this part; the
driver ships 5×4 overlays; mlir-aie stops at four. Unknown: whether column 0's shim has a
NoC DMA, and whether the driver grants a 5-column `hw_context` to a non-vendor xclbin.
Tooling: extend mlir-aie's NPU1 target model to five columns (`_MAX_COLS`, a
`VirtualizedNPU1TargetModel(5)`, the `AIEAttrs.td` enum) and build the memcpy design for
it. Measurement, in order: does the driver load it (`xrt-smi examine -r aie-partitions`
during the run must show a 5-column partition); does column 0's shim move data; if not, do
column 0's cores run when fed over the switch from column 1. Prize: 20 cores instead of
16, and the missing 18% of nameplate (2.1). The first experiment is the driver grant, not
a kernel.

**A2. Heterogeneous partitions in one process.**
Physical basis: `1x4.xclbin` gives independent per-column partitions that scale to 3.65×
(`results/multi_partition_yolov8n.log`), and an EP context and an IRON context coexist
(D4's evidence). Tooling: D4's custom op plus a one-column IRON design (`npu1_1col`) so
the DPU runs the convs on columns 1–3 while the norm, attention or requant kernel runs on
column 4, pipelined. Measurement: `resnetv2_50x3` again, and MobileViT's real backbone
handing its intermediate to an on-NPU attention block. Decides: whether the 49-boundary
graphs that lose today (3.2) can be pipelined instead of serialised.

**A3. A weight-resident, persistent small CNN.**
Physical basis: 3.0 MB of reachable SRAM (1.8) against yolov8n's ~3 MB of int8 weights;
the measured 2.40 ms fixed cost per yolov8n inference and 2.63 ms per ResNet50 inference
(`docs/BENCHMARKS.md`, "Input resolution") is work proportional to the graph, not the
pixels, and the EP's own dispatch is 0.089 ms of it. Tooling: D3's persistent design
applied to a real graph — weights loaded once into mem tiles and L1, activations streamed
per frame. Measurement: the same `ms = a + b × Mpixels` fit; the objective is `a` under
0.5 ms. Decides: whether the fixed cost is weight and layer setup (removable) or something
the graph's shape imposes; and it is the one route by which a MobileNetV2-class model could
stop losing to the CPU.

**A4. Five contexts on five columns.**
Follows A1 and the `1x4` result: the array today reaches 6.29 TOPS on four independent
contexts (`results/multi_partition_yolov8l.log`); a fifth column that behaves like the
other four adds 25% to that ceiling for independent-stream workloads, which is the one
workload shape where this NPU has beaten the CPU by the widest margin.

### Capstones — what the arithmetic says is possible, stated as targets

**C1. The repo's most accurate model, run as one NPU graph.** `resnetv2_50x3_bit` is 84.00%
top-1 and loses to the CPU (588.58 vs 470.79 ms) because 49 `InstanceNormalization` nodes
and their neighbours fall to the CPU and the graph is cut into subgraphs around them
(`results/bit/`). DERIVED order of magnitude only: scaling `resnet50`'s 4.24 GMACs
(`results/multi_partition_resnet50.log`) by 3² for width and 2² for 448² gives ~150 GMACs;
at the DPU's measured 1.65 TOPS per column that is ~180 ms on one column and ~45 ms on
four, plus the norms at the DRAM cap (~25 ms for all 49 if each is one read and one write,
zero if fused). Against 470.79 ms that is a 3–10× win on paper for the most accurate model
this repo has — reachable only through K1 (a DPU-class open conv) or A2 (the DPU plus a
column of norms), and the honest statement is that K1 is the long pole of this whole
document. Replace the MAC estimate with `onnx-tool`'s count before quoting it anywhere.

**C2. A transformer block at 7B scale on the array.** GEMMs already win 1.13–1.33× at
production shapes (2.4); the block needs K2 (tiles at the roofline), K5 (attention), D3 (so
the norms and activations between GEMMs don't each pay a dispatch), and one measured CPU
implementation named on the other side. The Llama/Mistral pair shows the verdict can flip
on one integer's factorisation (2.6), so the target is a block that carries its own DMA
decomposition per shape, not a generic one.

## 5. What the silicon cannot do, and what is merely blocked

**Cannot (design around, don't chase):**

- int16×int16, int8×int4, int16×int4 and bfp16 MACs — AIE2p only (SPEC 1.2).
- A vector fp32 multiply path — not in the table; fp32 products cost two bf16 MACs, fp32
  accumulation is free.
- More than 64 KB per core, 512 KB per mem tile, 16 BDs and 16 locks per core or shim tile,
  a shim BD stride over 4 MiB, a core-tile BD over 16,383 words, an iteration wrap over
  64 — field widths (1.2–1.4, 2.6).
- An off-chip rate above the shared cap S1 will pin (currently inferred at 26–28 GB/s per
  direction), whatever the channel count.
- Hardware `sqrtf` in Peano's AIE libc (software reciprocal square root is fine, 1.2).

**Blocked by runtime, firmware or packaging — i.e. work, not walls:**

- The fifth column through the 1.7.1 EP (fingerprint mismatch) — A1 goes around the EP.
- Batch > 1 through the EP writes only slot 0 (`results/batch/slot_probe_b2.log`, filed as
  amd/RyzenAI-SW#401) — N independent contexts is the measured alternative, and a custom
  design has no such limit.
- The `pyxrt` / Python 3.12 ABI wall — D4 removes Python from the loop.
- `A16W8` through the EP (opset-17 Q/DQ) — K3 bypasses the EP.
- `InstanceNormalization`, rank-5 tensors, `LayerNormalization` and `MatMul` on the EP
  (`results/bit/`, `results/mobilevit/quant_grid_audit.log`) — kernels, D4, A2.
- `aiecompiler`'s missing `physical_device.dll` — moot; mlir-aie/Peano does the place and
  route on this machine already.
- mlir-aie's 4-column NPU1 model — A1's tooling deliverable.
- The 617 µs IRON floor — D1–D3.
- Per-NPU voltage and power: `Estimated Power` reads N/A and the electrical query fails at
  the driver escape (`results/aie/xrt_api_live_clock_and_pdh_npu.log`); the Linux side
  reports the firmware telemetry buffer unpopulated too (`open-xdna`, third-party,
  amd/xdna-driver#1447) — S4 measures a package delta instead.

## 6. Rules every objective inherits

They are the repo's existing rules, restated because a silicon-level target makes each one
easier to break:

- **Name the CPU implementation** in every ratio. On this project the CPU kernel choice has
  decided the verdict more often than the NPU has: torch fp32 sat within 1% of the NPU on
  conv2x while ORT int8 was 6.3× ahead; numpy attention was 9× slower than torch
  (`results/aie/conv2x_int8_cpu_baseline.log`, `results/mobilevit/splice_wall_clock_npu.log`).
- **Decide on the fit, not per-point throughput.** `time = intercept + slope × work`;
  1/slope is the array's rate with every fixed cost removed, and per-point GOPS must rise
  with size on any accelerator (`results/aie/bottleneck_spatial_sweep_npu.log`).
- **Both sides in one sitting, machine named.** Latency drifts between sessions (1.7).
- **Say which bracket**: 169.8 µs hardware, 447.3 µs host, 617.0 µs wall — and after D1–D3,
  the replacements those measure.
- **A go/no-go before a kernel, not after**: the op's CPU time against the current floor.
- **Verify placement, not latency**: `deviceStat` for a `DPU` entry on the EP side; the
  design's own numeric check against a golden tensor on the IRON side, every shape, before
  a timing counts.
- **Keep the negative results in front.** Conv 12.75×, mobile attention 71–240×, MobileViT
  0.00%, GroupNorm 0/49 after handoff — each one is a constraint on an objective above, and
  the reason a win here would be believed.
