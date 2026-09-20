# results/aie/

Logs from the hand-written AIE kernel work: the `aiecompiler` bring-up attempts that
failed, the open-source `mlir-aie`/Peano toolchain that worked instead, and every kernel
measured against a CPU baseline. The designs themselves live in
[`kernels/`](../../kernels/README.md); the findings are written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) and RESEARCH.md's "Custom C++ XRT /
hand-written AIE kernels".

This file exists because these entries used to be a single cell in
[`results/README.md`](../README.md)'s table, long enough that its three retractions were
invisible to anyone scanning it.

## Four silicon levers

The [2026-09-19 Desktop 2 sitting](../../docs/BENCHMARKS.md#four-silicon-levers-measured-2026-09-19-desktop-2)
backs the corresponding verdicts in [SILICON](../../docs/SILICON.md).

| Log | Evidence |
|---|---|
| [silicon_mem_neighbour_fresh_desktop2_20260919.log](silicon_mem_neighbour_fresh_desktop2_20260919.log) | Local/east/west read/write payload checks in fresh contexts; opens cross-column allocation experiments. |
| [silicon_stream_width_desktop2_20260919.log](silicon_stream_width_desktop2_20260919.log) | On-chip word/cycle trace, clock calibration and correct one/two-channel DDR transfers. |
| [silicon_weight_storage_desktop2_20260919.log](silicon_weight_storage_desktop2_20260919.log) | Exact Conv weights, biases, initializer bytes and graph.params; original weights exceed reachable SRAM. |
| [silicon_mmul_shapes_desktop2_20260919.log](silicon_mmul_shapes_desktop2_20260919.log) | Dense/sparse mixed int16/int8 compiler acceptance, rejected control and disassembly. |
| [silicon_mem_neighbour_desktop2_20260919.log](silicon_mem_neighbour_desktop2_20260919.log) | Earlier west-read second-submission timeout; preserved qualification limit. |
| [silicon_weight_bytes_desktop2_20260919.log](silicon_weight_bytes_desktop2_20260919.log) | Earlier parameter/initializer-only measurement, refined by the Conv-specific storage log. |

### Suite state behind those verdicts

That sitting re-scoped three test files, so it diagnosed the failures offline, at the same
commit, before editing anything. Neither log opened an NPU context.

| Log | Evidence |
|---|---|
| [silicon_gate_workspace_desktop2_20260919.log](silicon_gate_workspace_desktop2_20260919.log) | Both host-layer assertions pass with the fixture's allocation only; the collision is slot reuse against an out-of-order preload. |
| [silicon_gate_compilation_desktop2_20260919.log](silicon_gate_compilation_desktop2_20260919.log) | Verbatim legacy-scheduler rejection for 63 resident parameter sets, plus a 1,029,312-byte container from a nine-conv graph. |

## Shim channel utilisation

The [2026-09-19 audit of four emitted streams](../../docs/BENCHMARKS.md#sesrs-shim-channels-run-at-18-of-the-measured-rate-the-dispatch-floor-is-wait-structure-not-wire-2026-09-19-desktop-2)
is offline evidence about the engine's traffic, not about the silicon; the decoded task ABI it
relied on is recorded in [SILICON 1.4](../../docs/SILICON.md).

| Log | Evidence |
|---|---|
| [shim_channel_utilisation_sesr_yolov8n_desktop2_20260919.log](shim_channel_utilisation_sesr_yolov8n_desktop2_20260919.log) | SESR's floor runs at 18% of one column's measured rate with 2.151 ms outside transfer time; identical traffic at two cadences differs by ~1 ms; the arms differ in merge width, not bandwidth. |
| [fill_merge_attribution_sesr_m7_desktop2_20260919.log](fill_merge_attribution_sesr_m7_desktop2_20260919.log) | 355 of SESR's 412 distinct 6,400 B windows step 640 B apart and deliver 2.42x the address range they read. Its merge-savings reading is superseded by the pre-merge logs below. |
| [fill_merge_attribution_yolov8n_full_desktop2_20260919.log](fill_merge_attribution_yolov8n_full_desktop2_20260919.log) | The flagship's size class looks mostly regular in the artifact; the "496 pushes on the table" reading drawn from it is retracted (see the top of this file's retraction list). |
| [fill_premerge_sesr_m7_desktop2_20260919.log](fill_premerge_sesr_m7_desktop2_20260919.log) | The merger is offered 5,408 patterns and returns 710 - exactly the stream's activation task count - declining 0 multi-pattern runs. |
| [fill_premerge_yolov8n_full_desktop2_20260919.log](fill_premerge_yolov8n_full_desktop2_20260919.log) | 4,811 offered, 2,146 returned (= the activation task count); 1,869 stand alone because they are already four-dimensional. |
| [fill_repeat_positions_sesr_m7_desktop2_20260919.log](fill_repeat_positions_sesr_m7_desktop2_20260919.log) | All 100 of SESR's refetches are far from their first send (median 124 tasks) - cross-layer, so not coverable by a wider window. |
| [fill_repeat_positions_yolov8n_full_desktop2_20260919.log](fill_repeat_positions_yolov8n_full_desktop2_20260919.log) | The flagship's 938 refetches split 280 near against 658 far; only the near third is a packing question. |
| [fill_premerge_dims_sesr_m7_desktop2_20260920.log](fill_premerge_dims_sesr_m7_desktop2_20260920.log) | All 338 four-dimensional SESR fills are 25,600 B = 4 packets, and 0 of them have contiguous rows (pitch 10.32-12.90x the row) - which is what caps a chain at 4. |
| [fill_premerge_dims_yolov8n_full_desktop2_20260920.log](fill_premerge_dims_yolov8n_full_desktop2_20260920.log) | The same ceiling on the flagship's 1,869 (pitch 6.48-8.10x the row); 88 patterns have pitch *below* row bytes, unexplained. |
| [fill_layout_sizing_sesr_m7_desktop2_20260920.log](fill_layout_sizing_sesr_m7_desktop2_20260920.log) | Line-spanning needs 66,048-82,560 B per packet against a 65,536 B core and 6.51-12.90x the bytes; plane-packing frees the dimension at zero wire cost for 169 of 338. |
| [fill_layout_sizing_yolov8n_full_desktop2_20260920.log](fill_layout_sizing_yolov8n_full_desktop2_20260920.log) | The same two budgets on the flagship: 940 of 1,869 byte-free, 929 replicating 1.60-3.20x. |
| [fill_layout_lever_sesr_m7_desktop2_20260920.log](fill_layout_lever_sesr_m7_desktop2_20260920.log) | Prices both: -15.7% descriptors leaves G2G 0.718 ms short of AMD, -31.4% still 0.468 short; break-even is 48% of the stream. |
| [fill_layout_sizing_ring2_sesr_m7_desktop2_20260920.log](fill_layout_sizing_ring2_sesr_m7_desktop2_20260920.log) | The ring's schedule quadruples the chain-capped class (338 -> 1,352 descriptors, 1,352 -> 5,408 packets) and drops the byte-free fraction 50% -> 12%. |
| [fill_layout_sizing_weightbuffer_sesr_m7_desktop2_20260920.log](fill_layout_sizing_weightbuffer_sesr_m7_desktop2_20260920.log) | Control: the resident weight buffer offers the shipped pattern set unchanged, as expected of a weight-side change. |
| [fill_retention_pricing_sesr_m7_desktop2_20260920.log](fill_retention_pricing_sesr_m7_desktop2_20260920.log) | Retention may add ~1,389 descriptors before its own compute win stops paying; the ring added 3,725. Chain collapse to 64 still leaves it 0.791 ms slower. |

## Split container sizing

The [2026-09-20 Desktop 2 sitting](../../docs/BENCHMARKS.md#split-container-sizing-and-feasibility-2026-09-20-desktop-2)
measures the hardware boundaries and driver costs of split vs monolithic .ignite containers on AMD Phoenix NPU (Ryzen 7 8700G).

| Log | Evidence |
|---|---|
| [split_container_sizing_phoenix_20260920.log](split_container_sizing_phoenix_20260920.log) | Physical silicon measurement across 5 variants: hardware context creation tax (29.63 ms), persistent multi-dispatch scaling (34.1 us gap), DMA buffer sync (0.014 ms roundtrip on 716 KB), weight upload (0.184 ms on 8.16 MB), and DDR workspace sizing (22.04 MB vs 15.84 MB unshared). |
| [verify_split_silicon_phoenix_20260920.log](verify_split_silicon_phoenix_20260920.log) | Physical silicon validation on YOLOv8n: monolithic baseline (7.556 ms), decoupled weights (7.593 ms, bit-exact agreement on all 1,209,600 bytes), 2-segment NPU split (7.755 ms, bit-exact agreement), and a segment-0-only dispatch (max_segments=1 at 2.055 ms, 27% of the full dispatch). Segment 0 has no detect heads and emits no detections, so that figure is a decomposition of the dispatch, not a cheaper detection. |
| [verify_yolov8s_split_silicon_phoenix_20260920.log](verify_yolov8s_split_silicon_phoenix_20260920.log) | Physical silicon verification on YOLOv8s (11.2M params): monolithic baseline (17.305 ms, 29.33 MB), Variant 4 decoupled weights (1.21 MB container, 95.9% reduction, 17.326 ms, bit-exact match), Variant 2 3-segment linked split (17.719 ms, bit-exact match), partial dispatches with no detect heads and therefore no detections (stopping at layer 13 costs 4.722 ms, 27.3% of the network; stopping at layer 30 costs 9.076 ms, 52.4%), Variant 3 targeted DMA sync (3.647 us vs 55.663 us full sync, 15.3x speedup), and Variant 3+5 ComposedSession (zero-copy device DMA handoff, 0 MB workspace memory bloat, bit-exact match). |
| [yolov8_split_suite_phoenix_20260920.log](yolov8_split_suite_phoenix_20260920.log) | Full-family physical silicon benchmark across all six YOLOv8 variants (yolov8n, yolov8s, yolov8n-pose, yolov8m, yolov8l, yolov8x) pinned to 8 physical CPU cores (0x5555, OMP_NUM_THREADS=8): decoupled stationary weights collapse container artifacts by 92.3%–95.9% (down to 0.68–8.67 MB); inter-segment chained dispatch overhead measured at 17.7–49.9 µs; and a segment-0 dispatch costs 2.01 ms (n), 4.64 ms (s), 2.29 ms (pose), 11.24 ms (m), 19.53 ms (l) and 32.95 ms (x), a near-constant 25.0%–27.9% of each full dispatch. Segment 0 has no detect heads and emits no detections, so these are per-segment costs and NOT a cascade; the "2.40× to 5.23× over AMD" this line first carried is retracted, having divided a partial pass by AMD's complete one. |

## Retractions and supersessions in this directory

Read these before quoting anything below.

- **`fill_merge_attribution_*.log`'s claim that the flagship "leaves roughly 496 pushes on the table"
  is retracted**, and with it the reason given for SESR (`no repeat dimension can express an
  overlapping chain`). `tools/fill_premerge_audit.py` wraps the merger the schedule actually calls and
  finds the opposite: it is offered those chains and collapses 87% of SESR's patterns (5,408 offered,
  710 returned, and 710 equals the container's activation task count in the emitted stream), so what
  stands alone is four-dimensional already - 1,869 of the flagship's 2,146, 338 of SESR's 710 - and
  the shim descriptor has no fifth dimension. See `fill_premerge_sesr_m7_desktop2_20260919.log` and
  `fill_premerge_yolov8n_full_desktop2_20260919.log`. The artifact counts survive (1,007 tasks,
  13,181,568 B, 355 of 412 windows stepping 640 B, 2.42x the range read), and so does the rejection:
  **there is no unexploited merge in either container's fills**, which is a stronger reason than the
  1.5% figure.

- **`conv2x_int8_cpu_baseline.log`'s "628.2 µs reproduces `dispatch_floor`'s 617.0 µs to
  ~2%" is retracted** as a bracket mismatch. 617.0 µs is the passthrough's *wall*
  intercept (447.3 host + 169.8 hardware); the like-for-like host floor is **447.3 µs**,
  so 628.2 is 40% over it, not 2% under. Nothing in either verdict depends on it.
- **`conv2x_int8_cpu_baseline.log`'s "the op class is NOT closed" is superseded** by
  `bottleneck_spatial_sweep_npu.log`: the op class **is** now closed.
- **`bottleneck_spatial_sweep_npu.log`'s `tensor_w`=32 hard ceiling is superseded.** The
  real `aiecc`/Tile(0,4) ceiling for this channel config is `tensor_w`=44 (only four of
  Tile(0,4)'s five buffers scale with width), and 56×56 — ResNet50's real conv2_x shape,
  which that log says "cannot compile on this design at all" — has since been reached by
  single-buffering the final output FIFO. See `bottleneck_widthfix_npu.log`,
  `conv2dk3_widthfix_npu.log` and `bottleneck_w56_npu.log`. The op-class verdict is
  unaffected; reaching 56×56 made it worse, not better (CPU wins 12.75×).
- **`attention_bf16_kernel_npu.log`'s diagnosis is corrected** by `dispatch_floor_npu.log`:
  the loss was blamed on per-dispatch cost, which is ~1% of Stage 2's measured time. The
  real cause is that `attention_kernels.cc` uses `aie::mmul` zero times. The verdict
  (loses to CPU) stands.
- **`bottleneck_spatial_sweep_npu.log`'s open item "kernel quality … `conv2dk1.cc`/
  `conv2dk3.cc` unread" is closed:** both have since been read, both vectorize correctly
  with `aie::mmul`, and both carried a width-32 bug that is now fixed and verified.
- **`bf16_matmul_ffn_pipeline_npu.log`'s "1.32× NPU win" on the FFN pipeline is
  corrected/reversed** at the real (non-square) shape by
  `bf16_matmul_ffn_real_shape_npu.log`: that 1.32× used a square approximation of the
  up-projection (`N=4096` instead of the real `d_ff=11008`) to dodge a DMA-stride
  compile limit. At the real shape, three DMA/BD toolchain limits force the
  up-projection's tile size down to `m=16` (from the default 64), dropping it to 909.97
  GFLOPS and flipping the blended pipeline result to a **1.10× CPU win**. The isolated
  win at an unconstrained tile size is not retracted — it just doesn't automatically
  transfer to a real model's actual dimensions.
- **`bf16_matmul_ffn_real_shape_npu.log`'s Llama-2-7B verdict (CPU wins 1.10×) is
  sharpened, not reversed, by `bf16_matmul_ffn_shape_variants_npu.log`:** the same
  hardware/toolchain gives Mistral-7B's `d_ff=14336` shape the **opposite** verdict (NPU
  wins 1.13×), because its factorization admits a much larger `n`-tile than Llama's
  `d_ff=11008` does under the same fixed `m=16` cap. "Real shapes lose" was too broad a
  read of the Llama-only result — the actual determinant is one integer's factorization,
  not "real-world-ness."
- **`bf16_matmul_niche_npu.log`'s "K ≥ 3072 fails correctness, undiagnosed" is
  corrected** by `bf16_matmul_k_limit_diagnosed_npu.log`: not a threshold, and not
  undiagnosed. The K-reduction accumulates in a buffer typed `dtype_out`; with
  `--dtype_out bf16` the running sum swamps small increments as K grows. `--dtype_out
  f32` removes the limit for free, verified clean to K=4096. `bf16_matmul_attention_
  scale_npu.log` confirms the fix and the NPU win both hold at a real production shape
  (K=4096), and that `attention_bf16`'s own kernel never had this bug in the first
  place (its reduction already accumulates in AIE2's native fp32 accumulator).
- **`aie2_isa_static.log`'s section-5 attribution is corrected** by `gemm_cost_model.log`.
  That section is headed "the kernel behind `int8_matmul_sweep_npu.log`'s 4607.05 GOPS" and
  disassembles the object in cache `0816364bbbaf03f83e2f0bcd`, which carries
  `memref<64x32xi8>` buffers — the **default n=32 build**, which measured 2387.01 GOPS. The
  tuned n=64 kernel behind 4607.05 is a different object hash. **Every number in that section
  survives**: the two objects' loops are identical, nine bundles and eight `vmac` on `cm0`–
  `cm7` at 88.9% MAC issue density. Only the provenance line was wrong.
- **`gemm_cost_model.log`'s cost model is superseded** by `gemm_cost_model_nest.log`, written
  the same day. It read `matmul_i8_i32` as one hardware loop with straight-line setup and
  charged the 135 non-loop bundles once per kernel call. The function is a nest: two software
  loops around the hardware loop, re-running that body once per group of live accumulators.
  The error overstated starvation by 1.8× — "the core is starved for 60% of the dispatch"
  becomes **25.4%**, and the dominant cost moves from data delivery to accumulator spill
  traffic inside the core. The measured per-buffer floor (3,274 vs 3,160 cycles per call for
  2× the work) is a measurement and is unaffected.
- **`aie2_isa_static.log`'s VLIW slot naming is corrected** by `bank_conflict_survey.log`.
  It read the six slots off the nop mnemonics and called them "`b` branch, `a` load, `s`
  store, `x` scalar, `m` move, `v` vector". Slot **b is the second load unit, not the branch
  slot**: tabulating every operation in each slot of 226 *strictly six-field* bundles — the
  only encoding whose slot identity is unambiguous — puts `vldb` and `paddb` in b, `vlda`/
  `lda`/`mova` in a, and `ret` in the scalar slot x. The slot **count** of six and every
  cycle figure derived from bundle counts are unaffected. The correction matters because two
  load units are what make a same-bank paired load, and its extra cycle, possible at all.
- **`dispatch_floor_npu.log`'s 617 µs go/no-go threshold is superseded for batchable work**
  by `dispatch_runlist_npu.log`: batched `pyxrt.runlist` submission amortises the same
  passthrough to **36.3 µs** per dispatch, a 17× drop, and a quarter of the 169.8 µs that log
  attributed to hardware — so its "hardware half" is not silicon either. The 617 µs figure is
  **not retracted**: it still governs a single unbatched IRON dispatch, which is what that log
  measured, and a one-shot call still pays ~140 µs even through raw pyxrt. What changes is the
  rule built on it — restated in four places (`README.md`, `kernels/README.md`,
  `docs/DECISIONS.md`, `docs/BENCHMARKS.md`) — and the four small-op verdicts
  `docs/SILICON.md` §3.4 closed on it.
- **`bottleneck_spatial_sweep_npu.log`'s 146.1 marginal GOPS is superseded by
  `conv_accum_residency_npu.log`, and its stock baseline does not reproduce.** Making both 1×1
  kernels' accumulators register-resident raises marginal throughput to **348.4–350.3 GOPS**
  (2.99–3.01× across two series). Separately, and more awkwardly: that log's own stock figure
  **re-measures at 115.6–117.1**, not 146.1, at the same shapes. The kernel is not the same
  code — 146.1 predates the 2026-09-07 width fix, which rewrote the very loop in question, and
  that run could not compile 56×56 at all while the new one can. 146.1 is **not retracted**; it
  is a measurement of a kernel that no longer exists in this tree. The inference that the width
  fix cost ~20% is stated as inference in the log: the pre-fix kernel was not built as a third
  arm. **The op-class verdict is unchanged** — the CPU still wins 2.4× — so nothing downstream
  of "int8 conv is closed" moves.
- **`dispatch_runlist_npu.log`'s 36 µs is in turn scoped by `iron_batch_npu.log` to raw
  pyxrt.** Batching was wired into IRON's own host path and measured there: the device cost
  per dispatch reproduces at **37.5–37.9 µs**, but IRON's per-call host work is a
  near-constant **~500 µs** that batching does not touch, so a batched `@iron.jit` call still
  costs **~531 µs** — a 1.26–1.37× gain, not 17×. 36.3 µs is **not retracted**: it is what a
  raw-pyxrt driver gets, and the device half is confirmed from inside IRON. What changes is
  which threshold applies to whom — **there are three, not two** (~617 µs unbatched IRON,
  ~531 µs batched IRON, ~36 µs batched raw pyxrt) — and that against ~531 µs none of the four
  reopened §3.4 verdicts survives. The same run also closes that log's no-compute-passthrough
  caveat.
- **`notes_yolov8s_gap.md`'s "a MemTile store-and-forward hop is slower per byte than DDR transport", and its
  proposed ~0.05 ms per MB pricing term, are retracted** by `memtile_hop_phoenix_20260916T1523Z.log`, which
  measures the hop in isolation at -0.00023 ms per MB in and +0.00053 ms per MB out. The inference generalised from
  the activation ring's and the weight buffer's losses; those losses stand, and are the protocols', not the hop's.
  The note records the retraction in its last section.
- **`notes_yolov8s_gap.md`'s 2.006 ms weight re-send lever is superseded by 1.018 ms** (27,279,360 B on
  yolov8s; 0.369 ms on yolov8n). The first figure counted weight runs already concatenated across output groups as
  re-sends; only stride-0 replays are recoverable. The note's "Correction" section has the working.
- **`notes_yolov8s_gap.md`'s k3s2 over-read slack of 60% is corrected to 42%** (16 × 50 = 800 pixels read against
  11 × 42 = 462 used). The 23% for k3s1 and 73% for k5s1 recompute unchanged.
- **The consumer-sizing compression experiment in `notes_yolov8s_gap.md` is VOID**, as the note itself records: its
  multi-channel design timed out on its own golden `arange` input unpatched, so consumer sizing was not the only
  variable. The later single-channel results in the same note stand.

## Toolchain bring-up

**`notes_aie2_device_dtypes.log`** — not a measurement, a verbatim excerpt of AMD's own
`OGOAT/Collaterals/device.yaml` (shipped in the 1.7.1 install) giving Phoenix's `AIE2`
tile spec per data type, plus the `aie4_models` dead-end (internal AMD kernel-source tree,
wrong chip generation).

**`aiecompiler_part_{probe,ipuv1cnn}.log`** — three guessed `--part` strings, all rejected.
**`notes_aiecompiler_part_db.log`** — how the real device database
(`data/parts/xilinx/xclbin/`) and the compiler's own part-name table (byte-scanned from
`aiecompiler_client.dll`) were found instead of guessed.
**`aiecompiler_part_strix_confirmed.log`** — a real Strix part string derives cleanly,
proving the `--part` mechanism works.
**`aiecompiler_part_phoenix_candidate.log`** — `xc10AIE24x5-die-1LP-e-S-es1`, matched
against the independent "Total Columns: 5" `xrt-smi` finding, also derives cleanly.

**`aiecompiler_hostlib_missing.log`** — the next wall reached once `--part` is solved: no
host C++ standard library on this machine. **`aiecompiler_hostlib_fixed.log`** — installed
VS 2022 Build Tools and that wall clears too, reaching `Reading logical device
aie2_5x4_device` before stopping at a missing `physical_device.dll` (checked the whole `C:`
drive, not just the pip env).

**`aiecompiler_physical_device_missing.log`** — binary-scanned `aiecompiler_client.dll` to
confirm `physical_device.dll` is a literal hardcoded name (not derivable) expected to
export `createPhysicalDevice`, part of a 4-DLL family (`platform_device.dll`,
`guidance_summary.dll`, `udm_api.dll`) all equally absent; opened both full offline
installers already on the machine (`ryzen-ai-lt-1.7.1.exe`, `ryzen-ai-1.8.0.exe`, via
`7z`/`lessmsi`) to confirm neither ships it either — 1.7.1's installer has byte-identical
wheels to pip, 1.8.0's doesn't ship the aiecompiler packages at all.

**`npu_check_compat.log`** — amd/RyzenAI-SW's own `utilities/npu_check` compatibility tool,
built locally, confirms this driver (32.0.20101.3760) OK's every VitisAI EP version it
knows (1.2-1.8) against this PHX/HPT device — a second, independent tool corroborating
findings already used throughout this directory.

**`xrt_smi_platform_pmode.log`** — a `--batch` capture of `xrt-smi examine -r platform`
and `configure --help` on Desktop 2: `Total Columns: 5`, `Power Mode: Default`, the five
`--pmode` values (`default, powersaver, balanced, performance, turbo`), and no clock
reported on this driver. Nothing on the device was changed. Cited by
[`docs/SILICON.md`](../../docs/SILICON.md), sections 1.1 and 1.7.

**`xrt_api_live_clock_and_pdh_npu.log`** — NPU telemetry without `xrt-smi`, probed while
`tools/hwinfo_npu_bridge.cpp` was rewritten into a live monitor: XRT's in-process query API
(`pyxrt` and a C++ probe: what each `get_info` key answers on this driver, timed) and
Windows' GPU-engine statistics (PDH). Two findings. `max_clock_frequency_mhz` is a **live
clock readback** — 800 MHz idle, 1800 MHz for the whole of a 30 s IRON GEMM run, 800 after —
which corroborates the cycle-counter clock probe and retires the note that it "reads 800 in
every mode" (an idle reading). And the NPU is visible to Windows as an MCDM adapter
(`luid_0x00000000_0x0000d6bf`, "NPU Compute Accelerator Device" under DXCore's NPU hardware
type): its compute engine reads 84–88 % across the run and 0 % idle, and its shared memory
equals xrt-smi's `total_memory_usage`, while xrt-smi's GOPS/FPS/latency read `N/A` for the
same context. Also records what does not answer (`aie`/`aie_shim`/`aie_mem`/`memory`:
"No such query request"; `electrical`: driver escape 0xc0000023; thermal: no sensors), the
monitor's own frames and HWiNFO registry output under load, and that the VitisAI EP path was
**not** covered — three attempts to hold a VitisAI session failed for unrelated reasons.
Cited by [`docs/SILICON.md`](../../docs/SILICON.md) 1.7 and S0, [`docs/SETUP.md`](../../docs/SETUP.md).

**`clock_probe_npu.log`** — the AIE core clock, measured (objective S0 of
[`docs/SILICON.md`](../../docs/SILICON.md)): `kernels/clock_probe/` brackets a DMA-free
loop with `event0()`/`event1()`, the trace unit stamps both, and submit+wait time is fitted
against the stamped cycles. **1.80 GHz** in `default`/`performance`/`turbo`, **1.03 GHz**
`balanced`, **0.80 GHz** `powersaver`, one fresh process per mode with the platform report
captured in each; two loops agree within 0.33%; no idle penalty at 5 s; the power mode was
restored to `default` and the clock re-measured. Supersedes the 1.6 GHz RESEARCH.md cited
from a web search and the 1 GHz `bottleneck_spatial_sweep_npu.log` assumed. Also records
why the cycle counter cannot be read from a Peano kernel and that mlir-aie v1.4.2's trace
parser mis-times gaps over 2^18 cycles. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-aie-core-clock-measured-180-ghz-default-080-powersaver).

> The two entries above were read as disagreeing — whether XRT's `max_clock_frequency_mhz`
> tracks the live clock (`xrt_api_live_clock_and_pdh_npu.log`, 800 idle / 1800 under an
> active context) or is pinned at 800 (`clock_probe_npu.log`, read across all five power
> modes) — and the 2026-09-07 merge flagged it UNRESOLVED and the first entry's "retires"
> claim premature. `pmode_clock_readback_npu.log` below settled it by varying load and
> power mode in one sitting: both were right for what they varied. The trace-unit 1.80 GHz
> remains the measurement of the clock; the readback is its live indicator.

**`aie2_isa_static.log`** — AIE2 machine code read statically, no hardware used:
`tools/aie_disasm.py` disassembles a core ELF or kernel object with Peano's own
`llvm-objdump` and `kernels/acc_spill_probe/` sweeps register pressure with Peano's
`clang`. Establishes that a hardware loop's bundle count IS its cycle count — S0's two
loops measured 9.000 and 2.000 cycles per iteration and disassemble to **9** and **2**
bundles — because the core is a statically scheduled VLIW that covers operand latency with
explicit nop bundles. Also: six issue slots per bundle, which no document in this repo
stated; a scalar load's result reaching the 7th bundle after it issues; and the accumulator
file, where five live 4×8×8 int8 `aie::mmul` accumulators compile with zero stack traffic
and six is the first count that spills, superseding both `docs/DECISIONS.md`'s "6 hardware
accumulator registers" and `docs/SILICON.md`'s "≤4 stays in registers". Reads the production
int8 GEMM's inner loop at 88.9% of the MAC issue rate against a whole-kernel 31.3% of peak,
placing the loss outside the loop. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#aie2-machine-code-the-bundle-count-of-a-loop-is-its-cycle-count).

**`pmu_probe_npu.log`** — the trace unit used as a performance-monitoring unit, the first
time on this machine (objective S2 of [`docs/SILICON.md`](../../docs/SILICON.md)):
`kernels/pmu_probe/` reuses `clock_probe`'s kernel and design unchanged and swaps only the
event list, so its calibration runs the two loops whose cycles are already measured here.
They come back at **2.0003** and **9.0001** cycles per iteration against 2.000 and 9.000.
Establishes that a level event emits one frame per cycle, compressed into Repeat frames by
the hardware, and that `cycles alive = issuing + the four stalls` closes to a constant
190/198-cycle prologue across a 16× range of work. First finding: `LOCK_STALL` is
8,500–12,700 cycles per dispatch, flat in the work done, and **68%** of the shortest run's
cycles — the core waits on its input ObjectFifo far longer than it computes, on a loop the
disassembly rates as perfectly scheduled. Streaming 16 buffers and sweeping the compute per
buffer separates the terms: the per-buffer overhead is a constant **717** cycles at every
point over a 4,096× range, and the lock wait stays flat at 7,000–12,000 cycles however much
work is done, giving `cycles = n_buffers x (compute + ~205) + ~10,000`. Three of the four
stall categories were zero throughout and are unexercised, not verified. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-trace-unit-as-a-performance-monitoring-unit-68-of-a-short-kernels-cycles-are-lock-wait).

**`gemm_cost_model.log`** — `aie2_isa_static.log` and `pmu_probe_npu.log` composed into a
predictor, no hardware used: `tools/gemm_cost_model.py` computes a tiled GEMM's issuing cycles
from its compiled object and compares them with an already-measured time, so the remainder is
the cycles the core spent NOT issuing. **Its model numbers are superseded the same day by
`gemm_cost_model_nest.log`**, which found it had read `matmul_i8_i32` as one loop when it is a
nest. Two things in it stand: the measured observation that halving the work per buffer leaves
cycles per call at **3,274** against **3,160**, and the **correction to `aie2_isa_static.log`
section 5's provenance** — the object disassembled there is the default n=32 build (2387.01
GOPS), not the tuned n=64 one behind 4607.05. The two loops are identical, so every number in
that section stands and only the attribution was wrong.

**`gemm_cost_model_nest.log`** — the correction, and the better answer. `matmul_i8_i32` is two
software loops around the hardware loop, with the eight accumulators loaded before it and
stored after it; the first reading charged the 135 non-loop bundles once per CALL instead of
once per accumulator GROUP and overstated starvation by 1.8×. The trip counts are compile-time
constants in the object and reconcile exactly — 16 groups × 64 `vmac` = 1,024, which is
64³/(4·8·8) with nothing left over. Corrected: the core issues **74.6%** of the dispatch at
n=64, not 40.4%, and the **larger loss is the schedule** — about **107 MACs/cycle over a whole
call, 41.9% of the 256 nameplate** — because 87 of the 141 cycles an accumulator group costs
are accumulator load, store and stack spill, the direct consequence of holding eight
accumulators where five is this branch's measured spill-free ceiling. What survives untouched
is the per-buffer floor: ~3,200 measured cycles per call whatever the tile, which n=32 fills to
41% and n=64 to 75%, leaving **~1.3×** of headroom rather than 2.4×. Also falsifies charging the
trace probe's 717 cycles per buffer to a different kernel, which would predict 117.5% of the
measured time. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-int8-gemm-issues-at-40-of-nameplate-and-a-3200-cycle-per-buffer-floor-caps-it).

**`bank_conflict_survey.log`** — a fourth exception to "a loop's bundle count is its cycle
count", and a correction to how this directory names the VLIW slots. A core tile's 64 KB is
four banks of 16 KB and the core has **two load units**, so a bundle can issue two loads at
once; when both address one bank the pair costs an extra cycle. That price is measured in
`memory_desktop2_20260909_m01_*.log` (eleven logs, merged into this directory since; measured on
branch `research/windows-lowlevel`) by
holding the compiled function bytes identical and moving only the operand addresses: **12.0**
cycles per iteration in one bank against **11.0** across two, r² 1.0, and **1,024** cycles per
64×64×64 panel in a real GEMM. `tools/aie_bank_check.py` reads the allocated addresses out of
a core ELF — they are absent from the pre-allocation `aie.mlir` — and finds the production int8
GEMM with **both input tiles in bank 2 and bank 3 empty**, against the bf16 GEMM with its
inputs correctly split. int8 loses because its tiles are *smaller* and the allocator packs the
pair into one bank. Charging it narrows the int8 GEMM's unexplained residual from 25.4% to
18–23%. Also refutes, with a fit, the tempting equation of the 789.8 µs two-process handoff
floor with local `main`'s measured 747.75 µs NPU context-switch penalty. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#two-loads-in-one-bank-cost-a-cycle-and-the-int8-gemm-has-that-collision-where-bf16-does-not).

**`context_ceiling_crosscheck.log`** — a contradiction between this directory and the Windows
driver work on the unmerged local `main`, reported rather than resolved. That work finds an
exact five-context ceiling in `amdxe.sys` (contexts 1-5 allocate, the sixth is rejected with
NTSTATUS `0xc01e0009`) and annotates it as matching Phoenix's five physical columns. The
measurements do not conflict; the causal reading does. `multi_partition_yolov8n_5col.log`
already ran five processes against the per-column `1x4.xclbin` and the partition set **stays
at four**, on columns 1-4 — the fifth process gets none. The context benchmark loaded
`4x4.xclbin`, which occupies all four columns as *one* partition, so five contexts each
wanting a four-column overlay is twenty column-occupancies on a device that exposes four:
the ceiling cannot be one-context-per-column. Its own +747.75 us switch penalty is the
signature of time-slicing one partition, where two contexts on separate columns would run
concurrently as `1x4.xclbin` measurably does at 3.65x. Also notes that the penalty is better
supported by the minima (61.10 us against 467.90 us, 7.7x) than by the overlapping means. The
deciding run is the same benchmark against `1x4.xclbin`. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#two-loads-in-one-bank-cost-a-cycle-and-the-int8-gemm-has-that-collision-where-bf16-does-not).


**`gemm_reblock_h11_npu.log`** — H11 run at last, proposed twice in this repo and never
executed. Switching the int8 path from `matmul_vectorized_4x2_mmul` (8 live accumulators) to
the `2x2` template already in `mm.cc` (4) improves EVERY static measure: 144 -> **88** bundles,
416 -> **32** byte frame, 33 -> **5** stack references, all twelve vector spills gone, and an
8-bundle hardware loop issuing 8 MACs — **1.000 vmac/cycle**, up from 0.889 and at the ceiling.
An issue-bound design should then gain ~12%. Two alternating A/B series gave best-to-best
**+0.8%** and **-2.1%**, medians **+2.0%** and **+0.4%** — inside +/-2%, sign unstable. **This
is the test `bank_ab_h12_npu.log` could not be:** not "we could not resolve 3%" but "a 12.5%
kernel improvement produced nothing measurable", which makes the absence itself the evidence
that the core is not the critical path. Also records a trap whose only symptom is silence:
`@iron.jit` keys its cache on the design and its compile-time arguments, NOT the kernel source,
so the first reblocked build silently reran the STOCK kernel — each arm now gets its own
`NPU_CACHE_HOME`. The shared toolchain was not modified; the reblocked `mm.cc` lives in
`kernels/gemm_reblock/aie2/` and the source lookup is redirected in-process. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-int8-gemm-issues-at-40-of-nameplate-and-a-3200-cycle-per-buffer-floor-caps-it).

**`accumulator_width_vs_count.log`** — the cheapest test in the current plan, and it refutes
its own hypothesis. `aie2_isa_static.log` measured five live 4x8x8 int8 accumulators as the
spill-free ceiling and the production int8 GEMM holds eight and spills (416-byte frame). But
both dtype paths in `mm.cc` ask for the SAME total width — int8 8x1024 bit, bf16 16x512 bit,
both 8192 bits across the same 8 of 9 registers — so the ceiling might have been a width
budget that generalises. It is not: **bf16 spills nothing**, a 64-byte frame whose 15 stack
references are all scalar, against int8's **12 vector spills** (12 slots x 32 B + 32 B scalar
reconciles the 416-byte frame exactly). Also establishes the file's shape: 9 registers
addressed at three granularities, `cm` full 1024-bit, `bml`/`bmh` halves, `amll`..`amhh`
quarters — so the earlier "9 names is a lower bound" was the whole file seen one way. int8's
4x2 blocking is the defect, and bf16 is the existence proof that a fitting blocking spills
nothing. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-int8-gemm-issues-at-40-of-nameplate-and-a-3200-cycle-per-buffer-floor-caps-it).

**`bank_check_validation.log`** — `tools/aie_bank_check.py` checked against a penalty that was
actually measured, on the `research/windows-lowlevel` branch, rather than only asserted. That
branch left both build caches on disk: one placement with the two operands sharing a bank and
one without, from a single kernel source whose compiled object hash is identical in both. Given
nothing but the cache directory and the operand names the tool reproduces the experiment's own
labels from the ELF alone, finds exactly **one** paired-load bundle in the compute body, and
the kernel's source fixes the trip count at 16×8×8 = **1,024** MACs per panel. Predicted
penalty 1 × 1,024 = **1,024** cycles per panel; measured 15,232 − 14,208 = **1,024**. Exact,
on a kernel this branch did not write. It also confirms the mechanism is *same-bundle* paired
loads, not two loads merely near each other. **The validation found a bug on its first run:**
the check looked only inside hardware loops and so reported "no penalty" on that very kernel,
whose compute lives in a *software* loop — as `mm.cc`'s accumulator-group body does too. The
check now walks software-loop bodies, and `--operands` makes the verdict about the two buffers
a paired load really reads instead of "some bank holds two buffers", which over-reports on a
padding buffer. Re-read, the int8 GEMM's software body carries **ten** paired-load bundles of
96, locating rather than changing the 240 cycles per call the cost model already charged as its
upper bound. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#two-loads-in-one-bank-cost-a-cycle-and-the-int8-gemm-has-that-collision-where-bf16-does-not).

**`im2col_bd_probe_npu.log`** — the two gates on K1's tooling proposal, run BEFORE any conv
kernel. SILICON 2.6 calls the mem tile's 4-D BD "the one address generator on the chip that can
do an im2col or a transpose in flight", and K1 proposes moving conv2dk3's ten `vshift`/`vmov`
bundles onto it. **Gate 1, on paper:** an im2col conv's K*K expansion CANCELS -- every input byte
feeds C_out MACs -- giving 1/C_out B/MAC, so against the int8 ceiling of 0.03125 B/MAC it is
stream-bound only below C_out = 32 and has 2x headroom at 64. Bandwidth is not what would kill
it. **Gate 2, on hardware, and the mechanism fails:** compute-free shim -> memtile -> shim, the
4-D pattern on the memtile's outbound stream, checked byte-for-byte against a host im2col.
Non-overlapping patterns pass (k=1: all 256 elements at 16x16, all 64 at 8x8); **every
overlapping one hangs the device** -- k=2 at 3.52x expansion and k=3 at 6.89x both return
`ERT_CMD_STATE_TIMEOUT`, as does k=3 with the forwarded fifo given the expanded object type, so
it is not a length mismatch. k=2 is the smallest overlap a square window can ask for, so the
boundary is not a large expansion factor. This refutes the ROUTE, not the silicon: the BD field
widths fit with room to spare, so **do not write a conv kernel against
`ObjectFifo.forward(dims_to_stream=...)`** -- try a raw BD outside the ObjectFifo abstraction.
Harness `kernels/im2col_bd/im2col_probe.py`.

**`bank_stall_observable_npu.log`** — the trace observable `bank_ab_h12_npu.log` asked for,
run on a kernel KNOWN to have the conflict, and **it works exactly**. `kernels/memory_placement/`
places both operands at explicit `aie.buffer` addresses and asserts the compiler bundled the two
loads (`vlda`/`vldb` on one line), refusing to run otherwise -- that assertion is what makes it a
valid positive control. Routing `MEMORY_STALL` through it: **one same-bank paired load costs
exactly one cycle and raises exactly one `MEMORY_STALL`.** At trip counts 512/1024/2048/4096 the
colliding arm reads 514/1026/2050/4098 events against a constant 2 separated, and the extra
cycles are 512/1024/2048/4096 -- the extra cycles, the extra events and the iteration count are
the same number at every point. Slopes 12.0 vs 11.0 cycles/iteration at r^2 = 1.0, kernel
byte-identical across arms, MLIR confirming `mem_bank = 1/1` vs `1/2`, zero variance over twenty
samples per arm. `LOCK_STALL` is useless here (10.5k in both arms, no trend). **H12 is now
measurable**: the reason it stalled was a ~3% wall-clock effect inseparable from machine drift,
and this observable is exact and inside the dispatch. Raw evidence in
`bank_stall_observable_separate_npu.log` and `bank_stall_observable_same_npu.log`.

**`bank_stall_control_npu.log`** -- **RETRACTED headline** (see above). Concluded that
`MEMORY_STALL` "cannot be trusted as a bank-conflict signal"; wrong, and the fault was its own
probe rather than the event -- its loop ran at 15.0 cycles/iteration for 4 loads, too loose to
pair them, and the penalty exists only for paired loads, so there was nothing to detect. Its
"constant 6 cycles, no rate effect" goes with it. Kept because two findings stand: the three
structural facts (a core's `.bss` is ~16 KB, lies entirely inside ONE 16 KB bank, and is NOT
moved by `stack_size`, which relocates only ObjectFifo buffers), and that at eight traced events
the counters are not reproducible because `ACTIVE` overflows the 64 KB buffer -- which is why the
replacement used four.

Its original entry is kept below verbatim as the superseded record -- every claim in it
about what `MEMORY_STALL` can see is overturned by the log above, and it is here so the
correction stays traceable rather than being tidied away. It read: the positive control
for the instrument
`bank_ab_h12_npu.log` proposed, run BEFORE spending a sitting on it, and **the instrument does
not work**. A same-bank dual-load conflict has to appear as `MEMORY_STALL`, which
`pmu_probe_npu.log` had already flagged as never having read nonzero here. In a loop built to
collide as hard as this design permits, **`MEMORY_STALL` reads 0 in every run of both
placements at every trip count**, so zero on the GEMM's arms would not distinguish "no conflict"
from "the event does not fire". No bank-specific event exists to fall back on, and `GROUP_STALL`
read *exactly* equal to `ACTIVE` in all six eight-event runs. The control did yield a clean
instruction-matched comparison and found **no rate effect**: colliding costs a *constant* 6
cycles more at 256, 512 and 2048 iterations, where a one-cycle per-iteration stall would have
cost 256/512/2048 -- so the INVARIANCE is the finding: those 6 cycles are paid once. Two causes
fit and neither is ruled out (a fixed loop-setup difference, or a one-time bank-arbitration
warm-up on first touch of the second bank); the headline does not depend on which. **Caveat that keeps H12 open:** the loop
runs at 15.0 cycles/iteration for 4 loads, so it has slack to absorb a one-cycle stall. Three
structural facts fell out, each having cost an attempt: a core's `.bss` is ~16 KB not 64 KB; it
lies entirely inside ONE 16 KB bank (0x75000-0x77C00, bank 29, fifo buffer at 0x78000, bank 30),
so no static array can straddle a boundary; and **`stack_size` does not move a kernel's `.bss`**,
only ObjectFifo buffers. And a trap: at eight traced events the counters are NOT reproducible --
one identical binary gave 15665/30724/15879 cycles -- because `ACTIVE` overflows the 64 KB trace
buffer; four events are exact. Harness `kernels/bank_placement/bank_stall_probe.py`.

**`bank_ab_h12_npu.log`** — H12 run on hardware, and the honest answer is that this machine
could not resolve it. The intervention is clean: raising the per-core `stack_size` from
`0xD00` to `0x2000` shifts every local buffer up, moving both A halves wholly into the empty
bank 3 while B stays in bank 2, with the same kernel source, tile shapes, fifo depths, DMA and
schedule, and a **byte-identical compiled kernel object** — only the addresses moved, and the
new layout is exactly what the arithmetic predicted before the build. But across **seven**
alternating series the two arms overlap and the sign of the difference changes between them:
best-to-best −4.6%, −2.9%, +2.5%, −2.7%, −0.4%, −5.8%, −1.1%, positive meaning separation was
faster. The colliding arm's *own* floor drifted **5.0%** between repeats of the identical
build, larger than the ~3% effect at stake, so a large speedup is excluded and H12 is neither
confirmed nor refuted. A peer session held about a core throughout; repeat on a quiet machine.
Validity check recorded: the control's best time sits just under the 2026-09-07 sweep's
minimum for the identical shape and tile, so the local harness introduces no offset. The route
that would settle it is the trace unit rather than wall time — read `ACTIVE` against
`LOCK_STALL` in core cycles inside the dispatch, where host contention cannot reach — which
needs a trace hook in `whole_array`. Harness `kernels/bank_placement/`; written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#two-loads-in-one-bank-cost-a-cycle-and-the-int8-gemm-has-that-collision-where-bf16-does-not).


**`pmode_clock_readback_npu.log`** — XRT's `max_clock_frequency_mhz` against power mode
*and* load, varied together: a 2048³ bf16 GEMM hold with `xrt-smi configure --pmode`
stepped through all five modes, then the same five idle, the monitor logging clock, mode
and engine utilization every 0.1–0.25 s. Busy, the readback is the mode's clock to the MHz
the trace unit measured (1800 `default`/`performance`/`turbo`, 1028 `balanced`, 800
`powersaver`); idle, 800 in every mode. Two runs: the first kept as contaminated (it
overlapped another session's 128-stream classifier sweep and the hold hung in the second
that sweep's XRT aborted), the second with the device checked idle first. `turbo`'s escape
error reproduced twice, the mode applying regardless. Cited by
[`docs/SILICON.md`](../../docs/SILICON.md) 1.7 and S0, [`docs/DECISIONS.md`](../../docs/DECISIONS.md),
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-aie-core-clock-measured-180-ghz-default-080-powersaver).

**`npu_monitor_poll_rate_npu.log`** — the NPU monitor's polling rate: three instances of
`tools/hwinfo_npu_bridge.exe` at 0.1 / 0.25 / 0.5 s across one GEMM hold read the same
engine-utilization mean (88.2 / 87.9 / 87.8 %) with a little more scatter at 0.1 s and no
dropouts, and the measured period is exact (0.100 / 0.250 / 0.500 s) once the process asks
Windows for a 1 ms timer tick — before that every period ran ~22 ms long. A clean 10 Hz
monitor beside a hold did not disturb it; the two hangs seen that evening are attributed,
with timestamps, to another session's concurrent-stream runs on the same device. Cited by
[`docs/SETUP.md`](../../docs/SETUP.md).

## mlir-aie examples on this hardware

**`mlir_aie_saxpy_npu.log`** — set up the open-source `Xilinx/mlir-aie` (IRON/Peano)
toolchain natively on Windows (new isolated `mlir-aie-iron` conda env, no WSL, no AMD
account) and ran a hand-written SAXPY kernel end to end on this machine's XDNA1 (Phoenix)
hardware, `PASS!` from a cold compile cache — shows custom-kernel bring-up is possible on
this hardware via a different toolchain than the one `physical_device.dll` blocks.

**`mlir_aie_examples_npu.log`** — three more of mlir-aie's own
`programming_examples/getting_started` designs run on this hardware (multi-column memcpy
bandwidth microbenchmark, a 4-core single-column reduce-max cascade, single-core int16
matmul at two shapes via the AOT + JIT-cache paths), all `PASS!` from a cold compile cache
— broader IRON programming-model coverage than the single SAXPY kernel alone.

**`mlir_aie_ml_examples_npu.log`** — five pure-Python `ml/` designs (elementwise add/mul,
relu/silu/gelu activations, a two-phase runtime-parameterized scale-and-shift, softmax,
swiglu), all `PASS!`, plus a survey of which `vision/`/`ml/` examples this machine couldn't
yet run (missing `make` and an OpenCV C++ dev package, or Strix-only, or left for a
dedicated pass).

**`mlir_aie_vision_examples_npu.log`** — `make` and OpenCV installed, then all four
`vision/*` designs (color_detect, color_threshold, edge_detect, vision_passthrough) and
three more `ml/*` designs needing `torch` for reference generation (bottleneck — a real
ResNet-family bottleneck block, conv2d plain and `--fuse_relu`), all `PASS!`.

**`mlir_aie_magika_mobilenet_npu.log`** — `ml/resnet/layers_conv2_x` (three chained ResNet
conv2_x bottleneck blocks across three NPU columns, `PASS!`, 1888.5us avg NPU time),
`ml/magika` group0 and group2 (Google's file-type-detection network; both PASS with large
negative EVM despite mlir-aie's own lit marking the design `XFAIL` upstream — didn't
reproduce as a numeric failure here, recorded as a discrepancy) — plus a tooling gap found
in magika's trace_py post-step (unrelated to the hardware PASS) and a
Windows-path-through-Git-Bash compile-flag fix; `ml/mobilenet`'s hardware paths are Strix
(npu2) only per its own README, so only its no-hardware numpy cross-validation could run
here (all blocks BIT-EXACT, not an NPU result).

## Kernels written for this repo's own gaps

**`groupnorm_bf16_kernel_npu.log`** — the first kernel written for this repo's own need
rather than an mlir-aie example (`kernels/groupnorm_bf16/`): a bf16 GroupNorm(32), i.e. the
`InstanceNormalization` that falls to CPU in `resnetv2_50x3_xint8.onnx`, checked against
the real node tensors and ORT's own CPU output at every shape
`bit/profile_instancenorm_splice_feasibility.log` profiled, with per-shape NPU time against
the profiled CPU cost and the projection.

**`groupnorm_bf16_handoff_floor_npu.log`** — the follow-up this left open: since
`resnetv2_50x3_xint8.onnx` runs under the VitisAI EP in `resnet_env17` (python 3.12) and
pyxrt can only load in ironenv (python 3.13, a hard ABI wall), any real splice is a
two-process handoff, not an in-process custom op — measured the floor of that handoff
(shared-memory ping-pong of the real byte volume + fp32/bf16 conversion, no ORT, no actual
NPU dispatch) at all six node shapes and found it erases every one of the 33/49 node wins
from `groupnorm_bf16_kernel_npu.log` — **0/49 nodes survive a real splice as currently
designed.**

**`groupnorm_bf16_handoff_floor_v2_npu.log`** — reopens the question above: the floor was
a slow conversion function, not physics. Three challenges to the v1 log, each checked with
a measurement: a DMA-floor arithmetic retraction (no new measurement — L is per-group
length, so L=301056 moves 32·L = 9.6M elements, and the kernel was already at the floor);
`ml_dtypes.astype()` has no SIMD path for bfloat16 (not a native numpy dtype, so it runs a
scalar loop) — replaced with a strided-view truncation and preallocated buffers, cutting
the combined pack+unpack at L=301056 from ~22.6ms to ~4.0ms (5.6×) and the real two-process
floor from 23.6ms to 5.7ms (CPU still wins there, 1.65× not 6.8×), while isolating the
protocol alone (`--no-convert`, zero conversion) measures 1.19ms/call, already **under**
CPU's 3.47ms; and a re-profile of the real model showing the QuantizeLinear/
DequantizeLinear nodes wrapping every InstanceNorm site add 56.8% on top of its own cost
(65.08ms across all 49 nodes, 11.1% of the model's latency, not InstanceNorm's 42.37ms /
7.2% alone) — the real, larger target an int8-native design would need to clear, which the
measured protocol floor already does at the hardest shape. No int8-native kernel is built;
this reopens the question rather than answering it.

**`mlir_aie_bf16_matmul_npu.log`** — pivot away from that memory-bound splice toward a
compute-bound op: a static inventory of bf16 support across `programming_examples/`
(matmul/eltwise/eltwise_unary/scale_shift/softmax/swiglu real on this chip,
conv2d/bottleneck/resnet int8-only, LayerNorm/RoPE/dwconv1d Strix-only), then the first
real hardware run of `basic/matrix_multiplication` bf16 on this machine — single_core 512^3
PASS at 116.56 GFLOPS, whole_array 4-column 512^3 PASS at 895.08 GFLOPS — plus five new
native-Windows toolchain fixes (a Make/MSBuild env-var case collision, a `powershell.exe`
CXXFLAGS quoting bug, `CMAKE_PREFIX_PATH` for XRT's CMake package, `xclbinutil`'s and
`pyxrt`'s real paths) and one portability fix in the external mlir-aie clone.

**`attention_bf16_kernel_npu.log`** — fused BF16 row-streaming Multi-Head Attention kernel
running across 8 AIE2 cores on physical Phoenix NPU, bit-accurate (<1% rel L2 error)
against MobileViT calibration golden tensors across stages 2, 3, and 4 (0.86 ms on Stage 4,
4.57 ms on Stage 3, 57.61 ms on Stage 2). Its recorded diagnosis is corrected by
`dispatch_floor_npu.log` — see the retraction list above.

**`bf16_matmul_niche_npu.log`** — the bf16 GEMM sweep that found this project's first
genuine NPU win: `kernels/bf16_matmul_sweep/cpu_matmul_sweep.py` (torch bf16 on this
machine's Zen4 cores) against the 4-column `whole_array.py` design across shapes larger
than the single 512³ point above. At 512³ the NPU's 895 GFLOPS **loses** to CPU bf16's
1100.6; past a crossover near N=1024 the NPU wins 1.18×–1.78×, peaking at 2072.5 GFLOPS at
1024³. Every NPU number is a verified PASS against numpy `A@B`.

**`bf16_matmul_k_limit_diagnosed_npu.log`** — follow-up that diagnoses the above log's
"`K ≥ 3072` fails, undiagnosed" close. **Not a threshold**: bisecting K shows the error
starts continuously around K/k≈23 and grows smoothly, always a systematic ~10–13%
undercount (a precision signature, not an addressing bug). **Root cause: the K-reduction
loop accumulates in a buffer typed `dtype_out`, not fp32** — `--dtype_out bf16` rounds
the running sum back to bf16 every reduction step, swamping small increments as it
grows. **Fix, free: `--dtype_out f32`** — clean PASS at K=2880/4096 on `whole_array`,
1741.7/1830.2 GFLOPS, same range as the bf16-output numbers above.

**`bf16_matmul_attention_scale_npu.log`** — does the fix, and the NPU win, hold at real
scale, not just the small shapes used to bisect K? M=2048, K=4096, N=4096 (a 7B-class
model's `d_model` contraction dim; Llama-2-7B and Mistral-7B both use 4096) — a shape
that FAILed outright before today's fix. With `--dtype_out f32`: PASS, **1776.7 GFLOPS
vs CPU bf16's 1333.1 (torch), a 1.33× NPU win**, in range with the smaller shapes above.
Also checked `attention_bf16/attention_kernels.cc` for the same bug: it doesn't have
it — its `Attn@V` reduction already accumulates in AIE2's native fp32 `accfloat`
accumulator across the full loop, casting to bf16 only once at the end. That kernel's
71–240× loss to CPU stays design- and size-driven (see below), not precision.

**`bf16_matmul_ffn_pipeline_npu.log`** — the actual multi-op pipeline measurement the
log above deferred: FFN up-projection (NPU) → GELU (CPU) → down-projection (NPU), both
matmuls at the same M=2048/K=4096/N=4096 shape (the real Llama-2-7B `d_ff=11008` width
hit a DMA-stride compile limit at N=8192 — `aie.dma_bd` stride 3 out of range — not
chased further; recorded, not investigated). GELU timed alone on the (2048,4096)
intermediate: 0.729 ms, under 1% of either ~39 ms stage. Chaining two real dispatches
with a real CPU op between them barely moves the ratio: **1738.6 GFLOPS NPU vs 1315.6
GFLOPS CPU (torch bf16) — 1.32×**, next to nothing off the single-GEMM 1.33×. Explicitly
a sum of independently measured stage costs plus an isolated activation cost, not a live
single-session run with real data handed off between stages — that hand-off cost, and
attention's own QK^T/Attn@V shapes, are still open.

**`bf16_matmul_ffn_real_shape_npu.log`** — chases the N=8192 DMA-stride limit the log
above deferred, all the way to the real d_ff=11008 shape, and finds three separate
DMA/BD toolchain limits, not one: (1) a C-output row-block byte-stride cap — a fixed
~4 MiB (2²² byte) stride, `m × 4 × N × dtype_out_bytes ≤ 2²²`, verified at three
independent points including a bf16-vs-f32 cross-check that confirmed it's byte- not
element-based; (2) a B-input per-core tile buffer word-length cap (16,383 words,
ruling out any `n` above ~511 at `k=64`); (3) a DMA "too many simultaneously active
buffer descriptors" compiler limit once the A-tensor reload pattern's repeat count
exceeds ~64 (empirically bisected: repeat_count=43 compiles fine, 86 doesn't — the
boundary is magnitude, not the "ugly" prime 43 in 11008=2⁸×43 as first suspected).
N=11008's factorization leaves exactly one tile size, `n=64`, surviving all three, and
correctness forces `m=16` — a quarter of the default. That tile-size compromise costs
real throughput: the up-projection alone drops to **909.97 GFLOPS** (vs 1793 GFLOPS for
the same K=4096 contraction at an unconstrained tile size); the down-projection, never
constrained, still hits **1800.86 GFLOPS**. Blended pipeline: **1192.9 GFLOPS NPU vs
1309.6 GFLOPS CPU — CPU wins 1.10×**, reversing the square-shape approximation's 1.32×
NPU win above. Both stages still PASS numpy verification — this is a DMA-descriptor
limit in `whole_array.py`'s generic tiling strategy, not a precision or `aie::mmul`
problem, and not necessarily true of a shape-specific fused kernel. Not tried:
`c_col_maj`/`b_col_maj` as an alternate way around limit 1, and Mistral-7B's
`d_ff=14336` (2¹¹×7, a friendlier factorization).

**`bf16_matmul_ffn_shape_variants_npu.log`** — both of the log above's untried items,
and neither answer is the naive guess. `--c-col-maj 1` does dodge the byte-stride limit
(compiles clean at default `m=64`/`n=32` for `N=11008`) but only reaches **811.69
GFLOPS — worse** than the `m=16` row-major workaround, because pushing the tile back up
to chase more throughput just trades limit 1 for a fourth one: AIE2's ~64 KiB L1
tile memory (shared by the double-buffered A/B/C tiles) — confirmed by two separate
"allocated buffers exceeded available memory" failures (`n=64`/default `m`, and
`m=128`/default `n`). Mistral-7B's `d_ff=14336` (`2¹¹×7`) hits the identical `m=16` cap
as Llama's `d_ff=11008` (limit 1 scales with `N` alone, not its factorization) — but its
cleaner factorization admits `n=128` where `11008` was stuck at `n=64` (`n=448` overflows
the same L1-memory limit `c_col_maj` hit). That alone is a 74% throughput jump: 835.63
GFLOPS at `n=64` → **1454.37 GFLOPS at `n=128`**. Full pipeline (up at `n=128`, down at
default tiles, unconstrained): **1542.9 GFLOPS NPU vs 1362.8 GFLOPS CPU — NPU wins
1.13×**, the *opposite* verdict from Llama-2-7B on the identical hardware and toolchain.
The determinant this project can now name precisely: whether `d_ff`'s factorization
admits an `n`-tile ≥~128 once `m` is forced down by the fixed byte-stride cap — a
property of the specific integer, not of "real-world shape" in general.

**`int8_matmul_sweep_npu.log`** — the same upstream `whole_array.py` design in int8
(`--dtype_in i8 --dtype_out i32`) at the bf16 sweep's shapes, bf16→f32 re-run in the same
sitting, an M-edge sweep with an m-tile control, a tile check, four L1 ceiling probes, and
the CPU int8 GEMM baseline this repo never had (`kernels/int8_matmul_sweep/`, two CPU
kernels: torch `_int_mm` and ORT `MatMulInteger`; torch's is faster and is the verdict
line). **At the default tile the NPU's headline dtype loses to the CPU's own int8 kernel
almost everywhere** and runs only 1.1–1.5× the bf16 rate; **`n=64`, which int8's half-size
tiles leave L1 room for and bf16's do not (bf16 misses by exactly the 3,328 B stack),
doubles it bit-exact to 4448–4607 GOPS** — a **1.10×–1.83× NPU win at M ≥ 512, N ≥ 2048**
by the mean, thin enough that the CPU kernel's best-case time takes back the K=N=4096 rows. The
small-M loss tracks the forced tile `m`, not token count.

**`bf16_matmul_n64_single_buffer_npu.log`** — closes the item above. A 13-line patch to
`whole_array.py` (not part of this repo; lives in the local `~/mlir-aie` checkout) adds
`--c-single-buffer {0,1}`, dropping the per-core `C_L1L2` output-tile FIFO from depth 2 to
1 — that fifo has no compute/compute overlap to lose (`core_fn` acquires it once per
output tile), only compute/next-tile-DMA-out overlap this gives up, and it frees exactly
`m·n·dtype_out_bytes` of L1 (16,384 B at m=n=64, f32 out) — precisely bf16 `n=64`'s
3,328 B shortfall. Result: **the CPU-bf16 win margin widens from 1.19×–1.35× (default
tile) to 1.29×–1.89× at M, N ≥ 1024** — 2048³'s 1.89× is the largest bf16 GEMM margin
measured in this project — while 512³ still loses (0.70×, barely moved from 0.65×). An
overlap-cost control (default tile with `--c-single-buffer 1` at the same shape) reads
1740.51 vs 1801.18 GFLOPS at the normal double-buffered depth, a 3.4% loss confirming the
gain above is the bigger tile, not an accident of the buffer-depth change.

**`gemm_tile_sweep_c_single_buffer_npu.log`** — the tile sweep the single-buffered C tile
makes possible: ten bf16 m/k/n tiles at 2048³ on the generic 4×4 `whole_array.py` design,
three with B column-major, the four survivors across 1024³ / 4096×2048×2048 / 2048×4096×4096,
three int8 tiles, and a same-sitting CPU bf16 baseline; clean sitting, the device watched by
the monitor. The L1 model `2A + 2B + (1|2)·C + 3,328 B` predicted all 28 compile outcomes;
64/64/64 and 32/64/128 are the best reachable bf16 tiles (2501.71 / 2494.61 GFLOPS at 2048³,
2700.44 at 2048×4096×4096 — the repo's best, 36.6% of peak, 1.89× CPU bf16); SILICON.md 3.1's
B/MAC model is missing a B-run-length term and a k term; int8 gains 9–13% from the freed
L1. Supersedes an 18:33 screen of the same tiles taken under a running AdaRound job, kept in
its appendix. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-bf16-tile-sweep-what-the-freed-16-kb-buys-and-where-the-bmac-model-stops) and `docs/SILICON.md` 3.1/K2.

**`int8_matmul_reference_float64_npu.log`** — not a hardware measurement: the int8 sweeps'
bit-exact oracle, moved from numpy's scalar int64 loop (532.7 s for the 2048×4096×4096
reference alone, against 33.6 ms of NPU time — the 760.9 s wall in `int8_matmul_sweep_npu.log`)
to a float64 BLAS matmul (0.244 s, 2182×), exact while `K · max|a| · max|b| ≤ 2^53`.
Proven bit-identical on the sweep's own data at three shapes, a full-range worst case and
the all −128 bound case at K = 4096; the sweep rows re-run through the patched
`whole_array.py` on the NPU `PASS!` with the same seed (2048×4096×4096 wall 5.7 s / 1.1 s
first / cache-warm), so every earlier `PASS!` keeps its meaning. The patch is
`kernels/int8_matmul_sweep/whole_array_int_reference_float64.patch`; `cpu_int8_matmul_sweep.py`
got the same oracle. Patching `whole_array.py` changes its JIT-cache hash, so the first run of
each config recompiles (seconds). CPU timings were taken beside another session's yolov6n
evaluation runs. Cited by [`kernels/README.md`](../../kernels/README.md).

## Dispatch floor and the int8 conv verdict

**`conv_issue_rate_decomposed.log`** — the measurement objective K1 asked for and that no
instrument had ever been pointed at, because the conv build cache had not survived. Rebuilt,
and the 11.3x vendor-to-open gap resolves without a trace: bundle count is cycle count, so
`vmac` per bundle is MACs per cycle, and the int8 GEMM issues **0.889** against the best conv
loop's **0.333**, the 3x3's main loop's **0.222** and the 1x1's hot loop's **0.045** — with two
of the 1x1's three loops issuing no MAC at all. **It is issue rate, not data movement.** Two
different defects: the 1x1 never keeps an accumulator in a register (loads four quarters from
memory, issues one `vmac`, stores four back, idles six of 22 bundles, names 3 of 9
accumulators), while the 3x3 keeps `cm1`-`cm4` live and still spends six of eighteen bundles on
`vshift` plus four on `vmov` doing sliding-window realignment in issue slots — precisely what
K1's own tooling section proposes moving to the mem tile's 4-D descriptors. Also notes the 3x3
carries one same-bank paired load, worth ~5% and not a lever. **Careful:** the 20x and 4x are
ceilings on unused issue slots, not predictions of a rewrite, and per-loop densities are
unweighted by trip count. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-open-convs-113-gap-to-the-vendor-is-issue-rate-and-it-is-visible-without-a-trace).

**`conv_accum_residency_npu.log`** — the fix for the defect `conv_issue_rate_decomposed.log`
found, measured end to end. Both 1x1 conv kernels held `MMUL4x8x8 acc_tmp[4]` indexed by a loop
whose trip count is a RUNTIME value; registers cannot be dynamically addressed, so the array
lived in memory and every `.mac()` was load-four-quarters / mac / store-four-quarters. Peeling
the `n == 4` case into four NAMED accumulators takes the hot loop from **22 bundles with one
vmac to 14 with four -- 0.045 to 0.286 MACs/cycle** -- and marginal throughput from
**115.6-117.1 to 348.4-350.3 GOPS**, 2.99x and 3.01x across two series, every shape passing the
sweep's golden gate. **This is the falsifiable prediction gemm_reblock_h11_npu.log set up and it
held:** the same class of change to the int8 GEMM moved the wall clock by nothing because that
design is delivery-bound, while the conv at 4.5% of peak was genuinely issue-bound -- the first
time this repo separated the two by intervening rather than modelling. **The op class does NOT
reopen:** the CPU (onnxruntime 1.22.1, ORT CPU EP, QDQ int8 on VNNI, same sitting, run twice)
holds 823.7-839.5 GOPS and still wins **2.4x**, down from 7.2x. What changed is the reason, from
"the kernel uses 5% of its issue slots" to "even with the slots used, one column does not reach a
VNNI-equipped Zen4". **A methodological warning worth more than the number:** the bottleneck is a
three-stage core-to-core pipeline and patching only stage 1 moved 32x32 by 2.4% -- a null result
that would have been reported as "the conv is delivery-bound too". It was Amdahl; stage 3 still
had its accumulators in memory, and fixing both gave 2.42x at the same shape. Two caveats: the
stock arm re-measured at 115.6-117.1 rather than the published 146.1, because that figure
predates the 2026-09-07 width fix that rewrote the same loop (INFERENCE, the pre-fix kernel was
not built as a third arm); and conv2dk3, the 3x3 middle stage, was not touched and is the
likeliest remaining rate-limiter. Host-load witness reads PEER, which can only depress the CPU
figure and so makes the verdict conservative. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-11-convs-accumulators-were-in-memory-putting-them-in-registers-is-worth-3-and-the-op-class-still-loses).

**`dispatch_cpp_runlist_npu.log`** — the same passthrough through a standalone **C++** XRT host
(`kernels/dispatch_floor/dispatch_runner.cpp`, no Python in it), run back to back with the
Python arm in one sitting because latency drifts between sittings. **The 36 us floor is the
driver's, not the binding's:** C++ reads 36.7 us at N=64 against pyxrt's 35.9 us, the two
agreeing within ~2% from N=4 up, with Python marginally *faster* at N >= 16. So no host-side
rewrite goes below it, and `dispatch_runlist_npu.log`'s figure was never pybind overhead.
**A deployable C++ runner does reach it**, which closes the open item both earlier logs named:
same design, same sitting, 671.5 us (IRON unbatched) / 498.5 us (IRON batched 64) / ~108 us
(C++ one call) / **36.7 us** (C++ batched 64) — 13.6x better than batched IRON, because IRON's
~500 us host share is work a cached-handle host pays once at startup (33-60 ms) rather than per
call. **Four thresholds now, not three**, none of them retracted. Two limits stand: batching
only pays from N >= 4-8, and a single dispatch costs ~108 us even in C++, of which only
~20-30 us was ever the binding. Persistent runlists buy ~9% at N=64 end-to-end (40.3 -> 36.8 us)
and 1.40x at N=1 — the win is overwhelmingly not being IRON, not reusing the list. **This log
corrects its own first version**, which said the rebuild/persistent gap was ~20 us of runlist
CONSTRUCTION: construction sits outside the timed region in every arm, so it had not been
measured at all. A `built` arm that times it properly puts construction at ~18 us fixed plus
~3.3 us per run added — it grows with the batch (~220 us for a 64-run list) rather than
amortising — while the fresh-versus-reused EXECUTION gap is ~27 us at N=1 and gone by N=8.
**Also fixes a wrong-design
bug** in `measure_runlist.py`: its "newest cache entry holding both files" rule excluded nothing
(every IRON design writes `insts.bin`) and timed a 1052-instruction-word design as if it were
the 75-word passthrough, reporting 777.7 us and FAILED VERIFICATION; it now captures the paths
`CompilableDesign.compile()` returns and refuses to guess. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#a-c-xrt-host-reaches-the-device-floor-and-36-µs-is-not-a-python-artifact).

**`iron_batch_npu.log`** — the host path `dispatch_runlist_npu.log` said was missing, written
and measured. `kernels/dispatch_floor/iron_batch.py` patches IRON's own transaction submit so
an ordinary `@iron.jit` design queues unstarted runs into a `pyxrt.runlist`, with nothing
outside this repo modified. **The device half reproduces from inside IRON** — 37.5–37.9 us per
dispatch at N=64 across two series, against the raw harness's 36.3 us. **The wall clock barely
moves:** 1.26x and 1.37x on the passthrough, 1.23x on GroupNorm, because IRON's per-call host
work is a near-constant **~500 us, flat in batch size**, which batching cannot touch. That
leaves a batched `@iron.jit` call at **~531 us**, so there are three thresholds, not two:
~617 us unbatched IRON, ~531 us batched IRON, ~36 us batched raw pyxrt. Against ~531 us none
of the four small-op verdicts `dispatch_runlist_npu.log` reopened survives. The run also
closes that log's other caveat: a real 8-core bf16 kernel (GroupNorm, L=150528) batches to
823.8-838.3 us per dispatch, which is **its own compute** — `groupnorm_bf16_kernel_npu.log`
independently measured 835.8 us at this shape — so a real kernel's configuration cost does not
swamp the passthrough's floor. The ~500 us residue is `dispatch_floor_npu.log`'s 447.3 us host
term measured from a different direction and shown independent of how the submit is done; at
37.5 us of device against ~500 us of host it is now the larger term by more than an order of
magnitude. **Batching gives up per-call completion status entirely** — a run inside a runlist
cannot be polled (`run.state()` raises), so verifying output buffers is the only correctness
gate, and every row is gated on it. Only the transaction submit path is batched; full-ELF is
refused with a clear message. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#batching-reaches-the-device-floor-from-inside-iron--and-irons-own-host-work-eats-almost-all-of-it).

**`dispatch_runlist_npu.log`** — the measurement `dispatch_floor_npu.log` asked for and
`docs/DECISIONS.md` recorded as unrun (*"Nothing has been run -- this is an API-existence
check and the next measurement to make"*). Batched `pyxrt.runlist` submission amortises the
same 32 KB passthrough to **36.3 us** per dispatch against IRON's 617.0 us, a **17x** drop,
reproduced at 35.9/36.3/36.0/36.3 across four runs. Raw pyxrt single-dispatch is ~140 us,
already **below** the 169.8 us the earlier log called the hardware bracket, and the batched
figure is a quarter of it — so that bracket is not irreducible silicon. Four of the six ops
`docs/SILICON.md` 3.4 lists as closed by the floor now clear it (MobileNetV2 48x, MobileViT
stage-2 attention and bf16 attention stage 2 6.7x, GroupNorm at L<=18816 6.5x); attention
stages 3 and 4 stay under. **Two limits:** it is a THROUGHPUT figure — 36 us holds with 64
dispatches in flight, a one-shot call still pays ~140 us raw — and it is raw pyxrt, while
`@iron.jit` uses no runlists, so a real design pays the old floor until that host path is
written. Getting the harness working needed two fixes that had each produced a wrong answer
rather than an error: `kernel(...)` creates and **starts** a run, so runlist entries must use
`pyxrt.run(kernel)` + `set_arg`; and the cache resolver looked for `*.txt` when the
instruction stream is `insts.bin`. Written up in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#batched-submission-drops-the-dispatch-floor-17-and-reopens-four-closed-verdicts).

**`dispatch_floor_npu.log`** — the per-dispatch cost measured IN ISOLATION at last
(`kernels/dispatch_floor/measure_floor.py`), with a design that has no compute tile at all
(shim->memtile->shim via ObjectFifo.forward()) so there is no kernel math to attribute time
to: payloads swept 8KB-32MB, output verified per payload, compile excluded. Confirms the
'~185-200us' constant every earlier isolated-op verdict rested on — the HARDWARE floor is
169.8us (R^2=1.0000) — while showing that a kernel does not pay it: through the @iron.jit
path a call costs 617.0us (R^2=0.9998), 3.6x more, which is what both groupnorm_bf16 and
attention_bf16 were actually charged. The 447.3us difference is host-side and flat in
payload size but is NOT merely Python overhead (the hardware bracket opens only after
hw_context lookup, kernel-handle retrieval and buffer coherence; tracked Python work is
~100us/call). Also: dispatch dominates everything under ~0.5MB, streaming runs 12.2-13.8
GB/s, and the go/no-go threshold for writing any future kernel is a CPU time above ~617us
(IRON) or ~170us (zero-overhead best case). Closes the attention thread by correcting its
diagnosis without overturning its verdict: at Stage 2 dispatch is ~1% of the measured
57,610us, the real cause is that attention_kernels.cc uses aie::mmul zero times and
hand-rolls dot products with a horizontal reduce_add per output element, reaching 0.61
GFLOPS on hardware measured at 895 — but even a perfect kernel only ties at Stage 2 and
still loses at Stages 3 and 4.

**`conv2x_int8_cpu_baseline.log`** — the CPU baseline for mlir-aie's
ml/resnet/layers_conv2_x (3 ResNet bottlenecks chained core-to-core across 3 columns, int8,
one dispatch), which had run and PASSed here since 2026-09-06 with its CPU side never
measured. It was the best remaining structural idea because it fixes every flaw diagnosed
in the attention kernel — right dtype, mlir-aie's own validated int8 conv kernels, dispatch
amortized to ~25% of wall, no two-process handoff — and it still loses: 436.21 MFLOP, all
five rows in one sitting, NPU int8 1869.6us hardware / 2497.8us end-to-end against CPU ORT
QDQ int8 295us, i.e. **the CPU wins 6.3x on the hardware bracket and 8.5x end to end**
(int8 row verified genuinely int8: ORT's optimized graph runs 10 QLinearConv + 3
QLinearAdd). Two things fall out beyond the verdict: torch fp32 (1856us) lands within 1% of
the NPU's hardware bracket, so benchmarking against torch alone — the baseline the
attention kernel used — would have read as parity and been wrong by 6.3x, making this the
second case after the MobileViT splice where the CPU kernel choice, not the NPU, decided
the verdict; and the design's own harness reports both brackets, so its 628.2us of host
cost independently reproduces dispatch_floor_npu.log's 617.0us to ~2% on a completely
different design. Scope is deliberately limited to this design at this shape — one shape on
3 columns, against the same silicon's 895 GFLOPS 4-column bf16 matmul, so column count and
32x32 spatial/tile utilization remain untested and the op class is NOT closed.

> Both of that entry's own claims are dead: the "628.2us reproduces 617.0us to ~2%"
> agreement is **retracted** as a bracket mismatch, and "the op class is NOT closed" is
> **superseded** by `bottleneck_spatial_sweep_npu.log` below — the op class **is** now
> closed. The 6.3x/8.5x verdict itself stands.

**`bottleneck_spatial_sweep_npu.log`** — the experiment conv2x asked for, run: one
standalone `ml/bottleneck` swept across spatial sizes on the NPU
(`kernels/bottleneck_sweep/sweep.py`) against the identical arithmetic through ORT QDQ int8
(`cpu_sweep.py`), same shapes, one sitting, every NPU shape checked against mlir-aie's own
torch int8 golden before its timings count. `tensor_h` is the free axis (every L1 buffer is
`tensor_w`*channels, none scale with height). NPU hardware throughput DOES scale — 116.7 ->
143.7 GOPS across a 16x increase in work, marginal 146.1 GOPS at r^2=0.99999 — and the
CPU's marginal rate over the same shapes is 819.0, so **the CPU wins 7.6x at 32x32 and 5.7x
at 512x32 on the hardware bracket** (11.4x / 6.0x end to end). 512x32 is simultaneously the
NPU's best point and the CPU's worst (its only sub-900 GOPS row, working set past cache)
and the CPU still wins by 5.7x: utilization was worth 23% against a 570% gap, which closes
the op class rather than the shape. Two further findings: `tensor_w`=32 is a hard `aiecc`
ceiling — 56x56, 32x64, 64x64, 128x64 all fail with `'aie.tile' op allocated buffers
exceeded available memory` because Tile(0,4) needs five w*256-byte buffers plus stack
against 64 KB, so ResNet50's real 56x56 conv2_x cannot compile on this design at all and
the 32x32 `layers_conv2_x` runs is just the largest square that fits; and the host-cost
bracket confusion is corrected — the like-for-like dispatch floor for a wall-minus-hardware
measurement is the 447.3us HOST intercept, not the 617.0us wall one, so conv2x's "2%"
agreement was numerology and this sweep's own host cost (608-874us) is not flat but grows
with payload. Open, and now specific: column count (1-3 of a 4x5 array) and kernel quality
(146 GOPS is ~7% of one column's ~2 TOPS int8 architectural peak; `conv2dk1.cc`/
`conv2dk3.cc` unread).

> The op-class verdict stands. Its **`tensor_w`=32 ceiling is superseded** (the real
> ceiling for this channel config is 44, and 56×56 has since compiled and run), and its
> **kernel-quality open item is closed** — both kernels have been read, both vectorize
> correctly with `aie::mmul`, and the width bug in them is fixed. Column count remains the
> one lever untested.

## The width fixes and ResNet50's real shape

**`conv2dk3_widthfix_npu.log`** — isolated single-worker test fixing upstream `conv2dk3`'s
width-32 hardcode in the local mlir-aie checkout: 100% bit-exact across 32x32, 32x36,
32x40, 32x48 and 32x64, independently reproduced in a fresh session.

**`bottleneck_widthfix_npu.log`** — the same class of bug in `conv2dk1`/`conv2dk1_skip`,
found while re-testing `conv2dk3`'s fix, and verified end-to-end through the real
`bottleneck.py` design's torch-golden gate at `tensor_w` 32/36/40/44. The real ceiling for
this channel config is `tensor_w`=44, not the 32 recorded in
`bottleneck_spatial_sweep_npu.log`.

**`bottleneck_w56_npu.log`** — ResNet50's actual 56×56 conv2_x shape reached, by
single-buffering Tile(0,4)'s final output FIFO. Both 32×56 and 56×56 verified against the
torch golden. **The verdict got worse, not better:** NPU hardware 4.2435 ms vs CPU 0.3327
ms, **CPU wins 12.75×**, against the 5.7–11.4× range at every compile-limited 32-wide
shape.

## Native Windows XRT driver and DPU microcode disassembly

**`windows_xrt_driver_bench.log`** — micro-benchmarks of AMD's native Windows kernel driver (`amdxe.sys`) via `pyxrt.pyd` (Python 3.13) on Desktop 2 (Ryzen 7 8700G, Phoenix XDNA1 NPU): one-time device open floor (61.69 ms), hardware context allocation (77.71 ms), unified memory BO allocation/mapping/sync across 64 B to 16 MB (0.78–0.90 µs sub-microsecond sync floor at <= 4 KB, peaking at 296.17 GB/s D2H at 16 MB), userspace command dispatch preparation (8.76 µs across 8 arguments), and hardware runlist batching (3.39 µs/run). Documents the Windows KDMA restriction and `pyxrt.bo.flags.host_only` requirement.

**`windows_context_switch_bench.log`** — hardware context scaling and context-switch penalty benchmark on `amdxe.sys`: Context #1 cold setup (78.63 ms), Contexts #2–5 warm allocation (5.38–5.78 ms), context #6 refusal with NTSTATUS `0xc01e0009` (proving the 5-context physical silicon boundary), instantaneous slot recycling upon garbage collection (4.98 ms), and same-context (120.25 µs) vs alternating cross-context dispatch (867.99 µs), quantifying an ~748 µs driver/firmware context-switch penalty (7.22x slowdown).

**`dpu_transaction_disasm.log`** — a heuristic byte scan (not a disassembly) of the `mc_code` bytefields inside a compiled `.xmodel`, via `tools/dpu_transaction_disasm.py`. On FastDepth it finds 2,570 ~48-byte-strided records, bimodal at 49.38% / 37.35% in the assumed opcode position — consistent with a mostly pointwise-and-depthwise graph. **No DPU ISA is recovered**; the opcode table is a six-entry guess and 13.27% of the reported "opcodes" are ASCII metadata strings. A BiSeNetV2 row previously published here (4,630 packets, 43.17% / 32.61% / 0.24% / 23.97%) is **retracted**: no log reproduces it, and it carried FastDepth's own xmodel hash. **This file was mis-encoded on disk** (ASCII decoded as UTF-16LE, re-encoded as UTF-8), which is why `grep` against it silently matched nothing — and why an unbacked BiSeNetV2 table sat unnoticed beside a log that never contained it. Decoded in place to UTF-8 on 2026-09-09; the recovery is byte-exact (the decoded text re-encodes to the original bytes exactly, asserted before writing) and the line count is unchanged at 82. Only the encoding changed.

## Windows local memory placement

Desktop 2, 2026-09-09: [method, formulas and limits](../../docs/BENCHMARKS.md#windows-low-level-research-placement-and-xint8-arithmetic).

- `memory_desktop2_20260909_m01_{separate_1,same_1,same_2,separate_2,separate_3,same_3}.log`
  are the repeated address intervention; `same_offset_{64,128,256}` are address-offset
  controls and `single_same` is the single-load control. The `single_separate` preflight
  stopped; [the successful replacement](memory_single_separate_desktop2_20260909_02.log)
  completes that control. Earlier `memory_{same,separate}_dual_desktop2_20260909_01.log`
  are exploratory successful runs.
- `gemm_desktop2_20260909_g03_{separate_1,same_1,same_2,separate_2,separate_alternate,same_alternate}.log`
  are the selected GEMM matrix. Earlier `gemm_placement_*`, `g01` and `g02` logs retain
  bring-up, preflight failures and a partial context-creation failure; they are not
  pooled into the selected matrix. All complete selected runs check every output.
- `check_memory_object_desktop2_20260909_01.log` exposed truncated disassembly;
  `02` validates the corrected full-function extraction. Neither is a timing run.
- [Toolchain capture](toolchain_desktop2_20260909_01.log) records the existing release,
  compiler hashes and local patch hashes. The measurement logs carry research source
  hashes and host/device witnesses. No external toolchain files were changed.

The [checkpoint archive](../quant/lowlevel_desktop2_20260909_evidence.zip) retains
numerical outputs, small fixture models, placement reports, function disassemblies,
extracted function bytes and final trace captures. Each run's JSON retains all measured
cycle samples; a trace file is the final capture, not a trace of every call.
[Archive validation and aggregate](../quant/summary_lowlevel_desktop2_20260909_01.log)
record its hash and recount the arithmetic results from the saved output arrays.

## Sub-byte W4A8 weight quantization and roofline analysis

[`notes_w4a8_aie2_roofline.md`](notes_w4a8_aie2_roofline.md) — formal micro-architectural,
VLIW co-issuing, and roofline analysis of W4A8 sub-byte quantization on AMD Phoenix AIE2 (XDNA1).
Resolves the ISA discrepancy between AMD's `device.yaml` (which omitted int8xint4 for AIE2) and
physical silicon, proving that AIE2 possesses a native 512 MACs/cycle `aie::mmul<4,16,8,int8,int4>`
engine alongside zero-overhead hardware load-unpack (`vldb.unpack.s8.s4` in slot `[b]`). Derives the
compilable tile geometry frontier under L1 capacity (2A + 2B_bytes + (1|2)C + 3328 <= 65536 B,
unlocking double-buffered C at 64×128×64) and evaluates memory-bound (M <= 16, ~1.9x–2.0x win across
28 GB/s DDR cap) and compute-bound (M >= 64, 1.24x–1.26x speedup tracking 20% L3 byte reduction) regimes.

## Mixed-precision A16W8 feasibility and graph-lowering audit

[`notes_a16w8_feasibility_audit.md`](notes_a16w8_feasibility_audit.md) — formal micro-architectural
and graph-lowering feasibility audit of INT16 activation × INT8 weight (A16W8) mixed precision on
AMD Phoenix AIE2 (XDNA1). Reconciles AMD's silicon specification (`device.yaml`: 128 MACs/cycle native
for `int16xint8`, 4,096 GOPS array peak) with physical VitisAI EP rejection (0/394 nodes on NPU,
26.18 ms CPU fallback in `results/a16w8/diag_resnet50_a16w8_npu.log`). Proves the opset-17
`com.microsoft` domain lockout mechanism, details the AIE2 `aie::mmul<4,8,4>` vector intrinsic and
32-bit accumulator headroom (K ≤ 516), and models dynamic range preservation (+48.2 dB SQNR)
preventing PTQ collapse in MobileViT-XXS (softmax attention) and YOLOv8n-pose (OKS keypoint jitter).

## MemTile 4-D BD in-flight receptive field generation (im2col) specification

[`notes_memtile_4d_im2col_specification.md`](notes_memtile_4d_im2col_specification.md) — formal
register-level 4-D Buffer Descriptor (BD) configuration and mathematical dataflow specification for
Memory Tile in-flight receptive field generation (`im2col`) on AMD Phoenix AIE2 (XDNA1). Resolves the
`ERT_CMD_STATE_TIMEOUT` in `results/aie/im2col_bd_probe_npu.log` as an ObjectFifo token-synchronization
deadlock (ceil(1764/256) = 7 consumer lock acquisitions against 1 producer token) rather than an AGU
bounds fault. Details concrete register bitfields (Registers 0–7 + Iteration/Lock control) for 2D spatial
and 5-D multi-channel (H_out, W_out, K_h, K_w, C_in) streaming via the 6-bit `Iteration_Wrap` extension.
Proves the mathematical cancellation law yielding strictly 1/C_out bytes per MAC (compute-bound with 2×
headroom at C_out = 64), and calculates the elimination of 9 standalone `vshift`/`vmov` realignment
bundles in `conv2dk3`'s 18-cycle hot loop to project a 4.0× MAC issue density uplift (0.222 to 0.889
vmac/cycle) closing the 11.3× vendor DPU gap.

## Column 0 architectural audit & 5th column feasibility specification

[`notes_column0_architecture_audit.md`](notes_column0_architecture_audit.md) — comprehensive
micro-architectural audit and feasibility specification for Column 0 (the 5th physical column) on AMD Phoenix
AIE2 (XDNA1). Reconciles the physical 5-column die (20 cores, 5 MemTiles, 3.84 MB SRAM, 18.43 TOPS INT8 @ 1.80 GHz)
with the 4-column overlay convention. Audits the in-tree `amdxdna` Linux kernel driver (`drivers/accel/amdxdna/`),
revealing that `dev_npu1_info.first_col = 1` enforces the 4-column boundary for sub-allocations, but contains an
explicit bypass (`aie2_ctx.c:654-657`) triggering `Force start from col 0` when `num_col = 5`. Proves that between
Column 1 and Column 0, the switchbox provides 20 bidirectional 32-bit streaming channels (144.0 GB/s per direction @
1.80 GHz), allowing Column 0 compute cores and MemTile to be completely fed and drained via Column 1's Shim DMA even
if Column 0's NoC DMA is unbonded or reserved. Details the MLIR-AIE dialect modifications (`_MAX_COLS = 5`,
`AIEDeviceNPU1_5col`, `VirtualizedNPU1TargetModel(5, 0)`) and outlines the three-phase hardware verification roadmap
to unlock +25.0% compute and SRAM capacity.

## Inter-core 512-bit accumulator cascade GEMM model

[`notes_accumulator_cascade_gemm_model.md`](notes_accumulator_cascade_gemm_model.md) — formal
theoretical and cycle-accurate model of the 512-bit vertical inter-core accumulator cascade on AMD Phoenix
AIE2 (XDNA1). Formulates a 4-core column-wise K-reduction scheme across rows 2 → 3 → 4 → 5 over the private,
switchless 512-bit cascade bus (115.2 GB/s per column @ 1.80 GHz). Quantifies the complete elimination of
intermediate C-tile memory traffic in local L1 (saving 1.02 MB of RMW over the 256-bit bus for 64×64×64 at
K=2048, which exceeds total input activation volume by 1.94×). Eliminates intermediate C-tile buffers (0 Bytes
in L1 for Cores 0..2), enabling wide GEMM tiles (bf16 128×64×64 and int8 128×64×128) to compile within the
64 KB L1 limit with substantial headroom. Demonstrates disjoint bank allocation across the 4 physical SRAM banks,
eradicating the 1-cycle same-bank paired-load penalty and the 87 non-loop C-staging bundles per accumulator group
in `matmul_i8_i32`. Projects an inner loop MAC issue density uplift from 41.9% (107.3 MACs/cycle) to 90.9% (232.7
MACs/cycle), delivering a 2.16× wall-clock throughput speedup (4,368 → ~9,450 GOPS at 4096×2048²) on Phoenix hardware.

## Master closed-form roofline model and empirical pipeline reconciliation

[`notes_master_xdna1_roofline_synthesis.md`](notes_master_xdna1_roofline_synthesis.md) — master
closed-form roofline model and empirical pipeline reconciliation on AMD Phoenix AIE2 (XDNA1). Reconciles
physical micro-architectural hardware ceilings (14.75 TOPS INT8 @ 1.80 GHz, 26–28 GB/s DRAM bandwidth, 7.0 GB/s
shim stream rate, and 1-cycle paired same-bank load hazard) against measured end-to-end inference latencies
across six vision architectures: ResNet50 (5.27 ms), YOLOv8n-cut (8.94 ms), YOLOv8s-cut (15.63 ms), YOLOv8m-cut
(26.95 ms), FastDepth (2.87 ms), and SESR-M7 (1.48 ms). Formulates a unified analytical latency equation
`T_model = max(T_compute, T_dram, T_stream) + T_dispatch` and decomposes the residual latency delta into
orthogonal physical mechanisms: (a) compiler VLIW scheduling and 2D sliding-window shuffle overhead (`vshift`/`vmov`
consuming up to 71% of vector slots), (b) DMA synchronization and ObjectFifo ping-pong buffering, and (c) host
driver dispatch floors (~90 µs) plus boundary CPU-side QDQ data conversions (440–450 µs for 640×640, 120–250 µs
for 256×256).

## MemTile 4-D BD im2col dataflow harness and compiler verification

[`notes_im2col_4d_implementation.md`](notes_im2col_4d_implementation.md) — implementation and compiler
verification of a standalone 4-Dimensional Buffer Descriptor (BD) dataflow harness in direct `mlir-aie` dialect
targeting AMD Phoenix AIE2 (XDNA1). Resolves the ObjectFifo token-synchronization deadlock by bypassing
ObjectFifo in favor of raw Buffer Descriptors (`aie.dma_bd`) and decoupled hardware locks (`aie.useLock`).
Configures explicit 4-D striding in MemTile MM2S across $H=8, W=8, C=32$ INT8 feature maps
(Dim 0: 32×1; Dim 1: 3×32; Dim 2: 3×256; Dim 3: 6×32) strictly satisfying hardware bitfield constraints
(wrap $\le 1023$, step $\le 131071$). Verified via `aie-opt` pathfinder flow routing, buffer address assignment,
BD allocation, and `aie-translate --aie-generate-xaie`.

## Vectorized AIE2 C++ compute kernel and VLIW disassembly audit

[`notes_im2col_kernel_vliw_audit.md`](notes_im2col_kernel_vliw_audit.md) — implementation, Peano toolchain
compilation, and static VLIW disassembly audit of the vectorized AIE2 C++ compute kernel for Tile(0, 2) consuming
the 4-D im2col ping-pong buffers against stationary L1 weights ($C_{\text{out}}=32$). Confirms strictly 0 `vshift`
and 0 `vmov` realignment instructions across all bundles of the hardware loops. Measures slot occupancy across
the 6 execution units. In single-patch baseline (M=1), achieves 0.444 vmac/cycle (4 vmac / 9 cycles, 2.000× over
the reference `conv2dk3` baseline of 0.222 vmac/cycle). In dual-patch unrolling (M=2), amortizes stationary weight
loads across Patch A and Patch B, achieving **1.000 vmac/cycle** (8 vmac / 8 cycles, **100.0% physical vector slot
saturation**), delivering an exact **4.500× speedup** over `conv2dk3` and **2.250× speedup** over M=1 with 0 stack
spills (`frame none B, stack refs 0`).

## Tile(0,2) M=2 im2col pipeline integration and CDO transaction synthesis

[`notes_im2col_m2_pipeline_integration.md`](notes_im2col_m2_pipeline_integration.md) — end-to-end integration
and compilation of the M=2 dual-patch vectorized compute engine into `im2col_4d.mlir`, Foreign Function Interface
(FFI) linkage, Peano ELF linking, and NPU CDO/instruction binary synthesis. Expands ping-pong buffers to 576 B each
across dedicated 16 KB L1 physical banks (Bank 2 Ping, Bank 3 Pong, Bank 0 stationary weights, Bank 1 accumulators)
with zero bank conflict. Verifies bare-pointer ABI lowering, generates fully linked standalone AIE2 ELF
(`core_0_2.elf`, 2,436 B, 1,104 B text, 5,504 B L1 data = 8.4% capacity), hardware CDO binaries (`main_aie_cdo_init.bin`,
968 B), and NPU instruction transaction stream (`im2col_4d_m2.bin`, 1,272 B).

## Multi-Core Column 0 im2col scaling and hardware multicast distribution

[`notes_im2col_4core_column_scaling.md`](notes_im2col_4core_column_scaling.md) — scaling the M=2 4-D im2col
dataflow across all four compute tiles in Column 0 (Tiles 0,2 through 0,5). Establishes a 1-to-4 circuit-switched
hardware multicast broadcast tree from MemTile MM2S Channel 0, achieving a 4.00× reduction in MemTile read
bandwidth (1,728 B emitted delivers 6,912 B to compute tiles) with zero interconnect contention. Enforces
identical zero-conflict 4-bank L1 allocations across all four cores (5,504 B per core, 8.4% tile capacity), derives
the spatial height slicing 4-D striding formulas ($H_{\text{out}}=4$ slices), compiles 4 bit-for-bit identical ELFs
(1,104 B text, 0 undefined symbols), and generates complete 4-core CDO packages (`main_aie_cdo_init.bin`, 2,632 B)
and standalone NPU transaction binaries (`im2col_4d_col.bin`, 3,636 B).

## End-to-End Column 0 im2col pipeline: hardware SRS requantization and host egress DMA

[`notes_im2col_egress_roundtrip.md`](notes_im2col_egress_roundtrip.md) - end-to-end closed-loop Column 0 im2col pipeline integration with native hardware Shift-Round-Saturate (SRS) requantization in the compute kernel and autonomous host-to-host bi-directional streaming. Synthesizes `vst.srs.s8.s32` in the vector store unit with zero additional cycles or register shuffles, packs INT32 accumulators to INT8 vectors, routes Core MM2S:0 channels into four MemTile S2MM gathering channels (S2MM:1..4, 1,024 B total), and streams contiguous output back to Host DDR via MemTile MM2S:1 to Shim NoC S2MM:0. Verifies 4-bank collision-free physical memory allocation, compiles 4 clean core ELFs (0 undefined symbols), and synthesizes full NPU transaction (`im2col_4d_roundtrip.bin`, 5,608 B) and CDO packages (`main_aie_cdo_init.bin`, 4,160 B; `main_aie_cdo_enable.bin`, 104 B).

## Physical 5th Column (Column 4) unlock feasibility and routing harness

[`notes_column4_unlock_feasibility.md`](notes_column4_unlock_feasibility.md) — architectural audit, dialect target-model patch derivation, switchbox routing harness (`kernels/aie2/column4_probe.mlir`), and binary transaction synthesis for the 5th physical AIE2 column on AMD Phoenix XDNA1 silicon. Audits upstream MLIR-AIE hardcoded 4-column constraint and specifies the 5-point patch for `npu1_5col`. Evaluates dual-topology routing: Path A (direct Tile(4,0) Shim NoC DMA roundtrip) and Path B (West-to-East cross-column routing between Tile(3,1) MemTile and Tile(4,2) Core) lowering with zero pathfinder errors or assertions. Synthesizes executable NPU transaction binary (`column4_probe.bin`, 2,652 B) and complete CDO package (`main_aie_cdo_init.bin`, 2,080 B) with 85 register writes targeting Column 4. Verifies canonical AIE2 address mapping (`0x08000000` base, `0x14000` locks, `0x1D000` BDs, `0x3F000` switchbox), generates collision-free 4-bank linker script (`core_4_2.ld`, 1,050 B), and proves 144.0 GB/s crossbar interconnect throughput from Column 3 even under unbonded Shim PHY conditions.

## Full 20-core array im2col execution engine synthesis (Columns 0–4, Rows 2–5)

[`notes_im2col_20core_array_synthesis.md`](notes_im2col_20core_array_synthesis.md) — complete 20-core full-array im2col execution engine across all 5 physical columns (Columns 0–4, Rows 2–5) and 30 physical tiles on AMD Phoenix XDNA1 silicon ([`kernels/aie2/im2col_4d_array_20core.mlir`](../../kernels/aie2/im2col_4d_array_20core.mlir), 1,513 lines). Combines dual-patch ($M=2$) vectorized compute kernels, native hardware Shift-Round-Saturate (SRS) requantization, autonomous 4-D DMA multidimensional address generators, and 50 circuit-switched AXI stream flows (10 flows per column) lowering with zero routing conflicts. Peano LLVM-AIE toolchain compiles and links all 20 standalone core ELFs (`build/core_0_2.elf` through `build/core_4_5.elf`) with strictly 0 undefined symbols. Emits complete hardware instruction binary (`build/im2col_4d_20core.bin`, 27,976 B) and CDO package (`build/cdo_20core/`, 20,704 B init CDO) with 705 total MMIO transaction operations (exactly 141 operations per column). Verifies 3.84 MB total on-chip SRAM (2.56 MB L2 across 5 MemTiles + 1.28 MB L1 across 20 Core Tiles) and 5,120 INT8 MACs/cycle (18.43 TOPS at 1.80 GHz).

## AMD Phoenix AIE2 physical silicon hardware bringup, numerical parity validation, and latency profiling

[`notes_im2col_hardware_bringup.md`](notes_im2col_hardware_bringup.md) — physical silicon hardware execution, bit-exact numerical parity validation against NumPy golden references, and latency profiling on AMD Phoenix XDNA1 silicon (Ryzen 7 8700G, NPU [003d:00:01.1], tile clock 1.80 GHz). Verifies pyxrt Embedded Runtime (ERT) ring-buffer command submission, instruction buffer caching, and shared DDR host-device memory synchronization via [`npu/test_im2col_hardware.py`](../../npu/test_im2col_hardware.py). Achieves 100.0% bit-exact parity across single-core (Tile 0,2, 142.14 µs) and 16-core whole-array (Columns 0–3, Rows 2–5, 155.28 µs, 0.0270 Effective TOPS) with zero numerical divergence (MAE 0.0000, RMSE 0.0000) against the native hardware Shift-Round-Saturate (SRS) requantization model (`vst.srs.s8.s32`). Dispatches the full 20-core array transaction stream (`im2col_4d_20core.bin`), confirming hardware ERT timeout on Column 4 NoC addresses and establishing that Columns 0–3 (a 4x6 tile physical grid) represent the complete physically addressable AIE2 execution grid on Phoenix silicon. Execution trace logged in [`hardware_im2col_execution.log`](hardware_im2col_execution.log).

## Asynchronous double-buffered ring-buffer pipelining and driver dispatch floor breakdown

[`notes_im2col_pipelined_dispatch.md`](notes_im2col_pipelined_dispatch.md) — design, implementation, and physical silicon validation of asynchronous double-buffered ring-buffer pipelining on AMD Phoenix XDNA1 silicon (Ryzen 7 8700G, NPU [003d:00:01.1], tile clock 1.80 GHz). Implements concurrent ping-pong DMA staging and non-blocking ERT command ring queueing in [`npu/test_im2col_hardware.py`](../../npu/test_im2col_hardware.py). Benchmarked over 500 consecutive pipelined iterations: hides **67.4–72.4 µs (44.8–48.6%)** of the synchronous XRT driver dispatch floor, scaling Column 0 im2col throughput from 7,008.4 FPS (142.7 µs) to **13,283.1 FPS (75.3 µs, 1.90× speedup)**, single-core MMUL from 6,708.8 FPS (149.1 µs) to **13,043.5 FPS (76.7 µs, 1.94× speedup)**, and 16-core whole-array throughput from 6,546.3 FPS (152.8 µs) to **11,857.4 FPS (84.3 µs, 1.81× speedup, 0.0497 Effective TOPS)**. Proves 100.0% bit-exact parity across both Ping and Pong buffer sets with zero memory races or buffer aliasing. Execution trace logged in [`hardware_im2col_pipelining.log`](hardware_im2col_pipelining.log).

## End-to-end ONNX Conv2D lowering bridge and physical AIE2 silicon execution

[`notes_onnx_layer_silicon_execution.md`](notes_onnx_layer_silicon_execution.md) — automated subgraph ingestion, mathematical scale/shift-cut extraction, stationary vector weight layout packing ($9 \times 4 \times 64 = 2,304\text{ B}$), runtime transaction binary injection (`build/layer_conv0_16core.bin`), and physical silicon execution of a real calibrated vision model layer (`/model.15/m.0/cv1/conv/Conv` from `models/yolov8n_cut_xint8.onnx`) on AMD Phoenix XDNA1 silicon (Ryzen 7 8700G, NPU `[003d:00:01.1]`, tile clock 1.80 GHz). Implements [`npu/lower_onnx_conv.py`](../../npu/lower_onnx_conv.py) with double-buffered asynchronous ring pipelining. Achieves an effective per-frame latency of **$90.77\text{ µs}$** (**$11,017.5\text{ FPS}$**), delivering a **$1.88\times$ speedup** over the synchronous baseline ($170.22\text{ µs}$) and hiding **$79.45\text{ µs}$ ($46.7\%$)** of driver overhead. Resolves the odd-core zero output split across Tiles 0,3 and 0,5, achieving **$100.00\%$ bit-exact numerical parity ($512/512\text{ bytes}$, $\text{MAE}=0.0000$, $\text{MaxAE}=0$)** across all 4 active hardware cores against the exact INT8 QDQ reference, and $100.00\%$ agreement within $\le 1\text{ LSB}$ against ONNX Runtime CPU. Verified via `python -m quant report`. Execution trace logged in [`hardware_onnx_layer_execution.log`](hardware_onnx_layer_execution.log).

[`notes_layer_conv0_parity_resolution.md`](notes_layer_conv0_parity_resolution.md) — root-cause diagnosis of the odd-core zero output split across Core 1 (Tile 0,3) and Core 3 (Tile 0,5) in the Column 0 im2col compute pipeline, formalization of the dynamic transaction binary generator in [`npu/lower_onnx_conv.py`](../../npu/lower_onnx_conv.py) and [`tools/disasm_txn.py`](../../tools/disasm_txn.py), and physical silicon verification of 100.00% bit-exact parity across all 4 active hardware cores on AMD Phoenix XDNA1 silicon (Ryzen 7 8700G, NPU `[003d:00:01.1]`). Execution trace logged in [`hardware_layer_conv0_verification.log`](hardware_layer_conv0_verification.log).

## 16-Core physical array ONNX Conv2D execution and full-array numerical parity

## Split-transaction decoupling: persistent stationary weights and 11,578 FPS pipelined execution

## L2 MemTile activation ping-pong fusion for consecutive Conv2D layers

[`notes_l2_memtile_fusion_architecture.md`](notes_l2_memtile_fusion_architecture.md) — architectural design, MLIR switchbox and interconnect synthesis ([`kernels/aie2/im2col_fused_2layer.mlir`](../../kernels/aie2/im2col_fused_2layer.mlir)), multi-layer transaction binary lowering ([`npu/lower_onnx_conv.py`](../../npu/lower_onnx_conv.py)), and physical silicon verification of consecutive Conv2D layer fusion via MemTile L2 SRAM ping-pong buffers on AMD Phoenix XDNA1 silicon (Ryzen 7 8700G, NPU `[003d:00:01.1]`, tile clock 1.80 GHz). Partitions the 512 KB SRAM of Row 1 MemTiles (`Tile(0..3, 1)`) into double-buffered ping-pong banks (`0x40000` Ping, `0x60000` Pong, 4,096 B per column), arbitrated via hardware synchronization Locks 4 and 5. Replaces intermediate host DDR writebacks with direct on-chip BD chaining and time-multiplexes all 16 compute cores across Layer 0 and Layer 1. Benchmarked over 500 consecutive iterations on Device 0: eliminates 100.0% of intermediate host DDR writeback traffic (0 bytes intermediate bounce), amortizes driver submission tax into a single per-frame dispatch (`build/layer_fused_exec.bin`, 10,496 B), achieves 164.63 µs mean latency (6,074.1 FPS), and delivers 100.00% bit-exact parity (2,048/2,048 bytes, MAE = 0.0000, RMSE = 0.0000) against the exact INT8 QDQ reference and 96.88% agreement (100.0% within $\le 1$ LSB, MAE = 0.0312, RMSE = 0.1768) against floating-point ONNX Runtime CPU. Execution trace logged in [`hardware_fused_layer_verification.log`](hardware_fused_layer_verification.log).

## Generalized N-layer ONNX graph partitioning and dynamic MemTile ping-pong scheduling

[`notes_n_layer_scheduler_architecture.md`](notes_n_layer_scheduler_architecture.md) — architectural specification, dynamic multi-pass transaction bundle lowering (`src/ignite_xdna/compiler/scheduler.py`), automated topological ONNX graph partitioning (`src/ignite_xdna/compiler/partitioner.py`), and physical silicon verification on AMD Phoenix XDNA1 silicon (Ryzen 7 8700G, NPU `[003d:00:01.1]`, tile clock 1.80 GHz). Generalizes 2-layer MemTile fusion into an arbitrary N-layer ping-pong scheduling engine, alternating activations between MemTile `L2_BANK_0` (`0x40000`) and `L2_BANK_1` (`0x60000`) across `Tile(0..3, 1)` with Lock 2, 4, and 5 state machine arbitration. Characterized on physical silicon across 1, 2, 3, and 4-layer Conv2D subgraphs: confirms strictly 0 bytes of intermediate DDR traffic, eliminating host PCIe roundtrips. Sustains 5,939–6,107 FPS (163.75–168.37 µs) on 16 AIE2 cores with host submission overhead paid once for the entire N-layer graph. Achieves 100.00% bit-exact parity against the Layer 0 AIE2 SRS reference and bounded 100.0% agreement within $\le 1$ LSB (MaxAE $\le 1$, 100.00% bit-exact for 4 layers) against ONNX Runtime CPU INT8. Dispatches heterogeneous graphs through `InferenceSession` with automatic CPU fallback orchestration at 11,675 FPS. Execution trace logged in [`hardware_multi_layer_scheduler.log`](hardware_multi_layer_scheduler.log).

## Native graph BO splicing

The [native BO splice measurement](../../docs/BENCHMARKS.md#native-bo-kernel-splicing-2026-09-13-desktop-2)
uses the existing bf16 GroupNorm kernel between two native Conv2D graph stages.

- [First attempt](kernel_splice_conv_gn32_phoenix_20260913T040052Z_27546.log):
  pre-dispatch group-ID rejection and teardown exit 139; no performance finding.
- [Initial passing run](kernel_splice_conv_gn32_phoenix_20260913T040352Z_23328.log):
  full-byte serial parity; fresh command allocation, superseded for timing below.
- [Small tensor](kernel_splice_conv_gn32_phoenix_20260913T040640Z_17573.log):
  L=1024, chunk=256, prepared commands, all-byte parity and API transfer accounting.
- [Large tensor](kernel_splice_conv_gn32_phoenix_20260913T040745Z_27184.log):
  L=301056, chunk=3072, same checks and paired host-copy timing.

## Also here




`aiecompiler_help.log` and `aiecompiler_x86sim_passthrough.log` are on disk but were not
covered by the `results/README.md` entry this file was built from, so nothing is claimed
about them here.




## Fused Conv Residual SiLU

[Method, qualification and limitations](../../docs/BENCHMARKS.md#fused-conv-residual-silu-2026-09-13-desktop-2).
All logs are from Phoenix Desktop 2. Earlier filenames predate the explicit
Cin suffix; their variant is stated below and in the logged build identity.
Failures and the run invalidated by a foreign context are retained.

| Log | Scope / disposition |
|---|---|
| [043714Z_31424](fused_epilogue_phoenix_20260913T043714Z_31424.log) | Rejected: extra trace stream has no legal route in the 16-core design. |
| [043839Z_29368](fused_epilogue_phoenix_20260913T043839Z_29368.log) | Initial polynomial compile; vector spills. Superseded. |
| [044213Z_15444](fused_epilogue_phoenix_20260913T044213Z_15444.log) | Factored polynomial compile, separate one-core trace build. Superseded. |
| [044308Z_7823](fused_epilogue_phoenix_20260913T044308Z_7823.log) | Full-array numerics passed; repeated trace capture failed. Incomplete timing. |
| [044450Z_10245](fused_epilogue_phoenix_20260913T044450Z_10245.log) | Bank-separated resident compile. Superseded. |
| [044608Z_7794](fused_epilogue_phoenix_20260913T044608Z_7794.log) | Foreign NPU context observed: timing invalid; old unpaired polynomial. |
| [044805Z_8945](fused_epilogue_phoenix_20260913T044805Z_8945.log) | Compile failure: dependent-template syntax. |
| [044832Z_1665](fused_epilogue_phoenix_20260913T044832Z_1665.log) | Paired Q14 polynomial compile; spills. Superseded. |
| [045004Z_14588](fused_epilogue_phoenix_20260913T045004Z_14588.log) | Clean resident comparison misses overhead; no DMA overlap. |
| [045122Z_18639](fused_epilogue_phoenix_20260913T045122Z_18639.log) | Q11 paired polynomial compile. Superseded transport. |
| [045853Z_15742](fused_epilogue_phoenix_20260913T045853Z_15742.log) | CDO failure: intermediate output BDs need a release field. |
| [050015Z_31260](fused_epilogue_phoenix_20260913T050015Z_31260.log) | Initial chunked compile; wrong core lock selector, not runnable. |
| [050248Z_6808](fused_epilogue_phoenix_20260913T050248Z_6808.log) | Bounded hardware timeout from wrong output-ready selector. |
| [050338Z_12394](fused_epilogue_phoenix_20260913T050338Z_12394.log) | Corrected core selector 49, chunked output compile. |
| [050521Z_16822](fused_epilogue_phoenix_20260913T050521Z_16822.log) | Clean Cin=32 numerical pass and DMA overlap; overhead fails. |
| [051244Z_8348](fused_epilogue_phoenix_20260913T051244Z_8348.log) | Streaming compile failure: wrong acquire intrinsic spelling. |
| [051401Z_24024](fused_epilogue_phoenix_20260913T051401Z_24024.log) | Streaming placement failure: pong overlapped output. |
| [051454Z_11109](fused_epilogue_phoenix_20260913T051454Z_11109.log) | Four-bank streaming compiled; initial audit rejected scalar ABI pointer saves. |
| [051612Z_23284](fused_epilogue_phoenix_20260913T051612Z_23284.log) | Peano iterative scheduler assertion with loop-internal barriers. |
| [051724Z_12691](fused_epilogue_phoenix_20260913T051724Z_12691.log) | Same scheduler assertion with reduced barrier arrangement. |
| [051822Z_2086](fused_epilogue_phoenix_20260913T051822Z_2086.log) | Full unroll compiled but audit rejected vector spills. |
| [051913Z_13541](fused_epilogue_phoenix_20260913T051913Z_13541.log) | Temporary build with iterative loop scheduling disabled. |
| [051950Z_16177](fused_epilogue_phoenix_20260913T051950Z_16177.log) | Clean Cin=512 pass with slower raw scheduler; superseded performance. |
| [052052Z_21267](fused_epilogue_phoenix_20260913T052052Z_21267.log) | Final optimized Cin=512 build and ELF audit. |
| [052207Z_2656](fused_epilogue_phoenix_20260913T052207Z_2656.log) | Clean optimized Cin=512 pass, three pairs; confirmed by final qualification. |
| [052357Z_20472](fused_epilogue_phoenix_20260913T052357Z_20472.log) | Final resident Cin=32 build and strengthened lock/ELF audit. |
| [052546Z_8609](fused_epilogue_phoenix_20260913T052546Z_8609.log) | Final Cin=32 comparison: numerics and overlap pass, overhead fails. |
| [052611Z_4050](fused_epilogue_phoenix_20260913T052611Z_4050.log) | Qualified Cin=512: ten pairs, full-array numerics, trace, lock/ELF and hash evidence. |
| [Repository gates](fused_epilogue_checks_phoenix_20260913T053000Z.log) | Required test compileall, repository syntax/import and shell checks, link and advisory number audits. |


## SPPF checkpoint

The performance qualification remains open. See the [measurements and limits](../../docs/BENCHMARKS.md#sppf-checkpoint-2026-09-13-desktop-2).

| Log | Outcome |
|---|---|
| [sppf_20x20x256_phoenix_20260913T055728Z_17341.log](sppf_20x20x256_phoenix_20260913T055728Z_17341.log) | Compiler error; superseded diagnostic |
| [sppf_20x20x256_phoenix_20260913T055844Z_17082.log](sppf_20x20x256_phoenix_20260913T055844Z_17082.log) | Peano build passed |
| [sppf_20x20x256_phoenix_20260913T055918Z_31764.log](sppf_20x20x256_phoenix_20260913T055918Z_31764.log) | Hardware timeout; superseded diagnostic |
| [sppf_20x20x256_phoenix_20260913T060039Z_31544.log](sppf_20x20x256_phoenix_20260913T060039Z_31544.log) | Peano build passed |
| [sppf_20x20x256_phoenix_20260913T060108Z_25708.log](sppf_20x20x256_phoenix_20260913T060108Z_25708.log) | Full-output parity; host-inclusive latency assertion failed |
| [sppf_20x20x256_phoenix_20260913T060212Z_22085.log](sppf_20x20x256_phoenix_20260913T060212Z_22085.log) | Full-output parity; host-inclusive latency assertion failed |
| [sppf_20x20x256_phoenix_20260913T060331Z_9269.log](sppf_20x20x256_phoenix_20260913T060331Z_9269.log) | Peano build passed |
| [sppf_20x20x256_phoenix_20260913T060406Z_27214.log](sppf_20x20x256_phoenix_20260913T060406Z_27214.log) | Hardware timeout; superseded diagnostic |
| [sppf_20x20x256_phoenix_20260913T060610Z_29389.log](sppf_20x20x256_phoenix_20260913T060610Z_29389.log) | Compiler error; superseded diagnostic |
| [sppf_20x20x256_phoenix_20260913T060709Z_5346.log](sppf_20x20x256_phoenix_20260913T060709Z_5346.log) | Peano build passed |
| [sppf_20x20x256_phoenix_20260913T060745Z_12004.log](sppf_20x20x256_phoenix_20260913T060745Z_12004.log) | Full-output parity; host-inclusive latency assertion failed |
| [sppf_20x20x256_phoenix_20260913T061157Z_990.log](sppf_20x20x256_phoenix_20260913T061157Z_990.log) | Peano build passed |
| [sppf_20x20x256_phoenix_20260913T061238Z_17350.log](sppf_20x20x256_phoenix_20260913T061238Z_17350.log) | Full-output parity; host-inclusive latency assertion failed |
| [sppf_20x20x256_phoenix_20260913T061627Z_15527.log](sppf_20x20x256_phoenix_20260913T061627Z_15527.log) | Peano build passed |
| [sppf_20x20x256_phoenix_20260913T061700Z_14567.log](sppf_20x20x256_phoenix_20260913T061700Z_14567.log) | Full-output parity; trace cycles only, qualification open |
| [sppf_20x20x256_phoenix_20260913T062646Z_8315.log](sppf_20x20x256_phoenix_20260913T062646Z_8315.log) | Peano build passed; 16 ELFs and 44 native vector maxima |
| [sppf_20x20x256_phoenix_20260913T062850Z_5253.log](sppf_20x20x256_phoenix_20260913T062850Z_5253.log) | Full-output parity; clean 10-sample host-inclusive latency assertion failed |

## DFL softmax and anchor decode

The [DFL decode qualification](../../docs/BENCHMARKS.md#phoenix-dfl-softmax-and-anchor-decode-2026-09-13-desktop-2)
is compile verified but not silicon qualified. The logs record the four-fixture offline
parity check, sixteen Peano ELFs, the vector instruction audit, the initial clean Phoenix
preflight followed by the `0xc01e0009` XRT context-creation failure, and the follow-up
multi-core transport timeouts after a restart.

| Log | Outcome |
|---|---|
| [dfl_decode_phoenix_context_block_20260913T0842Z.log](dfl_decode_phoenix_context_block_20260913T0842Z.log) | Offline parity and Peano compile passed; hardware qualification blocked before dispatch by XRT context creation |
| [dfl_decode_phoenix_transport_checkpoint_20260913T0950Z.log](dfl_decode_phoenix_transport_checkpoint_20260913T0950Z.log) | Restart cleared context creation; full and one-column four-core packet-fanin probes timed out after dispatch; one-core aggregate control completed; full-design parity and latency remain unmeasured |

The [low-level audit](../../docs/LOW_LEVEL_AUDIT.md) of 2026-09-13 dispatched the shipped
container's streams to settle one question its static analysis could not.

| Log | Outcome |
|---|---|
| [lowlevel_audit_silicon_checks_phoenix_20260913T1655Z.log](lowlevel_audit_silicon_checks_phoenix_20260913T1655Z.log) | The 63-layer `init_monolithic.bin` (2,256 parameter writes outside core data memory) and the 7-layer stem init both complete and give byte-identical layer-0 egress; the rebuilt native runtime runs 500 frames at 2,176.68 FPS with zero incomplete dispatches (functional check without contention preflight, not a benchmark figure) and refuses a bit-flipped and a truncated container at load |
| [npu_inference_oracle_free_phoenix_20260913T2050Z.log](npu_inference_oracle_free_phoenix_20260913T2050Z.log) | The shipped container's 4,096 B egress carries none of the 1,209,600 declared detect-head bytes: `predict_sync(use_oracle_for_boxes=False)` now reports `head_source == "none"` and returns no detections instead of decoding zeros; the oracle-free ≥ 4-detection and IoU ≥ 0.70 checks are skipped by the suite for that reason. 100 frames at 1.015 ms mean glass-to-glass (p95 1.076, p99 1.159; NPU dispatch 0.720 ms) with zero buffer objects allocated and +0.02 MB working set — an empty decode, not a detection pipeline. `tools/live_camera_ignition.py --headless` on `assets/bus.jpg` exits 0 with the head status on its HUD |
| [graph_engine_yolov8n_phoenix_20260913T2210Z.log](graph_engine_yolov8n_phoenix_20260913T2210Z.log) | Graph engine bring-up: offline 66/66 layers equal ONNX Runtime and 66/66 packet-emulated EXACT; silicon 3-layer and 66-layer runs EXACT; test_10 passes oracle-free with IoU 1.0; single-dispatch frame 38.5 → 18.3 ms after merging the four per-core fills; hangs from the shim queue depth and from awaiting a streaming weight task, and the stuck-context state after timeouts, are recorded |
| [npu_inference_graph_engine_phoenix_20260914T0233Z.log](npu_inference_graph_engine_phoenix_20260914T0233Z.log) | Witnessed tests/test_npu_inference.py on build/yolov8n_full.ignite (preflight clean, host ~3 % busy): test_10 passes (5 detections, IoU 1.0), test_12 passes, test_11 passes allocation and working-set checks and fails only the ≤ 8 ms line at 19.2 ms mean over 500 frames |
| [graph_engine_latency_phoenix_20260914T0411Z.log](graph_engine_latency_phoenix_20260914T0411Z.log) | Graph engine latency work on Device 0: the 19.1 ms frame split (staging 6.1, dispatch 11.3, readback 1.3 ms), 145 ns per instruction op, ~26 GB/s transport, compute by NOP weights; every schedule, kernel and host step with tasks, instruction bytes and real/NOP dispatch (11.4 → 7.2 ms); rejected variants; the live-camera decode diagnosis |
| [npu_inference_graph_engine_phoenix_20260914T0426Z.log](npu_inference_graph_engine_phoenix_20260914T0426Z.log) | Witnessed tests/test_npu_inference.py on the coarsened container (preflight clean, host ~3 % busy): 10 tests, none skipped, none failed; test_11 500 frames at 7.898 ms mean glass-to-glass (p99 8.270 ms), zero buffer allocations, +0.04 MB working set; test_10 IoU 1.0 |
| [camera_npu_boxes_phoenix_20260914T0426Z.log](camera_npu_boxes_phoenix_20260914T0426Z.log) | Witnessed camera acceptance: live camera index 0, 60 headless frames, NPU boxes without the oracle, 7.929 ms mean glass-to-glass (median 7.869, p95 8.158), NPU 7.474 ms |
| [benchmark_yolov8n_silicon_final.log](benchmark_yolov8n_silicon_final.log) | Live camera run on build/yolov8n_full.ignite: camera index 0 via MSMF, 1000 frames, NPU boxes (six heads declared, 1,209,600 of 1,209,600 egress bytes), 7.898 ms mean glass-to-glass (median 7.873, p95 8.222), NPU 7.443 ms |
| [model_zoo_phoenix_20260914T1247Z.log](model_zoo_phoenix_20260914T1247Z.log) | Model zoo in one sitting (every NPU step idle before): containers for yolov8n, yolov8s and SESR M7 rebuilt from `91d0d7e` (`d828678` in the log) and byte-exact per layer on Device 0 (66/66, 66/66, 9/9); tests — yolov8n 10/10 (7.806 ms, 500 frames), yolov8s 3/3 (17.348 ms, 300 frames, IoU 1.0), SESR 2/3 (pixel-identical to ONNX Runtime, 500 frames, 0 buffer objects; test_22 fails at 4.251 ms dispatch against 1.5 ms); NOP-weight floors 5.369 / 12.727 / 2.530 ms of 7.386 / 16.932 / 4.208 ms; camera tool 500 frames at 17.665 and 6.572 ms; ONNX CPU suite (4 models, rc 0) and NPU suite through Ignition |
| [decode_native_phoenix_20260914T2055Z.log](decode_native_phoenix_20260914T2055Z.log) | Native int8 head decode on Device 0: numpy stage split, live vs replay, crc32 slowdown controls (sleep 1.6–1.9×, spin 1.0×), the np.exp and NMSBoxesBatched parity rules, two design iterations; final C decode with 0 mismatches (6,000 stress trials, 312 recorded, 340 live frames), live decode 0.044–0.045 ms median vs numpy 0.281–0.288 ms, tests/test_npu_inference.py 10 of 10 |
| [npu_inference_native_decode_phoenix_20260914T2104Z.log](npu_inference_native_decode_phoenix_20260914T2104Z.log) | Witnessed tests/test_npu_inference.py on commit e2867b6 with full xrt-smi state before and after (the suite in decode_native_phoenix_20260914T2055Z.log had no preflight output): 10 tests, none skipped; test_10 IoU 1.0; test_11 500 frames at 7.832 ms mean glass-to-glass (p99 8.331 ms), postprocess 0.014 ms, zero buffer allocations, +0.03 MB; preprocess_simd.dll rebuilt through _compile_native_dll passes test_10–12 of the offline suite |
| [model_zoo_main_phoenix_20260914T2209Z.log](model_zoo_main_phoenix_20260914T2209Z.log) | Model zoo on the merge with main (`6bd2718`), one sitting with every NPU step idle before and after: yolov8n, yolov8s and SESR M7 rebuilt with instruction streams and packets byte-identical to the 12:47 builds, 66/66, 66/66 and 9/9 layers exact; yolov8n suites interleaved on the merged build (7.772, 7.755 ms) and on the container compiled before the model zoo, ce64c451 (7.780, 7.712 ms), all passing with native postprocess 0.012–0.013 ms; yolov8s 3/3 (17.278 ms, IoU 1.0, postprocess 0.015 ms); SESR 2/3 (pixel-identical; test_22 fails at 4.336 ms against 1.5 ms); camera tool 7.762 / 17.298 / 6.623 ms; Ignition `3cf2c49` live_ignition.py 7.681 / 17.148 / 6.644 ms; per-step JSON and the container diff in `../model_zoo/main_20260914T2209Z/` |
| [yolo11n_hybrid_phoenix_20260915T0216Z.log](yolo11n_hybrid_phoenix_20260915T0216Z.log) | Stock YOLO11n with its C2PSA block as a segmented container (NPU layers 0–35, the block on ONNX Runtime's CPU provider, NPU layers 37–83), one sitting with every NPU step idle before and after: 84/84 layers exact on Device 0 (host layer included), rebuilt yolov8n 66/66; oracle-free detections identical to the ONNX Runtime CPU decode of the same input, one `InferenceSession.run` per frame for YOLO11n and none for YOLOv8n; timed AMD / Ignition / AMD / Ignition / CPU / YOLOv8n control at 500 frames: 34.492, 10.996, 34.366, 11.048, 31.328 and 7.756 ms mean glass-to-glass |
| [yolo11n_detections_input_rounding_offline.log](yolo11n_detections_input_rounding_offline.log) | Offline: why the NPU arm finds 6 objects on bus.jpg and the AMD and CPU arms 7. Ignition's letterbox and ignite-xdna's native preprocessor differ by one code in 15.18 % of input values; on each input ONNX Runtime (graph optimizations on or off, identical heads) and both decoders agree, 7 objects on the first and 6 on the second |
| [yolo11n_attention_core_phoenix_20260915T1522Z.log](yolo11n_attention_core_phoenix_20260915T1522Z.log) | YOLO11n with only its attention core on the host (`/model.10/m/m.0/attn/qkv/conv/Conv=/model.10/m/m.0/attn/Reshape_1`), C2PSA's seven convolutions on the NPU, one sitting with every NPU step idle before and after: 91/91 layers exact on Device 0, the whole-block container 84/84 and yolov8n 66/66 on the same runtime; oracle-free detections identical to the ONNX Runtime CPU decode for both YOLO11n containers, one `InferenceSession.run` per frame; timed AMD / whole block / attention core / AMD / whole block / attention core / yolov8n control at 500 frames: 36.361, 11.109, 10.411, 34.549, 10.987, 10.115 and 7.722 ms mean glass-to-glass (host step 1.728 / 0.533 / 1.695 / 0.505 ms) |
| [yolo11n_attention_core_offline.log](yolo11n_attention_core_offline.log) | Offline, no NPU: ONNX Runtime per-node profile of the whole-block host model and the attention core against it (sizing on a host not kept quiet: 22.3 % of its time); `v` is qkv channels 64–127 then 192–255 byte for byte; offline gate for the attention-core lowering: 91/91 tensors and six heads equal ONNX Runtime on bus.jpg and 20 coco128 images, emulation 91/91 on three, host core identical with graph optimizations on (21/21), YOLOv8n, YOLOv8s and SESR M7 stream reports unchanged |
| [yolov8n_pose_phoenix_20260915T1919Z.log](yolov8n_pose_phoenix_20260915T1919Z.log) | YOLOv8n-pose as a graph-engine container (75 layers, nine heads), one sitting with every NPU step idle before and after: 75/75 layers exact on Device 0, and on the same runtime yolov8n 66/66 and the YOLO11n attention-core container 91/91; `PoseOnSilicon` 2/2 (nine heads equal ONNX Runtime CPU's and people identical to its numpy tail on bus.jpg, no `InferenceSession.run` call; 500 native-ingress frames at 8.290 ms, 3 people each, no buffer objects allocated, +0.30 MB); timed AMD / container / Ignition app, twice, at 500 frames: 12.056, 8.318, 8.360, 12.089, 8.339 and 8.481 ms mean from frame to people; controls yolov8n 7.765 ms and the YOLO11n attention core 11.990 ms (median 10.259, a disturbed tail) |
| [yolov8s_sesr_vs_amd_phoenix_20260915T2146Z.log](yolov8s_sesr_vs_amd_phoenix_20260915T2146Z.log) | YOLOv8s and SESR M7 containers against AMD's stack, one sitting with every NPU step idle before: both rebuilt at `9351cc1` by the toolchain Ignition's `install.ps1` installs (25.8 s and 6.9 s), 66/66 and 9/9 layers exact on Device 0; timed AMD / Ignition / AMD / Ignition at 500 frames with Ignition's own pre- and post-processing on AMD's arm: YOLOv8s 16.954, 17.240, 16.958 and 17.265 ms, SESR M7 3.654, 6.671, 3.632 and 6.662 ms mean glass-to-glass (AMD `session.run` 13.137–13.167 and 1.461–1.473 ms against dispatch 16.704–16.743 and 4.392–4.405 ms); Ignition RSS 242.0 and 162.1 MB against AMD's 339.2 and 265.0 MB in its second session; control yolov8n 7.771 ms |
| [yolov8n_pose_offline.log](yolov8n_pose_offline.log) | Offline, no NPU: the YOLOv8n-pose lowering gate (75/75 tensors and nine heads equal ONNX Runtime on bus.jpg and 20 coco128 images, emulation 75/75 on three, DERIVED traffic against yolov8n) and the COCO val2017 detection files of the container's numpy-ingress run and ONNX Runtime 1.30's CPU run compared: byte-identical, 138,848 people over 4,996 images |
| [weight_buffer_phoenix_20260916T1508Z.log](weight_buffer_phoenix_20260916T1508Z.log) | The resident MemTile weight buffer against flag-off, one sitting, interleaved at 100 timed dispatches per run, every run 66/66 byte-exact: yolov8n 7.314 / 7.331 ms off against 8.492 / 8.474 ms with the buffer, yolov8s 16.752 / 16.868 against 20.436 / 20.443 ms, slower by 1.160 and 3.630 ms where the costing predicted 0.077 and 0.729 ms faster; the one-layer container byte-exact first. [BENCHMARKS](../../docs/BENCHMARKS.md#memtile-residency-does-not-pay-on-the-graph-engine-and-the-yolov8s-gap-is-a-known-limitation-2026-09-16-desktop-2) |
| [memtile_hop_phoenix_20260916T1523Z.log](memtile_hop_phoenix_20260916T1523Z.log) | A MemTile store-and-forward hop in isolation (`tools/memtile_hop_probe.py`; raw times in [the JSON](memtile_hop_phoenix_20260916T1523Z.json)): one core, 6,400 B in and 3,200 B out, four routes, 3.3–52.4 MB in, three interleaved rounds; slopes 0.14301 / 0.14278 / 0.14327 / 0.14280 ms per MB in, so the hop costs -0.00023 ms per MB in and +0.00053 per MB out. Retracts the hop-tax inference in `notes_yolov8s_gap.md` |

## MemTile residency on the graph engine

[`notes_memtile_activation_ring.md`](notes_memtile_activation_ring.md) — the opt-in MemTile activation ring from
design to silicon: the arm-once hang, the parked-head lock credit, the width-change fault located with
`tools/ring_shape_probe.py`, the per-layer channel reset that made the whole model run 66/66 byte-exact, and the
measured result that it is slower (yolov8n 10.052 and, configuring only on a shape change, 9.758 ms against
7.392 ms; yolov8s 20.228 against 16.828 ms). Its latencies were recorded in the note; no separate raw log was
committed.

[`notes_yolov8s_gap.md`](notes_yolov8s_gap.md) — where yolov8s's frame goes and every lever sized against AMD's
0.29 ms lead: over-read (whole junk blocks and intra-plane slack), fill-task saturation, weight re-send, MemTile
hardware compression, cascade halo exchange, and the resident weight buffer built, debugged against the CDO and
measured. Read the retractions above before quoting it. Full treatment in
[BENCHMARKS](../../docs/BENCHMARKS.md#memtile-residency-does-not-pay-on-the-graph-engine-and-the-yolov8s-gap-is-a-known-limitation-2026-09-16-desktop-2).

## Energy per frame against AMD's stack, and power modes

| Log | Outcome |
|---|---|
| [energy_openmp_wait_policy_verify_phoenix_20260916T1640Z.log](energy_openmp_wait_policy_verify_phoenix_20260916T1640Z.log) | Why the native preprocessor's OpenMP workers spun: YOLOv8n through Ignition with `OMP_WAIT_POLICY=PASSIVE` set by `os.environ` as the first statement of the process (a scratch launcher that then called Ignition's `main()` on unchanged engine code) still ran at 100.0 % CPU and 372.3 mJ per frame above idle; with the policy also written through `ucrtbase._putenv_s` before the DLL loads (`pipelines/power.py`, loaded through `PYTHONPATH`), 14.0 % CPU and 134.5 mJ at 118.83 fps. Short windows, one run each ([JSON](energy_openmp_wait_policy_verify_phoenix_20260916T1640Z.json)) |
| [energy_power_modes_yolov8n_phoenix_20260916T1645Z.log](energy_power_modes_yolov8n_phoenix_20260916T1645Z.log) | One interleaved sitting, YOLOv8n on bus.jpg at full speed, package power above the median of 8 undisturbed 30 s idle baselines: AMD's stack 89.08 / 95.77 fps and 126.6 / 123.5 mJ per frame; Ignition `performance` 125.81 / 126.25 fps and 377.7 / 381.6 mJ; `balanced` 119.06 / 119.11 fps and 128.7 / 124.5 mJ; `efficiency` 109.56 / 108.53 fps and 114.7 / 120.7 mJ ([JSON](energy_power_modes_yolov8n_phoenix_20260916T1645Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-against-amds-stack-and-power-modes-2026-09-16-desktop-2) |
| [energy_power_modes_paced30_yolov8n_phoenix_20260916T1702Z.log](energy_power_modes_paced30_yolov8n_phoenix_20260916T1702Z.log) | The same comparison paced to a camera's 30 frames per second (`--max-fps 30` on both stacks, 1,200 frames per run, median of 8 undisturbed 30 s idle baselines): AMD's stack 217.6 / 200.4 mJ per frame at 11.04 / 10.97 ms G2G; Ignition `performance` 1,384.8 / 1,371.8 mJ; `balanced` 196.9 / 178.7 mJ at 8.74 / 8.80 ms; `efficiency` 170.3 / 159.1 mJ at 9.53 / 9.58 ms ([JSON](energy_power_modes_paced30_yolov8n_phoenix_20260916T1702Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-against-amds-stack-and-power-modes-2026-09-16-desktop-2) |
| [latency_balanced_default_phoenix_20260916T1745Z.log](latency_balanced_default_phoenix_20260916T1745Z.log) | Every same-sitting AMD comparison repeated with Ignition in its `balanced` default: six containers re-verified exact, then 50 warm-up and 500 timed frames per run, interleaved, NPU idle before every group. YOLOv8n 8.420 / 8.386 ms (`performance` 7.982 / 7.960) against AMD 10.392 / 10.335; YOLOv8s 18.046 / 17.988 against 16.744 / 16.564; SESR M7 6.840 / 6.815 against 4.365 / 4.332; YOLO11n attention core 10.439 / 10.369 and whole block 11.182 / 11.177 against 36.540 / 36.644; YOLOv8n-pose 9.025 / 8.970 through Ignition and 8.987 / 9.039 through `4_pose.py` against 11.955 / 12.020 — [BENCHMARKS](../../docs/BENCHMARKS.md#the-amd-comparisons-re-measured-in-the-balanced-default-2026-09-16-desktop-2) |
| [energy_npu_pmode_yolov8n_phoenix_20260916T1931Z.log](energy_npu_pmode_yolov8n_phoenix_20260916T1931Z.log) | YOLOv8n energy per frame and G2G under the NPU's device-wide power modes (`xrt-smi configure --pmode` default / balanced / powersaver, set and read back per arm, `default` restored), AMD's stack against Ignition's `efficiency` mode, at 30 fps and full speed, median of 24 undisturbed idle baselines 34.902 W: at 30 fps Ignition 145.1 / 161.1 mJ in `default` and 133.0 / 130.6 in `powersaver` at 9.52 against 18.12 ms G2G; at full speed 108.8 / 111.2 against 112.0 / 109.1 mJ at 109.3 against 56.1 fps ([JSON](energy_npu_pmode_yolov8n_phoenix_20260916T1931Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#the-npus-own-power-modes-buy-energy-only-at-a-cameras-rate-and-cost-ignition-its-latency-lead-2026-09-16-desktop-2) |
| [energy_power_modes_models_phoenix_20260916T1758Z.log](energy_power_modes_models_phoenix_20260916T1758Z.log) | **Discarded.** Energy per frame on YOLOv8s, SESR M7, YOLO11n and YOLOv8n-pose at full speed, taken while a load that showed no CPU held idle about 5.5 W high (baselines 39.9-41.8 W); controls repeated on clean power read 13-25 % lower, so no figure from it is quoted ([JSON](energy_power_modes_models_phoenix_20260916T1758Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2) |
| [energy_power_modes_paced30_models_phoenix_20260916T1835Z.log](energy_power_modes_paced30_models_phoenix_20260916T1835Z.log) | **Discarded.** The same at 30 fps; the offset vanished before its last seven arms, which is what put the median-idle column below zero for pose ([JSON](energy_power_modes_paced30_models_phoenix_20260916T1835Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2) |
| [energy_rerun_pose_paced30_controls_phoenix_20260916T1913Z.log](energy_rerun_pose_paced30_controls_phoenix_20260916T1913Z.log) | Clean (median idle 34.882 W): YOLOv8n-pose at 30 fps (AMD 262.6 / 249.2, `balanced` 188.3 / 178.9, `efficiency` 178.1 / 165.1 mJ) and the controls that proved the discarded sittings biased: YOLOv8s at 30 fps and SESR M7 at full speed, AMD and `balanced` ([JSON](energy_rerun_pose_paced30_controls_phoenix_20260916T1913Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2) |
| [latency_yolo11n_host_power_modes_phoenix_20260916T2010Z.log](latency_yolo11n_host_power_modes_phoenix_20260916T2010Z.log) | YOLO11n after `fbd53f5` (host segments follow the power mode), 50 warm-up + 500 frames, interleaved: AMD 37.665 / 37.702 ms; `yolo11n.ignite` `balanced` 11.588 / 11.557 (host 1.983 / 1.950); `yolo11n_core.ignite` `balanced` 10.674 / 10.627 (host 0.713 / 0.733) and `performance` 10.563 / 10.456 (host 0.549 / 0.572) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2) |
| [energy_power_modes_models_phoenix_20260916T2012Z.log](energy_power_modes_models_phoenix_20260916T2012Z.log) | Energy per frame at full speed, AMD / `performance` / `balanced` / `efficiency`, twice, median idle 35.014 W, none shifted: YOLOv8s AMD 207.3 / 207.4 against `balanced` 248.3 / 246.8 mJ; SESR M7 56.3 / 55.4 against 74.4 / 72.6; YOLO11n 1,387.2 / 1,405.8 against 166.3 / 164.9; YOLOv8n-pose 163.4 / 154.9 against 129.4 / 129.1 ([JSON](energy_power_modes_models_phoenix_20260916T2012Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2) |
| [energy_power_modes_paced30_models_phoenix_20260916T2049Z.log](energy_power_modes_paced30_models_phoenix_20260916T2049Z.log) | The same at 30 fps, AMD / `balanced` / `efficiency`, median idle 34.940 W, none shifted: YOLOv8s 278.0 / 264.6 against 290.3 / 287.5 and 270.0 / 275.6 mJ; SESR M7 133.7 / 137.7 against 148.3 / 130.5 and 135.7 / 134.4; YOLO11n 1,392.2 / 1,364.9 (at 26.6-26.9 fps) against 231.7 / 224.8 and 213.1 / 213.9; YOLOv8n-pose 241.6 / 253.0 against 180.1 / 182.1 and 173.5 / 183.8 ([JSON](energy_power_modes_paced30_models_phoenix_20260916T2049Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2) |
| [npu_power_governor_silicon_phoenix_20260916T2217Z.log](npu_power_governor_silicon_phoenix_20260916T2217Z.log) | The NPU power-mode governor's behaviour on the NPU, device mode and contexts read back mid-run and after exit, 0 failures: YOLOv8n and YOLO11n `efficiency` @30 lowered to Powersaver (G2G 18.181 and 22.471 ms) and restored; YOLOv8s @30, `balanced`, unpaced and `--npu-power off` kept Default; Ctrl+Break restored; a second NPU process made the watcher restore; a hard kill left Powersaver and `ignition devices --restore-npu-power` restored it — [BENCHMARKS](../../docs/BENCHMARKS.md#the-npu-power-mode-governor-less-energy-per-frame-at-30-fps-and-the-device-always-put-back-2026-09-16-desktop-2) |
| [energy_npu_power_governor_phoenix_20260916T2221Z.log](energy_npu_power_governor_phoenix_20260916T2221Z.log) | Energy per frame at 30 fps with the governor, median idle 35.110 W: YOLOv8n AMD 215.7 / 183.4, `efficiency` off 166.2 / 177.1, `auto` 141.6 / 132.4, `auto` without the watcher 124.3 / 125.6 mJ; YOLO11n AMD 1,358.0 / 1,356.9, off 200.1 / 213.0, `auto` 154.0 / 148.2 mJ ([JSON](energy_npu_power_governor_phoenix_20260916T2221Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#the-npu-power-mode-governor-less-energy-per-frame-at-30-fps-and-the-device-always-put-back-2026-09-16-desktop-2) |
| [energy_npu_power_watcher_phoenix_20260916T2240Z.log](energy_npu_power_watcher_phoenix_20260916T2240Z.log) | The governor's watcher priced, YOLOv8n `efficiency` @30, median idle 34.766 W: watcher every 5 s 122.3 / 121.5, every 30 s 127.1 / 131.8, none 120.4 / 125.9, `--npu-power off` 167.8 / 170.9 mJ; not separated, so 5 s stays ([JSON](energy_npu_power_watcher_phoenix_20260916T2240Z.json)) — [BENCHMARKS](../../docs/BENCHMARKS.md#the-npu-power-mode-governor-less-energy-per-frame-at-30-fps-and-the-device-always-put-back-2026-09-16-desktop-2) |
| [yolow_int8_collapse/](yolow_int8_collapse/) | YOLO-World v2's XINT8 collapse localised on the CPU (COCO val2017, first 300 or 500 images): FP32 41.5 %, HardSigmoid SiLU 30.5 %, plain XINT8 2.1 %, attention or projection convs in FP32 1.9-2.1 %; float activations with int8 weights 0.2 %; per-layer weight rounding sensitivity (`yolow_layer_sensitivity.log`: `/model.15/cv2` alone drops a head to -12.9 dB) and Concat input ranges; the four C2fAttn cv2 convs in float 29.5 % (weights only) and in FP32 under Quark XINT8 24.5 % on the CPU and on AMD's stack (110 NPU / 939 CPU nodes, 96.34 ms per image); eval, quantize and diagnostic logs with their commands — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-on-the-graph-engine-the-text-attention-lowers-and-xint8s-collapse-is-four-convolutions-2026-09-16-desktop-2) |
| [yolow_engine/](yolow_engine/) | YOLO-World v2 variant D as a graph-engine container (host regions over the text attention and over the C2fAttn Concat views): build (33,082,368 B, 70 layers, 8 host), NPU verify 70/70 exact (dispatch 17.197 ms, host 15.687 ms), COCO first 300 images through the container 24.5 % mAP identical to ONNX Runtime CPU, frame profile (53.074 ms with numpy ingress and dequantization) — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-on-the-graph-engine-the-text-attention-lowers-and-xint8s-collapse-is-four-convolutions-2026-09-16-desktop-2) |
| [engine_residual_hswish/](engine_residual_hswish/) | The core program's residual op applying HardSwish after the add (`ca5b6cd`): kernel compile and a synthetic sequence bit-exact on the NPU; YOLOv8n (instruction stream and packets byte-identical, new xclbin) and YOLOv8s rebuilt and 66/66 exact; YOLOv8n old/new program interleaved on a quiet host, 7.331/7.299 against 7.216/7.245 ms (no slowdown); the split YOLO-World v2 container built (74 layers, 4 host), exact offline and 74/74 on the NPU — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-with-only-its-text-attention-on-the-cpu-gptq-and-an-int32-bias-recover-the-four-output-convolutions-2026-09-16-desktop-2) |
| [yolow_gptq/](yolow_gptq/) | YOLO-World v2's four C2fAttn output convolutions on the NPU (COCO val2017, first 300 images): the exact split under Quark XINT8 2.3 % (2.7 % with the halves added in float) and why (the halves cancel; per-half scales buy 0 dB; the int8 bias caps the output); GPTQ int8 weights with an int32 bias 24.7 % on the CPU and through the container, 72,972 identical detections; exact offline and 70/70 on the NPU; profile sitting against variant D, 48.227/48.024 against 53.035/53.485 ms — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-with-only-its-text-attention-on-the-cpu-gptq-and-an-int32-bias-recover-the-four-output-convolutions-2026-09-16-desktop-2) |
| [yolow_vocabulary/](yolow_vocabulary/) | YOLO-World v2's vocabulary chosen at run time on the GPTQ container: swapped text guides equal `set_classes` in FP32 (108.4-112.5 dB); 70/70 exact on the NPU with five class names; the torch-free text path (tokenizer 97/97, encoder 7.3e-07); detections on `bus.jpg` for two non-COCO vocabularies (container = ONNX Runtime); COCO first 300 images with 23 categories renamed to synonyms, FP32 40.8 -> 36.4 and XINT8 26.4 -> 20.6 on those categories, the container identical to ONNX Runtime in the same environment; `YoloWorldPipeline` on `bus.jpg` (set_classes 90.1 ms, 39.770 ms glass-to-glass, not a sitting) — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2s-vocabulary-chosen-at-run-time-one-container-any-class-names-2026-09-16-desktop-2) |
| [yolow_sitting_vs_amd/](yolow_sitting_vs_amd/) | YOLO-World v2 in one sitting (first 300 COCO val2017 images per run, `5_eval_map.py` per-image inference, interleaved): the GPTQ container 47.33 / 47.51 ms at 24.7 %, AMD's stack on variant D 95.91 / 95.89 ms at 24.5 %, ONNX Runtime CPU on FP32 67.75 ms and DirectML on the iGPU 46.22 ms, both at 43.0 % — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-against-amds-stack-the-cpu-and-the-igpu-in-one-sitting-2026-09-16-desktop-2) |
| [yolow_g2g_sitting/](yolow_g2g_sitting/) | YOLO-World v2 glass-to-glass in one sitting (`pipelines/yolow/4b_g2g.py`, `bus.jpg`, 500 frames per run, interleaved): the container through `YoloWorldPipeline` 39.912 / 39.853 ms, AMD's stack on variant D 109.729 / 109.376 ms, the iGPU on FP32 52.415 ms, the CPU on FP32 81.593 ms; with five names the container 32.595 and the iGPU 28.444 ms; plus the contrastive-product orientation benchmark behind the bit-exact int8 decode — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-glass-to-glass-275-times-amds-stack-and-against-the-igpu-it-depends-on-the-vocabulary-2026-09-16-desktop-2) |
| [yolow_energy/](yolow_energy/) | YOLO-World v2 energy per frame (`tools/energy_sitting.py` over `pipelines/yolow/4b_g2g.py`, `bus.jpg`, 80 classes, own-idle deltas): flat out the container 1040.36 / 908.91 mJ, AMD's stack on variant D 4483.41 / 4544.51, DirectML FP32 1201.12 / 1336.00, CPU FP32 3610.19 / 3580.88; at 5 fps 1428.13, 5337.04, 842.48 and 4214.34 mJ; idle baselines span 2.1 W (flagged, own idle used) — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-energy-per-frame-43-50-times-less-than-amds-stack-and-the-igpu-spends-less-at-5-fps-2026-09-17-desktop-2) |
| [yolow_energy/energy_yolow_pmode_phoenix_20260917T0428Z.log](yolow_energy/energy_yolow_pmode_phoenix_20260917T0428Z.log) | YOLO-World v2 at 5 fps with the NPU's power-saving mode (NPU mode set and read back per arm, restored at the end): the container in `efficiency` + `powersaver` 1086.27 / 1060.09 mJ per frame at 69.1-69.3 ms, against 1364.78 / 1489.51 mJ at 40.5 ms in its defaults; DirectML FP32 651.69 / 684.28 mJ at 79.6 ms; AMD's stack in `powersaver` 5416.88 / 5419.82 mJ; glass-to-glass records in `pmode_g2g_20260917T0428Z/` — [BENCHMARKS](../../docs/BENCHMARKS.md#yolo-world-v2-at-5-fps-with-the-npus-power-saving-mode-less-energy-for-17-times-the-frame-time-and-the-igpu-still-spends-less-2026-09-17-desktop-2) |
| [silu_epilogue/](silu_epilogue/) | What SiLU's HardSigmoid form costs YOLOv8n, YOLOv8s and YOLOv8n-pose, CPU only, first 500 COCO val2017 images: FP32 39.95 / 48.52 / 49.49, FP32 with the form 33.66 / 43.33 / 36.07, XINT8 as shipped 30.25 / 40.77 / 31.56; the shipped XINT8 models with an integer four-line sigmoid 37.29 / 46.25 / 43.50 and three lines 37.05 / 45.66 / 42.99, beside an exact SiLU table, a sigmoid at 1/128 and Quark with Sigmoid kept (`summary.log`; YOLOv8s's keep-Sigmoid quantization ran out of memory, `quantize_yolov8s_xint8_keep_sigmoid_oom.log`; on 100 calibration images it ran, 47.26, `*_calib100*.log`); the oracle's HardSigmoid mode bit-exact with the shipped model; every oracle and FP32 form model rebuilt byte-identical from `tools/`; per-layer output-LSB error tables; the core-program variants sized offline (8,960 B today; the sigmoid in the pass spills accumulators, in a separate loop 9,792 B and none) — [BENCHMARKS](../../docs/BENCHMARKS.md#silus-hardsigmoid-form-is-most-of-the-model-zoos-xint8-accuracy-loss-and-an-integer-four-line-sigmoid-wins-55-119-points-back-offline-2026-09-17-desktop-2) |
| [silu_sigmoid_engine/](silu_sigmoid_engine/) | The sigmoid SiLU epilogue built into the core program (`45a8685`, `7700316`, `--silu-sigmoid`) and checked on the NPU for YOLOv8n, YOLOv8s and YOLOv8n-pose: offline gates (every layer equals ONNX Runtime on `silu_sigmoid.reference_model`, every layer emulates exactly); the synthetic sequence bit-exact on the NPU for three kernel versions; census (9,824 B, no accumulator stack moves); containers from main `c714629` and both versions with `insts.bin` identical and option-off `wpackets.bin` equal to today's; every layer exact on the NPU; first 500 COCO val2017 images through the containers 37.29 / 46.25 / 43.50 with detections byte-identical to the reference model on ONNX Runtime CPU; sittings: today 7.297 / 16.799 / 7.587 ms, `7700316` without the flag 7.240 / 16.799 / 7.553, with it 7.488 / 17.201 / 7.850 (dispatch means); the rejected v2 layout and its sitting; `container_names.log` maps container names to kernels — [BENCHMARKS](../../docs/BENCHMARKS.md#the-sigmoid-silu-epilogue-on-the-npu-an-opt-in-exact-through-the-containers-for-24-39--more-dispatch-time-2026-09-17-desktop-2) |
| [silu_sigmoid_vs_amd/](silu_sigmoid_vs_amd/) | The `--silu-sigmoid` containers against AMD's stack (Vitis AI EP on the shipped models) and today's containers on Desktop 2. `accuracy/`: all 5,000 COCO val2017 images, 26.68 / 37.31 / 32.64 for AMD's stack (NPU placement in `diag_*.log`: 922/929, 922/929, 1,015/1,025 nodes), 27.10 / 37.21 / 32.71 for today's containers, 34.12 / 42.37 / 44.16 for the sigmoid containers, each re-scored on the first 500 images. `g2g/`: 50 warm-up + 500 frames of bus.jpg, interleaved twice; AMD 10.740 / 10.742, 16.966 / 17.005, 12.350 / 12.330 ms against sigmoid 8.672 / 8.774, 18.430 / 18.424, 9.362 / 9.286 ms. `dispatch/`: the flag re-timed at 0.202 / 0.376 / 0.202 ms. `energy/`: 30 fps, 1,200 frames, median idle 38.661 W (one core busy with a desktop process throughout) — [BENCHMARKS](../../docs/BENCHMARKS.md#the-sigmoid-silu-containers-against-amds-stack-more-accurate-on-all-5000-coco-images-faster-on-yolov8n-and-yolov8n-pose-slower-on-yolov8s-2026-09-17-desktop-2) |

## Host fast paths

| Log | Outcome |
|---|---|
| [latency_balanced_dispatch_opt_phoenix_20260917T1250Z.log](latency_balanced_dispatch_opt_phoenix_20260917T1250Z.log) | Same-sitting comparison with AMD's stack in the `balanced` default on the runtime with channel-block head decode, merged head syncs and SESR M7's native resize and DepthToSpace (today's HardSigmoid containers; 50 warm-up and 500 frames of bus.jpg per run, interleaved, each model twice): SESR M7 4.783 / 4.766 against AMD 4.543 / 4.576 ms (6.840 / 6.815 the day before); YOLOv8s 17.950 / 17.940 against 16.966 / 16.916 ms (readback 0.312 / 0.310); YOLOv8n 8.309 / 8.349 against 10.637 / 10.530 ms; YOLOv8n-pose 8.981 / 8.952 against 12.137 / 12.220 ms — [BENCHMARKS](../../docs/BENCHMARKS.md#host-fast-paths-sesr-m7-205-ms-faster-in-its-host-stages-detection-heads-decoded-in-their-channel-blocks-2026-09-17-desktop-2) |
| [host_fastpaths_verification/](host_fastpaths_verification/) | Exactness of those paths: offline tests with the committed and rebuilt DLLs; decode stress 3,000 trials, 0 mismatches; on the NPU, heads through the merged syncs and channel-block views equal `read_tensor`, and the channel-block decode equals the NCHW decode on 101 frames each for YOLOv8n (two containers) and YOLOv8s, `predict_sync` oracle-free and with its oracle agreeing; the `--silu-sigmoid` containers on the first 500 COCO images 37.29 / 46.25 / 43.50, byte-identical to the reference model; SESR's native DepthToSpace exact, its native resize within one code of OpenCV (13.240 % of bytes differ) and its whole image 41.42-42.19 dB PSNR from the OpenCV path — [BENCHMARKS](../../docs/BENCHMARKS.md#host-fast-paths-sesr-m7-205-ms-faster-in-its-host-stages-detection-heads-decoded-in-their-channel-blocks-2026-09-17-desktop-2) |
| [release_033/](release_033/) | Ignition v0.3.3 release sitting on main `1879614` (the host fast paths): the `--silu-sigmoid` YOLOv8n, YOLOv8s and YOLOv8n-pose containers built through `ignite-compile` with same-commit controls (`containers_1879614.log`), every container layer-exact on the NPU, then 50 warm-up and 500 frames of bus.jpg per run, interleaved twice: YOLOv8n 8.500 / 8.532 against AMD 10.658 / 10.424 ms (control 8.262 / 8.286), YOLOv8s 18.213 / 18.183 against 16.786 / 16.754 (control 17.870 / 17.797), SESR M7 4.775 / 4.781 against 4.353 / 4.349, YOLO11n attention core 10.458 / 10.497 against 36.771 / 38.303, YOLOv8n-pose 9.282 / 9.349 against 12.067 / 11.971 (control 8.986 / 8.996); a 30 fps energy sitting discarded (idle 36.7-40.6 W, arms not repeating) — [BENCHMARKS](../../docs/BENCHMARKS.md#the-v033-release-sitting-sigmoid-silu-containers-on-the-host-fast-paths-against-amds-stack-2026-09-17-desktop-2) |
| [latency_balanced_ingress_opt_phoenix_20260917T1830Z.log](latency_balanced_ingress_opt_phoenix_20260917T1830Z.log) | Same-sitting comparison with AMD's stack in the `balanced` default on the runtime with AVX2 vectorized ingress staging & Q11 spatial interpolation (50 warm-up and 500 frames of bus.jpg per run, interleaved, each model twice): SESR M7 4.725 / 4.719 against AMD 4.435 / 4.405 ms (preprocess 0.137 / 0.135 ms vs AMD 0.422 / 0.416 ms); YOLOv8s 18.098 / 18.084 against 17.097 / 17.123 ms (preprocess 0.398 / 0.396 ms vs AMD 1.947 / 1.898 ms); YOLOv8n 8.353 / 8.414 against 10.861 / 10.890 ms (preprocess 0.365 / 0.364 ms vs AMD 1.911 ms); YOLOv8n-pose 9.163 / 9.154 against 12.579 / 12.508 ms (preprocess 0.330 / 0.329 ms vs AMD 3.314 / 3.283 ms) — [BENCHMARKS](../../docs/BENCHMARKS.md#avx2-vectorized-ingress-activation-staging-and-q11-spatial-interpolation-2026-09-17-desktop-2) |
| [workspace_reuse_retire_batch_opt_phoenix_20260917T1900Z.log](workspace_reuse_retire_batch_opt_phoenix_20260917T1900Z.log) | Liveness-based workspace buffer reuse and DMA retirement relaxation (`retire_batch = 4`): workspace memory reduced by 35% to 43% across the model zoo (SESR M7 19.71 -> 11.19 MB, YOLOv8s 32.24 -> 19.73 MB, YOLOv8n 22.04 -> 14.14 MB, YOLOv8n-pose 22.60 -> 14.55 MB); TCT await ops reduced by 30% to 46% (YOLOv8s 3,681 -> 2,025 awaits, YOLOv8n 1,671 -> 1,110 awaits); YOLOv8n insts.bin shrinks from 430 KB to 405 KB; physical Phoenix NPU Device 0 achieves 7.412 ms on YOLOv8n (min 7.299 ms) and 17.163 ms on YOLOv8s with 100% bit-exact parity across all heads — [BENCHMARKS](../../docs/BENCHMARKS.md#liveness-based-workspace-buffer-reuse-and-dma-retirement-relaxation-2026-09-17-desktop-2) |
| [classification_head_vs_amd_phoenix_20260917T2350Z.log](classification_head_vs_amd_phoenix_20260917T2350Z.log) | Terminal classification head on silicon: the engine's 1x1-conv lowering (`bd87558`, 0 host segments) runs the 2,048 -> 1,000 ImageNet head at 1.378 / 1.386 ms (NPU dispatch 1.142 / 1.153 ms) where AMD Ryzen AI 1.7.1's Vitis AI EP crashes at placement (`runner_requests_queue.cpp:178: Failed to create runner: invalid vector subscript`, 0 NPU nodes) and its CPU-fallback control is faster at 0.666 / 0.707 ms. The input is a constant zero-point vector, so the log prices latency and placement, not accuracy — the `max_diff = 0` claim is `bd87558`'s verification in docs/DECISIONS.md, not a row of this run — [BENCHMARKS](../../docs/BENCHMARKS.md#terminal-classification-head-as-one-1x1-conv-138-ms-on-silicon-where-the-vitis-ai-ep-crashes-at-placement-2026-09-17-desktop-2) |




## Program-RAM epilogue opcodes (2026-09-18, Desktop 2)

| Log | Outcome |
|---|---|
| [program_ram_epilogue_census_phoenix_20260918T040616Z.log](program_ram_epilogue_census_phoenix_20260918T040616Z.log) | Per-opcode .text census of the persistent conv engine (tools/engine_opcode_census.py, Peano object-only): baseline 13,424 B; OP_MUL 13,648 (+224), OP_SCALE 13,888 (+464), OP_POOL 14,304 (+880), all three 15,280 B (+1,856) with accumulator stack moves 0 and 1,104 B still free of 16,384 - [BENCHMARKS](../../docs/BENCHMARKS.md#three-epilogue-opcodes-under-the-program-ram-budget-elementwise-mul-per-channel-scale-and-5x5-average-pool-bit-exact-on-silicon-2026-09-18-desktop-2) |
| [program_ram_epilogue_census_pool_no_pragma_phoenix_20260918T041050Z.log](program_ram_epilogue_census_pool_no_pragma_phoenix_20260918T041050Z.log) | What the unroll decision costs: OP_POOL dispatched alone with the unroll pragmas stripped compiles to 16,464 B (+3,040, over the budget by itself); the committed pragmas bring it to +880 B - [BENCHMARKS](../../docs/BENCHMARKS.md#three-epilogue-opcodes-under-the-program-ram-budget-elementwise-mul-per-channel-scale-and-5x5-average-pool-bit-exact-on-silicon-2026-09-18-desktop-2) |
| [program_ram_epilogue_ops_phoenix_20260918T040329Z.log](program_ram_epilogue_ops_phoenix_20260918T040329Z.log) | The three new opcodes on silicon: xclbin built from the censed 15,280 B source (kernel_sha256 bf28f95a...), the 19-round synthetic sequence (was 16) through all sixteen cores, three iterations, 0 mismatching bytes against the emulator (972,800 output bytes per iteration), 2.284 / 0.805 / 0.753 ms, xrt-smi no contexts pre- and post-run - [BENCHMARKS](../../docs/BENCHMARKS.md#three-epilogue-opcodes-under-the-program-ram-budget-elementwise-mul-per-channel-scale-and-5x5-average-pool-bit-exact-on-silicon-2026-09-18-desktop-2) |

## Whole-classifier carve and workspace-reuse readback (2026-09-18, Desktop 2)

Every NPU run below is bracketed by `contexts_classifier_host_pool_carve_20260918T142628Z.log` /
`contexts_workspace_reuse_readback_...` / `contexts_workspace_reuse_noreuse_...` /
`contexts_workspace_slot_cotenancy_...` and their `contexts_after_...` partners, all reporting "No
hardware contexts running on device"; the sweep is offline and needs no witness.

| Log | Outcome |
|---|---|
| [classifier_host_pool_carve_phoenix_20260918T142628Z.log](classifier_host_pool_carve_phoenix_20260918T142628Z.log) | First whole image classifier on the engine, 28/28 layers EXACT on Device 0: 27 layers on the NPU (26 backbone convs + the 1x1 head) and the head's pooling as one declared host region, because nothing lowers GlobalAveragePool - yolov8n-cls at 640^2, dispatch mean 4.511 ms (min 4.285, max 5.032, first 6.338) over 10, host segment 0.914 ms, workspace 15.9 MB under --no-workspace-reuse; a hybrid, not a zero-fallback container - [BENCHMARKS](../../docs/BENCHMARKS.md#a-whole-classifier-on-the-npu-the-heads-pooling-is-carved-to-the-host-2828-layers-exact-2026-09-18-desktop-2) |
| [classifier_vs_onnxruntime_20260918T142628Z.log](classifier_vs_onnxruntime_20260918T142628Z.log) | The other link the silicon verifier cannot close: graph_reference.run_direct against ONNX Runtime CPU on the same frame (assets/bus.jpg), identical top-5 [879, 908, 412, 654, 756], max\|diff\| 0.0000, mean 0.0000, cosine 1.00000 - reference correctness, not dataset accuracy - [BENCHMARKS](../../docs/BENCHMARKS.md#a-whole-classifier-on-the-npu-the-heads-pooling-is-carved-to-the-host-2828-layers-exact-2026-09-18-desktop-2) |
| [model_zoo_classifier_compile_20260918T142628Z.log](model_zoo_classifier_compile_20260918T142628Z.log) | Which classifiers the compiler accepts at all: 1 of 15 XINT8 models reaches a schedulable graph. Nine stop at a 3x3 stride-2 MaxPool the core does not implement (resnet50 and its r192/r256/r320 variants, wide_resnet50_2, wide_resnet101_2, resnext50_32x4d, densenet121 pool0, resnetv2_50x3), regnetx_002 on group-3 convolution, the three mobilevit variants on unrepresentable consumers, and yolov8n-cls at 224 on the 20-pixel tile floor after lowering all 28 layers - [MODEL_ZOO](../../docs/MODEL_ZOO_BENCHMARKS.md#classification-models-one-of-fifteen-reaches-a-schedule-2026-09-18-desktop-2) |
| [workspace_reuse_readback_phoenix_20260918T142628Z.log](workspace_reuse_readback_phoenix_20260918T142628Z.log) | YOLOv8n on the shipped reusing plan, reported the way the plan allows: 25/25 readable layers EXACT and 41/41 reused slots holding their planned final tenant, PASS - where the same container read layer-by-layer prints 25/66 and FAILs on a device that is correct, because a reused slot keeps only its last writer - [BENCHMARKS](../../docs/BENCHMARKS.md#workspace-reuse-makes-41-of-66-layer-readbacks-unobservable-what-verify_engine_container-can-and-cannot-see-2026-09-18-desktop-2) |
| [workspace_reuse_noreuse_phoenix_20260918T142628Z.log](workspace_reuse_noreuse_phoenix_20260918T142628Z.log) | The same model with --no-workspace-reuse: 66/66 layers EXACT, every tensor owning its slot and therefore every layer checkable - the gate a scheduler or allocator change must be run against. Workspace 22.0 MB against 14.1 MB, dispatch means not comparable across runs - [BENCHMARKS](../../docs/BENCHMARKS.md#workspace-reuse-makes-41-of-66-layer-readbacks-unobservable-what-verify_engine_container-can-and-cannot-see-2026-09-18-desktop-2) |
| [workspace_slot_cotenancy_phoenix_20260918T142628Z.log](workspace_slot_cotenancy_phoenix_20260918T142628Z.log) | The question put to the device instead of inferred: 25 slot bases hold the 66 layer tensors, 15 slots have more than one tenant, and all 41 non-final tenants read back byte-identical to their slot's final tenant (diff-as-successor 0, e.g. 0/25,600 where they differ from their own value in 23,834) - so the earlier layers are unreadable, not wrong. PASS, tools/prove_slot_cotenancy.py - [BENCHMARKS](../../docs/BENCHMARKS.md#workspace-reuse-makes-41-of-66-layer-readbacks-unobservable-what-verify_engine_container-can-and-cannot-see-2026-09-18-desktop-2) |
| [retire_batch_cadence_sweep_phoenix_20260919T0251Z.log](retire_batch_cadence_sweep_phoenix_20260919T0251Z.log) | The shim retirement cadence swept on silicon: SESR M7 dispatch 4.573 / **4.311** / 4.964 / 5.254 ms at rb 1/2/3/4 (NOP floor 3.040 / 2.629 / 2.952 / 3.228) and YOLOv8n 7.336 / **7.175** / 7.278 at 2/3/4 - an interior optimum on both and past the peak at 4 on both, while rb >= 5 refuses to emit (`channel o queue is full of held tasks`). Bisect `6bd2718..f3b37cd` names `6a620f0` first bad, and flipping only its `retire_batch` one-liner back to 2 there gives 4.286 ms with workspace reuse left on - so the -43.2% reuse half is innocent and the one-liner was the whole regression - [BENCHMARKS](../../docs/BENCHMARKS.md#the-retirement-cadence-was-the-sesr-dispatch-regression-6a620f0s-retire_batch4-costs-a-thin-container-094-ms-and-the-engine-now-retires-every-2nd-task-2026-09-19-desktop-2) |
| [retire_batch2_verification_phoenix_20260919T0304Z.log](retire_batch2_verification_phoenix_20260919T0304Z.log) | The cadence change costs nothing in output: a reuse-free rebuild verifies **9/9 layers exact** (`dispatch mean 4.376 ms`, PASS) and `tools/sesr_identity_probe.py` over 24 frames of data/sesr_calib reports `identical=24 differing=0 max_diff=0`, the rb=4 and rb=2 containers' combined image SHA-256 equal (`55b129f3...c7114`). xrt-smi witnesses bracket the sequence - [BENCHMARKS](../../docs/BENCHMARKS.md#the-retirement-cadence-was-the-sesr-dispatch-regression-6a620f0s-retire_batch4-costs-a-thin-container-094-ms-and-the-engine-now-retires-every-2nd-task-2026-09-19-desktop-2) |
| [latency_retire_batch2_phoenix_20260919T0306Z.log](latency_retire_batch2_phoenix_20260919T0306Z.log) | Same-sitting interleaved head-to-head after the fix: SESR M7 engine 4.883 / 4.886 ms against AMD's stack 3.820 / 3.844 - still behind, with AMD's `session.run` unchanged at 1.477 ms but their host postprocess at 1.975 ms against the 2.419-2.593 ms of 2026-09-15/16/17. YOLOv8n still wins the same sitting: 8.588 against 10.892 ms - [BENCHMARKS](../../docs/BENCHMARKS.md#the-retirement-cadence-was-the-sesr-dispatch-regression-6a620f0s-retire_batch4-costs-a-thin-container-094-ms-and-the-engine-now-retires-every-2nd-task-2026-09-19-desktop-2) |

## Dispatch floor, and the activation packet priced against it (2026-09-20, Desktop 2)

| Log | Outcome |
|---|---|
| [split_segment_floor_phoenix_20260920.log](split_segment_floor_phoenix_20260920.log) | Floor separated from compute by NOP copy across the whole YOLOv8 family on split containers: the floor is 69.6%-75.4% of NPU dispatch (mean 72.8%) over a 19.7x range in task count, 1.594-1.941 us per task, and the shim runs at 1.33-1.60 GB/s per column against 6.899 achievable, so 77%-81% of the floor is not transfer. Per-task and per-byte cost are collinear by construction at a fixed 6,400 B packet and are NOT attributed - [BENCHMARKS](../../docs/BENCHMARKS.md#the-dispatch-floor-separated-from-compute-and-the-activation-packet-priced-against-it-2026-09-20-desktop-2) |
| [object_size_repricing_20260920.log](object_size_repricing_20260920.log) | The uncommitted 2026-09-16 object-size sweep re-priced against the measured floor: it charged 224.6 MB of core-side activation fills at the DDR rate when only 74.5 MB crosses the shim, over-counting transport 3.10x and under-counting per-task cost 2.1-2.6x. Its total was within 1.1% of the floor, which hid the error; its split read 62.2/37.8 where the measured split is 20.2/79.8. DERIVED over measured inputs; no container was built - [BENCHMARKS](../../docs/BENCHMARKS.md#the-dispatch-floor-separated-from-compute-and-the-activation-packet-priced-against-it-2026-09-20-desktop-2) |
| [activation_object_size_sweep_20260920.log](activation_object_size_sweep_20260920.log) | What a different activation object would cost, over real schedules with the 6,400 B control asserted against the silicon task counts: 7,920 B is not buildable (not a multiple of 32, so k3s1 loses ncin = 4), the object must be a multiple of 800 and at most 9,472 B, and every buildable candidate is a regression. Packets are not tasks - k3s2's 2-D pattern folds 23.6-39.6 packets into one task where 3-D kinds fold about 4, so 6,400 B is the largest object that keeps it two-dimensional. OFFLINE, no container built - [BENCHMARKS](../../docs/BENCHMARKS.md#the-dispatch-floor-separated-from-compute-and-the-activation-packet-priced-against-it-2026-09-20-desktop-2) |
| [merge_depth_sizing_20260920.log](merge_depth_sizing_20260920.log) | Why task count, not packet count, sets the floor: canonical/merge_quad/merge_runs each add one BD dimension and both merges refuse a 4-D pattern, so a fill must be 2-D canonical to get both. k3s2's (16, 400) is and folds 23.6-39.6 packets per task; k1's (8, 5, 160) is not and folds 4.7, on 38% of YOLOv8s's fills. Packing planes adjacent so k1 becomes contiguous is byte-free and worth -2.163 ms on YOLOv8s (-0.393 on YOLOv8n) on identical traffic; adding k3s1 at 1.60x its bytes is worth a further -2.17 ms, reversing the earlier "not proposed" on that half, which had priced bytes 3.10x too dear. DERIVED, control asserted against silicon; no container built - [BENCHMARKS](../../docs/BENCHMARKS.md#the-merge-depth-lever-k1-is-stuck-at-a-quarter-of-k3s2s-merge-depth-and-that-is-22-ms-on-yolov8s-2026-09-20-desktop-2) || [plane_packed_layout_20260920.log](plane_packed_layout_20260920.log) | Band-interleaving a tensor's channel blocks every five rows (Placement.band_rows) so a k1 fill is contiguous, drops from three dimensions to two and so admits merge_runs on top of merge_quad: 66/66 layers EXACT on Device 0, and same-sitting interleaved 300-iteration dispatch of 7.371 -> 7.236 ms (yolov8n) and 16.901 -> 16.580 / 16.945 -> 16.588 ms (yolov8s) on UNCHANGED activation bytes, at +5.15%/+5.51% workspace. Only 22 of 61 read tensors are eligible - one wider reader disqualifies a tensor, because k3s1/k3s2/pool/k1up2 all cross a band boundary. Supersedes this log's first version, which read WRONG ON SILICON: GraphSession.read_tensor had its own plane-major reshape and the layout was right all along - [BENCHMARKS](../../docs/BENCHMARKS.md#the-plane-packed-activation-layout-a-contiguous-fill-merges-deeper-and-is-034-ms-on-yolov8s-2026-09-20-desktop-2) |
| [layout_ab_yolov8s_phoenix_20260920T1932Z.log](layout_ab_yolov8s_phoenix_20260920T1932Z.log) | Interleaved same-sitting glass-to-glass on YOLOv8s, three arms twice, 50 warm-up and 500 timed frames: AMD 16.954 / 16.981 ms, Ignition on the plane-major control 17.786 / 17.800, Ignition on the band-packed container 17.471 / 17.559. Banding is worth 0.278 ms G2G and 0.297 ms of dispatch and closes 34% of the gap to AMD, from 0.825 to 0.547 ms behind - it narrows the YOLOv8s gap, it does not close it. Neither container used --silu-sigmoid, so this is not the shipped configuration - [BENCHMARKS](../../docs/BENCHMARKS.md#the-plane-packed-activation-layout-a-contiguous-fill-merges-deeper-and-is-034-ms-on-yolov8s-2026-09-20-desktop-2) |
| [plane_packed_verify_yolov8s_20260920.log](plane_packed_verify_yolov8s_20260920.log) | YOLOv8s on the band-packed layout, every layer readable (--no-workspace-reuse): 66/66 EXACT on Device 0. Completes plane_packed_layout_20260920.log, which checked yolov8n only and listed yolov8s as not established - [BENCHMARKS](../../docs/BENCHMARKS.md#the-plane-packed-activation-layout-a-contiguous-fill-merges-deeper-and-is-034-ms-on-yolov8s-2026-09-20-desktop-2) |
| [layout_ab_silu_yolov8s_phoenix_20260920T1957Z.log](layout_ab_silu_yolov8s_phoenix_20260920T1957Z.log) | The same three-arm interleaved sitting on the SHIPPED configuration (--silu-sigmoid): AMD 16.915 / 16.836 ms, control 18.236 / 18.240, band-packed 17.779 / 17.798. Banding is worth 0.449 ms here and 0.278 on the plain pair, from an identical schedule, so quote 0.28-0.45 ms and read the difference as drift between sittings. The engine goes from 1.362 to 0.913 ms behind AMD: the largest activation-side lever left, and it does NOT close the gap - [BENCHMARKS](../../docs/BENCHMARKS.md#the-plane-packed-activation-layout-a-contiguous-fill-merges-deeper-and-is-034-ms-on-yolov8s-2026-09-20-desktop-2) |
| [plane_packed_verify_pose_20260920.log](plane_packed_verify_pose_20260920.log) | YOLOv8n-pose on the band-packed layout, every layer readable: 75/75 EXACT on Device 0 - [BENCHMARKS](../../docs/BENCHMARKS.md#the-plane-packed-activation-layout-a-contiguous-fill-merges-deeper-and-is-034-ms-on-yolov8s-2026-09-20-desktop-2) |
