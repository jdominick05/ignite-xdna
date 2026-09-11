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

## Retractions and supersessions in this directory

Read these before quoting anything below.

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

## Also here


`aiecompiler_help.log` and `aiecompiler_x86sim_passthrough.log` are on disk but were not
covered by the `results/README.md` entry this file was built from, so nothing is claimed
about them here.


