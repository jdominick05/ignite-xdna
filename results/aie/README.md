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

## Dispatch floor and the int8 conv verdict

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

## Also here

`aiecompiler_help.log` and `aiecompiler_x86sim_passthrough.log` are on disk but were not
covered by the `results/README.md` entry this file was built from, so nothing is claimed
about them here.
