# Standalone BF16 transformer operations

Experimental single-token Phoenix kernels and a reusable `Bf16Ops` host wrapper.
Model arithmetic executes on the NPU through `ignite_xdna.runtime.driver`.
The host packs raw BF16 bits and dispatches each operation. Host round trips
remain in this baseline. These kernels are outside the ONNX
compiler and do not add operations or precision modes to `.ignite` containers.

| Method | Supported layout and semantics |
| --- | --- |
| `projection` | One BF16 vector times BF16 weights `[N,K]`; pads K to 256 and N to 16; FP32 chunk carry, BF16 output |
| `norm` | RMSNorm, epsilon `1e-6`; width 256 or 512, at most 2048 elements; FP32 normalization, BF16 output |
| `rope` | Four query heads, width 256 or 512; supplied BF16 cosine/sine; both product roundings preserved |
| `softmax` | BF16 score rows, each 1 to 32768 elements; streamed statistics, BF16 probabilities; exp tails below -80 become zero |
| `geglu` | 2048-element BF16 gate and up vectors; supplied lookup table for tanh GELU; rounds activation before multiplication |
| `residual` | BF16 addition followed by BF16 scalar multiplication; at most 2048 elements, multiple of 16 |
| `topk` | At most 4096 finite BF16 scores, k from 1 to 32; descending value, ascending index on ties |

Inputs and outputs use NumPy `uint16` arrays containing BF16 bits, not numerically
converted integers. `common.py` provides explicit bit conversions for reference
generation. Kernel entry symbols retain their original `draft_` ABI names; the
operations have no dependency on a model checkpoint or the external chat workspace.

```python
from kernels.transformer_bf16 import Bf16Ops

ops = Bf16Ops(cache_dir="scratch/bf16-cache")
y_bits = ops.projection(x_bits, weight_bits)
```

`precise_projection=True` selects a second GEMV implementation with compensated
FP32 accumulation and BF16 operands/output. It is an accuracy experiment; it
does not guarantee bit agreement with a CPU library's accumulation order.
The original matrix-unit implementation remains the default.

To reuse instructions and buffers, pass an explicitly owned session:

```python
from kernels.transformer_bf16.persistent import PacketSession

with PacketSession(max_contexts=2) as transport:
    ops = Bf16Ops("scratch/bf16-cache", transport=transport)
    y_bits = ops.projection(x_bits, weight_bits)
```

The caller must reserve exclusive NPU use for the session. Idle witnesses bracket
it; the research wrapper monitors foreign contexts during bounded checks. An LRU
cache retains at most two contexts with instruction/data BOs. Eviction and
context-manager exit release owned BOs before contexts. Image admission verifies
source/artifact hashes; successful exit checks that all admitted files remain
unchanged. Replays use already-loaded BOs. Each operation checks output guards
and input preservation. Calls are
synchronous and restricted to the thread that opened the session.

Each image records source and compiler hashes, transaction and xclbin hashes,
code size, stack frames and SRAM spans. Standalone dispatch checks hashes and
idle state each time; cached sessions check these at the boundaries described
above. Construct a new ops/session instance to adopt edited sources or images.
Build artifacts belong in `scratch/`, not alongside these sources. Toolchain paths
can be set with `MLIR_AIE_IRON`, `PEANO_INSTALL_DIR`, `MLIR_AIE_INSTALL_DIR` and
`XRT_ROOT`; defaults match the existing Windows research environment.

Prepare independent references in a torch environment:

```powershell
python kernels/transformer_bf16/probe.py prepare --fixtures scratch/bf16-fixtures
```

Run checks in the PyXRT environment, through the bounded research wrapper:

```bash
bash scripts/research-lowlevel.sh --npu --checks-only --seconds 180 --rss-gib 6 \
  --log results/aie/transformer_bf16_checks.log -- \
  python kernels/transformer_bf16/probe.py run --fixtures scratch/bf16-fixtures \
  --cache scratch/bf16-cache --out scratch/bf16-checks.json
```

The preparation checks the GELU table against every finite BF16 input. Device
checks cover each operation against independent CPU references. Passing these
checks establishes individual-operation correctness within the recorded criteria;
it does not establish whole-model accuracy, fused execution, serving latency or
hybrid GPU/NPU benefit. Model orchestration and checkpoint conversion stay with
their respective applications.

The [bounded device check](../../results/aie/transformer_bf16_checks_20261018_v2.log)
and [structured results](../../results/aie/transformer_bf16_checks_20261018.json)
are indexed with [method and limits](../../docs/BENCHMARKS.md#standalone-bf16-transformer-primitives-2026-10-18-desktop-2).

The cached transport has a separate replay/eviction check:

```bash
bash scripts/research-lowlevel.sh --npu --checks-only --seconds 180 --rss-gib 6 \
  --log results/aie/transformer_bf16_cached_checks.log -- \
  python kernels/transformer_bf16/session_probe.py --fixtures scratch/bf16-fixtures \
  --cache scratch/bf16-cached --out scratch/bf16-cached-checks.json --precise-projection
```

Its [28 passing checks](../../results/aie/transformer_bf16_cached_transport_20261018.log)
include three distinct inputs and an exact repeat of the first for each operation.
The consuming application's complete draft still fails its broader numerical
qualification; see the [full record](../../docs/BENCHMARKS.md#complete-draft-composition-and-cached-transport).

An experimental opcode dispatcher includes the seven operations in one core
program. Enable it with `single_image=True` on `Bf16Ops`
and `share_program=True` on `PacketSession`. Context reuse across packet counts
requires byte-identical PDI, kernel argument metadata and memory topology. A
matching ELF alone is insufficient. Each context retains at most 32 distinct
transaction/data-BO sets; allocation uses that context's full XRT group IDs.
This is a shared program, with host round trips between operations, not a fused
decoder graph. Its size-optimized build stays within the 16 KiB program limit.
The default projection uses the matrix unit; `precise_projection=True` selects
compensated accumulation through a separate opcode. Both store BF16 operands and
return BF16 output. Selecting compensated accumulation does not imply closer
agreement with every CPU-library rounding decision.

Add `--single-image --check-drift` to the cached probe to test opcode routing and
successful-exit rejection of a changed cached manifest. Only the probe's own
cache manifest is temporarily changed and then restored.

`state.py` provides an experimental `ArenaProgram` command stream. A 32 KiB
BF16 activation arena retains intermediate values on the compute tile; host
packets contain weights, metadata and explicit uploads. The builder rejects
out-of-range spans and reads of unwritten locations. Uploads must initialize
every value consumed by a new execution. It does not rely on state surviving
between dispatches. The size-optimized core program uses 15,648 bytes of the
16 KiB program memory; stack and all SRAM allocations remain checked by the
stream builder. This research path is separate from the ONNX compiler.

The [hardware replay check](../../results/aie/transformer_bf16_state_checks_20261018_v2.log)
and [structured evidence](../../results/aie/transformer_bf16_state_checks_20261018_v2.json)
cover seven outputs over A/B/A inputs, all bitwise equal to the existing native
packet operations. The check includes a GeGLU-to-residual chain without a host
round trip, odd-length softmax padding, bounds/unwritten-read rejection,
unchanged input BOs and released contexts. It does not establish whole-model
accuracy or serving throughput.

`ArenaCompiler` binds multiple command streams to one source snapshot and caches
their checked build metadata. Construct a new compiler to adopt source changes;
`PacketSession` still admits hashes and rechecks them at successful exit.
`program.prepare(cache)` creates an owned, immutable packet template. Execute it
with `prepared.execute(session)`, or provide BF16 replacements for explicitly
named uploads with `uploads={"inputs": bits}` after declaring
`program.upload(slot, bits, label="inputs")`. Patches must keep their original
element count and dtype and cannot alter command headers. Patching
copies the template; the template remains reusable. Every execution still
uploads, reads back and verifies the entire input BO. This does not cache
weights in device DDR or eliminate target-KV preparation.

The [prepared-upload check](../../results/aie/transformer_bf16_prepared_checks_20261018.log)
([JSON](../../results/aie/transformer_bf16_prepared_checks_20261018.json)) retains
the seven-output A/B/A check and adds three prepared executions with changed
named input data, plus unknown-label and wrong-shape rejection. Transport
evidence now separates input transfer, output poisoning/readback and input
verification costs from the dispatch wait. These are diagnostic observations;
see the [recorded limits](../../docs/BENCHMARKS.md#prepared-e2b-command-templates-2026-10-18).

Prepared streams also accept explicitly named matrix and rotary replacements.
Declare `projection(..., label="key")` or `rope(..., label="position")`, then
provide `matrices={"key": bits}` and `rotations={"position": (cosine, sine)}`
to `prepared.execute`. Matrix replacements must preserve both dimensions and
BF16 storage; the builder repacks them for the selected projection opcode.
Rotary replacements preserve the declared width. Undeclared weights, headers,
and the original template remain unchanged. A changed KV length requires a new
schedule unless the caller explicitly builds the fixed-capacity path below.

The [changed-state attention probe](../../results/aie/transformer_bf16_patch_checks_20261018.log)
([JSON](../../results/aie/transformer_bf16_patch_checks_20261018.json)) checks A/B/A
queries, keys, values and rotary factors for both projection implementations.
All six cases match native rotary/probability/context outputs bitwise. The
final A uses no patches, checking template preservation and state reset. Ten
invalid matrix/rotary patch cases are rejected before dispatch. This establishes
fixed-shape transport, not growing-context serving or model accuracy.

Before every hardware run, inspect CPU inference activity, GPU engines and NPU
contexts with `tools/check-inference-contention.ps1`. Add `-RequireIdle` before
performance measurements; a busy or unavailable signal prevents admission.
This read-only preflight never stops another job or prints its command line.
Continue using the bounded research wrapper to monitor contention during a run;
a preflight snapshot does not establish that the whole sitting was idle.

`Bf16Ops(..., refined_norm=True)` selects the separate experimental
`rmsnorm_refined.cc` image, including when other operations use `single_image`.
It is not enabled in the activation-arena dispatcher or adopted by default.
The candidate splits FP32 products to correct inverse-square-root residuals
without assuming a fused FP32 multiply-subtract. Isolated checks pass, but its
full-model qualification still fails; see the
[growing-cache diagnosis](../../docs/BENCHMARKS.md#e2b-growing-cache-trajectory-and-rmsnorm-diagnosis-2026-10-18).

`exact_norm=True` selects a second standalone norm candidate using
`bf16_product.hpp` for the two final products. The BF16 operand has an eight-bit
significand, so its product with a 24-bit FP32 significand fits a uint32. The
helper rounds that integer product to FP32 with ties to even before the final
BF16 conversion. It handles the boundary that rounds up to the smallest normal
FP32 value; exceptional operands and other underflow/overflow cases retain the
original float-product fallback. This is not an exact RMSNorm mean or inverse,
and the fallback paths are outside the helper's bitwise qualification.

The [product sweep](../../results/aie/transformer_bf16_product_bitwise_checks_20261018_v2.log)
([JSON](../../results/aie/transformer_bf16_product_bitwise_checks_20261018_v2.json))
passes 243,581 normal finite cases bitwise. The norm candidate passes all five
real growing-cache numerical cases and nine of twelve synthetic cases under
the unchanged gates. It remains opt-in; the command-stream option below now
embeds it. Three synthetic failures and serving integration remain open. Select at most
one of `refined_norm` and `exact_norm`.

For the command-stream path, `ArenaCompiler(exact_norm=True)` opts into a
matrix-only dispatcher with the corrected normalization products. It uses
`-Oz`, shares the larger arithmetic helpers, and inlines BF16 transport helpers
to keep conservative stack accounting inside the existing 4 KiB reserve. The
checked program occupies 12,144 bytes; summed stack frames plus the 256-byte
margin total 3,424 bytes. This mode rejects `projection(..., precise=True)`
before building packets because compensated GEMV is absent from its image.
The default compiler retains both projection variants and the original norm.

The [corrected arena check](../../results/aie/transformer_bf16_e2b_exact_arena_20261018.log)
([registered evidence](../../results/aie/transformer_bf16_e2b_exact_arena_20261018.json))
passes all five real growing-cache states and return-to-first-state replay,
bitwise equal to the standalone corrected composition at every checked stage.
Body and head need two submissions and reuse one exact program context without
eviction. The [generic checks](../../results/aie/transformer_bf16_exact_arena_checks_20261018.log)
([JSON](../../results/aie/transformer_bf16_exact_arena_checks_20261018.json)) cover
all seven outputs over A/B/A, prepared uploads and rejection of unsupported
compensated projections. This remains a captured-state research path, not a
live GPU bridge, sampled speculative verification or demonstrated speedup.

`norm_state_probe.py` is a dedicated diagnostic entry: the output prefix and
last sixteen words hold raw FP32 states/products, rather than ordinary guards.
Only that diagnostic kernel enables these stores. It verifies the untouched
guard gap and input BO and compares CPU values only after execution.

The external [GPU target bridge](../../tools/ggml_gpu_target/README.md) now feeds
these primitives live. The [bounded E2B hybrid experiment](../../docs/BENCHMARKS.md#live-e2b-gpu-target-and-npu-drafter-2026-10-18)
implements greedy draft chains, target verification and rejected-suffix rollback.
Its passing 32-token case requires an explicit fusion-disabled GPU control;
the default-fusion longer case fails equivalence. This does not supersede the
open synthetic numerical failures or qualify serving performance.

`ArenaCompiler(exact_norm=True, compact=True)` optionally returns only declared
exports plus one aggregate guard frame. Arithmetic still uses the same arena
kernel, with a local 4,224-byte result scratch buffer. Header word 11 carries
export/end markers inserted by `ArenaProgram.packet_array()`; prepared patches
cannot change them, and the transport rejects a changed schedule before DMA.
The core checks all sixteen prefix guards after every command, including
non-exported commands, and resets aggregate status at each stream boundary.
`output_blocks` in the build manifest describes the reduced output allocation;
legacy manifests retain their original command-count-sized result buffers.

The [generic compact check](../../results/aie/transformer_bf16_compact_checks_20261018.log)
([provenance](../../results/aie/transformer_bf16_compact_checks_20261018.json))
matches seven outputs over A/B/A and prepared input updates, detects an invalid
non-exported command, recovers on the next valid stream, and rejects changed
routing. The [live compact comparison](../../results/aie/gemma4_e2b_live_compact_dictionary32_20261018.log)
([JSON](../../results/aie/gemma4_e2b_live_compact_dictionary32_20261018.json))
matches every body export and all candidate logits against full-frame execution.
The [ordinary compact live run](../../results/aie/gemma4_e2b_live_compact_dictionary32_run_20261018.log)
([JSON](../../results/aie/gemma4_e2b_live_compact_dictionary32_run_20261018.json))
matches 32 GPU control tokens with one NPU context, zero evictions and idle exit.
Body readback is 42,240 bytes instead of at least 12,802,944 bytes; full input
uploads remain in the default mode; complete immutability checks remain in all
modes. These are checks-only results, not speedups.

`PacketSession(resident_inputs=True)` additionally enables changed-page input
updates for prepared command streams. Each stream owns a read-only host snapshot
of its last fully verified input BO. Exact uint32 comparison selects changed
4 KiB pages, merges adjacent ranges and writes/syncs only those ranges. Fresh or
evicted streams upload the full buffer; unchanged streams write zero bytes.
Rebuilt command execution retains full uploads as a comparison path. Complete
input readback verification, output poisoning and compact aggregate checks still
run on every dispatch. Transport and range-helper source hashes are frozen for
the session and included in evidence.

The [resident-input qualification](../../results/aie/transformer_bf16_resident_inputs_20261018.log)
([provenance](../../results/aie/transformer_bf16_resident_inputs_20261018.json))
covers page boundaries/tails, A/A/B/A/A, zero-byte repeat uploads, and complete
reinitialization after context eviction. The [live comparison](../../results/aie/gemma4_e2b_live_resident_dictionary32_compare_20261018.log)
([JSON](../../results/aie/gemma4_e2b_live_resident_dictionary32_compare_20261018.json))
matches all body exports and candidate logits against full uploads. Its second
draft updates 8,192 bytes of a 49,853,888-byte body input. The
[ordinary resident run](../../results/aie/gemma4_e2b_live_resident_dictionary32_run_20261018.log)
([JSON](../../results/aie/gemma4_e2b_live_resident_dictionary32_run_20261018.json))
matches 32 GPU tokens with one context and no evictions. Static packet weights
remain in the device-accessible input BO, not compute-tile SRAM; the NPU still
reads them each invocation. Host comparison/copies and full readback remain.

For a declared capacity that fits the arena, `softmax(..., capacity=N, label='name')`
records a validated true-length patch for the two fixed softmax passes.
`PreparedArena.execute(lengths={'name': actual_length}, ...)` accepts only
integer lengths in `1..N`; opcodes, routing and command count remain fixed.
It normalizes only the active range and does not clear the inactive arena tail.
A fixed-capacity attention caller must zero-pad its key/value data and establish
zero logits in the inactive range before consuming that whole range. The
[64-capacity](../../results/aie/transformer_bf16_padded_lengths_20261018.json)
and [512-capacity](../../results/aie/transformer_bf16_padded_lengths512_20261018.json)
device checks cover growth, shrinkage, restoration and invalid lengths bitwise
against exact-length execution.

Application templates remain immutable after data patches. Reusing one across
prefixes requires reapplying the current prefix's KV, rotary and length patches
on every execution, even when only query inputs change within the draft chain.
The [corrected live capacity comparison](../../results/aie/gemma4_e2b_live_capacity_dictionary32_compare_20261018_v3.json)
checks all body exports and candidate logits over 38 proposals. It builds one
body template and reuses it 37 times. The
[diagnostic latency record](../../docs/BENCHMARKS.md#diagnostic-loop-latency-and-padded-kv-schedules)
shows why this is experimental: warm hybrid loops remain slower than their
instrumented GPU-only controls; packing savings do not establish net acceleration.

`ArenaCompiler(exact_norm=True, small_packets=True)` optionally uses 2,192-word
packets rather than 4,112. Resident GELU opcode 11 reads the same 4,096-entry
table from protected arena slots 8,192–12,287; the first GELU call inserts one
table upload for the whole command stream. Table-region conflicts, replacement
tables and uploads over 4,352 BF16 elements are rejected. Device checks reject
out-of-bounds table reads and oversized uploads before memory access. The wide
opcode-5 format and numerical calculation remain available as the control.
Small packets currently require matrix projections with the exact-norm mode.

The [generic comparison](../../results/aie/transformer_bf16_small_packets_checks_20261018_v2.json)
matches all seven outputs bitwise over A/B/A and has zero GELU error against
the frozen Torch fixture. The [live comparison](../../results/aie/gemma4_e2b_live_small_packets_compare_20261018.json)
matches all body exports/candidate logits and 32 GPU tokens. Body input is
29,039,616 bytes instead of 54,459,328 in the wide padded schedule. The
[diagnostic profile and limitations](../../docs/BENCHMARKS.md#smaller-packets-with-an-arena-resident-gelu-table)
record a warm 4.611 s hybrid loop versus 1.668 s for its instrumented GPU control;
smaller packets improve this prototype's costs but do not establish net acceleration.

Named dynamic softmax schedules now span multiple 512-element chunks. Length
patches update each chunk's valid input/output counts; unused chunks have zero
valid elements and preserve the accumulated statistics. The
[2,560-capacity check](../../results/aie/transformer_bf16_multichunk_lengths_20261018.json)
matches exact-length attention across chunk boundaries, growth and shrinkage.
The caller must still zero-pad unused key/value rows. Gemma's larger-context
composition caps global capacity at 2,560 and sliding capacity at 768, leaving
the protected GELU table intact. This adds no ONNX graph-engine operator.

`ArenaCompiler(exact_norm=True, compact=True, small_packets=True, input_depth=2)`
adds a second input FIFO buffer, allowing DMA to prepare the next packet while
the core consumes the current one. The default remains one buffer. Other arena
formats reject depth two; their buffers do not fit the same SRAM budget.
Resource checks cover both input allocations, stack, state, output and scratch.
The [device check](../../results/aie/transformer_bf16_double_buffer_checks_20261018.json)
matches all exports over A/B/A, rejects a corrupt unexported command through the
aggregate guard and recovers on the next invocation. The
[live comparison](../../results/aie/gemma4_e2b_live_double_buffer_compare_20261018.json)
matches every body export/candidate logit and all target tokens. See the
[diagnostic profiles](../../docs/BENCHMARKS.md#double-buffered-bf16-arena-input)
for measured costs and limitations; this does not establish a serving speedup.

`ArenaCompiler(..., vector_copy=True)` optionally accelerates arena copies and
input padding in small-packet mode. Complete aligned 32-element BF16 spans use
vector loads/stores; unaligned copies and short tails retain scalar operations.
The [boundary check](../../results/aie/transformer_bf16_vector_copy_checks_20261018.json)
covers raw bit preservation, adjacent guards, final-arena-element accesses and
maximum uploads, plus all operation exports over A/B/A. The
[live comparison](../../results/aie/gemma4_e2b_live_vector_copy_compare_20261018.json)
preserves every body export/candidate logit and all target tokens. Arithmetic
and numerical gates are unchanged. The
[diagnostic profiles](../../docs/BENCHMARKS.md#vector-copies-in-the-bf16-arena)
show reduced costs while the hybrid still trails its GPU-only control.

Prepared matrix patches now pack tile payloads with batched NumPy reshapes and
one strided assignment instead of a Python loop per tile. The frozen scalar
layout matches exactly across both layouts, partial tiles, noncontiguous
matrices and all BF16 bit patterns. Templates remain immutable and every input
BO is still fully verified. This changes host packing only; repeated live
32-token requests match GPU controls at draft depths one and two. See
[evidence and limits](../../docs/BENCHMARKS.md#batched-host-matrix-packing).

The separate research modules `prefix_cache.py` and `prefix_binding.py` explore
reuse of prepared prefix patches without changing `PreparedArena` or the
default serving path. Content caching retains one immutable entry and rehashes
caller data to detect mutation. Binding instead transfers an exclusively owned
materialized command buffer to one thread; every subsequent call must replace
all declared upload payloads and cannot change matrices, rotary data or lengths.
A new prefix requires a new binding. Failed dispatches invalidate the bound
owner. Every dispatch retains complete input BO readback and verification.

The [live packet audit and diagnostic comparison](../../docs/BENCHMARKS.md#owned-prefix-binding-avoids-repeated-kv-packing)
compare complete NPU input/output hashes before collecting separate latency
observations. The initial content-cache attempt is a negative result; owned
binding reduces body packing costs in this bounded workload. These candidates
are not enabled in the frozen assistant runtime or its pending energy protocol.
