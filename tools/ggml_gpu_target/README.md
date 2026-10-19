# Bounded GPU target bridge

`worker.cpp` runs the Gemma main model and embedding lookups on ROCm. The Python
`Target` client owns its process and uses bounded binary stdin/stdout frames.
The experimental assistant retains loaded model weights across requests.

`worker_logits.cpp` and `worker_blocks_logits.cpp` are isolated research forks
for full-vocabulary output. Build with `build_logits.cmd serving|blocks
OUTPUT_EXE`; existing outputs are refused and `/W4 /WX` is enforced. Operation
11 exports the last computed output rows without inference, accepting 1..16
rows and the fixed 262,144-token vocabulary. The wire shape is 32768×8×rows×1
to retain existing dimension/frame bounds. `LogitTarget.logits()` restores
vocabulary order and validates it against greedy predictions in the study.
Read immediately after prepare/decode: rollback and
an identical block-cache hit explicitly invalidate export rather than returning
stale logits. These workers are outside the runtime and SDK.

`tools/qualify_gemma4_block_distribution.py --prepare-only` freezes 24 held-out
states, source/artifact identities, arm order and diagnostic criteria before
measurement. Without that flag it verifies the frozen inputs, compares export
workers to their unmodified counterparts, and saves the full FP32 vectors.
`tools/gemma4_distribution.py` computes FP64 host statistics for offline
analysis only. The existing BF16 reference supplies token IDs; its clipped
probabilities are not treated as an oracle. This sampled diagnostic cannot
waive the exact serving gate or establish corpus perplexity/agent quality.
The completed study rejects both block sizes: all 24 sampled greedy tokens
match, but KL and next-token NLL drift exceed the frozen limits. Exact export
fidelity passes for all three instrumented arms. See
[the negative quality result](../../docs/BENCHMARKS.md#frozen-distribution-diagnostic-for-fixed-block-prefill).

The terminal launcher accepts `-AdmissionWaitSeconds 0..60` (Python:
`--admission-wait 0..60`), default zero. It retries only `HardwareBusy` before
the first output chunk, closes each rejected stream before waiting, and keeps
the same assistant. The deadline bounds retry scheduling; telemetry and an
admitted generation can finish later. Partial output and numerical errors are
never replayed. One-shot `--report` diagnostics retain admission rejections and
are written after cleanup even when admission fails with exit code 2. Reports
can include private prompts/output; keep them local unless reviewed.

`build.cmd OUTPUT_EXE` compiles an explicitly named candidate with the current
host's MSVC and llama.cpp headers. It does not replace the serving executable.
The diagnostic build was compiled with `/W4` without warnings. Binary/source
identities and controls are registered by the qualification tools.

Operations 1/2/3/4/5/7 retain their existing decode, rollback, embedding, reset,
render and new-prompt semantics. Diagnostic operation 8 (`Target.timing()`)
returns cumulative counters for completed decode calls, grouped by operation
1/4/7. It does not perform inference or reset counters. It requires a worker
built with this operation; the older serving executable does not implement it.
Existing model response frames are unchanged.

`worker_stages.cpp` is a separate research fork built by
`build_stages.cmd OUTPUT_EXE` with `/W4 /WX`. It retains the five required
serving exports and captures one additional GPU FP32 tensor. Operation 10
accepts a bounded stage index and layer 0..13; it selects capture without model
arithmetic. It is unavailable in the serving worker. The existing client/frame
bounds remain in force. `tools/probe_gemma4_prefill_stages.py` compares every
regular state, shape, position and greedy prediction to an independent serving
worker before using a stage comparison. Its layer suite finds the first changed
layer output and scans that layer; its attention suite scans layer 0 scores,
softmax and weighted values. Score comparisons exclude future keys; softmax
also requires all future/padding probabilities to be bitwise positive zero.
Admission retries are bounded and retain rejected snapshots. A successful
diagnostic's `passed` field means instrumentation fidelity and cleanup passed,
not that its shared prefix is invariant. See
[the operation-level findings](../../docs/BENCHMARKS.md#prefill-divergence-localized-to-the-weighted-value-product).

`worker_prefix.cpp`, `prefix_history.hpp` and `prefix_client.py` are a separate
GPU prompt-reuse experiment, **not a serving replacement**. Build with
`build_prefix.cmd OUTPUT_EXE`; `build_prefix_test.cmd OUTPUT_EXE` builds the
hardware-free history oracle. Final builds enforce `/W4 /WX`. Original serving
sources and executables are retained.

`worker_blocks.cpp` is a separate fixed-block experiment built with
`build_blocks.cmd OUTPUT_EXE` (`/W4 /WX`, refuses existing output). Cold prepares
and cached suffixes use fixed batches with causal trailing padding. Future
rows are removed before replying. Cache reuse retains only complete shared
blocks from prior prefill, recomputing the affected tail; identical prompts
can return their saved state. Generated rows are not reused as prefill.
Operation 7 forces cold blocks; operation 9 requests reuse; ordinary token
decode remains unpadded. Host history copies GPU FP32 KV bytes only.
`tools/qualify_gemma4_block_prefill.py` compares independent serving, cold-block
and cached-block workers, including teacher-forced continuations and rollback.
It reports cache consistency separately and still requires exact unchanged
serving state for its overall pass. Consistency under a different prefill
schedule alone does not qualify a serving replacement.
The research setting `IGNITION_GPU_PREFILL_BLOCK` accepts 128 (default) or 256;
the qualifier exposes it as `--block-rows`. Both sizes passed cached-versus-cold
consistency at the 511/514 boundary but failed the unchanged-serving gate,
including one teacher-forced continuation prediction. See the
[fixed-block results](../../docs/BENCHMARKS.md#fixed-block-prefill-makes-reuse-consistent-but-changes-serving-state).

Experimental operation 9 (`PrefixTarget.prepare_cached`) retains a matching
GPU KV prefix and returns complete GPU-produced FP32 state for the drafter.
Host history only copies tensor bits. Cache reuse checks the retained GPU
position range; resets and rollback update the host history. Generated decode
rows are recomputed as part of a new prefill suffix. Padding/retention-class
heuristics are experiments, not a validated model of all backend decisions.
Operation 8's group 7 token count includes evaluated padding rows.

The short-prompt fixture passes its exact state/continuation gate after padding
and limiting reuse to the previous prefill span. Changed 511-token prompts
still fail, including a changed prediction in one variant. Graph-disable and
alignment attempts have not qualified the mode. No assistant frontend or SDK
selects this worker. See the [complete results and failed attempts](../../docs/BENCHMARKS.md#gpu-prompt-reuse-passes-short-cases-but-fails-the-long-prompt-gate).

`process_ms` covers `llama_process` plus synchronization. `capture_ms` is a
nested callback interval including tensor readback and possible GPU waiting;
it must not be added to `process_ms` or interpreted as isolated copy bandwidth.
Other fields cover finite-logit checking/argmax, tensor serialization and pipe
sending. Counters include processed tokens, callbacks, captured bytes and
response bytes. They are host wall diagnostics, not device kernel timings.
Timing responses have a versioned, bounded JSON payload, validated by the client.

`tools/profile_gemma4_target.py` compares a candidate against a preserved binary
at identical batch shapes, including all captured FP32 state bytes. The
`tools/profile_gemma4_coexistence.py` diagnostic compares fixed GPU inputs with
NPU contexts absent, resident or drafting between GPU calls, plus a deliberate
40 ms idle control. Its NPU proposals are discarded; it is not a speculative
decoding benchmark. Both tools require explicit local asset paths, enforce
admission, retain partial reports and leave foreign workloads untouched.
See [measurements and limits](../../docs/BENCHMARKS.md#gpu-target-decode-and-idle-gaps).

The coexistence tool also supports `--suite priming`, which measures an extra
GPU embedding lookup before verification and includes that lookup in loop time.
That experiment did not eliminate the idle-gap penalty and is not a serving
optimization. `--graph-debug` records runtime graph-use witnesses;
`--disable-graphs` requests graph scheduling off in that research child only.
`--reference-report` requires reference tokens and every GPU state snapshot to
match a prior complete raw report. Both graph configurations passed those
comparisons, but graph disabling also failed to eliminate the penalty.

`gpu_activity.cpp` and `build_activity.cmd OUTPUT_EXE` provide a research-only
D3D11 activity helper. No HIP compiler is required. It accepts the pinned ROCm
runtime directory and uses the exported `hipDeviceGetLuid` API to select the
same Windows adapter as HIP device zero, additionally requiring hardware vendor
1002/device 15bf. See AMD's [LUID API documentation](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/hip_runtime_api/modules/initialization_and_version.html).
Matching only PCI IDs was ambiguous on this host; that first check rejected use.

Each dispatch runs one group of 32 threads with a finite integer loop. Only one
dispatch is queued at a time, with a completion timeout. Activity windows accept
1–100 ms; completion includes a final dispatch and exact readback comparison
against a host integer reference, so elapsed time can slightly exceed the
requested window. `stop` joins early and `quit` cancels/joins before exit.
The helper client owns only its subprocess. It uses no model weights or CPU
model inference and changes no system power policy.

The coexistence tool's `--suite activity` requires `--activity-helper PATH`,
supports bounded `--activity-ms` values and optionally joins with
`--activity-stop-before-verify`. Its first attempt was blocked by foreign CPU
work; the subsequent two-sweep comparison passes all state checks. The current
opt-in runtime stops and joins before target verification; all helpers are idle
between requests. Early stopping reduces helper work, with some latency cost
in the NPU-interleaved comparison. Its energy benefit remains unmeasured. See the
[activity results](../../docs/BENCHMARKS.md#bounded-gpu-activity-during-draft-gaps).
