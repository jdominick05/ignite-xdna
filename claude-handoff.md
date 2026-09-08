# Ignition: handoff to Claude

## Name and scope

The user named the quantizer **Ignition**. Use that name in documentation and CLI
descriptions. Keep `quant/` as the internal Python package; the name does not require
an API/package rename. New emissions set ONNX `producer_name` to `Ignition`.
Measured older models retain `owned.xint8` metadata so their hashes remain evidence.

The user asked for the most beneficial next task after the proof of concept. This
session completed the first controlled EP acceptance study: isolate the properties
stock A8W8 changes together, then check numerical execution as well as placement.
It did not broaden the production emitter beyond folded ResNet without CLE.

This is a coordination snapshot. Measurements and caveats live in
[BENCHMARKS](docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance), engineering
rules in [DECISIONS](docs/DECISIONS.md#locked-decisions-do-not-reopen), and implementation
scope in [quant/DESIGN.md](quant/DESIGN.md). The old local `HANDOFF_CODEX.md` describes
the pre-code starting point and is now historical.

## Branch and workspace

Work is on **`quant-acceptance-map`**, based on `a326177` (independent no-CLE ResNet
calibration/emission), following scaffold commit `b7001c5`. On this machine its
worktree is `scratch/quant-worktree` beneath the primary repository. Find the latest
commit with `git log -1 quant-acceptance-map`; no push was performed.

The primary worktree has another session's dirty `pipeline-midas` work. Do not
reset, clean, overwrite or switch that worktree. `models/` and `data/` in the Ignition
worktree are junctions to shared ignored artifacts. Compilation uses the existing
worktree-local `modelcachekey`, cleared before each changed model. Existing primary
`AGENTS.md` was deliberately left unchanged, as the user requested. The local
Ignition-worktree `CLAUDE.md` and tracked `CONTRIBUTING.md` import gates were updated.

## What the earlier producer work established

- The [source/model audit](results/quant/notes_xint8_dialect.log) corrected the draft
  contract: refinement aligns connected Concat/pool positions to their minimum;
  Conv/Gemm own the cut/bias rules; bias starts with its own MinMSE position; stored
  signed weights/bias clip to `[-127,127]`; NumPy rounds ties to even. Producer rules
  read from Quark are not automatically compiler requirements.
- `quant.graph` and `quant.pow2` provide checked graph operations, fingerprints and
  fixed-position arithmetic. `sources`, `calib`, `qdq`, `passes`, `refine`, `verify`
  and `quantize` implement the folded ResNet path. Calibration stores exact float16
  samples, chooses scales with float32 MinMSE, and uses shared preprocessing.
- Independent calibration and float-weight re-emission match a fresh no-CLE Quark
  reference: graph connections, scales, zero points and all integer parameters are
  exact. The independent process blocks Quark and torch imports. See
  [graph diff](results/quant/diff_resnet50_own_nocle_c64.log) and the
  [producer method/full-set evidence](docs/BENCHMARKS.md#owned-resnet50-no-cle-re-emission-and-independent-calibration).
- The emitter remains deliberately narrow: Conv/Relu/Add/MaxPool/GAP/Flatten/Gemm,
  measured GAP shape, no CLE, no YOLO preparation or AdaRound. Design sketches in
  `quant/DESIGN.md` include future APIs; do not assume every signature exists.

## Acceptance findings

All results below concern the same no-CLE ResNet on Desktop 2, Ryzen 7 8700G,
Phoenix XDNA1, Ryzen AI 1.7.1, ORT `1.23.3.dev20260320`, opset 17 / IR 8,
static batch 1. The first matrix uses **32 diagnostic images, not accuracy scores**;
`c64` denotes 64 calibration images. The
[corrected summary](results/quant/probe_resnet50_acceptance_summary_v2.log) and
[archived EP diagnostic](results/quant/diag_ignition_acceptance_archived.log) bind the
following conclusions to the raw logs linked in BENCHMARKS.

| Change | Finding |
|---|---|
| Q/DQ domain alone → `com.microsoft` | CPU output unchanged; full fallback, 0/395 NPU nodes. Domain change alone suffices here; it need not be the sole problem in all A8W8 graphs. |
| UINT8/zp128 activations → INT8/zp0 | Same 393/395 placement; CPU and NPU outputs exact against their respective baselines, confirmed on all 1,000 evaluation images. |
| Strip model metadata / set producer to `Ignition` | Same placement and exact baseline NPU outputs. Vendor producer metadata is not required for this artifact. |
| Activation scales ×1.01 except final output | Still 393/395 NPU nodes, but NPU-vs-unoptimized-CPU RMSE 4.230485 and zero argmax agreement on the slice. |
| Weight scales ×1.01 | Partial fallback, 276/395 NPU nodes; RMSE 11.371655 and zero argmax agreement. These are controlled scale perturbations, not MinMax recalibration. |
| Bias dtype alone → INT32, same scale | Unoptimized CPU and NPU preserve their baselines exactly. Optimized CPU changes incorrectly relative to that reference; see below. |
| INT32 bias at input×weight scale | Both CPU modes preserve baseline outputs, but NPU RMSE is 8.025162 with zero argmax agreement despite 393/395 placement. |
| Remove GAP correction Constant/Mul | 392/394 placement and exact baseline NPU output; CPU approximation changes slightly. Keep the factor for producer parity. |
| Repeat scalar weight scales/zps per channel, set DQ axis | CPU output unchanged; compiler construction kept growing memory and was stopped. No placement or NPU numerical verdict. This did not test differing per-channel scale values. |

The full-set baseline and signed variant both score **CPU 62.00%/79.80% top-1/top-5,
NPU 59.90%/79.10%**, with every NPU logit element identical between the pair.
Their mean NPU latencies are 5.404 and 5.438 ms; do not claim a speedup from that
difference. These are no-CLE, small-calibration models, not the existing CLE/AdaRound
headline models. See the [baseline](results/quant/probe_resnet50_accept_full1000_baseline.log)
and [signed run](results/quant/probe_resnet50_accept_full1000_act_int8_zp0.log).

## Pitfalls and evidence corrections

**NPU placement is necessary, not sufficient.** The scale and product-scale bias
mutations show apparently normal NPU execution with bad outputs. Keep the current
XINT8 production recipe; experimental variants do not become supported options merely
because they compile.

**Optimized CPU is not always the reference.** For dtype-only INT32 bias, optimized
CPU differs from `ORT_DISABLE_ALL` by RMSE 5.035656 on the 32-image slice, while the
unoptimized computation and NPU preserve their respective baselines. The original
probe's `npu_vs_cpu` field compares against that misleading optimized result. Use the
[unoptimized audit v2](results/quant/probe_resnet50_accept_c64_cpu_reference_audit_v2.log).
The responsible ORT rewrite has not been isolated; do not invent a mechanism.

**The first combined summary is superseded.** It matched CPU audits by model hash
alone and reused slice RMSE 0.756329 in the full-set rows. The full-set raw logs always
reported 0.810127 correctly. Summary v2 also matches input hashes; it does not pretend
the slice's unoptimized audit covers all 1,000 images. Preserve the old logs and their
correction trail.

**Per-channel is unresolved.** The original child was manually stopped after 273.731
seconds at 11,973,251,072 bytes working set; session construction never finished.
The [stop witness](results/quant/probe_resnet50_accept_c64_weights_per_channel_resource_stop.log)
records the verified process and clean hardware-context report. Its old sidecar stays
`npu_build_started`; the summary attaches the stop witness. Do not repeat unbounded
compilation or call this CPU fallback. The bounded launcher now limits child RSS to
8 GiB and wall time to 300 seconds (600 for full evaluation).

## Artifacts and tools

The immutable baseline is `models/resnet50_own_nocle_c64.onnx`, SHA256
`c1945d2dce29e6afeaed16ef8c2bcc5689044f5210543440f84fb258c29e0ea5`.
Variants are `models/resnet50_accept_{c64,full1000}_<mutation>.onnx` with
`.probe.json`, `.outputs.npz` and completed `.ep.json` evidence. These are ignored;
tracked UTF-8 logs under `results/quant/` carry the measurements. The xclbin and
input hashes, exact timing brackets and pre-run context caveats are in BENCHMARKS.

- `quant/probe.py`: pure checked graph mutations and finite output comparison.
- `npu/ep_report.py`: shared placement parser, also used by `tools/diag_ep.py`.
- `tools/quant_probe.py`: one variant; optimized/unoptimized CPU references, fresh
  NPU session, archived report, output differences and optional full-set accuracy.
- `tools/quant_probe_bounded.py`: parent resource limits using existing `psutil`.
- `tools/quant_probe_cpu_audit.py`: audit archived results against unoptimized CPU.
- `tools/quant_probe_summary.py`: validate artifacts and aggregate matching cohorts.
- `scripts/quant-probe.sh`: activate the NPU env, preserve per-variant logs and run
  isolated bounded children. Omit `--mutation` only when intending the entire matrix.

From Git Bash, choose new names; existing model/log paths refuse overwrites:

```bash
./scripts/quant-probe.sh --tag resnet50_accept_next --mutation baseline
./scripts/quant-probe.sh --tag resnet50_accept_next --mutation act_int8_zp0
```

Add `--full-eval` to both with a separate tag for full-set accuracy. Use
`scripts/quant-own.sh` for fresh independent quantization; see the producer evidence
section for reference generation and comparison commands.

## Verification and next work

Both conda environments passed syntax, shell parsing and shared import gates without
Quark/torch imports. The integrated CPU-only check reproduces the reference discrepancy;
both time/RSS stop paths passed with CPU-only children. Archived EP reports and the
unchanged baseline hash were checked. Logs are indexed in [results/README.md](results/README.md).
The original NPU matrix predates the automatic unoptimized-reference addition; its
separate audit supplies that evidence. No new hardware behavior is claimed from the
later CPU-only implementation checks.

Recommended next producer milestone: **CLE parity on ResNet**, with fresh same-listing
Quark comparison and full-set CPU/NPU evaluation. This addresses a real gap in the
current no-CLE producer. YOLO preparation/refinement and AdaRound follow their own
parity gates. For compiler research, minimize the per-channel construction failure or
the INT32 product-scale numerical failure before trying them on MobileViT. Opset
changes, Concat alignment, unpruned Conv/Relu QDQ and HardSigmoid probes remain unrun.

Keep `quant/` Quark-free, `npu/` independent of `quant/`, preprocessing shared, cache
keys unchanged and fresh on model changes. Read the local invariants before running
hardware; use activated conda environments, coordinate shared NPU access, preserve
old evidence, and fold each new finding into BENCHMARKS plus affected decision/open
question documents. Do not push without the user's explicit authorization.
