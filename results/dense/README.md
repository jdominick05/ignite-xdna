# Dense hybrid tasks on Desktop 2

Evidence for [segmentation and matting](../../docs/BENCHMARKS.md#dense-segmentation-and-matting-against-amds-stack-2026-09-21-desktop-2).
Every validation set here is local and unlabeled. Agreement is not task accuracy.

There are **two sittings in this directory, and the 2026-09-21 one is the evidence**. The 2026-09-19
run is kept beside it because it is where the work was done, not because its numbers are the ones
reported.

## Which sitting backs which number

- **`*_20260921*` is the current evidence.** It was taken on the code that is committed: the boundary
  layer in `runtime/graph_session.py` had to be authored before any of this could import, so the
  2026-09-19 containers were built by a compiler that exists on no branch and could not be rebuilt
  until it was. The containers were rebuilt (`build/{family}_dense_20260921.ignite`) and the whole
  sitting re-run.
- **`*_20260919*` is the original run, superseded and kept.** Its numbers are not wrong; they simply
  belong to code that was never committed. Three things moved between the two sittings: the engine
  itself (`graph_ir`, `engine_schedule`, `engine_compile`, `graph_session`, `session`, `graph_reference`
  all differ), `npu/bisenetv2.py` changed on 2026-09-20 in `00c287e`, and the boundary layer was
  reconstructed. `compiler/dense_regions.py`, `runtime/dense_session.py` and
  `benchmarks/dense_compare.py` are byte-identical across both, which is why the partition reproduced
  exactly: 47 layers and 27 segments for BiSeNetV2, 53 and 47 for MODNet-Cut, same host/NPU split and
  same workspace sizes.

The 2026-09-21 sitting measures the same inputs: the pins files record identical model, FP32 reference
and image hashes, and `inputs_*_parity_20260921_*` checks the preprocessed tensors agree in both ORT
environments.

## Naming

- `*_pins_*.json` pins each XINT8 model, its FP32 reference, the canonical transform source and every
  validation image. A run refuses to start if any pinned artifact has changed.
- `pin_*` and `compile_*` record the pinning and the container builds.
- `inputs_*` pins the preprocessed tensor hashes in both ORT environments.
- `baseline_*` saves unoptimized CPU, optimized CPU, FP32 and backend outputs separately, and is where
  the agreement figures come from. AMD jobs compile fresh and embed the complete placement report.
- `verify_*` checks every extracted CPU region, every integer convolution, every silicon region and the
  complete output on all pinned images, through the public `InferenceSession.from_file` factory.
  Full-frame host-call counts must equal the manifest.
- `bench_*` is the alternating timed sweep: CPU in the vendor environment, CPU in the engine
  environment (`cpu_current`), AMD, then Ignition, twice per model, 50 warm-up and 500 timed frames on
  the same pinned first image. **Only `timing_eligible: true` qualifies as a timing result**, and every
  `bench_*` run here carries it.
- `*_summary_*` are the aggregated reports `dense_report.py` produces from the logs above.
- `*_initial*` and `*_opt*` (2026-09-19 only) are that sitting's own before/after for pruning internal
  CPU outputs. The 2026-09-21 containers correspond to the `_opt` form.

All device work runs through `scripts/research-lowlevel.sh --npu`, which captures NPU ownership and
idle witnesses before and after the child; compiler children go through `scripts/research-iron.sh`.
`benchmarks/dense_sitting.py` serializes the `inputs`, `baseline`, `bench` and `verify` phases;
re-pinning and rebuilding go through `benchmarks/dense_compare.py` directly. Raw outputs and containers
live under ignored `build/dense_validation/` and `build/`; the logs carry their hashes.

## What was scrubbed, and what is missing

Logs here are scrubbed before they land, the same way every other log in this repository replaces the
local profile directory with `C:\Users\<user>`. Two further substitutions were applied, and no number,
hash or verdict was touched:

- The checkout directory is folded back to the repository root. The 2026-09-21 runs were executed from
  a git worktree; it is the same commit with junctions to the same `models/` and `data/`.
- `HOST_LOAD_TOP`, which names the busiest process on the host during a run, had three developer tool
  names replaced with `<tool>`. What that line is evidence for - how loaded the host was, in cores and
  working set - is unchanged, and every other process name is intact.

**Four 2026-09-19 gate captures are deliberately not tracked**: `baseline_regression_failures`,
`current_regression_isolation`, `root_tests_hardware` and `root_tests_hardware_pack`. Each contains a
UTF-16LE section inside an otherwise UTF-8 file, roughly 44,000 NUL bytes, from a PowerShell redirect.
A log like that silently matches nothing under `grep` or `rg`, which is worse than an absent one. They
recorded regression gates rather than measurements, and those gates are re-runnable; they remain on
disk, untracked, on Desktop 2.

Failures and superseded attempts otherwise remain in this directory, including the 2026-09-19
runtime-only-environment compile attempt and its failed harness run.
