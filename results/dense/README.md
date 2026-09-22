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
- `acc_*` is the only LABELLED evidence here, from `benchmarks/dense_accuracy.py`: MODNet-Cut's alpha
  thresholded at 0.5 against every annotated COCO person instance, on all 2,693 val2017 images
  containing a person. Everything else in this directory is agreement. BiSeNetV2 has no `acc_*` log
  on purpose - its FP32 predicts 0.00-0.30 % person on images that are 84-96 % person, so a
  Cityscapes model on COCO photographs is out of domain and scoring it here would measure noise.

  Seven arms, same images, same threshold, same harness:

  | arm | log | person IoU | pixel accuracy |
  |---|---|---:|---:|
  | FP32 | `acc_modnet_cut_fp32_20260921.log` | 0.5030 | 90.22 % |
  | bf16 | `acc_modnet_cut_bf16_20260921.log` | 0.5029 | 90.26 % |
  | AMD, CLE + AdaRound | `acc_modnet_cut_cle_adaround_amd_20260922.log` | 0.4468 | 87.57 % |
  | Ignition, CLE + AdaRound | `acc_modnet_cut_cle_adaround_ignite_20260922.log` | 0.4272 | 88.02 % |
  | CPU, CLE + AdaRound | `acc_modnet_cut_cle_adaround_20260921.log` | 0.4272 | 88.02 % |
  | int8, plain XINT8 | `acc_modnet_cut_cpu_20260921.log`, `..._ignite_...` | 0.1609 | 83.69 % |
  | AMD, plain XINT8 | `acc_modnet_cut_amd_20260921.log` | 0.1700 | 82.08 % |

  The two 2026-09-21 additions are the bf16 and CLE+AdaRound arms, and together they correct a
  claim this directory used to support. "int8 costs this model 68 %" is true only of the plain
  XINT8 model: a better recipe reaches 84.9 % of FP32 without touching the hardware, so most of
  that collapse was the recipe. bf16 reaches 99.97 % and is the only arm here that keeps the whole
  model. The bf16 arm is a CPU simulation - `tools/onnx_bf16_cast.py` rounds weights and bias once
  and casts every convolution's activation input and output - not a device run, and it makes no
  timing claim.

  **The two 2026-09-22 arms are the control the CLE+AdaRound row needed, and they invert its
  reading.** The 0.4272 row is Ignition's recipe and the 0.1700 row is AMD's, so comparing them
  compares two models. On the *same* CLE+AdaRound ONNX, AMD scores 0.4468 and Ignition 0.4272:
  AMD is ahead by 4.6 % relative, the recipe is worth about 2.6x to both stacks, and there is no
  runtime accuracy win here. Ignition's arm equals its CPU reference to all sixteen digits
  (0.427203295160268), so the loss is not an exactness failure. The AMD arm has its own compile
  cache key: `modnet_cache_key` matches on "cut", so all four MODNet-Cut variants resolve to
  `modnetcutcachekey` and an unguarded run would have been served the previous model's compile.
  Its log embeds the EP report placing 502 of 507 nodes on the NPU, because a whole-graph CPU
  fallback would have scored the ONNX exactly and read as an NPU result.
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
