# Ignition todo list

This is the implementation backlog for Ignition. [README](README.md) defines the
supported Alpha scope; [DESIGN](DESIGN.md) keeps the detailed source contract and
research plan. A checkbox closes only with the evidence described beside it.

## Alpha 0.1.0a1

- [x] Independent folded-ResNet no-CLE calibration/emission with no Quark or torch
  imports; exact parity against the fresh same-listing oracle.
- [x] Versioned `python -m quant` interface, static inspection, compatibility entry
  point and ONNX/sidecar producer metadata.
- [x] Explicit unsupported-graph errors, artifact overwrite refusal and calibration
  disk guard; retain the repository's batch/opset/cache invariants.
- [x] Controlled EP probes with output comparisons, unoptimized CPU references,
  archived placement reports, input/model hash binding and child resource limits.
- [x] Full evaluation of the versioned artifact and both-environment release checks;
  evidence is in [Alpha validation](../docs/BENCHMARKS.md#ignition-alpha-release-validation).

The release branch is `ignition-alpha`; release verification and publication are
recorded in the handoff and Git history rather than treated as future features.

## Next milestone: default-preset ResNet parity

- [ ] Implement CLE from the audited source rules, including pair order, bias
  handling, thresholds and iteration limits. Match the oracle's pattern count and
  transformed weights before comparing calibration positions.
- [ ] Quantize with CLE on the same image listing as a fresh default XINT8 oracle.
  Require graph/parameter comparison, full labeled CPU/NPU accuracy, fresh EP reports
  and paired latency. Do not compare against an old session's latency.
- [x] Exercise refinement cases that move weight/bias positions. Resolve the audited
  change-tracking/raw-data hazards and record whether stored integers must change.
  Evidence: the [refinement probe](../docs/BENCHMARKS.md#ignition-refinement-rules-under-perturbation)
  (`results/quant/refine_probe_resnet50_quark_nocle_c64.log` and its `_wide` run): 20
  directed and 800 random perturbations of the oracle's positions give identical final
  tables from Quark and Ignition; stored integers are unchanged by both, so parity means
  no re-rounding; the raw-data write is a measured no-op in Quark and the Mul hazard is
  unreachable here. Rules and bridges absent from this graph stay untested.

## Broaden model support after parity gates

- [ ] Add YOLO preparation and handlers: Split→Slice, supported pooling rewrites,
  SiLU/HardSigmoid simulation, Concat sharing/alignment and graph-specific pruning.
  Gate on exact same-listing head-cut YOLOv8n comparison and full COCO evaluation.
- [ ] Add AdaRound as an isolated optional torch module. Match the audited schedule,
  layer selection, update behavior and full-set accuracy; record peak memory beside
  the Quark oracle. Keep all other core imports torch-free.
- [ ] Connect MODNet's calibration source to the shared inference preprocessing,
  then re-evaluate matte quality against the documented mismatched-preprocessing
  baseline. Do not claim this fixes the quality gap before measuring it.
- [ ] Consider histogram calibration only with measured error/accuracy and memory
  tradeoffs against the exact-sample store; label approximation explicitly.

## Compiler research, separate from production options

- [ ] Minimize the repeated-per-channel-parameter compiler memory growth under the
  bounded runner. Obtain a placement and numerical verdict before trying differing
  channel grids or proposing MobileViT recovery.
- [ ] Isolate product-scale INT32-bias NPU numerical failure. Separately isolate the
  optimizer-dependent CPU discrepancy for dtype-only INT32 bias; neither mechanism
  is established by the current output comparisons.
- [ ] Probe retained Conv/Relu QDQ, Concat alignment and HardSigmoid correction
  independently on suitable baselines. Run numerical checks for every placement result.
- [ ] Investigate opset/IR alternatives only as separate experiments with fresh
  artifacts; alpha's opset 17 / IR 8 contract stays pinned.

## Usability and distribution

- [ ] Expose a reusable comparison report with explicit acceptance criteria;
  distinguish parameter parity, placement, numerical error and dataset accuracy.
- [ ] Make a clean-checkout reproduction checklist for each additional model family;
  document source artifacts and licenses without redistributing model/data assets.
- [ ] Decide an installable package/distribution name when packaging is warranted.
  Keep the project name Ignition and internal package `quant` for this alpha.
- [ ] Add recovery guidance for hard-terminated calibration spools and partial output
  pairs without deleting unrelated shared artifacts.
