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

## Default-preset ResNet parity (closed 2026-09-08)

- [x] Implement CLE from the audited source rules, including pair order, bias
  handling, thresholds and iteration limits. Match the oracle's pattern count and
  transformed weights before comparing calibration positions. Evidence: `quant/cle.py`
  and the [float-level probe](../results/quant/cle_probe_resnet50_fp32.log): 33 patterns
  in the same order, byte-identical equalized initializers.
- [x] Quantize with CLE on the same image listing as a fresh default XINT8 oracle.
  Require graph/parameter comparison, full labeled CPU/NPU accuracy, fresh EP reports
  and paired latency. Evidence: [CLE parity](../docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset): empty position
  delta, 108/108 int8 exact, 72.80% CPU and 72.10% NPU top-1 for both, 393/395 placed,
  5.22 ms paired in one sitting; the oracle equals the repo's original resnet50 XINT8
  artifact. Depthwise/Gemm/Clip CLE paths raise and stay unmeasured.
- [x] Exercise refinement cases that move weight/bias positions. Resolve the audited
  change-tracking/raw-data hazards and record whether stored integers must change.
  Evidence: the [refinement probe](../docs/BENCHMARKS.md#ignition-refinement-rules-under-perturbation)
  (`results/quant/refine_probe_resnet50_quark_nocle_c64.log` and its `_wide` run): 20
  directed and 800 random perturbations of the oracle's positions give identical final
  tables from Quark and Ignition; stored integers are unchanged by both, so parity means
  no re-rounding; the raw-data write is a measured no-op in Quark and the Mul hazard is
  unreachable here. Rules and bridges absent from this graph stay untested.

## Broaden model support after parity gates

- [x] Add YOLO preparation and handlers: Split→Slice, supported pooling rewrites,
  SiLU/HardSigmoid simulation, Concat sharing/alignment and graph-specific pruning.
  Gate on exact same-listing head-cut YOLOv8n comparison and full COCO evaluation.
  Evidence (yolov8n-cut, closed 2026-09-08): `quant/passes.py`, the generalized
  `quant/qdq.py` marking and the full `quant/refine.py` transcription;
  [YOLO preparation parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity):
  the prepared graph equals Quark's pre-calibration graph and the committed artifact
  replays exactly (`tools/quant_prepare_probe.py`), a fresh same-listing c64 oracle
  matches position for position with 126/126 int8 byte-identical, and both files read
  27.43 CPU / 27.03 NPU mAP@50-95 with 922/929 nodes placed and byte-identical
  detections. Pooling rewrites and Conv→Relu pruning had no instance on this graph, and
  the HardSigmoid and swish shift bounds never fired.
- [x] Add AdaRound as an isolated optional torch module. Match the audited schedule,
  layer selection, update behavior and full-set accuracy; record peak memory beside
  the Quark oracle. Keep all other core imports torch-free. Evidence (ResNet, closed
  2026-09-08): `quant/adaround.py` and `python -m quant adaround`;
  [AdaRound parity](../docs/BENCHMARKS.md#ignition-adaround-parity): byte-identical to a
  fresh same-listing `XINT8_ADAROUND` oracle on Desktop 2 (108/108 int8 exact, 702
  log lines identical), 79.40% CPU top-1 for both, peak working set 3,055,075,328
  bytes against Quark's 3,582,218,240. yolov8n-cut (closed 2026-09-08):
  [YOLO AdaRound parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-adaround-parity):
  the layers are walked in the vendor's topological order of the float model
  (`Graph.vendor_order`, the order Quark's loop takes; Ignition's emitted file order
  diverges at the 39th conv), and the result is 126/126 int8 byte-identical to a fresh
  same-listing oracle with all 819 per-layer log lines equal, peak working set
  5,742,055,424 bytes against Quark's 5,393,625,088; both files read 32.21 CPU /
  32.04 NPU mAP@50-95 with 922/929 placed and byte-identical detections, 5.01 NPU
  points above the same-listing plain-XINT8 pair.
- [ ] Add GPU / ROCm acceleration for AdaRound (`quant/adaround.py`): enable
  `OptimDevice = "cuda"` for PyTorch training loop on Desktop 1 (RX 7900 XTX 24 GB)
  while keeping ORT activation caching on CPU. Validate paired convergence against
  Quark GPU oracle (`--device cuda`). See design note in [DESIGN.md §4.2](DESIGN.md#note-for-implementation-gpu--rocm-acceleration-for-adaround).
  **Plumbing landed 2026-09-09 (Desktop 2), the acceleration has not.** `OptimDevice` is
  wired through `--device`, ORT extraction stays on CPU (`InferDevice` still refuses to
  move), and the CPU path is shown bit-for-bit unchanged
  ([AdaRound gains an OptimDevice](../docs/BENCHMARKS.md#ignition-adaround-gains-an-optimdevice-and-the-cpu-path-does-not-move-2026-09-09-desktop-2)).
  What remains is entirely on Desktop 1 and needs a decision first: **no torch build on
  any machine here can reach a GPU** (`resnet_env` is `torch 2.4.1+cpu`; official ROCm
  wheels are Linux-only, so the Windows candidates are AMD's ROCm preview or
  `torch-directml`). Pick and install one, then run
  `python -m quant adaround ... --device cuda --accept-non-parity` and time it against
  the CPU wall on a wide model. Note the parity target changes with the device: the
  transcription is byte-identical to the **CPU** `XINT8_ADAROUND` oracle, and a GPU run
  can only be gated against a **GPU** oracle produced on the same box and runtime — the
  code refuses to let a GPU run be quoted as matching the CPU one
  (`AllowNonParityDevice`, `byte_parity_path` in the sidecar).
- [x] Connect MODNet's calibration source to the shared inference preprocessing,
  then re-evaluate matte quality against the documented mismatched-preprocessing
  baseline. Do not claim this fixes the quality gap before measuring it.
  Closed 2026-09-09. `quant/sources.py`'s `ModnetSource` reads through
  `npu.modnet.preprocess`, so the matting pipeline's calibration and its inference are one
  function rather than two copies, and the family is implemented (`passes.simplify`
  delegating to onnxslim as the vendor does, `Clip` marking and pruning, the vendor's full
  avgpool table, and a Q/DQ marking pass that follows the vendor's visit order). Evidence:
  [MODNet preparation parity](../docs/BENCHMARKS.md#ignition-modnet-preparation-and-replay-parity)
  (Quark's whole pre-process diffs empty against Ignition's; the committed
  `modnet_cut_xint8_calibfix.onnx` re-emits 140/140 int8 byte-identical from its own
  positions) and
  [MODNet independent calibration](../docs/BENCHMARKS.md#ignition-modnet-independent-calibration-and-paired-matte-evaluation)
  (fresh same-listing oracle, empty position delta, 140/140 int8 byte-identical, both files
  0.17122 MAD on CPU and 0.19021 on NPU over the 50 validation images at 502/507 placed).
  The re-evaluation half of this item was answered separately on the Quark side by the
  2026-09-08 OpenCV rerun; what is now also measured is Ignition reproducing it. The quality
  numbers here are **not** comparable with that rerun's 0.18629, whose listing was never
  recorded — that comparison needs a listing sweep, which is unrun. AdaRound is not wired
  for this family (the `adaround` command refuses it explicitly) and the Zero-Concat
  variant is untouched.
- [x] Wire AdaRound for MODNet. It could not reuse the ResNet/YOLO path unchanged: the
  layer walk runs on the **simplified** float graph, because `passes.simplify` reorders this
  family's node list. Closed 2026-09-09. `cli.py`'s adaround branch calls `simplify_for`
  before CLE and `prepare`, and `adaround.py` gained the vendor's `Clip` handling — a
  three-input Clip whose bounds are initializers of the quantized model becomes
  `torch.clamp`, not the `ActivationMapping` `nn.ReLU6` that names the same curve with a
  different gradient at the bounds; any other Clip shape is refused rather than
  approximated. MODNet is the first family here where a finetuned layer carries an
  activation at all (35 Clip, 17 Relu, 19 none over 71 Conv layers). Evidence:
  [MODNet AdaRound parity](../docs/BENCHMARKS.md#ignition-modnet-adaround-parity) — empty
  position delta against a fresh same-listing `XINT8_ADAROUND` oracle, 140/140 int8
  byte-identical, all 911 per-layer log lines equal, and 0.09361 CPU / 0.10072 NPU matte
  error for both files at 502/507 placed, which is a 45 and 47 percent cut against the same
  listing quantized plain. Peak working set 22,299,271,168 bytes, beyond the 16 GB laptop.
  The Zero-Concat variant and the listing sweep stay open.
- [ ] Consider histogram calibration only with measured error/accuracy and memory
  tradeoffs against the exact-sample store; label approximation explicitly.
- [x] Cut AdaRound's peak memory. Closed 2026-09-09 for ResNet50: the loop held roughly six
  copies of each layer's activation set, three per-image ORT lists and three stacked copies, of
  which two are ever read. Releasing the dead ones and collecting into one preallocated buffer
  instead of `np.array(list)` takes the peak from 3,055,075,328 to **2,543,427,584** bytes, −16.7%,
  with the emitted file bit-identical and the same 8,954,279 weights moved.
  [Evidence](../docs/BENCHMARKS.md#adaround-peak-memory-six-copies-of-the-activation-set-down-to-two-2026-09-09-desktop-2).
- [ ] Re-measure that on MODNet-Cut, which is the family where Ignition is actually worse than
  Quark (22,299,271,168 against 15,771,942,912). It was not re-run because free memory was 16.5 GB
  against a 22.3 GB peak, and peak working set reads low under memory pressure -- the measurement
  needs a quiet box or it will flatter itself.
- [x] Add a CLE stability guard. Closed 2026-09-09: `cross_layer_equalize(max_scale_log2=...)`,
  `python -m quant quantize --cle-guard BITS`, off by default. **The signal is not the
  post-transform weight range**, as this item originally assumed -- CLE narrows the weight
  positions even on the models it destroys. It is the per-channel scale, which multiplies the
  activation between the pair and is never rescaled: worst pair 2.52 bits on ResNet50 and 1.81 on
  MODNet against 42.20 on RegNetX-002 and 72.62 on ResNeXt-50 (corrected 2026-09-09: those two
  are depthwise triples, not pairs). At 4 bits the guard is
  byte-identical on both supported families, and firing it at 2 bits changes ResNet50's output as
  a positive control. [Spans, threshold and parity](../docs/BENCHMARKS.md#the-cle-stability-guard-what-to-threshold-on-and-what-it-costs-2026-09-09-desktop-2).
- [x] Implement the depthwise CLE triple path. Closed 2026-09-09: `equalize_triple` transcribes
  `_cle_set_with_depthwise_layers`, and `find_pairs` records 4-tuples instead of raising. Matching
  the vendor required reproducing its `check_conv_layers_support` early-break, where the failure
  `break` leaves the attribute loop rather than the node loop so the verdict is the last node's --
  worth 13 of RegNetX-002's 27 patterns. Pattern counts and every float initializer agree with
  Quark byte for byte on ResNet50, RegNetX-002 and ResNeXt-50. The guard then narrowed to triples
  only, since ten of ResNeXt-50's destructive triples sit inside ResNet50's beneficial pair range;
  at 2 bits it recovers 66.20% and 68.90% with both pair families byte-identical.
  [Evidence](../docs/BENCHMARKS.md#depthwise-cle-triples-implemented-and-the-guard-narrowed-to-them-2026-09-09-desktop-2).
- [ ] The guard's threshold rests on two collapse models and two healthy ones. A family whose
  depthwise triples are genuinely benign would be disarmed by `--cle-guard 2` and none has been
  found; widening the model set is what would justify a default rather than an opt-in.
- [x] RegNetX-002 is a fourth family reproduced with no code change. Its export is
  Conv/Relu/Add/GlobalAveragePool/Flatten/Gemm at 224, routes to `folded_resnet`, and Ignition's
  no-CLE artifact is byte-identical to a fresh Quark no-CLE oracle: 90/90 int8 initializers,
  empty node delta, `INT8_EXACT True`, `GRAPH_DIFF_PASS True`. 66.20% CPU / 66.40% NPU top-1 at
  324/326 placed. Closed 2026-09-09. DenseNet-121 is *not* reachable this way -- its export
  carries unfolded `BatchNormalization`, which `qdq.quantizable_tensors` rejects.

## Compiler research, separate from production options

- [x] Obtain the per-channel placement verdict. Closed 2026-09-09, and the answer is no: the
  EP places **0 of N** nodes for any weight scale with more than one element, at 1, 2, 4 and 8
  convolutions, sending the whole graph to CPU. A single-output-channel control places normally,
  so the `axis` attribute is fine and it is the vector length that is refused; the fixture values
  are identical repeats, so it is not about differing channel grids. Asked with one convolution
  instead of 54 the compile costs seconds, which is why the earlier attempt's memory growth never
  needed minimizing -- it was on the way to a rejection.
  [Verdict and controls](../docs/BENCHMARKS.md#per-channel-weight-scales-are-rejected-outright-2026-09-09-desktop-2). Differing channel grids and
  MobileViT recovery via per-channel are closed with it.
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
