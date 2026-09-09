# Ignition — an owned XINT8 quantizer for the X1 backend

The runnable release is **Alpha 0.1.0a1**: [supported scope and commands](README.md),
[implementation todo list](TODO.md), and [release validation](../docs/BENCHMARKS.md#ignition-alpha-release-validation).
The broader signatures and roadmap below remain a design, not an Alpha API promise.

**Status: ResNet emission and independent MinMSE calibration match fresh Quark
references both without CLE and with the default preset's CLE; the transcribed
refinement and equalization rules are probed against Quark's on identical inputs; the
first controlled ResNet acceptance study is measured; AdaRound is transcribed and
byte-identical to fresh `XINT8_ADAROUND` oracles on the same machine for both
families, walking the layers in the vendor's topological order of the float model;
head-cut YOLOv8n preparation (Split→Slice, the SiLU chain, Concat/Slice/pool alignment,
the HardSigmoid and swish shifts) is transcribed and position-for-position,
integer-for-integer equal to a fresh same-listing oracle. The broader acceptance map
remains open.** YOLO evidence is in the [YOLO preparation parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity)
and [YOLO AdaRound parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-adaround-parity) sections; ResNet AdaRound evidence is in the
[AdaRound parity section](../docs/BENCHMARKS.md#ignition-adaround-parity); CLE evidence is in the
[CLE parity section](../docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset). The audit is in
[`notes_xint8_dialect.log`](../results/quant/notes_xint8_dialect.log); remaining unknowns
are explicit in §2.6. Re-emission, exact graph/parameter comparison, full-set accuracy
and paired NPU evidence are in the [ResNet results](../docs/BENCHMARKS.md#owned-resnet50-no-cle-re-emission-and-independent-calibration).
Written 2026-09-08 on Desktop 2 from the Quark-produced
models in `models/`, the Quark 0.11rc1 source in `resnet_env`, and the VitisAI EP's
`vaip_config.json` in Ryzen AI 1.7.1. Every contract fact below carries a provenance tag;
a fact without one is a guess and is marked as such.

The first scaffold checks pass in both conda envs; source/model inspection and their
limitations are folded into
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md#an-owned-xint8-quantizer-scale-exact-reproduction-then-the-eps-acceptance-map).

Ignition is the quantizer's project name; `quant/` remains its internal Python package.
Earlier measured artifacts retain their original `owned.xint8` producer metadata;
newly emitted artifacts identify their producer as `Ignition`.

The question this directory answers is narrower than "write a quantizer": **can a
quantizer we own emit the dialect the VitisAI EP's X1 backend accepts — the same graph
Quark writes, every scale and zero point byte-identical, every weight within one LSB —
and then do the things Quark cannot be asked to do?** The EP, the xclbin and the DPU stay
exactly as they are. The interface being reproduced is the ONNX file.

Read [`docs/DECISIONS.md`](../docs/DECISIONS.md#locked-decisions-do-not-reopen) #3 and
#5 first: they are the two locked decisions this design lives inside (XINT8 as the measured default;
opset 17, static batch 1).

---

## 1. Why own it

Quark today does four jobs for this repo: fold and fuse the float graph, choose a
power-of-two scale per tensor from a calibration set, rewrite the graph into the QDQ form
the EP's `fuse_DPU` passes pattern-match, and (optionally) run AdaRound. It does them
behind `ModelQuantizer.quantize_model()` in `resnet_env`, pinned to 0.11rc1 by the same
1.7.1 freeze that pins `resnet_env17` for xclbins.

What owning those four jobs buys, in the order it would be collected:

1. **An acceptance map of the EP, one variable at a time.** DECISIONS #3 originally
   attributed A8W8 fallback to float scales. The A8W8 model on disk differs from XINT8
   in several ways at once (§2.5); the first probes now narrow that attribution. An
   emitter we control can flip one property per probe and read
   `vitisai_ep_report.json` after each.
2. **Per-tensor transparency at decision time.** The MobileViT collapse was diagnosed
   after the fact by reading the finished model (`results/mobilevit/quant_grid_audit.log`:
   depthwise weight grid Δ = 1.0). An owned calibrator writes the per-tensor error budget
   and every scale adjustment *as it decides*, into a sidecar beside the model.
3. **Calibration that is provably the pipeline's own preprocessing.** MODNet's every
   measured model was calibrated through a PIL resize the inference path never uses
   (`RESEARCH.md` open item). A calibration source that *is* `npu.modnet.preprocess`
   cannot drift from it.
4. **Longevity.** The dialect (§2) is a stable contract between two frozen artifacts.
   Owning the producer means the day `amd-quark 0.11rc1` stops installing is not the day
   this repo stops producing models.

What it does **not** change: latency. The EP compiles the same graph into the same DPU
code whichever tool wrote the Q/DQ nodes. Any latency delta between a Quark model and an
owned-quantizer model of the same graph is a bug or session drift, never a result.

---

## 2. The contract as measured today

Tags: **[M]** read from a Quark-produced model on disk (`models/`, this machine,
2026-09-07/08); **[Q]** Quark 0.11rc1 source, `resnet_env` site-packages
`quark/onnx/…`, file:line; **[P]** a tracked pipeline script under `pipelines/`, file:line; **[V]** `C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json`;
**[L]** a tracked log under `results/`; **[U]** unknown, to be resolved in Phase 0.

### 2.1 Graph-level fingerprint of a Quark XINT8 model

| Property | Value | Source |
|---|---|---|
| Opset / IR / domain | opset 17, `ir_version` 8, every node in the default domain | [M] `yolov8n_cut_xint8.onnx`, `resnet50_xint8_c64.onnx` |
| Node ordering | simulation nodes appended after consumers; sort in memory before the ONNX checker | [M][L] `check_resnet50_yolov8n_xint8_a8w8_scaffold_resnet_env17.log`; both XINT8 files fail before sorting and pass afterward |
| Format | QDQ: `QuantizeLinear` → `DequantizeLinear` pair on every quantized activation | [M] |
| Activations | `uint8`, zero point **128**, scalar power-of-two scale | [M] |
| Weights | `int8`, per-tensor, zero point 0, power-of-two scale; stored quantized as `<w>_quantized` feeding a `DequantizeLinear` (no `QuantizeLinear` ahead of it) | [M] |
| Bias | `int8`, zero point 0, **its own** power-of-two scale — not `in_scale × w_scale`; bounded by the `shift_bias` rule (§2.3) | [M]; [Q] `quantizers/npu_cnn_quantizer.py` `quantize_bias_tensor` |
| Initializer names | `<tensor>_scale`, `<tensor>_zero_point`, `<w>_quantized` | [M] |
| Q/DQ between Conv/Add and Relu | **pruned** (Relu consumes the Conv output directly; one Q/DQ after the Relu) | [M]; [Q] `quantization/quant_utils.py:739` `get_qdq_to_remove` |
| Concat | all available input/output positions take their **minimum** | [Q] `postprocess/refinement/refine.py:545-583`, `QuantPosManager.align_concat` |
| MaxPool / AveragePool / GlobalAveragePool / Slice / Pad | both sides take `min(ipos, opos)`; Resize is not in these rules | [Q] `refine.py:585-666`; handler parameter sharing is separate |
| GlobalAveragePool | followed by a `Mul` by a DPU factor: 7×7 → `49·21/1024 = 1.0048828125` | [M] `resnet50_xint8_c64.onnx`; [Q] `postprocess/simulation/simulate_dpu.py:108-185` (full table) |
| SiLU (`x·Sigmoid(x)`) | `HardSigmoid(alpha=1/6)` → `Mul(HARD_SIGMOID_SCALE)` → Q/DQ → `Mul(x)`, with `HARD_SIGMOID_SCALE = (2731/16384)/(1/6) = 1.0001220703125` | [M] `yolov8n_cut_xint8.onnx`; [Q] `quant_utils.py:101`, `:622` `check_hard_sigmoid_condition` |
| Split | rewritten to `Slice` nodes with `Constant` starts/ends/axes/steps | [M]; [Q] `optimizations/optimize.py:252-361` (`ConvertSplitToSlice`) |
| BatchNorm | folded into the preceding Conv before calibration; the timm resnet50 export already carries no `BatchNormalization` node, so the pass is a no-op there | [Q] `preprocess/preproc.py:149-185, 248-305`; [M] `resnet50_fp32.onnx` |
| CLE | applied by default (`include_cle=True`): 33 patterns on resnet50, 0 on the SiLU nets | [L] `results/res/quant_resnet50_xint8_c64.log`; `results/yolo_cut_quantize_xint8.log` |
| npu_cnn defaults | `ConvertSplitToSlice`, `ConvertBNToConv`, `ConvertReduceMeanToGlobalAvgPool`, `SplitLargeKernelPool` all forced on | [Q] `quantize.py:331-338` |
| HardSigmoid `beta` | omitted; installed opset-17 schema default is 0.5 | [M] audit's schema/model inspection; [Q] `simulate_dpu.py:86-98`, `quant_utils.py:622-632` accepts absence or approximately 0.5 |

The initial draft's Concat/pool rows described `QuantInfoManager`, the general
`align_quantize_info` path, and are superseded by the `QuantPosManager` rows above.
This is a source-audit correction, not a changed EP acceptance measurement [Q][L].
Stored signed weights/bias clip to **[-127, 127]**, despite INT8's wider representable
range [Q] `quant_utils.py:364-379, 527-535`; the model audit confirms those extrema [M][L].

### 2.2 How a scale is chosen

- **Position, not scale, is the variable.** `pos = rint(-log2(scale))`, `scale = 2^-pos`
  [Q] `quant_utils.py:814-833` `scale2pos` / `pos2scale`. Everything downstream (refine,
  the DPU shift rules) is integer arithmetic on `pos`.
- **Calibration data** is the float model run over the calibration set with every
  quantizable tensor exposed as an output; each tensor's samples are cast to `float16`
  and concatenated across all images ("All" mode) [Q] `calibration/calibrators.py:540+`
  `PowOfTwoCalibrater`. With `optimize_mem` they spool to `%TEMP%` — that is the
  ~105 MB/image (yolov8n at 640²) spool `CLAUDE.md` guards against.
- **MinMSE:** from the min/max position, try `pos-1 … pos+3` (five candidates) and keep
  the one with the smallest summed squared dequantization error over all samples [Q]
  `calibration/collectors.py:259-381` `compute_minmse_worker`, `:473`;
  `quant_utils.py:836-935` `compute_scale_zp`.
- **Weights take the same MinMSE path** as activations, on the weight tensor's values
  [Q] `npu_cnn_quantizer.py` `_add_qdq_pair_for_initializer`.
- **Bias initial position:** its own weight-style MinMSE on float bias values [Q]
  `npu_cnn_quantizer.py:200-216` calls `quantize_weight_tensor` when `Int32Bias=False`;
  `qdq_quantizer.py:931-956` calls `quantize_data`.

### 2.3 The DPU shift constraints (refine)

After Q/DQ insertion, `QuantPosManager` walks the graph and moves positions until every
rule holds, at most five loops [Q] `refine.py:670` `adjust_quantize_info`. `ipos`, `wpos`,
`bpos`, `opos` are the input, weight, bias and output positions of one node.

| Rule | Applies to | Constraint | Source |
|---|---|---|---|
| `shift_cut` | Conv, Gemm only | `wpos + ipos − opos ∈ [0, 16]` | [Q] `refine.py:164-203` `adjust_shift_cut` |
| `shift_bias` | same, with bias | `wpos + ipos − bpos ∈ [min(0, −(24 − (8 + shift_cut))), 15]` | [Q] `adjust_shift_bias` |
| `shift_swish` | the SiLU `Mul(x, hsig)` | `ipos0 + ipos1 − opos ∈ [0, 15]` | [Q] `adjust_shift_swish` |
| HardSigmoid | HardSigmoid | `ipos ∈ [0, 15]`, `opos ≥ 7`, `14 + ipos − opos ∈ [0, 31]` | [Q] `adjust_hard_sigmoid`; `quant_utils.py:622` |
| `shift_read` | Add, Sub | `max(ipos) − min(ipos) ∈ [0, 7]` | [Q] `adjust_shift_read` |
| `shift_write` | Add / Mul | Add: `min(ipos) − opos ∈ [−7, 25]`; Mul: `sum(ipos) − opos ∈ [0, 32]` | [Q] `adjust_shift_write` |
| alignment | Concat; MaxPool, Pad, Slice | as in §2.1 | [Q] `align_*` |

Adjustment directions are now sourced [Q] `refine.py:164-666`, reproduced with file:line
excerpts in the Phase 0 log: cut moves `wpos = clamp(sc,0,16)+opos-ipos`; bias moves
`bpos = wpos+ipos-clamp(sb,min_sb,15)` (a direct LeakyRelu successor forces `min_sb=0`);
read lowers one argmax input to `min(ipos)+7` per pass; write moves `opos` using the
clamped shift. HardSigmoid clamps input to `[0,15]`, output to `[7,14+new_ipos]`;
swish moves `opos = ipos0+ipos1-clamp(sw,0,15)`. Alignment chooses minimum positions.
Order is concat, pool, pad, slice, read, write, cut, bias, HardSigmoid, swish
[Q] `refine.py:686-724`. The draft's ConvTranspose/MatMul cut/bias coverage was incorrect.
These are producer rules, not independently measured compiler requirements.
All ten rules are transcribed in `quant/refine.py` as of 2026-09-08 and exercised on
yolov8n-cut, where the vendor's log and Ignition's sidecar record the same 20 moves (16
Concat alignments, 4 Slice alignments) in the same order. Ignition walks its own
topological order rather than the vendor's file order; the outcome is identical on this
graph, and equivalence where rules interact is unproven **[U]**. The HardSigmoid and swish
shift bounds are transcribed but no calibration has fired them yet
([YOLO preparation parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity)).

### 2.4 The consumer side

The X1 target is the pass list `['init', 'fuse_DPU']` [V]. `fuse_DPU`'s sub-passes, in
order [V]: `convert_ending_blacklist_ops_to_unknown_op`, `dynamic_input_batch`,
`create_const_op`, `convert_split_to_xir`, `convert_topk_to_xir`, `to_xir`, `convert_pad`,
`convert_in_to_gn`, `remove_extra_q_dq`, `merge_add_into_conv_bias`, `merge_fix`,
`layoutransform`, `fuse_transpose`, `remove_identity`, `add_fix_after_const`,
`const_fold_batchnorm_to_scale`, `const_fold_transpose`, `merge_pad`, `merge_hard_sigmoid`,
`merge_mul`, `final_gc`. Disabled in the shipped config: `merge_consecutive_fix`,
`convert_softmax_to_hard_softmax`, `merge_fix_fix_transpose`. `xcompilerAttrs`:
`debug_mode=performance`, `opt_level 0`, `disable_std_quant false`. 65 passes and 82 targets
are defined in the file; only the X1 slice matters here.

Two of these names are design inputs: `merge_hard_sigmoid` and `merge_mul` are why the
SiLU chain in §2.1 has the shape it has (the EP re-fuses what Quark spelled out), and
`merge_add_into_conv_bias` / `merge_fix` are why an `int8` bias with its own position is
acceptable to the DPU at all. What each pass *requires* of its input is not in the config;
that is what Phase 4 measures.

Evidence of engagement stays `<cacheKey>/vitisai_ep_report.json`, read through the
shared [`npu/ep_report.py`](../npu/ep_report.py) parser used by `tools/diag_ep.py` (schema `deviceStat` /
`nodeStat` / `shapeInfo`). A `_npu` log name means requested, not engaged.

**The QDQ file is not a bit-exact specification of what the DPU computes [L].** On the
unmutated no-CLE ResNet the NPU logits differ from unoptimized CPU QDQ execution, and
the 1,000-image top-1 drops 62.00% to 59.90%; with CLE the drop is 72.80% to 72.10%,
for Quark's and Ignition's artifacts alike
([acceptance study](../docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance),
[CLE parity](../docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset)).
Parity therefore means the same file as Quark, and "correct NPU output" in the probes
means the unmutated model's own NPU logits, never a CPU simulation. `pow2.py` is producer
arithmetic, not a model of the DPU's rounding; nothing in this design claims one.

### 2.5 The A8W8 confound

`models/resnet50_a8w8.onnx` [M] has its Q/DQ nodes in the **`com.microsoft` domain**,
`int8` zero-point-0 activations with **float** scales, and an **`int32`** bias whose scale
has shape `(1,)`. DECISIONS #3 originally attributed its CPU fallback to float scales.
Several properties changed together. **[L] The [controlled ResNet study](../docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance)
now shows domain-only fallback and exact signed-activation NPU parity.** Perturbing
activation scales by 1.01 retains placement but breaks numerical agreement; perturbing
weight scales partly falls back and also breaks agreement. INT32 bias dtype alone
preserves baseline NPU outputs, while the input×weight-scale representation does not.
This bounds the original attribution without claiming the domain is the only cause
in the stock A8W8 combination or every graph. The safe default remains unchanged.

### 2.6 Phase 0 resolutions and remaining unknowns

Every original unknown is resolved below or explicitly left open. The source excerpts,
installed versions, source/model SHA256 values, and raw fingerprints are in
[`notes_xint8_dialect.log`](../results/quant/notes_xint8_dialect.log) [L].

- **Default op list resolved [Q]:** `quantizers/interface.py:202-227` unions
  `QLinearOpsRegistry`, `QDQRegistry`, and `NPUCnnRegistry`. Exact sorted union:
  `Add, ArgMax, AveragePool, Clip, Concat, Conv, ConvTranspose, DepthToSpace, Div,
  EmbedLayerNormalization, Erf, Gather, Gelu, Gemm, GlobalAveragePool, HardSigmoid,
  InstanceNormalization, LayerNormalization, LeakyRelu, LpNormalization, MatMul, Max,
  MaxPool, Min, Mul, PRelu, Pad, ReduceMean, Relu, Reshape, Resize, Sigmoid, Slice,
  Softmax, SpaceToDepth, Split, Squeeze, Sub, Tanh, Transpose, Unsqueeze, Where`.
  Dispatch selects the NPU handler, then QDQ handler, then `QDQOperatorBase`
  (`registry.py:145-151`). Annotation's op list is Conv/Add/MaxPool/AveragePool/
  GlobalAveragePool/MatMul/Gemm/ConvTranspose (`quant_utils.py:102`); pruning also
  checks single-consumer topology and downstream activation (`:685-736`).
- **Sharing mechanism resolved [Q]:** `extended_quantizer.py:478-509` waits for the
  provider, selects its consumer-specific quantized value, and reuses its scale/zp
  initializer names. Initializer sharing is rejected. Resolved 2026-09-08 [M] for the
  head-cut YOLOv8n set: `QDQConv` marks input 0, the output and its initializers;
  `QDQMaxPool`/`QDQResize` (`QDQDirect8BitOp`) mark input 0 and share its parameters
  with the output, transitively through the SPPF chain; Sigmoid, Mul, Add, Concat and
  Slice fall to `QDQOperatorBase`, which marks every float activation input and output
  and skips INT64 Constant outputs and the empty Resize roi (`qdq_quantizer.py:256-276`).
  **[U]** handlers outside that set remain untranscribed.
- **Reader resolved [Q]:** `calibration/data_readers.py:544-579` returns the supplied
  object without `isinstance`; `calibrators.py:679` calls `get_next()`. The annotated
  base class is ORT's, so a Quark import is not needed in the adapter.
- **Bias and refinement resolved [Q]:** §2.2–2.3. **[L]** The two source hazards are
  measured on the fresh no-CLE oracle by the
  [refinement probe](../docs/BENCHMARKS.md#ignition-refinement-rules-under-perturbation):
  `refine.py:49-57` assigns raw-data scale updates to a temporary list, so Quark's
  refine is a silent no-op on `raw_data` scales (its own oracle stores `float_data`;
  Ignition's artifacts store `raw_data`); `:537-539` (Mul write without `has_change`)
  is unreachable on this graph. Twenty directed one-rule violations and 800 random
  perturbations give identical final tables from Quark's `adjust_quantize_info` and
  Ignition's `refine`, including the five-pass limit. Both producers change scale
  metadata without re-rounding stored integers, so parity requires not re-rounding;
  the DPU effect of a moved weight scale is unmeasured because no real calibration has
  fired shift-cut or shift-bias. The earlier
  [fresh no-CLE ResNet comparison](../results/quant/diff_resnet50_reemit_nocle_c64.log)
  moved only the GAP output position. Concat and Slice alignment and the
  HardSigmoid→Mul bridge are measured on yolov8n-cut (2026-09-08,
  [YOLO preparation parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity)): 20 moves identical to the vendor's. **[U]** Pad
  alignment, the HardSigmoid and swish shift bounds (transcribed, never fired),
  Gemm/ConvTranspose/MatMul cut/bias instances and the Clip/LeakyRelu/PRelu bridges
  remain untested.
- **CLE defaults resolved [Q]:** `algorithm/interface.py:61-66`: `CLESteps=1`,
  `CLEBalanceMethod="max"`, `CLEWeightThreshold=0.5`, `CLEScaleAppendBias=True`,
  `CLEScaleUseThreshold=True`, `CLETotalLayerDiffThreshold=2e-7`. **[L]** Transcribed
  in `quant/cle.py`; byte-identical equalized weights against `cle_transforms` on the
  float export, then an exact position/integer match with a fresh default-preset oracle
  that is itself identical to the repo's original resnet50 XINT8 artifact
  ([CLE parity](../docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset)). The matcher's insert-before-last sort lists
  the first pair twice (33 patterns, 32 unique); pairs are equalized in that order on
  the live initializers. **[U]** depthwise pairs/triples, Gemm pairs and Clip
  replacement raise; no measured graph exercises them.
- **Large pool resolved [Q]:** `optimizations/optimize.py:190-250`: rank-4 GAP,
  spatial area `>512`, one factorization near sqrt of each dimension; both resulting
  areas must be `<=512`. Prepend AveragePool with stride=kernel, retaining the GAP.
- **Simulation resolved [Q]:** `npu_cnn_quantizer.py:293-324` enables LeakyRelu,
  Sigmoid, HardSigmoid, AvgPool and ReduceMean; disables Softmax, InstanceNorm and
  Clip. `simulate_dpu.py:70-84` rounds LeakyRelu alpha onto a 1/256 grid; `:185-242`
  adds a nearest dyadic reciprocal correction to ReduceMean; `:276-295` maps optional
  InstanceNorm to the custom domain; `:297-345` rounds/clamps optional Clip bounds.
  **[U]** disabled Softmax's full expansion (`:261-274` delegates to
  `simulate_dpu_softmax.py`) remains open if a later phase enables it.
- **AdaRound preset and core schedule resolved [Q]:** `custom_config.py:20-30` matches
  the FastFinetune fields in §4.2. `algorithm/finetuning/train_torch/train_model_param.py:24-48`
  gives penalty 0.01, beta `(20,2)`, warmup 0.2; `train_model_loss.py:44-79` uses cosine
  decay after warmup. `create_torch/base_qdq_quantizers.py:192-263` uses gamma=-0.1,
  zeta=1.1, floor plus stretched sigmoid, and hard rounding at alpha>=0.
  `train_torch/train_model.py:55-158` optimizes alpha with Adam, random batches, and
  immediate early stop on non-improving rolling rounding loss. **Resolved 2026-09-08 [M]:**
  layer selection is every Conv/Gemm whose input arrives through Q->DQ and whose weight
  through a DQ (`onnx_subgraph.py` `find_start`/`find_end`), ending at the Relu output when
  one follows; the data path is sequential, the quantized sub-model under
  `ORT_DISABLE_ALL` and the float sub-model under default optimization; SelectiveUpdate
  (off in the preset) is not transcribed; `create_model_ops.py` builds the torch layer
  twice, which the transcription mirrors. Parity is measured bitwise in the
  [AdaRound parity section](../docs/BENCHMARKS.md#ignition-adaround-parity).
- **Consumer model metadata bounded [L]:** stripping model metadata and changing the
  producer name to `Ignition` preserve placement and NPU outputs on the measured
  no-CLE ResNet. See the [acceptance study](../docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance).
  **[U]** other graphs and opset-import/version sensitivity remain open.

---

## 3. Scope

**In:** the CNN XINT8 dialect for the X1 backend, for the models this repo already runs
(ResNet-family classifiers, yolov8 detect/pose head-cut, yolov6n, MODNet). Reproducing
Quark first, scale-exact, then departing from it deliberately and measurably.

**Out:** anything the EP is already measured to reject on this opset — INT16 / A16W8,
BFP, transformer paths (DECISIONS #3 and the MobileViT record); Strix / AIE2P; any change
to the EP, the xclbin or `npu/session.py`; QAT beyond a hook; a second copy of any
preprocessing transform. Per-channel weights and `int32` bias exist **only as Phase 4
probe mutations**, never as a default, until placement and numerical execution are validated.

**Invariants inherited from `CLAUDE.md`, plus three new ones:**

- Nothing under `quant/` imports Quark. Nothing under `npu/` imports `quant/`.
  `quant/` may import `npu/` (for the preprocessing sources and `build_session`).
- The quantizer runs in `resnet_env` (onnx 1.19, torch present). Its core — everything but
  `adaround.py` — needs only numpy, onnx and onnxruntime and must also import in
  `resnet_env17`, so `verify.py` can run beside an NPU session. (Observed, not acted on:
  `resnet_env17` on Desktop 2 has `torch 2.4.1+cpu` installed, contrary to `CLAUDE.md`'s
  "do not install torch into resnet_env17". This design does not rely on it.)
- When code lands, every `quant/*` module joins the `PIPELINE CHECKS` import list in
  `CLAUDE.md` and `CONTRIBUTING.md`.
- No new compile-cache key. Emitted models reuse the key of the pipeline whose graph they
  share (`modelcachekey`, `yolocutcachekey`, …) and every run passes `--fresh`, because
  the keys are names, not hashes.

---

## 4. Architecture

**Implemented ResNet slice:** `quant.graph` provides loading/checking, stable in-memory
topological sorting, producer/consumer/initializer and stored shape/dtype lookups, and
fingerprinting plus checked mutation/save helpers. `quant.pow2` provides `TensorQ` and arithmetic. Run
`python tools/quant_inspect.py models/resnet50_xint8_c64.onnx` in either conda env.
The inspector reports any node reordering; synced XINT8 files may append simulation
nodes after their consumers. It writes no model.

`sources.ImageFolderSource`, `calib.collect_and_choose`, `qdq.emit`,
`passes.avgpool_dpu_scale`, `refine.refine`, `verify.graph_diff` and
`quantize.quantize` now implement a folded ResNet path. Calibration uses CPU ORT
with optimization disabled, exact float16 samples on disk, float32 MinMSE, and
the shared classification transform. Weight/bias MinMSE lives in `calib.py`;
integer emission lives in `qdq.py`, without a separate `weights.py` yet.
The sidecar records the listing, sample sizes, candidate errors, initial positions,
parameter sharing, refinement moves, final positions and artifact hashes.

Use `scripts/quant-own.sh --out <new.onnx> --log <new.log>` for independent calibration
in `resnet_env17`; add `--scales-from <reference.onnx>` only for Phase 1 replay.
Both paths explicitly disable CLE. `scripts/quant-reference.sh` runs the separate
Quark oracle process; `scripts/quant-validate.sh` compares and evaluates the pair.
The shared `npu/ep_report.py` now validates EP counts and supplies placement evidence
to both `tools/diag_ep.py` and `tools/quant_probe.py`. It creates no sessions.
The probe runner records placement and output differences separately; successful
construction is not an automatic numerical acceptance verdict.

The remaining signatures below describe the broader design, not a promise that
every listed method exists. The current emitter accepts two families, read from the
float export's operators: folded ResNet (Conv/Relu/Add/MaxPool/GlobalAveragePool/
Flatten/Gemm) and head-cut YOLOv8 (Conv/Sigmoid/Mul/Add/Concat/MaxPool/Resize and
Split, rewritten to Slice). It rejects anything else, non-unit Gemm beta, MaxPool
indices, a float initializer feeding a non-Conv operator, and GAP shapes other than
the measured 7×7 case. Nested graphs remain unsupported; AdaRound is the separate
`adaround` command on an emitted file of either family. Probe mutations are separate from
the parity emitter. Degenerate
UINT8 calibration ranges are rejected because the vendor would emit zp0, outside
this slice's zp128 contract. No general XINT8 preset replacement is claimed.

### 4.1 Pipeline order

Mirrors Quark's stage order [Q] `quantization/quantize.py`, because §2.3's rules are
defined on the graph *after* Q/DQ insertion and *after* the SimulateDPU rewrites:

```
load ──► passes.prepare ──► cle ──► calib.collect ──► calib.choose ──► weights
   (fold BN, Split→Slice,     (opt)   (float16 samples)  (MinMSE pos)    (int8 pos)
    BN→Conv, ReduceMean→GAP,
    large-kernel pool split)
        ──► qdq.emit ──► qdq.prune_conv_relu ──► passes.simulate_dpu ──► refine
             (Q/DQ pairs,                          (SiLU chain, GAP Mul)    (§2.3, ≤5 loops)
              int8 weights/bias)
        ──► adaround (opt) ──► verify ──► save model + <out>.quant.json
```

Phase 1 runs only the right-hand half: it takes Quark's positions from an existing model
and exercises emit → prune → simulate → refine → verify. A calibrator cannot be validated
through an unvalidated emitter, so the emitter is validated first, alone.

Phase 1's starting graph is the **float export**, `models/resnet50_fp32.onnx` — never a
stripped Quark model, because re-quantizing dequantized int8 at the same position is an
identity and would make the weight gate vacuous. That export already needs none of
`passes.prepare`: it has 122 nodes (53 Conv, 49 Relu, 16 Add, 1 MaxPool,
1 GlobalAveragePool, 1 Flatten, 1 Gemm) and no `BatchNormalization`, `ReduceMean` or
`Split`; Quark's `resnet50_xint8_c64.onnx` is those 122 plus one `Mul` and its
`Constant` (the GAP factor) plus 74
`QuantizeLinear` / 182 `DequantizeLinear` [M, op-type counts read 2026-09-08]. So the
emitter, the GAP Mul and refine are the whole of Phase 1. yolov8n-cut is a Phase 2 target
for the same reason: its float export needs Split→Slice (8 → 16) and the SiLU chain
(57 Sigmoid → 57 HardSigmoid + 57 extra Mul, 57 → 114) [M] before its positions can be
compared. Both are transcribed (`passes.split_to_slice`, `passes.sigmoid_to_hardsigmoid`,
`passes.hardsigmoid_dpu_scale`); Ignition's prepared float graph diffs empty against
Quark's own pre-calibration graph, and the vendor's other pre-optimizations (onnxslim,
ORT basic optimization, BN folding) change nothing on this export [M 2026-09-08,
`results/quant/prepare_probe_yolov8n_cut.log`; the ResNet control is
`prepare_probe_resnet50_fp32.log`, where the same steps are no-ops and the committed
`resnet50_xint8_c64.onnx` replays exactly].

### 4.2 Modules, every function and form

Plain functions over `onnx` protos, `numpy` arrays and small dataclasses. No class
hierarchy, no plugin registry, no framework — the repo's convention. `Node` below is
`onnx.NodeProto`; `Graph` is the one wrapper class, and it exists only because the same
producer/consumer lookups are needed by every pass.

**`quant/graph.py`** — the graph wrapper

```
class Graph:
    model: onnx.ModelProto
    @classmethod load(path: Path, strict=True) -> Graph   # strict asserts opset 17, ir 8, static batch 1; strict=False records contract/checker errors for inspection; sorts in memory
    save(self, path: Path) -> None                  # checker + shape inference before write
    nodes(self) -> list[Node]                       # topological order
    producer(self, tensor: str) -> Node | None
    consumers(self, tensor: str) -> list[Node]
    initializer(self, name: str) -> np.ndarray | None
    set_initializer(self, name: str, value: np.ndarray, dtype=None) -> None
    remove_initializer(self, name: str) -> None
    value_shape(self, tensor: str) -> tuple[int, ...] | None   # stored metadata today; inference later
    value_dtype(self, tensor: str) -> int | None
    insert_after(self, node: Node, new_nodes: list[Node]) -> None
    insert_before(self, node: Node, new_nodes: list[Node]) -> None
    replace_node(self, old: Node, new: Node) -> None
    remove_node(self, node: Node, reconnect: bool = True) -> None   # bypasses 1-in/1-out nodes
    rename_tensor(self, old: str, new: str) -> None
    topo_sort(self) -> None
    check(self) -> None                             # onnx.checker.check_model + the opset assert
    fingerprint(self) -> dict                       # §2.1 table computed from the graph, for logs

def make_node(op: str, inputs, outputs, name: str, **attrs) -> Node
def const_node(name: str, value: np.ndarray) -> Node        # the Constant form Split→Slice uses
def unique_name(g: Graph, base: str) -> str
```

**`quant/config.py`** — configuration and presets

```
@dataclass
class FastFinetuneConfig:            # field names are Quark's extra_options["FastFinetune"] keys
    DataSize: int = 1000             # defaults = the ADAROUND dict in
    FixedSeed: int = 1705472343      # pipelines/yolov8n/3b_quantize_cut.py:36-40 [P]
    BatchSize: int = 2
    NumIterations: int = 1000
    LearningRate: float = 0.1
    OptimAlgorithm: str = "adaround"
    OptimDevice: str = "cpu"
    InferDevice: str = "cpu"
    EarlyStop: bool = True
    # pipelines/resnet50/3_quantize.py:143-148 [P] instead takes get_default_config(
    # "XINT8_ADAROUND")'s own FastFinetune dict and overrides only OptimDevice/InferDevice
    # (--device) and NumIterations (--iters, default 1000); other fields match above [Q custom_config.py:20-30]

@dataclass
class QuantConfig:
    activation_dtype: str = "uint8"; activation_zp: int = 128
    weight_dtype: str = "int8"; weight_per_channel: bool = False     # probe-only when True
    bias_dtype: str = "int8"                                          # "int32" is probe-only
    pow2: bool = True; minmse_range: tuple[int, int] = (-1, 3)
    calib_limit: int | None = None; calib_store: str = "exact"        # "exact" | "hist"
    cle: bool = True; cle_params: dict = field(default_factory=dict)  # resolved defaults in §2.6 [Q]
    convert_split_to_slice: bool = True; convert_bn_to_conv: bool = True
    convert_reduce_mean_to_gap: bool = True; split_large_kernel_pool: bool = True
    simulate_dpu: bool = True; refine: bool = True; refine_max_loops: int = 5
    prune_conv_relu: bool = True
    fast_finetune: FastFinetuneConfig | None = None
    opset: int = 17; ir_version: int = 8
    threads: int | None = None                                        # ORT intra-op for calibration

PRESETS: dict[str, QuantConfig]     # "XINT8", "XINT8_ADAROUND" — same names as get_default_config()
def preset(name: str) -> QuantConfig
def to_json(cfg: QuantConfig) -> dict; def from_json(d: dict) -> QuantConfig
```

**`quant/pow2.py`** — the fixed-point arithmetic, and nothing else

```
ACT_DTYPE, ACT_ZP = np.uint8, 128
W_DTYPE, B_DTYPE = np.int8, np.int8

@dataclass(frozen=True)
class TensorQ:
    name: str; dtype: str; pos: int; zp: int
    source: str            # "minmse" | "weight" | "bias" | "align_concat" | "shift_cut" | ... — who set it last

def qrange(dtype: str) -> tuple[int, int]         # producer bounds: int8 [-127,127], uint8 [0,255]
def scale2pos(scale: float) -> int              # rint(-log2(scale))            [Q quant_utils 814-833]
def pos2scale(pos: int) -> np.float32
def quantize(x: np.ndarray, pos: int, zp: int, dtype: str) -> np.ndarray     # clip(rint(x·2^pos) + zp)
def dequantize(q: np.ndarray, pos: int, zp: int) -> np.ndarray
def sqerr(x: np.ndarray, pos: int, zp: int, dtype: str) -> float             # Σ (x − dq(q(x)))²
```

Rounding mode is `np.rint` (half to even), confirmed by Quark's float32 divide followed
by NumPy `.round()` [Q] `quant_utils.py:527-535`. `sqerr` keeps float32 accumulation,
as `quantize_data` does [Q] `:1109`. The arithmetic API rejects nonfinite input, invalid
zero points, nonpositive scales and positions outside [-127,127]; valid positive
scales are clamped to [2^-127,2^127] before conversion, matching `:814-833`. This API
validation is our policy, not a claim about Quark's handling of invalid inputs.

**`quant/sources.py`** — calibration inputs that *are* the pipeline's preprocessing

```
class CalibSource(Protocol):
    input_name: str
    def listing(self) -> list[Path]            # sorted, after [:limit] — recorded in the sidecar
    def __len__(self) -> int
    def __iter__(self) -> Iterator[np.ndarray] # one NCHW float32 batch-1 array per image

class ImageFolderSource:   (calib_dir, cfg, limit, input_name="input")   → npu.preprocess.build_transform(cfg)
class CocoSource:          (folder, limit, imgsz, input_name="images")   → npu.yolo.letterbox; imgsz from npu.yolo.input_size
class ModnetSource:        (folder, limit, input_name)                   → npu.modnet.preprocess
class NpzSource:           (path)                                         # replay a recorded listing exactly

def as_reader(src: CalibSource):     # duck-typed get_next()/rewind() object for feeding the SAME
                                     # listing to Quark in the comparison harness (§2.6: no isinstance guard [Q])
```

Every source's `listing()` sorts and slices the way `pipelines/*/3*_quantize.py` do today,
so a Quark run and an owned run on one machine see identical images in identical order.
`data/` is not Syncthing-synced, so this only holds within a machine — a comparison
across machines is not scale-for-scale and must be labelled as such.

**`quant/calib.py`** — collect samples, choose positions

```
@dataclass
class TensorStats:
    name: str; count: int; vmin: float; vmax: float
    samples: np.ndarray | Path | None     # float16, concatenated ("exact") — or spooled path
    hist: tuple[np.ndarray, np.ndarray] | None     # ("hist" store) bins, counts

@dataclass
class CalibStats:
    tensors: dict[str, TensorStats]; listing: list[Path]; input_name: str; store: str

def quantizable_tensors(g: Graph) -> list[str]        # §2.6 resolved union [Q]; handler eligibility still [U]
def augment_outputs(g: Graph, tensors: list[str]) -> onnx.ModelProto
def collect(model: Path, src: CalibSource, cfg: QuantConfig, tmp: Path | None) -> CalibStats
        # ORT CPU, one sess.run per image, float16 cast, "exact" concatenates / spools like Quark,
        # "hist" keeps a fixed-bin histogram (memory-light, NOT byte-exact — label the model)
def choose_pow2_minmse(x: np.ndarray, dtype: str, zp: int, rng=(-1, 3)) -> int
        # exact replica of compute_minmse_worker [Q collectors 259-381]: candidates from the
        # min/max position, keep argmin Σ sqerr
def choose_pow2_minmax(vmin, vmax, dtype, zp) -> int
def choose_all(stats: CalibStats, cfg: QuantConfig) -> dict[str, TensorQ]
def share_params(g: Graph, q: dict[str, TensorQ]) -> int       # §2.6 resolved provider-name sharing [Q]
```

**`quant/weights.py`** — initializers

```
def quantize_weight(w: np.ndarray, cfg: QuantConfig) -> tuple[np.ndarray, TensorQ]
        # per-tensor MinMSE, same path as activations [Q]; per-channel only if cfg says (probe)
def quantize_bias(b: np.ndarray, w_pos: int, in_pos: int, cfg: QuantConfig) -> tuple[np.ndarray, TensorQ]
        # int8 with its own MinMSE pos (§2.2 [Q]); int32 in×w form is the probe variant
def conv_like(node: Node) -> bool                      # Conv, ConvTranspose, Gemm, MatMul with initializer weight
```

**`quant/qdq.py`** — the emitter

```
def insert_qdq(g: Graph, tensor: str, tq: TensorQ) -> tuple[Node, Node]
        # QuantizeLinear/DequantizeLinear pair, initializers <t>_scale (float32 scalar),
        # <t>_zero_point (uint8 scalar), consumers rewired to the DQ output — Quark's names [M]
def quantize_initializer(g: Graph, name: str, q: np.ndarray, tq: TensorQ) -> Node
        # replaces the float initializer with <w>_quantized int8 + one DequantizeLinear [M]
def emit(g: Graph, acts: dict[str, TensorQ], cfg: QuantConfig) -> EmitReport
        # walks quantizable_tensors, inserts pairs, quantizes weights and biases per node
def prune_conv_relu(g: Graph) -> int                   # removes the Q/DQ between Conv/Add and Relu [M][Q 739]
def read_pos_table(g: Graph) -> dict[str, TensorQ]     # inverse: positions out of an existing Quark model (Phase 1's input)
def strip_qdq(g: Graph) -> Graph                       # inverse: back to float, for graph_diff and the probes

@dataclass
class EmitReport: activations: int; weights: int; biases: int; pruned: int
```

**`quant/passes.py`** — graph rewrites, before and after Q/DQ

```
# before calibration (Quark's pre-optimization and npu_cnn defaults)
def fold_batchnorm(g: Graph) -> int                    # Conv+BN → Conv               [Q preproc 149-185]
def bn_to_conv(g: Graph) -> int                        # standalone BN → 1×1 depthwise Conv (ConvertBNToConv)
def split_to_slice(g: Graph) -> int                    # Split → Slice + Constant nodes [M][Q 331-338]
def reduce_mean_to_gap(g: Graph) -> int                # ReduceMean(2,3) → GlobalAveragePool
def split_large_kernel_pool(g: Graph) -> int           # rank-4 GAP area >512, one split; §2.6 [Q]
def prepare(g: Graph, cfg: QuantConfig) -> dict[str, int]

# after Q/DQ (Quark's SimulateDPU)                                          [Q simulate_dpu.py]
HARD_SIGMOID_SCALE = 1.0001220703125                   # (2731/16384)/(1/6)             [Q quant_utils 101]
AVGPOOL_FACTORS: dict[tuple[int, int], float]          # {(7, 7): 49·21/1024, ...} transcribed from :108-185
def sigmoid_to_hardsigmoid(g: Graph) -> int            # Sigmoid → HardSigmoid(alpha=1/6)
def hardsigmoid_dpu_scale(g: Graph, q) -> int          # + Mul(HARD_SIGMOID_SCALE) + Q/DQ before the SiLU Mul
def avgpool_dpu_scale(g: Graph) -> int                 # GlobalAveragePool → + Mul(factor)
def simulate_dpu(g: Graph, q: dict[str, TensorQ], cfg: QuantConfig) -> dict[str, int]
# stubs, raising NotImplementedError with the Quark line to transcribe: leaky_relu, reduce_mean,
# softmax, instance_norm, clip  (§2.6)
```

**`quant/cle.py`** — cross-layer equalization

```
def find_pairs(g: Graph) -> list[ClePair]                  # single-consumer walk through Relu/ReduceMean/Pad/LeakyRelu, plus the source's insert-before-last sort [Q algorithm/cle/equalization.py]
def equalize_pair(g, head, tail, weight_threshold, append_bias, use_threshold) -> dict   # _cross_layer_equalize for Conv→Conv group 1: bias column, "max" balance, tail * (1/scale)
def cross_layer_equalize(g: Graph, *, steps=1, ...) -> CleReport   # cle_transforms + process_cle_transforms step loop; report carries the ordered pairs and per-pair scale stats
```

Implemented as above [L]: 33 patterns (32 unique, the first listed twice) and
byte-identical equalized float weights on resnet50; yolov8n-cut's 0 patterns remains a
design expectation until that graph is prepared.
The matcher's rejections (Clip, Sigmoid, multi-consumer convs — `results/mobilevit/
cle_pattern_count.log`) are reproduced, not fixed; CLE across ReLU6 is not sound and the
record already says so.

**`quant/refine.py`** — the shift rules

```
class PosTable:                                        # dict[str, TensorQ] + an append-only change log
    get(tensor) -> TensorQ; set(tensor, pos, source) -> None; log: list[tuple[str, int, int, str]]

def node_positions(g: Graph, node: Node, pos: PosTable) -> tuple[ipos, wpos, bpos, opos]
def adjust_shift_cut(g, pos) -> int
def adjust_shift_bias(g, pos) -> int
def adjust_shift_swish(g, pos) -> int
def adjust_hard_sigmoid(g, pos) -> int
def adjust_shift_read(g, pos) -> int
def adjust_shift_write(g, pos) -> int
def align_concat(g, pos) -> int
def align_pool(g, pos) -> int; def align_pad(g, pos) -> int; def align_slice(g, pos) -> int
def refine(g: Graph, pos: PosTable, max_loops: int = 5) -> RefineReport      # loop until no change [Q 670]
def apply(g: Graph, pos: PosTable) -> None             # writes 2^-pos back into every <t>_scale initializer

@dataclass
class RefineReport: loops: int; changes_by_rule: dict[str, int]; log: list
```

Each `adjust_*` returns the number of positions it moved. The change log is what goes
into the sidecar: for every tensor whose final position is not its MinMSE position, the
rule that moved it and by how much. That list is the per-layer transparency of item 2 in §1.

**`quant/adaround.py`** — post-quantization rounding optimisation (torch, imported inside `finetune` only; landed 2026-09-08)

```
@dataclass
class FastFinetuneConfig          # Quark's extra_options["FastFinetune"] keys plus its fixed TrainParameters
def layer_targets(qg: Graph, fg: Graph) -> list[Layer]   # Conv/Gemm in the float graph's vendor_order (ORT topological_sort, Quark's loop order), with QDQ params
def finetune(float_graph: Graph, quant_graph: Graph, source: ImageFolderSource,
             cfg: FastFinetuneConfig | None = None, log=print) -> AdaRoundReport
        # per layer, sequentially: quantized pre-Q input (ORT_DISABLE_ALL) and float input/output
        # (default ORT) for DataSize images, all resident; torch module built like Quark's (two RNG
        # draws per layer); Adam on the rounding variable; hard rounding clamped to [-128, 127]
        # written to <w>_quantized in place; positions untouched; peak working set in the report
```

Parity was measured in bytes, not only in accuracy: with the same seed, module
construction, batch draws, data sessions and loss, Ignition's file is byte-identical to
a fresh `XINT8_ADAROUND` oracle on the same machine and runtime
([AdaRound parity](../docs/BENCHMARKS.md#ignition-adaround-parity)); across machines or runtimes the
comparison is statistical, as the Sep 5 control in that section shows.

#### Note for implementation: GPU / ROCm acceleration for AdaRound

When scaling to deep restoration networks (e.g. Real-ESRGAN 10-RRDB with 156 convolutions) or wide detectors (YOLOv8m/l/x), CPU AdaRound becomes the sole runtime bottleneck (~15 min at 64², projected ~1 hr at 128² on Zen 4). Enabling ROCm/HIP acceleration in Ignition should follow these guidelines:

1. **Strict scope isolation:** The rest of Ignition (`passes`, `calib`, `cle`, `refine`, `qdq`, `quantize`) must remain strictly pure Python / NumPy with zero PyTorch or GPU dependencies (`_check_imports()` invariant). Only `quant/adaround.py` imports torch and uses GPU devices.
2. **Hybrid execution architecture (high ROI, low complexity):**
   - **Keep ORT activation extraction on CPU:** `_run_quantized_inputs` and `_run_float` take only ~0.2s per layer on CPU (<2% of total runtime). Attempting to wire ONNX Runtime through `MIGraphXExecutionProvider` or `ROCMExecutionProvider` under Windows introduces severe configuration fragility for negligible wall-clock gain.
   - **Move only the PyTorch training loop to GPU:** 98%+ of runtime is spent inside the 1,000-iteration Adam optimization loop. Enable `OptimDevice = "cuda"` / `"cuda:0"` in `FastFinetuneConfig.check()`, transfer `Module`, `self.alpha`, `scale`, `zero_point`, and the drawn batch tensors to device inside the iteration loop, ensure `RoundHalfToEven` and `qdq` execute on CUDA/ROCm tensors, and retrieve weights via `.cpu().numpy()` at layer readout.
3. **Target machine routing:**
   - **Desktop 1** (Ryzen 7 7800X3D + Radeon RX 7900 XTX 24 GB) is the intended hardware target for GPU AdaRound. Discrete RDNA3 has official ROCm/CUDA support in PyTorch.
   - **Desktop 2** (Ryzen 7 8700G, Phoenix APU) runs Windows 11 where the Radeon 780M iGPU lacks official Windows PyTorch ROCm wheels (would require Linux/WSL2).
4. **Numerical parity contract:**
   - GPU floating-point parallel reductions in Adam loss computation have non-associative summation order compared to CPU, which may introduce $\pm 1$ LSB differences on edge weights.
   - Validation must be paired against a Quark GPU oracle run (`--device cuda`) on Desktop 1 rather than comparing cross-device against CPU.

**`quant/verify.py`** — the gates, as code

```
@dataclass
class GraphDiff:
    node_delta: list[str]          # (op_type, attrs, input dtypes) multiset difference
    init_exact_mismatch: list[str] # scales / zero points that differ at all
    weight_lsb: dict[str, int]     # max |Δ| in LSB per int8 initializer, only entries > 0
    ok(self, weight_tol_lsb: int = 1) -> bool

def graph_diff(a: Graph, b: Graph) -> GraphDiff        # semantic, ignores node names and order
def simulate_cpu(model: Path, src: CalibSource, n: int, threads=None) -> list[np.ndarray]
def compare_outputs(a: list[np.ndarray], b: list[np.ndarray]) -> dict   # max |Δ|, argmax agreement
def ep_report(model: Path, cache_key: str, xclbin: Path | None, fresh=True) -> EPReport
        # npu.session.build_session(ep="npu"), then <cacheKey>/vitisai_ep_report.json parsed by
        # shared npu/ep_report.py logic (implemented); this orchestration signature remains
        # a design sketch, while tools/quant_probe.py currently owns session construction
@dataclass
class EPReport: nodes_total: int; nodes_npu: int; nodes_cpu: int; device_stat: dict; cpu_nodes: list[str]
```

**`quant/quantize.py`** — the one entry point

```
@dataclass
class Report:
    config: QuantConfig; listing: list[Path]; versions: dict
    prepare: dict[str, int]; cle_patterns: int; emit: EmitReport; simulate: dict[str, int]
    refine: RefineReport; positions: dict[str, TensorQ]; adaround: dict | None
    wall_s: dict[str, float]; peak_rss: int

def quantize(model_in: Path, model_out: Path, src: CalibSource, cfg: QuantConfig, *,
             scales_from: Path | None = None,      # Phase 1: take positions from a Quark model, skip calib
             tmp: Path | None = None) -> Report
def write_sidecar(model_out: Path, report: Report) -> Path      # <out>.quant.json, next to the model
```

`<out>.quant.json` is the transparency artifact: config, the exact calibration listing,
every tensor's MinMSE position and its final position with the rule that moved it, the
CLE pattern count, tool versions, wall time per stage and peak RSS. It is git-ignored
with the model it describes; the numbers that matter are copied into the run log.

**`quant/probe.py`** — the acceptance map

**Implemented:** `mutate(Graph, name)` returns a checked copy and a change report;
`output_difference` compares finite matching output arrays. Actual names are
`baseline`, `strip_metadata`, `producer_ignition`, `domain_msft`, `act_int8_zp0`,
`float_act_scales`, `float_weight_scales`, `bias_int32_dtype`, `bias_int32_product`,
`weights_per_channel`, and `drop_gap_mul`. The scale probes multiply existing scales
by float32 1.01 (activation probe excludes the final output); they do not recalibrate.
The per-channel probe repeats scalar parameters, leaving stored weights unchanged.
The two bias probes distinguish dtype from scale representation. All are diagnostic
mutations, not production configuration options.

`tools/quant_probe.py` runs optimized and unoptimized CPU references, checks a clean
NPU context, clears the existing cache, archives the EP report and compares outputs.
`tools/quant_probe_bounded.py` uses the already-installed `psutil` to bound its child
at 8 GiB RSS and 300 seconds by default; the wrapper allows 600 seconds for full eval.
`tools/quant_probe_cpu_audit.py` audits older saved outputs; `tools/quant_probe_summary.py`
binds CPU audits to both model and input hashes. The measured slice and full-set
confirmation are in [BENCHMARKS](../docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance).

The catalog below is the remaining broader design, not the implemented registry:

```
Mutation = Callable[[Graph], Graph]      # pure: takes an emitted XINT8 graph, returns a variant
MUTATIONS: dict[str, Mutation] = {
    "baseline":            identity,
    "strip_metadata":      drop producer_name / metadata_props,
    "domain_msft":         move Q/DQ to com.microsoft (the A8W8 domain, alone),
    "float_scales":        replace 2^-pos scales with the MinMax float scale (alone),
    "act_int8_zp0":        int8 zero-point-0 activations (alone),
    "bias_int32":          int32 bias at in×w scale (alone),
    "weights_per_channel": per-channel int8 weights,
    "drop_gap_mul":        remove the GlobalAveragePool DPU Mul,
    "drop_hsig_mul":       remove the HARD_SIGMOID_SCALE Mul,
    "no_concat_align":     MinMSE positions on Concat inputs instead of the output's,
    "no_conv_relu_prune":  keep the Q/DQ between Conv and Relu,
    "opset_13": ..., "opset_19": ..., "opset_21": ...,
}
@dataclass
class ProbeResult:
    name: str; ep: EPReport; cpu_vs_npu: dict; latency_ms: float | None; note: str
def run_probe(base: Path, names: list[str], cache_key: str, src: CalibSource, n_images: int) -> list[ProbeResult]
        # per mutation: write models/<stem>_probe_<name>.onnx, clear the cache, build the NPU
        # session, copy the EP report into the log, compare NPU output to --ep cpu on the SAME
        # file (a plausible latency with wrong outputs is the batch-2 failure shape), latency
def table(results: list[ProbeResult]) -> str          # the markdown that goes into BENCHMARKS
```

The output comparison is not optional. DECISIONS #3's A16W8 record and the batch-2 record
both show the EP failing with a believable latency; a probe that reads only
`nodes_npu` would repeat that mistake.

**Entry points and wrappers** (thin, `argparse`, digit-prefixed so they cannot be imported):

- `pipelines/resnet50/3c_quantize_own.py`, `pipelines/yolov8n/3c_quantize_cut_own.py`,
  later one per pipeline: build the matching `CalibSource`, call `quant.quantize`, print
  the report.
- `pipelines/resnet50/3d_quantize_compare.py`: the Phase 1/2 harness. Runs Quark
  (`include_cle=False` for Phase 1; defaults for Phase 2) and `quant.quantize` on the
  **same** listing via `sources.as_reader`, then `graph_diff` and `compare_outputs`. This
  is the only script that imports both, and it lives under `pipelines/`, not `quant/`.
- `tools/quant_probe.py`: one controlled mutation per process, launched by the bounded wrapper.
- `scripts/quant-own.sh`, `scripts/quant-probe.sh`: `run_logged`, the per-variant disk
  guard (`calib_mb_per_image` applies to the "exact" store as much as to Quark), the
  Quark cleanup trap for the compare harness, `npu_verdict` after each probe.

---

## 5. Gates

A phase is done when its log exists under `results/quant/` and says what was run on which
machine. Log names carry model and variant, never a bare name (`CLAUDE.md`).

| Phase | Log(s) | What must be true |
|---|---|---|
| 0 | `notes_xint8_dialect.log`, subsequent scaffold inspection logs | §2 with every [U] either resolved (file:line quoted) or listed as still open; initial throwaway inspection followed by `Graph.fingerprint()` once the audit is folded. ResNet/YOLO XINT8, ResNet A8W8 and the ResNet float export. Vendor source read verbatim, no hardware — the `quant_grid_audit.log` precedent |
| 1 | `quant_resnet50_quark_nocle.log`, `quant_resnet50_own_reemit.log`, `diff_resnet50_own_vs_quark.log`, `run_resnet50_own_reemit_{cpu,npu}.log`, `diag_resnet50_own_reemit.log` | Starting from `models/resnet50_fp32.onnx`, against a **fresh** Quark run with `include_cle=False` on the same machine: `graph_diff.ok()` with the weight-LSB count printed; `refine()` run on the positions read out of the Quark model reports zero moves (every bound in §2.3 validated); identical full-set CPU top-1 through the existing `4_run.py`; EP report with the same node split as the Quark model's own `diag_*`; NPU latency captured in the same sitting as the Quark model's, and reported as a pair |
| 2 | `quant_resnet50_own_xint8.log`, `quant_yolov8n_cut_own_xint8.log`, `diff_*_own_vs_quark.log`, `run_*_own_xint8_npu.log`, `map_yolov8n_cut_own_xint8_npu.log` (full 5000), `diag_*` | Same machine, same sorted listing, same `--limit`: position table equal to Quark's for every tensor (or the diff listed and explained); full-set top-1 / full-5000 mAP beside Quark's from the same sitting. A slice is labelled a slice. ResNet closed (no-CLE and CLE sections); yolov8n-cut closed 2026-09-08 (`quant_yolov8n_cut_{quark,ignition}_cle_c64.log`, `diff_yolov8n_cut_ignition_cle_c64.log`, `map_yolov8n_cut_ignition_cle_c64_{reference,own}_{cpu,npu}.log`) |
| 3 | `quant_*_own_xint8_adaround.log`, `run_*`/`map_*` | Accuracy within session noise of `XINT8_ADAROUND` on resnet50 and yolov8n-cut; peak RSS logged beside Quark's on the same machine. Stretch: the laptop finishes where Quark SIGSEGVs. ResNet closed 2026-09-08 bitwise (`quant_resnet50_ignition_cle_adaround_c64.log`, `diff_*`, `run_*`); yolov8n-cut closed 2026-09-08 bitwise once the layer order followed the vendor's sort (`quant_yolov8n_cut_ignition_cle_adaround_c64.log`, `diff_*`, `map_*`); the laptop stretch is unrun |
| 4 | `probe_resnet50_<mutation>.log` per mutation, plus `probe_resnet50_summary.log` | For every mutation: EP node split, NPU-vs-CPU output agreement on N images, same-sitting latency; foreign-context check before each. The A8W8 attribution in DECISIONS #3 rewritten from the single-variable result — kept beside the old wording, not replacing it |
| 5 | one log per item | Each item measured and folded like any other experiment |

Machine routing follows `CLAUDE.md`'s table: quantization and the Quark comparison on
Desktop 1 or Desktop 2 (the comparison needs Quark *and* the NPU for its `diag_*`, so
Desktop 2 is the only box where Phases 1–2 close in one sitting); NPU gates on the laptop
or Desktop 2; wide/large variants never on the laptop. Before every NPU number:
`xrt-smi examine -r aie-partitions` says no hardware contexts, or the monitor is running
as a witness and the log says so.

---

## 6. Roadmap

| # | Phase | Builds | Done when | Where |
|---|---|---|---|---|
| 0 | Fingerprint | `notes_xint8_dialect.log`; §2 corrected | every [U] closed or explicitly open | any machine, no hardware |
| 1 | Re-emit | `graph.py`, `pow2.py`, `qdq.py`, `refine.py` (`apply` + rules), `passes.simulate_dpu`, `verify.py`, `quantize --scales-from`, `3d_quantize_compare.py`, `npu/ep_report.py` | resnet50 from the float export: `graph_diff.ok()` against a fresh no-CLE Quark model (scales exact, weights ≤1 LSB, count logged); `refine()` on Quark's positions moves nothing; identical CPU top-1; same EP split; paired latency | quantize D1/D2 → NPU L/D2 |
| 2 | Calibrate | `sources.py`, `calib.py` (exact store), `weights.py`, `cle.py`, `passes.prepare`, `3c_quantize_own.py` ×2, `scripts/quant-own.sh` | position-for-position equal to Quark on resnet50 and yolov8n-cut, same machine + listing; full-set top-1 and full-5000 mAP paired. ResNet and yolov8n-cut closed 2026-09-08 | Desktop 2 |
| 3 | AdaRound | `adaround.py`, `FastFinetuneConfig`, RSS logging | accuracy parity with `XINT8_ADAROUND`; RSS beside Quark's. ResNet and yolov8n-cut: byte-identical on Desktop 2 (2026-09-08) | D1/D2, then the laptop as the stretch |
| 4 | Acceptance map | `probe.py`, `tools/quant_probe.py`, `scripts/quant-probe.sh` | every mutation in §4.2 measured with output check; new BENCHMARKS section; DECISIONS #3 amended | L/D2 |
| 5 | Beyond Quark | per-layer error budget in the sidecar → a BENCHMARKS table; MODNet re-calibrated through `npu.modnet` (closes the RESEARCH open item); `calib_store="hist"` with its accuracy cost measured; MobileViT per-channel **only if** Phase 4 admits it; QAT hook (`sources` + `pow2` reused from torch) | each a logged, folded experiment | per `CLAUDE.md` routing |

The first ResNet slice of Phase 4 ran after Phase 1 and independent no-CLE calibration
passed: resolving the confounded acceptance rules was the next useful research task.
This does not complete Phase 4's broader catalog; opset changes, Concat, retained
Conv/Relu QDQ and HardSigmoid mutations remain untested.

Dependencies remain: Phase 2 does not start until Phase 1's diff log reads clean, and Phase 5
items each wait for the Phase 4 probe that licenses them. Phases 0 and 1 are one to two
sessions each; Phase 2 is where most of the transcription work sits (registry op list,
refine directions, the avgpool table). Nothing here is a latency experiment.

---

## 7. Risks and open questions

- **The dialect may be under-specified by what Quark emits.** Quark writes one point in
  the EP's acceptance region; Phase 4 maps its neighbourhood, not the region. A probe that
  passes says "this variant is accepted", never "this property is required".
- **Scale-exact equality may be blocked by a rounding-mode or float16 detail** deep in MinMSE.
  The Phase 1 gate tolerates ≤1 LSB in weights *and prints the count* so a systematic
  half-rounding difference shows up as a number, not as "close enough".
- **CLE order and thresholds** are transcribed, not designed; a different pair order gives
  different weights and a different (still valid) model. The pattern-count gate catches a
  matcher difference, not an ordering one — Phase 2's position table does.
- **RAM.** The exact store is Quark's ~105 MB/image; owning the code shrinks it only by
  what emission discards (36 percent on ResNet, the pruned pre-Relu tensors;
  [measured](../docs/BENCHMARKS.md#ignition-calibration-spool-without-the-pruned-pre-relu-tensors)).
  The `hist` store is the escape hatch and is labelled approximate until measured.
- **AdaRound parity is statistical across machines, bitwise on one.** Measured
  2026-09-08: byte-identical to the oracle on Desktop 2 under one runtime, while a
  Sep 5 Quark artifact from an unrecorded machine differs from the fresh oracle in
  4,149,415 weights by one LSB. If ours lands 0.3 points below Quark's on another box,
  that is a numerics-environment question first (the repo's drift record), a bug second.
- **The EP might key on something unobservable from the model** (a registry, an env var).
  The `strip_metadata` and `baseline` probes exist to bound that early.
- **Two live sessions share this worktree and NPU.** Every NPU log records the
  foreign-context check; every commit goes through `scripts/commit.sh`.

---

## 8. Where results fold

Same rule as every other experiment: the log in the same commit as the doc change.
`results/quant/` gets a row in [`results/README.md`](../results/README.md)'s index
pointing at a new `docs/BENCHMARKS.md` section ("An owned XINT8 quantizer: scale-exact reproduction,
then the EP's acceptance map"); Phase 4 amends DECISIONS #3 in place and adds a
"Rejected approaches" entry for every mutation the EP refuses; Phase 5's MODNet
re-calibration closes the RESEARCH.md open item it names. `python tools/check_links.py`
before every doc commit; `tools/audit_numbers.py` when a number is superseded — the A8W8
"float scales" attribution will be the first.
