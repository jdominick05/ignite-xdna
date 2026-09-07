# Ryzen AI XDNA1 NPU quantization pipelines

> **Research and characterization project, not a packaged tool.** All three pipelines
> are reproducible end to end, but the target is one specific chip generation, one
> specific SDK version, and Windows only — see [Compatibility](#compatibility) before
> assuming this runs on your machine as-is.

INT8 quantization and NPU deployment for **AMD Hawk Point** (Ryzen 8040-series,
XDNA1, 16 TOPS) on Windows, using AMD Quark and the VitisAI ONNX Runtime execution
provider.

Three pipelines:

| Pipeline | Model | Status |
|---|---|---|
| `pipelines/resnet50` | timm `resnet50.a1_in1k` classification | **Working** — 79.8% top-1 at 6.9 ms on the NPU |
| `pipelines/yolov8n` | YOLOv8 n / s / m object detection | **Working** — 8.9 / 15.6 / 30.8 ms on the NPU once the decode tail is cut off the graph |
| `pipelines/yolov8n-pose` | YOLOv8n-pose (17-point COCO keypoints) | **Working** — 9.8 ms on the NPU (1015/1025 nodes, `results/pose_cut_{npu,diag}.log`), OKS mAP@50-95 31.65 XINT8 vs 49.49 float (`results/map_kpts_*.log`, 500-image slice) |

The most useful part of this repository may not be the code. It is the set of facts
about XDNA1 that are undocumented, or documented incorrectly, collected under
[Key findings](#key-findings) and at greater length in
[`docs/DECISIONS.md`](docs/DECISIONS.md). [`RESEARCH.md`](RESEARCH.md) is the one-level-up
thesis — the question being asked and why — for framing before the numbers.

## How it works

1. **Export** — pull the model from timm or Ultralytics, resolve its own preprocessing
   config (never hardcoded ImageNet defaults), and export to ONNX with a static shape
   (opset 17, batch 1, no dynamic axes — see [Key findings](#key-findings) for why).
2. **Quantize** — AMD Quark converts the FP32 graph to XINT8 (power-of-two scales, the
   only scheme this backend accepts), calibrated on real images, with AdaRound as an
   optional accuracy-recovery pass.
3. **Run** — the VitisAI ONNX Runtime execution provider compiles and runs the graph on
   the NPU. `tools/diag_ep.py` reads its per-node assignment report, because nothing
   else in AMD's tooling surfaces what actually landed on-device.
4. **Measure** — latency and accuracy, always together, across every axis this repo has
   found to matter: resolution, model width, and (as a negative result) batch size. See
   [Results](#results).

---

## Results

**ResNet50** — 1000 ImageNet-1k validation images, the same images for every row.
Published top-1 for `resnet50.a1_in1k` is 80.4%.

| Model | top-1 | top-5 | Latency | Device |
|---|---|---|---|---|
| FP32 | 80.10% | 93.90% | 19.7 ms | CPU |
| XINT8 | 71.40% | 88.10% | 40.2 ms | CPU (ORT dequantizes — slower than FP32) |
| XINT8 | 71.70% | 88.40% | **5.63 ms** | NPU (393 ops NPU / 2 CPU, 1 subgraph) |
| A8W8 | 66.60% | 85.10% | 39.0 ms | CPU fallback despite requesting NPU |
| **XINT8 + AdaRound** | **79.80%** | **92.50%** | **6.93 ms** | **NPU** |

Two things worth pulling out of that table. Plain XINT8 costs 8.4 points of top-1,
which is a lot; AdaRound buys back all but 0.3 of it for about 1.3 ms. And NPU versus
CPU on the *same* INT8 model agree to within 0.3% — the NPU's numerics are faithful,
so any accuracy gap you see is the quantization, not the hardware.

**YOLOv8n** — 640×640, single image, inference only (post-processing listed separately).

| Model | Latency | Device |
|---|---|---|
| FP32, full graph | 37.0 ms | CPU |
| FP32, head-cut | 31.8 ms | CPU |
| XINT8, full graph | 39.1 ms | CPU — requested NPU, but the EP took zero nodes |
| **XINT8, head-cut** | **8.7–9.8 ms** | **NPU** (922 ops NPU / 7 CPU, ~110 fps) |

Cutting the float decode tail out of the graph is what makes the difference: the EP
refuses the full graph outright and accepts 99.2% of the cut one. Post-processing in
numpy costs a further 0.16 ms. Five runs, each recompiling from a cleared cache,
spanned 8.73–9.78 ms. See
[The YOLOv8n blocker, and how it was solved](#the-yolov8n-blocker-and-how-it-was-solved).

### Model size: n vs s, measured together

One calibration size (200 images), plain XINT8, **the full 5000-image val2017 set** —
conf 0.001, per-class NMS at IoU 0.7, max 300 detections. Latency is demo conditions
(conf 0.25), inference only, mean of 20 runs. `./scripts/yolo-bench.sh --no-adaround`
reproduces every cell.

| Model | Device | Latency | mAP@50-95 | mAP@50 | NPU nodes |
|---|---|---|---|---|---|
| yolov8n XINT8, head-cut | NPU | **8.94 ms** (112 fps) | 26.94 | 40.15 | 922 / 929 |
| **yolov8s XINT8, head-cut** | **NPU** | **15.63 ms** (64 fps) | **37.40** | **53.13** | 922 / 929 |
| yolov8n FP32 | CPU | 34.04 ms | 36.69 | 51.64 | — |
| yolov8s FP32 | CPU | 82.33 ms | 44.29 | 60.69 | — |

Three things fall out of that table, and they all point the same way.

**Quantized yolov8s beats float yolov8n on both axes at once** — 37.40 mAP against 36.69,
at 15.63 ms against 34.04. The intuition that a small model is the safe choice for an NPU
is backwards here: the bigger model is more accurate *and* 2.2× faster than the smaller one
on the CPU.

**Latency scales far below FLOPs.** yolov8s is 3.2× the arithmetic of yolov8n (28.6 vs
8.7 GFLOPs) but only 1.75× the latency. yolov8n was estimated at roughly 1.1 of the
array's 16 TOPS from FLOPs/latency alone; a direct measurement (see
[Direct NPU utilization](#direct-npu-utilization-what-gops-actually-says) below) instead
puts it at **9 GOPS — 0.06% of nameplate**, about two orders of magnitude lower, so the
idle headroom the extra width is landing in is far larger than this original estimate
implied. Both models partition identically — 922 NPU / 7 CPU — so this is width being
absorbed, not a different graph. **Do not quote a FLOPs ratio as a latency prediction on
this hardware.**

**The quantization penalty shrinks as the model gets wider:** −9.75 mAP for n
(36.69 → 26.94, a 27% relative loss) but −6.89 for s (44.29 → 37.40, 16%). Some of that
loss is structural and no calibration can remove it — Quark's `enable_npu_cnn` substitutes
hard-swish for SiLU, an architecture change on top of INT8 rounding. So width helps twice:
more accuracy to start with, and less of it lost on the way to INT8.

Not measured here: **AdaRound**. FastFinetune's memory high-water mark is layer 0, the only
layer still at full 640×640, and it exceeds the development machine's 13.8 GB — it takes SIGSEGV
rather than raising. On a machine with more RAM, drop `--no-adaround` and the same script
fills in the other half of the matrix. An earlier 200-image-slice run put AdaRound's
remaining loss at 2.9 mAP, which suggests it recovers most of the gap, but that figure is
from a different measurement and is not comparable to this table.

Earlier versions of this README reported a 200-image slice of val2017. Those numbers were
internally consistent but ran high against published figures — the slice scored FP32
yolov8n at 40.49 where the full 5000 gives 36.69. The table above supersedes them.

**A third width step: yolov8m.** `./scripts/yolo-cut.sh --variant m --limit 64` — head-cut,
plain XINT8, calibration 64 (not the 200 used above — see caveat below): **30.80 ms/frame**
(fixed-image, 20-run harness — matches the single-image live-demo figure), **1216/1223 NPU
nodes.** The FP32 CPU baseline for the same model, same harness: **204.38 ms/frame.** Full
5000-image val2017 mAP, same conf/IoU/NMS settings as the n/s table: **43.49 mAP@50-95,
59.79 mAP@50** — comfortably past yolov8s's 37.40/53.13, continuing the width trend on
accuracy as well as latency. (Small/medium/large-object AP: 26.71/48.58/58.05 — the usual
detector pattern of small objects being hardest, not something width-specific.)

| | Latency | vs FP32 CPU |
|---|---|---|
| yolov8m FP32 | 204.38 ms | CPU |
| yolov8m XINT8, head-cut | **30.80 ms** | **6.6× faster, on NPU** |

yolov8m is roughly 9.1× the FLOPs of yolov8n (78.9 vs 8.7 GFLOPs) for 3.46× the NPU
latency — the same sub-linear pattern as n→s (3.2× FLOPs, 1.75× latency) holding at a
third, considerably larger step, and partitioning is still clean (no new CPU fallback
beyond the usual head-boundary nodes). The CPU-vs-NPU speedup ratio itself grows with
model size — 3.8× at n, 5.3× at s, 6.6× at m — because CPU cost tracks FLOPs roughly
linearly while the NPU absorbs extra width into previously-idle lanes, so the two curves
diverge further apart the bigger the model gets.

Also measured alongside this run: **RAM stayed flat** (system free memory held at
5.0-6.5 GB across the NPU session, no leak, ~1.4 GB delta from the compiled session's
footprint) — no evidence of the AdaRound-style memory pressure this repo has seen
elsewhere. **NPU utilization via Windows' `GPU Engine` performance counter came back
unusable for this workload**: polling every 40-150 ms during a 500-run session (~15.5 s
of actual inference) caught zero nonzero samples, because `Get-Counter` itself costs
roughly 0.5-1 s per call — far coarser than the ~31 ms bursts it was trying to catch. This
is a sharper version of the existing caveat that Task Manager's NPU% is duty
cycle, not compute: for a workload this fast, the standard
Windows tooling can't resolve it at all, not just misrepresent it. The diag report
(1216/1223 nodes) and the measured 6.6× speedup remain the reliable evidence that the NPU
did the work — not this counter.

Caveat: yolov8m's mAP was measured on a calib-64 quantization, not calib-200 like n/s
above, so it isn't a perfectly like-for-like row — calibration size affects the
quantization's exact operating point, though this repo's own finding is that width
dominates the accuracy story far more than calibration sample count does.

**The last two width steps: l and x.** Disk was the blocker (`calib_mb_per_image`
extrapolates ~1-1.5 GB/calibration-image at 640×640), so both are calibrated smaller
than m — 32 images for l, 24 for x — another calibration-size caveat, same reasoning as
m's. `./scripts/yolo-cut.sh --variant l --limit 32` / `--variant x --limit 24`, full
5000-image mAP as above:

| Model | Latency | mAP@50-95 | mAP@50 | NPU nodes |
|---|---|---|---|---|
| yolov8l XINT8, head-cut | **49.67 ms** | 45.37 | 62.34 | 1510 / 1517 |
| yolov8x XINT8, head-cut | **117.11 ms** | 45.09 | 61.37 | 1510 / 1517 |

(`results/map_yolov8{l,x}_cut_xint8_npu.log`, `results/yolo_cut_{l,x}_{cpu,npu,diag}.log`.
`scripts/yolo-cut.sh` writes its demo/diag logs to a fixed filename shared across every
variant, so unlike the mAP logs they don't keep a variant in their name by default —
copied here to `_l_`/`_x_` filenames right after each run so switching variants again
doesn't silently overwrite them, the same trap that briefly clobbered the m row's own
backing log earlier in this session.) l and x share identical NPU node counts because they share the same
architecture depth — only channel width scales between them (width_multiple 1.00 vs
1.25) — so the latency gap (49.67 → 117.11 ms, 2.36×) is purely the extra channels
costing more MACs per node, the same story as every earlier width step.

**The width trend breaks here.** n→s→m each bought a clear mAP gain for its extra
latency (26.94 → 37.40 → 43.49). l→x does not: **45.37 → 45.09, a net loss**, for
2.36× the latency. Whatever accuracy this recipe can extract from width alone tops
out around l — x is pure extra cost with no return, at least under this quantization
recipe (XINT8, no AdaRound, calib 24). Worth checking whether AdaRound or a larger x
calibration run recovers a gap that plain XINT8 can't show here, but on the evidence
so far, model selection past yolov8l should look elsewhere (AdaRound, resolution,
license-plate-specific fine-tuning) rather than further width.

**yolov8l's full eval was flaky in a way nothing smaller has been.** Two of three
5000-image eval attempts crashed mid-run with a hardware `DPU timeout` (`aie_error`,
`ERT_CMD_STATE_TIMEOUT`) on the same subgraph both times; a standalone 500-image run
(152s of continuous inference) and the eventual successful 5000-image run (1532s) both
completed cleanly with **NPU memory flat at 231 MB throughout** (`xrt-smi examine -r
aie-partitions`, sampled every 15s) — so it is not a simple memory leak accumulating
over a long run. Root cause unresolved; not seen at all on n/s/m/x. Worth a closer look
before trusting long unattended l runs in a real deployment.

### iGPU vs NPU: is Ryzen AI worth it over DirectML?

The Radeon 780M/760M iGPU on these same chips is reachable through
`DmlExecutionProvider` with no extra install — this build of onnxruntime already lists
it (`onnxruntime.get_available_providers()` → `['VitisAIExecutionProvider',
'DmlExecutionProvider', 'CPUExecutionProvider']`) — so the real question isn't "can you
use DirectML instead," it's whether the NPU's extra pipeline work (head-cut, calibrate,
quantize, optionally AdaRound, always `--fresh`, all the footguns in
[`docs/DECISIONS.md`](docs/DECISIONS.md)) buys anything DirectML doesn't hand you for
free. `npu/session.py::build_session` now takes `--ep dml` alongside `cpu`/`npu`, so the
same script, same letterbox, same decode, same NMS runs on all three.

**A measurement bug would have inverted this comparison, so it's worth stating what got
fixed first.** `4_detect.py`/`5_eval_map.py` used to time `sess.run` and the cut model's
numpy DFL/anchor decode as one number called "infer." That decode cost is real numpy
work with nothing to do with the EP, and it scales with how many candidates survive
`--conf` — moving from the demo's 0.25 to the eval script's 0.001 alone took the cut NPU
model's measured "infer" from 8.94 ms to 15.01 ms on the same single image, no EP or
hardware change involved. A full-graph model decodes inside the ONNX graph, so its
"infer" never carried this cost — meaning the old numbers penalized exactly the models
this section needed to compare fairly. Fixed by splitting the timed region: "infer" is
now `sess.run` alone for every EP and every graph shape; decode (cut models only) moved
into "post" alongside NMS.

**Latency also drifted between sessions on this shared dev machine** — a NPU burst
measured 12.7 ms in isolation and 6.8-6.9 ms measured back-to-back with everything else
in this table minutes later, on the same model and cache, with no code change between
the two. Background CPU load (other sessions on this machine, not this project's code)
is the suspect. The fix for a comparison, not a single number: interleave every
configuration in one sitting rather than trust a config's latency against a number
committed on a different day. The table below is one such sweep
(`results/bench/lat_yolov8n_igpu_vs_npu_sweep.log`), each model run immediately after
the last, 150 warmed-up runs each.

| Model | Device | Effort to get here | Infer (sess.run only) | mAP@50-95 | mAP@50 |
|---|---|---|---|---|---|
| yolov8n FP32, full graph | CPU | none (baseline) | 22.9-27.9 ms | 36.69 | 51.64 |
| yolov8n FP32, full graph | **DML (iGPU)** | **none** — the export ONNX runs as-is | 12.1-12.8 ms | 36.69 | 51.64 |
| yolov8n FP16, full graph | **DML (iGPU)** | one `convert_float_to_float16` call | **9.9-10.5 ms** | 36.72 | 51.68 |
| yolov8n XINT8, head-cut | NPU | cut head + calibrate + quantize | 6.8-6.9 ms | 26.94 | 40.15 |
| yolov8n XINT8+AdaRound, head-cut | **NPU** | + AdaRound FastFinetune | **6.8-6.9 ms** | **32.19** | **47.04** |

Two things to check before trusting a GPU number at all, both done here rather than
assumed: DirectML registering is not the same as DirectML running the graph — the same
"EP claims the graph and still falls back" trap this repo already hit once with VitisAI
(see the YOLOv8 partitioning section above) is exactly as possible on DML. Every model
in this table logs `All nodes placed on [DmlExecutionProvider]`
(`results/bench/diag_dml_node_placement.log`, verbose session log, `--log 0`) — including
the XINT8 model, which DML happily accepts but gains nothing from: 16.6 ms, slower than
its own FP32 cut model, because DirectML has no dedicated INT8 fast path here and just
pays full QDQ dequantize→compute→quantize overhead around the same float math. And the
FP32→FP16 conversion (`onnxruntime.transformers.float16.convert_float_to_float16`,
`keep_io_types=True` so letterbox/decode stay float32 at the boundary) was checked
against a full 5000-image mAP, not assumed lossless: 36.72 vs FP32's 36.69, within noise.

**The honest answer has two columns, and they point in opposite directions.**
DirectML's best case (FP16, zero quantization work) is 1.3-1.5× slower than the NPU's
best case, full stop — the NPU wins the speed race even against an optimized iGPU path.
But NPU XINT8+AdaRound still costs **4.5 mAP points against FP32** even after the
accuracy-recovery step this repo's own locked decision calls for (32.19 vs 36.69 mAP@50-95,
and that gap is measured on the full 5000, not a slice: an earlier 200-image-slice
estimate for this same recovery step guessed a smaller 2.9-point remaining loss, and per
the standing rule that a slice is not the answer, the full-set 4.5 supersedes it) — against
DML's zero mAP loss at FP16 for one function call and no calibration, no head-cutting, no
`--fresh` discipline, none of `docs/DECISIONS.md`'s pitfall list. If the deployment can
tolerate ~10 ms instead of ~7 ms, DirectML is very plausibly the better trade: most of
the NPU's speed for none of its accuracy cost and none of its tooling burden. The NPU is
worth it specifically when the last ~30-40% of latency matters more than 4.5 mAP points
and the engineering time to chase it — a real case, just a narrower one than "NPU beats
iGPU" alone implies.

Not measured here: DML on the laptop's Radeon 760M (a different, smaller iGPU — this
table is Desktop 2 / Phoenix, Radeon 780M, only); DML with IOBinding (this repo's
existing methodology times plain `sess.run` for every EP including the NPU, so adding
IOBinding for DML alone would make the comparison less fair, not more, even though it
would make DML's own number smaller in isolation); yolov8s/m/l/x on DML.

### Model family: MobileNetV2 vs ResNet50 — does "NPU beats CPU" hold for a cheap-enough model?

Every NPU-wins-on-latency result above (ResNet50, yolov8n) starts from a CPU baseline
in the tens of milliseconds. `mobilenetv2_100.ra_in1k` (timm, depthwise-separable convs,
ReLU6, no SE block — the one op family neither existing pipeline had exercised) was
exported, quantized `XINT8_ADAROUND`, and run CPU/DML/NPU the same way as ResNet50, to
ask what happens once the CPU number itself is already small. Same methodology as the
iGPU-vs-NPU table above: same `npu/session.py::build_session`, same 1000-image eval
slice as ResNet50's own table, `--ep dml`/`--ep npu` added to `pipelines/resnet50/4_run.py`
for this comparison (`results/mobilenet/`).

| Config | Device | top-1 / top-5 | Latency |
|---|---|---|---|
| FP32, full graph | CPU | 74.20% / 89.70% | **1.72 ms** |
| FP32, full graph | DML (iGPU) | 74.20% / 89.70% | 3.19 ms |
| XINT8+AdaRound | NPU | 73.40% / 90.20% | 2.68 ms |

Node placement was checked before trusting the NPU number, same as everywhere else in
this doc: `tools/diag_ep.py` reads 347/349 nodes on the NPU
(`results/mobilenet/diag_mobilenetv2_npu.log`), the 2 CPU nodes being the input
QuantizeLinear/output DequantizeLinear boundary every XINT8 graph pays — not a partial
silent fallback dressed up as "high overhead."

**Plain CPU wins outright here — both accelerators are net negative.** Accuracy is a
wash (the 0.8-point top-1 drop is within the noise this 1000-image/623-class slice
already carries, and top-5 actually improved), so this isn't an accuracy-for-speed
trade like ResNet50 or yolov8n — DML and the NPU are simply slower, on a model already
83% smaller in node count than what either accelerator was built to be worth engaging
for. The likely mechanism: both DirectML dispatch and the VitisAI EP's per-call
setup/copy overhead are roughly fixed costs per `session.run`, and at 1.72 ms of actual
CPU compute there's nothing left for either accelerator's throughput advantage to
amortize against — the same shape as the XINT8-on-DML result above (16.6 ms, slower
than DML's own FP32), just reached from underneath instead of from the INT8-fast-path
side.

That puts a floor under every other result in this document: ResNet50's CPU baseline
(19.7 ms) and yolov8n's (22.9-27.9 ms) are both roughly one to two orders of magnitude
above MobileNetV2's 1.72 ms, and both are where the NPU wins convincingly. The crossover
between "NPU/DML worth it" and "plain CPU wins" sits somewhere between those two
regimes — not measured precisely, but bounded from both sides now. **The practical
answer to "what is this hardware good for": models expensive enough on CPU that a
few milliseconds of fixed accelerator overhead is small by comparison — not every
classifier a phone can already run fine.**

### Pushing width further: `resnetv2_50x3_bit`, and a second way to lose to CPU

The width lever paid off cleanly twice (resnet50 → wide_resnet50_2 → wide_resnet101_2),
so the obvious next step is to push it further and see if it keeps paying. `resnetv2_50x3
.goog_in21k_ft_in1k` (a "Big Transfer" ResNetv2, 3× resnet50's width, timm's own 448²
recommended input) is also the first model in this repo built on **GroupNorm-style
`InstanceNormalization`** instead of `BatchNormalization` — BatchNorm folds into the
preceding Conv at export time, but this architecture's norm layers do not, so they survive
as standalone graph nodes into the quantized model. Exported, quantized `XINT8_ADAROUND`,
and run CPU/NPU the same way as every model above (`results/bit/`):

| Config | Device | top-1 / top-5 | Latency |
|---|---|---|---|
| FP32, full graph | CPU | **84.00% / 96.00%** | **470.79 ms** |
| XINT8+AdaRound | NPU | 82.10% / 95.30% | 588.58 ms |

**84.00% top-1 is the best accuracy this repo has ever measured** — beating
wide_resnet101_2's 80.20% by 3.8 points, at the cost of a CPU baseline nearly 25× heavier
than resnet50's. And yet **the NPU loses to CPU again, by 25%** — a second, structurally
different way to fail the "NPU wins" pattern, this time on a model with plenty of compute
to amortize dispatch overhead. `tools/diag_ep.py` explains why (`results/bit/
diag_resnetv2_50x3_npu.log`): only **1010 of 1271 nodes (79.5%) place on the NPU** — far
below every other model in this doc (ResNet50 99.5%, wide_resnet101_2 99.7%, MobileNetV2
99.4%) — and the 261 CPU nodes are not the usual 2-node input/output boundary. **All 49
`InstanceNormalization` nodes fall to CPU**, along with 3 adjacent Conv nodes and their
Q/DQ scale nodes. The VitisAI EP for this backend has no NPU kernel for
`InstanceNormalization` — checked directly from the report, not inferred from latency.

The mechanism is different from MobileNetV2's: this is not dispatch overhead swamping a
tiny compute cost, it's **49 CPU-executed normalization ops interleaved between NPU
convs**, each transition paying a cross-EP hand-off (copy on/off the NPU) on top of doing
real work on the slower device — worse than running the whole graph on CPU natively,
where the same norm ops fuse efficiently into one runtime with no hand-off cost at all.
Accuracy also degrades further from quantization here (84.00% → 82.10%, a 1.9-point drop)
than any other model in this repo's XINT8+AdaRound rows, plausibly because the 49
unfused normalization layers add extra quantization boundaries AdaRound has to fit.

**Revises the "what is this hardware good for" answer once more**: being compute-heavy
is necessary but not sufficient. The graph also has to be built from ops the VitisAI EP
actually has NPU kernels for — Conv/Add/Mul/Relu/pooling place near-universally in this
repo, but `InstanceNormalization` (and, by the same logic, anything relying on non-fused
normalization — GroupNorm, LayerNorm) is a real, measured gap, not a hypothetical one.
**Every clean NPU win in this repo (ResNet50, wide_resnet50_2, wide_resnet101_2, yolov8n)
shares one property this model breaks: BatchNorm-only, so normalization disappears into
the Conv weights before the graph the NPU ever sees.** That, not just "heavy enough,"
is what "best suited for this hardware" actually means for a classifier.

**The gap is not closed, but it is now measured to be partly closable — with a kernel
the EP doesn't have.** The open-source `mlir-aie` toolchain (see "Key findings" below)
can run bf16 on this array, which Quark/VitisAI EP cannot, so `kernels/groupnorm_bf16/`
is a from-scratch bf16 GroupNorm(32) — exactly the op these 49 nodes are, via a reshape —
using all four columns' shim DMAs. Checked against the real node tensors pulled out of
this model (and ORT's own CPU output), its error is the bf16 output rounding and nothing
else. Per call, kernel NPU time vs the profiled CPU cost of the same node
(`results/aie/groupnorm_bf16_kernel_npu.log`, `results/bit/profile_instancenorm_splice_feasibility.log`):

| Shape (1, 32, L) | Nodes | CPU per call | Kernel per call | Verdict |
|---|---|---|---|---|
| L = 301056 | 3 | 3472 μs | **1535 μs** | kernel, −1937 μs |
| L = 150528 | 5 | 1899 μs | **836 μs** | kernel, −1063 μs |
| L = 75264 | 14 | 989 μs | **510 μs** | kernel, −479 μs |
| L = 37632 | 11 | 496 μs | **351 μs** | kernel, −145 μs |
| L = 18816 | 11 | 233 μs | 262 μs | CPU, by 28 μs |
| L = 9408 | 5 | 118 μs | not run | CPU (kernel floor is ~200 μs) |

33 of the 49 nodes win, worth ~19.4 ms of the op's 42.37 ms per inference — **at the
kernel alone**, single-process. Splicing it into the real model needs a second OS
process (the XRT Python binding is built against Python 3.13, the EP's env is 3.12,
a hard ABI wall), and the first attempt at measuring that handoff's floor
(`results/aie/groupnorm_bf16_handoff_floor_npu.log`) found 789 μs–23.6 ms per call
depending on shape — erasing every one of the 33 wins above. But that floor used
`ml_dtypes.astype()` for fp32/bf16 conversion, and bfloat16 isn't a native numpy
dtype: `.astype()` runs a scalar per-element loop with no SIMD path, so that number
measured a slow conversion function, not a hardware limit. Replacing it with a
strided-view truncation and preallocated buffers
(`results/aie/groupnorm_bf16_handoff_floor_v2_npu.log`) cuts the L=301056 floor from
23.6 ms to 5.7 ms — CPU still wins there, by 1.65× instead of 6.8× — but isolating
the protocol alone (`--no-convert`, zero conversion work, full bf16 payload) measures
**1.19 ms/call, already under CPU's 3.47 ms** at that shape. The shared-memory
handoff was never the obstacle; the host-side conversion was.

That reopens a further question: every one of the 49 InstanceNorm nodes sits inside a
`QuantizeLinear → DequantizeLinear → InstanceNorm → QuantizeLinear → DequantizeLinear`
sandwich (verified programmatically against the real graph, all 49/49, no other float
op riding along on either side) — and re-profiling the real model shows two of those
four QDQ nodes (the input-side quantize and output-side dequantize) are already
optimized away by ORT at runtime, while the other two (input-side dequantize,
output-side quantize) run for real, adding **56.8% on top of InstanceNorm's own
cost**: 65.08 ms total across all 49 nodes, not 42.37 ms, once those two nodes are
counted — 11.1% of the model's 586.32 ms/image latency, not 7.2%. An int8-native
kernel design (int8 in, on-core dequant fused into the stats pass, on-core requant
fused into the affine pass, int8 out) would need to beat that larger 65.08 ms number,
not InstanceNorm alone — a considerably easier bar, and the measured 1.19 ms
protocol floor already clears the hardest single-shape case (L=301056: 5.21 ms) with
room to spare, before any int8-specific payload reduction is even counted.

None of that kernel has been built: the on-core int8→bf16 widen-and-dequant cost is
real and unmeasured, not zero, and the host-side plumbing to extract/reinsert int8
tensors at these exact graph points doesn't exist yet. This is the first version of
this investigation where the projected numbers point to a win rather than a loss —
but it is still a projection. Full details:
`results/aie/groupnorm_bf16_handoff_floor_v2_npu.log`.

### Input resolution: the fixed cost of running the graph at all

The n-vs-s table says width is cheap. This one asks the complementary question — is the
*work* what costs the time? Resolution is the cleaner probe: it changes how much arithmetic
the model does without changing the graph, so the node count, operator mix and partitioning
all stay fixed and pixels are the only variable. Convolution FLOPs are exactly proportional
to input area, so a compute-bound accelerator would run 416² in (416/640)² = 42% of the
640² time.

yolov8n, plain XINT8 head-cut, calibration 32, mean of 20 runs. `./scripts/yolo-res.sh`
reproduces it.

| Input | Mpixels | GFLOPs | NPU nodes | Latency | If it scaled with pixels | Excess |
|---|---|---|---|---|---|---|
| 640² | 0.410 | 8.8 | 922 / 929 | 8.87 ms | — | — |
| 512² | 0.262 | 5.6 | 922 / 929 | 6.39 ms | 5.68 ms | +0.71 |
| 416² | 0.173 | 3.7 | 922 / 929 | 5.19 ms | 3.75 ms | +1.44 |

**It does not scale with pixels.** Cutting the input to 42% of its area cuts latency only to
59%. The excess grows monotonically as the input shrinks, which is the signature of a
constant rather than noise, and a least-squares fit puts a number on it:

```
ms = 2.40 + 15.68 × Mpixels
```

**~2.4 ms of every inference is fixed cost** — 27% of the 640² run, and it buys nothing as
the input gets smaller. That is weight streaming, DMA in and out, and per-layer invocation
overhead: work proportional to the *graph*, not to the data flowing through it. It sets a
floor. Shrinking yolov8n's input to 416² saves 3.7 ms; shrinking it to nothing would save
at most 6.5.

**The partitioning is resolution-invariant.** 922 of 929 nodes on the NPU at all three
sizes, the same split the 640 model has always had. The seven on the CPU are the input
QuantizeLinear and the six output DequantizeLinears — the boundary of the graph, not
anything the EP refused. So resolution is safe to change: it does not risk the wholesale
rejection that the float decode tail caused.

Put beside the width result, the two say one thing. Width is nearly free (3.2× the FLOPs
for 1.75× the time) and pixels are expensive to give back (58% fewer for 42% less time),
because a fixed per-inference cost dominates at this model size. **The lever is to fill
that fixed cost with useful work, not to shrink the work.** Run at full resolution and
spend the headroom on a wider model — which is exactly what the n-vs-s table measured
paying off in mAP.

The 640 row reads 8.87 ms here against 8.94 ms in the table above. Same model and same
recipe; the difference is calibration size (32 vs 200, which cannot affect latency) and
run-to-run variation. mAP is not filled in because this sweep was run for latency;
`./scripts/yolo-res.sh --map` adds it.

### ResNet50 input resolution: does the fixed-cost story hold for a classifier?

YOLO's finding was about detection, where objects stay visible however small the input
gets. A classifier trained at one fixed resolution (224², `crop_pct=0.95` for this
checkpoint) has no such guarantee — shrinking or growing the input changes what the
network sees, not just how much arithmetic it does. `./scripts/resnet-res.sh` runs the
same fixed-cost probe and reports top-1/top-5 in the same pass, so this table answers
latency *and* accuracy together.

Plain XINT8, no AdaRound (a probe recipe, not the 79.8% headline — see caveat below),
calibration 64 images, eval 1000 images, mean latency on NPU.

| Input | Mpixels | NPU nodes | Latency | top-1 | top-5 |
|---|---|---|---|---|---|
| 128² | 0.016 | 392 / 394 | 3.39 ms | 60.50% | 79.80% |
| 160² | 0.026 | 393 / 395 | 4.33 ms | 67.10% | 85.70% |
| 192² | 0.037 | 393 / 395 | 5.00 ms | 71.30% | 86.60% |
| **224²** | 0.050 | 393 / 395 | 5.68 ms | 72.10% | 88.10% |
| **256²** | 0.066 | 392 / 394 | 6.32 ms | **74.00%** | 88.40% |
| 288² | 0.083 | 393 / 395 | 9.32 ms | 71.50% | 89.50% |
| 320² | 0.102 | 393 / 395 | 9.80 ms | 71.20% | 89.30% |
| 384² | 0.147 | 393 / 395 | 11.51 ms | 69.90% | 88.10% |

320² and 384² are outliers past the range the first pass covered, added to check whether
the 288² turn was a real ceiling or a measurement blip. It's real: top-1 keeps falling
past it (71.50 → 71.20 → 69.90) while latency keeps climbing (9.32 → 9.80 → 11.51 ms), so
every size above 256² is strictly worse than 256² on **both** axes at once — not a
trade-off, a dominated region.

```
ms = 2.63 + 65.10 × Mpixels
```

**Fixed cost is 2.63 ms, 23% of the 384² run** — real, but proportionally smaller than
YOLO's 27%, and the fit's marginal cost (65 ms/Mpixel) is far steeper than YOLO's
(16 ms/Mpixel): this graph is comparatively more compute-bound. So resolution is a more
effective latency lever here than it was for detection — shrinking to 128² buys a real
2.3 ms, not YOLO's few-hundred-microsecond scraps.

**Accuracy does not peak at the training resolution.** 256² beats 224² on top-1 (74.00%
vs 72.10%) at only 0.64 ms more — a FixRes-style effect where test resolution slightly
above train resolution helps. It is not monotonic: 256² is the single best point measured
on the whole 128–384² range, and every size on either side of it gives up something. Below
it, latency drops but top-1 falls off fast (60.50% at 128²); above it, both latency and
top-1 get worse together — node count wobbles 392/394 at 256² and 288², suggesting a Q/DQ
boundary node shifts device near the peak rather than a clean scaling story. **256² is the
best measured point on the speed/accuracy frontier for this checkpoint** — beating the
default 224² on both axes at once, and beating every size tried above it on both axes too.

Caveat: this is plain XINT8 at a small calibration set, run to characterize the curve's
*shape* across resolution, not to produce a new headline number. AdaRound at 224² alone
(79.8%) is not comparable to any single row here; extending AdaRound across the sweep is
an open question.

### Model width: does "width is nearly free" hold for a classifier too?

The YOLOv8n-vs-s result said a wider detector costs far less latency than its FLOPs ratio
implies, because the NPU isn't compute-bound at these sizes. That was one data point on
one architecture. `wide_resnet50_2.racm_in1k` — same depth and topology as resnet50, 2×
the bottleneck channel width — is the direct test on the classification pipeline: same
224² input, same plain-XINT8 recipe (calibration 64 images, eval 1000), same
`4_run.py` measuring both latency and accuracy in one pass.

| Model | Params | NPU nodes | Latency | top-1 | top-5 |
|---|---|---|---|---|---|
| resnet50.a1_in1k | 25.6M | 393 / 395 | 5.68 ms | 72.10% | 88.10% |
| wide_resnet50_2.racm_in1k | 68.9M | 393 / 395 | 9.84 ms | 71.50% | 90.10% |
| wide_resnet101_2.tv2_in1k | 126.9M | 767 / 769 | 17.90 ms | **80.20%** | **92.90%** |

**resnet50 → wide_resnet50_2 is the clean test**: same depth (50 layers), only the
bottleneck channel width doubles. 2.7× the params for 1.73× the latency — the same shape
of result as yolov8n→s (3.2× FLOPs for 1.75× latency), on a different architecture and a
different task. Top-1 is a wash (71.50% vs 72.10%, within this recipe's noise) but top-5
improves clearly (90.10% vs 88.10%).

**wide_resnet101_2 pushes width further, but conflates it with depth and recipe** — 101
layers, not 50, and torchvision's newer `tv2` training recipe rather than `a1`/`racm`, so
its jump in accuracy isn't isolated to width the way the first step was. Still, the
latency scaling story holds even at nearly 5× the params: **4.96× resnet50's params for
3.15× the latency** — if anything a slightly *better* ratio than the pure-width step. And
it's the best accuracy number in this repo at any resolution or recipe: 80.20% top-1 with
plain XINT8 and no AdaRound, beating even resnet50's 79.8% AdaRound headline. Partitioning
stays clean at every step (767/769, same 2 CPU boundary nodes as 393/395 and 393/395
above) — three architectures now, and the EP has never refused a wider graph.

This is the second architecture and second task where the same pattern holds: **the lever
that works on this NPU is width, not a narrower model tuned harder or a better
quantization recipe** — and it keeps paying off at nearly 5× the params, not just the
first doubling.

Caveat: plain XINT8, calibration 64 — same caveat as the resolution sweep above. No
AdaRound run yet for either wide model, so none of this is a clean comparison against the
79.8% AdaRound headline for resnet50 on its own terms; wide_resnet101_2 beating it anyway,
without AdaRound, is the more striking reading of that number, not a like-for-like one.

### AdaRound at width: does it recover less on a wider model?

The YOLOv8 finding was that the quantization penalty *shrinks* with width (−27%
relative for n, −16% for s), which predicts AdaRound should have less to recover on a
wider model. `wide_resnet50_2` is also small enough at 224² that AdaRound's RAM wall
(which blocks it for YOLOv8s+ at 640×640) might not even apply here — worth checking
directly rather than assuming. Both models quantized at the same calibration size
(64 images) for a clean paired comparison, matching each one's own plain-XINT8 row
already measured above:

| Model | FP32 | XINT8 | XINT8 + AdaRound | Quantization loss | Recovered |
|---|---|---|---|---|---|
| resnet50 | 80.10% | 72.10% | 79.30% | −8.00 (10.0% rel.) | 7.20 (90.0% of the gap) |
| wide_resnet50_2 | 81.00% | 71.50% | **80.10%** | −9.50 (11.7% rel.) | 8.60 (90.5% of the gap) |

**The prediction doesn't hold, in either direction.** The initial quantization penalty
was actually *worse* for the wider model here (−9.50 vs −8.00, both absolute and
relative) — the opposite of the YOLO n→s pattern — and AdaRound's recovery efficiency
is essentially identical between them (90.0% vs 90.5% of the gap closed). Two
takeaways: first, "quantization penalty shrinks with width" doesn't generalize from
YOLOv8n→s to resnet50→wide_resnet50_2 — it was a real finding on that pair, not a law
of width in general. Second, and more useful in practice: **AdaRound also fits fine at
224² for a 2.7×-wider model** — the RAM wall that blocks it for YOLOv8s+ is specific to
640×640 inputs (where layer 0 is the memory peak and scales with input resolution, not
model width), not to width itself. Both AdaRound quantizations here ran through their
full layer stack on this same 13.8 GB, ~4-5 GB-free box without incident.

`wide_resnet50_2` + AdaRound is now **the best speed/accuracy point in this repo for
classification**: 80.10% top-1 at 9.66 ms, essentially matching `wide_resnet101_2`'s
80.20% (17.90 ms) at roughly half the latency — because `wide_resnet101_2`'s width step
confounded depth and recipe with width, while this one is the clean, isolated width
step, now with AdaRound applied on top of it.

### Width and resolution together: does a wide model at low resolution beat a narrow model at high resolution?

Both levers above were tested one at a time. The obvious next question: combine them —
does `wide_resnet50_2` at a low resolution beat `resnet50` at its own best resolution
(256², the peak from the earlier sweep), on both latency and accuracy at once? That would
be the actual configuration rule for a throughput-sensitive application with small
objects (licence-plate recognition on camera feeds, for example). `wide_resnet50_2` at 160²/224²/288², same
plain-XINT8 recipe:

| Model @ resolution | NPU nodes | Latency | top-1 | top-5 |
|---|---|---|---|---|
| wide_resnet50_2 @ 160² | 393 / 395 | 7.26 ms | 69.10% | 88.50% |
| wide_resnet50_2 @ 224² | 393 / 395 | 9.66 ms | 71.50% | 90.10% |
| wide_resnet50_2 @ 288² | 393 / 395 | 17.45 ms | 70.70% | 89.80% |
| **resnet50 @ 256²** (for reference) | 392 / 394 | **6.32 ms** | **74.00%** | 88.40% |

**The hypothesis is falsified.** resnet50 at its own peak resolution beats every
wide_resnet50_2 configuration tested — including the smallest, cheapest one — on
**both** latency and top-1 at once: 6.32 ms / 74.00% beats even 160²'s 7.26 ms / 69.10%.
Trading resolution for width is not a free swap here.

The reason traces back to the width table above: `wide_resnet50_2`'s top-1 gain over
resnet50 at matched resolution was already a wash in this recipe (71.50% vs 72.10% at
224², a loss if anything) — only top-5 improved. Shrinking its resolution to buy back
latency cannot recover an accuracy edge that was never really there for top-1 in the
first place. The one width step that *did* clearly buy top-1 (`wide_resnet101_2`,
+8.1 points) confounds width with depth and a different training recipe, so it isn't
a clean input to this test — a fair width-for-resolution trade would need a width step
that improves top-1 in isolation, which this repo hasn't measured yet. **The finding
isn't "width for resolution never pays off" — it's "it doesn't pay off for a width step
whose own accuracy gain was already marginal," which turns out to be true of the one
clean width step measured so far.**

### Is the fixed per-inference cost a hardware property or a graph property?

A second question the width×resolution data answers for free: the resolution sweep
found `ms = 2.63 + 65.10 × Mpixels` for resnet50. If that ~2.6 ms fixed cost is a
property of the *hardware* (weight streaming, DMA, per-layer invocation overhead that
doesn't care what the weights are), it should stay roughly the same for a 2.7×-wider
graph. If it's a property of the *graph*, it should scale up with width. Fitting the
same `ms = a + b × Mpixels` model to the three `wide_resnet50_2` points above:

```
ms = 1.88 + 180.94 × Mpixels
```

**The marginal cost nearly tripled** (65.10 → 180.94 ms/Mpixel) — squarely consistent
with a 2.7×-wider graph doing several times the arithmetic per pixel, and a solid result
given the fit residuals are small relative to that gap. **The intercept did not clearly
scale with width — if anything it came out lower** (1.88 ms vs resnet50's 2.63 ms), but
this is not a robust finding: three points fit each other only loosely here (predicted
vs. measured latency differs by 0.7-1.3 ms per point, non-trivial next to a ~2 ms
intercept), against the tighter eight-point fit resnet50's own number came from. **The
marginal-cost result is solid; the intercept comparison is inconclusive and would need
more resolutions per model to settle either way.** Three noisy points are not enough to
claim the fixed cost is hardware-invariant.

### Batching: does it help throughput?

Same question as width and resolution — is there another free lever on this NPU — but
this one comes back negative, and worse than "no speedup": a genuinely **static** (no
`dynamic_axes`, batch baked into the graph shape) batch-2 export of resnet50, quantized
and run the same way as every other row above.

| Config | NPU nodes | Latency/image | top-1 |
|---|---|---|---|
| batch 1, NPU | 393 / 395 | 5.68 ms | 72.10% |
| batch 2, "NPU" (pooled) | **80 / 395** | 71.58 ms | **36.00%** |
| batch 2, CPU (same weights) | — | 53.10 ms | 76.00% |

At batch 2 the EP doesn't reject the graph cleanly — it accepts a small fragment (Conv
itself falls entirely to CPU; only stray Add/Relu nodes stay on the NPU), runs to
completion with a plausible-looking latency, and the pooled top-1 (36.00%) looks like
ordinary degradation — until it's split per within-batch position:

| Batch slot | top-1 | top-5 | Notes |
|---|---|---|---|
| 0 | 72.00% | 88.60% | Correct — matches batch 1 within noise. |
| 1 | **0.00%** | 0.40% | Output is **identical across every image tested** regardless of input. |

**This isn't miscomputation, it's an unwritten output.** Slot 1's logits don't vary with
the input at all — same min/max/std across five different images — which is the
signature of a stale or uninitialized buffer, not corrupted math. The CPU row on the
identical quantized weights proves the model and quantization are fine (76.00%, both
slots correct); the bug is specific to VitisAI writing only the first slot of a batch>1
output tensor. Nothing errors, and the pooled number alone would have suggested "roughly
half wrong" rather than "one slot right, one slot never computed" — the split is what
makes the mechanism legible. The latency number alone (12.6× the batch-1 per-image cost)
already argues against batching; this makes it worse: a batch>1 config isn't just slow,
it silently drops every batch element after the first.

**Do not build a batch>1 config on this backend without an independent `--ep cpu` check,
split per batch index** — a "successful", timed NPU run at batch>1 is not evidence it
computed every element, and a pooled accuracy number can hide a per-slot failure that a
per-slot number would catch immediately.

### Two cameras: does independent concurrency work where batching doesn't?

A static batch axis fails above. A different way of asking for more than one image at
once — two independent `InferenceSession`s, same compiled model and cache, one Python
thread per "camera" — is not the same request to the backend, and behaves completely
differently: it works, and it's faster than doing the two streams one at a time.

`tools/dual_stream_bench.py` feeds session A a known-positive image (a person) and
session B a known-negative one (cars, no person) forever, on separate threads, and
counts any iteration where a result crosses streams — the same class of bug the
batch-2 finding above was, so it's checked directly rather than assumed away.

| Mode | camera A | camera B | combined | Cross-talk |
|---|---|---|---|---|
| solo (one stream at a time) | 75.2 fps | 75.4 fps | — | — |
| round-robin (1 thread, alternating) | 74.8 fps | 76.0 fps | 75.4 fps | 0 |
| **concurrent (2 threads)** | 69.9 fps | 69.9 fps | **139.8 fps** | **0** |

(yolov8n-pose XINT8, head-cut, `results/dual_stream_pose.log`; the detect model shows
the same 1.8× at `results/dual_stream_detect.log`.)

Round-robin gets no speedup at all — expected, if one NPU array serves both queues
strictly in turn. Concurrent threads get **1.8-1.9× the combined throughput** of
round-robin, with zero cross-talk on either model. Each individual call gets ~1 ms
slower under contention (13.3 → 14.3 ms), but two streams together clear far more
frames per second than one stream alone manages twice. That's consistent with the
width finding above: yolov8n only reaches about 6.6% of the array's 16 TOPS solo
(`tools/estimate_tops.py`; see "Splitting the array into independent partitions"
below — this retracts an earlier "~1.1" figure on this row, computed before that
script existed), so a single small model
leaves most of the array idle, and a second independent stream can use that
headroom — some other cost (dispatch/DMA setup, not compute) is what puts a floor
under single-stream latency, and that floor is what two threads partly hide behind
each other.

**Two cameras is a better fit for this hardware than one camera at double the frame
rate would be.** Both open questions above are now answered by `tools/nstream_bench.py`,
which generalizes the same known-positive/known-negative cross-talk check to N
concurrent streams instead of a fixed 2:

| streams | yolov8n combined fps | speedup vs 1 | yolov8m combined fps | speedup vs 1 |
|---|---|---|---|---|
| 1 | 78.7 | 1.00× | 29.2 | 1.00× |
| 2 | 143.2 | 1.82× | 37.9 | 1.30× |
| 3 | 167.2 | 2.12× | 37.8 | 1.29× |
| 4 | 167.4 | 2.13× | 37.8 | 1.29× |
| 6 | 166.6 | 2.12× | 37.7 | 1.29× |
| 8 | 167.4 | 2.13× | 37.7 | 1.29× |

(`results/nstream_yolov8n.log`, `results/nstream_yolov8m.log`; zero cross-talk at every
stream count on both models.)

Both open questions resolve the same way: **the multiplier saturates, and it
saturates lower for the wider model.** yolov8n's combined throughput climbs through 2
streams and flattens at 3 (~167 fps — the array's idle headroom is used up, not that
concurrency "stops working"); yolov8m, which already uses more of the array per call
and has less idle headroom to spare, saturates a stream earlier at a much smaller
1.29×. Neither model regresses below its 1-stream throughput at 8 concurrent streams —
extra streams past the ceiling are free, not harmful, they just don't add anything.
This is the direct, load-bearing confirmation of the "idle headroom" explanation above:
a model that leaves less idle capacity gets less benefit from a second stream, exactly
as the theory predicts, not a coincidence specific to yolov8n/yolov8n-pose.

**Is the saturation compute or memory?** Every concurrent session in the sweeps above
keeps its own runtime buffers, so more streams meaning more NPU memory footprint is a
real alternative (or additional) explanation for the throughput ceiling — not just
"compute headroom used up." Checked directly with `tools/session_hold.py`, which holds
N sessions busy and samples `xrt-smi examine -r aie-partitions` (the XRT tool's own
per-context memory report — Windows' `GPU Engine`/`GPU Adapter Memory` counters can't
see the NPU at all, since it registers as a `ComputeAccelerator` device, not a WDDM GPU
adapter):

| streams | yolov8n memory | yolov8m memory |
|---|---|---|
| 1 | 93 MB | 164 MB |
| 2 | 122 MB | 264 MB |
| 3 | 151 MB | — |
| 4 | 181 MB | 464 MB |
| 6 | 239 MB | — |
| 8 | 298 MB | 865 MB |

(`results/nstream_memory_yolov8{n,m}.log`.) Memory scales **linearly with stream
count** on both models (~29 MB/stream for yolov8n, ~100 MB/stream for yolov8m — wider
activations, more memory per session, consistent with everything else width does) —
and it keeps climbing cleanly through 8 streams with no sign of a ceiling, right past
the point (3 streams for yolov8n, 2 for yolov8m) where throughput already flattened.
**Memory and throughput are decoupled**: if memory capacity were the saturation
mechanism, throughput should have kept climbing until memory hit a limit, or the two
ceilings should track together. Neither happens — throughput caps out while memory
sails past that point still rising. This rules out memory pressure as the explanation
for the saturation curves above. What actually explains it, found later by reading
`xrt-smi`'s full partition report instead of just its memory/GOPS columns (see
[below](#direct-npu-utilization-what-gops-actually-says)): every session under
`4x4.xclbin` — any number of threads or processes — shares one single hardware
partition, so more streams were always time-slicing one physical resource, not
drawing down some abstract pool of "compute headroom."

**Can `xrt-smi`'s GOPS column build a real utilization-vs-TOPS story?** No — it
doesn't measure delivered compute at all on this backend, and that's worth stating
plainly rather than leaving it implied by an unused column. `tools/session_hold.py`
was extended to parse `xrt-smi`'s per-context GOPS field alongside memory, and to
independently count actual `session.run()` completions per stream in the same run
(a ground truth the tool didn't have before — GOPS alone can't be checked against
anything without it):

| streams | yolov8n GOPS | yolov8n measured completions/s | yolov8m GOPS | yolov8m measured completions/s |
|---|---|---|---|---|
| 1 | 9 | 151.4 | 80 | 37.9 |
| 2 | 18 | 171.6 | 160 | 36.1 |
| 3 | 27 | 169.2 | 240 | 36.1 |
| 4 | 36 | 171.1 | 320 | 39.2 |
| 6 | 54 | 168.9 | 480 | 36.2 |
| 8 | 72 | 171.8 | 640 | 39.0 |

(`results/gops_yolov8n.log`, `results/gops_yolov8m.log`.) GOPS is **exactly**
`9 × streams` for yolov8n and `80 × streams` for yolov8m, with no saturation at all
through 8 streams — while the measured completion rate in the same run is flat from
1 stream onward, matching the throughput ceiling already found above. If GOPS were
real delivered array throughput, it would flatten alongside the measured rate once
compute headroom ran out; instead it grows without bound, proportional only to
session count. The simplest explanation consistent with the data: `xrt-smi` credits
each HW context a notional GOPS figure (apparently the model's own nominal op count
times that context's submission rate), computed per-context in isolation, blind to
whatever shared-array contention is actually throttling real throughput. **GOPS is
not a usable proxy for utilization or saturation on this backend** — the honest
conclusion is a negative result, not a new axis of evidence.

That "simplest explanation" turned out to be exactly right, and checkable directly:
`xrt-smi examine -r aie-partitions` reports more than GOPS and memory — it names a
`Partition Index` and the array `Columns` each one claims, fields nothing in this
repo had read before this check. Under `4x4.xclbin`, every session ever built here —
any thread count, any process count — reports the same single
`Partition Index: 0, Columns: [1, 2, 3, 4]`. Two independent OS processes on that one
partition were measured to exactly halve each other's throughput (~149/s solo →
~76/s each running concurrently, combined ~152/s, matching the ceiling above): one
physical resource being time-sliced between contexts, not "compute headroom"
draining down. The "full utilization-vs-TOPS story" roadmap item is closed with
`4x4.xclbin` — but reading those same two fields also opened a real lever the GOPS
column never could have: see
[Splitting the array into independent partitions](#splitting-the-array-into-independent-partitions)
below.

**Does classification show the same shape, and does accuracy itself survive
contention?** `tools/nstream_cls_bench.py` runs the same N-stream sweep on resnet50,
past 8 streams this time (1 through 16), and — because every prior check here only
ever confirmed a binary found/not-found ground truth — every concurrent stream
classifies the exact same 60-image labeled slice the 1-session baseline used, so its
per-image predictions can be diffed against that baseline exactly, not just compared as
an aggregate top-1 number that a different sample could move on its own:

| streams | combined fps | speedup vs 1 | mean top-1 | mismatch vs solo |
|---|---|---|---|---|
| 1 | 67.1 | 1.00× | 81.67% | 0.00% |
| 2 | 102.6 | 1.53× | 81.67% | 0.00% |
| 4 | 105.0 | 1.57× | 81.67% | 0.00% |
| 8 | 107.1 | 1.60× | 81.67% | 0.00% |
| 12 | 107.1 | 1.60× | 81.67% | 0.00% |
| 16 | 107.3 | 1.60× | 81.67% | 0.00% |

(`results/nstream_resnet50.log`.) Same shape as detection: throughput climbs through 2
streams and flattens (1.60× by 8, resnet50's ceiling sitting between yolov8n's 2.13×
and yolov8m's 1.29×, consistent with its own idle-headroom budget), with **zero
throughput regression from 8 to 16 streams**. And critically: **every one of the 960
concurrent classifications (16 streams × 60 images) produced the bit-identical
prediction the uncontended baseline did** — not just "similar accuracy," the literal
same argmax on every image, at every stream count. Contention changes latency, not
outputs, on this backend, on both models tested.

### Direct NPU utilization: what GOPS actually says

Every earlier "not compute-bound" claim in this README was a FLOPs/latency estimate
(model FLOPs ÷ measured latency ÷ 16 TOPS nameplate), never a direct measurement — the
GOPS column in `xrt-smi examine -r aie-partitions`'s per-context report (already used
above for the concurrency memory numbers) had never been read. `tools/gops_sweep.py`
holds one NPU session busy per model and samples it:

| Model | GOPS | % of 16 TOPS | Memory |
|---|---|---|---|
| yolov8n | 9 | 0.056% | 93 MB |
| yolov8s | 29 | 0.181% | 118 MB |
| yolov8m (AdaRound) | 80 | 0.500% | 164 MB |
| yolov8l | 166 | 1.038% | 231 MB |
| yolov8x | 258 | 1.613% | 270 MB |
| resnet50 (AdaRound) | 9 | 0.056% | 99 MB |
| wide_resnet50_2 (AdaRound) | 23 | 0.144% | 144 MB |
| wide_resnet101_2 | 46 | 0.287% | 204 MB |

(`results/npu_utilization_gops.log`.) **The reading tracks FLOPs almost exactly across
the detection width steps** — yolov8n→s→m is 9→29→80 GOPS (3.2×, 8.9×) against FLOPs
ratios of 3.29× and 9.1×. At the time this looked like strong evidence of a real,
consistent relative measurement. It is instead exactly what a compile-time value
derived from the xmodel's own op count would also look like — and the concurrency
section above now shows directly that this is what it is: a per-context notional
figure, not delivered compute.

**Retracted: every absolute %-of-16-TOPS number this repo previously reported from
this table.** They anchored a real-sounding narrative (roughly two orders of
magnitude below the earlier FLOPs/latency estimate) on a counter since shown to be
blind to actual array contention. What survives: the relative FLOPs-tracking shape
above (still consistent with something real at the per-model-compile level), and the
observation that **GOPS keeps climbing from l to x (166 → 258) even though mAP does
not** (see [the width trend breaks here](#model-size-n-vs-s-measured-together) above,
45.37 → 45.09) — worth noting as a compile-time op-count fact about these two
graphs, not as a claim about delivered array utilization.

### Splitting the array into independent partitions

Reading `xrt-smi`'s full partition report (above) rather than only its GOPS/memory
columns showed *why* GOPS couldn't build a utilization story: `4x4.xclbin` always
compiles to one partition claiming the whole array, so every concurrency experiment
in this repo was time-slicing one physical resource, never accessing separate
hardware. `1x4.xclbin` — bundled with the SDK next to `4x4.xclbin`, unused by this
project until now — claims only one column per context. `tools/multi_partition_bench.py`
confirmed N independent OS processes against it get N *separate* `Partition Index`
entries on N different columns, and measured combined throughput with all N held in
a common, confirmed-active window (not reconstructed from staggered runs):

| processes | combined fps | speedup vs 1 | partitions | mismatches |
|---|---|---|---|---|
| 1 | 67.2 | 1.00× | 1 | 0 |
| 2 | 132.7 | 1.98× | 2 | 0 |
| 3 | 190.0 | 2.83× | 3 | 0 |
| 4 | 245.2 | 3.65× | 4 | 0 |

(`results/multi_partition_yolov8n.log`, yolov8n, Desktop 2 / Phoenix.) **245.2 fps at
4 processes beats the single 4x4 partition's own ~151–172 fps concurrency ceiling by
roughly 1.4–1.6×** — a real gain in aggregate throughput on the same silicon, with
zero cross-talk at every process count (checked the same known-positive/
known-negative way as every other concurrency tool here). The cost: each 1-column
context runs roughly half the solo speed of a 4-column one (~15 ms/call vs ~6.6
ms/call), so this is a throughput/latency trade for independent-stream workloads —
useful for N cheap cameras, not for making one stream faster.

**Caveat that has to travel with this result**: AMD deprecated `1x4.xclbin` starting
Ryzen AI 1.5 — "no longer supported and should not be used," per AMD's own release
notes (checked 2026-09-06). It still compiles and runs correctly, with verified zero
cross-talk, on the 1.7.1 install this project depends on — but this is unsupported
territory, not a sanctioned configuration, and nothing guarantees it survives a
future driver or SDK update.

**The 5th concurrent context question is answered: it shares, it doesn't queue or
refuse.** `xrt-smi examine -r platform` reports **Total Columns: 5** on Desktop 2's
Phoenix chip — one more than every sweep above assumed. Extending the process count
to 5 (`results/multi_partition_yolov8n_5col.log`):

| processes | combined fps | vs 1-proc | partitions reported | mismatches |
|---|---|---|---|---|
| 4 | 246.4 | 3.66× | 4 | 0 |
| 5 | 257.2 | 3.83× | **4** | 0 |

`1x4.xclbin` only ever exposes **4** independent partitions regardless of process
count — at N=5 the partition report still lists exactly 4 entries, and the 4th and
5th processes share column 4, each dropping to ~38 fps (half the ~60 fps the other
three columns keep solo) rather than getting a distinct 5th context. Combined
throughput barely moves past N=4 as a result (246.4 → 257.2 fps) — a shared-column
slowdown, not real 5-way scaling. The obvious next step — the driver's own
`5x4_*.xclbin` overlay family under `C:\Windows\System32\AMD`, none shipped in the
1.7.1 SDK's own `xclbins` folder — was tried directly: all three build and run
without error and produce output matching CPU, but `vitisai_ep_report.json` shows
every node on device `"CPU"` — a hardware-target fingerprint lookup fails silently
during session init (`Cannot find or create target with fingerprint=0x...`) and the
whole graph falls back to CPU rather than raising. Matching CPU output was, on its
own, *not* evidence of NPU execution here — caught only by checking the report's
`deviceStat` for a DPU entry, the same discipline the batch-2 and `yolov8s@1280`
findings above required. **Conclusion: the 5th physical column is real but not
reachable from this machine's 1.7.1 install through any xclbin tried** — see
`docs/DECISIONS.md` ("Rejected approaches") for the full record. Still untested:
whether the laptop's Hawk Point chip has the same column count and 5th-column
behavior — Phoenix-only so far.

Caveat, stated plainly: what exactly xrt-smi's GOPS counter counts — raw MACs, some
wider instruction count, a wall-clock average that folds in per-call dispatch overhead
the FLOPs/latency estimate never accounted for — is not documented anywhere this
project has found. The cross-model *ratios* are corroborated against known FLOPs ratios
above and are trustworthy; the absolute %-of-16-TOPS figure should be read as
directional, not a precise utilization number. The `xrt-smi` FPS/Latency columns next to
GOPS were always `N/A` in every sample taken, on this driver version.

### Achieved ops/s: a real answer to "% of 16 TOPS", not a GOPS estimate

`tools/estimate_tops.py` closes the gap the caveat above leaves open. It reads real
MACs per inference off the actual on-NPU graph via `onnx-tool`'s static analysis (the
head-cut model, not Ultralytics' published FLOPs — the cut removed the decode tail,
so the true NPU-side work is strictly less), and multiplies by a measured fps that is
cited from an existing log, never re-derived: `TOPS = MACs × 2 × fps` (AMD's 16 TOPS
nameplate is INT8, 2 ops/MAC). Nothing here depends on `xrt-smi`'s GOPS column at all.

| config | fps | TOPS | % of 16 |
|---|---|---|---|
| resnet50 XINT8 (solo, shared 4x4) | 176.06 | 1.49 | 9.3% |
| yolov8n cut XINT8 (solo, shared 4x4) | 111.86 | 1.05 | 6.6% |
| yolov8s cut XINT8 (solo, shared 4x4) | 63.98 | 1.91 | 11.9% |
| wide_resnet50_2 XINT8 (solo, shared 4x4) | 103.52 | 2.41 | 15.1% |
| yolov8m cut XINT8 (solo, shared 4x4) | 32.47 | 2.64 | 16.5% |
| yolov8x cut XINT8 (solo, shared 4x4) | 8.54 | 2.24 | 14.0% |
| **yolov8l cut XINT8 (solo, shared 4x4)** | 20.13 | 3.40 | **21.2%** (solo peak) |
| yolov8n cut XINT8 (4× independent 1x4 columns) | 245.20 | 2.31 | 14.4% |
| yolov8s cut XINT8 (4× independent 1x4 columns) | 132.70 | 3.96 | 24.8% |
| yolov8m cut XINT8 (4× independent 1x4 columns) | 65.30 | 5.30 | 33.1% |
| yolov8x cut XINT8 (4× independent 1x4 columns) | 23.20 | 6.08 | 38.0% |
| **yolov8l cut XINT8 (4× independent 1x4 columns)** | 37.30 | **6.29** | **39.3%** |
| yolov8x@1280 XINT8, throughput-only (4× independent 1x4 columns) | 6.00 | 6.29 | 39.3% — same ceiling at 6.2× the MACs |
| yolov8s@1280 XINT8 `--limit 4`, throughput-only, **quantization defect confirmed** (4× independent 1x4 columns) | 37.50 | 4.48 | 28.0% — below yolov8m despite more MACs |
| resnet50 XINT8 (4× independent 1x4 columns) | 354.60 | 3.00 | 18.8% |
| wide_resnet50_2 XINT8 (4× independent 1x4 columns) | 181.30 | 4.23 | 26.4% |

Two things fall out of this table that the retracted GOPS numbers never showed:

**Solo achieved compute rises with model size, then turns over at x** — the same
width-stops-paying-off shape already found for yolov8l→x's latency and mAP (see
[the width trend breaks here](#model-size-n-vs-s-measured-together)), now visible in
delivered compute too, not just in the FLOPs-per-latency ratio.

**Splitting across 4 independent columns changes which model wins, and by a lot.**
yolov8m and yolov8l both scale to ~3.8× combined throughput at 4 columns (`results/
multi_partition_yolov8{m,l}.log`) — not the same 3.65× as yolov8n by coincidence, but
better, because a heavier model still has per-column headroom left (the per-column
vs. shared-4x4 latency ratio — 1.66× at n, 1.88× at m, 2.07× at l — stays well under
the 4× a fully compute-bound single column would show, at every size tested). The
result: **yolov8l split across 4 columns reaches 39.3% of the 16 TOPS nameplate, the
best number this repo has measured** — nearly 6× yolov8n's old, now-retracted "~1.1"
GOPS-derived figure. yolov8x's solo regression does not reappear once it gets its own
column (38.0%, statistically tied with l) — confirming that regression was about
contending for the whole array, not a property of the model itself.

**Tested directly whether a heavier model climbs past 39.3%, and it doesn't — this
looks like a real ceiling, not unexhausted headroom.** yolov8x was re-exported at
1280² (imgsz doubled from the repo's usual 640) via `pipelines/yolov8n/1_export.py
--size 1280`, giving 524.1 GMACs/inference — almost exactly 4.0× the 640² model's
131.1 GMACs, confirming the resolution scaling. Split across 4 independent columns
it lands on **39.31% of nameplate** (`results/multi_partition_yolov8x_r1280.log`,
6.0 fps combined, 3.79× at N=4, zero cross-talk) — statistically identical to
yolov8l's 39.32%, despite 6.2× the per-inference compute. Two very different
model/resolution combinations converging on the same figure is real evidence of a
per-column ceiling near 39-40%, not a lever still waiting for a heavier model. The
per-column vs. shared-4x4 latency ratio (1.66× at n, 1.88× at m, 2.07× at l) implying
unused headroom does not translate into a climbing achieved-TOPS figure once a
model is heavy enough to actually test it. This 1280² model is calibration-thin
(`--limit 4`, plain XINT8, chosen deliberately small — see caveat below) and exists
purely to test this ceiling; it has no measured accuracy and should never be cited
for mAP.

**Which of the two candidate causes it is — narrowed, and it's not dispatch.**
`tools/percall_overhead_bench.py` (`results/percall_overhead_yolov8_1x4.log`) turns
on ORT's own profiler on a single held-open `1x4.xclbin` session per model size and
reads the duration ORT reports for the fused on-NPU compute node separately from the
full per-call time. Dispatch/sync overhead outside that node is negligible at every
size tested — 0.7% of wall time at yolov8n, down to 0.1% at yolov8l — so it is not a
fixed per-call cost failing to amortize. Essentially all wall time (95.8%–99.5%) is
the compute node's own reported duration, and *that* duration's efficiency against
an ideal 4-TOPS column climbs with model size the same way the combined-throughput
table above does (19.0%→29.3%→36.5%→41.3%, n→s→m→l, measured by a fully independent
method landing on the same numbers). The ~39-40% ceiling lives inside the compiled
kernel's own scheduled execution on a single column — a real compiler/scheduling
limit, not a host-side dispatch cost that could be amortized away by batching calls
differently.

Same caveats as the partition-splitting result above travel with every number in this
table that used `1x4.xclbin` (deprecated overlay, Desktop 2 / Phoenix only, unverified
on the laptop's Hawk Point chip). yolov8l and yolov8x additionally carry the known
DPU-timeout instability seen on 2 of 3 full 5000-image mAP attempts as an open risk on
long runs, though the ~12s throughput windows measured here did not trigger it.
Building the 1280² model surfaced a new, sharper version of the RAM-wall lesson: the
first attempt used this repo's usual `--limit 64` calibration count and spooled 169 GB
into Quark's calibration cache, driving this 32 GB machine's free RAM to 0.44 GB
before the run had to be killed. `--limit 4` (a throughput probe needs no real
accuracy, so a thin calibration set costs nothing here) kept the peak well clear of
the ceiling. The lesson generalizes the existing SIGSEGV note: **calibration memory
at this backend scales with resolution and calibration count together, not either
alone — a resolution jump needs the sample count re-checked, not carried over from a
lower-resolution recipe.** (`yolov8s@1280` at the same `--limit 4` recipe confirms this
generalizes to width too — its calibration cache peaked at only ~3 GB, since yolov8s's
activations are far smaller than yolov8x's at the same resolution.)

**Raw MACs/call alone does not predict the ceiling — filling the gap surfaced a
counterexample, not a clean second variable.** `yolov8s` split across 4 columns at
its native 640² reaches only 24.8% (`results/multi_partition_yolov8s.log`,
14.93 GMACs, zero cross-talk, the established calibration recipe), and re-exported
at 1280² (same weights, 4.0× the GMACs, confirming the resolution scaling again)
reaches only 28.0% — **below yolov8m's 33.1% despite `yolov8s@1280` having more raw
GMACs/call than yolov8m (59.66 vs 40.6)**. Checked directly against this repo's own
exported graphs (`onnx.load` on each `*_cut.onnx`, counting `Conv` nodes) rather than
assumed from Ultralytics' published multipliers:

| model | Conv nodes | GMACs/call (max tested) | best split-4 % |
|---|---|---|---|
| yolov8n | 63 | 4.70 | 14.4% |
| yolov8s | 63 | 59.66 (@1280) | 28.0% |
| yolov8m | 83 | 40.6 | 33.1% |
| yolov8l / x | 103 | 524.1 (x@1280) | 39.3% |

`yolov8n` and `yolov8s` share the same 63-node depth in this repo's own export (they
differ by channel width only), and achieved-% still climbs a long way within that
one depth class — 14.4% to 28.0%, a bigger swing than the 28.0%→33.1%→39.3% gap
between the depth classes themselves. **This repo has no pair that varies depth at
matched width**, or width at matched depth, across the stock YOLOv8 family: node
count, channel width, and total MACs move together in every variant Ultralytics
ships.

**The follow-up that isolates them ran, and it points at width, not depth.**
`resnet50` (50 layers, narrow) reaches 18.8% split-4 and `wide_resnet50_2` (50
layers — same depth, wider) reaches 26.4%: a clean +7.6-point width effect at
matched depth, similar in size to yolov8n→s's +10.4 points. `wide_resnet50_2` (50
layers) vs. `wide_resnet101_2` (101 layers) is the only width-*matched* pair
collected — checked directly via `onnx-tool` shape inference, not assumed: every
Conv stage's channel count is identical between the two graphs, differing only in
node count. **Doubling depth there buys only +2.1 points (26.4%→28.5%)**, far
weaker than the +8.3 points (24.8%→33.1%) the YOLO node-count table showed for a
similar jump — though that pair's GMACs/call also doubled alongside depth, so it's
"more depth and compute together" vs. nothing, not a depth-alone isolation.
**Width is the better-supported correlate of achieved-% in both families tested.**
*Why* width correlates more strongly was checked once more and left open: the
VitisAI EP's own `vitisai_ep_report.json` records only a static per-node NPU/CPU
routing decision, not cycle counts or tile/lane assignment, and virtually every
node in every model here (narrow or wide) already routes to the NPU — this repo's
tooling has nothing at the granularity that would explain the mechanism, and a
hardware topology narrative built from a spec sheet instead would be unfalsifiable
against these specific compiled graphs. See `RESEARCH.md` finding 5 for the full
numbers, the (weakened, not ruled out) weight-bandwidth-roofline check, the
padding check, and confirmation that AMD's 16 TOPS nameplate is independently
correct for both this machine's 8700G and the laptop's 8645HS.

The `yolov8s@1280` fps above needed its own correctness check first: `multi_partition_
bench.py`'s cross-talk oracle flagged every call as a mismatch, at every process
count *including N=1* where no concurrency is possible — the opposite signature of
real cross-talk. An independent `--ep cpu` comparison (same pattern as the batch>1
finding) confirmed a genuine quantization defect from the thin `--limit 4` recipe
meeting this narrower architecture, reproducible on plain CPU with no NPU involved:
the fps figure is unaffected (Q/DQ scale corruption doesn't change node or MAC count)
but this model must never be cited for mAP or detection accuracy, same as `yolov8x@1280`.

### A live demo: does the multi-partition finding hold on a real webcam?

Every number above for `1x4.xclbin` came from a static-image benchmark
(`tools/multi_partition_bench.py`) feeding pre-loaded frames as fast as each session
could accept them. `tools/webcam_multipartition_demo.py` asks the practical version of
the same question: does splitting into 4 independent single-column NPU sessions
(measured combined throughput 67.2 → 245.2 fps at N=4, yolov8n, above) turn into a
faster *live* demo than the single `4x4.xclbin` session `pipelines/yolov8n/4_detect.py`
uses? Four worker processes each build their own session against `1x4.xclbin` (one OS
process = one HW column, per the sweep above) and round-robin live webcam frames,
dropping a frame rather than queuing it if its assigned worker is still busy — so the
on-screen number is a genuine live rate, not an average over a growing backlog.

Measured on Desktop 2 / Phoenix, camera a Logitech C920s: all 4 workers reached ready,
and `xrt-smi` confirmed 4 active HW contexts — genuine separate-column parallelism, not
one partition being time-sliced. **Live HUD read combined 30.0 fps, ~18.0 ms per
worker.** 30.0 fps is exactly this camera's own native capture rate, measured
separately — the round-robin split is working as designed, but the webcam's frame
delivery, not NPU throughput, is what caps the on-screen number. The static-image
benchmark already put this same model's ceiling at 245.2 fps combined at N=4 — roughly
**8× of headroom sitting unused** here because the camera can't feed frames fast enough
to reach it. This resolves the Roadmap's open "webcam path… not exercised end to end"
item, but not the way that item anticipated: the finding isn't about the NPU or the
round-robin split at all, it's that camera capture is the bottleneck for this exact
demo, and the multi-partition throughput gain would need a faster frame source
(multiple cameras, a video file, or synthetic frames) to actually show up on screen —
*at this model size*.

**Repeating the same live demo at m, l, and x finds where that stops being true**
(`results/webcam_multipartition_yolov8{n,m,l,x}.log`, 15s capture windows each, same
camera): per-worker latency climbs with model size — 18.0 ms (n) → 62.0 ms (m) →
110.6 ms (l) → 176.6 ms (x) — and n/m/l all still land at the camera's 30 fps ceiling,
comfortably under the ~133 ms/worker four workers need to sustain it. x is the first
size where that budget is blown: **combined fps drops to 22.0–23.5, genuinely
NPU-bound rather than camera-bound for the first time in this series.** That live
number lines up closely with the static-image benchmark's 23.20 fps combined for the
same model/xclbin combination (above) — independent confirmation that both numbers are
measuring the same real throughput ceiling, just fed by a webcam instead of pre-loaded
frames.

**The ~90s camera open was OpenCV's backend, not the camera** — a correction to what
this section previously claimed. `cv2.VideoCapture(0)` taking ~90s, and
`cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)` adding another ~178s on top, were both
originally read as "a driver-level renegotiation cost specific to this camera."
They were measured only against OpenCV's *default* Windows backend, MSMF, so they
never separated a slow camera from a slow backend. `tools/cam_probe.py` separates them
by timing two consecutive opens per backend in a fresh process each
(`results/cam_probe_backends.log`, `results/cam_probe_setres.log`):

| backend | open 1 | open 2 | `set(1280×720)` | reports |
|---|---|---|---|---|
| MSMF (OpenCV's Windows default) | 90.02s | 90.20s | 179.34s / 178.24s | 640×480 @ 30.0 fps |
| MSMF, `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` | 0.22s | **0.07s** | **0.02s** | 640×480 @ 30.0 fps |
| DirectShow (`CAP_DSHOW`) | 0.75s | 0.56s | 1.05s | 640×480 @ **0.0 fps** |

Both MSMF opens cost ~90s, so this is a fixed per-open cost, not a cold Windows Frame
Server warmup that a second open would skip — and landing on 90.02s and 90.20s twice
looks like an internal timeout expiring rather than work being done. Disabling MSMF's
hardware transforms removes it, and removes the resolution-change cost with it (~178s →
0.02s), which is consistent with those being one cause rather than two: ~178s is
about twice ~90s, as an internal re-open paying the same timeout twice would be. The
variable has to be set **before `import cv2`** — OpenCV reads it at videoio init, so
setting it afterwards is a silent no-op. That is measured, not assumed: a fourth probe
case sets the variable immediately *after* importing cv2 and still opens in 89.99s and
89.50s, with `os.environ` reading it back as `"0"` the whole time
(`results/cam_probe_late_set.log`). A null result from that ordering is therefore not
evidence against the fix — it is the trap. It also means this cannot be hidden behind
a shared helper in `npu/`: the assignment has to sit above the first cv2 import in each
entry point, including the transitive one through `npu.yolo`. All three camera entry
points now carry it — the round-robin demo, `pipelines/yolov8n/4_detect.py`, and
`pipelines/yolov8n-pose/4_pose.py`. The latter two were worse off than the demo: both
request 1280×720 on the camera path, so they paid the open *and* the resolution change,
~270s of looking hung before the first frame. That is a plausible reason the Roadmap
still lists the `./scripts/yolo-demo.sh` webcam path as never exercised end to end.

The demo now sets it and pins `CAP_MSMF` explicitly. `CAP_DSHOW` is just as fast but
reports `CAP_PROP_FPS` as 0.0, which is precisely the number the camera-bound argument
above rests on. **Demo startup went from ~92s to 1.7s end to end** — and with per-phase
timing now printed, that 92s was all camera: compile-cache check 0.4s, four worker NPU
sessions 1.1s, camera open 0.2s. Re-running the 15s n capture on the fixed path
reproduces the original numbers exactly (30.0–30.5 fps combined, ~18.1 ms per worker,
same 640×480 @ 30 fps mode — `results/webcam_multipartition_yolov8n_msmf_nohw.log`), so
the four logs below remain comparable to anything measured after this change. The demo
still requests no explicit resolution, but that is now a free choice made to keep the
n/m/l/x numbers on one mode, not a cost being avoided; `letterbox()` already handles
arbitrary capture sizes. The n/m/l/x numbers above are backed by
`results/webcam_multipartition_yolov8{n,m,l,x}.log` — the tool now prints combined fps
and per-worker ms to stdout once a second (`--max-seconds` auto-quits an unattended
capture run) instead of only drawing them on the live HUD, which is what the first,
n-only pass through this section had to rely on.

---

## Compatibility

This is a narrow target, and most of the narrowness is not optional.

- **Hawk Point or Phoenix** (XDNA1, `AMD_AIE2_4x4_Overlay`, provider option
  `target: "X1"`). Strix is a different architecture and none of the firmware paths
  here apply to it.
- **Windows.** XDNA1 has no Linux userspace.
- **Ryzen AI 1.7.1 for inference.** Not 1.8.0 — see [Key findings](#key-findings).
- **PowerShell**, not cmd. The scripts print resolved paths at startup so you can see
  when an environment variable did not take.

XDNA1's support matrix is CNN INT8 only: no BF16, no transformer/NLP paths, no LLMs.
That is a vendor-level limit, not a configuration problem.

### Environments

Two conda environments, because the SDK version that can *export and quantize* is not
the one that can *run on the NPU*:

| Env | Clone of | Used for | Key contents |
|---|---|---|---|
| `resnet_env` | `ryzen-ai-1.8.0` | export + quantize | torch, timm, ultralytics, quark 0.11 |
| `resnet_env17` | `ryzen-ai-1.7.1` | **all NPU inference** | onnxruntime-vitisai 1.23.3, opencv, pillow |

```powershell
conda create -n resnet_env   --clone ryzen-ai-1.8.0
conda create -n resnet_env17 --clone ryzen-ai-1.7.1
```

Do not install torch or timm into `resnet_env17`. Keeping the inference environment
thin is what keeps Quark out of the inference import path.

**Always `conda activate` — never call `envs\<name>\python.exe` by its path.** Quark's
import needs `ninja.exe`, which lives in the environment's `Scripts\` directory and
only reaches `PATH` on activation. Without it, `import quark` fails with
`RuntimeError: Ninja is required to load C++ extensions`.

Before any NPU run:

```powershell
conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
```

The 1.8.0 installer sets that variable machine-wide and already-open shells keep the
stale value, so set it per session or pass `--xclbin` explicitly.

---

## Quick install

The common pipelines are wrapped in shell scripts under `scripts/`. They handle
conda activation, the Ryzen AI environment variables, preconditions, logging and
cleanup, so the usual answer is one command:

```bash
./scripts/setup.sh          # one-time: export, fetch datasets, quantize
./scripts/resnet-bench.sh   # the ResNet50 results table below
./scripts/yolo-cut.sh       # YOLOv8n on the NPU: cut, quantize, run, verify
./scripts/yolo-demo.sh      # live webcam detection, q to quit
./scripts/yolo-eval.sh      # COCO bbox mAP table
./scripts/pose-cut.sh       # YOLOv8n-pose on the NPU: same recipe, keypoints
./scripts/pose-eval.sh      # COCO OKS keypoint mAP table
./scripts/diag.sh           # what the VitisAI EP actually took
```

Run them from **Git Bash**, not WSL — the XDNA1 NPU has no Linux userspace, so a
WSL run would silently be CPU-only. Every script takes `--help`. Logs land in
`results/`, in UTF-8; PowerShell's `*>` writes UTF-16, which makes later greps
silently match nothing.

## Manual setup

Not on the scripted path, or want to drive a step by hand? Both pipelines follow the
same `1_export → 2_fetch_data → 3_quantize → 4_run` shape; the commands below reproduce
what the scripts above do.

### ResNet50 (working)

```powershell
conda activate resnet_env
python pipelines/resnet50/1_export.py
python pipelines/resnet50/2_fetch_imagenet.py --n-calib 300 --n-eval 1000 --shards 0 7
python pipelines/resnet50/3_quantize.py --calib-dir data/calib --config XINT8_ADAROUND

conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
python pipelines/resnet50/4_run.py --ep npu --images data/eval --n 1000 --model models/resnet50_xint8_adaround.onnx --fresh
```

For the FP32 CPU baseline: `--ep cpu --model models/resnet50_fp32.onnx`.

Step 2 needs a Hugging Face account approved for the gated `ILSVRC/imagenet-1k`
dataset, plus `hf auth login`.

### YOLOv8n

The head-cut path is the one that reaches the NPU. `1b` removes the float decode tail,
`3b` quantizes the result, and `4_detect.py` decodes in numpy — it switches on the
model's output count, so the same command drives either graph shape.

```powershell
conda activate resnet_env
python pipelines/yolov8n/1_export.py --size 640
python pipelines/yolov8n/2_fetch_coco.py --n-calib 300
python pipelines/yolov8n/1b_cut_head.py
python pipelines/yolov8n/3b_quantize_cut.py --calib-dir data/coco_calib --limit 300

conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx --ep npu --source assets/test_image.jpg --fresh --log 1
python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx --ep npu --source 0
```

`--source 0` is the webcam; press `q` to quit. The COCO download is about 1 GB and is
ungated.

Quantizing at 640×640 spools roughly 105 MB of calibration activations **per image**
into `%TEMP%`, so `--limit 300` peaks near 31 GB and leaves the cache behind if the
process is killed. `./scripts/yolo-cut.sh` checks free space first and cleans up on
exit, including on Ctrl-C; if you run `3b` by hand, watch your disk.

The original full-graph steps (`3_quantize.py`, and `4_detect.py` against
`yolov8n_xint8.onnx`) still work and still produce a model the EP refuses. They are kept
because they are the control that makes the cut model's result meaningful.

---

## Repo layout

```
npu/                    Shared, Quark-free library code
  session.py            CPU/VitisAI session construction, xclbin resolution, cache clearing
  preprocess.py         Classification transform, driven by timm's own data config
  yolo.py               Letterbox, NMS, drawing, COCO class maps, head tensor names
  yolo_decode.py        The yolov8 DFL/anchor decode tail, in numpy
  paths.py              Repo-relative paths, so scripts work from any directory
pipelines/resnet50/     1_export -> 2_fetch_imagenet -> 3_quantize -> 4_run
pipelines/yolov8n/      1_export -> 2_fetch_coco -> 3_quantize -> 4_detect
                        1b_cut_head and 3b_quantize_cut are the head-cut variant
tools/diag_ep.py        Which nodes the VitisAI EP actually took, and why a model may
                        be rejected. Works on both pipelines.
scripts/                Bash wrappers for the common pipelines; lib.sh holds the
                        conda/env/logging/disk-guard helpers they share
docs/DECISIONS.md       Engineering record: locked decisions, rejected approaches, and
                        the YOLOv8 partitioning investigation
results/                Measured outputs, detection images, captured NPU logs
assets/                 Test fixtures
models/  data/          Generated. Git-ignored.
```

`models/`, `data/`, and the `modelcachekey/`, `yolocachekey/` and `yolocutcachekey/`
compile caches are git-ignored: large, machine-specific, and fully reproducible from the
steps above.

Nothing under `npu/` may import Quark. The inference scripts import that package on
every run, and importing Quark triggers a custom-op build attempt each time.

---

## Key findings

Roughly ordered by how much time each one cost to discover.

**Ryzen AI 1.8.0 ships no Phoenix xclbin, and its documentation says otherwise.**
`voe-4.0-win_amd64\` contains only `vaip_config.json`; a recursive search of the whole
install tree finds no xclbin, and the environment's site-packages carries only a Strix
`base.xclbin`. The NPU driver does drop XDNA1 xclbins into `C:\Windows\System32\AMD\`,
but 1.8's EP rejects them with `Cannot find or create target with fingerprint=...`.
Version 1.7.1 still ships `voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin`, and that is
the only firmware observed to work on this chip. Hence the two-environment split.

**Without an explicit xclbin, the compiler silently targets Strix.** It selects
`AMD_AIE2P_4x4_Overlay` and the run then dies at inference with `DPU timeout ...
ERT_CMD_STATE_ERROR`. Setting `target: "X1"` alone does *not* select the chip
architecture; the xclbin does.

**The X1 backend is XINT8 or nothing.** Power-of-two scales, MinMSE calibration,
UINT8 activations with INT8 weights. A8W8 (float scales) does not raise an error — it
just falls back to CPU, and the only symptoms are CPU-level latency and lower accuracy.
**A16W8 (INT16 activations) now measured too, not just assumed dead: same silent
full-CPU fallback** — `tools/diag_ep.py` shows 0/394 nodes on NPU
(`results/a16w8/diag_resnet50_a16w8_npu.log`), and Quark's own quantize log names the
mechanism: this repo's export is pinned to opset 17 (below), and ONNX's `QuantizeLinear`/
`DequantizeLinear` don't support 16-bit types before opset 21, so Quark routes INT16 Q/DQ
through the `com.microsoft` domain instead — which the VitisAI EP's matcher evidently
doesn't recognize at all. Quark's own config dump also shows `A16W8` never sets
`enable_npu_cnn: True` the way `XINT8` does, so this wasn't a close call. Not worth
chasing further: fixing it means bumping export opset (its own trap, see below), for a
config with no shown accuracy edge over `XINT8_ADAROUND`. **BF16 was never attempted
through Quark, and that's a toolchain gap, not a silicon one** — Quark's quantizer for
this backend doesn't expose a BF16 config to try (only
`XINT8`/`A8W8`/`A16W8`/`XINT8_ADAROUND`/`XINT8_ADAQUANT` exist), and AMD's documented
support matrix says XDNA1's *shipped CNN/LLM runtime path* is INT8-only. But the tile
silicon itself is a different question, and now has a primary-source answer rather than
a spec-sheet assumption: AMD's own `OGOAT/Collaterals/device.yaml` (bundled in this same
1.7.1 install, see `results/aie/notes_aie2_device_dtypes.log`) gives Phoenix's `AIE2`
tile spec directly — `macs_per_cycle: bfloat16xbfloat16: 128, int16xint8: 128,
int8xint8: 256`. **The array natively does bfloat16 and int16 arithmetic; the absence
from this repo's results is Quark/VitisAI-EP not exposing it, not the hardware lacking
it.** See RESEARCH.md's "Custom C++ XRT / hand-written AIE kernels" for the full
citation, and for a measured yes on whether a custom kernel reaching those paths is
buildable at all: not through this SDK's own `aiecompiler` (a missing `physical_device.dll`
blocks it, confirmed absent from every AMD distribution channel checked), but through the
open-source `mlir-aie`/Peano toolchain instead — a hand-written kernel compiled and run
correctly on this machine's XDNA1 hardware, natively on Windows, no gated access required.
That path has since produced a kernel for this repo's own gap: a bf16 GroupNorm(32)
(`kernels/groupnorm_bf16/`) standing in for the `InstanceNormalization` that
`resnetv2_50x3_bit` leaves on CPU, measured on the real node tensors at 1535 μs vs the
CPU's 3472 μs per call for the largest shape, and a win on 4 of its 6 shapes (33 of 49
nodes) — but a follow-up measurement of the two-process handoff a real splice needs
found that floor alone (789 μs-23.6 ms/call) erases every one of those wins; see
"Pushing width further" below and `results/aie/groupnorm_bf16_kernel_npu.log` /
`results/aie/groupnorm_bf16_handoff_floor_npu.log`.

**Silent CPU fallback is the failure mode to watch for.** The `[Vitis AI EP]` banner,
`Target architecture:`, `Compile done.` and the operator table print **only during
compilation**, never on a cache load, so their absence in a normal run means nothing
by itself. Two reliable checks: the cache directory should contain a
`compiled.*.xmodel` (ResNet's does, YOLO's does not), and NPU latency should be several
times better than CPU. If it is not, you are running on the CPU. **Caveat found later
(MobileNetV2, see "Model family" below): that second check is a heuristic, not a
guarantee** — a genuinely engaged NPU (347/349 nodes, confirmed via
`vitisai_ep_report.json`) still lost to plain CPU (2.68 ms vs 1.72 ms) on a model cheap
enough that per-call dispatch overhead outweighs the compute saved. The report file is
still the only real evidence; "NPU should be faster" stops being a safe proxy once the
CPU number itself is in the low single-digit milliseconds.

**Always pass `--fresh` when changing model or xclbin.** The compile cache is keyed by
a hardcoded `cacheKey`, not by a model hash, so a stale entry is reused silently and
you end up debugging a wrong-architecture artifact.

**Export with opset 17, static batch 1, and the legacy exporter.** `dynamo=True` (the
default on torch >= 2.9) silently emits opset 18 even when asked for 17; the version
converter's assertion only warns. Dynamic batch axes inject `Shape`/`Concat`/`Reshape`
at the graph tail, which are prime CPU-fallback candidates. Static export yields a
clean `GlobalAveragePool -> Flatten -> Gemm`.

**Preprocessing must match calibration exactly**, which is why it lives in exactly one
place. The ResNet transform is driven by timm's `resolve_data_config` and written to
`models/preprocess_config.json` at export time — this checkpoint uses `crop_pct=0.95`,
not the 0.875 you would get by hardcoding ImageNet defaults.

**Quark's config object is a dataclass**, so assigning a misspelled option succeeds
silently and does nothing. Verify attribute names before trusting that an option took
effect:

```powershell
python -c "from quark.onnx.quantization.config import get_default_config as g; print([a for a in dir(g('XINT8')) if 'exclude' in a.lower()])"
```

**ImageNet-1k on Hugging Face is parquet, not tarballs.** The old
`data/val_images.tar.gz` path returns 404. Validation is 14 shards of roughly 480 MB
and 3570 rows each, in random class order, so a single shard already covers about 974
classes. Read it with `pyarrow.parquet.ParquetFile.iter_batches` and write
`image['bytes']` straight to disk. The `datasets` library pulls in more than a
gigabyte of Arrow and torch just to import, and a non-streaming `load_dataset` wants
150 GB.

**Neither accelerator is free — both carry a per-call floor.** MobileNetV2 (1.72 ms on
plain CPU) lost to both DML (3.19 ms) and a genuinely NPU-engaged run (2.68 ms,
347/349 nodes). ResNet50 and yolov8n's CPU baselines are 10-15x larger, and that is
exactly the regime where the NPU wins — this project's advice to reach for XDNA1 was
always implicitly scoped to "a model heavy enough that a few ms of dispatch overhead
is noise," not to every classifier a CPU already runs comfortably. See "Model family:
MobileNetV2 vs ResNet50" above.

**"Heavy enough" is necessary but not sufficient — the op set matters too.**
`resnetv2_50x3_bit` has a 470.79 ms CPU baseline (no dispatch-overhead excuse available)
and still loses to the NPU, by 25% (588.58 ms), because only 1010/1271 nodes (79.5%)
place on the NPU: every `InstanceNormalization` node in it falls to CPU, confirmed via
`tools/diag_ep.py`, not assumed. Every clean NPU win in this repo shares BatchNorm-only
normalization, which folds into the preceding Conv's weights at export and so never
reaches the graph as its own node. A non-fused normalization layer (GroupNorm,
LayerNorm, InstanceNorm) is a measured gap in this EP's NPU kernel coverage, not a
hypothetical one. See "Pushing width further: `resnetv2_50x3_bit`" above — including the
hand-written bf16 kernel that now beats the CPU fallback for that op on 33 of the 49
nodes, per node, with the whole-model splice still unmeasured.

---

## The YOLOv8n blocker, and how it was solved

For a while the quantized YOLOv8n ran at 37–39 ms on the NPU — indistinguishable from
CPU FP32 — while producing detections whose confidences were visibly shifted from FP32,
so the INT8 model clearly *was* executing. Something was running it, just not the NPU.

**Diagnosis.** The VitisAI EP writes an assignment report to
`<cacheKey>/vitisai_ep_report.json` on every session build — unlike the compile log,
which prints only on an actual compile. `tools/diag_ep.py` reads it. Against the working
ResNet50 pipeline:

```
yolov8n_xint8.onnx                    resnet50_xint8_adaround.onnx
  all           965 nodes               all            395 nodes
  CPU           298                     NPU            393     <-- no NPU entry for YOLO
  VITIS_EP_CPU  667                     VITIS_EP_CPU     2
```

No `NPU` entry at all, and every one of the 965 nodes marked `device: "CPU"`. The EP had
registered, walked the graph and claimed *nothing*. That is wholesale rejection rather
than bad partitioning — there is no partition — which is why no compile log, no operator
table and no `compiled.*.xmodel` were ever produced.

**Cause: the float decode tail.** The last 18 nodes of a YOLOv8 export are the DFL and
anchor-decode arithmetic, kept in float during quantization. Their presence made the EP
refuse the entire graph rather than partition around them.

**Fix: cut them out and decode in numpy.** `1b_cut_head.py` rewrites the model so its
outputs are the six raw detection convolutions (233 nodes → 209), and
`npu/yolo_decode.py` reproduces the removed tail. The result:

```
  yolov8n_cut_xint8.onnx
    all           929 nodes
    NPU           922            <-- 99.2% of the graph
    VITIS_EP_CPU    7
```

The seven CPU nodes are only the boundary conversions: the input `QuantizeLinear` and
the six output `DequantizeLinear`s — structurally the same as ResNet50's two.

| configuration | device | inference | note |
|---|---|---|---|
| full graph, FP32 | CPU | 37.0 ms | |
| head-cut, FP32 | CPU | 31.8 ms | |
| full graph, XINT8 | "NPU" | 39.1 ms | EP took 0 nodes; this was CPU |
| **head-cut, XINT8** | **NPU** | **8.7–9.8 ms** | **922/929 nodes, ~110 fps** |

Post-processing costs 0.16 ms, because the numpy decode filters on class logits before
the DFL softmax rather than decoding all 8400 anchors. Sigmoid is monotonic, so this
selects exactly the anchors NMS would have kept — verified identical, with and without
the filter.

**What was *not* the cause**, each checked rather than assumed:

- **Not the SiLU rewrite.** Quark's `enable_npu_cnn` turns `Sigmoid`+`Mul` into
  `HardSigmoid`+`Mul` (57 of them) and lowers `Split` to `Slice`. This was the leading
  suspect, since none of those ops appear in the ResNet50 graph. The report settles it:
  the EP put all 57 `HardSigmoid`, all 16 `Slice`, both `Resize` and all 13 `Concat` on
  the NPU. Every one of them is supported.
- **Not a misspelled Quark attribute.** `subgraphs_to_exclude` is a real field on the
  legacy `QuantizationConfig`, is consumed by `quantize.py`, and raises rather than
  no-ops when the subgraph doesn't match.
- **Not a missing xclbin.** The blocked run passed the correct Phoenix 4x4 xclbin from
  the 1.7.1 install. (`npu/session.py` now refuses to build an NPU session without one
  regardless — it previously warned and continued, which is a CPU run wearing an NPU
  costume.)
- **Not ORT's constant sharing.** The working ResNet report shows the same merged scalar
  initializers.

Reproduce the whole thing with `./scripts/yolo-cut.sh`, which cuts, quantizes, sanity-
checks on CPU, runs on the NPU and reads the report back.

> **Calibration caveat.** The model above was calibrated on only 32 images,
> because the point of that run was whether the EP accepts the graph — which calibration
> quality has no bearing on. Its detections drift from FP32 accordingly. For a model you
> would actually ship, re-run with `--limit 300 --adaround` and measure COCO mAP; expect
> real accuracy loss from the SiLU-to-hard-swish substitution on top of INT8 rounding.

---

## Known limitations

- **One chip generation, one SDK version.** Everything here targets Hawk Point/Phoenix
  (XDNA1) via Ryzen AI 1.7.1 specifically. Strix (XDNA2) uses a different xclbin and
  compiler target and has not been touched; 1.8.0 cannot run inference on this chip at
  all (no Phoenix xclbin — see [Key findings](#key-findings)).
- **Windows only.** XDNA1 has no Linux userspace; a WSL run is silently CPU-only rather
  than an error, which is the kind of failure that's easy to miss.
- **The full-graph YOLOv8 model is refused by the EP, on purpose left that way.** It's
  kept in the repo as the control proving the head-cut fix is real — see
  [The YOLOv8n blocker](#the-yolov8n-blocker-and-how-it-was-solved) — not a bug to fix.
- **Static batch >1 is unsafe on this backend, not just slow.** Measured: it silently
  drops every batch element after the first rather than raising an error (see
  [Batching](#batching-does-it-help-throughput)). Batch 1 only.
- **AdaRound needs more RAM than the 13.8 GB laptop has for YOLOv8s/m at 640×640** —
  FastFinetune's memory high-water mark is layer 0, the only layer at full resolution,
  and it takes SIGSEGV rather than raising there. Not a hard wall, though: both s and m
  now have AdaRound results (see Roadmap), quantized on Desktop 1's 32 GB + GPU-accelerated
  FastFinetune (`--device`). l/x AdaRound at 640×640 remains untried anywhere.
- **yolov8l's full-dataset eval is flaky.** Two of three 5000-image mAP attempts hit a
  hardware `DPU timeout` mid-run; the third, and a standalone 500-image run, completed
  cleanly with NPU memory flat throughout (ruling out a simple leak). Root cause
  unresolved — see the width section. Not seen on any other model size.
- **NPU utilization can't be measured through standard Windows tooling.** The `GPU
  Engine`/`GPU Adapter Memory` performance counters exist but can't see this device at
  all — the NPU registers as a `ComputeAccelerator`, not a WDDM GPU adapter, so it never
  shows up as an adapter LUID for those counters to poll, regardless of sampling rate.
  `xrt-smi examine -r aie-partitions` (bundled with the driver, `C:\Windows\System32\AMD\`)
  is the tool that actually sees it — live per-context memory (MB) and compute rate
  (GOPS) — and is what the concurrency measurements above use for memory. Its GOPS
  column was tried as a utilization signal too and turned out to be a dead end (scales
  linearly with stream count, decoupled from measured throughput); see the
  [GOPS section](#two-cameras-does-independent-concurrency-work-where-batching-doesnt) above and
  [Roadmap](#roadmap).
- **No formal test suite.** Verification here is empirical (`compileall` + import checks
  as a syntax gate, then real pipeline runs read from `results/`) rather than unit tests
  — there's no fixture NPU to test against in CI.

## Roadmap

Open measurements, roughly in the order they would resolve. `RESEARCH.md` carries the
reasoning behind each.

- **`pipelines/yolov8n-pose` end to end on the NPU — done.** Head-cut partitions
  1015/1025 (99.0%), 9.8 ms/frame, same clean pattern as detect. XINT8 costs 17.8
  points of OKS mAP@50-95 (49.49 → 31.65, a 36% relative loss — proportionally worse
  than bbox yolov8n's plain-XINT8 loss). AdaRound is untried for pose and is the
  obvious next lever, same as it was for detect.
- **AdaRound for YOLOv8s at 640×640 — done, and it barely helps.** Not blocked after
  all: `models/yolov8s_cut_xint8_adaround.onnx` compiles and runs (15.5 ms/frame,
  922/929 nodes). Full 5000-image mAP@50-95 is 39.98 against plain XINT8's 37.40
  (`results/map_yolov8s_cut_xint8_adaround_npu.log`) — 2.6 points, not the 90%
  recovery AdaRound gets on ResNet50/wide_resnet50_2. A 500-image slice run first
  suggested 45.19, which would have been a very different story; the full 5000 is
  the number to trust, consistent with this README's other slice-vs-full warnings.
  Worth understanding why detection AdaRound recovers so much less than
  classification's before spending the RAM on YOLOv8m/l/x or the wide ResNets.
- **AdaRound for YOLOv8m at 640×640 — done, and it recovers even less.** Quantized on
  Desktop 1 (GPU-accelerated FastFinetune, `--device`) and run on Desktop 2's XDNA1
  (Phoenix): `models/yolov8m_cut_xint8_adaround.onnx` runs at the same 30.46 ms/frame as
  plain XINT8 (1216/1223 nodes — no latency cost from AdaRound, only the weight rounding
  changes). Full 5000-image mAP@50-95 is 45.32 against plain XINT8's 43.49
  (`results/map_yolov8m_cut_xint8_adaround_npu.log`) — **+1.83 points**, a smaller
  absolute recovery than yolov8s's +2.58 despite m's much higher starting accuracy,
  extending the pattern that AdaRound has less room to recover as width increases.
  Caveat: this model arrived via Syncthing with no local log of its calibration count,
  so it isn't a clean like-for-like comparison against the calib-64 plain-XINT8 row —
  the exact "a model can arrive with no log explaining it" risk this repo's own
  machine notes warn about.
- **Concurrent streams past 2, and on a wider model — done.** Both saturate:
  yolov8n flattens at 3 streams (~167 fps, 2.1×); yolov8m, which uses more of the
  array per call, saturates a stream earlier at 1.29× (`results/nstream_*.log`).
  Table in the [Two cameras](#two-cameras-does-independent-concurrency-work-where-batching-doesnt)
  section above.
- **Is stream saturation compute or memory — done, it's compute.** `tools/session_hold.py`
  + `xrt-smi examine -r aie-partitions` show NPU memory scaling linearly with stream
  count on both models (~29 MB/stream yolov8n, ~100 MB/stream yolov8m), no ceiling
  through 8 streams — decoupled from the throughput plateau, which rules memory out.
  `results/nstream_memory_yolov8{n,m}.log`.
- **Does classification show the same saturation shape, and does accuracy survive
  contention — done, yes on both.** resnet50 saturates at 1.60× by 8 streams (between
  yolov8n's 2.13× and yolov8m's 1.29×) and holds flat through 16, with the *exact* same
  per-image predictions as the uncontended baseline at every stream count — not just
  similar top-1, bit-identical argmax on all 960 concurrent classifications tested.
  `tools/nstream_cls_bench.py`, `results/nstream_resnet50.log`. Open: `wide_resnet50_2`
  or a bigger classification model untested; >16 streams untested.
- **yolov8l and yolov8x — done.** 49.67 ms / 45.37 mAP@50-95 (l) and 117.11 ms / 45.09
  (x) — the width trend that held cleanly through m flattens here; x is pure extra cost
  for less accuracy than l. Calibrated smaller (32/24 images) than m's 64, a caveat in
  the same vein as m's own. l's full eval is also flaky in a way nothing smaller is —
  see [Known limitations](#known-limitations). Table in the width
  section above.
- **AdaRound across the ResNet50 resolution sweep.** The sweep is plain XINT8, and
  AdaRound's recovery could move where the accuracy peak sits.
- **A yolov8m mAP row at calibration 200**, so the detection width table is
  like-for-like at every size (the current 43.49 was calibrated on 64 images). Same
  caveat now applies to l (32) and x (24).
- **A full utilization-vs-TOPS story using `xrt-smi`'s GOPS column — done, and it's a
  dead end.** GOPS scales exactly linearly with stream count (9×/80× per stream for
  yolov8n/yolov8m) with no ceiling through 8 streams, while the same run's *measured*
  completion rate is flat from 1 stream on — decoupled from real throughput, so it
  can't be turned into a utilization-vs-16-TOPS number. `results/gops_yolov8{n,m}.log`.
  `xrt-smi`'s memory readout (used above) remains the one number from this tool that
  tracks something real; GOPS does not.
- **Explain ResNet50's AdaRound latency cost** (5.63 → 6.93 ms). The EP report shows the
  same 393 / 2 partition for both models, so extra CPU fallback is ruled out; a
  `--fresh` re-run of each and a diff of the two reports would settle it.
- **The webcam path (single `4x4.xclbin` session, `./scripts/yolo-demo.sh`) has not
  been exercised end to end.** The related but distinct round-robin-across-4-columns
  demo *has* — see
  [A live demo](#a-live-demo-does-the-multi-partition-finding-hold-on-a-real-webcam)
  above: camera-bound at 30 fps through n/m/l, genuinely NPU-bound (22.0–23.5 fps) at x.
- **5th AIE column on this Phoenix chip — done, and it's a dead end.** `1x4.xclbin`
  caps at 4 independent partitions regardless of process count; a 5th process shares
  column 4 rather than getting its own. The driver's `5x4_*.xclbin` overlays fall back
  silently to 100% CPU (fingerprint mismatch). See
  [Splitting the array into independent partitions](#splitting-the-array-into-independent-partitions)
  above and `docs/DECISIONS.md`.
- **Longer term:** a detector fine-tuned for fixed camera feeds (licence-plate
  recognition), reusing the head-cut + XINT8 + AdaRound recipe rather than re-deriving it.

## Acknowledgements

Quantization configuration and the AdaRound parameter set are adapted from AMD's Ryzen
AI `CNN-examples/object_detection` samples. Models are timm's `resnet50.a1_in1k`,
`wide_resnet50_2.racm_in1k` and `wide_resnet101_2.tv2_in1k`, and Ultralytics YOLOv8.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Measurements on
hardware not available here are especially useful: Strix, more RAM for the
AdaRound-blocked configurations, more disk for yolov8l/x. There is no unit-test suite,
because the hardware cannot be faked; changes must pass the syntax and import gate in
`CONTRIBUTING.md` and come with a logged measurement under `results/` for anything
behavioural.

## License

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). You're
free to use, study, modify, and share this code — including commercially — but any
version you distribute, or run as a network-accessible service, must also be
open-sourced under the AGPL. This choice follows from a dependency, not a preference:
`pipelines/yolov8n/1_export.py` imports `ultralytics` directly, and Ultralytics'
YOLOv8 code is itself AGPL-3.0 — matching that license here removes the ambiguity of
combining AGPL and permissively-licensed code in one repo, rather than leaving it
unresolved.

This covers the pipeline code only. The ResNet50, YOLOv8n, ImageNet and COCO artifacts
carry their own upstream licenses — none of them are redistributed by this repo
(`models/` and `data/` are git-ignored and regenerated locally).
