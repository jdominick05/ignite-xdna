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
8.7 GFLOPs) but only 1.75× the latency. yolov8n reaches roughly 1.1 of the array's 16 TOPS,
so most of the extra width lands in lanes that were already idle. Both models partition
identically — 922 NPU / 7 CPU — so this is width being absorbed, not a different graph.
**Do not quote a FLOPs ratio as a latency prediction on this hardware.**

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
width finding above: yolov8n only reaches about 1.1 of the array's 16 TOPS on paper
(see [Model width](#model-size-n-vs-s-measured-together)), so a single small model
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
for the saturation curves above and leaves compute headroom as the one still standing.

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
conclusion is a negative result, not a new axis of evidence, and the "full
utilization-vs-TOPS story" roadmap item is closed as not achievable with this tool
rather than left open.

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

**Silent CPU fallback is the failure mode to watch for.** The `[Vitis AI EP]` banner,
`Target architecture:`, `Compile done.` and the operator table print **only during
compilation**, never on a cache load, so their absence in a normal run means nothing
by itself. Two reliable checks: the cache directory should contain a
`compiled.*.xmodel` (ResNet's does, YOLO's does not), and NPU latency should be several
times better than CPU. If it is not, you are running on the CPU.

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
- **AdaRound doesn't fit for YOLOv8s and up at 640×640** on the 13.8 GB development
  machine — FastFinetune's memory high-water mark is layer 0, the only layer at full
  resolution, and it takes SIGSEGV rather than raising when it doesn't fit. Plain XINT8
  only for those configurations.
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
  [GOPS section](#is-the-saturation-compute-or-memory) above and
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
  see [Known limitations](#known-limitations-and-honest-caveats). Table in the width
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
- **The webcam path** (`./scripts/yolo-demo.sh`) has not been exercised end to end.
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
