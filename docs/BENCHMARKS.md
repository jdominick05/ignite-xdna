# Benchmarks

Every measurement in this project, with its full working: what was run, on which machine,
which log backs it, and every caveat attached to the number. `README.md` links into this
file rather than repeating it; if a figure here and a figure there disagree, this file is
the one with the method next to it.

Nothing in here is a summary. Sections are in the order they were measured.

## Results

**ResNet50** — 1000 ImageNet-1k validation images, the same images for every row.
Published top-1 for `resnet50.a1_in1k` is 80.4%.

| Model | top-1 | top-5 | Latency | Device |
|---|---|---|---|---|
| FP32 | 80.10% | 93.90% | 19.7 ms | CPU |
| XINT8 | 71.40% | 88.10% | 40.2 ms | CPU (ORT dequantizes — slower than FP32) |
| XINT8 | 71.70% | 88.40% | 5.63 ms *(superseded, see below)* | NPU (393 ops NPU / 2 CPU, 1 subgraph) |
| A8W8 | 66.60% | 85.10% | 39.0 ms | CPU fallback despite requesting NPU |
| **XINT8 + AdaRound** | **79.80%** | **92.50%** | 6.93 ms *(superseded, see below)* | **NPU** |

Two things worth pulling out of that table. Plain XINT8 costs 8.4 points of top-1,
which is a lot; AdaRound buys back almost all of it. And NPU versus CPU on the *same*
INT8 model agree to within 0.3% — the NPU's numerics are faithful, so any accuracy gap
you see is the quantization, not the hardware.

**The 1.3 ms AdaRound latency cost above does not reproduce, and is retracted.**
`RESEARCH.md` had left this open: same 393-NPU/2-CPU partition on both models, so the
gap couldn't be extra CPU fallback, and nothing else explained it. A same-sitting
`--fresh` rerun of both models, 1000 images each (2026-09-07, Desktop 2):

| Model | top-1 | top-5 | Latency | EP partition |
|---|---|---|---|---|
| XINT8 | 71.90% | 88.50% | **5.26 ms** | 393 NPU / 2 CPU |
| XINT8 + AdaRound | 79.80% | 92.50% | **5.27 ms** | 393 NPU / 2 CPU |

0.01 ms apart — no measured latency cost. `tools/diag_ep.py` against both
`vitisai_ep_report.json` captures shows the partitions aren't just the same size, they're
node-for-node, op-type-for-op-type, device-for-device identical (`results/
adaround_latency_diff_diag_xint8.log`, `results/adaround_latency_diff_diag_adaround.log`).
No log in this repo now reproduces the original 5.63/6.93 ms pair; the leading suspect is
the log-name collision this project has been burned by before (`CLAUDE.md`, "Never let two
machines silently overwrite the same result-log name") — `results/bench_xint8_npu.log` and
`results/bench_xint8_adaround_npu.log` exist today at only 100 images, not the 1000 the
headline table cites, meaning a smaller probe run reused those names after the original.
Treat 5.63/6.93 ms as unverified, not as the number to plan around; the accuracy figures
(71.70/79.80%) do still match a current log and stand. Latency also drifts session to
session on this shared machine (`CLAUDE.md`), so the fair comparison is always the
back-to-back pair above, not either number in isolation.
See `results/adaround_latency_diff_xint8_npu.log`,
`results/adaround_latency_diff_adaround_npu.log`.

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
| **yolov8m XINT8, head-cut** | **NPU** | **26.95 ms** (37 fps) | **43.38** | **59.62** | 1216 / 1223 |
| yolov8n FP32 | CPU | 34.04 ms | 36.69 | 51.64 | — |
| yolov8s FP32 | CPU | 82.33 ms | 44.29 | 60.69 | — |
| yolov8m FP32 | CPU | 144.12 ms | 49.54 | 66.09 | — |

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

Caveat: yolov8m's mAP was originally measured on a calib-64 quantization (`results/wide/map_yolov8m_npu.log`), not calib-200 like n/s above. Re-measuring it under `./scripts/yolo-bench.sh --variants "m" --calib 200 --no-adaround` produces **43.38 mAP@50-95** and **59.62 mAP@50** at **26.95 ms** (`results/bench/map_yolov8m_cut_xint8_c200_npu.log`, `results/bench/lat_yolov8m_cut_xint8_c200_npu.log`, `results/bench/diag_yolov8m_cut_xint8_c200.log`). The mAP delta is just -0.11 points, closing the caveat and proving that calibration sample count past 64 does not meaningfully change the quantization operating point. The full 5000-image FP32 CPU baseline for yolov8m was also measured in this run: **49.54 mAP@50-95, 66.09 mAP@50** at **144.12 ms** (`results/bench/map_yolov8m_cpu.log`, `results/bench/lat_yolov8m_cpu.log`).

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
[`docs/DECISIONS.md`](DECISIONS.md)) buys anything DirectML doesn't hand you for
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

> **The speed half of that has NOT reproduced (2026-09-07, Desktop 2).** A same-sitting
> four-way capture read CPU 27.18 ms, DML FP32 8.60, **DML FP16 6.08**, and **NPU
> XINT8+AdaRound 6.59** — the iGPU faster than the NPU, at 0.92×, while also holding
> 36.72 mAP against 32.19. It won on both axes, reversing this paragraph's conclusion
> (`results/demos/demo_tri_hardware_showdown.log`). What moved is the DML FP16 number:
> 6.08 ms here against the 9.9-10.5 ms in the table above; the NPU's ~6.6 ms is in line
> with what it has always read. The comparison also favours the NPU structurally — the DML
> rows time the full graph with decode inside the timed call, while the NPU row is head-cut
> with decode excluded — so the loss is not an artefact of measuring the NPU unfairly.
> Against that, it is one sitting on a machine whose latency drift is documented, and it
> was not repeated. Both numbers stand, neither is retracted, and the 1.3-1.5× lead should
> not be quoted again without a fresh capture of both paths together. The mAP half of the
> paragraph is unaffected and reproduced exactly (32.19 vs 36.72).
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
a hard ABI wall), and that handoff turns out to be the whole story: measuring its
floor (`results/aie/groupnorm_bf16_handoff_floor_npu.log` — shared-memory ping-pong
of the real byte volume plus fp32/bf16 conversion, no actual NPU dispatch, so this
can only understate the real cost) found **789 μs-23.6 ms per call depending on
shape, which erases every one of the 33 wins above** — 0/49 nodes survive a real
splice. ~90% of the floor at the largest shape is the fp32/bf16 conversion itself,
not the shared-memory transfer, and closing that gap wouldn't be enough either: the
non-conversion residual alone still exceeds every shape's margin except a near-wash
at L=301056. The per-node kernel numbers above stand as measured; the practical
payoff does not, absent an unbuilt cross-node batching scheme to amortize the
per-call floor. The full model's top-1 with bf16 in these nodes is now moot until
that scheme exists.

**Follow-up: this was the wrong shape of op for bf16, so the next check is a
compute-bound one instead.** GroupNorm has no arithmetic intensity — bf16 there only
ever bought a smaller payload, never a faster MAC, which is why the fixes above kept
moving a boundary cost around instead of the real constraint. Checked what a fused
self-attention block (QK²ᵀ → softmax → PV, one xclbin, no host round-trip) would take:
mlir-aie has real bf16 matmul, eltwise, activations, scale_shift, softmax, and swiglu
kernels already validated on this exact chip, but zero bf16 conv2d anywhere and
LayerNorm/RoPE gated to a different chip (Strix) — which picks attention over a CNN as
the next candidate. Ran bf16 matmul on this hardware for the first time to confirm the
primitive is real before building on it: 512×512×512 whole-array (4 columns), **895
GFLOPS, PASS.** See `results/aie/mlir_aie_bf16_matmul_npu.log`.

**Follow-up: Fused BF16 Attention Kernel Built, but Small Sequence Length Exposes the Arithmetic Floor (Negative Result).**
Investigating the hybrid CNN-Transformer architecture `mobilevit_xxs`: stock VitisAI EP
partitioned it into **49 metaDef (58 IPU) thrashing DPU subgraphs, 108.29 ms** (1,037 NPU / 156 CPU /
392 `VITIS_EP_CPU` nodes, the CPU compute being LayerNorm, MatMul, Slice, Squeeze,
Transpose, Reshape) because the DPU overlay has no kernel for those transformer
operators — **14.4× slower than the same FP32 graph under the ORT CPU EP (7.51 ms)**.
Cutting the attention blocks isolated the pure CNN backbone (409 nodes: 407 NPU in
**1 single subgraph**, 2 CPU boundary nodes), executing on NPU in **1.71 ms**
(**3.30×** faster than CPU 5.65 ms, like-for-like). Built the
custom fused BF16 multi-head attention kernel in `mlir-aie` (IRON + Peano): fixed Peano's linker
script stack collision with `Worker(stack_size=2048)`; implemented row-wise FlashAttention streaming
(scratchpad shrunk from 128 KB to 512 bytes); vectorized via 16-lane AIE2 SIMD (`aie_api`); and
scaled across 8 physical cores (Cols 0..3, Rows 2..3) mapped to all 8 physical Shim DMA channels.
The kernel passes bit-accurate numerical verification (<1% rel L2 error, 0 NaN) across all stages
(Stage 4: 0.86 ms, Stage 3: 4.57 ms, Stage 2: 57.61 ms), accounting for the 0.1700% FP32→BF16
quantization floor and 0.235% `fast_exp` approximation error on Stage 3's 10,240 active elements.

**However, the kernel heavily loses to CPU (Negative Result):**
On Stage 3 (8 heads), total arithmetic is only **2.79 MFLOP** — roughly 100× smaller than
MobileNetV2's ~300 MFLOP floor which already lost to CPU. Zen4 AVX-512 executes those 8 heads
in **0.034 ms**, making the 4.57 ms AIE2 kernel **134× slower than CPU** (0.61 GFLOPS achieved,
<0.1% of array compute peak). Running the full model entirely on NPU with this kernel would
take **>120 ms — a projection, not a measurement**: it is the per-stage kernel timings above
summed over the real block counts (2×57.61 + 4×4.57 + 3×0.86 ≈ 136 ms at 8 heads), never run
end to end. It is cited only to say the direction is hopeless, which the per-stage numbers
already establish on their own.

**Why it lost is not what this README first said — and the per-dispatch floor is now
measured.** Every isolated-op verdict here rested on a "~185–200 µs" per-dispatch constant
inherited from one 96 KB probe and never measured on its own.
`kernels/dispatch_floor/measure_floor.py` measures it with a design that has **no compute
tile at all** (shim→memtile→shim), so there is no kernel math to attribute time to —
payloads swept 8 KB–32 MB, output verified per payload, compile excluded
(`results/aie/dispatch_floor_npu.log`):

| | |
|---|---|
| Hardware floor (submit+wait only) | **169.8 µs** (R²=1.0000) |
| Wall floor through `@iron.jit` | **617.0 µs** (R²=0.9998) |
| Host-side, flat in payload size | 447.3 µs |
| Streaming bandwidth | 12.2–13.8 GB/s |

**The old constant was right about the hardware.** What nobody had separated is that a
kernel doesn't *pay* the hardware floor — through the IRON call path it pays **3.6× more**,
and both hand-written kernels were charged that while their write-ups reasoned with 185 µs.

That correction cuts the other way too. At Stage 2, dispatch is **~1% of the measured
57,610 µs** — so "dominated by dispatch overhead" was never true here. The real
cause is kernel design: `attention_kernels.cc` uses **`aie::mmul` zero times**, hand-rolling
dot products with a horizontal `aie::reduce_add` per output element, reaching 0.61 GFLOPS on
hardware this repo measured at **895 GFLOPS**. The row-wise streaming that solved the 128 KB
scratchpad overflow is the same edit that destroyed the arithmetic intensity.
**Rewriting it with `mmul` still would not save it**, which is why the verdict stands: a
perfect 895 GFLOPS kernel gives Stage 2 40 µs + 617 µs = 657 µs against CPU's 240 µs, and
even at the 170 µs hardware floor 210 vs 240 µs is a wash; Stages 3 and 4 lose at both.

The reusable output is a **go/no-go test to run before writing a kernel at all**: the op's
CPU time must exceed **~617 µs** through IRON, or **~170 µs** on a hypothetical
zero-overhead resubmit path. Both are lower bounds — this is a no-compute passthrough, and a
real multi-core kernel's own configuration cost sits inside the hardware bracket. Dispatch
also dominates *everything* below ~0.5 MB: wall time is flat across a 64× payload range.

### The chained int8 CNN also loses — and this time it was measured before anything was built

`ml/resnet/layers_conv2_x` (three ResNet bottlenecks chained core-to-core across three
columns, int8, ObjectFifo→ObjectFifo, **one dispatch for the whole chain**) had run and
PASSed here since 2026-09-06, but its CPU side had never been measured. It was the best
remaining structural idea precisely because it fixes every flaw diagnosed above: the dtype
this repo's XINT8 thesis is about, mlir-aie's own validated int8 conv kernels instead of a
hand-rolled loop, dispatch amortized to ~25% of wall, and no two-process handoff.
436.21 MFLOP, every row captured in one sitting
(`kernels/conv2x_baseline/cpu_baseline.py`, `results/aie/conv2x_int8_cpu_baseline.log`):

| | time | throughput |
|---|---|---|
| NPU int8, hardware bracket | 1869.6 µs | 233 GOPS |
| NPU int8, **end-to-end** | 2497.8 µs | 175 GOPS |
| CPU torch fp32 (8 threads) | 1856 µs | 235 GFLOPS |
| CPU ORT CPU EP, fp32 | 815 µs | 535 GFLOPS |
| **CPU ORT CPU EP, QDQ int8 (VNNI)** | **295 µs** | **1481 GOPS** |

**Like for like — int8 against int8 — the CPU is 6.3× faster than the NPU's hardware
bracket and 8.5× end to end.** The int8 row was verified to genuinely be int8 (ORT's
optimized graph executes 10 `QLinearConv` + 3 `QLinearAdd`), since the whole ratio rests
on it.

**The CPU baseline choice nearly inverted the conclusion.** torch fp32 lands at 1856 µs —
within 1% of the NPU's 1869.6 µs. Benchmarking against torch alone, which is what the
attention kernel did, would have read as *parity* and been wrong by 6.3×. ORT beats torch
by 2.3× on the identical fp32 graph, and its int8 path by another 2.8× on top. Together
with the splice's torch-vs-numpy 9×, that is two for two: **on this project the CPU kernel
choice has decided the verdict more often than the NPU has.** Any NPU-vs-CPU claim here has
to name which CPU implementation it beat.

The scope of that run was narrow on purpose — one shape (32²×64) on three columns — so it
closed *this design at this shape*, not the op class, and named the experiment that would
settle it: sweep standalone `ml/bottleneck` and watch whether GOPS scales.

### It scales, by 23%, against a 570% gap — the op class is closed

`kernels/bottleneck_sweep/` sweeps one bottleneck across spatial sizes on the NPU and runs
the identical arithmetic through ORT int8 on the CPU at the same shapes, both sides in one
sitting (`results/aie/bottleneck_spatial_sweep_npu.log`). `tensor_h` is the free axis —
every L1 buffer scales with `tensor_w`, none with `tensor_h` — and every NPU shape is
checked against mlir-aie's own torch int8 golden before its timings count.

| H×W | MFLOP | NPU hw | NPU e2e | CPU int8 | hw ratio |
|---|---|---|---|---|---|
| 32×32 | 142.6 | 1.222 ms | 1.830 ms | 0.161 ms | **7.6×** |
| 64×32 | 285.2 | 2.238 ms | 2.933 ms | 0.270 ms | 8.3× |
| 128×32 | 570.4 | 4.172 ms | 5.003 ms | 0.583 ms | 7.2× |
| 256×32 | 1140.9 | 8.098 ms | 8.910 ms | 1.219 ms | 6.6× |
| 512×32 | 2281.7 | 15.875 ms | 16.750 ms | 2.783 ms | **5.7×** |

Per-point GOPS *has* to climb with size on any accelerator, because the fixed host cost
amortizes — so the verdict rests on a least-squares fit of `time = intercept + slope ×
FLOPs`, whose `1/slope` is throughput with every fixed cost removed. **NPU marginal 146.1
GOPS against CPU 819.0** (NPU fit r²=0.99999). NPU hardware throughput does rise, 116.7 →
143.7 GOPS over 16× more work, and that 23% is the whole prize for fixing utilization
against a 570% gap. 512×32 is simultaneously the NPU's best point and the CPU's *worst*
(the working set has outgrown cache there) and the CPU still wins 5.7×; every other shape
is worse for the NPU.

**`tensor_w` = 32 was recorded here as a hard ceiling; corrected to 44 (2026-09-07).**
56×56, 32×64, 64×64 and 128×64 all failed in `aiecc`, not at runtime: the skip-add core
needs five buffers plus stack against 64 KB of AIE2 tile memory (`'aie.tile' op allocated
buffers exceeded available memory`). But only four of those five buffers actually scale
with `w`; the fifth (the 1×1+skip weights) is sized by channels and stays fixed — the
"5×w×256 B" estimate overstated the ceiling everywhere except the one width (64) it was
checked at. `conv2dk1.cc`/`conv2dk3.cc` have since been read and both had a real bug: a
width-32-only restriction (`conv2dk3`'s pointer-stride hardcode, `conv2dk1`/
`conv2dk1_skip`'s dead remainder path), fixed and verified end-to-end through
`bottleneck.py` itself at `tensor_w`=36/40/44 (`results/aie/conv2dk3_widthfix_npu.log`,
`results/aie/bottleneck_widthfix_npu.log`, `docs/DECISIONS.md`). So the 32² that
`layers_conv2_x` runs still is not the network's real 56×56 shape — the corrected ceiling
of 44 is closer but still short of it.

**56×56 itself has since been reached (2026-09-07), and the verdict got worse, not
better.** Single-buffering Tile(0,4)'s final output FIFO (`depth=1` instead of 2, in the
local `bottleneck.py` — `skip_buf`'s depth is load-bearing for the skip connection's
timing and was left alone) frees just enough L1 headroom to compile 56×56. Both 32×56 and
56×56 verified against the torch golden. At the real shape: NPU hardware 4.2435 ms vs CPU
0.3327 ms — **CPU wins 12.75×**, worse than the 5.7–11.4× range found at every
compile-limited 32-wide shape (marginal, fixed-cost-removed rate: 111.1 vs 1678.8 GOPS,
15.1×). Reaching ResNet50's actual shape did not narrow the gap; it widened it
(`results/aie/bottleneck_w56_npu.log`).

What that leaves open is narrower and more specific than before: **column count** (both
measurements use 1–3 columns of a 4×5 array; 4×146 ≈ 584 GOPS would still lose, but not by
5.6×) — the one lever the 56×56 result doesn't touch, since it's still a 1-column design.
Kernel quality is no longer an open question in the attention-kernel sense: `conv2dk1.cc`/
`conv2dk3.cc` do vectorize correctly with `aie::mmul`, and the width bug that was in them
is now fixed and verified at the real shape.

**A correction that came out of this sweep:** the conv2x log called its 628.2 µs of host
cost a reproduction of the passthrough's 617.0 µs "to ~2%". Those are different quantities.
617.0 µs is the passthrough's *wall* intercept (447.3 host + 169.8 hardware); the
like-for-like host floor is **447.3 µs**, so 628.2 is 40% over it, not 2% under. The
sweep's own host cost runs 608–874 µs and *grows* with payload, so it is not a flat floor
either. Nothing in either verdict depends on this — dispatch was never the thing to fix —
but the agreement was numerology and is retracted here.

### bf16 GEMM: the first genuine NPU win in this project

Conv lost at 12.75×, even at ResNet50's real shape. Mobile-vision attention lost at
71–240×. At that point the goal stopped being "make the NPU match CPU at every op" and
became "find where it actually has an edge" — the NPU and the CPU are different
hardware and shouldn't be expected to be good at the same things. No CPU bf16/fp32
GEMM baseline existed anywhere in this repo to check that against — every other CPU
number here is int8 QDQ conv. Built one
(`kernels/bf16_matmul_sweep/cpu_matmul_sweep.py`, torch bf16 on this machine's Zen4
cores) and swept the same 4-column bf16 `whole_array.py` design across shapes larger
than the single 512³ point measured earlier.

**At that one shape, the NPU's 895 GFLOPS actually loses to CPU bf16 (1100.6 GFLOPS)**
on this machine (Ryzen 7 8700G, no discrete GPU) — the "895 is a strength" framing was
incomplete without this comparison. But NPU throughput climbs with M/N while CPU bf16
stays close to flat:

| MxKxN | NPU GFLOPS | CPU bf16 GFLOPS | winner |
|---|---|---|---|
| 512×512×512 | 895.1 | 1100.6 | CPU, 1.23× |
| 512×512×1024 | 1339.0 | 1134.5 | NPU, 1.18× |
| 512×512×2048 | 1675.8 | 1146.6 | NPU, 1.46× |
| 1024×1024×1024 | 2072.5 | 1161.2 | NPU, 1.78× |
| 2048×2048×2048 | 1791.9 | 1197.5 | NPU, 1.50× |
| 4096×2048×2048 | 1847.0 | 1247.7 | NPU, 1.48× |

Every NPU number is a verified PASS against numpy `A@B`, not just timed. The crossover
is around N=1024 at M=K=512; past it the NPU wins by 1.18×–1.78×.

**Correction (2026-09-07): "K ≥ 3072 fails correctness, undiagnosed" was the wrong
framing — it isn't a threshold, and it isn't undiagnosed.** Bisecting K in small steps
(not the original coarse 512/1024/2048/4096 grid) shows the error starts continuously
around K/k≈23 tile-iterations (K=1472 at the design's default k=32 tile) and grows
smoothly — 1 element wrong of 262,144 at onset, 4.0% wrong by K=2048, 99.99% wrong by
K=3072 — always a systematic ~10–13% *undercount*, never NaN/garbage. **Root cause: the
K-reduction loop accumulates the running sum directly in a buffer typed `dtype_out`, not
fp32** — with `--dtype_out bf16` (every number in the table above), the partial sum
round-trips through bf16's 8-bit mantissa on every one of the K/k reduction steps, and
once its magnitude swamps a new tile's marginal contribution, that increment is dropped.
AIE2's `aie::mmul` does accumulate one MAC to fp32 natively, but that result is rounded
back to bf16 the instant it's stored for the next iteration to add onto — the fp32
accumulation the hardware is capable of never spans more than one step.
**Fix, and it's free: `--dtype_out f32`** — verified 0/262,144 mismatches at K=3072
(single-core) and clean PASSes at K=2880 and K=4096 on the real 4-column `whole_array`
design, at 1741.7 and 1830.2 GFLOPS — in the same range as the bf16-output numbers
above, no measured throughput cost. K was never the real ceiling; the output dtype was.
See `results/aie/bf16_matmul_k_limit_diagnosed_npu.log`.

This is the "LLM-scale, not mobile-vision" shape `attention_bf16`'s own math predicted
would be needed to make kernel quality (not dispatch overhead) the deciding factor. It
does not by itself mean a fused attention block would win — attention is softmax plus
two data-dependent matmuls, not one static GEMM — but the K ceiling that would have
capped a head_dim×seq_len block near LLM scale is gone with `--dtype_out f32`.
**Checked: `attention_bf16`'s own kernel does not have the bug.** Its `Attn@V` reduction
(`attention_kernels.cc`) already accumulates into `aie::accum<accfloat,16>` — AIE2's
native fp32 hardware accumulator — across the full loop and casts to bf16 once at the
end, the same pattern the matmul fix required. That kernel's 71–240× loss to CPU is
design (zero `aie::mmul` calls) and op size, not precision — no change needed there.

**And the win holds at real production scale, past the old K ceiling.** M=2048,
K=4096, N=4096 — a QKVO/gate-projection-sized contraction dim (Llama-2-7B and
Mistral-7B both use `d_model=4096`), previously an outright FAIL under the bf16-output
default — passes clean with `--dtype_out f32` at **1776.7 GFLOPS vs 1333.1 GFLOPS CPU
bf16 (torch), a 1.33× NPU win**, squarely inside the 1.18×–1.78× range measured at
smaller shapes above. See `results/aie/bf16_matmul_attention_scale_npu.log`. This is
one GEMM in isolation, not a multi-op pipeline measurement — the dispatch-floor
caveats below still apply to any real transformer block built from it.
See `results/aie/bf16_matmul_niche_npu.log`.

**And a real two-matmul pipeline holds too: 1.32× on FFN up-projection → GELU →
down-projection, barely moved from the single-GEMM 1.33×.** Both stages at the same
M=2048/K=4096/N=4096 shape (the real Llama-2-7B `d_ff=11008` width hit a DMA-stride
compile limit at N=8192, not chased further this session — see the log), GELU timed
separately on the (2048,4096) intermediate (0.729 ms — under 1% of either stage's
~39 ms, so it doesn't meaningfully dilute the ratio). Effective pipeline throughput:
**1738.6 GFLOPS NPU vs 1315.6 GFLOPS CPU (torch bf16)**. This is a sum of two
independently measured stage costs plus an isolated activation cost, **not** a live
single-session run with real data handed off between stages — the host-side cost of
reading a real NPU output, running GELU on it, and staging it as the next real input
was not measured, only GELU on a fresh random tensor of the same shape was. Still open:
the wider (non-square) FFN shape, a live single-session pipeline, and attention's own
QK^T/Attn@V shapes (small K=head_dim, K=seq_len) rather than this square GEMM.
See `results/aie/bf16_matmul_ffn_pipeline_npu.log`.

**Correction: at the real (non-square) shape, that pipeline win reverses — CPU wins
1.10×.** The square approximation above dodged a real limit: Llama-2-7B's actual
`d_ff=11008` up-projection (`N=11008`) hit a DMA-stride compile error chasing this down
found **three separate DMA/BD toolchain limits**, not one — a C-output row-block
byte-stride cap of a fixed ~4 MiB (`m × 4 × N × dtype_out_bytes ≤ 2²²`, verified at three
independent points), a B-input per-core tile buffer word-length cap (16,383 words), and a
DMA "too many simultaneously active buffer descriptors" compiler limit on the A-tensor
reload pattern once its repeat count exceeds ~64. `N=11008`'s factorization (2⁸ × 43)
leaves exactly one tile size, `n=64`, that survives all three, and correctness (limit 1)
forces `m=16` — a quarter of the default. That tile-size compromise is what costs the
win: the up-projection alone drops to **909.97 GFLOPS** (vs 1793 GFLOPS for the same
`K=4096` contraction at an unconstrained tile size), while the down-projection, never
tile-constrained, still hits **1800.86 GFLOPS**. Blended: **1192.9 GFLOPS NPU vs 1309.6
GFLOPS CPU (torch bf16) — CPU wins 1.10×**, reversing the square-shape's 1.32× NPU win.
Both stages still verify PASS against numpy — this is a DMA-descriptor/toolchain limit
in `whole_array.py`'s generic tiling, not a precision or `aie::mmul` correctness problem,
and not necessarily true of a shape-specific fused kernel that could pick a different DMA
decomposition. See `results/aie/bf16_matmul_ffn_real_shape_npu.log`.

**Sharpened, not just extended: it isn't "real shapes lose," it's "this specific
integer's factorization decides it."** Two follow-ups on the deferred items above.
(1) `--c-col-maj 1` does dodge the byte-stride limit (compiles clean at default tiles for
`N=11008`) but tops out at **811.69 GFLOPS — worse** than the `m=16` row-major workaround,
because pushing the tile back up to recover throughput hits a fourth limit instead (AIE2's
~64 KiB L1 tile memory, shared by the double-buffered A/B/C tiles) — not a useful lever
here. (2) Mistral-7B's `d_ff=14336` (`2¹¹ × 7`, vs Llama's `2⁸ × 43`) hits the *identical*
`m=16` cap — that limit scales with `N` alone, not its factorization — but its cleaner
factorization admits `n=128` where `11008` was stuck at `n=64`, and that alone is a 74%
throughput jump (835.63 → 1454.37 GFLOPS). Full Mistral pipeline: **1542.9 GFLOPS NPU vs
1362.8 GFLOPS CPU — NPU wins 1.13×**, the opposite verdict from Llama-2-7B on the same
hardware and toolchain. **The real determinant this project can now name precisely:
whether `d_ff`'s factorization admits an `n`-tile ≥~128 once `m` is forced down by the
fixed byte-stride cap** — a property of the specific integer a model architect picked for
unrelated reasons, not of "real-world shape" in general. See
`results/aie/bf16_matmul_ffn_shape_variants_npu.log`.

**Heterogeneous splice, now measured: 3.25 ms, and 2.31× — not the 4.47 ms / 4.1× once
published here.** `tools/splice_wall_clock.py` puts a `perf_counter` around a real
in-process loop (`results/mobilevit/splice_wall_clock_npu.log`, 100 iterations, every row
from the same run because NPU latency drifts between sessions):

| | Latency | |
|---|---|---|
| Cut CNN backbone, **NPU** | **1.71 ms** | 407/409 nodes, **1** subgraph |
| Cut CNN backbone, CPU | 5.65 ms | like-for-like, **3.30× for the NPU** |
| Attention ×9 blocks, CPU (torch, 8 threads) | 1.41 ms | |
| **Splice: NPU backbone + CPU attention** | **3.25 ms** | in-process residual **+0.13 ms** |
| Full model FP32, **ORT CPU EP** | 7.51 ms | → splice is **2.31×** |
| Full model XINT8, stock graph on NPU | 108.29 ms | 1037/156/392 nodes, **49 metaDef (58 IPU)** subgraphs |

Three corrections fall out. **The residual is 0.13 ms, not 0.72** — the old figure was
back-solved from a hardcoded total, and in-process handoff is very nearly free. **The
speedup is 2.31×, not 4.1×**, because the old 18.37 ms baseline was *PyTorch eager* while
the splice ran under ORT; measured here, PyTorch eager is 15.71 ms and ORT CPU runs the
same FP32 graph in **7.51 ms**. Comparing an ORT splice to a PyTorch baseline inflated the
win by ~1.8×. The 108 ms stock-EP figure, by contrast, **reproduced almost exactly**
(108.29 ms).

**The CPU kernel decides the verdict, not the NPU.** The same nine attention blocks cost
1.41 ms in torch and 12.73 ms in plain numpy — 9× — from multithreaded batched GEMM and a
fused softmax. Swap numpy in and the identical splice becomes 14.57 ms, i.e. **0.52×: it
loses to plain CPU.** Any "NPU beats CPU" claim on a heterogeneous pipeline is really a
claim about which CPU kernel you chose to lose to.

> **This is a cost model, not a working pipeline.** `mobilevit_cut_backbone_xint8.onnx` is
> `[1,3,256,256] → [1,1000]` — a complete classifier with the transformer blocks *deleted*,
> not a backbone that hands intermediates to attention. There is no tensor for the CPU half
> to consume, so the two halves are unconnected and the composite computes nothing valid
> (the quantized backbone scores 0% top-1 on its own). It measures what the pipeline *would*
> cost, which is also all the 4.47 ms ever meant. It does **not** use the AIE attention
> kernel, and splitting it across processes (the pyxrt ABI wall) would meet the
> `groupnorm_bf16` IPC floor of 789 µs–23.6 ms and erase the margin outright.

Demo in `scripts/attention-demo.sh` and `tools/demo_attention.py`. See
`kernels/attention_bf16/README.md` and `results/aie/attention_bf16_kernel_npu.log`.

### int8 GEMM: the NPU's headline dtype needs a tile bf16 can't fit

bf16 was the niche found above, but this project is named for int8, XDNA1's 16 TOPS
nameplate is an int8 figure, and AMD's own tile spec gives the AIE2 core twice the int8
MACs per cycle (256 vs 128 bf16). Every int8 number under `kernels/` came from the chained
conv design; nobody had run the same `whole_array.py` GEMM in int8, and there was no CPU
int8 GEMM baseline to read it against — every CPU int8 number here is ORT QDQ conv. Both
were run in one sitting (`kernels/int8_matmul_sweep/`, 2026-09-07, Desktop 2): the
upstream design unmodified with `--dtype_in i8 --dtype_out i32` (i32 because the K
reduction accumulates in a `dtype_out` buffer — the bf16 lesson above), bf16→f32 re-run
alongside it, and a CPU script that times **two** int8 kernels — torch `_int_mm`
(int8×int8→int32) and ORT `MatMulInteger` u8s8 (MLAS, the library behind every other CPU
int8 number here) — because this repo has twice lost a verdict to the slower CPU kernel.
torch's is faster at every shape (1.05–1.68×) and is the verdict line. Every int8 NPU row
is a bit-exact PASS (`np.array_equal`), stronger than the bf16 tolerance check.

**At the tile the design ships with (m=64/k=64/n=32), int8 loses to the CPU's own int8
kernel almost everywhere, and runs only 1.1–1.5× the bf16 rate, not 2×.** Same sitting:

| MxKxN | NPU int8 GOPS | NPU bf16 GFLOPS | int8/bf16 | CPU int8 GOPS (torch, mean) | NPU/CPU int8 |
|---|---|---|---|---|---|
| 512³ | 982.52 | 846.40 | 1.16× | 2053.4 | 0.48× |
| 1024³ | 2743.72 | 1875.30 | 1.46× | 2267.7 | 1.21× |
| 2048³ | 2373.66 | 1775.65 | 1.34× | 2427.3 | 0.98× |
| 4096×2048×2048 | 2387.01 | — | — | 2690.4 | 0.89× |
| 2048×4096×4096 | 2047.66 | 1801.18 | 1.14× | 2757.9 | 0.74× |
| 512×4096×4096 | 1875.49 | 1671.94 | 1.12× | 2647.8 | 0.71× |
| 1024×4096×4096 | 1956.97 | 1779.78 | 1.10× | 2563.3 | 0.76× |

What bounds the default tile is dtype-blind: the design re-streams the whole A matrix from
DDR once per column block (N/(n·4) times) and B once per row block (M/(m·4) times), each
k-step handshakes two tiles through two FIFO levels, and the output tile it
read-modify-writes is 4 bytes per element for i32 and f32 alike. Halving the MAC
instruction count buys little against that.

**What int8 has that bf16 does not is L1 headroom, and it is worth 2×.** int8's A/B tiles
are half the bytes, so its default tile uses 32 KB of the 64 KB tile memory (bf16: 44 KB)
and `n=64` fits at 52 KB. A tile check at 2048³ (all bit-exact): `k=128` +20%, `m=128`
+37%, **`n=64` +98% — 4603 GOPS**, because every doubling of `n` halves the A re-streaming.
The same `n=64` in bf16 needs 68,864 B against a 65,536 B tile: it misses by exactly the
3,328 B stack (`'aie.tile' op Basic sequential allocation failed`, and so do `m=128` bf16,
`m=128 n=64` int8 and `k=128 n=64` int8). The full int8 sweep at `n=64`:

| MxKxN | m | NPU int8 GOPS, n=64 | vs default | CPU int8 GOPS (mean) | NPU/CPU int8 | CPU int8 best case (min) | NPU/CPU at CPU's min |
|---|---|---|---|---|---|---|---|
| 512³ | 64 | 974.29 | 0.99× | 2053.4 | 0.47× | 2218.5 | 0.44× |
| 512×512×1024 | 64 | 1764.51 | 1.15× | 2161.4 | 0.82× | 2200.3 | 0.80× |
| 512×512×2048 | 64 | 2544.29 | 1.28× | 2312.9 | 1.10× | 2359.9 | 1.08× |
| 1024³ | 64 | 3544.29 | 1.29× | 2267.7 | 1.56× | 2304.2 | 1.54× |
| 2048³ | 64 | 4447.97 | 1.87× | 2427.3 | 1.83× | 4200.5 | 1.06× |
| 4096×2048×2048 | 64 | 4607.05 | 1.93× | 2690.4 | 1.71× | 4189.2 | 1.10× |
| 2048×4096×4096 | 64 | 3263.99 | 1.59× | 2757.9 | 1.18× | 4033.9 | 0.81× |
| 128×4096×4096 | 16 (forced) | 1091.53 | 1.86× | 2868.7 | 0.38× | 3028.9 | 0.36× |
| 256×4096×4096 | 32 (forced) | 1888.39 | 1.81× | 2859.2 | 0.66× | 3167.5 | 0.60× |
| 512×4096×4096 | 64 | 3160.96 | 1.69× | 2647.8 | 1.19× | 3610.7 | 0.88× |
| 1024×4096×4096 | 64 | 3282.40 | 1.68× | 2563.3 | 1.28× | 3698.2 | 0.89× |

**The verdict, by this repo's mean-based convention: with `n=64`, NPU int8 beats the CPU's
best int8 kernel 1.10×–1.83× at every shape with M ≥ 512 and N ≥ 2048**, and still loses at
512³, 512×512×1024 and short prefill (M=128: 2.6×, M=256: 1.5×). 4607 GOPS is the highest
ops rate any single dispatch has reached in this project (28.8% of the 16 TOPS nameplate;
the 39.3% above is a whole-graph figure across four columns), and 2.5× the bf16 rate of the
same sitting at 2048³ — the 2× the MAC count promised, but only once the tile bf16 cannot
have. **The margin is thin at the largest shapes:** torch's kernel has a large mean/min
spread there (2048³: 7.08 ms mean, 4.09 ms min), and read against its best case the
2048-class wins hold at 1.06–1.10× while the K=N=4096 rows become a 1.13–1.24× CPU win.
In the same sitting the same design in bf16 beat CPU bf16 by 1.13×–1.35× at M ≥ 512, N ≥ 1024 (CPU
bf16 came in ~15% higher than in the sweep above — drift on the CPU side, which is why that
range is narrower than 1.18×–1.78×). So int8 GEMM is a second genuine niche, about twice
the bf16 one in absolute throughput, and relative to its own CPU competitor no wider than
bf16's at the largest shapes (1.18× vs 1.19× at 2048×4096×4096).

**The M-edge is a tile artifact.** `whole_array` needs `M % (m·4) == 0` and `(M/m/4) % 2
== 0`, so M=128 forces m=16 and M=256 forces m=32; control rows at M=512 with the same
forced m match the small-M rows to within 6% at both dtypes (int8 m=16: 585 at M=128 vs
610 at M=512; m=32: 1041 vs 1046). Throughput tracks `m` — 610 → 1046 → 1875 for 16 → 32
→ 64 — not token count. A prefill shorter than ~512 tokens at d_model=4096 loses at either
dtype on this design; only a differently tiled design could change that.

### bf16 GEMM at n=64: the same fix int8 used

int8's `n=64` win above came from L1 headroom bf16 didn't have — bf16's own `n=64` misses
the 65,536 B tile by exactly 3,328 B (68,864 B needed). The fix is a 13-line patch to
`whole_array.py` (not part of this repo; lives in the local `~/mlir-aie` checkout), added
2026-09-07: a `--c-single-buffer {0,1}` flag that drops the per-core `C_L1L2` output-tile
FIFO from depth 2 to depth 1. That FIFO is not the A/B DMA re-stream path double-buffering
earns its keep on — `core_fn` acquires it once, accumulates `K/k` matmuls into it, and
releases it once per output tile, so there is no compute/compute overlap to lose, only
compute/next-tile-DMA-out overlap this patch gives up. It frees exactly
`m·n·dtype_out_bytes` of L1 (16,384 B at m=n=64, f32 out) — precisely the shortfall.

| MxKxN | NPU bf16 GFLOPS, n=64 | vs default tile (n=32) | CPU bf16 GFLOPS (mean) | NPU/CPU bf16 |
|---|---|---|---|---|
| 512³ | 910.72 | 1.08× | 1309.7 | 0.70× |
| 1024³ | 2029.93 | 1.08× | 1570.8 | 1.29× |
| 2048³ | 2477.23 | 1.39× | 1313.7 | **1.89×** |
| 2048×4096×4096 | 2641.41 | 1.47× | 1517.5 | 1.74× |

All four PASS at this repo's standing bf16/f32 tolerance. An overlap-cost control —
default tile (`n=32`) with `--c-single-buffer 1` at the last row's shape — reads 1740.51
GFLOPS against 1801.18 at the normal double-buffered depth, a 3.4% loss: single-buffering
`C_L1L2` does cost a little overlap when L1 headroom was never the constraint, which is
what confirms the `n=64` gain above is the bigger tile reaching the array more
efficiently, not an accident of the buffer-depth change itself.

**The win margin against CPU bf16 widens from 1.19×–1.35× (default tile, the standing bf16
GEMM headline) to 1.29×–1.89× at M, N ≥ 1024** — 2048³'s 1.89× is the largest bf16 GEMM
margin measured in this project. 512³ still loses (0.70×, barely moved from the default
tile's 0.65×) — the same small-shape verdict every GEMM result here has shown. See
`results/aie/bf16_matmul_n64_single_buffer_npu.log`.

**Superseded the same day (kept as written):** the C tile was single-buffered after all, once the other session had finished with `whole_array.py` — the section above — and the tile sweep it enables is the next section. As it stood: bf16 at `n=64` is one line away — single-buffering the C output
FIFO (`depths=[1]`, the change that got `bottleneck.py` to 56×56) frees 16 KB — but
`whole_array.py` is the shared upstream file another live session was running its FFN
measurements through, and editing it under them would silently change their numbers. If
bf16 gains at `n=64` what int8 gained, the bf16 niche roughly doubles too. Also not
measured: the int8 requantization epilogue a real quantized layer needs (upstream's i8→i8
path accumulates in an int8 buffer across K and is unusable past one k-tile), `--n-aie-cols`
< 4, int16. ORT MatMulInteger's 1.05–1.68× deficit to torch here does not reopen the closed
int8 conv class — GEMM is not conv. See `results/aie/int8_matmul_sweep_npu.log`.

### The bf16 tile sweep: what the freed 16 KB buys, and where the B/MAC model stops

Single-buffering the C output tile (`--c-single-buffer 1`, the local `whole_array.py` patch
in `kernels/gemm_tile_sweep/`) frees 16 KB of the 64 KB L1 per core. The section above spent
it on `n=64`. This sweep asks what else it reaches, and tests `docs/SILICON.md` 3.1's
bytes-per-MAC ceiling model against every tile the generic 4×4 design can now compile:
ten bf16 m/k/n tiles at 2048³, three of them with B column-major, the four survivors across
1024³, 4096×2048×2048 and 2048×4096×4096, three int8 tiles, and a same-sitting CPU bf16
baseline (torch 2.14, 8 threads). Desktop 2, clean sitting 2026-09-07 23:59 – 2026-09-08
00:05, the device watched by the monitor throughout (only the sweep's own contexts) after an
earlier screen under a running AdaRound job was discarded. Every NPU point is 10 iterations
after 3 warm-ups; GFLOPS is 2MKN over the NPU-bracket average. `results/aie/gemm_tile_sweep_c_single_buffer_npu.log`.

**L1 is exactly the model.** `2A + 2B + (1|2)·C + 3,328 B stack` decided all 28 compiles: 25 of
the 26 tiles it put under 65,536 B compiled (the 26th, m=128 at N=4096, hit the C-output stride
cap of SILICON.md 2.6 instead), and both it put over — 128/64/64 and 64/128/64 in bf16, 85,248 B —
died in the allocator. 128×64 and 64×128 are out of bf16's reach even single-buffered, now
measured rather than derived.

**The 2048³ tile screen** (bf16; predicted = 3.1's ceiling × the 7.37 TFLOPS peak at 1.80 GHz):

| tile m/k/n | C buffer | B/MAC | 3.1 ceiling | predicted | measured GFLOPS | % of peak | vs default |
|---|---|---|---|---|---|---|---|
| 64/64/32 (default) | double | 0.0938 | 67% | 4913 | 1715.59 | 23.3% | 1.00× |
| 64/64/32 | single | 0.0938 | 67% | 4913 | 1603.26 | 21.8% | 0.93× |
| 128/64/32 | single | 0.0781 | 80% | 5896 | 2136.09 | 29.0% | 1.25× |
| **64/64/64** | single | 0.0625 | 100% | 7370 | **2501.71** | 33.9% | 1.46× |
| **32/64/128** | single | 0.0781 | 80% | 5896 | **2494.61** | 33.8% | 1.45× |
| 64/128/32 | single | 0.0938 | 67% | 4913 | 1809.65 | 24.6% | 1.05× |
| 128/32/64 | single | 0.0469 | 100% | 7370 | 2070.87 | 28.1% | 1.21× |
| 64/32/128 | single | 0.0469 | 100% | 7370 | 2146.76 | 29.1% | 1.25× |
| 128/64/64, 64/128/64 | single | 0.0469, 0.0625 | 100% | 7370 | L1: 85,248 B | — | — |

Single-buffering alone costs 6.5% at the default tile (the overlap it removes is real); what it
buys is 64/64/64 and 32/64/128, equal within 0.3% and 1.46× the default. **The B/MAC model is
missing two terms.** 128×32 and 32×128 have identical B/MAC and measure 1.17× apart; with
`--b-col-maj 1`, which makes B's contiguous DMA run k elements instead of n for both, 128/64/32
gains 4.3% (2228.17), 32/64/128 loses 12.6% (2180.36), 64/64/64 moves −2.9% (2429.80), and the
first two swap order — B's run length is a term. And k is a term: 128/32/64 and 64/32/128 have a
better B/MAC than 64/64/64 and land below it, while 64/128/32 beats the default tile at the
same B/MAC by 5.5%. The model still orders tiles that differ only in B/MAC correctly; at 64×64
the measured 33.9% of peak against its 100% input-bound ceiling says the next two thirds are
not input bandwidth.

**Shapes × tiles against the same-sitting CPU bf16** (CPU GFLOPS from the mean, and from the
best-case min, of 20 iterations):

| MxKxN | CPU bf16 mean / min | 64/64/32 dbl | 64/64/64 | 32/64/128 | 128/64/32 | best NPU ÷ CPU mean / min |
|---|---|---|---|---|---|---|
| 1024³ | 1419.8 / 2294.6 | 1872.95 | 2080.98 | 1972.14 | 1885.68 | 1.47× / 0.91× |
| 2048³ | 1210.0 / 1424.4 | 1715.59 | 2501.71 | 2494.61 | 2136.09 | 2.07× / 1.76× |
| 4096×2048×2048 | 1315.3 / 1610.3 | 1663.31 | 2508.97 | 2521.46 | 2123.49 | 1.92× / 1.57× |
| 2048×4096×4096 | 1431.4 / 1760.2 | 1731.85 | 2653.05 | **2700.44** | stride cap | 1.89× / 1.53× |

2700.44 GFLOPS at 2048×4096×4096 is the best bf16 figure in this repo, 36.6% of peak. The two
best tiles track each other at every shape; 1024³ is the one shape where the CPU's best-case
min beats the NPU's mean (0.91×) — the small-shape caveat every GEMM result here carries. The
2048³ 64/64/64 point agrees with the `n=64` section's 2477.23 within 1%.

**int8 gains too, and more than the contaminated screen suggested:** at 2048³, 64/64/64
double-buffered 4293.63 GOPS, 128/64/64 single 4683.89 (1.091×), 64/128/64 single 4852.06
(1.130×) — the k=128 tile is the better use of the 16 KB in int8, consistent with the k term
above. (The discarded screen, kept in the log's appendix, had every control 6–11% low and the
int8 gains at +4%; its ranking was right and its sizes were not.)

**Not tested:** any tile past L1 (a design change — C through the mem tile or a smaller
accumulator — not a flag); m=128 at N=4096 (the 2.6 stride cap); 32×64 and 32×32; int8 across
shapes; `--b-col-maj` at other shapes; whether 32/64/128 keeps its lead below M=1024.

### The AIE core clock, measured: 1.80 GHz default, 0.80 powersaver

Every per-second ceiling this repo derives for the array — TOPS per column, bytes per
cycle on a stream, MACs per cycle per core — multiplied a per-cycle figure by a clock
nothing on this machine had measured. `RESEARCH.md` cited 1.6 GHz from a web search,
`results/aie/bottleneck_spatial_sweep_npu.log` reasoned at 1 GHz and said so, and
`xrt-smi examine -r platform` prints no clock. `docs/SILICON.md` carried it as unmeasured
and made measuring it objective S0. This is that measurement (Desktop 2 / Phoenix,
2026-09-07, `results/aie/clock_probe_npu.log`, `kernels/clock_probe/`).

**Method.** One Worker on one core tile runs `event0()`, a DMA-free loop of N iterations,
`event1()`. The tile's trace unit stamps both instruction events with its 64-bit timer and
streams the packets to a host buffer (`Program.enable_trace`). The host times the same
call with the runtime's own submit+wait bracket — the `hw` column
`results/aie/dispatch_floor_npu.log` used — and fits `hw_ms = intercept + slope × (stamp1 − stamp0)`
across N = 2^18 … 2^25, so the fixed per-dispatch cost lands in the intercept and the
clock is 1/slope. Two loops with different costs (a volatile scalar add: 9 cycles per
iteration; a dependent 16-lane `aie::add` chain: 2) must fit to the same clock. Every call
is verified before its timing is kept (mode and length echoed back, a checksum of the
loop's arithmetic, the core-tile row, the real event pair present ahead of the filler
pairs in the trace); compile is excluded. One fresh process per power mode, because `@iron.jit` holds one hardware
context for the life of the process; each run captures `xrt-smi examine -r platform` in
its own output so the mode is evidenced, not asserted.

| Power mode | Core clock, scalar loop | vector loop | R² (scalar / vector) | Loops agree within |
|---|---|---|---|---|
| `default` | **1.7983 GHz** | 1.7924 GHz | 1.000000 / 0.999995 | 0.33% |
| `powersaver` | **0.7985 GHz** | 0.7984 GHz | 1.000000 / 1.000000 | 0.01% |
| `balanced` | **1.0274 GHz** | 1.0278 GHz | 1.000000 / 1.000000 | 0.04% |
| `performance` | **1.8002 GHz** | 1.7986 GHz | 0.999999 / 1.000000 | 0.09% |
| `turbo` | **1.7998 GHz** | 1.7989 GHz | 1.000000 / 1.000000 | 0.05% |
| `default`, re-measured after the sweep | **1.7990 GHz** | 1.7993 GHz | 0.999999 / 1.000000 | 0.02% |

Cycles per iteration came out exactly constant at every length — 9.000 and 2.000 from 2^18
to 2^25 iterations — which is what makes the stamps trustworthy as core cycles: a timer at
k times the clock would need both 9/k and 2/k to be whole numbers, which only k = 1
satisfies, and a timer at a fraction of it would put the independently measured 7.0 GB/s
shim stream at under half of `device.yaml`'s 4 bytes per cycle (at 1.80 GHz it is 3.9).
The raw ratio cycles ÷ hw at
the longest `default` point (168 ms) reads 1.7947 GHz, converging on the fit from below as
the intercept amortizes.

**What it changes.** Nothing measured in milliseconds, and no %-of-nameplate figure: those
divide by AMD's 16 TOPS, not by a clock. What moves is every ceiling `docs/SILICON.md`
derived from a clock: the 16 TOPS nameplate is what 20 cores do at 1.6 GHz, and in
`default` this part runs at 1.80 — 18.4 TOPS for the full array, 14.7 for the 16 cores the
`4x4` overlay reaches, so that overlay's physical ceiling is 92% of nameplate, not 82%; a
column is 3.69 int8 TOPS, the vendor DPU's measured 1.650 is 44.8% of it; the 16-core bf16
peak is 7.37 TFLOPS and the best GEMM here (2072.54 GFLOPS) is 28.1% of it. The
conversion from any figure quoted "at 1.6 GHz" is the ratio 1.6 ÷ 1.8 = 0.889; the old
columns stay in that file beside the new one. Power mode is a **2.25× lever on the clock**
(0.80 → 1.80 GHz) that no log in this repo recorded: any future comparison across sessions
should capture the platform report alongside, as `clock_probe.py` does.

**What else the runs showed.**
- Three calls each after 5 s of idle read 1.7851 / 1.7716 / 1.7683 GHz, against 1.759–1.787
  for ten back-to-back calls. No idle penalty at that scale, so the session-to-session
  latency drift in `docs/DECISIONS.md` is not an idle clock state at 5 s.
- pyxrt's `device.get_info(max_clock_frequency_mhz)` reads **800 in every mode**, the
  `powersaver` clock; it is not the live clock.
- `xrt-smi configure --pmode turbo` printed `[xrt-smi] ERROR: Failed to escape
  (0xc0000001): A device attached to the system is not functioning.`, yet the platform
  report then read `Turbo` and the clock matched `performance` and `default`. Restoring
  `default` from `turbo` printed the same error and worked; the device stayed healthy (the
  last table row). `powersaver`, `balanced` and `performance` switched without error, all
  from an unelevated shell. Whether `turbo` is a real fourth state on this part is open.
- The traced core reports itself at physical row 2, column 1 (both `get_coreid()` and the
  trace packet header): IRON's logical column 0 is physical column 1 on this xclbin.
- Untraced (`--trace-size 0`, three matching points), the same design's hardware-bracket
  intercept is 210.6 / 281.9 µs (vector / scalar loop): 40–110 µs above the no-compute
  passthrough's 169.8 µs, the cost of a core to load and start. Trace adds 65–110 µs on
  top (321.4 / 346.9 µs traced). Both cancel in the fit. The untraced slopes, 1.1156 and
  5.0006 ns per iteration, with the traced 2 and 9 cycles per iteration, give 1.7928 and
  1.7998 GHz — the clock a third way, from timing alone.

**Tooling found on the way, all of it load-bearing for the trace objectives in
`docs/SILICON.md`.** Peano (llvm-aie 22) declares `get_cycles()` in its aie_api compat
header and never defines it (`ld.lld: error: undefined symbol: get_cycles()`);
`__builtin_readcyclecounter()` dies in the legalizer and inline asm in IRTranslator, so the
trace unit is the one path to the tile timer the open toolchain exposes — this was also
the first end-to-end hardware trace on npu1 on this machine. One `event0`/`event1` pair
alone never reached host memory: the trace unit packs frames into 32-byte packets and the
shim DMA writes 64-byte bursts, exactly the "too few events to create a valid trace
packet" case mlir-aie's programming guide names; the kernel emits 256 filler pairs after
the real one. And mlir-aie v1.4.2's `aie.utils.trace.parse` mis-times any gap longer than
2^18 cycles: the hardware encodes it as an `0xff` sync frame (one wrap of the 18-bit delta
counter) plus a repeat count, the parser treats `0xff` as a no-op and the repeat as
re-issuing the last event, and a 2,097,172-cycle gap came back as 45,017 with eight
spurious events. `clock_probe.py` decodes the frames itself and cross-checks against
upstream on a run short enough to hold no sync frame (both: 73,732 cycles). At 1.8 GHz the
upstream limit is 146 µs between consecutive events; any real kernel trace here will hit
it.

**Caveats.** The measurement assumes the trace timer ticks at the core clock; the integer
cycles per iteration support it and do not prove it. One core tile (logical (0,2)) was
measured; other tiles and columns are assumed to share the clock domain. The
concurrent-VitisAI-EP leg of objective S0 was not run — the worktree that ran this had no
`models/` directory. Power mode was changed and restored; nothing else on the device was
touched. Wall time is not reported: with trace on, IRON allocates and dumps a 64 KB trace
buffer inside the wall bracket.

**The XRT clock readback, reconciled (2026-09-08).** This run read `max_clock_frequency_mhz`
as a flat 800 in every power mode; the NPU-monitor work read it as 800 idle → 1800 under an
active context. Both axes varied in one sitting (`results/aie/pmode_clock_readback_npu.log`; a
2048³ bf16 GEMM hold with `xrt-smi configure --pmode` stepped through all five modes, then
the five modes idle, the monitor logging the clock, the mode and the engine utilization every
0.1–0.25 s, twice): busy, the readback is the mode's clock to the MHz of the table above —
1800 `default`/`performance`/`turbo`, 1028 `balanced`, 800 `powersaver`; idle, 800 in every
mode. The "flat 800" was an idle reading. Every switch took effect within one poll with the
GEMM running; `turbo` printed its escape error under load and idle and applied anyway. The
same log's first run is kept as contaminated: it overlapped another session's 128-stream
classifier sweep, and the hold hung in the second that sweep's XRT aborted.

### MobileViT-XXS does not survive per-tensor INT8, and AdaRound cannot save it

The accuracy question the attention work deferred, now measured on the full 1000-image
eval set (`./scripts/mobilevit-eval.sh`, all rows CPU, `results/mobilevit/eval_*.log`):

| Model | Quantization | Top-1 | Top-5 | Latency/img |
|---|---|---|---|---|
| MobileViT-XXS | FP32 baseline | **68.30%** | 88.20% | 8.77 ms |
| MobileViT-XXS | Full XINT8 | **0.00%** | 0.10% | 20.14 ms |
| MobileViT-XXS | Hybrid (CNN XINT8, transformer FP32) | **0.10%** | 0.30% | 11.65 ms |
| MobileViT-XXS | Hybrid + AdaRound (500 iters, real data) | **0.80%** | 2.50% | 11.40 ms |

This is a **total collapse, and real-data calibration does not fix it.** That the models
above really are a different calibration from the earlier `UseRandomData=True` probes was
checked, not assumed — a 0% score is also what a random probe gives. The old probe is
still on disk and the two are nothing alike: it has activation scales spanning
0.000122–**4.0** (32768×) against the real calibration's 0.0078–0.5 (64×), and **zero**
dead depthwise channels against 28. It scores ~0% anyway — a third strike against the
channel-death story below. AdaRound moves top-1 from 0.00% to 0.80%; that is the ceiling
of what rounding buys here.

> Only the FP32 and full-XINT8 rows are reproducible from this repo
> (`pipelines/mobilevit/1_export.py` → `2_quantize.py`). The two hybrid models came from a
> cut + AdaRound path that is not committed, so "300 images" and "500 iters" are reported
> rather than verified; the eval rows measure them as found in `models/`.

**Retraction: the FP32 baseline is 68.30%, not the 75.0% previously published here.**
75.0% was the first *100* images; the full 1000 settle at 68.30%, which matches the
published MobileViT-XXS paper figure (~69.0%). Reproduced deliberately as the last row of
`./scripts/mobilevit-eval.sh --slice`: same weights, same code, 75.00% on 100 images and
68.30% on 1000. This is the yolov8s AdaRound slice trap (45.19 → 39.98 mAP) a second time.

**Why it collapses, when MobileNetV2 in the same repo survives the same recipe at 73.40%.**
`tools/audit_quant_grid.py` reads this out of the `.onnx` files themselves — no hardware,
no accuracy run, reproducible by anyone holding `models/`
(`results/mobilevit/quant_grid_audit.log`):

| | MobileNetV2 (recovers) | MobileViT-XXS (collapses) |
|---|---|---|
| Weight scale granularity | per-tensor | per-tensor |
| Depthwise scale grid | 0.0156 … **0.25** | 0.125 … **1.0** |
| Depthwise channels quantized to all-zero | 196/7136, worst block 25.0% | 28/432, worst block 34.4% |
| Other-conv dead channels | 186/9920 | 3/1864 |
| Coarsest activation scale | 0.5 | 0.5 |

The intuitive explanation — "depthwise channels die under per-tensor quantization" — **is
not what separates them.** MobileNetV2 carries a 25%-dead depthwise block of its own and
still reaches 73.40%, and its *other* convs are considerably more damaged than
MobileViT's (186/9920 vs 3/1864). Nor is it activation range: both models top out at the
same coarsest activation scale of 0.5.

What separates them is the **depthwise weight-scale grid**. MobileNetV2's depthwise scales
never exceed Δ=0.25; MobileViT-XXS reaches **Δ=1.0** on a 3×3 depthwise kernel, a 4–16×
coarser grid. At Δ=1.0 essentially every real weight rounds to 0 or ±1, so the layer stops
being a convolution and becomes a sign map.

**What sets Δ is now an open question again — Cross-Layer Equalization has been measured
and ruled out.** The standing explanation was that CLE needs a positive-homogeneous
activation (`f(αx) = αf(x)`), that ReLU6 has it and SiLU doesn't, so Quark equalizes
MobileNetV2 and skips MobileViT. Capturing the counts
(`results/mobilevit/cle_pattern_count.log`) confirms the headline and destroys the
argument: MobileNetV2 matches **3** CLE patterns, MobileViT-XXS **0**. But 3 is really 2
distinct pairs against 52 convs, and — decisively — **neither pair contains a depthwise
conv.** Both are pointwise (a `conv_pw`→`conv_pw` pair and `conv_pwl`→`conv_head`). CLE
never touches a depthwise layer in either model, so it cannot be what bounds the depthwise
grid, in either direction.

The premise was also wrong on its own terms: **ReLU6 is not positive-homogeneous** —
`ReLU6(2·4) = 6`, not `2·ReLU6(4) = 8`. Quark's matcher agrees with the math rather than
with the old claim; it accepts only `['Relu','ReduceMean','Pad','LeakyRelu']` between two
convs, and `Clip` is deliberately absent. MobileNetV2's ReLU6 exports as `Clip` (35 of
them, zero `Relu`), so its activations block the walk for the same structural reason
SiLU's do; the 3-vs-0 gap is residual activation-free `Conv`→`Conv` adjacency, not a
statement about homogeneity. MobileViT is rejected twice over: `Sigmoid` isn't in the
accepted set, and 29 of its 36 convs feed *both* the `Sigmoid` and the `Mul`, failing the
matcher's single-consumer precondition before the op-type test is even reached. So the
discriminator is measured, CLE is excluded as its mechanism, and **the upstream cause of
the depthwise grid difference is unexplained.**

AdaRound's failure is now mechanically explained rather than asserted: it picks between
`floor(w/Δ)` and `ceil(w/Δ)` and **never changes Δ**. The audit confirms the scale grid is
byte-identical before and after AdaRound; only the dead-channel count shifts at the
rounding boundary (28/432 → 24/432), worth 0.8 points. **Deploying a SiLU/GELU backbone to
XDNA1 needs QAT or per-channel scale support, not a better PTQ recipe.**

**A rank-5 tensor never reaches the NPU.** Across `mobilevit_stock`'s 1585 nodes, **207
touch a rank-5 tensor and all 207 are on CPU — zero exceptions** (`--rank-audit`). The
contrast holds: YOLOv8's 16 rank-4 `Slice` nodes all place on NPU, and no other model in
this repo produces a rank-5 tensor at all. Rank-5 is *sufficient* to force CPU fallback,
but not the whole story — 18 rank-4 `Transpose`, 18 rank-4 `MatMul` and 21 rank-3
`LayerNormalization` nodes fall back too, on op support rather than rank.


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

**Does AdaRound change the curve?** AdaRound at 224² reaches 79.80% top-1 / 92.50%
top-5 (5.27 ms). Testing AdaRound across the resolutions spanning 160² to 288²
reveals how rounding recovery interacts with resolution:

| Resolution | Plain XINT8 top-1 | AdaRound top-1 | AdaRound top-5 | NPU Latency | NPU Nodes | Backing Logs |
|---|---|---|---|---|---|---|
| **160²** | 67.10% | **76.60%** | 91.70% | **4.55 ms** | 393 / 395 | `results/res/run_resnet50_r160_adaround_npu.log`, `diag_resnet50_r160_adaround.log` |
| **192²** | 71.30% | 79.30% | 92.40% | 5.15 ms | 393 / 395 | `results/res/run_resnet50_r192_adaround_npu.log`, `diag_resnet50_r192_adaround.log` |
| **224²** | 72.10% | **79.80%** | 92.50% | 5.27 ms | 393 / 395 | `results/adaround_latency_diff_adaround_npu.log` |
| **256²** | **74.00%** | **79.80%** | **93.40%** | 5.85 ms | 392 / 394 | `results/res/run_resnet50_r256_adaround_npu.log`, `diag_resnet50_r256_adaround.log` |
| 288² | 71.50% | 78.10% | 93.40% | 8.78 ms | 393 / 395 | `results/res/run_resnet50_r288_adaround_npu.log`, `diag_resnet50_r288_adaround.log` |

**Key findings on AdaRound across resolution**:
- **A broad 79.3%–79.8% top-1 accuracy plateau from 192² to 256²**: Under plain XINT8, accuracy
  sharply collapsed as resolution shrank (74.00% at 256² → 72.10% at 224² → 71.30% at 192²) because
  lower-resolution activation maps were more susceptible to per-tensor INT8 rounding noise. AdaRound
  delivers massive recoveries across all three sizes (+8.00% at 192², +7.70% at 224², +5.80% at 256²),
  flattening the accuracy response into a tight 0.5-point band (79.30%–79.80%).
- **160² marks the true inflection point (+9.50% recovery, 76.60% top-1 at 4.55 ms)**: At 160²,
  AdaRound yields its largest single recovery (+9.50% top-1 from 67.10% to 76.60%, and +6.00% top-5
  from 85.70% to 91.70%). Here the network finally steps off the 79% plateau, reflecting the physical
  limit of spatial downsampling (a 5×5 final spatial grid before pooling) rather than rounding noise.
  Crucially, 160² AdaRound still **beats default 224² plain XINT8 on both axes** (+4.50% top-1,
  +3.60% top-5, and 20% lower latency: 4.55 ms vs 5.68 ms, ~220 img/s).
- **192² offers maximum throughput at near-peak accuracy**: At **5.15 ms** (~194.1 img/s), 192² AdaRound
  reaches **79.30% top-1 and 92.40% top-5**, sacrificing only 0.50% top-1 against 224² (79.80%) and matching
  its top-5 (92.40% vs 92.50%) while running at higher throughput. Against stock plain XINT8 at 224²
  (72.10% at 5.68 ms), 192² AdaRound is **both significantly more accurate (+7.20% top-1) and faster
  (5.15 ms vs 5.68 ms)**.
- **Top-1 peaks at 79.80% across 224² and 256²**: Both 224² and 256² converge to the identical **79.80%**
  ceiling of the float model.
- **Top-5 improves by +0.90% at 256²**: 256² AdaRound achieves **93.40% top-5**, clearly outperforming
  224² (92.50%), 192² (92.40%), and 160² (91.70%) at only **5.85 ms** (~171.0 img/s, +0.58 ms over 224²).
- **Comparison to `wide_resnet50_2`**: `wide_resnet50_2` + AdaRound scores 80.10% top-1 / 93.40% top-5
  at 9.66 ms. ResNet50 at 256² with AdaRound reaches virtually identical accuracy (79.80% / 93.40%)
  at **39% lower latency** (5.85 ms vs 9.66 ms, 171 img/s vs 103 img/s).
- **288² remains dominated**: Top-1 falls off to 78.10% while latency jumps to 8.78 ms (a 50% latency
  increase over 256² for 1.7 points lower top-1).

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

Caveat: plain XINT8, calibration 64 — same caveat as the resolution sweep above.
(AdaRound has since been evaluated for both wide models; see the next section for
the like-for-like comparison against the float baselines.)

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
| wide_resnet101_2 | 82.00% | 80.20% | 79.90% | −1.80 (2.2% rel.) | −0.30 (wash on top-1; top-5 recovers 92.90% → 93.90%, closing 66.7% of gap) |

**The prediction doesn't hold, in either direction.** The initial quantization penalty
was actually *worse* for the wider model at 50 layers (−9.50 vs −8.00, both absolute and
relative) — the opposite of the YOLO n→s pattern — and AdaRound's recovery efficiency
is essentially identical between them (90.0% vs 90.5% of the gap closed).

**At `wide_resnet101_2` (101 layers, 126.9M params, 104 Conv layers), the story shifts again**:
- **Plain XINT8 was already remarkably immune to quantization loss**: only −1.80 points of
  top-1 loss from the 82.00% FP32 baseline (`results/wide/run101_fp32_cpu.log`), compared to
  −8.00 for resnet50 and −9.50 for wide_resnet50_2. The massive parameter capacity absorbs
  per-tensor rounding noise.
- **AdaRound on `wide_resnet101_2`** (`results/wide/run101_adaround_npu.log`,
  `results/wide/diag101_adaround.log`): Top-1 sits at **79.90%** (vs 80.20% plain XINT8, within
  statistical noise on 1000 images), but **top-5 recovers clearly**: **92.90% → 93.90%**, closing
  66.7% of the gap to the 94.40% FP32 ceiling.
- **Latency & Placement**: Runs at **16.64 ms** on NPU (767 / 769 nodes, 99.7% on NPU),
  delivering a **4.97× speedup over the 8-core Zen 4 CPU FP32 baseline** (82.78 ms).
- **Practical comparison**: `wide_resnet50_2` + AdaRound remains **the best speed/accuracy point
  in this repo for classification**: 80.10% top-1 at 9.66 ms, essentially matching
  `wide_resnet101_2`'s 79.90% / 80.20% accuracy at nearly half the latency (9.66 ms vs 16.64 ms),
  because `wide_resnet101_2` doubles depth (101 layers vs 50) without buying more top-1 accuracy.
- **Resource requirement**: AdaRound on `wide_resnet101_2` ran 104 Conv layers through CPU FastFinetune
  at 224² on this same 13.8 GB, ~4-5 GB-free box without incident, confirming once more that
  AdaRound's memory limit is resolution-specific (640×640), not parameter- or depth-specific.

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
### Alternative classification topologies: DenseNet-121 (Concat) and ResNeXt-50 (Grouped Convs)

Testing the hardware execution boundaries and quantization limits of the XDNA1 compiler on
two alternative convolutional wiring topologies: channel concatenation across dense blocks
(`densenet121`, 7.98M params, 120 Convs, 62 Concats, 3 AveragePools) and grouped convolutions
(`resnext50_32x4d`, 25.0M params, 53 Convs with `groups=32`, 32 groups of 4 channels each).
Both evaluated on 1,000 ImageNet-1k validation images against their FP32 Zen 4 CPU baselines:

| Model | Topology | FP32 CPU Latency | FP32 top-1 | XINT8 NPU Latency | Plain XINT8 top-1 | NPU Nodes | Backing Logs |
|---|---|---|---|---|---|---|---|
| resnet50 | Residual (`Add`) | 12.18 ms | 80.10% | **5.68 ms** | **72.10%** | 393 / 395 | `results/bench_xint8_npu.log` |
| **densenet121** | Dense (`Concat`) | 21.70 ms | 78.00% | **8.06 ms** | **0.10%** | **1703 / 1705** | `results/res/run_densenet121_xint8_npu.log`, `diag_densenet121_xint8.log`, `run_densenet121_fp32_cpu.log` |
| **resnext50_32x4d** | Grouped (`group=32`) | 17.40 ms | 81.00% | **9.37 ms** | **0.10%** | **393 / 395** | `results/res/run_resnext50_32x4d_xint8_npu.log`, `diag_resnext50_32x4d_xint8.log`, `run_resnext50_32x4d_fp32_cpu.log` |

**What the compiler accepts vs what quantization destroys**:

1. **Hardware offload is virtually complete (99.5%–99.9%)**:
   - `densenet121`: **1,703 of 1,705 nodes (99.9%)** execute on the NPU. All 58 `Concat` nodes,
     all 3 `AveragePool` nodes, all 182 quantized Conv units, and all 121 Relu activations map
     natively to the AIE array. Only the input/output Q/DQ boundary sits on CPU.
   - `resnext50_32x4d`: **393 of 395 nodes (99.5%)** execute on the NPU. All 53 convolutional
     layers with `group=32` compile to native AIE kernels with zero CPU fallback.
   - **Both topologies run faster than Zen 4 CPU FP32**: DenseNet-121 achieves **8.06 ms**
     (~124.1 img/s, a 2.69× speedup over 21.70 ms CPU); ResNeXt-50 achieves **9.37 ms**
     (~106.7 img/s, a 1.86× speedup over 17.40 ms CPU).
   - **Concat memory overhead is well-managed**: DenseNet's 8.06 ms across 120 convs + 58 concats
     confirms that channel concatenation does not choke the AIE DMA or trigger excessive copy overhead.
   - **Grouped convolution efficiency penalty**: Compared to standard ResNet-50 (5.68 ms), ResNeXt-50
     takes 9.37 ms (+65% latency for identical depth and FLOPs), demonstrating that 32 narrow 4-channel
     group micro-kernels achieve lower SIMD lane utilization on the 4×4 AIE array than standard wide convs.

2. **Both topologies suffer catastrophic PTQ collapse under plain XINT8 (0.10% top-1)**:
   - While stock ResNet-50 retains 72.10% top-1 under plain XINT8, DenseNet-121 collapses to **0.10% top-1
     / 0.70% top-5** and ResNeXt-50 collapses to **0.10% top-1 / 0.60% top-5**.
   - Running `densenet121_xint8.onnx` under `--ep cpu` confirms the identical collapse (0.00% top-1 / 2.00% top-5),
     proving the failure is intrinsic to the INT8 scale representation, not an NPU execution bug.
   - **The mechanism for DenseNet**: Concatenating feature maps across blocks forces up to 32 disparate
     activation representations to share common power-of-two scales. Quark's compiler constraint adjuster
     reports shift cuts exceeding `[0, 16]` by up to 7 powers of 2 (a 128× scaling error), resulting
     in severe clipping and zeroed activations.
   - **The mechanism for ResNeXt**: Grouped convs with small channel counts per group (4 channels)
     produce wide dynamic range divergence across groups. A single per-tensor INT8 scale cannot span
     all 32 groups simultaneously, triggering numerical overflow (`rmax/rmin set to inf/-inf`) and
     extreme weight shift cut adjustments (up to 128, far outside `[0, 16]`).

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

**Does a wider classifier saturate the same way, or earlier — like yolov8m does
against yolov8n?** `wide_resnet50_2` (2.7× resnet50's params, same recipe, calib 64)
run through the same sweep:

| streams | combined fps | speedup vs 1 | mean top-1 | mismatch vs solo |
|---|---|---|---|---|
| 1 | 73.9 | 1.00× | 76.67% | 0.00% |
| 2 | 113.2 | 1.53× | 76.67% | 0.00% |
| 3 | 114.9 | 1.55× | 76.67% | 0.00% |
| 4 | 116.0 | 1.57× | 76.67% | 0.00% |
| 6 | 116.7 | 1.58× | 76.67% | 0.00% |
| 8 | 117.1 | 1.58× | 76.67% | 0.00% |
| 12 | 117.2 | 1.58× | 76.67% | 0.00% |
| 16 | 117.2 | 1.59× | 76.67% | 0.00% |

(`results/nstream_wide_resnet50_2.log`, `--fresh`.) Confirms the pattern from
detection: the wider model saturates to essentially the same final multiplier as
resnet50 (1.59× vs 1.60×) but gets there faster — flat by 2 streams instead of 8 —
because it already uses more of the array per call, leaving less idle headroom for a
second context to fill. And the accuracy guarantee holds again: every concurrent
classification at every stream count matched the uncontended baseline exactly
(0.00% mismatch throughout), zero regressions on a second, wider architecture.

**Pushed further on both open ends at once: a third, still-wider classifier
(`wide_resnet101_2`, plain XINT8 calib 64, 767/769 nodes), and streams past 16, up
to 32:**

| streams | combined fps | speedup vs 1 | mean top-1 | mismatch vs solo |
|---|---|---|---|---|
| 1 | 47.1 | 1.00× | 81.67% | 0.00% |
| 2 | 61.3 | 1.30× | 81.67% | 0.00% |
| 3 | 62.0 | 1.32× | 81.67% | 0.00% |
| 4 | 62.3 | 1.32× | 81.67% | 0.00% |
| 6 | 62.1 | 1.32× | 81.67% | 0.00% |
| 8 | 62.2 | 1.32× | 81.67% | 0.00% |
| 12 | 62.1 | 1.32× | 81.67% | 0.00% |
| 16 | 62.2 | 1.32× | 81.67% | 0.00% |
| 24 | 62.2 | 1.32× | 81.67% | 0.00% |
| 32 | 62.3 | 1.32× | 81.67% | 0.00% |

(`results/nstream_wide_resnet101_2.log`, `--fresh`.) Both open ends close the same
way: **the ceiling holds at 32 streams with zero regression** (dropping the "is 16
enough to see the plateau end" question), and the trend across all three
classifiers now reads as monotonic with per-call array usage, not just a
two-point pattern — resnet50's 1.60× (flat by 8) → wide_resnet50_2's 1.59× (flat by
2) → wide_resnet101_2's **lower** 1.32× (flat by 3, its lowest idle headroom of the
three, consistent with it also being the heaviest single-stream call at 47.1 fps
vs the other two's 67-74 fps). Accuracy is unaffected here too: still 0.00%
mismatch against the uncontended baseline at every stream count, all the way to
32. Concurrent-stream saturation on this backend is now checked on three
classifiers spanning 2.7× to 5× resnet50's params, plus two detection models, with
the same shape and the same zero-corruption guarantee every time.

**Pushed to find where the ceiling itself gives out, not just where fps plateaus:**
`wide_resnet101_2` again, streams 32/48/64/96/128 in one run. 32/48/64/96 all land
on the same plateau already seen above (61.9 / 62.1 / 62.3 / 61.9 fps, flat within
noise, 0.00% mismatch throughout) — one more confirmation that the fps ceiling
itself doesn't move past the point it's already reached by 3-8 streams. **128
streams does not complete: it hits a real hardware/driver ceiling, not a script
failure.** Session construction gets to roughly 121 of the 128 requested sessions,
then XRT fatally aborts: `Failed to submit command to hw queue (0xc01e0200): Even
after the video memory manager split the DMA buffer, the video memory manager
could not page-in all of the required allocations into video memory at the same
time. The device is unable to continue.` This is WDDM failing to page in enough
concurrent hardware-context allocations, not an OOM in the Python process itself —
though host RAM was also under real pressure at the time (free memory dropped to
~6.4 GB on this 32 GB box while ~120+ sessions were live, climbing back to ~20 GB
within seconds once the process aborted), and this machine had another concurrent
session's build process running at the same time, so the exact session count where
this breaks is not a clean, isolated ceiling — call it "somewhere in the 96-128
range, and lower under memory pressure from other processes," not a precise number.
`results/nstream_wide_resnet101_2_128_ceiling.log`. Concurrent-stream headroom on
this hardware is generous but not unlimited, and the failure mode when it runs out
is a hard XRT abort, not a graceful queue or slowdown.

**Does the memory/throughput decoupling found on yolov8n/yolov8m above ("Is the
saturation compute or memory?") hold for a classifier too, and at this much larger
memory footprint?** Checked
directly with `tools/session_hold.py` against `wide_resnet101_2` at conservative
stream counts (1/8/16/32 — deliberately not repeating the 128-stream WDDM abort
just found above), holding sessions busy on synthetic input and sampling
`xrt-smi examine -r aie-partitions` every 3s:

| streams | NPU memory | GOPS | measured completions/s |
|---|---|---|---|
| 1 | 204 MB | 46 | 61.1 |
| 8 | 1188 MB | 368 | 62.4 |
| 16 | 2313 MB | 736 | 62.1 |
| 32 | 4562 MB | 1472 | 62.1 |

(`results/session_hold_wide_resnet101_2.log`.) Same shape as yolov8n/yolov8m,
extended to a classifier and to a far larger absolute footprint: memory scales
**exactly linearly** at ~140.6 MB/stream (steeper than yolov8m's ~100 MB/stream
and yolov8n's ~29 MB/stream — consistent with it being the largest model checked
this way), and GOPS is again **exactly** `46 × streams` with no saturation through
32 streams — while measured completions/s is flat from 1 stream onward, matching
the same ~62 fps ceiling `nstream_cls_bench.py` found independently above. Memory
climbing to 4.5 GB by 32 streams with throughput never moving off its 1-stream
value is the same decoupling already established, now confirmed on a third model
family: neither NPU memory pressure nor `xrt-smi`'s GOPS column explains where the
classifier throughput ceiling comes from, any more than they did for detection.

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

### AdaRound on detection: how much does it actually recover?

Moved here from RESEARCH.md's roadmap, where both results were recorded and nowhere else.
AdaRound buys back ~90% of the quantization loss on this repo's classifiers; on detection
it does not come close, at either width measured.

**yolov8s at 640×640 — it barely helps.** Not blocked after all:
`models/yolov8s_cut_xint8_adaround.onnx` compiles and runs (15.5 ms/frame, 922/929 nodes).
Full 5000-image mAP@50-95 is 39.98 against plain XINT8's 37.40
(`results/map_yolov8s_cut_xint8_adaround_npu.log`) — 2.6 points, not the 90% recovery
AdaRound gets on ResNet50/wide_resnet50_2. A 500-image slice run first suggested 45.19,
which would have been a very different story; the full 5000 is the number to trust,
consistent with this repo's other slice-vs-full warnings. Worth understanding why
detection AdaRound recovers so much less than classification's before spending the RAM on
YOLOv8m/l/x or the wide ResNets.

**yolov8m at 640×640 — it recovers even less.** Quantized on Desktop 1 (GPU-accelerated
FastFinetune, `--device`) and run on Desktop 2's XDNA1 (Phoenix):
`models/yolov8m_cut_xint8_adaround.onnx` runs at the same 30.46 ms/frame as plain XINT8
(1216/1223 nodes — no latency cost from AdaRound, only the weight rounding changes). Full
5000-image mAP@50-95 is 45.32 against plain XINT8's 43.49
(`results/map_yolov8m_cut_xint8_adaround_npu.log`) — **+1.83 points**, a smaller absolute
recovery than yolov8s's +2.58 despite m's much higher starting accuracy, extending the
pattern that AdaRound has less room to recover as width increases.

> **Caveat.** The yolov8m AdaRound model arrived via Syncthing with no local log of its
> calibration count, so it isn't a clean like-for-like comparison against the calib-64
> plain-XINT8 row — the exact "a model can arrive with no log explaining it" risk this
> repo's own machine notes warn about.

### yolov8n-pose end to end on the NPU

Head-cut partitions 1015/1025 (99.0%), ~9.1–9.4 ms/frame, same clean pattern as detect:
only the input `QuantizeLinear` and 9 output `DequantizeLinear` nodes land on CPU, with
every Conv (72), Mul (126), HardSigmoid (63), Slice (16), Concat (13), and MaxPool (3)
placing on the NPU (`results/pose_cut_adaround_diag.log`).

Plain XINT8 costs 17.2 points of OKS mAP@50-95 on the full set (49.86 → 32.64, a 35%
relative loss) — proportionally worse than bbox yolov8n's plain-XINT8 loss, because
keypoint coordinate regression and heatmap peaks are especially sensitive to per-tensor
quantization grids. AdaRound (`models/yolov8n-pose_cut_xint8_adaround.onnx`, 300 calib
images, 72 layers optimized on Desktop 2) recovers a meaningful portion of this gap with
zero latency cost.

Full COCO val2017 evaluation (5000 images, conf 0.001, IoU 0.7, max_det 300, single-class
class-agnostic NMS):

| Precision | Device | EP partition | Latency | OKS mAP@50-95 | OKS mAP@50 | Backing log |
|---|---|---|---|---|---|---|
| FP32 (float) | CPU | all CPU | 28.89 ms | 49.86 | 78.69 | `results/map_kpts_yolov8n-pose_cut_full5000_cpu.log` |
| Plain XINT8 | NPU | 1015 NPU / 10 CPU | 9.10 ms | 32.64 | 66.90 | `results/map_kpts_yolov8n-pose_cut_xint8_full5000_npu.log` |
| **XINT8 + AdaRound** | **NPU** | **1015 NPU / 10 CPU** | **9.35 ms** | **34.32** (+1.68) | **71.85** (+4.95) | `results/map_kpts_yolov8n-pose_cut_xint8_adaround_npu.log` |

The earlier 500-image slice run (kept beside the full set per this repo's history-of-being-wrong
contract):

| Precision | Device | EP partition | Latency | OKS mAP@50-95 | OKS mAP@50 | Backing log |
|---|---|---|---|---|---|---|
| FP32 (float) | CPU | all CPU | 36.47 ms | 49.49 | 79.05 | `results/map_kpts_yolov8n-pose_cut_cpu.log` |
| Plain XINT8 | NPU | 1015 NPU / 10 CPU | 9.27 ms | 31.65 | 66.39 | `results/map_kpts_yolov8n-pose_cut_xint8_npu.log` |
| **XINT8 + AdaRound** | **NPU** | **1015 NPU / 10 CPU** | **9.15 ms** | **33.94** (+2.29) | **71.88** (+5.49) | `results/map_kpts_yolov8n-pose_cut_xint8_adaround_slice500_npu.log` |

Two findings from these tables:

1. **AdaRound buys back accuracy with zero latency penalty.** On the full 5,000 images,
   AdaRound recovers **+1.68 points** of OKS mAP@50-95 and **+4.95 points** of OKS mAP@50
   (and +2.29 / +5.49 on the 500-image slice). Mean inference latency is 9.35 ms vs 9.10 ms
   (single-image timed run reads 9.19 ms, `results/pose_cut_adaround_npu.log`), well within
   normal session drift.
2. **Keypoint recovery resembles detection, not classification.** AdaRound recovers
   ~90% of the quantization loss on ResNet50; on pose it recovers ~10% of the mAP@50-95
   drop and ~42% of the mAP@50 drop. As with YOLO detection, the spatial heads remain
   fundamentally limited by per-tensor power-of-two scale granularity.

Quantization log: `results/pose_cut_quantize_xint8_adaround.log` (FastFinetune 1319.2s, total
quantization 2169.5s on Desktop 2's 8700G CPU). Single-image verification:
`results/pose_cut_adaround_{cpu,npu}.log`, with outputs drawn to
`results/out_yolov8n-pose_cut_xint8_adaround_{cpu,npu}.jpg`.

### Category C, first candidate: YOLOv6n (RepVGG backbone)

`pipelines/yolov6n/` — new pipeline, built against Meituan's official 0.4.0 release
(`yolov6n.pt`). Tests the Category C hypothesis directly: YOLOv6's backbone is
`RepVGGBlock`s, which `switch_to_deploy()` collapses at export time from a multi-branch
training graph into a single 3x3 `Conv` + `Relu` per block — no residual `Add`, unlike
YOLOv8's CSPDarknet. `configs/yolov6n.py` also ships `use_dfl=False, reg_max=0`: the box
head regresses raw `ltrb` directly, so there is no DFL softmax in the exported graph at
all (confirmed: zero `Softmax` nodes). The head's own `stem`/`cls_conv`/`reg_conv` layers
still use `ConvBNSiLU` (`Sigmoid`+`Mul`, no native ONNX SiLU op), so the "avoids
SiLU→HardSwish distortion" half of the hypothesis holds for the backbone only, not the
head.

Head-cut at the six raw per-level conv outputs (3 `reg_preds`, 3 `cls_preds`), same
rationale as yolov8n's `1b_cut_head.py`. The removed decode tail (`dist2bbox` + anchor
grid + sigmoid) is reimplemented in `npu/yolov6_decode.py`, verified bit-exact against the
full-graph ONNX output on a random input (xywh max abs diff 0.0, cls max abs diff
1.16e-7 — float sigmoid rounding only) and verified functionally identical on a real
image: full-graph CPU and head-cut CPU produce the same 22 detections, same classes,
boxes and scores.

Node placement, quantized (`models/yolov6n_cut_xint8.onnx`, plain XINT8, 300-image
calibration, no AdaRound): **518/525 nodes (98.7%) on NPU**, a single clean subgraph —
only the input `QuantizeLinear` and the six output `DequantizeLinear` nodes stay on CPU
(`results/diag_yolov6n_cut_xint8.log`). Quark's compiler substitutes `HardSigmoid` for the
head's `Sigmoid`, the same treatment YOLOv8's SiLU gets.

NPU single-image latency at demo settings (conf 0.25): **6.60 ms mean, 151.4 fps**, 518/525
nodes on NPU (`results/lat_yolov6n_cut_xint8_npu.log`) — the same 22 detections as the CPU
cross-check above (person, chair, tv, potted plant, ...).

Full COCO val2017 evaluation (5000 images, conf 0.001, IoU 0.7, max_det 300, per-class NMS).
`4_detect.py`/`5_eval_map.py`'s "infer" is `sess.run` alone (see Invariants), so this
latency is comparable across rows even though conf differs from the demo setting above:

| Precision | Device | Latency (eval, conf 0.001) | mAP@50-95 | mAP@50 | Backing log |
|---|---|---|---|---|---|
| FP32 (float) | CPU | 20.04 ms | 36.95 | 51.98 | `results/map_yolov6n_fp32_cpu.log` |
| Plain XINT8 | NPU | 6.62 ms | 22.92 (-14.03) | 34.98 (-17.00) | `results/map_yolov6n_cut_xint8_npu.log` |
| **XINT8 + AdaRound** | **NPU** | **6.62 ms** | **33.57** (-3.38) | **49.84** (-2.14) | `results/map_yolov6n_cut_xint8_adaround_npu.log` |

AdaRound (`models/yolov6n_cut_xint8_adaround.onnx`, 300 calib images, 71 layers optimized
on Desktop 2's 8700G CPU, `results/quant_yolov6n_cut_xint8_adaround.log`, e2e 1522.0s)
placement and latency are unchanged from plain XINT8 — **518/525 nodes (98.7%), the same
single subgraph** (`results/diag_yolov6n_cut_xint8_adaround.log`), 6.53 ms demo-conf
single-image latency (`results/lat_yolov6n_cut_xint8_adaround_npu.log`, vs plain XINT8's
6.60 ms — within normal session drift, not a regression).

Three findings:

1. **The structural hypothesis holds on placement and speed.** 98.7% single-subgraph
   placement and a 3.0x latency win (20.04 -> 6.62 ms at eval settings) are in the same
   range as yolov8n's own head-cut numbers — RepVGG's Add-free backbone compiles and runs
   as cleanly as CSPDarknet's does here, neither better nor worse on this axis.
2. **Plain XINT8 costs more accuracy here than it does on yolov8n**, but AdaRound recovers
   most of it, at zero latency cost. -14.03 points of mAP@50-95 (38% relative) and -17.00
   of mAP@50 (33% relative) is a substantially larger plain-XINT8 drop than yolov8n's on
   the same convention. AdaRound recovers **+10.65 points of mAP@50-95 (76% of the loss)**
   and **+14.86 points of mAP@50 (87% of the loss)**, closing to within 3.38 / 2.14 points
   of FP32 — much closer to ResNet50's ~90% AdaRound recovery than to yolov8n-pose's ~10%
   (see the pose section above). Re-parameterized RepVGG weights evidently don't carry the
   AdaRound-resistant outliers the falsification criterion below worried about.
3. **This refutes the "AdaRound can't save it" branch of the Category C criterion.**
   `RESEARCH.md`'s falsification criteria asked whether re-parameterized weight
   distributions would degrade INT8 PTQ "beyond the recovery capacity of AdaRound" —
   measured, they don't: the same AdaRound recipe used elsewhere in this repo, with no
   yolov6n-specific tuning, recovers the large majority of the plain-XINT8 loss.

### Category C, second candidate: YOLOv11n (C2PSA Attention Block & Decoupled DWConv Head)

`pipelines/yolov11/` — new pipeline, built against Ultralytics YOLOv11 (`yolo11n.pt`).
Tests the Category C hypothesis on successor YOLO architectures: C3k2 blocks (faster
CSP implementations with optional 2-stage convolutions), decoupled depthwise-convolution
detection heads (`model.23.cv2` for box regression and `model.23.cv3` for classification),
and the C2PSA (Convolutional 2-Stage Pointwise Spatial Attention) block (`model.10`)
positioned at the deepest stage of the backbone.

Head-cut at the six raw per-level convolution outputs (3 box heads from `cv2.{0,1,2}.2`,
3 class heads from `cv3.{0,1,2}.2`), following the established `1b_cut_head.py` recipe.
The 24-node decode tail (DFL softmax, anchor coordinate regression, and class sigmoid)
is stripped via `onnx.utils.extract_model` (`models/yolo11n_cut.onnx`, opset 17, 187 nodes).
NumPy decode in `npu/yolo_decode.py` reproduces full-graph CPU detections exactly (13 identical
detections on `assets/test_image.jpg`, see `results/lat_yolo11n_cut_fp32_cpu.log`).

Node placement and the C2PSA rejection:
- **Stock quantized YOLOv11n** (`models/yolo11n_cut_xint8.onnx`, plain XINT8, 200-image COCO
  calibration): **6 / 1300 nodes (0.46%) on NPU**, 1294 nodes on CPU
  (`results/diag_yolo11n_cut_xint8.log`). The VitisAI level-1 DPU compiler rejects the 4D
  `MatMul` operations ($B=1, \text{heads}=2, N=400$) inside the C2PSA spatial attention loop
  (`/model.10/m/m.0/attn`). Unlike MobileViT which fragmented into 58 subgraphs, the EP here
  refuses all 87 Convolutions outright, placing only the isolated `Softmax` and `Transpose` ops
  on NPU. The resulting host-NPU round trips cause massive context thrashing, inflating single-image
  latency to **34.63 ms** (`results/lat_yolo11n_cut_xint8_npu.log`) — slower than host CPU FP32 (21.59 ms).
- **C2PSA-Ablated YOLOv11n** (`models/yolo11n_no_c2psa_cut_xint8.onnx`, exported with `--no-c2psa`
  where `/model.10/m/m.0` is replaced with an identity passthrough): **1,173 / 1,180 nodes (99.4%)
  on NPU**, a single monolithic DPU subgraph (`results/diag_yolo11n_no_c2psa_cut_xint8.log`).
  Only the 1 input `QuantizeLinear` and 6 output `DequantizeLinear` nodes stay on CPU, with zero
  internal CPU fallbacks. Single-image demo latency drops to **6.95–7.02 ms (143.9 fps)**.

Single-image latency on Desktop 2 (Phoenix 8700G, 640×640):
- Stock CPU (FP32): **21.59 ms** (`results/lat_yolo11n_cut_fp32_cpu.log`)
- Stock DirectML (Radeon 780M iGPU, FP32): **8.58 ms** (`results/lat_yolo11n_cut_fp32_dml.log`)
- Stock NPU (XINT8, fractured): **34.63 ms** (`results/lat_yolo11n_cut_xint8_npu.log`)
- Ablated NPU (XINT8, monolithic): **6.95–7.02 ms**

Full COCO val2017 evaluation (5000 images, conf 0.001, IoU 0.7, max_det 300, per-class NMS;
inference is `sess.run` alone):

| Variant | Precision | Device | Latency (eval) | mAP@50-95 | mAP@50 | NPU nodes | Backing log |
|---|---|---|---|---|---|---|---|
| Stock | FP32 | CPU | 22.82 ms | 38.72 | 54.24 | — | `results/eval_yolo11n_cut_fp32_cpu.log` |
| Stock | Plain XINT8 | NPU | 33.29 ms | 25.82 (-12.90) | 38.76 (-15.48) | 6 / 1300 | `results/eval_yolo11n_cut_xint8_npu.log` |
| No-C2PSA (ablated) | Plain XINT8 | NPU | **7.08 ms** (141.2 fps) | 0.19 | 0.34 | **1173 / 1180** | `results/eval_yolo11n_no_c2psa_cut_xint8_npu.log` |

Three findings:

1. **C2PSA spatial self-attention is rejected by the DPU compiler.** The 4D MatMuls embedded
   within the C2PSA attention block cannot be scheduled onto the XDNA1 DPU array by the VitisAI
   compiler. Rather than isolating the attention block and keeping the remaining convolutions on
   NPU, the EP fractures catastrophically: all 87 Convolutions remain on CPU, and only 6 ancillary
   nodes land on NPU. The resulting hand-off overhead inflates inference to 33.29 ms, making stock
   NPU deployment 1.46× slower than FP32 CPU execution.
2. **The C3k2 backbone and decoupled DWConv heads are the fastest YOLO architecture on XDNA1.**
   When C2PSA is ablated, YOLOv11n executes at **7.08 ms over the full 5,000-image evaluation**
   (141.2 fps) in a single monolithic DPU subgraph. This sets the all-time speed record for YOLO
   models on this silicon:
   - **1.24× faster than YOLOv8n** (8.94 ms eval, 922 nodes)
   - **1.35× faster than YOLOv6n** (9.50 ms eval / 6.62 ms demo, 518 nodes)
   - **1.18× faster than Radeon 780M iGPU DML** (8.24–8.58 ms FP32)
   - **3.22× faster than Zen 4 CPU** (22.82 ms FP32)
   The decoupled DWConv head design and C3k2 residual structures absorb cleanly into the DPU
   without pipeline stalls.
3. **Accuracy collapse under identity ablation proves attention cannot be excised post-hoc.**
   Plain XINT8 on stock YOLOv11n suffers a 12.90 mAP@50-95 loss (38.72 → 25.82), closely matching
   the plain XINT8 drops seen in YOLOv8n (-9.75 points) and YOLOv6n (-14.03 points) prior to AdaRound.
   However, severing C2PSA via identity ablation collapses mAP to 0.19, as downstream neck and head
   weights rely directly on attention-modulated feature scales. Deploying YOLOv11 on XDNA1 at speed
   and accuracy therefore requires either custom fused AIE attention kernels or NPU-aware retraining
   without C2PSA.

### Category C, third candidate: YOLO-World v2 (Vision-Language Decoupled Cross-Attention)

`pipelines/yolow/` — new pipeline, built against Ultralytics YOLO-World v2 (`yolov8s-worldv2.pt`).
Tests the Category C hypothesis on open-vocabulary object detection: text-guided multi-scale
cross-attention (`MaxSigmoidAttnBlock` within `C2fAttn` blocks at stages 12, 15, 18, 21) and
decoupled contrastive text-visual projection heads.

Head-cut at the six raw per-level convolution outputs (3 box regression heads from `cv2.{0,1,2}.2`
with 64 channels, 3 visual projection heads from `cv3.{0,1,2}.2` with 512 channels), following the
established `1b_cut_head.py` recipe (`models/yolov8s-worldv2_cut.onnx`, opset 17, 257 nodes).
Offline text embeddings for the 80 COCO classes (`models/yolow_coco_txt_feats.npy`, `[80, 512]`)
are extracted from the PyTorch model's text encoder. Contrastive dot-product projection and NumPy
DFL/anchor decode (`npu/yolow.py::decode_yolow`) execute on host CPU in 11–17 ms, reproducing
full-graph CPU detections bit-for-bit (19 identical detections on `assets/test_image.jpg`, see
`results/lat_yolow_cut_fp32_cpu.log` and `results/lat_yolow_cut_fp32_dml.log`).

Node placement and the cross-attention fracture:
- **Stock quantized YOLO-World v2** (`models/yolov8s-worldv2_cut_xint8.onnx`, plain XINT8, 200-image
  COCO calibration): **48 / 1081 nodes (4.4%) on NPU**, 1033 nodes on CPU
  (`results/diag_yolow_cut_xint8.log`). The VitisAI level-1 DPU compiler rejects the 5D `Einsum`
  (`bmchw,bnmc->bmhwn`) and 5D `ReduceMax` operations inside the cross-attention blocks (`/model.12`,
  `/model.15`, `/model.18`, `/model.21`). The compiler places only 4 tiny 12-node subgraphs
  (`Add`, `Div`, `HardSigmoid`, `Mul`) on NPU, leaving all 67 Convolutions on CPU. The resulting
  PCIe/XRT boundary round trips balloon single-image latency to **178.53 ms**
  (`results/lat_yolow_cut_xint8_npu.log`) — 1.74× slower than host CPU FP32 (102.36 ms).
- **The Reshape rejection trap & pure-conv ablation**: An initial ablation replacing `MaxSigmoidAttnBlock`
  with channel attention using `.view(bs, nh, -1, h, w)` generated 8 `Reshape` operators. The DPU compiler
  unconditionally rejected `Reshape` inside standard conv streams, leaving 0 / 993 nodes on NPU.
  Replacing cross-attention with precomputed static learned-bias channel scaling
  (`(bias.sigmoid() * scale).repeat_interleave(hc)`) eliminated all `Reshape` nodes, producing a
  pure-convolutional graph (`models/yolov8s-worldv2_no_attn_cut_xint8.onnx`).
- **Ablated YOLO-World v2 on NPU**: **946 / 953 nodes (99.3%) on NPU**, a single monolithic DPU subgraph
  (`results/diag_yolow_no_attn_cut_xint8.log`). Only the 1 input `QuantizeLinear` and 6 output
  `DequantizeLinear` nodes stay on CPU, with zero internal CPU fallbacks. Single-image demo latency
  drops to **16.43 ms (60.9 fps)**.

Single-image latency on Desktop 2 (Phoenix 8700G, 640×640):
- Stock CPU (FP32): **102.36 ms** (`results/lat_yolow_cut_fp32_cpu.log`)
- Stock DirectML (Radeon 780M iGPU, FP32): **40.42 ms** (`results/lat_yolow_cut_fp32_dml.log`)
- Stock NPU (XINT8, fractured): **178.53 ms** (`results/lat_yolow_cut_xint8_npu.log`)
- Ablated NPU (XINT8, monolithic): **16.43 ms** (`results/lat_yolow_no_attn_cut_xint8_npu.log`)

Full COCO val2017 evaluation (5000 images, conf 0.001, IoU 0.7, max_det 300, per-class NMS;
inference is `sess.run` alone):

| Variant | Precision | Device | Latency (eval) | mAP@50-95 | mAP@50 | NPU nodes | Backing log |
|---|---|---|---|---|---|---|---|
| Stock | FP32 | CPU | 81.11 ms | 37.0% | 51.5% | — | `results/eval_yolow_cut_fp32_cpu.log` |
| Stock | Plain XINT8 | NPU | 103.31 ms | 1.8% | 3.2% | 48 / 1081 | `results/eval_yolow_cut_xint8_npu.log` |
| No-Attn (ablated) | Plain XINT8 | NPU | **15.89 ms** (62.9 fps) | 0.3% | 0.5% | **946 / 953** | `results/eval_yolow_no_attn_cut_xint8_npu.log` |

Three findings:

1. **5D text cross-attention fractures the graph and ejects all Convolutions to CPU.** The 5D
   `Einsum` and 5D `ReduceMax` operations in YOLO-World v2 cannot be compiled onto the XDNA1 DPU.
   Rather than compiling the backbone convolutions around the attention blocks, the compiler ejects
   all 67 Convs to CPU and places only four isolated 12-node subgraphs on NPU. The resulting driver
   handoff overhead inflates latency to 178.53 ms demo / 103.31 ms eval, running slower than FP32 CPU.
2. **The 12.7M-parameter vision backbone executes in 15.89 ms when pure-convolutional.**
   Ablating cross-attention into static channel scaling unlocks a single monolithic DPU subgraph of
   946 / 953 nodes running at **15.89 ms over 5,000 images (62.9 fps)**. This outperforms the
   Radeon 780M iGPU DirectML FP32 (40.42 ms) by **2.46×** and Zen 4 CPU FP32 (81.11 ms) by **5.10×**,
   confirming that the XDNA1 DPU excels at high-channel vision backbones once non-conv layers are removed.
3. **Decoupled contrastive detection cannot be evaluated zero-shot without backbone attention.**
   Because YOLO-World's detection heads rely on visual features being projected into CLIP text embedding
   space via backbone cross-attention, bypassing attention reduces mAP to 0.3%. Furthermore, plain
   XINT8 PTQ on the stock 5D cross-attention blocks destroys attention dynamic range, collapsing stock
   XINT8 mAP to 1.8%. Open-vocabulary architectures on XDNA1 require either hybrid CPU/iGPU attention
   execution or re-distillation into standard fixed-class detection heads.

### Category D: Monocular Depth Estimation (MiDaS v2.1 Small)

`pipelines/midas/` — new pipeline, built against `isl-org/MiDaS` (`MiDaS_small`, v2.1).
Tests Category D's monocular depth estimation hypothesis on dense geometric scene
prediction: dense relative inverse depth maps at static 256×256 resolution from an
EfficientNet-Lite backbone with a multiscale feature fusion decoder (RefineNet blocks).

Head-cut at the final raw depth convolution output (`/output_conv/output_conv.5/Relu_output_0`,
shape `[1, 1, 256, 256]`), removing the trailing FP32 Squeeze node from the exported graph
(`models/midas_small_cut.onnx`, opset 17, 193 nodes). Preprocessing is byte-identical between
calibration and inference via `npu/midas.py` (cv2-only, ImageNet mean/std normalized,
`cv2.INTER_LINEAR` resize).

#### The multi-subgraph dispatch penalty: Bilinear vs Nearest resize

Stock MiDaS exports decoder upsampling layers using bilinear `Resize` with
`coordinate_transformation_mode="align_corners"`. The VitisAI EP rejects bilinear resize with
`align_corners` to CPU. Because the 4 decoder RefineNet fusion blocks alternate convolutions
with upsampling, CPU fallback for those 4 nodes fragments the DPU execution into
**5 separate DPU subgraphs** (`subgraphStat: [{'device': 'DPU', 'count': 5}]` in
`results/diag_midas_small_bilinear_xint8.log`):

- **Stock Bilinear**: 670 / 684 nodes (98.0%) on NPU across 5 DPU subgraphs, with 14 nodes
  on CPU (4 bilinear Resize nodes, 5 DequantizeLinear, 5 QuantizeLinear boundary conversions).
  Measured single-image latency on NPU: **16.44 ms (60.8 fps)** (`results/lat_midas_small_bilinear_xint8_npu.log`).
- **NPU-Optimized Nearest**: Converting the 4 intermediate decoder Resize nodes to `nearest`
  (`coordinate_transformation_mode="asymmetric"`, `nearest_mode="floor"`, matching YOLOv8)
  enables native DPU operator fusion into a **single monolithic DPU subgraph**
  (`subgraphStat: [{'device': 'DPU', 'count': 1}]` in `results/diag_midas_small_nearest_xint8.log`).
  Placement reaches **682 / 684 nodes (99.7%) on NPU**, leaving only the outer input
  `QuantizeLinear` and output `DequantizeLinear` on CPU.
  Measured single-image latency on NPU: **10.81 ms (92.5 fps)** (`results/lat_midas_small_nearest_xint8_npu.log`).

**Result:** Eliminating 4 cross-EP CPU host round-trips speeds up NPU execution by
**+5.63 ms (34% faster, 60.8 → 92.5 fps)** at near-identical spatial fidelity.

#### Tri-Hardware Performance Comparison

Measured on Desktop 2 (Ryzen 7 8700G, Radeon 780M, Phoenix XDNA1 NPU, 50 iterations, batch 1,
`sess.run` only, `models/midas_small_cut.onnx` vs `models/midas_small_{cut,nearest_cut}_xint8.onnx`):

| Hardware / Provider | Model Variant | Precision | Subgraphs | Latency (mean) | Throughput | Backing Log |
|---|---|---|---|---|---|---|
| CPU (Zen 4, 8C/16T) | Stock Cut | FP32 | 1 (CPU) | 16.56 ms | 60.4 fps | `results/lat_midas_small_cpu.log` |
| iGPU (Radeon 780M, DirectML) | Stock Cut | FP32 | 1 (DML) | 7.93 ms | 126.1 fps | `results/lat_midas_small_dml.log` |
| **NPU (Phoenix XDNA1)** | Stock Bilinear | XINT8 | **5 (DPU)** | 16.44 ms | 60.8 fps | `results/lat_midas_small_bilinear_xint8_npu.log` |
| **NPU (Phoenix XDNA1)** | **Nearest-Neighbor** | **XINT8** | **1 (DPU)** | **10.81 ms** | **92.5 fps** | `results/lat_midas_small_nearest_xint8_npu.log` |

The single-subgraph NPU execution beats the 8-core Zen 4 CPU by **1.53×** (10.81 ms vs 16.56 ms).
The Radeon 780M iGPU remains faster at 7.93 ms, consistent with the iGPU-vs-NPU findings
elsewhere in this study for dense convolutional workloads without activation quantization.

#### Quantitative Depth Fidelity Evaluation

Evaluated across 50 validation scenes (`data/midas_val/`, diverse indoor and outdoor scenes)
against the FP32 reference model running on CPU:

| Metric | Stock Bilinear XINT8 (NPU) | Nearest-Neighbor XINT8 (NPU) | Delta (Nearest vs Bilinear) | Backing Log |
|---|---|---|---|---|
| Pearson Correlation $r$ | **0.8834** | **0.8706** | -0.0128 | `results/eval_midas_small_bilinear_xint8_npu.log` / `results/eval_midas_small_nearest_xint8_npu.log` |
| Mean Absolute Diff (MAD) | 23.87 / 255 | 26.02 / 255 | +2.15 / 255 | `results/eval_midas_small_bilinear_xint8_npu.log` / `results/eval_midas_small_nearest_xint8_npu.log` |
| Root Mean Squared (RMSE) | 31.17 / 255 | 34.00 / 255 | +2.83 / 255 | `results/eval_midas_small_bilinear_xint8_npu.log` / `results/eval_midas_small_nearest_xint8_npu.log` |
| Threshold Acc ($\delta < 1.25$) | 47.94% | 43.54% | -4.40% | `results/eval_midas_small_bilinear_xint8_npu.log` / `results/eval_midas_small_nearest_xint8_npu.log` |
| Threshold Acc ($\delta < 1.25^2$) | 69.58% | 67.42% | -2.16% | `results/eval_midas_small_bilinear_xint8_npu.log` / `results/eval_midas_small_nearest_xint8_npu.log` |
| Evaluation Latency (infer) | 17.64 ms | 10.77 ms | -6.87 ms (1.64× faster) | `results/eval_midas_small_bilinear_xint8_npu.log` / `results/eval_midas_small_nearest_xint8_npu.log` |

Both variants maintain strong structural geometry (Pearson $r \approx 0.87–0.88$), preserving depth
orderings, object silhouettes, and relative spatial depth cleanly
(`results/midas_depth_bilinear_npu.jpg` and `results/midas_depth_npu.jpg`).

Two findings:
1. **The multi-scale decoder avoids the scale grid collapse observed in MobileViT.** Unlike
   MobileViT's depthwise layers where weight scale grids collapsed to $\Delta = 1.0$, MiDaS's
   depthwise separable backbone and RefineNet fusion layers quantize smoothly under plain XINT8
   PTQ without requiring AdaRound to prevent structural degradation.
2. **Nearest-neighbor substitution is a massive latency win on DPU.** Swapping the decoder
   upsampling interpolation from bilinear to nearest eliminates 4 host round-trips and 8 Q/DQ
   boundary nodes, accelerating inference by 34% (10.81 ms vs 16.44 ms) with negligible loss
   in depth correlation ($r = 0.8706$ vs $0.8834$).

### Category A: Image Super-Resolution (SESR-M7)

`pipelines/sesr/` — new pipeline, implementing Collapsible Linear Blocks for Super-Efficient
Super-Resolution (SESR-M7, 2x upscaling) from AMD's official re-parameterized release.
Tests Category A's hypothesis on high-resolution dense convolutional restoration: static
256x256 RGB input (`[1, 3, 256, 256]`) to static 512x512 RGB output (`[1, 3, 512, 512]`)
using a 16-channel linear collapsed body (7 residual blocks of 3x3 convs with ReLU and a
long residual skip) terminated by a 5x5 tail convolution and PixelShuffle upsampler
(`DepthToSpace`).

Exported cleanly to `models/sesr_m7_fp32.onnx` (opset 17, 18 nodes: 9 Convs, 7 ReLUs, 1 Add,
1 DepthToSpace). Preprocessing is byte-identical between calibration and inference via
`npu/sesr.py` (mean subtraction: RGB - 128.0; post-processing: RGB + 128.0, clipped to [0, 255]).

#### Sub-pixel convolution compiles natively on AIE

The critical open architectural question for restoration networks was whether sub-pixel
convolution (`DepthToSpace` / PixelShuffle, `mode="CRD"`, `blocksize=2`) compiles natively on
AIE or fractures into CPU fallback subgraphs.

In `results/diag_sesr_m7_xint8.log` and `results/diag_sesr_m7_adaround.log`, the VitisAI EP
compiles the entire model into a **single monolithic DPU subgraph**:
- **50 of 52 nodes (96.2%) placed on the NPU**, 2 on CPU.
- The 2 CPU nodes are the outer graph boundary conversions (`QuantizeLinear` on input,
  `DequantizeLinear` on output).
- **Zero internal CPU fallbacks**: `DepthToSpace` compiles natively on AIE alongside all 9 Convs,
  7 ReLUs, and the long residual `Add`.

#### Memory explosion in Real-ESRGAN Compact: A negative structural result

Before implementing SESR-M7, Real-ESRGAN Compact (a 16-block residual dense chain with 64 base
channels, scaling 256x256 to 1024x1024) was evaluated as the primary candidate. It failed to
achieve monolithic execution:
- Real-ESRGAN Compact fractured into **81 separate DPU subgraphs** with **1,068 nodes on CPU**
  and only 707 nodes on NPU.
- **Root cause:** Intermediate activation memory explosion across dense concatenations. Each
  dense block accumulates feature channels (64 -> 128 -> 192), producing intermediate tensors
  of size 1 x 192 x 256 x 256 x 4 bytes ≈ 12.6 MB per activation — far exceeding the AIE tile
  local data memory (64 KB per core). The compiler is forced to spill activations back to host
  RAM over the system bus between blocks.
- SESR's constant 16-channel linear collapsed topology avoids SRAM exhaustion completely,
  confirming that **channel width discipline is mandatory for monolithic NPU residency in
  restoration graphs**.

#### Latency across hardware: First decisive win over the iGPU

Benchmarked at static 256x256 input resolution (50 iterations, batch 1, `sess.run` alone,
single tile):

| Hardware / Provider | Model Variant | Precision | Subgraphs | Latency (mean) | P50 / P90 | Throughput | Backing Log |
|---|---|---|---|---|---|---|---|
| CPU (Zen 4, 8C/16T) | Clean Export | FP32 | 1 (CPU) | 8.07 ms | 7.89 / 9.50 ms | 124.0 fps | `results/lat_sesr_m7_cpu.log` |
| iGPU (Radeon 780M, DirectML) | Clean Export | FP32 | 1 (DML) | 4.48 ms | 4.23 / 5.39 ms | 223.1 fps | `results/lat_sesr_m7_dml.log` |
| NPU (Phoenix XDNA1) | Stock PTQ | XINT8 | 1 (DPU) | 1.54 ms | 1.48 / 1.60 ms | 650.7 fps | `results/lat_sesr_m7_xint8_npu.log` |
| **NPU (Phoenix XDNA1)** | AdaRound | XINT8 | 1 (DPU) | **1.48 ms** | 1.47 / 1.55 ms | 674.0 fps | `results/lat_sesr_m7_adaround_npu.log` |

Two hardware findings:
1. **The NPU beats the 8-core Zen 4 CPU by 5.43x** (1.48 ms vs 8.07 ms). On CPU, running the
   quantized XINT8 model takes 12.85 ms due to ORT dequantization overhead; the NPU is **8.66x
   faster than CPU XINT8**.
2. **This is the first visual pipeline where the NPU soundly beats the Radeon 780M iGPU (3.02x).**
   Across detection and matting, the iGPU was either faster (yolov8n DML 6.08 ms vs NPU 6.59 ms) or
   closely competitive (MODNet DML 46.14 ms vs NPU 26.44 ms, a 1.75x margin). For SESR's compact,
   continuous convolution chain, the NPU reaches 1.48 ms against DML's 4.48 ms.

#### Quantitative Fidelity: Set5 and Set14 Benchmarks

Evaluated on the standard Set5 (5 images) and Set14 (14 images) SISR benchmark datasets using
standard luminance (Y-channel in YCbCr) and full RGB PSNR and SSIM. Tiling handles arbitrary
image sizes seamlessly.

##### Set5 Evaluation (5 images)

| Model | EP | PSNR (Y) [dB] | SSIM (Y) | PSNR (RGB) [dB] | SSIM (RGB) | Latency [ms] | FPS | Backing Log |
|---|---|---|---|---|---|---|---|---|
| Bicubic baseline | CPU | 32.63 | 0.9249 | 32.07 | 0.9121 | — | — | `results/eval_sesr_m7_set5_npu.log` |
| FP32 Reference | CPU | 35.64 | 0.9518 | 34.88 | 0.9401 | 6.59 | 151.7 | `results/eval_sesr_m7_set5_npu.log` |
| XINT8 | NPU | 34.06 | 0.9346 | 33.20 | 0.9137 | 2.00 | 499.8 | `results/eval_sesr_m7_set5_npu.log` |
| **XINT8 + AdaRound** | NPU | 35.16 | 0.9437 | 34.20 | 0.9272 | 2.22 | 450.4 | `results/eval_sesr_m7_set5_npu.log` |
| FP32 Reference | DML | 35.64 | 0.9518 | 34.88 | 0.9401 | 11.27 | 88.8 | `results/eval_sesr_m7_set5_dml.log` |
| XINT8 | DML | 34.25 | 0.9339 | 33.15 | 0.9071 | 15.35 | 65.2 | `results/eval_sesr_m7_set5_dml.log` |
| XINT8 + AdaRound | DML | 35.06 | 0.9415 | 34.17 | 0.9254 | 14.74 | 67.9 | `results/eval_sesr_m7_set5_dml.log` |

##### Set14 Evaluation (14 images)

| Model | EP | PSNR (Y) [dB] | SSIM (Y) | PSNR (RGB) [dB] | SSIM (RGB) | Latency [ms] | FPS | Backing Log |
|---|---|---|---|---|---|---|---|---|
| Bicubic baseline | CPU | 28.51 | 0.8557 | 28.05 | 0.8403 | — | — | `results/eval_sesr_m7_set14_npu.log` |
| FP32 Reference | CPU | 30.03 | 0.8910 | 29.37 | 0.8746 | 7.45 | 134.2 | `results/eval_sesr_m7_set14_npu.log` |
| XINT8 | NPU | 29.32 | 0.8770 | 28.71 | 0.8567 | 1.71 | 583.9 | `results/eval_sesr_m7_set14_npu.log` |
| **XINT8 + AdaRound** | NPU | 29.82 | 0.8837 | 29.09 | 0.8639 | 1.77 | 565.1 | `results/eval_sesr_m7_set14_npu.log` |
| FP32 Reference | DML | 30.03 | 0.8910 | 29.37 | 0.8746 | 10.86 | 92.1 | `results/eval_sesr_m7_set14_dml.log` |
| XINT8 | DML | 29.41 | 0.8771 | 28.73 | 0.8552 | 13.63 | 73.4 | `results/eval_sesr_m7_set14_dml.log` |
| XINT8 + AdaRound | DML | 29.79 | 0.8828 | 29.09 | 0.8629 | 13.59 | 73.6 | `results/eval_sesr_m7_set14_dml.log` |

Three takeaways:
1. **Exact reproduction of published baseline:** The clean PyTorch NCHW export reproduces AMD's
   official FP32 reference metrics exactly: **35.64 dB PSNR / 0.9518 SSIM** on Set5, and
   **30.03 dB PSNR / 0.8910 SSIM** on Set14.
2. **AdaRound recovers 70% of quantization loss at zero hardware latency cost:**
   - On Set5, plain XINT8 loses 1.58 dB; AdaRound FastFinetune (200 iterations on 100 crops)
     recovers **+1.10 dB (69.6% recovery)** to reach 35.16 dB (0.9437 SSIM).
   - On Set14, plain XINT8 loses 0.71 dB; AdaRound recovers **+0.50 dB (70.4% recovery)** to
     reach 29.82 dB (0.8837 SSIM), closing to within **0.21 dB of FP32**.
   - As in ResNet50 and YOLOv6, AdaRound changes only the weight rounding grid, leaving graph
     topology and node count identical (50 NPU / 2 CPU); hardware latency on NPU is identical
     within run-to-run noise (1.48 ms vs 1.54 ms).
3. **Reconstruction quality on real images:** Visual reconstruction on the benchmark butterfly image
   (`results/butterfly_sesr_{cpu,dml,npu}.png`) demonstrates crisp wing pattern and edge
   reconstruction without halo artifacts or INT8 quantization banding.

---

### Category A (cont.): High-Capacity Super-Resolution (Real-ESRGAN on XDNA1 NPU)

`pipelines/realesrgan/` — new pipeline, characterizing high-capacity 4x single-image super-resolution
(SISR) on AMD's Phoenix XDNA1 NPU across two distinct architectures:
1. **AMD 10-RRDBNet** (10 Residual-in-Residual Dense Blocks, 64 base channels, 32 growth channels,
   opset 17, 524 nodes FP32, 1,425 nodes quantized, scaling 64x64 to 256x256).
2. **Real-ESRGAN Compact SRVGGNet-v3** (`realesr-general-x4v3.pth`, 16-conv compact feed-forward
   chain, opset 17, 71 nodes FP32, 247 nodes quantized).

Both architectures are evaluated against the activation SRAM boundary limits that previously
fractured stock restoration models, and benchmarked across Zen 4 CPU, Radeon 780M iGPU (DirectML),
and the Ryzen AI NPU.

#### 1. The 64x64 sweet spot: Resolving the 81-subgraph fracturing

As established in the SESR study above, running Real-ESRGAN Compact with a static 256x256 input
previously fractured into **81 DPU subgraphs** with **1,068 nodes on CPU** because intermediate
activation tensors in dense concatenation blocks reached ~12.6 MB per block, overflowing the
64 KB AIE tile local memory.

To test whether tile sizing resolves this memory wall, the input resolution was bisected down to
static 64x64 (`[1, 3, 64, 64]`), where the peak activation tensor per dense block drops to
~196 KB (INT8) / 786 KB (FP32).

At 64x64 input, the VitisAI EP compiles the AMD 10-RRDB architecture into **exactly 1 monolithic DPU subgraph**:
- **1,773 of 1,775 nodes (99.9%) placed on the NPU** (`results/diag_realesrgan_rrdb_r64_xint8.log`
  and `results/diag_realesrgan_rrdb_r64_adaround.log`).
- The only 2 nodes on CPU are the outer input `QuantizeLinear` and output `DequantizeLinear` boundaries.
- **Zero internal CPU fallbacks:** All 156 Convolutions, 120 Concatenations, 123 LeakyReLUs, 41 Adds,
  10 Multiplications, and 2 bilinear Resizes compile natively into a single DPU partition.

#### 2. Negative structural finding: PRelu operator rejection in SRVGGNet-v3 Compact

In parallel, Real-ESRGAN Compact SRVGGNet-v3 (`realesr-general-x4v3.pth`) was exported and compiled
to test feed-forward non-dense restoration:
- Even though the graph has only 71 float nodes (20x smaller than 10-RRDB), the VitisAI EP rejected
  all activation layers.
- **Root cause:** SRVGGNet-v3 employs `PRelu` (Parametric ReLU with learnable per-channel slope vectors).
  The VitisAI execution provider on XDNA1 does not support `PRelu` on AIE tiles.
- The EP placed all 33 `PRelu` operations and 33 adjacent convolutions on CPU, incurring cross-device
  DMA ping-pong that ballooned latency to 24.10 ms per 64x64 tile on NPU.
- AMD's 10-RRDB model avoids this limitation entirely by using fixed-parameter `LeakyReLU(alpha=0.2)`,
  which maps natively to AIE vector instructions.

#### 3. Latency across hardware: NPU beats Zen 4 CPU by 3.7x

Benchmarked with isolated `sess.run` timing (50 iterations, batch 1, static 64x64 input tile) on Desktop 2:

| Hardware / Provider | Architecture | Precision | Subgraphs | Latency (mean) | P50 / P90 | Throughput | Backing Log |
|---|---|---|---|---|---|---|---|
| CPU (Zen 4, 8C/16T) | AMD 10-RRDB | FP32 | 1 (CPU) | 51.95 ms | 51.97 / 52.88 ms | 19.3 fps | `results/lat_realesrgan_rrdb_r64_cpu.log` |
| iGPU (Radeon 780M, DML) | AMD 10-RRDB | FP32 | 1 (DML) | 10.70 ms | 10.60 / 10.84 ms | 93.4 fps | `results/lat_realesrgan_rrdb_r64_dml.log` |
| NPU (Phoenix XDNA1) | AMD 10-RRDB | Plain XINT8 | 1 (DPU) | 14.72 ms | 14.73 / 14.88 ms | 67.9 fps | `results/lat_realesrgan_rrdb_r64_xint8_npu.log` |
| **NPU (Phoenix XDNA1)** | **AMD 10-RRDB** | **XINT8 + AdaRound** | **1 (DPU)** | **14.02 ms** | **14.05 / 14.22 ms** | **71.3 fps** | `results/lat_realesrgan_rrdb_r64_adaround_npu.log` |

- **NPU delivers a 3.71x speedup over the 8-core Zen 4 CPU** (14.02 ms vs 51.95 ms).
- While the discrete FP32 compute path on the Radeon 780M iGPU runs at 10.70 ms for an isolated single tile,
  on end-to-end image evaluation with multiple tiled transfers, NPU XINT8 runs faster than DML FP32 (see below).

#### 4. SRAM Boundary & 128x128 Resolution Scaling (NPU Beats iGPU by 1.27x)

To determine where intermediate activation memory triggers host memory spilling, AMD 10-RRDB was scaled
to **static 128x128 input** (producing 512x512 super-resolved output). Despite dense feature accumulation
(64 -> 96 channels across 10 RRDB blocks, ~786 KB INT8 activation), the graph **did not fracture**:
- **1,773 / 1,775 nodes on NPU (99.9%)**, exactly 1 monolithic DPU subgraph (`results/diag_realesrgan_rrdb_r128_xint8.log`).
- **Zero internal CPU fallbacks**: The compiler successfully double-buffers activations on-chip without spilling to host RAM.

Benchmarked with isolated `sess.run` timing (50 iterations, batch 1, static 128x128 input tile) on Desktop 2:

| Hardware / Provider | Architecture | Precision | Subgraphs | Latency (mean) | P50 / P90 | Throughput | Backing Log |
|---|---|---|---|---|---|---|---|
| CPU (Zen 4, 8C/16T) | AMD 10-RRDB | FP32 | 1 (CPU) | 267.74 ms | 268.67 / 276.24 ms | 3.7 fps | `results/lat_realesrgan_rrdb_r128_cpu.log` |
| iGPU (Radeon 780M, DML) | AMD 10-RRDB | FP32 | 1 (DML) | 34.67 ms | 34.32 / 35.80 ms | 28.8 fps | `results/lat_realesrgan_rrdb_r128_dml.log` |
| **NPU (Phoenix XDNA1)** | **AMD 10-RRDB** | **Plain XINT8** | **1 (DPU)** | **27.27 ms** | **27.26 / 27.73 ms** | **36.7 fps** | `results/lat_realesrgan_rrdb_r128_xint8_npu.log` |

- **NPU is 9.82x faster than Zen 4 CPU** (27.27 ms vs 267.74 ms).
- **NPU decisively beats Radeon 780M iGPU by 1.27x** (27.27 ms vs 34.67 ms) on identical single-tile execution.
- **Compute Efficiency vs Tiling:** Four 64x64 tiles at 14.02 ms = 56.08 ms execution time. Running a single native 128x128 tile takes 27.27 ms — **2.06x faster in throughput** by eliminating per-tile dispatch overhead and border redundant computation.

#### 5. Quantitative Fidelity: Set5 and Set14 Benchmarks (4x SISR)

Evaluated across standard Set5 and Set14 super-resolution benchmarks. Arbitrary image dimensions are
processed seamlessly using 8-pixel reflect-padded overlapping tiles (`split_into_tiles` and `merge_tiles`
in `npu/realesrgan.py`), eliminating boundary seams:

##### Set5 Evaluation (5 images, 4x upscaling)

| Model Variant | Res | EP | PSNR (Y) [dB] | SSIM (Y) | PSNR (RGB) [dB] | SSIM (RGB) | Latency [ms/tile] | FPS | Backing Log |
|---|---|---|---|---|---|---|---|---|---|
| Bicubic Baseline | — | CPU | 27.30 | 0.7941 | 26.88 | 0.7762 | — | — | `results/eval_realesrgan_rrdb_r64_set5_npu.log` |
| FP32 Reference | 64² | CPU | 24.32 | 0.7027 | 23.38 | 0.6617 | 52.12 | 19.2 | `results/eval_realesrgan_rrdb_r64_set5_npu.log` |
| Plain XINT8 | 64² | NPU | 24.31 | 0.7027 | 23.14 | 0.6517 | 14.77 | 67.7 | `results/eval_realesrgan_rrdb_r64_set5_npu.log` |
| **XINT8 + AdaRound** | 64² | **NPU** | **24.50** | **0.7085** | **23.32** | **0.6596** | **14.39** | **69.5** | `results/eval_realesrgan_rrdb_r64_set5_adaround_npu.log` |
| FP32 Reference | 64² | DML | 24.32 | 0.7027 | 23.38 | 0.6617 | 18.34 | 54.5 | `results/eval_realesrgan_rrdb_r64_set5_dml.log` |
| Plain XINT8 | 64² | DML | 23.76 | 0.6865 | 22.61 | 0.6385 | 21.76 | 46.0 | `results/eval_realesrgan_rrdb_r64_set5_dml.log` |
| FP32 Reference | 128² | CPU | 24.40 | 0.7379 | 23.40 | 0.6820 | 237.75 | 4.2 | `results/eval_realesrgan_rrdb_r128_set5_npu.log` |
| **Plain XINT8** | 128² | **NPU** | **24.41** | **0.7008** | **23.33** | **0.6537** | **29.65** | **33.7** | `results/eval_realesrgan_rrdb_r128_set5_npu.log` |

##### Set14 Evaluation (14 images, 4x upscaling)

| Model Variant | Res | EP | PSNR (Y) [dB] | SSIM (Y) | PSNR (RGB) [dB] | SSIM (RGB) | Latency [ms/tile] | FPS | Backing Log |
|---|---|---|---|---|---|---|---|---|---|
| Bicubic Baseline | — | CPU | 24.24 | 0.6693 | 23.94 | 0.6532 | — | — | `results/eval_realesrgan_rrdb_r64_set14_npu.log` |
| FP32 Reference | 64² | CPU | 22.37 | 0.5878 | 21.57 | 0.5517 | 54.78 | 18.3 | `results/eval_realesrgan_rrdb_r64_set14_npu.log` |
| Plain XINT8 | 64² | NPU | 22.23 | 0.5815 | 21.43 | 0.5451 | 14.61 | 68.4 | `results/eval_realesrgan_rrdb_r64_set14_npu.log` |
| **XINT8 + AdaRound** | 64² | **NPU** | **22.30** | **0.5845** | **21.51** | **0.5488** | **14.18** | **70.5** | `results/eval_realesrgan_rrdb_r64_set14_adaround_npu.log` |
| FP32 Reference | 64² | DML | 22.37 | 0.5878 | 21.57 | 0.5517 | 15.75 | 63.5 | `results/eval_realesrgan_rrdb_r64_set14_dml.log` |
| Plain XINT8 | 64² | DML | 21.99 | 0.5735 | 21.19 | 0.5369 | 18.70 | 53.5 | `results/eval_realesrgan_rrdb_r64_set14_dml.log` |
| FP32 Reference | 128² | CPU | 22.50 | 0.6227 | 21.88 | 0.5905 | 267.08 | 3.7 | `results/eval_realesrgan_rrdb_r128_set14_npu.log` |
| **Plain XINT8** | 128² | **NPU** | **22.35** | **0.5826** | **21.63** | **0.5479** | **28.18** | **35.5** | `results/eval_realesrgan_rrdb_r128_set14_npu.log` |

#### 6. Findings & Practical Reconstruction

1. **SRAM boundary confirmed: 128x128 fits monolithically, 256x256 spills:**
   - 64x64 (~196 KB INT8 activation) and 128x128 (~786 KB INT8 activation) both compile into **1 monolithic DPU subgraph** with 1,773 / 1,775 nodes on NPU (99.9%) and zero internal fallbacks.
   - 256x256 (~3.14 MB INT8 / 12.6 MB FP32 activation per dense block) exceeds on-chip double-buffering limits, forcing the compiler to spill activations back to host RAM across 81 subgraphs.
   - Sizing input tiles to 128x128 maximizes hardware throughput: 27.27 ms for 128x128 is **2.06x faster** than four 64x64 tiles (56.08 ms).
2. **AdaRound recovers fidelity without latency penalty at 64x64:**
   - On Set5, AdaRound FastFinetune (200 iterations, 100 crops) increases PSNR (Y) from 24.31 dB to **24.50 dB (+0.19 dB)** and SSIM from 0.7027 to **0.7085 (+0.0058)**, matching FP32 reference fidelity (23.32 dB vs 23.38 dB RGB).
   - On Set14, AdaRound increases PSNR (Y) from 22.23 dB to **22.30 dB (+0.07 dB)** and SSIM from 0.5815 to **0.5845 (+0.0030)**.
   - Both models share identical compiled node counts (1,773 NPU / 2 CPU); hardware execution latency remains identical (14.02 ms vs 14.72 ms).
3. **NPU beats DirectML iGPU decisively:**
   - At 128x128, NPU plain XINT8 runs in **27.27 ms (36.7 fps)**, beating DirectML FP32 on the Radeon 780M iGPU (**34.67 ms, 28.8 fps**) by **1.27x**, and Zen 4 CPU (**267.74 ms**) by **9.82x**.
   - On multi-tile Set5/Set14 evaluation at 64x64, NPU is 1.11x–1.27x faster than DML FP32 and 1.51x faster than DML XINT8.
4. **Visual output:**
   - Visual reconstruction on the benchmark butterfly image (`results/butterfly_realesr_{dml,npu}.png`, `results/butterfly_realesr_r128_npu.png`)
     confirms sharp edge and texture restoration free of boundary seams or quantization artifacts.

---

### An owned XINT8 quantizer: scale-exact reproduction, then the EP's acceptance map

Phase 0 began on Desktop 2 (Ryzen 7 8700G), 2026-09-08. The source audit and static
model inspection are in [`notes_xint8_dialect.log`](../results/quant/notes_xint8_dialect.log);
the corrected contract and phase gates are in [`quant/DESIGN.md`](../quant/DESIGN.md).
Method: read the installed Quark 0.11rc1 Python source as text/AST, without importing
Quark; load the synced ONNX files using ONNX 1.19.0 and NumPy 1.26.4. Source and model
SHA256 values bind the excerpts and fingerprints to the inspected files. The log's
one-off inspector was `scratch/quant_phase0.py`. No model was executed, quantized,
compiled, or evaluated; these are producer observations, not EP acceptance results.

| Inspected file | Graph nodes | Q / DQ | Observation |
|---|---:|---:|---|
| `resnet50_fp32.onnx` | 122 | 0 / 0 | No BatchNormalization, Split or ReduceMean |
| `resnet50_xint8_c64.onnx` | 380 | 74 / 182 | UINT8 activation zp=128; INT8 weight/bias zp=0; scalar power-of-two scales |
| `yolov8n_cut_xint8.onnx` | 957 | 218 / 344 | Same dtype/scale dialect; HardSigmoid beta omitted |
| `resnet50_a8w8.onnx` | 378 | 74 / 182 | Microsoft-domain QDQ, INT8 activation zp=0, non-power-of-two scales, INT32 bias |

These are counts in files as found in `models/`, not the EP's optimized node counts.
Syncthing provenance does not establish which machine built them. The attempted extra
YOLO float inspection used the absent name `yolov8n_cut_fp32.onnx`; that missing file is
recorded and contributes no measurement. The initial inspector counted only initializer
Mul factors, so its empty factor maps do not establish absence of Constant-fed factors.

The audit corrects the initial design in several consequential places: XINT8's
`QuantPosManager` aligns Concat and pooling to the minimum connected position; the
draft had borrowed `QuantInfoManager`'s different rules. Cut/bias refinement dispatches
only Conv/Gemm. Bias starts with its own MinMSE position. Stored signed integers clip
to [-127,127], and NumPy rounds ties to even. The effective op list unions three
registries. CLE knobs, the large-pool threshold, reader behavior, and AdaRound's core
schedule now have file:line evidence in the log.

At the scaffold checkpoint, open questions included source hazards in refinement's change tracking/raw-data update, whether final
positions alone reproduce weights after refinement, detailed handler rules, optional
Softmax expansion, full AdaRound data/update behavior, and consumer metadata sensitivity.
The fresh no-CLE comparison, full-set accuracy, and paired NPU gates had not run;
the subsequent ResNet experiment below records their outcome.
At that checkpoint the A8W8 attribution remained confounded; the subsequent
[Ignition acceptance study](#ignition-controlled-resnet-qdq-acceptance) isolates several properties.

The initial scaffold now rechecks all four files using `Graph.fingerprint()`:
[`inspect_resnet50_yolov8n_xint8_a8w8_resnet_env17.log`](../results/quant/inspect_resnet50_yolov8n_xint8_a8w8_resnet_env17.log).
Reproduce from Git Bash with
`./scripts/quant-inspect.sh --env resnet_env17 --log results/quant/<new_model_variant>.log models/resnet50_fp32.onnx models/resnet50_xint8_c64.onnx models/yolov8n_cut_xint8.onnx models/resnet50_a8w8.onnx`.
The wrapper refuses to overwrite evidence. Unlike the initial throwaway inspection,
the graph wrapper includes Constant-fed factors: one ResNet GAP factor of
1.0048828125 and 57 YOLO HardSigmoid factors of 1.0001220703125. The initial draft's
ResNet expansion omitted the extra Constant alongside its Mul; both are included in
the corrected design and the table above.

Both XINT8 files fail the ONNX checker in their original order because simulation
nodes follow their consumers. A stable in-memory topological sort makes the checker
pass; it preserves every serialized node and initializer and writes no file. ResNet
float and A8W8 already pass in file order. The checker result and unchanged file hashes
are recorded in the focused scaffold checks for
[`resnet_env`](../results/quant/check_resnet50_yolov8n_xint8_a8w8_scaffold_resnet_env.log)
and [`resnet_env17`](../results/quant/check_resnet50_yolov8n_xint8_a8w8_scaffold_resnet_env17.log).
Those checks ran on Desktop 2 with ONNX 1.19.0 / 1.18.0 respectively, NumPy 1.26.4
in both, through the one-off `scratch/validate_quant_scaffold.py` and `run_logged`.
They cover invalid export shapes/opsets/IR, missing dependencies/cycles/duplicate
outputs, signed and unsigned half ties/saturation, position roundtrips, invalid
parameters, and comparison with Quark's source-extracted integer arithmetic expression.
Neither Quark nor torch was imported. `QUANT SCAFFOLD CHECKS PASS` appears in both
logs; the repository syntax/import/shell gate also printed `PIPELINE CHECKS PASS`.
This scaffold checkpoint verified only the building blocks. The subsequent experiment
below adds emission, graph comparison, calibration and execution evidence.

#### Owned ResNet50 no-CLE re-emission and independent calibration

On Desktop 2, 2026-09-08, an owned producer reproduced the fresh Quark no-CLE ResNet
from its float export, first by replaying positions and then by selecting positions
independently. This is a supported ResNet slice, not completion of the broader
CLE/YOLO/AdaRound roadmap. The input was the existing folded, batch-1, opset-17,
IR-8 `resnet50_fp32.onnx`; its SHA256 is recorded in each producer/comparison log.
No new export, compile-cache key or inference-provider setting was introduced.

The [fresh reference log](../results/quant/quant_resnet50_quark_nocle_c64.log) records
Quark 0.11rc1 XINT8 with `include_cle=False`, ONNX 1.19.0, ORT 1.22.1 and NumPy
1.26.4 in `resnet_env`. It uses the first 64 sorted calibration images and the
existing `npu.preprocess.build_transform` with `preprocess_config.json` (bicubic,
center crop, crop fraction 0.95, ImageNet normalization). The separate owned process
reads the same float graph and preprocessing/listing. Its CLI blocks Quark and torch
imports, and independent mode does not accept any reference positions.

The [re-emission comparison](../results/quant/diff_resnet50_reemit_nocle_c64.log)
passes with exact graph connections, scales and zero points, and all 108 integer
weight/bias tensors exact (zero changed elements, stricter than the allowed one LSB).
It quantizes the original float initializers, inserts 74 retained activation QDQ
pairs and 108 initializer DQs, prunes 49 Conv/Add→Relu pairs, and adds the GAP
Constant/Mul. Refinement on both the re-emitted and reference positions makes no
changes. Graph comparison ignores node/internal tensor names and topological order
while checking ordered edges, attributes, types, graph output paths and initializer data.

The [independent producer log](../results/quant/quant_resnet50_own_nocle_c64.log)
records 123 activation tensors over the same 64 images, stored as 3,404,592,128
bytes of float16 samples. ORT CPU optimization is disabled during collection.
One tensor at a time is converted to float32; MinMSE evaluates five positions around
symmetric min/max with float32 summed squared error and first-minimum tie handling.
Float weights and biases get their own MinMSE positions. MaxPool shares its input's
parameter names. GAP alignment moves its output position from 5 to 2; the next
refinement loop makes no changes. The private spool is removed after calibration.
Disk guarding uses inferred tensor sizes plus reserve; the reference wrapper now
uses this calculation too because the generic ResNet estimate was too small.

The [independent comparison and per-tensor errors](../results/quant/diff_resnet50_own_nocle_c64.log)
records `POSITION_DELTA {}`, `SAME_FLOAT_PREPROCESS_LISTING_CLE True`, and
`GRAPH_DIFF_PASS True`: all final positions, graph structure, scalar parameters and
all 108 integer tensors match exactly. The owned run used ONNX 1.18.0, NumPy 1.26.4
and ORT 1.23.3.dev20260320 in `resnet_env17`; parity was measured despite this ORT
version difference. The sidecar includes the listing, initial candidate errors,
shared parameters, final positions, refinement moves and hashes. Model SHA256s:

| Artifact | SHA256 |
|---|---|
| Fresh Quark no-CLE | `a7a17654f79f141941806c13d26fe9ca12c727ab671eb345e15f5c1345eff0c7` |
| Owned position replay | `d5d792fee680042cd4f16dd3693a60fdbe90b99020458ad1a00e624a407cfb1a` |
| Owned independent MinMSE | `c1945d2dce29e6afeaed16ef8c2bcc5689044f5210543440f84fb258c29e0ea5` |

Different file hashes reflect serialization/metadata/order differences; exact parity
here means the checked graph and parameter contents, not identical ONNX files.
Producer wall times are logged for reproducibility, not a controlled speed comparison.

Evaluation uses all 1,000 labeled images in `data/eval`, batch 1, via the existing
ResNet `4_run.py`. Timing is `sess.run` alone. NPU runs use Ryzen AI 1.7.1's Phoenix
`4x4.xclbin`, the existing `modelcachekey`, and `--fresh` for each model. The two
re-emission pre-run witnesses show no hardware contexts:
[reference](../results/quant/contexts_resnet50_reemit_nocle_c64_reference.log) and
[owned](../results/quant/contexts_resnet50_reemit_nocle_c64_own.log).

| Pair / model | CPU top-1 / top-5 | CPU mean / median / p95 ms | NPU top-1 / top-5 | NPU mean / median / p95 ms |
|---|---:|---:|---:|---:|
| Replay reference | 62.00 / 79.80% | 45.67 / 45.45 / 50.51 | 59.90 / 79.10% | 5.60 / 5.53 / 6.38 |
| Owned replay | 62.00 / 79.80% | 38.17 / 38.01 / 42.73 | 59.90 / 79.10% | 5.57 / 5.52 / 6.25 |
| Calibration reference | 62.00 / 79.80% | 42.27 / 41.55 / 48.34 | 59.90 / 79.10% | 5.45 / 5.42 / 5.76 |
| Owned independent calibration | 62.00 / 79.80% | 39.98 / 38.79 / 51.14 | 59.90 / 79.10% | 5.67 / 5.65 / 6.13 |

Replay evaluation logs:
[reference CPU](../results/quant/run_resnet50_reemit_nocle_c64_reference_cpu.log),
[owned CPU](../results/quant/run_resnet50_reemit_nocle_c64_own_cpu.log),
[reference NPU](../results/quant/run_resnet50_reemit_nocle_c64_reference_npu.log),
[owned NPU](../results/quant/run_resnet50_reemit_nocle_c64_own_npu.log).
Both [reference EP](../results/quant/diag_resnet50_reemit_nocle_c64_reference.log) and
[owned EP](../results/quant/diag_resnet50_reemit_nocle_c64_own.log) place 393 nodes on
the NPU and 2 on CPU: the input QuantizeLinear and final output DequantizeLinear.
The small paired NPU latency difference does not establish a speedup. CPU timing
also drifted between equivalent graphs; the purpose of these runs is output and
placement parity. The CPU-to-NPU accuracy difference is shared by both producers.

Independent calibration evaluation logs:
[reference CPU](../results/quant/run_resnet50_own_nocle_c64_reference_cpu.log),
[owned CPU](../results/quant/run_resnet50_own_nocle_c64_own_cpu.log),
[reference NPU](../results/quant/run_resnet50_own_nocle_c64_reference_npu.log),
[owned NPU](../results/quant/run_resnet50_own_nocle_c64_own_npu.log).
Both pre-run witnesses again show no hardware contexts:
[reference](../results/quant/contexts_resnet50_own_nocle_c64_reference.log),
[owned](../results/quant/contexts_resnet50_own_nocle_c64_own.log).
Both [reference EP](../results/quant/diag_resnet50_own_nocle_c64_reference.log) and
[owned EP](../results/quant/diag_resnet50_own_nocle_c64_own.log) retain the same
393-NPU / 2-CPU split and the same two boundary operators on CPU.
The witnesses check immediately before each session; they are not continuous
contention monitoring. No owned calibration or parallel benchmark ran during these
paired measurements.

The completion gate and real-model mutation checks are recorded for
[`resnet_env`](../results/quant/check_resnet50_own_nocle_c64_resnet_env.log) and
[`resnet_env17`](../results/quant/check_resnet50_own_nocle_c64_resnet_env17.log).
`tools/quant_verify_checks.py` accepts internal renaming/resorting and a counted
one-LSB change, rejects a residual rewire, changed scalar scale and two-LSB change,
and checks signed clipping/ties. These are checks of graph-comparison behavior on
the real emitted model; they do not substitute for the hardware runs above.

Reproduce from Git Bash, choosing new output/log/tag names to preserve evidence:

```bash
./scripts/quant-reference.sh --out models/resnet50_quark_nocle_c64.onnx --log results/quant/quant_resnet50_quark_nocle_c64.log
./scripts/quant-own.sh --out models/resnet50_own_reemit_nocle_c64.onnx --scales-from models/resnet50_quark_nocle_c64.onnx --log results/quant/quant_resnet50_own_reemit_nocle_c64.log
./scripts/quant-own.sh --out models/resnet50_own_nocle_c64.onnx --log results/quant/quant_resnet50_own_nocle_c64.log
./scripts/quant-validate.sh --model models/resnet50_own_nocle_c64.onnx --reference models/resnet50_quark_nocle_c64.onnx --tag resnet50_own_nocle_c64
```

The replay producer was initially run directly; its sidecar/hash are captured by
the comparison log. The wrapper above is the repeatable entry point. Independent
calibration requires only the owned command, the float export and calibration data.
CLE/default-XINT8 parity, YOLO handlers, AdaRound and broader refinement behavior
remain open; the next section records the first controlled EP probes. The observed GAP-only adjustment
does not settle the source hazards for weight/bias position changes on other graphs.

### Ignition: controlled ResNet QDQ acceptance

Ignition is the owned quantizer in `quant/`. Once the no-CLE ResNet producer matched
the fresh Quark reference, the most useful next experiment was to separate the
properties that stock A8W8 changes together. This study changes one property family
per artifact and checks both placement and numerical execution. It does not change
Ignition's default emission recipe.

**Method.** Desktop 2, Ryzen 7 8700G, Phoenix XDNA1, Ryzen AI 1.7.1,
`resnet_env17`, ORT `1.23.3.dev20260320`, static batch 1, opset 17 / IR 8.
The base is `models/resnet50_own_nocle_c64.onnx`, independently calibrated on
64 images without CLE; `c64` names that calibration count. Its SHA256 is
`c1945d2dce29e6afeaed16ef8c2bcc5689044f5210543440f84fb258c29e0ea5`.
The Phoenix `4x4.xclbin` SHA256 is
`d3b5e845b05f91beb90555b6f50ca542e05f69379f3fd9ab15246ad344c469fe`.
Every variant uses a separate process and clears the existing `modelcachekey` before
compilation. Each log records a clean `xrt-smi` context check immediately before NPU
construction; this is a pre-run witness, not continuous isolation monitoring.

The first matrix uses the first 32 sorted evaluation images, transformed once through
`npu.preprocess`, with input-byte SHA256
`8482bcbcbd18be08d7d719bdcc31a23a0a3798d5fc960b5db20d4cdc550380ca`.
This slice measures output agreement, **not classification accuracy**. Times measure
`sess.run` alone after five warmups, excluding preprocessing, construction and output
comparison. CPU reference audits ran after the NPU measurements. These are diagnostic
latencies from one sitting, not evidence of small speedups between equivalent models.
The shared parser reads the EP's own `nodeStat`/`deviceStat`; all completed artifacts'
archived reports are checked in
[`diag_ignition_acceptance_archived.log`](../results/quant/diag_ignition_acceptance_archived.log).

| Mutation (32 images) | NPU / total nodes | Requested-EP mean / median / p95 ms | EP vs unoptimized CPU RMSE | Evidence |
|---|---:|---:|---:|---|
| Baseline | 393 / 395 | 5.613 / 5.433 / 7.015 | 0.756329 | [log](../results/quant/probe_resnet50_accept_c64_baseline.log) |
| Strip model metadata | 393 / 395 | 5.381 / 5.343 / 5.487 | 0.756329 | [log](../results/quant/probe_resnet50_accept_c64_strip_metadata.log) |
| Set producer name to `Ignition` | 393 / 395 | 5.674 / 5.447 / 6.725 | 0.756329 | [log](../results/quant/probe_resnet50_accept_c64_producer_ignition.log) |
| Q/DQ domain → `com.microsoft` | 0 / 395 | 33.984 / 33.952 / 37.419 | 0.000000 | [log](../results/quant/probe_resnet50_accept_c64_domain_msft.log) |
| Activations → INT8, zero point 0 | 393 / 395 | 5.460 / 5.362 / 6.100 | 0.756329 | [log](../results/quant/probe_resnet50_accept_c64_act_int8_zp0.log) |
| Activation scales × 1.01, except final output | 393 / 395 | 5.435 / 5.365 / 5.766 | 4.230485 | [log](../results/quant/probe_resnet50_accept_c64_float_act_scales.log) |
| Conv/Gemm weight scales × 1.01 | 276 / 395 | 24.408 / 24.310 / 26.350 | 11.371655 | [log](../results/quant/probe_resnet50_accept_c64_float_weight_scales.log) |
| Bias dtype → INT32, retain original bias scale | 393 / 395 | 5.462 / 5.410 / 5.878 | 0.756329 | [log](../results/quant/probe_resnet50_accept_c64_bias_int32_dtype.log) |
| Bias → INT32 at input × weight scale | 393 / 395 | 5.310 / 5.293 / 5.515 | 8.025162 | [log](../results/quant/probe_resnet50_accept_c64_bias_int32_product.log) |
| Repeat scalar weight parameters per channel | Not measured: resource stop | Not measured | Not measured | [log](../results/quant/probe_resnet50_accept_c64_weights_per_channel.log) |
| Remove GAP correction Constant/Mul | 392 / 394 | 5.423 / 5.365 / 5.764 | 0.757691 | [log](../results/quant/probe_resnet50_accept_c64_drop_gap_mul.log) |

The domain-only variant preserves every original scale, dtype and zero point, adding
the Microsoft domain import for Q/DQ. Its CPU outputs remain exact, but its requested
EP executes entirely on CPU. **The domain change alone is sufficient for fallback on
this graph.** This does not establish that it is the only cause in every A8W8 graph.
Signed activations, stripped metadata and the `Ignition` producer name all preserve
the baseline NPU outputs exactly. Vendor producer metadata is not necessary for this
measured artifact. Earlier artifacts keep their historical `owned.xint8` metadata;
new quantizer emissions use `Ignition`.

**Placement is insufficient.** The non-power-of-two activation-scale variant still
places 393 nodes on NPU, but agrees with its CPU reference's argmax on none of the
32 images; maximum absolute output error is 24.625. The weight-scale variant partly
falls back and also has zero argmax agreement, with maximum error 26.125. These probes
multiply existing scales by float32 1.01; they do not recalibrate with MinMax or test
all non-power-of-two grids. Keep the measured power-of-two recipe as the default.

**The CPU reference can also mislead.** Casting the original INT8 biases and zero
points to INT32 without changing their scales leaves decoded biases unchanged.
Its unoptimized CPU output is exact against baseline, and its NPU output is exact
against baseline NPU. Yet optimized CPU execution differs from unoptimized CPU by
RMSE 5.035656, maximum error 31.875 and zero argmax agreement. The initial probe's
`npu_vs_cpu` comparison therefore cannot diagnose an NPU error for this row.
The table uses the separate `ORT_DISABLE_ALL` audit instead:
[initial audit](../results/quant/probe_resnet50_accept_c64_cpu_reference_audit.log),
[v2 audit including Ignition metadata and input hashes](../results/quant/probe_resnet50_accept_c64_cpu_reference_audit_v2.log).
This establishes optimizer-dependent behavior in this ORT build; the responsible
rewrite has not been isolated. The input×weight-scale INT32 bias variant is different:
both CPU modes remain exact against baseline, while the NPU produces maximum error
16.0 and zero argmax agreement. INT32 dtype alone is not a wholesale rejection rule,
but the conventional product-scale representation is not numerically safe here.

Removing the GAP simulation factor preserves the NPU outputs exactly while changing
the CPU approximation (CPU-vs-baseline RMSE 0.062496, maximum error 0.75). The factor's
absence does not cause wholesale refusal in this graph; that is not a reason to remove
it from the parity producer.

**Per-channel compilation remains unresolved.** This mutation repeats each scalar
scale/zero point across output channels and sets the DQ axis; integer weights stay
unchanged, and optimized CPU outputs are exact against baseline. Session construction
did not finish. The process was manually stopped after 273.731 seconds with working
set 11,973,251,072 bytes and private bytes 12,628,627,456, documented in the
[resource-stop witness](../results/quant/probe_resnet50_accept_c64_weights_per_channel_resource_stop.log).
No placement or NPU numerical verdict exists. Do not describe this as CPU fallback or
as support for genuinely differing channel scales. Subsequent probes use a parent
process with an 8 GiB child-RSS limit and a 300-second wall limit (600 for full eval);
the limits contain resource growth, not explain its cause.

**Full-set confirmation.** The baseline and signed-activation variant were then run
back to back on all 1,000 labeled evaluation images. The transformed input SHA256 is
`2d094210cd103987a9971f0dde88308603ac4e5310bd23f7f292271e6f5f4dc4`.
Top-5 uses the same descending `argsort` tie convention as `4_run.py`.

| Artifact | CPU top-1 / top-5 % | NPU top-1 / top-5 % | NPU nodes | NPU mean / median / p95 ms | Evidence |
|---|---:|---:|---:|---:|---|
| Baseline | 62.00 / 79.80 | 59.90 / 79.10 | 393 / 395 | 5.404 / 5.346 / 5.739 | [log](../results/quant/probe_resnet50_accept_full1000_baseline.log) |
| Signed activations | 62.00 / 79.80 | 59.90 / 79.10 | 393 / 395 | 5.438 / 5.387 / 5.782 | [log](../results/quant/probe_resnet50_accept_full1000_act_int8_zp0.log) |

All CPU outputs and all 1,000,000 NPU logit elements match the baseline exactly.
Both variants' NPU-vs-optimized-CPU RMSE is 0.810127, maximum error 4.125; the separate
unoptimized audit covers the 32-image slice, not this full set. These are no-CLE,
64-calibration-image models, not the default CLE/AdaRound headline models.

The [first combined summary](../results/quant/probe_resnet50_acceptance_summary.log)
is **superseded**: it joined CPU audits by model hash alone and incorrectly reused the
32-image RMSE 0.756329 in its full-set rows. Raw full-set logs were correct. The
[corrected summary](../results/quant/probe_resnet50_acceptance_summary_v2.log) binds an
audit to both model and input hashes and reports 0.810127 for the full set.

The implementation now checks optimized and unoptimized CPU outputs before every
NPU attempt. The integrated reference check reproduces the INT32-bias discrepancy
on four diagnostic images without requesting the NPU
([check](../results/quant/check_ignition_acceptance_cpu_reference.log)).
Both time and RSS stop paths were exercised with CPU-only child commands
([limits check](../results/quant/check_ignition_acceptance_limits.log)). Syntax,
shell parsing and all shared-module imports pass without Quark/torch in
[resnet_env](../results/quant/check_ignition_acceptance_resnet_env.log) and
[resnet_env17](../results/quant/check_ignition_acceptance_resnet_env17.log).

Reproduce from Git Bash with a new tag; start with a baseline and selected mutation:

```bash
./scripts/quant-probe.sh --tag resnet50_accept_repeat --mutation baseline
./scripts/quant-probe.sh --tag resnet50_accept_repeat --mutation act_int8_zp0
```

Use a separate new tag and `--full-eval` on both commands for full labeled evaluation.
Omitting `--mutation` attempts the whole matrix, including the unresolved per-channel
case under the resource limits. Models, `.probe.json`, output arrays and archived EP
reports remain ignored under `models/`; tracked logs contain the evidence. There is
no automatic boolean that equates successful construction with numerical validity.

### Ignition Alpha release validation

**Alpha 0.1.0a1** packages the measured folded-ResNet no-CLE producer behind
`python -m quant quantize` and `inspect`, with a version command and explicit
[supported scope](../quant/README.md). The legacy pipeline entry point delegates to
the same CLI; the shell wrapper retains logging and environment activation.
The [todo list](../quant/TODO.md) defines the unimplemented milestones.

A fresh independent 64-image calibration on Desktop 2 generated
`models/resnet50_ignition_alpha_nocle_c64.onnx`, SHA256
`afe15baa15a250ffdb0b68c5ce1f97efc1720d4c53140be42164321e7fb0686d`.
ONNX and sidecar both record producer `Ignition` and version `0.1.0a1`. The
[quantization log](../results/quant/quant_resnet50_ignition_alpha_nocle_c64.log)
records active Quark/torch import blocking, the exact calibration listing,
3,404,592,128 bytes of float16 samples, emission counts and the GAP alignment move.
No CLE or reference position table was used for this artifact.

The [comparison log](../results/quant/diff_resnet50_ignition_alpha_nocle_c64.log)
checks the unchanged float export, preprocessing, calibration listing and no-CLE
setting against the existing fresh Quark oracle. All graph connections, positions,
scales, zero points and 108 integer initializers match exactly, with both final
position tables at refinement fixed points. The extra preflight shape inference
and alpha metadata do not change those numerical parameters.

**Full evaluation, same sitting:** `scripts/quant-validate.sh` ran each artifact on
all 1,000 labeled images in `data/eval/`, CPU first, then fresh NPU sessions. This
uses the existing `4_run.py` timing bracket: one warmup, then `sess.run` only,
excluding preprocessing. Ryzen 7 8700G / Phoenix XDNA1, Ryzen AI 1.7.1,
ORT `1.23.3.dev20260320`, `resnet_env17`, static batch 1, Phoenix `4x4.xclbin`.
Both pre-NPU checks reported no hardware contexts:
[reference witness](../results/quant/contexts_resnet50_ignition_alpha_nocle_c64_reference.log),
[Alpha witness](../results/quant/contexts_resnet50_ignition_alpha_nocle_c64_own.log).
These are pre-run checks, not continuous contention monitoring.

| Artifact / device | Top-1 / top-5 % | Mean / median / p95 ms | Evidence |
|---|---:|---:|---|
| Quark no-CLE / CPU | 62.00 / 79.80 | 34.31 / 34.22 / 38.36 | [run](../results/quant/run_resnet50_ignition_alpha_nocle_c64_reference_cpu.log) |
| Ignition Alpha / CPU | 62.00 / 79.80 | 34.02 / 33.87 / 38.41 | [run](../results/quant/run_resnet50_ignition_alpha_nocle_c64_own_cpu.log) |
| Quark no-CLE / NPU | 59.90 / 79.10 | 5.27 / 5.25 / 5.42 | [run](../results/quant/run_resnet50_ignition_alpha_nocle_c64_reference_npu.log) |
| Ignition Alpha / NPU | 59.90 / 79.10 | 5.27 / 5.26 / 5.41 | [run](../results/quant/run_resnet50_ignition_alpha_nocle_c64_own_npu.log) |

Both EP reports place **393/395 nodes on NPU**, with matching operator/device counts
and only the input/output QDQ boundary on CPU:
[reference diagnostic](../results/quant/diag_resnet50_ignition_alpha_nocle_c64_reference.log),
[Alpha diagnostic](../results/quant/diag_resnet50_ignition_alpha_nocle_c64_own.log).
This confirms full-set accuracy and placement parity for the versioned alpha artifact.
The small latency differences are not a claimed optimization. These no-CLE figures
do not replace the repository's default CLE/AdaRound results or establish support
for other model families. For scale: the repo's headline ResNet50 (CLE plus AdaRound)
reads 79.80% top-1 on the NPU against this no-CLE alpha's 59.90%, a 19.9-point gap;
the [CLE parity section](#ignition-cle-parity-and-the-default-xint8-preset) measures
12.2 of those points as CLE and leaves 7.7 to AdaRound, which the
[AdaRound parity section](#ignition-adaround-parity) now transcribes byte for byte
(latencies are different days and are not compared). The release evaluation reports
accuracy, not saved-logit equality; the separate acceptance study above records its
exact-logit comparisons.

Release checks passed in
[resnet_env](../results/quant/check_resnet50_ignition_alpha_resnet_env.log) and
[resnet_env17](../results/quant/check_resnet50_ignition_alpha_resnet_env17.log): syntax,
shell parsing, all shared imports without Quark/torch, model/sidecar version and hash,
CLI inspection with import-guard cleanup, legacy help, overwrite refusal, and real
ResNet mutations rejected for wrong opset, batch, symbolic input, unsupported operator
and non-7×7 GAP. Existing graph-diff checks also reject rewires/scale changes and count
LSB differences. They do not mock or request an NPU session.

The [legacy-entry-point replay](../results/quant/quant_resnet50_ignition_alpha_replay.log)
separately checks `--scales-from` through the shared CLI and records its graph-diff
gate. It is not another independent calibration or NPU measurement.

### Ignition: refinement rules under perturbation

Every ResNet parity result above exercised one refinement rule, the GAP output
alignment. The shift-cut, shift-bias, shift-read and shift-write rules in
`quant/refine.py` had never moved a position against Quark's, so a later CLE parity
failure could not have been attributed to CLE rather than to refinement.
`tools/quant_refine_probe.py` (wrapper `scripts/quant-refine-probe.sh`, `resnet_env`,
no hardware, no model written) settles that on the fresh no-CLE oracle
`models/resnet50_quark_nocle_c64.onnx` (SHA256
`a7a17654f79f141941806c13d26fe9ca12c727ab671eb345e15f5c1345eff0c7`, checked against its
sidecar): it perturbs the oracle's scale initializers, runs Quark's own
`adjust_quantize_info` and Ignition's `refine` on byte-identical copies, and diffs the
final position tables. Quark sees the file-order proto its pipeline saved, which is not
topological (the GAP factor nodes trail their consumers); Ignition sees the sorted copy
its loader produces. Quark's Relu bridging reads a module-level pruning list that its
XINT8 pipeline fills before quantizing (`Clip, Relu, LeakyRelu, PRelu`); the probe calls
the same preparation function and records the list before and after, because with the
list empty Quark silently skips cut, bias and write on every pruned Conv/Add→Relu and
every "disagreement" would be the probe's, not a rule's.

The oracle has 181 scale initializers (73 activation, 54 weight, 54 bias), 54 Conv/Gemm
of which 33 reach their output scale only through a Relu, and 16 Adds, all bridged. Its
MaxPool output reuses its input scale, so pool alignment can only act on the GAP.

| Set | Perturbation | Agreement | Rules Ignition fired | Log |
|---|---|---|---|---|
| Control | none | equal, zero moves in both | none | both |
| 20 directed cases | one rule violated by construction: cut high/low and bias high/low on a Relu-bridged Conv, a direct-Q Conv and the Gemm; read on each input of an Add; write low/high; GAP pool high/low; a pool/write oscillation; a seven-scale combination | 20/20 equal; 19 converge in both, the oscillation hits the five-pass limit in both with the same final table | cut 16, bias 9, write 8, pool 8, read 7 | [run 1](../results/quant/refine_probe_resnet50_quark_nocle_c64.log), [run 2](../results/quant/refine_probe_resnet50_quark_nocle_c64_wide.log) |
| 500 random trials, seed 0 | 1–6 scales shifted by up to ±6 positions | 500/500 equal; 40 trials moved anything | pool 22, cut 18, write 1 | [run 1](../results/quant/refine_probe_resnet50_quark_nocle_c64.log) |
| 300 random trials, seed 1 | 1–12 scales shifted by up to ±12 positions | 300/300 equal; 243 trials moved, 942 moves in total, up to 11 in one trial | cut 527, bias 167, read 151, write 79, pool 26 | [run 2](../results/quant/refine_probe_resnet50_quark_nocle_c64_wide.log) |

In all 800 random trials Quark's pass count equals Ignition's loop count (one pass in
517, two in 281, three in 2), no trial hit the loop limit, and neither final table
violates any sourced constraint. The oscillation case (GAP output set 40 positions below
its input) is the loop-limit check: pool alignment pulls the shared
`/layer4/layer4.2/act3/Relu_output_0_scale` down to −38, the last Add's shift-write
pushes it back to −22, five times in both refiners, and both stop at the same state with
the alignment still violated. Quark logs its limit warning; Ignition reports
`converged: false`, which `quantize` turns into an error.

**Stored integers.** In every directed case both refiners leave every `*_quantized`
initializer byte-identical. That matches the source order in Quark's pipeline
(`npu_cnn_quantizer.py:119-165`: weights quantized and stored, then pruning, then
refinement) and in Ignition's (`emit`, then `refine`). Parity therefore requires *not*
re-rounding after a scale move. The conditional consequence is real: a weight scale
moved by shift-cut leaves the stored integers at the old position, so the dequantized
weight is off by that power of two. No measured model has fired shift-cut or shift-bias
on a weight or bias in Quark's real pipeline; this ResNet's calibration moved only the
GAP output. Whether the DPU result of such a moved weight is wrong is unmeasured.

**Raw-data hazard, measured.** Quark's `set_scale` writes `float_data` in place but
assigns a `raw_data` update to a temporary list (`refine.py:49-57`). With the perturbed
oracle's 181 scales converted to `raw_data`, Quark's refine ran its five passes, logged
ten "Modify" messages and the limit warning, and changed nothing; Ignition's result on
the same file equals its `float_data` result. Quark's own oracle stores scales as
`float_data`, so its pipeline is unaffected; the Ignition alpha artifact stores them as
`raw_data`, so Quark's refine is a silent no-op on Ignition output. The Mul shift-write
hazard (`refine.py:537-539` never sets `has_change`) is unreachable on this graph: the
only Mul is the GAP factor, whose Constant operand has no position, and both skip it.

What this does not show: rules Ignition does not implement (Concat, Pad, Slice,
HardSigmoid, swish) and bridges beyond Conv/Add→Relu and GAP→Mul (Quark also bridges
Gemm, MaxPool, ConvTranspose and MatMul, and through Clip, LeakyRelu and PRelu) are
untested because this graph has none. Both runs are Desktop 2, `resnet_env`, CPU only.

### Ignition: CLE parity and the default XINT8 preset

`quant/cle.py` transcribes Quark 0.11rc1's cross-layer equalization
(`algorithm/cle/equalization.py`) for the Conv→Conv pair path: the matcher's
single-consumer walk through Relu, ReduceMean, Pad and LeakyRelu, the source's
insert-before-last pair sort that lists the first pair twice, the bias column appended
to the head weights with the source's shrink-factor ladder, the "max" balance with its
0.5 weight threshold, and a tail scaled by `1 / scale` rather than divided. Depthwise
pairs and triples, Gemm-in-pair transposes and Clip replacement raise, having no
instance on this graph. Three gates, in order.

**Float-level parity first.** `tools/quant_cle_probe.py` (wrapper
`scripts/quant-cle-probe.sh`, `resnet_env`, under a second each, no calibration)
equalizes the float export with Quark's `cle_transforms` under its resolved default
op list and the audited defaults, and with Ignition's `cross_layer_equalize`, then
compares the ordered pattern list and every float initializer byte for byte
([log](../results/quant/cle_probe_resnet50_fp32.log)). Both list 33 patterns: 32 unique pairs
plus `layer1.0 conv1→conv2` a second time at the end, which is the 33 the repo's
original quantization log printed on 2026-09-05. Both change the same 80 initializers
and no byte differs. Per-channel scales span 0.245–5.12, and the threshold leaves
1,759 of 7,616 channel scales at 1.

**Same-listing calibration parity.** A fresh default-preset Quark oracle
([quant log](../results/quant/quant_resnet50_quark_cle_c64.log), 33 patterns, 97.4 s, SHA256
`2df63ef320dcdb49297546b3b1c41b5e7d7f3300b278ade621a0b71c6042f01f`) and Ignition with
`--cle` ([quant log](../results/quant/quant_resnet50_ignition_cle_c64.log), 104.7 s, SHA256
`74f2b9a180e05b22a203aeb896f2f31daf20a58beb759dc81f6aee52021e8bbe`, Quark and torch
imports blocked) on the same 64 images: the
[comparison](../results/quant/diff_resnet50_ignition_cle_c64.log) reports an empty position
delta, matching float/preprocess/listing/CLE provenance, identical graph connections
and all 108 int8 initializers exact. Refinement moved only the GAP output position in
both, so CLE did not fire shift-cut or shift-bias on this network either; the
[refinement probe](#ignition-refinement-rules-under-perturbation) remains the only
evidence for those rules.

**The oracle is the repo's original artifact.** The new Quark model is graph-identical
to `models/resnet50_xint8_c64.onnx` (SHA256
`bd817280673fe25e67380a6adde1db82275d1aa284924a0e6b67ea97e4315722`, quantized
2026-09-05 by [`quant_resnet50_xint8_c64.log`](../results/res/quant_resnet50_xint8_c64.log)
on the same 64 images): empty position delta, 108/108 int8 initializers exact.
Ignition's `--cle` output therefore reproduces, to the integer, the plain-XINT8 ResNet50
that every earlier ResNet figure in this document was measured on. The files differ in
producer metadata and initializer order, not in any parameter.

**Full evaluation, same sitting** (`scripts/quant-validate.sh`, Desktop 2, Ryzen 7 8700G /
Phoenix XDNA1, Ryzen AI 1.7.1, `resnet_env17`, 1,000 labeled images, `sess.run` only,
static batch 1, fresh compile; both pre-NPU witnesses idle:
[reference](../results/quant/contexts_resnet50_ignition_cle_c64_reference.log),
[own](../results/quant/contexts_resnet50_ignition_cle_c64_own.log)):

| Artifact / device | Top-1 / top-5 % | Mean / median / p95 ms | Placement | Evidence |
|---|---:|---:|---|---|
| Quark default (CLE) / CPU | 72.80 / 88.40 | 31.40 / 31.80 / 38.26 | CPU | [run](../results/quant/run_resnet50_ignition_cle_c64_reference_cpu.log) |
| Ignition `--cle` / CPU | 72.80 / 88.40 | 30.62 / 30.68 / 39.86 | CPU | [run](../results/quant/run_resnet50_ignition_cle_c64_own_cpu.log) |
| Quark default (CLE) / NPU | 72.10 / 88.10 | 5.22 / 5.17 / 5.54 | 393/395 | [run](../results/quant/run_resnet50_ignition_cle_c64_reference_npu.log), [diag](../results/quant/diag_resnet50_ignition_cle_c64_reference.log) |
| Ignition `--cle` / NPU | 72.10 / 88.10 | 5.22 / 5.17 / 5.48 | 393/395 | [run](../results/quant/run_resnet50_ignition_cle_c64_own_npu.log), [diag](../results/quant/diag_resnet50_ignition_cle_c64_own.log) |
| Ignition alpha no-CLE / CPU and NPU | 62.00 / 79.80 and 59.90 / 79.10 | 34.02 and 5.27 | 393/395 | [Alpha validation](#ignition-alpha-release-validation) |
| Repo headline, CLE + AdaRound / NPU | 79.80 | 5.27 | 393/395 | README pipeline table |

CLE alone is worth 10.8 points of top-1 on CPU and 12.2 on the NPU over the no-CLE
alpha; the remaining 7.7 points to the headline are AdaRound, which Ignition does not
have. The NPU accuracy equals the original artifact's 2026-09-05 measurement
(72.10 / 88.10, [run](../results/res/run_resnet50_npu.log)); its latency then (5.68 ms)
and now (5.22 ms) are different days on the shared machine and are not compared. The
0.7-point CPU-to-NPU drop is the NPU-versus-CPU floor the acceptance study measured.

What this does not show: CLE on grouped or depthwise convolutions, on Gemm pairs, or on
graphs whose matcher walk crosses Pad or ReduceMean; each raises until it has a gate.

### Ignition: calibration spool without the pruned pre-Relu tensors

Both producers' calibration had spooled all 123 activations of the folded ResNet,
including the 49 Conv/Add outputs that feed only a Relu. Those tensors get a
temporary Q/DQ pair that emission removes (Quark's `get_qdq_to_remove`, Ignition's
`prune_conv_relu`), so their MinMSE positions never reach the file. `quant/calib.py`
now skips them: 74 tensors are spooled and searched, the sample store drops from
3,404,592,128 to 2,174,678,016 bytes (36 percent) for 64 images, and the CLE
calibration's wall time from 104.7 s to 72.0 s on Desktop 2 (no-CLE: 102.6 s to 71.1 s). The gate is byte
identity of the output file, not a graph diff: the trimmed
[CLE run](../results/quant/quant_resnet50_ignition_cle_c64_lean.log) writes SHA256
`74f2b9a180e05b22a203aeb896f2f31daf20a58beb759dc81f6aee52021e8bbe`, the same file as
the [full-spool CLE run](../results/quant/quant_resnet50_ignition_cle_c64.log) above,
and the trimmed [no-CLE run](../results/quant/quant_resnet50_ignition_nocle_c64_lean.log)
writes `afe15baa15a250ffdb0b68c5ce1f97efc1720d4c53140be42164321e7fb0686d`, the
released alpha artifact. Quark's All-mode calibrator still spools every tensor, so
`scripts/quant-reference.sh` keeps sizing its disk guard from the full list. The
sidecar records `spooled_tensors` and `skipped_prunable_tensors`; the skipped set is
exactly the set `emit` prunes, by construction from the same `prunable_tensors`.

### Ignition: permissive inspection

`python -m quant inspect` and `tools/quant_inspect.py` used the same loader as
`quantize`, so any file outside the export contract (IR 8, opset 17, static batch 1)
could not be fingerprinted at all; the batch-2 ResNet that measured the EP's stale
slot-1 buffer was one such file. Inspection now loads with `strict=False`: the
contract check and the ONNX checker run, their failures are reported per file as
`export_contract` and `onnx_checker` instead of raised, and the fingerprint follows.
[Inspection of the batch-2 XINT8 and float exports and the A8W8 model](../results/quant/inspect_resnet50_b2_permissive_resnet_env17.log)
reports the batch violation for the first two and `ok` for the third, with all three
fingerprints. `quantize` keeps the strict loader:
[the alpha boundary checks](../results/quant/check_resnet50_ignition_nocle_c64_lean_resnet_env17.log),
re-run against the fresh artifact, still reject wrong opset, batch 2, a symbolic batch,
an unsupported operator and a non-7×7 GAP, and the CLI still refuses a missing CLE
choice, a non-positive limit and an existing output. A direct `quantize` on the batch-2
float export exits with the batch error and writes nothing.

### Ignition: AdaRound parity

Quark's `XINT8_ADAROUND` preset is the default XINT8 pipeline followed by one
post-process, `fast_finetune` (`quark/onnx/algorithm/finetuning/`), which rewrites the
integer weights of every Conv/Gemm in the finished, refined QDQ file and touches nothing
else. `quant/adaround.py` transcribes that path and `python -m quant adaround` runs it on
an emitted Ignition file, importing torch inside the finetune only; `quantize` and
`inspect` still block torch, and Quark stays blocked throughout. The transcription:
layers in the quantized file's node order, one at a time, each seeing the rounding
already chosen upstream; the layer's pre-QuantizeLinear input from the current quantized
graph (ONNX Runtime CPU, `ORT_DISABLE_ALL`) and its float input and output from the
equalized float graph (ONNX Runtime CPU, default optimization) for every calibration
image; a torch module with the file's UINT8/zp128 input scale, the INT8 weight scale
behind the AdaRound rounding variable (floor plus a rectified sigmoid, gamma −0.1, zeta
1.1), the INT8 bias scale and the Relu; Adam at 0.1 on the rounding variable for 1,000
iterations of two-image batches drawn with `torch.randperm`, a squared-Frobenius
reconstruction loss plus the annealed rounding regulariser after a 20 percent warm
start, early stop on the windowed rounding loss; hard rounding clamped to [−128, 127]
written back to `<w>_quantized`. One construction detail decides whether the two
implementations can agree bitwise at all: Quark's module wrapper runs the torch layer's
initialiser twice, so its global generator advances twice per layer before the first
batch draw ([RNG probe](../results/quant/adaround_rng_probe_resnet_env.log): Conv with
and without bias, 1×1 Conv and Gemm all read `double`); Ignition constructs the layer
and calls `reset_parameters()` once more.

Fresh same-listing oracle: [`XINT8_ADAROUND` on the c64 listing](../results/quant/quant_resnet50_quark_cle_adaround_c64.log)
(`scripts/quant-reference.sh --cle --adaround`), 54 modules, no early stop, 88.7 s of
ONNX inference plus 445.1 s of torch training, 629.4 s end to end, peak working set
3,582,218,240 bytes, SHA256 `a9250fae…62a3d`. Ignition on its CLE artifact
([log](../results/quant/quant_resnet50_ignition_cle_adaround_c64.log),
`scripts/quant-adaround.sh`): 82.7 s of data plus 444.1 s of training, 528.2 s for the
finetune alone, peak working set 3,055,075,328 bytes, SHA256 `bf605321…fc2bc`. Both
ran in `resnet_env` (torch 2.4.1+cpu, 8 threads, ONNX Runtime 1.22.1) on purpose: the
oracle's data sessions run under that runtime, so the graph diff compares the
algorithm, not the runtime. Result
([diff](../results/quant/diff_resnet50_ignition_cle_adaround_c64.log)): empty position
delta, provenance equal (listing, preprocessing, float hash, CLE and every FastFinetune
parameter), **108/108 int8 initializers byte-identical**, refinement fixed point on
both. The two logs agree line for line: all 702 per-layer lines (the module banner,
eleven loss lines and the reconstruction metric for each of 54 layers) are identical
to the last printed digit. AdaRound moved 8,954,279 of the 25,529,472 weight elements
by exactly one LSB relative to the CLE base (35.07 percent), the same count on both
sides, and 624 of them sit at −128, Quark's dtype clamp rather than the producer's
[−127, 127] clip. Full-set CPU top-1 is 79.40% / 93.00% top-5 for both files
([reference](../results/quant/run_resnet50_ignition_cle_adaround_c64_reference_cpu.log),
[own](../results/quant/run_resnet50_ignition_cle_adaround_c64_own_cpu.log)), as
identical bytes require. On the NPU, paired in one sitting with a clean
`xrt-smi` context witness before each model and `--fresh` compilation, both read
**79.50% top-1 / 93.30% top-5** at 5.21 ms (reference) and 5.23 ms (own) mean latency,
393 of 395 nodes placed
([reference](../results/quant/run_resnet50_ignition_cle_adaround_c64_reference_npu.log),
[own](../results/quant/run_resnet50_ignition_cle_adaround_c64_own_npu.log),
[EP reports](../results/quant/diag_resnet50_ignition_cle_adaround_c64_own.log)); the
0.02 ms latency gap is session noise, not a finding. The NPU sits 0.1 point above CPU
here, the opposite sign from the no-CLE and CLE floors above, so the DPU-vs-QDQ
difference is a drift of either sign, not a fixed penalty.

Three controls bound what "bitwise" means here. A second Quark oracle on the same
machine minutes later ([rerun log](../results/quant/quant_resnet50_quark_cle_adaround_c64_rerun.log),
633.4 s, peak working set 3,581,181,952 bytes) hashes to the same SHA256 as the first,
so Quark reproduces itself here. A second Ignition run
([rerun log](../results/quant/quant_resnet50_ignition_cle_adaround_c64_rerun.log), 523.2 s,
peak working set 3,053,993,984 bytes) hashes to the same SHA256 as the first, so
Ignition reproduces itself too, and its sidecar carries the corrected version
provenance described below. The
[Sep 5 `XINT8_ADAROUND` artifact](../results/adaround/quant_resnet50_c64.log), built
from the same listing and seed, differs from the fresh oracle in 4,149,415 weight
elements (max 1 LSB; topology and scales identical); its log records no machine and its
layer-0 losses differ from the fresh run's in the sixth decimal (0.347939 against
0.347940 at iteration 100, during the warm start when only the reconstruction loss
exists), so that difference began in the float arithmetic (machine or thread count,
neither recorded), not in a rounding decision. Parity is therefore stated as: same
machine, same torch and ONNX Runtime
builds, same thread count; across machines the comparison is statistical. Two
provenance notes: this run's sidecar inherited the base's numpy/onnx versions (onnx
1.18.0 from `resnet_env17`) although the finetune process ran onnx 1.19.0, fixed for
later runs (`base_versions` now keeps the base's); and the finetuned file carries the
base's `positions` table unchanged, which is correct because AdaRound never moves a
scale. Against the alpha's no-CLE 62.00% CPU top-1, the c64 Ignition artifact with CLE
and AdaRound reads 79.40%; the repo's 79.80% headline is a separate run whose
calibration count no log here records, and it is not compared.

### Ignition: YOLOv8n-cut preparation parity

The head-cut YOLOv8n export is the first graph outside folded ResNet, and it needs
everything the ResNet path never exercised: `quant/passes.py` rewrites each `Split`
into one `Slice` per output plus four unnamed INT64 `Constant` nodes for
starts/ends/axes/steps and drops the split initializer; after emission and before
refinement it replaces each `Sigmoid` with `HardSigmoid(alpha=1/6)` under the same node
name and inserts a `Mul` by `HARD_SIGMOID_SCALE = (2731/16384)/(1/6)` with its
`<out>_Scale` constant, the vendor's DPU simulation. `quant/qdq.py` marks Conv input,
output, weights and bias; `MaxPool` and `Resize` outputs share their input's scale and
zero-point initializers, resolved through the SPPF pool chain to the root provider; and
Sigmoid, Mul, Add, Concat and Slice quantize every float activation input and output.
`quant/refine.py` is now a full transcription of Quark's `QuantPosManager`: concat,
pool, pad and slice alignment, then the read, write (a `Mul` write does not raise the
change flag), cut, bias, HardSigmoid and swish shifts, in the vendor's order, at most
five loops; it walks Ignition's topological order rather than the vendor's file order.
`quant/sources.py::CocoSource` replays `3b_quantize_cut.py`'s reader (sorted jpgs,
`npu.yolo.letterbox` at the graph's input size) and the family is read from the
operators, never a flag. Three gates, in order.

**Preparation parity first.** `tools/quant_prepare_probe.py` (wrapper
`scripts/quant-prepare-probe.sh`, `resnet_env`, no calibration, no hardware) runs
Quark's `apply_pre_process` with the resolved static op types, the hardware-compatibility
conversions its `quantize()` forces on and its own CLE, against Ignition's load, CLE and
prepare in the order `quantize()` applies them
([log](../results/quant/prepare_probe_yolov8n_cut.log)). The graph diff is empty: the
209-node float export becomes 281 nodes on both sides (8 `Split` to 16 `Slice` plus 64
`Constant`, the four `onnx::Split_*` initializers removed, 131 to 127), CLE finds zero
patterns on both (the SiLU net has no Conv-to-Conv pair), node names are equal as sets
and differ only in order, so the comparison is structural. Each of the vendor's other
preparation steps applied alone to the export, onnxslim, ORT basic optimization as
Quark runs it and BatchNorm folding, leaves 209 nodes and 131 initializers with no byte
changed (onnxslim adds `value_info` only, 218 to 428). The probe then replays the
repo's committed `yolov8n_cut_xint8.onnx` (SHA256 `f02e86ba…`) from its own positions
through the new emitter, HardSigmoid bridge and refinement: 344 positions, 218
activations, 126/126 int8 initializers byte-identical, empty graph diff, zero
refinement moves on either side, 3.8 s. The same probe on the ResNet export
([control](../results/quant/prepare_probe_resnet50_fp32.log)) finds all four steps
no-ops on 122 nodes / 108 initializers, 33 CLE patterns on both sides, an empty whole
pre-process diff, and `resnet50_xint8_c64.onnx` replaying exactly (182 positions,
108/108 int8, zero moves), so generalizing the marking and refinement did not move the
ResNet path.

**Same-listing calibration parity.** A fresh default-preset Quark oracle on the sorted
first 64 COCO calibration images
([log](../results/quant/quant_yolov8n_cut_quark_cle_c64.log),
`scripts/quant-reference.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --cle`):
0 CLE patterns, 20 refinement moves, peak working set 3,725,996,032 bytes, SHA256
`763e61cb…6ec46`. Ignition on the same listing
([log](../results/quant/quant_yolov8n_cut_ignition_cle_c64.log), `scripts/quant-own.sh --cle`,
Quark and torch imports blocked): 218 tensors spooled, 7,106,560,000 bytes, 57
Sigmoid replaced and scaled, refinement converged in two loops with 16 Concat
alignments and 4 Slice alignments. The two calibrations ran at the same time on the
same eight cores, so their wall times (232.2 s and 244.5 s) are not a timing comparison
and are not compared. Result ([diff](../results/quant/diff_yolov8n_cut_ignition_cle_c64.log)):
empty position delta; listing, preprocessing (letterbox 640, input `images`), float
hash and CLE provenance equal; **126/126 int8 initializers byte-identical**, maximum 0
LSB; refinement fixed point on both. The files themselves hash differently
(`b4b7e0fa…` against `763e61cb…`): the gate is the graph and every parameter, not the
serialization. The vendor's 20 logged moves, node, direction, old and new position,
match Ignition's sidecar entry for entry in the same order, a check made in-session on
the two logs (the vendor's line names the node but not the tensor, so the comparison is
node, direction, old and new). The HardSigmoid and swish shift bounds, the read, write,
cut and bias shifts and pool and pad alignment were transcribed but never fired on this
calibration, and the graph has no large-kernel pooling or Conv-to-Relu pruning instance,
so those handlers remain transcriptions without a measured case.

**Full-set evaluation.** `scripts/quant-validate.sh --family yolo` runs
`pipelines/yolov8n/5_eval_map.py` on all 5,000 val2017 images at conf 0.001, IoU 0.7,
max_det 300 with per-class NMS, the numpy decode outside "infer". CPU: both files read
**27.43 mAP@50-95 / 40.87 mAP@50** (small/medium/large 13.96 / 31.04 / 37.63;
[reference](../results/quant/map_yolov8n_cut_ignition_cle_c64_reference_cpu.log),
[own](../results/quant/map_yolov8n_cut_ignition_cle_c64_own_cpu.log)), as identical
parameters require; the detection files (66,904,327 bytes, git-ignored) compared
byte-identical in-session. NPU, paired in one sitting with a clean `xrt-smi` context
witness before each model and `--fresh` compilation
([reference](../results/quant/map_yolov8n_cut_ignition_cle_c64_reference_npu.log),
[own](../results/quant/map_yolov8n_cut_ignition_cle_c64_own_npu.log),
[EP reports](../results/quant/diag_yolov8n_cut_ignition_cle_c64_own.log)): both read
**27.03 mAP@50-95 / 40.19 mAP@50** (13.21 / 30.68 / 37.02), 922 of 929 nodes on the
NPU with the input `QuantizeLinear` and the six head `DequantizeLinear` on CPU,
identical EP reports, and byte-identical NPU detection files (65,873,014 bytes), so the
compiler built the same executable from both. Mean `sess.run` at eval conf was 7.30 ms
(reference) and 7.28 ms (own), not comparable to demo latency, and the 0.02 ms gap is
session noise. The 0.40-point CPU-to-NPU drop is the DPU-versus-QDQ drift seen on
ResNet, downward here. The repo's c200 head-cut artifact reads 26.94 / 40.15 on the
NPU (the yolov8n XINT8 head-cut row above); the c64 pair is not compared to it because
the calibration count differs.

What remains open on YOLO: AdaRound (`python -m quant adaround` raises for a non-ResNet
base), a calibration that fires the shift rules, and traversal-order equivalence where
refinement rules interact, which this graph does not test.

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

**Historical rule, now narrowed: "The X1 backend is XINT8 or nothing."** The
[Ignition probes](#ignition-controlled-resnet-qdq-acceptance) supersede the float-scale-only
attribution: domain-only fallback is measured, signed activations work, and some scale
changes compile but execute incorrectly. Keep XINT8 as the default: power-of-two scales, MinMSE calibration,
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

### Quantization CPU threading: SMT contention and barrier thrashing during FastFinetune

During AdaRound FastFinetune on the 8-core / 16-thread Ryzen 7 8700G, Task Manager shows
unusual CPU behavior: low sustained aggregate utilization with cores appearing to "take turns"
rather than running saturated.

To isolate the cause, `tools/bench_quant_threads.py` microbenchmarks AdaRound FastFinetune
(`models/mobilenetv2_fp32.onnx`, 52 Conv layers, 32 calibration images, 100 iterations/layer,
`batch_size=2`, logged in `results/bench_quant_threads.log`) across four distinct configurations:
1. **1 pinned core** (affinity mask `0x1`, `--threads 1`, `OMP_NUM_THREADS=1`)
2. **4 real physical cores** (affinity mask `0x55` [cores 0, 2, 4, 6], `--threads 4`, `OMP_NUM_THREADS=4`)
3. **8 real physical cores / no SMT** (affinity mask `0x5555` [cores 0, 2, 4, 6, 8, 10, 12, 14], `--threads 8`, `OMP_NUM_THREADS=8`)
4. **16 logical threads** (unconstrained mask `0xFFFF`, `--threads 16`, `OMP_NUM_THREADS=16`, default behavior)

| Configuration | Mask | Threads | FastFinetune (s) | Torch training (s) | ONNX eval (s) | Wall clock (s) |
|---|---|---|---|---|---|---|
| **1 pinned core** | `0x1` | 1 | 216.9 s | 43.6 s | 171.2 s | 257.6 s |
| **4 real cores** | `0x55` | 4 | 81.5 s | 28.4 s | 51.3 s | 110.5 s |
| **8 real cores (no SMT)** | `0x5555` | 8 | **49.3 s** | **27.0 s** | 20.4 s | **76.7 s** |
| **16 logical threads** | `0xFFFF` | 16 | 54.4 s | 32.8 s | **19.8 s** | 81.6 s |

**Findings & Root Cause:**
- **8 real cores with no SMT threads is the fastest overall across the entire run (49.3 s FastFinetune, 76.7 s wall clock).**
  It beats 16 logical threads on wall clock (76.7 s vs 81.6 s, +6.0% faster) and beats 4 real cores (76.7 s vs 110.5 s, +30.6% faster).
  Running exactly one thread per physical Zen 4 core provides maximum dedicated L1/L2 cache capacity and execution units without SMT sibling pipeline resource sharing.
- **In the per-layer optimization loop (`Torch training`), 8 real cores (27.0 s) and 4 real cores (28.4 s) both beat 16 threads (32.8 s).**
  FastFinetune operates layer-by-layer with `batch_size=2`. The tensors being optimized are small enough
  that individual layer forward/backward steps execute in milliseconds. Spreading these tiny workloads across
  16 threads creates severe OpenMP barrier overhead (`#pragma omp barrier`), lock contention, and SMT
  pipeline sharing between logical sibling threads on the same physical core. Threads spin-wait and bounce
  across cores, causing the "taking turns" effect seen in Task Manager. 8 physical cores cuts training time from 32.8 s to 27.0 s (**+17.7% faster**).
- **In full-graph calibration inference (`ONNX eval`), 8 physical cores (20.4 s) matches 16 threads (19.8 s) to within 3%.**
  Here ONNX Runtime evaluates the entire model graph where large matrix multiplications saturate execution units, and 8 dedicated physical cores achieve virtually identical throughput to 16 SMT threads while avoiding thread contention.
- **1 pinned core suffers compute starvation (257.6 s wall clock, 3.4× slower than 8 cores).**
  Pinning strictly to a single core eliminates OpenMP synchronization overhead, but severely bottlenecks the BLAS/GEMM routines.
- **Repository configuration:** All 5 quantization pipelines (`pipelines/resnet50/3_quantize.py`, `pipelines/yolov8n/3b_quantize_cut.py`, `pipelines/yolov8n/3_quantize.py`, `pipelines/yolov8n-pose/3b_quantize_cut.py`, `pipelines/mobilevit/2_quantize.py`) now support `--threads` and default to `OMP_NUM_THREADS=4` (or 8 on 8-core CPUs) with `OMP_WAIT_POLICY=PASSIVE` in `scripts/lib.sh`.

---

### Category B: Real-Time Portrait Matting (MODNet on XDNA1 NPU)

MODNet evaluates real-time portrait matting at 512x512 with an objective-oriented architecture: MobileNetV2 backbone, Low-Resolution (LR) semantic branch, High-Resolution (HR) boundary detail branch, and Fusion branch.

#### 1. The Wholesale Rejection & Root-Cause Bisection

The stock MODNet ONNX graph was rejected outright by the VitisAI EP (`vitisai_ep_report.json` showed **0 NPU nodes, 236 CPU, 636 VITIS_EP_CPU**). The reported 132 ms was CPU INT8 execution inside VitisAI EP, not hardware NPU execution.

To diagnose the failure, individual subgraphs were exported, quantized with Quark `XINT8`, and compiled through the VitisAI EP against physical hardware (`tools/diag_ep.py`):

| Subgraph Tested | Total Nodes | NPU Nodes | Non-NPU Nodes | NPU % | Compilation Verdict |
|---|---|---|---|---|---|
| **MobileNetV2 Backbone** | 337 | **335** | 2 | **99.4%** | Accepted (Q/DQ boundary only) |
| **SEBlock (1x1 Conv refactor)** | 357 | **355** | 2 | **99.4%** | Accepted |
| **Resize (`F.interpolate` bilinear)** | 340 | **338** | 2 | **99.4%** | Accepted |
| **Standard BatchNorm + Conv** | 343 | **341** | 2 | **99.4%** | Accepted |
| **`IBNorm` (Slice -> BN + IN -> Concat)** | 365 | **0** | 365 | **0.0%** | **Wholesale Rejection** |

The bisection isolated the failure directly to `IBNorm`. Each `IBNorm` splits channels in half, executing `BatchNorm2d` on one half and `InstanceNorm2d` on the other half before concatenating them. Because `InstanceNorm` is unsupported on the NPU and runs on the host CPU, the graph split requires synchronizing across CPU and NPU within every single normalization layer. The VitisAI compiler cannot partition this intra-layer diamond across heterogeneous devices and refuses the entire model.

#### 2. The NPU Architecture Refactor (`modnet_cut`)

To enable native NPU execution, three architectural refactors were applied:
1. **Calibrated Unified Normalization**: Evaluated empirical running mean and variance for the 17 `InstanceNorm` layers across 100 portrait calibration images. The running statistics match the uncalibrated reference model to **MAD = 0.0384** (under 3.9% deviation).
2. **Conv + BN Parameter Folding**: The dual-branch normalization was merged into a unified `BatchNorm2d` and mathematically folded into the preceding `Conv2d` weight and bias (W_fused = W * gamma / sqrt(var + eps)). This eliminated all 17 `Slice`, 17 `InstanceNorm`, and 17 `Concat` layers.
3. **SEBlock 1x1 Conv**: Replaced `nn.Linear` with 1x1 `nn.Conv2d`, eliminating `MatMul`, `Reshape`, and `Expand`.
4. **Tail Cut**: The final `Sigmoid` was removed from the ONNX graph so the NPU outputs raw logits; sigmoid is evaluated in numpy postprocessing (<0.2 ms).

#### 3. Hardware Execution & Accuracy Metrics

Tested across 50 full validation portrait images (`data/modnet_val/`):

| Model & Runtime | Device | Latency | FPS | NPU Node Placement | Accuracy vs FP32 Ref |
|---|---|---|---|---|---|
| **MODNet Cut XINT8** | **Ryzen AI NPU** (Phoenix 4x4) | **28.45 ms** | **35.1 fps** | **502 / 507 (99.0%)** | MAD: 0.1902, SAD: 50.75k, MSE: 0.1818 |
| MODNet FP32 | Radeon 780M iGPU (DirectML) | 39.05 ms | 23.2 fps | — | Reference baseline |
| MODNet FP32 | Ryzen 7 8700G CPU (8 Zen 4 cores) | 256.89 ms | 3.8 fps | — | Reference baseline |
| MODNet Stock XINT8 | CPU fallback (VitisAI EP) | 132.07 ms | 7.3 fps | 0 / 872 (0.0%) | Rejected graph |

- **NPU Node Placement**: **502 of 507 nodes (99.0%)** compiled on the physical NPU (`modnetcutcachekey/vitisai_ep_report.json`). The only 5 non-NPU nodes are input/output boundary Q/DQ conversions and a single initial 4x image downsampling `Resize`.
- **Speedup**: **9.03x over 8-core Zen 4 CPU**, and **1.37x faster than the 12 CU Radeon 780M iGPU**.
- **End-to-End Frame Pipeline**: 1.90 ms preprocess + 28.13 ms NPU infer + 2.33 ms postprocess = **32.36 ms total frame time (~30.9 real-time FPS)** with live bokeh blur.

**The four latency rows above shipped with no backing log** — they were written from a
session whose output was never captured under `results/`, against this repo's rule that
every figure trace to a log. They are kept here rather than deleted, and re-measured
below; read the re-measurement as the citable set.

#### 4. Same-sitting re-measurement, with logs (2026-09-07, Desktop 2)

All four configurations captured back to back in one session, because NPU and DML latency
on this machine drift between sessions independently of any code change. Same 50
validation images, same harness (`pipelines/modnet/5_eval.py`), `--fresh` on both NPU runs.

| Model & Runtime | Device | Latency | FPS | NPU Node Placement | MAD vs FP32 ref | Log |
|---|---|---|---|---|---|---|
| MODNet Zero-Concat XINT8 | Ryzen AI NPU (Phoenix 4x4) | **17.75 ms** | 56.3 fps | **533 / 538 (99.1%)** | **0.35269** | `results/modnet/eval_modnet_zero_concat_xint8_npu.log` |
| MODNet Cut XINT8 | Ryzen AI NPU (Phoenix 4x4) | 26.44 ms | 37.8 fps | 502 / 507 (99.0%) | 0.19022 | `results/modnet/eval_modnet_cut_xint8_npu.log` |
| MODNet FP32 | Radeon 780M iGPU (DirectML) | 46.14 ms | 21.7 fps | — | 0.00000 | `results/modnet/lat_modnet_fp32_dml.log` |
| MODNet FP32 | Ryzen 7 8700G CPU (Zen 4) | 209.60 ms | 4.8 fps | — | 0.00000 | `results/modnet/lat_modnet_fp32_cpu.log` |

- **The accuracy metrics reproduce exactly.** Cut XINT8 re-measured MAD 0.19022 / SAD
  50.75k against the 0.1902 / 50.75k recorded above — the quality numbers in section 3
  were right, only their evidence was missing. The two FP32 rows score MAD 0.00000
  because reference and test model are the same file; that is the harness identity check,
  not a result.
- **The latencies do not reproduce, and the speedup multiples move with them.** Cut XINT8
  read 26.44 ms here against 28.45 ms above, CPU 209.60 against 256.89, DML 46.14 against
  39.05. Recomputed from this sitting, Cut XINT8 is **7.93x the CPU** (not 9.03x) and
  **1.75x the iGPU** (not 1.37x) — the iGPU margin is the one that moved most, and it
  moved in the NPU's favour. Neither set is wrong; they are different sessions, which is
  exactly why the two are kept separate rather than merged.
- **Zero-Concat is the faster graph and the worse matte.** It places 31 more nodes on the
  NPU (533/538) and runs 1.49x faster than Cut, but its MAD against the FP32 reference is
  **0.35269 — 1.85x Cut's 0.19022**. Buying 8.7 ms costs nearly double the alpha error.
  `demos/portrait_matting_demo.py` defaults to this variant, so what the demo shows on
  screen is the fast-and-loose end of that trade, not the accurate one.
- **Calibration and inference preprocessing were not byte-identical** for any MODNet model
  measured so far. `pipelines/modnet/3_quantize.py` resized calibration images through
  `PIL.Image.BILINEAR` while every inference path used `cv2.INTER_LINEAR`; Pillow
  antialiases on downscale and OpenCV does not, so the two disagree pixel-for-pixel. Both
  now share `npu/modnet.py`, but **every MODNet number on this page was measured with
  models calibrated through the old PIL path** — re-quantizing to close the gap is untried,
  and the MAD figures above are the ones to beat when someone does.

#### 5. OpenCV calibration fix: error reduction and the structural Zero-Concat gap (2026-09-08, Desktop 2)

Section 4 called out that calibration and inference preprocessing were not byte-identical:
`pipelines/modnet/3_quantize.py` calibrated through `PIL.Image.BILINEAR` (antialiasing on
downscale) while inference used `cv2.INTER_LINEAR`. Both models were re-quantized with unified
OpenCV preprocessing and evaluated back to back on Desktop 2 under the same 50 validation images:

| Model & Runtime | Device | Latency | FPS | NPU Node Placement | MAD vs FP32 ref | SAD (1e3) | Log |
|---|---|---|---|---|---|---|---|
| MODNet Cut XINT8 (calibfix) | Ryzen AI NPU (Phoenix 4x4) | **27.51 ms** | 36.3 fps | 502 / 507 (99.0%) | **0.18629** | **49.11** | `results/modnet/eval_modnet_cut_xint8_calibfix_npu.log` |
| MODNet Cut XINT8 (PIL, superseded) | Ryzen AI NPU (Phoenix 4x4) | 26.44 ms | 37.8 fps | 502 / 507 (99.0%) | 0.19022 | 50.75 | `results/modnet/eval_modnet_cut_xint8_npu.log` |
| MODNet Zero-Concat XINT8 (calibfix) | Ryzen AI NPU (Phoenix 4x4) | **18.46 ms** | 54.2 fps | 533 / 538 (99.1%) | **0.33187** | **90.03** | `results/modnet/eval_modnet_zero_concat_xint8_calibfix_npu.log` |
| MODNet Zero-Concat XINT8 (PIL, superseded) | Ryzen AI NPU (Phoenix 4x4) | 17.75 ms | 56.3 fps | 533 / 538 (99.1%) | 0.35269 | 93.18 | `results/modnet/eval_modnet_zero_concat_xint8_npu.log` |

- **Error dropped across both variants.** Cut MAD fell from 0.19022 to 0.18629 (-2.1%) and SAD
  from 50.75k to 49.11k; Zero-Concat MAD fell from 0.35269 to 0.33187 (-5.9%) and SAD from 93.18k
  to 90.03k. Eliminating downsampling antialiasing differences during calibration directly improves
  quantized alpha reproduction.
- **The Zero-Concat quality penalty is structural, not calibration drift.** Zero-Concat's error
  remains 1.78x higher than Cut's under identical calibration (0.33187 vs 0.18629). Replacing skip-
  connections with zero-padded channels in the fusion stage permanently discards boundary spatial
  detail; the ~8.7 ms speedup continues to trade half the alpha quality.

---

### Alternative classification topologies: DenseNet-121 (concat) and ResNeXt-50 (grouped convs)

Investigating compiler placement and post-training quantization fidelity across non-standard
convolutional topologies: dense channel concatenation (DenseNet-121) and grouped convolutions
(ResNeXt-50 32x4d). 1000 ImageNet-1k validation images, static shape `(1, 3, 224, 224)` on Desktop 2:

| Model | Architecture Feature | NPU Placement | Latency (NPU) | Latency (Zen 4 FP32) | Top-1 (FP32) | Top-1 (Plain XINT8) | Log |
|---|---|---|---|---|---|---|---|
| DenseNet-121 | 58 `Concat`, 3 `AveragePool` | **1703 / 1705 (99.9%)** | **8.06 ms** | 21.70 ms (2.69x) | 78.00% | 0.10% | `results/res/run_densenet121_xint8_npu.log` |
| ResNeXt-50 32x4d | 53 grouped convs (`groups=32`) | **393 / 395 (99.5%)** | **9.37 ms** | 17.40 ms (1.86x) | 81.00% | 0.10% | `results/res/run_resnext50_32x4d_xint8_npu.log` |

- **Hardware offload is complete**: Both models achieve >99.5% NPU placement with zero op-level
  refusal. All 58 Concat nodes in DenseNet-121 execute on AIE tiles without DMA bottlenecks, and
  all 53 grouped convs in ResNeXt-50 compile natively without scalar fallback.
- **Plain per-tensor XINT8 collapses completely**: Both drop to 0.10% top-1 (random chance on 1000
  classes).

#### RegNetX-002: regular channels, shift-cut scale explosion, and AdaRound limits (2026-09-08, Desktop 2)

To isolate whether Category E's collapse was driven by DenseNet's accumulating channel concatenation
or ResNeXt's narrow 4-channel groups, `regnetx_002` (2.68M parameters, regular linear channel
capacity, no concat, standard grouped convs) was exported and evaluated on Desktop 2 across 200
ImageNet-1k validation images:

| Model Variant | Execution Target | Top-1 | Top-5 | Latency | Placement | Log |
|---|---|---|---|---|---|---|
| RegNetX-002 FP32 | CPU (Zen 4) | **68.50%** | **90.50%** | 1.89 ms | Reference baseline | `results/regnet/eval_regnetx_002_fp32_cpu.log` |
| RegNetX-002 Plain XINT8 | **Ryzen AI NPU** | **0.50%** | **1.00%** | **2.45 ms** (407.9 FPS) | **324 / 326 (99.4%)** | `results/regnet/eval_regnetx_002_xint8_npu.log` |
| RegNetX-002 AdaRound | CPU (ORT) | 0.50% | 1.00% | 4.20 ms | — | `results/regnet/eval_regnetx_002_xint8_adaround_cpu.log` |

- **Placement and speed excel**: 324 of 326 nodes (99.4%) compile onto the Phoenix NPU (`results/regnet/diag_regnetx_002_xint8.log`),
  with only input QuantizeLinear and output DequantizeLinear boundary nodes on CPU. Inference runs
  at 2.45 ms (407.9 FPS), delivering a 1.39x speedup over Zen 4 CPU INT8 (4.20 ms).
- **Total accuracy collapse persists**: Both plain XINT8 and AdaRound (300 iters/layer, 45 layers,
  77s FastFinetune on 8 pinned CPU cores) score 0.50% top-1 (pure random guessing).
- **Scale explosion root cause (`tools/audit_quant_grid.py`)**:
  Static inspection (`results/regnet/audit_regnetx_002_quant_grid.log`) revealed astronomical scale
  distortion under Quark's power-of-two quantizer:
  - Activation scales span 0.015625 to 1.329e+36 (an 8.5e35 range across 61 sites).
  - Depthwise scale grid reaches 3.245e+32 (2^108).
  - Other conv scales collapse to 9.40e-38 (2^-123).
- **The DPU shift-cut clamp mechanism**:
  During compilation Quark logs `Shift cut of layer onnx::Conv_418 exceeds range [0, 16] (131). Modify wpos from 7 to -108.`
  The Phoenix DPU accumulator shift register only allows shifts in `[0, 16]`. To avoid hardware
  overflow, Quark modifies weight positions by 100+ powers of 2. Since scale is 2^-pos,
  shifting `wpos` to -108 forces scale to 2^108, annihilating activation resolution.
- **Why AdaRound cannot rescue this**:
  As established in the MobileViT study, AdaRound optimizes ternary rounding {-1, 0, 1} over
  fixed quantization intervals Delta. It never changes Delta. When Delta has suffered
  floating-point scale explosion, integer rounding cannot recover the network.

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
  [Roadmap](../RESEARCH.md#roadmap).
- **No formal test suite.** Verification here is empirical (`compileall` + import checks
  as a syntax gate, then real pipeline runs read from `results/`) rather than unit tests
  — there's no fixture NPU to test against in CI.
- **`scripts/lib.sh` now exports `OMP_NUM_THREADS=8` (etc.) for every script that sources
  it, not just quantization.** Added 2026-09-07 to fix FastFinetune's thread-thrashing on
  tiny per-layer batches (see "Quantization CPU threading" above), but the export has no
  scope guard, so it also pins CPU-EP inference in `4_run.py`/eval scripts run through
  `scripts/*.sh`. Measured effect: yolov8m's FP32 CPU baseline dropped from 204.38 ms
  (`results/wide/yolo_m_fp32_cpu.log`, pre-change, unconstrained threads) to 144.12 ms
  (`results/bench/lat_yolov8m_cpu.log`, post-change, 8 physical cores) on the same
  8700G — an 8-core pin beating 16 unconstrained threads for CPU inference too, not only
  FastFinetune. Any CPU latency number measured before 2026-09-07 was not pinned; any
  measured after, through a `scripts/*.sh` wrapper, is. Don't diff a pre- and post-change
  CPU number as if the only variable were the model.

