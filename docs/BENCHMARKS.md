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

**It does not reproduce on Desktop 2, under a witness — and the re-run turned up a
6.2× timing gap that is now the more interesting question**
(`results/map_yolov8l_cut_xint8_npu_witnessed.log`,
`results/witness_yolov8l_cut_xint8_npu.jsonl`, 2026-09-09). The same model
(`yolov8l_cut_xint8.onnx`), the same 5000 images, with `tools/hwinfo_npu_bridge.exe`
sampling the device once a second for the whole run: **it completed**, and produced
**486694 detections and 45.37 mAP@50-95 — identical, to the detection, to the original
run.** So the numerics are the same and only the environment differs.

The witness rules out, *for this run on this machine*, four of the candidates:

| candidate | what the 347 samples show |
|---|---|
| a foreign hardware context | exactly one context throughout — pid 33088, 4 columns, `start_col` 1; never a second |
| a power-mode change or downclock | `power_mode` `Default` in all 347; clock 1800 MHz whenever submitting, 800 MHz only when idle |
| NPU memory growth | flat 231 MB while running (64 MB during session setup) — matching the original run's flat 231 MB |
| a progressive stall | submissions 16.71–19.36/s across 273 working samples, no trend; the only zero-throughput stretch is the 34 s tail, which is the CPU-side pycocotools accumulate |

**None of that explains the original failure, and it must not be read as doing so.**
Two things block that. First, the machine the failures happened on is **unestablished**:
the original log records a compile cache under `C:\Users\<user>\src\ryzen-ai-xdna1-quantization`,
a checkout that does not exist on Desktop 2 (which uses `PycharmProjects\`), and 22
tracked logs share that `src\` prefix. Desktop 1 has no XDNA1 device, so those NPU runs
were most likely the laptop — but commit `533b83a` predates this repo's name-the-machine
rule and does not say. A clean run on Phoenix/32 GB cannot clear a failure that may have
happened on Hawk Point/16 GB, and host RAM pressure there is untouched by "NPU memory
flat at 231 MB": NPU memory and host memory are different pools.

Second, and the reason this re-run is worth more than a null result: **the original took
1532 s at 287.94 ms mean inference; this one took 273 s at 46.33 ms — 5.6× the wall clock
and 6.2× the per-image figure, for byte-identical output.** Part of that is a timing
definition, and that part is now settled rather than guessed: at `533b83a` the timed call
was `forward = lambda x: decode_heads([sess.run(...)…])`, so the original's 287.94 ms
**included the numpy DFL decode**, where this run's 46.33 ms is `sess.run` alone. The
per-image figures are therefore not comparable as they stand.

**The wall clock is, and it does not go away.** 1532 s against 273 s is the same loop
doing the same work — and `npu/yolo_decode.py` and `npu/yolo.py` have had no commits
since `533b83a`, so the decode being blamed is byte-identical code on both sides. Two
readings survive, and this repo cannot yet choose between them:

- decode really did cost ~240 ms/image there against ~8 ms/image here, which would be a
  30× gap in identical numpy on two Zen 4-class CPUs — implausible on its face; or
- `sess.run` itself was ~270–280 ms there against 46.33 ms here, i.e. the **device** was
  genuinely ~6× slower, which is what a lower NPU power mode would look like (S0 measured
  1.80 GHz `default` against 1.03 `balanced` and 0.80 `powersaver` — a 2.25× span on its
  own) possibly compounded by the contention that this whole line of enquiry started from.

The second is the more likely, and it keeps a **command watchdog** in play as the
mechanism: a ~280 ms command sits far closer to any fixed timeout than a 46 ms one, and a
downclocked, contended device is exactly where a long command would get longer still.
Nothing here measures that, and it is not claimed. What settles it is one run nobody has
done: **the same model on the laptop, under the current infer/post split and a witness**,
which yields `sess.run` alone on the machine where the failures actually happened.

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

> **Superseded 2026-09-09 for batchable work: the threshold is ~36 µs, not 617 µs.** The
> "hypothetical zero-overhead resubmit path" was measured and is not hypothetical. Batched
> submission through `pyxrt.runlist` amortises a dispatch to **36.3 µs**, a **17×** drop, and
> that is below the 169.8 µs this section calls the hardware floor. See
> [Batched submission drops the dispatch floor 17×](#batched-submission-drops-the-dispatch-floor-17-and-reopens-four-closed-verdicts).
> The 617 µs figure still governs a **single** unbatched IRON dispatch, which is what this
> section measured, so it is superseded rather than retracted.
>
> **Scoped 2026-09-09 (same day): ~36 µs is a raw-pyxrt figure. Through IRON the batched
> floor is ~531 µs.** Batching was then wired into IRON's own host path and measured there:
> the device cost per dispatch does fall to 37.5–37.9 µs, but IRON's per-call host work is a
> near-constant ~500 µs that batching never touches, so a batched `@iron.jit` call still costs
> ~531 µs. See [Batching reaches the device floor from inside IRON](#batching-reaches-the-device-floor-from-inside-iron--and-irons-own-host-work-eats-almost-all-of-it).

### The open conv's 11.3× gap to the vendor is issue rate, and it is visible without a trace

§2.3 of `docs/SILICON.md` puts the vendor DPU at **1650 GOPS per column** on int8 conv against
the open `ml/bottleneck` kernel's **146.1** — 4.5% of what the same silicon does under AMD's
compiler. Objective K1 said "the first trace will say whether it is data movement or issue
rate", and no trace was ever run, because *"the conv kernels have no surviving build cache, so
this repo's most-lost op class was not surveyed"*. The cache was rebuilt and the question
answered from the instruction schedule alone. Backing log
`results/aie/conv_issue_rate_decomposed.log`.

Bundle count is cycle count on this core, so `vmac` per bundle **is** MACs per cycle:

| Loop | Bundles | `vmac` | Per cycle |
|---|---|---|---|
| int8 GEMM, hardware loop | 9 | 8 | **0.889** |
| conv2dk3 (3×3), main loop | 18 | 4 | 0.222 |
| conv2dk3, best loop | 3 | 1 | 0.333 |
| conv2dk1 (1×1), hot loop | 22 | 1 | **0.045** |
| conv2dk1, other two loops | 15, 15 | 0, 0 | 0.000 |

**A 4×–20× shortfall in MAC issue density against an 11.3× throughput gap.** The schedule
alone more than accounts for it; nothing about data movement needs invoking.

**The two kernels fail differently, and only one is subtle.** The 1×1 never keeps an
accumulator in a register — its loop loads all four quarters from memory (`vlda amhh1/amhl1/
amlh1/amll1`), issues **one** `vmac`, stores four quarters back, and idles **six of 22
bundles** on load-to-use latency, while naming 3 of the file's 9 accumulators. *(The cause was
found and fixed the same day — a runtime-indexed accumulator array — and the fix is worth
2.99×; see [the 1×1 conv's accumulators](#the-11-convs-accumulators-were-in-memory-putting-them-in-registers-is-worth-3-and-the-op-class-still-loses).)* The 3×3 *does*
keep `cm1`–`cm4` live with no accumulator traffic, and still reaches only 0.222, because six
of its eighteen bundles are `vshift` and four more `vmov`: sliding-window realignment spent in
issue slots. That is the classic conv-on-SIMD cost, and it is exactly what K1's tooling
section proposes removing by moving row shifting to the mem tile's 4-D descriptors.

One minor third finding, recorded so it is not mistaken for a lever: the 3×3's hot loop
carries one paired-load bundle and a bank holds two distinct buffers, so the same-bank penalty
applies — about 5%, noise beside a 4× issue shortfall.

**Note what this does and does not overturn.** `kernels/README.md` closed a "kernel quality"
item with *"both vectorize correctly with `aie::mmul`"*. That is a **correctness** statement
and it remains true; these are the throughput numbers, which were never taken. Reaching K1's
1 TOPS bar needs 6.8× and the 1×1's defect alone is worth up to 20× on that kernel — but the
20× and 4× are **ceilings on unused issue slots, not predictions of a rewrite**, and the
per-loop densities are unweighted by trip count, so which loop dominates runtime is still
unmeasured. What this removes is the excuse that nobody knew where the 11.3× went.

### The 1×1 conv's accumulators were in memory. Putting them in registers is worth 3×, and the op class still loses

The section above located the open int8 conv's 11.3× gap to the vendor DPU as issue rate, with
the 1×1's hot loop the worst of it at **0.045 MACs/cycle** against the GEMM's 0.889, and named
the cause: the kernel keeps its accumulator in *memory*. This is the fix, measured. Backing log
`results/aie/conv_accum_residency_npu.log`.

**The defect is one declaration.** `conv2dk1_i8_vector` holds `MMUL4x8x8 acc_tmp[4]` and indexes
it with `for (int x = 0; x < n; x++)` where `n` is a **runtime** value. Registers cannot be
dynamically addressed, so the array is forced to memory and every `.mac()` becomes
load-four-quarters / mac / store-four-quarters. Peeling the `n == 4` case into four **named**
accumulators — the pattern `mm.cc`'s `matmul_vectorized_2x2_mmul` already uses — fixes it. The
array loop is kept as the tail and is genuinely reached: `total_chunks = iw/4`, so `iw=56` gives
14 = 4+4+4+2, and the 56×56 shape runs the tail at n=2 and verifies.

| `conv2dk1_i8_vector` hot loop | Stock | Peeled |
|---|---|---|
| Bundles per iteration | 22 | 14 |
| `vmac` per iteration | 1 | 4 |
| **MAC issue rate** | **0.045/cyc** | **0.286/cyc** |
| Accumulator quarter loads / stores | 4 / 4 | 0 / 0 |
| Bundles issuing nothing | 6 | 3 |
| Accumulator registers named | 3 (`cm0`–`cm2`) | 5 (`cm0`–`cm4`) |

**This is the falsifiable prediction H11 set up, and it held.** H11 improved the int8 GEMM's
kernel by 12.5% — every spill gone, a full MAC every cycle — and the wall clock did not move,
because that design is bound by a per-buffer delivery floor. The conv sat at 4.5% of peak rather
than 41.9%, so it should be genuinely issue-bound and should actually speed up. It does. The two
designs are bound by different things, and this is the first result here that shows it by
*intervening* rather than by modelling.

| Marginal GOPS (fit slope, shape-free) | Series A | Series B |
|---|---|---|
| Stock | 117.1 | 115.6 |
| Both 1×1 stages peeled | **350.3** | **348.4** |
| Gain | 2.99× | 3.01× |

Both series agree in sign and magnitude, well outside this machine's ~5% drift, r² ≥ 0.9987, and
every shape in every arm passes the sweep's own correctness gate.

**A single-stage fix would have been reported as a null result, and that is the transferable
lesson.** The bottleneck is a three-stage core-to-core pipeline — `conv2dk1`, `conv2dk3`,
`conv2dk1_skip` — and the third stage carries the *same* defect. Patching only `conv2dk1.cc`, at
32×32:

| Arm | hw ms | GOPS |
|---|---|---|
| Stock | 1.4844 | 96.1 |
| `conv2dk1.cc` only | 1.4486 | 98.4 |
| Both 1×1 stages | **0.6133** | **232.5** |

2.4% alone, 2.42× together. A pipeline runs at the rate of its slowest stage, so a single-stage
intervention measures the pipeline's balance, not the intervention.

**And the op class stays closed.** The CPU baseline, named and measured in the same sitting:
onnxruntime 1.22.1, ORT CPU EP, QDQ int8 reaching VNNI, same six shapes, run twice — **823.7 and
839.5 marginal GOPS** (consistent with the 819.0 already published for this sweep).

| | Marginal GOPS | CPU wins by |
|---|---|---|
| Stock NPU | 115.6–117.1 | 7.1–7.2× |
| Peeled NPU | 348.4–350.3 | **2.4×** |
| CPU (VNNI int8) | 823.7–839.5 | — |

A 3× kernel improvement moves the deficit from 7.2× to 2.4× and does not close it. What changes
is the *reason* the op class is closed: it was "the kernel uses 5% of its issue slots"; it is now
"even with the slots used, one column of this array does not reach a VNNI-equipped Zen4." Against
the vendor DPU the per-column gap narrows from ~14× (this sitting's stock arm) to **~4.7×**.

**A discrepancy that has to be stated because it looks like a contradiction.** The published
stock figure is **146.1** GOPS; this sitting's stock arm measures **115.6–117.1** at the same
shapes. The kernel is not the same code — that sweep predates the 2026-09-07 width fix, which
rewrote this very loop to walk the width in ≤4-chunk blocks where upstream used eight concurrent
accumulators, and the published run could not compile 56×56 at all while this one can. *Inference,
not measurement:* the width fix appears to have cost ~20% at w=32 while making non-multiple-of-32
widths correct, and was never measured at the time. The A/B is like-for-like within one sitting,
so the 2.99× is unaffected — but 350 should be read as **2.4× the last published figure**, not 3×
it. The pre-fix kernel was not built as a third arm: a peer producer held ~13 GB with free RAM at
zero throughout the window, and contending with another session's job was not worth it.

**What this does not show.** One design, one column, int8, one machine, one sitting; the w=64
shapes fail to compile in *both* arms (a memtile limit, unrelated) and are excluded from both
fits. The host-load witness reads **PEER**, not CLEAR — but contention can only *depress* the CPU
figure, which makes "the CPU still wins by 2.4×" conservative rather than flattered. The MAC issue
rates are static ceilings from the schedule, not counters, which is why the whole-kernel 2.99× is
smaller than the hot loop's 6.29× — the loop is not all of the kernel. The correctness gate is
`np.allclose(rtol=0, atol=INP_SCALE)` against a torch int8 golden, a tolerance check rather than a
bit-exact one. **`conv2dk3` was not touched** and is the likeliest remaining rate-limiter; nothing
here establishes where the residual 2.4× to the CPU sits.

### Batched submission drops the dispatch floor 17×, and reopens four closed verdicts

The dispatch floor above is the single most consequential number in this repo: `docs/SILICON.md`
§3.4 states that **every** small-op verdict here is conditional on it. The same log named its
own fix, and `docs/DECISIONS.md` recorded `pyxrt.runlist` as bound but explicitly unmeasured —
*"Nothing has been run — this is an API-existence check and the next measurement to make, not
a result."* This is that measurement. Backing log `results/aie/dispatch_runlist_npu.log`.

| Path, same 32 KB passthrough | Per dispatch |
|---|---|
| IRON `@iron.jit` wall | 617.0 µs |
| IRON hardware bracket | 169.8 µs |
| **raw pyxrt, single** | **~140 µs** |
| **`pyxrt.runlist`, N=64, amortised** | **36.3 µs** |

**The go/no-go threshold moves from 617 µs to about 36 µs, a factor of 17 — for a caller
driving raw pyxrt.** The amortised figure reproduced at 35.9, 36.3, 36.0 and 36.3 µs across
four runs. It was later measured *through IRON* as well, where the device half reproduces but
the end-to-end threshold only falls to ~531 µs; see
[Batching reaches the device floor from inside IRON](#batching-reaches-the-device-floor-from-inside-iron--and-irons-own-host-work-eats-almost-all-of-it).

**The "hardware" half was not all hardware.** Raw pyxrt submits the same design in ~140 µs,
below the 169.8 µs the earlier work called the hardware bracket, and the batched figure is a
*quarter* of that bracket. That settles the question the earlier log left open: 169.8 µs is
not irreducible silicon cost.

**This is a throughput result, not a latency one, and that is the binding limit.** 36 µs is
what a dispatch costs when 64 are in flight together; a single op still pays ~140 µs raw or
617 µs through IRON. It reopens work that can be batched — many independent tiles, frames or
graph nodes — and does nothing for a latency-critical single call. N=1 *through* a runlist is
slightly slower than a raw single dispatch, so batching only pays from N=2.

**What reopens.** Against §3.4's own list of what the old floor closed, with its CPU times:

| Op | CPU time | Against a 36 µs raw-pyxrt floor | Against a ~531 µs batched-IRON floor |
|---|---|---|---|
| MobileNetV2, whole model | 1720 µs | 48× above | whole-model time, not per dispatch — see below |
| MobileViT stage-2 attention | 240 µs | 6.7× above | still under |
| bf16 attention stage 2 | 240 µs | 6.7× above | still under |
| GroupNorm at L ≤ 18816 | 233 µs | 6.5× above | still under |
| bf16 attention stage 3 | 34 µs | still under | still under |
| bf16 attention stage 4 | 12 µs | still under | still under |

Four of six reopen, without a line of kernel code. That does **not** mean they now win — it
means the floor is no longer why they lose, and kernel quality becomes the deciding question
for the first time. The attention kernel's README argued *"the op has to be ~20× larger before
the kernel quality is what decides the outcome"*; at a 36 µs floor that multiple is ~1.4× for
stage 2.

**The fourth column is the one to read if you are writing an `@iron.jit` design**, and it says
that none of the four reopens survive there. Three are single ops under 250 µs against a
~531 µs batched-IRON floor. MobileNetV2's 1720 µs is a *whole-model* CPU time being compared
against a *per-dispatch* floor — a comparison the 36 µs column inherits from §3.4 and does not
justify; the model is many dispatches, and whether it clears the floor needs per-layer
arithmetic nobody has done. Reopening any of these at 36 µs means committing to a raw-pyxrt
driver that gives up IRON's argument handling.

**Two real defects were fixed to get here**, both of which produced a wrong answer rather than
an error. `kernel(...)` in pyxrt creates *and starts* a run, so adding one to a runlist hands
`execute()` a run already in flight — this is why every batch had failed its output check;
runs must be built with `pyxrt.run(kernel)` plus `set_arg`. And the cache resolver looked for
the instruction stream as `*.txt` when the file is `insts.bin`, and took the newest xclbin
anywhere in the cache, which after any other design is compiled is a different design.

**What this does not show.** One design, one payload, one machine. It is a no-compute
passthrough, so a real kernel's configuration cost lands inside the dispatch and pushes its
floor above this one — treat 36 µs as a floor, not a constant. Nothing here is an IRON result:
the batched path is raw pyxrt, `@iron.jit` does not use runlists, and reaching 36 µs from a
real design means writing that host path. *(Both of those caveats were closed the same day —
the host path was written and a real kernel run through it; see the next section.)* The 617.0
and 169.8 µs comparators are quoted from
2026-09-07, not re-run. The reopened verdicts are arithmetic against published CPU times, not
re-measurements — each still needs its own paired run.

### Batching reaches the device floor from inside IRON — and IRON's own host work eats almost all of it

The section above measured 36.3 µs by driving raw pyxrt and closed with two caveats: *"Nothing
here is an IRON result… reaching 36 µs from a real design means writing that host path"*, and
*"this is a no-compute passthrough: a real kernel's own configuration cost lands inside the
dispatch and pushes its floor above this one."* Both are now closed. `kernels/dispatch_floor/
iron_batch.py` puts runlist submission inside IRON's own host path, and a real 8-core bf16
kernel was run through it. Backing log `results/aie/iron_batch_npu.log`.

The mechanism: `XRTHostRuntime.run()` submits with `kernel_handle.kernel(3, insts_bo, …)`, and
in pyxrt `kernel(…)` creates *and starts* a run — exactly what cannot go into a runlist. A
context manager swaps that kernel for a proxy which builds the run **unstarted**
(`pyxrt.run(kernel)` + `set_arg`) and queues it. Every other step of IRON's `run()` executes
unchanged: same ABI validation, same instruction buffer, same argument order. Nothing outside
this repo is modified, and the patch is removed on leaving the block.

| No-compute passthrough, 32 KB | Series A | Series B |
|---|---|---|
| Unbatched, median | 676.2 µs | 729.0 µs |
| Batched N=64, **device** (runlist bracket) | **37.9 µs** | **37.5 µs** |
| Batched N=64, **wall per call** | **537.1 µs** | **530.9 µs** |
| Batched N=64, host share | 499.2 µs | 493.4 µs |
| End-to-end gain | 1.26× | 1.37× |

**The device half works, and it reproduces the raw-pyxrt number from inside IRON.** The
runlist bracket falls monotonically — 167.3/170.7 µs at N=1, 45.7/45.5 at N=16, 37.9/37.5 at
N=64 — against the raw-pyxrt harness's 36.3 µs on the same design. The 17× survives going
*through* IRON's argument handling rather than around it.

**The real kernel closes the other caveat and confirms the reading from the other side.** bf16
GroupNorm at L=150528 (8 cores, 32 groups) batches to 838.3, 823.8 and 829.8 µs per dispatch
at N=4, 16 and 64, then stops. That plateau is not a measurement floor — it is the kernel's own
compute: `results/aie/groupnorm_bf16_kernel_npu.log` independently measured this design at this
L at **835.8 µs** per call (min 807.7) against 1899.3 µs on the CPU. All three batched figures
land within 1.5% of it. Batching removed ~140 µs of device-side dispatch overhead and left the
work. So a real kernel's dispatch cost is the *same order* as the passthrough's; its
configuration cost does not swamp it.

**And the end-to-end gain is 1.2–1.4×, not 17×.** That is the half a caller feels. Wall time
per call improves 1.26× and 1.37× on the passthrough and 1.23× on GroupNorm (1844.5 → 1495.7
µs). The reason is the host-share column, and it is **flat in batch size**: 478–529 µs on the
passthrough at every N ≥ 4 across both series, 666–781 µs on GroupNorm. Batching cannot touch
it because it is not dispatch — it is what IRON does per call *before* the submit: ABI
validation, buffer preparation, instruction-buffer setup.

**This reproduces and localises the 447 µs host term.** `dispatch_floor_npu.log` split the
617.0 µs floor into a 169.8 µs hardware bracket and 447.3 µs of host cost. This measures that
term from a different direction — 478–529 µs on the same design, at every batch size — and
shows it is independent of *how* the submit is done. Batching fixed the dispatch half of the
floor; the host half needs a different fix and is now the larger by more than an order of
magnitude: **37.5 µs of device against ~500 µs of host.**

**There are three thresholds, not two.** This is the correction to the section above:

| Path | Per-dispatch threshold |
|---|---|
| Unbatched `@iron.jit` call | 617.0 µs published; 676–729 µs this sitting |
| **Batched `@iron.jit` call** | **~531–537 µs** |
| Batched raw-pyxrt driver | 36.3 µs |

~531 µs is 14% below the published 617.0 µs and 21–27% below this sitting's own unbatched
medians — not 17× below either. Reopening a verdict at 36 µs means committing to a raw-pyxrt
driver that gives up IRON's argument handling.

**N=1 and N=2 are slower than unbatched, and that is not noise.** A one-call batch pays to
construct a runlist and amortises nothing: 0.91×/0.86× on the passthrough, 0.96× on GroupNorm.
Batching is only worth reaching for at N ≥ 4.

**What this does not show.** Two designs, one machine, one sitting; the passthrough was run
twice and both series are printed in the log, with the N=1 and N=2 rows moving between them.
The host share is a *subtraction* (wall minus runlist bracket), not a profile of `run()`, so it
is an upper bound that includes Python loop and buffer-allocation time. Nothing here *reduces*
the host share — identifying which part of IRON's per-call work dominates it would need a
profile, and is now worth more than any further dispatch work. Only the transaction submit path
is batched; the full-ELF path builds its own `pyxrt.run()` and is refused with a clear message.
Verification is per batch after the flush, so a batch whose runs executed in the wrong *order*
would still pass. The GroupNorm arm checks finiteness and non-zeroness — enough to catch a
batch that silently did nothing, which is the failure mode batching introduces, but not an
accuracy check.

**Batching gives up per-call completion status entirely, and it cannot be recovered.** A run
inside a runlist cannot be polled on this binding — `run.state()` raises *"Cannot poll a command
that has not been submitted"* — so `runlist.wait()` is the only completion signal. **Verifying
output buffers is the only correctness gate under batching**, and a batch that silently did
nothing would otherwise look extremely fast.

### A C++ XRT host reaches the device floor, and 36 µs is not a Python artifact

The two sections above both measured through pybind11, which left one question open and it was
the one that mattered: is the residual per-call cost the *binding* or the *driver*? A C++ host
answers it, and the answer decides whether a deployable runner can reach 36 µs or whether
~500 µs is what a real caller always pays. `kernels/dispatch_floor/dispatch_runner.cpp` is a
standalone C++ XRT host with no Python in it, driving the **same** cache entry — same
`final.xclbin`, same `insts.bin`, same argument layout. `scripts/run-dispatch-cpp.sh` runs the
Python and C++ arms back to back **in one sitting**, because this repo's own rule is that NPU
latency drifts between sittings independent of any code change. Backing log
`results/aie/dispatch_cpp_runlist_npu.log`.

Per-dispatch µs on the 32 KB no-compute passthrough, two independent series each:

| N | Python runlist (A / B) | C++ rebuild (A / B) | C++ persistent (A / B) |
|---|---|---|---|
| 1 | 144.0 / 146.9 | 148.5 / 152.4 | 125.7 / 123.0 |
| 4 | 59.8 / 59.7 | 61.5 / 63.2 | 58.4 / 57.9 |
| 8 | 47.1 / 47.3 | 48.7 / 48.8 | 47.4 / 47.6 |
| 64 | **35.9 / 35.9** | **36.7 / 36.8** | **36.7 / 36.7** |

**The 36 µs floor belongs to the driver, not to Python.** At N=64 C++ reads 36.7 µs against
pyxrt's 35.9 µs, in both series, and from N=4 upward the two languages agree within ~2% — with
Python marginally the *faster* of the two at every N ≥ 16. Removing the binding entirely does
not move the batched floor. This closes the question the runlist section left open in the
direction that makes its figure **more** trustworthy: 36.3 µs was never a measurement of pybind
overhead.

**A deployable C++ runner reaches that floor — the open item is closed.** Same design, same
sitting, per dispatch: **671.5 µs** unbatched through IRON, **498.5 µs** batched through IRON,
**36.7 µs** through the C++ host. That is 18.3× better than IRON unbatched and 13.6× better
than IRON batched. IRON's 460–573 µs host share is *not* irreducible; it is work a cached-handle
host does once at startup (33–60 ms here) instead of on every call.

**So there are four thresholds, not three**, and which applies depends entirely on the host:
671.5 µs (IRON, one call) · 498.5 µs (IRON, batched 64) · ~108 µs (C++, one call) · **36.7 µs**
(C++ or pyxrt, batched 64). Nothing above is retracted — each still governs its own host path.

**The single-dispatch path is only partly Python.** C++ single runs 108.1 / 109.8 µs mean
against raw pyxrt's 140.8 / 127.0, so roughly 20–30 µs of it is binding overhead — real, but a
minority. C++ still pays ~108 µs for one dispatch against 36.7 µs batched, so **~70 µs per
submission is driver-side work that only batching amortises, in any language.** Latency-critical
single calls do not benefit from the rewrite; throughput does.

**Persistent runlists buy a little, and the mechanism is not the obvious one.** Building the list
once and re-executing beats rebuilding only at N=1 (125.7 vs 148.5 µs) and N=2, and the two are
indistinguishable from N=8 up. The first version of this section attributed that to runlist
*construction* — wrongly: in every arm, and in the Python harness, the timer starts *after* the
`set_arg`/`add` loop, so construction was never timed at all. A `built` arm that starts the timer
before the build loop measures it properly: construction costs ~18 µs fixed **plus ~3.3 µs per
run added**, so it *grows* with the batch (21.7 µs at N=1, ~220 µs to build a 64-run list) rather
than amortising away. The fresh-versus-reused *execution* gap — what the earlier claim was
actually pointing at — is ~27 µs at N=1 and gone by N=8. End to end, which is what a caller who
rebuilds every time faces, `built` vs `persistent` is 169.2 vs 120.8 µs at N=1 (1.40×) and
40.3 vs 36.8 µs at N=64 (**1.09×**). Reusing the list is worth ~9% at N=64 — real, but small
beside the 13.6× that comes from not being IRON.

**A wrong-design bug had to be fixed first, and it produced a plausible wrong number rather than
an error.** The first attempt read FAILED VERIFICATION in both arms with a 777.7 µs single
dispatch. `measure_runlist.py` resolved the wrong design: its rule was "newest cache entry
holding both `final.xclbin` and `insts.bin`", and its own comment claimed the `insts.bin`
condition stopped "a GEMM, say" being picked up — but *every* IRON design writes `insts.bin`, so
the condition excluded nothing. Once `bank_placement/` and `gemm_reblock/` were compiled later
the same day, "newest" was one of those. The two designs are not subtle: the passthrough is 75
instruction words, what it resolved was 1052. It now captures the paths IRON itself returns from
`CompilableDesign.compile()` and **refuses** to fall back to "newest".

**What this does not show.** One design (a no-compute passthrough), one payload, one machine —
this bounds the *host*, not any model's latency; a real kernel plateaus on its own compute. The
C++ host is a benchmark host, not an inference runtime, and nothing here shows the VitisAI EP or
any ONNX path can use a runlist. Batching still gives up per-call completion status, so any
deployment at 36 µs inherits output verification as its only correctness gate. The one-time
setup cost varied 1.8× between two runs of the identical binary and was not investigated.

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

### AIE2 machine code: the bundle count of a loop is its cycle count

Backing log: `results/aie/aie2_isa_static.log`. Tools: `tools/aie_disasm.py`,
`kernels/acc_spill_probe/`. **No hardware was used.** Every figure here comes from
disassembling object code with Peano's own `llvm-objdump` or compiling with Peano's `clang`,
so the whole thing runs in seconds against a busy device.

The clock measurement above made cycles convertible to seconds. It did not say where the
cycles go. `docs/SILICON.md` 1.2 carried MACs per cycle and the vector width as SPEC rows
copied from AMD's `device.yaml`, issue width appeared in no document in this repo, and two
documents disagreed about the accumulator file. All three are now read off the machine code.

**The bundle format.** The nop mnemonics name the slots: `nopb ; nopa ; nops ; nopx ; nopm ;
nopv`, so six slots — branch, load, store, scalar, move, vector. `nopxm` is the fused
encoding printed when x and m are both idle, so a five-field bundle still occupies six slots.
Bundles using few slots are emitted compressed, shorter than 16 bytes, and still issue in one
cycle, so cycles count by bundle and never by byte.

**The calibration, and the finding that comes out of it.** S0 measured two loops at exactly
9.000 and 2.000 cycles per iteration, constant from 2^18 to 2^25 iterations. Disassembled,
the same two loops are 9 and 2 bundles. Both exact.

| Loop | Measured cycles/iteration | Bundles in the loop body |
|---|---|---|
| scalar, `volatile` load-add-store | 9.000 | 9 |
| vector, dependent 16-lane `aie::add` | 2.000 | **2** |

That equality is the point. AIE2 is a statically scheduled VLIW with an exposed pipeline, so
Peano covers every operand latency with explicit nop bundles instead of leaving it to a
hardware interlock. The scalar loop shows the mechanism: six consecutive all-nop bundles sit
between the load and the add that consumes it, so a scalar load's result reaches the seventh
bundle after it issues, and that latency is the whole reason the loop costs 9 cycles to do
one add. **An inner loop's cycles per iteration can therefore be read before the kernel is
ever run.** The exception is a loop that waits on a lock, a stream or a DMA, which takes
longer than its bundle count; the disassembly cannot say how much longer, and that is what
the trace unit's stall events are for.

**Compiling for the core without IRON.** `clang++ --target=aie2-none-unknown-elf -std=c++20
-O2 -D__AIE_API_AIE_ADF_HPP__=1 -c -I <mlir_aie>/include` builds a kernel object directly.
The flag predefines the include guard of `aie_api`'s graph-level ADF header so its body is
skipped; that header includes `<adf.h>`, which ships with Vitis and exists nowhere on this
machine. Recompiling the clock probe's own source this way reproduces the object IRON built
for the hardware run — same 102 bundles, same 29 full-width and 73 compressed, same 32-byte
frame, same four loops at the same addresses — which is what makes the flag safe to use.

**The accumulator file, and two documents corrected.** `kernels/acc_spill_probe/` holds K
live `aie::mmul<4,8,8,int8,int8,acc32>` accumulators across a k-reduction loop, the shape
upstream's `conv2dk3` uses, and sweeps K.

| Live accumulators | Accumulator registers named | Stack references |
|---|---|---|
| 1–5 | 1 to 6 | **0** |
| 6 | 9 | 5 |
| 7 | 9 | 17 |
| 8–12 | 9 | 25 to 96 |

The allocator names nine accumulator registers, `cm0`–`cm8`, reaches nine at six live
accumulators and never goes past it however many more are asked for. Five live accumulators
of this shape compile with no stack traffic at all; six is the first count that touches the
stack. Both prior claims were wrong in opposite directions: `docs/DECISIONS.md`'s "only 6
hardware accumulator registers" is below the nine names that appear, and `docs/SILICON.md`'s
"≤4 stays in registers" is one below the real spill-free ceiling. Both were inferred from the
single `conv2dk3` kernel that spilled at 8, and both are now marked superseded rather than
removed. The width fix in `kernels/conv2dk3_widthfix/` was written to N ≤ 4 for safety, so it
is correct but one accumulator short of what fits.

Caveat: one accumulator shape, one optimisation level, one compiler version. A wider
accumulator fits fewer, and nine register names is a lower bound on the architectural file
since the allocator may simply never have needed a tenth.

**The production int8 GEMM, read the same way.** The kernel behind the 4607.05 GOPS above has
a nine-bundle inner loop issuing eight `vmac` instructions, one per live accumulator
`cm0`–`cm7`, with its operands arriving on the load and store slots of the same bundles. One
int8 `vmac` is the 256-MAC operation the 256 MACs/cycle nameplate describes, so the loop
issues 0.889 vector MACs per cycle, **88.9% of the machine's MAC issue rate**.

Set that against the measured whole-kernel figure. 4607.05 GOPS over 16 cores at 1.7983 GHz
is 31.3% of the 14,730 GOPS those cores can issue. The inner loop is at 88.9%. **The missing
factor is not the inner loop's instruction schedule**, so rewriting it is not where the time
is — the question is how much of the elapsed time is spent inside that loop at all, which is
a dispatch, DMA and occupancy question rather than a kernel-quality one. The function does
spill, a 416-byte frame and 37 stack references, consistent with it holding eight live
accumulators where five is the ceiling; but none of that traffic is in the nine loop bundles,
so the spills cost setup per call and not per-iteration throughput.

**Hand-written assembly is available; the cycle counter still is not.** `docs/DECISIONS.md`
recorded that Peano "rejects inline asm", which closed hand-scheduling on this part. That is
true only of statement-level inline asm inside a C++ function, which dies in the IRTranslator.
A standalone `.s` file never enters instruction selection: `kernels/asm_probe/` assembles one,
compiles a C++ caller, links them, and both symbols resolve with nothing undefined. So a
hand-scheduled inner loop is available wherever the compiler's schedule is the binding
constraint, which the tool above can now identify.

It does not rescue the cycle counter. Enumerating the special registers the assembler accepts
as a `mov` source, by trying to assemble each, yields only `CORE_ID` — even `PC`, `SP` and `LR`
are refused there. The register database puts the tile timer at memory-mapped `0x340F8` and
`0x340FC`, in the configuration space reached over AXI-MM from the host or a DMA, not in the
core's data space, whose stack this toolchain places at `0x70000`. The trace unit remains the
only path to it, which is what the clock work concluded from three other failures; this is a
fourth independent route to the same answer.

**What this does not show.** Nothing here is a hardware measurement, the assembly result
included — the object assembles, disassembles and links, but no hand-written kernel has been
run on the NPU. The bundle-equals-cycle identity is checked against two measured loops and no
more. The 88.9% is the inner loop's
issue density, not the kernel's utilisation. The slot names come from the nop mnemonics
`llvm-objdump` prints, not from a published AIE-ML ISA document, which this project does not
have.

### The trace unit as a performance-monitoring unit: 68% of a short kernel's cycles are lock wait

Backing log: `results/aie/pmu_probe_npu.log`. Tool: `kernels/pmu_probe/`.

The clock made cycles convertible to seconds and the disassembly made an issuing loop's cost
readable. Neither says anything about a core that is *not* issuing, which is where every
losing verdict in this repo actually lives. The AIE2 trace unit carries a stall taxonomy
(`MEMORY_STALL`, `STREAM_STALL`, `LOCK_STALL`, `CASCADE_STALL`), an occupancy signal
(`ACTIVE`, `DISABLED`) and an instruction mix, eight events at a time per tile. None had been
used on this machine.

`kernels/pmu_probe/` reuses the clock probe's kernel and design unchanged and swaps only the
event list, so its loops are the two whose cycles per iteration are already measured here.
That makes the first run a calibration, not a measurement.

**How a level event is encoded.** One frame per cycle. `ACTIVE` returns 27,864 hits over a
span of 27,863 cycles. The trace unit compresses consecutive identical frames into Repeat
frames itself, which is why this does not overflow: the vector loop at 65,536 iterations spans
142,730 cycles and still fits in 1,344 bytes.

**Calibration.** Cycles per iteration converge on the measured values as the loop grows and
the fixed entry cost amortises.

| Loop | Iterations | Cycles/iteration | Measured by S0 |
|---|---|---|---|
| vector | 4,096 | 2.0051 | 2.000 |
| vector | 16,384 | 2.0013 | 2.000 |
| vector | 65,536 | **2.0003** | 2.000 |
| scalar | 2,048 | 9.0020 | 9.000 |
| scalar | 8,192 | 9.0005 | 9.000 |
| scalar | 32,768 | **9.0001** | 9.000 |

**The accounting, which is the stronger result.** Subtracting the traced stall cycles and the
loop's own cycles from the cycles the core was alive leaves exactly 190 cycles on every vector
run and exactly 198 on every scalar run, across a 16× range of work. A residual that is
constant rather than proportional is the kernel's prologue and epilogue, and it is what says

```
cycles the core is alive = issuing + memory + stream + lock + cascade stalls
```

closes on this hardware. `ACTIVE` is inclusive of stall cycles, not exclusive of them — a core
waiting on a lock is still enabled and not halted — so issuing cycles are what remains after
the stalls are subtracted rather than a figure the hardware reports directly.

**What it found.** `LOCK_STALL` is the only non-zero stall term in any run, and it is large.
The core waits 8,500–12,700 cycles per dispatch on the input ObjectFifo's lock, about 5–7 µs
at 1.80 GHz, and that barely moves as the loop grows 16×, so it is a fixed cost of getting
data to the core rather than a function of the work. On the shortest run it is 18,926 cycles
against 8,915 of issuing: **the core spends 68% of its life waiting and 32% computing**, on a
kernel whose inner loop the disassembly rates as perfectly scheduled. That is the mechanism
this project has been inferring from throughput fits since the first kernel lost.

It also sharpens the 169.8 µs hardware dispatch floor above. Some of that floor is visible
from inside the core as lock wait, but only a little: 10,000 cycles is about 3% of 169.8 µs,
so the rest is outside the core entirely.

**What a buffer costs, and the rule that comes out of it.** A single dispatch does not
amortise anything. Streaming 16 buffers through the same core and sweeping the compute per
buffer separates the fixed and per-buffer terms.

| Compute per buffer (cycles) | Issuing cycles per buffer | Difference | Lock stall (total) |
|---|---|---|---|
| 32 | 740 | 708 | 7,523 |
| 128 | 824 | 696 | 12,059 |
| 512 | 1,229 | 717 | 11,912 |
| 2,048 | 2,765 | 717 | 11,876 |
| 8,192 | 8,909 | 717 | 6,948 |
| 32,768 | 33,485 | 717 | 12,173 |
| 131,072 | 131,789 | **717** | 9,835 |

The per-buffer overhead is a constant 717 cycles — not a fit or a trend, the same residual at
131,072 cycles of compute as at 512, four thousand times smaller. The lock wait stays flat too,
7,000–12,000 cycles regardless of the work, and varies as much between repeats of one point as
across the whole sweep, because it is DMA timing. The issuing side is deterministic: the large
points come back bit-identical between runs.

That 717 decomposes entirely into things already measured here. 512 cycles are this probe
kernel's own flush loop, the 256 event pairs it emits so the trace packet reaches host memory,
which the disassembly rates at 8 bundles per 4 pairs. 190–198 cycles are the kernel prologue
and epilogue isolated above. What is left, about 15 cycles, is the genuine ObjectFifo acquire
and release. So for a kernel shaped like this one, on one core:

```
cycles = n_buffers x (compute_per_buffer + ~205) + ~10,000
```

where the 205 is the per-buffer handoff including the kernel call and the 10,000 is the fixed
dispatch lock wait. **A buffer carrying less than a few hundred cycles of work is mostly
handoff, and a dispatch carrying less than about 10,000 cycles of work in total is mostly
waiting.** Both are lower bounds, measured on the easiest kernel available: one core, no
cascade, no neighbour traffic, operands already in registers. A real kernel pays more.

**What this does not show.** Only `LOCK_STALL` has been seen non-zero, so three of the four
stall categories are unexercised and are not shown to work by this run. One core tile, one
power mode. The traced window starts when the trace unit is enabled rather than when the
dispatch begins, so the cycles the core was alive are not the whole submit-to-wait bracket and
must not be compared against it directly. `ACTIVE` being inclusive of stalls is inferred from
the accounting closing, not from a document.

### The int8 GEMM issues at 40% of nameplate, and a ~3,200-cycle per-buffer floor caps it

The two results above compose into something neither gives alone. If a hardware loop's bundle
count is its cycle count, and a buffer costs a constant on top of its work, then a kernel's
**issuing time is computable from its object file without running it**. Subtracting that from a
measured time leaves the cycles the core spent not issuing — the quantity every losing verdict
in this repo has been missing. `tools/gemm_cost_model.py` does that computation for the
`mm.cc`-shaped tiled GEMM; backing log `results/aie/gemm_cost_model.log`.

**A correction first.** The static-ISA section above is headed "the kernel behind
`results/aie/int8_matmul_sweep_npu.log`'s 4607.05 GOPS" and disassembles the object in cache
`0816364bbbaf03f83e2f0bcd`. That cache carries `memref<64x32xi8>` buffers, so it is the
**default n=32 build**, which measured 2387.01 GOPS — not the tuned n=64 build that produced
4607.05. The tuned kernel is a different object hash, `matmul_i8_i32_86901378.o`. Every number
in that section survives, because the two objects' loops are identical: nine bundles, eight
`vmac`, `cm0`–`cm7`, 88.9% MAC issue density. Only the attribution was wrong, and it is
corrected here rather than edited out of the log.

**A first reading of this was wrong, and the correction moves the answer.** The kernel was
modelled as one hardware loop with straight-line setup, charging its 135 non-loop bundles once
per call. `matmul_i8_i32` is a **nest**: two software loops around the hardware loop, with the
accumulators loaded before it and stored after it, and that whole body re-run once per group of
live accumulators. Three things in the disassembly say so — two backward branches after the
hardware loop each with their own induction update and bound test, a loop body with eight
`vmac` and **no accumulator store**, and compile-time loop bounds (`mova r3, #0x8` →
6 hardware-loop trips, `mova r7, #0x6` → 4 inner, `mov r8, #0xc` → 4 outer). The trip counts
reconcile exactly: 16 groups × 64 `vmac` = 1,024 = 64³/(4·8·8), nothing left over.
Superseded numbers are kept below; backing log `results/aie/gemm_cost_model_nest.log`.

Both tiles, same 4096×2048×2048 problem, same 16 cores, same sitting in the source log:

| Tile | `vmac`/call | Issuing cycles/call | Measured cycles/call | Issuing | MAC rate over the call |
|---|---|---|---|---|---|
| m64 k64 **n64** | 1024 | 2442 | 3274 | **74.6%** | 107.3 (41.9% of 256) |
| m64 k64 **n32** | 512 | 1306 | 3160 | **41.3%** | 100.4 (39.2% of 256) |
| *superseded, one-loop reading* | | *1323 / 738* | | *40.4% / 23.4%* | *198.1 / 177.5* |

**The schedule is the larger loss, and the first reading put it in the wrong place.** Over a
whole call the kernel issues about 100–107 MACs per cycle against the 256 the tile can retire,
roughly 40% — not the 77.4% the one-loop reading gave. The 88.9% figure for the inner loop is
correct and unchanged, but it covers only 54 of the 141 cycles an accumulator group costs. The
other 87 bundles per group are accumulator loads, accumulator stores and stack spill traffic,
run 16 times per call at n=64. **That is a direct consequence of a number measured two sections
above:** five live 4×8×8 int8 accumulators is the spill-free ceiling and this kernel holds
eight, with a 416-byte frame and 33 stack references.

**And the spill is a blocking defect, not a width limit — bf16 proves it.** The two dtype paths
in `mm.cc` ask the register file for the *same* total accumulator width: int8 takes 8
accumulators of 1024 bit, bf16 takes 16 of 512 bit, both 8192 bits across the same 8 of the
file's 9 registers. If the ceiling were a width budget, bf16 would spill too and reblocking
int8 would buy nothing. It does not spill at all: a **64-byte frame** whose 15 stack references
are every one of them scalar, against int8's **416-byte frame** carrying **12 vector spills**
(12 slots × 32 B + 32 B of scalar reconciles 416 exactly). The file itself is 9 registers
addressed at three granularities — `cm` full, `bml`/`bmh` halves, `amll`…`amhh` quarters —
so it was never 9 *or more*. Backing log `results/aie/accumulator_width_vs_count.log`. This is
the existence proof H11 needed: a blocking that fits spills nothing.

**The per-buffer floor is the finding that survived the correction.** Measured cycles per call
are 3,274 at n=64 and 3,160 at n=32 — a 3.6% difference for buffers whose compute differs by
2×. That is a measurement, not a model output, and neither reading changes it. What the
corrected model changes is how full the slot is: n=32 puts 1,306 issuing cycles into a
~3,200-cycle slot and n=64 puts 2,442 into it. So n=64 is 1.93× faster because it nearly fills
a slot whose length barely moves, and the remaining headroom is about **1.3×, not 2.4×**.

**A candidate constant is now falsified.** The first reading bounded the handoff by charging
the trace probe's entire measured 717 cycles per buffer. At the corrected issuing cost that
bound predicts **117.5%** of the measured time at n=64, which is impossible. The probe's 717
does not transfer to another kernel — exactly as its own decomposition said, since 512 of it
was that probe's trace-flush loop and 190 its kernel prologue, leaving only the ~15-cycle
acquire/release as a property of the ObjectFifo.

Note that the schedule × issuing identity reproducing the measured fraction of peak is
**arithmetic, not evidence**: the per-call cost cancels, so it holds for any value and checks
only the tool's bookkeeping. The evidence for the nest reading is the trip-count reconciliation
above, and the tool refuses to produce a number when that reconciliation fails.

Three hypotheses:

- **H9.** Stream-port tracing measures a sustained input rate at or below **2.5 B/cycle** into a
  core. Bytes into L1 per call are 8,192 (n=64) and 6,144 (n=32); over the measured cycles per
  call that is 2.50 and 1.94 B/cycle, against the 3.35 and 4.71 a never-starved core would
  need. Fails if the ports read faster, which would move the missing time elsewhere.
- **H10.** Per-buffer wall time is a floor set by the data path, so throughput rises with work
  per buffer until the issuing cost approaches ~3,200 cycles — about 1.3× headroom at n=64.
  Fails if a larger tile does not raise throughput. **Already obstructed:**
  `results/aie/int8_matmul_sweep_npu.log`'s probes at m=128 and at k=128 both failed to build
  with `'aie.tile' op Basic sequential allocation failed`, an L1 capacity limit. Reaching the
  headroom means changing what occupies L1 — buffer depth, or the 16 KB single-buffered output
  tile — not asking for a bigger tile.
- **H11 — RUN 2026-09-09. The kernel improved on every static measure and the wall clock did
  not move.** Switching the int8 path from `matmul_vectorized_4x2_mmul` (8 live accumulators)
  to the `2x2` template already in `mm.cc` (4) gives: 144 → **88** bundles, a 416 → **32**-byte
  frame, 33 → **5** stack references, **every one of the twelve vector spills gone**, and a
  hardware loop of 8 bundles issuing 8 MACs — **1.000 `vmac`/cycle**, up from 0.889 and at the
  ceiling. An issue-bound design should then run ~12% faster. Two alternating A/B series gave
  best-to-best **+0.8%** and **−2.1%**, medians **+2.0%** and **+0.4%** — inside ±2%, with the
  sign not even stable. Backing log `results/aie/gemm_reblock_h11_npu.log`; harness
  `kernels/gemm_reblock/`. **This is the test H12 could not be:** not "we could not resolve 3%"
  but "a 12.5% kernel improvement produced nothing measurable". The core is not the critical
  path, and the per-buffer floor now rests on an intervention large enough that its absence is
  the evidence. Keep the two lines — they are free and strictly better — but stop expecting
  wall clock from inner-loop work on this design.

**What this does not show.** Nothing here was measured on hardware; the issuing-cycle figures
are computed from object code and the microseconds come from a run two days earlier. "Not
issuing" is a residual, not an observation — consistent with lock stall, the only category seen
non-zero so far, but not attributed by this run. The nest walk assumes the two backward branches
target the two labels in order; the branch targets are unresolved relocations in an object file
and were not read directly, so the trip-count reconciliation is the evidence. One power mode,
one dtype, one design.

### Two loads in one bank cost a cycle, and the int8 GEMM has that collision where bf16 does not

An AIE2 core tile has 64 KB of local data memory in four banks of 16 KB, and **two load
units**, so a bundle can issue two loads in one cycle. When both address the same bank the
pair costs one extra cycle. That price is measured, not assumed: a controlled experiment on
branch `research/windows-lowlevel` holds the compiled function bytes identical and changes only
the operand addresses, fitting a length sweep at r² 1.0.

| Case | Core cycles per iteration |
|---|---|
| Two loads, same bank | **12.0** |
| Two loads, separate banks | **11.0** |
| One load, same bank | 11.0 |
| Same bank, operand offset 64 / 128 / 256 B | 12.0 / 12.0 / 12.0 |

The second load is free across banks and costs exactly one cycle inside one, and the
granularity is the bank rather than the address. The same branch carries it into a single-core
tiled GEMM at this section's own 64×64×64 panel geometry: **15,232 cycles per panel with the
operands in one bank against 14,208 separated**, a 1,024-cycle difference which is one cycle
for each of that kernel's 1,024 inner iterations. Those logs are not merged here, so they are
named rather than linked; backing log for the survey below is
`results/aie/bank_conflict_survey.log`.

**A correction to this repo's own issue-width row falls out first.** The six VLIW slots were
named "`b` branch, `a` load, `s` store, `x` scalar, `m` move, `v` vector" from the nop
mnemonics alone. Tabulating every operation in each slot of 226 *strictly six-field* bundles —
the only encoding whose slot identity is unambiguous — puts `vldb` and `paddb` in slot b,
`vlda`/`lda`/`mova` in slot a, and `ret` in the scalar slot. **Slot b is the second load unit,
not the branch slot.** That is not a footnote: it is the reason a bank conflict can happen.

`tools/aie_bank_check.py` reads the allocated buffer addresses out of a core ELF's symbol
table — they are absent from the cached `aie.mlir`, which is pre-allocation — assigns each to
a bank, and counts the bundles that issue two loads:

| Build | C | A | B | Empty | Paired loads in the loop | Verdict |
|---|---|---|---|---|---|---|
| int8 GEMM, 4607.05 GOPS | banks 0, 1 | **bank 2** | **bank 2** | bank 3 | 1 of 9 bundles | **hazard** |
| bf16 GEMM, the repo's NPU win | bank 0 | bank 1 | bank 2 | bank 3 | 1 of 32 bundles | clean |

**The reason is inverted from what you would guess.** int8's operand tiles are half the size of
bf16's, so the pair fits inside one 16 KB bank and the allocator packs them there, while bf16's
larger tiles are forced apart. The int8 kernel is penalised *because* its data is smaller. The
exposure is worse than the count suggests, too: both kernels have exactly one paired-load bundle
in their steady-state loop, but that is 3.1% of the bf16 loop's 32 bundles and 11.1% of the
int8 loop's 9.

**The check predicts a number someone else measured, exactly.** That branch left both build
caches on disk — one placement with the operands sharing a bank, one without, from a single
kernel source whose compiled object hash is identical in both. Given nothing but the cache
directory and the two operand names, `tools/aie_bank_check.py` reproduces the experiment's own
labels from the ELF alone: banks 1 and 2 in the build it calls separate, both bank 1 in the
build it calls same. It finds exactly **one** paired-load bundle in the compute loop's body,
and the kernel's source fixes the trip count without inference — a 64×(64·panels)×64 int8 GEMM
over `aie::mmul<4,8,8>` with bounds 16 × 8 × 8, so one panel issues 1,024 MACs and that body
runs once per MAC.

```
predicted   1 paired-load bundle x 1,024 iterations = 1,024 cycles per panel
measured    15,232 - 14,208                         = 1,024 cycles per panel
```

Backing log `results/aie/bank_check_validation.log`. It also confirms the mechanism is
**same-bundle** paired loads specifically, not two loads merely near each other: the body is
eight bundles and only one names both ports.

**The validation found a bug in the check on its first run**, which is the point of doing it.
The first version looked only inside *hardware* loops and reported "no penalty" on that very
kernel — because the kernel keeps its compute in a **software** loop, its only hardware loop
being the trace-flush loop. `mm.cc` is the same shape: its hardware loop is only the innermost
k-reduction, and the accumulator-group body around it is a software loop. The check now walks
software-loop bodies too, and a `--operands` flag makes the verdict about the two buffers a
paired load really reads, rather than "some bank holds two buffers" — which over-reports, since
a padding buffer sharing a bank is harmless. Re-read that way, the int8 GEMM's software body
carries **ten** paired-load bundles across its 96, the one in the hardware loop plus nine in
the accumulator spill traffic. That does not change the cost model, it locates it: nine per
group over 16 groups plus one per hardware-loop iteration over 96 iterations is 240 cycles per
call, exactly the `--bank-collision all` figure below.

**What it does to the cost model.** Charging the loop's paired load raises the int8 GEMM's
issuing cost per call from 2,442 to 2,538 cycles and its issuing fraction from 74.6% to 77.5%;
charging every paired-load bundle in the function, an upper bound since this tool does not
resolve which buffers the ones outside the loop read, gives 2,682 and 81.9%. So the residual
this document has been calling starvation narrows from 25.4% to between 18% and 23%.

**And the fix is free, which makes it a controlled experiment.** Bank 3 is empty in both builds.
Moving one input tile into it changes the core's issuing time by a known amount and changes
nothing about data movement: same bytes, same DMA, same fifo depth, same function bytes.

- **H12 — attempted 2026-09-09, and the machine could not resolve it.** The intervention
  worked exactly as designed: raising the per-core `stack_size` from `0xD00` to `0x2000` shifts
  every buffer up, moving both A halves wholly into the empty bank 3 while B stays in bank 2,
  with the same kernel source, tile shapes, fifo depths, DMA and schedule, and a **byte-identical
  compiled kernel object**. Only the addresses moved. But across **seven** alternating series
  the two arms overlap and the sign of the difference changes between series — best-to-best
  −4.6%, −2.9%, +2.5%, −2.7%, −0.4%, −5.8%, −1.1%, where positive means separating was faster.
  The colliding arm's *own* floor drifted 5.0% between repeats of the identical build, which is
  larger than the ~3% effect being looked for. **So a large speedup is excluded and H12 is
  neither confirmed nor refuted.** Backing log `results/aie/bank_ab_h12_npu.log`; harness
  `kernels/bank_placement/`. A peer session held about a core throughout, and this should be
  repeated on a quiet machine.
  **What would settle it is an instrument this branch already has.** Wall time is the wrong
  observable for a 3% core-side change on a shared machine; the trace unit is not. Pointing
  `kernels/pmu_probe/`'s event routing at one core of this design reads `ACTIVE` against
  `LOCK_STALL` in core cycles, inside the dispatch, where host contention cannot reach. If the
  floor is real, the separated build's issuing cycles fall by ~96 per call and its lock stall
  rises by the same, leaving `ACTIVE` unchanged — an equality needing no timing at all. The
  obstacle is the one `RESEARCH.md` already names: `whole_array` carries no trace hook.
  **That plan was tested as an instrument on 2026-09-09 and it does not work — do not run it as
  written.** A same-bank dual-load conflict has to surface as `MEMORY_STALL`, and
  `results/aie/pmu_probe_npu.log` had already flagged that event as never having read nonzero
  here. `kernels/bank_placement/bank_stall_probe.py` is the positive control: one loop, two
  independent 16-lane loads, one compile-time flag moving stream A's base and nothing else, with
  the arm read back from the addresses the kernel reports. **`MEMORY_STALL` reads 0 in every run
  of both placements, at every trip count** — including an arm built to collide as hard as this
  design permits. So zero on the GEMM's arms would not distinguish "no conflict" from "the event
  does not fire", and the sitting would buy nothing. There is no bank-specific event to fall back
  on (the enum has `GROUP_STALL` 22, `MEMORY_STALL` 23, `DM_ACCESS_TO_UNAVAILABLE` 66 and nothing
  else), and `GROUP_STALL` read *exactly* equal to `ACTIVE` in all six eight-event runs, so it is
  not an independent signal either. Backing log `results/aie/bank_stall_control_npu.log`.
  **The control did produce a clean instruction-matched comparison, and it found no rate effect:**
  with identical `INSTR_LOAD` in both arms, colliding costs a *constant* **6 cycles** more at 256,
  512 and 2048 iterations (3850/3844, 7690/7684, 30730/30724). A one-cycle per-iteration stall
  would have cost 256, 512 and 2048, so **the invariance is the finding** — those 6 cycles are
  paid once, whatever they are. Two causes fit and this run separates neither: a fixed loop-setup
  difference (one base is a static address, the other a function argument), or a one-time
  bank-arbitration warm-up on first touch of the second bank, which only the separated arm
  touches. Neither is ruled out; the headline does not depend on which. **The
  caveat that keeps H12 open:** the loop runs at 15.0 cycles/iteration for 4 loads, so it is not
  load-bound and has slack to absorb a one-cycle stall — this bounds the penalty in a *slack*
  loop, not in the GEMM's tight inner loop.
  **Three structural facts from the same run, each of which cost an attempt**: a core's `.bss` is
  ~16 KB, not 64 KB (a 32 KB static array fails to link); it lies *entirely inside one 16 KB
  bank* (measured 0x75000–0x77C00, bank 29, with the input fifo buffer starting exactly at
  0x78000, bank 30), so no static array can straddle a boundary and the second stream had to come
  from the fifo buffer; and **`stack_size` does not move a kernel's `.bss`** — at 0x400 and
  0x1400 the streams landed identically — because it moves ObjectFifo buffers, which is what H12
  shifted, not linker-placed statics.
  **A methodological trap worth carrying forward:** at eight traced events the counters are *not
  reproducible*. Three dispatches of one identical binary gave 15665 / 30724 / 15879 cycles and
  4179 / 8195 / 4235 loads, because `ACTIVE` alone emits 38–54k frames into a 64 KB buffer, which
  overflows and truncates differently each run. Every figure above is from a four-event capture,
  where all of it is exact and repeatable. An eight-event capture at this loop length would have
  made the colliding arm look 2× slower — a pure artefact of the *separated* arm being truncated.

**A tempting join with the driver work, tested and refuted.** Local `main` measures an NPU
hardware-context-switch penalty of **+747.75 µs** (same-context dispatch 120.25 µs, alternating
across two contexts 867.99 µs, a 7.22× slowdown) and an exact five-context ceiling in
`amdxe.sys` matching Phoenix's five columns. This repo's largest unexplained blocker is the
two-process handoff floor that erased all 33 bf16 GroupNorm node wins, whose smallest shape is
**789.8 µs**. The two numbers are close enough to be worth testing, and the test refutes it:
fitting that log's whole table against element count gives a slope of 78.4 ns per element, an
intercept of **147.2 µs**, and r² 0.9997. The floor is 99.97% a per-element cost, its fixed
component is a fifth of the context-switch penalty, and the agreement at the smallest shape is
a coincidence of one row. The original diagnosis stands — the floor is conversion-bound, bf16
pack and unpack being ~90% of the round trip at the largest shape. **Do not write that the
handoff floor is the context switch.**

What the driver work *does* settle here: its userspace dispatch-preparation floor is 8.76 µs
and its hardware runlist batching overhead 3.39 µs per run. The int8 GEMM is **one** dispatch
containing 65,536 kernel calls, so those are paid once over 7,458 µs and cannot be the per-call
residual. That eliminates the host and the driver, and leaves on-chip data movement — which is
what H9 predicts and what a stream-port trace would confirm.

**A contradiction to report, not resolve.** The same driver log finds an exact five-context
ceiling in `amdxe.sys` — contexts 1–5 allocate, the sixth is rejected with NTSTATUS
`0xc01e0009` — and annotates it as "exactly matches physical Phoenix silicon column count (5
columns)". **That reading conflicts with a measurement this repo already holds.** The
measurements themselves do not conflict; only the causal claim does. Backing log
`results/aie/context_ceiling_crosscheck.log`.

- `results/multi_partition_yolov8n_5col.log` ran N processes against the per-column
  `1x4.xclbin` and recorded the partitions actually handed out. At N=5 the set **stays at
  four**, on columns 1–4. The fifth process gets no fifth partition. This is already in §1.1 of
  `docs/SILICON.md` as "Columns any path on this machine can drive: 4".
- The context benchmark loaded **`4x4.xclbin`**, which this repo has measured as occupying all
  four columns as *one* partition. Five contexts each wanting a four-column overlay is twenty
  column-occupancies on a device that exposes four. They cannot be one-per-column, so the
  ceiling of five cannot be a column count. It is a driver context-table limit.
- The benchmark's own second half agrees. Two contexts on separate columns would run
  concurrently — which is what `1x4.xclbin` measurably does, scaling to 3.65×. Instead
  alternating between two contexts costs +747.75 µs, and a large switch penalty is the
  signature of time-slicing one partition. The driver work's own conclusion, that multi-stream
  execution needs physical column isolation, is the right reading of its own data.

Worth adding in the other direction: that penalty is better supported than its headline 7.22×
suggests. The mean ratio is taken over overlapping distributions — the same-context *maximum*,
881.50 µs, exceeds the cross-context *mean* of 867.99 µs — but the minima separate cleanly at
61.10 µs against 467.90 µs, a factor of 7.7, and a minimum is the right statistic for a floor.
**The deciding run** is the same context-scaling benchmark against `1x4.xclbin`: a ceiling
still at five makes it a driver context-table limit outright, a ceiling at four makes it track
partitions. Neither outcome makes it five columns, and A1 in `docs/SILICON.md` — reach the
fifth column — stays open either way.

**What this does not show.** Nothing here was measured on hardware by this run; the cycle costs,
panel slopes and driver floors are quoted from logs on two unmerged branches. The collision is
a hazard, not a measured cost, for these two kernels — the tool does not resolve which buffers
a given paired load reads, so it is certain only inside the hardware loop where the operands are
the `mmul` tiles. Whether the penalty composes linearly for several paired loads per iteration
is untested. The bank map is read from the first of 16 core ELFs and allocation is per core. The
conv kernels have no surviving build cache, so this repo's most-lost op class was **not**
surveyed.

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
XDNA1 needs QAT or per-channel scale support, not a better PTQ recipe.** *(Narrowed 2026-09-09:
per-channel scale support is measured to be unavailable -- the EP places 0 of N nodes for any
weight scale with more than one element, so QAT is the remaining half.
[Verdict](#per-channel-weight-scales-are-rejected-outright-2026-09-09-desktop-2).)*

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
     **Superseded 2026-09-09, same correction as RegNetX-002:** the shift-cut adjustments are
     downstream of cross-layer equalization, not of the grouped-conv structure. With CLE off,
     ResNeXt-50 recovers to **68.90% top-1**
     (`results/quant/eval_resnext50_32x4d_quark_nocle_c64_cpu.log`), and the corrected analyzer
     finds zero sigma violations in the graph
     (`results/quant/shift_cut_reaudit_20260909_desktop2.log`). The dynamic-range divergence
     across groups is real; blaming the clamp for the collapse is not. DenseNet-121's bullet
     above carries the same caveat — its shift-cut observation has not been re-tested with CLE
     off.

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

**Scope correction.** This paragraph previously said the run "resolves the Roadmap's open
'webcam path… not exercised end to end' item". It resolved the round-robin half of it.
RESEARCH.md's item named a *single* `4x4.xclbin` session through `./scripts/yolo-demo.sh`
— a different demo, a different overlay, a different partition — and that stayed open
until the run in [The single 4x4.xclbin session on a real
webcam](#the-single-4x4xclbin-session-on-a-real-webcam) below. The two documents
disagreed about whether the item was closed for two days; both halves are now measured.

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

### The single 4x4.xclbin session on a real webcam

The section above answers the *round-robin* webcam question. This one answers the other
one RESEARCH.md carried: the ordinary path, one session on the shared `4x4.xclbin`
partition, `./scripts/yolo-demo.sh`, which had never been exercised end to end even
though every piece of code for it existed. What kept it open was not the hardware — it
was that the camera branch of `pipelines/yolov8n/4_detect.py` was display-only, so an
attended run left nothing behind to fold in. `--seconds N` now makes a bounded run a
logged run, and this is the first one.

Measured on Desktop 2 / Phoenix, Logitech C920s, `yolov8n_cut_xint8_adaround.onnx` at
640×640, one 15s window (`results/webcam_single_4x4_yolov8n_cut_xint8_adaround_npu.log`):

| | value |
|---|---|
| EP placement | **922/929 nodes on the NPU** |
| frames | 439 in 15.0s = **29.2 fps end to end** |
| infer (`sess.run` only) | mean **6.86 ms**, median 6.79, p95 7.39 → 145.7 fps if infer-bound |
| post (numpy DFL decode + NMS) | mean 2.90 ms |
| loop (capture → draw) | mean 24.53 ms, median 19.82, p95 43.90 |
| detections | 579 over 439 frames (person 394, couch 82, chair 64, laptop 39) |

**It works, and the NPU is not the limit — the camera is.** 29.2 fps end to end against
a camera delivering 30.0 fps: the demo is camera-bound, with the NPU turning frames
around in 6.86 ms and roughly 5× of unused headroom sitting behind a 30 fps webcam. That
is the same conclusion the round-robin demo reached at this model size, by a different
route.

**The host contention is visible in the numbers, and it is confined to the host.** A
peer session's `3b_quantize_cut.py` on yolov8x held ~3.4 of 16 cores for the whole
capture (`results/load_webcam_single_4x4_yolov8n_cut_xint8_adaround_npu.log`, verdict
`PEER`); `xrt-smi` read "No hardware contexts running" beforehand, so the NPU itself was
uncontended. It shows: per-second `loop` swings between 16.6 and 44.6 ms across the run
while `infer` never leaves 6.8–6.9 ms. The end-to-end 29.2 fps and the loop p95 of 43.90
ms therefore carry that caveat and the infer figure does not — but the run should still
be repeated on a quiet host before the fps number is treated as this path's ceiling.

**What this does not settle: the single-session vs round-robin latency comparison.** The
obvious reading — 6.86 ms on the shared 4×4 partition against ~18.0 ms per worker on a
single `1x4` column, so splitting into four columns costs per-frame latency and buys
nothing while the camera is the ceiling — is *consistent* with the static-image sweep's
own per-column vs shared-4x4 ratio (14.9/8.9 ≈ 1.66× at n, above), but the two live
numbers were captured on different days, and NPU latency on this machine drifts across
sessions independent of any code change. Treat the direction as established and the
multiple as not measured: a real comparison needs both configurations captured together.
The two runs also differ in capture mode — this one delivered 1280×720 (`4_detect.py`
requests 720p; the multipartition demo requests nothing and got 640×480), both at 30.0
fps — which leaves host-side letterbox cost, not NPU work, as the affected term.

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

### Category D, second candidate: FastDepth (MobileNet-NNConv5dw)

`pipelines/fastdepth/` — new pipeline, built against MIT's FastDepth architecture (Wofk et al., ICRA 2019).
Tests Category D's depthwise-separable convolutional decoder hypothesis: pure convolutional encoder-decoder
monocular depth estimation using a MobileNet encoder and a depthwise separable decoder (`NNConv5dw-skipadd`).
All 5 upsampling stages natively use nearest-neighbor resize, avoiding the multi-subgraph fragmentation observed
in stock bilinear MiDaS.

Exported cleanly to `models/fastdepth_fp32.onnx` (opset 17, 84 nodes: 38 Convs, 27 Clips, 11 Relus, 5 Resizes, 3 Adds;
static batch 1, input shape `[1, 3, 256, 256]`, output shape `[1, 1, 256, 256]`). Preprocessing is byte-identical
between calibration and inference via `npu/fastdepth.py` (cv2-only, standard `[0, 1]` scaling RGB / 255.0,
`cv2.INTER_LINEAR` resize).

#### Monolithic DPU offload and op placement

Quantized to Quark XINT8 with 300 calibration images (`data/fastdepth_calib/`, `results/quant_fastdepth_xint8.log`):
254 nodes in quantized ONNX graph.

The VitisAI EP accepts **255 of 257 nodes (99.2%) on NPU** (`results/diag_fastdepth_xint8.log`), compiling into
**exactly 1 monolithic DPU subgraph** (`subgraphStat: [{'device': 'DPU', 'count': 1}]`). Only the outer input
`QuantizeLinear` and output `DequantizeLinear` execute on CPU:
- All 38 Convolutions execute natively on AIE.
- All 27 `Clip` (ReLU6) and 11 `Relu` activations execute natively on AIE.
- All 5 nearest-neighbor `Resize` layers compile natively on AIE with zero internal CPU fallbacks.
- All 3 residual skip `Add` layers compile natively on AIE.

#### Tri-Hardware Performance Comparison

Measured on Desktop 2 (Ryzen 7 8700G, Radeon 780M, Phoenix XDNA1 NPU, 50 iterations, batch 1,
`sess.run` only, `models/fastdepth_fp32.onnx` vs `models/fastdepth_fp32_xint8.onnx`):

| Hardware / Provider | Precision | Subgraphs | Latency (mean) | Latency (median) | Throughput | Backing Log |
|---|---|---|---|---|---|---|
| CPU (Zen 4, 8C/16T) | FP32 | 1 (CPU) | 3.22 ms | 3.17 ms | 310.2 fps | `results/lat_fastdepth_cpu.log` |
| iGPU (Radeon 780M, DirectML) | FP32 | 1 (DML) | 3.02 ms | 2.63 ms | 331.1 fps | `results/lat_fastdepth_dml.log` |
| **NPU (Phoenix XDNA1)** | **XINT8** | **1 (DPU)** | **2.87 ms** | **2.78 ms** | **348.1 fps** | `results/lat_fastdepth_xint8_npu.log` |

**Findings:**
1. **NPU beats both Zen 4 CPU and Radeon 780M iGPU**: At **2.87 ms (348.1 fps)**, FastDepth on Phoenix XDNA1
   is **1.12× faster than 8-core Zen 4 CPU** (3.22 ms) and **1.05× faster than Radeon 780M iGPU DirectML FP32** (3.02 ms).
   This establishes FastDepth alongside SESR-M7 and Real-ESRGAN 128² as vision pipelines where the NPU beats the integrated GPU.
2. **3.77× faster than MiDaS v2.1 Small**: Pure depthwise separable decoding cuts latency from MiDaS's 10.81 ms
   to 2.87 ms on the same silicon, delivering over 340 frames per second of continuous depth estimation.

#### Quantitative Depth Fidelity Evaluation

Evaluated across 50 validation scenes (`data/fastdepth_val/`) against the FP32 reference model running on CPU:

| Metric | CPU XINT8 | NPU XINT8 | Delta (NPU vs CPU) | Backing Log |
|---|---|---|---|---|
| Pearson Correlation $r$ | 0.9363 +/- 0.0751 | **0.9383 +/- 0.0738** | +0.0020 | `results/eval_fastdepth_xint8_cpu.log` / `results/eval_fastdepth_xint8_npu.log` |
| Mean Absolute Diff (MAD) | 16.33 / 255 | **16.14 / 255** | -0.19 / 255 | `results/eval_fastdepth_xint8_cpu.log` / `results/eval_fastdepth_xint8_npu.log` |
| Root Mean Squared (RMSE) | 21.36 / 255 | **21.06 / 255** | -0.30 / 255 | `results/eval_fastdepth_xint8_cpu.log` / `results/eval_fastdepth_xint8_npu.log` |
| Threshold Acc ($\delta < 1.25$) | 68.21% | **68.07%** | -0.14% | `results/eval_fastdepth_xint8_cpu.log` / `results/eval_fastdepth_xint8_npu.log` |
| Threshold Acc ($\delta < 1.25^2$) | 85.40% | **85.72%** | +0.32% | `results/eval_fastdepth_xint8_cpu.log` / `results/eval_fastdepth_xint8_npu.log` |
| Evaluation Latency (infer) | 10.49 ms | **2.80 ms** | -7.69 ms (3.75× faster) | `results/eval_fastdepth_xint8_cpu.log` / `results/eval_fastdepth_xint8_npu.log` |

FastDepth preserves relative scene depth and structural geometry exceptionally well under plain XINT8 PTQ:
Pearson $r = 0.9383$ (substantially higher than MiDaS v2.1 Small's 0.8706) and MAD of $16.14 / 255$ (vs MiDaS's $26.02 / 255$).
Visual inspection (`results/fastdepth_depth_npu.jpg` vs `results/fastdepth_depth_cpu.jpg`) confirms sharp depth boundaries
around foreground objects and consistent planar surfaces without quantization contouring.

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

### Does byte parity survive the vendor compiler?

Backing log: `results/quant/ignition_quark_pair_diff.log`. Tool: `tools/quant_pair_diff.py`.
No NPU session was built.

Every Ignition parity result above is reported as `INT8_EXACT`, which is the gate in
`quant/verify.py`: zero structural node delta, zero byte mismatch on any scale, zero-point or
non-int8 initializer, and at most 1 LSB on int8 weights, in practice 0. That is a claim about
the numbers. It is not a claim that the two files are byte-identical, and they are not —
`resnet50_ignition_cle_c64.onnx` and its Quark oracle differ by 8,171 bytes.

Splitting that difference into numeric and non-numeric parts:

| | Ignition | Quark |
|---|---|---|
| Nodes / initializers | 380 / 470 | 380 / 470 |
| Structural node delta | — | **0** |
| Initializers differing by a byte | — | 0 |
| int8 initializers over 0 LSB | — | 0 of 108 |
| `producer_name` | `Ignition` 0.1.0a1 | `quark.onnx` 0.11rc1 |
| `opset_import` entries | 1 | **9** |
| Node names matching in order | — | 209 of 380 |

Every number and every edge matches. The whole 8,171 bytes is producer metadata, eight extra
opset domain declarations, and 171 renamed nodes — exactly the material a compiler is entitled
to ignore, and exactly the material it is entitled not to.

**The test this sets up was not run.** Compiling both files through the EP with a fresh cache
and comparing the resulting `compiled.*.xmodel` and `4x4.xclbin` would say whether parity
extends from the file to the program the silicon actually runs, which is a stronger claim than
this repo makes anywhere. It was skipped because the host-load check reported
`HOST_LOAD_VERDICT PEER` — another session was part-way through a bisenetv2 quantize on the
CPU, an EP compile is CPU-heavy, and the repo's own wrappers refuse to start on a contended
machine. The NPU itself was idle. Re-run it on a clear machine; the eight extra opset domains
are the first thing to suspect if the two compiles differ.

### The shift-cut hazard predictor calls a working architecture 100% infeasible

`quant/shift_cut.py` (branch `main`, unmerged at the time of writing) formulates a real
hardware constraint: the DPU maps its 32-bit accumulator to int8 through a 15-bit multiplier
and an arithmetic right shift confined to σ ∈ [0, 31], so a scale triple whose ideal factor
cannot be written in that window is infeasible on the silicon. The audit flags such
convolutions, and its flags line up with two documented failures — RegNetX-002's collapse to
0.50% top-1, and BiSeNetV2's fall from 59.47% pixel accuracy under CPU simulation to 15.33%
on hardware. That second case is exactly the class this repo keeps being burned by: fully
placed, fast, and numerically wrong only on the device.

But all of that evidence was **retrodictive** — every model the audit had been pointed at
already had a known outcome. This is the forward test. Predictions for a fixed candidate set
were committed before anything ran (`results/quant/shift_cut_forward_predictions.log`), and
the results are in `results/quant/shift_cut_forward_results.log`.

| Model | Prediction | NPU nodes | Correlation vs CPU | Max/peak |
|---|---|---|---|---|
| `sesr_m7_fp32_xint8` | **hazard, 9 of 9** | 50 / 52 | **0.99912** | 0.040 |
| `sesr_m7_nchw_xint8` | **hazard, 9 of 9** | 50 / 52 | **0.99913** | 0.041 |
| `test_sr_xint8` | clean, 0 of 2 | 8 / 17 | 1.00000 | 0.008 |
| `yolov8n_cut_xint8_c32` | clean, 0 of 177 | 922 / 929 | 0.880–0.979 | ≤ 0.543 |

**The boldest prediction is refuted, twice.** Both SESR artifacts were called infeasible on
every one of their nine operations. Both place 50 of 52 nodes on the NPU, the two exceptions
being a quantize/dequantize pair rather than compute, and both track their own CPU reference
at a correlation of 0.999 with a worst-case deviation of 4% of peak on a smooth pixel
mapping. That is ordinary requantization divergence, not a requantizer that cannot represent
its scales.

**A third case was already inside the audit's own evidence.** Its committed log shows
`sesr_m7_xint8.onnx` at 9 of 9 violations. That artifact's measured NPU quality is **34.06 dB
PSNR on Set5** against a 35.64 dB float reference, rising to 35.16 dB with AdaRound. So three
SESR artifacts are called 100% hardware-infeasible and all three work. The rule fires on the
whole architecture family, and the commit that introduced it lists ResNet-50, YOLOv8n,
YOLOv8n-pose and FastDepth as sitting cleanly without mentioning this row.

**The clean direction is not settled here, and the metric is why.** `test_sr_xint8` agrees to
a correlation of 1.00000 but reaches the NPU with only 8 of 17 nodes, so it exercises little
of the DPU. `yolov8n_cut_xint8_c32` places 922 of 929 and still shows 0.880–0.979 on its raw
head tensors — yet this repo's own known-good yolov8n-cut reports **byte-identical decoded
detections** between CPU and NPU. Raw-tensor correlation is too sensitive for a detection
head; it moves for reasons that never reach the task output. The metric is sound for
super-resolution, where the tensor *is* the output, which is where the decisive result sits.

**What this means for the quantizer.** The audit must not gate Ignition in its current form:
a rule that rejects an entire working architecture family would silently discard good
artifacts, which is worse than missing bad ones. It does **not** show the bound is wrong as
physics — the multiplier width and shift window are read from the hardware — only that the
classifier built on them has a false-positive mode of the widest possible kind. Its
BiSeNetV2 and RegNetX-002 flags are undisturbed by this and remain retrodictive.

**What this does not show.** One seeded input per model, one real image for the detection
model; an output-agreement probe, not a dataset evaluation. `realesrgan_compact_r64_xint8`
was in the prediction set and is reported as discarded rather than quoted: its run fed a
0–255 input to a model that `npu/realesrgan.py` scales to 0–1, saturating both paths, and it
places only 12 of 248 nodes. Every other model in the prediction set is still unrun and those
predictions stand as recorded.

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
layers one at a time in the order Quark's loop visits them (its quantized file's node
order, which is ORT's `topological_sort` of the pre-processed float model, not the
export's file order: on this graph the downsample Conv precedes conv2 of its block; the
first run took Ignition's own file order, which coincides on ResNet, and the
[YOLO AdaRound section](#ignition-yolov8n-cut-adaround-parity) below records where it does
not), each seeing the rounding
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

What remains open on YOLO: a calibration that fires the shift rules, and traversal-order
equivalence where refinement rules interact, which this graph does not test. AdaRound
on this base is the next section.

### Ignition: AdaRound gains an OptimDevice, and the cpu path does not move (2026-09-09, Desktop 2)

`results/quant/adaround_device_cpu_parity_20260909_desktop2.log`. **No GPU measurement —
none was possible here.** `resnet_env` carries `torch 2.4.1+cpu`, a CPU-only build with
neither CUDA nor HIP, and Desktop 2's GPU is a Radeon 780M iGPU; the RX 7900 XTX this path
targets is on **Desktop 1**. Nothing here says AdaRound is faster on a GPU, or runs on one.

What is measured is the precondition. `quant/adaround.py` now takes Quark's `OptimDevice`
for the Adam rounding loop (ORT activation extraction stays on the CPU, as `InferDevice`
still refuses to move), and the one serious risk that carried was silently perturbing the
CPU path — the path that is byte-identical to a fresh `XINT8_ADAROUND` oracle.

`tools/quant_adaround_device_checks.py` runs the transcription on a synthetic QDQ Conv from
`tools/quant_fixtures.py`, pointed by `--repo` at the pre-change code in the main checkout
at `311a672` and at the post-change code, and compares emitted int8 weights, per-layer
report figures and every transcribed log line. At **8 iterations / 4 images** and at
**200 iterations / 8 images** (which moves 16 of 36 weights, so the loop is genuinely
exercised) the two are **identical** — weights, recon metrics, changed-element counts and
trace. Ignition's own `ADAROUND` banner is excluded and named as such in the tool: it is
this project's telemetry, not transcribed Quark output, and it gained `optim_device` /
`byte_parity_path` fields deliberately.

Three fail-closed boundaries are checked and hold:

- `OptimDevice` off the CPU **raises** unless `AllowNonParityDevice` is set
  (CLI: `--device <dev> --accept-non-parity`). A GPU changes float reduction order in Adam
  and in the convolutions, so the weights will differ from the oracle in the low bits.
- A non-CPU `InferDevice` is still refused outright.
- An acknowledged device torch cannot reach raises `RuntimeError` naming the torch build,
  **instead of silently running on the CPU**. A "GPU run" that is quietly a CPU run is the
  same failure shape as a CPU run in an NPU costume (`npu.session.resolve_xclbin`), and it
  is refused the same way.

The module is built and alpha-initialised on the CPU whatever the device, then moved: Quark
draws twice from the torch RNG per layer in `reset_parameters`
(`tools/quant_adaround_rng_probe.py`), and the byte-parity result depends on that stream.
Per-layer QDQ constants and the staged activation tensors are placed on the device once,
before the loop, rather than rebuilt per call.

**Still open.** Whether a 7900 XTX shortens the 1000-iteration loop on a wide model is
unmeasured, and so is whether a torch build that can reach it exists on Windows — torch's
official ROCm wheels are Linux-only, leaving AMD's ROCm-on-Windows preview or
`torch-directml`, neither installed on any machine here. Installing one is a Desktop 1
decision. A GPU run will not be byte-identical to the oracle by construction; the report
records `byte_parity_path=false` and the run logs a warning. Compare accuracy, never bytes.

### Ignition: YOLOv8n-cut AdaRound parity

`python -m quant adaround` now takes a head-cut YOLOv8n base: the family comes from the
base's sidecar, the float reference is the export letterboxed through `CocoSource` on
the sidecar's own listing, equalized (zero patterns here) and prepared (Split to Slice)
in the order `quantize` applies them, which is Quark's pre-processed float model. The
Conv/Q/DQ/HardSigmoid graph needed nothing new in the module: every Conv output feeds
its `QuantizeLinear` directly, so no layer carries an activation and each subgraph ends
at the Conv output, as Quark's `find_end` decides.

What the graph did need was the layer order. Quark's loop walks its quantized file, and
that file keeps the original nodes in the order ORT's quantization `topological_sort`
gave the pre-processed float model (Quark sorts it before quantizing: nodes without an
input first, then the consumers of the sorted initializer and input names, then
breadth-first by output, consumers in file order). On folded ResNet that order puts the
downsample Conv before conv2 of its block, and Ignition's own emitted order happened to
agree, which is why the ResNet run above was bitwise without anyone deciding the
question. On yolov8n-cut the two diverge at the 39th conv: the export interleaves the
P3 head convs with the neck (`/model.22/cv2.0/cv2.0.1/conv/Conv` before
`/model.18/cv1/conv/Conv`), the vendor sort keeps that, and Ignition's emitted file does
not. `Graph.vendor_order` transcribes the vendor sort and `layer_targets` now walks the
float graph in that order; checked in-session against ORT's own routine on eight files
(both float exports, both c64 pairs, both repo AdaRound artifacts: identical node order
on all), against the ResNet log above (the 54 `ADAROUND_LAYER` names in the same
order, so that run needs no repeat) and against the fresh oracle's file (63 convs in
the same order). Layer order is a parity condition in its own right: each layer's
rounding is chosen against inputs that already carry the rounding of every layer
visited before it, and the torch generator advances per layer, so a different order
draws different batches.

Fresh same-listing oracle: [`XINT8_ADAROUND` on the sorted first 64 COCO calibration images](../results/quant/quant_yolov8n_cut_quark_cle_adaround_c64.log)
(`scripts/quant-reference.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --cle --adaround`):
0 CLE patterns, 63 modules, 16 early stops (every one at the last windowed check,
iteration 999, so each dropped a single final update rather than truncating a
schedule; ResNet's run above had none), 161.0 s of ONNX inference plus 361.6 s of
torch training, 696.6 s end to end including calibration, peak working set
5,393,625,088 bytes, SHA256 `505cf451…6d86`. Ignition on its CLE c64 artifact
([log](../results/quant/quant_yolov8n_cut_ignition_cle_adaround_c64.log),
`scripts/quant-adaround.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib`,
Quark import-blocked): 157.3 s of data plus 353.2 s of training, 512.0 s for the
finetune alone, peak working set 5,742,055,424 bytes, SHA256 `7de7e9f9…d535`. Both in
`resnet_env` (torch 2.4.1+cpu, 8 threads, ONNX Runtime 1.22.1), one after the other on
an otherwise idle box. Result ([diff](../results/quant/diff_yolov8n_cut_ignition_cle_adaround_c64.log)):
empty position delta; listing, preprocessing, float hash, CLE and every FastFinetune
parameter equal; **126/126 int8 initializers byte-identical**; refinement fixed point on
both. The two logs agree line for line: all 819 per-layer lines (63 module banners in
the same order, the loss lines, the 16 early-stop lines and 63 reconstruction metrics)
are identical to the last printed digit. AdaRound moved 1,177,194 of the 3,146,160
weight elements by exactly one LSB relative to the CLE base (37.42 percent), the same
count on both sides, biases untouched, and 90 of them sit at −128, Quark's dtype clamp;
the base had none there. The files hash differently, as the CLE pair did: the gate is
the graph and every parameter, not the serialization.

**Full-set evaluation.** `scripts/quant-validate.sh --family yolo`, all 5,000 val2017
images at conf 0.001, IoU 0.7, max_det 300, per-class NMS, decode outside "infer". CPU:
both files read **32.21 mAP@50-95 / 46.97 mAP@50** (small/medium/large 15.41 / 34.52 /
47.13; [reference](../results/quant/map_yolov8n_cut_ignition_cle_adaround_c64_reference_cpu.log),
[own](../results/quant/map_yolov8n_cut_ignition_cle_adaround_c64_own_cpu.log)), 614,151
detections each, detection files byte-identical in-session (60,003,158 bytes,
git-ignored). NPU, paired in one sitting with a clean `xrt-smi` context witness before
each model and `--fresh` compilation
([reference](../results/quant/map_yolov8n_cut_ignition_cle_adaround_c64_reference_npu.log),
[own](../results/quant/map_yolov8n_cut_ignition_cle_adaround_c64_own_npu.log),
[EP reports](../results/quant/diag_yolov8n_cut_ignition_cle_adaround_c64_own.log)): both
read **32.04 mAP@50-95 / 46.78 mAP@50** (14.81 / 34.29 / 46.37), 922 of 929 nodes on
the NPU with the same seven on CPU as the XINT8 pair (the two EP reports differ only in
their timing lines, checked in-session), identical EP reports for the pair, 611,937
detections each and byte-identical NPU detection files (59,796,355 bytes). Mean
`sess.run` at eval conf was 6.94 ms (reference) and 6.86 ms (own); the CPU means
(34.58 and 36.80 ms, medians 34.20 and 34.11 ms) differ by session noise, not by
model, since the files compute identically.

This pair is also the first like-for-like AdaRound toggle on yolov8n-cut. The repo's
earlier artifacts varied the calibration count with the algorithm (32 images plain, 300
with AdaRound, the caveat `scripts/yolo-bench.sh` carries), so their gap could not be
attributed. Here the same 64-image listing quantized plain reads 27.43 CPU / 27.03 NPU
([above](#ignition-yolov8n-cut-preparation-parity)) and with AdaRound 32.21 / 32.04:
AdaRound alone adds 4.78 points on CPU and 5.01 on the NPU, recovering 5.01 of the 9.66
points plain XINT8 loses against the 36.69 FP32 baseline, at the same placement and
the same latency band as the plain file. The repo's c300 AdaRound row (32.19 / 47.04)
is 0.15 points above this c64 pair on the NPU; that is a different calibration count in
a different session, so the two are not compared beyond noting that 64 images reach
within session noise of 300 here.

What remains open on YOLO AdaRound: it ran on the yolov8n-cut graph only (no Gemm, no
activation-bearing layer), the laptop stretch is unrun, and GPU finetune waits on
Desktop 1.

### Ignition: MODNet preparation and replay parity

MODNet is Ignition's third model family and the first that is not a plain feed-forward
convolutional stack. It brings four things neither gated export has: 35 `Clip(0,6)`
activations, 17 depthwise convolutions, a `GlobalAveragePool` over a 16x16 map rather
than 7x7, and `Resize` nodes with fractional scales, two of them fed straight from the
graph input. It is also the graph the
[preprocessing question](../RESEARCH.md) was about: its calibration reader was a copy
of the inference transform, and the two had silently diverged. `quant/sources.py`'s
`ModnetSource` calls `npu.modnet.preprocess`, so calibration and inference are the same
function rather than two copies of one.

Two gates ran on Desktop 2 (Ryzen 7 8700G, XDNA1 Phoenix) in `resnet_env`, both from
`models/modnet/modnet_cut_fp32.onnx` with CLE on, the vendor's default. Neither touches
hardware and neither calibrates: a difference here is attributable to graph preparation
or emission alone. Log: `results/quant/prepare_probe_modnet_cut_fp32.log`.

**Preparation.** Quark's whole `apply_pre_process` (the resolved static op types, the
hardware-compatibility conversions its `quantize()` forces on for `enable_npu_cnn`, and
its CLE) against Ignition's `simplify -> CLE -> prepare`, compared with `graph_diff`:
empty node delta, no initializer mismatch, `whole_pre_process_equal` true. Node names
match as sets and the order does not, exactly as on the two earlier families, because
Quark sorts with onnxruntime's `topological_sort` and Ignition with its own stable sort;
the position table is keyed by tensor name, so order is not part of this gate.

The isolated steps say where the work is on this graph. `onnxslim` takes the export from
230 nodes to 150 and 140 initializers to 145: it lowers all 79 `Constant` nodes into
initializers, ties the duplicates (the 35 `Clip` bound pairs collapse to one, six
`Resize` scale tensors to one), and removes one `Resize` as a common subexpression of
another with the same input and attributes. Quark's `optimize_model` passes are no-ops
here: BatchNorm folding and the hardware-compatibility conversions each report
`structure_equal` with no node or initializer change, so nothing in Quark's own optimizer
touches this graph and the whole preparation delta is onnxslim's plus CLE's.

**Ignition calls onnxslim rather than reimplementing it.** Quark delegates its
`SimplifyModel` step to the same library, so a transcription would be reproducing a
third-party optimizer rather than the vendor's XINT8 dialect, and the gate would still be
"matches onnxslim". This is a weaker dependency than Quark: `quant/`'s core still imports
with only numpy, onnx and onnxruntime in both environments, and only the MODNet family
needs onnxslim at run time (0.1.96, present in `resnet_env` and `resnet_env17`). It is
recorded as a decision in [`quant/DESIGN.md`](../quant/DESIGN.md), not as an oversight.
The two earlier families do not run the step: onnxslim is measured to be a structural
no-op on both exports, so adding it there would change nothing already gated.

**CLE fires on this graph, and only in pairs.** 9 patterns over 8 unique `Conv -> Conv`
pairs, every one at group 1 — the first measured graph where the transcribed CLE
actually moves weights (ResNet's 33 patterns did; the SiLU YOLO net had none). The
depthwise triple path that `quant/cle.py` raises on is *not* reached even though the
graph has 17 depthwise convolutions, because the vendor's triple matcher links only
through `Relu` and MODNet's inverted residuals are separated by `Clip`. Depthwise CLE
therefore remains unimplemented and unmeasured; this graph does not exercise it.

**Replay.** The committed `modnet_cut_xint8_calibfix.onnx` artifact's positions were read
back and re-emitted through Ignition's own emitter, then compared to the artifact:
**140 of 140 int8 initializers byte-identical**, empty node delta, no initializer
mismatch, and refinement a fixed point on both sides (zero moves on the reference, zero
on the replay, converged in one loop). 239 positions, 99 activation Q/DQ pairs emitted
and **52 pruned** where a Conv or Add feeds a single `Relu` or `Clip(0,6)`. The GAP
correction `Mul` was inserted with factor 1.0 for the 16x16 pool, which is what the
vendor's dyadic search returns for a 256-element window and what its own artifact
carries; the `Sigmoid` became `HardSigmoid` and picked up its `HARD_SIGMOID_SCALE` `Mul`.

**The rule this graph exposed.** `QDQDirect8BitOp` hands a `MaxPool` or `Resize` output
its input's quantization parameters only when that input is already marked at the moment
the node is visited, and otherwise marks neither input nor output, leaving the output to
be marked plainly by whichever consumer reaches it. Six of MODNet's eight `Resize` nodes
share, and the two fed by the graph input do not: the vendor's artifact gives them their
own scales, 0.25 and 0.0625, where the graph input's is 0.25. Ignition had assumed
sharing was unconditional, which is invisible on ResNet and yolov8n-cut because every
pooling and resize input there is produced by an earlier quantized node. `quant/qdq.py`
now walks the marking pass in `Graph.vendor_order`, the order onnxruntime's
`quantize_model` visits nodes, and applies the rule positionally. That order was verified
against onnxruntime's own `topological_sort` node for node on both **slimmed** MODNet
exports, which matters because simplification reorders this graph — the raw export's order
is not the vendor's. Ten files now agree on `vendor_order`. Re-running the two
earlier replays under the new rule reproduces them unchanged — ResNet 108/108 int8 exact
with 49 pruned, yolov8n-cut 126/126 with 0 pruned, both `PREPARE_PROBE_PASS True`
(`results/quant/prepare_probe_resnet50_fp32_vendor_order.log`,
`results/quant/prepare_probe_yolov8n_cut_vendor_order.log`).

Not measured in this probe: independent calibration of this graph, and any hardware run of
an Ignition-produced MODNet file. Replay proves the emitter and the refinement, not the
calibrator. That gate follows in
[independent calibration](#ignition-modnet-independent-calibration-and-paired-matte-evaluation).

### Ignition: MODNet independent calibration and paired matte evaluation

The [preparation gate](#ignition-modnet-preparation-and-replay-parity) proved the emitter
and the refinement on this graph by replaying a committed artifact's positions. It could
not prove the calibrator. This closes that: Ignition chose every position itself, from the
float export, with Quark and torch blocked, against a fresh Quark oracle built from the
same float file, the same 64-image listing and the same `include_cle=True` default preset.
Both ran on Desktop 2 (Ryzen 7 8700G, XDNA1 Phoenix) in one sitting, **sequentially**, so
their wall times and peak memory are comparable to each other.

Producer logs: `results/quant/quant_modnet_cut_quark_cle_c64.log` and
`results/quant/quant_modnet_cut_ignition_cle_c64.log`. Comparison:
`results/quant/diff_modnet_cut_ignition_cle_c64.log`.

**The graph comparison is exact.** Empty position delta, **140 of 140 int8 initializers
byte-identical**, no node delta, and `SAME_FLOAT_PREPROCESS_LISTING_CLE True` binding both
sides to the same float SHA256, the same preprocessing record and the same 64 filenames.
The two files still hash differently (`7d012cbe` against `71eb3f5f`): the gate is the
graph and its parameters, never the serialization, as on the two earlier families.

Refinement moved 9 positions on this graph and converged in 3 loops: 8 `align_concat` and
1 `align_pool`, the latter on the SE block's `GlobalAveragePool`. For contrast the same
transcription needs 2 loops elsewhere and fires one rule each — ResNet 1 `align_pool`,
yolov8n-cut 16 `align_concat` plus 4 `align_slice`. MODNet is the first measured graph
where two alignment rules interact across loops, which is the case
[`quant/DESIGN.md`](../quant/DESIGN.md) lists as the open traversal-order question; it
converges here, which is evidence for this graph and not a proof of the general case.

**Producer cost, same sitting, sequential:**

| | Quark oracle | Ignition |
|---|---|---|
| Wall | 488.8 s | 333.3 s |
| Peak working set | 15,817,965,568 B | 13,879,128,064 B |
| Activation store | 18.55 GB cached, 151 tensors | 12,714,516,480 B spooled, 99 tensors |
| Environment | `resnet_env`, onnx 1.19.0, onnxruntime 1.22.1 | `resnet_env17`, onnx 1.18.0, onnxruntime 1.23.3.dev |

Two caveats on that table. The environments differ by design — the oracle needs Quark,
which lives in `resnet_env`, and Ignition runs where the NPU runtime is — so the wall-time
and memory gap is across two onnx/onnxruntime versions as well as two producers, and is
not a like-for-like producer benchmark. And the two peak figures were taken differently:
Quark's is `psutil`'s in-process `peak_wset`, Ignition's was sampled from outside every
5 seconds, so it is a lower bound. The **store** row is the one clean comparison, and it
is the pruning effect this repo already
[measured on ResNet](#ignition-calibration-spool-without-the-pruned-pre-relu-tensors):
Ignition skips the 52 activations that feed a single `Relu` or `Clip(0,6)` and are pruned
at emission, spooling 99 tensors where the vendor's All mode caches all 151.

**Paired evaluation, 50 validation images, against the FP32 export**
(`pipelines/modnet/5_eval.py`, both NPU runs `--fresh`, both preceded by an `xrt-smi`
witness reading no hardware contexts):

| File | EP | MAD | SAD (1e3) | MSE | mean ms | P50 | P90 | Placement |
|---|---|---|---|---|---|---|---|---|
| Quark oracle | CPU | 0.17122 | 45.60 | 0.163283 | 103.40 | 101.18 | 115.30 | — |
| Ignition | CPU | 0.17122 | 45.60 | 0.163283 | 101.98 | 100.62 | 115.24 | — |
| Quark oracle | NPU | **0.19021** | 50.75 | 0.181764 | 26.69 | 26.46 | 27.12 | 502 / 507 |
| Ignition | NPU | **0.19021** | 50.75 | 0.181764 | 26.50 | 26.40 | 26.78 | 502 / 507 |

Every error figure is identical to five decimal places on both providers, which is what
byte-identical parameters should give. Placement is identical too, 502 of 507 nodes, the
five elsewhere being four boundary Q/DQ nodes and the initial 4x downsampling `Resize`
(`results/quant/diag_modnet_cut_ignition_cle_c64_{reference,own}.log`). The latency
differences between the two rows of a pair are session noise on a shared machine, not
model differences: the two files compute the same thing.

**Two things this table is not.** It is **not** comparable with the
[calibration-fix row](#5-opencv-calibration-fix-error-reduction-and-the-structural-zero-concat-gap-2026-09-08-desktop-2)'s
0.18629. That artifact has no committed quantize log and no recorded listing or count, so
the only honest statement is that a different, unrecorded listing produced a different
number; 0.19021 here is not a regression against it, and neither figure can be attributed
to the producer. Reading the two as a calibration-count effect would need a listing sweep
nobody has run. And the CPU/NPU gap within a single file — 0.17122 against 0.19021 on
byte-identical parameters — is the same finding the
[acceptance study](#ignition-controlled-resnet-qdq-acceptance) recorded on ResNet: a QDQ
file is not a bit-exact specification of what the DPU computes. MODNet is the third graph
to show it, and it means "parity" here continues to mean the same file as Quark, never a
claim about DPU arithmetic.

Still open on MODNet: AdaRound is not wired for this family; the Zero-Concat variant is
untouched, a second graph with its own cache key rather than a checkbox; and the listing
sweep that would let this row be compared with the calibration-fix row is unrun.

### Ignition: MODNet AdaRound parity

`python -m quant adaround` now takes a MODNet-Cut base. This is the first family in which a
finetuned layer carries an activation, so it is the first time the trained subgraph is
anything but Conv into Q/DQ: of the 71 Conv layers, 35 are followed by a three-input
`Clip(0, 6)`, 17 by a `Relu`, and 19 feed their `QuantizeLinear` directly. Quark's own
module banner ends the subgraph at the activation output rather than the Conv output
(`Module (input)->(/lr_branch/backbone/features.0/features.0.2/Clip_output_0)`), which is
what `layer_targets` reproduces.

The Clip is a fork, not a detail. `create_model_ops.py` rewrites a three-input Clip whose
bounds are initializers of the **quantized** model into attribute form and returns Quark's
own `Clip` module, whose forward is `torch.clamp`. Its `ActivationMapping` fallback for any
other Clip shape is `nn.ReLU6`, which draws the same curve but is a different module with a
different gradient at the bounds, and AdaRound trains through it. Ignition builds the
`torch.clamp` form and refuses anything else rather than approximating it. The oracle's log
carries no `Not supported activation node` warning, which is the evidence that the vendor
took the same branch on all 35.

The layer walk runs on the **simplified** float graph. `passes.simplify` (onnxslim, the
vendor's own SimplifyModel step) reorders this family's node list, so the adaround branch
calls `simplify_for` before CLE and `prepare`, in the order `quantize` applies them. This
was checked before the run rather than after: the 71 `(start, end)` pairs Ignition derives
are identical to the oracle's 71 banners, in the same order, so the layer-order condition
that had to be fixed for yolov8n-cut holds here unchanged.

Fresh same-listing oracle: [`XINT8_ADAROUND` on the sorted 64-image portrait listing](../results/quant/quant_modnet_cut_quark_cle_adaround_c64.log)
(`scripts/quant-reference.sh --in-model models/modnet/modnet_cut_fp32.onnx --calib-dir
data/modnet_calib --cfg-path models/modnet/preprocess_config.json --cle --adaround`): 9 CLE
patterns, 71 modules, 33 early stops, 2,881.1 s end to end including calibration and the
MinMSE search, peak working set 15,801,147,392 bytes, SHA256 `8ec93db8…1524d`. That run was
contended and was repeated on a quiet box; the timings here and below are **not** a producer
comparison, for the reasons set out after the diff. Ignition on
its CLE c64 artifact ([log](../results/quant/quant_modnet_cut_ignition_cle_adaround_c64.log),
`scripts/quant-adaround.sh --in-model models/modnet/modnet_cut_fp32.onnx --calib-dir
data/modnet_calib --cfg-path models/modnet/preprocess_config.json`, Quark import-blocked):
281.7 s of data plus 1,858.6 s of training, 2,140.9 s for the finetune, peak working set
22,299,271,168 bytes, SHA256 `a4ee1083…b267f`. Both in `resnet_env` (torch 2.4.1+cpu, 8
threads, ONNX Runtime 1.22.1).

Result ([diff](../results/quant/diff_modnet_cut_ignition_cle_adaround_c64.log)): empty
position delta; listing, preprocessing, float hash, CLE and every FastFinetune parameter
equal; **140/140 int8 initializers byte-identical**; refinement a fixed point on both. The
two logs agree line for line: all 911 per-layer lines — 71 module banners in the same
order, the loss lines and the 33 early stops — are identical to the last printed digit.
Against the CLE base, AdaRound moved 2,484,637 of the 6,441,120 weight elements by exactly
one LSB (38.57 percent), the same count on both sides, biases untouched (0 of 17,885), and
430 of them sit at −128, Quark's dtype clamp; the base had none there. The files hash
differently, as on the two earlier families: the gate is the graph and every parameter,
never the serialization.

**The oracle was re-run on a quiet box, and Quark's AdaRound is reproducible here.** The
first oracle shared the machine with several other sessions' producers, so a second run was
made on the same listing and seed
([log](../results/quant/quant_modnet_cut_quark_cle_adaround_c64_rerun.log)). It produced a
**byte-identical file** — SHA256 `8ec93db8…1524d` both times — with the same 71 modules, the
same 33 early stops, and per-layer logs identical to the first run's and to Ignition's, all
911 lines. The parity gate above is therefore against a reproducible reference rather than
one lucky run, and load reaches the cost of these runs but not their result.

**The wall times are not a comparison, in either direction.** Host-load witnesses were
sampled every 10 s through all three runs (`tools/host_load.ps1`), and each saw a different
machine:

| run | peer cores, mean | free RAM, min | finetune onnx + torch | peak working set |
|---|---|---|---|---|
| [oracle, first](../results/quant/load_modnet_cut_quark_cle_adaround_c64.log) | 2.96 | 7.6 GB | 313.8 + 2,004.6 = 2,318.4 s | 15,801,147,392 B |
| [oracle, re-run](../results/quant/load_modnet_cut_quark_cle_adaround_c64_rerun.log) | 0.68 | 9.1 GB | 266.8 + 1,798.3 = 2,065.1 s | 15,771,942,912 B |
| [Ignition](../results/quant/load_modnet_cut_ignition_cle_adaround_c64.log) | 2.39 | 1.7 GB | 281.7 + 1,858.6 = 2,140.3 s | 22,299,271,168 B |

Read in order, those rows are a warning about drawing a producer conclusion from any one of
them. Against the *first* oracle, Ignition looks 7.7 percent faster; against the *re-run* it
looks 3.6 percent slower; and the re-run itself ran with a third the background load
Ignition had, so neither figure isolates the producer. Quark's own end-to-end wall, which
also covers calibration and the MinMSE search, was 2,881.1 s contended against 2,562.0 s
clean — an 11 percent spread on identical work, which is the size of the effect being
mistaken for a producer difference. **No claim is made here about which producer is
faster**; settling it needs both run back to back under the same load, which has not been
done.

The **peak working sets** are the one row that does compare. Both Quark figures and
Ignition's are the same measurement (`psutil.Process().memory_info().peak_wset`, in-process,
unlike the externally sampled figure in the
[calibration section](#ignition-modnet-independent-calibration-and-paired-matte-evaluation)),
and the two Quark runs agree to 0.18 percent across very different memory pressure — 7.6 GB
free against 9.1 GB. That retires the hypothesis that the contended run's peak was trimmed
low by the OS, and leaves Ignition's 41 percent higher peak as a real difference in
allocation rather than an artifact.

**Full evaluation, 50 validation images against the FP32 export** (`scripts/quant-validate.sh
--family modnet`, both NPU runs `--fresh`, both preceded by an `xrt-smi` witness reading no
hardware contexts):

| File | EP | MAD | SAD (1e3) | MSE | mean ms | P50 | Placement |
|---|---|---|---|---|---|---|---|
| Quark oracle | CPU | 0.09361 | 24.48 | 0.084456 | 120.59 | 118.81 | — |
| Ignition | CPU | 0.09361 | 24.48 | 0.084456 | 119.00 | 113.06 | — |
| Quark oracle | NPU | **0.10072** | 26.71 | 0.091831 | 28.17 | 27.67 | 502 / 507 |
| Ignition | NPU | **0.10072** | 26.71 | 0.091831 | 27.76 | 27.49 | 502 / 507 |

Every error figure is identical to five decimals on both providers, and placement is
identical at 502 of 507 nodes — the five on CPU being two `QuantizeLinear`, two
`DequantizeLinear` and one `Resize` (`/hr_branch/Resize_1_output_0`). The two EP reports
differ in exactly one line, the generated name of a duplicated boundary DQ node
(`input_DequantizeLinear_Output/duplicated` against `…/duplicated_token_0`); node counts and
per-op device assignments are identical
([reports](../results/quant/diag_modnet_cut_ignition_cle_adaround_c64_own.log)). Latency
differences within a pair are session noise on a shared machine, not model differences.

**This is a like-for-like AdaRound toggle on MODNet.** The same 64-image listing quantized
plain reads 0.17122 CPU / 0.19021 NPU
([above](#ignition-modnet-independent-calibration-and-paired-matte-evaluation)); with
AdaRound it reads 0.09361 / 0.10072. AdaRound cuts the matte error by 45 percent on CPU and
47 percent on the NPU, at identical placement and in the same latency band. That is a larger
relative move than AdaRound produces on the other two families here, though MAD and mAP are
different metrics and the comparison is directional only. It does not close the CPU/NPU gap
within a single file (0.09361 against 0.10072), which remains the
[recurring finding](#ignition-controlled-resnet-qdq-acceptance) that a QDQ file is not a
bit-exact specification of what the DPU computes.

**A machine-routing consequence.** Ignition's MODNet AdaRound peaked at 22,299,271,168 bytes
and drove this 32 GB box down to 1.7 GB free. That puts it beyond the 16 GB laptop, which
SIGSEGVs without a traceback rather than raising under this load, and lowering the image
count does not help because the peak is at the full-resolution layers. Run this one on a
desktop.

Still open on MODNet AdaRound: the Zero-Concat variant is untouched, the calibration-listing
sweep that would let these rows be compared with the 2026-09-08 calibration-fix row is
unrun, and GPU FastFinetune (`--device`) waits on Desktop 1 as it does for the other
families.

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

### Category B, second candidate: BiSeNetV2 (Bilateral Segmentation Network)

`pipelines/bisenetv2/` — new pipeline, built against the official BiSeNetV2 architecture (Yu et al., IJCV 2021)
with Cityscapes 19-class weights (`models/model_final_v2_city.pth`).
Tests Category B's bilateral segmentation hypothesis: separate wide shallow Detail Branch (preserving $256 \times 256$
spatial detail at 64–128 channels) and deep narrow Semantic Branch (downsampling to $16 \times 16$ at 128 channels with
Gather-and-Expansion and Context Embedding blocks), fused by Bilateral Guided Aggregation (BGA) with HardSigmoid gating
and upsampled to full resolution ($512 \times 512$).

Exported to `models/bisenetv2_fp32.onnx` (nearest-neighbor head upsample, 118 nodes) and `models/bisenetv2_bilinear_fp32.onnx`
(stock bilinear head upsample, 118 nodes; static batch 1, input shape `[1, 3, 512, 512]`, output shape `[1, 19, 512, 512]`).
Preprocessing is byte-identical between calibration and inference via `npu/bisenetv2.py` (cv2-only, ImageNet mean/std
normalization `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`, `cv2.INTER_LINEAR` resize).

#### Compiler placement and DPU fusion

Quantized to Quark XINT8 with 300 calibration images (`data/bisenetv2_calib/`, `results/quant_bisenetv2_xint8.log`):
396 nodes in the quantized ONNX graph.

The VitisAI EP accepts **402 of 404 nodes (99.5%) on NPU** (`results/diag_bisenetv2_xint8.log`), compiling into
**exactly 1 monolithic DPU subgraph** (`subgraphStat: [{'device': 'DPU', 'count': 1}]`). Only the outer input
`QuantizeLinear` and output `DequantizeLinear` boundaries execute on CPU:
- All 57 Convolutions execute natively on AIE.
- All 40 `Relu` activations execute natively on AIE.
- All 10 `Add` and 5 `Mul` nodes execute natively on AIE.
- Both BGA gating activations compile natively to AIE: Quark's `enable_npu_cnn` detects `left * sigmoid(right)`
  and automatically lowers `Sigmoid` to `HardSigmoid` with DPU-compatible alpha (`alpha=0.166667`).
- All 3 nearest-neighbor `Resize` layers compile natively on AIE with zero internal CPU fallbacks.
- The `StemBlock` MaxPool and Concat, `CEBlock` GlobalAveragePool, and BGA `AveragePool` all compile natively on AIE.

**Bilinear vs. Nearest Head Ablation:**
In `models/bisenetv2_bilinear_fp32_xint8.onnx` (`results/diag_bisenetv2_bilinear_xint8.log`), the final 8× upsampling
Resize in `SegmentHead` uses stock `mode='linear'`. The VitisAI EP rejects this node to CPU (399/404 on NPU, 1 CPU Resize),
adding 0.26 ms of host dispatch latency (13.38 ms vs 13.12 ms). Converting the head upsample to nearest-neighbor
fuses all 3 Resize nodes directly into the monolithic DPU engine.

#### Tri-Hardware Performance Comparison

Measured on Desktop 2 (Ryzen 7 8700G, Radeon 780M, Phoenix XDNA1 NPU, 50 iterations, batch 1, 512×512,
`sess.run` only, `models/bisenetv2_fp32.onnx` vs `models/bisenetv2_fp32_xint8.onnx`):

| Hardware / Provider | Precision | Subgraphs | Latency (mean) | Latency (median) | Throughput | Backing Log |
|---|---|---|---|---|---|---|
| CPU (Zen 4, 8C/16T) | FP32 | 1 (CPU) | 58.07 ms | 58.16 ms | 17.2 fps | `results/lat_bisenetv2_cpu.log` |
| iGPU (Radeon 780M, DirectML) | FP32 | 1 (DML) | 14.25 ms | 12.91 ms | 70.2 fps | `results/lat_bisenetv2_dml.log` |
| **NPU (Phoenix XDNA1, nearest)** | **XINT8** | **1 (DPU)** | **13.12 ms** | **13.04 ms** | **76.2 fps** | `results/lat_bisenetv2_xint8_npu.log` |
| NPU (Phoenix XDNA1, bilinear) | XINT8 | 1 DPU + 1 CPU | 13.38 ms | 13.09 ms | 74.7 fps | `results/lat_bisenetv2_bilinear_xint8_npu.log` |

**Findings:**
1. **NPU beats both Zen 4 CPU and Radeon 780M iGPU**: At **13.12 ms (76.2 fps)**, BiSeNetV2 on Phoenix XDNA1
   is **4.43× faster than 8-core Zen 4 CPU** (58.07 ms) and **1.09× faster than Radeon 780M iGPU DirectML FP32** (14.25 ms mean).
   This establishes BiSeNetV2 alongside SESR-M7, FastDepth, and Real-ESRGAN 128² as vision workloads where the NPU outpaces the integrated GPU.
2. **2.17× faster than MODNet Cut**: BiSeNetV2 runs in 13.12 ms vs MODNet Cut's 28.45 ms (27.51 ms calibfix)
   at the same 512×512 resolution, delivering over 76 full frames per second of dense multi-class segmentation.

#### Quantitative Segmentation Fidelity Evaluation

Evaluated across 50 validation scenes (`data/bisenetv2_val/`) against the FP32 reference model running on CPU:

| Metric | CPU XINT8 | NPU XINT8 | Delta (NPU vs CPU) | Backing Log |
|---|---|---|---|---|
| Pixel Accuracy | **59.47% +/- 9.44%** | 15.33% +/- 10.82% | -44.14% | `results/eval_bisenetv2_xint8_cpu.log` / `results/eval_bisenetv2_xint8_npu.log` |
| Mean IoU (mIoU) | **25.72% +/- 6.96%** | 2.44% +/- 1.28% | -23.28% | `results/eval_bisenetv2_xint8_cpu.log` / `results/eval_bisenetv2_xint8_npu.log` |
| Softmax Prob MAD | **0.00357** | 0.01476 | +0.01119 | `results/eval_bisenetv2_xint8_cpu.log` / `results/eval_bisenetv2_xint8_npu.log` |
| Softmax Prob RMSE | **0.00464** | 0.01924 | +0.01460 | `results/eval_bisenetv2_xint8_cpu.log` / `results/eval_bisenetv2_xint8_npu.log` |
| Evaluation Latency (infer) | 99.01 ms | **13.01 ms** | -86.00 ms (7.61× faster) | `results/eval_bisenetv2_xint8_cpu.log` / `results/eval_bisenetv2_xint8_npu.log` |

**Bilateral Gating Fixed-Point Distortion Diagnosis:**
- Under floating-point QDQ simulation on CPU, XINT8 preserves segmentation structure cleanly (59.47% pixel accuracy,
  0.00357 probability MAD, and 77.86% agreement on single scenes with 0.9036 correlation).
- On physical DPU hardware, elementwise tensor multiplication between the Detail Branch and the HardSigmoid-gated
  Semantic Branch (`left * HardSigmoid(right)`) suffers fixed-point dynamic range truncation. Because the two branches
  span divergent activation scales, the fixed-point product attenuates minority classes (e.g. vehicles drop from 95.6k pixels
  to 128 pixels, while stationary background classes dominate).
- This confirms the Category B falsification hypothesis: multi-branch bilateral aggregation requires fine-tuning
  (or AdaRound scale optimization) to balance inter-branch power-of-two scale multipliers on physical systolic hardware,
  even though pure DPU compilation and speed (13.12 ms, 76.2 fps) are flawless.

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

> **Superseded on 2026-09-09, in two ways.** The figures below are a 200-image slice; the full
> 1,000-image set reads 69.50% FP32 and 0.10% XINT8. And the root cause named in this section --
> a hardware shift-cut bound that RegNetX inherently violates -- is wrong: the same graph, the
> same producer and the same calibration listing **without CLE** produce positions 2..10, no
> shift-cut adjustment at all, and 66.20% top-1. See
> [the collapse is CLE](#regnetx-002-and-resnext-50-recovered-the-collapse-is-cle-not-a-hardware-bound-2026-09-09-desktop-2).
> The measurements here stand; the attribution does not.

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
- **The DPU shift-cut clamp mechanism** — *observation stands, attribution retracted
  2026-09-09*:
  During compilation Quark logs `Shift cut of layer onnx::Conv_418 exceeds range [0, 16] (131). Modify wpos from 7 to -108.`
  Quark modifies weight positions by 100+ powers of 2; since scale is 2^-pos, shifting `wpos`
  to -108 forces scale to 2^108, annihilating activation resolution. That much is reproducible.
  **What is wrong is calling it a property of the architecture.** The clamp fires only because
  cross-layer equalization has already inflated the per-channel ranges: the same graph, producer
  and calibration listing with **CLE off** logs *no* shift-cut adjustment at all and scores
  **66.20% top-1** against **69.50%** FP32, where the CLE-on artifact scores **0.10%**
  (`results/quant/quant_regnetx_002_ignition_nocle_c64.log`,
  `eval_regnetx_002_ignition_nocle_c64_cpu.log`, `eval_regnetx_002_fp32_full1000_cpu.log`,
  `eval_regnetx_002_quark_cle_full1000_cpu.log`). The corrected analyzer finds **zero sigma
  violations** in this graph (`results/quant/shift_cut_reaudit_20260909_desktop2.log`). Also
  note the `[0, 16]` here is Quark's own producer-side contract, not the `[0, 31]` hardware
  bound asserted elsewhere in this file — the two were being used interchangeably, and neither
  is measured.
- **Why AdaRound cannot rescue this**:
  As established in the MobileViT study, AdaRound optimizes ternary rounding {-1, 0, 1} over
  fixed quantization intervals Delta. It never changes Delta. When Delta has suffered
  floating-point scale explosion, integer rounding cannot recover the network.

#### RegNetX-002 and ResNeXt-50 recovered: the collapse is CLE, not a hardware bound (2026-09-09, Desktop 2)

Cross-layer equalization is on in the repo's default XINT8 preset, and on grouped architectures it
is what destroys them. Holding the producer, the graph and the 64-image calibration listing fixed
and changing only whether CLE runs moves RegNetX-002 from **0.10%** to **66.20%** top-1. Every row
below is the full 1,000-image labeled set with the model's own preprocessing config, CPU unless
stated, measured in one sitting on Desktop 2.

| RegNetX-002 | EP | Top-1 | Top-5 | Log |
|---|---|---|---|---|
| FP32 reference | CPU | **69.50%** | 88.60% | [log](../results/quant/eval_regnetx_002_fp32_full1000_cpu.log) |
| Quark, default preset (CLE) | CPU | **0.10%** | 0.60% | [log](../results/quant/eval_regnetx_002_quark_cle_full1000_cpu.log) |
| Quark, default preset + AdaRound | CPU | 0.30% | 0.70% | [log](../results/quant/eval_regnetx_002_quark_cle_adaround_full1000_cpu.log) |
| Quark, **no CLE** | CPU | **66.20%** | 86.70% | [log](../results/quant/eval_regnetx_002_quark_nocle_c64_cpu.log) |
| Ignition, **no CLE** | CPU | **66.20%** | 86.70% | [log](../results/quant/eval_regnetx_002_ignition_nocle_c64_cpu.log) |
| Ignition, **no CLE** | **NPU** | **66.40%** | 86.20% | [log](../results/quant/eval_regnetx_002_ignition_nocle_c64_npu.log) |

The recovered model is not slower and not worse placed: **324 of 326 nodes on the NPU**, the same
count the collapsed artifact reaches, with only the boundary Q/DQ pair on CPU
([placement](../results/quant/diag_regnetx_002_ignition_nocle_c64.log), device idle beforehand per
its [witness](../results/quant/contexts_regnetx_002_ignition_nocle_c64.log)), at 2.47 ms and about
405 images/s. Placement was never the problem, and this is another instance of the recurring
finding that a fully placed graph says nothing about whether it computes anything.

**The scale grids say it plainly.** All three artifacts have 151 scalar scales:

| Artifact | Scale range | Position range | Negative positions |
|---|---|---|---|
| Quark, default preset (CLE) | 9.40e-38 .. 1.32923e+36 | **-120 .. 123** | 17 |
| Quark, no CLE | 0.000976562 .. 0.25 | 2 .. 10 | 0 |
| Ignition, no CLE | 0.000976562 .. 0.25 | 2 .. 10 | 0 |

Without CLE nothing is out of range, so Quark's `Shift cut ... exceeds range [0, 16] (131). Modify
wpos from 7 to -108` never fires and Ignition's own transcribed `shift_cut` rule never fires either
-- its refinement converges in 2 loops having made a single `align_pool` change
([log](../results/quant/quant_regnetx_002_ignition_nocle_c64.log)). The shift-cut clamp was
reacting to a range CLE created, not enforcing a bound this network inherently hits.

**This is not Ignition beating Quark.** The two producers emit the *same file*: 90 of 90 int8
initializers byte-identical, empty node delta, `INT8_EXACT True` and `GRAPH_DIFF_PASS True`. What
that buys is a fourth family reproduced bit-for-bit at zero cost -- RegNetX-002's export is
Conv/Relu/Add/GlobalAveragePool/Flatten/Gemm and routes to `folded_resnet` with no new code -- and
an attribution, not an accuracy win.

The one behavioural difference is worth stating precisely, because it is smaller than it looks.
Asked for `--cle` on this graph Ignition **refuses and writes nothing** (`Depthwise CLE triples are
not implemented`), where Quark applies CLE and emits a plausible-looking 0.10% model. Failing
closed is the better outcome, but Ignition gets there because that CLE path is unimplemented, not
because it detected an instability. A designed guard -- measure the post-transform range and skip
or damp the pair -- is what would turn an accident into a capability, and it does not exist yet.

**It generalizes to the second grouped collapse.** ResNeXt-50 32x4d, same treatment, same 1,000
images:

| ResNeXt-50 32x4d | EP | Top-1 | Top-5 | Log |
|---|---|---|---|---|
| FP32 reference | CPU | **80.90%** | 93.80% | [log](../results/quant/eval_resnext50_32x4d_fp32_full1000_cpu.log) |
| Quark, default preset (CLE) | CPU | **0.10%** | 0.50% | [log](../results/quant/eval_resnext50_32x4d_quark_cle_full1000_cpu.log) |
| Quark, **no CLE** | CPU | **68.90%** | 88.20% | [log](../results/quant/eval_resnext50_32x4d_quark_nocle_c64_cpu.log) |

68.8 points recovered, though ResNeXt still gives up 12.0 points against FP32 where RegNetX gives
up 3.3 -- so grouped convolutions remain genuinely harder to quantize, and only the *catastrophe*
is CLE's doing.

**DenseNet-121, the third collapse, is untested here** and stays open. `scripts/quant-reference.sh`
sizes its disk guard through Ignition's `prepared_graph`, and that export carries unfolded
`BatchNormalization`, which `qdq.quantizable_tensors` rejects
(`ValueError: Unsupported float operator: :BatchNormalization`). The wrapper therefore cannot start
a run for a family Ignition does not yet parse, which is a limitation of the harness rather than a
result about DenseNet.

**MobileViT is a different failure and is not addressed by this.** Its discriminator is a depthwise
weight-scale grid reaching delta = 1.0, measured independently, and it has only two equalized pairs
to begin with. Nothing here suggests turning CLE off would help it.

---

### BiSeNetV2's CPU/NPU gap, localised to one Mul (2026-09-09, Desktop 2)

BiSeNetV2 is the repo's largest divergence between providers on a byte-identical file, and it
reproduces exactly: 50 images, same artifact, same sitting
([CPU](../results/quant/bga_bisenetv2_xint8_cpu_desktop2_20260909.log),
[NPU](../results/quant/bga_bisenetv2_xint8_npu_desktop2_20260909.log)).

| | Pixel accuracy | mIoU | Softmax MAD | Softmax RMSE | Infer |
|---|---|---|---|---|---|
| CPU | 59.47% | 25.72% | 0.00357 | 0.00464 | 99.80 ms |
| NPU | 15.33% | 2.44% | **0.01476** | **0.01924** | 13.06 ms |

**Read the right number.** BiSeNetV2 is a Cityscapes model scored on 50 COCO scenes against a
reference session pinned to CPU whatever `--ep` says, so 59.47% and 15.33% are *not* segmentation
accuracy -- they are two models disagreeing on out-of-domain images. What is valid is the
**divergence between providers on identical parameters**, and the softmax MAD/RMSE, which need no
ground truth. The NPU's softmax error is **4.13x** the CPU's.

**It is not a gain error.** Fitting `npu = k * cpu` on unsaturated elements gives k = 0.73, 0.60,
0.61 across three images -- not a power of two -- with correlation **0.34 to 0.37**, and removing
the fitted gain barely moves the residual (0.267 against 0.273). The NPU output is largely
*decorrelated* from the CPU's, which is a stronger failure than a misplaced requantisation shift.

**It is not localised in space or class either.** The worst 10% of pixels carry only 13-14% of the
total error, against 10% for a perfectly uniform spread, and the worst class is only 1.8-3.0x the
median class. The NPU output does sit on the full output grid -- 256 distinct values pinned at
[-1.000, 0.992] with 0.27-0.63% of elements at the rails -- where the CPU uses 175-217 values and
0.003% at the rails.

**Where it enters.** Cutting the graph at each `DequantizeLinear` and running the sub-model on both
providers ([log](../results/quant/bga_tail_scan_desktop2_20260909.log)) points at `/bga/Mul`:

> **The "single node" reading below is superseded (same day).** Cuts 181 and 182 differ by **60
> nodes**, not one -- cut 182 adds the whole `left1` detail branch along with the Mul -- so this
> scan never isolated the operator. Measured separately, the Mul is not at fault at all:
> [the Mul is not the fault](#the-mul-is-not-the-fault-a-long-lived-activation-is-2026-09-09-desktop-2). The correlations in the table are correct; the
> attribution is not.

| Cut | Tensor | Correlation | Mean abs diff | Placed |
|---|---|---|---|---|
| 180 | `/bga/right2/right2.2/Conv` | 0.9913 | 0.041 | 285/287 |
| 181 | `/bga/Sigmoid_output_0` (gate) | 0.9858 | 0.019 | 288/290 |
| **182** | **`/bga/Mul_output_0`** | **0.6869** | **0.640** | 349/351 |
| 183 | `/bga/Sigmoid_1_output_0` (sibling gate) | 0.9904 | 0.007 | 289/291 |
| **184** | **`/bga/Mul_1_output_0`** (sibling) | **0.9984** | 0.036 | 350/352 |
| 185 | `/bga/up2/Resize` | 0.9984 | 0.036 | 353/355 |
| 186 | `/bga/Add` (the two branches meet) | 0.8516 | 0.646 | 382/384 |
| 187 | `/bga/conv/conv.2/Relu` | 0.6462 | 1.703 | 388/390 |
| 188 | `/head/conv/relu` | 0.2552 | 0.324 | 394/396 |
| 190 | `logits` | 0.3455 | 0.273 | 402/404 |

Both of `/bga/Mul`'s inputs arrive correlated at 0.986 and above. Its output is 0.687. Its sibling
gate `/bga/Mul_1`, the same operation in the same block, goes 0.990 into 0.998 and loses nothing.
Everything after cut 186, where the two branches are added, is downstream amplification. **What
this does not show, contrary to the first reading, is that the Mul is responsible** -- see below.

**This does not support Theorem 2 as stated.** The claim is that divergent power-of-two scales on a
multi-branch elementwise operation truncate dynamic range. But the sibling has the *larger*
divergence and is the one that works:

| | Feature branch | Gate branch | Position gap | sigma | Shape | Correlation out |
|---|---|---|---|---|---|---|
| `/bga/Mul` | pos 3, from Conv | pos 7, HardSigmoid | **4** | 20 | `[1,128,64,64]` | **0.687** |
| `/bga/Mul_1` | pos 1, from AveragePool | pos 7, HardSigmoid | **6** | 18 | `[1,128,16,16]` | 0.998 |

Both sigmas are inside [0, 31], so this is not a shift-cut hazard either -- consistent with the
audit finding no violation in this model.

**What is left standing** is a structural difference the scan cannot decide between: the failing Mul
is **16x larger spatially** (524,288 elements against 32,768, at the same 128 channels), and its
feature branch sits at position 3 rather than 1. Which of those is the trigger -- or whether it is
the `Conv` versus `AveragePool` provenance of the feature branch -- is not established here. A
minimal two-input Mul fixture, varying one of those at a time the way the XINT8 arithmetic probe
does for Conv, is what would decide it, and it has not been run.

Method notes worth keeping. A binary search over cut depth was tried first and its assumption is
false: divergence is **not monotonic** in depth -- cut 145 reads 0.9731 while cuts 148 and 154 read
0.9859 and 0.9889 -- so the "first divergent cut" it returned was a threshold crossing, not a
culprit, and the linear scan above replaces it. Sub-models also have to be topologically sorted
before `onnx.utils.extract_model` will accept them, because `passes.avgpool_dpu_scale` appends its
`Mul` after the consumer; `Graph.topo_sort` is the repo's own fix and changes node order only.
Every sub-model's placement is reported above because one that fell back to CPU would be comparing
CPU against CPU and would look perfect.

---

### The Mul is not the fault, a long-lived activation is (2026-09-09, Desktop 2)

Four measurements, each holding everything else fixed, take `/bga/Mul` off the hook.

**The operator is exact in isolation.** A minimal QDQ graph -- two quantised inputs, one
elementwise `Mul`, one quantised output -- was built at the failing node's exact shape and
positions and walked toward the working sibling's
([log](../results/quant/mul_fixture_desktop2_20260909.log)). Every configuration is perfect:

| Fixture | C | HxW | positions | Correlation | Mean abs diff | Placed |
|---|---|---|---|---|---|---|
| the failing node's configuration | 128 | 64 | 3 x 7 -> 4 | **1.0000** | 0.00122 | 4/7 |
| spatial 32 | 128 | 32 | 3 x 7 -> 4 | 1.0000 | 0.00124 | 4/7 |
| spatial 16 | 128 | 16 | 3 x 7 -> 4 | 1.0000 | 0.00123 | 4/7 |
| feature position 1 | 128 | 64 | 1 x 7 -> 4 | 1.0000 | 0.00151 | 4/7 |
| the working sibling's configuration | 128 | 16 | 1 x 7 -> 4 | 1.0000 | 0.00150 | 4/7 |
| 32 channels | 32 | 64 | 3 x 7 -> 4 | 1.0000 | 0.00124 | 4/7 |

Neither the 64x64 spatial size nor the position gap of 4 breaks a Mul on its own.

**Both operands are clean measured separately**
([log](../results/quant/bga_branches_desktop2_20260909.log)). The detail branch that feeds the Mul
reads 0.9979 at its own output, and the gate reads 0.9858 -- yet their product reads 0.6869.

| Cut | Tensor | Correlation |
|---|---|---|
| 133 | `/bga/left1/left1.0/Conv` | 0.9929 |
| 136 | `/bga/left1/left1.2/Conv` (the Mul's feature input) | **0.9979** |
| 171 | `/bga/left2/left2.2/AveragePool` (sibling's feature input) | 0.9987 |
| 182 | `/bga/Mul_output_0` | **0.6869** |
| 184 | `/bga/Mul_1_output_0` | 0.9984 |

**Scale choice does not fix it.** Walking the feature branch's position from 3 to 7 raises the
correlation from 0.687 to 0.787, which looks like progress until the signal is accounted for: the
CPU reference's own standard deviation falls from 0.6200 to 0.2432 over the same walk, because a
finer scale simply clips the feature. Normalised, the error is flat at essentially 100% of the
signal throughout -- 1.03, 1.03, 1.07, 0.99, 1.00
([log](../results/quant/mul_repair_desktop2_20260909.log)). There is no position that helps, which
also means this is not something the quantiser can choose its way out of.

**The deep branch is what breaks it.** Rebuilding the same Mul with the same `left1` branch, but
feeding the gate in as a graph input so the ~280-node semantic branch is never compiled
([log](../results/quant/mul_context_desktop2_20260909.log)):

| Sub-model | Nodes | Correlation | Mean abs diff | CPU std | Normalised error |
|---|---|---|---|---|---|
| full cut 182 | 343 | 0.6869 | 0.63986 | 0.6200 | **1.032** |
| gate supplied as an input | 62 | **0.9981** | 0.01814 | 0.6200 | **0.029** |

The CPU reference is identical in both -- same standard deviation to four decimals -- so the
arithmetic being asked for is the same. Only the compilation differs, and the error falls by 35x.

**What this points at.** `/bga/left1/left1.2/Conv`'s output is `[1,128,64,64]`, **512 KB** at int8,
and it has to stay live from early in the network until the semantic branch has finished so the two
can be multiplied. The sibling's equivalent is `[1,128,16,16]`, **32 KB**, sixteen times smaller,
crosses the same span, and is unaffected. AIE tile local memory is 64 KB per core. A 512 KB
activation held across ~280 nodes of unrelated computation is the one thing the failing case has
that none of the working cases do.

**Not established.** That the mechanism is specifically a spill or an overwrite, which nothing here
observes directly; the exact live-size threshold, since only 512 KB and 32 KB were measured; whether
other models carry a long-lived activation of this size without a visible problem. This is a
compiler and allocator behaviour, so the ordinary levers -- calibration, scales, AdaRound -- have
nothing to act on. The fix directions that follow from it are structural: shrink the long-lived
tensor, or force a subgraph boundary so it round-trips through DDR instead of living on-chip.
Neither is measured here, and the second trades latency for correctness.

---

### The CLE stability guard: what to threshold on, and what it costs (2026-09-09, Desktop 2)

Cross-layer equalization destroys RegNetX-002 and ResNeXt-50 and is worth 10.8 top-1 points on
ResNet50, so the useful question is what separates the two cases. The obvious candidate is wrong.

**It is not the weight range.** Applying Quark's own `cle_transforms` to the float exports and
measuring every Conv weight before and after, CLE **narrows** the power-of-two positions on both
models it destroys -- RegNetX-002 from 5..10 into 6..8, ResNeXt-50 from 4..9 into 5..8, the largest
single move being three positions
([log](../results/quant/cle_pair_spans_desktop2_20260909.log) covers the pair analysis; the range
scan is `scratch/cle_ranges.py`). That is CLE working as designed, and it is why this was never
caught by looking at weights: the transform multiplies the head by a per-channel vector `s` and the
tail by its reciprocal, so weight ranges stay tidy no matter how extreme `s` gets.

**It is the per-channel scale itself.** What nothing rescales is the activation *between* the two
layers, which is multiplied by `s` and then calibrated. Measuring the widest power-of-two excursion
in `s` per equalized pair, on the same axis for all four models:

> **Superseded the same day.** These spans were computed by reading Quark's pattern list as
> pairs, but 14 of RegNetX-002's 27 patterns and 17 of ResNeXt-50's 33 are **four-element
> depthwise triples**, and applying the pair formula to them gives the wrong number. Measured
> per pattern with the triple path implemented, the worst is **42.20 bits** on RegNetX-002 and
> **72.62** on ResNeXt-50, and a single threshold does not separate the populations at all:
> [depthwise triples and the narrowed guard](#depthwise-cle-triples-implemented-and-the-guard-narrowed-to-them-2026-09-09-desktop-2).

| Model | CLE's effect | Pairs | Worst pair | Median | Pairs over 4 bits |
|---|---|---|---|---|---|
| ResNet50 | **helps**, +10.8 top-1 | 33 | **2.52 bits** | 1.63 | **0** |
| MODNet-Cut | used by default | 9 | **1.81 bits** | 0.97 | **0** |
| RegNetX-002 | **destroys**, 66.20 -> 0.10 | 27 | **17.11 bits** | 1.58 | **9** |
| ResNeXt-50 32x4d | **destroys**, 68.90 -> 0.10 | 33 | **33.21 bits** | 1.95 | **4** |

The medians are almost identical across all four -- 1.63, 0.97, 1.58, 1.95 -- so this is not a
global difference in how equalizable the architectures are. It is a handful of extreme pairs, which
is exactly what a per-pair test catches. A scale span of 33 bits means an activation multiplied by
about 8.6 billion before anything measures its range, and `quant/calib.py` clamps such a range to
`float32.max / 2`, which lands at position -120 -- the number the RegNetX audit reports.

**The guard.** `quant/cle.py` now takes `max_scale_log2`, exposed as `python -m quant quantize
--cle-guard BITS` and `scripts/quant-own.sh --cle-guard BITS`. A pair whose span exceeds the
threshold is left untouched and recorded in `CleReport.skipped_unstable` with its span and channel
count. **It is off by default**, so the parity path is unchanged.

At **4 bits** it is free on every family Ignition supports -- 1.6x above the worst healthy pair and
4x below the damage:

| Run | Guard | Result |
|---|---|---|
| ResNet50 `--cle`, replayed against the Quark oracle | off | `GRAPH_DIFF_PASS True` |
| ResNet50 `--cle` | 16 bits | `GRAPH_DIFF_PASS True` |
| ResNet50 `--cle` | 3 bits | `GRAPH_DIFF_PASS True` |
| ResNet50 `--cle` | **2 bits** | **`GRAPH_DIFF_PASS False`** -- fires as intended, 2 bits is below ResNet50's 2.52 |
| MODNet-Cut `--cle` | 4 bits | `GRAPH_DIFF_PASS True` |

The 2-bit row is the positive control: the guard does change the output when it fires, and names
the pairs it skipped ([logs](../results/quant/) under `cleguard_*`).

**What is not shown, and it is the important caveat.** The guard has **not** been demonstrated to
recover RegNetX-002 or ResNeXt-50, because Ignition cannot equalize either graph at all: both raise
`Depthwise CLE triples are not implemented` in `find_pairs`, and that path is unwritten. Their spans
above come from Quark's pattern matcher with Ignition's `_calc_scale` applied to the float weights,
which is the same arithmetic Ignition would use but is not the same thing as Ignition running it.
So this guard is a defence for graphs Ignition already supports, and the measured separation is the
argument for its threshold -- not an accuracy result. Closing that gap needs the depthwise triple
path implemented and gated against a fresh oracle first.

---

### Per-channel weight scales are rejected outright (2026-09-09, Desktop 2)

Per-channel weight quantization is named in two places as the remedy for MobileViT-XXS, and the
only evidence about it was a ResNet50 session build that reached 11,973,251,072 bytes and was
stopped by hand at 273.7 s **without ever returning a verdict**. That experiment asked the question
with 54 convolutions at once. Asked with one, a compile costs seconds.

The mutation is `quant/probe.py`'s own `weights_per_channel`: each scalar weight scale and zero
point is repeated across the output channels and the DequantizeLinear gains an `axis`. The integers
and the real grid are untouched, so a correct implementation must return exactly the per-tensor
result ([log](../results/quant/perchannel_verdict_desktop2_20260909.log), device idle per its
[witness](../results/quant/contexts_perchannel_desktop2_20260909.log)):

| Fixture | Channels | Per-tensor placed | Per-channel placed |
|---|---|---|---|
| 1 Conv | 64 | 5 / 7 | **0 / 7** |
| 2 Conv | 64 | 10 / 12 | **0 / 12** |
| 4 Conv | 64 | 20 / 22 | **0 / 22** |
| 8 Conv | 64 | 40 / 42 | **0 / 42** |
| 1 Conv | **1** | 5 / 7 | **5 / 7** |

**The EP takes nothing.** Not a partial placement, not a fallback of the affected Convs: the whole
graph goes to CPU, `deviceStat` reading `{'CPU': 22}` for the four-Conv case with all four Convs,
13 DequantizeLinear and 5 QuantizeLinear on CPU. Note this is plain `CPU`, not the `VITIS_EP_CPU`
that appears in the A8W8 and A16W8 fallbacks. The per-tensor baseline of the *same* fixture places
normally at every size.

**The last row is the control, and it changes what the verdict means.** With a single output
channel the scale vector has length 1 and only the `axis` attribute distinguishes it from the
per-tensor form -- and it places 5/7, with NPU output identical to per-tensor to the last bit. So
the rejection is not the `axis` attribute, and it is not a schema or parsing failure. It is also
not about the values differing, because every value in these fixtures is the *same repeated
scalar*. **What the EP rejects is a weight scale with more than one element, whatever it contains.**

Two supporting checks confirm the fixtures are honest. On CPU the per-tensor and per-channel models
agree exactly (`cpu_pt_vs_pc_max = 0`), which is the mutation's design. And the rejected models'
"NPU" output equals their CPU output exactly (`npu_vs_cpu_perchannel_max = 0`), which is the
signature of full fallback rather than a silent miscompute.

**Consequence.** MobileViT-XXS's collapse is driven by a depthwise weight-scale grid reaching
delta = 1.0, and per-channel scales are the standard fix for exactly that. They are not available
on this hardware through this EP at opset 17. The recorded verdict that a SiLU/GELU backbone
"requires QAT or per-channel scale support" therefore narrows to **QAT**, since the other half is
now measured to be unreachable. This also retires the open question of whether the compiler's
memory growth was hiding a usable feature: the growth was on the way to a rejection.

**Scope.** One fixture family, 3x3 Convs with equal input and output channels, batch 1, opset 17,
Ryzen AI 1.7.1 on Phoenix. Per-channel *activation* scales and per-channel *bias* were not tested,
and neither was a genuinely differing channel grid -- though a vector of identical values being
refused makes a differing one very unlikely to fare better.

---

### Depthwise CLE triples implemented, and the guard narrowed to them (2026-09-09, Desktop 2)

`find_pairs` used to raise `Depthwise CLE triples are not implemented`, so Ignition could not
equalize RegNetX-002 or ResNeXt-50 at all and the stability guard could not be tested against the
failure that motivated it. The triple path is now transcribed from
`_cle_set_with_depthwise_layers`: a Conv, a depthwise Conv and a pointwise Conv are balanced
together against the geometric mean of their per-channel maxima, with the first two biases scaled
and the third layer's left alone. That path takes none of the pair path's options -- the source
passes it no balance method, weight threshold, bias flag or threshold flag -- so `equalize_triple`
accepts none either.

**Matching the vendor required reproducing a bug in it.** `check_conv_layers_support` sets one
`conv_support` flag per node and, when a grouped Conv fails the depthwise test, breaks out of the
*attribute* loop rather than the node loop. The flag is then overwritten by the next node, so the
returned verdict is whichever the **last** node produced: a grouped head followed by a group-1 tail
is accepted. Ignition's reading was the strict one, and on RegNetX-002 that is the difference
between 13 matched pairs and none -- thirteen of the vendor's 27 patterns silently dropped.

| Model | Quark pairs / triples | Ignition pairs / triples |
|---|---|---|
| ResNet50 | 33 / 0 | 33 / 0 |
| RegNetX-002 | 13 / 14 | 13 / 14 |
| ResNeXt-50 32x4d | 16 / 17 | 16 / 17 |

The float-level probe agrees byte for byte on all three
([RegNetX](../results/quant/cle_probe_regnetx_002_fp32_triples.log),
[ResNeXt](../results/quant/cle_probe_resnext50_32x4d_fp32_triples.log),
[ResNet50](../results/quant/cle_probe_resnet50_fp32_triples.log)): 27 patterns from both sides on
RegNetX, `patterns_equal: true`, 65 changed initializers each, `byte_mismatches: {}`.

**With the triple path in place, Ignition refuses to emit these models at all.** Quantizing
RegNetX-002 with `--cle` stops at
`Nonfinite float16 calibration samples for /s1/b1/conv2/bn/act/Relu_output_0`
([log](../results/quant/quant_regnetx_002_ignition_cle.log)). That is the mechanism closing:
a per-channel scale spanning 42 bits multiplies the activation past float16's range, and
`quant/calib.py` asserts its samples are finite. Quark, which stores samples differently, emits a
plausible-looking model that scores 0.10%. This is fail-closed **by design** rather than by the
unimplemented-path accident recorded earlier.

**The correct spans, per pattern, with the triple formula where it belongs:**

| Model | Pairs | Triples |
|---|---|---|
| ResNet50 | 33, worst **2.36 bits**, median 1.63 | none |
| MODNet-Cut | 9, worst **1.81 bits**, median 0.93 | none |
| RegNetX-002 | 13, all skipped as unsupported heads | 14, worst **42.20 bits**, median **23.51** |
| ResNeXt-50 32x4d | 16, same | 17, worst **72.62 bits**, median 3.14 |

**A single threshold cannot work, and that is the finding.** ResNeXt-50's seventeen triples span
2.2, 2.4, 2.6, 2.7, 2.7, 2.8, 3.0, 3.1, 3.4, 3.4, then 32.7, 40.0, 40.7, 41.4, 43.9, 72.6, 72.6.
**Ten of them sit inside ResNet50's beneficial range**, whose worst pair is 2.36. Any cut low
enough to disarm those ten also disarms the pairs CLE is worth 10.8 points for. The guard
therefore applies to **triples only**, and pairs are never guarded.

**What that buys**, all on the full 1,000-image labeled set:

| Model | CLE | Guard | Top-1 | Log |
|---|---|---|---|---|
| RegNetX-002 | on | none | **refuses to emit** | [log](../results/quant/quant_regnetx_002_ignition_cle.log) |
| RegNetX-002 | on | 4 bits, 9 of 14 skipped | **25.60%** | [log](../results/quant/eval_regnetx_002_ignition_cle_guard4_cpu.log) |
| RegNetX-002 | on | **2 bits, 14 of 14** | **66.20%** | [log](../results/quant/eval_regnetx_002_ignition_cle_g2_cpu.log) |
| ResNeXt-50 | on | **2 bits, 17 of 17** | **68.90%** | [log](../results/quant/eval_resnext50_ignition_cle_g2_cpu.log) |
| ResNet50 | on | 2 bits | byte-identical to the oracle | [log](../results/quant/cleguard_rn50_g2_triplesonly.log) |
| MODNet-Cut | on | 2 bits | byte-identical to the oracle | [log](../results/quant/cleguard_modnet_g2_triplesonly.log) |

The 4-bit row is the useful counter-example: a looser threshold is **worse than either extreme**.
Letting 5 of RegNetX's 14 triples through leaves the chain partly rebalanced and reads 25.60%,
against 66.20% for skipping all of them and a refusal for skipping none.

**Be precise about the win.** At 2 bits the guarded result *equals* the no-CLE result on both
models -- 66.20% and 68.90% -- it does not beat it. What the guard buys is that `--cle` can stay on
as a default for the families it helps without destroying the ones it does not, and that the
decision is made per pattern from a measured quantity rather than per model by hand. The threshold
itself rests on two collapse models and two healthy ones; a family whose triples are genuinely
benign would be disarmed by it and nothing here has found one.

---

### AdaRound peak memory: six copies of the activation set down to two (2026-09-09, Desktop 2)

AdaRound is the one place Ignition was measured *worse* than Quark -- 22,299,271,168 bytes against
15,771,942,912 on MODNet-Cut, same `psutil peak_wset` on both sides. Reading the loop, each layer
held the activation set roughly six times over:

- the three per-image lists ORT returns (`q_inputs`, `f_inputs`, `f_outputs`),
- the three stacked arrays `_stack` builds from them, which is `np.array(list)` and therefore a
  full copy while the list is still live,
- plus torch views, which cost nothing because `torch.from_numpy(np.expand_dims(arr[i], 0))` does
  not copy.

Only two of those are read after the shape check: `q_input` feeds the module and `f_output` is the
target. `f_input` exists solely for that check -- `cfg.check()` pins `DropRatio` at 1, so the float
inputs are never fed to anything -- and the source's `inputs_f` list is built and never read.

Two changes, neither touching a value, an RNG draw or a log line:

1. Release `f_input` after the shape check and stop building `inputs_f`.
2. Collect activations straight into one preallocated buffer (`_Collector`) instead of appending to
   a list and copying it with `_stack`.

The second is what matters. Releasing the lists moves the steady state but not the high-water mark,
because at the moment `_stack` runs the list and its copy are both alive -- that *is* the peak.

**ResNet50 AdaRound, 54 layers, 64-image listing, host witness `CLEAR` for both new runs:**

| Build | Peak working set | Against baseline | Output SHA-256 | Weights moved |
|---|---|---|---|---|
| Baseline (as committed) | 3,055,075,328 | — | `bf605321…` | 8,954,279 |
| Release the lists only | 2,960,109,568 | **−3.1%** | `bf605321…` | 8,954,279 |
| **Collect into one buffer** | **2,543,427,584** | **−16.7%** | `bf605321…` | 8,954,279 |

**The output is bit-identical at every step** -- same file hash, same count of weights moved by one
LSB -- which is the point: this is an allocation change, not an algorithm change. `_Collector` is
checked against `_stack` directly and produces the same array, same dtype, same order.

Against Quark's own ResNet50 AdaRound peak of 3,582,218,240, Ignition moves from 0.85x to **0.71x**.

**Not verified where the regression actually is.** MODNet-Cut is the case where Ignition loses, and
it was deliberately not re-run: free memory on the box was 16.5 GB against a 22.3 GB baseline peak,
and this repo has already recorded that peak working set reads *low* under memory pressure because
the OS trims the set. Measuring it there and then would have produced a number that looks like an
improvement for the wrong reason. It needs a quiet box, and until it has one the 22.3 GB figure
stands unchallenged.

Wall times were 528 s, 568 s and 584 s in that order, but the baseline is from a different day and
each row is a single run, so that is not a controlled timing comparison and no cost is claimed from
it. What can be said is that the direction is the one to expect: filling a buffer element by element
does more Python-level work than one bulk `np.array`.

---

## Native Windows XRT driver latency and DPU microcode disassembly

A characterization of AMD's native Windows kernel driver (`amdxe.sys`) and userspace runtime (`pyxrt.pyd`, Python 3.13) on Desktop 2 (Ryzen 7 8700G, Phoenix XDNA1 NPU), measuring the driver floor, unified memory synchronization bandwidth, command submission overhead, and reverse-engineering the compiled DPU microcode transaction stream.

Backing logs:
- `results/aie/windows_xrt_driver_bench.log`: device initialization, BO allocation/map/sync, kernel argument binding, and runlist queuing.
- `results/aie/dpu_transaction_disasm.log`: binary disassembly of DPU instruction packets from compiled `.xmodel` archives.

### Driver and runtime initialization floor

One-time setup latency measured via native `pyxrt`:

| Operation | Latency | Target / Context | Log |
|---|---|---|---|
| Device Open (`pyxrt.device(0)`) | **61.69 ms** (61690.90 µs) | `amdxe.sys` adapter handle | `results/aie/windows_xrt_driver_bench.log` |
| XCLBIN UUID Registration | **3.20 ms** (3198.90 µs) | `fastdepthcachekey/4x4.xclbin` | `results/aie/windows_xrt_driver_bench.log` |
| Hardware Context Creation | **77.71 ms** (77712.90 µs) | `pyxrt.hw_context` on device 0 | `results/aie/windows_xrt_driver_bench.log` |
| Kernel Instantiation (`DPU_PDI_0`) | **73.60 µs** | Compute unit handle | `results/aie/windows_xrt_driver_bench.log` |

Device opening and hardware context creation together cost 139.40 ms. This cost is paid once per process session; subsequent dispatches execute on the established context.

### Unified memory (BO) synchronization bandwidth

Buffer Object allocation, pointer mapping, and bidirectional host-device synchronization across buffer sizes (64 B to 16 MB) using `pyxrt.bo.flags.host_only`:

| Buffer Size | Alloc (µs) | Map (µs) | H2D Sync (µs) | H2D Bandwidth | D2H Sync (µs) | D2H Bandwidth |
|---|---|---|---|---|---|---|
| 64 B | 31.53 | 1.76 | 0.83 | 0.07 GB/s | **0.78** | 0.08 GB/s |
| 256 B | 29.74 | 1.58 | 0.87 | 0.27 GB/s | **0.79** | 0.30 GB/s |
| 1.0 KB | 28.65 | 1.90 | 0.85 | 1.12 GB/s | **0.82** | 1.16 GB/s |
| 4.0 KB | 30.19 | 0.94 | 0.90 | 4.23 GB/s | **0.82** | 4.63 GB/s |
| 16.0 KB | 26.78 | 0.84 | 0.97 | 15.67 GB/s | **0.95** | 16.03 GB/s |
| 64.0 KB | 30.38 | 1.54 | 1.48 | 41.18 GB/s | **1.46** | 41.72 GB/s |
| 256.0 KB | 37.52 | 1.42 | 3.59 | 68.08 GB/s | **3.50** | 69.83 GB/s |
| 1.0 MB | 69.64 | 2.28 | 13.34 | 73.18 GB/s | **11.34** | 86.12 GB/s |
| 4.0 MB | 164.13 | 3.27 | 46.14 | 84.66 GB/s | **60.08** | 65.02 GB/s |
| 16.0 MB | 541.97 | 6.66 | 59.46 | 262.79 GB/s | **52.76** | 296.17 GB/s |

Key findings from the memory sweep:
- **Sub-microsecond synchronization floor**: At tile sizes <= 16 KB, `bo.sync` completes in 0.78-0.97 µs. On unified APU memory, host-to-device and device-to-host syncs do not perform PCIe/DMA bus transfers; they are CPU cache line writeback (`clflushopt`) and invalidation operations.
- **Large buffer saturation**: Device-to-host bandwidth peaks at 296.17 GB/s at 16 MB (52.76 µs), reflecting the APU coherent fabric bandwidth.

### Userspace command dispatch floor and Windows driver constraints

Micro-benchmarking the userspace call path for kernel execution:
- **Run Object Allocation**: 1.85 µs
- **Argument Binding**: 6.91 µs total across 8 kernel arguments (0.86 µs per argument via `run.set_arg`)
- **Total Userspace Preparation Floor**: 8.76 µs
- **Hardware Runlist Batching**: `pyxrt.runlist.add` requires 3.39 µs per run across 10 batched dispatches.

Two Windows driver constraints identified:
1. **`pyxrt.bo.flags.normal` is rejected**: `amdxe.sys` throws `invalid argument` on `normal` allocation. On Phoenix APUs, memory is unified host RAM and must be allocated with `pyxrt.bo.flags.host_only`.
2. **KDMA is unsupported on Windows**: XRT emits `[XRT] WARNING: Reverting to host copy of buffers (KDMA not supported on windows)` if a buffer's memory group ID does not match the kernel compute unit's connected bank. To prevent fallback copies, all zero-copy buffers must be allocated using `group_id = kern.group_id(arg_idx)`.

### Hardware context scaling and driver context-switch penalty

Benchmarked via `tools/windows_context_switch_bench.py` (`results/aie/windows_context_switch_bench.log`) on Desktop 2 (Phoenix XDNA1 NPU):

Virtual hardware context capacity allocation:

| Context Instance | Allocation Time | Status | Hardware Meaning |
|---|---|---|---|
| Context #1 | **78.63 ms** | Allocated | Cold firmware partition allocation and descriptor mapping |
| Context #2 | **5.78 ms** | Allocated | Warm slot mapping (13.6x faster than cold setup) |
| Context #3 | **5.66 ms** | Allocated | Warm slot mapping |
| Context #4 | **5.38 ms** | Allocated | Warm slot mapping |
| Context #5 | **5.72 ms** | Allocated | Warm slot mapping (matches 5 physical Phoenix columns) |
| Context #6 | **1.71 ms** | **REJECTED (0xc01e0009)** | Hardware resource exhaustion / 5-column capacity ceiling |

When Context #5 is deleted and garbage collected from userspace, reallocation succeeds in 4.98 ms, confirming clean slot recycling.

Interleaved dispatch latency and context-switch penalty (25 iterations):

| Dispatch Configuration | Mean Latency | Min Latency | Max Latency | Context-Switch Penalty |
|---|---|---|---|---|
| Same-Context (Baseline) | **120.25 µs** | 61.10 µs | 881.50 µs | Baseline |
| Cross-Context Alternation | **867.99 µs** | 467.90 µs | 933.50 µs | **+747.75 µs (+0.748 ms, 7.22x slowdown)** |

Alternating between two distinct hardware contexts on the Phoenix NPU incurs a 747.75 µs kernel driver / ERT firmware context-switch penalty due to DMA stream quiescing, micro-register state invalidation, and base register reprogramming. This explains why time-sliced multi-tenancy collapses throughput and why independent multi-process execution requires physical partition isolation across separate columns (`1x4.xclbin`).

### DPU microcode transaction stream disassembly

> **Scope correction, 2026-09-09 (Desktop 2).** An earlier version of this section
> described `tools/dpu_transaction_disasm.py` as decoding a DPU instruction set, and
> published a BiSeNetV2 row. The decode claim is narrowed to what the method supports and
> the BiSeNetV2 row is **retracted** — no log backs it. The superseded figures are kept
> below rather than deleted, per this repo's practice.

The VitisAI compiler bundles compiled DPU instruction streams inside `.xmodel` Protobuf
archives under an `mc_code` bytefield. `tools/dpu_transaction_disasm.py` locates that
field with a literal ASCII `b"mc_code"` search, then walks the archive one byte at a time
looking for any 32-bit word whose high byte is `0x0B`, consuming 48 bytes from each hit
without validating alignment. Its `OPCODE_MAP` is a **six-entry hand-written guess
table**; no AIE-ML or DPU ISA document is cited anywhere in this repo.

**What the measurement supports.** FastDepth's `compiled.0x800020500148acb.xmodel`
(7,004,134 bytes) contains two `mc_code` segments, at offsets 6,731,604 and 6,782,410,
holding a dense stream of ~48-byte-strided records — 1,659 + 911 = 2,570 of them by this
scan. The distribution is strongly bimodal: 49.38% of records carry `3` in the assumed
opcode position and 37.35% carry `6`. That is *consistent with* a graph built mostly from
pointwise and depthwise convolution, which FastDepth is (255/257 nodes, one DPU subgraph).
Backing log: `results/aie/dpu_transaction_disasm.log`.

**What it does not support, and why.** The remaining 13.27% of reported "opcodes" are
ASCII text being read as instruction words:

| Reported "opcode" | The bytes, as ASCII |
|---|---|
| `0x74746F62` | `bott` |
| `0x6E617274` | `tran` |
| `0x68736572` | `resh` |
| `0x69642D78` | `id-x` |
| `0x6E696C71` | `nilq` |
| `0x2D676F6C` | `-gol` |
| `0x6F630A0A` | `oc` followed by **two newline bytes** |

The last row settles it: no instruction opcode contains `0x0A0A`. These are xmodel
metadata strings, so an unquantified share of the 2,570 "packets" are not instructions at
all and the record boundary is not established. Every packet in the listing also carries
the identical `Inst Ptr 0x0001D214`, which is what a misaligned constant-offset read looks
like. **No DPU ISA has been recovered.** Packet boundary, opcode field position, opcode
semantics and column encoding are all unverified inference from a byte-frequency scan.

What *is* solidly measured about column identity is a different result entirely: IRON
logical `Tile(0,2)` reports `get_coreid()` row 2, column 1, confirmed independently by the
trace packet header (`results/aie/clock_probe_npu.log`).

**Superseded FastDepth listing** (kept for the record; the percentages are a byte-scan
histogram, not an instruction mix):

- Opcode 3: 1,269 packets (49.38%) · Opcode 6: 960 (37.35%) · `0xA0801A2`: 100 (3.89%)
  · `0x74746F62`: 72 (2.80%) · `0xFFFF8028`: 32 (1.25%) · control/DMA: 137 (5.33%)

**Retracted: the BiSeNetV2 row.** It reported 4,630 packets at 43.17% opcode 3, 32.61%
opcode 6, 0.24% "Bilateral Guided Aggregation gating" and 23.97% DMA/barrier. **No log in
this repo or in any of its worktrees reproduces those numbers** — the only committed
disassembly log names exactly one model, and it is FastDepth's. The row was also
attributed to `compiled.0x800020500148acb.xmodel`, which is FastDepth's own xmodel hash
copied verbatim from the entry above it. The 0.24% "gating" attribution further asserts a
semantic decode the six-entry guess table cannot support. Retracted rather than revised:
re-run and re-log before citing any BiSeNetV2 microcode figure.

**Where real ISA knowledge does exist.** `tools/aie_disasm.py` (branch `silicon-isa-pmu`)
reads AIE2 *core* machine code with Peano's own `llvm-objdump` — a real disassembler for a
documented target, calibrated against the clock probe's exact 9.000- and 2.000-cycle
loops. That is the tool to build on; this one is a byte-frequency probe.

**Log encoding.** `results/aie/dpu_transaction_disasm.log` was written as ASCII, decoded as
UTF-16LE and re-encoded as UTF-8, so it renders as CJK mojibake and every `grep` against it
silently matches nothing. The corrupt original is kept unmodified, per the rule that a log
under `results/` protects a measurement's *content*, not its byte encoding, and a file no
`grep` can read is not serving as evidence. The recovery is byte-exact: the decoded text
re-encodes to the original bytes exactly, asserted before anything was written, and the line
count is unchanged at 82. Only the encoding changed; no number moved. The same fix was
applied to `results/quant_fastdepth_xint8.log` in merge `4308a62`, and the rule is now
recorded in `docs/DECISIONS.md` so there is one policy rather than two precedents.

---

## AIE-ML systolic shift-cut feasibility theorem for Project Ignition

> **Substantially retracted on 2026-09-09 (Desktop 2). The section is kept in full,
> superseded numbers included, because what it got wrong is the useful part.**
>
> 1. **The audit does not reproduce.** The corrected analyzer over eleven quantized models
>    finds **zero sigma violations on every one**
>    (`results/quant/shift_cut_reaudit_20260909_desktop2.log`). RegNetX-002 reads
>    min 11 / median 21 / max 30, not the -90 / 24 / 27 tabulated below. FastDepth reads
>    min 18 / median 21 / **max 23** and never approaches the sigma = 32 that this section's
>    "shift clamp" case rests on. `results/quant/shift_cut_feasibility.log` no longer matches
>    the tool that wrote it.
> 2. **Both worked examples were confounds.** RegNetX-002's collapse is cross-layer
>    equalization, not the shifter: CLE **off** on the same graph, producer and calibration
>    listing takes zero shift-cut adjustments and scores **66.20% top-1** against **69.50%**
>    FP32, where the CLE-on artifact scores **0.10%**. FastDepth's "doubled layer outputs"
>    was never measured, and its NPU fidelity is in fact better than its CPU-QDQ fidelity.
> 3. **The repair was breaking working models.** `project_scale_to_feasible_basin` clamped
>    pos_x/pos_w into [0, 31] while sigma came from the unclamped scales, so it projected an
>    already-feasible RegNetX-002 layer at sigma = 30 to **sigma = -90** — manufacturing the
>    overflow it exists to prevent. Repairs on real models went SESR-M7 9 -> 0,
>    MODNet-Cut 1 -> 0, RegNetX-002 5 -> 0; each had targeted a working op.
> 4. **The "if and only if" has never been tested at either edge.** Fixtures built to reach
>    sigma 0, 31 and 32 are refused by the VitisAI EP and execute on the CPU EP instead
>    (`"tested_conv_on_npu": false`), so they cannot speak to the DPU in either direction.
>    **The highest sigma ever executed on this hardware is 30.** Neither the 15-bit
>    multiplier nor the 5-bit shifter is measured, and no ISA document in this repo states
>    either width.
>
> 5. **The zero was a tautology, and a third of the "analyzed" operations were not
>    analyzed.** Added 2026-09-09 (Desktop 2), see the subsection below and
>    `results/quant/shift_cut_contract_20260909_desktop2.log`. Quark's own
>    `adjust_shift_cut` clamps `shift_cut = wpos + ipos - opos` into `[0, 16]`, and this
>    module's sigma is the same quantity plus 14 — so the producer contract is
>    **sigma in [14, 30]**, strictly inside Theorem 1's [0, 31], and an in-contract file
>    *cannot* violate the bound the audit tests. Separately, the analyzer defaulted
>    unreadable scales to 1.0 and still scored the operation: **166 of 1102** candidate
>    operations across the eleven models were passing on invented scales, 57 of 177 on
>    yolov8n-cut alone.
>
> What survives: sigma as a computable property of a QDQ triad, and the audit as an
> **advisory** report. It must not gate the quantizer.

Analytical formulation and verification of the post-accumulator scaling unit on XDNA1 AIE-ML, isolating the mathematical mechanism causing catastrophic accuracy collapse in quantized topologies.

Backing logs:
- `results/quant/shift_cut_contract_20260909_desktop2.log`: the producer-contract identity,
  the coverage correction, and contract-edge saturation across the same eleven models.
  Supersedes neither log below — every violation count in them still reads zero — but it is
  why that zero was never evidence.
- `results/quant/shift_cut_reaudit_20260909_desktop2.log`: the corrected audit across eleven
  quantized ONNX models, zero violations on all of them. **This supersedes the log below.**
- `results/quant/shift_cut_feasibility.log`: the original audit across 7 quantized ONNX
  models. **Superseded — it no longer matches its own tool.**
- `results/quant/eval_regnetx_002_ignition_nocle_c64_cpu.log`,
  `eval_regnetx_002_fp32_full1000_cpu.log`, `eval_regnetx_002_quark_cle_full1000_cpu.log`,
  `quant_regnetx_002_ignition_nocle_c64.log`,
  `eval_resnext50_32x4d_quark_nocle_c64_cpu.log`: the CLE re-attribution (measured on branch
  `research/windows-lowlevel`, imported here as the evidence for the retraction above).

### Mathematical formulation

On the XDNA1 AIE-ML architecture, integer convolution and matrix multiplication accumulate into 32-bit registers. The post-multiplication ALU maps the 32-bit accumulator to an 8-bit output tensor using an integer multiplier M (15-bit) and an arithmetic right-shift register sigma in [0, 31]:

    out_8 = clamp( floor( (acc_32 * M + 2^(sigma - 1)) / 2^sigma ), -128, 127 )

For an ONNX QuantizeLinear/DequantizeLinear triad with input scale S_x, weight scale S_w, and output scale S_y, the ideal analytical scale factor is:

    A = (S_x * S_w) / S_y

The hardware compiler approximates A using (M, sigma):

    A ≈ M * 2^(-sigma), where M in [16384, 32767] and sigma in [0, 31].

### Theorems

**Theorem 1 (Systolic Shift-Cut Bound):**
An operation is physically executable without numerical distortion on XDNA1 if and only if
(**hypothesis, not a result — see the retraction at the top of this section; neither edge has
been reached on hardware and the highest sigma ever executed is 30**):

    0 <= sigma <= 31

If sigma < 0, the operation requires an arithmetic left-shift exceeding the 32-bit accumulator, resulting in accumulator overflow. If sigma > 31, the hardware 5-bit shift register overflows or clamps.

**Theorem 2 (Multi-Branch Inter-Scale Feasibility):**
For multi-branch elementwise tensor operations C = A * B or C = A + B:

    A_elem = (S_A * S_B) / S_C

Both input branches must satisfy identical power-of-two scale alignments; divergent scale grids cause dynamic range truncation in the fixed-point ALU.

> **Not supported as stated (2026-09-09).** BiSeNetV2's two Bilateral Guided Aggregation gates were
> measured node by node. The one that fails has a position gap of 4; the one that works has a gap of
> **6**. Divergence alone therefore does not predict which Mul breaks, and both sit at feasible
> sigma. [Evidence](#bisenetv2s-cpunpu-gap-localised-to-one-mul-2026-09-09-desktop-2).

### Empirical audit across 7 models

> **Superseded 2026-09-09 — this whole table.** Re-running `check-shift-cut` over these
> same files with the corrected analyzer gives **zero violations on every model**
> (`results/quant/shift_cut_reaudit_20260909_desktop2.log`). The two CRITICAL rows do not
> reproduce: **RegNetX-002 reads min 11 / median 21 / max 30**, not -90 / 24 / 27, and
> **FastDepth reads min 18 / median 21 / max 23**, not 25 / 29 / 32 — its sigma never gets
> near the 32 the clamp story needs. Three defects caused the original numbers: the position
> check sat in an `elif` after the sigma checks so it only fired when sigma was already
> feasible; `--repair` gated on different positions than the analyzer flagged; and the
> projection clamped pos_x/pos_w into [0, 31] while sigma came from the unclamped scales.
> The rows below are kept as the record of what was reported.

Evaluated with `python -m quant check-shift-cut` (`results/quant/shift_cut_feasibility.log`):

> Two sessions re-ran this independently on Desktop 2 the same day and agree: the
> `research/windows-lowlevel` re-audit
> ([log](../results/quant/shift_cut_feasibility_desktop2_20260909.log)) covers eleven models and
> also reads zero violations, with the same RegNetX-002 11 / 21 / 30 and FastDepth 18 / 21 / 23.
> The defects behind the original numbers are itemised in
> [the audit re-run](#the-shift-cut-audit-re-run-theorem-3-retracted-and-three-defects-in-the-verifier-2026-09-09-desktop-2).


| Model | Quantized Ops | Violations | Sigma Range (min / median / max) | Hardware Status | Backing Log |
|---|---|---|---|---|---|
| **RegNetX-002** | 46 | **1 (2.2%)** | **-90 / 24 / 27** | **CRITICAL: Accumulator Overflow (sigma = -90 < 0)** | `results/quant/shift_cut_feasibility.log` |
| **FastDepth** | 38 | **1 (2.6%)** | **25 / 29 / 32** | **CRITICAL: Shift Clamp (sigma = 32 > 31)** | `results/quant/shift_cut_feasibility.log` |
| **MODNet** (`modnet_cut_xint8`) | 74 | **0 (0.0%)** | 7 / 24 / 31 | PASS: Reaches upper register bound (sigma = 31) | `results/quant/shift_cut_feasibility.log` |
| **ResNet50** (`resnet50_xint8_c64`) | 55 | **0 (0.0%)** | 12 / 24 / 26 | PASS: Centered in systolic basin | `results/quant/shift_cut_feasibility.log` |
| **YOLOv8n** (`yolov8n_cut_xint8`) | 177 | **0 (0.0%)** | 7 / 20 / 23 | PASS: Centered in systolic basin | `results/quant/shift_cut_feasibility.log` |
| **MiDaS Small** (`midas_small_cut_xint8`) | 97 | **0 (0.0%)** | 17 / 23 / 27 | PASS: Centered in systolic basin | `results/quant/shift_cut_feasibility.log` |
| **BiSeNetV2** (`bisenetv2_fp32_xint8`) | 63 | **0 (0.0%)** | 7 / 22 / 26 | PASS: Centered in systolic basin | `results/quant/shift_cut_feasibility.log` |

Diagnosis of identified violations — **all three retracted 2026-09-09, kept as the record**:
- **RegNetX-002**: Layer `/s1/b1/conv2/conv/Conv` has scale factor A = 2.028241e+31, yielding M = 16384 and sigma = -90. Because sigma < 0, the post-accumulator ALU cannot scale the 32-bit register down to int8; this forces top-1 accuracy to collapse to 0.50% (random chance). — **Retracted.** The corrected analyzer reads that layer at **sigma = 30**, inside the basin; the -90 was produced by the defective projection, not by the graph. The accuracy collapse is real but is caused by cross-layer equalization: CLE off gives **66.20% top-1** against **69.50%** FP32 with no shift-cut adjustment logged, CLE on gives **0.10%** ([full matrix](#regnetx-002-and-resnext-50-recovered-the-collapse-is-cle-not-a-hardware-bound-2026-09-09-desktop-2)).
- **FastDepth**: Layer `Conv_96` has scale factor A = 3.814697e-06, yielding M = 16384 and sigma = 32. The hardware shifter is 5 bits wide (maximum shift 31); sigma = 32 overflows by exactly 1 bit, clamping to 31 and doubling the layer's output activations. — **Retracted.** The corrected analyzer reads FastDepth at max sigma **23**. The "doubling" was never measured, and FastDepth's NPU fidelity is in fact better than its CPU-QDQ fidelity (r = 0.9383 vs 0.9363), which is the opposite of what a clamped layer would give.
- **MODNet**: Upper bound hits sigma = 31 exactly. Any scale perturbation exceeding 1 bit would push it into the clamp hazard regime. — **Not reproduced**; the re-audit reads MODNet-Cut with zero violations, and in any case "the clamp hazard regime" above sigma 31 has never been observed on hardware.

### Closed-form systolic scale feasibility window and repair projection

In power-of-two quantization where scale is parameterized by position (S = 2^(-pos)), the post-accumulator scaling formula reduces to:

    sigma = pos_x + pos_w - pos_y + 14

Because the physical shift register is bounded by 0 <= sigma <= 31, the output scale position pos_y must satisfy the **Systolic Scale Feasibility Window**:

    pos_x + pos_w - 17 <= pos_y <= pos_x + pos_w + 14

Or equivalently in real scales:

    2^(-14) * (S_x * S_w) <= S_y <= 2^(17) * (S_x * S_w)

### The shift-cut audit re-run: Theorem 3 retracted, and three defects in the verifier (2026-09-09, Desktop 2)

Re-auditing eleven quantized models with the current tool finds **no sigma hazard anywhere**
([log](../results/quant/shift_cut_feasibility_desktop2_20260909.log)). Sigma spans 7 to 30 across
every model and never reaches either edge of [0, 31]. Every flag the original audit raised was
Theorem 3 -- a position rule, not a sigma rule -- which means **`--repair` has never had a genuine
hazard to act on, and the standing "verify before hardware execution" rule has never caught
anything real.**

| Model | Ops | Sigma violations | Sigma min / median / max | Advisory position notes |
|---|---|---|---|---|
| ResNet50 `xint8_c64` | 55 | 0 | 12 / 21 / 24 | 0 |
| YOLOv8n-cut | 177 | 0 | 7 / 20 / 23 | 0 |
| YOLOv8n-pose-cut | 198 | 0 | 7 / 21 / 23 | 0 |
| MODNet-Cut | 74 | 0 | 7 / 22 / 27 | 1 |
| SESR-M7 | 9 | 0 | 17 / 21 / 23 | **9** |
| FastDepth | 38 | 0 | 18 / 21 / 23 | 0 |
| RegNetX-002 | 46 | 0 | 11 / 21 / 30 | 8 |
| MiDaS-Small nearest-cut | 97 | 0 | 15 / 22 / 26 | 16 |
| BiSeNetV2 | 63 | 0 | 7 / 21 / 24 | 1 |
| MobileViT-XXS | 177 | 0 | 7 / 21 / 23 | 0 |
| ResNeXt-50 32x4d | 55 | 0 | 11 / 21 / 30 | 8 |

MobileViT is worth noting: 0 violations and 0 notes, so the shift-cut theory has nothing to say
about its 0.00% collapse either way.

**Theorem 3 is retracted.** It held that positions must lie in [0, 31] and that `pos < 0`
(scale > 1.0) causes dynamic range overflow. Three independent measurements contradict it:

- `tools/xint8_arithmetic_probe.py` sets the output position to `-sc`. **Fifteen of its seventeen
  NPU fixtures ran at pos_y in {-1, -4, -8, -16}** -- output scales up to 2^16 -- placed on the DPU,
  and matched an independent integer reference apart from the uniform one-code half-up rounding
  difference documented above.
- SESR-M7 was flagged **9/9** while placing 50 of 52 nodes and scoring 34.06 dB; MODNet-Cut was
  flagged 1/74 while placing 502 of 507.
- RegNetX-002's `pos = -120`, cited as the theorem's evidence, is
  [CLE's doing](#regnetx-002-and-resnext-50-recovered-the-collapse-is-cle-not-a-hardware-bound-2026-09-09-desktop-2),
  not a hardware bound.

Positions outside [0, 31] are now reported as an **advisory note**, never a hazard. Theorem 1's
sigma window remains the executability criterion -- as Theorem 1 itself always said, with an *iff*.
Note the sigma edges remain **unmeasured**: the probe only ever reached sigma in
{14, 15, 18, 22, 30}.

**Three defects in `quant/shift_cut.py`, all fixed here.**

1. **In-range sigma reported as CRITICAL.** The position check sat in an `elif` after the sigma
   checks, so it fired *only* when sigma was feasible -- flagging exactly the operations Theorem 1
   calls executable. SESR-M7 read `9/9 CRITICAL HARDWARE INFEASIBILITY` for a model that runs.
2. **`--repair` skipped hazards it had just reported, and its two operator branches disagreed.**
   The repair gate tested `pos_y` only, while the analyzer flagged on `pos_x`, `pos_w` or `pos_y`,
   so a node flagged for an input position was reported and then silently not repaired -- with a
   repair count printed regardless. Separately, the `Mul` analyzer checked `< 0` but not `> 31`,
   which the `Conv` branch did. Both branches now share one `position_note` helper, and the repair
   gate is exactly the hazard criterion.
3. **The projection could manufacture the hazard it exists to prevent.**
   `project_scale_to_feasible_basin` clamped `pos_x` and `pos_w` into [0, 31] to build its window,
   but recomputed sigma from the *unclamped* scales, so the window and sigma came from different
   numbers. Measured on RegNetX-002 `/s1/b1/conv2/conv/Conv`: a layer at **sigma = 30, already
   feasible**, was projected to **sigma = -90** -- the accumulator-overflow case the module exists
   to catch. Positions are now used unclamped and the result is checked, raising rather than
   emitting a scale the analyzer would flag. The documented FastDepth demo still reproduces
   exactly (pos_y 0 -> 1, sigma -> 31).

### Theorem 1's sigma window: the edges are unreachable, not measured (2026-09-09, Desktop 2)

> **This section originally claimed sigma = 32 executes exactly and sigma = 0 does not, and both
> claims are RETRACTED.** Every fixture built outside the producer's shift-cut rule was refused by
> the VitisAI EP and ran entirely on CPU -- `"tested_conv_on_npu": false`, 7 of 7 nodes off the NPU
> -- so those rows compared CPU against CPU and say nothing about the DPU. The mistake was reading
> mismatch counts without checking placement, which is the failure this repo warns about most
> often. Caught by the forward test on `main` (`79ee4fe`, `311a672`), which reached the same
> conclusion independently. What the sweep does establish is below, and it is a better result.

**The EP's acceptance boundary is the producer's shift-cut rule.** Across all sixteen fixtures, the
Conv places if and only if its shift-cut lies in `[0, 16]` -- exactly the bound `quant/refine.py`
enforces and Quark logs. There are no exceptions in either direction:

| shift-cut | sigma | inside [0, 16] | Conv on NPU | Nodes off NPU |
|---|---|---|---|---|
| -20, -17, -15, -14, -8 | -6, -3, -1, 0, 6 | no | **false** | 7 of 7 |
| 0, 1, 3, 4, 8, 16 | 14, 15, 17, 18, 22, 30 | yes | **true** | 2 |
| 17, 18, 20, 24, 31 | 31, 32, 34, 38, 45 | no | **false** | 7 of 7 |

So sigma is reachable on this hardware only in **[14, 30]**, and Theorem 1's `[0, 31]` window can
never be tested at either edge with a Conv the DPU will actually run. That also explains, without
any appeal to a shift register's width, why no model in this repo has ever shown a sigma outside
7..30: the compiler will not accept the graph in the first place. The highest sigma ever executed
here is 30.

The rounding results from the in-contract sweep are unaffected -- those fixtures did place, at 5 of
7 nodes, and are reported in the arithmetic section above.



Theorem 1 claims an operation is executable **iff** `0 <= sigma <= 31`, and it had never been
tested at either edge: the arithmetic probe's guard is the producer's own `[0, 16]` shift-cut
rule (`quant/refine.py`), which caps sigma at 30. `--out-of-contract` widens that guard for
synthetic 7-node fixtures only, leaving the default untouched. Eleven cases, `--checks-only`,
each `--fresh` with an idle-device precheck
([first pass](../results/quant/arithmetic_desktop2_20260909_s01_sigma17_sc3.log),
[second pass](../results/quant/arithmetic_desktop2_20260909_s02_sigma32_sc18.log)):

| sigma | shift-cut | NPU mismatches | max abs diff | Optimized CPU | Distinct reference outputs | Reading |
|---|---|---|---|---|---|---|
| -6 | -20 | 7763 / 57344 | **255** | 0 | 3 | diverges |
| -3 | -17 | 4488 / 57344 | **255** | 0 | 3 | diverges |
| -1 | -15 | 3564 / 57344 | **255** | 0 | 3 | diverges |
| **0** | -14 | **3052 / 57344** | **255** | 0 | 3 | **diverges, though Theorem 1 calls it feasible** |
| 6 | -8 | 0 / 57344 | 0 | 0 | 3 | agrees |
| 17 | 3 | 1056 / 57344 | 1 | 0 | 101 | agrees, up to the known half-up rounding |
| **31** | 17 | **0 / 57344** | 0 | 0 | 9 | **agrees exactly at the claimed upper bound** |
| **32** | 18 | **0 / 57344** | 0 | 0 | 5 | **agrees exactly one past it** |
| 34 | 20 | 0 / 57344 | 0 | 0 | **1** | degenerate, proves nothing |
| 38 | 24 | 0 / 57344 | 0 | 0 | **1** | degenerate, proves nothing |
| 45 | 31 | 0 / 57344 | 0 | 0 | **1** | degenerate, proves nothing |

**On the retracted upper-bound row.** At sigma = 32 the fixture produced five distinct output
levels and the two providers agreed on all 57,344 elements -- but both were the CPU, so this says
nothing about the shifter. Retained only to show what the numbers were.

**On the retracted lower-bound row.** Sigma = 0 showed 3,052 elements wrong against the plain CPU
session, worst case 255 -- but that graph also ran wholly on CPU, through the VitisAI EP's fallback
rather than the DPU, so the divergence is between two CPU paths and not evidence about hardware
saturation. It is left here as the record of what was measured and how it was misread.

Saturation alone does not explain it. Sigma = 6 saturates identically -- the same three distinct
reference values, 0/128/255 -- and reads **zero** mismatches. Something changes between sigma = 6
and sigma = 0, and the direction is consistent with the int32 accumulator overflowing once
`A = M * 2^-sigma` approaches `2^14`. So Theorem 1's *mechanism* survives while its *boundary*
does not: the measured window is `sigma >= 1`, not `sigma >= 0`.

**The last three rows are worthless, and that is a property of the fixture, not the hardware.**
Because the probe holds `pos_x = pos_w = 0` and moves only `pos_y`, sigma and the output magnitude
are the same knob: `out ~ acc * 2^(14 - sigma)`. With C = 32 the accumulator caps at
`32 * 127 * 127 = 2^19`, so the predicted peak output is 3.94 at sigma = 31, 1.97 at 32 and 0.49 at
34 -- against 9, 5 and 1 distinct values actually observed. The model is exact. Beyond the mid-30s
every output rounds to the zero point and the comparison is vacuous.

That also bounds what any single int8 Conv can ever test. An int32 accumulator caps at `2^31`, so a
non-constant output needs `sigma <= 14 + 31 = 45`, and reaching sigma = 34 non-degenerately would
take roughly 6,500 input channels against the probe's 64. **This is why no model in the repo has
ever exceeded sigma = 30**: real layers sit where their accumulators put them, and the upper edge
of the window is not somewhere a convolution can go.

**What this leaves.** Theorem 1 is untested at both edges and cannot be tested with a single Conv
on this stack, because the compiler refuses every graph that would reach them. What is *not*
claimed here: that sigma = 33 or beyond is safe, which no fixture
in this family can show; that the divergence at sigma <= 0 is specifically shifter behaviour rather
than accumulator overflow, which these fixtures cannot separate; or anything at all about
multi-layer graphs, since every case is one 1x1 Conv at batch 1 with 5 of 7 nodes on the NPU.

**What the repair fix prevents, measured on three shipped models.** Running the old and new
`repair_model_shift_cut` over the same files:

| Model | Scales rewritten before | After | What it would have done |
|---|---|---|---|
| SESR-M7 (50/52 placed, 34.06 dB) | **9** | 0 | scale 2 -> 1 on `/head/Conv`, sigma 21 -> 20 |
| MODNet-Cut (502/507 placed) | **1** | 0 | scale 2 -> 1 on `/f_branch/conv_f/conv_f.2/Conv` |
| RegNetX-002 | **5** | 0 | sigma 30 -> **-90** on `/s1/b1/conv2/conv/Conv` |

Every one of those rewrites targeted a feasible operation on a model that works. The standing rule
in [DECISIONS](DECISIONS.md#aie-ml-systolic-shift-cut-bound-0-31--substantially-retracted-2026-09-09) that every emitted graph be
verified before hardware execution is kept, but it now means the sigma window alone, and `--repair`
is not something to run on a model that already places and scores.

If pos_y < pos_x + pos_w - 17, sigma > 31 and the hardware shifter clamps/overflows. If pos_y > pos_x + pos_w + 14, sigma < 0 and the 32-bit accumulator overflows.

**Automated Scale Repair Projection:**
When an ONNX graph contains violating nodes, `quant/shift_cut.py::project_scale_to_feasible_basin` projects pos_y to the nearest boundary:

    pos_y_repaired = clamp(pos_y, pos_x + pos_w - 17, pos_x + pos_w + 14)

Tested on FastDepth `Conv_96`:
- Original: pos_x = 7 (S_x = 2^-7), pos_w = 11 (S_w = 2^-11), pos_y = 0 (S_y = 1.0).
- Analytical sigma: 7 + 11 - 0 + 14 = 32 > 31 (overflows 5-bit shifter by 1 bit).
- Feasible pos_y window: [18 - 17, 18 + 14] = [1, 32].
- Projected: pos_y = 1 (S_y = 0.5), yielding sigma = 31 <= 31.
- Outcome: The layer is 100% physically compliant with zero shift-cut violations, eliminating the clamp hazard without retraining.

> **Retracted 2026-09-09.** This projection was measured to move layers that were already
> feasible. On RegNetX-002 `/s1/b1/conv2/conv/Conv` it took sigma = 30 to **sigma = -90**,
> because the window was built from pos_x/pos_w clamped into [0, 31] while sigma was computed
> from the unclamped scales. Across real models the "repairs" it reported were SESR-M7 9,
> MODNet-Cut 1 and RegNetX-002 5 — all of which drop to 0 once the analyzer is corrected,
> i.e. every one had targeted a working op. `quant/shift_cut.py` now uses unclamped positions
> and raises rather than emitting a scale the analyzer would flag. Do not run `--repair` on a
> model that already places and scores.

---

### The producer contract is the binding constraint, not Theorem 1 (2026-09-09, Desktop 2)

Static ONNX inspection only, no hardware context:
`results/quant/shift_cut_contract_20260909_desktop2.log`. Same eleven models as the
re-audit, so the two are directly comparable.

**sigma and the vendor's own quantity differ by a constant.** Quark's `adjust_shift_cut`
— transcribed at `quant/refine.py::shift_cut`, sourced in
`results/quant/notes_xint8_dialect.log:37` and `:1216-1253` — defines, for Conv and Gemm,
`shift_cut = wpos + ipos - opos`, clamped into `[0, 16]`. `quant/shift_cut.py`'s sigma over
the same three positions is `pos_x + pos_w - pos_y + 14`. So `sigma == shift_cut + 14`, and
the producer contract is exactly **sigma in [14, 30]** — a strict subset of Theorem 1's
`[0, 31]`.

A Conv or Gemm in a file Quark or Ignition emitted therefore cannot reach the `[0, 31]`
edges without the producer's clamp having failed first. **The "zero violations on 11 of 11
models" result is a tautology of that clamp, not a measurement of the hardware bound.**
Measured here and now checked rather than assumed: every analyzed Conv/Gemm on all eleven
models lies in `[14, 30]`, zero outside.

This also explains, with no new hardware, the fact recorded in `311a672` that the highest
sigma ever executed on this device is 30 — 30 is the contract's upper edge (`shift_cut` 16),
i.e. the producer's ceiling rather than the silicon's. Both register widths behind Theorem 1
remain unmeasured and both edges of `[0, 31]` remain unreached.

**166 of 1102 candidate operations were being scored on scales that were never read.** The
analyzer initialised `scale_x`/`scale_w`/`scale_y` to 1.0 and overwrote each only where a
`DequantizeLinear` producer with an initializer scale existed; where none existed the
operation was still scored and still counted as passing. Those rows are the `<out>_Scale` /
`<out>_Mul` pairs `quant/passes.py::_insert_mul` writes for DPU simulation, whose inputs are
a `Constant` and a `HardSigmoid`. They are the sub-14 minima in the published distributions —
no Conv ever produced one.

| model | candidates | analyzed | unresolved | Conv/Gemm n | Conv/Gemm sigma min/med/max |
|---|---|---|---|---|---|
| resnet50_xint8_c64 | 55 | 54 | 1 | 54 | 18 / 21 / 24 |
| yolov8n_cut_xint8 | 177 | 120 | **57** | 63 | 19 / 21 / 23 |
| yolov8n-pose_cut_xint8 | 198 | 135 | **63** | 72 | 20 / 21 / 23 |
| regnetx_002_xint8 | 46 | 45 | 1 | 45 | **14** / 21 / **30** |
| resnext50_32x4d_xint8 | 55 | 54 | 1 | 54 | **14** / 21 / **30** |
| bisenetv2_fp32_xint8 | 63 | 59 | 4 | 57 | 16 / 21 / 24 |
| fastdepth_fp32_xint8 | 38 | 38 | 0 | 38 | 18 / 21 / 23 |
| sesr_m7_xint8 | 9 | 9 | 0 | 9 | 17 / 21 / 23 |
| midas_small_cut_xint8 | 97 | 97 | 0 | 97 | 15 / 22 / 26 |
| mobilevit_xint8 | 177 | 142 | 35 | 72 (+18 MatMul) | 18 / 22 / 26 |
| densenet121_xint8 | 187 | 183 | 4 | 183 | **14** / 21 / 29 |

**Scope, because getting this wrong would be the same defect again.** The contract is
asserted of **Conv and Gemm only** — `adjust_shift_cut` skips every other `op_type`
(`quant/refine.py::shift_cut`). All 744 analyzed Conv/Gemm across the eleven models sit in
`[14, 30]`. The 18 analyzed MatMuls and 174 analyzed Muls get a sigma and the `[0, 31]`
hazard test but **no contract verdict**: they happen to land inside `[14, 30]` here, which
is a coincidence of these models rather than a rule. A Mul is refined by `shift_write_mul`
(clamp 0..32) and `shift_swish` (clamp 0..15) — different rules over different quantities —
so judging one against `[14, 30]` would report "this file was not emitted by
Quark/Ignition" about a file that was.

Unresolved operations are now excluded from the violation denominator and from every
distribution, and counted on their own line. `--repair` skips them too: projecting from an
invented 1.0 is the same class of defect as the sigma = −90 incident in `311a672`.

**Contract-edge saturation, and what it does not prove.** An operation at sigma 14 or 30 is
one the clamp had hard against an edge. Only three models touch an edge at all, and only two
touch *both*: regnetx_002 (4 low, 2 high) and resnext50_32x4d (4 low, 4 high) — exactly the
two identified above as CLE confounds, at 0.10% top-1 each and recovering to 66.20% and
68.90% with CLE off. densenet121 is the informative near-miss: 18 at the low edge, none at
the high edge, and not a known accuracy casualty.

This is a **correlation at n = 2 and the cause is not established.** The alternative not
ruled out is architecture: both saturating models are grouped-convolution designs, whose
per-channel weight ranges are far wider than the dense convolutions in the seven interior
models, and in the sample available on this machine no grouped-conv model is CLE-free and no
CLE-on model is group-free — the two explanations are perfectly confounded. The deciding
experiment is a single-graph A/B in the shape of `scripts/quant-cle-probe.sh`: quantize
regnetx_002 through Ignition twice from the same float export and calibration listing, once
`--cle` and once `--no-cle`, and re-measure saturation. Ignition emits both, so it needs no
vendor run. Not done here — it is a producer pass and belongs on Desktop 1.

**Multi-branch inter-scale divergence** is now reported by `check-shift-cut --branches` as
the inter-branch position spread at every quantized `Add`/`Concat`, and is **advisory
only**: a DPU aligns branches to one output scale before adding them, so a wide spread is
where that alignment costs most, but no divergence has been measured to change an output on
this device. BiSeNetV2 — the model the check was asked for — reads zero violations under
every criterion in the module, and its Add/Concat spread is unremarkable.

**Fixture checks.** `tools/quant_shift_cut_checks.py` (65 checks) builds ONNX graphs whose
sigma is chosen by construction and asserts the analyzer reads it back, bands it correctly,
declines to band a Mul at all, refuses to score an operation whose scales it cannot read,
and leaves an in-contract model byte-identical under `--repair`. `tools/quant_passes_checks.py` (54 checks) pins each
`quant/passes.py` rewrite against a fixture whose expected node list is known. Neither
establishes vendor parity — the oracle diff remains that gate — and neither touches hardware.

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

## Windows low-level research: placement and XINT8 arithmetic

**2026-09-09, Desktop 2, Ryzen 7 8700G / Phoenix, Windows.** These experiments
intervene on local operand addresses, probe observable Conv arithmetic, and evaluate
an exact-count calibration alternative. They use an isolated checkout: at
`1e27c5d48b30b9e83a821d36a27a0e8b81fa2bbc` for the placement, arithmetic and first
calibration runs, and at later commits on the same branch for the `c04` and `t01`
calibration runs. Each log's `RESEARCH_PLATFORM` line carries the commit it ran at.
Production Ignition defaults are unchanged.
The machine, source hashes, command, resource limits and host witness are embedded in
each log. NPU runs require an idle-device precheck and record sampled context ownership.
The [toolchain capture](../results/aie/toolchain_desktop2_20260909_01.log) identifies the
existing mlir-aie release and local patches; this work did not modify that toolchain.

### Local operand placement: an address intervention

The [paired-load probe](../kernels/memory_placement/probe.py) fixes the external
function and changes only buffer placement in the array program. On the measured
physical tile `(row=2, col=1)`, operand A starts at `0x4000`; B starts at `0x4800`
(same allocator bank) or `0x8000` (separate allocator bank). Each iteration loads
two vectors and accumulates their product. Input transfer, initialization, checksum
reduction, output transfer and trace-flush fillers lie outside the event bracket.

| Timed body | B placement | Fitted core cycles for N iterations |
|---|---|---|
| Two loads and MAC | Same | `40 + 12*N` |
| Two loads and MAC | Separate | `40 + 11*N` |
| One load and MAC | Either | `41 + 11*N` |

The dual-load matrix runs three sessions per placement in alternating order, with
N = 512, 1024, 2048, 4096, 8192 and 16384 and five measured samples per N.
Every checksum passes, every selected fit has R-squared 1, and no foreign context
was observed. A offsets of 64, 128 and 256 bytes retain the same-bank slope of 12.
The complete final function is 1760 bytes, SHA-256
`87ab8a827f0ec7852b1418be7a6cef374e96dec2d0cc32c6c5eee7107cb9c10e`,
identical across the intervention, including the single-load control. Object and final
function equality are controls against compiler scheduling changes; the array programs
necessarily differ. The disassembly shows bundled operand loads. This supports a
placement-dependent load conflict for this access pattern, without determining the
complete physical banking map or a bandwidth ceiling.

Evidence: [same](../results/aie/memory_desktop2_20260909_m01_same_1.log),
[separate](../results/aie/memory_desktop2_20260909_m01_separate_1.log),
[single same](../results/aie/memory_desktop2_20260909_m01_single_same.log),
[single separate replacement](../results/aie/memory_single_separate_desktop2_20260909_02.log).
The [AIE index](../results/aie/README.md#windows-local-memory-placement) identifies all
repeats, offsets and failed preflights.

### Transfer to a tiled INT8 GEMM

The [GEMM intervention](../kernels/memory_placement/gemm.py) uses a single core,
`aie::mmul<4,8,8,int8,int8,acc32>`, and a `64 x (64*P) x 64` product. A starts at
`0x4000`, B at `0x6000` or `0x8000`, and C stays at `0xC000`, verified from the
lowered buffers. Every call checks all 4096 output integers against an independent
int64 NumPy matrix product. The timed bracket includes the GEMM and local C stores;
operand initialization and DMA are outside it.

| Placement | Core-cycle fit, P = 1, 2, 4, 8 |
|---|---|
| Same allocator bank | `2659 + 15232*P` |
| Separate allocator banks | `2659 + 14208*P` |

Two sessions per placement, followed by one per placement alternating between two
resident operand panels, all give the same fits with R-squared 1 and exact repeated
cycle counts. Each target has five measured samples and a warmup. Host pacing of
100 ms is outside the event bracket. The complete final function is 1632 bytes,
SHA-256 `29b4b56e0bc42df03e7e8325937425c7237c26429faa7edc93cba21b36f1a044`,
identical for every variant.

The measured difference is `1024*P` cycles. There are
`(64/4)*(64/8)*8*P = 1024*P` logical mmul updates: **one cycle saved per logical
update in this layout**, or **6.72% of the reduction slope**. This is a local-kernel
result; alternating resident panels does not test DMA overlap, and it establishes
no whole-array or application speedup.

Evidence: [separate](../results/aie/gemm_desktop2_20260909_g03_separate_1.log),
[same](../results/aie/gemm_desktop2_20260909_g03_same_1.log),
[alternating separate](../results/aie/gemm_desktop2_20260909_g03_separate_alternate.log),
[alternating same](../results/aie/gemm_desktop2_20260909_g03_same_alternate.log).
The unpaced exploratory `g02` attempt ended in XRT context-creation error
`0xc01e0009`; its cause remains unknown. It contributes no performance claim.

**Testable optimization hypothesis:** construct a weighted graph of operand pairs
loaded together and minimize `sum(weight[u,v] * same_bank[u,v])`, subject to buffer
size, alignment and fixed output placement. Use dynamic joint-load counts as weights.
For this GEMM the measured penalty coefficient is one cycle per logical update.
The next test is to predict an unseen layout or panel shape, freeze the function
bytes, and compare predicted versus measured cycle differences. The present evidence
does not validate this objective for arbitrary kernels, address strides or DMA traffic.
The address mechanism is documented by the upstream
[buffer allocator](https://xilinx.github.io/mlir-aie/dev/doxygen/html/AIEAssignBuffers_8cpp_source.html);
the cycle results above come from these local experiments.

### Observable XINT8 Conv rounding

[The arithmetic probe](../tools/xint8_arithmetic_probe.py) generates static batch-one
QDQ models containing one 1x1 Conv. Channel counts 1, 31, 32, 33 and 64, output
shifts 0, 1, 4, 8 and 16, fractional/integer bias shifts, saturation, cancellation
and reversed input/weight channels distinguish 150 arithmetic candidates. Each model
uses three fitting fixtures and four held-out fixtures, including adjacent float32
values on both sides of input half ties. An independent scalar rational check
validates the vectorized reference. Both CPU optimization settings agree exactly
with the nearest-even ONNX reference.

Every one of the 17 NPU models places the tested Conv: the EP report assigns 5/7
nodes to the NPU, with input QuantizeLinear and terminal DequantizeLinear on CPU.
Therefore input rounding is a CPU boundary observation. Across 974848 output
elements, 94412 differ from the ONNX reference, each by exactly one output code.
Repeated NPU outputs agree exactly; reversing channels at C = 31, 32 and 33
preserves every output. No foreign NPU context was observed.

One candidate survives the intersection across all models and held-out fixtures:

```text
qx = clip(round_even(x), -128, 127)
yq = clip(128 + floor((sum(qx*qw) + qb*2^shift_bias)/2^shift_cut + 0.5), 0, 255)
```

This is half-up output rounding, including negative ties, with the fractional bias
retained in the tested combined expression. It describes the **observable compiled
stack** among the candidates tested; it does not uniquely identify the silicon's
internal accumulator implementation. No accuracy, latency or general Conv-kernel
claim follows. ONNX specifies nearest-even rounding in
[QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13).
Ignition's existing producer rounding remains unchanged.

Evidence: [baseline](../results/quant/arithmetic_desktop2_20260909_a01_c32_sc1_sb0.log),
[fractional bias at zero output shift](../results/quant/arithmetic_desktop2_20260909_a02_c32_sc0_sb-1.log),
[large output/bias shift](../results/quant/arithmetic_desktop2_20260909_a02_c32_sc16_sb15.log),
[independent oracle check](../results/quant/check_arithmetic_desktop2_20260909_01.log).
All `a01` and `a02` fixtures, raw outputs and per-node placement reports are retained
in the evidence archive described below.

**Testable optimization hypothesis:** use an observed-stack rounding surrogate when
ranking quantization positions, while preserving the emitted ONNX contract. On these
fixtures, output-space discrepancy from nearest-even obeys
`MSE = output_scale^2 * mismatch_fraction`, since every difference is one code.
That identity measures disagreement between execution paths, not accuracy loss against
float output. A next test would compare the two position rankings on saved real
activations, then run both resulting models on the same full evaluation set; a lower
surrogate loss alone would not establish an accuracy improvement.

### Exact-count calibration: certificate and fallback

[The calibration experiment](../tools/quant_calib_alphabet.py) counts every float16
bit pattern using uint64 frequencies and evaluates the same float32 element errors
as the ordered-spool producer. Counts lose sample order. A conservative interval
using `gamma_(N-1)` bounds any float32 reduction tree; a separate float64 bound covers
the weighted count reduction. Every interval operation rounds outward. Disjoint loss
intervals certify a winner; pointwise-identical errors preserve first-minimum ties;
all other cases use the exact ordered legacy reduction. The guard also rejects
nonfinite samples and counter overflow. It does not assume a particular
[NumPy reduction tree](https://raw.githubusercontent.com/numpy/numpy/v1.26.4/numpy/core/src/umath/loops_utils.h.src).

In `dual` mode every certified position is checked against the full ordered spool.
In `alphabet` mode ambiguous tensors are spooled by replaying the same unoptimized
ORT graph and output list; ordered float16 hashes must match the first pass. Both
modes leave preparation, CLE, QDQ emission and refinement unchanged. The research
collector replacement exists only inside the experiment process.

All four declared pairs are complete on the same 64-image listings, each `dual` run
compared against a legacy run started fresh in the same scratch base. Every one emits
byte-identical ONNX. No certificate was ever unsound: `dual` also reduces every tensor
from the ordered spool and asserts on any disagreement with a certified position.

| Model | Activation tensors | Certified | Exact fallback | Sample bytes | Fallback share | Count tables | ONNX identical |
|---|---|---|---|---|---|---|---|
| ResNet50, CLE | 74 | 17 | 57 | 2174678016 | 96.57% | 38797312 | yes |
| ResNet50, no CLE | 74 | 18 | 56 | 2174678016 | 96.28% | 38797312 | yes |
| YOLOv8n-cut, CLE | 218 | 33 | 185 | 7106560000 | 97.52% | 114294784 | yes |
| MODNet-Cut, CLE | 99 | 15 | 84 | 12714516480 | 99.30% | 51904512 | yes |

Storage was the hypothesis, and the conservative bound does not deliver it. Certified
tensors are a minority everywhere -- 23.0%, 24.3%, 15.1% and 15.2% of tensors -- and
they are the small ones, so the byte fractions are worse than the counts suggest.
Sample bytes avoidable before counting are 3.43%, 3.72%, 2.48% and 0.70%. A count table
is a fixed 524288 bytes per activation tensor whatever that tensor's size, costing
1.78%, 1.78%, 1.61% and 0.41% of the same totals, which leaves a net 1.64%, 1.94%,
0.87% and 0.29%.

The direction is the finding. Table overhead shrinks as activations grow, but avoidable
bytes shrink faster, so the method saves least on exactly the model whose spool is the
problem: MODNet-Cut spools 12714516480 bytes and gives back 36544512 of them, 0.29%.
Nothing here justifies replacing the ordered spool.

Evidence, legacy and dual per family: ResNet50 CLE
[legacy](../results/quant/alphabet_desktop2_20260909_c02_resnet50_cle_legacy_1.log),
[dual](../results/quant/alphabet_desktop2_20260909_c02_resnet50_cle_dual_1.log);
ResNet50 no-CLE
[legacy](../results/quant/alphabet_desktop2_20260909_c03_resnet50_no-cle_legacy_1.log),
[dual](../results/quant/alphabet_desktop2_20260909_c03_resnet50_no-cle_dual_1.log);
YOLOv8n-cut
[legacy](../results/quant/alphabet_desktop2_20260909_c03_yolov8n_cle_legacy_1.log),
[dual](../results/quant/alphabet_desktop2_20260909_c03_yolov8n_cle_dual_1.log);
MODNet-Cut
[legacy](../results/quant/alphabet_desktop2_20260909_c04_modnet_cle_legacy_1.log),
[dual](../results/quant/alphabet_desktop2_20260909_c04_modnet_cle_dual_1.log);
[certificate checks](../results/quant/check_alphabet_desktop2_20260909_02.log).

The MODNet-Cut pair was rerun under a fresh tag after a first attempt at its dual case
was interrupted 24 s in, with no process result recorded. Its fresh legacy run
reproduced the interrupted attempt's legacy output exactly, SHA-256
`71eb3f5f07ddce1facb47f5987fb735a64579ca3890c98f73bac7c848840bdb9` on both, so the
reference the byte comparison is made against is itself reproducible across runs.
An earlier ResNet50 comparison against a historical artifact failed file identity
despite identical initializers; graph node order differed. It is retained as a failed
comparison, not pooled with fresh same-producer pairs. Resource-guard failures and the
interrupted runs are likewise retained. The result index distinguishes completed checks
from these attempts.

### Exact-count calibration: what it costs

The four pairs above are `--checks-only` correctness runs and carry no timing claim.
The cost question is answered separately, by `alphabet` mode -- count tables first,
then an ordered replay for whatever fails to certify -- against legacy, on ResNet50
CLE, in five independently started blocks alternating which method goes first.

Every one of the ten runs emitted the same file, SHA-256
`eafed96b7096caed5a306a94ed1e6edcdae6adc81c4d13a3fb2abbc9ee8e46ab`, the same hash as
the correctness pair above; all five `alphabet` runs report `onnx_bytes_identical`.
That is the only correctness gate `alphabet` mode has, because unlike `dual` it never
reduces the certified tensors the legacy way, and its replay hash check only proves the
second inference pass reproduced the first.

Nine of the ten runs are timing-eligible. Block 5's `alphabet` run is excluded: the
monitor recorded PyCharm at 2.065 cores mid-run, which is contention, not a result.

| Method | Blocks used | Mean wall seconds | Standard deviation |
|---|---|---|---|
| Legacy ordered spool | 1-4 | 66.41 | 0.09 |
| Exact counts with fallback | 1-4 | 74.69 | 2.35 |

The paired differences are +10.15, +9.67, +8.22 and +5.10 seconds: `alphabet` mode is
slower in every block, by a mean 8.29 s, **1.125x**. Including block 5 anyway would
read 7.99 s and 1.120x, so the verdict does not depend on the exclusion. Peak working
set is unchanged, about 10.1 GB either way.

Those differences shrink monotonically across the matrix and nothing here explains it.
The method order alternates between blocks, so it is not an order effect; the legacy
runs are flat to 0.09 s, so it is not the machine warming up under both arms. Only the
`alphabet` side moves, from 76.62 s down to 71.43 s. It never comes close to changing
the sign in any block, so it does not affect the verdict, but it is unaccounted for.

The cost is structural, not incidental. 57 of 74 tensors fail to certify, so the second
inference pass runs nearly the whole model again -- 6.65 to 7.03 s of replay -- and the
ordered MinMSE reduction still has to cover 96.57% of the samples: 46.89 to 51.86 s
here, against the 52.998 s the same reduction takes over all 74 tensors in the `dual`
run above. Paying a full extra inference pass to skip 3.43% of a reduction is the
entire trade, and it loses.

**Verdict: rejected.** The certificate is sound, which is the result worth keeping, and
it held on four model families -- but a conservative interval that certifies a sixth to
a quarter of tensors, all of them small, buys 0.29% to 1.94% of spool bytes and costs
12% more wall time. Ignition's calibration is unchanged. A less conservative bound would
have to certify the large tensors to matter, and nothing here shows one exists.

Evidence: the ten `alphabet_desktop2_20260909_t01_resnet50_cle_{legacy,alphabet}_{1..5}`
logs under [`results/quant/`](../results/quant/), each carrying its own host timeline
and process result.

### Evidence and reproduction

The [checkpoint evidence archive](../results/quant/lowlevel_desktop2_20260909_evidence.zip)
contains raw arithmetic arrays, tiny QDQ models, placement reports, kernel result
arrays, function disassembly/bytes and final trace captures. Calibration sidecars
record the input listing and hashes; full generated model weights stay in ignored
scratch storage. The [archive validation log](../results/quant/summary_lowlevel_desktop2_20260909_01.log)
recounts NPU mismatches, intersects held-out candidates and checks channel permutations
from those arrays. It records unsuccessful attempts separately from completed cases.

From Git Bash, `bash scripts/research-matrix.sh <suite> <new_machine_date_tag>` runs a
bounded sequential matrix. Suites are `memory`, `gemm`, `arithmetic`,
`arithmetic-boundaries`, `calibration-dual` and `calibration-time`; see `--help` for
asset paths and selections. Existing output directories and log names are refused.
The AIE wrapper activates the existing ironenv and uses a private checkout-local JIT
cache. The parent runs in activated `resnet_env17`. Every NPU fixture uses a fresh
private compile cache. `--checks-only` records correctness evidence while explicitly
disqualifying performance claims. No NPU test here substitutes CPU placement evidence.

