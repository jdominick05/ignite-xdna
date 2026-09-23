# Ryzen AI XDNA1 NPU quantization pipelines

INT8 quantization and NPU deployment for **AMD Hawk Point / Phoenix** (Ryzen 8040 / 8000G,
XDNA1, 16 TOPS) on Windows, using Quark and the VitisAI ONNX Runtime execution provider.
Pipelines spanning classification, detection, pose, depth, and super-resolution.

**This is a hardware-characterization study, not a packaged tool.** Everything is
reproducible end to end, but the deliverable is measurements: every number below is
backed by a log under [`results/`](results/README.md). Which of its XDNA1 facts are new and which
were already public is [audited against prior art](results/aie/notes_tnzr_cross_audit.md#d-novelty).

## Does this run on your machine?

Most of this narrowness is not optional.

| | Required | If you don't have it                                                                                                                                                                                                            |
|---|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Chip | **Hawk Point or Phoenix** (XDNA1, `AMD_AIE2_4x4_Overlay`, provider option `target: "X1"`) | Strix (XDNA2) is a different architecture and none of the firmware paths here apply; it has not been touched                                                                                                                    |
| OS | **Windows** | Linux untested.                                                                                                                                                                                                                 |
| SDK | **Ryzen AI 1.7.1** for inference, 1.8.0 for export/quantize | 1.8.0 ships no Phoenix xclbin at all and cannot run inference on this chip                                                                                                                                                      |
| Shell | PowerShell, and Git Bash for `scripts/` | cmd prints `%VAR%` back instead of erroring on an unset variable                                                                                                                                                                |
| Workload | **CNN INT8 only** | No BF16, no transformer/NLP paths, no LLMs through the shipped runtime. That is a vendor-level limit, not a configuration problem — though the silicon itself is a different question, see [kernels](#hand-written-aie-kernels) |

Full environment split, install steps and footguns: [`docs/SETUP.md`](docs/SETUP.md).

## What works

| Pipeline | Model | Status |
|---|---|---|
| `pipelines/resnet50` | timm `resnet50.a1_in1k` | **Working** — 79.80% top-1 at 5.27 ms on the NPU (XINT8+AdaRound) |
| `pipelines/yolov8n` | YOLOv8 n/s/m/l/x detection | **Working** — 8.94–117.11 ms on the NPU once the decode tail is cut off the graph |
| `pipelines/yolov8n-pose` | YOLOv8n-pose, 17-point COCO keypoints | **Working** — 9.35 ms on the NPU (1015/1025 nodes), OKS mAP@50-95 34.32 AdaRound vs 32.64 plain XINT8 and 49.86 float (5000 images); the graph engine reads 44.16 on the same 5000 |
| `pipelines/yolov6n` | YOLOv6n detection (RepVGG backbone, no DFL) | **Working** — 6.62 ms on the NPU (518/525 nodes), mAP@50-95 33.57 AdaRound vs 22.92 plain XINT8 and 36.95 float (5000 images) |
| `pipelines/yolov11` | YOLOv11n detection (C3k2 + C2PSA attention) | **Fractured (6/1300 nodes, 33.29 ms)** on AMD's stack due to C2PSA 4D MatMul; ablated backbone runs at **7.08 ms (1173/1180 nodes)**. The graph engine runs it with only the attention core on the CPU, at 34.63 mAP@50-95 against AMD's 25.82 and 2.17x its speed on the container benchmark ([carve](docs/BENCHMARKS.md#yolo11ns-carve-the-narrow-one-is-free-on-accuracy-074-ms-faster-and-unlocks-883-map-2026-09-21-desktop-2)) |
| `pipelines/yolow` | YOLO-World v2 open-vocabulary (cross-attention) | **Fractured on AMD's stack (48/1081 nodes, 103.31 ms)** due to 5D Einsum/ReduceMax; ablated backbone runs at 15.89 ms (946/953 nodes). On the graph engine everything but the four text attention blocks runs on the NPU (63 of 67 convolutions), at 24.7% mAP on the first 300 val images after GPTQ on four convolutions, with class names chosen at run time ([how](docs/BENCHMARKS.md#yolo-world-v2-with-only-its-text-attention-on-the-cpu-gptq-and-an-int32-bias-recover-the-four-output-convolutions-2026-09-16-desktop-2)); frame to detections that is 2.75× AMD's stack at equal accuracy, and against the iGPU running FP32 at 43.0% faster with 80 classes but slower with five ([sitting](docs/BENCHMARKS.md#yolo-world-v2-glass-to-glass-275-times-amds-stack-and-against-the-igpu-it-depends-on-the-vocabulary-2026-09-16-desktop-2)), at 4.3–5.0× less energy per frame than AMD's stack flat out and 3.7× at 5 fps ([energy](docs/BENCHMARKS.md#yolo-world-v2-energy-per-frame-43-50-times-less-than-amds-stack-and-the-igpu-spends-less-at-5-fps-2026-09-17-desktop-2)) |
| `pipelines/{midas,fastdepth}` | MiDaS / FastDepth (monocular depth) | **Working** — FastDepth 2.87 ms (beats 3.02 ms iGPU, r=0.9383); MiDaS 10.81 ms (1.53x CPU, r=0.8706) |
| `pipelines/{sesr,realesrgan}` | SESR / Real-ESRGAN (super-resolution) | **Working** — SESR-M7 1.48 ms (beats 4.48 ms iGPU); Real-ESRGAN 14.02 ms (beats 18.34 ms iGPU) |
| `pipelines/bisenetv2` | BiSeNetV2 (bilateral segmentation) | **Working** — 13.12 ms (beats 14.25 ms iGPU, monolithic 402/404 nodes, 76.2 fps) |
| `pipelines/mobilevit` | MobileViT-XXS (hybrid CNN/transformer) | **Does not survive INT8** — 0.00% top-1, kept as the negative result |
| `quant/` (Ignition Alpha, not the SDK below) | Folded ResNet, head-cut YOLOv8n and MODNet, with or without CLE, plus AdaRound on all three | **Alpha 0.1.0a1** — independent calibration/emission without Quark or torch; `--cle` reproduces the repo's plain-XINT8 ResNet50 to the integer and a fresh yolov8n-cut oracle position for position, and `adaround` reproduces Quark's `XINT8_ADAROUND` byte for byte on both graphs on the same machine (the layer order is the vendor's topological sort, not the file's). MODNet matches a fresh oracle to the integer too, with identical matte error on both providers, and its AdaRound — the first family here where a finetuned layer carries an activation, so the first to need the vendor's `Clip` module rather than the `ReLU6` that names the same curve — is byte-identical as well, cutting matte error 45 percent on CPU and 47 percent on the NPU against the same listing quantized plain. [Quickstart](quant/README.md), [todo list](quant/TODO.md), [validation](docs/BENCHMARKS.md#ignition-alpha-release-validation), [CLE parity](docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset), [AdaRound parity](docs/BENCHMARKS.md#ignition-adaround-parity), [YOLO preparation](docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity), [YOLO AdaRound](docs/BENCHMARKS.md#ignition-yolov8n-cut-adaround-parity), [MODNet](docs/BENCHMARKS.md#ignition-modnet-independent-calibration-and-paired-matte-evaluation), [MODNet AdaRound](docs/BENCHMARKS.md#ignition-modnet-adaround-parity), [acceptance findings](docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance) |

Detection width sweep, head-cut plain XINT8, full 5000-image val2017 mAP
(conf 0.001, IoU 0.7, max_det 300, per-class NMS):

| Model | NPU latency | mAP@50-95 | mAP@50 | NPU nodes | Calibration images |
|---|---|---|---|---|---|
| yolov8n | **8.94 ms** | 26.94 | 40.15 | 922 / 929 | 200 |
| yolov8s | **15.63 ms** | 37.40 | 53.13 | 922 / 929 | 200 |
| yolov8m | **26.95 ms** | 43.38 | 59.62 | 1216 / 1223 | 200 |
| yolov8l | **49.67 ms** | 45.37 | 62.34 | 1510 / 1517 | 32 |
| yolov8x | **117.11 ms** | 45.09 | 61.37 | 1510 / 1517 | 24 |

> **These rows are not like-for-like.** Calibration count drops across the table
> (~1–1.5 GB activations/image at 640²), so l (32) and x (24) are calibrated thinner than
> n/s/m (200). Width dominates accuracy: yolov8m shifted mAP by only -0.11 (43.49 → 43.38)
> vs calib 64. l's full eval is flaky (two of three attempts hit hardware `DPU timeout`).
> Working: [Model size: n vs s](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together).

## Headline findings

**A float decode tail makes the EP refuse the entire graph, not partition around it.**
Quantized YOLOv8n ran at 39.1 ms — CPU speed — and `vitisai_ep_report.json` showed why:
**0 of 965 nodes** on the NPU, no partition at all. Cutting the last 18 DFL/anchor nodes
out of the graph and decoding them in numpy instead gives **922 of 929 nodes on the NPU**
and 8.7–9.8 ms. The seven CPU nodes are only the input/output Q/DQ boundary.
[Working](docs/BENCHMARKS.md#the-yolov8n-blocker-and-how-it-was-solved).

**Silent CPU fallback is the failure mode to watch for.** The `[Vitis AI EP]` banner and
operator table print only during compilation, never on a cache load, so their absence
means nothing. `<cacheKey>/vitisai_ep_report.json` — written on every session build, read
by `tools/diag_ep.py` — is the only real evidence of placement, and placement alone is not correctness: [Ignition's probes](docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance) catch wrong NPU logits. A `_npu` suffix in a log name means the
EP was *requested*. A8W8 falls back silently (39.0 ms, CPU speed); so does A16W8 (0/394 nodes); so does batch 2.

**AdaRound recovers classification, but not detection.** ResNet50 loses 8.4 points of top-1 to
plain XINT8 (80.10% → 71.70%) and AdaRound buys back all but 0.3 of it (79.80%) at no measured
latency cost (5.26 vs 5.27 ms, back-to-back). On detection it barely moves: yolov8s **+2.58
mAP** (37.40 → 39.98) and yolov8m **+1.83** (43.49 → 45.32), nowhere near the ~90% recovery it
gets on classifiers. A 500-image slice first suggested 45.19 for yolov8s, which would have been
a different story — the full 5000 is the number to trust.
[Working](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together).

**Three ways to lose to a Zen4 CPU, all measured, all different.** (1) MobileNetV2 is too cheap to
be worth accelerating: 1.72 ms on plain CPU against 2.68 ms on a genuinely engaged NPU (347/349
nodes) and 3.19 ms on the iGPU — per-call dispatch cost has nothing to amortize against. (2)
`resnetv2_50x3_bit` is heavy enough (470.79 ms CPU) and still loses by 25% (588.58 ms), because
all 49 `InstanceNormalization` nodes fall to CPU and each transition pays a cross-EP hand-off;
only 1010/1271 nodes place. Every clean NPU win here shares BatchNorm-only normalization, which
folds into the preceding Conv at export. (3) MobileViT-XXS survives INT8 at **0.00% top-1** — a
total collapse, and AdaRound moves it only to 0.80%. Being compute-heavy is necessary but not
sufficient; the graph also has to be built from ops the EP has kernels for.
[Working](docs/BENCHMARKS.md#pushing-width-further-resnetv2_50x3_bit-and-a-second-way-to-lose-to-cpu).

**The iGPU on the same chip is a serious competitor.** yolov8n FP16 through DirectML runs at
9.9–10.5 ms for one `convert_float_to_float16` call and **zero mAP loss** (36.72 vs FP32's 36.69).
The NPU's best case is 6.8–6.9 ms — faster, but XINT8+AdaRound still costs **4.5 mAP points**
(32.19 vs 36.69) after the recovery step, plus head-cutting, calibration and every footgun in
`docs/DECISIONS.md`. The NPU is worth it when the last 30–40% of latency matters more than 4.5 mAP
and the engineering time to chase it — on AMD's stack; the next finding asks it of the engine. [Working](docs/BENCHMARKS.md#igpu-vs-npu-is-ryzen-ai-worth-it-over-directml).

**The graph engine is more accurate than AMD's stack, not just faster.** Most of the zoo's XINT8
loss is Quark rendering SiLU as a HardSigmoid. Computed instead as a four-line integer sigmoid on
the NPU, YOLOv8n, YOLOv8s and YOLOv8n-pose read **34.12 / 42.37 / 44.16** mAP@50-95 over all 5000
COCO images against AMD's **26.68 / 37.31 / 32.64**, with no AdaRound — and YOLOv8n and pose are
faster too (8.50 vs 10.66 ms, 9.28 vs 12.07). AMD stays ahead on YOLOv8s and SESR M7.
[Accuracy](docs/BENCHMARKS.md#the-sigmoid-silu-containers-against-amds-stack-more-accurate-on-all-5000-coco-images-faster-on-yolov8n-and-yolov8n-pose-slower-on-yolov8s-2026-09-17-desktop-2) · [sitting](docs/BENCHMARKS.md#the-v033-release-sitting-sigmoid-silu-containers-on-the-host-fast-paths-against-amds-stack-2026-09-17-desktop-2).

**Batching is unsafe; independent concurrency is not.** A static batch-2 export doesn't fail
cleanly — the EP takes 80/395 nodes, runs with a plausible latency, and **writes only slot 0**:
slot 1's logits are byte-identical across every input, a stale buffer rather than a
miscomputation. Two independent `InferenceSession`s on separate threads, by contrast, give
**1.8–1.9× combined throughput with zero cross-talk** (139.8 fps on yolov8n-pose), because one
small model leaves most of the array idle.
[Batching](docs/BENCHMARKS.md#batching-does-it-help-throughput) ·
[concurrency](docs/BENCHMARKS.md#two-cameras-does-independent-concurrency-work-where-batching-doesnt).

**The array tops out near 39% of its 16 TOPS nameplate.** Measured as `MACs × 2 × fps` off the
real on-NPU graph (`tools/estimate_tops.py`), not estimated: 6.6% for yolov8n solo, 21.2% for
yolov8l solo, and **39.3% for yolov8l split across four independent 1x4 columns** — the best
figure here. yolov8x re-exported at 1280², with 6.2× the MACs, lands on the same 39.3%, which
makes it look like a real per-column ceiling rather than unexhausted headroom; ORT's own
profiler puts dispatch at 0.1–0.7% of wall time, so it is a compiler/scheduling limit, not host
overhead. **Two earlier answers to this question are retracted**: the ~1.1 TOPS FLOPs-derived
estimate for yolov8n, and every %-of-nameplate figure derived from `xrt-smi`'s GOPS column,
which scales linearly with stream count while measured throughput stays flat.
[Working](docs/BENCHMARKS.md#achieved-opss-a-real-answer-to--of-16-tops-not-a-gops-estimate).

**Do not quote a FLOPs ratio as a latency prediction on this hardware.** yolov8s is 3.2×
the arithmetic of yolov8n for 1.75× the latency; yolov8m is 9.1× for 3.46×. The CPU-vs-NPU
speedup ratio grows with model size — 3.8× at n, 5.3× at s, 6.6× at m — because CPU cost
tracks FLOPs roughly linearly while the NPU absorbs extra width into idle lanes. But the
trend does break: l → x buys **no** accuracy (45.37 → 45.09) for 2.36× the latency.

## Quickstart

Two conda environments (`resnet_env` for 1.8.0 export/quantize; `resnet_env17` for 1.7.1 inference; full setup in [`docs/SETUP.md`](docs/SETUP.md)). Run from **Git Bash**:

```bash
./scripts/setup.sh          # one-time: export, fetch datasets, quantize
./scripts/resnet-bench.sh   # the ResNet50 table
./scripts/yolo-cut.sh       # YOLO on the NPU: cut, quantize, run, verify
./scripts/yolo-eval.sh      # COCO bbox mAP          ./scripts/pose-eval.sh  # OKS mAP
./scripts/diag.sh           # what the VitisAI EP actually took
```

Python runtime (`ignite_xdna.InferenceSession` on AMD Phoenix silicon):

```python
from ignite_xdna import InferenceSession
with InferenceSession.from_file("build/yolov8n_full.ignite") as s:   # ignite-compile --engine graph
    heads = s.run_yolo_monolithic(input_data)  # whole network on the NPU, every layer bit-exact
    preds = s.decode_yolo_predictions(heads)
```

`ignite-compile --engine graph` builds that container (ironenv); the older `build/yolov8n.ignite` runs one
conv layer per stage and carries no heads ([measured](docs/BENCHMARKS.md#whole-network-yolov8n-on-a-16-core-convolution-engine-every-layer-on-the-npu-bit-exact-2026-09-13-desktop-2)). The same engine runs YOLOv8s and SESR M7 ([model zoo](docs/MODEL_ZOO_BENCHMARKS.md)), and stock YOLO11n with only its attention core (two MatMuls and a softmax) on the CPU between two NPU dispatches: 10.46–10.50 ms against 36.8–38.3 ms on AMD's stack in one sitting ([sitting](docs/BENCHMARKS.md#the-v033-release-sitting-sigmoid-silu-containers-on-the-host-fast-paths-against-amds-stack-2026-09-17-desktop-2)), superseding the 10.1–10.4 against 34.5–36.4 first measured on 2026-09-15 ([how](docs/BENCHMARKS.md#yolo11ns-c2psa-convolutions-on-the-npu-only-its-attention-core-on-the-host-2026-09-15-desktop-2)). That carve also takes the sigmoid epilogue, which lifts it to 34.63 mAP@50-95 against AMD's 25.82 — 8.83 points over the container shipping today, and 0.46 ms faster than it ([carve](docs/BENCHMARKS.md#yolo11ns-carve-the-narrow-one-is-free-on-accuracy-074-ms-faster-and-unlocks-883-map-2026-09-21-desktop-2)). YOLO-World v2 runs the same way with four attention cores on the CPU, and `YoloWorldPipeline.set_classes` changes its class names on an open container in about 90 ms ([how](docs/BENCHMARKS.md#yolo-world-v2s-vocabulary-chosen-at-run-time-one-container-any-class-names-2026-09-16-desktop-2)).

Native graph stages can also share BOs with custom AIE kernels through
`ignite_xdna.runtime.KernelSplicer`. The Phoenix Conv2D/GroupNorm/Conv2D validation
checks every stage against serial execution. [Scope, measurements and usage](docs/BENCHMARKS.md#native-bo-kernel-splicing-2026-09-13-desktop-2).

## Where the detail lives

| Document | What it is for |
|---|---|
| `README.md` | This page: whether the repo is for you, and what was found |
| [`RESEARCH.md`](RESEARCH.md) | The question being asked, why it is worth asking, and what is still open |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Every measurement with its full working, caveats and superseded history |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Locked decisions, rejected approaches, and the traps behind each |
| [`docs/SETUP.md`](docs/SETUP.md) | Compatibility, the two environments, install, manual steps |
| [`results/`](results/README.md) | The logs themselves — the evidence for every number above |
| [`docs/MODEL_ZOO_BENCHMARKS.md`](docs/MODEL_ZOO_BENCHMARKS.md) | Per-model engine results: what compiles to a container, and what it measured |
| [`kernels/`](kernels/README.md) | Hand-written AIE kernels and what each one measured |
| [Ignition](https://github.com/jdominick05/Ignition) | The consumer SDK over this engine — a separate repository, and not `quant/` above |

## Hand-written AIE kernels

The XINT8-only ceiling above is the *shipped runtime's*, not the silicon's. AMD's own
`device.yaml` (bundled with the same 1.7.1 install) gives Phoenix's AIE2 tile spec directly:
`bfloat16xbfloat16: 128` MACs/cycle, `int16xint8: 128`, `int8xint8: 256`. The array natively does
bf16; Quark and the VitisAI EP just don't expose it. The open-source `mlir-aie`/Peano toolchain
does, natively on Windows, with no gated access — so `kernels/` holds bf16 and int8 kernels
written against the bare array. Outcomes, mostly negative and all measured:

- **bf16 GEMM is the first genuine NPU win in this project.** 2072.5 GFLOPS at 1024³
  against CPU bf16's 1161.2 — the NPU wins 1.18×–1.78× once M/N ≥ 1024, and loses at
  512³ (895.1 vs 1100.6). "`K ≥ 3072` fails" was the reduction loop accumulating in
  `dtype_out` instead of fp32; `--dtype_out f32` fixes it free, and **the win holds at a
  7B-model projection shape** (M2048/K4096/N4096, 1.33×) — but a real FFN hinges on
  `d_ff`'s factorization: Llama-2-7B (`d_ff=11008`) flips it to **1.10× CPU**, Mistral-7B
  (`d_ff=14336`) keeps **1.13× NPU**. `attention_bf16`'s own kernel never had this bug.
- **int8 GEMM wins too, but only with a tile bf16 couldn't fit — until it could.** At the
  default tile the NPU's headline dtype **loses** to CPU's own int8 kernel almost everywhere;
  `n=64` fits int8's half-size tiles for **4448–4607 GOPS**, a **1.10×–1.83× win at M ≥ 512,
  N ≥ 2048** (thin at K=N=4096; prefill loses). Single-buffering the C output tile frees bf16's
  missing 16 KB too: **2700 GFLOPS** at 2048×4096×4096, **1.89×** same-sitting CPU bf16; int8 gains 13%.
- **Int8 conv loses, and the op class is closed** — still, after a **3×** fix (register-resident
  accumulators, 116 → 350 GOPS): CPU wins **2.4×** on marginal rate, **12.75×** at 56×56.
- **bf16 attention for MobileViT loses by 71×–240×**, and its recorded diagnosis was
  wrong: not dispatch cost, but `attention_kernels.cc` never calling `aie::mmul` — 0.61 GFLOPS vs 895.
- **A bf16 GroupNorm beat the CPU on 33 of 49 nodes** of `resnetv2_50x3_bit` — and then
  the measured two-process handoff floor (789 µs–23.6 ms per call) erased all 33.
- **Go/no-go:** CPU time must exceed a *host-set* dispatch floor — **617 µs** one-shot IRON,
  **~531 µs** batched IRON, **36.7 µs** batched C++ (≈ pyxrt: it's the driver's). **1.80 GHz**.

See [`kernels/README.md`](kernels/README.md) and `results/aie/`.

## Repo layout

```
src/ignite_xdna/      Bare-metal AIE2 vector compute engine (compiler/ & runtime/)
npu/                  Shared library code & backwards-compatibility shims
pipelines/<name>/     1_export -> 2_fetch_data -> 3_quantize -> 4_run/detect/pose -> 5_eval_map
benchmarks/           Silicon benchmark suites & Vitis AI vs ignite-xdna comparison
kernels/ tools/       Hand-written AIE2 kernels (kernels/aie2/), disassemblers, auditors
scripts/ results/     Bash wrappers & tracked logs — the evidence for every number
models/  data/        Generated. Git-ignored, and expensive to regenerate
```

`models/`, `data/` and compile caches are git-ignored: large, machine-specific, and
reproducible from the steps above. Pass `--fresh` when the model or xclbin changes.

## Known limitations

One chip generation and one SDK version (Hawk Point/Phoenix via Ryzen AI 1.7.1); Windows
only; batch 1 only; the full-graph YOLOv8 model is deliberately left in a state the EP
refuses, as the control that makes the head-cut result meaningful; AdaRound at 640×640
needs more RAM than the 13.8 GB laptop has; yolov8l's full eval is flaky; NPU utilization
can't be read through standard Windows tooling; the study half has no unit-test suite and cannot
have one, its subject being the hardware, while the engine's `tests/` cover offline invariants only. The full list, with the measurement behind each: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#known-limitations).

## Acknowledgements

Adapted from AMD's Ryzen AI `CNN-examples/object_detection` samples. Models are timm's
`resnet50.a1_in1k`, `wide_resnet50_2.racm_in1k`, `wide_resnet101_2.tv2_in1k`, and Ultralytics YOLOv8.

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Measurements on Strix or with
more RAM/disk are especially useful. Changes must pass the syntax/import gate in
`CONTRIBUTING.md` and include logged measurements under `results/`.

## License

The whole repository, engine under `src/ignite_xdna/` included, is licensed under the [GNU Affero
General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later): use, study, modify and share it
freely, including commercially, but any version you distribute or run as a network service must also
be open-sourced under the AGPL. That follows from a dependency: `pipelines/yolov8n/1_export.py`
imports `ultralytics`, whose YOLOv8 code is AGPL-3.0. Model artifacts keep their own licenses and are
not redistributed (`models/` and `data/` are git-ignored); the few files copied from mlir-aie keep
AMD's notice and licence, and say so in their header.
