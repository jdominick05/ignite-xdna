# Ryzen AI XDNA1 NPU quantization pipelines

INT8 quantization and NPU deployment for **AMD Hawk Point / Phoenix** (Ryzen 8040 / 8000G,
XDNA1, 16 TOPS) on Windows, using Quark and the VitisAI ONNX Runtime execution provider.
Pipelines spanning classification, detection, pose, depth, and super-resolution.

**This is a hardware-characterization study, not a packaged tool.** Everything is
reproducible end to end, but the deliverable is measurements: every number below is
backed by a log under [`results/`](results/README.md), and the most useful part of the
repo is probably the set of XDNA1 facts that are undocumented or documented incorrectly.

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
| `pipelines/yolov8n-pose` | YOLOv8n-pose, 17-point COCO keypoints | **Working** — 9.35 ms on the NPU (1015/1025 nodes), OKS mAP@50-95 34.32 AdaRound vs 32.64 plain XINT8 and 49.86 float (5000 images) |
| `pipelines/yolov6n` | YOLOv6n detection (RepVGG backbone, no DFL) | **Working** — 6.62 ms on the NPU (518/525 nodes), mAP@50-95 33.57 AdaRound vs 22.92 plain XINT8 and 36.95 float (5000 images) |
| `pipelines/midas` | MiDaS v2.1 Small (monocular depth) | **Working** — 10.81 ms on NPU (682/684 nodes, single subgraph), r = 0.8706 vs FP32 (50 scenes) |
| `pipelines/sesr` | SESR-M7 (2x super-resolution) | **Working** — 1.48 ms on NPU (50/52 nodes, single subgraph), 35.16 dB PSNR on Set5 (XINT8+AdaRound) |
| `pipelines/mobilevit` | MobileViT-XXS (hybrid CNN/transformer) | **Does not survive INT8** — 0.00% top-1, kept as the negative result |

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
> because disk was the blocker (~1–1.5 GB of spooled activations per calibration
> image at 640×640), so l (32) and x (24) are calibrated thinner than n/s/m (200).
> Width dominates the accuracy story far more than sample count does: yolov8m at
> 200 images shifted mAP by only -0.11 points (43.49 → 43.38) vs calib 64.
> l's full eval is also flaky — two of three 5000-image attempts hit a hardware `DPU timeout`.
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
by `tools/diag_ep.py` — is the only real evidence. A `_npu` suffix in a log name means the
EP was *requested*. A8W8 falls back silently (39.0 ms, CPU speed); so does A16W8
(0/394 nodes); so does batch 2.

**AdaRound recovers classification, but not detection.** ResNet50 loses 8.4 points of
top-1 to plain XINT8 (80.10% → 71.70%) and AdaRound buys back all but 0.3 of it (79.80%)
at no measured latency cost (5.26 vs 5.27 ms, back-to-back). On detection it barely moves: yolov8s **+2.58 mAP** (37.40 → 39.98) and
yolov8m **+1.83** (43.49 → 45.32), nowhere near the ~90% recovery it gets on classifiers.
A 500-image slice first suggested 45.19 for yolov8s, which would have been a different
story — the full 5000 is the number to trust.
[Working](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together).

**Three ways to lose to a Zen4 CPU, all measured, all different.** (1) MobileNetV2 is too
cheap to be worth accelerating: 1.72 ms on plain CPU against 2.68 ms on a genuinely
engaged NPU (347/349 nodes) and 3.19 ms on the iGPU — per-call dispatch cost has nothing
to amortize against. (2) `resnetv2_50x3_bit` is heavy enough (470.79 ms CPU) and still
loses by 25% (588.58 ms), because all 49 `InstanceNormalization` nodes fall to CPU and
each transition pays a cross-EP hand-off; only 1010/1271 nodes place. Every clean NPU win
here shares BatchNorm-only normalization, which folds into the preceding Conv at export.
(3) MobileViT-XXS survives INT8 at **0.00% top-1** — a total collapse, and AdaRound moves
it only to 0.80%. Being compute-heavy is necessary but not sufficient; the graph also has
to be built from ops the EP has kernels for.
[Working](docs/BENCHMARKS.md#pushing-width-further-resnetv2_50x3_bit-and-a-second-way-to-lose-to-cpu).

**The iGPU on the same chip is a serious competitor.** yolov8n FP16 through DirectML runs
at 9.9–10.5 ms for one `convert_float_to_float16` call and **zero mAP loss** (36.72 vs
FP32's 36.69). The NPU's best case is 6.8–6.9 ms — faster, but XINT8+AdaRound still costs
**4.5 mAP points** (32.19 vs 36.69) after the recovery step, plus head-cutting,
calibration and every footgun in `docs/DECISIONS.md`. The NPU is worth it when the last
30–40% of latency matters more than 4.5 mAP and the engineering time to chase it.
[Working](docs/BENCHMARKS.md#igpu-vs-npu-is-ryzen-ai-worth-it-over-directml).

**Batching is unsafe; independent concurrency is not.** A static batch-2 export doesn't
fail cleanly — the EP takes 80/395 nodes, runs with a plausible latency, and **writes only
slot 0**: slot 1's logits are byte-identical across every input, a stale buffer rather
than a miscomputation. Two independent `InferenceSession`s on separate threads, by
contrast, give **1.8–1.9× combined throughput with zero cross-talk** (139.8 fps on
yolov8n-pose), because one small model leaves most of the array idle.
[Batching](docs/BENCHMARKS.md#batching-does-it-help-throughput) ·
[concurrency](docs/BENCHMARKS.md#two-cameras-does-independent-concurrency-work-where-batching-doesnt).

**The array tops out near 39% of its 16 TOPS nameplate.** Measured as `MACs × 2 × fps`
off the real on-NPU graph (`tools/estimate_tops.py`), not estimated: 6.6% for yolov8n
solo, 21.2% for yolov8l solo, and **39.3% for yolov8l split across four independent 1x4
columns** — the best figure here. yolov8x re-exported at 1280², with 6.2× the MACs, lands
on the same 39.3%, which makes it look like a real per-column ceiling rather than
unexhausted headroom; ORT's own profiler puts dispatch at 0.1–0.7% of wall time, so it is
a compiler/scheduling limit, not host overhead. **Two earlier answers to this question are
retracted**: the ~1.1 TOPS FLOPs-derived estimate for yolov8n, and every %-of-nameplate
figure derived from `xrt-smi`'s GOPS column, which scales linearly with stream count while
measured throughput stays flat.
[Working](docs/BENCHMARKS.md#achieved-opss-a-real-answer-to--of-16-tops-not-a-gops-estimate).

**Do not quote a FLOPs ratio as a latency prediction on this hardware.** yolov8s is 3.2×
the arithmetic of yolov8n for 1.75× the latency; yolov8m is 9.1× for 3.46×. The CPU-vs-NPU
speedup ratio grows with model size — 3.8× at n, 5.3× at s, 6.6× at m — because CPU cost
tracks FLOPs roughly linearly while the NPU absorbs extra width into idle lanes. But the
trend does break: l → x buys **no** accuracy (45.37 → 45.09) for 2.36× the latency.

## Quickstart

Two conda environments, because the SDK version that can export and quantize is not the
one that can run on the NPU:

```powershell
conda create -n resnet_env   --clone ryzen-ai-1.8.0    # export + quantize (torch, timm, quark)
conda create -n resnet_env17 --clone ryzen-ai-1.7.1    # all NPU inference
```

Then, from **Git Bash** (not WSL — a WSL run is silently CPU-only). Every script takes
`--help`, handles conda activation and logging, and writes UTF-8 logs to `results/`:

```bash
./scripts/setup.sh          # one-time: export, fetch datasets, quantize
./scripts/resnet-bench.sh   # the ResNet50 table
./scripts/yolo-cut.sh       # YOLOv8 on the NPU: cut, quantize, run, verify
./scripts/yolo-eval.sh      # COCO bbox mAP          ./scripts/pose-eval.sh  # OKS mAP
./scripts/yolo-demo.sh      # live webcam detection, q to quit
./scripts/diag.sh           # what the VitisAI EP actually took
```

Everything else — the environment split in full, manual per-step commands, and the
footguns that cost the most time here — is in [`docs/SETUP.md`](docs/SETUP.md).

## Where the detail lives

| Document | What it is for |
|---|---|
| `README.md` | This page: whether the repo is for you, and what was found |
| [`RESEARCH.md`](RESEARCH.md) | The question being asked, why it is worth asking, and what is still open |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Every measurement with its full working, caveats and superseded history |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Locked decisions, rejected approaches, and the traps behind each |
| [`docs/SETUP.md`](docs/SETUP.md) | Compatibility, the two environments, install, manual steps |
| [`results/`](results/README.md) | The logs themselves — the evidence for every number above |
| [`kernels/`](kernels/README.md) | Hand-written AIE kernels and what each one measured |

## Hand-written AIE kernels

The XINT8-only ceiling above is the *shipped runtime's*, not the silicon's. AMD's own
`device.yaml` (bundled with the same 1.7.1 install) gives Phoenix's AIE2 tile spec
directly: `bfloat16xbfloat16: 128` MACs/cycle, `int16xint8: 128`, `int8xint8: 256`. The
array natively does bf16; Quark and the VitisAI EP just don't expose it. The open-source
`mlir-aie`/Peano toolchain does, natively on Windows, with no gated access — so
`kernels/` holds bf16 and int8 kernels written against the bare array.

Outcomes, mostly negative and all measured:

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
- **Int8 conv loses, and the op class is closed.** NPU marginal throughput 146.1 GOPS
  against the CPU's 819.0; at ResNet50's real 56×56 conv2_x shape the CPU wins **12.75×**.
- **bf16 attention for MobileViT loses by 71×–240×**, and its recorded diagnosis was
  wrong: not dispatch cost, but `attention_kernels.cc` never calling `aie::mmul` — 0.61 GFLOPS vs 895.
- **A bf16 GroupNorm beat the CPU on 33 of 49 nodes** of `resnetv2_50x3_bit` — and then
  the measured two-process handoff floor (789 µs–23.6 ms per call) erased all 33.
- **Go/no-go before writing any kernel:** the op's CPU time must exceed the measured
  dispatch floor, **617.0 µs** IRON / **169.8 µs** hardware. Core clock: **1.80 GHz** (0.80 in powersaver).

See [`kernels/README.md`](kernels/README.md) and `results/aie/`.

## Repo layout

```
npu/                  Shared library code. MUST NOT import Quark
pipelines/<name>/     1_export -> 2_fetch_data -> 3_quantize -> 4_run/detect/pose -> 5_eval_map
                      1b_cut_head / 3b_quantize_cut are the head-cut variant
tools/                diag_ep.py, estimate_tops.py, the *_bench.py harnesses
kernels/              Hand-written mlir-aie/IRON kernels; run from the ironenv, not resnet_env17
scripts/              Bash wrappers: env activation, logging, disk guards
results/              Tracked logs — the evidence for every number in the docs
models/  data/        Generated. Git-ignored, and expensive to regenerate
```

`models/`, `data/` and the `*cachekey/` compile caches are git-ignored: large,
machine-specific, and reproducible from the steps above. A compile cache is keyed by name
rather than by model hash, so pass `--fresh` whenever the model or xclbin changes.

## Known limitations

One chip generation and one SDK version (Hawk Point/Phoenix via Ryzen AI 1.7.1); Windows
only; batch 1 only; the full-graph YOLOv8 model is deliberately left in a state the EP
refuses, as the control that makes the head-cut result meaningful; AdaRound at 640×640
needs more RAM than the 13.8 GB laptop has; yolov8l's full eval is flaky; NPU utilization
can't be read through standard Windows tooling; no unit-test suite (the hardware cannot
be faked). The full list, with the measurement behind each: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#known-limitations).

## Acknowledgements

Quantization configuration and the AdaRound parameter set are adapted from AMD's Ryzen
AI `CNN-examples/object_detection` samples. Models are timm's `resnet50.a1_in1k`,
`wide_resnet50_2.racm_in1k` and `wide_resnet101_2.tv2_in1k`, and Ultralytics YOLOv8.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Measurements on
hardware not available here are especially useful: Strix, more RAM for the
AdaRound-blocked configurations, more disk for yolov8l/x. Changes must pass the syntax
and import gate in `CONTRIBUTING.md` and come with a logged measurement under `results/`
for anything behavioural — there is no unit-test suite, because the hardware cannot be faked.

## License

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). You're
free to use, study, modify, and share this code — including commercially — but any
version you distribute, or run as a network-accessible service, must also be
open-sourced under the AGPL. This choice follows from a dependency, not a preference:
`pipelines/yolov8n/1_export.py` imports `ultralytics` directly, and Ultralytics'
YOLOv8 code is itself AGPL-3.0 — matching that license here removes the ambiguity of
combining AGPL and permissively-licensed code in one repo.

This covers the pipeline code only. Upstream model artifacts carry their own
licenses and are not redistributed here (`models/` and `data/` are git-ignored).
