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
