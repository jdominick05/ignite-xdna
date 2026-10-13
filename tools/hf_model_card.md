---
license: agpl-3.0
library_name: onnx
pipeline_tag: object-detection
tags:
- ryzen-ai
- xdna
- npu
- amd
- int8
- xint8
- onnx
- yolov8
- yolo11
- yolo-world
- super-resolution
- portrait-matting
---

# ignite-xdna: models and NPU containers for AMD XDNA1

These are the quantized models and the compiled NPU containers from
[ignite-xdna](https://github.com/jdominick05/ignite-xdna), a bare-metal engine for the AIE2 array in AMD's
**XDNA1** NPU (Phoenix and Hawk Point; developed on a Ryzen 7 8700G and a Ryzen 5 8645HS) on **Windows**.
A `.ignite` container is the whole program the engine dispatches: the NPU instruction stream, the packed
weights and the kernel image. It runs through ignite-xdna's runtime or the
[Ignition](https://github.com/jdominick05/Ignition) SDK, **not** through ONNX Runtime's VitisAI EP.

The `*_xint8.onnx` files are AMD Quark XINT8 QDQ models, and the containers were compiled from them. The ONNX
files also run on AMD's own Ryzen AI 1.7.1 stack, which is the baseline the measurements compare against.

## Contents

| Folder | ONNX | Container | NPU only? | Upstream licence |
|---|---|---|---|---|
| `yolov8n/` | `yolov8n_cut_xint8.onnx` | `yolov8n_full.ignite` | yes: 0 host segments | AGPL-3.0 |
| `yolov8s/` | `yolov8s_cut_xint8.onnx` | `yolov8s.ignite` | yes: 0 host segments | AGPL-3.0 |
| `yolov8n-pose/` | `yolov8n-pose_cut_xint8.onnx` | `yolov8n_pose.ignite` | yes: 0 host segments | AGPL-3.0 |
| `yolo11n/` | `yolo11n_cut_xint8.onnx` | `yolo11n.ignite` | no: 1 host segment (the C2PSA attention core) | AGPL-3.0 |
| `yolo-world/` | `yolov8s-worldv2_cut_xint8_gptqcv2.onnx`, `yolow_text_encoder.onnx`, `yolow_text.npz` | `yolow.ignite` | no: 4 host segments (the text cross-attention) | AGPL-3.0, plus MIT (CLIP) |
| `sesr_m7/` | `sesr_m7_xint8.onnx`, `sesr_m7_fp32.onnx` | `sesr_m7.ignite` | yes: 0 host segments | Apache-2.0 |
| `resnet50/` | `resnet50_xint8_c64.onnx`, `test_resnet50_head.onnx` | `resnet50_head.ignite` (classifier head only) | yes: 0 host segments | Apache-2.0 |
| `modnet/` | `modnet_cut_xint8_calibfix.onnx` | `modnet_cut_dense.ignite` | **no: 24 host segments** | Apache-2.0 |
| `fixtures/` | inputs for ignite-xdna's test suite | | | per subfolder |

A "host segment" is a region the container declares for the CPU, run through ONNX Runtime once per frame. "0
host segments" means the whole network runs on the NPU. That is read from each container's own manifest, and
it is a claim about that file, not about the model family.

**`modnet_cut_dense.ignite` is not a whole-network NPU result, and not a speed result.** About half of its
segments are CPU regions. In ignite-xdna's measurements, the CPU-region compute alone exceeds AMD's whole frame
on the same model. It is published because its output is exact where AMD's stack is not.

**`resnet50_head.ignite` is the classifier head alone**: it takes the pooled 2048-channel features (declared
input `[1,2048,20,20]`) and returns 1000 logits. It is not a ResNet50 backbone. `resnet50_xint8_c64.onnx` is the whole network, for AMD's VitisAI EP.

Every folder has a `LICENSE.md` naming the upstream weights, their licence, and where that licence was read.
`manifest.json` lists every file with its sha256 and its evidence.

## How these were checked

Each `.ignite` here was compiled by ignite-xdna at commit `394fc3d`, with the flags in `manifest.json`. Each was
then run on a Phoenix NPU (Ryzen 7 8700G) with `tools/verify_engine_container.py`. That check compares every
layer tensor still readable after one dispatch with the integer reference built from the ONNX next to it,
byte for byte. All seven read **EXACT**:

| Container | Readable layers exact | Reused slots correct |
|---|---|---|
| `yolov8n_full` | 28 / 28 | 38 / 38 |
| `yolov8s` | 28 / 28 | 38 / 38 |
| `yolov8n_pose` | 37 / 37 | 38 / 38 |
| `yolo11n` | 31 / 31 | 60 / 60 |
| `yolow` | 29 / 29 | 41 / 41 |
| `sesr_m7` | 5 / 5 | 4 / 4 |
| `resnet50_head` | 1 / 1 | — |

The log is
[`results/hf/verify_published_containers_desktop2_20261013.log`](https://huggingface.co/datasets/jdominick05/ignite-xdna-results/blob/main/results/hf/verify_published_containers_desktop2_20261013.log).
Workspace reuse means a later layer overwrites some layers' tensors, so this checks the readable ones plus
every reused slot's final contents; it is not all-layers coverage. `modnet_cut_dense.ignite` is the
2026-09-19 container from `results/dense/compile_modnet_20260919.log`, which records this ONNX's sha256.
The verifier cannot lower dense host regions, so it was not re-run for this upload.

## Accuracy

All on Desktop 2 (Ryzen 7 8700G, Phoenix NPU). COCO protocol: conf 0.001, IoU 0.7, max_det 300, per-class NMS.
"AMD's stack" is the same ONNX through Ryzen AI 1.7.1's VitisAI EP.

| Model | Metric | Engine container | AMD's stack | Dataset | Log |
|---|---|---|---|---|---|
| YOLOv8n | mAP@50-95 / @50 | **34.12** / 48.78 | 26.68 / 39.73 | COCO val2017, 5,000 | `results/aie/silu_sigmoid_vs_amd/accuracy/eval_yolov8n_sigmoid_5000.log`, `eval_yolov8n_amd_5000.log` |
| YOLOv8s | mAP@50-95 / @50 | **42.37** / 58.70 | 37.31 / 53.08 | COCO val2017, 5,000 | `.../eval_yolov8s_sigmoid_5000.log`, `eval_yolov8s_amd_5000.log` |
| YOLOv8n-pose | OKS mAP@50-95 / @50 | **44.16** / 76.39 | 32.64 / 66.90 | COCO val2017, 5,000 | `.../eval_pose_sigmoid_5000.log`, `eval_pose_amd_5000.log` |
| YOLO11n | mAP@50-95 / @50 | **34.63** / 49.59 ¹ | 25.82 / 38.76 | COCO val2017, 5,000 | `results/aie/epilogue_yolo11n_yolow_20260921.log`; AMD: `results/eval_yolo11n_cut_xint8_npu.log` |
| SESR M7 (x2) | PSNR-Y, dB | **34.25** Set5 / **29.41** Set14 ² | 34.06 / 29.32 | Set5, Set14 | `results/aie/sesr_m7_bf16_silicon_desktop2_20260922.log`; AMD: `results/eval_sesr_m7_set{5,14}_npu.log` |

¹ Copied into that log as a summary table of the full-set run. The raw pycocotools output is not in the log.
² Identical to the same XINT8 model on ONNX Runtime CPU (34.2492 / 29.4098), as an exact engine should be.

The engine reads higher than AMD's stack on the YOLO models mainly because of `--silu-sigmoid`. AMD's XINT8
model replaces each SiLU's sigmoid with a straight-line HardSigmoid, and the flag computes a closer integer
sigmoid on the NPU instead. Without the flag the containers read like AMD's stack (YOLOv8n 27.10, YOLOv8s 37.21,
pose 32.71, YOLO11n 25.80).

**Which build was measured.** These figures were measured on earlier compiles of these same ONNX files with
the same flags (YOLOv8n/s and pose at ignite-xdna `7700316` on 2026-09-17, YOLO11n on 2026-09-21), not on the
files uploaded here. The uploaded files were checked layer-exact against the engine's integer reference (above).
But the compiler's IR and reference code changed between the two builds (`7700316..394fc3d`), and the mAP was
not re-run on the uploaded files. That is why this card carries no `model-index` metadata: Hugging Face would
present these figures as measurements of these files, and they are not.

**Other families, and why they are not in the table:**
- **ResNet50** (`resnet50_xint8_c64.onnx` on AMD's VitisAI EP): top-1 72.10 / top-5 88.10 on the repo's
  **1,000-image** ImageNet-val evaluation set, not the full 50,000 (`results/res/run_resnet50_npu.log`).
  `resnet50_head.ignite` has no accuracy figure.
- **YOLO-World:** only a 300-image slice exists (24.7 mAP@50-95, `results/aie/yolow_gptq/eval_gptqcv2_ignite300.log`),
  and slices of this model read higher than the full set.
- **MODNet:** not a benchmark-standard matting set.
  - Labelled: against COCO person masks on all 2,693 val2017 images with a person, the engine scores 0.1609
    person IoU, matching the XINT8 CPU reference to every digit. AMD's stack scores 0.1700 and FP32 scores
    0.5030. XINT8 itself costs the model most of its quality, and neither stack wins
    (`results/dense/acc_modnet_cut_*_20260921.log`).
  - Agreement with FP32 on 50 unlabelled images: mean alpha MAD 0.1486 for the engine, 0.1672 for AMD's stack
    (`results/dense/agreement_summary_20260921.log`). On those 50 images the engine equalled ONNX Runtime CPU
    every time, and AMD's stack never did.

## Latency

This card makes no latency claim. The engine's timings depend on the compiler version, the power mode and the
machine, and they drift between sessions on the same machine. ignite-xdna's
[`docs/BENCHMARKS.md`](https://github.com/jdominick05/ignite-xdna/blob/main/docs/BENCHMARKS.md) and
[`docs/MODEL_ZOO_BENCHMARKS.md`](https://github.com/jdominick05/ignite-xdna/blob/main/docs/MODEL_ZOO_BENCHMARKS.md)
give every measured figure with its machine, build and log. Those figures were measured on earlier compiles of
these same ONNX files, not on these exact containers.

## Use

Requirements: an XDNA1 NPU on Windows with AMD's NPU driver and XRT, and ignite-xdna installed in its
`mlir-aie-iron` environment (see its `docs/SETUP.md`).

```python
from ignite_xdna import InferenceSession

with InferenceSession.from_file("yolov8n/yolov8n_full.ignite") as s:
    heads = s.run_yolo_monolithic(input_data)      # every layer on the NPU
    preds = s.decode_yolo_predictions(heads)
```

To put everything where ignite-xdna's tests and pipelines expect it (`models/` and `build/`), pinned and
checked by sha256, run `python tools/fetch_hf_artifacts.py` from a checkout.

To rebuild a container from its ONNX:

```bash
ignite-compile --engine graph --input yolov8n/yolov8n_cut_xint8.onnx --output yolov8n_full.ignite --silu-sigmoid
```

## Not here, and why

- **BiSeNetV2:** its weights are trained on Cityscapes, whose terms are non-commercial and restrict
  redistributing derivatives.
- **YOLO26n:** the only source ONNX on the build machine postdates its container, so the two could not be shown
  to pair.
- **Upstream `.pt`, `.pth` and `.ckpt` weights:** fetch them from their owners (Ultralytics, timm, AMD, ZHKKKe).
- **Calibration and evaluation data** (ImageNet, COCO, Cityscapes): never redistributed.

## Measurement logs

Every number in ignite-xdna's documents is backed by a log in
[jdominick05/ignite-xdna-results](https://huggingface.co/datasets/jdominick05/ignite-xdna-results).
