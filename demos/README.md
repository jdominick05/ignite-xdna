# demos/

Interactive, reproducible benchmark demos illustrating the central architectural findings,
performance trade-offs, and failure modes of the AMD XDNA1 NPU on Phoenix / Hawk Point.

Every demo in this directory executes directly against the physical hardware in a single sitting,
following the repo's measurement invariants: `sess.run` is timed strictly alone as hardware
compute, preprocessing is byte-identical to calibration, compile caches are cleared between
graph variations, and all comparative figures are captured together to eliminate cross-session
latency drift (`CLAUDE.md`).

All demos run in `resnet_env17` with the Phoenix xclbin path configured:

```powershell
conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
```

## Index of Demos

| Script | Finding / Topic | Section it backs | Visual Output |
|---|---|---|---|
| [`width_ladder_demo.py`](width_ladder_demo.py) | **YOLO Width Scaling**: `yolov8n` -> `s` -> `m` -> `x` on NPU; sub-linear scaling ($3.3\times$ FLOPs for $1.88\times$ latency) and saturation past $m$ | [Model size: n vs s](../docs/BENCHMARKS.md#model-size-n-vs-s-measured-together) | `results/width_ladder_{n,s,m,x}_npu.jpg` |
| [`adaround_diff_demo.py`](adaround_diff_demo.py) | **AdaRound False-Positive Suppression**: Bipartite IoU matching between plain XINT8 and AdaRound; reveals XINT8 over-confidence and score recalibration | [AdaRound at width](../docs/BENCHMARKS.md#adaround-at-width-does-it-recover-less-on-a-wider-model) | `results/adaround_diff_yolov8n_npu.jpg` |
| [`conf_sweep_demo.py`](conf_sweep_demo.py) | **Confidence Threshold Ladder**: Sweeps conf from 0.001 to 0.90; visualises the 148 long-tail bounding boxes evaluated during mAP that demo thresholds discard | [Model size: n vs s](../docs/BENCHMARKS.md#model-size-n-vs-s-measured-together) | `results/conf_sweep_yolov8m_npu.jpg` |
| [`tri_hardware_showdown_demo.py`](tri_hardware_showdown_demo.py) | **Tri-Hardware Showdown**: Zen 4 CPU vs Radeon 780M iGPU (DirectML FP32/FP16) vs Phoenix NPU; NPU is $1.35\times$ faster than iGPU FP16, but iGPU preserves full FP32 accuracy | [iGPU vs NPU](../docs/BENCHMARKS.md#igpu-vs-npu-is-ryzen-ai-worth-it-over-directml) | `results/tri_hardware_showdown.jpg` |
| [`pose_adaround_demo.py`](pose_adaround_demo.py) | **Keypoint Regression Jitter & AdaRound**: Evaluates `yolov8n-pose` on NPU; quantifies 8.82 px joint drift on limbs and visualises stabilization vectors | [yolov8n-pose end to end on the NPU](../docs/BENCHMARKS.md#yolov8n-pose-end-to-end-on-the-npu) | `results/pose_adaround_diff_npu.jpg` |
| [`batch_failure_demo.py`](batch_failure_demo.py) | **Batch > 1 Silent Hardware Memory Corruption**: Evaluates static batch 2 ResNet50; demonstrates Slot 0 updates while Slot 1 is byte-identical ($\Delta = 0.0$) to previous calls | [Batching](../docs/BENCHMARKS.md#batching-does-it-help-throughput) | `results/batch_failure_npu.jpg` |
| [`resolution_ladder_demo.py`](resolution_ladder_demo.py) | **Fixed Dispatch Floor & Compute Knee**: ResNet50 from 128² to 384²; $3.07\times$ FLOPs scaling costs only $1.61\times$ latency below 224² before compute overtakes dispatch floor | [ResNet50 input resolution](../docs/BENCHMARKS.md#resnet50-input-resolution-does-the-fixed-cost-story-hold-for-a-classifier) | `results/resolution_ladder_npu.jpg` |
| [`classifier_width_demo.py`](classifier_width_demo.py) | **Classifier Width Generality**: ResNet50 vs Wide-ResNet50-2 vs Wide-ResNet101-2; confirms sub-linear scaling holds across CNN classifiers ($2.7\times$ params costs $1.64\times$ latency) | [Model width](../docs/BENCHMARKS.md#model-width-does-width-is-nearly-free-hold-for-a-classifier-too) | `results/classifier_width_npu.jpg` |
| [`webcam_multipartition_demo.py`](webcam_multipartition_demo.py) | **Multi-Partition Concurrency**: Live webcam round-robin across 4 independent `1x4.xclbin` column partitions | [A live demo](../docs/BENCHMARKS.md#a-live-demo-does-the-multi-partition-finding-hold-on-a-real-webcam) | — |
| [`portrait_matting_demo.py`](portrait_matting_demo.py) | **Real-Time Portrait Matting**: Live webcam matting (MODNet Zero-Concat XINT8 512x512) on NPU at 17.75 ms / 56.3 fps, 533/538 nodes — the fast variant, at 1.85x Cut's alpha error; bokeh blur, green-screen studio, EP cycling | [Category B: Real-Time Portrait Matting](../docs/BENCHMARKS.md#category-b-real-time-portrait-matting-modnet-on-xdna1-npu) | `results/modnet/000000001000_npu_composite.png` |

## Running the Demos

### 1. Model Width Scaling: YOLO Ladder
Runs `yolov8n`, `s`, `m`, and `x` sequentially on the NPU, flushing the compiled cache between variants:
```bash
python demos/width_ladder_demo.py --out-dir results/
```

### 2. AdaRound Diff & False-Positive Inspection
Runs plain XINT8 and XINT8+AdaRound side-by-side, performs bipartite IoU matching, and isolates spurious vs recovered bounding boxes:
```bash
python demos/adaround_diff_demo.py --variant n --out-dir results/
```

### 3. Confidence Threshold Ladder
Runs a single inference pass at low confidence threshold on the NPU, then sweeps thresholds in post-processing from $0.001$ to $0.90$:
```bash
python demos/conf_sweep_demo.py --variant m --out-dir results/
```

### 4. Tri-Hardware Showdown (CPU vs iGPU vs NPU)
Pits Zen 4 CPU, Radeon 780M iGPU (FP32 & FP16 via DirectML), and Phoenix XDNA1 NPU against each other in the exact same sitting:
```bash
python demos/tri_hardware_showdown_demo.py --out-dir results/
```

### 5. Pose Estimation Keypoint Jitter & AdaRound Stabilization
Evaluates `yolov8n-pose` on the NPU, measures per-joint Euclidean pixel drift ($\|p_{\text{adaround}} - p_{\text{xint8}}\|_2$), and renders skeleton overlays with displacement arrows:
```bash
python demos/pose_adaround_demo.py --out-dir results/
```

### 6. Negative Result: Batch > 1 Unwritten Stale Buffer
Feeds two distinct pairs of calibration images through static batch 2 ResNet50, demonstrating that the NPU hardware writes only Slot 0 and leaves Slot 1 as unwritten, stale RAM:
```bash
python demos/batch_failure_demo.py --out-dir results/
```

### 7. Fixed Hardware Dispatch Floor & Resolution Knee
Sweeps ResNet50 input resolutions from 128² to 384², exposing the $\sim 3.5\text{ ms}$ fixed AIE dispatch/DMA floor:
```bash
python demos/resolution_ladder_demo.py --out-dir results/
```

### 8. Classifier Width Scaling Generality
Evaluates ResNet50, Wide-ResNet50-2, and Wide-ResNet101-2 back-to-back on the NPU, validating that the spatial fabric absorbs width across model architectures:
```bash
python demos/classifier_width_demo.py --out-dir results/
```

### 9. Live Portrait Matting on Webcam
Runs MODNet **Zero-Concat** XINT8 at 512x512 on the physical NPU with live interactive controls:
```powershell
python demos/portrait_matting_demo.py --source 0
```
- **Key controls**: `b` (Bokeh blur), `[` / `]` (blur radius), `g` (Green screen), `m` (Raw alpha trimap), `s` (Split screen), `c` (Cycle NPU / iGPU / CPU), `p` (Save snapshot), `SPACE` (Pause), `q` (Quit).
- **It defaults to the fast, less accurate graph.** Zero-Concat runs 17.75 ms to Cut's
  26.44 ms, but its alpha error against the FP32 reference is 1.85x Cut's (MAD 0.35269 vs
  0.19022) — see [Category B](../docs/BENCHMARKS.md#4-same-sitting-re-measurement-with-logs-2026-09-07-desktop-2).
  Pass `--model models/modnet/modnet_cut_xint8.onnx` for the accurate one.
- **`p` writes a snapshot of whatever the camera sees** into `results/modnet/`. That
  directory is committed, so delete snapshots you don't want published — the ones from the
  original demo session were of a person and were kept out of the repo deliberately.
