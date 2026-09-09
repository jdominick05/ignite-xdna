# demos/

Interactive, reproducible benchmark demos illustrating the central architectural findings,
performance trade-offs, and failure modes of the AMD XDNA1 NPU on Phoenix / Hawk Point.

The eight non-interactive demos below execute directly against the physical hardware,
timing `sess.run` alone, clearing the compile cache between graph variants, and capturing
every figure in a comparison together so cross-session latency drift cannot leak into a
ratio (`CLAUDE.md`). The two webcam demos are interactive and need a person at the machine.

**Every figure in the table below is quoted from a log in
[`results/demos/`](../results/demos), captured 2026-09-07 on Desktop 2.** They were
previously quoted from a session that was never captured, and four of them did not
survive being re-run -- see [What changed when these were re-run](#what-changed-when-these-were-re-run).
Latency on this hardware drifts between sessions, so read each row as one sitting rather
than a constant.

All demos run in `resnet_env17` with the Phoenix xclbin path configured:

```powershell
conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
```

## Index of Demos

| Script | Finding / Topic | Section it backs | Visual Output |
|---|---|---|---|
| [`width_ladder_demo.py`](width_ladder_demo.py) | **YOLO width scaling**: `yolov8n` 6.63 ms, `s` 12.87, `m` 27.78, `x` 74.46 on NPU. n->s is 3.3x the FLOPs for 1.94x the latency, n->m 9.1x for 4.19x -- sub-linear at both steps (`demo_width_ladder_npu.log`) | [Model size: n vs s](../docs/BENCHMARKS.md#model-size-n-vs-s-measured-together) | `results/width_ladder_{n,s,m,x}_npu.jpg` |
| [`adaround_diff_demo.py`](adaround_diff_demo.py) | **AdaRound False-Positive Suppression**: Bipartite IoU matching between plain XINT8 and AdaRound; reveals XINT8 over-confidence and score recalibration | [AdaRound at width](../docs/BENCHMARKS.md#adaround-at-width-does-it-recover-less-on-a-wider-model) | `results/adaround_diff_yolov8n_npu.jpg` |
| [`conf_sweep_demo.py`](conf_sweep_demo.py) | **Confidence threshold ladder**: 148 of 171 boxes are present at conf 0.001 and gone by 0.25 -- the long tail mAP integrates over and a demo window never shows (`demo_conf_sweep_yolov8m_npu.log`) | [Model size: n vs s](../docs/BENCHMARKS.md#model-size-n-vs-s-measured-together) | `results/conf_sweep_yolov8m_npu.jpg` |
| [`tri_hardware_showdown_demo.py`](tri_hardware_showdown_demo.py) | **Tri-hardware showdown**: CPU 27.18 ms, DML FP32 8.60, DML FP16 6.08, NPU XINT8 6.59. **The iGPU FP16 beat the NPU on both axes in this sitting** -- 0.92x the latency and 36.72 vs 32.19 mAP -- reversing the 1.3-1.5x NPU lead in BENCHMARKS (`demo_tri_hardware_showdown.log`) | [iGPU vs NPU](../docs/BENCHMARKS.md#igpu-vs-npu-is-ryzen-ai-worth-it-over-directml) | `results/tri_hardware_showdown.jpg` |
| [`pose_adaround_demo.py`](pose_adaround_demo.py) | **Keypoint jitter and AdaRound**: `yolov8n-pose` on NPU, 8.82 px mean joint drift between plain XINT8 and AdaRound, drawn as displacement vectors (`demo_pose_adaround_npu.log`) | [yolov8n-pose end to end on the NPU](../docs/BENCHMARKS.md#yolov8n-pose-end-to-end-on-the-npu) | `results/pose_adaround_diff_npu.jpg` |
| [`batch_failure_demo.py`](batch_failure_demo.py) | **Batch > 1 returns a stale buffer**: static batch-2 ResNet50. The second call moves slot 0 by 19.75 and slot 1 by exactly 0.000000 -- unwritten RAM, returned with a plausible ~14 ms latency and no error (`demo_batch_failure_npu.log`) | [Batching](../docs/BENCHMARKS.md#batching-does-it-help-throughput) | `results/batch_failure_npu.jpg` |
| [`resolution_ladder_demo.py`](resolution_ladder_demo.py) | **Dispatch floor and compute knee**: ResNet50 128² to 384². Below 224², 3.07x the FLOPs costs 1.63x the latency (128² is 3.24 ms, ~80% of it fixed dispatch/DMA); above it 2.94x FLOPs costs 2.05x (`demo_resolution_ladder_npu.log`) | [ResNet50 input resolution](../docs/BENCHMARKS.md#resnet50-input-resolution-does-the-fixed-cost-story-hold-for-a-classifier) | `results/resolution_ladder_npu.jpg` |
| [`classifier_width_demo.py`](classifier_width_demo.py) | **Width generality across classifiers**: ResNet50 5.38 ms, Wide-ResNet50-2 8.94. 2.69x the parameters costs 1.66x the latency; the Wide-ResNet101-2 step is 4.96x params for 3.10x (`demo_classifier_width_npu.log`) | [Model width](../docs/BENCHMARKS.md#model-width-does-width-is-nearly-free-hold-for-a-classifier-too) | `results/classifier_width_npu.jpg` |
| [`webcam_multipartition_demo.py`](webcam_multipartition_demo.py) | **Multi-Partition Concurrency**: Live webcam round-robin across 4 independent `1x4.xclbin` column partitions | [A live demo](../docs/BENCHMARKS.md#a-live-demo-does-the-multi-partition-finding-hold-on-a-real-webcam) | — |
| [`portrait_matting_demo.py`](portrait_matting_demo.py) | **Real-Time Portrait Matting**: Live webcam matting (MODNet Zero-Concat XINT8 512x512) on NPU at 17.75 ms / 56.3 fps, 533/538 nodes — the fast variant, at 1.85x Cut's alpha error; bokeh blur, green-screen studio, EP cycling | [Category B: Real-Time Portrait Matting](../docs/BENCHMARKS.md#category-b-real-time-portrait-matting-modnet-on-xdna1-npu) | `results/modnet/000000001000_npu_composite.png` |

## Running the Demos

`./scripts/tui.sh` (or `python -m tui` from PowerShell) is a menu over everything
below, plus the models as tasks. It shows the command line before running it, warns
about the two traps in the next section, sends output to `outputs/` instead of
`results/`, and reads `<cacheKey>/vitisai_ep_report.json` afterwards so "did the NPU
actually take it" is answered rather than assumed. `python -m tui --selftest` checks
the whole catalogue without opening a hardware context.

### Two things to know before running any of these by hand

**`--ep cpu` still destroys the NPU compile cache.** Six of the eight non-interactive
demos call `clear_cache()` unconditionally rather than only for an NPU run --
`width_ladder:69`, `adaround_diff:44`, `conf_sweep:72`, `pose_adaround:50`,
`classifier_width:86`, `resolution_ladder:63`. Only `tri_hardware_showdown:92` guards
it with `if ep == "npu"`. So a quick CPU look costs a full recompile on the next NPU
run of that cache key. `batch_failure` clears unconditionally too, but it always runs
an NPU pass, so there is nothing to warn about there.

**`portrait_matting_demo.py` writes camera frames to disk in two places, and only one
of them is documented above.** The `p` key saves a snapshot, and the `t` (calibrate)
key writes **25 frames automatically** to `data/user_calib/`. `data/` is git-ignored
so those stay local, but snapshots went to `results/modnet/`, which *is* tracked and
holds images this page links. `--snapshot-dir` now exists to send them somewhere
else; the default is unchanged.


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
Runs a single inference pass at low confidence threshold on the NPU, then sweeps thresholds in post-processing from 0.001 to 0.90:
```bash
python demos/conf_sweep_demo.py --variant m --out-dir results/
```

### 4. Tri-Hardware Showdown (CPU vs iGPU vs NPU)
Pits Zen 4 CPU, Radeon 780M iGPU (FP32 & FP16 via DirectML), and Phoenix XDNA1 NPU against each other in the exact same sitting:
```bash
python demos/tri_hardware_showdown_demo.py --out-dir results/
```

### 5. Pose Estimation Keypoint Jitter & AdaRound Stabilization
Evaluates `yolov8n-pose` on the NPU, measures per-joint Euclidean pixel drift (the L2 distance between the AdaRound and plain-XINT8 joint positions), and renders skeleton overlays with displacement arrows:
```bash
python demos/pose_adaround_demo.py --out-dir results/
```

### 6. Negative Result: Batch > 1 Unwritten Stale Buffer
Feeds two distinct pairs of calibration images through static batch 2 ResNet50, demonstrating that the NPU hardware writes only Slot 0 and leaves Slot 1 as unwritten, stale RAM:
```bash
python demos/batch_failure_demo.py --out-dir results/
```

### 7. Fixed Hardware Dispatch Floor & Resolution Knee
Sweeps ResNet50 input resolutions from 128² to 384², exposing the fixed AIE dispatch/DMA floor (3.24 ms at 128² in the run logged here):
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

## What changed when these were re-run

Every figure above was originally quoted from a demo session whose stdout was never
written to `results/`, so none of it could be checked. Re-running all eight
non-interactive demos on 2026-09-07 (Desktop 2, Phoenix) produced the logs in
`results/demos/`. Two figures reproduced exactly, four moved, and one reversed.

| Claim as written | Re-run | Verdict |
|---|---|---|
| 148 long-tail boxes below conf 0.25 | 148 of 171 | reproduces exactly |
| 8.82 px mean keypoint drift | 8.82 px | reproduces exactly |
| n->s: 3.3x FLOPs for 1.88x latency | 1.94x | moved; session drift |
| 128^2->224^2: 3.07x FLOPs for 1.61x latency | 1.63x | moved; session drift |
| ~3.5 ms dispatch floor at 128^2 | 3.24 ms | moved; session drift |
| 2.7x params for 1.64x latency | 2.69x for 1.66x | moved; session drift |
| **NPU 1.35x faster than iGPU FP16** | **NPU 0.92x -- the iGPU won** | **reversed** |

The reversal is the one that matters. `docs/BENCHMARKS.md` records DirectML FP16 at
9.9-10.5 ms against the NPU's ~6.9 ms and concludes the iGPU's best case is 1.3-1.5x
slower. In this sitting DML FP16 ran at **6.08 ms** against the NPU's **6.59 ms**, and the
iGPU also held 36.72 mAP against the NPU's 32.19 -- it won on both axes. The old DML
number is what failed to reproduce, not the NPU one.

Two things stop this from settling the question. The DML rows time the **full graph**, so
decode happens inside the timed call, while the NPU row is **head-cut** and its decode is
excluded -- the NPU is flattered here, not penalised, which makes the loss harder to
explain away. Against that, this is a single sitting on a machine whose latency is
documented to drift, and the run was not repeated. Treat the published 1.3-1.5x lead as
**not reproduced** rather than retracted, and re-run both before quoting either.

`tri_hardware_showdown_demo.py` also printed "NPU wins by 0.92x" -- it called any ratio a
win. Fixed to name the direction, and to state the head-cut asymmetry in its own output.

## Reproducing the logs

```bash
python demos/batch_failure_demo.py          --out-dir results/
python demos/conf_sweep_demo.py --variant m --out-dir results/
python demos/adaround_diff_demo.py --variant n --out-dir results/
python demos/pose_adaround_demo.py          --out-dir results/
python demos/resolution_ladder_demo.py      --out-dir results/
python demos/classifier_width_demo.py       --out-dir results/
python demos/tri_hardware_showdown_demo.py  --out-dir results/
python demos/width_ladder_demo.py           --out-dir results/
```

Redirect each to `results/demos/<name>.log` to refresh the evidence; the `.jpg` artifacts
land in `results/` either way. The two webcam demos are excluded on purpose -- they need a
person at the machine, and `portrait_matting_demo.py` writes camera frames to disk.
