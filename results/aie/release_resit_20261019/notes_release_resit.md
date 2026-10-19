# Release re-sit against AMD's stack, 2026-10-19 (Desktop 2)

Desktop 2 (`DESKTOP-CBL5NUA`, Ryzen 7 8700G, Phoenix NPU `[003d:00:01.1]`), balanced power mode, `bus.jpg`, 50
warm-up + 500 timed frames per run, two interleaved rounds (round 2 in reverse order), `xrt-smi` idle before every
run. Ignition ran `live_ignition.py --headless` (Ignition `c16abb2`) on containers built by their recipes (compiler
`v0.1.0-phoenix-npu-717-gb58994f71373`, each manifest's `build` record names its recipe step). AMD's stack ran Ryzen AI
1.7.1 through `tools/amd_vitisai_yolo.py`, `tools/amd_vitisai_sesr.py` and `pipelines/yolov8n-pose/4_pose.py`, each
on fresh cache keys compiled by an untimed run first. Before any timed run, every container matched its reference on
every layer on the NPU (0 mismatches) and was checked to be compiled from the same quantized ONNX file AMD's arm runs.

| Log | What it is |
|---|---|
| [`release_resit_desktop2_20261019T151835Z.log`](release_resit_desktop2_20261019T151835Z.log) | First attempt, stopped before any timed run: the current runtime refuses `yolo11n_core_silu.ignite` (built 2026-09-20, no placement `dtype`). Led to rebuilding every container through its recipe. |
| [`release_resit_desktop2_20261019T160617Z.log`](release_resit_desktop2_20261019T160617Z.log) | The sitting, all nine models. |
| [`release_resit_clean_desktop2_20261019T162436Z.log`](release_resit_clean_desktop2_20261019T162436Z.log) | The clean re-sit of the four models whose runs overlapped operator activity (below). |

## Pre-registration

Written before any timed run (v2, sha256 `d80beac3b380c69682b8c4210bd5752df01a3f05f43aabc408ced3d34eed05aa`; a v1
written against the old containers was superseded before any timed run). Metric: each tool's glass-to-glass mean.
Rule: "Ignition faster" only if its mean is below the AMD arm's in **both** rounds, "AMD faster" only if above in
both, otherwise "level". AMD "tuned" (`enable_cost_model_tiling`) is a second AMD arm for YOLOv8n and YOLOv8s.

**Addendum, written after the first sitting and before the re-sit** (sha256
`21295b548f075283e29da23adf99edbb1d136e12f4aea710b2b9ad2c672ea185`):
1. **Overlap.** The operator ran file edits, link/word/badge checks, greps and git commits during some timed runs.
   Matching those commands' timestamps against each run's window: pose, YOLO26n, YOLOv8m, YOLOv8l and YOLOv8x were
   clean in every run; YOLOv8n's round-2 AMD runs, five of YOLOv8s's six runs, SESR M7's two AMD runs and all four
   YOLO11n runs overlapped. Those four models were re-sat in full, with nothing else run until it ended (0 overlapping
   commands), and the re-sit's verdicts replace the first sitting's whichever way they read. The overlapped numbers
   stay in the first log and are listed below as overlapped.
2. **Pose metric.** `4_pose.py` prints no glass-to-glass mean (only per-100-frame samples, an `infer` mean that is
   `session.run` alone, and a `pre` mean). Pose is therefore judged against AMD's `infer` mean, a lower bound on its
   frame time, and "Ignition faster" holds only if Ignition's whole frame is below it in both rounds.

## Verdicts

Mean ms per run, round 1 / round 2.

| Model | Ignition | AMD default | AMD tuned | Verdict | Source |
|---|---|---|---|---|---|
| YOLOv8n | 7.157 / 7.147 | 10.834 / 10.837 | 10.165 / 10.262 | **Ignition faster** than both | clean re-sit |
| YOLOv8s | 15.765 / 15.770 | 17.220 / 17.248 | 14.599 / 14.661 | **Ignition faster** than default; **AMD tuned faster** | clean re-sit |
| YOLOv8n-pose | 7.499 / 7.497 | ≥ 9.18 / 9.15 (`infer` only) | — | **Ignition faster** | sitting |
| SESR M7 | 4.464 / 4.541 | 3.974 / 4.255 | — | **AMD faster** | clean re-sit |
| YOLO11n | 9.850 / 9.878 | 36.976 / 36.216 | — | **Ignition faster** | clean re-sit |
| YOLO26n | 10.825 / 10.765 | 37.188 / 37.182 | — | **Ignition faster** | sitting |
| YOLOv8m | 39.182 / 39.230 | 31.950 / 31.950 | — | **AMD faster** | sitting |
| YOLOv8l | 69.960 / 69.970 | 51.470 / 51.473 | — | **AMD faster** | sitting |
| YOLOv8x | 112.900 / 112.998 | 78.927 / 78.996 | — | **AMD faster** | sitting |

Overlapped first-sitting runs of the re-sat models (superseded by the re-sit; same verdict in every case): YOLOv8n
7.108 / 7.100 against 10.857 / 10.936 (default) and 10.263 / 10.223 (tuned); YOLOv8s 15.744 / 15.747 against
17.215 / 17.204 and 14.705 / 14.655; SESR M7 4.611 / 4.617 against 4.388 / 4.156; YOLO11n 9.917 / 9.906 against
36.960 / 37.889.

## Where the time goes

| Model | Ignition stages (round 1, ms) | AMD stages (round 1, ms) | Detections, Ignition / AMD |
|---|---|---|---|
| YOLOv8n | preprocess 0.300, NPU 6.692 (dispatch 6.312), decode+NMS 0.158 | letterbox 1.766, `session.run` 6.884, decode+NMS 2.184 | 5 / 5 |
| YOLOv8s | preprocess 0.300, NPU 15.296 (dispatch 14.891), decode+NMS 0.160 | letterbox 1.799, `session.run` 13.221 (tuned 10.617), decode+NMS 2.200 | 5 / 5 |
| YOLOv8n-pose | preprocess 0.264, NPU 6.859 (dispatch 6.332), decode+NMS 0.370 | pre 3.148, `infer` 9.18 | 4 / 3 people |
| SESR M7 | preprocess 0.137, NPU 3.968, image output 0.354 | preprocess 0.344, `session.run` 1.469, postprocess 2.161 | image / image |
| YOLO11n | preprocess 0.257, NPU + host 9.438 (dispatch 8.457, host 0.685), decode+NMS 0.148 | letterbox 1.914, `session.run` 32.973, decode+NMS 2.089 | 5 / 7 |
| YOLO26n | preprocess 0.201, NPU + host 10.530 (dispatch 8.981, host 1.338), decode+NMS 0.089 | letterbox 1.905, `session.run` 33.988, decode+NMS 1.296 | 5 / 6 |
| YOLOv8m | preprocess 0.308, NPU 38.698 (dispatch 38.271), decode+NMS 0.167 | letterbox 1.753, `session.run` 27.996, decode+NMS 2.201 | 5 / 6 |
| YOLOv8l | preprocess 0.308, NPU 69.475 (dispatch 69.047), decode+NMS 0.169 | letterbox 1.778, `session.run` 47.485, decode+NMS 2.207 | 5 / 6 |
| YOLOv8x | preprocess 0.311, NPU 112.409 (dispatch 111.982), decode+NMS 0.171 | letterbox 1.770, `session.run` 74.898, decode+NMS 2.259 | 5 / 8 |

AMD's pre- and post-processing columns use Ignition's numpy reference code (pose: the pipeline's own), so the host
part of Ignition's lead comes from its native code, not from the NPU. On every model where AMD wins, its `session.run`
is shorter than Ignition's NPU stage. The detection counts differ between stacks on several models; whether that is
the SiLU approximation, the decode or the threshold path was not separated here (unexplained).
