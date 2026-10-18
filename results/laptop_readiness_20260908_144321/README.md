# Laptop readiness — 2026-09-08

Updated the Ryzen 5 8645HS / Radeon 760M / Hawk Point XDNA1 laptop from
`a39a60e` to `2f041ec21e3108172c7ec39279452a4ae1f2ceaa` (118 commits on main).
The local RESEARCH heading edit was backed up and restored. Existing environments
and driver required no changes. No export, quantization, or dataset download ran.

The syntax, shell, and shared-module import gate passed in `resnet_env17`, including
the new MODNet and YOLOv6 modules. Python syntax checks also covered `kernels/` and
`demos/`; neither was executed. See [checks_final.log](checks_final.log).

All four selected models completed CPU and fresh-cache NPU smoke runs sequentially.
Each CPU/NPU pair used the same quantized model. These are readiness checks, not
FP32 comparisons, full-dataset accuracy evaluations, or controlled benchmarks.

| Model | CPU mean (ms) | NPU mean (ms) | NPU nodes / total | Placement evidence |
|---|---:|---:|---:|---|
| ResNet50 XINT8 AdaRound | 39.68 | 5.54 | 393 / 395 | [report](diag_resnet50_xint8_adaround_npu.log) |
| YOLOv8n cut XINT8 AdaRound | 41.56 | 6.90 | 922 / 929 | [report](diag_yolov8n_cut_xint8_adaround_npu.log) |
| YOLOv8n pose cut XINT8 AdaRound | 47.66 | 8.93 | 1015 / 1025 | [report](diag_yolov8n_pose_cut_xint8_adaround_npu.log) |
| YOLOv6n cut XINT8 | 38.74 | 6.51 | 518 / 525 | [report](diag_yolov6n_cut_xint8_npu.log) |

Timings come from the corresponding `run_<model>_<ep>.log` files. ResNet used the
first 20 sorted images under `data/eval`, static batch 1; its 80% top-1 in both runs
is only a 20-image slice. Detection and pose used `assets/test_image.jpg`, a warm-up,
then 20 iterations at confidence 0.25 / IoU 0.5. Detection inference means
`sess.run` only; pose includes numpy decode. Power mode was not captured and timing
was not controlled for background load. Do not compare these timings to other machines.

Output images are included. YOLOv8n emitted five CPU detections and four NPU
detections at the chosen threshold; this check does not establish output equivalence.
Pose found one person with 12/17 visible keypoints on both providers. The NPU pose
and YOLOv6 output images were visually inspected for rendered output.

[environment.log](environment.log) records the commit, runtime versions, model/input
SHA-256 hashes, graph input shapes, xclbin, driver, and available disk. The repeated
[environment_after.log](environment_after.log) confirms that the selected model,
preprocess-config, image, and xclbin hashes did not change during the run.
`*_ep_report.json` are snapshots of each just-built cache's assignment report;
`verdict_*.log` independently count nodes on the NPU. Logs and report snapshots are
UTF-8 with local profile paths redacted. Existing measurement logs were not modified.

Ryzen AI 1.7.1 remains the inference SDK; the separate 1.8.0 export/quantization
environment is retained. Optional IRON/Peano and monitor build environments were not
installed. MODNet, MobileViT, wide models, webcam, DirectML and full-dataset accuracy
were not exercised. New evidence is left uncommitted for review.
