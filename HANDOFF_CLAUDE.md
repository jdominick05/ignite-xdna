# Handoff: Session State & Remaining Tasks

This document provides a clean handoff of recent work, hardware findings, current repository state, and remaining measurement tasks.

---

## 1. Executive Summary of Work Completed

Five commits have been completed since the previous handoff. The most recent two (`3ba760f`, `6b42568`) are local-only — **not yet pushed to `origin/main`**, which has diverged with one unmerged commit in the other direction (see Section 3).

### Commit 1: `f46b523`
1. **ResNet50 288² AdaRound Evaluation**:
   - Quantized `models/resnet50_r288_fp32.onnx` -> `models/resnet50_r288_xint8_adaround.onnx` (300 calib images, 54 layers optimized via CPU FastFinetune in 1233.3 s, total 1898.5 s).
   - NPU partition verified: **393 / 395 nodes on NPU (99.5%)**.
   - Evaluated on 1000 validation images (`results/res/run_resnet50_r288_adaround_npu.log`):
     - Plain XINT8 (calib 64): 71.50% top-1 / 89.50% top-5 (9.32 ms).
     - XINT8 + AdaRound (calib 300): **78.10% top-1 / 93.40% top-5 (8.78 ms, ~113.9 fps)**.
     - **Delta: +6.60% top-1, +3.90% top-5, -0.54 ms latency**.
     - Confirmed: The drop at 288² was quantization noise rather than resolution mismatch, though 224² remains optimal on both axes (79.80% at 5.27 ms).
2. **Quantization CPU Thread Scaling Microbenchmark & Pipeline Optimizations**:
   - Discovered and diagnosed CPU thread thrashing ("cores taking turns" in Task Manager) during FastFinetune on small per-layer mini-batches (`batch_size=2`).
   - Benchmarked 1 pinned core vs 4 real cores vs 8 real cores vs 16 logical threads in `tools/bench_quant_threads.py` (`results/bench_quant_threads.log`).
   - Result: **8 real physical cores (mask `0x5555`)** is the fastest configuration overall (49.3 s FastFT, 76.7 s wall clock, beating 16 threads by 6% and 4 cores by 31%).

### Commit 2: `29481ae`
1. **YOLOv8m at Calibration 200 (Task 2 Closed)**:
   - Quantized `models/yolov8m_cut.onnx` -> `models/yolov8m_cut_xint8_c200.onnx` with plain XINT8 (200 images, 83 Conv layers, ~68.7 GB NVMe spool).
   - **NPU Latency**: **26.95 ms (37.1 fps, 1216/1223 nodes, 99.4%)** — beats previous calib-64 (30.80 ms) and provides a **5.35× speedup over the 8-core CPU** (144.12 ms).
   - **5,000-image COCO val2017 mAP**: **43.38 mAP@50-95, 59.62 mAP@50** (`results/bench/map_yolov8m_cut_xint8_c200_npu.log`).
   - Compared to calib-64 (43.49 / 59.79): delta is only **-0.11 mAP**, closing the caveat and proving that model width, not calibration count past 64, governs quantization accuracy.
2. **8-Core Physical Affinity Default (`0x5555`)**:
   - `scripts/lib.sh` and all 5 quantization pipelines now default to **8 threads**.
   - Automatic Win32 affinity pinning (`SetProcessAffinityMask(0x5555)`) pins execution to the 8 physical Zen 4 cores (avoiding hyperthreads 1, 3, 5, 7, 9, 11, 13, 15).
3. **`tools/audit_numbers.py` Optimization**:
   - Skips multi-megabyte `dets_*.json` detection dumps, bringing execution from hanging/thrashing to under 2 seconds.

### Commit 3: `38d28b3`
- **Audit of Gemini's three closed items** — verified numbers from `29481ae` (yolov8m calib-200), `db518e2` (pose AdaRound), and `f46b523` (ResNet50 288² AdaRound) against their backing logs. Fixed minor doc inconsistencies found during the audit.

### Commit 4: `3ba760f` (local only, not on origin)
- **NPU monitor rewrite** — `tools/hwinfo_npu_bridge.cpp` rewritten from scratch into a live ANSI dashboard (utilization bar, sparkline, clock, memory, per-context table, HWiNFO64 sensor publishing via registry). Two measurement findings:
  - `max_clock_frequency_mhz` (`pyxrt`) is a **live clock readback** (800 MHz idle, 1800 MHz under load), not a static value.
  - NPU is visible to Windows as an MCDM adapter; its compute engine reads 84–88% across a GEMM run.
  - Logged to `results/aie/xrt_api_live_clock_and_pdh_npu.log`.
- `docs/SETUP.md` updated with NPU monitor subsection.
- `docs/SILICON.md` 1.7 updated with live-clock and live-utilization rows.

### Commit 5: `6b42568` (local only, not on origin)
- **NPU monitor cleanup** — two HWiNFO-visible bugs fixed:
  1. Frozen "XDNA NPU" ghost group (old schema, never cleaned up at startup).
  2. Zeros fed to HWiNFO's Average column when the NPU is idle. Added `--idle hide` mode: activity sensors are removed while no hardware context is active, so HWiNFO averages only real NPU-active time. Default stays `zero` (0% and 800 MHz are the true idle readings).
- `docs/SETUP.md` documents the `--idle hide` flag.

### Commit 6: `986030f` (local only, not on origin) — **Current HEAD**
- **Created `demos/` directory and standalone benchmark suite** with `demos/README.md` documenting and indexing 8 dedicated interactive demos covering all core XDNA1 findings:
  1. `demos/width_ladder_demo.py`: YOLOv8 n->s->m->x width scaling and saturation on NPU (`results/width_ladder_{n,s,m,x}_npu.jpg`).
  2. `demos/adaround_diff_demo.py`: Bipartite IoU matching between plain XINT8 and AdaRound (`results/adaround_diff_yolov8n_npu.jpg`).
  3. `demos/conf_sweep_demo.py`: Confidence ladder sweeping 0.001 to 0.90, quantifying the 148 tail boxes mAP evaluates (`results/conf_sweep_yolov8m_npu.jpg`).
  4. `demos/tri_hardware_showdown_demo.py`: CPU vs Radeon 780M iGPU (DML FP32 & FP16) vs Phoenix NPU (`results/tri_hardware_showdown.jpg`).
  5. `demos/pose_adaround_demo.py`: YOLOv8n-pose 17-joint Euclidean coordinate drift (8.82 px mean) and skeleton stabilization (`results/pose_adaround_diff_npu.jpg`).
  6. `demos/batch_failure_demo.py`: Negative result demo proving the batch>1 silent hardware corruption bug on NPU (`results/batch_failure_npu.jpg`).
  7. `demos/resolution_ladder_demo.py`: ResNet50 128²->384² resolution sweep exposing the ~3.5 ms fixed AIE dispatch floor (`results/resolution_ladder_npu.jpg`).
  8. `demos/classifier_width_demo.py`: ResNet50 vs Wide-ResNet50-2 vs Wide-ResNet101-2 width ladder on NPU (`results/classifier_width_npu.jpg`).
- Link integrity confirmed: `python tools/check_links.py` passes on all 12 markdown files.

---

## 2. Current Machine & Environment Setup

- **Machine**: Desktop 2 (AMD Ryzen 7 8700G, Zen 4 8 cores / 16 threads, Radeon 780M iGPU, Phoenix XDNA1 NPU, 32 GB RAM, Windows 11).
- **Disk**: Drive `C:` has **>759 GB free** (as of session end).
- **Conda Environments**:
  - `resnet_env`: Python 3.12, PyTorch, Quark 0.11 (for ONNX export, quantization, CPU microbenchmarks).
  - `resnet_env17`: Python 3.12, ONNX Runtime 1.23.3.dev with VitisAI EP, Ryzen AI 1.7.1, pycocotools (for all NPU/CPU inference, latency, and mAP evaluation).
- **Core Invariants from `CLAUDE.md`**:
  - Out of scope: Do not touch int64 or `docs/SILICON.md` (reserved for Fable).
  - Strictly measured numbers only; `--fresh` on model/xclbin changes.
  - UTF-8 logs with user profile paths scrubbed (`C:\Users\<user>`).
  - Never `git push` without asking for user confirmation first.
  - All markdown links must resolve (`python tools/check_links.py` exit 0), and `README.md` must stay under the 250-line budget.

---

## 3. Current Working Tree Status

- **Branch**: `main`, **DIVERGED from `origin/main`**.
  - Local has 2 commits not on origin: `3ba760f`, `6b42568` (NPU monitor work).
  - `origin/main` has 1 commit not local: `a4e7eea` — "Measure the AIE core clock: 1.80 GHz default, 0.80 powersaver, via trace-unit stamps" (by Claude Fable on Desktop 2's other session, closes SILICON.md objective S0). This commit touches `SILICON.md`, `BENCHMARKS.md`, `RESEARCH.md`, `DECISIONS.md`, `README.md`, `kernels/README.md`, and both results index files, plus adds `kernels/clock_probe/` and `results/aie/clock_probe_npu.log`.
  - **Merging will conflict** in at least `SILICON.md`, `RESEARCH.md`, `DECISIONS.md`, `results/README.md`, and `results/aie/README.md`. Both sides have new findings that must survive the merge (CLAUDE.md rule: treat both as evidence to combine).
- **Unstaged working tree changes** (WIP — do not touch):
  - `docs/SILICON.md`
  - `kernels/bf16_matmul_sweep/cpu_matmul_sweep.py`
  - `kernels/int8_matmul_sweep/npu_matmul_sweep.py`
  - `tools/hwinfo_npu_bridge.cpp`
- **Untracked files** (WIP — do not touch):
  - `kernels/gemm_tile_sweep/whole_array_c_single_buffer.patch`
- **Link check**: `python tools/check_links.py` passes (11 markdown files checked, 0 broken links, README within 250-line budget).
- **Numeric audit**: `python tools/audit_numbers.py` runs cleanly (65 notices, all correctly labelled retraction/supersession entries — no orphans).

---

## 4. In-Progress Background Job

At session hand-off time, **PID 6332** was running:
```
resnet_env python pipelines/resnet50/3_quantize.py --calib-dir data/calib --limit 300 \
    --config XINT8_ADAROUND --in-model models/wide_resnet101_2_fp32.onnx \
    --cfg-path models/preprocess_config_wide101.json \
    --out models/wide_resnet101_2_xint8_adaround.onnx
```
- Started ~6:31 PM, was ~77 minutes into a 104-layer FastFinetune run.
- This addresses the open question in `RESEARCH.md` line 522: *"AdaRound for `wide_resnet101_2` (would it also land near ~90%…)"*
- **If this has finished**, `models/wide_resnet101_2_xint8_adaround.onnx` will be on disk. The next step is:
  1. Run `pipelines/resnet50/4_run.py --ep npu --fresh --model models/wide_resnet101_2_xint8_adaround.onnx --n 1000` in `resnet_env17`.
  2. Read the log output before writing anything.
  3. Fold into `docs/BENCHMARKS.md` (under the AdaRound at width section), close the open item in `RESEARCH.md` line 522, and add the log to `results/aie/README.md`'s index (or `results/adaround/`).
  4. Run `check_links.py` and `audit_numbers.py` before committing.

---

## 5. Remaining Tasks & Next Steps

### Task A: Webcam Path End to End (`./scripts/yolo-demo.sh`)
- **Status**: Open in `RESEARCH.md`.
- **Requirement**: Requires a physical webcam connected and a human at the machine to eyeball the output window. Do not run unattended.

### Task B: Merge `origin/main` (Clock Probe Commit)
- **Status**: Needed before pushing anything.
- The merge brings in the AIE core clock measurement (1.80 GHz, objective S0) from the other session.
- Will conflict in multiple doc files. Both sides must survive — read both hunks before resolving.

### Task C: Width Scaling Calibration Sweep for l and x
- **Status**: `yolov8l` at calib-32 and `yolov8x` at calib-24 are the last rows without like-for-like calibration.
- `l` has hit hardware `DPU timeout` on two of three previous 5000-image mAP attempts — if it times out again, that is the result to record.
- Disk space is fine (>759 GB free; ~200 GB spool expected for l at calib-200).
- **Do not start Task C until Task B (merge) is resolved** — both touch the same doc sections.

### Task D: Verification Check Before Any Future Commits
Always run:
```bash
python tools/check_links.py
python tools/audit_numbers.py
```
And ensure user paths are scrubbed from any newly created log files under `results/`.
