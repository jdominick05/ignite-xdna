# Model zoo on Phoenix: ONNX CPU baselines and bare-metal `.ignite` containers

Four ONNX models timed on the CPU through Ignition, and two of them (YOLOv8s and SESR M7)
lowered onto the 16-core convolution engine that already runs YOLOv8n
([engine design](BENCHMARKS.md#whole-network-yolov8n-on-a-16-core-convolution-engine-every-layer-on-the-npu-bit-exact-2026-09-13-desktop-2),
[latency work](BENCHMARKS.md#graph-engine-latency-from-192-to-79-ms-glass-to-glass-2026-09-14-desktop-2)).
Every number above [On main with the native decode](#on-main-with-the-native-decode-2026-09-14-2209-utc) was measured in one sitting on Desktop 2 (`DESKTOP-CBL5NUA`,
Phoenix NPU, Device 0 `[003d:00:01.1]`) on 2026-09-14 between 12:47 and
12:51 UTC, from ignite-xdna `91d0d7e` and Ignition `91ace94`. `xrt-smi examine -r
aie-partitions` reported no hardware contexts before every NPU step. The log and the generated
tables below quote ignite-xdna `91d0d7e` by `d828678`, its ID before the history rewrite. Ignition
`91ace94` is on no branch now; its `live_ignition.py` and pipelines are the files of `3cf2c49`, the
commit on Ignition's `model-zoo` branch.

Evidence: [`results/aie/model_zoo_phoenix_20260914T1247Z.log`](../results/aie/model_zoo_phoenix_20260914T1247Z.log)
(every step of the sitting, with commands and return codes),
[`results/benchmarks_onnx_cpu.json`](../results/benchmarks_onnx_cpu.json),
[`results/benchmarks_npu_silicon.json`](../results/benchmarks_npu_silicon.json) and the per-step
artifacts in [`results/model_zoo/`](../results/model_zoo/).

## Acceptance

| Criterion | Status | Measured |
|---|---|---|
| 1. CPU baselines for yolov8s, yolo11n (no C2PSA), SESR M7 and ResNet50, 300 frames each, return code 0, table here | **Met** | all four return 0; [table](#cpu-baselines-onnx-runtime-through-ignition) |
| 2. `build/yolov8s.ignite` with `head_status` present; 300 oracle-free frames on Device 0 with no buffer-object leaks; glass-to-glass ≤ 25.0 ms | **Met** | heads present (P3/P4/P5 box and class heads, 1,209,600 of 1,209,600 egress bytes); 66 of 66 layers byte-exact on silicon; 300 frames at **17.35 ms** mean (p99 17.73), 0 buffer objects allocated, +0.03 MB working set; IoU 1.0 against the CPU oracle for all six detections |
| 3a. `build/sesr_m7.ignite` runs 500 frames without hangs or queue desyncs | **Met** | 9 of 9 layers byte-exact; the image equals ONNX Runtime's in all 786,432 values; 500 frames, 0 buffer objects allocated, no dispatch errors |
| 3b. SESR raw NPU dispatch ≤ 1.5 ms | **Not met** | **4.25 ms** mean (p99 4.74); the same stream with no core compute takes 2.53 ms, and all weight traffic accounts for ~0.26 ms of that, so neither a kernel change nor resident weights can reach 1.5 ms on this engine ([why](#sesr-the-15-ms-dispatch-and-sram-resident-weights-are-not-met)) |
| 3c. SESR parameters resident in core / MemTile SRAM | **Not met** | the 18 weight packets (170,496 B) stream from DDR every frame, as for every engine container ([why](#sesr-the-15-ms-dispatch-and-sram-resident-weights-are-not-met)) |
| 3d. SESR egress descriptor is a dense H × W × C image | **Met, with a host tail** | the manifest's `dense_output` declares the tail convolution as a dense 12 × 256 × 256 uint8 map (scale 2.0, zero point 128, 786,432 egress bytes, layout `blocks_hw8`); the model's final `DepthToSpace` × 2 (CRD) runs on the host through a lookup table and gives the 512 × 512 × 3 image (`output_shapes.image`), 1.93 ms of the 6.57 ms frame ([manifest](../results/model_zoo/manifest_sesr_m7.json)) |

`tests/test_npu_inference.py`'s `SuperResolutionOnSilicon.test_22_dispatch_budget` asserts the
1.5 ms line and fails; it is left failing rather than relaxed.

## CPU baselines (ONNX Runtime, through Ignition)

Ignition's `live_ignition.py --headless --frames 300 --warmup 10 --json` on `assets/bus.jpg`
(810 × 1080), one process per model. Glass-to-glass is frame in memory to finished output:
preprocess, `session.run`, and the task's postprocess (decode and NMS for detection, softmax
top-k for classification, depth-to-space and the output image for super-resolution). RSS drift
is the resident set at the end minus at the first timed frame. ONNX Runtime CPU provider with
its default threading on 16 logical CPUs.

<!-- BEGIN onnx-cpu (tools/model_zoo_bench.py) -->
Suite `onnx-cpu`, 2026-09-14T12:49:17Z, 300 timed frames after 10 warm-up, source `bus.jpg`, host `DESKTOP-CBL5NUA`, onnxruntime 1.30.0, ignite-xdna `v0.1.0-phoenix-npu-1-gd828678`, Ignition `v0.2.0-15-g91ace94`. All return codes zero: **True**.

| Model | Task | Backend | Frames | G2G mean (ms) | P50 | P95 | P99 | FPS | Preprocess | Network | Post | Dispatch | Readback | RSS drift (MB) | rc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolov8s_cut_xint8.onnx` | detect | onnxruntime-cpu | 300 | 83.66 | 83.90 | 93.59 | 98.08 | 12.0 | 1.90 | 79.61 | 2.15 | n/a | n/a | 0.19 | 0 |
| `yolo11n_no_c2psa_cut_xint8.onnx` | detect | onnxruntime-cpu | 300 | 35.79 | 35.82 | 40.77 | 42.19 | 27.9 | 1.93 | 31.90 | 1.96 | n/a | n/a | 0.02 | 0 |
| `sesr_m7_xint8.onnx` | super_resolution | onnxruntime-cpu | 300 | 15.58 | 15.20 | 18.84 | 19.64 | 64.2 | 0.59 | 12.95 | 2.03 | n/a | n/a | -0.71 | 0 |
| `resnet50_xint8_c64.onnx` | classify | onnxruntime-cpu | 300 | 34.24 | 34.39 | 39.31 | 41.04 | 29.2 | 0.94 | 33.22 | 0.08 | n/a | n/a | 0.06 | 0 |
<!-- END onnx-cpu -->

`yolo11n_no_c2psa_cut_xint8.onnx` returns no detections on `bus.jpg` (0.00 per frame at
confidence 0.25): it is the C2PSA-ablated backbone, kept for its latency, not a working
detector. The ResNet50 row is 224 × 224 classification with the model's timm transform.

## NPU containers

### Build

`python -m ignite_xdna.compiler.cli compile --model <onnx> --output build/<name>.ignite` under
`scripts/research-iron.sh`, from the committed sources (the log's `[pin]` lines show the
imported modules). YOLOv8n is rebuilt as the regression reference.

| Container | Layers | Rounds | Weight packets | Instructions | DDR workspace | Container size | Egress |
|---|---:|---:|---:|---:|---:|---:|---|
| `yolov8n_full.ignite` | 66 | 1,415 | 862 (8.16 MB) | 430,180 B | 22.0 MB | 8,823,680 B | six heads, 1,209,600 B |
| `yolov8s.ignite` | 66 | 2,173 | 3,114 (29.50 MB) | 1,019,140 B | 32.2 MB | 30,743,680 B | six heads, 1,209,600 B, `head_status: present` |
| `sesr_m7.ignite` | 9 | 1,521 | 18 (0.17 MB) | 144,880 B | 19.7 MB | 511,168 B | dense 12 × 256 × 256 uint8; host depth-to-space × 2 → 512 × 512 × 3 image |

What the compiler and kernel needed beyond YOLOv8n (commit `91d0d7e`):

- **Residual adds with a left-shifted main branch.** Some yolov8s adds put the main branch
  at a finer power-of-two scale than the sum, which `rne(main + (res << rsh))` cannot express
  exactly. The residual op is now `rne((main << lsh_m) + (res << lsh_r), rsh)` when a packet
  sets header flag 32; every existing packet keeps its meaning.
- **Model-sized DDR extents.** yolov8s carries 29.5 MB of weight packets, over the 16 MB
  default; the runtime buffers are sized from the container.
- **SESR operators:** ReLU as an exact integer epilogue (`max(q, 128)`), bias-less and linear
  convolutions, 5 × 5 stride-1 convolutions, a long-skip Add attached only once both operands
  exist, and the final `DepthToSpace` (CRD) recorded as a host output transform.
- **Maps the 20-pixel tile does not divide.** SESR's 256 × 256 maps use overlapping edge
  tiles (origins 0, 20, …, 220, 236): 169 rounds per layer.

### Exactness on Device 0

- `tools/verify_engine_container.py` compares every layer tensor after one dispatch with the
  integer reference (itself equal to ONNX Runtime's uint8 intermediates offline):
  **66 / 66** for yolov8n, **66 / 66** for yolov8s, **9 / 9** for SESR M7.
- yolov8s, oracle-free: six detections on `bus.jpg`, IoU 1.0 against the ONNX Runtime pass
  over `yolov8s_cut_xint8.onnx`.
- SESR M7: 0 of 786,432 output values differ from ONNX Runtime (graph optimizations off).
- Regression: the unchanged YOLOv8n suite passes all ten tests at its original budgets,
  500 frames at 7.806 ms mean glass-to-glass (p99 8.166).

### Silicon latency audit (500 frames, `tools/live_camera_ignition.py`)

`--source assets/bus.jpg --headless --warmup 10 --frames 500 --json`. Host staging is the
resize and quantization straight into the mapped input plane; dispatch is `run.wait()` of the
one instruction stream; readback is the egress read (heads and per-anchor class maxima for
detection, the 12-channel tail for SESR); postprocess is decode and NMS, or depth-to-space
through a lookup table.

| Container | Staging | Dispatch mean / p99 | Readback | Postprocess | Glass-to-glass mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolov8s.ignite` (6 objects / frame) | 0.360 | 16.745 / 16.901 | 0.243 | 0.316 | **17.665** | 17.619 | 18.019 | 18.152 |
| `sesr_m7.ignite` (512 × 512 × 3 out) | 0.364 | 4.256 / 4.828 | 0.023 | 1.929 | **6.572** | 6.584 | 7.041 | 7.255 |

All times in ms. `tests/test_npu_inference.py` in the same sitting (1280 × 720 and 640 × 480
synthetic frames): yolov8s 300 frames at 17.348 ms mean (p99 17.732), 0 buffer objects,
+0.03 MB; SESR 500 frames at 6.558 ms mean (p99 7.341), dispatch 4.251 ms mean (p99 4.741),
0 buffer objects, +0.84 MB.

### Through Ignition (`live_ignition.py`, 500 frames)

<!-- BEGIN npu (tools/model_zoo_bench.py) -->
Suite `npu`, 2026-09-14T12:50:31Z, 500 timed frames after 10 warm-up, source `bus.jpg`, host `DESKTOP-CBL5NUA`, onnxruntime 1.30.0, ignite-xdna `v0.1.0-phoenix-npu-1-gd828678-dirty`, Ignition `v0.2.0-15-g91ace94`. All return codes zero: **True**.

| Model | Task | Backend | Frames | G2G mean (ms) | P50 | P95 | P99 | FPS | Preprocess | Network | Post | Dispatch | Readback | RSS drift (MB) | rc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolov8s.ignite` | detect | npu | 500 | 17.73 | 17.69 | 18.12 | 18.35 | 56.4 | 0.36 | 17.06 | 0.31 | 16.81 | 0.25 | 0.10 | 0 |
| `sesr_m7.ignite` | super_resolution | npu | 500 | 6.57 | 6.57 | 7.08 | 7.37 | 152.1 | 0.35 | 4.26 | 1.96 | 4.24 | 0.02 | 0.00 | 0 |
<!-- END npu -->

The `-dirty` on this suite's ignite-xdna commit is a one-line `README.md` link edited while
the sitting ran; no source file differed from `91d0d7e`. Against the CPU rows above, the same
models run **4.7×** faster end to end on the NPU for yolov8s (17.73 against 83.66 ms) and
**2.4×** for SESR M7 (6.57 against 15.58 ms).

### Where the dispatch goes

`tools/engine_dispatch_floor.py` copies a container with every weight packet's op set to NOP:
same xclbin, instruction stream and packets, so the cores still take every packet and emit
every output object, but compute nothing. 300 dispatches each, same frame.

| Container | Dispatch | NOP-weight floor | Core compute | Weight packets | Instructions |
|---|---:|---:|---:|---:|---:|
| `yolov8n_full.ignite` | 7.386 ms | 5.369 ms | 2.017 ms | 862 | 430,180 B |
| `yolov8s.ignite` | 16.932 ms | 12.727 ms | 4.205 ms | 3,114 | 1,019,140 B |
| `sesr_m7.ignite` | 4.208 ms | 2.530 ms | 1.679 ms | 18 | 144,880 B |

yolov8s has ~3.5× yolov8n's parameters and ~2.1× its core compute; most of its 17 ms is the
larger packet traffic (3.6× the weight packets, 2.4× the instruction bytes).

### Tile memory with 3.5× the parameters

`tools/engine_memory_report.py` over the placed design (`design.prj/input_with_addresses.mlir`),
[`memory_yolov8s.json`](../results/model_zoo/memory_yolov8s.json) and
[`memory_sesr_m7.json`](../results/model_zoo/memory_sesr_m7.json):

| Tile | Used | Capacity | Largest buffers |
|---|---:|---:|---|
| compute (each of 16) | 59,392 B incl. 2,048 B stack | 65,536 B | 16,000 B accumulator scratch, two 9,472 B weight objects, 6,400 B activation object |
| MemTile (each of 4) | 76,800 B | 524,288 B | two 25,600 B activation splits, two 12,800 B output joins |

The two reports are identical. Weight, activation and output objects have fixed sizes, so
yolov8s's extra parameters become more weight packets streamed from DDR per frame, never a
larger object on a tile: weight streaming stays inside the 64 KB core and 512 KB MemTile
budgets by construction, and the placer would refuse a design that did not.

## SESR: the 1.5 ms dispatch and SRAM-resident weights are not met

**The floor is not weights or compute.** With every weight op a NOP, SESR's dispatch is still
2.530 ms, above the 1.5 ms target, so removing all core compute (1.68 ms) and all weight
traffic would not reach it. Per frame the schedule issues 1,007 DMA tasks: 36 weight fills
moving 6,403,072 B (the 18 packets repeated once per round), 710 activation fills moving
47,590,400 B and 261 drains moving 19,468,800 B
([`tools/engine_stream_report.py`](../results/model_zoo/stream_sesr_m7.log); packet counts in the
[manifest](../results/model_zoo/manifest_sesr_m7.json)). Weights are 3.6 % of the tasks and
8.7 % of the bytes; at the measured 26.8 GB/s fill rate and 145 ns per instruction op they come
to about 0.26 ms (DERIVED), so perfectly resident weights would leave a floor near 2.27 ms,
still above 1.5 ms. The floor is the engine's design: every layer's activations leave DDR, pass the MemTile to the cores and
return to DDR, and SESR's 256 × 256 maps at 20-pixel tiles make 1,521 rounds and 7,436
activation packets for a 22 KB network (the fixed dispatch cost, ~145 ns per instruction op
and transport measured in the [latency work](BENCHMARKS.md#graph-engine-latency-from-192-to-79-ms-glass-to-glass-2026-09-14-desktop-2)
all scale with that traffic). The VitisAI EP runs the same model in
[1.48–1.54 ms](BENCHMARKS.md#category-a-image-super-resolution-sesr-m7); reaching that bare-metal
would need feature maps kept on the chip between layers, a different engine. One 256 × 256 × 16
uint8 map is 1,048,576 B, twice a MemTile.

**Why the weights still stream (not run on silicon).** The resident layout was designed and
dropped before building, from reading mlir-aie IRON 1.4.2's `ObjectFifo` code, not from a
measurement:

- An ObjectFIFO with `init_values` on a MemTile hands its objects to the consumer once; the
  producer side never refills them, so the second frame's acquire waits on a lock forever.
  Replaying them per frame needs a DMA channel reset in the runtime sequence that IRON does
  not wrap and that is untested with `init_values`; `disable_synchronization` removes the
  locks on both ends and lets a core read a half-updated object.
- The core program acquires one weight object per round. Holding SESR's 18 layer packets
  (170,496 B, which would fit) needs a different core program that keeps a layer's weight
  object across all its rounds; the per-round stream as scheduled today is 6.4 MB a frame,
  about 1.6 MB per column against a 512 KB MemTile.
- It would not change the result above: removing all weight traffic takes about 0.26 ms off
  the 2.53 ms floor (DERIVED above).

## On main with the native decode (2026-09-14, 22:09 UTC)

The branch was merged with ignite-xdna `main` `5688f6d`, which carries the native int8 head
decode, as `6bd2718`, and the checks above were repeated on that commit in one sitting from
22:09:56 to 22:13:05 UTC, with Ignition's `model-zoo` branch at `3cf2c49`. `xrt-smi examine -r
aie-partitions` reported no hardware contexts before and after every NPU step. Evidence:
[`results/aie/model_zoo_main_phoenix_20260914T2209Z.log`](../results/aie/model_zoo_main_phoenix_20260914T2209Z.log)
and the per-step JSON in [`results/model_zoo/main_20260914T2209Z/`](../results/model_zoo/main_20260914T2209Z/).

- **Builds.** The three containers compiled from `6bd2718` have the 12:47 builds' sizes, and
  their instruction streams and weight packets are byte-identical to them. The manifests differ
  only in build time, compile seconds and xclbin hash; the xclbins differ in 79 to 84 bytes and
  carry the same kernel hash. Which xclbin fields those bytes belong to was not decoded
  ([`container_diff.json`](../results/model_zoo/main_20260914T2209Z/container_diff.json)).
- **Exactness.** 66 / 66, 66 / 66 and 9 / 9 layers byte-exact on Device 0; yolov8n and yolov8s
  detections at IoU 1.0 against the CPU oracle; SESR 0 of 786,432 values differ from ONNX Runtime.
- **The yolov8n container compiled before the model zoo still runs.** The main checkout's
  `build/yolov8n_full.ignite` (sha256[:16] `ce64c451ea9bd620`, `ignite-compile` v0.3.0) has the
  merged build's instruction stream and packets, an older kernel and xclbin (187,518 against
  188,542 B), and no `task` or `ddr_extents_bytes` in its manifest. It loads on the merged runtime
  and passes `NpuInferenceOnSilicon`. Interleaved suites, 500 synthetic 1280 × 720 frames each:

| `tests/test_npu_inference.py` run | Container | G2G mean | p99 | NPU | Postprocess |
|---|---|---:|---:|---:|---:|
| 1 (10 tests, all pass) | merged build | 7.772 | 8.013 | 7.403 | 0.013 |
| 2 (3 tests, all pass) | compiled before the model zoo | 7.780 | 8.032 | 7.413 | 0.013 |
| 3 (3 tests, all pass) | merged build | 7.755 | 8.071 | 7.386 | 0.012 |
| 4 (3 tests, all pass) | compiled before the model zoo | 7.712 | 8.096 | 7.340 | 0.012 |

All times in ms. The two containers fall within the 0.07 ms spread of these four runs.

- **YOLOv8s decodes natively.** Postprocess is 0.041 ms in the camera tool (six objects) against
  0.316 ms at 12:47, and 0.015 ms against 0.060 ms on the suite's synthetic frames. Every step
  loaded the native decode library; the other host stages moved by at most 0.05 ms between the
  two sittings, so the 7.7× fall is not sitting-to-sitting drift.

`tools/live_camera_ignition.py --source assets/bus.jpg --headless --warmup 10 --frames 500`:

| Container | Staging | Dispatch mean / p99 | Readback | Postprocess | Glass-to-glass mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolov8n_full.ignite` (5 objects / frame) | 0.317 | 7.218 / 7.316 | 0.191 | 0.036 | **7.762** | 7.762 | 7.938 | 8.225 |
| `yolov8s.ignite` (6 objects / frame) | 0.319 | 16.745 / 16.826 | 0.194 | 0.041 | **17.298** | 17.270 | 17.490 | 17.629 |
| `sesr_m7.ignite` (512 × 512 × 3 out) | 0.356 | 4.344 / 4.913 | 0.026 | 1.897 | **6.623** | 6.632 | 7.066 | 7.319 |

Through Ignition's `live_ignition.py` (`3cf2c49`, `--headless --warmup 10 --frames 500 --json`
on `bus.jpg`; resident memory unchanged over each run):

| Container | Glass-to-glass mean | P50 | P95 | P99 | Preprocess | Dispatch | Readback | Post | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `yolov8n_full.ignite` | **7.681** | 7.686 | 7.872 | 8.037 | 0.301 | 7.161 | 0.184 | 0.031 | 5.00 detections / frame |
| `yolov8s.ignite` | **17.148** | 17.119 | 17.329 | 17.492 | 0.305 | 16.609 | 0.191 | 0.038 | 6.00 detections / frame |
| `sesr_m7.ignite` | **6.644** | 6.635 | 7.114 | 7.468 | 0.327 | 4.369 | 0.027 | 1.916 | 512 × 512 × 3 image |

- **Suites.** yolov8s `NpuInferenceOnSilicon` 3 / 3: 300 frames at 17.278 ms mean (p99 17.525),
  no buffer objects allocated. SESR `SuperResolutionOnSilicon` 2 / 3: 500 frames at 6.606 ms mean
  (p99 7.363), no buffer objects allocated; `test_22` still fails, at a 4.336 ms mean dispatch
  against 1.5 ms.
- **Not re-run:** the NOP-weight dispatch floors, the tile memory reports and the stream report
  (the instruction streams and packets match the 12:47 builds), and the ONNX CPU suite.

The two sittings ran on the same day but were not interleaved, so glass-to-glass differences
between this section and the rows above are not attributed to the merge.

## YOLO11n with C2PSA: two NPU segments and a host step (2026-09-15, 02:16 UTC)

Stock `yolo11n_cut_xint8.onnx`, attention block included, as a segmented container: layers 0–35 and
37–83 on the NPU, the C2PSA block (`/model.10/`) on ONNX Runtime's CPU provider between them. Design,
exactness and the full sitting are in
[BENCHMARKS](BENCHMARKS.md#stock-yolo11n-with-its-c2psa-attention-block-npu-segments-around-a-host-step-2026-09-15-desktop-2);
evidence [`results/aie/yolo11n_hybrid_phoenix_20260915T0216Z.log`](../results/aie/yolo11n_hybrid_phoenix_20260915T0216Z.log).

| Container | Layers | Rounds | Weight packets | Instructions | DDR workspace | Container size | Egress |
|---|---:|---:|---:|---:|---:|---:|---|
| `yolo11n.ignite` | 83 NPU + 1 host | 1,606 | 9.24 MB | 200,160 + 287,684 B (1,376 + 1,984 tasks) | 24.5 MB | 10,259,008 B | six heads; host model 295,565 B |

- **Exactness:** 84 / 84 layers byte-exact on Device 0, host layer included; oracle-free detections
  identical to ONNX Runtime's CPU decode of the same input.
- **Through Ignition's `live_ignition.py`** (50 warm-up and 500 frames of `bus.jpg`, interleaved with
  AMD's stack): **10.996** and **11.048** ms mean glass-to-glass (P99 11.293 and 11.581), NPU dispatch
  8.452 and 8.442 ms, C2PSA on the CPU 1.685 and 1.688 ms, six objects per frame. In the same sitting
  ONNX Runtime with the Vitis AI EP took 34.492 and 34.366 ms and the CPU provider 31.328 ms (seven
  objects: the input preprocessing differs by one pixel code in 15 % of values, which is enough to
  change this model's borderline boxes).

```bash
bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolo11n_cut_xint8.onnx --output build/yolo11n.ignite --host-region /model.10/
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolo11n.ignite --model models/yolo11n_cut_xint8.onnx
python tests/test_engine_host_layer.py
```

Build one container per process: a second graph container compiled in the same Python process fails
to link its kernel object ([DECISIONS](DECISIONS.md)).

## Reproduce

```bash
# CPU suite through Ignition's `ignition suite` (Ignition f9c85fb or later next to this repo, or --ignition <path>)
python tools/model_zoo_bench.py run --suite onnx-cpu
# containers (mlir-aie ironenv)
bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolov8s_cut_xint8.onnx --output build/yolov8s.ignite
bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/sesr_m7_xint8.onnx --output build/sesr_m7.ignite
# silicon: per-layer exactness, tests, dispatch floor, 500-frame audit, Ignition suite
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/sesr_m7.ignite --model models/sesr_m7_xint8.onnx
IGNITE_MODEL=build/yolov8s.ignite IGNITE_TEST_FRAMES=300 IGNITE_G2G_MEAN_MS=25 IGNITE_G2G_P99_MS=27 \
    bash scripts/research-iron.sh tests/test_npu_inference.py NpuInferenceOnSilicon
bash scripts/research-iron.sh tests/test_npu_inference.py SuperResolutionOnSilicon
bash scripts/research-iron.sh tools/engine_dispatch_floor.py build/sesr_m7.ignite
bash scripts/research-iron.sh tools/live_camera_ignition.py --model build/sesr_m7.ignite --source assets/bus.jpg --headless --warmup 10 --frames 500
python tools/model_zoo_bench.py run --suite npu
python tools/engine_memory_report.py build/conv_engine/yolov8s
```

`research-iron.sh` replaces `PYTHONPATH`; from a worktree whose package is installed editable
elsewhere, put the checkout's `src` first on `sys.path` (the log's compile steps do). The
tables between the markers above are rewritten from the JSON by
`python tools/model_zoo_bench.py report`.

`run` now hands the models to Ignition's `ignition suite`, which still runs each one in its own
process, writes every record and log into a new `results/model_zoo/<suite>_<UTC time>/`
directory, and records `xrt-smi` before and after each container. The tool checks that the suite
loads Ignition from that checkout and ignite-xdna from this one, and rewrites the suite JSON and
the tables only when every run is clean. Both tables above were produced before that change, when the tool ran
`live_ignition.py` itself and wrote `results/model_zoo/<suite>_<model>.log`; they have not been
re-run.

## Caveats

- One still image for every run: detection decode cost depends on what is in the frame
  (six objects here), and host staging includes the 810 × 1080 resize.
- NPU latency on this machine drifts between sessions; compare these rows with each other,
  not with numbers from another day.
- The CPU rows are ONNX Runtime's CPU provider on the same XINT8 QDQ models, not a tuned
  CPU implementation.
