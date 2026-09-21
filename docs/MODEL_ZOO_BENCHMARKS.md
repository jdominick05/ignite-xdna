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
  *Those three figures predate `6a620f0`'s workspace buffer reuse, and a reuse-built container
  can no longer be read that way: after one dispatch a reused slot holds only its last writer,
  so 41 of yolov8n's 66 layers are unreadable and the tool reports them as slots, not as
  mismatches. The like-for-like number today is `25/25 readable layers exact; 41/41 reused slots
  hold their planned final tenant`, and the full per-layer check needs
  `ignite-compile --no-workspace-reuse`, which still gives **66 / 66**. See
  [what the verifier can and cannot see](BENCHMARKS.md#workspace-reuse-makes-41-of-66-layer-readbacks-unobservable-what-verify_engine_container-can-and-cannot-see-2026-09-18-desktop-2).*
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

### Against AMD's stack (2026-09-15)

One sitting on Desktop 2 ([log](../results/aie/yolov8s_sesr_vs_amd_phoenix_20260915T2146Z.log)): both
containers rebuilt at `9351cc1` by the toolchain Ignition's `install.ps1` installs (25.8 s and 6.9 s), 66/66 and 9/9
layers exact on Device 0, then AMD / Ignition / AMD / Ignition per model at 50 warm-up and 500 timed frames of
Ignition's `examples/assets/bus.jpg`, with `xrt-smi` idle before every run (host CPU 1.2–3.6 %). AMD's arm is ONNX
Runtime + Vitis AI EP (Ryzen AI 1.7.1, `resnet_env17`) with Ignition's own letterbox and decode, or its
`sr_preprocess` / `sr_postprocess`; Ignition's arm is `live_ignition.py` at `6985ef7`.

| Model | AMD glass-to-glass, runs 1 / 2 | AMD `session.run` | Ignition glass-to-glass, runs 1 / 2 | Ignition dispatch | RSS, AMD 2nd session / Ignition |
|---|---:|---:|---:|---:|---:|
| YOLOv8s | **16.954 / 16.958** | 13.137 / 13.167 | 17.240 / 17.265 | 16.704 / 16.743 | 339.2 / 242.0 MB |
| SESR M7 | **3.654 / 3.632** | 1.461 / 1.473 | 6.671 / 6.662 | 4.405 / 4.392 | 265.0 / 162.1 MB |

All times in ms. AMD's stack is faster on both: its NPU stage is 3.6 ms shorter on YOLOv8s and 2.9 ms shorter on
SESR M7, more than Ignition's native letterbox and decode recover on YOLOv8s (0.34 ms of host work against 3.82 ms).
The control, `build/yolov8n_full.ignite` through `live_ignition.py`, ran at 7.771 ms.

The YOLOv8s gap is now a known limitation of this runtime: every runtime-side way of shortening its NPU stage was
built, measured or sized on 2026-09-16 and none survived
([BENCHMARKS](BENCHMARKS.md#memtile-residency-does-not-pay-on-the-graph-engine-and-the-yolov8s-gap-is-a-known-limitation-2026-09-16-desktop-2)).

**Re-measured in the `balanced` power mode (2026-09-16).** The sitting above ran the preprocessor's workers spinning,
which is `--power-mode performance` since `3bfebcd`. In the `balanced` default, one sitting with every container
re-verified exact ([log](../results/aie/latency_balanced_default_phoenix_20260916T1745Z.log),
[BENCHMARKS](BENCHMARKS.md#the-amd-comparisons-re-measured-in-the-balanced-default-2026-09-16-desktop-2)):

| Model | AMD glass-to-glass, runs 1 / 2 | AMD `session.run` | Ignition glass-to-glass, runs 1 / 2 | Ignition dispatch | RSS, AMD / Ignition |
|---|---:|---:|---:|---:|---:|
| YOLOv8s | **16.744 / 16.564** | 12.792 / 12.669 | 18.046 / 17.988 | 16.792 / 16.755 | 339.6 / 242.9 MB |
| SESR M7 | **4.365 / 4.332** | 1.492 / 1.495 | 6.840 / 6.815 | 4.178 / 4.178 | 265.3 / 162.9 MB |

All times in ms, RSS from each stack's second run. The dispatch barely moved; the gap on YOLOv8s is 1.36 ms on the
means, against 0.30 ms above, and on SESR M7 2.48 ms against 3.02 ms. The environment changed too (the
`mlir-aie-iron` env rather than an `install.ps1` installation), and AMD's SESR image output drifted 0.6 ms slower with
nothing changed, so compare these rows with each other, not with the table above.

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

(2026-09-16: a hand-written MemTile weight buffer with raw locks, not an `init_values` ObjectFIFO, was later built
for YOLOv8n and YOLOv8s. It is 66/66 byte-exact and slower on both, by 1.160 and 3.630 ms
([BENCHMARKS](BENCHMARKS.md#memtile-residency-does-not-pay-on-the-graph-engine-and-the-yolov8s-gap-is-a-known-limitation-2026-09-16-desktop-2)).
It was not tried on SESR, so 3c above stays Not met.)

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
| `yolo11n_core.ignite` | 90 NPU + 1 host | 1,650 | 10.64 MB | 206,016 + 303,812 B (1,416 + 2,092 tasks) | 25.1 MB | 11,406,848 B | six heads; attention-core host model 15,724 B |

- **Exactness:** 84 / 84 layers byte-exact on Device 0, host layer included; oracle-free detections
  identical to ONNX Runtime's CPU decode of the same input.
- **Through Ignition's `live_ignition.py`** (50 warm-up and 500 frames of `bus.jpg`, interleaved with
  AMD's stack): **10.996** and **11.048** ms mean glass-to-glass (P99 11.293 and 11.581), NPU dispatch
  8.452 and 8.442 ms, C2PSA on the CPU 1.685 and 1.688 ms, six objects per frame. In the same sitting
  ONNX Runtime with the Vitis AI EP took 34.492 and 34.366 ms and the CPU provider 31.328 ms (seven
  objects: the input preprocessing differs by one pixel code in 15 % of values, which is enough to
  change this model's borderline boxes).
- **Only the attention core on the host** (`yolo11n_core.ignite`, `--host-region /model.10/m/m.0/attn/qkv/conv/Conv=/model.10/m/m.0/attn/Reshape_1`,
  2026-09-15 from 15:22 UTC): C2PSA's seven convolutions on the NPU, 91 / 91 layers exact on Device 0.
  **10.411** and **10.115** ms mean glass-to-glass (NPU dispatch 8.784 and 8.708 ms, host 0.533 and 0.505 ms)
  against 11.109 and 10.987 ms for `yolo11n.ignite` and 36.361 and 34.549 ms on AMD's stack in the same
  sitting ([BENCHMARKS](BENCHMARKS.md#yolo11ns-c2psa-convolutions-on-the-npu-only-its-attention-core-on-the-host-2026-09-15-desktop-2)).
- **Both, in the `balanced` power mode** (2026-09-16; the runs above spun the preprocessor's workers): 10.439 and
  10.369 ms with the attention core on the host, 11.182 and 11.177 ms with the whole block, against 36.540 and
  36.644 ms on AMD's stack, every container re-verified exact first
  ([BENCHMARKS](BENCHMARKS.md#the-amd-comparisons-re-measured-in-the-balanced-default-2026-09-16-desktop-2)). That
  host segment's ONNX Runtime threads still spun; since `fbd53f5` they follow the power mode, and `balanced` read
  10.674 and 10.627 ms (attention core) and 11.588 and 11.557 ms (whole block) against 37.665 and 37.702 ms, at 165 mJ
  per frame against AMD's 1,396
  ([BENCHMARKS](BENCHMARKS.md#energy-per-frame-on-yolov8s-sesr-m7-yolo11n-and-yolov8n-pose-2026-09-16-desktop-2)).

```bash
MSYS_NO_PATHCONV=1 bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolo11n_cut_xint8.onnx --output build/yolo11n.ignite --host-region /model.10/
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolo11n.ignite --model models/yolo11n_cut_xint8.onnx
MSYS_NO_PATHCONV=1 bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolo11n_cut_xint8.onnx --output build/yolo11n_core.ignite --host-region "/model.10/m/m.0/attn/qkv/conv/Conv=/model.10/m/m.0/attn/Reshape_1"
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolo11n_core.ignite --model models/yolo11n_cut_xint8.onnx
python tests/test_engine_host_layer.py
```

Build one container per process: a second graph container compiled in the same Python process fails
to link its kernel object ([DECISIONS](DECISIONS.md)).

## YOLOv8n-pose: every layer on the NPU (2026-09-15, 19:19 UTC)

The head-cut `yolov8n-pose_cut_xint8.onnx` as a single-dispatch container whose nine heads carry the box, a person
score and 17 keypoints at three strides. Design, exactness, COCO accuracy and the full sitting are in
[BENCHMARKS](BENCHMARKS.md#yolov8n-pose-on-the-graph-engine-every-layer-on-the-npu-keypoints-through-the-container-2026-09-15-desktop-2);
evidence [`results/aie/yolov8n_pose_phoenix_20260915T1919Z.log`](../results/aie/yolov8n_pose_phoenix_20260915T1919Z.log).

| Container | Layers | Rounds | Weight packets | Instructions | DDR workspace | Container size | Egress |
|---|---:|---:|---:|---:|---:|---:|---|
| `yolov8n_pose.ignite` | 75 | 1,457 | 864 (8.18 MB) | 426,564 B (2,947 tasks) | 22.6 MB | 8,845,248 B | nine heads, 974,400 B, `head_status: present` |

- **Exactness:** 75 / 75 layers byte-exact on Device 0. With `npu/yolo.py`'s letterbox, its detections on the 5,000
  COCO val2017 images are the same bytes as ONNX Runtime's CPU provider's.
- **COCO keypoints, 5,000 images:** OKS mAP@50-95 **32.77** and mAP@50 **67.82** through the native letterbox, 32.71
  and 67.77 through the numpy one (ONNX Runtime CPU's figures), against 32.64 and 66.90 recorded for AMD's Vitis AI EP.
- **Against AMD's stack** (`4_pose.py`, 50 warm-up and 500 frames of `bus.jpg`, interleaved): **8.318** and **8.339**
  ms from frame to people, against 12.056 and 12.089 ms; through Ignition's `live_ignition.py` 8.360 and 8.481 ms
  (NPU dispatch 7.548 and 7.572 ms, decode and NMS 0.288 and 0.320 ms), three people per frame.
- **The same, in the `balanced` power mode** (2026-09-16; the runs above spun the preprocessor's workers): 8.987 and
  9.039 ms through `4_pose.py`, 9.025 and 8.970 ms through `live_ignition.py`, against 11.955 and 12.020 ms on AMD's
  stack ([BENCHMARKS](BENCHMARKS.md#the-amd-comparisons-re-measured-in-the-balanced-default-2026-09-16-desktop-2)).

```bash
bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolov8n-pose_cut_xint8.onnx --output build/yolov8n_pose.ignite
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolov8n_pose.ignite --model models/yolov8n-pose_cut_xint8.onnx
bash scripts/research-iron.sh tests/test_npu_inference.py PoseOnSilicon
bash scripts/research-iron.sh pipelines/yolov8n-pose/4_pose.py --model build/yolov8n_pose.ignite --ep ignite --source assets/bus.jpg --warmup 50 --runs 500
conda activate mlir-aie-iron   # pyxrt and pycocotools
python pipelines/yolov8n-pose/5_eval_map.py --model build/yolov8n_pose.ignite --ep ignite [--ingress numpy]
python tests/test_pose_pipeline_offline.py
```

## YOLO-World v2: only the text attention on the CPU, class names at run time (2026-09-16)

The head-cut YOLO-World v2 (yolov8s-worldv2) after Quark XINT8, with its four C2fAttn output convolutions requantized
by GPTQ with int32 biases. Its four text cross-attention cores run on ONNX Runtime's CPU provider between five NPU
segments. Design, accuracy, exactness and the profile sitting are in
[BENCHMARKS](BENCHMARKS.md#yolo-world-v2-with-only-its-text-attention-on-the-cpu-gptq-and-an-int32-bias-recover-the-four-output-convolutions-2026-09-16-desktop-2),
the run-time vocabulary in
[BENCHMARKS](BENCHMARKS.md#yolo-world-v2s-vocabulary-chosen-at-run-time-one-container-any-class-names-2026-09-16-desktop-2);
evidence `results/aie/yolow_gptq/` and `results/aie/yolow_vocabulary/`.

| Container | Layers | Rounds | Weight packets | Instructions | DDR workspace | Container size | Segments |
|---|---:|---:|---:|---:|---:|---:|---|
| `yolow_gptqcv2.ignite` | 70 (4 host) | 2,446 | 31,181,824 B | 1,145,856 B | 36.7 MB | 33,725,760 B | NhNhNhNhN |

- **Exactness:** 70 / 70 layers byte-exact on Device 0, with COCO's names and with another vocabulary set at run time.
- **COCO, first 300 val2017 images:** mAP@50-95 **24.7** through the container, the same detections as ONNX Runtime CPU;
  43.0 in FP32. AMD's stack scores 1.8 on the 5,000 images in plain XINT8.
- **Frame:** 48.0-48.2 ms profiled against 53.0-53.5 ms with the four convolutions as FP32 host steps, in one sitting,
  with numpy input quantization and head dequantization. That is not a comparison with AMD's stack.
- **Against AMD's stack, the CPU and the iGPU** (one sitting, `5_eval_map.py` per-image inference, 300 images each):
  - container 47.33 and 47.51 ms at 24.7 %;
  - AMD's stack on variant D (its best usable model) 95.91 and 95.89 ms at 24.5 %;
  - ONNX Runtime CPU on FP32 67.75 ms at 43.0 %;
  - DirectML on the iGPU on FP32 46.22 ms at 43.0 %.

  So twice AMD's speed at equal accuracy, and level with the iGPU at far lower accuracy
  ([BENCHMARKS](BENCHMARKS.md#yolo-world-v2-against-amds-stack-the-cpu-and-the-igpu-in-one-sitting-2026-09-16-desktop-2)).
- **Glass-to-glass** (`pipelines/yolow/4b_g2g.py`, `bus.jpg`, 500 frames, one sitting):
  - with 80 classes: container 39.912 and 39.853 ms, AMD's stack 109.729 and 109.376 ms, iGPU FP32 52.415 ms, CPU FP32
    81.593 ms;
  - with five names: container 32.595 ms, iGPU 28.444 ms.

  So 2.75 times AMD's stack, and faster than the iGPU only with the larger vocabulary
  ([BENCHMARKS](BENCHMARKS.md#yolo-world-v2-glass-to-glass-275-times-amds-stack-and-against-the-igpu-it-depends-on-the-vocabulary-2026-09-16-desktop-2)).
- **Energy per frame** (`tools/energy_sitting.py`, same loop, 80 classes):
  - flat out: container 1040.36 and 908.91 mJ, AMD's stack 4483.41 and 4544.51 mJ, iGPU 1201.12 and 1336.00 mJ, CPU
    3610.19 and 3580.88 mJ;
  - at 5 fps: container 1428.13 mJ, AMD's stack 5337.04, iGPU 842.48, CPU 4214.34.

  ([BENCHMARKS](BENCHMARKS.md#yolo-world-v2-energy-per-frame-43-50-times-less-than-amds-stack-and-the-igpu-spends-less-at-5-fps-2026-09-17-desktop-2)).
- **What the compiler, runtime and kernel needed beyond YOLO11n:**
  - host regions whose constants are shared with another region;
  - host inputs that are Concat views;
  - int32 convolution biases;
  - host constants replaced at run time (`EngineSession.set_host_constants`);
  - a residual flag for HardSwish after the add, which no model uses.

```bash
# model tools (resnet_env): export, head cut, XINT8, then the text side for run-time vocabularies
python pipelines/yolow/1_export.py && python pipelines/yolow/1b_cut_head.py
python pipelines/yolow/3b_quantize_cut.py --calib-dir data/coco_calib --limit 200
python pipelines/yolow/6_text_encoder.py
# mlir-aie-iron: GPTQ on the four output convolutions, then compile (MSYS_NO_PATHCONV=1 under Git Bash)
python pipelines/yolow/3c_gptq_cv2.py
MSYS_NO_PATHCONV=1 bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolov8s-worldv2_cut_xint8_gptqcv2.onnx --output build/yolow.ignite --host-region /model.12/attn/ --host-region /model.15/attn/ --host-region /model.18/attn/ --host-region /model.21/attn/
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolow.ignite --model models/yolov8s-worldv2_cut_xint8_gptqcv2.onnx [--host-constants guides.npz]
conda activate mlir-aie-iron   # pyxrt and pycocotools; worktree or checkout src first on PYTHONPATH
python pipelines/yolow/5_eval_map.py --model build/yolow.ignite --ep ignite --n 300 [--vocabulary pipelines/yolow/vocabularies/coco_synonyms.txt]
python tests/test_yolow_pipeline_offline.py
```

## YOLOv8n, YOLOv8s and YOLOv8n-pose with `--silu-sigmoid` (2026-09-17)

`ignite-compile --silu-sigmoid` computes SiLU through the core program's four-line sigmoid instead of Quark's
HardSigmoid form. Every layer is exact on the NPU against `silu_sigmoid.reference_model`. The table covers all 5,000
COCO val2017 images with the ONNX Runtime path's letterbox for every stack; glass-to-glass is the mean of two runs of
bus.jpg in one sitting. Details are in
[BENCHMARKS](BENCHMARKS.md#the-sigmoid-silu-containers-against-amds-stack-more-accurate-on-all-5000-coco-images-faster-on-yolov8n-and-yolov8n-pose-slower-on-yolov8s-2026-09-17-desktop-2).

| Model | AMD's stack mAP | Container mAP | `--silu-sigmoid` mAP | AMD's stack G2G | `--silu-sigmoid` G2G |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 26.68 | 27.10 | 34.12 | 10.741 ms | 8.723 ms |
| YOLOv8s | 37.31 | 37.21 | 42.37 | 16.986 ms | 18.427 ms |
| YOLOv8n-pose (OKS) | 32.64 | 32.71 | 44.16 | 12.340 ms | 9.324 ms |

- **The flag costs 2.2-3.9 % of the NPU dispatch** across the sittings, which is 0.30-0.42 ms glass-to-glass against
  the same container without it.
- **Opt-in:** YOLO11n and YOLO-World v2 cannot use it (host regions), and SESR M7 has no SiLU.

```bash
bash scripts/research-iron.sh -m ignite_xdna.compiler.cli compile --model models/yolov8n_cut_xint8.onnx --output build/yolov8n_sigmoid.ignite --silu-sigmoid
bash scripts/research-iron.sh tools/verify_engine_container.py --container build/yolov8n_sigmoid.ignite --model models/yolov8n_cut_xint8.onnx
```

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

## Classification models: one of fifteen reaches a schedule (2026-09-18, Desktop 2)

`tools/sweep_model_zoo_classifiers.py` puts every XINT8 classification model under `models/` through the real
compiler stages offline — `lower_yolov8n`, then `plan_workspace`, then `schedule_graph` — and records where each
stops. No device, no container written, nothing about cost or accuracy. Log:
[`results/aie/model_zoo_classifier_compile_20260918T142628Z.log`](../results/aie/model_zoo_classifier_compile_20260918T142628Z.log).
The filter keeps one variant per family; the `resnet50_r192/r256/r320` rows survived it and are kept because they
show resolution is not what blocks ResNet.

| Model | Stage reached | Layers | What stopped it |
|---|---|---:|---|
| `resnet50_xint8_c64` | lower | 0 | `/maxpool/MaxPool: unsupported maxpool` |
| `resnet50_r192 / _r256 / _r320_xint8_c64` | lower | 0 | same — resolution does not move the blocker |
| `wide_resnet50_2_xint8_c64` | lower | 0 | same |
| `wide_resnet101_2_xint8_c64` | lower | 0 | same |
| `resnext50_32x4d_xint8` | lower | 0 | same |
| `densenet121_xint8` | lower | 0 | `/features/pool0/MaxPool: unsupported maxpool` |
| `resnetv2_50x3_xint8` | lower | 0 | `/stem/conv/Conv: unexpected consumers ['MaxPool']` |
| `regnetx_002_xint8` | lower | 0 | `/s1/b1/conv2/conv/Conv: group 3 convolution (24 -> 24) is not supported; only depthwise` |
| `mobilevit_xint8`, `mobilevit_xxs_xint8`, `mobilevit_cut_backbone_xint8` | lower | 0 | `/stages.2.0/conv3_1x1/conv/Conv: unexpected consumers ['Concat', 'Conv']` |
| `yolov8n-cls_cut_xint8` (224) | schedule | 28 | `a 14-pixel map is smaller than one 20-pixel tile` |
| **`yolov8n-cls_640_cut_xint8`** | **scheduled** | **28** | — 27 layers on the device, 1 host region, 12.6 MB |

- **The stem pool is the wall.** Nine of the fifteen stop at a 3x3 stride-2 `MaxPool` the core does not implement
  (it takes only the SPPF 5x5 pad-2 form), before pooling, grouping or the head are ever reached. `regnetx_002`
  adds grouped convolutions, and the `resnetv2` / `mobilevit` refusals are the graph walk declining nodes whose
  consumers it cannot express.
- **The 20-pixel tile floor decides the input size.** A /32 network at 224 ends at 7x7, and `yolov8n-cls` at 224
  lowers all 28 layers and then dies at 14x14; only the 640 re-export clears it. That is why the one survivor is
  8x the arithmetic of a 224 classifier and why its latency is not comparable to the Vitis AI EP rows above.
- **The survivor is a hybrid.** Its pooling runs as a declared host region — nothing in the engine computes a
  global average — so it is not the zero-CPU-fallback result the graph-engine bar asks for. Measurement and the
  reasoning: [A whole classifier on the NPU](BENCHMARKS.md#a-whole-classifier-on-the-npu-the-heads-pooling-is-carved-to-the-host-2828-layers-exact-2026-09-18-desktop-2).
- **Not swept:** families with no XINT8 file on disk (vgg, efficientnet, inception, senet, convnext, swin,
  resnet18/34, googlenet), and `mobilenetv2`, which exists here only as an AdaRound variant.

## Caveats

- One still image for every run: detection decode cost depends on what is in the frame
  (six objects here), and host staging includes the 810 × 1080 resize.
- NPU latency on this machine drifts between sessions; compare these rows with each other,
  not with numbers from another day.
- The CPU rows are ONNX Runtime's CPU provider on the same XINT8 QDQ models, not a tuned
  CPU implementation.

## Which current detection architectures could lower at all (2026-09-20, Desktop 2)

A node census of the families `ultralytics` 8.4.142 ships, built from their YAMLs with random
weights and exported at 640, scored against what the graph engine accepts. Structural only: it
can rule a model out, never in. `tools/arch_compat_audit.py`,
[`arch_compat_current_models_20260920.log`](../results/aie/arch_compat_current_models_20260920.log).

**YOLO26 is the newest family in this toolchain.** The families present are 26, 12, 11, v10,
v9, v8, v6, v5, v3 and rt-detr; there is no YOLO27 here, and whether one exists elsewhere was
not established.

Every row reports Div, Gather, Shape, Sub and one Softmax - that is the DFL and anchor decode
tail the head cut removes, and YOLOv8n lowers today carrying exactly it. The column that
matters is what is left after that, because it is what would go to a host segment.

| model | nodes | convs | unsupported conv shapes | host burden beyond decode |
|---|---:|---:|---|---|
| YOLO26n | 496 | 102 | **none** | MatMul 4, Softmax 2 |
| YOLO26s | 497 | 102 | **none** | MatMul 4, Softmax 2 |
| YOLO12n | 744 | 120 | 7x7 stride 1 x8 | MatMul 16, Softmax 9 |
| YOLO11n | 431 | 88 | none | MatMul 2, Softmax 2 |
| YOLOv10n | 399 | 83 | 7x7 stride 1 x1 | MatMul 2, Softmax 2 |
| YOLOv8n | 316 | 64 | none | Softmax 1 |

YOLO26 is the most engine-compatible modern family: no conv shape it needs is missing, and its
non-decode remainder is one attention block's worth - the same shape of problem as YOLO11n's
C2PSA, which the engine already carves to the host at one ONNX Runtime call and 0.51-0.53 ms
per frame. YOLO12 is a much larger lift: area attention runs through the whole backbone, and
eight 7x7 convolutions are unsupported at any stride.

That matters beyond compatibility. The engine's widest margin over AMD's stack anywhere in the
zoo is YOLO11n, 10.46 / 10.50 ms against 36.77 / 38.30, and the reason is that the Vitis AI EP
places almost none of an attention block. Modern detectors are attention-heavy, so the
architectures that are hardest for AMD's stack are the ones this engine already has a route
for.

Not established: nothing here was exported with real weights, head-cut, quantized or lowered;
whether YOLO26's 8 grouped convolutions are all depthwise (anything else is refused); whether
an NMS-free family's head cut removes the same tail; and channel counts against the
multiple-of-32 rule or map sizes against the 20-pixel tile floor.

### YOLO26n is bit-exact on the engine (2026-09-20, Desktop 2)

107 of 107 layers exact on Device 0, with both attention cores carved to the host -
[`yolo26n_exact_20260920.log`](../results/aie/yolo26n_exact_20260920.log).

| arm | layers exact | NPU dispatch (20) | host segments |
|---|---|---:|---:|
| band-packed | **107/107** | 8.963 ms | 1.342 ms |
| plane-major control | **107/107** | 9.319 ms | 1.380 ms |

107 layers, 2 on the host, container 12.27 MB; segments npu 0..38 (1,392 tasks), host,
npu 39..93 (1,820), host, npu 94..107 (266). One compiler change was needed: YOLO26's SPPF
`cv1` has no activation and feeds MaxPool/Concat directly, where YOLOv8's carries SiLU before
the same fan-out.

This first read 38/107 and was recorded as a broken attention carve. That was wrong. The fault
was a regression in the band-packed layout: `HostStep.run` stages a host segment through a
plane-major reshape, and banding the segment's output tensor wrote the right values to the
wrong addresses. Host-written tensors are now excluded from banding, since the host path is
CPU numpy and gains nothing from a layout whose only purpose is DMA merge depth. YOLO11n, the
one shipped model with a host segment, re-checks at 84/84 exact.

Layer-exactness says the container reproduces the quantized ONNX graph. It says nothing about
detection quality, and there is none yet: **YOLO26 needs a new decoder**, because
`decode_native.c` assumes DFL and YOLO26 regresses four box values directly and is NMS-free.
No mAP, no detections, no comparison against AMD's stack.

### YOLO26n detects on the NPU (2026-09-20, Desktop 2)

`npu/yolo26_decode.py` is the anchor decode the head cut removes. YOLO26 dropped DFL, so its
box branch regresses four values per anchor where YOLOv8 emits `4 x REG_MAX` bins reduced
through a softmax-weighted sum; everything else is identical, so the result is a drop-in for
`npu.yolo.postprocess`. Checked against the float export's own `output0`: box channels
bit-exact, class channels `1.4e-07` -
[`yolo26n_detections_20260920.log`](../results/aie/yolo26n_detections_20260920.log).

| arm | detections on bus.jpg |
|---|---|
| full float export | 5 - c5@0.90, c0@0.85, c0@0.85, c0@0.84, c0@0.56 |
| float cut, this decoder | 5 - identical to the export |
| XINT8 cut, ONNX Runtime CPU | 6 - c5@0.92, c0@0.82, c0@0.82, c0@0.73, c0@0.73, c11@0.32 |
| **the container on Device 0** | **6 - c5@0.924, c0@0.818, c0@0.818, c0@0.731, c0@0.731, c11@0.321** |

The device reproduces the quantized CPU model exactly: a bus, four people and one
low-confidence false positive that XINT8 also produces on CPU, so the extra box is the
quantizer and not the engine.

~~No mAP: six detections on one image is a smoke test, and this repo's own rule is that even a
500-image slice is not the answer.~~ **Measured 2026-09-21 on the full val2017 5,000: the engine
reads 23.74 mAP@50-95 / 39.40 @50, AMD's EP on the same XINT8 model 23.64 / 39.16, and the FLOAT
cut model 39.66 / 55.87 - so XINT8 costs 15.92 mAP, a 40 % relative drop, while the two
accelerators agree to 0.10.** The loss is the quantization recipe, not the engine. **Most of it was
recovered the same day:** the activation form is 10.11 of the 15.92 (isolated by putting Quark's
HardSigmoid into the float graph alone), and `--silu-sigmoid` refused only because of an
over-broad guard - the real conflict needs a host region *containing a SiLU*, and an attention core
contains none. With the guard corrected the container builds, verifies 107/107 exact and reads
**32.55 mAP at 2.49x AMD**, against 23.74 at 2.53x: **8.81 mAP for 0.26 ms**. See
[BENCHMARKS](BENCHMARKS.md#yolo26n-coco-map-253x-faster-than-amd-and-1592-map-poorer-than-its-own-float-model-2026-09-21-desktop-2).
The original caveat is kept below as written. No latency figure either - the dispatch in that log is one
cold run through a debug script that reads six tensors back separately. And the decoder is
numpy only, where YOLOv8's decode has a native C path, so a fair glass-to-glass comparison
must either add one or say plainly that the host tail is unoptimized.

### YOLO26n against AMD's stack: 1.43x glass to glass, because attention stops the Vitis AI EP (2026-09-20, Desktop 2)

How much of each model the two stacks place on the NPU, from AMD's own
`vitisai_ep_report.json`:

| model | total nodes | on the NPU | CPU | VITIS_EP_CPU |
|---|---:|---:|---:|---:|
| YOLO26n | 1,526 | **12** | 445 | 1,069 |
| YOLO11n | 1,300 | 6 | 382 | 912 |
| YOLOv8n | 929 | 922 | 0 | 7 |

AMD's EP takes 12 nodes of 1,526. The engine runs 107 of 107 layers, two of them - the
attention cores - on the host by declaration. That is the same collapse YOLO11n shows, and it
is the reason a modern detector is worth measuring here at all.

`tools/bench_container_vs_amd.py`, bus.jpg, 50 warm-up and 500 timed frames per run, arms
alternating, `xrt-smi` idle before each -
[`yolo26n_vs_amd_20260920.log`](../results/aie/yolo26n_vs_amd_20260920.log):

| arm | round 1 | round 2 | mean | preprocess | forward | decode |
|---|---:|---:|---:|---:|---:|---:|
| AMD Ryzen AI 1.7.1 | 43.099 | 43.118 | 43.109 | 3.50 | 34.47 | 5.13 |
| the engine | 30.124 | 30.260 | **30.192** | 3.16 | 22.46 | 4.57 |

**12.92 ms faster glass to glass, 1.43x**; on the network alone, 34.47 to 22.46 ms, 1.53x.
Both arms returned 6 detections on every frame and shared the letterbox, the decoder and the
NMS, so the only difference is what executes the network.

**RESOLVED 2026-09-21: the corrected figure is 2.53x, and it is measured.** The model was
re-exported, re-cut, re-quantized and the container rebuilt (the artifacts had been lost), giving
an identical lowering - 364 cut nodes, identical op census, 107 layers, the same `insts.bin` size,
107/107 layers exact on Device 0. In one sitting of three alternating arms: AMD **44.039 ms**, the
engine through the old tool **30.434 ms** (1.45x, replicating the 1.43x below to 0.8 %), and the
engine through the corrected tool **17.414 ms** - **2.53x**, six detections on every arm. The
benchmark's host tax on this model was 13.020 ms. The engine did not get faster; its dispatch is
unchanged. See `results/aie/yolo26n_requant_vs_amd_20260921.log` and
[BENCHMARKS](BENCHMARKS.md#yolo26n-re-derived-and-253x-amd-once-the-benchmark-stops-charging-it-13-ms-of-numpy-2026-09-21-desktop-2).
The figures below stand as measured and are superseded, not deleted.

**Original caveat, 2026-09-21.** The benchmark was
filling the engine arm's input plane with numpy rather than calling the AVX2 ingress the shipped
pipeline uses, which costs about 12 ms on a 640x640 frame and sat inside the `network` column
above. Both numbers here are therefore an **understatement** of the container, not an
overstatement. The corrected figure was **not measured** and none is invented: the sitting could
not be re-run because `yolo26n_cut_xint8.onnx` was lost with a removed worktree and AMD's arm
needs it, though `build/yolo26n.ignite` survives. See
[BENCHMARKS](BENCHMARKS.md#the-benchmark-was-charging-the-engine-arm-12-ms-of-numpy-its-own-pipeline-never-pays-2026-09-21-desktop-2),
where the same fix is measured on YOLOv8n and YOLOv8s. The figures above stand as measured.

What this is not. It is **not an Ignition number**: this is ignite-xdna's benchmark driving the
container directly, and Ignition's pipeline has no YOLO26 decode path, so the app cannot run
this model and no README badge may carry these figures. Neither arm has a native decode -
`decode_native.c` assumes DFL, which YOLO26 dropped - so preprocess and decode are numpy on
both sides and cost about 8.7 ms of every frame. The engine arm additionally pays one readback
per head where the EP returns all six from one call, which makes 1.43x a floor. And there is
**no accuracy figure**: six detections on one image is a smoke test, not mAP.
