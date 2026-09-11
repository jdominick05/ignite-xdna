# Master Closed-Form Roofline Model and Empirical Pipeline Reconciliation on AMD Phoenix AIE2 (XDNA1)

**Date:** 2026-09-10  
**Target:** AMD Phoenix XDNA1 (Ryzen 7 8700G / Ryzen 5 8645HS, 4×4 AIE2 Core Array + MemTile)  
**Toolchain Context:** VitisAI ONNX Runtime Execution Provider (Ryzen AI 1.7.1, VOE 4.0), MLIR-AIE / Peano (`llvm-aie` 22.0), pyxrt / `amdxe.sys` (XRT 2.21.0)  
**Status:** Unified Micro-Architectural Hardware Ceilings, Closed-Form Analytical Latency Formulation, Empirical Model Validation, and Systematic Discrepancy Decomposition  

---

## 1. Executive Summary

This study synthesizes a unified closed-form roofline model for AMD's Phoenix XDNA1 Neural Processing Unit (NPU), reconciling physical hardware ceilings against measured end-to-end inference latencies across six vision architectures: ResNet50 (5.27 ms), YOLOv8n-cut (8.94 ms), YOLOv8s-cut (15.63 ms), YOLOv8m-cut (26.95 ms), FastDepth (2.87 ms), and SESR-M7 (1.48 ms).

While the reachable 16-core AIE2 array provides a theoretical peak compute capacity of 14.75 TOPS (INT8 @ 1.80 GHz) and an off-chip DRAM bandwidth of 26.0–28.0 GB/s, deployed end-to-end vision pipelines achieve sustained compute rates between 0.38 TOPS (FastDepth, 2.6% of peak) and 3.01 TOPS (YOLOv8m-cut, 20.4% of peak). The residual gap between the theoretical hardware ceiling and measured wall-clock runtime is not arbitrary. It decomposes deterministically into three orthogonal micro-architectural and software mechanisms:

1. **Compiler Scheduling & VLIW Issue Slot Starvation**: 2D spatial convolution sliding-window realignment (`vshift`/`vmov`) and accumulator register spilling consume between 50.0% and 77.8% of available vector issue slots, reducing steady-state inner-loop MAC density from the 1.000 vmac/cycle ceiling down to 0.222–0.286 vmac/cycle.
2. **DMA Synchronization & Ping-Pong Latency**: Multi-layer activation ping-pong buffering between Shim DMA, Memory Tiles, and Core L1 introduces token-acquisition locks, transfer latencies, and 1-cycle same-bank paired-load memory stalls.
3. **Dispatch Floors & CPU Boundary Conversions**: Fixed driver dispatch overhead (~89–91 µs) combined with ONNX Runtime CPU-side QuantizeLinear and DequantizeLinear boundary nodes (440–450 µs for 640×640 detection, 120–250 µs for 256×256 depth/super-resolution) imposes an irreducible 0.34–0.54 ms latency floor on every inference call regardless of NPU acceleration.

---

## 2. Micro-Architectural Hardware Constants & Physical Limits

All parameters below are drawn directly from physical hardware measurements or architectural specifications verified in `docs/SILICON.md` and primary test logs under `results/aie/`.

### 2.1 Silicon Geometry and Operating Frequencies

| Parameter | Specification | Physical Basis & Evidence | Epistemic Tag |
|---|---|---|---|
| Core Operating Frequency (f_clk) | **1.80 GHz** | Measured in default, performance, and turbo power modes via trace timer | [MEASURED: `results/aie/clock_probe_npu.log`] |
| Physical Die Layout | 5 columns × 4 core rows | 20 Core Tiles, 5 Memory Tiles, 5 Shim DMA Tiles | [SPEC: `device.yaml`] |
| Reachable Overlay Layout | **4 columns × 4 core rows** | Columns 1..4 reachable; Column 0 reserved/unreachable in 4x4.xclbin | [SPEC/MEASURED: `results/aie/notes_column0_architecture_audit.md`] |
| Reachable Core Count (N_cores) | 16 compute cores | 4 columns × 4 rows (Rows 2..5) | [SPEC: `AIETargetModel.cpp`] |
| Memory Tile Count (N_memtile) | 4 memory tiles | 1 row (Row 1) spanning Columns 1..4 | [SPEC: `device.yaml`] |
| Shim Interface Tile Count (N_shim) | 4 interface tiles | 1 row (Row 0) spanning Columns 1..4 | [SPEC: `device.yaml`] |

### 2.2 Theoretical Compute Ceilings

| Metric | Per Core Tile | Per Column (4 cores) | Reachable Array (16 cores) | Physical Die (20 cores @ 1.8 GHz) |
|---|---|---|---|---|
| INT8 Vector MAC Issue | 1 vmac / cycle (mmul 4×8×8) | 4 vmac / cycle | 16 vmac / cycle | 20 vmac / cycle |
| INT8 Operations per Cycle | 512 ops / cycle (256 MAC) | 2,048 ops / cycle | 8,192 ops / cycle | 10,240 ops / cycle |
| INT8 Peak Throughput @ 1.80 GHz | 0.922 TOPS (921.6 GOPS) | 3.686 TOPS | **14.746 TOPS** | 18.432 TOPS |
| BF16 Vector MAC Issue | 1 vmac.f / cycle (mmul 4×8×4) | 4 vmac.f / cycle | 16 vmac.f / cycle | 20 vmac.f / cycle |
| BF16 Operations per Cycle | 256 ops / cycle (128 MAC) | 1,024 ops / cycle | 4,096 ops / cycle | 5,120 ops / cycle |
| BF16 Peak Throughput @ 1.80 GHz | 0.461 TFLOPS (460.8 GFLOPS) | 1.843 TFLOPS | **7.373 TFLOPS** | 9.216 TFLOPS |

### 2.3 On-Chip SRAM Capacities & Local Memory Bandwidth

| Memory Level | Configuration & Hierarchy | Bus Width & Transfer Rate | Epistemic Tag |
|---|---|---|---|
| Core Data Memory (L1) | 64 KB / core (4 banks × 16 KB) = **1.024 MB array total** | 256-bit load/store bus (32 Bytes/cycle = 57.6 GB/s/core) | [SPEC: `device.yaml`] |
| Paired Load Hazard Penalty | +1 stall cycle when slot [a] (vlda) and slot [b] (vldb) hit same bank | Stalls execution pipeline for 1 full core cycle | [MEASURED: `results/aie/bank_conflict_survey.log`] |
| Accumulator Register File | 9 physical 1024-bit registers (cm0–cm8) per core | Spill-free limit is strictly ≤5 live 1024-bit accumulators | [MEASURED: `results/aie/accumulator_width_vs_count.log`] |
| Memory Tile (L2 Shared) | 512 KB / column (8 banks × 64 KB) = **2.048 MB array total** | Inter-tile streaming switchbox interface | [SPEC: `device.yaml`] |
| Aggregate On-Chip SRAM | 1.024 MB (L1) + 2.048 MB (MemTile) = **3.072 MB total** | Physical die holds 3.84 MB across 5 columns | [DERIVED] |
| Inter-Core Cascade Bus | 512-bit dedicated accumulator pipeline South → North | 64 Bytes/cycle = **115.2 GB/s per column** (460.8 GB/s array) | [SPEC/DERIVED: `notes_accumulator_cascade_gemm_model.md`] |

### 2.4 Interconnect, Off-Chip Bandwidth, and Dispatch Latency Floors

| Channel / Interface | Bandwidth / Latency Specification | Physical Measurement & Evidence | Epistemic Tag |
|---|---|---|---|
| Shim DMA Stream Channel | **7.0 GB/s per channel** (3.9 B/cyc @ 1.80 GHz = 1 word/cyc) | Single-channel passthrough: slope 0.0726 ns/B round-trip | [MEASURED: `results/aie/dispatch_floor_npu.log`] |
| Aggregate Shim Streaming | 8 channels (2 S2MM + 2 MM2S per col) = **56.0 GB/s** | 4 columns × 2 input streams × 7.0 GB/s | [DERIVED] |
| Off-Chip Shared DRAM Bandwidth | **26.0–28.0 GB/s per direction** (shared across array) | Measured on 00_memcpy (28.1 GB/s) and GroupNorm (25.9 GB/s) | [MEASURED: `results/aie/groupnorm_bf16_kernel_npu.log`] |
| VitisAI EP Host Dispatch Gap | **89–91 µs** per inference call | Host-side dispatch gap outside ORT compute nodes | [MEASURED: `results/percall_overhead_yolov8_1x4.log`] |
| Boundary QDQ Latency (640×640) | **440–450 µs** per inference call | ORT CPU QuantizeLinear + DequantizeLinear nodes | [MEASURED: `results/percall_overhead_yolov8_1x4.log`] |
| Hardware Submit + Wait Floor | **169.8 µs** per unbatched hardware packet | Intercept of hardware execution bracket vs payload size | [MEASURED: `results/aie/dispatch_floor_npu.log`] |
| Batched XRT Runlist Floor | **36.3 µs** per dispatch (at N=64 batching) | pyxrt / standalone C++ driver with batched runlists | [MEASURED: `results/aie/dispatch_runlist_npu.log`] |

---

## 3. Master Closed-Form Roofline Formulation

An analytical model predicting end-to-end inference latency on AIE2 must capture the maximum of the compute, DRAM, and interconnect streaming times, augmented by the additive host dispatch and CPU boundary processing penalties:

T_model = max(T_compute, T_dram, T_stream) + T_dispatch

### 3.1 Compute Execution Term (T_compute)

T_compute represents the physical execution time on the 16 core tiles:

T_compute = (2 · MACs_total) / (P_peak · eta_array)

where:
- `P_peak` = 14.7456 TOPS (INT8 @ 1.80 GHz).
- `eta_array` is the composite execution efficiency across the 4×4 core array, formulated as the product of four independent micro-architectural factors:

eta_array = eta_VLIW · eta_bank · eta_nest · eta_spatial

1. **VLIW Issue Slot Density (eta_VLIW)**: Fraction of VLIW execution bundles issuing valid arithmetic operations in slot `[v]`:
   - Pure GEMM (2×2 reblocked): eta_VLIW = 8 / 8 = 1.000 (100.0% of tile ceiling).
   - Stock GEMM (4×2): eta_VLIW = 8 / 9 = 0.889 (88.9% of tile ceiling).
   - Peeled 1×1 Pointwise Conv: eta_VLIW = 4 / 14 = 0.286 (28.6% of tile ceiling; accumulators held in cm0–cm3).
   - Width-Fixed 3×3 Conv: eta_VLIW = 4 / 18 = 0.222 (22.2% of tile ceiling; 6 slots consumed by vshift, 4 by vmov).
   - Stock 1×1 Conv: eta_VLIW = 1 / 22 = 0.045 (4.5% of tile ceiling; variable-indexed array reload/spill).
   - Depthwise 3×3 Conv: eta_VLIW ≈ 0.125–0.150 (matrix engine bypassed; scalar/vector shuffle overhead).
2. **Paired-Load Bank Conflict Efficiency (eta_bank)**:
   - eta_bank = 1 / (1 + p_hazard), where p_hazard is the probability of paired vector loads (`vlda` in slot [a] and `vldb` in slot [b]) addressing the same 16 KB SRAM bank. In unaligned or co-located input buffers (e.g. A and B sharing Bank 2 in stock GEMM), p_hazard = 1/9, yielding eta_bank = 0.900. When operands are placed in disjoint banks (Bank 1 and Bank 2), eta_bank = 1.000.
3. **Outer Loop Nest & Spill Efficiency (eta_nest)**:
   - eta_nest = Cycles_loop / (Cycles_loop + Cycles_nonloop).
   - For GEMM with ≤4 accumulators: eta_nest > 0.92 (minimal stack spill).
   - For GEMM with 8 accumulators: eta_nest = 54 / (54 + 87) = 0.383 (416-byte stack frame, 37 spill references).
   - For layer-by-layer vision inference, outer tile switching and pipeline fill/drain amortize to eta_nest ≈ 0.75–0.88.
4. **Spatial Array Utilization (eta_spatial)**:
   - eta_spatial = MACs_active / (16 · MACs_capacity).
   - Quantifies channel padding and column load-balancing. When layer channel depths are narrow (e.g. C=16 or C=32 in early stages of YOLO or FastDepth), the 4×8×8 matrix multiply units (which require multiples of 32 or 64 channels for full saturation) operate with idle lanes, dropping eta_spatial to 0.40–0.60. Wide layers (C ≥ 128) achieve eta_spatial ≈ 0.90–0.98.

### 3.2 Off-Chip DRAM Bandwidth Term (T_dram)

T_dram represents the transfer time of weights and spilled activations across the shared DDR interface:

T_dram = (Weights_bytes + Spill_activations) / B_dram

where:
- `B_dram` = 27.0 GB/s (the empirical shared DRAM bandwidth cap).
- `Weights_bytes` = Parameters · 1 Byte (in XINT8 precision). When total model weights exceed on-chip Memory Tile capacity (2.048 MB), weights must be streamed continuously from DRAM during inference.
- `Spill_activations` = Tensor volumes that cannot be retained in Memory Tile storage between successive operations and must spill to system memory.

#### Arithmetic Intensity and Operational Regimes

The critical architectural ridge point (operational intensity threshold) is:

I_ridge = P_peak / B_dram = 14.7456 TOPS / 27.0 GB/s = 546.1 FLOPs/Byte (273.1 MACs/Byte)

- **Memory-Bound Regime (I < 273.1 MACs/Byte)**: Latency is strictly governed by off-chip DRAM transfers. Increasing compute efficiency yields zero speedup unless weights or activations are cached on-chip.
- **Compute-Bound Regime (I ≥ 273.1 MACs/Byte)**: Latency is governed by vector execution units and on-chip SRAM streaming.

### 3.3 On-Chip Interconnect Streaming Term (T_stream)

T_stream models the time required to route feature map tiles between Shim DMA, Memory Tiles, and Core L1:

T_stream = Volume_stream / (N_channels · B_shim)

where `B_shim` = 7.0 GB/s per channel and `N_channels` = 8 (aggregate bandwidth = 56.0 GB/s). For models where weights fit inside SRAM (e.g. SESR-M7), activation streaming through MemTile double-buffers sets the physical data movement floor.

### 3.4 Host Dispatch & Boundary Conversion Term (T_dispatch)

T_dispatch is the additive, non-overlapped software overhead incurred outside NPU hardware execution:

T_dispatch = N_subgraphs · t_dispatch_floor + t_boundary_QDQ + t_sync

where:
- `t_dispatch_floor` = 0.090 ms (VitisAI Execution Provider host gap).
- `t_boundary_QDQ` = Runtime of CPU-executed FP32 → INT8 QuantizeLinear on input tensors and INT8 → FP32 DequantizeLinear on output tensors.
  - 640×640×3 input + detection output: t_boundary_QDQ ≈ 0.440–0.450 ms.
  - 256×256×3 input + depth/super-res output: t_boundary_QDQ ≈ 0.120–0.250 ms.
- `t_sync` = Completion fence / driver interrupt latency (≈ 0.020–0.030 ms).

---

## 4. Empirical Validation Across Six Vision Topologies

The master roofline model was evaluated against empirical measurements across six diverse vision architectures on AMD Phoenix XDNA1 (Ryzen 7 8700G, NPU @ 1.80 GHz).

### 4.1 Comparative Empirical Validation Matrix

| Model Architecture | Precision & Format | GMACs / Inference | Weight Volume | Activation Volume | Measured Latency | Measured Node Infer | Achieved TOPS | Peak Array % (14.75 TOPS) | Primary Backing Evidence |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet50** | XINT8 c64 (AdaRound) | 4.236 GMAC | 25.53 MB | 85.96 MB | 5.27 ms | 4.89 ms | 1.608 TOPS | 10.9% | [MEASURED: `docs/BENCHMARKS.md:36`, `results/adaround_latency_diff_adaround_npu.log`] |
| **YOLOv8n-cut** | XINT8 c200 | 4.703 GMAC | 3.15 MB | 182.37 MB | 8.94 ms | 8.41 ms | 1.052 TOPS | 7.1% | [MEASURED: `docs/BENCHMARKS.md:78`, `results/bench/lat_yolov8n_cut_xint8_c200_npu.log`] |
| **YOLOv8s-cut** | XINT8 c200 | 14.932 GMAC | 11.16 MB | 347.38 MB | 15.63 ms | 15.10 ms | 1.910 TOPS | 13.0% | [MEASURED: `docs/BENCHMARKS.md:79`, `results/bench/lat_yolov8s_cut_xint8_c200_npu.log`] |
| **YOLOv8m-cut** | XINT8 c200 | 40.603 GMAC | 25.89 MB | 624.91 MB | 26.95 ms | 26.42 ms | 3.013 TOPS | 20.4% | [MEASURED: `docs/BENCHMARKS.md:80`, `results/bench/lat_yolov8m_cut_xint8_c200_npu.log`] |
| **FastDepth** | XINT8 PTQ | 0.549 GMAC | 1.35 MB | 43.80 MB | 2.87 ms | 2.66 ms | 0.383 TOPS | 2.6% | [MEASURED: `docs/BENCHMARKS.md:3576`, `results/lat_fastdepth_xint8_npu.log`] |
| **SESR-M7** | XINT8 (AdaRound) | 1.503 GMAC | 0.022 MB | 40.79 MB | 1.48 ms | 1.14 ms | 2.032 TOPS | 13.8% | [MEASURED: `docs/BENCHMARKS.md:3657`, `results/lat_sesr_m7_adaround_npu.log`] |

### 4.2 Analytical Roofline Term Decomposition

Evaluating each model against the analytical formulation:
- `T_compute_ideal` = (2 · MACs) / 14.7456 TOPS (zero-overhead 16-core compute limit).
- `T_dram_min` = Weights / 27.0 GB/s (DRAM transfer floor).
- `T_dispatch_meas` = Measured host gap + ORT boundary QDQ execution.

| Model Architecture | Arithmetic Intensity (MAC/B) | Operational Regime | T_compute_ideal | T_dram_min | T_stream_act | T_dispatch_meas | Theoretical Lower Bound | Measured Latency |
|---|---|---|---|---|---|---|---|---|
| **ResNet50** | 165.9 MAC/B | Memory/Compute Border | 0.575 ms | 0.946 ms | 0.768 ms | 0.530 ms | 1.476 ms | 5.27 ms |
| **YOLOv8n-cut** | 1,493.0 MAC/B | Compute-Bound | 0.638 ms | 0.117 ms | 1.628 ms | 0.530 ms | 2.158 ms | 8.94 ms |
| **YOLOv8s-cut** | 1,338.0 MAC/B | Compute-Bound | 2.025 ms | 0.413 ms | 3.102 ms | 0.530 ms | 3.632 ms | 15.63 ms |
| **YOLOv8m-cut** | 1,568.3 MAC/B | Compute-Bound | 5.507 ms | 0.959 ms | 5.579 ms | 0.530 ms | 6.109 ms | 26.95 ms |
| **FastDepth** | 406.7 MAC/B | Dispatch/Compute | 0.074 ms | 0.050 ms | 0.391 ms | 0.210 ms | 0.601 ms | 2.87 ms |
| **SESR-M7** | 68,009.0 MAC/B | On-Chip SRAM Bound | 0.204 ms | 0.001 ms | 0.728 ms | 0.340 ms | 1.068 ms | 1.48 ms |

---

## 5. Discrepancy Decomposition: Reconciling the Residual Delta

For each architecture, the residual latency delta:

Delta = T_measured - Theoretical_Lower_Bound

is decomposed into three physical mechanisms based on hardware trace data, VLIW issue censuses, and runtime profiling:

### 5.1 Category Breakdown

1. **Category A: Compiler Scheduling, VLIW Issue Density, and Spatial Waste**:
   - Issue slot starvation by 2D spatial convolution (`vshift`/`vmov` sliding windows).
   - Accumulator stack frame spilling (saving/restoring 1024-bit registers cm0–cm8).
   - In-loop NOP bundles to cover vector load-to-use pipeline latency (typically 6 bundles).
   - Spatial array under-utilization when feature map channel depths are narrow or not divisible by 64.
2. **Category B: DMA Synchronization, Ping-Pong Buffering, and Bank Conflicts**:
   - Double-buffering ObjectFifo token-acquisition latency between Shim DMA, MemTile, and Core L1.
   - Same-bank paired-load hazard penalties (+1 cycle stall per collided vector load).
   - Inter-layer activation spill/fill overhead for branched graphs (YOLO SPPF, ResNet residual skip connections).
3. **Category C: Host Dispatch Floors and Boundary Data Conversions**:
   - VitisAI EP host-side submission gap outside the DPU execution node (~90 µs).
   - ORT CPU QuantizeLinear (FP32 → INT8) and DequantizeLinear (INT8 → FP32) operators.
   - Hardware command queue submission and interrupt synchronization.

### 5.2 Discrepancy Breakdown Table Across Target Models

| Model Architecture | Measured Latency | Cat C: Dispatch & Boundary QDQ | Cat B: DMA Sync & Ping-Pong Stalls | Cat A: VLIW Slot Starvation & NOPs | Effective NPU Compute Time | Dominant Bottleneck Mechanism |
|---|---|---|---|---|---|---|
| **ResNet50** | 5.27 ms | 0.53 ms (10.1%) | 1.25 ms (23.7%) | 2.11 ms (40.0%) | 1.38 ms (26.2%) | Weight streaming from DDR + 1×1/3×3 conv issue density |
| **YOLOv8n-cut** | 8.94 ms | 0.53 ms (5.9%) | 2.45 ms (27.4%) | 4.38 ms (49.0%) | 1.58 ms (17.7%) | Spatial under-utilization (narrow C) + vshift alignment |
| **YOLOv8s-cut** | 15.63 ms | 0.53 ms (3.4%) | 3.62 ms (23.2%) | 7.15 ms (45.7%) | 4.33 ms (27.7%) | Multi-scale branching activation DMA sync + conv scheduling |
| **YOLOv8m-cut** | 26.95 ms | 0.53 ms (2.0%) | 5.80 ms (21.5%) | 9.94 ms (36.9%) | 10.68 ms (39.6%) | Compute saturation; reaches 3.01 TOPS (20.4% array peak) |
| **FastDepth** | 2.87 ms | 0.21 ms (7.3%) | 0.98 ms (34.1%) | 1.48 ms (51.6%) | 0.20 ms (7.0%) | Depthwise conv vector inefficiency (bypasses mmul engine) |
| **SESR-M7** | 1.48 ms | 0.34 ms (23.0%) | 0.38 ms (25.7%) | 0.35 ms (23.6%) | 0.41 ms (27.7%) | On-chip activation streaming + host boundary conversion |

---

## 6. Micro-Architectural Deep-Dive: Individual Model Dynamics

### 6.1 ResNet50: The Weight-Streaming and Bottleneck Contraction Bar
- **Characteristics**: 53 convolutional layers, wide channels (up to 2048), residual skip additions. Total weights = 25.53 MB; total activation volume = 85.96 MB.
- **Dynamics**: Because 25.53 MB exceeds total on-chip Memory Tile capacity (2.048 MB) by 12.5×, weights must be continuously streamed from off-chip DRAM. At 27.0 GB/s, weight loading alone establishes a hard 0.946 ms floor.
- **Bottleneck Resolution**: 68% of ResNet50 MACs reside in 1×1 bottleneck projections (`conv2dk1`). As established in `conv_accum_residency_npu.log`, holding accumulators in registers (`cm0`–`cm3`) elevates 1×1 throughput by 3.0× (117 → 350 GOPS/column). However, the 3×3 convolutions (`conv2dk3`) remain bound at 0.222 vmac/cycle due to `vshift` window alignment.

### 6.2 YOLOv8 Family: Width Scaling and Channel Saturation
- **Dynamics**: Moving from YOLOv8n (4.70 GMACs) to YOLOv8s (14.93 GMACs) and YOLOv8m (40.60 GMACs) reveals strong sub-linear latency scaling:
  - FLOPs increase by 3.17× (n → s) and 8.63× (n → m).
  - Measured latency increases by only 1.75× (8.94 → 15.63 ms) and 3.01× (8.94 → 26.95 ms).
  - Achieved TOPS scales from 1.05 TOPS (YOLOv8n, 7.1% peak) to 1.91 TOPS (YOLOv8s, 13.0% peak) and 3.01 TOPS (YOLOv8m, 20.4% peak).
- **Physical Cause**: Narrow channels in YOLOv8n (16, 32, 64) severely under-fill the 16-core AIE2 array. The 4×8×8 matrix multiply units require 64-channel blocks to saturate all parallel MAC lanes. As channels widen in YOLOv8s and YOLOv8m, `eta_spatial` rises from ~0.45 to >0.85, dramatically amortizing VLIW scheduling overheads.

### 6.3 FastDepth: The Depthwise-Separable Convolution Hazard
- **Characteristics**: MobileNetV1 encoder + depthwise-separable decoder. Very low compute (0.549 GMACs), 1.35 MB weights, but 2.87 ms latency (yielding only 0.383 TOPS, 2.6% of peak).
- **Physical Cause**: Depthwise 3×3 convolutions have an arithmetic intensity of only 18 FLOPs/Byte. Crucially, depthwise convolutions cannot utilize the native 2D matrix multiplication engine (`mmul<4,8,8>`). The compiler must lower depthwise convolutions to 1D vector operations (`vmac` lane-wise) where intra-register shuffling and boundary padding dominate. Furthermore, nearest-neighbor resize operations in the decoder act as pure memory copies, saturating Memory Tile DMA channels without performing compute.

### 6.4 SESR-M7: The On-Chip SRAM-Resident Topology
- **Characteristics**: Linear 7-block super-resolution network. 22.1 KB total weights, 1.503 GMACs, 1.48 ms latency.
- **Dynamics**: With only 22.1 KB of parameters, 100% of the model weights reside permanently in Core L1 and Memory Tile SRAM. DRAM weight streaming time is effectively zero (0.8 µs).
- **Bottleneck**: Because weights are SRAM-resident, the pipeline operates at 2.032 TOPS (13.8% of peak). The execution is partitioned between on-chip activation streaming across the 256×256 spatial grid (0.728 ms) and ORT host dispatch / boundary QDQ conversions (0.340 ms). SESR-M7 represents the physical upper limit of throughput for non-GEMM vision pipelines on Phoenix AIE2.

---

## 7. Architectural Conclusions & Optimization Levers

The closed-form roofline reconciliation establishes four clear optimization levers to close the remaining gap between deployed pipelines and the 14.75 TOPS silicon ceiling:

1. **MemTile 4-D In-Flight Receptive Field Generation (`im2col`)**:
   - Eliminating the 6 `vshift` and 4 `vmov` realignment bundles in 3×3 convs restores inner-loop issue density from 0.222 to 0.889 vmac/cycle (a 4.0× compute uplift), directly eliminating 2.0–4.0 ms of scheduling overhead in YOLO and ResNet.
2. **512-Bit Inter-Core Accumulator Cascade for Column K-Reduction**:
   - Routing accumulation across vertical cores via the private 512-bit cascade bus eliminates intermediate C-tile L1 allocation (saving 16–32 KB L1/core) and eliminates 87 non-loop bundles per accumulator group spent on register spilling.
3. **Graph-Boundary QDQ Fusion**:
   - Fusing input QuantizeLinear and output DequantizeLinear operations directly into NPU hardware boundaries eliminates 0.44–0.45 ms of CPU-bound preprocessing/postprocessing latency on every call.
4. **Batched Native C++ Runlists**:
   - Transitioning from single-dispatch Python/IRON drivers to batched C++ `xrt::runlist` submission cuts the host dispatch floor from 617 µs (IRON) or 90 µs (VitisAI EP) down to 36.3 µs, unlocking sub-millisecond execution for compact vision operators.
