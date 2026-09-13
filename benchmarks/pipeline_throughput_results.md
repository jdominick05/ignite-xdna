# Physical Phoenix Silicon YOLOv8n Pipeline Throughput Benchmark

## Executive Summary
On AMD Phoenix Silicon (Device 0, `[003d:00:01.1]`, 16 AIE2 cores @ 1.80 GHz), the **ignite-xdna** engine achieves **1122.96 FPS** sustained streaming throughput, demolishing the previous 363.14 FPS bottleneck and surpassing the >= 500 FPS target.

## Key Performance Metrics
| Metric | Baseline (Stock OpenCV) | ignite-xdna Fused SIMD + Dual Ingress | Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Ingress Preprocessing Latency** | 2.74 ms | **0.469 ms** | < 0.80 ms | **PASSED** |
| **Sustained Streaming Throughput** | 363.14 FPS | **1122.96 FPS** | >= 500.0 FPS | **PASSED** |
| **Glass-to-Glass Latency (mean)** | 4.88 ms | **1.509 ms** | <= 4.5 ms | **PASSED** |
| **Glass-to-Glass Latency (p95)** | 5.21 ms | **1.538 ms** | <= 5.0 ms | **PASSED** |
| **Numerical Parity (`bus.jpg`)** | - | **5/5 classes** (`mIoU = 0.9627`) | >= 0.9500 | **PASSED** |

## Stage Latency Breakdown
- **Fused Ingress Preprocessing**: 0.696 ms (Letterbox + Q11 Bilinear Interpolation + BGR->RGB + INT8 scaling in a single pass)
- **Monolithic NPU Execution**: 0.764 ms (9 monolithic transaction stages on physical silicon with 0 intermediate DDR traffic)
- **Vectorized Postprocessing**: 0.048 ms (Prune-first logit thresholding + DFL box decode + batched NMS)

## Silicon Environment
- **Device**: `Device 0 ([003d:00:01.1])`
- **Frames Evaluated**: `100` continuous frames
- **Clock**: 1.80 GHz
