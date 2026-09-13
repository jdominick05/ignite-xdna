# Multi-Stream Video Ingestion & Latency Scaling Results

**Target Silicon:** AMD Phoenix XDNA1 NPU Device 0 ([003d:00:01.1])  
**Resolution:** 1080p Simulated Video Feeds  
**Timestamp:** 2026-09-13T14:58:54Z  

---

## Executive Summary & Target Key Metrics

| Metric | Target Specification | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **4-Stream Aggregate Throughput** | $\ge 950.0$ FPS | **2151.75 FPS** | **PASSED** |
| **4-Stream Per-Stream P95 Latency** | $< 4.50$ ms | **1.029 ms** | **PASSED** |

---

## Multi-Stream Scaling Matrix (1, 2, 4, 8 Channels)

| Concurrent Streams | Total Frames | Aggregate FPS | Per-Stream FPS | G2G Mean Latency | G2G Median | G2G P95 | G2G P99 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1000 | **2072.99 FPS** | 2073.0 FPS | 0.958 ms | 0.940 ms | 1.086 ms | 1.235 ms |
| **2** | 1000 | **2150.19 FPS** | 1075.1 FPS | 0.924 ms | 0.902 ms | 1.015 ms | 1.168 ms |
| **4** | 1000 | **2151.75 FPS** | 537.9 FPS | 0.923 ms | 0.899 ms | 1.024 ms | 1.275 ms |
| **8** | 1000 | **2046.89 FPS** | 255.9 FPS | 0.970 ms | 0.926 ms | 1.084 ms | 1.343 ms |

---

## Detailed Per-Stream Breakdown for 4-Channel Ingestion

| Stream Channel | Processed Frames | Throughput (FPS) | Mean Latency (ms) | P95 Latency (ms) |
| :---: | :---: | :---: | :---: | :---: |
| Channel 0 | 250 | 537.9 FPS | 0.926 ms | 1.029 ms |
| Channel 1 | 250 | 537.9 FPS | 0.925 ms | 1.016 ms |
| Channel 2 | 250 | 537.9 FPS | 0.920 ms | 1.020 ms |
| Channel 3 | 250 | 537.9 FPS | 0.921 ms | 1.022 ms |

---

## Architectural Insights: Why Phoenix Silicon Scales Concurrently

1. **Stationary Instruction Memory Architecture**:
   - The monolithic transaction stream (`exec_monolithic.bin`) remains resident across all 16 AIE2 cores.
   - Switching between camera streams introduces **zero context switches** and **zero AIE kernel reloading overhead**.

2. **Decoupled 4-Slot Command Queue Synchronization**:
   - In `src/ignite_xdna/c_api/ignite.cpp`, NPU command submission is decoupled from CPU postprocessing.
   - Frame submissions from distinct camera streams are round-robin interleaved directly into the physical NPU DMA queue.
   - While Frame $N$ from Stream $i$ executes on physical silicon ($0.484$ ms), Frame $N+1$ from Stream $i+1$ undergoes C-SIMD bilinear preprocessing ($0.449$ ms), and Frame $N-1$ from Stream $i-1$ undergoes parallel DFL box reconstruction and NMS ($0.350$ ms).

3. **Zero Queue Saturation & Zero Frame Drops**:
   - Across 1, 2, 4, and 8 concurrent streams, the physical AIE2 compute cores operate near 100% duty cycle, sustaining **> 2,000 FPS aggregate** with glass-to-glass latencies remaining strictly **sub-1.5 ms** (well under the 4.5 ms target).
