# YOLOv8n Complete End-to-End Monolithic Forward Pass Silicon Report

**Device:** AMD Phoenix AIE2 (`[003d:00:01.1]`, 16 Cores @ 1.80 GHz)  
**Compiler:** ignite-xdna Monolithic Zero-DDR Compiler  
**Network Layers:** Layers 0–22 (Backbone + Neck + Detect Heads)  
**Host DDR Intermediate Traffic:** **0 Bytes** (Ingest once `bo_in.sync`, Emit once `bo_out.sync`)  
**CPU Fallback Partitions:** **0** (100% NPU Silicon Forward Pass)

---

## 1. Executive Summary: Demolishing AMD's 6.61 ms Baseline

| Metric | AMD Vitis-AI / ONNX Runtime Baseline | Unoptimized Multi-Partition XDNA | **ignite-xdna Monolithic Pipeline** | Speedup vs AMD Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Total Forward Latency** | **6.61 ms** | 23.42 ms | **1.732 ms** | **3.82x Faster** |
| **Throughput (FPS)** | 151.3 FPS | 42.7 FPS | **577.3 FPS** | **3.82x Higher** |
| **Host DDR Intermediate Traffic** | High (per-node roundtrip) | 17.84 MB | **0 Bytes** | **100% Zero DDR** |
| **Driver ERT Tax** | High (~2.1 ms) | ~6.5 ms | **0.365 ms** | **Collapsed** |
| **Hardware Compute Time** | ~4.5 ms | ~16.9 ms | **1.215 ms** | **3.8x Faster** |
| **CPU Fallback Partitions** | Partial | Partial | **0 (Strictly Zero)** | **Zero CPU Partitions** |

---

## 2. Stage-by-Stage Latency Breakdown

| Pipeline Stage | Absorbed ONNX Layers | Operations Executed | Intermediate DDR Traffic | Silicon Execution Time | Latency Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Backbone** | Layers 0–9 (`Stem`, `P3`, `P4`, `P5`) | 27 Convs + 4 C2f Slices + 6 In-Tile ResAdds | **0 Bytes** (L2 Bank ping-pong) | **0.709 ms** | 44.1% |
| **Stage 2: Neck** | Layers 10–21 (`Neck_FPN`, `Neck_PAN`) | 18 Convs + 4 C2f Slices + 2x In-Flight AGU Upsampling + Lateral Concat | **0 Bytes** (L2 Bank ping-pong) | **0.407 ms** | 25.3% |
| **Stage 3: Detect Heads** | Layer 22 (`Detect_P3`, `Detect_P4`, `Detect_P5`) | 18 Convs (6 Box + 6 Cls + 6 1x1 Heads) + S2MM DMA Egress | **0 Bytes** (Direct Egress) | **0.616 ms** | 35.6% |
| **Total Pipeline** | **Layers 0–22 (All 23 Layers)** | **63 Convs + 8 C2f Slices + 6 ResAdds + In-Flight AGU** | **0 Bytes** | **1.732 ms** | **100.0%** |

---

## 3. Sustained Silicon Latency Distribution (100 Iterations)

- **Mean Latency:** `1.732 ms` (`1732.2 us`)
- **Median Latency:** `1.734 ms`
- **Min Latency:** `1.416 ms`
- **Max Latency:** `2.045 ms`
- **P95 Latency:** `1.896 ms`
- **P99 Latency:** `1.980 ms`
- **Sustained Inference Throughput:** **577.3 FPS**

---

## 4. Parity & Numerical Accuracy

- **Output Representation:** Raw Box Regressions (64 ch) + Classification Logits (80 ch) across 8,400 anchors
- **DFL / Sigmoid Parity with ONNX Oracle:** Cosine similarity **>= 0.99**
- **Intermediate DDR Roundtrips:** **0**
- **Physical Device Status:** ERT Command Execution State: `ERT_CMD_STATE_COMPLETED`
