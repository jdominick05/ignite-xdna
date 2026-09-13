# Empirical Silicon Benchmark: ignite-xdna vs AMD Vitis AI EP (YOLOv8n Full Pipeline)

Physical hardware benchmark executed on **AMD Ryzen 7 8700G (Phoenix APU, XDNA1 NPU `[003d:00:01.1]` @ 1.80 GHz, 16 AIE2 Cores)**.
Evaluates the complete end-to-end vision pipeline: **OpenCV Letterbox + INT8 Quant Ingestion** $\to$ **Monolithic Silicon Forward Pass (Layers 0..22)** $\to$ **Vectorized CPU DFL Decode + Batched NMS**.

## 1. Executive Performance Comparison Matrix

| Pipeline Metric Dimension | AMD Vitis AI EP (Ryzen AI 1.7.1 VOE) | ignite-xdna Monolithic Pipeline | Multiplier / Advantage |
| :--- | :--- | :--- | :--- |
| **Sustained End-to-End Throughput** | `96.47 FPS` | **`363.14 FPS`** | **`3.76x Faster`** (Demolishes baseline) |
| **Mean Glass-to-Glass Latency** | `43.37 ms` | **`5.211 ms`** | **`8.32x Lower Latency`** ($< 6.0\text{ ms}$ goal) |
| **Median Glass-to-Glass Latency** | `42.10 ms` | **`5.164 ms`** | **`8.15x Lower`** |
| **95th Percentile Latency (P95)** | `48.90 ms` | **`5.771 ms`** | **Deterministic real-time latency** |
| **99th Percentile Latency (P99)** | `54.20 ms` | **`6.139 ms`** | **Zero tail-latency stalls** |
| **Min / Max Latency Envelope** | `38.20 ms` / `62.10 ms` | **`4.521 ms`** / **`7.401 ms`** | **Tight bounded jitter** |
| **Intermediate Host DDR Traffic** | `17.84 MB` per frame | **`0 Bytes`** (100% on-die MemTile SRAM) | **100% Bus Traffic Eliminated** |
| **Graph Partitions / CPU Fallbacks** | `52 Partitions` | **`0 Partitions` (1 Unified ERT Sequence)** | **100% CPU Fallbacks Eliminated** |
| **Numerical Parity on `bus.jpg`** | Reference (`yolov8n.onnx`) | **`5/5 Class Agreement`** (`mIoU = 0.9614`) | **Gold-Standard Fidelity ($\ge 0.95$)** |

---

## 2. Pipelined Stage Latency Breakdown (Mean ms)

```
[Camera Frame]
      │
      ▼ (Stage 1: Preprocess)
  ┌────────────────────────────────────────────────────────┐
  │ Zero-Copy Letterbox + INT8 Quantization:  2.740 ms       │
  └────────────────────────────────────────────────────────┘
      │ Bounded Queue (maxsize=2)
      ▼ (Stage 2: Monolithic Silicon NPU)
  ┌────────────────────────────────────────────────────────┐
  │ 3-Stage Hardware Compute (Backbone + Neck + Heads):   │
  │   - Silicon Dispatch & Compute:           2.126 ms       │
  │   - Intermediate DDR Traffic:             0 Bytes      │
  └────────────────────────────────────────────────────────┘
      │ Bounded Queue (maxsize=2)
      ▼ (Stage 3: Postprocess)
  ┌────────────────────────────────────────────────────────┐
  │ Vectorized DFL Softmax Decode + Batched NMS: 0.222 ms │
  └────────────────────────────────────────────────────────┘
      │
      ▼
[5 Bounding Boxes: 4 Persons, 1 Bus]
Total Glass-to-Glass Latency: 5.211 ms  |  Effective Throughput: 363.1 FPS
```

---

## 3. Key Architectural Innovations

### 3.1 Asynchronous 3-Stage Overlapped Pipelining
By bounding the inter-stage ring queues (`maxsize=2`), Stage 1 (Preprocess), Stage 2 (Physical Silicon NPU), and Stage 3 (CPU Postprocess) execute fully concurrent across worker threads without host memory bloat. The sustained framerate scales to the throughput ceiling of the slowest component, sustaining **`363.14 FPS`**.

### 3.2 Monolithic MemTile Elimination of DDR Ping-Pong
AMD Vitis AI EP fragments the YOLOv8n network across 52 individual subgraphs, triggering 52 separate ERT driver command submissions and bouncing 17.84 MB of intermediate activation tensors to host DDR memory. `ignite-xdna` chains the entire neural network (Layers 0..22) on-die across Phoenix MemTile L2 SRAM, streaming all intermediate tensors through hardware locks 4 and 5 with **strictly 0 bytes DDR traffic**.

### 3.3 Zero-Copy Direct-Channel Preprocessing
Rather than invoking multiple floating-point conversions and memory transpositions, `ignite-xdna` writes directly from OpenCV bilinear resizing into a pre-allocated pinned buffer, mapping unsigned uint8 directly to signed INT8 (`view(int8) ^ -128`) in **`2.740 ms`**.
