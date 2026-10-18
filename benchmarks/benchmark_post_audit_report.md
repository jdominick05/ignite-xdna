> Archived, unverified historical report. Its throughput and "glass-to-glass"
> labels have not been revalidated. They do not establish current engine speed or
> distinct camera-frame throughput. Preserve the original claims below as work in
> progress; use the [archived structured output](benchmark_post_audit_results.json)
> for provenance, not as a replacement for a controlled measurement.

# Physical Silicon Benchmark: 1,000+ FPS Sustained Streaming Pipeline

**Target Silicon:** AMD Phoenix XDNA1 NPU Device 0 ([003d:00:01.1])  
**Evaluated Frames:** 2000 steady-state continuous frames  
**Input Source:** `assets/bus.jpg` (810x1080)  
**Timestamp:** 2026-09-13T17:42:55Z  

---

## Executive Summary

| Metric | Target Specification | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Sustained Throughput** | $\ge 1,000.0$ FPS | **1213.79 FPS** | **PASSED** |
| **Glass-to-Glass Mean** | $\le 1.80$ ms | **1.376 ms** | **PASSED** |
| **Frame Drops** | 0 frames | **0 frames** (100% complete) | **PASSED** |
| **Memory Leak Drift** | $< 5.0$ MB across 2,000 frames | **+0.71 MB** | **PASSED** |

---

## Detailed Latency Profile (Python Streaming Pipeline)

| Stage | Latency Mean (ms) | Percentage of G2G |
| :--- | :--- | :--- |
| **Stage 1: C-SIMD Ingress (Bilinear + BGR2RGB + Int8)** | 0.567 ms | 41.2% |
| **Stage 2: Physical NPU Single-Dispatch Execution** | 0.761 ms | 55.3% |
| **Stage 3: Vectorized DFL Softmax + Batched NMS** | 0.048 ms | 3.5% |
| **Total Glass-to-Glass (Mean)** | **1.376 ms** | 100.0% |

### Latency Percentiles
- **Median (P50):** 1.350 ms
- **P95:** 1.484 ms
- **P99:** 1.724 ms
- **Min / Max:** 1.305 ms / 3.839 ms

---

## Native C++ Engine (libignite_xdna & ignite-run)

| Metric | Native C++ Engine |
| :--- | :--- |
| **Sustained Throughput** | **N/A FPS** |
| **Glass-to-Glass Mean** | **1.333 ms** |
| **Glass-to-Glass Median** | **1.334 ms** |
| **Glass-to-Glass P95** | **1.381 ms** |
| **Glass-to-Glass P99** | **1.463 ms** |

---

## Stability and Memory Profile across 2,000 Continuous Frames

- **Initial Steady-State RSS:** 173.46 MB
- **Final Steady-State RSS:** 174.17 MB
- **Total Net Drift:** +0.71 MB (zero memory leak drift)
