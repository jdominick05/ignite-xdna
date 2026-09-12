# Split-Transaction Decoupling: Persistent Weights & 11,578 FPS Pipelined Execution

## Overview & Objective

In monolithic transaction stream architectures (`build/layer_conv0_16core.bin`, 62,528 bytes), every inference frame incurs massive MMIO blockwrite overhead:
1. **Stationary Weights, Biases & Shift Parameters:** 39,024 bytes (16 cores $\times$ [2,304 B weights + 128 B bias + 4 B shift + header words]) are re-written into Core L1 SRAM (`0x70400`, `0x70380`, `0x7037C`) via slow host MMIO writes every single frame.
2. **Static Interconnect Routing:** 208 register writes (3,328 bytes) re-program identical crossbar switchbox routes in MemTiles (row 1) and Core tiles (rows 2–5) despite switchbox configuration registers remaining static in hardware.
3. **Redundant BD Zeroing:** 4,608 bytes of register writes re-clear DMA Buffer Descriptors that do not change.

This document records the design, implementation, and physical Phoenix silicon validation of **Split-Transaction Decoupling**: separating the transaction stream into a **one-time parameter initialization binary** (`build/layer_conv0_init.bin`, 62,528 bytes) and a **lightweight per-frame execution binary** (`build/layer_conv0_exec.bin`, 10,496 bytes, with an ultra-compact 1,920-byte `< 2.5 KB` variant `build/layer_conv0_exec_minimal.bin`).

Silicon measurements on physical AMD Phoenix NPU (Ryzen 7 8700G, NPU `[003d:00:01.1]`, tile clock 1.80 GHz) demonstrate:
- **Pipelined Throughput:** Scaled from **5,054.7 FPS** (197.84 $\mu$s) to **11,578.1 FPS** (**86.37 $\mu$s**), delivering a **2.29× end-to-end speedup**.
- **Synchronous Latency:** Reduced from 308.98 $\mu$s down to **164.04 $\mu$s** (**1.88× speedup**).
- **Overhead Elimination:** **46,960 bytes eliminated** per frame (from 62,528 bytes down to 10,496 bytes).
- **Bit-Exact Numerical Parity:** **100.00% exact match** ($2,048/2,048$ bytes, $\text{MAE} = 0.0000$, $\text{RMSE} = 0.0000$, $\text{MaxAE} = 0$) against the exact INT8 QDQ reference on steady-state execution.
- **Ring-Buffer Determinism:** **100.00% bit-agreement** ($\text{MAE} = 0.0000$) between Ping and Pong asynchronous buffers across 500 consecutive iterations.

---

## Transaction Decoupling Architecture

The transaction stream lowering pipeline in [`npu/lower_onnx_conv.py`](../../npu/lower_onnx_conv.py) refactors monolithic emission into two specialized emitters:

### 1. One-Time Parameter Initialization Binary (`emit_layer_init_binary`)
- **Target Output:** `build/layer_conv0_init.bin` (62,528 bytes, 856 operations).
- **Dispatched:** Exactly once during session startup via a dedicated instruction BO.
- **Contents:**
  * **Core Tile Resets:** Tile un-reset sequences across rows 2–5 and columns 0–3.
  * **MemTile Credit Initialization:** Patches MemTile Lock 2 (`0x1C0020`) to `val = 4` across all 4 MemTiles ((0,1), (1,1), (2,1), (3,1)) to initialize S2MM gather credits.
  * **Static Switchbox Routing:** 208 register writes configuring MemTile and Core crossbars.
  * **Stationary Weights & Parameters:** Injects all 39,024 bytes of packed weights (`0x70400`), vector bias words (`0x70380`), and shift cut parameters (`0x7037C`) into Bank 0 L1 SRAM.
  * **Terminal Wait Token:** `TXN_OPC_TCT` (`0x80`) wait token.

### 2. Lightweight Per-Frame Execution Binary (`emit_layer_exec_binary`)
- **Target Output:** `build/layer_conv0_exec.bin` (10,496 bytes, 397 operations).
- **Dispatched:** Every frame inside the asynchronous double-buffered ERT ring pipeline.
- **Optimizations Applied:**
  * **Stationary Weights Stripped:** All 39,024 bytes of Bank 0 parameter blockwrites are omitted. The weights remain stationary in tile L1 SRAM across iterations.
  * **Switchbox Routing Stripped:** All 208 static switchbox writes (3,328 bytes) are omitted. Crossbars retain routing states.
  * **BD Zeroing Writes Stripped:** Redundant BD zeroing writes (4,608 bytes) are omitted.
  * **Credit Maintenance:** MemTile Lock 2 credit restore (`val = 4`) is retained to prevent S2MM channel starvation across multi-frame bursts.
  * **Shim DMA BD & Patch Records:** Retains Shim DMA BD 0/4 blockwrites and eight `0x81 TXN_OPC_DDR_PATCH` tokens for host-device address binding.
  * **Terminal Wait Token:** `TXN_OPC_TCT` (`0x80`).

### 3. Minimal Command Buffer Budget (`< 2.5 KB`)
- **Target Output:** `build/layer_conv0_exec_minimal.bin` (**1,920 bytes**, 57 operations).
- **Budget Compliance:** At 1.88 KB, this command buffer is well within the 2,560-byte (< 2.5 KB) hard constraint.
- **Composition:**
  * 16 Core Reset writes (`0x32000`, mask=3, val=2).
  * 4 MemTile Lock 2 credit restores (`0x1C0020`, val=4).
  * 4 Shim DMA BD configurations (`0x1D000`) with 8 DDR patches (`0x81`).
  * 12 Shim DMA channel queue pushes (`0x1D214` MM2S, `0x1D204` S2MM).
  * 16 Core Enable writes (`0x32000`, mask=3, val=1).
  * 1 Terminal TCT token (`0x80`).

---

## Hardware Execution Lifecycle & Priming

Investigation of AIE2 tile execution lifecycle under MLIR-AIE revealed critical hardware dynamics:
1. **Core Lifecycle at `aie.end`:** In the generated kernel (`im2col_4d_16core_src.mlir`), each core executes its processing loop and hits `aie.end`, transitioning the processor to a halted state.
2. **Core Reset/Enable Cycle:** To re-trigger compute on subsequent dispatches without wiping Bank 0 SRAM (`0x00400` weights, `0x00380` biases, `0x0037C` shift), the host asserts and deasserts reset via `0x32000`:
   - Assert Reset: `MASKWRITE (col, row, 0x32000) mask=3, val=2`
   - Deassert Reset + Core Enable: `MASKWRITE (col, row, 0x32000) mask=3, val=1`
3. **Pipeline Priming:** When starting an unprimed streaming pipeline:
   - **Dispatch 0 (Init):** Writes weights and interconnect. Hardware streaming buffers from host DDR $\to$ Shim $\to$ MemTile $\to$ Core L1 SRAM are empty; egress produces `bias >> 7` padding.
   - **Dispatch 1 (Prime):** Activations fill the multi-stage hardware streaming pipeline.
   - **Dispatch 2 (Steady State):** Validates 100.00% bit-exact parity across all 16 cores.
   - **Dispatch 3+ (Continuous Pipelining):** Sustains 100.00% deterministic parity continuously over 500+ iterations.

---

## Physical Silicon Benchmark Comparison

Measured on physical AMD Phoenix NPU (Ryzen 7 8700G, NPU `[003d:00:01.1]`, tile clock 1.80 GHz) on real vision model layer `/model.15/m.0/cv1/conv/Conv` (32 Cout $\times$ 32 Cin, 3x3 Conv):

| Architecture / Execution Mode | Txn Binary Size | Driver Overhead | Sync Latency (Mean) | Pipelined Latency (Mean) | Effective FPS | Speedup vs Monolithic |
|---|---|---|---|---|---|---|
| **Monolithic Baseline** | 62,528 B | 111.14 $\mu$s (36.0%) | 308.98 $\mu$s (3,236.5 FPS) | 197.84 $\mu$s | 5,054.7 FPS | 1.00× (Baseline) |
| **Split Txn (Persistent Weights)** | **10,496 B** | **77.67 $\mu$s (47.3%)** | **164.04 $\mu$s (6,095.9 FPS)** | **86.37 $\mu$s** | **11,578.1 FPS** | **2.29× Pipelined / 1.88× Sync** |
| **Minimal Command Buffer** | **1,920 B** | — | — | — | — | **< 2.5 KB Budget Met (1.88 KB)** |

### Latency Distribution across 500 Pipelined Iterations
- **Step Latency Mean:** 85.72 $\mu$s
- **Step Latency Min:** 40.40 $\mu$s
- **Step Latency P95:** 124.50 $\mu$s
- **Driver Overhead Hidden:** 77.67 $\mu$s (47.3% of synchronous runtime hidden by asynchronous double-buffered ring execution).

---

## Numerical Parity Verification

Evaluated across all 16 active cores ($64\text{ pixels} \times 32\text{ channels} = 2,048\text{ B}$ active output egress):

| Comparison Target | Bit-Agreement | MAE | RMSE | MaxAE | Parity Verdict |
|---|---|---|---|---|---|
| **Silicon vs Exact INT8 QDQ Reference** | **100.00%** | **0.0000** | **0.0000** | **0** | **Bit-Exact (100.0% Parity, 2048/2048 Bytes)** |
| **Silicon vs Floating-Point ORT CPU** | **51.56%** | **0.4844** | **0.6960** | **1** | **Tie-Break Bound ($\le 1\text{ LSB}$ Rounding, $\text{MAE} \le 0.50$)** |
| **Ping Ring vs Pong Ring Determinism** | **100.00%** | **0.0000** | **0.0000** | **0** | **100.0% Deterministic Ring Parity** |

Backing silicon verification logs:
- Split Persistent Weights Log: [`results/aie/hardware_16core_persistent_verification.log`](hardware_16core_persistent_verification.log)
- Monolithic Baseline Log: [`results/aie/hardware_16core_layer_verification.log`](hardware_16core_layer_verification.log)
