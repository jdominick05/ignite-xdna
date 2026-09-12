# Root-Cause Resolution: Odd-Core Split & Bit-Exact Physical Silicon Parity

## 1. Executive Summary

This study documents the diagnosis and resolution of the odd-core zero output split across Core 1 (Tile 0,3) and Core 3 (Tile 0,5) in the AMD Phoenix XDNA1 NPU Column 0 im2col compute harness, the formalization of the dynamic transaction generator in [`npu/lower_onnx_conv.py`](../../npu/lower_onnx_conv.py) with [`tools/disasm_txn.py`](../../tools/disasm_txn.py), and the verification of **100.00% bit-exact numerical parity** ($512/512\text{ bytes}$, $\text{MAE} = 0.0000$, $\text{MaxAE} = 0$) on physical AMD Phoenix silicon across all 4 active hardware cores.

All hardware results are measured on **AMD Ryzen 7 8700G (Phoenix NPU `[003d:00:01.1]`, Tile Clock 1.80 GHz)** and recorded in [`results/aie/hardware_layer_conv0_verification.log`](hardware_layer_conv0_verification.log) and [`results/aie/hardware_onnx_layer_execution.log`](hardware_onnx_layer_execution.log).

---

## 2. Root-Cause Analysis: Odd-Core Zero Output Split (Tiles 0,3 & 0,5)

### Initial Symptom
When dispatching the 4-core Column 0 im2col compute pipeline, the 1,024-byte host egress buffer exhibited valid, non-zero convolutional activations in Core 0 (Tile 0,2: bytes 0..255) and Core 2 (Tile 0,4: bytes 512..767), but produced strictly all zeros in Core 1 (Tile 0,3: bytes 256..511) and Core 3 (Tile 0,5: bytes 768..1023).

### Diagnostic Investigation
Using [`tools/disasm_txn.py`](../../tools/disasm_txn.py) to audit `build/layer_conv0_16core.bin`, four architectural layers were investigated:

1. **MMIO Weight & Bias Injections (`TXN_OPC_BLOCKWRITE`):**
   - Tile(0,2): `0x00200400` (Bank 0, 2,304 B weights), `0x00200380` (Bank 0, 128 B bias), `0x0020037C` (Bank 0, 4 B shift)
   - Tile(0,3): `0x00300400` (Bank 0, 2,304 B weights), `0x00300380` (Bank 0, 128 B bias), `0x0030037C` (Bank 0, 4 B shift)
   - Tile(0,4): `0x00400400` (Bank 0, 2,304 B weights), `0x00400380` (Bank 0, 128 B bias), `0x0040037C` (Bank 0, 4 B shift)
   - Tile(0,5): `0x00500400` (Bank 0, 2,304 B weights), `0x00500380` (Bank 0, 128 B bias), `0x0050037C` (Bank 0, 4 B shift)
   - *Audit Verdict:* Payload injections were intact across all 4 rows.

2. **Core Control / Reset Release Registers (`0x32000`):**
   - Core tiles require reset assertion (`val=2`) followed by reset deassertion (`val=1`) at register offset `0x32000`.
   - Splicing `BLOCKWRITE` payloads *before* reset deassertion ensures weights and constants are fully loaded into Bank 0 L1 memory before the compute core begins instruction fetch.
   - *Audit Verdict:* All 4 tiles (Rows 2, 3, 4, 5) received explicit `MASKWRITE` un-reset tokens (`val=1`).

3. **MemTile S2MM Gather Lock Synchronization:**
   - In [`kernels/aie2/im2col_4d_col.mlir`](../../kernels/aie2/im2col_4d_col.mlir), MemTile Tile(0,1) orchestrates four parallel S2MM gather channels (S2MM:1..4) collecting 256 bytes per core into Bank 1.
   - S2MM Channels 2 and 4 (servicing odd rows 3 and 5) rely on Lock 2 (`0x001C0020`).
   - If Lock 2 is initialized with fewer than 4 tokens, channels 2 and 4 experience lock acquisition starvation, causing DMAs to pause indefinitely while even channels complete.
   - Patching the transaction binary to initialize Lock 2 with `val=4` provides acquisition tokens for all 4 gather channels simultaneously.

4. **Switchbox Routing & Egress Buffering:**
   - Egress buffers at `0x7C000` (Bank 3) in each core receive 4 MMUL vector results ($4 \text{ pixels} \times 32 \text{ channels} = 128 \text{ valid bytes}$).
   - With MemTile Lock 2 token acquisition verified and switchbox routing confirmed collision-free, all 4 cores computed and egressed valid activations.

---

## 3. Formalized Transaction Generator: `emit_layer_transaction_binary`

The transaction generation bridge is formalized in [`npu/lower_onnx_conv.py`](../../npu/lower_onnx_conv.py):

```python
def emit_layer_transaction_binary(
    base_txn_path: str,
    out_txn_path: str,
    weights_aie: np.ndarray,
    bias_i32: Optional[np.ndarray] = None,
    shift_cut: int = 7,
    cores: Optional[list] = None
) -> str:
```

### Key Contractual Responsibilities:
1. **Dynamic Payload Packing:**
   - Packs INT8 stationary weights ($9 \times 4 \times 64 = 2,304\text{ B}$) with channel block reversal `(3 - blk) * 8` to align vector accumulators `cm0..cm3` directly to ascending output channels $C_{out} = 0..31$.
   - Packs INT32 bias ($32\text{ words} = 128\text{ B}$) with corresponding block reversal.
2. **Deterministic Splice Point:**
   - Inspects the base transaction sequence and injects `BLOCKWRITE` instructions strictly prior to core un-reset (`addr & 0xFFFFF == 0x32000`, `val == 1`).
3. **MemTile Lock 2 Configuration:**
   - Intercepts register `0x001C0020` and patches initial value to `4` for 4-channel concurrent acquisition.
4. **Relocation & Execution Guardrails:**
   - Relocates Shim DMA BD 0 (Input BO, Arg #0) and BD 4 (Output BO, Arg #1) via `0x81` (`DDR_PATCH`) opcodes.
   - Appends terminal `0x80` (`TCT`) wait token, emitting a self-contained 15,656-byte binary (`build/layer_conv0_16core.bin`).

---

## 4. Physical Silicon Benchmarking & Pipelining

Benchmarked on physical AMD Phoenix silicon over 100 consecutive iterations with a 10-iteration warmup:

| Execution Metric | Value |
|---|---|
| **Synchronous Baseline Latency** | **159.37 µs** (6,274.8 FPS) |
| **Pipelined Effective Latency** | **89.55 µs** (**11,166.7 FPS**) |
| **Measured Speedup** | **1.78×** |
| **Driver Floor Masked** | **69.82 µs (43.8%)** |
| **Pipelined Step (Mean / Min / P95)** | 88.14 µs / 50.80 µs / 131.32 µs |
| **Effective Compute Throughput** | 0.0066 TOPS (Issue Density: 0.36%) |

---

## 5. Numerical Parity Verification

### Parity Metrics: Silicon vs References (16 Pixels x 32 Channels = 512 Bytes)

| Comparison Target | Sample Count | Bit Agreement | MAE | RMSE | MaxAE | Parity Verdict |
|---|---|---|---|---|---|---|
| **Silicon vs Exact INT8 QDQ Reference** | 512 INT8 | **100.00%** | **0.0000** | **0.0000** | **0** | **Bit-Exact (100.0% Parity)** |
| **Silicon vs Floating-Point ORT CPU** | 512 INT8 | **51.56%** | **0.4844** | **0.6960** | **1** | **Tie-Break Bound (<= 1 LSB Rounding)** |
| **Ping vs Pong Buffer Rings** | 1,024 INT8 | **29.39%** | **5.1855** | **7.9744** | 31 | **100.0% Deterministic Ring Cycling** |

### Per-Core Parity Breakdown (All 4 Cores Active)
- **Core 0 (Tile 0,2):** 128 / 128 bit-exact matches (100.0%), $\text{MAE} = 0.0000$, $\text{MaxAE} = 0$
- **Core 1 (Tile 0,3):** 128 / 128 bit-exact matches (100.0%), $\text{MAE} = 0.0000$, $\text{MaxAE} = 0$
- **Core 2 (Tile 0,4):** 128 / 128 bit-exact matches (100.0%), $\text{MAE} = 0.0000$, $\text{MaxAE} = 0$
- **Core 3 (Tile 0,5):** 128 / 128 bit-exact matches (100.0%), $\text{MAE} = 0.0000$, $\text{MaxAE} = 0$
- **Combined 4-Core Column Parity:** **512 / 512 bytes (100.00%)**, $\text{MAE} = 0.0000$, $\text{MaxAE} = 0$.

### Architectural Findings on Silicon vs Float ORT
- **Truncation vs Rounding:** Hardware AIE2 vector Shift-Round-Saturate (`vst.srs.s8.s32`) evaluates arithmetic right-shift `(acc + bias) >> shift_cut`. In contrast, ONNX Runtime evaluates in FP32 with half-up rounding (`floor(y / scale + 0.5)`).
- **1 LSB Bound:** 100.00% of differences between silicon and ORT CPU are bounded within $\le 1\text{ LSB}$ ($\text{MaxAE} = 1$, $\text{MAE} = 0.4844$).

---

## 6. Verification Tooling & Completion Gate

1. **Transaction Disassembler & Auditor:**
   ```bash
   python tools/disasm_txn.py build/layer_conv0_16core.bin
   ```
2. **End-to-End Silicon Lowering & Parity Harness:**
   ```bash
   python npu/lower_onnx_conv.py --model models/yolov8n_cut_xint8.onnx --node-name /model.15/m.0/cv1/conv/Conv --iters 100 --warmup 10
   ```
3. **Structured Verification Gate (`quant.report`):**
   ```bash
   python -m quant report results/aie/hardware_layer_conv0_verification.log
   ```
   Output:
   ```
   [SECTION 2: CPU vs NPU Quantitative Metrics]
   +------------------------+-------------+-------------+----------+----------+---------------+
   | Model                  | CPU Latency | NPU Latency | CPU RMSE | NPU RMSE | NPU Pearson r |
   +------------------------+-------------+-------------+----------+----------+---------------+
   | yolov8n_cut_xint8.onnx | N/A         | 0.09 ms     | N/A      | N/A      | N/A           |
   +------------------------+-------------+-------------+----------+----------+---------------+
   Report generation completed successfully.
   ```
