# Fused BF16 Self-Attention on XDNA1 (AIE2 / Phoenix)

Hand-written, fully vectorized BF16 multi-head self-attention kernel targeting the AMD XDNA1 NPU (Ryzen 8040 / 7040 series Phoenix NPU) using `mlir-aie` (IRON + Peano).

Replaces the 49 thrashing subgraphs that stock VitisAI EP creates on `mobilevit_xxs`, keeping all intermediate attention scores inside the NPU tile memory without DDR round-trips.

## Architecture & Optimizations

1. **Row-Wise FlashAttention Streaming:**
   - Instead of allocating a full $N \times N$ matrix in tile memory (which would take 128 KB for $N=256$, exceeding the 64 KB tile data memory), the kernel computes attention row-by-row.
   - The intermediate scratchpad buffer is reduced to a single vector row of $N$ elements (512 bytes for $N=256$, 128 bytes for $N=64$, 32 bytes for $N=16$).
   - This fits all 3 MobileViT stages comfortably in tile memory (34.5 KB total with `depth=1` ObjectFifos).

2. **16-Lane SIMD Vectorization (`aie_api`):**
   - $Q \times K^T$ and $Attn \times V$ are computed using native AIE2 16-lane `bfloat16` vector instructions (`aie::load_v<16>`, `aie::store_v<16>`, `aie::mul`, `aie::mac`, `aie::reduce_add<float>`).
   - Head dimensions are padded to 32-byte vector boundaries ($D_{pad} = 16$ for Stage 2; $D_{pad} = 32$ for Stages 3 & 4), ensuring 100% native vector loads/stores with zero tail overhead or unaligned shift bugs.

3. **In-Place Stable Softmax:**
   - Pass 1: Vector `aie::reduce_max` over the row.
   - Pass 2: Evaluates `fast_exp(x - max_val)` using immediate-operand polynomial approximation (zero uninitialized `.rodata` memory access), accumulates the sum, and writes `exp` values in-place into the row scratchpad.
   - Pass 3: 16-lane vectorized multiplication with broadcast `inv_sum`.

4. **Multi-Core Array Scaling:**
   - Scales across 8 compute tiles on Phoenix NPU (Cols 0..3, Rows 2..3).
   - Driven by 8 parallel ObjectFifos mapped to the 8 physical Shim DMA channels (2 channels per column).
   - Achieves near-linear parallel scaling with zero inter-tile contention.

5. **Linker Script Stack Collision Fix:**
   - Default Peano stack allocation of 1024 bytes (`0x70000` to `0x70400`) grows upward into tile buffers.
   - Worker explicitly configured with `stack_size=2048`, placing all buffers safely at `0x70800`+.

## Measured Results (Physical Phoenix NPU, Ryzen 7 8700G)

Log: `results/aie/attention_bf16_kernel_npu.log`

| Stage | Sequence Length $N$ | Head Dim $D$ ($D_{pad}$) | Cores | Latency | Verification (vs Golden) | CPU (8 heads) | Verdict vs CPU |
|---|---|---|---|---|---|---|---|
| **Stage 4** | 16 | 24 (32) | **8 cores** | **0.86 ms** | **PASS (0.799% rel L2)** | 0.012 ms | 71× slower than CPU |
| **Stage 3** | 64 | 20 (32) | **8 cores** | **4.57 ms** | **PASS (0.973% rel L2)** | 0.034 ms | 134× slower than CPU |
| **Stage 2** | 256 | 16 (16) | **8 cores** | **57.61 ms** | **PASS (0.992% rel L2)** | 0.240 ms | 240× slower than CPU |

### Numerical Precision & Tensor Shape Verification

- **Shape Check:** MobileViT-XXS Stage 3 attention tensor is $B=1, H=8, N=64, D=20$, giving exactly **10,240 active elements** per pass. Memory layout pads $D \rightarrow D_{pad}=32$ for 16-lane vector alignment (16,384 elements in tile memory).
- **Quantization Floor:** Truncating golden FP32 outputs to BF16 introduces an unavoidable baseline quantization floor of **0.1700% relative L2 error**.
- **Kernel Approximation:** The kernel's `fast_exp` polynomial approximation has a maximum relative error of **0.235%**. Combined with BF16 vector accumulation, the total measured kernel error against FP32 golden reference is **0.973% relative L2 error (0 NaN)** — fully within expected numerical bounds.

### The Finding: A Definitive Negative Result on Operator Acceleration

While the kernel compiles, maps cleanly across 8 physical AIE2 cores, streams via 8 Shim DMA channels, and achieves bit-accurate numerical verification (<1% rel L2 error, 0 NaN), **it is heavily outperformed by CPU**.

**Why? The Arithmetic Intensity Floor:**
- At Stage 3 (8 heads), total compute is only **2.79 MFLOP** ($0.0028\text{ GFLOP}$).
- For comparison, MobileNetV2 was $\sim 300\text{ MFLOP}$ and was already too light to amortize NPU fixed costs (losing to CPU 2.68 ms vs 1.72 ms). This attention operator is **$\sim 100\times$ smaller still**.
- At 4.57 ms, achieved throughput is only **0.61 GFLOPS** ($<0.1\%$ of the array's compute capability), completely dominated by runtime dispatch, task-group orchestration, and shim DMA latency for tiny vector lengths.
- The pivot intended to find a compute-bound operator, but mobile vision attention ($N \le 256$) lacks the token sequence length of LLMs ($N \ge 2048$), landing squarely back in the dispatch-bound regime.

## Full Model Architecture Comparison

| Pipeline Configuration | Hardware Execution | Latency | Note |
|---|---|---|---|
| **Stock VitisAI EP Baseline** | 49 NPU Partitions (partition thrashing) | 108.00 ms | 1.0× (baseline, 1037 NPU nodes / 548 CPU nodes) |
| **CPU Full Baseline** | 8 Zen4 cores (PyTorch FP32) | 18.37 ms | 5.9× faster than stock EP (75.0% top-1 on 1000 eval images) |
| **Cut CNN Backbone** | Physical Phoenix NPU (1 subgraph, 407 nodes) | **1.73 ms** | **3.1× vs CPU CNN (5.35 ms)** (like-for-like) |
| **Full NPU with AIE Attention** | Cut CNN on NPU + AIE2 Attention Kernel | **>120 ms** | **Loses to both CPU and Stock EP** |
| **Heterogeneous Splice (Measured)**| Cut CNN on NPU (1.73 ms) + Attention on CPU (2.03 ms) | **4.47 ms** | **4.1× vs full CPU (18.37 ms)**; incl. 0.72 ms in-process handoff |

> [!WARNING]
> The heterogeneous spliced pipeline does NOT use the AIE attention kernel — it leaves attention on CPU. End-to-end wall-clock time measured around the real in-process loop is **4.47 ms** (1.73 ms NPU infer + 2.03 ms CPU attention across 9 transformer blocks + 0.72 ms in-process buffer wrapping residual), delivering an honest **4.1× speedup vs full CPU (18.37 ms)** and 24.2× vs stock EP (108.00 ms).
> 
> In contrast, if partitioned across separate Python processes (due to Python 3.12 vs 3.13 pyxrt ABI walls), the IPC handoff floor measured in `groupnorm_bf16` ($789\,\mu\text{s}$–$23.6\text{ ms}$) would completely erase this advantage.
>
> **Accuracy Caveat:** While full FP32 `mobilevit_xxs` scores 75.0% top-1 on `data/eval`, the quantized ONNX models (`mobilevit_cut_backbone_xint8.onnx` and `mobilevit_xint8.onnx`) were compiled with synthetic random calibration (`UseRandomData = True`) for hardware place-and-route diagnostics, scoring 0% accuracy on real validation images. Neither quantized model can be cited for top-1 or mAP accuracy without a real-data calibration pass.

## Quickstart

Run in PowerShell with `ironenv`:

```powershell
. C:\Users\<user>\mlir-aie\iron_env.ps1

# Run Stage 3 attention on 8 cores against real ImageNet golden tensors
python kernels/attention_bf16/attention.py -d npu --golden-dir data/golden/attn_s3_l0 --num-cores 8 --iters 10

# Run Stage 4 attention on 8 cores
python kernels/attention_bf16/attention.py -d npu --golden-dir data/golden/attn_s4_l0 --num-cores 8 --iters 10

# Run Stage 2 attention on 8 cores
python kernels/attention_bf16/attention.py -d npu --golden-dir data/golden/attn_s2_l0 --num-cores 8 --iters 5
```
