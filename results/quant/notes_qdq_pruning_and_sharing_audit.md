# Static Graph and Compiler-Rule Audit: Retained QDQ Pruning and Activation Scale-Sharing Across Vision Topologies

**Status**: Complete [SPEC & MEASURED & DERIVED]  
**Author**: Project Ignition  
**Date**: 2026-09-10  
**References**:
- [`quant/qdq.py`](../../quant/qdq.py) (dialect emission, operator marking, parameter sharing, pruning)
- [`quant/refine.py`](../../quant/refine.py) (Quark refinement transcription, Concat/Add/Mul alignment)
- [`quant/graph.py`](../../quant/graph.py) (topological ordering and vendor traversal order)
- [`quant/DESIGN.md`](../../quant/DESIGN.md) (Ignition architecture specification, Section 2)
- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#the-mul-is-not-the-fault-a-long-lived-activation-is-2026-09-09-desktop-2) (BiSeNetV2 provider divergence and bilateral aggregation analysis)
- [`results/quant/bga_tail_scan_desktop2_20260909.log`](bga_tail_scan_desktop2_20260909.log) (BiSeNetV2 linear cut scan)
- [`results/quant/mul_fixture_desktop2_20260909.log`](mul_fixture_desktop2_20260909.log) (isolated Mul fixture controls)

---

## 1. Executive Summary & Purpose

In Project Ignition, QuantizeLinear (Q) and DequantizeLinear (DQ) node placement transforms a floating-point ONNX graph into the AMD XDNA1 XINT8 dialect. For the VitisAI Execution Provider (VAI-EP) to successfully lower an ONNX graph to the XIR compiler target (`fuse_DPU` pass in `vaip_config.json`), QDQ pairs must strictly adhere to the DPU's systolic array execution model.

Two critical mechanisms govern this lowering:
1. **Retained QDQ Pruning (Activation Fusion)**: The DPU hardware natively evaluates activation functions (such as ReLU and bounded Clip) inside the accumulator writeback stage of convolution and arithmetic operations. If an intermediate QDQ pair is retained between a producer and a fused activation, the VAI-EP pattern matcher fails to recognize the fused operator, causing either catastrophic CPU fallback or redundant int8 writeback/requantization round-trips.
2. **Activation Scale-Sharing Invariants**: Memoryless operations (such as MaxPool and Resize) cannot rescale tensor data. They implement `QDQDirect8BitOp`, forcing the output tensor to share the scale and zero-point parameters of their input. This sharing is order-dependent: outputs inherit parameters only when their producer has already been marked during topological iteration (`Graph.vendor_order()`). Furthermore, multi-branch operations (Concat, Add, Mul) impose strict fixed-point exponent alignment rules (`align_concat`, `shift_read`, `shift_write`) that can truncate dynamic range.

This audit documents the static graph marking rules across six vision architectures (ResNet50, YOLOv8n-cut, YOLOv6n-cut, MODNet-Cut, FastDepth, and BiSeNetV2), analyzes the mathematical mechanics of DPU fusion, formalizes the scale-sharing topological dependencies, and cross-references multi-branch dynamic range truncation against the measured BiSeNetV2 bilateral aggregation failure (15.33% pixel accuracy on hardware vs 59.47% in CPU simulation).

---

## 2. Operator-Type Marking Survey Across Vision Topologies

### 2.1 Operator Placement Rules in `quant/qdq.py`

In the standard XINT8 dialect, activations are quantized to unsigned 8-bit integers (`uint8`, zero point 128) and weights to signed 8-bit integers (`int8`, zero point 0), both parameterized by scalar power-of-two scales `S = 2^(-pos)`. Operators are classified into four structural categories:

- **Annotate Operators (`ANNOTATE_OPS`)**: `Conv`, `Gemm`, `ConvTranspose`, `Add`, `MatMul`, `MaxPool`, `AveragePool`, `GlobalAveragePool`. When single-consumer outputs of these operators feed an annotated activation, the intermediate QDQ pair is pruned.
- **Direct 8-Bit Sharing Operators (`SHARING_OPS`)**: `MaxPool`, `Resize`. These operators mark input 0 and force the output tensor to share input 0's quantization parameters (`scale` and `zero_point`), computing no independent calibrated scale.
- **Parameter Operators (`PARAMETER_OPS`)**: `Clip`. The activation input (input 0) is quantized as `uint8`, while parameter inputs (min/max bounds at inputs 1 and 2) remain unquantized float initializers.
- **General Operators (`QDQOperatorBase`)**: `Add`, `Mul`, `Relu`, `Sigmoid`, `HardSigmoid`, `Concat`, `Slice`. Every non-constant float activation input and output receives an independent calibrated QDQ pair.

| Operator | Input Q/DQ Placement | Weight/Parameter Handling | Output Q/DQ Placement | Fusion Eligibility | Parameter Sharing Role |
|---|---|---|---|---|---|
| **Conv** | Required on input 0 (`uint8`, zp 128) | INT8 zp 0 weights; INT8/INT32 bias | Required (`uint8`, zp 128) | Yes: fuses following Relu/Clip | Producer (defines activation scale) |
| **Gemm** | Required on input 0 (`uint8`, zp 128) | INT8 zp 0 weights; INT8/INT32 bias | Required (`uint8`, zp 128) | Yes: fuses following Relu/Clip | Producer (defines activation scale) |
| **ConvTranspose** | Required on input 0 (`uint8`, zp 128) | INT8 zp 0 weights; INT8/INT32 bias | Required (`uint8`, zp 128) | Yes: fuses following Relu/Clip | Producer (defines activation scale) |
| **Add** | Required on inputs 0 and 1 | No weight parameters | Required (`uint8`, zp 128) | Yes: fuses following Relu/Clip | Producer; subject to `shift_read`/`shift_write` |
| **Mul** | Required on inputs 0 and 1 | No weight parameters | Required (`uint8`, zp 128) | No: activation evaluated separately | Producer; subject to `shift_write` |
| **MaxPool** | Required on input 0 | No parameters | Reuses input 0 parameters | Eligible as producer in `ANNOTATE_OPS` | Consumer: inherits input 0 scale/zp |
| **Resize** | Required on input 0 | ROI / scales / sizes as float initializers | Reuses input 0 parameters | No | Consumer: inherits input 0 scale/zp |
| **AveragePool** | Required on input 0 | No parameters | Required (independent scale) | Eligible as producer in `ANNOTATE_OPS` | Independent calibrated scale |
| **GlobalAveragePool** | Required on input 0 | Followed by DPU scale `Mul` | Required (independent scale) | Eligible as producer in `ANNOTATE_OPS` | Independent calibrated scale |
| **Relu** | Skipped when preceded by Conv/Add | No parameters | Required (`uint8`, zp 128) | Fused into preceding Conv/Add | Independent scale (or defines fused output) |
| **Clip** | Skipped when preceded by Conv/Add | Min and Max remain float initializers | Required (`uint8`, zp 128) | Fused into preceding Conv/Add | Parameter op: min/max float initializers |
| **HardSigmoid** | Required on input 0 | Internal alpha=1/6, beta=0.5 | Followed by scale `Mul` | Fused in DPU pass `merge_hard_sigmoid` | Independent scale |

### 2.2 Empirical Topology Survey Across 6 Architectures

The operator placement and QDQ counts across all six vision models in Project Ignition are measured from their compiled XINT8 graphs [MEASURED]:

| Architecture | Total Nodes | Quantize Nodes | Dequantize Nodes | Conv / Gemm | Relu / Clip | Add / Mul | MaxPool | AvgPool / GAP | Resize |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet50** | 380 | 74 | 182 | 54 (53/1) | 49 (49/0) | 17 (16/1) | 1 | 1 | 0 |
| **YOLOv8n-cut** | 957 | 218 | 344 | 63 (63/0) | 57 (0/0; 57 HS) | 120 (6/114) | 3 | 0 | 2 |
| **YOLOv6n-cut** | 510 | 99 | 241 | 71 (69/0 + 2 CT) | 54 (54/0) | 18 (0/18) | 3 | 0 | 0 |
| **MODNet-Cut** | 492 | 99 | 239 | 71 (71/0) | 52 (17/35) | 13 (10/3) | 0 | 1 | 8 |
| **FastDepth** | 254 | 47 | 123 | 38 (38/0) | 38 (11/27) | 3 (3/0) | 0 | 0 | 5 |
| **BiSeNetV2** | 396 | 79 | 191 | 57 (57/0) | 40 (40/0) | 16 (10/6) | 1 | 2 (1/1) | 3 |

*Notes on operator counts*:
- In ResNet50, YOLOv6n-cut, MODNet-Cut, FastDepth, and BiSeNetV2, all Relu and Clip nodes following Convolutions and Adds have their intermediate QDQ pairs completely pruned.
- In YOLOv8n-cut, activation functions are SiLU (`x · Sigmoid(x)`), which are expanded into `HardSigmoid(alpha=1/6, beta=0.5) → Mul(scale) → Q → DQ → Mul(x)`. Thus, YOLOv8n-cut contains 0 standard Relu/Clip nodes and 57 HardSigmoid nodes.
- In MODNet-Cut and FastDepth, MobileNetV2 inverted residual blocks use `Clip(0, 6)` (Relu6), which are identified by `needs_annotated` and pruned with zero unpruned activations remaining.

---

## 3. Pruning and Fusion Mechanics

### 3.1 The Graph-Rewriter Rules (`prunable_tensors` and `prune_conv_relu`)

In `quant/qdq.py`, activation pruning is executed in two stages: identification during calibration/emission, and structural removal post-insertion.

```
Pattern Matcher:
[Producer Node] (Conv, Add, Gemm, MaxPool, AvgPool)
      │
      ▼ (node.output[0])
[QuantizeLinear]
      │
      ▼
[DequantizeLinear]
      │
      ▼
[Activation Node] (Relu, LeakyRelu, PRelu, Clip[0,6], Clip[0,1])
      │
      ▼
[Downstream Consumers]
```

The matching rules enforce four strict structural invariants:
1. **Producer Op-Type**: The producer node must belong to `ANNOTATE_OPS`:
   `ANNOTATE_OPS = ("Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool", "MatMul", "Gemm", "ConvTranspose")`
2. **Single-Consumer Producer Output**: `len(g.consumers(node.output[0])) == 1`. If the convolution output branches to multiple destinations (for example, feeding both a skip connection and a ReLU), the output cannot be fused without duplicating or altering the branch.
3. **Activation Compatibility (`needs_annotated`)**:
   - `Relu`, `LeakyRelu`, `PRelu`: Always match.
   - `Clip`: Matches if and only if the min/max parameters satisfy `min == 0.0` and `max in (1.0, 6.0)`. Arbitrary clipping bounds do not map to the hardware ALU saturation modes.
4. **Single-Consumer Dequantize Output**: `len(g.consumers(dq.output[0])) == 1`. The activation must be the sole consumer of the dequantized tensor.

When these conditions are met, `prune_conv_relu` modifies the graph in-place:
1. The activation's input is rewired directly to the producer output (`followers[0].input[0] = node.output[0]`).
2. The intermediate `QuantizeLinear` and `DequantizeLinear` nodes are removed from the graph.
3. Their corresponding scale and zero-point initializers are deleted via `clean_initializers()`.

### 3.2 Pruning Statistics Across Architectures

The number of pruned intermediate QDQ pairs across the six models is measured as follows [MEASURED]:

| Architecture | Fused Conv → Relu/Clip | Fused Add → Relu/Clip | Total Pruned QDQ Pairs | Unpruned Residual Activations |
|---|---|---|---|---|
| **ResNet50** | 33 | 16 | 49 | 0 |
| **YOLOv8n-cut** | 0 (SiLU chain) | 0 | 0 | 0 |
| **YOLOv6n-cut** | 54 | 0 | 54 | 0 |
| **MODNet-Cut** | 52 (17 Relu / 35 Clip) | 0 | 52 | 0 |
| **FastDepth** | 38 (11 Relu / 27 Clip) | 0 | 38 | 0 |
| **BiSeNetV2** | 32 | 8 | 40 | 0 |

In ResNet50, 33 residual convolutions feed private ReLUs, while 16 bottleneck `Add` operations feed the block output ReLUs. Both are completely pruned, leaving zero unpruned activations.

### 3.3 Hardware Rationale and Unpruned Failure Modes

Why must intermediate QDQ pairs be pruned?

1. **Native DPU Systolic Array Writeback Fusion**:
   In Phoenix AIE2 cores, each vector MAC accumulator maintains 32-bit (or 48-bit) integer precision. The output pipeline contains an integrated Shift-Round-Saturate (SRS) unit and vector ALU:
   ```
   Accumulator (32-bit) ──► SRS Unit (shift_cut) ──► Activation ALU (ReLU / Clip) ──► Saturation to uint8 [0, 255]
   ```
   The ReLU or Clip operation is executed directly on the post-shift accumulator registers before writeback to local data memory. This hardware execution has **zero latency overhead** and **zero extra memory bandwidth**.

2. **Compiler Failure Mode (VitisAI EP `fuse_DPU` Pass)**:
   The VAI-EP compiler pass `fuse_DPU` pattern-matches subgraph templates in ONNX. It expects a single QDQ pair terminating a fused operator: `Conv → Relu → Q → DQ`.
   If an intermediate QDQ pair is retained (`Conv → Q → DQ → Relu → Q → DQ`), the compiler:
   - Fails to recognize `Conv` as an open fusion target for `Relu`.
   - Rejects the standalone `Relu` node on the DPU, forcing it to fall back to `CPUExecutionProvider`.
   - Triggers PCIe/AXI Shim DMA transfers between NPU tile memory and system RAM for every convolution output, collapsing inference throughput.

3. **Requantization Distortion and Precision Loss**:
   When intermediate QDQ is retained, the accumulator is shifted and rounded to an intermediate 8-bit grid (`pos_conv`), written to memory, reloaded, passed through a zero-clamping ALU, and shifted/rounded a second time to `pos_relu`. Each quantization stage introduces an independent rounding error of up to `±0.5 LSB`. Cascading two QDQ operations doubles the maximum quantization error bound without changing the mathematical operation.

---

## 4. Scale-Sharing Invariants and Visit-Order Dependencies

### 4.1 Memoryless Operators and `QDQDirect8BitOp`

Operators such as `MaxPool` and `Resize` do not perform arithmetic scaling, multiplication, or accumulation. They select, duplicate, or interpolate existing spatial samples:
- `MaxPool`: Evaluates `y[c, h, w] = max_(i, j in K) x[c, h+i, w+j]`.
- `Resize` (nearest / bilinear): Evaluates `y[c, h, w] = sum_(i, j) w_(i,j) x[c, h_i, w_j]`.

In the DPU fixed-point architecture, memoryless operators cannot modify tensor quantization parameters. They must operate with identical input and output representations:
```
Scale(output) = Scale(input)
ZeroPoint(output) = ZeroPoint(input)
pos(output) = pos(input)
```

In `quant/qdq.py`, this is implemented by assigning `sharing[node.output[0]] = node.input[0]`. When Q/DQ nodes are inserted, `insert_qdq` parameterizes the output QDQ using the input's scale and zero-point initializer names (`<input>_scale`, `<input>_zero_point`), eliminating duplicate initializers.

### 4.2 The Visit-Order Dependency (`Graph.vendor_order`)

A critical architectural invariant in Project Ignition is that **scale-sharing is visit-order dependent** [SPEC & DERIVED]:

In `quant/qdq.py::quantizable_tensors`:
```python
        elif node.op_type in SHARING_OPS:
            if node.input[0] not in acts:
                # is_tensor_quantized is false: skip marking, defer to consumer
                continue
            sharing[node.output[0]] = node.input[0]
            acts[node.output[0]] = None
            continue
```

When walking the graph, `quantizable_tensors` checks whether `node.input[0]` is already present in `acts`:
- **Producer Already Visited (`input[0] in acts`)**: If `input[0]` was emitted by an earlier node in topological order, it is already marked. `sharing[output] = input[0]` is recorded, and the output inherits the producer's scale.
- **Producer Not Yet Visited (`input[0] not in acts`)**: If `input[0]` has not been marked (for example, when `Resize` consumes a raw graph input), `is_tensor_quantized` is false. The quantizer **skips marking both input and output**. The output is only marked later when a downstream consumer (e.g. a Conv) visits it, causing the output to receive its own independent calibrated scale!

#### The MODNet Graph-Input Anomaly
This visit-order dependency explains a key property of MODNet:
- In MODNet-Cut, there are 8 `Resize` nodes.
- 6 `Resize` nodes follow internal convolutions or ReLUs (`/lr_branch/se_block/Mul`, `/lr_branch/conv_lr16x/.../Relu`, etc.). Because their producers are already marked, all 6 share their producers' scale parameters.
- 2 `Resize` nodes (`/hr_branch/Resize` and `/f_branch/Resize`) consume raw graph inputs directly. Because graph inputs are not produced by preceding nodes, they are not in `acts` when `Resize` is visited. They skip sharing, and downstream consumers assign them independent calibrated scales.

### 4.3 Transitive Sharing Chains in SPPF Topologies

When multiple memoryless operations are chained sequentially, parameter sharing must resolve transitively to the original root producer:

```
[Conv] ──► [Relu/Mul] (Root Scale S_0)
                 │
                 ├──► [MaxPool 1] (S_0)
                 │          │
                 │          ▼
                 ├──► [MaxPool 2] (S_0)
                 │          │
                 │          ▼
                 └──► [MaxPool 3] (S_0)
```

In YOLOv8n-cut (SPPF block `/model.9`) and YOLOv6n-cut (CSPSPPF block `/backbone/ERBlock_5/.../cspsppf`):
- Three consecutive 5x5 MaxPool nodes are cascaded.
- `MaxPool 1` shares with `cv1/act/Mul`.
- `MaxPool 2` consumes `MaxPool 1` output and records sharing with `MaxPool 1`.
- `MaxPool 3` consumes `MaxPool 2` output and records sharing with `MaxPool 2`.
- Transitive resolution in `quant/qdq.py`:
  ```python
  for name in list(sharing):
      root = sharing[name]
      while root in sharing:
          root = sharing[root]
      sharing[name] = root
  ```
  collapses `MaxPool 2` and `MaxPool 3` directly to `cv1/act/Mul`. All three MaxPool outputs and the root activation share a single scale initializer:
  - In YOLOv8n-cut: `/model.9/cv1/act/Mul_output_0_scale` is shared across 4 distinct tensors.
  - In YOLOv6n-cut: `/backbone/ERBlock_5/.../Relu_output_0_scale` is shared across 4 distinct tensors.

### 4.4 Measured Scale-Sharing Groups Across 6 Models

The distinct shared scale initializers and their consumer counts are measured across all six models [MEASURED]:

| Architecture | Total Distinct Scale Initializers | Shared Scale Initializers | Sharing Operators & Tensors |
|---|---|---|---|
| **ResNet50** | 181 | 1 | MaxPool: `/maxpool/MaxPool` shares `/act1/Relu_output_0_scale` |
| **YOLOv8n-cut** | 339 | 3 | MaxPool (3 nodes in SPPF share `/model.9/cv1/act/Mul` scale); Resize (2 nodes share preceding Mul scales) |
| **YOLOv6n-cut** | 238 | 1 | MaxPool (3 nodes in CSPSPPF share cv4 Relu scale) |
| **MODNet-Cut** | 233 | 6 | Resize (6 internal Resize nodes share preceding Mul/Relu scales) |
| **FastDepth** | 118 | 5 | Resize (5 decoder Resize nodes share preceding Relu scales) |
| **BiSeNetV2** | 187 | 4 | MaxPool (1 node shares Relu scale); Resize (3 nodes share Conv/Mul scales) |

---

## 5. Multi-Branch Alignment Rules and Dynamic Range Truncation

### 5.1 Fixed-Point Constraints in Multi-Branch Junctions

When multiple computational branches converge at an elementwise operation (`Concat`, `Add`, `Mul`), the DPU vector units impose rigid fixed-point alignment constraints transcribed in `quant/refine.py`:

1. **Concatenation Alignment (`align_concat`)**:
   In a Concat operation, channel tensors from disparate branches are packed into a single contiguous memory buffer. The DPU cannot apply per-channel scales across a unified spatial activation tensor. Therefore:
   ```
   pos_target = min(pos_output, pos_input_0, pos_input_1, ..., pos_input_k)
   For all i: pos_input_i := pos_target
   pos_output := pos_target
   ```
   Because scale is `S = 2^(-pos)`, a lower position corresponds to a **coarser quantization grid** with wider dynamic range (`S_coarse > S_fine`). The minimum position operator forces all converging branches to the coarsest scale among them.

2. **Addition Read Alignment (`shift_read`)**:
   The DPU vector ALU shifter for addition operands has a limited shift range of 7 bits:
   ```
   |pos_input_max - pos_input_min| <= 7
   ```
   If the scale difference between two addition inputs exceeds 7 powers of two (a dynamic range ratio > 128), `shift_read` clamps the finer input:
   ```
   pos_input_max := pos_input_min + 7
   ```

3. **Addition Write Alignment (`shift_write`)**:
   For `Add`, the writeback shift `sw = min(pos_inputs) - pos_output` must satisfy:
   ```
   sw in [-7, 25]
   ```

4. **Multiplication Write Alignment (`shift_write_mul`)**:
   For elementwise `Mul`, the product of two fixed-point integers produces an implicit scale position `pos_prod = pos_input_0 + pos_input_1`. The output writeback shift `sw = (pos_input_0 + pos_input_1) - pos_output` must satisfy:
   ```
   sw in [0, 32]
   ```

### 5.2 Dynamic Range Truncation Theorem

**Theorem (Branch Attenuation under Minimum-Position Alignment)** [DERIVED]:
Let branch A have dynamic range `R_A = max |x_A|` with calibrated scale `S_A = 2^(-pos_A)`, and branch B have dynamic range `R_B = max |x_B|` with calibrated scale `S_B = 2^(-pos_B)`. Assume branch A carries fine, low-amplitude features (`R_A << R_B`), such that `pos_A >> pos_B`.

When branch A and branch B converge at a Concat node (or an Add node requiring scale alignment):
1. The alignment rule forces `pos_A' = pos_B`.
2. The number of active quantization levels for branch A shrinks from 256 to:
   ```
   N_active = 256 * (S_A / S_B) = 256 * 2^(-(pos_A - pos_B))
   ```
3. For every power of two that `pos_A` is lowered, exactly one bit of precision is truncated from branch A's representation. If `pos_A - pos_B >= 8`, branch A is quantized entirely to zero (complete signal extinction).

---

## 6. Requantization Distortion Risk: The BiSeNetV2 Case Study

### 6.1 The Empirical Discrepancy

BiSeNetV2 presents the most severe provider divergence in the Project Ignition benchmark suite [MEASURED]:

| Target / Provider | Pixel Accuracy | Mean IoU (mIoU) | Softmax MAD | Softmax RMSE | Mean Latency |
|---|---|---|---|---|---|
| **CPU Simulation (ORT)** | 59.47% +/- 9.44% | 25.72% +/- 6.96% | 0.00357 | 0.00464 | 99.01 ms |
| **NPU (VitisAI EP)** | 15.33% +/- 10.82% | 2.44% +/- 1.28% | 0.01476 | 0.01924 | 13.01 ms |

On identical weights and identical QDQ parameters, the NPU output is largely decorrelated from the CPU reference (correlation 0.3455 at final logits; Softmax error 4.13x higher on NPU).

### 6.2 Subgraph Cut Localization: The Bilateral Aggregation Module (BGA)

BiSeNetV2 fuses a shallow Detail Path (high spatial resolution, 64x64) with a deep Semantic Path (low spatial resolution, 16x16) through the Bilateral Aggregation Module (BGA). In BGA:
- `/bga/Mul`: Detail branch features (`left1`, 64x64) multiplied by upsampled semantic gate (`up1`, 64x64).
- `/bga/Mul_1`: Downsampled detail features (`left2`, 16x16) multiplied by semantic gate (`right2`, 16x16).
- `/bga/Add`: Sums `/bga/Mul` output with upsampled `/bga/Mul_1` (`/bga/up2/Resize`).

Linear cut scanning across the graph ([`bga_tail_scan_desktop2_20260909.log`](bga_tail_scan_desktop2_20260909.log)) isolated the first catastrophic divergence to `/bga/Mul`:

| Cut Index | Node / Tensor | Correlation (NPU vs CPU) | Mean Abs Diff | Placed Nodes |
|---|---|---|---|---|
| 180 | `/bga/right2/right2.2/Conv` | 0.9913 | 0.041 | 285/287 |
| 181 | `/bga/Sigmoid_output_0` (gate) | 0.9858 | 0.019 | 288/290 |
| **182** | **`/bga/Mul_output_0`** | **0.6869** | **0.640** | **349/351** |
| 183 | `/bga/Sigmoid_1_output_0` (sibling gate) | 0.9904 | 0.007 | 289/291 |
| 184 | `/bga/Mul_1_output_0` (sibling product) | 0.9984 | 0.036 | 350/352 |
| 185 | `/bga/up2/Resize` | 0.9984 | 0.036 | 353/355 |
| 186 | `/bga/Add` (sum of branches) | 0.8516 | 0.646 | 382/384 |
| 190 | `logits` | 0.3455 | 0.273 | 402/404 |

Both inputs to `/bga/Mul` arrive highly correlated (>0.985), yet their product collapses to 0.6869. Meanwhile, sibling node `/bga/Mul_1` evaluates with 0.9984 correlation!

### 6.3 Fixed-Point Multiplier Arithmetic at `/bga/Mul`

Let us examine the exact quantization parameters of `/bga/Mul`:
- Input 0 (`/bga/left1/left1.2/Conv_output_0`): `pos_feat = 3` (`S_feat = 2^(-3) = 0.125`).
- Input 1 (`/bga/Sigmoid_output_0`): `pos_gate = 7` (`S_gate = 2^(-7) = 0.0078125`).
- Output (`/bga/Mul_output_0`): `pos_out = 4` (`S_out = 2^(-4) = 0.0625`).

Mathematical evaluation in fixed point [DERIVED]:
1. Real product: `y = x_feat * x_gate`.
2. Quantized inputs:
   ```
   q_feat = round(x_feat * 8) in [-128, 127]
   q_gate = round(x_gate * 128) in [0, 127] (since Sigmoid in [0, 1])
   ```
3. Integer accumulator product:
   ```
   Acc = q_feat * q_gate in [-16256, 16129]
   ```
4. Output scaling:
   ```
   pos_prod = pos_feat + pos_gate = 3 + 7 = 10
   sw = pos_prod - pos_out = 10 - 4 = 6 bits
   q_out = round(Acc * 2^(-6)) = round(Acc / 64)
   ```
5. Range of `q_out`:
   ```
   q_out in [-16256/64, 16129/64] = [-254, 252]
   ```
   Because `q_out` must be saturated to signed int8 `[-128, 127]`, any feature where `|q_feat * q_gate| > 8128` (e.g. `x_feat > 1.0` when `gate ≈ 1.0`) is severely clipped!

### 6.4 Operator Isolation vs. Long-Lived Memory Pressure

Does this fixed-point shift explain the 15.33% failure?

**Crucially, controlled experiments prove the multiplier operator is NOT at fault [MEASURED]**:
1. **Isolated Operator Control (`mul_fixture`)**:
   A minimal QDQ graph executing `Mul` at the exact shape (`[1, 128, 64, 64]`) and positions (`3 × 7 → 4`) was compiled and run on the NPU ([`mul_fixture_desktop2_20260909.log`](mul_fixture_desktop2_20260909.log)).
   - Measured Correlation: **1.0000**
   - Mean Absolute Difference: **0.00122**
   In isolation, the DPU evaluates this exact arithmetic perfectly!
2. **Bypassing the Semantic Branch (`mul_context`)**:
   When the gate input was supplied as a pre-computed graph input (cutting the ~280-node semantic branch), the exact same `/bga/Mul` compiled with:
   - Correlation: **0.9981** (against 0.6869 in full model)
   - Normalized Error: **0.029** (against 1.032 in full model)
3. **The True Physical Bottleneck: Long-Lived Activation Memory Pressure**:
   - The detail feature tensor `/bga/left1/left1.2/Conv_output_0` has shape `[1, 128, 64, 64]`. At 8-bit precision, this single activation buffer requires **512 KB** of memory.
   - This 512 KB activation must remain **live across ~280 nodes** while the entire semantic branch executes.
   - Sibling tensor `/bga/left2/left2.2/AveragePool_output_0` has shape `[1, 128, 16, 16]`, requiring only **32 KB** of memory.
   - **Phoenix AIE2 Hardware Constraint**: Each AIE tile contains exactly **64 KB of local data memory**. The 32 KB activation fits comfortably in a single local tile memory bank. The 512 KB activation cannot fit in on-chip tile memory and forces the compiler/allocator to spill across tiles or round-trip through external DDR memory. Under heavy multi-branch register pressure, memory corruption or buffer overwrites occur on the NPU, destroying the 512 KB activation while leaving the 32 KB activation pristine.

---

## 7. Master Graph-Rewriting and Scale-Sharing Reference Matrix

The following master reference matrix unifies all graph-rewriter rules, parameter-sharing behaviors, pruning requirements, and compiler constraints across Project Ignition [SPEC & MEASURED]:

| Node Type | Graph-Rewriter Action | Parameter Sharing Rule | Pruning Condition | DPU Compiler Constraint (`fuse_DPU`) |
|---|---|---|---|---|
| **Conv** | Insert QDQ on input 0; quantize weight/bias initializers | None (defines root activation scale) | Pruned if single-consumer feeds `Relu`/`Clip` | `shift_cut in [0, 16]`; `sigma in [14, 30]` |
| **Gemm** | Insert QDQ on input 0; quantize weight/bias initializers | None (defines root activation scale) | Pruned if single-consumer feeds `Relu`/`Clip` | Unit beta required (`beta=1.0`) |
| **ConvTranspose** | Insert QDQ on input 0; quantize weight/bias initializers | None (defines root activation scale) | Pruned if single-consumer feeds `Relu`/`Clip` | Valid spatial strides and kernel attributes |
| **Add** | Insert QDQ on inputs 0 and 1 | None (producer of new scale) | Pruned if single-consumer feeds `Relu`/`Clip` | `shift_read <= 7`; `shift_write in [-7, 25]` |
| **Mul** | Insert QDQ on inputs 0 and 1 | None (producer of new scale) | Never pruned (evaluated separately) | `shift_write_mul in [0, 32]` |
| **MaxPool** | Insert QDQ on input 0; output reuses input scale/zp | Inherits input 0 parameters (`QDQDirect8BitOp`) | Eligible as producer in `ANNOTATE_OPS` | Output scale must match input scale bitwise |
| **Resize** | Insert QDQ on input 0; output reuses input scale/zp | Inherits input 0 parameters if producer marked | Never pruned | Skips sharing if input is raw graph input |
| **AveragePool** | Insert QDQ on input 0 and output | Independent calibrated scale | Eligible as producer in `ANNOTATE_OPS` | Kernel area `<= 512`; scale multiplier |
| **GlobalAveragePool** | Insert QDQ on input 0 and output | Followed by dyadic DPU correction `Mul` | Eligible as producer in `ANNOTATE_OPS` | Kernel area `<= 512`; dyadic factor |
| **Relu** | Insert QDQ on input 0 and output | Inherits producer scale if fused | Input QDQ pruned if preceded by Conv/Add | Native ALU writeback saturation (zero latency) |
| **Clip** | Insert QDQ on input 0; min/max stay float initializers | `PARAMETER_OPS`: parameters unquantized | Input QDQ pruned if `min=0`, `max in (1, 6)` | Maps to native ALU clamp; unpruned falls back |
| **HardSigmoid** | Transformed to DPU `HardSigmoid → Mul` chain | Re-fused by DPU pass `merge_hard_sigmoid` | Never directly pruned | Clamped input `[0, 15]`, output `[7, 14+ipos]` |
| **Concat** | Insert QDQ on all inputs and output | Forced alignment: `pos = min(ipos_k, opos)` | Never pruned | Minimum-position lattice; dynamic range truncation |
| **Slice** | Insert QDQ on input 0 and output | Forced alignment: `pos_in = pos_out` | Never pruned | Coinciding scale constraint |
| **Pad** | Insert QDQ on input 0 and output | Forced alignment: `pos_in = pos_out` | Never pruned | Coinciding scale constraint |

---

## 8. Architectural Implications for Project Ignition

1. **Safety of Activation Pruning**:
   All 49 ReLUs in ResNet50, 54 ReLUs in YOLOv6n-cut, 52 activations in MODNet-Cut, 38 activations in FastDepth, and 40 activations in BiSeNetV2 are safely and correctly pruned by `quant/qdq.py`. In every case, this matches the hardware execution model of the Phoenix AIE2 systolic array, preventing catastrophic CPU fallback.
2. **Preservation of Root-Level Resizes**:
   The order-dependent check `if node.input[0] not in acts: continue` in `quant/qdq.py` correctly handles models where `Resize` operates on raw graph inputs (MODNet). Hand-rolling an order-independent sharing map would incorrectly force graph-input Resizes to share an uncalibrated dummy scale.
3. **Mitigating Multi-Branch Attenuation**:
   When designing multi-branch architectures for XDNA1 NPU deployment, long-lived activations (>64 KB) crossing hundreds of nodes must be strictly avoided. If a multi-branch network requires bilateral aggregation (like BiSeNetV2), inserting a subgraph split or forcing an intermediate host round-trip prevents local tile memory corruption, trading minor memory transfer overhead for complete numerical fidelity.
