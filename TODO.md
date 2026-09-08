# TODO: Ryzen AI XDNA1 Characterization & Acceleration Roadmap

This backlog organizes tasks by **immediate engineering dependencies and logical phases**, moving from low-hanging empirical characterization to foundational tools, hardware kernels, hybrid execution, and custom NPU-native models.

---

## 🎯 Phase 1: Immediate Measurements & Closing Open Research
*Tasks directly resolving open items in RESEARCH.md and completing the empirical benchmark matrix. Can be executed right now with existing tooling.*

### 1.1 In-Flight & Verification Sweeps
- [ ] **MODNet Alpha Calibration Retest**
  - **Issue:** Previous runs had a calibration/inference mismatch (PIL bilinear with antialiasing during calib vs OpenCV bilinear at infer). Consolidated into 
pu/modnet.py.
  - **Task:** Re-quantize modnet_cut and modnet_zero_concat under identical OpenCV preprocessing.
  - **Question:** Does MAD error drop below 0.19022 (Cut) / 0.35269 (Zero-Concat), and is Zero-Concat's degradation inherent or calibration drift?
- [ ] **AdaRound Across ResNet50 Input Resolution Sweep**
  - **Context:** The resolution sweep in docs/BENCHMARKS.md was plain XINT8, peaking at 256² (74.00% top-1).
  - **Task:** Run AdaRound on 224² and 288² base models (esnet50_r224_fp32.onnx and esnet50_r288_fp32.onnx).
  - **Question:** Does AdaRound recovery shift the optimal resolution/accuracy peak?
- [ ] **YOLOv8m / YOLOv8l / YOLOv8x Full Calibration Sweep**
  - **Context:** yolov8m (43.49 mAP) was calibrated on only 64 images due to disk/memory constraints; l and x used 32 and 24.
  - **Task:** Run ./scripts/yolo-bench.sh --variants m l x --calib 200 on Desktop 1 or Desktop 2 (needs >120 GB in %TEMP%). Evaluate full 5000-image COCO mAP.
- [ ] **AdaRound for YOLOv8-Pose**
  - **Task:** Run 3b_quantize_cut.py --adaround on yolov8n-pose and evaluate full 5000-image OKS mAP via ./scripts/pose-eval.sh.
- [ ] **Single-Session 4x4.xclbin Live Webcam Verification**
  - **Task:** Test ./scripts/yolo-demo.sh end-to-end with an active webcam and human inspection to confirm real-time bounding box visualization on a single partition.

### 1.2 Model Architecture Frontier (Category Pipelines)
- [ ] **Category C: YOLO-World v2 & YOLOv11**
  - Test whether YOLO-World v2 text-visual cross-attention or YOLOv11 C3k2/C2PSA (Pointwise Spatial Attention) blocks compile on the X1 backend or trigger CPU fallbacks.
- [ ] **Category D: FastDepth**
  - Lightweight MobileNet-NN depth estimator. Benchmark against MiDaS v2.1 Small (10.81 ms baseline).
- [ ] **Category E: RegNetX**
  - Test regular quantized conv channels against ResNeXt-50's narrow 4-channel groups to isolate PTQ scale collapse.

---

## 🏗️ Phase 2: Foundational Tooling & Infrastructure (Breaking Lock-in)
*Software engineering milestones that eliminate external vendor dependencies and make the repo bulletproof.*

### 2.1 Owned Quark-Free XINT8 Quantizer (quant/)
- [ ] **Phase 0 Audit:** Complete esults/quant/notes_xint8_dialect.log, resolving all [U] tags in quant/DESIGN.md (op registry, bias positions, CLE knobs, symmetric clipping [-127, 127]).
- [ ] **Phase 1 Core Arithmetic:** Finalize quant/graph.py, quant/pow2.py, quant/qdq.py, and quant/refine.py. Must pass syntax and import gates in both esnet_env and esnet_env17.
- [ ] **Phase 2 Calibration & Verification:** Run table generation on models/resnet50_fp32.onnx and verify bit-level parity against Quark 0.11rc1 output via quant/verify.py.
- [ ] **Phase 3 NPU Hardware Acceptance:** Build session on laptop/Desktop 2 with --fresh; verify diag_ep.py reports ~393/395 nodes accepted on NPU.

### 2.2 Pipeline & Cache Hygiene
- [ ] **Standardize MobileViT and MODNet Compile Cache Keys**
  - Consolidate all cache paths in 
pu/paths.py to enforce clean root-level directories (modnetcachekey, mobilevit_cut) and eliminate nested modelcachekey/mobilevit_cut duplicates.
- [ ] **Pre-Flight Hardware Contention Guard**
  - Add a pre-flight assertion in scripts/lib.sh and benchmark harnesses querying xrt-smi examine -r aie-partitions to abort immediately if foreign contexts are detected.

---

## ⚡ Phase 3: Hardware Kernels & Precision Exploration (kernels/)
*Bare-metal AIE tile programming bypassing the Vitis AI INT8 compiler via mlir-aie / IRON.*

### 3.1 Kernel Splicing & Pipeline Integration
- [ ] **AIE Kernel Splicing (kernels/dispatch_floor/):**
  - Connect the custom bf16 GroupNorm(32) kernel to an ONNX graph output buffer without bouncing through CPU host memory.

### 3.2 Unlocking Native Tile Datatypes (BF16 & INT16)
- [ ] **BF16 Scaled Dot-Product Attention (SDPA):**
  - Implement a tiled BF16 matrix multiply + online softmax kernel on AIE-ML tiles to run transformer attention heads natively without INT8 quantization collapse.
- [ ] **Mixed-Precision Pipeline (INT8 Conv Backbone + BF16 Head/Norm):**
  - Pipe intermediate INT8 activations directly into custom BF16 AIE kernels for normalization and regression heads (YOLO DFL decode, depth refinement).
- [ ] **A16W8 / A16W16 High-Dynamic-Range INT16 Kernels:**
  - Benchmark 16-bit x 16-bit vector GEMMs on AIE tiles to measure the exact TOPS and throughput penalty vs INT8 for audio DSP and depth estimation.
- [ ] **BF16 Continuous Activation Engine:**
  - Implement vector BF16 polynomial/lookup activation kernels (GELU, SiLU, Sigmoid) on AIE vector units, eliminating Quark's HardSwish/HardSigmoid structural modifications.

---

## 🔀 Phase 4: System Integration & Heterogeneous Hybrid Compute
*Leveraging the entire SoC (NPU + iGPU + CPU) over unified LPDDR5/DDR5 system memory.*

### 4.1 Hybrid Execution & Post-Processing
- [ ] **Asynchronous Pipelined Hand-off (NPU INT8 + iGPU FP16):**
  - Partition hybrid models (e.g. MobileViT, YOLOv11): dispatch conv backbone to NPU (INT8) and attention heads to Radeon 780M/760M iGPU via DirectML (FP16).
  - Double-buffer the pipeline so frame N-1 runs on iGPU while frame N runs on NPU.
- [ ] **Zero-Copy Shared Memory Ring Buffer (xrt::bo <-> Direct3D 12 / Vulkan):**
  - Use DirectX 12 shared handles or Vulkan external memory mapped to XRT host-visible pages to eliminate CPU intermediate copies.
- [ ] **Zero-Copy Post-Processing Engine:**
  - Write a native GPU compute shader or CPU AVX-512 SIMD worker to ingest NPU output buffers directly for YOLO DFL decode, NMS, and letterbox reversal.
- [ ] **NPU + CPU AVX-512 Cooperative Pipeline:**
  - Dedicate Zen 4 AVX-512 cores to camera decode, preprocessing, and NMS in parallel with active NPU matrix computation.

### 4.2 Comprehensive Efficiency Frontier (Energy & Battery Benchmarks)
- [ ] **Joules-per-Frame & Battery Drain Study (	ools/hwinfo_npu_bridge.cpp):**
  - Run continuous thermal steady-state benchmarks (15–30 min continuous webcam matting and YOLO streams).
  - Measure Package Power (Watts) vs. FPS and Joules-per-frame across:
    - NPU (VitisAIExecutionProvider via 4x4.xclbin)
    - iGPU (DmlExecutionProvider on Radeon 780M/760M)
    - CPU (CPUExecutionProvider on Zen 4)
  - Map the exact crossover point where NPU wins on sustained battery life vs iGPU burst performance.

---

## 🧬 Phase 5: NPU-Native Custom Architectures (ROCm on 7900 XTX)
*Designing, training, and distilling custom models on Desktop 1 tailored from epoch 0 to XDNA1 hardware constraints.*

- [ ] **1. XDNA-Net: Fixed-Point Native Micro-Backbone**
  - Train a 5M–15M parameter vision backbone from scratch with **Quantization-Aware Training (QAT)** on the 7900 XTX:
    - Native HardSwish / ReLU activations only (no compiler rewriting).
    - Power-of-2 clamping enforced in forward pass.
    - Channel dimensions padded to multiples of 16/32/64 to match AIE SIMD vector lanes.
    - RepVGG structural reparameterization.
  - **Target:** 100% NPU node placement with **0.0% accuracy loss under plain XINT8**.
- [ ] **2. XDNA-Matte: Distilled Native Portrait Matting**
  - Distill a clean single-branch portrait matting model on the 7900 XTX:
    - Eliminate all InstanceNorm / BatchNorm split-concat branches.
    - Use depthwise-separable convs and sub-pixel conv (DepthToSpace) upsampling.
  - **Target:** 60 FPS webcam background removal at <3 Watts package power.
- [ ] **3. NPU-Tailored License Plate & OCR Engine (LPRNet)**
  - Train a dedicated edge vehicle detector and plate recognizer on the 7900 XTX:
    - Avoid recurrent LSTM layers (which fail on NPU), using pure 1D/2D spatial convs with CPU CTC decode.
- [ ] **4. Ultra-Low-Latency Keyword Spotting / VAD**
  - Train a tiny 1D dilated CNN (MatchboxNet-style) on CommonVoice/LibriSpeech for always-on microphone gating in <1 ms on NPU.

---

## 💡 Phase 6: Unconventional & Non-Vision NPU Workloads
*Exploring the AIE-ML systolic array for non-traditional dataflow tasks.*

- [ ] **1. Real-Time Audio DSP & Noise Cancellation (1D Convolutions)**
  - Map 1D depthwise separable audio models (RNNoise / DTLN) to NPU tiles for sub-millisecond 10ms–20ms audio frame processing.
- [ ] **2. Always-On Sensor Telemetry & Anomaly Detection**
  - Evaluate streaming sensor telemetry via quantized Temporal Convolutional Networks (TCNs) without waking the host CPU from power-saving C-states.
- [ ] **3. Software-Defined Radio (SDR) & RF Signal Processing**
  - Ingest streaming I/Q samples via DMA to run matched FIR filters and deep modulation classification on the original telecom-derived AIE architecture.
- [ ] **4. Database Acceleration & Approximate Vector Search (Local RAG)**
  - Accelerate high-dimensional INT8 vector cosine/Euclidean distance calculations across local document embeddings via dense matrix-vector streaming ( \cdot x$).
- [ ] **5. 2D Grid Physics & Cellular Automata Simulation**
  - Frame discrete 2D grid updates (fluid diffusion, Conway's Game of Life, reaction-diffusion) as fixed-stencil convolutions executing on the 4x4 AIE tile grid.
