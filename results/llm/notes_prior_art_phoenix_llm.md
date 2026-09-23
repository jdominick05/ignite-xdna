# Prior art: LLMs on Phoenix (XDNA1), before this repo measures anything

**Date:** 2026-09-23 (Desktop 2, `DESKTOP-CBL5NUA`, Ryzen 7 8700G).
**Status:** sources only. Nothing in this note was run on this machine. It sets up the LLM study's
Phase 1 (CPU and DirectML yardsticks) and bounds what the NPU can win.

## How to read this

Every claim carries one of two tags:
- **verified**: read by this repo at the source and ref given, on 2026-09-23. It is quoted, or it is a
  file listing.
- **inferred**: a conclusion drawn from verified material. The tag says from what.

Numbers are tagged the way the rest of the repo does it:
- SPEC(source): a vendor document;
- DERIVED: arithmetic here;
- "published, not reproduced here": someone else's measurement.

GitHub refs were read through the GitHub API; the Ryzen AI documentation pages through
`ryzenai.docs.amd.com`.

## 1. AMD shipped an LLM path for Phoenix's NPU in 2024, in RyzenAI-SW 1.1 to 1.3.1

Source: [amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW), `example/transformers`, branches `1.1`,
`1.2.0` and `1.3.1`.
- The `1.1` release commit `46947467ff` is dated 2024-02-17.
- The last `example/transformers` commit on `1.3.1` is `557e527b68`, 2024-11-27.
- GitHub releases v1.2.0 and v1.3.1 were published 2025-03-25.

| Claim | Where | Tag |
|---|---|---|
| The 1.1 setup targets Phoenix by default: `set DEVICE=phx`, `set XLNX_VART_FIRMWARE=%PWD%/xclbin/phx`. | `1.1:example/transformers/setup.bat:23-24` | verified |
| 1.1 ships three GEMM overlays for PHX: `gemm_4x4_a16fw4acc32f.xclbin`, `gemm_4x4_a16fw3acc32f.xclbin` and `gemm_4x4_a8w8acc32.xclbin`, beside `1x4.xclbin`, `aieml_gemm_asr.xclbin` and `aieml_gemm_vm_phx_4x4.xclbin`. STX has the same three GEMM overlays plus `gemm_4x4_a16w8acc64.xclbin`. | `1.1:example/transformers/xclbin/{phx,stx}` | verified |
| The overlay name decodes from source. `gemm_4x4_` + activation (`a8` int8, `a16` int16, `a16f` bfloat16) + weight (`w8` int8, `w3` int4, `w4` uint4, `w16f` bfloat16) + accumulator (`acc32` int32, `acc64` int64, `acc32f` float32). So **`a16fw4acc32f` is bfloat16 activations × uint4 weights, float32 accumulation**. | `1.1:example/transformers/ops/cpp/qlinear_2/qlinear_2.hpp:276-337` | verified |
| That operator has a **dedicated M = 1 path**. `execute` picks `instr_bo_1_` and separate `a_bo_token_` / `c_bo_token_` buffers when `input_shape[0] == 1`. M ≤ 8 and M ≤ 32 get their own instruction streams. | `qlinear_2.hpp:837-857` | verified |
| The M = 1 instruction streams exist for Phoenix at exactly Llama-2-7B's shapes: `a16fw4acc32f_1_4k_4k.txt`, `_1_4k_11k.txt` and `_1_11k_4k.txt` (plus `_8_` and `_32_`). The int8 overlay has `a8w8acc32_{1,8,16,32}_{2k_2k,2k_8k,8k_2k}.txt`. The weight shapes select the kernel shape: 4096 × 4096, 4096 × 11008 and 11008 × 4096, with K = 11008 padded to 11264, and `KERNEL_M_MAX = 32`. | `1.1:example/transformers/dll/phx/qlinear_2/`; `qlinear_2.hpp:368-428` | verified |
| The host API takes each int4 weight and each int4 zero point in a full byte, with fp32 scales and a group size (default 32). The on-device layout is produced by `wgt_matrix.h` / `matrix_formatting.h`; this repo has not read them. | `qlinear_2.hpp:193-208` | verified (API); layout not read |
| The AIE kernel source is not in the tree: `example/transformers` ships xclbins and `.txt` instruction streams, and `dll/phx/README.md` reads "This is where dlls are generated". | `1.1:example/transformers/{xclbin,dll}` | verified (absence in this tree) |
| So the expansion of uint4 into a bf16 multiply on the AIE core cannot be read from this release. Whether the M = 1 stream is a true GEMV or a padded tile of the M = 8/32 GEMM is also unknown. | from the two rows above | inferred |
| Llama-2-7B: "Currently the supported precision on NPU is "w4abf16" and "w4abf16 + FA". **w4abf16** uses **AWQ** PerGrp quantization". Run with `run_awq.py --target aie --w_bit 4` (and `--w_bit 3`). | `1.1:example/transformers/models/llama2/README.MD:27,129-160` | verified |
| That README's perplexity table (wikitext2-raw): w4abf16 AWQ 3-bit, group 128: CPU 7.807, NPU 7.809; 4-bit, group 128: NPU 6.925. The NPU and CPU agree at 3-bit; there is no accuracy win on either side. The README's prose line "4-bit AWQ has higher perplexity than 3-bit AWQ" contradicts its own table; this note does not resolve it. | `README.MD:92,163-179` | verified; published, not reproduced here |
| OPT-1.3b, w4abf16 (`run_awq.py --target aie --task benchmark`): 117–120 ms/token, 8.3–8.5 tokens/s at 6–9-token prompts; 5.8–7.4 tokens/s at 512–2000-token prompts. | `1.1:example/transformers/models/opt/README.MD:123-143` | verified; published, not reproduced here |
| That OPT run was on Phoenix. | the README names no processor; 1.1's `setup.bat` defaults to `phx` | **inferred** |
| 1.3.1 lists 18 models for its PyTorch flow ("a just representative collection"): OPT 125m–13b, Llama-2-7B/13B (and -chat), Meta-Llama-3-8B-Instruct, StarCoder, CodeLlama-7B, Gemma-2B/7B and ChatGLM/ChatGLM3-6B. Llama-2-7B's "Quant Model Size" is 3.9 (unit not stated) and OPT-1.3b's is 0.8. The table does not say which models ran on which device. | `1.3.1:example/transformers/models/llm/docs/README.md:11-30` | verified |
| 1.3.1 says of that flow: "This flow is not optimized for performance and should not be used for benchmarking purposes." | same file, line 3 | verified |
| "`w8a16` only supported on STX". Setup is per device: `setup_phx.bat` / `setup_stx.bat` (and `.ps1`). | same file, lines 79-92, 113 | verified |
| 1.3.1's `xclbin/phx` adds `mladf_gemm_4x4_a8w8acc32.xclbin`, `bfloat16_256x2048_2048x2048.xclbin` (with a `_txn.bin`), `4x4.xclbin` and `int32_scalar.xclbin`, and drops the 3-bit overlay. So a bf16 GEMM overlay also shipped for Phoenix. | `1.3.1:example/transformers/xclbin/phx` | verified |
| 1.2.0 adds `models/llm_gguf` (a GGUF / llama.cpp path) next to `llm`, `llm_onnx` and `rag`. | `1.2.0:example/transformers/models` | verified (presence only; not read) |

**Consequence.** "No LLMs on XDNA1" was never a property of the silicon. AMD shipped a Phoenix
bf16 × uint4 GEMM overlay with an M = 1 token path at 7B shapes in 2024. This repo cannot claim a
first W4A16 or a first M = 1 GEMV on Phoenix. What stays open is published *bandwidth*: no source
read here gives a Phoenix GEMV's bytes per second or a 7B tokens/s.

## 2. In the Ryzen AI releases read (1.4, 1.7.1, 1.8.0), Phoenix gets no NPU LLM path

1.5, 1.6 and 1.7.0 were not read.

| Claim | Source | Tag |
|---|---|---|
| Release dates: v1.4.0 2025-04-01, v1.7.1 2026-03-27 (the inference SDK on this machine), v1.8.0 2026-07-23. | GitHub releases, `amd/RyzenAI-SW` | verified |
| 1.4: "Only Ryzen AI 300-series Strix Point (STX) and Krackan Point (KRK) processors support OGA-based hybrid execution." "The OGA-based NPU-only execution mode is supported on STX and KRK platforms." "Developers with Ryzen AI 7000- and 8000-series processors can get started using the CPU-based examples linked in the Featured LLMs table." | [ryzenai.docs.amd.com/en/1.4/llm/overview.html](https://ryzenai.docs.amd.com/en/1.4/llm/overview.html) | verified |
| 1.7.1 and 1.8.0 (identical table), "Supported Processor Configurations": Ryzen AI 300 (STX/KRK) ✓ NPU-Only, ✓ Hybrid, ✓ GPU/CPU; **Ryzen AI 7000/8000 ✗ NPU-Only, ✗ Hybrid, ✓ GPU/CPU**. | [en/1.7.1/llm/overview.html](https://ryzenai.docs.amd.com/en/1.7.1/llm/overview.html), [en/latest/llm/overview.html](https://ryzenai.docs.amd.com/en/latest/llm/overview.html) (reads "Ryzen AI Software 1.8.0") | verified |
| `example/transformers` is gone from 1.4.0; 1.7.1 and 1.8.0 have no `example/` directory at all. | `amd/RyzenAI-SW` refs `1.4.0`, `1.7.1`, `1.8.0` | verified |
| The 8700G is a Ryzen 8000-series part, so the current release's LLM flows give it GPU and CPU only. | the table above | inferred |

So README:22 and docs/SETUP.md:22 ("no LLMs") hold for the NPU under the current release, 1.8.0
(and 1.7.1, used here). They do not hold for the chip's history, and not for the GPU/CPU LLM flows
the same table lists.

## 3. Open toolchains that could express an M = 1 GEMV on Phoenix

| Claim | Source | Tag |
|---|---|---|
| mlir-aie `matrix_vector`: "A single AI Engine compute core computes `c = A @ b` … Default config: `int16` inputs / `int32` outputs, `M`=`K`=`288`", and "This is a single-core design; multi-core extension is left for a future revision." It takes the device from `iron.get_current_device()`. | `Xilinx/mlir-aie@84d5df981f` (2026-09-23), `programming_examples/basic/matrix_multiplication/matrix_vector/{README.md:10,17, matrix_vector.py:127}` | verified; not run here |
| amd/IRON has `iron/operators/gemv` (bf16 in, bf16 out) and `iron/operators/dequant` (packed int4 plus one bf16 scale per group, default group 32, out bf16). | `amd/IRON@7b8fba7b57` (default branch `devel`, 2026-09-18), `iron/operators/{gemv,dequant}/{design,op}.py` | verified |
| Both compile mlir-aie's **generic** kernels: `generic/mv.cc` and `generic/expand.cc`. IRON's `get_kernel_dir` returns "'aie2' for NPU1 (Phoenix)". The only NPU2-only gate is the GEMV's fused GELU epilogue ("only available on NPU2 (aie2p)"). mlir-aie also has `aie_kernels/aie2/mv.cc` and `aie_kernels/generic/q4nx_dequant.cc`. | `amd/IRON@7b8fba7b57:iron/common/device_utils.py:9`, `gemv/op.py:124-141`, `dequant/op.py:68-70`; `Xilinx/mlir-aie@84d5df981f:aie_kernels/` | verified |
| So IRON's bf16 GEMV and its int4 → bf16 expansion have an aie2 (Phoenix) path in code. Neither has been compiled or run here. `generic/expand.cc` and `generic/q4nx_dequant.cc` are the reference int4 → bf16 expansion for the study's compile-only ISA gate (Phase 2a), rather than something to write from scratch. | from the row above | inferred |
| FastFlowLM: "FastFlowLM (FLM) supports all Ryzen™ AI Series chips with XDNA2 NPUs (Strix, Strix Halo, Kraken, and Gorgon Point)." The repository is now `ROCm/FastFlowLM`. | `ROCm/FastFlowLM:README.md:20` | verified |

## 4. The bound that decides NPU-only decode (DERIVED)

Llama-2-7B has 6,476,005,376 linear-layer weights (32 layers × (4 × 4096² + 3 × 4096 × 11008)). Its
`lm_head` has 131,072,000 more; the embedding is a one-row lookup and is not counted. At 4 bits the
weights read per token are:

| Storage | GB per token | at 26.8 GB/s | tokens/s ceiling |
|---|---:|---:|---:|
| group 128, bf16 scales, no zeros, `lm_head` int4 | 3.407 | 127.1 ms | 7.87 |
| group 128, bf16 scales + int4 zeros, `lm_head` bf16 | 3.627 | 135.3 ms | 7.39 |
| group 32, fp16 scales + int4 zeros, `lm_head` bf16 | 4.006 | 149.5 ms | 6.69 |

- 26.8 GB/s is the MEASURED fill transport of this repo's engine: 78.7 MB in 2.94 ms, no compute
  ([BENCHMARKS, graph engine latency](../../docs/BENCHMARKS.md#graph-engine-latency-from-192-to-79-ms-glass-to-glass-2026-09-14-desktop-2),
  lines 8343-8344). It is not a measured NPU read ceiling.
- [SILICON 1.6](../../docs/SILICON.md#16-off-chip-bandwidth) has 25.9 GB/s of reads beside concurrent
  writes, and 28.1 GB/s per direction in a round trip, and says none of its figures is "a clean
  single-direction read test". Its 26–28 GB/s shared cap is DERIVED (SILICON.md:136-140). At the best
  rate seen, 28.1 GB/s, the smallest row is 121.2 ms, 8.25 tokens/s.
- KV-cache reads add bytes and only lower the ceiling.
- 1.3.1's "Quant Model Size" of 3.9 for Llama-2-7B, if in GB, sits inside this range.
- **So at the DRAM rates this repo has measured for the NPU (26.8 GB/s fill, 28.1 best), and at
  SILICON 1.6's derived 26–28 GB/s shared cap, NPU-only 7B decode stays under about 8 tokens/s**
  (DERIVED). The read-only ceiling is unmeasured, so this is not yet a hard cap.
- This machine runs DDR5-6000 on two channels: two 16 GiB DIMMs on channels A and B, JEDEC speed
  4800, configured 6000 (Win32_PhysicalMemory, 2026-09-23). That is 96 GB/s theoretical, DERIVED from
  the configured speed. It is above AMD's rated maximum for the 8700G, "2x1R DDR5-5200" / "2x2R
  DDR5-5200" (SPEC([AMD Ryzen 7 8700G product page](https://www.amd.com/en/products/processors/desktops/ryzen/8000-series/amd-ryzen-7-8700g.html)),
  read 2026-09-23), which would be 83.2 GB/s.
- If the CPU or the 780M reads this DDR5 faster than the NPU in its own int4 GEMV, NPU-only decode
  cannot win on speed. Neither has been measured on this machine; that is the study's Phase 1
  (`tools/cpu_mem_bw.py` measures the CPU's read rate).

For scale, the published OPT-1.3b figure above implies at most 0.8 GB / 0.117–0.120 s =
6.67–6.84 GB/s of weight traffic (DERIVED). That assumes every byte of the 0.8 quantized model is
read per token, and mixes a 1.3.1 size with a 1.1 timing. The flow that produced it is the one 1.3.1
calls "not optimized for performance".

## What this note does not settle

- How AMD's closed Phoenix kernel turns uint4 into a bf16 multiply, and whether its M = 1 stream is a
  GEMV. The `.txt` instruction streams are in the tree and could be disassembled; this note did not.
- Whether IRON's `gemv` / `dequant` compile and run on npu1.
- What any of the three chips on this APU reaches in an int4 GEMV. That is Phase 1 (CPU, DirectML)
  and Phase 3 (NPU).
