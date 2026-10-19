# GPU GGUF token embeddings

`embedding.cpp` loads only `token_embd.weight` from a GGUF into a ROCm0 device
buffer. Public GGML APIs run `GET_ROWS` and a broadcast `MUL`; there is no CPU
dequantization or arithmetic fallback and no target-model evaluation. The caller
provides the row scale explicitly. Using `MUL` matches the pinned llama.cpp
embedding path, including negative zero; `SCALE` adds a zero bias and differs.

From a Visual Studio developer shell, with the same GGML headers as the runtime:

```powershell
$ggmlInclude = 'L:\llm-work\src\llama.cpp\ggml\include'
cl /nologo /std:c++17 /EHsc /O2 /W4 /I $ggmlInclude tools/ggml_gpu_embedding/embedding.cpp /Foscratch/ggml-gpu-embedding.obj /Fescratch/ggml-gpu-embedding.exe
```

The executable resolves GGML exports from `ggml-base.dll` and `ggml.dll` and
loads the runtime's backend modules. It needs no import library. Usage:

```text
ggml_gpu_embedding BIN MODEL TOKENS_I32 NEW_OUTPUT_DIR SCALE
```

The token file contains 1-256 little-endian int32 IDs. Output is contiguous FP32
rows in `embeddings.f32` plus `lookup.json`. An existing output directory is
rejected. The CLI also checks single-token A/B/A lookups after its batch, using
the same backend and resident weight objects. It compares raw bits, including
signed zeros. GPU support and device buffers are checked for both graph nodes.

Define `IGNITE_GGML_EMBEDDING_LIBRARY` before including the source to omit its
CLI and reuse `Runtime`, `GpuEmbedding` and `lookup(tokens, scale)`. Keep one
runtime/table alive across serial lookups; libraries belong to the backend
registry's process lifetime. The current implementation makes its own device
copy of the embedding table: 330,301,440 bytes for the qualified E2B GGUF. It
does not share the target model's existing buffer or apply adapters, rotated
embedding transforms or multimodal preprocessing.

Before GPU use, run `tools/check-inference-contention.ps1`; require idle for
performance measurements. The [qualification record](../../docs/BENCHMARKS.md#gpu-token-embedding-lookup-and-npu-handoff-2026-10-18)
covers five public E2B trajectory tokens, 7,680 bitwise-equal FP32 values,
persistent lookup replay and the NPU handoff. This is a reusable research
component, not a live assistant or a throughput result.
