# llm_gemv: the whole-array M=1 bf16 GEMV

Step 1 of the small-LLM decode on XDNA1: SmolLM2-135M-Instruct, every neural op on the NPU. A decoder token is
mostly matrix-vector products whose weights cannot stay on chip (269 MB per token against 3.0 MB of reachable SRAM,
`docs/SILICON.md`), so the projection has to stream weights from DDR as fast as the array can read them. This
directory is that projection. Nothing else of the decoder is here yet.

- `pack.py`: the DDR layout. Weights are packed once at load into 8 channel streams of mmul<4,8,4> tiles. A call's
  x travels as the first object on every core. Outputs land in natural order.
- `design.py`: the array program. It uses 4 columns x 2 shim MM2S, the layout that read DDR at 47.62 GB/s in
  `tools/npu_read_bw_probe.py`. Each column splits through its MemTile to 4 cores and joins back to one shim S2MM.
  The core program is shape-free, reading counts from each call's header. So one xclbin serves every shape, and a
  program of calls is only a runtime sequence.
- `gemv.cc`: one core's share. fp32 accumulation over all of K, one round-to-nearest-even to bf16.
- `oracle.py`: the exact expected output under the emulator's MAC model (`engine_bf16_emulator.MAC_MODEL`). Device
  results must equal it bit for bit; there is no tolerance.

`tools/llm_gemv_probe.py` compiles the programs, computes the oracle on the real weights, and runs the sitting:
exactness on every one of a token's 121 GEMVs, then one token's projections issued three ways (one dispatch per
GEMV, per layer, per token). `tests/test_llm_gemv_offline.py` checks the layout against the oracle with no device.

Shapes: N a multiple of 192 (16 cores x 12 outputs) and K a multiple of 192 up to 2112. Every SmolLM2-135M
projection fits with no padding: qkv 960x576, o 576x576, gate+up 3072x576, down 576x1536, head 49152x576.
