# Fused Conv + Residual Add + SiLU on Phoenix

Standalone AIE2/Peano kernel and routed transport. The performance qualification
is **Cin=512, Cout=32, 3x3, eight spatial outputs per core**. The Cin=32 comparison
passes numerical checks but misses the overhead target. See the
[measurements and limitations](../../../docs/BENCHMARKS.md#fused-conv-residual-silu-2026-09-13-desktop-2).
This does not register an ONNX Runtime operator or change model placement.

Run from the repository root in Git Bash:

```bash
./scripts/fused-epilogue.sh --compile --cin 512
./scripts/fused-epilogue.sh --hardware --cin 512 --iters 10
./scripts/fused-epilogue.sh --offline --cin 512
```

Alternatively, activate the existing mlir-aie ironenv and use
`make -f kernels/aie2/fused_conv_epilogue/Makefile all hardware CIN=512`.
The hardware command fails on numerical mismatch, overhead >=8%, missing DMA
overlap, memory-stall events in fused compute, changed artifacts, or a busy device.
`--cin 32` is an explicitly failing performance comparison, not another qualified shape.

## Numerical contract

Batch one, NHWC, valid stride-one 3x3 Conv, dilation one, groups one, zero points
zero, INT8 inputs/weights/residual, INT32 bias and accumulation. Scales are
input=weight=residual=output=1/16; bias and Conv accumulator=1/256. Bias plus the
complete reduction plus residual aligned by a left shift of four must fit acc32.

Eight `aie::mmul<4,8,8,...,acc32>` objects cover both four-pixel groups and all
32 output channels. Residual addition occurs in acc32 before any INT8 clamp.
For `t=min(abs(x),8)`, SiLU is approximated as
`max(x,0) - (t/2)*(1-t/8)^5`, using Q8 values and Q11 polynomial factors.
All shifts use ties-to-even rounding. Final Q8-to-Q4 requantization uses multiplier
one and saturating INT8 SRS stores with bounds [-128,127]. The intermediate INT16
clamp is safe for these scales because its tails already saturate the final output.
The test checks every representable Q8 input against an independent float32 SiLU;
the acceptance tolerance is one output LSB. Hardware must also match the integer
polynomial oracle exactly.

## Packet and lock ABI

For Cin=512, each core consumes 64 panels of 3264 bytes. Panel order is increasing
eight-channel block; each panel contains all nine taps:

| Byte offset | Contents |
|---|---|
| 0 | INT8 A[patch group=2][tap=9][pixel=4][ci=8], 576 bytes |
| 576 | INT8 W[tap=9][Cout block=4][ci=8][co=8], 2304 bytes |
| 2880 | INT32 bias[32], 128 bytes |
| 3008 | INT8 residual[group=2][Cout block=4][pixel=4][co=8], 256 bytes |

Only the first panel's metadata is consumed; subsequent metadata is padding.
The Shim input order is [column][panel][row][packet byte]. Output order is
[column][row][group][Cout block][pixel][co]. Each core writes all 256 bytes.
The resident Cin=32 layout is documented in `conv2d_fused.cc` and packed separately.

Streamed core-local buffers occupy four different 16-KiB banks: ping at 0x400,
activation/metadata staging at 0x4000, pong at 0x8000, output at 0xc000.
Input-ready local lock 3 is core selector 51; input-free lock 2 is selector 50.
The compiler wrapper owns output-free lock 0 (selector 48). C++ releases
output-ready lock 1 (selector 49) after each pair of 32-byte stores.
Four output BDs acquire one credit each, reading consecutive 64-byte chunks.
Only the final BD returns the output-free credit. Never substitute local lock IDs
directly for core selectors. `transport.py` checks the lowering before modifying it.

## Build and evidence

Peano compiles for `aie2-none-unknown-elf` at `-O3` with its default VLIW scheduler.
The build writes explicit `raw16`, `fused16`, `raw1`, and `fused1` artifacts under
`build/fused_conv_epilogue/<source-and-Cin-hash>/`. Each has final core ELFs,
disassembly, xclbin, `insts.bin`, and a generated `transaction.h` with byte count.
The manifest binds source, transport, compiler identity and binary hashes.
`_bootstrap/` is a compiler placement template with an unfinished lock protocol;
**never execute its xclbin or instructions**. Only the final artifacts are audited
and accepted by the runner.

The audit checks every final ELF, lock selectors, chunk publication after stores,
buffer banks and Shim argument mapping. It rejects vector spills and stack
accesses inside compute. Peano's scalar pointer save/restore at function entry/exit
is recorded separately. Each variant's one-core and 16-core builds must have
identical kernel function bytes.
Cycle/stall/DMA evidence comes from the one-core trace build; all-core numerical
evidence comes from independent data on the 16-core build. This separation is
necessary because the full array leaves no legal route for the extra trace stream.

Each traced sample uses a fresh hardware context. The measured sum covers
accumulator initialization, all MAC panels and the epilogue; input transfer,
staging, function entry/exit and trace flush are excluded. The runner also records
the whole first-to-last event span, retaining input waits in that separate value.
Logs are tracked under `results/aie/`; full tensor and trace witnesses stay in the
ignored build directory with their hashes recorded in the log.
