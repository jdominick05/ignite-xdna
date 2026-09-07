# kernels/

Hand-written AIE kernels for this machine's XDNA1 NPU, built with the open-source
`Xilinx/mlir-aie` (IRON + Peano) toolchain rather than Quark/VitisAI EP. They exist to
reach data types the standard pipeline structurally cannot (bf16, int16 -- the X1
backend is XINT8 or nothing, see `docs/DECISIONS.md`), for the specific ops VitisAI EP
leaves on the CPU. Nothing here changes how `resnet50`/`yolov8n`/`yolov8n-pose` are
quantized or run; each kernel is a standalone, measured artifact with its own log
under `results/aie/`.

These do **not** run in `resnet_env17`. They need the mlir-aie `ironenv` described in
`RESEARCH.md` ("Custom C++ XRT / hand-written AIE kernels"), activated from
PowerShell:

```powershell
. C:\Users\<user>\mlir-aie\iron_env.ps1
python kernels/groupnorm_bf16/groupnorm.py -d npu --L 301056
```

`aiecc`'s final link needs `xclbinutil`, which ships with the XRT SDK and is **not** in
`ironenv/Scripts`. Put that directory on PATH as well, or a build that has already done
all its real work dies at step 36/37 with `tool 'xclbinutil' not found`:

```powershell
$env:PATH = "C:\Xilinx\XRT\xrt_sdk\xrt;$env:PATH"
```

The first run of a given shape JIT-compiles (Peano for the C++ kernel, aiecc for the
array program) into `~/.npu/cache/<hash>/`; later runs of the same shape hit the cache.

| Kernel | Op it replaces | Status |
|---|---|---|
| `groupnorm_bf16/` | `InstanceNormalization` (really `GroupNorm(32)`) in `resnetv2_50x3_xint8.onnx`, the op that falls to CPU on that model | Kernel alone beats CPU on 33/49 nodes (`results/aie/groupnorm_bf16_kernel_npu.log`), but the measured two-process handoff floor erases the win at every shape -- 0/49 once spliced (`results/aie/groupnorm_bf16_handoff_floor_npu.log`) |
| `attention_bf16/` | Multi-Head Attention (`MatMul` + `Softmax` + `MatMul`) in `mobilevit_xxs`, replacing 58 partition-thrashing EP subgraphs | Numerically correct (<1% rel L2, 0 NaN) on 8 AIE2 cores across all 3 MobileViT stages (Stage 2: 57.6 ms, Stage 3: 4.57 ms, Stage 4: 0.86 ms at 8 heads) and a **negative result**: it loses to Zen4 CPU by 71×–240×. **Its recorded diagnosis was wrong** — the loss was blamed on per-dispatch cost, which is ~1% of Stage 2's time now that the floor is measured. The real cause is that `attention_kernels.cc` uses `aie::mmul` **zero** times, hand-rolling dot products with a horizontal `reduce_add` per output element for 0.61 GFLOPS on hardware measured at 895. **Do not reuse its inner loop as a template**; see its own README (`results/aie/attention_bf16_kernel_npu.log`). |
| `dispatch_floor/` | Not an operator — measures the **per-dispatch fixed cost itself**, the constant every isolated-op verdict in this repo rests on | No compute tile at all (shim→memtile→shim), payloads swept 8 KB–32 MB, output verified per payload. **Hardware floor 169.8 µs** — confirming the "~185–200 µs" constant — but **617.0 µs wall** through the `@iron.jit` path, 3.6× more, which is what the two kernels above were actually charged. Go/no-go for any future kernel: the op's CPU time must exceed ~617 µs (IRON) or ~170 µs (zero-overhead best case). Run it *before* writing a kernel (`results/aie/dispatch_floor_npu.log`). |
| `conv2x_baseline/` | Not an operator — the **CPU baseline** for mlir-aie's `ml/resnet/layers_conv2_x`, the chained int8 design that had run here since 2026-09-06 with its CPU side never measured | 3 bottlenecks, 436.21 MFLOP, all rows one sitting: NPU int8 **1869.6 µs** hw / **2497.8 µs** end-to-end vs CPU ORT QDQ int8 **295 µs** — **CPU wins 6.3–8.5×** like-for-like. torch fp32 (1856 µs) sits within 1% of the NPU, so benchmarking against torch alone would have read as parity. Closed *this design at this shape*; the op class is now closed too by `bottleneck_sweep/` below. Its host-cost claim ("628.2 µs reproduces 617.0 µs to ~2%") is **retracted** — different brackets; see that row (`results/aie/conv2x_int8_cpu_baseline.log`). |
| `bottleneck_sweep/` | Not an operator — the **spatial sweep** the conv2x log asked for: does NPU GOPS scale with problem size, or is conv simply slower here? `sweep.py` runs one standalone `ml/bottleneck` on the NPU; `cpu_sweep.py` runs the same arithmetic through ORT QDQ int8; same shapes, one sitting | `tensor_h` is the free axis (every L1 buffer is `tensor_w`×channels). NPU hw throughput rises **116.7 → 143.7 GOPS** over 16× more work, marginal **146.1** vs the CPU's **819.0** — the CPU wins **7.6×** at 32×32 and **5.7×** at 512×32, its own worst point. Utilization was worth 23% against a 570% gap, so **the op class is closed**. `tensor_w`=32 is a hard `aiecc` ceiling (Tile(0,4) needs 5×`w`×256 B + stack > 64 KB), so ResNet50's real 56×56 conv2_x **cannot compile** on this design. Open: column count, and kernel quality — 146 GOPS is ~7% of one column's peak (`results/aie/bottleneck_spatial_sweep_npu.log`). |

`groupnorm_bf16/extract_golden.py` and `groupnorm_bf16/measure_handoff_floor.py --role ep`
are the two scripts here that run in `resnet_env17`, not ironenv: the former pulls a
real node's input, params and ORT's own CPU output out of the model into `data/golden/`
(git-ignored) so the kernel is checked against the actual tensors, not random data; the
latter is one side of the two-process handoff-floor measurement above (the other side,
`--role kernel`, runs in ironenv like everything else here).
