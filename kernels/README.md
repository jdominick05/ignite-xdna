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

The first run of a given shape JIT-compiles (Peano for the C++ kernel, aiecc for the
array program) into `~/.npu/cache/<hash>/`; later runs of the same shape hit the cache.

| Kernel | Op it replaces | Status |
|---|---|---|
| `groupnorm_bf16/` | `InstanceNormalization` (really `GroupNorm(32)`) in `resnetv2_50x3_xint8.onnx`, the op that falls to CPU on that model | PASS on real node tensors at L = 75264 / 150528 / 301056; `results/aie/groupnorm_bf16_kernel_npu.log` |

`groupnorm_bf16/extract_golden.py` is the one script here that runs in `resnet_env17`:
it pulls a real node's input, params and ORT's own CPU output out of the model into
`data/golden/` (git-ignored) so the kernel is checked against the actual tensors, not
random data.
