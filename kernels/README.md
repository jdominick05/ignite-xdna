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
| `groupnorm_bf16/` | `InstanceNormalization` (really `GroupNorm(32)`) in `resnetv2_50x3_xint8.onnx`, the op that falls to CPU on that model | Kernel alone beats CPU on 33/49 nodes (`results/aie/groupnorm_bf16_kernel_npu.log`), but the measured two-process handoff floor erases the win at every shape -- 0/49 once spliced (`results/aie/groupnorm_bf16_handoff_floor_npu.log`) |

`groupnorm_bf16/extract_golden.py` and `groupnorm_bf16/measure_handoff_floor.py --role ep`
are the two scripts here that run in `resnet_env17`, not ironenv: the former pulls a
real node's input, params and ORT's own CPU output out of the model into `data/golden/`
(git-ignored) so the kernel is checked against the actual tensors, not random data; the
latter is one side of the two-process handoff-floor measurement above (the other side,
`--role kernel`, runs in ironenv like everything else here).
