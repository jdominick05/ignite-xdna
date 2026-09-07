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
| `groupnorm_bf16/` | `InstanceNormalization` (really `GroupNorm(32)`) in `resnetv2_50x3_xint8.onnx`, the op that falls to CPU on that model | Kernel alone beats CPU on 33/49 nodes (`results/aie/groupnorm_bf16_kernel_npu.log`); the v1 two-process handoff floor erased the win at every shape, but that floor used a conversion function with no SIMD path, not a hardware limit -- a faster conversion plus the real int8-boundary target reopen the question, unbuilt. See `results/aie/groupnorm_bf16_handoff_floor_v2_npu.log` |

`groupnorm_bf16/extract_golden.py` and `groupnorm_bf16/measure_handoff_floor.py --role ep`
/ `measure_handoff_floor_v2.py --role ep` are the scripts here that run in
`resnet_env17`, not ironenv: the first pulls a real node's input, params and ORT's own
CPU output out of the model into `data/golden/` (git-ignored) so the kernel is checked
against the actual tensors, not random data; the other two are one side each of the
two-process handoff-floor measurements above (the other side, `--role kernel`, runs in
ironenv like everything else here). `measure_handoff_floor_v2.py` replaces
`ml_dtypes.astype()` (a scalar per-element loop -- bfloat16 isn't a native numpy dtype,
so there's no SIMD path) with a strided-view truncation and preallocated buffers,
cutting the measured floor at L=301056 from 23.6ms to 5.7ms, and adds `--no-convert`
to isolate the protocol cost alone (1.19ms -- already under CPU's 3.47ms on its own).
See `results/aie/groupnorm_bf16_handoff_floor_v2_npu.log` for the full writeup,
including why the real CPU cost this design should be compared against is higher than
InstanceNorm alone (56.8% higher -- the QuantizeLinear/DequantizeLinear nodes wrapping
every site are a real target too, not just InstanceNorm).
