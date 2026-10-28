# results/linux/

The first runs of the engine on Linux. All of them are on the **laptop** (Ryzen 5 8645HS, Hawk Point XDNA1,
16 GB) under Arch Linux, kernel 7.2.11, with the in-tree `amdxdna` driver and its firmware **1.5.5.391**. XRT
**2.21.75** (XRT hash `4eb1f439`) and the `amdxdna` shim plugin come from `amd/xdna-driver` at tag `2.21.75`,
built from source into `/opt/xilinx/xrt`. The shim needed one gcc 16 fix: two `std::memory_order::memory_order_seq_cst`
became `std::memory_order_seq_cst`.

Software: Python 3.13.18 (uv) and mlir-aie **1.4.2**. Peano is **llvm-aie 22.0.0.2026090201+a36c62b9**, the build
the Windows logs name, for everything from `compile_yolov8n_full_…180553Z` on. The first clock probe
(`…180335Z`) and the crashed compile (`…180421Z`) ran under mlir-aie 1.4.2's own pin, `21.0.0.2026080301+c9c5ecb7`.
The container verifications use no Peano at all. Every log here opens with `RESEARCH_PLATFORM`, which records the Linux stack it ran under: kernel,
NPU firmware, XRT hash, platform profile, EPP and AC. Each log also carries idle `xrt-smi` witnesses before and after.

**Every run here is `--checks-only`.** These logs establish correctness on Linux. The dispatch times they print are
2-iteration smoke figures, and none of them is a timing claim or a comparison with Windows.

## The engine runs exact on Linux (2026-10-28)
- **Published containers, compiled on Windows** (Hugging Face revision `e2dd0f21d490`). Each one's xclbin and
  instruction stream run unchanged and **layer-exact** under Linux XRT. "Layers" counts readable layers, then reused
  workspace slots holding their planned final tenant:

  | Container | Layers | Log |
  |---|---|---|
  | `yolov8n_full` | 28/28, 38/38 | [log](verify_yolov8n_full_laptop-linux_20261028T180251Z.log) |
  | `yolov8s` | 28/28, 38/38 | [log](verify_yolov8s_laptop-linux_20261028T180253Z.log) |
  | `yolov8n_pose` | 37/37, 38/38 | [log](verify_yolov8n_pose_laptop-linux_20261028T180255Z.log) |
  | `yolo11n` | 31/31, 60/60, plus its one ONNX Runtime host segment | [log](verify_yolo11n_laptop-linux_20261028T180256Z.log) |
  | `sesr_m7` | 5/5, 4/4 | [log](verify_sesr_m7_laptop-linux_20261028T180258Z.log) |

  The engine passes the instruction size in **bytes** (`runtime/driver.py`). amdxdna accepts that, and the IRON
  passthrough's **word** count works too.
- **Compiled on Linux.** `ignite-compile --engine graph --silu-sigmoid` builds `yolov8n_full` from the published
  ONNX in 9.0 s ([log](compile_yolov8n_full_laptop-linux_20261028T180553Z.log)). The result is layer-exact on the
  device ([log](verify_linuxbuilt_yolov8n_full_laptop-linux_20261028T180615Z.log)).
  - It is not byte-comparable with the published container. That container was compiled at `394fc3d`, before layer
    overlap became the default.
  - Whether the compiler is deterministic across operating systems needs a Windows compile at the same commit, and
    is not established here.
- **Peano pin.** mlir-aie 1.4.2's own pin, llvm-aie `21.0.0.2026080301+c9c5ecb7`, **crashes** compiling
  `kernels/aie2/conv_engine/engine.cc`: a SIGSEGV in `InstructionSelect` on `@engine`
  ([log](compile_yolov8n_full_laptop-linux_20261028T180421Z.log)). The engine needs the a36c62b9 nightly the Windows
  builds used.

## The NPU on Linux
- **Clock.** In `xrt-smi` power mode `Default`, the trace-timer probe fits the core clock at **1.7995 GHz** (scalar)
  and **1.7996 GHz** (vector), R² 0.999999 and 0.999954 ([log](clock_probe_default_laptop-linux_20261028T180335Z.log),
  Peano 21).
  - Re-run with the a36c62b9 Peano, it gives **1.7957 / 1.7864 GHz**, 0.52% apart
    ([log](clock_probe_default_a36c62b9_laptop-linux_20261028T181724Z.log)).
  - Both builds' scalar loop costs exactly **9.000 cycles per iteration** (cycles = 9 × iterations + 4 at every
    length). That is the constant `tools/os_bench.py` uses to turn an untraced replay into a clock.
  - That is the 1.80 GHz Desktop 2 measured under Windows (`docs/SILICON.md`).
  - The worker lands on physical column 1 (`r2c1`), as on Windows: column 0 stays outside the partition.
- **Contexts.** One process gets **6** hardware contexts on one xclbin. The 7th fails with
  `DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... err=-22`, on both a 1-column passthrough and the YOLOv8n engine xclbin
  ([passthrough](context_limit_5dfdf62b365d1cccc4c93120_final_laptop-linux_20261028T180640Z.log),
  [engine](context_limit_ignite_cache_engine_2a1700172dc22a73_laptop-linux_20261028T180640Z.log)).
  - Windows allows 5 (`docs/DECISIONS.md`, NTSTATUS `0xc01e0009` on the sixth).
  - `xrt-smi` lists one active row at the peak, so this does not establish that all six are resident at once.
- **pyxrt.** The 2.21.75 build, with pybind11 3.1.0, prints `ert_cmd_state.ERT_CMD_STATE_COMPLETED`. That is the
  string the runtime compares against.
