# Setup

Which hardware and SDK this runs on, the two conda environments and why they are two,
and how to install and build the models by hand. The footguns in here are the ones that
cost real debugging time on this project — the `ninja.exe`/PATH activation trap, a stale
machine-wide `RYZEN_AI_INSTALLATION_PATH`, PowerShell writing UTF-16 logs, Git Bash
versus WSL. `README.md` links here; `docs/DECISIONS.md` has the reasoning behind the
version pins.

## Compatibility

This is a narrow target, and most of the narrowness is not optional.

- **Hawk Point or Phoenix** (XDNA1, `AMD_AIE2_4x4_Overlay`, provider option
  `target: "X1"`). Strix is a different architecture and none of the firmware paths
  here apply to it.
- **Windows.** XDNA1 has no Linux userspace.
- **Ryzen AI 1.7.1 for inference.** Not 1.8.0 — see [Key findings](BENCHMARKS.md#key-findings).
- **PowerShell**, not cmd. The scripts print resolved paths at startup so you can see
  when an environment variable did not take.

XDNA1's support matrix is CNN INT8 only: no BF16, no transformer/NLP paths, no LLMs.
That is a vendor-level limit, not a configuration problem.

### Environments

Two conda environments, because the SDK version that can *export and quantize* is not
the one that can *run on the NPU*:

| Env | Clone of | Used for | Key contents |
|---|---|---|---|
| `resnet_env` | `ryzen-ai-1.8.0` | export + quantize | torch, timm, ultralytics, quark 0.11 |
| `resnet_env17` | `ryzen-ai-1.7.1` | **all NPU inference** | onnxruntime-vitisai 1.23.3, opencv, pillow |

```powershell
conda create -n resnet_env   --clone ryzen-ai-1.8.0
conda create -n resnet_env17 --clone ryzen-ai-1.7.1
```

Do not install torch or timm into `resnet_env17`. Keeping the inference environment
thin is what keeps Quark out of the inference import path.

**Always `conda activate` — never call `envs\<name>\python.exe` by its path.** Quark's
import needs `ninja.exe`, which lives in the environment's `Scripts\` directory and
only reaches `PATH` on activation. Without it, `import quark` fails with
`RuntimeError: Ninja is required to load C++ extensions`.

Before any NPU run:

```powershell
conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
```

The 1.8.0 installer sets that variable machine-wide and already-open shells keep the
stale value, so set it per session or pass `--xclbin` explicitly.

### NPU monitor: `tools/hwinfo_npu_bridge.exe`

A live dashboard for the NPU, and a bridge that publishes the same numbers to HWiNFO64's
custom-sensor registry interface. Everything it shows is read from one of three sources,
and the screen says which (`results/aie/xrt_api_live_clock_and_pdh_npu.log` is the probe
behind each claim):

- **Utilization and memory** — Windows' own GPU-engine statistics for the NPU adapter (the
  D3DKMT counters Task Manager reads, via PDH). The NPU is an MCDM adapter with a LUID and a
  compute engine like any GPU; an IRON/XRT GEMM run reads 84–88 %, idle reads 0 %, and the
  adapter's shared memory equals xrt-smi's figure. This is the only live utilization number
  on this stack — xrt-smi's GOPS/FPS/latency columns read `N/A` for such contexts. Not yet
  confirmed under a VitisAI EP session (three attempts failed for unrelated reasons; see the
  log).
- **Clock and power mode** — XRT's in-process query API. `max_clock_frequency_mhz` is a live
  readback on this driver: 800 MHz idle in every power mode, and the mode's clock while a
  hardware context is active (1800 `default`/`performance`/`turbo`, 1028 `balanced`, 800
  `powersaver`; `results/aie/pmode_clock_readback_npu.log`).
- **Contexts, columns, counters** — `xrt-smi examine -r aie-partitions`, read on its own
  thread every `--smi-interval` seconds (a frame shows the age of the sample it used): pid,
  process, status, submissions/completions (and their per-second deltas between reads),
  migrations, suspensions, errors, priority, memory, and whatever GOPS the context reports.

Not shown, because no documented interface exposes them on this NPU: voltage and power
(xrt-smi's electrical query fails at the driver escape; thermal reports no sensors).

```powershell
# one-time: header-only build deps (nlohmann/json + boost, which the XRT SDK headers need)
conda create -n npu_monitor_build -c conda-forge libboost-headers nlohmann_json
# build (Visual Studio 2022 Build Tools; links the XRT SDK from C:\Xilinx\XRT\xrt_sdk when present)
scripts\build_hwinfo_bridge.bat            # or ./scripts/build-hwinfo-bridge.sh from Git Bash
# run
tools\hwinfo_npu_bridge.exe                # live dashboard: 0.5 s polls, xrt-smi every 2 s, publishes to HWiNFO
tools\hwinfo_npu_bridge.exe --interval 0.1 # ten utilization/clock samples a second (--smi-interval <s> for xrt-smi, 0 = never)
tools\hwinfo_npu_bridge.exe --once        # one plain sample;  --json / --plain for scripts
tools\hwinfo_npu_bridge.exe --no-hwinfo   # monitor only, touches no registry key
tools\hwinfo_npu_bridge.exe --idle hide   # drop the activity sensors from HWiNFO while the NPU is idle
```

Two polling cadences, because xrt-smi is a child process that costs a few hundred ms per
report while the engine counters and the clock readback cost a few ms: `--interval` (default
0.5 s, minimum 0.1) drives utilization, memory and clock; `--smi-interval` (default 2 s,
0 = never) drives xrt-smi, and completions/s are deltas between xrt-smi reads rather than
between frames. In the dashboard `+`/`-` halve and double the poll interval, `[`/`]` the
xrt-smi interval, `s` turns xrt-smi off and on, `p` pauses (the activity sensors leave HWiNFO
while paused) and `q` quits; the footer prints the requested and the measured period, and
`--json` carries the same per sample (`period_s`, and `sample`/`age_s` in the xrt-smi section),
which is the route for logging a run. Measured on Desktop 2 with three monitors side by side
across one 2048³ bf16 GEMM hold (`results/aie/npu_monitor_poll_rate_npu.log`): the engine
counter read a mean 88.2 % over 312 polls at 0.1 s (single polls 82–94 %), 87.9 % at 0.25 s
and 87.8 % at 0.5 s (86–90 %) — the same figure at every rate, a little more scatter at the
fastest, no dropouts — and the period is exact at all three (0.100 / 0.250 / 0.500 s) once the
process asks Windows for a 1 ms timer tick; before that, 20 ms naps ran ~31 ms on the default
15.6 ms tick and every period carried ~22 ms over the request (0.122 / 0.275 / 0.527 s).

A running `hwinfo_npu_bridge.exe` locks its own file, so stop it before rebuilding. HWiNFO
picks the sensors up from `HKCU\Software\HWiNFO64\Sensors\Custom\<device name>` while its
Sensors window is open; `--clean` removes that group on exit. HWiNFO's Min/Max/Average
columns average every sample a custom sensor is given and there is no way to hand it "no
reading", so two rules apply: a value xrt-smi reports as `N/A` is never published as 0 (the
key is removed instead), and `--idle hide` removes utilization, clock, completions/s,
submissions/s and GOPS while no hardware context is active, so the Average covers the time
the NPU was actually doing something. The default keeps publishing the true idle readings
(0 %, 800 MHz, 0/s), which is what drags a whole-session Average down. An earlier build of
the bridge left a frozen second group, "XDNA NPU", with its old sensor schema; the current
build removes it at start-up.

---

## Quick install

The common pipelines are wrapped in shell scripts under `scripts/`. They handle
conda activation, the Ryzen AI environment variables, preconditions, logging and
cleanup, so the usual answer is one command:

```bash
./scripts/setup.sh          # one-time: export, fetch datasets, quantize
./scripts/resnet-bench.sh   # the ResNet50 results table below
./scripts/yolo-cut.sh       # YOLOv8n on the NPU: cut, quantize, run, verify
./scripts/yolo-demo.sh      # live webcam detection, q to quit
./scripts/yolo-eval.sh      # COCO bbox mAP table
./scripts/pose-cut.sh       # YOLOv8n-pose on the NPU: same recipe, keypoints
./scripts/pose-eval.sh      # COCO OKS keypoint mAP table
./scripts/diag.sh           # what the VitisAI EP actually took
```

Run them from **Git Bash**, not WSL — the XDNA1 NPU has no Linux userspace, so a
WSL run would silently be CPU-only. Every script takes `--help`. Logs land in
`results/`, in UTF-8; PowerShell's `*>` writes UTF-16, which makes later greps
silently match nothing.

## Manual setup

Not on the scripted path, or want to drive a step by hand? Both pipelines follow the
same `1_export → 2_fetch_data → 3_quantize → 4_run` shape; the commands below reproduce
what the scripts above do.

### ResNet50 (working)

```powershell
conda activate resnet_env
python pipelines/resnet50/1_export.py
python pipelines/resnet50/2_fetch_imagenet.py --n-calib 300 --n-eval 1000 --shards 0 7
python pipelines/resnet50/3_quantize.py --calib-dir data/calib --config XINT8_ADAROUND

conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
python pipelines/resnet50/4_run.py --ep npu --images data/eval --n 1000 --model models/resnet50_xint8_adaround.onnx --fresh
```

For the FP32 CPU baseline: `--ep cpu --model models/resnet50_fp32.onnx`.

Step 2 needs a Hugging Face account approved for the gated `ILSVRC/imagenet-1k`
dataset, plus `hf auth login`.

### YOLOv8n

The head-cut path is the one that reaches the NPU. `1b` removes the float decode tail,
`3b` quantizes the result, and `4_detect.py` decodes in numpy — it switches on the
model's output count, so the same command drives either graph shape.

```powershell
conda activate resnet_env
python pipelines/yolov8n/1_export.py --size 640
python pipelines/yolov8n/2_fetch_coco.py --n-calib 300
python pipelines/yolov8n/1b_cut_head.py
python pipelines/yolov8n/3b_quantize_cut.py --calib-dir data/coco_calib --limit 300

conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx --ep npu --source assets/test_image.jpg --fresh --log 1
python pipelines/yolov8n/4_detect.py --model models/yolov8n_cut_xint8.onnx --ep npu --source 0
```

`--source 0` is the webcam; press `q` to quit. The COCO download is about 1 GB and is
ungated.

Quantizing at 640×640 spools roughly 105 MB of calibration activations **per image**
into `%TEMP%`, so `--limit 300` peaks near 31 GB and leaves the cache behind if the
process is killed. `./scripts/yolo-cut.sh` checks free space first and cleans up on
exit, including on Ctrl-C; if you run `3b` by hand, watch your disk.

The original full-graph steps (`3_quantize.py`, and `4_detect.py` against
`yolov8n_xint8.onnx`) still work and still produce a model the EP refuses. They are kept
because they are the control that makes the cut model's result meaningful.

---

