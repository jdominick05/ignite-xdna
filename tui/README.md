# tui/ — a launcher for the models

```powershell
conda activate resnet_env17
$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
python -m tui
```

or `./scripts/tui.sh` from Git Bash, which sets that up for you but gives numbered
menus instead of arrow keys (mintty is not a real Windows console, so `msvcrt` cannot
read keys there). Needs `rich` — see [SETUP](../docs/SETUP.md#the-launcher-optional).

Two lanes. **Tasks** point a model at your own image and write a usable result:
remove a background, detect objects, estimate pose, make a depth map, upscale, segment
a street scene. **Demos** launch the ten scripts in [`demos/`](../demos/README.md).

### Live camera

Six tasks take a live source as well as an image — **remove background, detect objects,
detect (YOLOv6n), estimate pose, depth (FastDepth), depth (MiDaS)**. Answer the input
prompt with a camera index (`0` is the default camera) instead of a path; detect,
detect-v6 and pose take a video file too. `q` or `ESC` quits, `s` saves a snapshot.

```powershell
python -m tui.task detect --input 0 --ep npu          # until you press q
python -m tui.task detect --input 0 --ep npu --frames 60 --no-window
```

`--frames N` stops after N frames and `--no-window` runs headless — together they are
how the live path gets smoke-tested with nobody at the machine. Headless removes the
only way to quit early, so use them as a pair.

Each frame goes through the same `DISPATCH[family]` adapter a still image does, never a
second preprocess path: MODNet shipped once with its transform copied five times and two
of them disagreed. A live run leaves its last frame as a PNG with the numbers burned in,
plus the usual sidecar carrying `frames`, mean/median/p95 `sess.run` and the loop fps.
**The two numbers are not the same quantity** — `sess.run` is the hardware, the loop fps
includes capture, pre/post and drawing — and both are `quotable: false`.

A camera that will not open is a hard error. It does **not** fall back to stock images,
because a result you cannot distinguish from a live one is worse than no result — the
same reason [`npu.session.resolve_xclbin`](../npu/session.py) raises rather than warns.

## What it is not

Not a measurement tool. [`results/`](../results/README.md) is the tracked evidence
base with a naming contract, so nothing here writes there — everything lands in
git-ignored `outputs/`. Every latency printed is `sess.run` alone, one image, one
sitting, and its sidecar says `quotable: false`. Numbers with their method and caveats
live in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

It also does not export or quantize — that needs `resnet_env` and Quark, a different
process — and it does not wrap `scripts/*-bench.sh`, because a logged run belongs in
`results/` and should be a deliberate act.

## Why it exists

The run-stage entry points do not agree with each other. `yolov8n-pose` has no `dml`
choice; `--ep` defaults to `cpu` in `resnet50/4_run.py` and the yolo `4_detect`
scripts but `npu` everywhere else; `--fresh` is NPU-gated in bisenetv2/fastdepth/
midas/realesrgan/sesr and unconditional in resnet50/modnet/all-yolo; verbosity is
`--log {0,1,2}` in yolo and `--verbose` elsewhere; and five pipelines compute their
cache key internally with no override. `registry.py` records those differences once
so a command line is built correctly rather than plausibly.

The other half is that a convenient launcher is exactly how a CPU run in an NPU
costume gets made. So every run re-checks for foreign hardware contexts and host load
before it starts, and reads `<cacheKey>/vitisai_ep_report.json` afterwards — a `_npu`
in a filename means the EP was *requested*, not that it engaged. For `cpu` and `dml`
it says so plainly instead of reporting `0/N`, because only VitisAI writes a report.

**Stale caches are caught, not guessed at.** The compile cache is keyed by name, not
by model hash, so picking a different model in the same family silently reuses the
previous compile. `<cacheKey>/context.json` records what that compile was built from,
so switching model forces `--fresh` on the evidence rather than on a rule someone has
to remember.

## Layout

| file | what |
|---|---|
| `registry.py` | the catalogue: entries, models, cache keys, flag dialects |
| `guards.py` | machine, env, xclbin, contention, host load, staleness, EP verdict |
| `runner.py` | builds a demo's argv and spawns it |
| `task.py` | `python -m tui.task <entry>` — one model, one input, one result |
| `app.py` | the menu loop and the preflight panel |
| `ui.py` | rendering; arrow keys where possible, numbered menus otherwise |
| `selftest.py` | `--selftest`: validates the catalogue, opens no hardware context |

Everything that touches the device is a subprocess, so the launcher never holds a
hardware context of its own and a DPU timeout takes down a child rather than the UI.

## Checking it

```bash
python -m tui --selftest     # every model path, cache key and argv; no hardware
```

That is the regression test — it is safe to run while the NPU is busy. Anything
behavioural still needs a real run; see [CONTRIBUTING](../CONTRIBUTING.md#testing).
