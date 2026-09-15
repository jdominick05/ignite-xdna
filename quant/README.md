# Ignition Alpha

Ignition **0.1.0a1** independently calibrates and quantizes this repository's folded
ResNet50 and head-cut YOLOv8n ONNX exports into the XINT8 QDQ dialect used by the
XDNA1 VitisAI backend.
The quantization command imports neither Quark nor torch. It writes an ONNX model
and a `.quant.json` sidecar with version, hashes, calibration listing, candidate
errors, scales, parameter sharing and refinement decisions.

This alpha is a runnable repository feature, not an installable package or a general
ONNX quantizer. Its internal package remains `quant`. See the
[todo list](TODO.md) for the next milestones and [design](DESIGN.md) for the contract.

## Supported scope

| Area | Alpha contract |
|---|---|
| Measured models | `resnet50.a1_in1k`, folded FP32 export, default 224px input; `yolov8n_cut`, the head-cut YOLOv8n export at 640px; `modnet_cut`, the refactored MODNet matting export at 512px |
| Graph | Folded ResNet: standard-domain Conv/Relu/Add/MaxPool/GlobalAveragePool/Flatten/Gemm, one input/output, GAP receives a 7×7 spatial tensor. Head-cut YOLOv8: Conv/Sigmoid/Mul/Add/Concat/MaxPool/Resize/Split, one input, the six head outputs; Split is rewritten to Slice and SiLU to the DPU HardSigmoid chain as the vendor does. MODNet: Conv (including depthwise) / Relu / Clip / Add / Mul / Concat / Resize / GlobalAveragePool / Sigmoid, one input and one matte-logits output, any square GAP window; the vendor's onnxslim simplification runs first. The family is read from the operators |
| Export | Opset 17, IR 8, fully static batch 1 |
| Quantization | Exact-sample MinMSE over each pipeline's own reader (the timm transform, `npu.yolo.letterbox`, or `npu.modnet.preprocess`, all at the graph's input size); scalar power-of-two scales; UINT8/zp128 activations and INT8/zp0 weights/biases; a MaxPool/Resize output shares its input's parameters only when that input is already marked at the vendor's visit order, and otherwise gets its own; optional transcribed CLE (`--cle`, Conv→Conv pairs only; 33 patterns on ResNet, 9 on MODNet, zero on the SiLU net, as in the vendor's default preset) |
| Execution target | Windows, Phoenix/Hawk Point XDNA1, Ryzen AI 1.7.1; measured on Phoenix |
| AdaRound | `python -m quant adaround` on an emitted file of the ResNet or YOLO families (MODNet is not wired): Quark's FastFinetune AdaRound transcribed (torch, `resnet_env`), layers walked in the vendor's topological order of the float model; byte-identical to fresh same-listing `XINT8_ADAROUND` oracles on ResNet50 and yolov8n-cut, same machine and runtime. The rounding loop takes Quark's `OptimDevice` via `--device`; **cpu is the default and the only byte-parity path**, and anything else needs `--accept-non-parity` |
| Additional tools | Static inspection of any ONNX file (contract violations reported, not enforced), position-table replay, graph comparison, full classification evaluation, controlled EP probes, a refinement probe and a preparation probe against Quark. `check-shift-cut` is **advisory and must not gate**: it reports sigma per operation, the producer's `[14, 30]` band for Conv/Gemm only, operations whose scales it could not read (never counted as passing), and with `--branches` the inter-branch scale spread at Add/Concat |

Other graphs are unvalidated even if they share those operators. Unsupported operators
and batch/opset contracts fail explicitly. The MODNet family additionally needs
`onnxslim` at run time: the vendor delegates its own simplification step to it and so
does Ignition, rather than transcribing a third-party optimizer. The core still imports
with only numpy, onnx and onnxruntime in both environments. Exactly one of `--cle` and
`--no-cle` is required; `--cle` applies the transcribed default-preset equalization
(Conv→Conv pairs; depthwise pairs, Gemm pairs and Clip replacement raise). Per-channel
weights, INT32 bias and arbitrary scales are outside the alpha's production scope;
AdaRound is the separate `adaround` command, and is wired for the ResNet and YOLO
families only.
Probe mutations are experiments, not presets.

## Prepare the local artifacts

Follow the repository's [environment setup](../docs/SETUP.md#environments). Run from
the repository root in an activated environment. Export uses `resnet_env`; Ignition
and all NPU runs use `resnet_env17`. Do not replace its pinned VitisAI runtime with
a generic ORT install.

Required inputs are `models/resnet50_fp32.onnx`, `models/preprocess_config.json`,
and calibration images under `data/calib/`. The existing
[export script](../pipelines/resnet50/1_export.py) creates the graph and its matching
preprocessing configuration. Keep that configuration with the model. Generated models
and datasets are ignored by Git and are not bundled with the alpha.

For a fresh workspace, export in `resnet_env` with
`python pipelines/resnet50/1_export.py`; the
[ImageNet fetcher](../pipelines/resnet50/2_fetch_imagenet.py) prepares disjoint
calibration/evaluation folders (`--n-calib 300 --n-eval 1000`) using your authorized
dataset access. Existing local artifacts can be reused; no download is needed by
the quantization command itself.

## Quantize and inspect

In Git Bash, the wrapper activates the inference environment, records a UTF-8 log,
and calibrates without CLE unless `--cle` is given. Choose new output and log names on each run:

```bash
./scripts/quant-own.sh --out models/resnet50_ignition_alpha.onnx --log results/quant/quant_resnet50_ignition_alpha.log --limit 64
```

For custom paths, run the CLI directly after activation:

```bash
conda activate resnet_env17
python -m quant --version
python -m quant quantize --in-model models/resnet50_fp32.onnx --out models/resnet50_ignition_custom.onnx --no-cle --calib-dir data/calib --cfg-path models/preprocess_config.json --limit 64 --scratch scratch
python -m quant inspect models/resnet50_ignition_custom.onnx
```

Replace `--no-cle` with `--cle` for the default-preset recipe; the sidecar then carries
the ordered pair list and per-pair scale statistics under `cle_report`.

For the head-cut YOLOv8n export, name the model and the COCO calibration folder; the
letterbox size is read from the graph input and `--cfg-path` is ignored:

```bash
./scripts/quant-own.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --out models/yolov8n_cut_ignition_cle_c64.onnx --log results/quant/quant_yolov8n_cut_ignition_cle_c64.log --cle
```

The sidecar records `prepare.split_to_slice` (8 on yolov8n), the `hardsigmoid`
replacement/scaling counts (57 each) and every refinement move. Calibration spools
all 218 activation tensors, 7,106,560,000 bytes for 64 images, under `scratch/`.

`inspect` loads permissively and reports `export_contract` and `onnx_checker` per
file; `quantize` keeps the strict loader. Direct Python commands print to the
terminal; use the shell wrappers when collecting repository evidence. Existing output models and sidecars are never overwritten.
Calibration checks free disk from inferred tensor sizes and removes its private spool
on normal completion or a Python exception. Hard process termination can leave a spool
and a partial model/sidecar pair; see [Recover from a killed run](#recover-from-a-killed-run).
The historical `pipelines/resnet50/3c_quantize_own.py` entry point delegates to this CLI.

`--scales-from <reference.onnx>` selects position-table replay instead of independent
calibration. It is a parity/debugging mode, not needed to quantize from the float model.
It does not copy the reference's integer weights or topology.

## AdaRound

`adaround` finetunes the weight rounding of an emitted file the way Quark's
`XINT8_ADAROUND` preset does after calibration. It reads the CLE flag and the
calibration listing from the base's sidecar, rebuilds the equalized float reference,
checks the float model's hash, and writes a new model plus sidecar (an `adaround`
section with per-layer reconstruction metrics, iterations, changed elements, peak
working set and versions; scales and positions are unchanged). It imports torch, so
the wrapper uses `resnet_env` (`--env` overrides that, for device timing only); Quark
stays blocked:

`--device` selects the torch device for the Adam rounding loop (Quark's `OptimDevice`).
ORT activation extraction always stays on the CPU. **The default `cpu` is the only
byte-parity path**: a GPU changes float reduction order in Adam and in the convolutions,
so `--device` off `cpu` requires `--accept-non-parity`, records `byte_parity_path: false`
in the sidecar's `adaround` section, and logs a warning. A run made that way must be
compared on accuracy, never quoted as matching a CPU oracle. A device torch cannot reach
raises rather than silently falling back to the CPU. Until 2026-09-10 no GPU run had been
made. Since then ResNet50 has run on Desktop 1's RX 7900 XTX
(`scripts/quant-adaround.sh --env resnet_env_rocm --device cuda`): 2.27× faster end to
end, and 1 LSB off the same-env CPU run in 12.58 % of int8 elements
([BENCHMARKS](../docs/BENCHMARKS.md#ignition-adaround-on-the-rx-7900-xtx-2026-09-10-desktop-1)).
See `quant/TODO.md` for what remains.

```bash
./scripts/quant-adaround.sh --quant models/resnet50_ignition_cle_c64.onnx --out models/resnet50_ignition_cle_adaround_c64.onnx --log results/quant/quant_resnet50_ignition_cle_adaround_c64.log
./scripts/quant-adaround.sh --in-model models/yolov8n_cut.onnx --calib-dir data/coco_calib --quant models/yolov8n_cut_ignition_cle_c64.onnx --out models/yolov8n_cut_ignition_cle_adaround_c64.onnx --log results/quant/quant_yolov8n_cut_ignition_cle_adaround_c64.log
```

The family is read from the base's sidecar; a head-cut YOLO base needs its float export
and COCO folder and letterboxes to the graph input (`--cfg-path` is not read). The
layers are visited in the order Quark's loop takes, ORT's topological sort of the float
model (`Graph.vendor_order`), which is not the export's file order. About nine minutes
on Desktop 2 for ResNet50's 54 layers, peak working set about 3 GB; about eight and a
half minutes and 5.7 GB for yolov8n-cut's 63 layers on 64 images. The matching oracle
is `./scripts/quant-reference.sh --cle --adaround` (with `--in-model` and `--calib-dir`
for YOLO), and `scripts/quant-validate.sh` gates the pair like any other artifact;
`INT8_EXACT` in its diff log is the bitwise verdict.

## Recover from a killed run

Normal completion and a Python exception clean up after themselves. A hard termination
(`taskkill /F`, a closed terminal, a crash or a power loss) ends the process before
Python's cleanup or a wrapper's `trap` can run. It can leave a calibration spool, a
partial model/sidecar pair and a partial log. Remove only what the killed run created,
by exact name. `scratch/`, `%TEMP%`, `models/` and `results/quant/` also hold other
runs' files, so never delete any of them wholesale or by a broad glob.

First confirm the run has really ended; a live run's files look the same as a stranded
run's. In PowerShell, none of the listed command lines may be a quantization
(`-m quant`, `3d_quantize_compare.py` or another Quark script). Also look for the
`tools/torch_device_info.py --watch` GPU sampler: `quant-adaround.sh --device` starts it,
it runs until killed, and the wrapper stops it only from its EXIT trap
(`scripts/quant-adaround.sh:78-83`). If one is still listed after its wrapper died, stop
it with `Stop-Process -Id <ProcessId>`.

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine | Format-List
```

### Calibration spools

| Killed command | Spool it can leave |
|---|---|
| `quant-own.sh`, `quant-adaround.sh`, or `python -m quant quantize` without `--calib-method exact` | None: `hist` calibration writes nothing to disk, `quant-own.sh` has no `--calib-method` flag, and `adaround` writes no spool |
| `python -m quant quantize --calib-method exact` | `owned-calib-*` directories of `tensor_<i>.f16` files under the `--scratch` root, `scratch/` by default (`quant/calib.py:205`) |
| `scripts/quant-reference.sh` | `quark_onnx.*` directories inside its private `scratch/quant-reference.*` (`scripts/quant-reference.sh:51-54`) |
| Any other script that calls `quark_guard`, or a Quark pipeline script run by hand | `quark_onnx.*` directories directly under `%TEMP%` |

Only those commands create `owned-calib-*` and `quant-reference.*`, so once no
quantization is running, every such directory under the checkout's `scratch/` (or the
`--scratch` root you passed) is stranded. List them with sizes, then delete each by the
exact path `du` printed:

```bash
du -sh scratch/owned-calib-*/ scratch/quant-reference.*/ 2>/dev/null
rm -rf -- scratch/owned-calib-XXXXXXXX
```

A normal `quant-reference.sh` exit leaves its `quant-reference.*` directory behind too:
`quark_cleanup` (`scripts/lib.sh:320-336`) removes the `quark_onnx.*` directories and
the marker inside it, not the directory itself.

`%TEMP%` is shared with every other program and checkout. A killed wrapper leaves its
`.quark_guard.<pid>` marker there, created when the run armed its cleanup
(`scripts/lib.sh:314-318`). Its spools are among the `quark_onnx.*` directories newer
than that marker, the test `quark_cleanup` applies on a normal exit. That test alone
does not prove ownership, because a later or overlapping Quark run passes it too, so
delete only the directories whose times also fall before the kill. A Quark script run
by hand leaves no marker; use the time it started instead.

```bash
t="$(cygpath -u "$TEMP")"
find "$t" -maxdepth 1 \( -name 'quark_onnx.*' -o -name '.quark_guard.*' \) -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort
find "$t" -maxdepth 1 -type d -name 'quark_onnx.*' -newer "$t/.quark_guard.PID" -print0 | xargs -0 -r du -sh
rm -rf -- "$t/quark_onnx.calib.XXXXXXXX" "$t/quark_onnx.quant.XXXXXXXX" "$t/.quark_guard.PID"
```

### Partial output pairs

Each producer writes the model first and its sidecar last, so a sidecar that matches
the model marks a finished run:

| Producer | Sidecar | Model hash field | Writes |
|---|---|---|---|
| `python -m quant quantize` (`quant-own.sh`) | `<out>.quant.json` | `output_sha256` | `quant/quantize.py:166-172` |
| `python -m quant adaround` (`quant-adaround.sh`) | `<out>.quant.json` | `output_sha256` | `quant/cli.py:213-233` |
| `quant-reference.sh` | `<out>` with `.onnx` replaced by `.reference.json` | `model_sha256` | `pipelines/resnet50/3d_quantize_compare.py:97-107` |

A pair is complete only when the sidecar parses and that field equals the model's hash
(any Python will do):

```bash
sha256sum models/resnet50_ignition_alpha.onnx
python -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['output_sha256'])" models/resnet50_ignition_alpha.onnx.quant.json
```

A missing sidecar, a JSON error from a truncated one, or two different hashes mark a
partial pair. The producers refuse an existing model or sidecar
(`quant/quantize.py:105-106`, `quant/cli.py:147`,
`pipelines/resnet50/3d_quantize_compare.py:43-44`), so a retry under the same `--out`
fails until that one pair's two paths are deleted by name; or choose a new `--out`.
Output models are gitignored (`models/`, `*.onnx`), so nothing restores a deleted file.

A killed wrapper also leaves its `--log` and, beside it, the `load_<name>` host-load
witness (`scripts/lib.sh:162-166`); the wrappers refuse an existing `--log`.
`quant-adaround.sh --device` also leaves a `gpuload_<name>` witness there and refuses to
start while it exists (`scripts/quant-adaround.sh:58-59`). Logs under
`results/` are tracked evidence: `git ls-files -- <path>` prints nothing for a file git
does not track. Delete or rename only such a file, never a tracked log.

## Verify an output

The optional Quark comparison requires a separately generated same-listing reference
built with the same CLE setting (`scripts/quant-reference.sh`, `--cle` or not) and its
provenance file; the [producer experiment](../docs/BENCHMARKS.md#owned-resnet50-no-cle-re-emission-and-independent-calibration)
documents those commands. With that reference and labeled `data/eval/` available:

```bash
./scripts/quant-validate.sh --model models/resnet50_ignition_alpha.onnx --reference models/resnet50_quark_nocle_c64.onnx --tag resnet50_ignition_alpha_verify
./scripts/quant-validate.sh --family yolo --model models/yolov8n_cut_ignition_cle_c64.onnx --reference models/yolov8n_cut_quark_cle_c64.onnx --tag yolov8n_cut_ignition_cle_c64
```

This compares parameters, runs full labeled CPU/NPU evaluation (all 5,000 val2017
images through `pipelines/yolov8n/5_eval_map.py` with `--family yolo`), requires a clean
pre-run hardware-context check, uses fresh compilation and records the EP report.
For CPU-only verification add `--cpu-only`. Quark is needed only to generate the
optional oracle, not by Ignition or its inference commands.

`./scripts/quant-refine-probe.sh --log results/quant/refine_probe_<tag>.log` perturbs
the oracle's positions and diffs Quark's refinement against Ignition's on identical
inputs. It runs in `resnet_env` because it imports Quark, and writes only its log.

`./scripts/quant-prepare-probe.sh --log results/quant/prepare_probe_<tag>.log --in-model <float export> [--replay-from <committed XINT8 file>] [--cle]`
compares Quark's pre-calibration graph with Ignition's prepared graph, reports each
vendor preparation step in isolation and, with `--replay-from`, re-emits a committed
artifact from its own positions and diffs it. Same environment rule, same single output.

Read [Alpha validation](../docs/BENCHMARKS.md#ignition-alpha-release-validation) for
the versioned artifact's evidence and [acceptance findings](../docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance)
for numerical traps. Successful EP placement does not establish correct outputs;
even an optimized CPU reference can disagree with unoptimized ONNX computation.
With `--cle` the output reproduces the repository's plain-XINT8 ResNet50 to the integer
([CLE parity](../docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset)); `adaround` on that file
reproduces a fresh `XINT8_ADAROUND` oracle byte for byte on the same machine
([AdaRound parity](../docs/BENCHMARKS.md#ignition-adaround-parity)). On the head-cut
YOLOv8n export the same `--cle` run matches a fresh same-listing oracle position for
position and integer for integer ([YOLO preparation parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity)),
and `adaround` on that file is byte-identical in every integer to a fresh
`XINT8_ADAROUND` oracle as well ([YOLO AdaRound parity](../docs/BENCHMARKS.md#ignition-yolov8n-cut-adaround-parity)).
