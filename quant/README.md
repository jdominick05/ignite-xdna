# Ignition Alpha

Ignition **0.1.0a1** independently calibrates and quantizes this repository's folded
ResNet50 ONNX export into the XINT8 QDQ dialect used by the XDNA1 VitisAI backend.
The quantization command imports neither Quark nor torch. It writes an ONNX model
and a `.quant.json` sidecar with version, hashes, calibration listing, candidate
errors, scales, parameter sharing and refinement decisions.

This alpha is a runnable repository feature, not an installable package or a general
ONNX quantizer. Its internal package remains `quant`. See the
[todo list](TODO.md) for the next milestones and [design](DESIGN.md) for the contract.

## Supported scope

| Area | Alpha contract |
|---|---|
| Measured model | `resnet50.a1_in1k`, folded FP32 export, default 224px input |
| Graph | Standard-domain Conv/Relu/Add/MaxPool/GlobalAveragePool/Flatten/Gemm; one input/output; GAP receives a 7×7 spatial tensor |
| Export | Opset 17, IR 8, fully static batch 1 |
| Quantization | Exact-sample MinMSE; scalar power-of-two scales; UINT8/zp128 activations and INT8/zp0 weights/biases; no CLE |
| Execution target | Windows, Phoenix/Hawk Point XDNA1, Ryzen AI 1.7.1; measured on Phoenix |
| Additional tools | Static inspection, position-table replay, graph comparison, full classification evaluation and controlled EP probes |

Other graphs are unvalidated even if they share those operators. Unsupported operators,
batch/opset contracts and GAP shapes fail explicitly. The `--no-cle` acknowledgement
is required. CLE, YOLO, AdaRound, per-channel weights, INT32 bias and arbitrary scales
are outside the alpha's production scope. Probe mutations are experiments, not presets.

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
and uses the alpha's no-CLE recipe. Choose new output and log names on each run:

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

Direct Python commands print to the terminal; use the shell wrappers when collecting
repository evidence. Existing output models and sidecars are never overwritten.
Calibration checks free disk from inferred tensor sizes and removes its private spool
on normal completion or a Python exception. Hard process termination can leave a spool.
The historical `pipelines/resnet50/3c_quantize_own.py` entry point delegates to this CLI.

`--scales-from <reference.onnx>` selects position-table replay instead of independent
calibration. It is a parity/debugging mode, not needed to quantize from the float model.
It does not copy the reference's integer weights or topology.

## Verify an output

The optional Quark comparison requires a separately generated same-listing no-CLE
reference and its provenance file; the [producer experiment](../docs/BENCHMARKS.md#owned-resnet50-no-cle-re-emission-and-independent-calibration)
documents those commands. With that reference and labeled `data/eval/` available:

```bash
./scripts/quant-validate.sh --model models/resnet50_ignition_alpha.onnx --reference models/resnet50_quark_nocle_c64.onnx --tag resnet50_ignition_alpha_verify
```

This compares parameters, runs full labeled CPU/NPU evaluation, requires a clean
pre-run hardware-context check, uses fresh compilation and records the EP report.
For CPU-only verification add `--cpu-only`. Quark is needed only to generate the
optional oracle, not by Ignition or its inference commands.

Read [Alpha validation](../docs/BENCHMARKS.md#ignition-alpha-release-validation) for
the versioned artifact's evidence and [acceptance findings](../docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance)
for numerical traps. Successful EP placement does not establish correct outputs;
even an optimized CPU reference can disagree with unoptimized ONNX computation.
The no-CLE alpha does not claim the accuracy of the repository's CLE/AdaRound models.
