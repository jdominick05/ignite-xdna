# Contributing

This repository is a hardware-characterization study of AMD's XDNA1 NPU: what the
device can actually run, measured empirically, because the vendor documentation is thin
and occasionally wrong. It is not a maintained tool with a feature roadmap.
Contributions are welcome, particularly measurements on hardware configurations that
were not available to the original work.

## Ground rules

- **Measure, don't assume.** Every claim in `README.md`, `RESEARCH.md` and
  `docs/BENCHMARKS.md` is backed by a logged run under `results/`. If you change a number, change it because you ran
  something, and say what you ran.
- Keep changes focused: one logical change per merge request.
- Prefer simple, explicit code over abstraction. There is no framework here on purpose.
  Entry-point scripts (`1_export.py`, `2_fetch_*.py`, ...) start with a digit so they
  cannot be imported, which is what forces anything two scripts share into `npu/`.

## Getting set up

You need real Ryzen AI hardware (Hawk Point or Phoenix, XDNA1) to run anything past
export. There is no simulator, and CPU-only runs do not exercise what this repository is
about. See [Compatibility](docs/SETUP.md#compatibility) before assuming your machine
qualifies.

```powershell
conda create -n resnet_env   --clone ryzen-ai-1.8.0
conda create -n resnet_env17 --clone ryzen-ai-1.7.1
```

Two environments, not one, because the SDK version that can export and quantize is not
the one that can run on the NPU. See [Environments](docs/SETUP.md#environments) for the
reasoning; do not try to collapse them into one.

### What to read before changing anything

- **`docs/DECISIONS.md`**, in particular its "LOCKED DECISIONS" and "Rejected
  approaches and known pitfalls" sections. Several plausible-looking changes (dynamic
  batch axes, A8W8 quantization, Ryzen AI 1.8.0 for inference, a smaller calibration set
  to fix an AdaRound crash) have already been tried here and measured to fail, with the
  measurement recorded. Reopening one of these without new evidence repeats the
  debugging time it already cost.
- **`RESEARCH.md`** for the one-level-up framing: what question this project is
  answering and why, before diving into a specific pipeline.

## Making changes

- **Preprocessing must stay byte-identical between calibration and inference.** It lives
  in exactly one place per pipeline (`npu/preprocess.py`, `npu/yolo.py`) for this reason.
  Never inline a second copy, even a supposedly equivalent one.
- **Input resolution comes from the model, never from a flag.** `npu.yolo.input_size()`
  reads the NCHW input shape off the graph, and calibration, inference and evaluation all
  letterbox to that number. A `--size` flag that anyone can forget is exactly how
  calibration and inference drift apart.
- **Nothing under `npu/` may import Quark.** The inference scripts import that package on
  every run, and importing Quark triggers a custom-op build attempt each time. This is
  what keeps NPU inference fast to start up.
- **`npu.session.resolve_xclbin` must keep raising when no xclbin is found**, not warn
  and continue. A driver-resolved run silently targets the wrong chip architecture and
  fails with a confusing DPU timeout instead of a clear error. This was a real incident,
  not a hypothetical one.
- **Pass `--fresh` whenever the model or xclbin changes.** The compile cache is keyed by
  a hardcoded `cacheKey`, never by model hash, so a stale cache silently reuses a wrong
  artifact.
- **NMS is per-class** (`cv2.dnn.NMSBoxesBatched`). Class-agnostic NMS lets an
  overlapping person and chair suppress each other and silently costs mAP; `agnostic=True`
  is opt-in.
- **Evaluate mAP at conf 0.001, IoU 0.7, max_det 300**, not the demo's 0.25. Average
  precision integrates the whole precision-recall curve, so a high threshold deletes the
  tail the metric needs. Evaluation latency is therefore not comparable to demo latency.
- Match the surrounding style: entry points take `argparse` flags with sane defaults,
  shared logic lives in `npu/`, and shell wrappers in `scripts/` handle environment
  activation, logging, and disk guards so the underlying Python scripts stay runnable
  standalone too.

## Testing

There is no unit-test suite. There is no way to fake this hardware, and a mocked NPU
session would test nothing this project cares about. Verification is empirical instead:

```bash
python -m compileall -q npu pipelines tools kernels quant \
  && for s in scripts/*.sh; do bash -n "$s" || exit 1; done \
  && python -c "import npu.preprocess, npu.yolo, npu.yolo_decode, npu.yolo_pose, npu.yolo_pose_decode, npu.session, npu.ep_report, npu.paths, npu.modnet, npu.yolov6, npu.yolov6_decode, quant, quant.graph, quant.pow2, quant.sources, quant.calib, quant.qdq, quant.passes, quant.refine, quant.verify, quant.quantize, quant.probe" \
  && echo "PIPELINE CHECKS PASS"
```

Run this (in `resnet_env17`) after touching anything under `npu/`, `pipelines/`,
`tools/`, `scripts/`, `quant/`, or `kernels/` (`kernels/` only gets the syntax pass here — its
designs import mlir-aie, which lives in a different env; see `kernels/README.md`). It
only catches syntax and import breakage; it is **not**
evidence that a behavioural claim is true. If your change could affect latency,
accuracy, or whether the VitisAI EP accepts a graph, run the relevant pipeline and read
the log in `results/` before describing the change as verified. Never report a number
you did not measure.

Every new `npu/` or `quant/` module joins the import list here and in the local
`CLAUDE.md`. `quant/` must never import Quark, and `npu/` must never import `quant/`.
The owned quantizer's core also imports in `resnet_env`; torch belongs only in its
future AdaRound module. See [`quant/DESIGN.md`](quant/DESIGN.md) for the ordered
comparison gates; passing imports does not validate a quantized model.

## Submitting a change

1. Create a branch for your change.
2. Open a merge request with a clear description: what changed, why, and what you ran to
   verify it. Paste the relevant `results/` output or log line, not just "it works".
3. If your change touches a locked decision or a documented finding, update
   `docs/DECISIONS.md`, `RESEARCH.md`, `docs/BENCHMARKS.md` or `README.md` in the same
   change. A document that contradicts the code misleads the next person to read it.
4. State explicitly what was tested and what was not. Untested code handed over as
   verified has caused real errors in this project; a merge request that says "steps 1
   and 2 were run on hardware, step 3 was not" is more useful than one that claims
   everything works.

Contributions may be produced with whatever tooling you prefer, including AI coding
assistants. The same standard applies regardless of how a change was written: the
submitter is accountable for understanding it, and every measurement claim must come
from a run on real hardware.

## Licensing

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE)
(AGPL-3.0), chosen because `pipelines/yolov8n/1_export.py` imports `ultralytics`
directly, and Ultralytics' own YOLOv8 code is AGPL-3.0; matching that license here
avoids leaving an unresolved copyleft ambiguity between this repository and a
dependency it imports at runtime. By contributing, you agree your contributions are
licensed under the same terms. Note the license covers the pipeline code only. The
ResNet50, YOLOv8, ImageNet, and COCO artifacts it uses carry their own upstream licenses
and are never redistributed by this repository.

## Reporting bugs and ideas

Open an issue. If you have reproduced a finding differently than this repository reports
it (a different latency, a different partition, a different accuracy number), that is
especially useful. Include your Ryzen AI SDK version, chip (Hawk Point or Phoenix), and
the relevant `results/` log or `vitisai_ep_report.json`.
