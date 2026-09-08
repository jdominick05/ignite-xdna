# Handoff: small, self-contained measurement tasks

Not part of the doc contract (`README.md`/`RESEARCH.md`/`docs/`) — this file is a scratch
worklist, untracked by git, safe to delete once picked through. Read `CLAUDE.md` first;
everything below assumes its invariants (measured numbers only, `--fresh` on model/xclbin
change, UTF-8 logs with the profile path scrubbed, fold a new log into the docs in the
*same* commit, `python tools/check_links.py` exit 0 before any doc commit).

**Out of scope — do not touch:** anything int64-related (Fable is on it), and
`docs/SILICON.md` / any objective in it (K0-K5, A1-A4, S0-S3, D1-D4, C1-C2) — also Fable's.
If a task below turns out to brush against either, stop and leave it.

**Machine check first.** `CLAUDE.md`'s "Task routing" table governs where each of these
can run — most need an XDNA1 device (Laptop or Desktop 2), none need Desktop 1's GPU.

Each item below follows the same shape as the AdaRound-latency-cost item just closed
(`8ef3c62`): pick one, run it, read the log before writing anything, fold the result into
`docs/BENCHMARKS.md` (full working) + close/open the matching line in `RESEARCH.md`'s
Closed/Still-open lists + `README.md` only if it changes the 60-second summary + add the
log to `results/README.md`'s index — all in one commit, `--session-url` set, never `git push`
without asking first (see `CLAUDE.md`, "Sync at the edges of a session").

## 1. AdaRound across the ResNet50 resolution sweep

`RESEARCH.md` "Still open": the resolution sweep (`docs/BENCHMARKS.md#resnet50-input-resolution...`)
is plain XINT8 at every size; AdaRound's recovery could move where the accuracy peak sits
(currently 256² at 74.00%, per that section).

No AdaRound model exists yet at any non-224² resolution — confirmed via
`ls models/*.onnx | grep -iE "r[0-9]+.*adaround"` (empty). 224² already has one
(`models/resnet50_xint8_adaround.onnx`, just re-measured in `8ef3c62`).

Smallest useful slice: quantize+AdaRound at the two resolutions bracketing the current
peak (224² and 288²) rather than redoing all eight. `scripts/resnet-res.sh` built the
FP32 base models and plain-XINT8 quant per resolution already
(`models/resnet50_r*_fp32.onnx`, `models/resnet50_r*_xint8_c64.onnx`); reuse the FP32 base
as `--in-model` and add AdaRound directly, mirroring how `scripts/setup.sh` builds the 224²
AdaRound model:

```
python pipelines/resnet50/3_quantize.py --calib-dir data/calib --config XINT8_ADAROUND \
    --in-model models/resnet50_r288_fp32.onnx --cfg-path models/preprocess_config_r288.json \
    --out models/resnet50_r288_xint8_adaround.onnx
```

`models/preprocess_config_r288.json` already exists from the original sweep, same for every
other resolution's cfg (`ls models/preprocess_config_r*.json`). Then `4_run.py --ep npu
--fresh` on 1000 images, same as every other row in that table.

## 2. yolov8m mAP row at calibration 200

`RESEARCH.md` "Still open": the detection width table's current yolov8m row (43.49 mAP)
was calibrated on only 64 images, unlike the rest of the table; l (32) and x (24) have the
same caveat, cheapest first.

```
./scripts/yolo-bench.sh --variants "m l x" --calib 200
```

m at `--calib 200` needs ~120 GB of `%TEMP%` spool per the script's own header comment —
Desktop 1 or Desktop 2 only, not the laptop (`CLAUDE.md`'s machine table). `mAP over all
5000` is the default (matches `--n 0`); don't shrink it — see `CLAUDE.md`'s "A 500-image
slice is not the answer" invariant, yolov8s already got burned by exactly this (45.19 slice
vs 39.98 full).

## 3. AdaRound for pose

`RESEARCH.md` "Still open", called out as "the obvious next lever." Confirmed untried —
`models/yolov8n-pose_cut_xint8.onnx` exists, no `_adaround` sibling.

```
python pipelines/yolov8n-pose/3b_quantize_cut.py --calib-dir data/coco_calib --adaround
./scripts/pose-eval.sh --n 5000 --model models/yolov8n-pose_cut_xint8_adaround.onnx
```

Compare against the existing cut-XINT8 pose row in `docs/BENCHMARKS.md`. Full 5000 for the
number that goes in the docs, same rule as above.

## 4. Webcam path end to end (needs a person at the machine)

`RESEARCH.md` "Still open": `./scripts/yolo-demo.sh`'s single `4x4.xclbin` session has
never been exercised end to end — distinct from the round-robin 4-column demo, which
*has* been measured (`docs/BENCHMARKS.md#a-live-demo...`). This one needs an actual webcam
and a human to eyeball it, not something to run unattended; pick this one only if a person
is at the NPU machine anyway.

---

Not included here on purpose: "column count for the int8 conv kernels" and "what the array
physically is" are both `docs/SILICON.md` objectives (K1, S0) — Fable's.
"Longer term: a fixed-camera detector" is a real project, not a small task — worth a
separate conversation with the user before anyone starts it.
