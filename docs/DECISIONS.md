# DECISIONS — settled choices, rejected approaches, and the YOLOv8 partitioning investigation

> Companion to `README.md` (results, quickstart) and `RESEARCH.md` (the thesis). This
> file is the record of *why*: things already tried and rejected, and the values
> that must not be silently changed. Update it when a decision changes, not when a
> number does.

## LOCKED DECISIONS (do not reopen)

1. **Ryzen AI 1.7.1 for NPU inference, not 1.8.0.** 1.8.0 SDK ships ZERO xclbins
   (`voe-4.0-win_amd64\` contains only `vaip_config.json`; recursive search of the
   install tree finds none; env site-packages only has
   `data\parts\xilinx\xclbin\strx\base.xclbin`). This is a packaging bug, not a
   documentation error: AMD's 1.8.0 docs are internally consistent and correctly say
   to use `phoenix\4x4.xclbin` for PHX/HPT INT8 CNN models — the installed package
   just doesn't contain that file (verified against the published 1.8.0 docs
   2026-09-06). The driver drops xclbins in `C:\Windows\System32\AMD\` (`1x4_*`,
   `4x4_*`, `5x4_*` = XDNA1; `AMD_AIE2P_*` = Strix) but 1.8's EP rejects them:
   `Cannot find or create target with fingerprint=0x080002050018eec1`
   (4x4_3.5.0.0-2352) / `0x0a000205001c8d4c` (4x4_3.5.0.0-2160_ipu_2). Not tried:
   other 4x4 variants, or grepping 1.8's DLLs for `AMD_AIE2_4x4_Overlay` — moot since
   1.7.1 works. Filed upstream: [amd/RyzenAI-SW#400](https://github.com/amd/RyzenAI-SW/issues/400).
2. **Provider options for PHX/HPT** (in `npu/session.py::build_session`): `cacheDir`,
   `cacheKey`, `enable_cache_file_io_in_mem: "0"`, `target: "X1"`,
   `xlnx_enable_py3_round: "0"`, `xclbin: <phoenix 4x4 path>`. camelCase
   `cacheDir`/`cacheKey` (tutorial form) is what's used; the docs table shows
   snake_case — both appear in AMD's own documentation. With no xclbin, the compiler
   defaults to `Target architecture: AMD_AIE2P_4x4_Overlay` (Strix) and dies at
   runtime: `DPU timeout ... Timeout layer
   name:[subgraph_(/conv1/Conv_output_0_vaip_3)] ... ERT_CMD_STATE_ERROR`.
   `target: X1` alone does NOT select chip arch; the xclbin does.
3. **X1 backend = XINT8 only.** Power-of-two scales, MinMSE calib, UINT8 activations /
   INT8 weights+bias. A8W8 (float scales) silently falls back to CPU (39 ms latency =
   CPU speed, accuracy 66.6%). A16W8 not tried, assumed dead on X1.
4. **Accuracy recovery = XINT8 + AdaRound**, i.e. `get_default_config("XINT8_ADAROUND")`.
   Recovered ResNet50 from 71.7% → 79.8% (FP32 80.1%). Config names verified:
   `get_default_config()` accepts `"XINT8"`, `"A8W8"`, `"A16W8"`, `"XINT8_ADAROUND"`,
   `"XINT8_ADAQUANT"`. Quark API: `from quark.onnx.quantization.config import Config,
   get_default_config; from quark.onnx import ModelQuantizer;
   ModelQuantizer(Config(global_quant_config=qc)).quantize_model(in, out, reader)`.
   Calibration reader subclasses `onnxruntime.quantization.CalibrationDataReader` with
   `get_next()`/`rewind()`.
5. **Export settings:** `torch.onnx.export(..., opset_version=17, dynamo=False)`,
   static batch 1, NO `dynamic_axes`. The dynamo exporter (default in torch ≥2.9)
   silently emits opset 18 even when asked for 17 (its version_converter assertion
   fails, only warns). Dynamic batch injects `Shape/Concat/Reshape` at the graph tail
   (CPU-fallback risk). Static export yields clean `GlobalAveragePool/Flatten/Gemm`.
   Ultralytics export: `YOLO(w).export(format="onnx", opset=17, imgsz=640,
   simplify=True, dynamic=False, batch=1)` — runs onnxslim, gives opset 17, input
   `images [1,3,640,640]`, output `output0 [1,84,8400]`. **Batch 1 is not just an
   export-simplicity choice** — measured: a STATIC (non-dynamic) batch-2 resnet50
   export drops the EP's partition from 393/395 to 80/395 nodes (Conv itself falls
   entirely to CPU; only stray Add/Relu stay on NPU), 12.6× the per-image latency
   (71.58 ms vs 5.68 ms), **and the pooled accuracy (36.0% top-1) turned out to be
   exactly half the batch-1 number (72.1%) for a specific, mechanistic reason, not
   generic corruption.** Splitting the accounting per within-batch position (`i=0`,
   `i=1`) shows: index 0 gets 72.00% top-1 — correct, matching batch-1 within noise.
   Index 1 gets 0.00% top-1, 0.40% top-5, and its raw logits are **byte-identical
   across every image tested** (same min/max/std regardless of input) — the second
   batch slot's output is a fixed, input-independent buffer, never actually written
   by the graph, not merely miscomputed. Pure `--ep cpu` on the same weights gets
   76.0% (both slots correct), so the model and export are fine — the bug is specific
   to VitisAI writing only slot 0 of a batch>1 output tensor. No error is raised
   anywhere; the run reports a plausible-looking latency and completes normally.
   **Never trust a batch>1 NPU result on this backend without an independent accuracy
   check** — it fails silently, not loudly, and it fails by not writing every batch
   element, not by miscomputing them. Filed upstream:
   [amd/RyzenAI-SW#401](https://github.com/amd/RyzenAI-SW/issues/401). A label-free,
   ImageNet-free reproduction lives at `tools/probe_batch_slot_write.py`: it reads the
   static batch size N off the graph (not assumed to be 2), fills every one of the N
   slots with independent random noise in two separate batches, and reports per slot
   whether the output actually changed — a slot identical across two draws of
   continuous random floats is a probability-zero coincidence, so "identical" means
   unwritten. Logged at `results/batch/slot_probe_b2.log` (2026-09-06): at batch 2 on
   the NPU EP, slot 0 changes (max abs diff 2.125) and slot 1 does not (max abs diff
   0.0, flagged STALE/UNWRITTEN); the CPU EP control on the same weights changes in
   both slots (2.5 and 1.625) with none flagged stale. **Only batch 2 has been run —
   batch 4 is untested.** The tool itself is not limited to 2 (point it at any static
   batch-N model to get a per-slot table for that N), but no batch-4 XINT8 export has
   been produced or tried yet, so whether batch 4 leaves only slot 0 live or some
   larger prefix is an open question, not an assumed generalization from batch 2.
6. **Preprocessing** must match calibration exactly. ResNet: timm
   `resolve_data_config` → `{'input_size':(3,224,224),'interpolation':'bicubic',
   'mean':(0.485,0.456,0.406),'std':(0.229,0.224,0.225),'crop_pct':0.95,
   'crop_mode':'center'}`; resize shorter side to `int(224/0.95)=235` (floor, matches
   timm), center crop 224, bicubic. YOLO: letterbox to 640 with gray 114 pad,
   BGR→RGB, /255, no mean/std.
7. **Always `--fresh` (delete cache dir) when changing model or xclbin.** The compile
   cache is keyed by a hardcoded `cacheKey`, not model hash. A stale cache means a
   wrong-arch artifact and confusing errors. Compile stats (`[Vitis AI EP] No. of
   Operators / Subgraphs`, `stat.cpp`) print ONLY during compile, not on cache load.
8. **ImageNet data source:** HF `ILSVRC/imagenet-1k` is parquet:
   `data/validation-{00000..00013}-of-00014.parquet` (~480MB each, ~3571 rows, random
   class order — one shard covers 974 classes). NOT `val_images.tar.gz` (old loader
   script, 404). Read with `pyarrow.parquet.ParquetFile.iter_batches`, write
   `image['bytes']` directly. `label` column is int class index (sorted-wnid order,
   matches timm). Never use the `datasets` lib (1GB+ RAM from Arrow + torch
   autoimport). Never `load_dataset` without streaming (150GB).

## Rejected approaches and known pitfalls

- **`4_detect.py`/`5_eval_map.py` used to time a cut model's numpy DFL/anchor decode as
  part of "infer."** For a head-cut model `forward` was `sess.run` + `decode_heads`
  timed as one block; a full-graph model's decode runs inside the ONNX graph, so its
  "infer" never carried this cost. The two "infer" numbers were never measuring the same
  thing, and the gap is not small: moving `--conf` from the demo's 0.25 to the eval
  script's 0.001 alone took the cut NPU model's single-image "infer" from 8.94ms to
  15.01ms with zero EP or hardware change. Fixed by splitting the timed region in both
  scripts: "infer" is `sess.run` alone (comparable across every EP and graph shape),
  decode moved into "post" alongside NMS. Found while building a fair CPU/DML/NPU
  comparison (see `README.md`'s "iGPU vs NPU" section) — a DML full-graph model would
  have been compared against an NPU cut model's inflated number, understating the NPU
  specifically because it needed the head-cut workaround DirectML doesn't.
- **NPU single-instance latency drifts session to session on this shared dev machine
  independent of any code or model change** — 12.7ms measured for
  `yolov8n_cut_xint8_c200.onnx` in isolation, 6.8-6.9ms measured minutes later
  back-to-back with a CPU/DML/NPU sweep, same model, same compile cache, nothing
  changed. Background CPU load from other processes on the machine is the suspect, not
  diagnosed further. Consequence for any future timing comparison: don't trust a
  latency number against one committed on a different day: capture every configuration
  being compared in one interleaved sitting.
- **DirectML on this iGPU (Radeon 780M) gains nothing from a plain QDQ INT8 model.**
  `yolov8n_cut_xint8_c200.onnx` on `DmlExecutionProvider`: 16.6ms, slower than the same
  graph shape's own FP32 (14.5ms) — DirectML has no dedicated INT8 fast path exercised
  by ORT's standard QDQ lowering here, so it just pays dequantize→float-compute→quantize
  overhead around the same math the FP32 model already does. Registering and "All nodes
  placed on [DmlExecutionProvider]" (`--log 0`, no CPU fallback) is not evidence of a
  speedup — check the number, not just whether it engaged. FP16 (via
  `onnxruntime.transformers.float16.convert_float_to_float16`, `keep_io_types=True`) is
  DML's real speed lever on this hardware: ~9.9-10.5ms, and mAP moved only 36.69 → 36.72
  on the full 5000-image set, i.e. free.
- **DML has no `vitisai_ep_report.json` equivalent.** "Did the EP actually take the
  graph" for a `_dml` result means a `--log 0` capture of ORT's own
  `VerifyEachNodeIsAssignedToAnEp` log line, not a report file — see
  `results/bench/diag_dml_node_placement.log`.

- Small language models on this NPU: unsupported by the vendor support matrix (no
  BF16, no NLP path on XDNA1). Scope redirected to CNN inference.
- Ryzen AI 1.8.0 for inference: no Phoenix xclbin (see locked #1).
- Driver-resolved firmware (no xclbin) on Phoenix: compiles for Strix, DPU timeout.
- Driver xclbins from `System32\AMD` with the 1.8 EP: fingerprint mismatch.
- A8W8 on X1: CPU fallback.
- `datasets` lib / `.shuffle(buffer_size=5000)` for ImageNet: 3.5GB RAM. Tar path
  `data/val_images.tar.gz`: 404 (stale).
- `echo %RYZEN_AI_INSTALLATION_PATH%` (cmd syntax) in PowerShell masks an unset var —
  it prints the literal string back instead of erroring. Use `$env:`.
- An early draft hardcoded the INT8 model in the eval script, so the FP32 baseline
  was silently skipped; caused a detour. Fixed with a `--model` flag.
- ResNet quantization note: Quark folds ReLU into the preceding Conv (ReLU absent
  from the quantized op table) — expected. This does NOT transfer to YOLO: SiLU is
  rewritten to `HardSigmoid`+`Mul`, not folded.
- **Quark's calibration cache filling the disk.** Measured: quantizing the 640×640
  YOLO model spools **~105 MB per calibration image for yolov8n, and ~198 MB for
  yolov8s** (6.18 GB for 32 images, measured directly) into
  `%TEMP%\quark_onnx.calib.*`. The rate tracks **activation width, not graph size** —
  n and s are both 929-node, depth-0.33 graphs and differ only in width — so a single
  constant is wrong. `scripts/lib.sh` exposes `calib_mb_per_image <variant>`; n and s
  are measured, m/l/x are deliberately pessimistic extrapolations, since
  over-estimating only makes the check refuse sooner. Using n's 105 for s
  understates a 300-image run by 25 GB, which was enough to fill the disk while
  `require_disk` still reported fine. `--limit 300` peaks around 31 GB, and it is
  **not** cleaned up if the process is killed. The space is released normally once
  MinMSE thresholding finishes, so this is a *peak* requirement, not a permanent one.
  ResNet at 224×224 is a small fraction, which is why `--limit 300` was always fine
  there. `scripts/yolo-cut.sh` now checks free space against
  `calib_mb_per_image "$VARIANT"` before starting and traps EXIT/INT/TERM to clean
  up; its default `--limit` is 100, and `--limit 32` is plenty for the only question
  that experiment asks (does the EP take the graph — calibration quality is
  irrelevant to that).
- **The same calibration-cache growth above, but at a resolution `calib_mb_per_image`
  doesn't cover.** Quantizing yolov8x re-exported at 1280² (double the repo's usual
  640, for the multi-partition achieved-TOPS ceiling check — see `RESEARCH.md`
  finding 5) with the repo's usual `--limit 64` spooled Quark's calibration cache to
  **169 GB across 64 images (~2.64 GB/image)** and drove this 32 GB machine's free
  RAM to 0.44 GB before the run had to be killed manually. `3b_quantize_cut.py` was
  run directly, the same documented gap as the disk-guard note above (`scripts/
  yolo-cut.sh`'s checks are skipped by hand). Per-image footprint at 1280² is roughly
  4× the 640² rate (matches the resolution-squared scaling the MACs also showed:
  524.1 GMACs vs. 131.1 GMACs, almost exactly 4.0×), so `calib_mb_per_image`'s
  n/s-at-640 figures cannot be reused unscaled at a different resolution. Fixed here
  with `--limit 4` — this specific model was built to test a throughput ceiling, not
  to report accuracy, so a thin calibration set costs nothing real; it would be the
  wrong fix for a model whose accuracy will actually be cited.
- **Confirms the per-image footprint above scales with model width too, not only
  resolution.** `yolov8s` re-exported at the same 1280² (filling the achieved-TOPS
  gap between yolov8m and yolov8l — `RESEARCH.md` finding 5) at the same `--limit 4`
  spooled only **~3 GB total** (~0.77 GB/image) — `yolov8s`'s activations are far
  smaller than `yolov8x`'s at matched resolution, since it is both narrower and
  shallower (63 Conv nodes vs. `yolov8x`'s 103). Same `--limit 4` fix, no incident this
  time; cited here as the confirming data point, not a new pitfall. This calibration
  choice also produced a real, separate accuracy defect (not a RAM issue) — see
  `RESEARCH.md` finding 5's cross-talk-oracle write-up: the thin calibration set was
  too thin for this narrower architecture, confirmed via `--ep cpu` comparison, and
  the resulting model must never be cited for mAP.
- **AdaRound running out of RAM, and taking SIGSEGV instead of raising.** Measured:
  `--adaround --limit 300` succeeded, then `--limit 200` segfaulted twice at exactly
  the same point with another memory-heavy app resident. *Less* calibration data
  failing where more had succeeded is the confusing part, and it is not a Quark bug —
  the peak is **layer 0**, the only layer at full 640×640 resolution, where
  FastFinetune holds the float and quantized activations for `DataSize` samples at
  once. A run that clears layer 0 will finish; one that does not dies with the log
  ending at `Module (images)->(/model.0/conv/Conv_output_0) will be optimized by
  adaround on cpu` and no traceback. Free the memory rather than lowering `--limit`,
  which does not help in the direction you expect. `scripts/yolo-bench.sh` warns
  below 6 GB free and no longer lets one crashed model abort the rest of a matrix.
- Grepping `results/npu_log.txt` with plain `grep`/`rg`: the file is UTF-16
  (PowerShell `*>`), so every ASCII pattern misses and you conclude a line is absent
  when it is there. Decode first (`open(p,'rb').read().decode('utf-16')`), or use
  `Select-String -Path f -SimpleMatch`, which handles it. Logs written by
  `scripts/*.sh` are UTF-8 and safe.
- `conda run` as a substitute for `conda activate`: it cannot take a multi-line
  `python -c` (`NotImplementedError`), and it crashes on non-ASCII child stdout
  (cp1252 `UnicodeEncodeError`), exiting 1 *after* the script has already succeeded.
  Check for the output file before believing it failed. Write a temporary script and
  use `conda activate`, or prefix `PYTHONIOENCODING=utf-8`.
- Reading Quark's `custom_ops.dll does NOT exist` / `where cl` warnings on import as
  a failure. They are harmless; every quantization log in `results/` carries them.
- Reading `deviceStat[].supportedOpType` as "ops this device supports": it is the set
  of ops *assigned* to that device. The report names no reasons at all.
- Assuming a missing compile log means the EP never ran. The EP writes
  `<cacheKey>/vitisai_ep_report.json` on every session build; the compile log only
  prints on a real compile. Check the report, and check for `compiled.*.xmodel`.
- An early draft of the cut-model quantize script set `extra_options["FastFinetune"]`
  without `qc.include_fast_ft = True`, which would have made `--adaround` a silent
  no-op. Fixed in `3b_quantize_cut.py`. Watch for this whenever AdaRound is wired up.
- **Static batch >1 on this EP: not a speed tradeoff, and a more specific bug than
  "correctness issue."** Tested resnet50 at a genuinely static (no `dynamic_axes`)
  batch of 2 to check whether "batch 1 only" (locked #5) was measured or assumed.
  It's now measured, and the failure mode is precise: the EP accepts a *fragment* of
  the graph (80/395 nodes, Conv itself entirely CPU), and **only the first batch
  slot's output is ever written** — index 0 scores 72.00% top-1 (correct, matches
  batch-1), index 1 scores ~0% and its raw logits are identical across every input
  image tried, i.e. a stale/uninitialized buffer, not a miscomputation. The pooled
  number (36.0%) is deceptively close to "half of everything a bit wrong," which
  reads like generic corruption until you split it per batch index — do that split
  before describing any batch>1 failure as "corruption" rather than "unwritten."
  This is the same class of danger as A8W8's silent CPU fallback, but worse: A8W8
  falls back to CPU cleanly and correctly, batch>1 fills only slot 0 and leaves the
  rest stale. Do not ship a batch>1 config on this backend without comparing against
  a `--ep cpu` run on the exact same weights, split per batch index.
- Quark 0.12 forward-looking risk: it also ships a *new* `config.QConfig` whose
  equivalent of `subgraphs_to_exclude` is called `exclude` and takes node names and
  subgraph tuples in one list. `get_default_config` still returns the legacy
  dataclass today, but if a future Ryzen AI bumps Quark and it starts returning
  `QConfig`, every `qc.<attr> = ...` assignment becomes a silent no-op.
  `3b_quantize_cut.py` type-checks for exactly this.
- **The 5th physical AIE column on this Phoenix chip cannot be reached through this
  repo's tooling.** `xrt-smi examine -r platform` reports **Total Columns: 5** here
  (Desktop 2), not the 4 every `tools/multi_partition_bench.py` sweep has assumed.
  Extending that tool to `--procs 1 2 3 4 5` against the existing `1x4.xclbin` shows
  why: `xrt-smi examine -r aie-partitions` during the n=5 window still reports only
  **4** distinct partitions (columns 1–4); the 5th and 4th process share one partition
  and both drop to ~38 fps (half of ~61 fps solo) while columns 1–3 stay full-rate —
  `1x4.xclbin` itself exposes only 4 of the 5 columns as independent contexts, so a
  5th process just gets time-sliced onto column 4 rather than a genuine 5th column.
  Combined throughput at n=5 (257.2 fps) barely improves on n=4 (246.4 fps) for
  exactly this reason.
  Tried the obvious next step — the driver's own 5-column overlay family
  (`C:\Windows\System32\AMD\5x4_3.5.0.0-799.xclbin`, `-950.xclbin`,
  `5x4_3.5.0.050-651.xclbin`; none shipped in the 1.7.1 SDK's own xclbins folder) —
  and **all three build without error and produce numerically correct output that
  matches CPU, but run 100% on CPU.** `vitisai_ep_report.json`'s `deviceStat` for
  every one of them has no `DPU` entry at all (only `CPU` and `VITIS_EP_CPU`), and
  session build logs a glog `F`-severity line, `target_factory.cpp:161] Cannot find
  or create target with fingerprint=0x...`, for each — a hardware-target lookup that
  fails and silently falls through to full CPU rather than raising. This is the exact
  "CPU run in an NPU costume" failure `npu.session.resolve_xclbin`'s docstring warns
  about, just from a different cause (xclbin/firmware fingerprint mismatch, not a
  missing xclbin) — **matching CPU output is not evidence of NPU execution; check
  `deviceStat` for a `DPU` entry before trusting any result from an unfamiliar
  xclbin.** Conclusion: the 5th column is real but not reachable from this machine's
  1.7.1 install with any xclbin tried so far. Getting to it would need either a
  driver/firmware revision this repo doesn't have, or the separate ahead-of-time
  AIE-compiler flow (`vaitrace`'s `-cp`/`vaiml` flags) rather than the lightweight
  ORT/VitisAI-EP JIT-compile path every pipeline here uses — out of scope unless that
  flow gets set up deliberately. Desktop 2 / Phoenix.

## The YOLOv8 partitioning failure (resolved)

For a while, YOLOv8 would not reach the NPU at all: the VitisAI EP claimed **zero**
nodes and silently fell back to CPU while still reporting "NPU" in the logs. **The
actual finding here is that failure mode, not the fix that followed it** — a handful
of unsupported float ops (DFL/anchor decode) caused the EP to reject the *entire*
965-node graph wholesale rather than partitioning around the unsupported tail and
running the rest, which is the more common and far less surprising failure shape.
Moving decode off-graph is not itself novel: cutting post-processing out of a
detector and running it in inference code is the standard move for edge NPU
toolchains generally (Rockchip's RKNN and Hailo's compiler both require the same
pattern for YOLO). What's specific to this backend is that skipping the cut doesn't
degrade gracefully into a partial partition — it's all-or-nothing.

Cutting the 18-node DFL/anchor tail out of the ONNX graph and doing it in numpy
takes the VitisAI EP from claiming **zero** nodes to claiming **922 of 929**, and
yolov8n from 39.1 ms (CPU in disguise) to **8.7–9.8 ms on the NPU**, ~110 fps. Five
runs, each with `--fresh` so each recompiled, spanned 8.73–9.78 ms.

```
                        yolov8n_xint8.onnx      yolov8n_cut_xint8.onnx
  deviceStat  all              965                     929
              NPU        (absent)                      922      <-- 99.2%
              CPU              298                       -
              VITIS_EP_CPU     667                       7
  compiled.*.xmodel      not present              present
  inference              39.1 ms                  8.7-9.8 ms
```

The 7 non-NPU nodes are exactly the boundary conversions — the input `QuantizeLinear`
and the six output `DequantizeLinear`s — structurally identical to ResNet50's 2.

### How it was found

`<cacheKey>/vitisai_ep_report.json` — the Operator Assignment Report, a documented
AMD feature (auto-generation became opt-in rather than automatic as of Ryzen AI 1.5;
`XLNX_ONNX_EP_REPORT_FILE` is what turns it back on). It's real, but nothing in the
tooling points here at the moment a graph is silently rejected, and neither AMD's
CNN tutorial nor the failure path in the EP's own logging says "check this file" —
that gap, not the file's existence, is what cost the time here. The EP writes it on
**every** session build, including a cache hit, which makes it far better evidence
than the compile log (which prints only on a real compile). `tools/diag_ep.py
--cache-key` reads it; `scripts/*.sh` call `npu_verdict` to summarise it in one line.

The report showed no `NPU` entry in `deviceStat` at all and all 965 nodes marked
`device: "CPU"` — wholesale rejection, not bad partitioning. That reframed the
question from "why is the partition bad" to "why is the graph refused outright",
which is what made the head cut the obvious next move rather than one option among
four.

Caveat on reading the report: `deviceStat[].supportedOpType` is the set of ops
*assigned to* that device, not a capability list, and there is no reason field
anywhere. It tells you what landed where, never why.

### What it was NOT — all four checked, not assumed

- **Not the SiLU rewrite.** This was the leading hypothesis and it was wrong. Quark's
  `enable_npu_cnn` turns `Sigmoid`+`Mul` into `HardSigmoid`+`Mul` (57×) and lowers
  `Split` (8) to `Slice` (16), none of which appear in the ResNet50 graph that
  compiles. The cut model's report puts **all 57 `HardSigmoid`, all 16 `Slice`, both
  `Resize` and all 13 `Concat` on the NPU**. Every one is supported by the X1
  overlay. Do not re-run this hypothesis.
- **Not `subgraphs_to_exclude`.** Real field on the legacy `QuantizationConfig`,
  consumed by `quantize.py:391`, and `quant_utils.py::match_subgraphs` *raises*
  rather than no-ops if the subgraph doesn't match. It either works or it throws; it
  worked.
- **Not a missing xclbin.** The captured NPU log showed the correct 1.7.1 Phoenix 4x4
  xclbin on the blocked run. (`npu/session.py` was hardened anyway: `resolve_xclbin`
  raises instead of returning None, and `build_session` asserts the EP registered. It
  previously warned and continued, which is a CPU run in an NPU costume.)
- **Not ORT's `ConstantSharing`.** The working ResNet report shows the same merged
  scalar initializers.

### The numbers

| configuration | device | infer | post |
|---|---|---|---|
| full graph FP32 | CPU | 37.0 ms | 1.62 ms |
| head-cut FP32 | CPU | 31.8 ms | 0.12 ms |
| full graph XINT8 | "NPU" (was CPU) | 39.1 ms | — |
| **head-cut XINT8** | **NPU** | **8.7-9.8 ms** | **0.16 ms** |

Decode parity of `npu/yolo_decode.py` against the full graph's `output0`, FP32 on
CPU: box coords max abs diff **1.22e-4 px**, class scores **1.39e-7**, detections
identical to the integer pixel. The `conf_thres` prefilter — which selects on class
logits before the DFL softmax, valid because sigmoid is monotonic — gives
byte-identical detections and takes post from 1.62 ms to 0.12 ms.

**That first model was not shippable.** It was calibrated on 32 images, because that
run existed only to answer whether the EP accepts the graph, which calibration
quality does not affect. Requantizing with `--limit 300 --adaround` before measuring
accuracy is what produced the numbers in `README.md`.

### Stability

Six NPU runs of the cut model, each with `--fresh` so each recompiled: five clean at
8.73–9.78 ms, one hard crash (Windows fatal exception, C-level stack, no Python
traceback) that did not reproduce in the three consecutive runs after it. Not
diagnosed; noted so a recurrence doesn't read as a new defect. If it comes back,
capture with `--log 0` and check whether it lands during session teardown.

### Reproducing

`./scripts/yolo-cut.sh` does the whole thing: cut → quantize → CPU sanity gate → NPU
run → read the report back → print the verdict. `./scripts/diag.sh` shows all three
caches.
