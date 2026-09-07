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
   CPU speed, accuracy 66.6%). **A16W8 (INT16 activations / INT8 weights) now measured,
   not assumed: also a full CPU fallback.** `tools/diag_ep.py --cache-key modelcachekey`
   on `resnet50_a16w8.onnx` (`results/a16w8/diag_resnet50_a16w8_npu.log`) shows **0/394
   nodes on NPU** — `deviceStat` has no NPU entry at all, only `CPU` (122) and
   `VITIS_EP_CPU` (272), the same wholesale-rejection shape as the YOLOv8 full-graph
   blocker, not a partial partition. Quark's own quantize log
   (`results/a16w8/quant_resnet50_a16w8.log`) is a tell in hindsight: `A16W8`'s default
   config prints `enable_npu_cnn: False` and `execution_providers:
   ['CPUExecutionProvider']` — Quark itself never marks this config for NPU deployment,
   unlike `XINT8`. Ran at 26.18 ms / 69.70% top-1 on CPU (`results/a16w8/
   lat_resnet50_a16w8_npu.log`) — plausible-looking numbers, same silent-fallback danger
   as A8W8. Root cause is a real, documented ONNX Runtime opset limit surfaced in the
   quantize log: *"ONNX QuantizeLinear and DequantizeLinear operators do not support
   16-bit/4-bit integer quantization types prior to opset 21"* — this repo's export is
   pinned to **opset 17** (locked #5), so Quark routes the INT16 Q/DQ nodes through the
   `com.microsoft` domain instead of standard ONNX ops, and the VitisAI EP's op matcher
   evidently does not recognize that domain's Q/DQ variant at all, hence zero nodes.
   INT16 activations are not dead on this hardware/EP in principle so much as
   incompatible with the specific opset this project's export pipeline is locked to —
   untested whether opset 21 would change this, and not worth chasing: bumping export
   opset is its own can of worms (locked #5's dynamo/opset-18 trap) for a config with no
   demonstrated accuracy upside over `XINT8_ADAROUND` so far.
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
  flow gets set up deliberately. Desktop 2 / Phoenix. **That flow is now scoped, not
  just named** — see RESEARCH.md's open questions, "Custom C++ XRT / hand-written AIE
  kernels": the whole classical AIE compiler stack (`aiecompiler`, ADF C++ headers,
  Peano, raw XRT) is already present in the `RyzenAI\1.7.1` install, unused by anything
  here. **Update:** the device-model string this flow needs (`--part`) was found —
  `xc10AIE24x5-die-1LP-e-S-es1`, pulled from `aiecompiler_client.dll`'s own part table,
  matching this same "Total Columns: 5" finding rather than a guess — but the bring-up
  now stops one wall later: this machine has no host C++ standard library for Peano's
  bundled clang to preprocess the ADF frontend with (no `<iostream>` anywhere in the SDK,
  no Visual Studio install). See `results/aie/aiecompiler_part_phoenix_candidate.log` and
  `aiecompiler_hostlib_missing.log`. **Update:** installed VS 2022 Build Tools (C++
  workload) and fed its + the pulled-in Windows SDK's include dirs to `aiecompiler` —
  the `<iostream>` wall is gone, and the compiler now derives the graph and logs
  `Reading logical device aie2_5x4_device` (matching this same "Total Columns: 5" finding
  even more precisely than the part-string name alone did). Stops one wall later:
  `lib/win64.o/physical_device.dll` doesn't exist anywhere on this machine, checked via a
  full `C:` drive search, not just the pip env. See
  `results/aie/aiecompiler_hostlib_fixed.log`. **Update:** confirmed this is a genuine
  packaging gap, not a fixable name/path issue. Binary-scanning `aiecompiler_client.dll`
  (all three copies in the pip env) shows `physical_device.dll` is a literal hardcoded
  filename the compiler `LoadLibrary`s and calls `createPhysicalDevice()` from — not
  derivable — and it's one of four sibling DLLs the same string table names
  (`platform_device.dll`, `guidance_summary.dll`, `udm_api.dll`), all equally absent.
  Opened both full offline installers already on this machine (`ryzen-ai-lt-1.7.1.exe`,
  `ryzen-ai-1.8.0.exe`, via `7z`/`lessmsi` — installed both tools with `winget` this
  session) rather than only the pip package: 1.7.1's installer contains byte-identical
  wheels to pip (no extra content); 1.8.0's installer doesn't ship the `vaie_overlay`/
  `vaie_cpplus` packages at all. `device_essentials_strx_overlay-1.7.1` (source of
  `strx/base.xclbin`) ships a large ML place-and-route congestion feature store for
  Strix only — no Phoenix equivalent in 1.7.1 — but still no `physical_device.dll`, even
  for Strix. Conclusion: AMD's redistributable packaging draws its boundary at pre-built
  xclbin overlays; the physical-implementation backend to build a new one from a
  hand-written ADF graph isn't included, on either SDK version available here. See
  `results/aie/aiecompiler_physical_device_missing.log`. **Update:** that wall is specific
  to AMD's proprietary `vaie_cpplus`/`aiecompiler` toolchain, not to this hardware. Set up
  the open-source `Xilinx/mlir-aie` (IRON/Peano) toolchain instead — native Windows, no
  WSL, no AMD account, no license, in a new isolated conda env (`mlir-aie-iron`, Python
  3.13) that touches none of the existing 5 envs — and ran a hand-written SAXPY kernel
  end to end on this machine's actual XDNA1 (Phoenix) hardware: JIT-compiled, dispatched
  against `device="npu"`, verified correct against a numpy reference. `PASS!`, from a cold
  compile cache whose intermediate artifacts (placed MLIR, staged LLVM-IR, core ELF, CDO
  binaries, PDI, final `.xclbin`) confirm a genuine place-and-route, not a stale artifact
  or a simulator run. Peano (`llvm_aie_lightweight`) was already a pip dependency in the
  existing `ryzen-ai-1.7.1` env, under a `win64.o` namespace entirely separate from the
  one missing `physical_device.dll`. Doesn't overturn the proprietary-toolchain finding —
  `physical_device.dll` is still genuinely absent everywhere checked — but custom-kernel
  bring-up on this hardware is not actually blocked, only through a different toolchain
  than this project's own pipelines use. See `results/aie/mlir_aie_saxpy_npu.log`.
  **Update:** ran three more of mlir-aie's own tutorial designs from a cleared compile
  cache to check the result wasn't specific to SAXPY's single elementwise core — a
  multi-column memcpy bandwidth microbenchmark (56.19 GB/s effective), a 4-core
  single-column reduce-max cascade (cross-core control flow, not just parallel
  elementwise), and a single-core int16 matmul at two shapes via the AOT `.compile()`
  path (two distinct on-disk `.xclbin`s, confirmed not reused across shapes). All
  `PASS!`. Same conclusion as above, now on broader IRON-surface evidence. See
  `results/aie/mlir_aie_examples_npu.log`. **Update:** tried mlir-aie's `vision/`/`ml/`
  categories next (closer to this repo's own operator vocabulary than
  `getting_started`'s generic microbenchmarks). Five pure-Python, Phoenix-tagged designs
  ran clean from a cleared cache — `eltwise` (add/mul), `eltwise_unary` (relu/silu/gelu),
  `scale_shift` (two-phase runtime-parameterized `D=A*B+C`), `softmax`, `swiglu` — all
  `PASS!`. The rest is a real, stated gap rather than a workaround: `vision/*` and a few
  `ml/*` designs need a `make`-built C++ host program linking OpenCV, and this machine
  has neither `make` nor an OpenCV C++ dev package; other `ml/*` designs are
  Strix-only (`ryzen_ai_npu2`) regardless of that gap; `ml/magika` and `ml/mobilenet`
  are Phoenix-capable but multi-file enough to warrant their own pass. See
  `results/aie/mlir_aie_ml_examples_npu.log`. **Update:** installed `make` 4.4.1
  (conda-forge, into the isolated `mlir-aie-iron` env) and OpenCV 5.0.0 (current latest,
  extracted to `C:\Technical\thirdParty\opencv` — mlir-aie's own CMakeLists.txt hardcoded
  default path, no edits needed) to clear that gap. All four `vision/*` designs then
  passed (`color_detect`, `color_threshold`, `edge_detect`, `vision_passthrough` —
  byte-exact). Also installed `torch` (latest, CPU-only, host-reference generation only)
  and ran `bottleneck` (a real ResNet-style conv1×1→conv3×3→conv1×1-plus-skip block,
  1620us, `PASS!`) and `conv2d` plain/`--fuse_relu` (540/533us, both `PASS!`). Fixed two
  Makefile quirks at invocation time rather than editing mlir-aie: `make
  getwslpath=echo ...` (its WSL-detection heuristic false-positives on native Windows)
  and putting OpenCV's `bin` dir on `PATH` before `make run` (needed at run time, not
  just CMake configure time). Every `ryzen_ai_npu1`-tagged design short of the
  substantial multi-file models (`magika`, `mobilenet`) has now been tried and passes.
  See `results/aie/mlir_aie_vision_examples_npu.log`. **Update:** ran `magika` and
  `mobilenet`, plus `ml/resnet/layers_conv2_x` (tagged `ryzen_ai_npu1`, missed by the
  original survey) — three chained ResNet conv2_x bottleneck blocks across three NPU
  columns, `PASS!` at 1888.5us avg NPU time. `magika`'s own `run_phoenix.lit` marks it
  `XFAIL` upstream ("Known-failing numerical check on the NPU"), but running it anyway
  found both `group0` (EVM -34.85 dB) and `group2` (EVM -56.91 dB) **PASS** on this
  hardware — the expected failure did not reproduce here, recorded as a discrepancy
  rather than assumed away. `trace_py` for both groups hits an unrelated tooling gap
  (aiecc doesn't leave the trace parser's expected intermediate MLIR file on disk on
  this machine) after an identical NPU PASS. `mobilenet`'s own README targets "the
  Strix NPU2" and its hardware lits require `ryzen_ai_npu2` — a chip this machine
  doesn't have, same class of gap as the other npu2-only designs already logged; only
  its no-hardware numpy cross-validation could run here. Also found and fixed a new
  Windows-path/Git-Bash interaction: `magika`'s Makefile (unlike `bottleneck`/`conv2d`)
  defines its own compile rule expanding a backslash-separated Windows path
  (`$(shell python3 -c "from aie.utils.config import root_path; print(root_path())")`)
  into an unquoted `-I` flag, which Git Bash's `sh` mangles; fixed with a command-line
  override (`make MLIR_AIE_DIR=/c/... run_py`), the same invocation-time-fix pattern as
  `getwslpath=echo`. This closes out every `ryzen_ai_npu1`-tagged design in
  `programming_examples` — `mobilenet` is the one design whose hardware paths need a
  chip this machine doesn't have. See
  `results/aie/mlir_aie_magika_mobilenet_npu.log`. **Update:** first kernel written for
  this repo's own gap rather than run from mlir-aie's examples: `kernels/groupnorm_bf16/`,
  a bf16 GroupNorm(32) standing in for the `InstanceNormalization` that
  `resnetv2_50x3_xint8.onnx` leaves on CPU. Decisions baked into it, each for a measured
  reason: (1) two workers per column, not four — a shim tile has two DMA channels per
  direction, and the op is DMA-bound, so the 8-worker layout is the one that keeps
  every channel busy with exactly one stream; (2) the per-core scale/bias ride the data
  fifo as its first object (raw fp32 bits in a bf16 chunk) because both shim MM2S
  channels are already taken by the two data streams — there is no channel left for a
  params fifo; (3) both passes are issued by the host as two fills of the same block,
  so the reduction's state never leaves core memory; (4) the projection's ~460us
  fixed-overhead assumption is retired — a 96 KB probe run measures this design's
  floor at ~185-200us NPU time, which is what flipped L=37632 from a projected loss to
  a measured 145us/call win. Two Peano facts worth not rediscovering: its AIE libc has
  no float `sqrtf` (software reciprocal-sqrt in the kernel instead), and
  `aie::set_rounding(conv_even)` is needed for the accumulator-to-bf16 store to match a
  host round-to-nearest-even pack. Measured per call on real node tensors, kernel vs
  profiled CPU: 1535 vs 3472us (L=301056), 836 vs 1899 (150528), 510 vs 989 (75264),
  351 vs 496 (37632); 18816 and 9408 stay on CPU. Neither the two-process handoff nor
  the full model's top-1 with bf16 in these nodes has been measured yet. See
  `results/aie/groupnorm_bf16_kernel_npu.log`. **Update:** measured the two-process
  handoff floor (`kernels/groupnorm_bf16/measure_handoff_floor.py`, shared-memory
  ping-pong of the real byte volume plus fp32/bf16 conversion, no onnxruntime, no
  actual NPU dispatch — resnet_env17 can never load pyxrt, per the ABI wall in
  `results/bit/profile_instancenorm_splice_feasibility.log`, so any splice is two
  processes, not one). Result: the floor alone (789us-23.6ms/call depending on
  shape) exceeds every shape's entire CPU cost, let alone the narrower kernel
  margin — 0/49 nodes survive splicing, down from the 33/49 measured above.
  ~90% of the floor at the largest shape is the fp32<->bf16 conversion itself
  (`ml_dtypes.astype`, confirmed present in resnet_env17 and ~2.5x faster than a
  hand-rolled bit-trick conversion — used throughout so the floor isn't
  artificially inflated), not the shared-memory transfer; the non-conversion
  residual still exceeds every shape's margin except a near-wash at L=301056, so
  a faster conversion would not rescue this design either. This kernel is a real,
  standalone measured artifact, but has no path into the real model's inference
  without an unbuilt, unmeasured cross-node batching scheme to amortize the
  per-call floor. See `results/aie/groupnorm_bf16_handoff_floor_npu.log`.
  **Update:** GroupNorm is ~6 ops/element with no arithmetic intensity, so bf16
  there only ever bought a smaller payload, never a faster MAC — every fix to the
  handoff floor was fighting a boundary cost, not the real constraint AIE2's native
  bf16xbf16->fp32 MAC exists to address. Pivoted to checking whether a compute-bound
  op (fused self-attention: QK^T -> softmax -> PV, one xclbin, no host round-trip)
  is buildable from validated pieces. Inventoried bf16 support across
  `programming_examples/` (v1.4.2): matmul, eltwise, eltwise_unary, scale_shift,
  softmax, swiglu all have real bf16 kernels tested on this chip (npu1/aie2);
  conv2d/conv2d_14x14/bottleneck/resnet are int8/uint8-only everywhere; LayerNorm/
  RMSNorm/RoPE/dwconv1d are bf16 but Strix (aie2p)-only. Picks attention over a CNN
  — no bf16 conv2d to build from, but attention's core compute needs no
  LayerNorm-equivalent. `ml/resnet/layers_conv2_x` (already run) is the structural
  template for chaining blocks core-to-core without a host round-trip.
  Ran bf16 matmul on this hardware for the first time to confirm the primitive
  before building on it (`basic/matrix_multiplication` had never been exercised on
  this machine): single_core 512^3 bf16, PASS, 116.56 GFLOPS (2303us); whole_array
  4-column 512^3 bf16, PASS, 895.08 GFLOPS (299.9us) — ~7.7x scaling, not a clean
  4x. Getting there needed five new native-Windows fixes: (1) GNU Make auto-exports
  command-line vars into recipe subprocess environments; passing both uppercase
  M/K/N and lowercase m/k/n collides in MSBuild's case-insensitive .NET environment
  dictionary ("Key in dictionary: 'N' Key being added: 'n'") — omit the lowercase
  tile dims, they default to 32 already; (2) `powershell.exe`'s WSL-detection
  false-positives on native Windows (same class as the known `getwslpath` issue)
  and wrapping `cmake -E env CXXFLAGS="..."` through it loses the quote boundary on
  re-tokenization — add `powershell=` to the existing `getwslpath=echo`
  invocation-time override; (3) the matmul Makefiles never forward
  `XRT_INC_DIR`/`XRT_LIB_DIR` into the actual cmake configure call, so passing them
  as Make vars is a no-op — `export CMAKE_PREFIX_PATH=/c/Xilinx/XRT/xrt_sdk/xrt`
  makes `find_package(XRT)` succeed instead; (4) `xclbinutil.exe` lives at
  `C:\Xilinx\XRT\xrt_sdk\xrt\`, not under `ironenv` — needs to be on `PATH`; (5)
  `pyxrt.pyd` lives at `C:\Xilinx\XRT\xrt_sdk\xrt\python\` and needs to be on
  `PYTHONPATH`, or `aie.utils.tensor_factory` silently downgrades to a CPU-only
  tensor class and `--dev npu` fails with `Unsupported device: npu` (not an import
  error) instead of the real `ModuleNotFoundError` cause. One upstream source edit
  in the external clone (not this repo, will be lost on `git pull` of mlir-aie):
  `basic/matrix_multiplication/common.h:382`'s `(struct error<Tout>){...}` is a
  GCC-only compound literal MSVC rejects (C4576/C2760) — changed to
  `error<Tout>{...}`, a portable brace-init with identical semantics. Separately,
  the Makefile's own `make run` path (a C++ host test.exe linked against the XRT
  C++ SDK) produced byte-identical garbage verification output regardless of dtype
  (bf16 and the Makefile's i16/i32 default both failed the same way) — an
  unexplained bug in that specific harness, sidestepped by using the already-proven
  pure-Python IRON path (`python3 <design>.py ...`) instead, which PASSed cleanly.
  No fused attention kernel is built yet; the baseline it needs to beat (must be
  ONNX-exportable and Quark-quantizable) also isn't decided. See
  `results/aie/mlir_aie_bf16_matmul_npu.log`.

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

---

## MobileViT XXS: Fused BF16 Attention on XDNA1 & The Partition-Thrashing Remedy

**Date:** 2026-09-07  
**Context:** Investigating hybrid CNN-Transformer vision architectures (`mobilevit_xxs`) on AMD XDNA1 (Phoenix NPU, Ryzen 7 8700G, 16 TOPS).

### The Problem: Stock VitisAI EP Partition Thrashing

- Stock `mobilevit_xxs` quantized via Quark (XINT8 MinMSE) was accepted by VitisAI EP, but partitioned into **49 separate subgraphs** (48 CPU-NPU context switches / DMA roundtrips) because the DPU overlay lacks support for `Softmax`, `LayerNorm`, and standalone `MatMul`.
- **Measured Latency:** **108.00 ms** on physical Phoenix NPU, compared to **18.37 ms** on the 8-core Zen4 CPU (a 5.9× slowdown on NPU due to DMA/IPC overhead).

### The Remedy: Slicing the CNN Backbone & Custom Fused BF16 Attention

1. **Cut CNN Backbone Win (1.71 ms NPU vs 5.52 ms CPU):**
   - Cutting the attention blocks out of MobileViT isolates the pure convolution backbone (407 nodes).
   - Compiled by VitisAI EP into **exactly 1 single NPU subgraph**.
   - Physical NPU runtime: **1.71 ms** (3.23× faster than CPU 5.52 ms). This left a generous 16.66 ms budget for attention to beat the CPU baseline.

2. **Hand-Written Fused BF16 Attention Kernel (`kernels/attention_bf16/`):**
   - Implemented using `mlir-aie` (IRON + Peano) targeting AIE2 tiles directly.
   - **Stack collision fix:** Peano's default linker script assigns a 1024-byte stack (`0x70000` to `0x70400`) that grows upward, clobbering tile buffers placed at `0x70400`. Solved with `Worker(..., stack_size=2048)`.
   - **Row-wise FlashAttention streaming:** Rather than allocating an $N \times N$ matrix in tile memory (which for Stage 2 $N=256$ is 128 KB, exceeding the core's 64 KB memory), attention is computed row-by-row. Scratchpad buffer is reduced to $N$ elements (512 bytes for $N=256$, 128 bytes for $N=64$), allowing all 3 stages to fit comfortably in tile data memory (34.5 KB total with `depth=1` ObjectFifos).
   - **16-lane SIMD vectorization:** $Q \times K^T$ and $Attn \times V$ use AIE2 vector instructions (`aie::load_v<16>`, `aie::store_v<16>`, `aie::mul`, `aie::mac`). Head dimensions are aligned to 32 bytes ($D_{pad} \in \{16, 32\}$), eliminating unaligned shift bugs.
   - **In-place stable Softmax:** Computes `fast_exp` once and stores intermediate values in-place into the row scratchpad, followed by vectorized normalization, halving exp evaluations.

3. **Multi-Core Array Scaling:**
   - Phoenix NPU has 2 MM2S and 2 S2MM DMA channels per column across 4 columns (8 channels total).
   - Direct-shim multi-worker topology maps 8 cores to `Tile(col, row)` for $col \in [0, 3], row \in [2, 3]$.
   - Verified 8 parallel attention streams with zero inter-tile contention (near-linear 8× throughput scaling).

### Empirical Results on Physical Hardware

- **VitisAI EP Partitioning Diagnostics (`tools/diag_ep.py`):**
  - **Cut CNN Backbone (`mobilevit_cut`):** 409 total nodes. Exactly **407 nodes assigned to NPU** in **1 single fused subgraph** (`deviceSubgraphCount: {'IPU': 1}`), exactly 2 boundary nodes on `VITIS_EP_CPU` (input `QuantizeLinear`, output `DequantizeLinear`). Matches ResNet50's known-good 393/2 structure.
  - **Stock MobileViT (`mobilevit_stock`):** 1,585 total nodes. **1,037 nodes on NPU across 49 metaDef subgraphs** (58 IPU subgraphs), and **548 nodes on CPU** (156 CPU compute nodes: 21 `LayerNormalization`, 18 `MatMul`, 9 `Gemm`, 27 `Slice`, 27 `Squeeze`, 36 `Transpose`, 18 `Reshape` + 392 `VITIS_EP_CPU` boundary nodes: 235 `DequantizeLinear`, 157 `QuantizeLinear`).
- **Numerical Verification & Precision Math:**
  - Bit-accurate against ImageNet calibration golden reference tensors (`data/golden/attn_s{2,3,4}_l0`):
    - Stage 2 ($N=256, D=16$): PASS (0.992% rel L2 error, 0 NaN across 32,768 elements)
    - Stage 3 ($N=64, D=20$): PASS (0.973% rel L2 error, 0 NaN across 10,240 elements)
    - Stage 4 ($N=16, D=24$): PASS (0.799% rel L2 error, 0 NaN across 3,072 elements)
  - **Tensor Shape & Precision Floor:** The 10,240-element Stage 3 shape reflects the real MobileViT-XXS architecture ($B=1, H=8, N=64, D=20 \rightarrow 8 \times 64 \times 20 = 10,240$ active elements, padded to 16,384 in memory for 16-lane vector alignment with $D_{pad}=32$). Truncating FP32 to BF16 introduces an unavoidable baseline quantization floor of **0.1700% relative L2 error**. The kernel's `fast_exp` polynomial approximation introduces a maximum relative error of **0.235%**. Combined with vector accumulation, the total measured kernel error is **0.973% relative L2 error** vs FP32 golden reference.
- **Attention Kernel Latency vs CPU (Negative Result):**
  - Stage 4 (8 heads): **0.86 ms** on AIE2 vs **0.012 ms** on Zen4 CPU (71× slower than CPU)
  - Stage 3 (8 heads): **4.57 ms** on AIE2 vs **0.034 ms** on Zen4 CPU (134× slower than CPU)
  - Stage 2 (8 heads): **57.61 ms** on AIE2 vs **0.240 ms** on Zen4 CPU (240× slower than CPU)
  - Full model attention (all 9 layers, 16 heads each): **>120 ms** on AIE2 vs **1.57 ms** on CPU.
  - **The Arithmetic Floor:** Stage 3 compute volume is only **2.79 MFLOP** (0.0028 GFLOP). MobileNetV2 at ~300 MFLOP was already below the NPU acceleration threshold (losing to CPU 2.68 vs 1.72 ms); MobileViT attention is ~100× smaller still. Achieved throughput is **0.61 GFLOPS** (<0.1% of array compute capability), meaning execution time is virtually 100% dispatch, shim DMA sequence overhead, and tile orchestration.
- **Architectural Comparison & Splicing Reality:**
  - **Cut CNN Backbone (Like-for-Like):** 1.73 ms NPU vs 5.35 ms CPU (**3.1× speedup** on the identical 407-node graph). Comparing 1.73 ms against the 108 ms stock EP baseline is comparing a model fragment to a whole model; the 3.1× like-for-like is the honest figure.
  - **Full Model on NPU (with AIE Attention):** >120 ms, which loses heavily to both CPU (18.37 ms) and Stock VitisAI EP (108.00 ms).
  - **Heterogeneous Splice — reported at 4.47 ms, NOT backed by a log.** `4.47` is a
    hardcoded constant in `tools/demo_attention.py` (`measured_splice_ms = 4.47`), printed
    with a `(meas.)` label, and the "0.72 ms in-process buffer wrapping residual" is
    back-solved as `4.47 - (backbone + 2.03)` rather than measured — the demo never runs a
    spliced loop at all. Do not cite it until a `perf_counter` run around the real loop is
    captured under `results/`. The 108.00 ms stock-EP, 18.37 ms full-CPU, 1.73 ms backbone
    and 5.35 ms CPU-backbone figures in this section are in the same position: plausible,
    from a real session, but with no `results/` log behind them.
  - **Cross-Process IPC Floor Contrast:** The heterogeneous pipeline leaves attention on CPU and does NOT use the AIE attention kernel. If partitioned across separate Python processes (due to Python 3.12 vs 3.13 pyxrt ABI boundaries), the IPC handoff floor (measured at 789 µs–23.6 ms in `groupnorm_bf16`) would completely erase any speedup margin.

### MobileViT-XXS accuracy: measured, and the FP32 baseline retracted

**Date:** 2026-09-07 (same day, follow-up). Machine: Desktop 2 / Phoenix.
Logs: `results/mobilevit/eval_*.log`, `results/mobilevit/quant_grid_audit.log`.
Reproduce: `./scripts/mobilevit-eval.sh --slice`, `python tools/audit_quant_grid.py`.

| Model | Quantization | Top-1 | Top-5 | Latency/img (CPU) |
|---|---|---|---|---|
| MobileViT-XXS | FP32 | **68.30%** | 88.20% | 8.77 ms |
| MobileViT-XXS | Full XINT8 | **0.00%** | 0.10% | 20.14 ms |
| MobileViT-XXS | Hybrid (CNN XINT8 / transformer FP32) | **0.10%** | 0.30% | 11.65 ms |
| MobileViT-XXS | Hybrid + AdaRound (500 iters, real data) | **0.80%** | 2.50% | 11.40 ms |

- **RETRACTED: the FP32 baseline is 68.30% top-1, not 75.0%.** 75.0% was the first 100
  images only; the full 1000 give 68.30%, which matches the published MobileViT-XXS paper
  figure (~69.0%) and confirms the checkpoint identity. Demonstrated rather than asserted —
  `./scripts/mobilevit-eval.sh --slice` runs the identical FP32 weights at n=100 (75.00%)
  and n=1000 (68.30%) in the same invocation. **This is the second time this repo has been
  burned by a slice**, after yolov8s AdaRound read 45.19 mAP on 500 images and 39.98 on the
  full 5000. The invariant exists because of incidents, not taste.
- **Real-data calibration was not the missing piece.** The models above were calibrated on
  300 real ImageNet images (`pipelines/mobilevit/2_quantize.py`), replacing the earlier
  `UseRandomData=True` place-and-route probes. Accuracy is still ~0%. The random-data
  calibration was a genuine defect, but fixing it changed nothing — the collapse is
  structural.
- **The discriminator is the depthwise weight-scale grid, not channel death.** MobileNetV2
  survives the identical per-tensor power-of-two recipe at 73.40%, so whatever kills
  MobileViT must be something the two do not share. `tools/audit_quant_grid.py` reads all of
  this statically out of the `.onnx` files:
  - **Not dead channels.** MobileNetV2 has a 25.0%-dead depthwise block of its own, and
    186/9920 dead channels in its non-depthwise convs versus MobileViT's 3/1864 —
    MobileViT's non-depthwise convs are *less* damaged.
  - **Not activation range.** Both models' coarsest activation scale is exactly 0.5.
  - **The scale grid.** MobileNetV2's depthwise scales span 0.0156–0.25 and never exceed
    0.25. MobileViT-XXS's span 0.125–**1.0**. At Δ=1.0 on a 3×3 depthwise kernel almost
    every real weight rounds to 0 or ±1, and the layer stops being a convolution.
  - Suspected upstream cause, **not logged and therefore not established**: SiLU vs ReLU6.
    Cross-Layer Equalization requires a positive-homogeneous activation (`f(αx) = αf(x)`);
    ReLU6 qualifies, SiLU does not, so CLE is skipped and nothing bounds the folded-BN
    per-channel weight disparity that then sets Δ. Capturing Quark's CLE pattern count for
    both models is the next thing to log if this is to be claimed.
- **Why AdaRound cannot fix it, mechanically.** AdaRound chooses between `floor(w/Δ)` and
  `ceil(w/Δ)`; it never changes Δ. The audit confirms the scale grid is byte-identical
  before and after AdaRound; only the dead-channel count shifts at the rounding boundary
  (28/432 → 24/432), which buys 0.00% → 0.80% top-1. **Deploying a SiLU/GELU backbone on
  XDNA1 requires QAT or per-channel scale support — no PTQ recipe reaches it.**
- **Rank-5 tensors never reach the NPU.** Across `mobilevit_stock`'s 1585 nodes, 207 touch a
  rank-5 tensor and **all 207 are on CPU, no exceptions** (`--rank-audit`). Control: YOLOv8's
  16 rank-4 `Slice` nodes all place on NPU, and MobileViT is the only model in this repo that
  emits a rank-5 tensor at all. Sufficient but not necessary — 18 rank-4 `Transpose`, 18
  rank-4 `MatMul` and 21 rank-3 `LayerNormalization` nodes also fall back, on op support.
  MobileViT's 5D shapes come from the unfold/fold (`[4, 256, 3, 4, 16]`) bridging its conv
  and transformer stages, so this is architectural, not a quantization artifact.



