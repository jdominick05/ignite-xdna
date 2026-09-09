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
3. **Keep XINT8 as the measured default recipe.** Power-of-two scales, MinMSE calib,
   UINT8 activations / INT8 weights+bias. The original rule was "X1 backend = XINT8
   only": A8W8 (float scales) silently fell back to CPU (39 ms latency = CPU speed,
   accuracy 66.6%). **That measurement stands; attributing it to float scales alone
   is superseded by [Ignition's controlled probes](BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance).**
   On the no-CLE ResNet, moving Q/DQ to `com.microsoft` alone causes full fallback;
   signed INT8 activations preserve baseline NPU outputs over the full evaluation set.
   Perturbing activation scales off the power-of-two grid can retain NPU placement
   while breaking numerical agreement. INT32 bias dtype alone also retains placement
   and baseline NPU outputs, but the input×weight-scale representation produces wrong
   NPU results. Keep both as experimental mutations. **Per-channel weight scales are now
   measured and rejected (2026-09-09): the EP places 0 of N nodes -- whole graph to CPU, all
   Convs included -- for any weight scale with more than one element, at 1, 2, 4 and 8
   convolutions. A single-output-channel control, where the vector has length 1 and only the
   `axis` attribute differs, places normally with identical output, so it is the vector length
   and not the attribute. The values in these fixtures are identical repeats, so it is not about
   channel grids differing either.**
   [Verdict and controls](BENCHMARKS.md#per-channel-weight-scales-are-rejected-outright-2026-09-09-desktop-2). Stripping metadata or naming
   the producer `Ignition` preserves measured placement and outputs. These findings
   bound this artifact and runtime, not every graph the compiler may see.
   **A16W8 (INT16 activations / INT8 weights) now measured,
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

- **The launcher (`tui/`) is a front end, not a measurement tool, and the boundary is
  load-bearing.** It writes only to `outputs/` (git-ignored): every demo defaults
  `--out-dir` to `results/`, which is the tracked evidence base -- 69 of its images
  are cited from the docs, and `adaround_diff_demo.py` writes exactly
  `results/adaround_diff_yolov8n_npu.jpg`, which `demos/README.md` links. A launcher
  that did not override that flag would overwrite cited evidence on its first
  successful run, so `tui/runner.py` passes an absolute `--out-dir` every time and
  `--selftest` asserts no output path resolves inside `results/`. Latencies it prints
  carry `quotable: false` in their sidecar next to the contention verdict, because a
  number-shaped artifact taken under an unknown host load is not a measurement.
  Rejected along the way: a cross-hardware compare mode (a same-sitting CPU/iGPU/NPU
  table is exactly the thing that should be a logged run, not a menu item); an
  in-process task lane (an ORT session inside the UI means a DPU timeout kills the
  launcher, a lingering session holds a context on single-tenant silicon and blocks
  the next run, and structured results would have to come from parsing stdout -- so
  tasks are `python -m tui.task`, spawned like a demo, returning a sidecar JSON); and
  a `classify` task, because no ImageNet index-to-name mapping exists anywhere in this
  repo and a task that answers `285` is not one anyone can use.

- **Compile-cache staleness is exactly detectable, and nothing was using the
  detection.** The cache is keyed by name rather than by model hash, so switching
  model within a family silently reuses the previous compile -- the documented
  wrong-weights failure. But `<cacheKey>/context.json` records `config.onnxPath`, the
  model that compile was actually built from, and it is present in all 31 caches on
  Desktop 2. Comparing it against the model about to run turns "always pass
  `--fresh`" from a rule people remember into a check, with no sidecar state to
  drift. Measured on 2026-09-09 while writing this: `modelcachekey` (ResNet50's key)
  held a compile of `wide_resnet101_2_xint8_c64.onnx`, left by `classifier_width_demo`
  which shares `RESNET_CACHE_KEY` across all three classifiers, and `yolocutcachekey`
  held `yolov8n_cut_xint8_adaround.onnx`. A `4_run.py --ep npu` or a plain yolov8n run
  without `--fresh` at that moment would have executed the wrong weights and reported
  a plausible number. Paths in that field are recorded three ways (relative posix,
  relative windows, absolute), so any comparison has to resolve against `ROOT` first.


- **A wall-time or peak-memory figure from Desktop 2 without a host-load witness is a
  guess.** That box runs three Claude sessions, `agy` and PyCharm against the same 16
  threads, and `xrt-smi` answers only the device question -- nothing was watching the CPU.
  On 2026-09-08 the MODNet AdaRound oracle spent its whole calibration and MinMSE phase
  beside another session's `pipelines/yolow/3b_quantize_cut.py`, which nobody knew until a
  check was written for it; the suspicion at the time was `agy` compiling, and `agy`
  measured 0.2 cores. `tools/host_load.ps1` is the host-side counterpart to the `xrt-smi`
  check: one-shot it classifies build tools, this repo's own producers (by command line,
  because every producer here is `python.exe`) and any process holding 2+ cores, and
  `-Watch` samples a timeline. `check_host_load` in `scripts/lib.sh` warns on every NPU run
  through `npu_env`, and refuses to start `quant-reference.sh` / `quant-own.sh` /
  `quant-adaround.sh` unless `--allow-busy`, which records the override in the witness.
  What contention does and does not reach matters: parity and accuracy are fixed-seed,
  fixed-thread computations and are untouched by it; wall time and peak working set are
  not comparable across runs that saw different load, and peak working set can read *low*
  under memory pressure because the OS trims the set.

- **Do not apply CLE to a grouped or depthwise architecture without checking what it did to the
  ranges.** Cross-layer equalization is on in the default XINT8 preset and it is what destroys
  this repo's grouped classifiers -- not the DPU, not the calibrator, not the rounding. Holding
  producer, graph and 64-image listing fixed and changing only CLE: RegNetX-002 goes 0.10% ->
  **66.20%** top-1 and ResNeXt-50 32x4d 0.10% -> **68.90%**, both on the full 1,000-image set.
  With CLE the emitted scale positions span -120..123 and Quark's own shift-cut rule has to move
  a weight position from 7 to -108 to keep them representable; without it every position is in
  2..10 and no adjustment fires at all. Placement is identical either way (324/326 for RegNetX),
  which is why this is invisible unless accuracy is actually measured. CLE remains the right
  default for the plain-convolution families where it is worth 10.8-12.2 points on ResNet50 --
  the rule is to check, not to abandon it. **What to check is the per-channel scale, not the
  weight range**: CLE keeps weights tidy by construction and in fact *narrows* the weight
  positions on both models it destroys, while the activation between the two layers is
  multiplied by the scale and never rescaled. Measured per equalized pair, the worst is
  2.36 bits on ResNet50 and 1.81 on MODNet against **42.20** on RegNetX-002 and **72.62** on
  ResNeXt-50 -- the last two measured with the depthwise **triple** formula, which is what those
  patterns actually are. `--cle-guard BITS` skips a triple over the threshold and never touches a
  pair, because no single cut separates the populations: ten of ResNeXt-50's destructive triples
  sit at 2.2-3.4 bits, inside ResNet50's beneficial range. At **2 bits** it recovers RegNetX-002 to
  66.20% and ResNeXt-50 to 68.90% while leaving ResNet50 and MODNet byte-identical; at 4 it lets
  five of RegNetX's fourteen through and reads 25.60%, worse than either extreme. Unguarded,
  Ignition now **refuses to emit** these models -- the post-CLE activation is nonfinite in float16.
  [Triple path, spans and results](BENCHMARKS.md#depthwise-cle-triples-implemented-and-the-guard-narrowed-to-them-2026-09-09-desktop-2). Ignition currently refuses `--cle` on these graphs, but
  only because the depthwise path is unimplemented, so that is fail-closed by accident rather than
  a guard. [Full matrix, scale grids and caveats](BENCHMARKS.md#regnetx-002-and-resnext-50-recovered-the-collapse-is-cle-not-a-hardware-bound-2026-09-09-desktop-2).

- **Exact float16 frequency tables do not replace the calibration spool.** Counting every
  distinct stored float16 value, and bounding the reduction conservatively enough that a
  certified position cannot be wrong, is sound: it reproduced the emitted ONNX byte for
  byte on ResNet50 with and without CLE, on YOLOv8n-cut and on MODNet-Cut, and never
  certified a position the ordered reduction disagreed with. It is also not worth having.
  The bound certifies 15-24% of activation tensors and they are the small ones, so it
  avoids 0.70-3.72% of spool bytes, while a count table costs a fixed 524288 bytes per
  tensor -- a net 0.29-1.94%. Worse, the uncertified majority still needs an ordered
  spool, so counting alone has to replay a second full inference pass: five
  order-balanced blocks put it 1.125x slower than the spool it would replace. The saving
  shrinks as activations grow, so it is smallest on exactly the models whose spool is the
  problem. [Method, four-family parity and the timing blocks](BENCHMARKS.md#exact-count-calibration-certificate-and-fallback).

- **An optimized CPU session is not sufficient as a QDQ numerical reference.**
  Ignition's INT32-bias dtype-only mutation preserves decoded biases and unoptimized
  CPU outputs, but this ORT build's optimized CPU output differs. Comparing the NPU
  only with that optimized result would blame the wrong execution path. Keep
  `ORT_DISABLE_ALL` as a separate reference, and require output checks in addition
  to EP placement. Product-scale INT32 bias and non-power-of-two scale mutations
  expose the converse failure: NPU placement with incorrect numerical execution.
  [Controlled evidence and limits](BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance).

- **Quark's refinement is a silent no-op on `raw_data` scale initializers.** Its
  `set_scale` writes `float_data` in place but assigns a `raw_data` update to a
  temporary list, so on such a file it runs five passes, logs "Modify" lines and the
  loop-limit warning, and changes nothing. Quark's own output stores scales as
  `float_data`; Ignition's stores `raw_data`, so running Quark's `adjust_quantize_info`
  on an Ignition artifact refines nothing while claiming to. Ignition's `refine` reads
  either form. [Measured on the perturbed oracle](BENCHMARKS.md#ignition-refinement-rules-under-perturbation).

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
  being compared in one interleaved sitting. **Tested 2026-09-07 for one candidate cause,
  an NPU clock that decays with idle:** three probes each after 5 s of idle read the same
  1.77–1.79 GHz as back-to-back calls (`results/aie/clock_probe_npu.log`), so idle at
  that scale is not it. Power mode *is* a 2.25× clock lever (0.80 → 1.80 GHz), and nothing
  in this repo records the mode a number was taken under — a sweep should capture
  `xrt-smi examine -r platform` alongside, as `kernels/clock_probe/clock_probe.py` does.
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
  floor at ~185-200us NPU time (CONFIRMED 2026-09-07 by direct measurement at 169.8us —
  but see the dispatch-floor entry below: a design does not *pay* the hardware floor, it
  pays 617us through the IRON call path), which is what flipped L=37632 from a projected loss to
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
  **Update:** that floor's ~90% conversion cost turned out to be `ml_dtypes.astype()`
  itself, not physics — bfloat16 isn't a native numpy dtype, so it runs a scalar
  loop with no SIMD path. A strided-view truncation with preallocated buffers
  (`measure_handoff_floor_v2.py`) cuts the L=301056 floor from 23.6ms to 5.7ms, and
  isolating the shared-memory protocol alone (zero conversion) measures 1.19ms —
  already under CPU's 3.47ms at that shape. Separately, re-profiling the real model
  found the QuantizeLinear/DequantizeLinear nodes wrapping every InstanceNorm site
  add 56.8% on top of its own cost (65.08ms across all 49 nodes, not 42.37ms) — the
  real bar an int8-native design (on-core dequant/requant, folding those nodes'
  scales into the kernel) would need to clear, which the measured protocol floor
  already does at the hardest shape. No int8-native kernel is built; this reopens
  the question rather than answering it. See
  `results/aie/groupnorm_bf16_handoff_floor_v2_npu.log`.
  **Update, written after the above:** GroupNorm is ~6 ops/element with no arithmetic intensity, so bf16
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
- **bf16 GEMM at M/N >= ~1024 is the first genuine NPU win measured anywhere in this
  project (2026-09-07).** After conv closed at a 12.75x loss even at ResNet50's real
  56×56 shape, and mobile-vision attention closed at 71–240×, the user reframed the
  goal explicitly: stop trying to make the NPU match CPU at every op, find where it
  actually has an edge. No CPU bf16/fp32 GEMM baseline existed in this repo to check
  the 895 GFLOPS anchor against — every other CPU number here is int8 QDQ conv. Built
  one (`kernels/bf16_matmul_sweep/cpu_matmul_sweep.py`, torch bf16/fp32, this
  machine's Zen4 cores) and swept `whole_array.py` (4-column bf16) across shapes.
  **At the one shape previously measured (512³), the NPU's 895 GFLOPS actually *loses*
  to plain CPU torch bf16 (1100.6 GFLOPS)** on this specific machine (Ryzen 7 8700G, no
  discrete GPU) — the "895 is a strength" framing was incomplete without this
  comparison. But NPU throughput keeps climbing with M/N (895 → ~1800–2070 GFLOPS)
  while CPU bf16 stays close to flat (1100–1360 GFLOPS); the crossover is around
  N=1024 at M=K=512, and from there the NPU wins by **1.18×–1.78×** up to a K limit
  (below). Every NPU number is a verified PASS against numpy `A@B`, not just timed.
  **Diagnosed (2026-09-07): the "K >= 3072 fails, undiagnosed" limit was mis-framed —
  not a threshold, and not undiagnosed.** Bisecting K in small steps shows the error
  starts continuously around K/k≈23 reduction steps and grows smoothly to near-total by
  K/k=96, always a systematic ~10–13% undercount, never NaN/garbage — a precision
  signature, not an addressing bug. **Root cause: the K-reduction loop accumulates the
  running sum in a buffer typed `dtype_out`, not fp32.** `aie::mmul` does accumulate one
  MAC to fp32 natively, but with `--dtype_out bf16` that fp32 result is rounded back to
  bf16 before the next reduction step adds onto it, so the running sum round-trips
  through bf16's 8-bit mantissa on every step and swamps small increments once its
  magnitude grows. **Fix, and it's free: `--dtype_out f32`** — clean PASS at K=2880 and
  K=4096 on the real `whole_array` design, 1741.7 and 1830.2 GFLOPS, in the same range
  as the bf16-output numbers above. K was never the ceiling; the output dtype was. See
  `results/aie/bf16_matmul_k_limit_diagnosed_npu.log`. This is the "LLM-scale, not
  mobile-vision" shape `attention_bf16/README.md`'s own math already predicted would be
  needed — but does not by itself mean a fused attention block would win (softmax + two
  data-dependent matmuls, not one static GEMM), and whether `attention_bf16`'s own
  kernel has the same accumulate-in-`dtype_out` pattern is unchecked. See
  `results/aie/bf16_matmul_niche_npu.log`.
- **int8 GEMM (2026-09-07): the NPU's headline dtype loses at `whole_array`'s default tile
  and wins only with a tile bf16 can't fit — use `n=64` for int8.** Same upstream design,
  `--dtype_in i8 --dtype_out i32` (i32 for the same accumulate-in-`dtype_out` reason as
  above), against the CPU's own int8 GEMM kernels — torch `_int_mm` and ORT
  `MatMulInteger` u8s8, both timed because this repo has twice lost a verdict to the slower
  CPU kernel; torch's is 1.05–1.68× faster here and is the verdict line. At the default
  m=64/k=64/n=32: int8 runs only 1.1–1.5× the bf16 rate (not the 2× the MAC count
  promises) and loses to the CPU's int8 kernel at every shape but 1024³. The default tile
  is bound by dtype-blind costs — the design re-streams A from DDR N/(n·4) times and B
  M/(m·4) times, each k-step handshakes two tiles through two FIFO levels, and the output
  tile it read-modify-writes is 4 bytes per element for i32 and f32 alike. int8's
  half-size A/B tiles put its default at 32 KB of the 64 KB L1 (bf16: 44 KB), so `n=64`
  fits (52 KB) and doubles the rate bit-exact (2048³: 2374 → 4448 GOPS; `m=128` +37%,
  `k=128` +20%). bf16 at `n=64` needs 68,864 B — over by exactly the 3,328 B stack
  (`'aie.tile' op Basic sequential allocation failed`; the allocator dump is in the log).
  Verdict by the mean: NPU int8 at `n=64` wins 1.10×–1.83× at M ≥ 512 and N ≥ 2048, loses
  at 512³ and M ≤ 256; against torch's best-case (min) time the K=N=4096 rows flip to a
  1.13–1.24× CPU win. **The small-M loss is a tile artifact, not a hardware property:**
  `whole_array` forces m = M/8 (4 rows × 2 transfer blocks), and M=128 at m=16 equals
  M=512 at m=16 to within 6% at both dtypes. **Rejected for now, deliberately: bf16 at
  `n=64` by single-buffering the C output FIFO** (`depths=[1]` in the C join — the change
  that got `bottleneck.py` to 56×56 and would free 16 KB) — `whole_array.py` is the shared
  upstream file another live session was running FFN measurements through, and editing it
  under them would silently change their numbers. Next session that owns the file: try it;
  if bf16 gains what int8 gained, the bf16 niche roughly doubles. **Superseded the same
  day:** the other session took the file and measured it — bf16 at `n=64` with the C tile
  single-buffered reads 2477.23 GFLOPS at 2048³ against the default tile's 1775.65, and
  the CPU-bf16 margin widens to 1.29×–1.89× at M, N ≥ 1024
  (`results/aie/bf16_matmul_n64_single_buffer_npu.log`; the patch is kept as
  `kernels/gemm_tile_sweep/whole_array_c_single_buffer.patch`). The rejection above was a
  file-ownership call, not a measurement, and is kept here as written. Also not measured: the
  requantize-to-int8 epilogue a real quantized layer needs (upstream's i8→i8 kernel path
  accumulates in an int8 buffer across K and is unusable past one k-tile). See
  `results/aie/int8_matmul_sweep_npu.log`.
- **NPU monitoring sources (2026-09-07):** `tools/hwinfo_npu_bridge.cpp` reads utilization
  and adapter memory from Windows' GPU-engine statistics (PDH over the D3DKMT counters; the NPU
  is an MCDM adapter with its own LUID), the clock and power mode from XRT's in-process query
  API (`max_clock_frequency_mhz` is a live readback here: 800 MHz idle, 1800 MHz with an active
  context), and the per-context table from `xrt-smi examine -r aie-partitions`
  (`results/aie/xrt_api_live_clock_and_pdh_npu.log`). Tried and dropped: DXCore's
  `D3D12_CORE_COMPUTE` adapter list (returns only the iGPU and the basic render driver; the NPU
  is listed under `DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU`, with a PDH-LUID fallback when even that
  is absent); XRT's `aie`/`aie_shim`/`aie_mem`/`memory` queries ("No such query request" on this
  driver) and `electrical`/`thermal` (fails / no sensors); linking `xrt_coreutil_static.lib`
  (169 MB; the DLL ships with the driver in `System32`, so the import lib is used); installing
  boost into `resnet_env17` (the XRT SDK headers need `boost/any.hpp` — a separate header-only
  env, `npu_monitor_build`, keeps the pinned inference env untouched). Rejected for good:
  inventing a voltage or power figure — no documented interface exposes one on this NPU.
- **Reading the AIE cycle counter from a Peano kernel does not work; use the trace unit
  (2026-09-07).** `aie::tile::current().cycles()` links against a `get_cycles()` that
  llvm-aie 22 declares and never defines (`ld.lld: undefined symbol`);
  `__builtin_readcyclecounter()` fails in the legalizer; inline asm fails in IRTranslator.
  *(Narrowed 2026-09-09: it is statement-level inline asm inside a C++ function that fails
  there. A standalone `.s` file assembles and links fine, so hand-written AIE2 assembly is
  available — `kernels/asm_probe/`, `results/aie/aie2_isa_static.log`. It does not rescue the
  cycle counter: the assembler exposes no timer register name, and the timer is memory-mapped
  at `0x340F8`/`0x340FC` in the tile's configuration space, not in the core's data space. The
  conclusion below is unchanged; only the reason "inline asm" was recorded is more specific.)*
  What works: `event0()`/`event1()` in the kernel, `Program.enable_trace` with
  `INSTR_EVENT_0/1`, and the stamps decoded from the trace stream — with two traps. Fewer
  events than fill a 32-byte packet never reach host memory (emit filler events after the
  real ones), and mlir-aie v1.4.2's `aie.utils.trace.parse` mis-times any gap longer than
  2^18 cycles (146 µs at 1.8 GHz) because it treats the `0xff` sync frame as a timer no-op
  and the following repeat as re-issued events; `kernels/clock_probe/clock_probe.py` has
  the corrected decoder, cross-checked against upstream on a sync-free run. Also learned
  there: pyxrt's `max_clock_frequency_mhz` reads 800 in every power mode and is not the
  live clock (an idle reading, it turned out — see the resolution below); `xrt-smi configure
  --pmode turbo` prints a device error and switches anyway. `results/aie/clock_probe_npu.log`.
- **RESOLVED 2026-09-08 (flagged UNRESOLVED on 2026-09-07) — the two bullets above disagreed
  about `max_clock_frequency_mhz`, and both were right for what they varied.** As flagged: the
  monitoring bullet calls it a live readback (800 idle, 1800 with an active context); the
  clock-probe bullet calls it "800 in every power mode and not the live clock". Both were
  measured on Desktop 2 on the same day by different sessions, and neither has been
  retracted. They may not actually conflict — one varied *load* at a fixed power mode, the
  other varied *power mode* with no workload running, and a value that tracks load but
  ignores `xrt-smi configure --pmode` would satisfy both readings. That is a hypothesis,
  not a measurement: nobody has yet varied both axes in one sitting. Until someone does,
  do not cite `max_clock_frequency_mhz` as the core clock — the trace-unit figure
  (1.80 GHz `default`) is the measured one. A `--once` monitor sample taken while the
  device was idle read 800 MHz, which is consistent with either account and settles
  nothing. **Resolution, both axes in one sitting** (`results/aie/pmode_clock_readback_npu.log`: a 2048³
  bf16 GEMM hold with `xrt-smi configure --pmode` stepped through all five modes, then the
  same five idle, the monitor reading the clock, the mode and the engine utilization every
  0.1–0.25 s; run twice, the second with the device checked idle first): with a context
  active the readback is the mode's clock to the MHz the trace unit measured — 1800
  `default`/`performance`/`turbo`, 1028 `balanced`, 800 `powersaver` — and with no hardware
  context on the device it reads 800 in every mode (a context that exists but sits idle holds
  the mode's clock: the first run's idle pass read 1028 in `balanced` until another process's
  leftover context left the device). The clock-probe's "800 in every mode" was an idle
  reading; the hypothesis above ("tracks load but ignores `--pmode`") is wrong, it tracks
  both. So: the readback is a valid busy-clock indicator (what the monitor and HWiNFO show),
  an idle reading says nothing about the mode, and the trace unit stays the measurement of
  the clock itself. `--pmode turbo`'s escape error reproduced under load and idle, the mode
  applying regardless. Two of the evening's holds hung (`ERT_CMD_STATE_TIMEOUT`) — both
  inside another session's 32- and 128-stream classifier sweeps on the same device, the
  second's timeout expiring in the same second that session's XRT aborted; a clean 10 Hz
  monitor beside a hold did not (`results/aie/npu_monitor_poll_rate_npu.log`).
- **C2PSA spatial self-attention in YOLOv11 fractures into CPU fallback on XDNA1 (2026-09-08):**
  YOLOv11 introduces the C2PSA (Convolutional 2-Stage Pointwise Spatial Attention) block at the
  deepest stage of the backbone (`/model.10`). In the XINT8 quantized model, the VitisAI EP rejects
  the 4D `MatMul` operations ($B=1, \text{heads}=2, N=400$) embedded inside the attention loop,
  placing only 6 out of 1,300 nodes on NPU (0.46%) and leaving all 87 Convolutions on CPU
  (`results/diag_yolo11n_cut_xint8.log`). The resulting CPU-NPU context ping-pong degrades
  single-image latency to 33.29–34.63 ms (eval/demo) — 1.46× slower than host CPU FP32 (21.59–22.82 ms).
  Ablating C2PSA into an identity skip (`/model.10/m/m.0`) proves the remainder of the architecture is
  exceptionally NPU-friendly: the backbone (C3k2) and decoupled depthwise-convolution heads fuse into
  a single monolithic DPU subgraph of 1,173 / 1,180 nodes (99.4%, `results/diag_yolo11n_no_c2psa_cut_xint8.log`)
  executing in **7.08 ms** on 5,000 val2017 images (141.2 fps) — the fastest YOLO model recorded on
  XDNA1. However, because identity ablation collapses unweighted mAP to 0.19, deploying stock YOLOv11
  natively on XDNA1 without DPU attention kernel fusion is rejected.
- **5D Einsum/ReduceMax in YOLO-World v2 forces CPU fallback; Reshape inside conv streams rejected (2026-09-08):**
  YOLO-World v2 embeds multi-scale text cross-attention (`MaxSigmoidAttnBlock`) inside `C2fAttn` blocks at
  layers 12, 15, 18, 21. The VitisAI level-1 DPU compiler rejects the 5D `Einsum` (`bmchw,bnmc->bmhwn`) and
  5D `ReduceMax` operations, placing only 48 of 1081 nodes (4.4%) on NPU while all 67 Convolutions fall
  back to CPU (`results/diag_yolow_cut_xint8.log`), inflating latency to 178.53 ms. An initial ablation
  using `.view(bs, nh, -1, h, w)` revealed a second DPU compiler trap: `Reshape` operations inside conv
  streams are unconditionally rejected on Phoenix, leaving 0 / 993 nodes on NPU. Eliminating `Reshape` via
  precomputed static channel scaling (`(bias.sigmoid() * scale).repeat_interleave(hc)`) unlocked a
  monolithic DPU subgraph of 946 / 953 nodes (99.3%, `results/diag_yolow_no_attn_cut_xint8.log`) running at
  **15.89 ms (62.9 fps)**. However, because bypassing attention destroys open-vocabulary alignment (mAP 0.3%)
  and plain XINT8 PTQ scrambles 5D attention weights (mAP 1.8%), running stock YOLO-World cross-attention
  directly on the DPU is rejected.
- **Depthwise-separable decoder achieves monolithic DPU compilation for monocular depth (2026-09-09):**
  MiDaS v2.1 Small required substituting stock bilinear upsampling with nearest-neighbor resize to avoid
  a 5-subgraph partitioning trap that added 5.63 ms of host round-trips. FastDepth (Wofk et al., ICRA 2019)
  natively pairs nearest-neighbor upsampling with depthwise-separable 5×5 convolutions (`NNConv5dw-skipadd`),
  bypassing the bilinear dispatch trap by design. In the quantized XINT8 graph (`models/fastdepth_fp32_xint8.onnx`),
  the VitisAI EP accepts 255 of 257 nodes (99.2%) into exactly 1 monolithic DPU subgraph
  (`subgraphStat: [{'device': 'DPU', 'count': 1}]` in `results/diag_fastdepth_xint8.log`), with zero internal
  CPU fallbacks across all 38 Convs, 27 Clips, 11 Relus, 5 Resizes, and 3 Adds. Executes in **2.87 ms (348.1 fps)**
  on Phoenix XDNA1 — outperforming both Zen 4 CPU (3.22 ms, 1.12× speedup) and Radeon 780M iGPU DML FP32 (3.02 ms, 1.05× win).
  Unlike MobileViT, depthwise separable layers quantize smoothly under plain XINT8 PTQ without scale grid collapse
  (Pearson $r = 0.9383$, MAD $16.14 / 255$, $\delta < 1.25 = 68.07\%$), proving that lightweight depthwise-separable
  decoders are optimal for real-time dense spatial prediction on XDNA1.
- **Bilateral Guided Aggregation achieves monolithic DPU compilation for semantic segmentation (2026-09-09):**
  BiSeNetV2 (Yu et al., IJCV 2021) couples a wide shallow Detail Branch with a deep narrow Semantic Branch,
  fusing them via Bilateral Guided Aggregation (BGA) using elementwise multiplications gated by Sigmoid.
  When exported with nearest-neighbor upsampling (`models/bisenetv2_fp32.onnx`), Quark's `enable_npu_cnn`
  automatically lowers `left * sigmoid(right)` to DPU-compatible `HardSigmoid` with `alpha=0.166667`.
  In `models/bisenetv2_fp32_xint8.onnx`, the VitisAI EP compiles 402 of 404 nodes (99.5%) into **exactly 1
  monolithic DPU subgraph** (`subgraphStat: [{'device': 'DPU', 'count': 1}]` in `results/diag_bisenetv2_xint8.log`),
  placing all 57 Convs, 40 Relus, 10 Adds, 5 Muls, 2 HardSigmoids, 3 Resizes, 1 MaxPool, 1 AveragePool, and 1 GlobalAveragePool
  natively on AIE. Executes in **13.12 ms (76.2 fps)** on Phoenix XDNA1 — **4.43× faster than Zen 4 CPU** (58.07 ms)
  and **1.09× faster than Radeon 780M iGPU DirectML FP32** (14.25 ms). Stock bilinear upsampling at the head
  ejects 1 Resize node to CPU (399/404 on NPU in `results/diag_bisenetv2_bilinear_xint8.log`), adding 0.26 ms of host dispatch.
  However, physical DPU fixed-point execution reveals a dynamic range limitation: while CPU QDQ simulation
  maintains 59.47% pixel accuracy and 25.72% mIoU, on-device fixed-point elementwise multiplication across disparate
  inter-branch activation scales attenuates minority classes (15.33% pixel accuracy, 2.44% mIoU), confirming that
  multi-branch bilateral gating requires fine-tuning or AdaRound to balance inter-branch scale multipliers on physical systolic hardware.

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

- Stock `mobilevit_xxs` quantized via Quark (XINT8 MinMSE) was accepted by VitisAI EP, but partitioned into **49 metaDef subgraphs (58 IPU subgraphs)** (48 CPU-NPU context switches / DMA roundtrips) because the DPU overlay lacks support for `Softmax`, `LayerNorm`, and standalone `MatMul`.
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
- **The per-dispatch floor, measured in isolation (2026-09-07).** Every isolated-op
  verdict in this repo rested on a "~185-200us" constant inherited from one 96 KB probe
  during the GroupNorm work and never measured on its own. It has now been measured with
  a no-compute passthrough design (shim→memtile→shim, no compute tile, so no math to
  attribute time to), sweeping 8 KB–32 MB, output verified per payload, compile excluded:
  `kernels/dispatch_floor/measure_floor.py`, `results/aie/dispatch_floor_npu.log`.
  - **Hardware floor 169.8 µs** (R²=1.0000) — the old constant was **right**. This is a
    confirmation, not a retraction.
  - **Wall floor 617.0 µs** (R²=0.9998) through the `@iron.jit` path — **3.6× the number
    the write-ups were quoting.** Both `groupnorm_bf16` and `attention_bf16` were charged
    the wall figure while their analyses reasoned with the hardware one. That gap, not the
    constant's value, is what was wrong.
  - **447.3 µs/call is host-side and flat in payload size.** Do *not* call this "Python
    overhead": the hardware bracket is narrow (it opens only after the hw_context lookup,
    kernel-handle retrieval and buffer-coherence work), and cProfile — whose own overhead
    is ~0.36 ms/call here — attributes its largest entry to a C call it cannot see into.
    Tracked Python work is only ~100 µs/call. The `nt.stat`×8 / `_getfinalpathname`×4 /
    `inspect._signature_from_callable`×6 **per launch** (the `artifacts_present` check in
    `callabledesign.py`, and `xrtruntime/hostruntime.py:721`) is real waste worth hoisting,
    but it is the minority. How much of the 447 µs is recoverable is **unmeasured**.
  - **Dispatch dominates everything below ~0.5 MB.** Wall time is flat 8 KB→512 KB across a
    64× payload range; transfer only becomes visible past ~2 MB. Streaming bandwidth
    12.2–13.8 GB/s.
  - **Go/no-go, to run BEFORE writing a kernel:** the op's CPU time must exceed **~617 µs**
    through IRON today, or **~170 µs** on a hypothetical zero-overhead resubmit path. Both
    are **lower bounds** — this is a no-compute passthrough, and a multi-core kernel's own
    configuration cost lands inside the hardware bracket and pushes its floor above 169.8 µs.
    One design, one data point: a floor, not a universal constant.
  - **SUPERSEDED 2026-09-09 for batchable work: ~36 µs.** The "hypothetical zero-overhead
    resubmit path" above was measured. Batched `pyxrt.runlist` submission amortises the same
    passthrough to **36.3 µs** per dispatch, 17× below the IRON figure and a quarter of the
    169.8 µs hardware bracket — so that bracket is not irreducible silicon cost either.
    Reproduced at 35.9 / 36.3 / 36.0 / 36.3 µs across four runs
    (`results/aie/dispatch_runlist_npu.log`). **It is a throughput figure**: it holds with 64
    dispatches in flight, while one unbatched call still costs ~140 µs raw or 617 µs through
    IRON, and N=1 through a runlist is slightly *slower* than a raw single dispatch. The old
    rule governs one-shot latency-critical work; ~36 µs governs anything batchable.
  - **SCOPED the same day: ~36 µs is a raw-pyxrt figure, and through IRON the batched floor is
    ~531 µs.** Batching was wired into IRON's own host path (`kernels/dispatch_floor/
    iron_batch.py`, `results/aie/iron_batch_npu.log`) rather than measured around it. The
    device half reproduces from inside IRON — 37.5–37.9 µs per dispatch at N=64, against the
    raw harness's 36.3 — but IRON's **per-call host work is a near-constant ~500 µs that
    batching never touches**, so a batched `@iron.jit` call still costs ~531 µs end to end, a
    1.26–1.37× gain rather than 17×. That ~500 µs is the same term `dispatch_floor_npu.log`
    called 447.3 µs of host-side cost, measured from a different direction and shown to be
    independent of how the submit is done. **There are three thresholds, not two:** ~617 µs
    unbatched through IRON, ~531 µs batched through IRON, ~36 µs batched through raw pyxrt.
    The 36 µs number is real but it is only available to a caller willing to give up IRON's
    argument handling. The open question is no longer dispatch — it is IRON's host path, now
    the larger term by more than an order of magnitude.
  - **THAT CALLER IS NOW WRITTEN, AND THE FLOOR IS THE DRIVER'S — measured 2026-09-09 in C++.**
    `kernels/dispatch_floor/dispatch_runner.cpp` is a standalone C++ XRT host with no Python in
    it, driving the same cache entry back to back with the Python arm in one sitting
    (`results/aie/dispatch_cpp_runlist_npu.log`). It reaches **36.7 µs** at N=64 against pyxrt's
    35.9 µs, the two agreeing within ~2% from N=4 up — so **~36 µs is a property of the driver
    and the device, not of pybind**, and no host-side rewrite goes below it. **Four thresholds
    now, none retracted:** 671.5 µs (IRON unbatched) · 498.5 µs (IRON batched 64) · ~108 µs
    (C++, one call) · **36.7 µs** (C++ batched 64), all same-design same-sitting. So the
    decision for a new small-op design is: *a C++ XRT host is the sub-100 µs path, and the only
    one*; IRON's ~500 µs is per-call work a cached-handle host pays once at startup (33–60 ms).
    Two limits stand — it needs ≥ 4–8 dispatches in flight, and a single dispatch still costs
    ~108 µs in C++, of which only ~20–30 µs was ever the binding. **Persistent runlists are not
    worth reaching for**: ~9% at N=64 end to end. Note the correction folded into that log — the
    rebuild/persistent gap is *not* runlist construction, which sits outside the timed region in
    every arm; timed properly, construction is ~18 µs fixed plus ~3.3 µs per run added and
    *grows* with the batch rather than amortising.
  - **REPRODUCED on an independent design the same day.** `ml/resnet/layers_conv2_x` (a
    3-block int8 CNN with real weights, nothing like a passthrough) reports both brackets
    from its own harness: end-to-end 2497.8 µs − hardware 1869.6 µs = **628.2 µs** of
    host-side cost, against the passthrough's 617.0 µs wall floor. Two unrelated designs
    agree to ~2%, which upgrades this from "one data point" to a reproducible constant.
    The slightly higher figure is consistent with carrying more buffers (activations plus
    three blocks' weights). See `results/aie/conv2x_int8_cpu_baseline.log`.
  - **The batched-submit path exists — verified, not measured.** `xrt::runlist` is present
    in this install's `include/xrt/experimental/xrt_kernel.h` (experimental namespace in
    XRT 2.21.75, so the signature is not stable), and **`pyxrt.runlist` is bound**, exposing
    `add` / `execute` / `wait`. N dispatches can therefore be queued and submitted once,
    which attacks both measured terms: the 447 µs host cost would be paid per batch rather
    than per call, and part of the 169.8 µs may amortize since it brackets submit+wait.
    **Nothing has been run** — this is an API-existence check and the next measurement to
    make, not a result. It does not rescue attention (see above), but it would move the
    go/no-go threshold for every future kernel.
    **RUN 2026-09-09, and both guesses in this paragraph were right.** Batching amortises a
    dispatch to **36.3 µs**, 17× below the IRON floor; and part of the 169.8 µs did amortize —
    the batched figure is a quarter of it. It still does not rescue attention stages 3 and 4
    (34 µs and 12 µs of CPU time, both under the new floor), but stage 2 at 240 µs now clears
    it 6.7×. `results/aie/dispatch_runlist_npu.log`. Getting there needed two fixes to
    `kernels/dispatch_floor/measure_runlist.py`, both of which had produced a *wrong answer*
    rather than an error: `kernel(...)` creates and **starts** a run, so runlist entries must
    be built with `pyxrt.run(kernel)` + `set_arg` instead; and the cache resolver looked for
    `*.txt` when the instruction stream is `insts.bin`, and took the newest xclbin in the
    cache rather than one belonging to this design.
- **Chained int8 CNN vs CPU — measured before building anything (2026-09-07).**
  `ml/resnet/layers_conv2_x` (3 ResNet bottlenecks chained core-to-core across 3 columns,
  int8, ObjectFifo→ObjectFifo, **one dispatch for the chain**) had run and PASSed here since
  2026-09-06 at 1888.5 µs, but nobody had measured the CPU side. It was the best remaining
  structural idea: it fixes every flaw diagnosed in `attention_bf16` — right dtype for this
  repo's XINT8 thesis, mlir-aie's own validated int8 conv kernels rather than a hand-rolled
  loop, dispatch amortized to ~25% of wall, and no two-process handoff. All five rows below
  captured in one sitting per the drift invariant
  (`kernels/conv2x_baseline/cpu_baseline.py`, `results/aie/conv2x_int8_cpu_baseline.log`).
  Workload: 1×64×32×32, stride 1, spatial 32×32 throughout, **436.21 MFLOP**.

  | | time | throughput |
  |---|---|---|
  | NPU int8, hardware bracket | 1869.6 µs | 233 GOPS |
  | NPU int8, **end-to-end** | 2497.8 µs | 175 GOPS |
  | CPU torch fp32 (8 thr) | 1856 µs | 235 GFLOPS |
  | CPU ORT CPU EP fp32 | 815 µs | 535 GFLOPS |
  | **CPU ORT CPU EP QDQ int8 (VNNI)** | **295 µs** | **1481 GOPS** |

  **Like for like, int8 vs int8, the CPU is 6.3× faster than the NPU's hardware bracket and
  8.5× end to end.** Verified the int8 row really is int8 — after ORT's own optimization the
  executed graph is 10 `QLinearConv` + 3 `QLinearAdd`, not an fp32 graph with stray Q/DQ.
  - **The CPU baseline choice nearly inverted this.** torch fp32 is 1856 µs against the NPU's
    1869.6 µs — benchmarking against torch alone (the baseline `attention_bf16` used) would
    have read as "parity", off by 6.3× from the like-for-like answer. ORT beats torch by 2.3×
    on the identical fp32 graph and its int8 path by another 2.8×. Together with the
    MobileViT splice's torch-vs-numpy 9× this is now two for two: **on this project the CPU
    kernel choice has decided the verdict more often than the NPU has.** Any NPU-vs-CPU claim
    here must name which CPU implementation it beat.
  - **Scope — closed for this design at this shape; the op class is NOT closed.** One shape
    (32²×64, 436 MFLOP) and 3 columns. The same silicon reached 895 GFLOPS on 4-column bf16
    matmul — ~4× this design's 233 GOPS — and that gap is unexplained. Two untested
    candidates this run does not distinguish: **column count** (3 vs 4, part of it but not
    all), and **spatial size / tile utilization** (32×32 gives short rows and little work per
    DMA transfer — structurally attention's problem again, which would mean the conv kernels
    underutilize the array at *this* shape, not at all shapes). The experiment that would
    settle it: run standalone `ml/bottleneck` at 32² against a larger spatial and see whether
    GOPS scales. **Do not record this as "int8 chained conv is dead."**
    *(Superseded 2026-09-07 by the spatial sweep below — the op class IS now closed, on the
    evidence that experiment asked for. The caution above stands as written for the state of
    knowledge at the time; do not read it as still-open.)*
- **int8 ResNet-bottleneck convolution on XDNA1 via mlir-aie — CLOSED, loses at every
  compilable shape (2026-09-07).** The experiment the entry above asked for, run:
  `kernels/bottleneck_sweep/sweep.py` sweeps one standalone `ml/bottleneck` on the NPU
  across spatial sizes; `cpu_sweep.py` runs the identical arithmetic through ORT QDQ int8 at
  the same shapes; both sides in one sitting per the drift invariant
  (`results/aie/bottleneck_spatial_sweep_npu.log`, Desktop 2 / Phoenix). Every NPU shape is
  checked against mlir-aie's own torch int8 golden before its timings are kept.
  - **`tensor_h` is the free axis, `tensor_w` is not.** Every L1 buffer in `bottleneck.py`
    is `tensor_w * channels` bytes; none scale with height. Sweep h, not w.
  - **Throughput scales, and nowhere near enough.** NPU hardware GOPS rises 116.7 → 143.7
    across a 16× increase in work, converging on a **marginal 146.1 GOPS** (fit
    `hw_time = intercept + slope × FLOPs`, r² = 0.99999). The CPU's marginal rate over the
    same shapes is **819.0 GOPS**. Per shape the CPU wins **7.6×** at 32×32 and **5.7×** at
    512×32 on the hardware bracket (11.4× / 6.0× end to end). 512×32 is the NPU's best point
    *and* the CPU's worst (cache-limited, its only sub-900 GOPS row) and the CPU still wins
    by 5.7×. Fixing utilization entirely buys 23% against a 570% gap.
  - **Decide on the fit, never on per-point GOPS.** Per-point end-to-end GOPS must climb
    with problem size on any accelerator as the fixed host cost amortizes — reading that
    rise as "the array scales" is a measurement artifact. `1/slope` removes fixed cost.
  - **`tensor_w` = 32 was recorded as a structural ceiling here; corrected below to 44
    once the actual blocker (a separate conv2dk1/conv2dk1_skip bug, not this buffer
    budget) was fixed — see the 2026-09-07 correction bullet further down.** 56×56,
    32×64, 64×64 and 128×64 all fail in `aiecc`, not at runtime: Tile(0,4) (the skip-add
    core) needs five buffers plus a 2560-byte stack against 64 KB of AIE2 tile memory —
    `'aie.tile' op allocated buffers exceeded available memory`, bank-aware *and* basic
    sequential allocation, 8 error lines across the 4 shapes. But four of those five
    buffers (`outOFL2L3`×2, `skip_buf`×2) scale as `w`×256 B; the fifth, `wts_buf_02`
    (the 1×1+skip weights), is sized by channels only (`input_channels/8 *
    output_channels` = constant 16384 B) and does not grow with `w` at all. The original
    "5×w×256 B" framing held exactly at `w`=64 only because 64×256 happens to equal that
    fixed 16384 B by coincidence — it overstated the ceiling everywhere else. ResNet50's
    real conv2_x (56×56) still cannot compile — the corrected ceiling of 44 is short of
    56 — so the practical conclusion is unchanged, but the stated mechanism and the exact
    ceiling value were both off.
  - **`conv2dk1.cc`/`conv2dk3.cc` read (2026-09-07): they vectorize correctly (use
    `aie::mmul`, not a scalar fallback), ruling out the attention-kernel failure mode —
    but the "8 live pipelined accumulators" read below was wrong; corrected 2026-09-07.**
    Both `conv2dk1_i8_vector` and `conv2dk3_i8_vector` (the paths `bottleneck.py` actually
    compiles — its `kernels/conv.py` wrapper passes no `-DSCALAR`) use
    `aie::mmul<4,8,8,int8,int8>`. Zero resemblance to `attention_bf16`'s zero-`mmul` scalar
    fallback — the 146 GOPS is not a coding bug of *that* kind. Placement (`Tile(0,3)`,
    `Tile(0,4)`, `Tile(0,5)` + one more) already spans 4 cores of one column, so the "~7%
    of one column's peak" framing was already accounting for multi-core, not comparing
    against a single core. **What was wrong:** the "8 live pipelined accumulators" were not
    a performance-only design choice — AIE2 spills past 5 live accumulators of that shape
    (measured 2026-09-09; this entry originally said 6 registers, see below), so
    8 concurrent accumulators is itself the correctness bug fixed below (register
    spill/pointer corruption in conv2dk3, dead remainder code in conv2dk1/conv2dk1_skip).
    The throughput gap at width 32 (where the bug never fired) reads as structural —
    dispatch overhead and short pipeline fill/drain — not something a kernel rewrite
    trivially fixes; but "8 accumulators is fine, it's just slow" was not a safe reading
    at any other width, which the fix below confirms.
  - **`conv2dk3_i8_vector` hardcodes `const int iw = 32;` (both variants, lines 449 and
    904) and ignores its own `input_width` parameter — VERIFIED as a real stride bug, not
    dead code.** `iw` is not cosmetic: it is the actual line-advance stride for `line[i] +=
    (iw * 8)` / `line[i] -= (input_channels/8) * (iw * 8)` and the matching `output +=/-=
    iw * 8` resets, used ~20 times through the function to walk input/output pointers row to
    row. Feed this kernel an `input_width` other than 32 and every one of those pointer
    resets computes the wrong offset — silent memory corruption, not a crash, and not
    caught by any assert in this function. The kernel author's own comment: *"TODO temporary
    workaround. When assigned to input_width, the results are wrong. ???"* Cross-checked
    against `conv2dk3_i8_scalar` (line 33+), which *does* use `input_width` correctly in its
    address math — the bug is vector-path-only. In practice this never fires today: the
    `aiecc` L1-memory ceiling above already rejects every width-≠-32 shape at compile time,
    before this stride bug could run. But it means fixing the L1 budget alone would not be
    enough to unlock a wider `bottleneck` — this specific upstream kernel would need its own
    fix (or independent verification) before trusting output at any width but 32. Not this
    project's file to patch; flag upstream if pursued.
  - **The conv2dk3 width fix (2026-09-07) — RESOLVED, 100% bit-exact across all widths
    (32, 36, 40, 48, 64).** Root cause identified, diagnosed via self-tagging identity
    kernels (`kernels/conv2dk3_widthfix/test_single.py`, `diag.py`), and fully resolved
    in both `conv2dk3_ui8_vector` and `conv2dk3_i8_vector` in both wheel and clone copies:
    - **Mechanism of the bug:**
      1. **Hardware accumulator limit vs compile-time unrolling:** The AIE2 vector unit has
         6 hardware accumulator registers. *(Superseded 2026-09-09. The count was inferred
         from this kernel alone and is wrong. Sweeping the live-accumulator count and
         reading the object code shows the allocator names nine accumulator registers,
         `cm0`–`cm8`, and that five live 4×8×8 int8 accumulators compile with no stack
         traffic while six is the first count that spills — `kernels/acc_spill_probe/`,
         `results/aie/aie2_isa_static.log`. The spill this bug rests on is real and the fix
         is unchanged; only the register count named here was wrong.)* Upstream mlir-aie
         attempted to fully unroll the
         middle section by 8 chunks (`acc_tmp[8]`, 8 concurrent `MMUL4x8x8` accumulators).
         At width 36, `iw_32_rem = 7` allocated 7 accumulators; at width ≥ 40, the aligned
         block allocated 8 accumulators. When > 6 accumulators are live concurrently, Peano
         LLVM spills vector accumulators to the stack (`paddb [sp], #0x580`), but Peano's
         spill/reload generation produces vector misalignments and register clobbering that
         blew away channel-block 0's leftmost 4 pixels (rotating channel 0 data into
         channels 2 and 6). When tested with `iw_32_rem = 6`, stack usage shrank to `0x120`
         (288 bytes) with zero spills and channel-block 0 was immediately 100% clean.
      2. **Upstream aligned-block pointer arithmetic bugs (`iw_32 > 0`):** Upstream's
         aligned block had never been tested or exercised before. It had three compounding
         pointer math bugs:
         - Line buffer reset was `line[i] -= 320; // (8+2)*32` instead of `(8+1)*32 = 288`,
           overshooting backwards by 32 bytes on every tap.
         - `line[i]` was never advanced by 32 pixels between `iw_32c` iterations, causing
           the core to re-read the exact same input pixels repeatedly (`[1,2,3,4,1,2,3,4]`).
         - `wtsLine[i]` was never reset per `iw_32c` iteration, reading next-channel weights.
         - Output pointer reset did `output -= ... - (iw_32 * 32)` with comment `// 32 = 4*8`,
           confusing 4 pixels with 32 pixels (`8 * 4 = 32` pixels = 256 bytes), losing 224
           bytes per oc.
    - **The unified solution:**
      - Replaced both the separate aligned and remainder loops with a single modular template
        helper: `template <typename ActType, int N> run_middle_block(...)` where `N <= 4`.
      - Middle pattern processes in blocks of 4 chunks (`N = 4`, 16 pixels per block), followed
        by a remainder block of `rem_middle_chunks` (`N` = 1, 2, or 3 chunks).
      - Because `N <= 4`, at most 4 accumulators are ever live at any moment anywhere in the
        kernel (Leftmost: 1, Middle: N ≤ 4, Rightmost: 1). Hardware registers never spill,
        stack stays at 288 bytes (`0x120`), and Peano LLVM compiles clean register-only code.
      - Fixed all pointer stride invariants: per-block line shift `+4*32`, per-block weight
        reset `-(input_channels/8)*576`, per-oc line reset `-total_middle_chunks*32`,
        per-oc weight advance `+(input_channels/8)*576`, per-oc output advance
        `+(iw*8) - total_middle_chunks*32`, and post-loop Rightmost alignment.
    - **Empirical verification (`test_width.py --shapes 32x32,32x36,32x40,32x48,32x64`):**
      - `32x32`: PASS (0.000 max |diff|)
      - `32x36`: PASS (0.000 max |diff|)
      - `32x40`: PASS (0.000 max |diff|)
      - `32x48`: PASS (0.000 max |diff|)
      - `32x64`: PASS (0.000 max |diff|)
      - All 64 channels across all rows bit-exact against PyTorch ground truth.
    - **Files updated:**
      - Wheel: `~/mlir-aie/ironenv/Lib/site-packages/mlir_aie/include/aie_kernels/aie2/conv2dk3.cc`
      - Clone: `~/mlir-aie/aie_kernels/aie2/conv2dk3.cc` (100% mirrored)
      - Both `conv2dk3_ui8_vector` and `conv2dk3_i8_vector` patched and verified. Local-only;
        never pushed upstream.
    - **Independently reproduced** in a fresh session (Desktop 2, same day): env rebuilt
      from `iron_env.ps1` by hand, `~/.npu/cache` cleared, `test_width.py` rerun end to
      end — same bit-exact result at all five widths. Log: `results/aie/conv2dk3_widthfix_npu.log`.
  - **Correction to the "they vectorize correctly" read above: `conv2dk1_i8_vector`/
    `conv2dk1_ui8_vector`/`conv2dk1_skip_i8_vector` had the same width restriction as
    conv2dk3, and worse — a dead remainder path, not a corrupted one (2026-09-07).**
    Testing intermediate widths on the already-fixed `conv2dk3` found `bottleneck.py`
    compiles fine at `tensor_w`=36/40/44 (the "5×w×256 B" ceiling above was itself an
    approximation — one of Tile(0,4)'s five buffers, `wts_buf_02`, is sized by channels,
    not width, so it doesn't grow with `w`) but produces wrong output at every one of
    them. Root cause: `conv2dk1_i8_vector`/`conv2dk1_ui8_vector` hardcode
    `iw_32_rem = 0` (a real value, never computed from `input_width`), and
    `conv2dk1_skip_i8_vector`'s remainder path was never implemented at all — a
    commented-out stub. Both guarded only by `assert((input_width/4)%8==0)`, compiled
    out in the release `aiecc` build. Any `input_width` not a multiple of 32 silently
    dropped its remainder columns from output (and, in the skip case, from the residual
    add too) — never exercised before because every prior run used `tensor_w`=32 exactly.
    **Fixed**: all three kernels rewritten to walk `input_width` in blocks of N≤4
    4-pixel chunks addressed directly from base pointers (index arithmetic, not
    incremental pointer state — the class of bug conv2dk3's aligned block had),
    keeping live accumulators within AIE2's 6 hardware registers (upstream used 8), and
    `input_width` made a compile-time constant via `-DINPUT_WIDTH` for the same reason
    conv2dk3 needed it. `conv2dk1_skip_ui8_vector` (the uint8-skip twin, not exercised
    by `bottleneck.py`) was left unfixed and is flagged, not touched. **Verified
    end-to-end** through the actual `bottleneck.py` design via
    `kernels/bottleneck_sweep/sweep.py`'s torch-golden gate: 32×32/36/40/44 all `ok=yes`
    (previously NO at 36/40/44). The *real* ceiling for this channel config (256/64/64/256)
    is `tensor_w`=44, not 32 — 45 fails only because it isn't a multiple of 4 (VMAC's
    fundamental granularity, a real hardware limit), and 46/48 exceed Tile(0,4)'s 64 KB
    (also real). This does **not** reopen the CLOSED throughput verdict above (99.1 GOPS
    at w=44, still far below the CPU's 819–1094). Log: `results/aie/bottleneck_widthfix_npu.log`.
  - **56×56 reached (2026-09-07) — the item the sweep above left explicitly open
    ("untestable without rewriting the design's buffering") is now tested, and it makes
    the verdict worse, not better.** Of Tile(0,4)'s five buffers, the final output
    ObjectFifo (`outOFL2L3`) was the one safe one to shrink — single-buffering it
    (`depth=1` instead of the default 2, in `bottleneck.py` itself, local checkout only)
    trades some throughput (compute stalls until the previous row's DMA-out drains) for
    3584 B of headroom at w=56, just enough. `skip_buf`'s depth was left alone — that one
    is load-bearing for the skip connection's timing against the conv3x3 stage's latency,
    not safe to shrink casually. **Result, both shapes verified against the torch golden:**
    at 56×56 (ResNet50's actual conv2_x, not a compile-constrained stand-in), NPU hw
    4.2435 ms vs CPU 0.3327 ms — **CPU wins 12.75×**, worse than the 5.7–11.4× range found
    at every compile-limited 32-wide shape. Marginal (fixed-cost-removed) rate: NPU 111.1
    GOPS vs CPU 1678.8 GOPS — **15.1×**. Reaching the real shape did not narrow the gap,
    it widened it. Log: `results/aie/bottleneck_w56_npu.log`. Caveat: this NPU fit is 2
    points on the w-axis, not the 5-point h-axis fit above (146.1 GOPS marginal) — not
    strictly the same quantity, and single-buffering the output costs some of that
    difference — but both read as far below the CPU regardless.
  - **Tooling:** `aiecc` needs `xclbinutil`, which is NOT in `ironenv/Scripts`. Put the XRT
    SDK directory (`/c/Xilinx/XRT/xrt_sdk/xrt`) on PATH too, or the build dies at the final
    link with `tool 'xclbinutil' not found`.
- **Correction — "628.2 µs reproduces the 617 µs dispatch floor to ~2%" was numerology
  (2026-09-07).** `results/aie/conv2x_int8_cpu_baseline.log` compared its measured *host*
  cost (wall − hardware) against the passthrough's *wall* intercept. Those are different
  quantities: 617.0 µs = 447.3 µs host + 169.8 µs hardware, so the like-for-like host floor
  is **447.3 µs** and 628.2 is 40% over it, not 2% under. The bottleneck sweep's own host
  cost runs 608–874 µs and *grows* with payload (44% across a 16× byte range), consistent
  with buffer sync charged outside the hardware bracket — so it is not a flat floor either.
  No verdict in this repo depends on the agreement. When quoting the dispatch floor, say
  which bracket: **169.8 µs hardware, 447.3 µs host, 617.0 µs wall.**
- **Attention Kernel Latency vs CPU (Negative Result):**
  - Stage 4 (8 heads): **0.86 ms** on AIE2 vs **0.012 ms** on Zen4 CPU (71× slower than CPU)
  - Stage 3 (8 heads): **4.57 ms** on AIE2 vs **0.034 ms** on Zen4 CPU (134× slower than CPU)
  - Stage 2 (8 heads): **57.61 ms** on AIE2 vs **0.240 ms** on Zen4 CPU (240× slower than CPU)
  - Full model attention (all 9 layers): **>120 ms** on AIE2 vs ~1.4 ms on CPU. NOTE: the >120 ms is a PROJECTION -- the per-stage kernel timings summed over the real block counts (2x57.61 + 4x4.57 + 3x0.86 ~= 136 ms at 8 heads), never run end to end.
  - **The Arithmetic Floor:** Stage 3 compute volume is only **2.79 MFLOP** (0.0028 GFLOP). MobileNetV2 at ~300 MFLOP was already below the NPU acceleration threshold (losing to CPU 2.68 vs 1.72 ms); MobileViT attention is ~100× smaller still. Achieved throughput is **0.61 GFLOPS** (<0.1% of array compute capability).
  - **DIAGNOSIS CORRECTED (2026-09-07) — the verdict stands, the stated cause was wrong.**
    The clause "execution time is virtually 100% dispatch, shim DMA sequence overhead and
    tile orchestration" was written before the dispatch floor was ever measured. It now is
    (`results/aie/dispatch_floor_npu.log`, `kernels/dispatch_floor/measure_floor.py`):
    **617 µs wall / 170 µs hardware** per call. Against Stage 2's measured 57,610 µs that
    is **~1%**, not ~100%. What actually happened is kernel design:
    `attention_kernels.cc` uses **`aie::mmul` zero times**, hand-rolling dot products with
    a horizontal `aie::reduce_add` per output element, so it reaches 0.61 GFLOPS on
    hardware this repo measured at **895 GFLOPS** — 0.07% of demonstrated throughput.
    ~99.9% of the gap is the kernel, not fixed cost. The row-wise FlashAttention streaming
    that fixed the 128 KB scratchpad overflow is the same edit that destroyed the
    arithmetic intensity; **do not reuse that inner loop as a template.**
    Rewriting it with `aie::mmul` still would not save it, which is why this remains a
    negative result: a *perfect* kernel at 895 GFLOPS gives Stage 2 40 µs + 617 µs =
    657 µs against CPU's 240 µs (loses 2.7×), and even on a hypothetical zero-overhead
    resubmit path 210 µs vs 240 µs is a wash; Stages 3 and 4 lose at both floors. The op
    must be ~20× larger before kernel quality is what decides the outcome.
- **Architectural Comparison & Splicing Reality:**
  - **Cut CNN Backbone (Like-for-Like):** 1.73 ms NPU vs 5.35 ms CPU (**3.1× speedup** on the identical 407-node graph). Comparing 1.73 ms against the 108 ms stock EP baseline is comparing a model fragment to a whole model; the 3.1× like-for-like is the honest figure.
  - **Full Model on NPU (with AIE Attention):** >120 ms (projected, see above), losing heavily to both the ORT CPU EP (7.51 ms measured) and the stock VitisAI EP (108.29 ms measured).
  - **Heterogeneous Splice — MEASURED 2026-09-07: 3.25 ms and 2.31×, superseding the
    reported 4.47 ms / 4.1×.** `tools/splice_wall_clock.py`,
    `results/mobilevit/splice_wall_clock_npu.log`, 100 iterations, all rows from one run.
    Backbone NPU 1.71 ms / CPU 5.65 ms (3.30× like-for-like); attention ×9 on CPU 1.41 ms
    (torch, 8 threads); splice 3.25 ms with an in-process residual of **+0.13 ms**; full
    FP32 under the ORT CPU EP 7.51 ms; stock 49-subgraph (metaDef; 58 IPU subgraphs) graph on NPU 108.29 ms.
    What the old numbers got wrong: the 0.72 ms residual was back-solved from a hardcoded
    total and is really 0.13 ms, and the 4.1× compared an ORT splice against a **PyTorch
    eager** baseline (18.37 ms). PyTorch eager measures 15.71 ms here, but ORT CPU runs the
    same FP32 graph in 7.51 ms — same-runtime, the speedup is 2.31×. What they got right:
    108.00 ms stock EP reproduced at 108.29 ms, the 1.73 ms backbone and its 407/2-node
    single-subgraph placement both reproduced.
  - **The CPU kernel choice decides the verdict, not the NPU.** The same nine attention
    blocks cost 1.41 ms under torch and 12.73 ms under plain numpy — 9×, from
    multithreaded batched GEMM and a fused softmax. Substituting numpy makes the identical
    splice 14.57 ms, i.e. **0.52× — it loses to plain CPU.** When this repo reports a
    heterogeneous-pipeline speedup it must say which CPU implementation the NPU was
    allowed to beat; against a naive one, almost anything wins.
  - **It is a cost model, not a functional pipeline, and cannot be made into one from what
    is on disk.** `mobilevit_cut_backbone_xint8.onnx` is `[1,3,256,256] → [1,1000]` — a
    complete classifier with the transformer blocks *deleted*, not a backbone that emits
    intermediates for an attention stage. There is no tensor for the CPU half to consume,
    so the halves are unconnected and the composite computes nothing valid (the quantized
    backbone is 0% top-1 alone). Building a real splice would need a genuinely cut graph
    that exposes the pre-transformer activations; that does not exist yet.
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
- **Real-data calibration was not the missing piece.** The models above are a genuinely
  different calibration from the earlier `UseRandomData=True` place-and-route probes, and
  accuracy is still ~0%. Verified rather than assumed, since a 0% score is also what a
  random-data probe gives: the old probe `models/mobilevit_xint8.onnx` is still on disk,
  and `audit_quant_grid.py` shows the two are nothing alike —

  | | random-data probe | real-data calibration |
  |---|---|---|
  | Activation scales | 0.000122 … **4.0** (32768× range, 12 distinct) | 0.0078 … 0.5 (64× range, 7 distinct) |
  | Depthwise scale grid | 0.0156 (single value) | 0.125 … 1.0 |
  | Dead depthwise channels | **0/432** | 28/432 |

  **The probe also scores ~0% with zero dead channels**, which is a third independent
  strike against the channel-death explanation: two models fail the same way, one with 28
  dead channels and one with none. The probe's failure mode is different again — Gaussian
  inputs drive activation scales to 4.0 and produce a 32768× spread no real image
  distribution justifies.
- **Provenance caveat.** Only the FP32 and full-XINT8 models are reproducible from this
  repo (`pipelines/mobilevit/1_export.py` → `2_quantize.py`, which quantizes the whole
  graph with one `get_default_config`). The two **hybrid** models were produced by a
  cut + AdaRound path that is **not committed here** — `2_quantize.py` has no cut path and
  no `include_fast_ft` wiring — so their calibration set size and AdaRound iteration count
  are reported, not verified, and the eval rows above evaluate them as found in `models/`.
  Committing that build path is the outstanding work.
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
  - **What sets Δ is open. Cross-Layer Equalization was the suspect, was measured, and is
    ruled out** (`results/mobilevit/cle_pattern_count.log`; raw Quark logs
    `results/mobilevit/cle_probe_{mobilenetv2,mobilevit}_quant.log`, line 33 of each).
    The old claim: CLE needs a positive-homogeneous activation (`f(αx) = αf(x)`), ReLU6
    qualifies and SiLU doesn't, so CLE is skipped on MobileViT and nothing bounds the
    folded-BN per-channel weight disparity that sets Δ. The measured counts are
    MobileNetV2 **3**, MobileViT-XXS **0** — the predicted direction, which makes this an
    easy thing to mis-read as confirmation. It isn't, on two grounds:
    - **Neither matched pair is depthwise.** Re-running `get_cle_pattern_pair` directly,
      the 3 entries are 2 distinct pairs (one is emitted twice), both pointwise:
      `/blocks/blocks.0/.../conv_pw` → `/blocks/blocks.1/.../conv_pw`, and
      `/blocks/blocks.6/.../conv_pwl` → `/conv_head`. CLE touches no depthwise conv in
      either model, so it cannot bound the depthwise grid in either direction. (The
      weaker supporting point: 2 pairs against 52 convs is nearly inert anyway.)
    - **ReLU6 is not positive-homogeneous.** `ReLU6(2·4)=6`, not `2·ReLU6(4)=8`; it
      saturates. The premise had ReLU6 on the wrong side of the property it invoked.
      Quark's matcher tracks the real math: `equalization.py:475` accepts only
      `Linear_node = ['Relu','ReduceMean','Pad','LeakyRelu']` between two convs and
      `break`s on anything else — `Clip` is absent, and `replace_all_clip6_to_relu`
      exists as an **opt-in approximation** (`ReplaceClip6Relu`, default `False`, never
      set in this repo) precisely because ReLU6 fails the precondition natively.
      MobileNetV2's ReLU6 exports as 35 `Clip` nodes with **zero** `Relu`, so its
      activations block the walk for the same structural reason SiLU's do. MobileViT is
      rejected twice over: `Sigmoid` is not in the accepted set, and 29 of its 36
      `Conv`/`Gemm` nodes feed *both* the `Sigmoid` and the `Mul` (that is what
      `x·sigmoid(x)` is), so the `len(...) == 1` single-consumer precondition fails before
      the op-type test runs; the remaining 6 feed `Add`/`Reshape`, also not accepted.
    The 3-vs-0 gap is therefore residual activation-free `Conv`→`Conv` adjacency
    (linear-bottleneck exits), not a statement about homogeneity. **Net: the
    discriminator is measured (depthwise grid), CLE is excluded as its mechanism, and the
    upstream cause is unexplained.** Do not fill the slot with the `HardSigmoid(34)`
    substitution visible in MobileViT's XINT8 op table — a post-hoc activation swap
    cannot set a weight scale. Note `--limit` is irrelevant to any of this: CLE matching
    is structural and runs before calibration.
- **Why AdaRound cannot fix it, mechanically.** AdaRound chooses between `floor(w/Δ)` and
  `ceil(w/Δ)`; it never changes Δ. The audit confirms the scale grid is byte-identical
  before and after AdaRound; only the dead-channel count shifts at the rounding boundary
  (28/432 → 24/432), which buys 0.00% → 0.80% top-1. **Deploying a SiLU/GELU backbone on
  XDNA1 requires QAT or per-channel scale support — no PTQ recipe reaches it.** *(Narrowed
  2026-09-09: per-channel scale support is measured to be unavailable on this EP, so QAT is the
  remaining half. [Verdict](BENCHMARKS.md#per-channel-weight-scales-are-rejected-outright-2026-09-09-desktop-2).)*
- **Rank-5 tensors never reach the NPU.** Across `mobilevit_stock`'s 1585 nodes, 207 touch a
  rank-5 tensor and **all 207 are on CPU, no exceptions** (`--rank-audit`). Control: YOLOv8's
  16 rank-4 `Slice` nodes all place on NPU, and MobileViT is the only model in this repo that
  emits a rank-5 tensor at all. Sufficient but not necessary — 18 rank-4 `Transpose`, 18
  rank-4 `MatMul` and 21 rank-3 `LayerNormalization` nodes also fall back, on op support.
  MobileViT's 5D shapes come from the unfold/fold (`[4, 256, 3, 4, 16]`) bridging its conv
  and transformer stages, so this is architectural, not a quantization artifact.

### Native Windows XRT driver constraints: KDMA and unified memory flags

Investigated via `tools/windows_xrt_driver_probe.py` (`results/aie/windows_xrt_driver_bench.log`) on Desktop 2 (Ryzen 7 8700G, Phoenix XDNA1 NPU):

- **`pyxrt.bo.flags.normal` fails on Windows**: Calling `pyxrt.bo(device, size, pyxrt.bo.flags.normal, group_id)` causes the native `amdxe.sys` kernel driver to fail with `invalid argument`. Memory on Phoenix APUs is host-managed unified RAM; buffer allocations must use `pyxrt.bo.flags.host_only`.
- **KDMA is unsupported on Windows**: Attempting to execute a kernel with buffer objects allocated on a generic or default memory group emits `[XRT] WARNING: Reverting to host copy of buffers (KDMA not supported on windows)`. The driver falls back to an expensive host bounce buffer. To achieve zero-copy execution on Windows, every BO must be allocated on the kernel argument's connected bank via `group_id = kern.group_id(arg_idx)`.
- **Sub-microsecond synchronization floor**: Unified memory buffer sync (`bo.sync`) requires 0.78-0.97 µs at sizes <= 16 KB (0.85 µs for a 4 KB frame tile). On unified APU memory, host-device synchronization is purely a CPU cache line flush (`clflushopt`) and invalidation, not a physical PCIe/DMA transfer.
- **Userspace dispatch preparation floor**: Direct userspace dispatch preparation via `pyxrt` requires 8.76 µs (1.85 µs run allocation + 6.91 µs for 8 argument bindings). Pipelining via `pyxrt.runlist` requires 3.39 µs per run.
- **Virtual hardware context capacity bound (5 columns)**: `amdxe.sys` enforces a physical ceiling of 5 active virtual hardware contexts per Phoenix device (`results/aie/windows_context_switch_bench.log`). Initial context setup requires 78.63 ms, while subsequent contexts allocate in 5.38-5.78 ms. Attempting a 6th context triggers driver failure with NTSTATUS `0xc01e0009` (hardware capacity exhaustion). Context deletion and userspace garbage collection cleanly recycles the slot in 4.98 ms.
- **Hardware context-switch penalty (~748 µs)**: Interleaving dispatches across two distinct hardware contexts on the Phoenix NPU increases mean latency from 120.25 µs to 867.99 µs (+747.75 µs penalty, 7.22x slowdown) due to partition state teardown, instruction stream flushing, and base register reprogramming. Multi-tenant concurrency therefore requires dedicated column partitioning (`1x4.xclbin`) rather than time-sliced virtualization on a single partition.

### AIE-ML systolic shift-cut bound [0, 31] — substantially retracted 2026-09-09

**Numbering warning, because two files disagree.** In `quant/shift_cut.py`, Theorem 2 is
the `pos_y` window and Theorem 3 is the position rule. In this entry, Theorem 2 is
Multi-Branch and Theorem 3 is the `pos_y` window. So "Theorem 3 is retracted" means
different things in the two files. What is retracted in the *code* is the rule that
positions must lie in `[0, 31]`. The `pos_y` window itself is not retracted; the
*projection onto it* was defective. Statements below are kept and marked, not rewritten —
this file is the record of why.

Investigated via `quant/shift_cut.py` and `python -m quant check-shift-cut`
(`results/quant/shift_cut_reaudit_20260909_desktop2.log`, which supersedes
`results/quant/shift_cut_feasibility.log`):

- **Hardware accumulator shift constraint**: On XDNA1 AIE-ML, the post-accumulator scaling unit uses a 15-bit multiplier M in [16384, 32767] and a 5-bit arithmetic right-shift register sigma in [0, 31]. The effective scaling factor is A ≈ M * 2^(-sigma). — **UNMEASURED (2026-09-09).** Both register widths are asserted; no AIE-ML or DPU ISA document in this repo states either, and AMD's AI Engine documentation describes the path as SRS (shift-round-saturate) without giving the shift field's width.
- **Theorem 1 (Shift-Cut Feasibility Bound)**: If an ONNX QDQ triad requires sigma < 0, the operation requires an arithmetic left-shift exceeding the 32-bit accumulator, resulting in immediate accumulator overflow (as observed in RegNetX-002, where sigma = -90 forces top-1 accuracy to collapse to 0.50%). If sigma > 31, the 5-bit shift register overflows/clamps (as observed in FastDepth, where sigma = 32 overflows by 1 bit, clamping to 31 and doubling layer outputs). — **BOTH EXAMPLES RETRACTED (2026-09-09).** The corrected analyzer reads RegNetX-002 at sigma min 11 / median 21 / max 30 and FastDepth at min 18 / median 21 / max 23; neither -90 nor 32 reproduces. RegNetX-002's collapse is cross-layer equalization (CLE off: 66.20% top-1 vs 69.50% FP32, no shift-cut adjustment logged; CLE on: 0.10%). FastDepth's "doubling" was never measured. **The theorem itself is untested at both edges**: fixtures built to reach sigma 0, 31 and 32 are refused by the VitisAI EP and execute on the CPU EP (`"tested_conv_on_npu": false`), so they say nothing about the DPU. The highest sigma ever executed on this hardware is **30**.
- **Theorem 2 (Multi-Branch Inter-Scale Alignment)**: In multi-branch elementwise operations (such as Bilateral Guided Aggregation in BiSeNetV2), divergent scale grids truncate dynamic range in hardware. — **NOT SUPPORTED AS STATED.** It rests on a single flagged op out of 63 in that graph and has never been tested forward; a scale-position sweep on the real BiSeNetV2 `/bga/Mul` moved correlation only in step with the signal's own standard deviation, so no scale position repairs it. Localised further on `research/windows-lowlevel`: both of that Mul's operands measure clean on their own (0.9979 and 0.9858) and its output reads 0.687, but a minimal Mul fixture at the same shape and positions is exact, and feeding the gate in as a graph input -- so the ~280-node semantic branch never compiles -- takes the same node to 0.9981 against an identical CPU reference. What distinguishes the failing case is a 512 KB activation held live across the whole semantic branch against 32 KB for its sibling, on a device with 64 KB of tile local memory. That is compiler and allocator behaviour, so no scale or rounding choice acts on it. [Localisation and the refuted hypotheses](BENCHMARKS.md#the-mul-is-not-the-fault-a-long-lived-activation-is-2026-09-09-desktop-2).
- **Theorem 3 (Systolic Scale Feasibility Window)**: In power-of-two quantization with scale positions pos = -log2(S), the output position must satisfy:
    pos_x + pos_w - 17 <= pos_y <= pos_x + pos_w + 14
  Violations are projectable to the nearest bound via `project_scale_to_feasible_basin` (demonstrated on FastDepth `Conv_96`: pos_y 0 -> 1, bringing sigma from 32 down to 31, eliminating the clamp without retraining). — **THE PROJECTION WAS DEFECTIVE (2026-09-09).** It clamped pos_x/pos_w into [0, 31] while sigma was computed from the unclamped scales, so on RegNetX-002 `/s1/b1/conv2/conv/Conv` it took an already-feasible layer at sigma = 30 to **sigma = -90**, manufacturing the overflow it exists to prevent. Repairs on real models went SESR-M7 9 -> 0, MODNet-Cut 1 -> 0, RegNetX-002 5 -> 0 once the analyzer was corrected — every one had targeted a working op. Now uses unclamped positions and raises rather than emitting a scale the analyzer would flag.
- **Rule for Project Ignition**: `python -m quant check-shift-cut` is an **advisory report, not a gate**. Run it, read it, and do not let it block or rewrite a graph: it has a demonstrated false-positive mode of the widest kind (SESR-M7 flagged 9 of 9 operations, then placed 50 / 52 nodes and tracked its CPU reference at r = 0.99912). That bound means **sigma only** — positions outside [0, 31] are advisory notes and never violations. **Do not run `--repair` on a model that already places and scores.**

### A log whose *encoding* is corrupt is fixed in place, not duplicated

Two logs arrived written as UTF-16LE behind a mangled byte-order mark
(`results/quant_fastdepth_xint8.log`, `results/aie/dpu_transaction_disasm.log`). Git treats
them as binary, `commit.sh`'s gate rejects them, and — the part that actually costs
something — every `grep`/`rg` against them silently matches nothing, which is how an
unbacked BiSeNetV2 microcode table sat unnoticed beside a log that never contained it.

**The rule: decode in place to UTF-8, scrub the profile path, and say in the commit body
that only the encoding changed.** The "never rewrite a log under `results/`" rule protects
the *content* of a measurement, not its byte encoding; a file nothing can read is not
serving as evidence. Verify by asserting the round trip (the recovered text must re-encode
to the original bytes exactly) before writing, and diff line counts across the change.

This settles a split precedent: `results/quant_fastdepth_xint8.log` was fixed in place in
merge `4308a62`, while `results/aie/dpu_transaction_disasm.log` was first handled by adding
a readable sibling and leaving the corrupt original. The sibling approach is withdrawn —
one canonical, greppable file per measurement.




