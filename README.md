# Ryzen AI XDNA1 NPU quantization pipelines

> **Research and characterization project, not a packaged tool.** All three pipelines
> are reproducible end to end, but the target is one specific chip generation, one
> specific SDK version, and Windows only — see [Compatibility](#compatibility) before
> assuming this runs on your machine as-is.

INT8 quantization and NPU deployment for **AMD Hawk Point** (Ryzen 8040-series,
XDNA1, 16 TOPS) on Windows, using AMD Quark and the VitisAI ONNX Runtime execution
provider.

Three pipelines:

| Pipeline | Model | Status |
|---|---|---|
| `pipelines/resnet50` | timm `resnet50.a1_in1k` classification | **Working** — 79.8% top-1 at 6.9 ms on the NPU |
| `pipelines/yolov8n` | YOLOv8 n / s / m object detection | **Working** — 8.9 / 15.6 / 30.8 ms on the NPU once the decode tail is cut off the graph |
| `pipelines/yolov8n-pose` | YOLOv8n-pose (17-point COCO keypoints) | **Working** — 9.8 ms on the NPU (1015/1025 nodes, `results/pose_cut_{npu,diag}.log`), OKS mAP@50-95 31.65 XINT8 vs 49.49 float (`results/map_kpts_*.log`, 500-image slice) |

The most useful part of this repository may not be the code. It is the set of facts
about XDNA1 that are undocumented, or documented incorrectly, collected under
[Key findings](#key-findings) and at greater length in
[`docs/DECISIONS.md`](docs/DECISIONS.md). [`RESEARCH.md`](RESEARCH.md) is the one-level-up
thesis — the question being asked and why — for framing before the numbers.

## How it works

1. **Export** — pull the model from timm or Ultralytics, resolve its own preprocessing
   config (never hardcoded ImageNet defaults), and export to ONNX with a static shape
   (opset 17, batch 1, no dynamic axes — see [Key findings](#key-findings) for why).
2. **Quantize** — AMD Quark converts the FP32 graph to XINT8 (power-of-two scales, the
   only scheme this backend accepts), calibrated on real images, with AdaRound as an
   optional accuracy-recovery pass.
3. **Run** — the VitisAI ONNX Runtime execution provider compiles and runs the graph on
   the NPU. `tools/diag_ep.py` reads its per-node assignment report, because nothing
   else in AMD's tooling surfaces what actually landed on-device.
4. **Measure** — latency and accuracy, always together, across every axis this repo has
   found to matter: resolution, model width, and (as a negative result) batch size. See
   [Results](#results).

---

## Repo layout

```
npu/                    Shared, Quark-free library code
  session.py            CPU/VitisAI session construction, xclbin resolution, cache clearing
  preprocess.py         Classification transform, driven by timm's own data config
  yolo.py               Letterbox, NMS, drawing, COCO class maps, head tensor names
  yolo_decode.py        The yolov8 DFL/anchor decode tail, in numpy
  paths.py              Repo-relative paths, so scripts work from any directory
pipelines/resnet50/     1_export -> 2_fetch_imagenet -> 3_quantize -> 4_run
pipelines/yolov8n/      1_export -> 2_fetch_coco -> 3_quantize -> 4_detect
                        1b_cut_head and 3b_quantize_cut are the head-cut variant
tools/diag_ep.py        Which nodes the VitisAI EP actually took, and why a model may
                        be rejected. Works on both pipelines.
scripts/                Bash wrappers for the common pipelines; lib.sh holds the
                        conda/env/logging/disk-guard helpers they share
docs/DECISIONS.md       Engineering record: locked decisions, rejected approaches, and
                        the YOLOv8 partitioning investigation
results/                Measured outputs, detection images, captured NPU logs
assets/                 Test fixtures
models/  data/          Generated. Git-ignored.
```

`models/`, `data/`, and the `modelcachekey/`, `yolocachekey/` and `yolocutcachekey/`
compile caches are git-ignored: large, machine-specific, and fully reproducible from the
steps above.

Nothing under `npu/` may import Quark. The inference scripts import that package on
every run, and importing Quark triggers a custom-op build attempt each time.

---

## Key findings

Roughly ordered by how much time each one cost to discover.

**Ryzen AI 1.8.0 ships no Phoenix xclbin, and its documentation says otherwise.**
`voe-4.0-win_amd64\` contains only `vaip_config.json`; a recursive search of the whole
install tree finds no xclbin, and the environment's site-packages carries only a Strix
`base.xclbin`. The NPU driver does drop XDNA1 xclbins into `C:\Windows\System32\AMD\`,
but 1.8's EP rejects them with `Cannot find or create target with fingerprint=...`.
Version 1.7.1 still ships `voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin`, and that is
the only firmware observed to work on this chip. Hence the two-environment split.

**Without an explicit xclbin, the compiler silently targets Strix.** It selects
`AMD_AIE2P_4x4_Overlay` and the run then dies at inference with `DPU timeout ...
ERT_CMD_STATE_ERROR`. Setting `target: "X1"` alone does *not* select the chip
architecture; the xclbin does.

**The X1 backend is XINT8 or nothing.** Power-of-two scales, MinMSE calibration,
UINT8 activations with INT8 weights. A8W8 (float scales) does not raise an error — it
just falls back to CPU, and the only symptoms are CPU-level latency and lower accuracy.
**A16W8 (INT16 activations) now measured too, not just assumed dead: same silent
full-CPU fallback** — `tools/diag_ep.py` shows 0/394 nodes on NPU
(`results/a16w8/diag_resnet50_a16w8_npu.log`), and Quark's own quantize log names the
mechanism: this repo's export is pinned to opset 17 (below), and ONNX's `QuantizeLinear`/
`DequantizeLinear` don't support 16-bit types before opset 21, so Quark routes INT16 Q/DQ
through the `com.microsoft` domain instead — which the VitisAI EP's matcher evidently
doesn't recognize at all. Quark's own config dump also shows `A16W8` never sets
`enable_npu_cnn: True` the way `XINT8` does, so this wasn't a close call. Not worth
chasing further: fixing it means bumping export opset (its own trap, see below), for a
config with no shown accuracy edge over `XINT8_ADAROUND`. **BF16 was never attempted
through Quark, and that's a toolchain gap, not a silicon one** — Quark's quantizer for
this backend doesn't expose a BF16 config to try (only
`XINT8`/`A8W8`/`A16W8`/`XINT8_ADAROUND`/`XINT8_ADAQUANT` exist), and AMD's documented
support matrix says XDNA1's *shipped CNN/LLM runtime path* is INT8-only. But the tile
silicon itself is a different question, and now has a primary-source answer rather than
a spec-sheet assumption: AMD's own `OGOAT/Collaterals/device.yaml` (bundled in this same
1.7.1 install, see `results/aie/notes_aie2_device_dtypes.log`) gives Phoenix's `AIE2`
tile spec directly — `macs_per_cycle: bfloat16xbfloat16: 128, int16xint8: 128,
int8xint8: 256`. **The array natively does bfloat16 and int16 arithmetic; the absence
from this repo's results is Quark/VitisAI-EP not exposing it, not the hardware lacking
it.** See RESEARCH.md's "Custom C++ XRT / hand-written AIE kernels" for the full
citation, and for a measured yes on whether a custom kernel reaching those paths is
buildable at all: not through this SDK's own `aiecompiler` (a missing `physical_device.dll`
blocks it, confirmed absent from every AMD distribution channel checked), but through the
open-source `mlir-aie`/Peano toolchain instead — a hand-written kernel compiled and run
correctly on this machine's XDNA1 hardware, natively on Windows, no gated access required.
That path has since produced a kernel for this repo's own gap: a bf16 GroupNorm(32)
(`kernels/groupnorm_bf16/`) standing in for the `InstanceNormalization` that
`resnetv2_50x3_bit` leaves on CPU, measured on the real node tensors at 1535 μs vs the
CPU's 3472 μs per call for the largest shape, and a win on 4 of its 6 shapes (33 of 49
nodes) — but a follow-up measurement of the two-process handoff a real splice needs
found that floor alone (789 μs-23.6 ms/call) erases every one of those wins; see
"Pushing width further" below and `results/aie/groupnorm_bf16_kernel_npu.log` /
`results/aie/groupnorm_bf16_handoff_floor_npu.log`.

**Silent CPU fallback is the failure mode to watch for.** The `[Vitis AI EP]` banner,
`Target architecture:`, `Compile done.` and the operator table print **only during
compilation**, never on a cache load, so their absence in a normal run means nothing
by itself. Two reliable checks: the cache directory should contain a
`compiled.*.xmodel` (ResNet's does, YOLO's does not), and NPU latency should be several
times better than CPU. If it is not, you are running on the CPU. **Caveat found later
(MobileNetV2, see "Model family" below): that second check is a heuristic, not a
guarantee** — a genuinely engaged NPU (347/349 nodes, confirmed via
`vitisai_ep_report.json`) still lost to plain CPU (2.68 ms vs 1.72 ms) on a model cheap
enough that per-call dispatch overhead outweighs the compute saved. The report file is
still the only real evidence; "NPU should be faster" stops being a safe proxy once the
CPU number itself is in the low single-digit milliseconds.

**Always pass `--fresh` when changing model or xclbin.** The compile cache is keyed by
a hardcoded `cacheKey`, not by a model hash, so a stale entry is reused silently and
you end up debugging a wrong-architecture artifact.

**Export with opset 17, static batch 1, and the legacy exporter.** `dynamo=True` (the
default on torch >= 2.9) silently emits opset 18 even when asked for 17; the version
converter's assertion only warns. Dynamic batch axes inject `Shape`/`Concat`/`Reshape`
at the graph tail, which are prime CPU-fallback candidates. Static export yields a
clean `GlobalAveragePool -> Flatten -> Gemm`.

**Preprocessing must match calibration exactly**, which is why it lives in exactly one
place. The ResNet transform is driven by timm's `resolve_data_config` and written to
`models/preprocess_config.json` at export time — this checkpoint uses `crop_pct=0.95`,
not the 0.875 you would get by hardcoding ImageNet defaults.

**Quark's config object is a dataclass**, so assigning a misspelled option succeeds
silently and does nothing. Verify attribute names before trusting that an option took
effect:

```powershell
python -c "from quark.onnx.quantization.config import get_default_config as g; print([a for a in dir(g('XINT8')) if 'exclude' in a.lower()])"
```

**ImageNet-1k on Hugging Face is parquet, not tarballs.** The old
`data/val_images.tar.gz` path returns 404. Validation is 14 shards of roughly 480 MB
and 3570 rows each, in random class order, so a single shard already covers about 974
classes. Read it with `pyarrow.parquet.ParquetFile.iter_batches` and write
`image['bytes']` straight to disk. The `datasets` library pulls in more than a
gigabyte of Arrow and torch just to import, and a non-streaming `load_dataset` wants
150 GB.

**Neither accelerator is free — both carry a per-call floor.** MobileNetV2 (1.72 ms on
plain CPU) lost to both DML (3.19 ms) and a genuinely NPU-engaged run (2.68 ms,
347/349 nodes). ResNet50 and yolov8n's CPU baselines are 10-15x larger, and that is
exactly the regime where the NPU wins — this project's advice to reach for XDNA1 was
always implicitly scoped to "a model heavy enough that a few ms of dispatch overhead
is noise," not to every classifier a CPU already runs comfortably. See "Model family:
MobileNetV2 vs ResNet50" above.

**"Heavy enough" is necessary but not sufficient — the op set matters too.**
`resnetv2_50x3_bit` has a 470.79 ms CPU baseline (no dispatch-overhead excuse available)
and still loses to the NPU, by 25% (588.58 ms), because only 1010/1271 nodes (79.5%)
place on the NPU: every `InstanceNormalization` node in it falls to CPU, confirmed via
`tools/diag_ep.py`, not assumed. Every clean NPU win in this repo shares BatchNorm-only
normalization, which folds into the preceding Conv's weights at export and so never
reaches the graph as its own node. A non-fused normalization layer (GroupNorm,
LayerNorm, InstanceNorm) is a measured gap in this EP's NPU kernel coverage, not a
hypothetical one. See "Pushing width further: `resnetv2_50x3_bit`" above — including the
hand-written bf16 kernel that now beats the CPU fallback for that op on 33 of the 49
nodes, per node, with the whole-model splice still unmeasured.

---

## The YOLOv8n blocker, and how it was solved

For a while the quantized YOLOv8n ran at 37–39 ms on the NPU — indistinguishable from
CPU FP32 — while producing detections whose confidences were visibly shifted from FP32,
so the INT8 model clearly *was* executing. Something was running it, just not the NPU.

**Diagnosis.** The VitisAI EP writes an assignment report to
`<cacheKey>/vitisai_ep_report.json` on every session build — unlike the compile log,
which prints only on an actual compile. `tools/diag_ep.py` reads it. Against the working
ResNet50 pipeline:

```
yolov8n_xint8.onnx                    resnet50_xint8_adaround.onnx
  all           965 nodes               all            395 nodes
  CPU           298                     NPU            393     <-- no NPU entry for YOLO
  VITIS_EP_CPU  667                     VITIS_EP_CPU     2
```

No `NPU` entry at all, and every one of the 965 nodes marked `device: "CPU"`. The EP had
registered, walked the graph and claimed *nothing*. That is wholesale rejection rather
than bad partitioning — there is no partition — which is why no compile log, no operator
table and no `compiled.*.xmodel` were ever produced.

**Cause: the float decode tail.** The last 18 nodes of a YOLOv8 export are the DFL and
anchor-decode arithmetic, kept in float during quantization. Their presence made the EP
refuse the entire graph rather than partition around them.

**Fix: cut them out and decode in numpy.** `1b_cut_head.py` rewrites the model so its
outputs are the six raw detection convolutions (233 nodes → 209), and
`npu/yolo_decode.py` reproduces the removed tail. The result:

```
  yolov8n_cut_xint8.onnx
    all           929 nodes
    NPU           922            <-- 99.2% of the graph
    VITIS_EP_CPU    7
```

The seven CPU nodes are only the boundary conversions: the input `QuantizeLinear` and
the six output `DequantizeLinear`s — structurally the same as ResNet50's two.

| configuration | device | inference | note |
|---|---|---|---|
| full graph, FP32 | CPU | 37.0 ms | |
| head-cut, FP32 | CPU | 31.8 ms | |
| full graph, XINT8 | "NPU" | 39.1 ms | EP took 0 nodes; this was CPU |
| **head-cut, XINT8** | **NPU** | **8.7–9.8 ms** | **922/929 nodes, ~110 fps** |

Post-processing costs 0.16 ms, because the numpy decode filters on class logits before
the DFL softmax rather than decoding all 8400 anchors. Sigmoid is monotonic, so this
selects exactly the anchors NMS would have kept — verified identical, with and without
the filter.

**What was *not* the cause**, each checked rather than assumed:

- **Not the SiLU rewrite.** Quark's `enable_npu_cnn` turns `Sigmoid`+`Mul` into
  `HardSigmoid`+`Mul` (57 of them) and lowers `Split` to `Slice`. This was the leading
  suspect, since none of those ops appear in the ResNet50 graph. The report settles it:
  the EP put all 57 `HardSigmoid`, all 16 `Slice`, both `Resize` and all 13 `Concat` on
  the NPU. Every one of them is supported.
- **Not a misspelled Quark attribute.** `subgraphs_to_exclude` is a real field on the
  legacy `QuantizationConfig`, is consumed by `quantize.py`, and raises rather than
  no-ops when the subgraph doesn't match.
- **Not a missing xclbin.** The blocked run passed the correct Phoenix 4x4 xclbin from
  the 1.7.1 install. (`npu/session.py` now refuses to build an NPU session without one
  regardless — it previously warned and continued, which is a CPU run wearing an NPU
  costume.)
- **Not ORT's constant sharing.** The working ResNet report shows the same merged scalar
  initializers.

Reproduce the whole thing with `./scripts/yolo-cut.sh`, which cuts, quantizes, sanity-
checks on CPU, runs on the NPU and reads the report back.

> **Calibration caveat.** The model above was calibrated on only 32 images,
> because the point of that run was whether the EP accepts the graph — which calibration
> quality has no bearing on. Its detections drift from FP32 accordingly. For a model you
> would actually ship, re-run with `--limit 300 --adaround` and measure COCO mAP; expect
> real accuracy loss from the SiLU-to-hard-swish substitution on top of INT8 rounding.

---

## Known limitations

- **One chip generation, one SDK version.** Everything here targets Hawk Point/Phoenix
  (XDNA1) via Ryzen AI 1.7.1 specifically. Strix (XDNA2) uses a different xclbin and
  compiler target and has not been touched; 1.8.0 cannot run inference on this chip at
  all (no Phoenix xclbin — see [Key findings](#key-findings)).
- **Windows only.** XDNA1 has no Linux userspace; a WSL run is silently CPU-only rather
  than an error, which is the kind of failure that's easy to miss.
- **The full-graph YOLOv8 model is refused by the EP, on purpose left that way.** It's
  kept in the repo as the control proving the head-cut fix is real — see
  [The YOLOv8n blocker](#the-yolov8n-blocker-and-how-it-was-solved) — not a bug to fix.
- **Static batch >1 is unsafe on this backend, not just slow.** Measured: it silently
  drops every batch element after the first rather than raising an error (see
  [Batching](#batching-does-it-help-throughput)). Batch 1 only.
- **AdaRound needs more RAM than the 13.8 GB laptop has for YOLOv8s/m at 640×640** —
  FastFinetune's memory high-water mark is layer 0, the only layer at full resolution,
  and it takes SIGSEGV rather than raising there. Not a hard wall, though: both s and m
  now have AdaRound results (see Roadmap), quantized on Desktop 1's 32 GB + GPU-accelerated
  FastFinetune (`--device`). l/x AdaRound at 640×640 remains untried anywhere.
- **yolov8l's full-dataset eval is flaky.** Two of three 5000-image mAP attempts hit a
  hardware `DPU timeout` mid-run; the third, and a standalone 500-image run, completed
  cleanly with NPU memory flat throughout (ruling out a simple leak). Root cause
  unresolved — see the width section. Not seen on any other model size.
- **NPU utilization can't be measured through standard Windows tooling.** The `GPU
  Engine`/`GPU Adapter Memory` performance counters exist but can't see this device at
  all — the NPU registers as a `ComputeAccelerator`, not a WDDM GPU adapter, so it never
  shows up as an adapter LUID for those counters to poll, regardless of sampling rate.
  `xrt-smi examine -r aie-partitions` (bundled with the driver, `C:\Windows\System32\AMD\`)
  is the tool that actually sees it — live per-context memory (MB) and compute rate
  (GOPS) — and is what the concurrency measurements above use for memory. Its GOPS
  column was tried as a utilization signal too and turned out to be a dead end (scales
  linearly with stream count, decoupled from measured throughput); see the
  [GOPS section](#two-cameras-does-independent-concurrency-work-where-batching-doesnt) above and
  [Roadmap](#roadmap).
- **No formal test suite.** Verification here is empirical (`compileall` + import checks
  as a syntax gate, then real pipeline runs read from `results/`) rather than unit tests
  — there's no fixture NPU to test against in CI.

## Roadmap

Open measurements, roughly in the order they would resolve. `RESEARCH.md` carries the
reasoning behind each.

- **`pipelines/yolov8n-pose` end to end on the NPU — done.** Head-cut partitions
  1015/1025 (99.0%), 9.8 ms/frame, same clean pattern as detect. XINT8 costs 17.8
  points of OKS mAP@50-95 (49.49 → 31.65, a 36% relative loss — proportionally worse
  than bbox yolov8n's plain-XINT8 loss). AdaRound is untried for pose and is the
  obvious next lever, same as it was for detect.
- **AdaRound for YOLOv8s at 640×640 — done, and it barely helps.** Not blocked after
  all: `models/yolov8s_cut_xint8_adaround.onnx` compiles and runs (15.5 ms/frame,
  922/929 nodes). Full 5000-image mAP@50-95 is 39.98 against plain XINT8's 37.40
  (`results/map_yolov8s_cut_xint8_adaround_npu.log`) — 2.6 points, not the 90%
  recovery AdaRound gets on ResNet50/wide_resnet50_2. A 500-image slice run first
  suggested 45.19, which would have been a very different story; the full 5000 is
  the number to trust, consistent with this README's other slice-vs-full warnings.
  Worth understanding why detection AdaRound recovers so much less than
  classification's before spending the RAM on YOLOv8m/l/x or the wide ResNets.
- **AdaRound for YOLOv8m at 640×640 — done, and it recovers even less.** Quantized on
  Desktop 1 (GPU-accelerated FastFinetune, `--device`) and run on Desktop 2's XDNA1
  (Phoenix): `models/yolov8m_cut_xint8_adaround.onnx` runs at the same 30.46 ms/frame as
  plain XINT8 (1216/1223 nodes — no latency cost from AdaRound, only the weight rounding
  changes). Full 5000-image mAP@50-95 is 45.32 against plain XINT8's 43.49
  (`results/map_yolov8m_cut_xint8_adaround_npu.log`) — **+1.83 points**, a smaller
  absolute recovery than yolov8s's +2.58 despite m's much higher starting accuracy,
  extending the pattern that AdaRound has less room to recover as width increases.
  Caveat: this model arrived via Syncthing with no local log of its calibration count,
  so it isn't a clean like-for-like comparison against the calib-64 plain-XINT8 row —
  the exact "a model can arrive with no log explaining it" risk this repo's own
  machine notes warn about.
- **Concurrent streams past 2, and on a wider model — done.** Both saturate:
  yolov8n flattens at 3 streams (~167 fps, 2.1×); yolov8m, which uses more of the
  array per call, saturates a stream earlier at 1.29× (`results/nstream_*.log`).
  Table in the [Two cameras](#two-cameras-does-independent-concurrency-work-where-batching-doesnt)
  section above.
- **Is stream saturation compute or memory — done, it's compute.** `tools/session_hold.py`
  + `xrt-smi examine -r aie-partitions` show NPU memory scaling linearly with stream
  count on both models (~29 MB/stream yolov8n, ~100 MB/stream yolov8m), no ceiling
  through 8 streams — decoupled from the throughput plateau, which rules memory out.
  `results/nstream_memory_yolov8{n,m}.log`.
- **Does classification show the same saturation shape, and does accuracy survive
  contention — done, yes on both.** resnet50 saturates at 1.60× by 8 streams (between
  yolov8n's 2.13× and yolov8m's 1.29×) and holds flat through 16, with the *exact* same
  per-image predictions as the uncontended baseline at every stream count — not just
  similar top-1, bit-identical argmax on all 960 concurrent classifications tested.
  `tools/nstream_cls_bench.py`, `results/nstream_resnet50.log`. Open: `wide_resnet50_2`
  or a bigger classification model untested; >16 streams untested.
- **yolov8l and yolov8x — done.** 49.67 ms / 45.37 mAP@50-95 (l) and 117.11 ms / 45.09
  (x) — the width trend that held cleanly through m flattens here; x is pure extra cost
  for less accuracy than l. Calibrated smaller (32/24 images) than m's 64, a caveat in
  the same vein as m's own. l's full eval is also flaky in a way nothing smaller is —
  see [Known limitations](#known-limitations). Table in the width
  section above.
- **AdaRound across the ResNet50 resolution sweep.** The sweep is plain XINT8, and
  AdaRound's recovery could move where the accuracy peak sits.
- **A yolov8m mAP row at calibration 200**, so the detection width table is
  like-for-like at every size (the current 43.49 was calibrated on 64 images). Same
  caveat now applies to l (32) and x (24).
- **A full utilization-vs-TOPS story using `xrt-smi`'s GOPS column — done, and it's a
  dead end.** GOPS scales exactly linearly with stream count (9×/80× per stream for
  yolov8n/yolov8m) with no ceiling through 8 streams, while the same run's *measured*
  completion rate is flat from 1 stream on — decoupled from real throughput, so it
  can't be turned into a utilization-vs-16-TOPS number. `results/gops_yolov8{n,m}.log`.
  `xrt-smi`'s memory readout (used above) remains the one number from this tool that
  tracks something real; GOPS does not.
- **Explain ResNet50's AdaRound latency cost** (5.63 → 6.93 ms). The EP report shows the
  same 393 / 2 partition for both models, so extra CPU fallback is ruled out; a
  `--fresh` re-run of each and a diff of the two reports would settle it.
- **The webcam path (single `4x4.xclbin` session, `./scripts/yolo-demo.sh`) has not
  been exercised end to end.** The related but distinct round-robin-across-4-columns
  demo *has* — see
  [A live demo](#a-live-demo-does-the-multi-partition-finding-hold-on-a-real-webcam)
  above: camera-bound at 30 fps through n/m/l, genuinely NPU-bound (22.0–23.5 fps) at x.
- **5th AIE column on this Phoenix chip — done, and it's a dead end.** `1x4.xclbin`
  caps at 4 independent partitions regardless of process count; a 5th process shares
  column 4 rather than getting its own. The driver's `5x4_*.xclbin` overlays fall back
  silently to 100% CPU (fingerprint mismatch). See
  [Splitting the array into independent partitions](#splitting-the-array-into-independent-partitions)
  above and `docs/DECISIONS.md`.
- **Longer term:** a detector fine-tuned for fixed camera feeds (licence-plate
  recognition), reusing the head-cut + XINT8 + AdaRound recipe rather than re-deriving it.

## Acknowledgements

Quantization configuration and the AdaRound parameter set are adapted from AMD's Ryzen
AI `CNN-examples/object_detection` samples. Models are timm's `resnet50.a1_in1k`,
`wide_resnet50_2.racm_in1k` and `wide_resnet101_2.tv2_in1k`, and Ultralytics YOLOv8.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Measurements on
hardware not available here are especially useful: Strix, more RAM for the
AdaRound-blocked configurations, more disk for yolov8l/x. There is no unit-test suite,
because the hardware cannot be faked; changes must pass the syntax and import gate in
`CONTRIBUTING.md` and come with a logged measurement under `results/` for anything
behavioural.

## License

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). You're
free to use, study, modify, and share this code — including commercially — but any
version you distribute, or run as a network-accessible service, must also be
open-sourced under the AGPL. This choice follows from a dependency, not a preference:
`pipelines/yolov8n/1_export.py` imports `ultralytics` directly, and Ultralytics'
YOLOv8 code is itself AGPL-3.0 — matching that license here removes the ambiguity of
combining AGPL and permissively-licensed code in one repo, rather than leaving it
unresolved.

This covers the pipeline code only. The ResNet50, YOLOv8n, ImageNet and COCO artifacts
carry their own upstream licenses — none of them are redistributed by this repo
(`models/` and `data/` are git-ignored and regenerated locally).
