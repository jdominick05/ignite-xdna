# RESEARCH — what this repo is trying to find out

> Companion to `README.md` (results, quickstart) and `docs/DECISIONS.md` (engineering
> record — every decision, rejection, and non-obvious fact). This file is neither: it's
> the thesis — the question being asked, why, and how far the answer has gotten.
> Update it when the question changes, not when a number does.

## The question

**Can a real, unmodified consumer NPU — AMD's XDNA1 in the Hawk Point chip shipped in
consumer laptops, 16 TOPS on paper — actually run useful CNN vision workloads, end to end,
using only what AMD ships?** Not a datacenter accelerator, not a reference board: the
NPU sitting idle in an off-the-shelf Ryzen 8040-series machine.

That's a narrower question than it sounds, because the honest answer going in was
"unclear." AMD's own documentation is thin, occasionally wrong (the 1.8.0 SDK's docs
describe a Phoenix xclbin that doesn't exist in that install — see LOCKED DECISIONS #1
in `docs/DECISIONS.md`), and none of it says what actually happens when a real graph hits the
VitisAI execution provider. The vendor support matrix rules out the flashy target
(LLMs — no BF16, no NLP path on this silicon) almost immediately, which leaves CNN
inference as the entire remaining scope. This project is the process of finding out,
empirically, whether that scope is real: whether "CNN INT8 only" means "a batch of toy
examples work" or "arbitrary CNN architectures can be made to work, at a cost you can
measure and reason about."

## Why this, why now

The hardware was already on hand, already idle, and already the subject of vendor claims
(16 TOPS, "AI PC") that don't come with a way to verify them. This is ground-up
exploration of a piece of hardware, not the build-out of a planned application — the
guiding principle throughout has been to prefer a generalizable measurement (does
latency track pixel count? does accuracy scale with model width?) over a fix
narrowly scoped to make one demo run. Findings here are meant to outlive any single
model.

There is a longer-range motivation sitting behind the pure characterization work: an
eventual detector fine-tuned for fixed camera feeds (licence-plate recognition), feeding
a separate downstream project, was named early on as the kind of thing this capability
would eventually be used for. Nothing in this repo commits to that path yet — no
labeled dataset for it exists — but it's the reason detection (not just classification)
was worth pursuing at all once classification proved the hardware was real.

## The method: two pipelines, each answering a different half

**ResNet50 classification** was the first target because it's the simplest possible
falsification test. A stock timm checkpoint, standard preprocessing, one well-known
accuracy number to check against (80.4% published top-1). If this doesn't work, nothing
harder will. It worked, cleanly: 79.8% top-1 at 6.9 ms on the NPU with AdaRound, 393 of
395 graph nodes accepted by the EP. This pipeline is now the **known-good control** —
whenever something is unclear about the YOLO pipeline (is a partition bad? is the
xclbin wrong?), the question becomes "does ResNet50 still show its normal 393/395
split?" If yes, the problem is specific to the other graph, not the environment.

**YOLOv8 detection** was the harder, more realistic test: a graph the EP had never seen
that includes operations (DFL decode, anchor math) with no analog in a classifier. It
initially failed completely — the EP claimed **zero** nodes, silently falling back to
CPU while reporting "NPU" in the logs. Diagnosing *why* a graph is refused outright,
with no error and no reason given anywhere in AMD's tooling, was the real research work
of this half of the project (see `docs/DECISIONS.md`, "The YOLOv8 partitioning failure",
for the full investigation). The answer — cut the float decode tail out of the graph and do it
in numpy instead — is now a repeatable pattern (`1b_cut_head` → `3b_quantize_cut`), not
a one-off workaround: it took the EP from 0/965 nodes to 922/929, and the model runs at
8.9 ms, faster than the classifier despite doing much more work per frame.

## What's actually been learned (not just measured)

The numbers live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). What's worth stating as findings, because they
generalize beyond any one model:

1. **Graph shape matters more than graph size, and the EP gives you nothing to debug
   it with.** A model can be entirely rejected — not partitioned badly, rejected
   wholesale — and the only ground truth is `vitisai_ep_report.json`, which the
   tooling doesn't surface anywhere obvious. `tools/diag_ep.py` exists because nothing
   else reads that file.
2. **Quantization has a real, structural accuracy cost on this backend** (Quark's
   `enable_npu_cnn` rewrites SiLU to hard-swish — an architecture change, not just
   rounding), **and AdaRound recovers most but not all of it.** ResNet50: 71.7% → 79.8%
   against an 80.4% ceiling. This is a repeatable recovery technique, not a lucky
   result on one model.
3. **This NPU is not compute-bound at the model sizes tested, but `xrt-smi`'s GOPS
   column cannot tell you by how much — it's a per-context notional figure, not a
   delivered-compute measurement, and every absolute number this repo previously read
   off it is retracted, not revised.** `tools/gops_sweep.py`
   (`results/npu_utilization_gops.log`) found GOPS ratios that track FLOPs almost
   exactly across yolov8n→s→m (9→29→80 against 3.29×/9.1× FLOPs ratios) — which reads
   like a real measurement, but is equally what a *compile-time* value derived from
   the xmodel's own op count would look like, since that also scales with FLOPs. Two
   independent checks settled which one it is, in the same direction: GOPS scales
   exactly linearly with concurrent stream count while measured completions/s stays
   flat in the same run (`tools/session_hold.py`, `results/gops_yolov8{n,m}.log`); and
   reading `xrt-smi examine -r aie-partitions`'s *full* report, not just its GOPS and
   memory columns, shows every session this repo has ever built under the `4x4.xclbin`
   overlay — any thread count, any process count — lands on one
   `Partition Index: 0, Columns: [1, 2, 3, 4]`. Two independent processes on that
   partition were measured to exactly halve each other's throughput (~149/s solo →
   ~76/s each concurrent, combined ~152/s): a single physical resource being
   time-sliced. That is also the actual mechanism behind the concurrency-saturation
   finding below, which previously named "compute headroom" only by eliminating
   memory as the cause — the real cause is one shared partition, not abstract
   headroom. No %-of-16-TOPS number from this repo's earlier GOPS readings should be
   cited going forward.
4. **Splitting the array into independent per-column partitions beats the single
   4x4 partition on combined throughput — a real way to use more of this NPU, with a
   real caveat attached.** `1x4.xclbin`, bundled with the SDK right next to
   `4x4.xclbin` and unused by this project until now, claims only one column per HW
   context. `tools/multi_partition_bench.py` (`results/multi_partition_yolov8n.log`)
   confirmed via the same full partition report that N independent OS processes
   against it get N *separate* `Partition Index` entries on N different columns —
   not the single shared partition the 4x4 overlay always produces — with throughput
   measured across all N held in a common, confirmed-active window (every process
   started, `xrt-smi` polled until all N contexts report Active, only then does the
   clock start), not reconstructed from staggered runs with uncontrolled overlap.
   Result on yolov8n: 67.2 fps solo, scaling to 245.2 fps at N=4 (3.65×), zero
   cross-talk at every process count checked the same known-positive/known-negative
   way as every other concurrency tool in this repo. 245.2 fps beats the single 4x4
   partition's own ~151–172 fps ceiling by roughly 1.4–1.6×, at the cost of
   per-inference latency roughly doubling (~15 ms/call on one column vs 4x4's
   ~6.6 ms/call) — a genuine throughput/latency trade for independent-stream
   workloads, not free capacity. **Caveat that must travel with this result**: AMD
   deprecated `1x4.xclbin` starting Ryzen AI 1.5 ("no longer supported and should not
   be used" — checked against AMD's published release notes 2026-09-06); it still
   compiles and runs correctly on the 1.7.1 install this project depends on, but this
   is unsupported territory, not a sanctioned configuration, and nothing guarantees
   it survives a future driver or SDK update. Measured on Desktop 2 / Phoenix only —
   the laptop's Hawk Point chip may have a different column count and has not been
   tested.

   **Checked whether this scales to a 5th column — it doesn't, on this install.**
   `xrt-smi examine -r platform` reports **Total Columns: 5** on Desktop 2's Phoenix
   chip, one more than the 4 assumed above. Extending the sweep to `--procs 1 2 3 4 5`
   against the same `1x4.xclbin` shows why that 5th column was never in play:
   `xrt-smi`'s own partition report at N=5 still lists only 4 distinct partitions —
   the 5th process shares column 4 with the 4th (both drop to ~38 fps, half of solo
   rate) rather than getting an independent 5th context, so combined throughput barely
   moves (246.4 → 257.2 fps). The driver's own 5-column overlay family
   (`5x4_*.xclbin` under `C:\Windows\System32\AMD`, not shipped in the 1.7.1 SDK's own
   xclbins folder) was tried directly as the obvious next step and is a dead end here:
   all three versioned candidates build and run without error and produce output that
   matches CPU, but **every node reports `device: CPU`** in
   `vitisai_ep_report.json` — a hardware-target fingerprint lookup fails silently
   (`target_factory.cpp: Cannot find or create target with fingerprint=0x...`) and the
   whole graph falls back to CPU, not the NPU. See `docs/DECISIONS.md`'s "Rejected
   approaches" for the full write-up. The physical 5th column is real; reaching it
   would need a different driver/firmware revision or the ahead-of-time AIE-compiler
   flow this repo doesn't use, not more xclbin-hunting on this install.

5. **A real achieved-ops/s number now exists, replacing every retracted GOPS
   citation — and it names a config well past anything measured before.**
   `tools/estimate_tops.py` computes `TOPS = MACs_per_inference × 2 × fps`, with MACs
   read from `onnx-tool`'s static analysis of the actual on-NPU (head-cut) graph, not
   Ultralytics' published FLOPs — the head cut removed the decode tail, so the real
   per-inference NPU work is strictly less than the published figure. Every fps input
   is a hardcoded citation of an already-measured, already-logged number; the script
   only does the arithmetic. Solo (one session, the shared 4x4 partition), achieved
   TOPS climbs with model size and then turns over: resnet50 9.3%, yolov8n 6.6%,
   yolov8s 11.9%, wide_resnet50_2 15.1%, yolov8m 16.5%, **yolov8l 21.2% (the solo
   peak)**, yolov8x 14.0% (down from l — the same width-stops-paying-off turn already found for
   latency and mAP in the status table's "Width beyond yolov8m" row, now visible in
   achieved compute too). Splitting
   across 4 independent `1x4.xclbin` columns (finding 4 above) does not just add a
   constant multiplier — it changes which model wins: yolov8m scales to 3.77× at N=4
   (`results/multi_partition_yolov8m.log`, 33.1% of nameplate, beating every solo
   config including yolov8l), and yolov8l scales to 3.86× (`results/multi_partition_
   yolov8l.log`, **39.3% of nameplate, the best number this repo has measured** —
   6.29 of the nameplate 16 TOPS). yolov8x still scales to 3.87× on 4 columns (`results
   /multi_partition_yolov8x.log`, 38.0%) — its solo regression does not reappear here,
   so it was specific to contending for the whole array, not a property of the model
   itself. The per-column vs. shared-4x4 latency ratio (14.9/8.9≈1.66× at n, 57.8/30.8
   ≈1.88× at m, 103.1/49.7≈2.07× at l) rises with model size but stays well under the
   4× a fully compute-bound single column would show at every size tested — meaning
   even yolov8l/x leave real per-column headroom unused. **Tested directly: yolov8x
   re-exported at 1280² (imgsz doubled, 524.1 GMACs/inference — 4.0× the 640² model's
   131.1 GMACs, confirming the resolution scaling) still lands on 39.31% of nameplate
   at N=4 (`results/multi_partition_yolov8x_r1280.log`, 6.0 fps combined, zero
   cross-talk), statistically identical to yolov8l's 39.32%. Two model/resolution
   combinations 6.2× apart in per-inference compute converging on the same figure is
   evidence of a real per-column ceiling near 39-40% of the 16 TOPS nameplate for this
   overlay/toolchain, not a still-open "keep climbing" lever — the remaining headroom
   the latency ratio implies is not being left on the table by an insufficiently heavy
   model; something else (fixed per-call dispatch cost that does not shrink relative
   to a longer compute-bound call the way it should, or a ceiling in how the compiler
   schedules a single column) is capping it. This calibration-thin (`--limit 4`, plain
   XINT8) 1280² model was built purely to test this ceiling — it has no measured
   accuracy and should never be cited for mAP.** Caveat from building it: quantizing
   at 1280² with the repo's usual `--limit 64` spooled the machine's calibration cache
   to 169 GB and drove free system RAM to 0.44 GB before being killed — `--limit 4`
   (this is a throughput probe, not an accuracy claim, so a thin calibration set costs
   nothing real) kept the peak comfortably below the 32 GB ceiling. **Standing "~1.1 of 16 TOPS" citations
   for yolov8n solo elsewhere in this repo (status table, README) are retracted by
   this same number, not merely superseded** — 6.6%, not "~1.1", is yolov8n's correct
   solo figure; the old phrasing predates this script and should not be repeated.
   Same caveats as finding 4 travel with every 1x4 number here (deprecated overlay,
   Desktop 2 / Phoenix only); yolov8l/x additionally carry the known DPU-timeout
   instability seen on 2 of 3 full 5000-image mAP attempts (finding in the status
   table below) as an open risk on longer runs, though the ~12s throughput windows
   used here did not trigger it.

   **Follow-up that closes the yolov8m/yolov8l gap, but weakens rather than confirms
   the obvious "raw MACs/call" story — no clean second variable survives it.** Two
   more points were added to fill the gap between yolov8m (33.1%) and yolov8l (39.3%):
   `yolov8s` split across 4 columns at its native 640² (`results/
   multi_partition_yolov8s.log`, zero cross-talk, the established calibration recipe,
   no accuracy caveat) reaches only **24.8%** (14.93 GMACs, 132.7 fps combined) — and
   `yolov8s` re-exported at 1280² (`results/multi_partition_yolov8s_r1280.log`, same
   weights, 59.66 GMACs — 4.0× the 640² model's, confirming the resolution scaling
   again) reaches only **28.0%**, *below yolov8m's 33.1% despite more raw GMACs/call
   (59.66 vs 40.6)*. Raw compute-per-call is not the predictor: yolov8s@1280 has more
   of it than yolov8m and still lands lower. Checked directly against this repo's own
   exported graphs (`onnx.load` on each `*_cut.onnx`, counting `Conv` nodes and reading
   each first/last Conv weight's initializer shape) rather than assumed from
   Ultralytics' published width/depth multipliers:
   | model | Conv nodes | first-conv weight | GMACs/call (max tested) | best split-4 % |
   |---|---|---|---|---|
   | yolov8n | 63 | [16,3,3,3] | 4.70 | 14.4% |
   | yolov8s | 63 | [32,3,3,3] | 59.66 (@1280) | 28.0% |
   | yolov8m | 83 | [48,3,3,3] | 40.6 | 33.1% |
   | yolov8l / yolov8x | 103 | [64,3,3,3] / [80,3,3,3] | 524.1 (x@1280) | 39.3% |

   An earlier draft of this finding read the 63/83/103 node-count split as depth
   *setting* the ceiling. That overshoots what these three points support: within the
   single 63-node class, achieved-% nearly doubles (14.4% → 24.8% → 28.0%) — a bigger
   swing than the 28.0%→33.1%→39.3% gap *between* the depth classes. `yolov8n` and
   `yolov8s` share the same node count and differ by channel width only, so this table
   shows achieved-% rising with **both** width (within a depth class) and depth
   (across classes) — but every stock YOLOv8 variant scales both together along with
   total MACs, so **no pair here isolates depth from width**, or either from raw
   compute. Building a synthetic model to isolate them cleanly (custom-scaled
   untrained architecture, or graph surgery duplicating a block) was considered and
   rejected: random-initialized weights make Quark's calibration meaningless — this
   session already found that even *real* weights under thin calibration
   (`yolov8s@1280` below) produce a model whose correctness silently collapses, and a
   synthetic graph has no working oracle to catch the same failure, so its fps
   wouldn't be citable under this repo's own evidence standard. The cheaper, already-
   available check is a second architecture family with different confounds: see the
   ResNet-depth follow-up two entries below, which reuses already-built, already-
   quantized models (`resnet50`, `wide_resnet50_2`, `wide_resnet101_2`) through
   `multi_partition_cls_bench.py` with no new export, quantize, or RAM risk.

   **That follow-up ran, and it undercuts the depth reading rather than confirming
   it.** `resnet50` (50 layers, narrow) reaches 18.8% split-4 (`results/
   multi_partition_resnet50.log`, 354.6 fps, 4.24 GMACs, zero mismatches);
   `wide_resnet50_2` (50 layers — same depth, wider) reaches 26.4% (`results/
   multi_partition_wide_resnet50_2.log`, 181.3 fps, 11.65 GMACs, zero mismatches);
   `wide_resnet101_2` (101 layers — double the depth, similar width to
   `wide_resnet50_2`) reaches 28.5% (already measured, `results/
   multi_partition_wide_resnet101_2.log`). The width match was checked directly
   (`onnx-tool` shape inference on both quantized graphs' Conv output tensors, not
   assumed from torchvision's `width_per_group` documentation): the two graphs'
   per-stage channel counts are byte-identical at every one of 53 shared stages
   (`[64,128,256,...,2048]`, same list both models), differing only in node count
   (53 vs. 104) — this genuinely is a width-matched pair, the only one collected
   this session. `wide_resnet50_2` vs. `wide_resnet101_2` is that pair, and **doubling
   depth there buys only +2.1 points (26.4%→28.5%)**, a far weaker relationship than
   YOLO's node-count table suggested (63→83 nodes moved 24.8%→33.1%, +8.3 points). One
   precision this pair does *not* buy, though: its GMACs/call also roughly doubled
   alongside depth (11.65→23.18) — width is held fixed, but compute and depth still
   move together, so this is "doubling depth-and-compute at fixed width" vs. nothing,
   not a depth-alone isolation either. Meanwhile width at *matched* depth is a real,
   comparably-sized effect in both families:
   `resnet50`→`wide_resnet50_2` (50 layers, width only) is +7.6 points on 2.75× the
   GMACs; `yolov8n`→`yolov8s` (63 nodes, width only) is +10.4 points on 3.2× the
   GMACs. **The honest conclusion across both families: achieved-% correlates clearly
   with channel width at matched depth; its correlation with depth at matched width is
   much weaker in the one family that actually isolates it.** The strong-looking
   depth pattern in the YOLO node-count table most likely reflects Ultralytics' width
   and depth scaling moving together (and total MACs/call rising with both), not an
   independent depth effect — this finding does not carry over to a family where depth
   was checked on its own. Depth is downgraded from "leading hypothesis" back to
   "correlated but not shown to be causal here"; channel width is the better-supported
   correlate of the two, in both families, though neither is established as the
   mechanism (vs., e.g., total GMACs/call itself, which also rises with width in every
   pair above and hasn't been cleanly separated from it either).

   **Checked whether this repo's own tooling can explain *why* width correlates with
   achieved-% — it can't, and this is where that question stops rather than moves to
   speculation.** `vitisai_ep_report.json`'s `nodeStat` (a per-node list already
   partly used for the padding check above) carries each node's op type, I/O tensor
   shapes, and a `device: NPU|CPU` field — a static compile-time routing decision, not
   a cycle count, timing, or tile/lane assignment. Checked on the freshest available
   builds (`yolocut1x4cachekey`, which held `wide_resnet50_2`'s report after the run
   above): 393/395 nodes route to NPU, only 2 to `VITIS_EP_CPU` (boundary Q/DQ) —
   effectively the same near-100% NPU routing already seen for every classification
   and YOLO model in this repo (929-node YOLO graphs: 922 NPU/7 CPU). Every model
   tested, narrow or wide, routes almost entirely to the NPU, so this file cannot be
   distinguishing "narrow models fall back to CPU more" (they don't) from any
   tile-level utilization story — and it carries no data at the tile/lane granularity
   that would test one. No file this repo's own toolchain writes exposes that level of
   detail. Rather than reach for AMD's published AIE tile-count/topology figures — a
   spec sheet has no way to be checked against which specific layers of *these*
   compiled graphs landed on which tiles, so a topology narrative built from it would
   be unfalsifiable with what's on hand — **this thread stops here**: achieved-%
   correlates with channel width across two architecture families, more weakly with
   depth, and more weakly still (not cleanly separable) with total MACs/call; *why*
   width is the strongest correlate is an open hardware question this repo's tooling
   cannot currently answer, not a mechanism this repo has established.

   **One piece of that thread reopened, though: not *why* width correlates, but
   *where* the missing time goes, which `nodeStat`'s static routing could never show
   and a new tool can.** `tools/percall_overhead_bench.py`
   (`results/percall_overhead_yolov8_1x4.log`) turns on ORT's own profiler
   (`session_options.enable_profiling`) on a single held-open `1x4.xclbin` (one
   column) session and reads the duration ORT itself reports for the fused
   `vitis_ai_ep_*_kernel_time` node — the entire on-NPU subgraph as one op — separately
   from the full per-call `model_run` duration. This distinguishes the two candidate
   explanations finding 5 above left open ("fixed per-call dispatch cost that doesn't
   shrink proportionally... or a limit in how the compiler schedules a single column
   ... nothing measured so far distinguishes which"): **dispatch/sync overhead outside
   the compute node is negligible at every size tested — 0.7% of wall time at yolov8n,
   shrinking to 0.1% at yolov8l** (consistent with a small, roughly-fixed number of
   microseconds, not a cost that fails to amortize). Essentially all wall time (95.8%
   to 99.5%) is the `vitis_ep` node's own reported duration, and *that* duration's
   efficiency against an ideal 4-TOPS column (16 TOPS / 4 columns) rises monotonically
   with model size — 19.0% (n) → 29.3% (s) → 36.5% (m) → 41.3% (l) — closely matching
   the combined-4-column figures already measured by a completely independent method
   (`multi_partition_bench.py`'s multi-process throughput: 14.4% / 24.8–28.0% / 33.1% /
   39.3%). Two different measurements — solo per-call node timing here, combined
   multi-process throughput there — land on the same numbers. **The dispatch-overhead
   hypothesis is now ruled out, not merely left undistinguished: the ~39-40% ceiling
   and its rise with width live entirely inside the compiled kernel's own scheduled
   execution on a single column, a real compiler/scheduling limit, not a fixable
   host-side cost.** Going further than this would need instruction/tile-level
   profiling this repo's toolchain does not expose — the same wall the paragraph above
   already hit from the `nodeStat` side, reached again here from the timing side.

   **The `yolov8s@1280` fps number above needed its own correctness check, and the
   check found a real (separate) defect worth naming plainly.**
   `multi_partition_bench.py`'s built-in cross-talk oracle (alternate a known-positive
   and known-negative image across workers, flag any crossed result) flagged **every
   single call as a mismatch, at every process count including N=1** where no
   concurrency is even possible — the opposite of what cross-talk would look like
   (cross-talk needs ≥2 contending workers). Traced with the cheapest available check
   (an independent `--ep cpu` comparison, the same pattern already established in
   finding 7): the FP32 cut model correctly finds class 0 in the positive image and not
   in the negative one at 1280²; the **XINT8 model reproduces the same false positive
   on the negative image on plain CPU**, with no NPU or concurrency involved at all.
   This is a genuine under-calibration accuracy defect from the deliberately thin
   `--limit 4` calibration recipe (chosen for RAM safety, see the caveat above) meeting
   a narrower/shallower architecture than the `yolov8x@1280` probe that established the
   same recipe safely — not a hardware fault. Q/DQ scale corruption does not change
   node count or MAC count, so the fps figure is unaffected and still usable for the
   ceiling question; the model's *correctness* is not, and — like `yolov8x@1280` before
   it — it must never be cited for mAP or detection accuracy. Filed as a minor tooling
   gap rather than fixed: `multi_partition_bench.py`'s cross-talk heuristic does not
   distinguish "fails identically solo" from "fails only when contended," and a future
   revision should gate on the N=1 case before attributing anything to concurrency.

   **The weight-bandwidth-roofline hypothesis raised when this ceiling was first
   measured is weakened, not ruled out — the pair that looked like a clean test of it
   wasn't.** `tools/estimate_tops.py` now also reports MACs per INT8 weight-byte
   (`onnx-tool`'s `graph.params`) alongside MACs per graph-memory-byte
   (`graph.memory`, its sum of every tensor's byte size — a proxy for data movement,
   not a measurement of actual DDR traffic, since the compiler may keep tensors
   on-chip). The `yolov8l` (1931 MACs/weight-byte, 39.3%) vs. `yolov8x@1280` (7685
   MACs/weight-byte — 4.0× more reuse of the same weight bytes, still 39.3%)
   comparison originally cited as ruling this out is **confounded**: MACs/call and
   weight-intensity moved together by the same 4.0× (both are driven by the same H×W
   resolution term for a fixed graph), so that pair cannot separate "more spatial
   reuse didn't help" from "more compute per call didn't help." The pair that actually
   discriminates was already in the table: **`wide_resnet101_2` (182.8 MACs/weight-byte,
   23.2 GMACs) reaches 28.5%, vs. `yolov8m` (1568.5 MACs/weight-byte — 8.6× more reuse
   per weight byte, but only 1.75× the GMACs/call) reaches 33.1%.** An 8.6× spread in
   weight-arithmetic-intensity producing only a 4.6-point spread in achieved-% is a
   real weakening of a weight-bandwidth-roofline story (if weight streaming from DDR
   every call dominated, that 8.6× gap should show up as a far larger effect than 4.6
   points) — but it is not a clean refutation, since `yolov8m`'s higher GMACs/call is
   itself a live confound in this pair too. No experiment run so far isolates
   weight-intensity from GMACs/call cleanly; the depth finding above is the stronger,
   better-isolated result from this round.

   **Checked the channel-padding hypothesis once, cheaply, with no new hardware
   time.** If the compiler pads channel counts to a tile-granularity boundary, the
   array executes more MACs than the graph nominally contains, and "39% of nameplate"
   would understate how busy the silicon actually is. `yolocutcachekey/
   vitisai_ep_report.json` (written by the EP on every session build, read here for a
   model already compiled from an earlier run — no new build needed) includes a
   `shapeInfo` list naming every compiled tensor's shape; `yolov8n`'s first conv weight
   appears as `[16, 3, 3, 3]`, exactly the nominal, unpadded channel count from the
   graph itself. One data point is not proof padding never happens elsewhere in the
   graph, but it found no evidence for it where checked, which weakens (does not
   confirm or rule out) padding as the explanation for the gap between 39% useful ops
   and 100% busy silicon.

   **Denominator checked, not just inherited: AMD's 16 TOPS nameplate is confirmed
   correct for this specific chip, not a mobile-Phoenix figure carried over by
   mistake.** Web search on 2026-09-06 confirms both machines' NPUs are independently
   rated 16 TOPS INT8 by AMD/reviewers: the Ryzen 7 8700G (Desktop 2, this machine)
   at 1.6 GHz XDNA clock, and the laptop's Ryzen 5 8645HS (Hawk Point). The 10 TOPS
   figure belongs only to the original Ryzen 7040 mobile series at its lower mobile
   NPU clock — not to either machine this repo measures on. Every "% of 16" number in
   this repo divides by the correct figure for the hardware it was measured on.
   **Update 2026-09-07: the clock is now measured, not searched** — 1.80 GHz in the
   `default` power mode on Desktop 2, 0.80 in `powersaver`, 1.03 in `balanced`
   (`results/aie/clock_probe_npu.log`). The 16 TOPS nameplate is what all 20 cores do
   at 1.6 GHz; at the measured clock the array is 18.4 TOPS and the 16 reachable cores
   14.7. The %-of-nameplate figures here divide by 16 TOPS and are unchanged; the
   physical-ceiling reading is in `docs/SILICON.md` section 2.

   Consequences below follow from the width of the compute-bound margin (still real,
   independent of the retracted GOPS number — see the concurrency and width findings
   throughout this repo), and both have been measured rather than assumed:
   - **Width is nearly free — confirmed on a second architecture and a second task,
     and it keeps paying off further out.** yolov8s costs 3.2× the FLOPs of yolov8n
     for only 1.75× the latency, and yolov8m pushes a third step further: 9.1× the
     FLOPs of yolov8n for 3.46× the latency (30.80 ms, 1216/1223 nodes). The full
     5000-image mAP now confirms the accuracy side holds too: 43.49 mAP@50-95 —
     comfortably past yolov8s's 37.40. Measured against its own FP32 CPU
     baseline (204.38 ms), yolov8m is 6.6× faster on the NPU — and this ratio itself
     grows with model size (3.8× at n, 5.3× at s, 6.6× at m), because CPU cost tracks
     FLOPs roughly linearly while the NPU absorbs the extra width into idle lanes, so
     the gap widens the bigger the model gets. The quantization accuracy penalty also
     *shrinks* as the model gets wider (−27% relative for n, −16% for s). Classification
     shows the same pattern at two width steps: `wide_resnet50_2` (2.7× resnet50's
     params, same depth, isolated width) costs 1.73× the latency; `wide_resnet101_2`
     (4.96× resnet50's params, deeper as well as wider) costs 3.15× the latency — if
     anything a *better* ratio at the larger step. Partitioning stays clean at every
     step measured (393/395, 393/395, 767/769, 1216/1223). And `wide_resnet101_2` at
     plain XINT8, no AdaRound, reaches 80.2% top-1 — beating resnet50's own 79.8%
     AdaRound headline. Three architectures, two tasks, three-to-four width steps,
     same shape of result: a wider model is close to a strictly better trade here
     than a narrower one tuned harder.
   - **Resolution has a floor that shrinking the input doesn't touch — for detection.**
     ~2.4 ms of every YOLOv8n inference is fixed cost (weight streaming, DMA, per-layer
     invocation) independent of pixel count — 27% of the 640px run. The same probe on
     ResNet50 (`scripts/resnet-res.sh`) found a smaller, still-real fixed cost (2.0 ms,
     22% of its largest measured run) and a steeper compute-scaling term — this graph
     is more compute-bound, so shrinking its input is a more effective latency lever
     than it was for YOLO. It also surfaced something YOLO's probe couldn't, because
     detection accuracy doesn't move much with resolution: **top-1 accuracy peaks
     above the training resolution** (256px beats the native 224px), a FixRes-style
     effect, then falls again at 288px — not a monotonic trade.
6. **The full-graph YOLO model is kept in the repo on purpose, not cleaned up.** It's
   the control that still fails — proof that the head-cut result is the fix, not an
   artifact of measuring differently.
7. **Batching is not just inefficient here, it's dangerous — and the exact mechanism
   is now isolated, not just observed.** A genuinely static (non-dynamic) batch-2
   export of resnet50 was tested to check whether "batch 1 only" was a measured
   constraint or an inherited assumption. It's measured now: the EP accepts a
   *fragment* of the graph (80/395 nodes; Conv itself falls entirely to CPU), runs to
   completion with a plausible-looking latency (71.6 ms/image, 12.6× the batch-1
   number), and the pooled accuracy (36.0% top-1) is deceptively close to "everything
   a bit wrong" — until split per within-batch position. Batch slot 0 scores 72.00%
   top-1, matching batch-1 within noise; slot 1 scores ~0%, and its raw output is
   **identical across every image tested regardless of input** — a stale or
   uninitialized buffer that the graph never writes, not a miscomputation. Pure
   `--ep cpu` on the same weights gets 76.0% (both slots correct), confirming the
   model and export are fine. This is a sharper version of the A8W8 finding (locked
   #3 in `docs/DECISIONS.md`): silent CPU fallback is at least numerically correct;
   batch>1 here only fills the first slot and leaves the rest stale. The practical
   rule: never trust a batch>1 result on this backend without an independent
   `--ep cpu` check, split per batch index — a pooled number can look like ordinary
   degradation and hide a per-slot failure entirely.
8. **Width and resolution don't trade off cleanly — the obvious combined-lever
   hypothesis was tested and falsified.** If a wider model at lower resolution beat a
   narrower model at higher resolution on both axes, that would be the practical
   configuration rule for a throughput-sensitive application. Measured
   `wide_resnet50_2` at 160²/224²/288² against `resnet50`'s own measured peak (256²):
   resnet50@256² beats every `wide_resnet50_2` configuration on **both** latency and
   top-1 at once, including the cheapest one (160²). The reason traces back to finding
   3 above: `wide_resnet50_2`'s top-1 gain over resnet50 at matched resolution was
   already a wash in this recipe (only top-5 improved), so there was no real accuracy
   edge left to buy back by shrinking its resolution. This doesn't kill the width×
   resolution idea in general — it says the trade needs a width step that clearly
   improves top-1 in isolation, which only `wide_resnet101_2` did here, and that step
   confounds width with depth and recipe, so it isn't a clean input to this
   particular test.
9. **Whether the fixed per-inference cost is a hardware or graph property is still
   open — the three-point check was inconclusive.** Fit the same
   `ms = a + b × Mpixels` model to `wide_resnet50_2` (2.7× resnet50's width): the
   marginal cost `b` nearly tripled (65.10 → 180.94 ms/Mpixel), a solid result
   consistent with a wider graph doing more arithmetic per pixel. The intercept `a`
   came out lower, not higher or the same (1.88 ms vs 2.63 ms) — but on only three
   points with residuals of 0.7-1.3 ms against a ~2 ms intercept, that comparison
   isn't trustworthy either way. More resolutions per model would be needed to
   settle whether the fixed cost is hardware-invariant.
10. **"Quantization penalty shrinks with width" does not generalize — it was a fact
   about yolov8n→s, not a law of width.** Tested the prediction directly: AdaRound at
   matched calibration (64 images) for resnet50 (loss −8.00, recovers 90.0% of it) and
   `wide_resnet50_2` (loss −9.50, recovers 90.5% of it). The wider model's initial
   quantization penalty was *larger*, not smaller, and AdaRound's recovery efficiency
   was essentially identical between them — both predictions from the YOLO finding
   were wrong for this pair. Useful side effect: AdaRound's RAM wall (which blocks it
   for YOLOv8s+ at 640×640) turned out to be specific to that resolution, not to model
   width — both AdaRound runs here, one at 2.7× the width, finished fine on the same
   ~4-5 GB-free development machine, because layer 0 (AdaRound's memory peak) scales
   with input resolution, not with the bottleneck width that changes between these two
   models.

## Status per pipeline

| Pipeline | Question it was built to answer | Status |
|---|---|---|
| ResNet50 | Does *any* real CNN reach the NPU with acceptable accuracy? | **Answered: yes.** 79.8% top-1, 6.9 ms. Serves as the ongoing control. |
| YOLOv8 (full graph) | Does a harder, decode-tail-included graph reach the NPU? | **Answered: no**, and understood why. Kept as a control, not a bug to fix. |
| YOLOv8 (head-cut) | Can the same model reach the NPU with the decode moved off-graph? | **Answered: yes.** 8.9–15.6 ms depending on width, beats FP32 on both axes at width s. |
| Resolution sensitivity (classification) | Does ResNet50 have the same fixed-cost floor YOLO does, and does accuracy survive moving off the training resolution? | **Answered, 128–384px measured.** Smaller fixed cost (2.6 ms vs YOLO's 2.4, but a smaller share — 23% vs 27% — of a slower run), more compute-bound, and accuracy peaks above training resolution at 256px rather than at it. The 288px turn is a real ceiling, not noise: everything above 256px is strictly worse on both latency and accuracy at once. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#resnet50-input-resolution-does-the-fixed-cost-story-hold-for-a-classifier). |
| Width sensitivity (classification) | Does "width is nearly free" (found on yolov8n→s) generalize past YOLO, and does it hold at a bigger step? | **Answered: yes, at two steps.** `wide_resnet50_2` (isolated width, 2.7× params, 1.73× latency) and `wide_resnet101_2` (deeper+wider, 4.96× params, 3.15× latency, 80.2% top-1 beating resnet50's AdaRound headline with plain XINT8) both hold. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#model-width-does-width-is-nearly-free-hold-for-a-classifier-too). |
| Width sensitivity (detection) | Does the yolov8n→s ratio hold at a third, bigger width step? | **Answered: yes, latency and accuracy both.** yolov8m: 9.1× the FLOPs of yolov8n for 3.46× the latency (30.80 ms, 1216/1223 nodes), 43.49 mAP@50-95 past yolov8s's 37.40 (calib 64, not the 200 n/s used — noted as a caveat, not a confound). Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together). |
| Batch efficiency | Is a static batch >1 an efficient way to use this NPU? | **Answered: no, and it's unsafe in a specific, isolated way.** Batch 2 on resnet50 drops the EP's partition to 80/395 nodes, costs 12.6× the per-image latency, and only writes the first batch slot — slot 0 scores 72.00% (correct), slot 1 scores ~0% with input-independent output. Do not batch on this backend. |
| Width×resolution interaction | Does a wide model at low resolution beat a narrow model at high resolution on both axes? | **Answered: no, falsified.** resnet50@256² (6.32 ms, 74.00% top-1) beats every `wide_resnet50_2` configuration tested (160²/224²/288²) on both axes at once. Traces back to `wide_resnet50_2` never having a real top-1 edge at matched resolution to begin with. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#width-and-resolution-together-does-a-wide-model-at-low-resolution-beat-a-narrow-model-at-high-resolution). |
| Fixed-cost intercept: hardware or graph property? | Does the ~2.6 ms fixed cost stay fixed at 2.7× the width? | **Inconclusive.** Marginal cost triples with width (65.10 → 180.94 ms/Mpixel, solid). Intercept comes out lower, not the same or higher (1.88 ms vs 2.63 ms) — but on only 3 points, not a trustworthy comparison either way. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#is-the-fixed-per-inference-cost-a-hardware-property-or-a-graph-property). |
| AdaRound at width | Does AdaRound recover less on a wider model, as the YOLO n→s pattern predicts? | **Answered: no, the prediction was wrong.** resnet50 recovers 90.0% of its quantization loss, `wide_resnet50_2` recovers 90.5% — essentially identical, and the wider model's initial loss was actually larger, not smaller. At `wide_resnet101_2`, plain XINT8 is already so resilient (-1.80% top-1 loss) that AdaRound has little top-1 gap to close (79.90% vs 80.20%), while recovering top-5 from 92.90% to 93.90% (closing 66.7% of the gap to 94.40% FP32) at 16.64 ms (4.97× CPU speedup). Also settled: AdaRound's RAM wall is resolution-specific (640×640), not width-specific — both fit fine at 224². `wide_resnet50_2`+AdaRound (80.10% top-1, 9.66 ms) remains this repo's best classification speed/accuracy point. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#adaround-at-width-does-it-recover-less-on-a-wider-model). |
| AdaRound for YOLOv8s at 640² | Is this actually RAM-blocked, and does it recover as much as classification's ~90%? | **Answered: not blocked, and recovers far less.** `yolov8s_cut_xint8_adaround.onnx` compiles and runs fine (15.5 ms, 922/929 nodes) — the earlier RAM-wall note didn't hold. Full 5000-image mAP@50-95 is 39.98 vs plain XINT8's 37.40 — 2.6 points, nowhere near classification's ~90% recovery. A 500-image slice first suggested 45.19, another instance of the slice-vs-full trap this repo already flags elsewhere. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#adaround-on-detection-how-much-does-it-actually-recover). |
| AdaRound for YOLOv8m at 640² | Does AdaRound's recovery keep shrinking as width increases past s, matching the n→s quantization-penalty trend? | **Answered: yes, recovers even less in absolute terms.** Quantized on Desktop 1 (GPU-accelerated FastFinetune) and run on Desktop 2's XDNA1: 45.32 mAP@50-95 vs plain XINT8's 43.49 — +1.83 points, smaller than s's +2.58 despite m's much higher baseline accuracy, and no latency cost (30.46 ms, same as plain XINT8's 30.80). Caveat: the quantized model synced in with no local log of its calibration count, so it isn't a clean like-for-like comparison against the calib-64 plain-XINT8 row. `results/map_yolov8m_cut_xint8_adaround_npu.log`. |
| Concurrent camera streams | Is a static batch>1 the only way to ask this NPU for more than one image at once, and does it fail the same way? | **Answered: no — two independent sessions on two threads is a different request than batching, and it works.** 1.8-1.9× the combined throughput of round-robin, zero cross-talk between streams (checked directly, the same way the batch-2 bug was found, not assumed away). Consistent with yolov8n only reaching about 6.6% of the array's 16 TOPS solo (`tools/estimate_tops.py`, finding 5 above — retracts this row's earlier "~1.1" figure) — there's headroom for a second stream. `tools/dual_stream_bench.py`, `results/dual_stream_{pose,detect}.log`. |
| Concurrent streams beyond 2, and at width | Does the multiplier keep climbing past 2 streams, and does a wider model with less idle headroom get the same multiplier? | **Answered: no on both counts, and the two answers explain each other.** yolov8n's combined throughput saturates at 3 streams (~167 fps, 2.1×) — headroom runs out, concurrency doesn't stop working. yolov8m, which already uses more of the array per call, saturates a stream earlier at a much smaller 1.29×. Zero cross-talk at up to 8 concurrent streams on either model; no throughput regression past the ceiling. `tools/nstream_bench.py`, `results/nstream_{yolov8n,yolov8m}.log`. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#two-cameras-does-independent-concurrency-work-where-batching-doesnt). |
| Is stream saturation compute-bound or memory-bound? | Concurrent sessions each hold their own runtime buffers — is the throughput ceiling actually a memory ceiling in disguise? | **Answered: memory, not the cause.** `xrt-smi examine -r aie-partitions` (the one tool found this session that can actually see NPU memory — Windows' `GPU Engine`/`GPU Adapter Memory` counters can't, the device is a `ComputeAccelerator`, not a WDDM GPU adapter) shows memory scaling linearly with stream count on both yolov8n (~29 MB/stream) and yolov8m (~100 MB/stream), climbing cleanly through 8 streams with no ceiling — well past the 2-3 stream point where throughput already flattened. Memory and throughput are decoupled, ruling memory out and leaving compute headroom as the standing explanation. `tools/session_hold.py`, `results/nstream_memory_yolov8{n,m}.log`. **Confirmed on a classifier too**: `wide_resnet101_2` at 1/8/16/32 streams scales exactly linearly at ~140.6 MB/stream (reaching 4.5 GB by 32 streams) with completions/s flat at ~62 fps throughout — same decoupling, third model family. `results/session_hold_wide_resnet101_2.log`. |
| Does classification show the same concurrency shape; does accuracy survive contention past a binary found/not-found check? | Every prior concurrency check only confirmed a binary ground truth (found a person, or didn't) — does real accuracy hold under N-way contention, and does classification saturate the same way detection does? | **Answered: yes on both, cleanly.** resnet50 saturates at 1.60× by 8 concurrent streams (between yolov8n's 2.13× and yolov8m's 1.29×, its own idle-headroom budget) and holds flat through 16 with zero regression. Every concurrent stream ran the identical labeled slice the solo baseline used, so predictions could be diffed exactly rather than compared as an aggregate top-1 a different sample could move on its own — result: bit-identical argmax on all 960 concurrent classifications tested (16 streams × 60 images), at every stream count. `tools/nstream_cls_bench.py`, `results/nstream_resnet50.log`. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#two-cameras-does-independent-concurrency-work-where-batching-doesnt). |
| Width beyond yolov8m: does the n→s→m trend continue at l/x? | Two width steps (n→s, s→m) both bought a clear mAP gain for extra latency — does l→x keep paying off? | **Answered: no, the trend breaks.** l: 49.67 ms, 45.37 mAP@50-95, 1510/1517 nodes. x: 117.11 ms, 45.09 mAP@50-95, same 1510/1517 nodes (l/x share architecture depth, only channel width differs). x is 2.36× the latency of l for a net *loss* in mAP — width alone stops paying off somewhere around l under this recipe (plain XINT8, calib 24-32). Also found: yolov8l's full 5000-image eval failed with a hardware DPU timeout on 2 of 3 attempts, memory confirmed flat (231 MB) during the runs that succeeded — ruling out a simple leak, root cause still unresolved, not seen on any other size. `results/map_yolov8{l,x}_cut_xint8_npu.log`, `results/yolo_cut_{l,x}_{cpu,npu,diag}.log`. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together). |
| Can `xrt-smi`'s GOPS column build a utilization-vs-16-TOPS story? | GOPS is the one other live NPU-side reading `xrt-smi` exposes besides memory — does it track real compute headroom running out the way the throughput ceiling does? | **Answered: no, it's a dead end.** `tools/session_hold.py` extended to parse GOPS and independently count actual completions/s in the same run. GOPS is exactly `9 × streams` (yolov8n) / `80 × streams` (yolov8m) with zero saturation through 8 streams, while measured completions/s is flat from 1 stream onward in the same run — decoupled from real throughput. Simplest explanation: `xrt-smi` credits each context a notional per-context GOPS figure blind to shared-array contention, not a measurement of delivered compute. `results/gops_yolov8{n,m}.log`. Table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#direct-npu-utilization-what-gops-actually-says). |
| Is there any live NPU utilization or clock reading at all, xrt-smi or not? | `xrt-smi`'s GOPS is a dead end (row above), HWiNFO's native NPU rows are placeholders, and every live number this repo had came from polling `xrt-smi`. | **Answered: yes, two, neither from xrt-smi.** Windows' own GPU-engine statistics track the NPU as an MCDM adapter: `\GPU Engine(pid_*_luid_0x00000000_0x0000d6bf_*_engtype_compute)\Utilization Percentage` reads 84–88 % across an IRON GEMM run and 0 % idle, with the adapter's shared memory equal to xrt-smi's figure, while xrt-smi's own GOPS/FPS/latency read `N/A` for that context. And XRT's `max_clock_frequency_mhz` query is a live readback: 800 MHz idle, 1800 MHz while a hardware context is active, ~0.05 ms per read — and it follows the power mode too: busy it reads 1800 `default`/`performance`/`turbo`, 1028 `balanced`, 800 `powersaver`, the trace-unit clocks to the MHz, while idle it reads 800 in every mode (`results/aie/pmode_clock_readback_npu.log`, both axes in one sitting; this reconciles it with the clock probe's "800 in every mode", an idle reading). Both drive `tools/hwinfo_npu_bridge.exe` now (`results/aie/xrt_api_live_clock_and_pdh_npu.log`; `docs/SETUP.md`). Not yet confirmed under a VitisAI EP session — three attempts to hold one failed for unrelated reasons, so the earlier "0 % under a VitisAI session" note in the bridge's docstring stands untested either way. |

## Open questions

These are the threads this project would pull next, roughly in the order they'd
resolve, carried forward from earlier engineering notes and updated for what has since
been closed:

- **Resolution vs. accuracy for classification — done, 128–384px.** 256px is the
  measured peak of the speed/accuracy frontier under plain XINT8 (74.00% at 6.32 ms);
  everything above it is strictly dominated. AdaRound across the resolution range (160², 192²,
  224², 256², 288²) maps the complete landscape: 160² marks the physical inflection point
  recovering +9.50% top-1 (67.10% → 76.60%) and +6.00% top-5 (85.70% → 91.70%) at 4.55 ms
  (~220 img/s); 192²–256² forms a broad 79.30%–79.80% accuracy plateau (192² at 79.30% / 5.15 ms;
  256² at 79.80% / 5.85 ms, matching 224² top-1 with +0.90% higher top-5 at 93.40% and rivaling
  wide_resnet50_2 at 39% lower latency); while 288² AdaRound (78.10% at 8.78 ms) confirms that
  resolutions above 256² remain dominated even after rounding optimization.
- **AdaRound for `wide_resnet50_2` and `wide_resnet101_2` — done.** `wide_resnet50_2`
  recovers 90.5% of its quantization loss, essentially matching resnet50's 90.0% (see the
  findings section above); the predicted "recovers less at width" effect didn't materialize,
  and the result is now this repo's best classification speed/accuracy point (80.10% top-1,
  9.66 ms). `wide_resnet101_2` is also done: plain XINT8 had already absorbed most loss
  (80.20% vs 82.00% FP32), so AdaRound top-1 is a wash (79.90%), but top-5 recovers from
  92.90% to 93.90% (66.7% of the gap to 94.40% FP32) at 16.64 ms on NPU (767/769 nodes,
  4.97× speedup over Zen 4 CPU FP32 at 82.78 ms). AdaRound for YOLOv8s and m are both done
  now (see the status table) — neither was actually RAM-blocked once run on a machine with
  more headroom (Desktop 1's GPU-accelerated FastFinetune), confirming the wall found
  earlier was resolution-specific (640×640), not width-specific, since the ResNet
  AdaRound runs above hit no such wall at 224². AdaRound for YOLOv8l/x at 640×640 is
  still untried anywhere.
- **Direct NPU utilization measurement — the GOPS path is closed and retracted, but
  reading the full partition report instead of just that column opened a real one.**
  Windows' `GPU Engine` counter was a dead end for a reason stronger than polling
  granularity: the NPU registers as a `ComputeAccelerator` device, not a WDDM GPU
  adapter, so it never appears as an adapter LUID for that counter (or `GPU Adapter
  Memory`) to read at all, regardless of sampling rate. `xrt-smi examine -r
  aie-partitions`'s GOPS column looked like a utilization signal (cross-model ratios
  track FLOPs closely, `tools/gops_sweep.py`, `results/npu_utilization_gops.log`) but
  is a per-context notional figure, not delivered compute — disproved by linear
  scaling with stream count while measured completions/s stays flat
  (`tools/session_hold.py`, `results/gops_yolov8{n,m}.log`), and explained by finally
  reading the report's Partition Index / Columns fields (never parsed before): every
  session this repo has built under `4x4.xclbin` lands on one shared
  `Partition Index: 0, Columns: [1, 2, 3, 4]`, so "GOPS vs 16 TOPS" was never a
  meaningful ratio to begin with. See finding 3 — every prior absolute number from
  GOPS is retracted, not revised. That same discovery opened a real lever instead:
  `1x4.xclbin` gives independent OS processes *separate* partitions on separate
  columns, and `tools/multi_partition_bench.py`
  (`results/multi_partition_yolov8n.log`) measured 3.65× combined throughput at 4
  concurrent processes vs 1, beating the single 4x4 partition's own ceiling by
  roughly 1.4–1.6× — see finding 4, caveat about `1x4.xclbin` being deprecated by AMD
  included. **Whether it holds for a wider model than yolov8n — done, and it holds
  better.** yolov8m and yolov8l both scale to ~3.8× at 4 columns (`results/
  multi_partition_yolov8{m,l}.log`), and a real achieved-ops/s number now exists to
  measure it with (finding 5, `tools/estimate_tops.py`): yolov8l split across 4
  columns reaches 39.3% of the 16 TOPS nameplate, the best figure this repo has
  measured. **Whether a model heavier than yolov8l/x would climb past 39.3% — done,
  and it doesn't.** yolov8x re-exported at 1280² (6.2× the per-inference MACs of
  yolov8l at 640²) lands on 39.31%, statistically identical to yolov8l's 39.32%
  (`results/multi_partition_yolov8x_r1280.log`) — this reads as a real per-column
  ceiling near 39-40%, not headroom still waiting for a heavier model. **Whether a
  5th concurrent context queues, refuses, or shares a column — done: it shares.**
  This Phoenix chip actually has 5 physical columns (`xrt-smi examine -r platform`);
  `1x4.xclbin` still only ever exposes 4 partitions, and the 5th process pairs up
  with the 4th on column 4 rather than getting its own (`results/
  multi_partition_yolov8n_5col.log`). The driver's own `5x4_*.xclbin` overlay family
  was tried directly as the obvious follow-up and falls back to 100% CPU on this
  install (`docs/DECISIONS.md`, "Rejected approaches") — the 5th column is real but
  unreachable through anything this repo's tooling can drive. **What specifically
  caps a single column at ~39-40% rather than higher — narrowed: it is not fixed
  per-call dispatch overhead.** `tools/percall_overhead_bench.py`
  (`results/percall_overhead_yolov8_1x4.log`) profiled a single held-open
  `1x4.xclbin` session per model size and found dispatch/sync overhead outside the
  compute node negligible at every size (0.7% of wall time at n, down to 0.1% at l)
  — essentially all wall time is the compute node's own reported duration, and that
  duration's efficiency against an ideal 4-TOPS column climbs with model size
  (19.0%→29.3%→36.5%→41.3%, n→s→m→l), matching the combined-throughput figures
  above via a fully independent measurement. The ceiling lives inside the compiled
  kernel's own scheduled execution, not host-side dispatch — a real compiler/
  scheduling limit. Still unconfirmed: whether any of this holds on the laptop's
  Hawk Point chip (different column count, untested).
- **Width beyond yolov8m — done for detection (l/x); classification untested.**
  yolov8l/x are measured (see the findings table above) and the trend breaks at x. Two
  width steps still confirm the classification trend (resnet50→wide_resnet50_2→
  wide_resnet101_2) with nothing past that tested.
- **Full 5000-image mAP for every model size — done for n/s/m/l/x.** All five detection
  sizes now have a full-dataset number; only calibration sample count differs between
  them (200 for n/s, 64 for m, 32/24 for l/x), a caveat already flagged in `README.md`
  and `docs/BENCHMARKS.md`.
- **The eventual application.** Nothing here commits to it yet, but the shape of a
  next step is visible: a labeled licence-plate dataset from fixed camera feeds,
  fine-tuning a YOLO variant on it, and reusing this same head-cut + XINT8 + AdaRound
  recipe rather than re-deriving it.
- **Custom C++ XRT / hand-written AIE kernels — scoped, and the silicon-vs-toolchain
  question now has a primary-source answer.** Everything above runs through ONNX
  Runtime's VitisAI EP against Quark-quantized INT8 graphs; it answers "what can this
  hardware do through the toolchain AMD ships for CNN inference," not "what can the AIE
  array itself do." Prompted by asking whether XDNA1's silicon supports INT16/BF16 at
  all (the "A16W8" and BF16 findings above are both about this EP's own config surface,
  not the tile ISA) — checked what it would actually take to reach the array directly,
  and what the tile ISA actually supports.
  **`C:\Program Files\RyzenAI\1.7.1` ships an AI Engine compiler toolchain never touched
  by anything in this repo:** `vaie_overlay` (its RECORD lists exactly two executables,
  `aiecompiler.exe` and `graph_preprocessor.exe`, plus .bat wrappers and a client DLL —
  `vaie_overlay/cli.py`'s `__all__` also names `mesimulator`, `xchesscc`/`xchessmk`,
  `xclbinutil`, `bootgen`, `iss_dbg`, but that list is a shared entrypoint list used
  across AMD's Vitis Python packaging convention, not a manifest of this wheel's actual
  contents — an earlier draft of this note stated those tools ship here; they don't,
  confirmed from the RECORD, not assumed); `vaie_cpplus` (the ADF C++ dataflow-graph API
  headers — `adf.h`, tile control for both `aie2gen`, this project's own Hawk
  Point/Phoenix chips, and `aie4gen`); and `llvm_aie_lightweight` (Peano: `clang.exe` +
  `ld.lld.exe` targeting `aie2-none-unknown-elf` directly). Raw XRT
  (`xrt\xrt_coreutil.dll`) is present too. This is the same Vitis/Vivado-lineage AI
  Engine toolchain used for Versal FPGA+AIE designs, repackaged for XDNA1 — not the
  open-source `mlir-aie`/IRON project, which this SDK does not ship at all.
  **The dtype question has a better answer than compiling anything: `device.yaml`**, a
  config file inside the `waic` wheel (also in this same install, `OGOAT/Collaterals/
  device.yaml` — see `results/aie/notes_aie2_device_dtypes.log` for the full excerpt),
  gives AMD's own per-architecture tile spec. Phoenix mixes in the `AIE2` block:
  `macs_per_cycle` of 128 for `bfloat16xbfloat16` and `int16xint8`, 256 for
  `int8xint8`; `adds_per_cycle` of 32 (`bfloat16`, `int16`) and 64 (`int8`, `int4`).
  **This settles it: the silicon supports bfloat16, int16, int8, and int4 arithmetic
  natively — this repo's INT8-only toolchain limit (locked #3) is a Quark/VitisAI-EP
  config-surface restriction, not a hardware one**, now backed by an AMD-authored
  in-SDK citation rather than a spec-sheet assumption. `int16xint8` at 128 macs/cycle is
  exactly A16W8's shape (INT16 activation/INT8 weight) — confirms the A16W8 finding
  above was correctly scoped to the EP (opset-17 Q/DQ domain routing), not the hardware.
  Strix's `AIE2p` block, for contrast, adds `bfp16` and `int16xint16`/`int8xint4`
  combinations AIE2 lacks and roughly doubles most throughput figures — an asymmetry
  that is itself evidence this is a real per-chip table, not a copy-pasted default.
  **Checked whether an actual custom kernel could be built and run on Phoenix from
  material already in this install — a real dead end, confirmed rather than assumed.**
  The same `waic` wheel also bundles `aie4_models/`, a large internal AMD kernel-source
  tree (per-dtype Conv/GEMM/MaxPool/GroupNorm/Softmax/etc. C++ kernels in int8x8,
  int16x8, int16x16, and bf16 variants — a real catalog of "every kernel, every data
  type" for *some* chip). But `aie4_models/buildscripts/build_conv.py` hardcodes
  `set_dev_gen(DevGen.Aie4)` at import time; `model_cfg.yaml`'s only documented
  `device:` choices are `mds` (Medusa) and `swv` (SoundWave), both mapping to
  `device.yaml`'s `TestHW` block — no `macs_per_cycle` table at all, i.e. pre-silicon
  test hardware, not a shipping product; and `settings.sh` references AMD-internal
  infrastructure (`/proj/primebuilds/...`, `aiengine-eng` license servers). This is an
  internal engineering source tree for a different, unreleased chip generation,
  incidentally bundled into a public wheel — not a path to a Phoenix kernel, checked
  from the source before attempting a build, not assumed after one failed.
  Checked whether any existing example already exercises the *reachable* part of this
  toolchain (`aiecompiler`/ADF/Peano) for CNN inference: no — `LLM\` (the only other
  non-CNN use of this same install) goes through `onnxruntime-genai`'s own C API, not
  custom AIE kernels, so there is no worked example anywhere in this SDK to build a
  Phoenix kernel from. Reaching it would still be a from-zero bring-up — ADF graph in
  C++, kernel compile (CHESS or Peano), `aiecompiler`, `xclbinutil` packaging, a small
  XRT host program — the same falsification-first shape this project itself started
  with on resnet50, one level lower in the stack.
  **Bring-up started, and it stopped at a specific, well-defined wall — a real result,
  not an abandoned attempt.** Wrote the smallest possible ADF graph (`tools/aie_probe/`:
  one `int8` passthrough kernel, two ports, one connect) using the confirmed API surface
  from `vaie_cpplus/include/adf.h`/`adf/window/window.h` (`window_readincr`/
  `window_writeincr` verified present, not assumed from memory of the public AI Engine
  docs). `aiecompiler` (invoked via the installed `vaie-overlay` package, in
  `ryzen-ai-1.7.1`, not `resnet_env17` — that env stays untouched) runs completely
  self-contained on this install: full version banner (`AI Engine Compiler 2026.1`,
  `SW Build ab5caf8 (release_rai_1_7)`), full `--help` listing, no missing-dependency
  error (`results/aie/aiecompiler_help.log`) — the `data/baseline.txt` failure an earlier
  pass in this session worried about never materialized.
  **First wall: no valid `--part`/`--platform` device-model string exists anywhere in
  this SDK's help output or config files.** `--target=x86sim` (compiles to native x86
  threads — validates ADF graph structure and kernel logic only, proves nothing about the
  AIE array itself) got as far as requiring one of those two flags
  (`results/aie/aiecompiler_x86sim_passthrough.log`). Checked three guessed candidates,
  all failed identically ("AIE architecture could not be auto-derived"): a deliberately
  bogus string (confirms the flag works, no valid-parts list is ever printed);
  `vaip_config.json`'s `"target"` strings (`PROCYON-MHA-QDQ`, `PSD`/`PSO`/`PSV`,
  `RyzenAI_transformer_cxx_*`) — these are the VitisAI EP's own op-fusion subgraph
  labels, a different namespace entirely; and `IPUV1CNN`, a real string pulled directly
  from `phoenix\4x4.xclbin`'s own `aie_partition` metadata section (this SDK ships no
  `xclbinutil` to read that section properly, so this came from a raw byte scan) — still
  rejected (`results/aie/aiecompiler_part_{probe,ipuv1cnn}.log`).
  **That wall is now solved, not just named.** Rather than keep guessing, read
  `aiecompiler.bat`/`setupEnv.bat` directly to find where the tool actually looks for a
  device database: `RDI_APPROOT/data/parts/xilinx/xclbin/`, which this install ships with
  exactly one populated entry — `strx/base.xclbin` (Strix). Its embedded `build_metadata`
  gives a real, well-formed part string, `xc10AIE2P_ML-die-0x-e-S`, and feeding that to
  `--part` makes the "could not be auto-derived" error disappear entirely — it derives
  `__AIE_ARCH__=21` and moves on to a completely different failure
  (`results/aie/aiecompiler_part_strix_confirmed.log`). The `--part` mechanism works; this
  install's shipped database simply has no Phoenix entry, only Strix's. Byte-scanning the
  132 MB `aiecompiler_client.dll` that actually implements derivation (found by grepping
  every `.dll`/`.exe` in the env for the literal error string) turned up
  `BuildDevice::isPHXPart()` alongside `isSTRXPart()` — the compiler's own code recognizes
  a PHX device family — plus a table of real `xc10`-prefixed part strings for many AMD
  chips, including `xc10AIE24x5-die-1LP-e-S-es1`. That name matches an existing,
  independent measurement in `docs/DECISIONS.md`: `xrt-smi examine -r platform` reports
  **Total Columns: 5** on this exact Phoenix desktop, not the 4 every tool here assumes —
  i.e. the physical array really is 4×5, of which this project's own xclbins expose only a
  4×4 subset. Fed to `--part`, `xc10AIE24x5`/its full form both derive cleanly
  (`__AIE_ARCH__=20`, no "could not be derived" error —
  `results/aie/aiecompiler_part_phoenix_candidate.log`; full detail on the string-table
  discovery in `results/aie/notes_aiecompiler_part_db.log`). Not confirmed against an
  AMD-published part list (none exists anywhere in this SDK), but the strongest evidence
  available on this machine, and corroborated by hardware measurement that predates and
  wasn't chosen to fit this investigation.
  **Second wall, newly reached: no host C++ standard library exists on this machine.**
  Every part string that now derives successfully — Strix's confirmed one and Phoenix's
  candidate — reaches the identical next failure, independent of `--target={x86sim,hw}`:
  `adf/new_frontend/adf.h`'s own `#include <iostream>` fails to resolve, because the ADF
  C++ frontend's preprocessing pass runs through Peano's bundled `clang.exe`, and nothing
  in this pip package (`find ... -iname iostream` → no results) or on this machine (no
  Visual Studio / Build Tools install anywhere under `Program Files`, no `vcvarsall.bat`)
  supplies a C++ standard library for it to find (`results/aie/aiecompiler_hostlib_missing.log`).
  This wall is not a data-population gap like the first one — it would block a Strix build
  identically.
  **That wall is solved too.** Installed Visual Studio 2022 Build Tools (C++/VCTools
  workload) via `winget`, then passed MSVC's and the pulled-in Windows 10 SDK's (`10.0.
  26100.0` — an independent version number from this machine's Windows 11 build, not a
  mismatch) include directories to `aiecompiler` as extra `--include` flags (it accepts
  the flag repeated, accumulating a search list). The `<iostream>` failure disappears
  completely (`results/aie/aiecompiler_hostlib_fixed.log`). Along the way, the trivial
  graph in `tools/aie_probe/graph.h` hit its own, unrelated bug — `connect<>(...)` with an
  empty template list is rejected for window ports — fixed with an explicit
  `adf::window<32>` argument (32 int8 elements, matching the kernel's read loop). With
  both fixed, the compiler goes further than at any point in this investigation: it reads
  and derives the graph, and logs **`Reading logical device aie2_5x4_device`** — the
  strongest confirmation yet that `xc10AIE24x5-die-1LP-e-S-es1` is genuinely Phoenix,
  since "aie2_5x4" names exactly the architecture and 5-column×4-row physical layout the
  pre-existing, independent `xrt-smi` measurement already established for this chip.
  **Third wall: `physical_device.dll` is a real, hardcoded, missing file — not one AMD
  distribution on this machine ships it.** One step past device derivation, `aiecompiler`
  needs `lib/win64.o/physical_device.dll` and can't find it. Binary-scanning all three
  copies of `aiecompiler_client.dll` in the pip env confirms this is a literal filename
  the compiler `LoadLibrary`s and calls an exported `createPhysicalDevice()` from — not a
  `%s`-templated name a differently-named file could satisfy — and it sits beside three
  sibling DLLs the same table expects (`platform_device.dll`, `guidance_summary.dll`,
  `udm_api.dll`), all of which are equally absent. Checked the whole `C:` drive (zero
  matches), then went looking inside the two full offline installers already on this
  machine (`ryzen-ai-lt-1.7.1.exe`, `ryzen-ai-1.8.0.exe` — each a 7z-SFX wrapping an MSI
  plus ~20 cabs; opened with `7z`/`lessmsi`, the latter needed because MSI cabinets store
  files under opaque per-file IDs, not real names). The 1.7.1 installer turned out to
  install the *exact same pip wheels*, byte-for-byte — no extra content. The 1.8.0
  installer doesn't even ship the `vaie_overlay`/`vaie_cpplus` packages that carry
  `aiecompiler_client.dll` at all. Also opened `device_essentials_strx_overlay-1.7.1`
  (the package that supplies `strx/base.xclbin`): it ships a large ML-based
  place-and-route congestion feature store (`data/FeatureStore/`, `data/hierdb/
  optstrategy/*.pb`) for Strix only — no Phoenix equivalent exists in 1.7.1 at all — but
  even for Strix it contains no `physical_device.dll`. The data and the code that would
  consume it are both incomplete, in different ways. This reads as a deliberate boundary
  in AMD's redistributable packaging: pre-built xclbin overlays ship to end users; the
  physical-implementation backend needed to build a new one from a hand-written ADF
  graph does not. **This is where AMD's own proprietary toolchain (`vaie_cpplus`/
  `aiecompiler_client.dll`) stops: two walls solved in sequence (device-model string,
  host C++ toolchain), a third that neither offline installer on this machine, at either
  SDK version, can supply.** See `results/aie/aiecompiler_physical_device_missing.log`.
  **Chasing "a full Vitis/Vivado install" (the avenue this log left open) turned up a
  different, unblocked door instead of the gated one implied.** The physical
  place-and-route backend AMD ships standalone (Vitis AIE Essentials) is gated behind an
  early-access account + a MAC-locked license, frozen at SDK 1.3.1, and every Windows
  reference to it runs under WSL (which this project's own findings already rule out for
  reaching XDNA1 hardware) — not pursued further. But AMD's current docs also say the
  Vitis step can be skipped entirely for AIE2/AIE2P using **Peano**, the open-source
  LLVM-based AIE backend bundled with `Xilinx/mlir-aie` on GitHub — a project that
  implements its own place-and-route in MLIR passes and never calls `aiecompiler`/
  `physical_device.dll` at all. Peano was already on this machine as a pip dependency
  (`llvm_aie_lightweight`, native Windows `clang.exe`/`ld.lld.exe` under a completely
  separate `win64.o` namespace from the one missing its backend).
  **Set up mlir-aie's native-Windows path (no WSL) in a new, isolated conda env
  (`mlir-aie-iron`, Python 3.13; none of the existing 5 envs touched) and ran a
  hand-written kernel end to end on this machine's actual XDNA1 (Phoenix) hardware —
  measured, not simulated.** `xrt-smi examine` already reported exactly the XRT/driver
  version (2.21.0 / 32.0.20101.3760) mlir-aie's guide requires, so no driver change was
  needed. Two setup snags, both fixed without touching the existing pip envs (a missing
  `llvm-objcopy.exe` the setup script looks for in the wrong directory — the `mlir_aie`
  wheel bundles its own copy; a missing Windows XRT *SDK*, distinct from the driver/
  runtime XRT already installed — downloaded from the matching XRT GitHub release). One
  version-skew failure (`main` branch example code vs. a rolling wheel channel —
  `Runtime.__init__()` signature mismatch, the exact pitfall the project's own README
  warns about) fixed by checking out a tagged release (`v1.4.2`) and installing its
  matching wheel instead of `main` + latest. With both in sync, the SAXPY example
  (`z = 3*x + y`, N=4096, bfloat16, one AIE core) JIT-compiled, dispatched against
  `device="npu"`, and verified against a numpy reference: **`PASS!`**
  (`results/aie/mlir_aie_saxpy_npu.log`). The fresh compile cache is the evidence this
  was a real compile, not a stale artifact or a simulator run: placed MLIR, three staged
  LLVM-IR dumps, a compiled kernel object, a linked core ELF, three CDO binaries, a PDI,
  and a 8,870-byte `final.xclbin` — every place-and-route stage done by mlir-aie's own
  tools. **This does not overturn the proprietary-toolchain finding — `physical_device.dll`
  is still genuinely absent from every AMD distribution channel checked — but it means
  custom-kernel bring-up on this hardware is not actually blocked; it just needs a
  different toolchain than the one this project's own pipelines (Quark + VitisAI EP) use
  for quantized-model inference.** The two are not interchangeable: mlir-aie/IRON is for
  hand-written kernels from scratch, Quark/VitisAI EP is for running existing quantized
  ONNX models — this project's own pipelines still have no reason to move off the latter.
  Closes the "is custom-kernel bring-up possible on this hardware" question with a
  measured yes, via a different door than the one that was walled off.

  **Follow-up: three more `programming_examples/getting_started` designs, same result.**
  SAXPY is the simplest design in mlir-aie's own tutorial set, so it left open whether the
  result generalizes. From a cleared compile cache, three more designs were run and all
  `PASS!`ed: `00_memcpy` (a multi-column DMA-bound bandwidth microbenchmark, every shim DMA
  in/out pair on the device, 56.19 GB/s effective NPU-side bandwidth on 64 MiB round-trip);
  `02_vector_reduce_max` (a 4-core single-column cascade — each core reduces a quarter of
  the input and forwards its running max to the next core, exercising cross-core control
  flow, not just parallel elementwise work); and `03_matrix_multiplication_single_core`
  (int16 matmul at two shapes, 256³ and 512³, via the design's opt-in AOT path —
  `.specialize(...).compile()` produces distinct on-disk xclbin/insts artifacts for each
  shape before either kernel runs, confirmed by two differently-sized `final.xclbin`s in
  the compile cache, not one reused across both). Full stdout and cache-directory evidence
  in `results/aie/mlir_aie_examples_npu.log`. This broadens the exercised IRON surface
  (`ObjectFifo` split/forward, multi-core `Worker` cascades, `TaskGroup`-scoped multi-tile
  DMA, AOT `compile()`) well past SAXPY's single elementwise core, with the same
  conclusion: still a different toolchain from this project's own pipelines, not a reason
  to move off Quark/VitisAI EP for quantized-model inference.

  **Follow-up: mlir-aie's `ml/` examples — closer to this repo's own operator
  vocabulary.** `getting_started` designs are generic (memcpy, reduce, matmul);
  mlir-aie also ships a `vision/` and an `ml/` category, closer to what YOLOv8/ResNet
  actually run. Five pure-Python, Phoenix-tagged (`ryzen_ai_npu1`) designs from this
  category ran, all from a cleared compile cache: `eltwise` (bf16 add and mul, each its
  own compiled kernel), `eltwise_unary` (relu, silu, gelu — the exact activation
  functions this project's own quantized models use), `scale_shift` (a two-phase
  `D = A*B+C` design synchronized across phases with a `WorkerRuntimeBarrier`, a
  different structural pattern than the split-and-parallelize style of the other
  designs), `softmax`, and `swiglu`. All **`PASS!`** — see
  `results/aie/mlir_aie_ml_examples_npu.log`. The rest of `vision/`/`ml/` was not run
  this pass, for reasons the log states plainly rather than working around: the
  `vision/*` designs and a few `ml/*` designs (`bottleneck`, `conv2d`, `conv2d_14x14`)
  drive the NPU from a `make`-built C++ host program linking OpenCV, and this machine
  has neither `make` nor an OpenCV C++ dev package installed; several other `ml/*`
  designs are tagged `ryzen_ai_npu2` (Strix) only in mlir-aie's own test metadata and
  do not apply to this Phoenix machine regardless; `ml/magika` and `ml/mobilenet` are
  Phoenix-capable and pure-Python but are multi-file models substantial enough to
  warrant their own dedicated pass rather than folding into this one.

  **Follow-up: installed `make` and OpenCV, ran the rest.** Both gaps above were filled
  — GNU Make 4.4.1 into the isolated `mlir-aie-iron` conda env, and OpenCV 5.0.0 (current
  latest, Windows prebuilt) extracted to `C:\Technical\thirdParty\opencv`, which happens
  to be mlir-aie's own `CMakeLists.txt` hardcoded default `OpenCV_DIR` for every
  `vision/*` example, so nothing in mlir-aie itself needed editing. All four `vision/*`
  designs then ran and passed: `color_detect`, `color_threshold`, `edge_detect`, and
  `vision_passthrough` (byte-exact, 0 differences — the tightest of the four, as expected
  for a pure passthrough). Also installed `torch` (latest, CPU-only) into mlir-aie's own
  venv — used only to generate a host-side reference, not for anything
  performance-relevant — and ran two more `ml/*` designs that needed it: `bottleneck`
  (**Avg NPU time: 1620us**, `PASS!`) and `conv2d` at its default 1×1 32×32×64→64 shape,
  both plain (540us) and with `--fuse_relu` (533us), both `PASS!`. `bottleneck` is the
  closest match yet to this repo's actual model family — a real ResNet-style
  conv1×1→conv3×3→conv1×1-plus-skip block, built from mlir-aie's own
  `kernels.conv2dk1`/`conv2dk3`/`conv2dk1_skip` library functions, still via a
  hand-written-kernel toolchain rather than a quantized ONNX graph through Quark/VitisAI
  EP. Two Makefile-specific quirks were fixed at invocation time (not by editing
  mlir-aie): `make getwslpath=echo ...` (its own WSL-detection heuristic — "is
  `powershell.exe` on PATH" — false-positives on native Windows), and adding the OpenCV
  `bin` directory to `PATH` before `make run` (needed by the built `.exe` at run time,
  not just at CMake configure time). Full detail in
  `results/aie/mlir_aie_vision_examples_npu.log`.
- **Follow-up: `magika` and `mobilenet` — the two multi-file models left for their own
  pass — plus one design missed in the original survey.** `ml/resnet/layers_conv2_x`
  (tagged `ryzen_ai_npu1`, not caught by the first `vision`/`ml` sweep) chains three
  ResNet conv2_x bottleneck blocks depth-first across three separate NPU columns —
  **PASS!**, 1888.5us avg NPU time. `ml/magika` (Google's file-type-detection network)
  is tagged `ryzen_ai_npu1` too but mlir-aie's own `run_phoenix.lit` marks it `XFAIL`
  ("Known-failing numerical check on the NPU"); ran every target in that lit by hand
  anyway rather than trusting the label, and it did not reproduce as a numeric failure
  on this hardware — both `group0` (Avg NPU time 665us, EVM -34.85 dB) and `group2`
  (455us, EVM -56.91 dB) **PASS**, a discrepancy from the upstream expectation worth
  recording rather than silently matching. `trace_py` for both groups reproduces the
  identical NPU PASS but then fails in an unrelated downstream step — the trace-JSON
  parser's expected intermediate MLIR file isn't left on disk by aiecc's pipeline on
  this machine, a tooling gap in the visualization step, not the hardware result.
  `ml/mobilenet`'s own README states it targets "the Strix NPU2," and both its
  hardware-driving lits (`run_e2e.lit`, `run_strix_makefile.lit`) require
  `ryzen_ai_npu2` — inapplicable to this Phoenix machine by chip generation, the same
  as the other npu2-only designs already logged. Only its numpy-only cross-validation
  (`run_numpy_per_bn.lit`, no hardware tag) could run here — all 8 verified blocks
  bit-exact against the brevitas fixtures — recorded as a no-hardware check, not an NPU
  measurement. Full detail, including the Windows-path-through-Git-Bash compile-flag
  fix magika needed, in `results/aie/mlir_aie_magika_mobilenet_npu.log`. With this,
  every `programming_examples` design tagged for this machine's chip (`ryzen_ai_npu1`)
  has now been tried on the actual hardware; `mobilenet` remains the one design whose
  hardware-relevant paths require a chip (Strix/npu2) this machine doesn't have.
- **Follow-up: does any of this actually help the models this repo cares about? Not by
  beating the quantized path's raw latency — by reaching data types XINT8 structurally
  can't.** VitisAI EP is XINT8-or-nothing (`A16W8` falls back to CPU entirely, see
  "LOCKED DECISIONS"); every conv-shaped IRON design tried above (`bottleneck`,
  `conv2d`, `resnet/layers_conv2_x`) is itself int8/uint8, matching rather than
  exceeding what Quark already gives. But `eltwise`/`eltwise_unary`/`scale_shift`/
  `softmax`/`swiglu` (bf16) and `magika`/the `getting_started` matmul (int16) prove
  those precisions genuinely execute on this NPU via IRON — something the standard
  pipeline cannot do at all. That points at a narrower, real idea: hand-written kernels
  for the specific ops XINT8 already can't reach — like `InstanceNormalization` in
  `resnetv2_50x3_bit`, which falls back to CPU entirely (79.5% of nodes place; see
  `results/bit/`) and is the documented reason that model loses to CPU despite being
  this repo's best-accuracy result. Checked two things before writing any kernel code:
  (1) a literal in-process ORT custom op calling `pyxrt` is dead — `pyxrt.pyd` hard-
  depends on `python313.dll` (`dumpbin /dependents`), and `resnet_env17` is Python
  3.12 — so any splice has to be a two-process pipeline; (2) that two-process design is
  hardware-viable — a VitisAI-EP session and a separately-compiled IRON xclbin both
  held active NPU contexts at once, in both acquisition orders, with zero contention
  (steady 149-150 completions/s throughout a 35s hold in one direction; 8/8 IRON PASSes
  during a session build in the other). Then profiled a real inference on
  `resnetv2_50x3_xint8.onnx` (ORT's own `enable_profiling`) to get real per-node CPU
  cost for all 49 `InstanceNormalization` nodes (really `GroupNorm(32)` via a reshape
  trick — confirmed in the graph itself), grouped by their 6 distinct shapes, and
  compared each against a DMA-floor-plus-round-trip-overhead projection. Verdict: not
  uniform. The 3 smallest shapes (27/49 nodes, ~8.6 of the total 42.37ms CPU cost)
  should stay on CPU — fixed per-call dispatch overhead alone matches or beats their
  real cost. The 3 largest shapes (22/49 nodes, ~33.8ms) are real candidates, projected
  to cut InstanceNorm's total cost from 42.37ms to roughly 28ms if a kernel hits the
  projection — a meaningful reduction, not an elimination. Full numbers in
  `results/bit/profile_instancenorm_splice_feasibility.log`.
- **Follow-up: the kernel, written and measured.** `kernels/groupnorm_bf16/` is a
  from-scratch bf16 GroupNorm(32) IRON design (no mlir-aie template exists for this op
  on npu1; `ml/norm` is Strix-only): 8 workers, two per column so every one of the
  device's 8 shim DMA channels per direction carries one stream, each worker owning 4
  groups and seeing its block twice through one ObjectFifo (statistics pass, then
  normalise pass) with the per-group stats staying in core memory between passes.
  Written for the largest shape first (L=301056: widest margin, fewest nodes, the row
  that tests whether the projection model is real), then built down the table. Checked
  at every shape against the real node tensors pulled from the model (input, params,
  and ORT's own CPU fp32 output for one image): the kernel's entire error is the bf16
  output rounding (per-group rel-L2 ~0.17%; bit-exact against the bf16-rounded
  reference bar a few round-half cases). Measured NPU time per call vs. the profiled
  CPU cost: L=301056 **1535us vs 3472** (the projection said 1489 — landed on it, though
  its two halves were both off in cancelling directions: fixed overhead measured at
  ~200us, not the 460 borrowed from magika, and this design's traffic moves at 37-39
  GB/s, not memcpy's 56), L=150528 **836 vs 1899**, L=75264 **510 vs 989**, and L=37632
  **351 vs 496** — that last row flipping from the projection's "CPU wins (barely)" to a
  narrow measured kernel win because the real overhead is smaller. L=18816 loses by
  28us/call and L=9408 sits under the ~200us floor, both staying on CPU. Net: 33 of the
  49 nodes are now measured wins, ~19.4ms of the op's 42.37ms per inference (the
  feasibility log projected 22 nodes and ~14ms). `results/aie/groupnorm_bf16_kernel_npu.log`.
- **Follow-up: the handoff cost that erases it.** The kernel wins above are single-
  process (`iron.jit`'s own buffer marshaling only). A real splice needs two OS
  processes — `resnet_env17`'s python 3.12 can never load pyxrt (a hard ABI wall
  against python313.dll, not a PATH issue), so the EP session and the IRON kernel
  can't share an interpreter. Measured the floor of that handoff — shared-memory
  ping-pong of the real per-shape byte volume plus fp32/bf16 conversion, no
  onnxruntime, no actual NPU dispatch (an identity copy stands in for the kernel) —
  at all six node shapes: the floor alone is **789us-23.6ms/call**, and every single
  shape flips. 0 of the 49 nodes survive splicing once this floor is added, down
  from the 33/49 measured as kernel-alone wins. The floor is conversion-bound, not
  transfer-bound (an isolated timing at L=301056 found ~90% of the 23.6ms is the
  fp32<->bf16 cast itself, using `ml_dtypes` — confirmed installed in
  `resnet_env17`, previously unknown — which measured ~2.5x faster than a
  hand-rolled bit-trick conversion); the residual, non-conversion cost still
  exceeds every shape's kernel-vs-CPU margin except a near-wash at L=301056, so a
  faster conversion alone would not rescue this. The per-node kernel wins stand as
  measured; the practical splice does not survive contact with the real two-process
  pipeline this hardware/toolchain split forces, and no batching-across-nodes
  design has been attempted to change that arithmetic. Not yet measured: the full
  model's top-1 with bf16 in these nodes (now moot unless a batched splice is
  built and shown to change the handoff numbers above).
  `results/aie/groupnorm_bf16_handoff_floor_npu.log`.
- **Follow-up: the floor was a conversion function, not physics -- and the real
  target is bigger than assumed.** Three challenges to the log above, each checked
  with a measurement: (1) DMA-floor retraction -- L is per-group length, so
  L=301056 moves 32*L = 9.6M elements; the kernel was already at the floor (1535us
  measured vs ~1489us projected), not 28x off it, an earlier arithmetic error, not a
  re-measurement. (2) `ml_dtypes.bfloat16` isn't a native numpy dtype, so
  `.astype()` runs a scalar loop with no SIMD path -- `measure_handoff_floor_v2.py`
  replaces it with a strided-view truncation (`x.view(uint16)[1::2]`, bf16-by-
  truncation rather than RNE) plus preallocated `out=` buffers: combined pack+unpack
  at L=301056 drops from ~22.6ms to ~4.0ms (5.6x), and the real two-process floor at
  that shape drops from 23.6ms to 5.7ms (CPU still wins, 1.65x not 6.8x) --
  isolating the protocol alone (`--no-convert`, zero conversion, full bf16 payload)
  measures 1.19ms/call, already UNDER CPU's 3.47ms. The shared-memory handoff was
  never the obstacle; the conversion function was. (3) Every one of the 49
  InstanceNorm nodes sits inside a `QuantizeLinear -> DequantizeLinear ->
  InstanceNorm -> QuantizeLinear -> DequantizeLinear` sandwich (verified
  programmatically against the real graph, all 49/49, nothing else riding along) --
  re-profiling the real model (same ORT chrome-trace methodology as
  profile_instancenorm_splice_feasibility.log) shows the input-side quantize and
  output-side dequantize are already optimized away by ORT at runtime (0/49 sites),
  but the input-side dequantize and output-side quantize DO run, adding 56.8% on top
  of InstanceNorm's own cost -- 65.08ms across all 49 nodes, not 42.37ms, i.e. 11.1%
  of the model's 586.32ms/image latency, not 7.2%. Folding those two nodes' scales
  into the kernel's existing stats/affine passes (raised in review, not built) would
  let an int8-native design (int8 in, on-core dequant, on-core requant, int8 out)
  target that larger 65.08ms number instead of InstanceNorm alone -- a bar the
  measured 1.19ms protocol floor already clears at the hardest shape (5.21ms target
  at L=301056) before any int8-specific payload reduction is counted. Nothing about
  the actual int8-native kernel is built: the on-core int8->bf16 widen-and-dequant
  cost is real and unmeasured, not assumed zero, and the host-side plumbing to
  extract/reinsert int8 tensors at these graph points doesn't exist. First version
  of this chain where the projected numbers point to a win, not a loss -- still a
  projection, not a measurement of the thing itself.
  `results/aie/groupnorm_bf16_handoff_floor_v2_npu.log`.
- **Follow-up: GroupNorm was the wrong shape of op for bf16 to begin with -- pivoting
  to a compute-bound one.** Written after the reopening above, not unaware of it: even
  taking the reopened floor at face value, GroupNorm is ~6 ops per element with no
  arithmetic intensity; bf16 there only ever bought a 2x byte-size reduction over fp32,
  never a compute win, which is why every fix to the handoff floor was fighting a
  boundary cost rather than the real constraint. AIE2's actual bf16 advantage is a
  native bf16xbf16->fp32 MAC, which a memory-bound op never exercises. Decision:
  check whether a fused self-attention block (QK^T -> softmax -> PV, one xclbin, no
  host round-trip between stages) is buildable entirely from mlir-aie's own
  bf16-validated building blocks on this chip, so that MAC path has something to
  land on. Inventoried bf16 support across `programming_examples/` (v1.4.2, static
  code reading): matmul (`basic/matrix_multiplication/{single_core,whole_array}`),
  `ml/eltwise`, `ml/eltwise_unary`, `ml/scale_shift`, `ml/softmax`, and `ml/swiglu`
  all have real bf16 kernels already tested on this chip (npu1/aie2); `ml/conv2d`,
  `ml/conv2d_14x14`, `ml/bottleneck`, and `ml/resnet` are int8/uint8-only with no
  bf16 path anywhere in the tree; `ml/norm` (LayerNorm/RMSNorm), `ml/rope`, and
  `ml/dwconv1d` are bf16 but Strix (aie2p)-only, "no aie2 counterpart." This picks
  the net: a CNN would mean writing bf16 conv2d from zero with no reference: a
  self-attention block needs no LayerNorm-equivalent for its core compute and is
  assemblable entirely from already-validated pieces. `ml/resnet/layers_conv2_x`
  (already run, see the entry above) is the structural template for chaining
  without a host round-trip: block N's output ObjectFifo is literally block N+1's
  input ObjectFifo, core to core, and the whole chain is one dispatch.
  Before building anything, ran the actual bf16 matmul primitive on this hardware
  for the first time (`basic/matrix_multiplication` had never been run on this
  machine; only the unrelated `getting_started/03_matrix_multiplication_single_core`
  int16 design had): single_core, 512x512x512, bf16 in/out, **PASS, 116.56
  GFLOPS (2303us NPU time)**; whole_array, 4 columns, same shape, **PASS, 895.08
  GFLOPS (299.9us NPU time)** -- ~7.7x scaling across 4 columns, not a clean 4x. The
  headline is the PASS, not the GFLOPS: it answers whether there is a real, correct
  bf16 compute primitive to build an attention block on before committing engineering
  time to one. Getting there needed five new native-Windows toolchain fixes (a
  Make/MSBuild environment-variable case collision, a `powershell.exe`
  re-tokenization bug that loses a multi-word `CXXFLAGS` value, `CMAKE_PREFIX_PATH`
  for XRT's own CMake package discovery, `xclbinutil.exe`'s real path, and
  `pyxrt.pyd`'s real path/PYTHONPATH requirement) plus one one-line portability fix
  in the external mlir-aie clone (a GCC-only compound-literal cast MSVC rejects) --
  see `results/aie/mlir_aie_bf16_matmul_npu.log` for all five, and for a separate,
  still-unexplained bug in this design's C++ host-test harness (identical garbage
  verification output regardless of dtype) that the pure-Python IRON path sidesteps
  cleanly.
- **Follow-up: Fused BF16 Attention Kernel Built, but Small Sequence Length Exposes the Arithmetic Floor (Negative Result).**
  Addressed the hybrid CNN-Transformer architecture `mobilevit_xxs`. Stock VitisAI EP
  quantized via Quark partitioned it into **49 metaDef (58 IPU) thrashing DPU subgraphs, 108.29 ms** (1,037 NPU /
  156 CPU / 392 `VITIS_EP_CPU` nodes; the CPU compute is LayerNorm, MatMul, Slice, Squeeze,
  Transpose, Reshape) because the DPU overlay has no kernel for those transformer operators --
  **14.4x slower than the same FP32 graph under the ORT CPU EP (7.51 ms)**. Cutting the
  attention blocks left a pure convolution backbone (409 nodes: 407 NPU in **1 single
  subgraph**, 2 CPU boundary nodes) compiled into NPU at **1.71 ms** (**3.30x** faster than
  CPU 5.65 ms, like-for-like). All four figures re-measured in one run, see the splice entry
  below. Built the custom
  fused BF16 attention kernel in `mlir-aie` (IRON + Peano): solved Peano's linker script upward
  stack collision via `Worker(stack_size=2048)`; implemented row-wise FlashAttention streaming
  shrinking tile memory from 128 KB to 512 bytes; applied 16-lane AIE2 vector intrinsics (`aie_api`)
  with aligned padding ($D_{pad} \in \{16, 32\}$); and evaluated in-place stable Softmax. Scaled
  across 8 physical cores (Cols 0..3, Rows 2..3) mapped to all 8 physical Shim DMA channels. The kernel
  passes bit-accurate numerical verification (<1% rel L2 error, 0 NaN) across all stages (Stage 4:
  0.86 ms, Stage 3: 4.57 ms, Stage 2: 57.61 ms), accounting for the 0.1700% FP32→BF16 quantization floor
  and 0.235% `fast_exp` approximation error on Stage 3's 10,240 active elements.
  
  **The Finding (Negative Result):**
  The kernel loses heavily to CPU. At Stage 3 (8 heads), total compute is only **2.79 MFLOP**
  (~100x smaller than MobileNetV2's ~300 MFLOP floor, which already lost to CPU). Zen4 AVX-512
  runs those 8 heads in **0.034 ms**, making the 4.57 ms AIE2 kernel **134x slower than CPU**
  (achieving 0.61 GFLOPS, <0.1% of array peak). Mobile vision self-attention lacks the token
  sequence length ($N \ge 2048$) of LLMs.
  Running the full model on NPU with this kernel takes **>120 ms -- a projection, not a
  measurement**: the per-stage timings summed over the real block counts
  (2x57.61 + 4x4.57 + 3x0.86 ~= 136 ms at 8 heads), never run end to end. Cited only for
  direction; the per-stage numbers already settle it.

  **Diagnosis corrected 2026-09-07 -- the verdict holds, the stated cause did not.**
  This entry originally attributed the loss to "AIE2 launch and DMA sequencing overhead."
  The per-dispatch floor has since been measured in isolation (see the next finding):
  **617 us wall / 170 us hardware** per call, which against Stage 2's measured 57,610 us is
  **~1%**, not the dominant term. The actual cause is the kernel: `attention_kernels.cc`
  contains **zero uses of `aie::mmul`** and instead hand-rolls dot products with a
  horizontal `aie::reduce_add` per output element, forfeiting the native bf16xbf16->fp32
  MAC the whole pivot was chasing. It reaches 0.61 GFLOPS on hardware this repo measured at
  **895 GFLOPS**, so ~99.9% of the gap is design, not fixed cost. The row-wise
  FlashAttention streaming that solved the 128 KB scratchpad overflow is the same edit that
  destroyed the arithmetic intensity -- that inner loop is not a template to reuse.
  Rewriting it with `aie::mmul` would still not save it, which is why this stays a negative
  result: a perfect 895 GFLOPS kernel gives Stage 2 40 us + 617 us = 657 us against CPU's
  240 us, and even at the 170 us hardware floor 210 vs 240 us is a wash; Stages 3 and 4 lose
  at both floors. The op needs to be ~20x larger before kernel quality decides anything.

- **The chained int8 CNN loses to the CPU by 6.3x, measured before any kernel was written.**
  `ml/resnet/layers_conv2_x` -- 3 ResNet bottlenecks chained core-to-core across 3 columns,
  int8, ObjectFifo to ObjectFifo, one dispatch for the whole chain -- had run and PASSed on
  this machine since 2026-09-06, but the CPU side had never been measured. It was the best
  remaining structural idea, because it fixes every flaw diagnosed in the attention kernel:
  the dtype the repo's whole XINT8 thesis is about, mlir-aie's own validated int8 conv
  kernels instead of a hand-rolled inner loop, dispatch amortized to ~25% of wall, and no
  two-process handoff. Workload 1x64x32x32, spatial 32x32 throughout, **436.21 MFLOP**; all
  rows in one sitting (`kernels/conv2x_baseline/cpu_baseline.py`,
  `results/aie/conv2x_int8_cpu_baseline.log`):
  NPU int8 **1869.6us** hardware / **2497.8us** end-to-end; CPU torch fp32 **1856us**;
  CPU ORT fp32 **815us**; CPU ORT QDQ int8 **295us** (1481 GOPS, VNNI).
  **Like for like the CPU wins by 6.3x on the hardware bracket and 8.5x end to end.** The
  int8 row was verified to actually be int8 -- ORT's optimized graph executes 10
  `QLinearConv` + 3 `QLinearAdd` -- since the whole ratio rests on it.
  **The CPU baseline choice nearly inverted the conclusion:** torch fp32 (1856us) sits within
  1% of the NPU's hardware bracket (1869.6us), so benchmarking against torch alone -- the
  baseline the attention kernel used -- would have read as parity, off by 6.3x. ORT is 2.3x
  faster than torch on the identical fp32 graph and its int8 path another 2.8x on top. With
  the MobileViT splice's torch-vs-numpy 9x, that is two for two: **on this project the CPU
  kernel choice has decided the verdict more often than the NPU has**, and any NPU-vs-CPU
  claim here has to name which CPU implementation it beat.
  **Scope, stated deliberately:** this closes *this design at this shape*, not the op class.
  It is one shape (32^2 x 64) on 3 columns, and the same silicon reached 895 GFLOPS on
  4-column bf16 matmul -- ~4x this design's 233 GOPS -- a gap this run does not explain. The
  two untested candidates are column count (3 vs 4) and spatial size / tile utilization
  (32x32 means short rows and little work per DMA transfer, structurally attention's problem
  again). Running standalone `ml/bottleneck` at 32^2 against a larger spatial and watching
  whether GOPS scales is what would decide it. Third consecutive negative on hand-written
  kernel/subgraph acceleration, and the first to cost an afternoon rather than weeks.

- **The spatial sweep settles it: GOPS scales by 23%, against a 570% gap. The op class is
  closed.** `kernels/bottleneck_sweep/sweep.py` runs one standalone `ml/bottleneck` on the
  NPU across spatial sizes, and `cpu_sweep.py` runs the identical arithmetic through ORT
  QDQ int8 at the same shapes, both sides in one sitting
  (`results/aie/bottleneck_spatial_sweep_npu.log`, Desktop 2 / Phoenix). `tensor_h` is the
  free axis -- every L1 buffer in the design is `tensor_w * channels` bytes and none scale
  with height -- and each NPU shape is checked against mlir-aie's own torch int8 golden
  before its timings enter the fit.
  NPU hardware throughput rises monotonically **116.7 -> 143.7 GOPS** across a 16x increase
  in work (32x32 -> 512x32), converging on a **marginal 146.1 GOPS** (r^2 = 0.99999). The
  CPU's marginal rate over the same shapes is **819.0 GOPS**. Per-shape the CPU wins
  **7.6x** at 32x32 and **5.7x** at 512x32 on the hardware bracket (11.4x / 6.0x end to
  end). 512x32 is simultaneously the NPU's best point and the CPU's *worst* -- the only row
  where the CPU drops below 900 GOPS, its working set having outgrown cache -- and the CPU
  still wins by 5.7x. So 32x32 *was* underutilizing the array, by about 23%, against a loss
  of 570%.
  The verdict deliberately rests on the fit rather than per-point GOPS: per-point
  end-to-end GOPS must climb with size on any accelerator as fixed host cost amortizes, so
  reading that rise as "the array scales" would be a measurement artifact. `1/slope` removes
  every fixed cost. (The CPU fit is the weaker of the two -- r^2 0.98870 with an unphysical
  -116.9us intercept, because CPU per-FLOP cost varies with cache behaviour -- so the
  per-shape table is the primary evidence and the CPU marginal rate is corroboration. Both
  agree.)
  **`tensor_w` = 32 is a hard ceiling for this design, and ResNet50's real conv2_x is
  56x56.** 56x56, 32x64, 64x64 and 128x64 all fail in `aiecc`, not at runtime: the skip-add
  core Tile(0,4) needs five `w`*256-byte buffers plus a 2560-byte stack against 64 KB of
  AIE2 tile local memory (`'aie.tile' op allocated buffers exceeded available memory`, both
  bank-aware and basic sequential allocation, 8 error lines across the 4 shapes). At w=56
  the buffers are 14336 B each for 74240 B total -- still over. So the 32x32 that
  `layers_conv2_x` runs is not the network's shape; it is the largest square that fits, and
  widening past it needs the design's buffering restructured, not a parameter changed.
  **What stays open, and it is now specific:** column count (both measurements use 1-3
  columns of a 4x5 array -- 4 x 146 = 584 GOPS would still lose, but not by 5.6x), and
  kernel quality -- 146.1 GOPS is roughly 7% of one column's ~2 TOPS int8 peak (256 int8
  MACs/cycle/core x 2 ops x 4 cores at 1 GHz; architectural, **not** measured on this
  machine -- measured 2026-09-07: 1.80 GHz in `default`, so one column is 3.69 TOPS and
  146.1 GOPS is 4% of it, `results/aie/clock_probe_npu.log`). That is the same shape of finding as the attention kernel's 0.61 against 895
  GFLOPS, and nobody has yet read `conv2dk1.cc`/`conv2dk3.cc` to see how they vectorize.
  **Correction carried by this run:** the conv2x log called its 628.2us of host cost a
  reproduction of the passthrough's 617.0us "to ~2%". Different quantities -- 617.0us is the
  passthrough's *wall* intercept (447.3 host + 169.8 hardware), so the like-for-like host
  floor is **447.3us** and 628.2 is 40% over it, not 2% under. This sweep's own host cost
  runs 608-874us and *grows* with payload (44% across a 16x byte range), so it is not a flat
  floor either. No verdict depends on it; the agreement was numerology and is retracted.
- **The core clock, measured: 1.80 GHz, and power mode moves it 2.25x.** Every
  per-second ceiling in `docs/SILICON.md` divided by a clock nothing here had measured
  (1.6 GHz from a web search; 1 GHz assumed in the bottleneck log). `kernels/clock_probe/`
  brackets a DMA-free loop with `event0()`/`event1()`, lets the tile's trace unit stamp
  both with its timer, and fits the runtime's submit+wait time against the stamped cycles
  across 2^18-2^25 iterations so the dispatch cost cancels: **1.7983 GHz in `default`**
  (R^2 = 1.000000), 0.7985 `powersaver`, 1.0274 `balanced`, 1.8002 `performance`, 1.7998
  `turbo`; a second loop with a different cost (2.000 vs 9.000 cycles per iteration, both
  exactly constant at every length) agrees within 0.33% in every mode
  (`results/aie/clock_probe_npu.log`, Desktop 2 / Phoenix, 2026-09-07). Three things fell
  out. The 16 TOPS nameplate is what 20 cores do at 1.6 GHz; at the measured clock the
  array is 18.4 TOPS and the 16 reachable cores 14.7, so the `4x4` overlay's physical
  ceiling is 92% of nameplate, not 82%. No idle penalty at the 5 s scale (1.77-1.79 GHz
  after 5 s idle), so the session-to-session drift is not an idle clock state -- but a
  power-mode change between sessions would produce exactly that symptom, and no log here
  records the mode. And the tooling: Peano cannot read the cycle counter at all
  (`get_cycles()` is declared, never defined; `__builtin_readcyclecounter` and inline asm
  both die in the backend), pyxrt's `max_clock_frequency_mhz` reads 800 in every mode,
  `xrt-smi configure --pmode turbo` errors and switches anyway, and mlir-aie v1.4.2's
  trace parser mis-times any gap over 2^18 cycles (146 us) -- the harness carries a
  corrected decoder, cross-checked against upstream on a sync-free run. Not run: the
  concurrent-VitisAI-EP leg; only one core tile was measured. Full treatment in
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#the-aie-core-clock-measured-180-ghz-default-080-powersaver).
- **The per-dispatch floor, finally measured in isolation.** Every isolated-op verdict
  above rested on a "~185-200us" constant inherited from one 96 KB probe during the
  GroupNorm work and never measured on its own. `kernels/dispatch_floor/measure_floor.py`
  measures it with a design that has **no compute tile at all** (shim->memtile->shim via
  `ObjectFifo.forward()`), so there is no kernel math to attribute time to; payloads sweep
  8 KB-32 MB, output is verified per payload, and compile is excluded
  (`results/aie/dispatch_floor_npu.log`, Desktop 2 / Phoenix).
  **The old constant was right about the hardware: 169.8 us** (R^2 = 1.0000). What nobody
  had separated is that a kernel does not *pay* the hardware floor -- through the
  `@iron.jit` path it pays **617.0 us** (R^2 = 0.9998), 3.6x more, and both hand-written
  kernels were charged that while their write-ups reasoned with the hardware number.
  The 447.3 us difference is host-side and flat in payload size, but it is **not** simply
  "Python overhead": the hardware bracket is narrow (opening only after the hw_context
  lookup, kernel-handle retrieval and buffer coherence), and cProfile -- whose own overhead
  is ~0.36 ms/call here -- puts its largest entry inside a C call it cannot see into.
  Tracked Python work is ~100 us/call; the `nt.stat`x8 / `_getfinalpathname`x4 /
  `inspect._signature_from_callable`x6 *per launch* is real waste worth hoisting but is the
  minority. How much of the 447 us is recoverable is **unmeasured** -- a cached-handle
  resubmit, a C++ host, or a batched submit are the candidates, and that is the next
  measurement rather than a claim. Two further results fall out: **dispatch dominates
  everything below ~0.5 MB** (wall time flat across a 64x payload range, 8 KB->512 KB;
  transfer only visible past ~2 MB; 12.2-13.8 GB/s streaming), and a **go/no-go threshold
  to apply before writing a kernel** -- the op's CPU time must exceed ~617 us through IRON,
  or ~170 us on a hypothetical zero-overhead path. Both are lower bounds: this is a
  no-compute passthrough, and a real multi-core kernel's configuration cost sits inside the
  hardware bracket and pushes its own floor above 169.8 us.

  **Heterogeneous splice: now measured at 3.25 ms / 2.31x, replacing the reported
  4.47 ms / 4.1x.** `tools/splice_wall_clock.py` wraps a `perf_counter` around a real
  in-process loop; every row below comes from the same run, since NPU latency drifts
  between sessions on this machine (`results/mobilevit/splice_wall_clock_npu.log`,
  100 iterations, Desktop 2 / Phoenix):

      cut CNN backbone, NPU                    1.71 ms   (407/409 nodes, 1 subgraph)
      cut CNN backbone, CPU                    5.65 ms   -> 3.30x like-for-like
      attention x9 blocks, CPU (torch, 8 thr)  1.41 ms
      SPLICE (NPU backbone + CPU attention)    3.25 ms   residual +0.13 ms
      full model FP32, ORT CPU EP              7.51 ms   -> splice 2.31x
      full model XINT8, stock graph on NPU   108.29 ms   (1037/156/392, 49 metaDef / 58 IPU subgraphs)

  Three things change. (1) **The in-process residual is 0.13 ms, not 0.72** -- the old
  number was back-solved from a hardcoded total; real in-process handoff is nearly free.
  (2) **The speedup is 2.31x, not 4.1x**: the old 18.37 ms baseline was PyTorch eager
  while the splice ran under ORT. Measured here, PyTorch eager is 15.71 ms (consistent
  with 18.37 after drift) but ORT CPU runs the same FP32 graph in 7.51 ms, so comparing
  an ORT splice against a PyTorch baseline inflated the win ~1.8x. (3) The 108.00 ms
  stock-EP claim **reproduced almost exactly** at 108.29 ms, as did the 1.73 ms backbone
  and its 1-subgraph placement.
  **The CPU-side kernel choice, not the NPU, decides the verdict.** The same nine blocks
  cost 1.41 ms in torch and 12.73 ms in numpy -- 9x, from multithreaded batched GEMM and
  a fused softmax. With numpy the identical splice is 14.57 ms, i.e. **0.52x -- it loses
  to plain CPU.** Every "the NPU wins" claim about a heterogeneous pipeline is partly a
  claim about which CPU implementation it was allowed to beat, and this repo should state
  which one it used.
  **It is a cost model, not a pipeline.** `mobilevit_cut_backbone_xint8.onnx` is
  `[1,3,256,256] -> [1,1000]`: a complete classifier with the transformer blocks deleted,
  not a backbone emitting intermediates for an attention stage. Nothing connects the two
  halves and the composite computes nothing valid (the quantized backbone is 0% top-1 by
  itself). That is also all the 4.47 ms ever meant. It does NOT use the AIE attention
  kernel, and splitting it across processes (the python 3.12 vs 3.13 pyxrt ABI wall)
  would meet the `groupnorm_bf16` IPC floor (789 µs-23.6 ms) and lose the margin outright.

- **Follow-up: MobileViT-XXS collapses under per-tensor INT8, and the cause is the
  depthwise scale grid — not channel death, and not activation range.** Measured on the
  full 1000-image eval set with real-data (300-image) calibration, all CPU
  (`./scripts/mobilevit-eval.sh`, `results/mobilevit/eval_*.log`): FP32 **68.30%/88.20%**
  (8.77 ms); full XINT8 **0.00%/0.10%**; hybrid CNN-XINT8/transformer-FP32 **0.10%/0.30%**;
  hybrid + AdaRound (500 iters, real data) **0.80%/2.50%**. Real-data calibration does not
  rescue it — the earlier `UseRandomData=True` probes were not the problem. Checked rather
  than assumed, since a random probe also scores 0%: the old probe is still on disk and the
  two calibrations are nothing alike (activation scales 0.000122–4.0, a 32768× spread, vs
  0.0078–0.5; and **zero** dead depthwise channels vs 28). The probe fails anyway, which is
  a third independent strike against the channel-death explanation below. Provenance
  caveat: only the FP32 and full-XINT8 models are reproducible from this repo; the two
  hybrids came from a cut + AdaRound path that is not committed, so their calibration size
  and iteration count are reported, not verified.
  **Retraction: the FP32 baseline is 68.30%, not the 75.0% previously published.** 75.0%
  was the first 100 images; 1000 images give 68.30%, matching the paper's ~69.0%.
  Reproduced on purpose as a row of `--slice`: identical weights, 75.00% at n=100 and
  68.30% at n=1000. Same trap as yolov8s AdaRound (45.19 on a slice, 39.98 on 5000);
  this repo has now been bitten by it twice, which is why "a slice is not the answer" is
  an invariant and not a style note.
  The mechanism is static and reproducible from the artifacts alone
  (`tools/audit_quant_grid.py`, `results/mobilevit/quant_grid_audit.log`). MobileNetV2
  survives the identical recipe at 73.40%, so the discriminator had to be something both
  models do not share. It is **not** dead channels: MobileNetV2 carries a 25.0%-dead
  depthwise block and 186/9920 dead channels in its other convs, against MobileViT's
  3/1864 — MobileViT's non-depthwise convs are *healthier*. It is **not** activation
  range: both bottom out at the same coarsest activation scale, 0.5. What differs is the
  **depthwise weight-scale grid**: MobileNetV2 never exceeds Δ=0.25, MobileViT-XXS reaches
  **Δ=1.0** on a 3×3 depthwise kernel — 4–16× coarser — at which point nearly every weight
  rounds to 0 or ±1 and the layer degenerates into a sign map. The upstream cause is
  **open**: the standing SiLU-vs-ReLU6 / Cross-Layer-Equalization explanation was logged
  and **refuted** (`results/mobilevit/cle_pattern_count.log`, raw Quark logs in
  `cle_probe_mobilenetv2_quant.log` / `cle_probe_mobilevit_quant.log`). The counts come
  out as predicted — MobileNetV2 **3**, MobileViT-XXS **0** — which reads as confirmation
  and is not one. Re-running Quark's own `get_cle_pattern_pair` shows the 3 are 2 distinct
  pairs against 52 convs, and **neither contains a depthwise conv** (both pointwise:
  `conv_pw`→`conv_pw`, `conv_pwl`→`conv_head`). CLE never reaches a depthwise layer in
  either model, so it cannot bound the depthwise grid that *is* the measured
  discriminator. The premise was independently wrong: **ReLU6 is not positive-homogeneous**
  (`ReLU6(2·4)=6 ≠ 2·ReLU6(4)=8`). Quark's matcher accepts only
  `['Relu','ReduceMean','Pad','LeakyRelu']` between convs — `Clip` absent — and
  MobileNetV2's ReLU6 exports as 35 `Clip` nodes with zero `Relu`, so its activations block
  the walk exactly as SiLU does; the opt-in `ReplaceClip6Relu` rewrite (default `False`,
  never set here) exists precisely *because* ReLU6 fails the precondition. MobileViT is
  rejected twice: `Sigmoid` is not in the accepted set, and 29/36 of its convs feed both
  the `Sigmoid` and the `Mul`, breaking the matcher's single-consumer precondition first.
  The 3-vs-0 gap is residual activation-free `Conv`→`Conv` adjacency, nothing more.
  Incidental and unmeasured: the MobileViT XINT8 op table shows `HardSigmoid(34)` where
  FP32 has `Sigmoid(34)`; that is a post-hoc activation swap and cannot set a weight scale,
  so it is not a candidate for the vacated slot.
  Why AdaRound can't help, now mechanical rather than asserted: it selects between
  `floor(w/Δ)` and `ceil(w/Δ)` and never alters Δ. The audit shows the scale grid
  byte-identical pre/post AdaRound, with only the dead-channel count shifting at the
  rounding boundary (28/432 → 24/432) — worth 0.8 points of top-1. **A SiLU/GELU backbone
  on XDNA1 needs QAT or per-channel scales, not a better PTQ recipe.**
- **Follow-up: the VitisAI EP places no rank-5 tensor on the NPU, ever.** Across
  `mobilevit_stock`'s 1585 nodes, **207 nodes touch a rank-5 tensor and all 207 are on
  CPU — no exceptions** (`python tools/audit_quant_grid.py --rank-audit`, reading the EP's
  own `vitisai_ep_report.json`, which it writes on every session build). The control holds:
  YOLOv8's 16 rank-4 `Slice` nodes all place on NPU, and MobileViT is the only model in
  this repo that emits a rank-5 tensor at all — which is why this never surfaced before.
  Sufficient, but not necessary: 18 rank-4 `Transpose`, 18 rank-4 `MatMul` and 21 rank-3
  `LayerNormalization` nodes also fall back, on op support rather than rank. MobileViT's
  5D tensors come from its unfold/fold (`[4, 256, 3, 4, 16]`) between conv and transformer
  stages, so this is a structural property of the architecture, not a quantization
  artifact.

## Future pipeline test plans (Categories A through E)

Based on the hardware invariants established across classification, detection, pose estimation, and kernel sweeps on AMD XDNA1 (Phoenix/Hawk Point, `4x4.xclbin`, VOE 4.0, VitisAI EP), candidate models must satisfy five structural criteria to run efficiently: arithmetic intensity >= 150 MACs/weight-byte, channel widths >= 64, pure standard 2D convolutions/pooling/elementwise ops without LayerNorm/Softmax in the compiled subgraph, single monolithic subgraph placement (>98% in `vitisai_ep_report.json`), and static batch-1 shapes.

The following test plans define candidate models, hypotheses, verification metrics, and falsification criteria across five untested application domains.

### Category A: Image Super-Resolution and Restoration

Super-resolution models are structurally matched to XDNA1: 100% convolutional, zero LayerNorm or Softmax, large spatial activations (H x W), and high arithmetic intensity that stresses AIE MAC utilization rather than DMA dispatch overhead.

- **Candidate architectures:**
  1. Real-ESRGAN Compact (16-block residual convolution chain + PixelShuffle / ConvTranspose upsampler; 4x scaling).
  2. SESR-M7 / ESPCN (Efficient Sub-Pixel Convolutional Neural Network; 3x3 and 5x5 convs with PReLU; 2x to 4x scaling).
- **Hypothesis:** High-resolution conv-only upscaling graphs sustain >30% of the 16 TOPS nameplate without graph fragmentation, achieving sub-10 ms per-tile latency.
- **Target shapes and pipeline:** Static input `(1, 3, 256, 256)` -> static output `(1, 3, 1024, 1024)` (4x) or `(1, 3, 512, 512)` (2x).
- **Quantization:** Quark XINT8 PTQ + AdaRound calibrated on high-frequency image crops (DIV2K / Set14).
- **Verification protocol:**
  1. Inspect `vitisai_ep_report.json` via `tools/diag_ep.py` to confirm 100% NPU assignment. Specifically test whether ONNX `Resize` (nearest/bilinear) or `ConvTranspose` / `DepthToSpace` (PixelShuffle) compiles natively on AIE or fractures into CPU subgraphs.
  2. Measure PSNR and SSIM on standard benchmarks (Set5, Set14) across FP32, plain XINT8, and AdaRound to evaluate quantization loss on image reconstruction.
  3. Profile latency and achieved TOPS via `tools/estimate_tops.py`.
- **Falsification criteria:** The pipeline fails if sub-pixel shuffling or spatial upsampling operations fall back to CPU, incurring cross-device transfer overhead that negates convolutional acceleration.
- **Negative result (Real-ESRGAN Compact at 256x256):** Real-ESRGAN Compact (64-channel residual dense chain) explodes intermediate activation memory across residual concatenations (12.6 MB per activation tensor), exceeding on-chip tile memory and fracturing into **81 DPU subgraphs** with **1,068 nodes on CPU** and only 707 on NPU.
- **Real-ESRGAN 10-RRDB at 64x64 and 128x128 monolithic scaling:** Sizing static input tiles to 64x64 (~196 KB INT8) and 128x128 (~786 KB INT8) drops activation tensors below the host-spill threshold, allowing AMD's 10-RRDB architecture (156 Convs, 120 Concats, 123 LeakyReLUs) to compile into **exactly 1 monolithic DPU subgraph (1,773 / 1,775 nodes on NPU, 99.9%)** with 0 internal CPU fallbacks. At 64x64, runs at **14.02 ms per tile (71.3 fps)** on Phoenix XDNA1 with AdaRound — **3.71x faster than 8-core Zen 4 CPU (51.95 ms)** and outperforming Radeon 780M iGPU DML on full Set5/Set14 tiled evaluation (14.39 ms vs 18.34 ms on Set5). At 128x128, runs at **27.27 ms per tile (36.7 fps)** on NPU plain XINT8 — **9.82x faster than Zen 4 CPU (267.74 ms)** and **1.27x faster than Radeon 780M iGPU DML FP32 (34.67 ms)** on single-tile execution, while delivering **2.06x faster throughput** than four stitched 64x64 tiles (56.08 ms). Conversely, SRVGGNet-v3 Compact revealed an unsupported op rejection: `PRelu` is refused by the VitisAI EP on AIE, falling back to CPU. Full working: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-a-cont-high-capacity-super-resolution-real-esrgan-on-xdna1-npu).

### Category B: Real-Time Portrait Matting and Semantic Segmentation

Matting and bilateral segmentation provide zero-trimap background separation for real-time video conferencing, pairing with the multi-partition camera capture infrastructure in `scripts/yolo-demo.sh`.

- **Candidate architectures:**
  1. MODNet (Objective-Oriented Trimap-Free Portrait Matting; MobileNetV2-derived backbone with semantic, detail, and fusion branches). **Tested** —
     see [docs/BENCHMARKS.md](docs/BENCHMARKS.md#5-opencv-calibration-fix-error-reduction-and-the-structural-zero-concat-gap-2026-09-08-desktop-2).
  2. BiSeNetV2 / STDC (Bilateral Segmentation Network; separate wide shallow detail branch and deep semantic branch). **Tested** —
     see below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-b-second-candidate-bisenetv2-bilateral-segmentation-network).
- **Hypothesis:** Multi-branch convolutional matting executes trimap-free at 512x512 with >98% NPU node residency, delivering sub-10 ms alpha matte generation suitable for 30+ fps webcam background replacement with near-zero CPU load.
- **Target shapes and pipeline:** Static input `(1, 3, 512, 512)` -> static output `(1, 1, 512, 512)` alpha matte in `[0, 1]`, or `(1, 19, 512, 512)` multi-class segmentation logits.
- **Quantization:** Quark XINT8 PTQ with portrait calibration (PPM-100 / portrait subsets) + AdaRound for fine boundary refinement.
- **Verification protocol:**
  1. Verify compiler node acceptance in `vitisai_ep_report.json`. Check whether bilinear upsampling in the fusion branch triggers subgraph partitioning.
  2. Evaluate alpha matte boundary fidelity: Mean Absolute Difference (MAD), Sum of Absolute Differences (SAD), and Mean Squared Error (MSE) relative to FP32 reference.
  3. Deploy in an interactive video pipeline (`pipelines/modnet/4_matte.py`) under `resnet_env17` to measure end-to-end webcam frame latency, alpha composition overhead, and NPU utilization.
- **Falsification criteria:** If depthwise separable layers in MODNet's backbone exhibit the Delta=1.0 scale grid collapse observed in MobileViT, or if multi-scale feature fusion forces CPU round-trips, the model requires backbone replacement (e.g. standard ResNet/BiSeNet detail branch).
- **BiSeNetV2 result:** Bilateral Guided Aggregation (BGA) with nearest-neighbor upsampling compiles natively into **1 monolithic DPU subgraph of 402 / 404 nodes (99.5%)** with 0 internal CPU fallbacks (`subgraphStat: [{'device': 'DPU', 'count': 1}]`). All 57 Convs, 40 Relus, 10 Adds, 5 Muls, 3 nearest-neighbor Resizes, and both HardSigmoid gating activations execute natively on AIE. Runs in **13.12 ms (76.2 fps)** on Phoenix XDNA1 — **4.43× faster than 8-core Zen 4 CPU (58.07 ms)** and **1.09× faster than Radeon 780M iGPU DML FP32 (14.25 ms)**, and 2.17× faster than MODNet Cut (28.45 ms). Bilinear upsampling at the head triggers host CPU fallback on 1 Resize (399/404 on NPU, +0.26 ms). Under CPU floating-point QDQ simulation, XINT8 maintains good fidelity (59.47% Pixel Accuracy, 25.72% mIoU against FP32 across 50 scenes); however, on physical DPU hardware, fixed-point dynamic range truncation across the elementwise bilateral multiplication (`left * HardSigmoid(right)`) attenuates minority class activations (15.33% Pixel Accuracy, 2.44% mIoU), confirming that multi-branch bilateral gating requires fine-tuning or AdaRound to balance inter-branch scale grids on physical systolic hardware. Full working: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-b-second-candidate-bisenetv2-bilateral-segmentation-network).

### Category C: Advanced Detection and RepVGG Backbones

Structurally re-parameterized networks collapse multi-branch training graphs into a single linear sequence of standard 3x3 convolutions with ReLU at inference, eliminating residual Add branches.

- **Candidate architectures:**
  1. YOLOv6 (Meituan RepVGG backbone; pure 3x3 convs + ReLU in inference mode). **Tested** —
     see below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-c-first-candidate-yolov6n-repvgg-backbone).
  2. YOLO-World v2 (Open-vocabulary detection; decoupled CPU text embedding + NPU vision backbone). **Tested** —
     see below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-c-third-candidate-yolo-world-v2-vision-language-decoupled-cross-attention).
  3. YOLOv11 (Successor detection architecture with C3k2 blocks, evaluated under the established 6-output head-cut pattern). **Tested** —
     see below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-c-second-candidate-yolov11n-c2psa-attention-block--decoupled-dwconv-head).
- **Hypothesis:** Eliminating residual `Add` branches via structural re-parameterization reduces SRAM buffer contention and DMA ping-ponging, improving single-column execution efficiency relative to YOLOv8 CSPDarknet blocks while plain ReLU avoids SiLU->HardSwish quantization distortion.
- **Target shapes and pipeline:** Static input `(1, 3, 640, 640)`, head-cut architecture exporting raw box and class tensors directly (`1b_cut_head` recipe).
- **Quantization:** Quark XINT8 + AdaRound calibrated on COCO val2017.
- **Verification protocol:**
  1. Re-parameterize YOLOv6 to inference mode before export.
  2. Cut decode heads and verify 6-output shape contract.
  3. Measure compiled node placement, per-frame latency, and mAP@50-95 on the full 5000-image COCO val2017 benchmark.
- **Falsification criteria:** If re-parameterized weight distributions exhibit high dynamic range outliers that degrade INT8 PTQ accuracy beyond the recovery capacity of AdaRound, or if the compiler fails to fuse adjacent Conv+ReLU layers efficiently.
- **YOLOv6n result:** placement (98.7%, 518/525 nodes, single subgraph) and speed (3.0x
  over FP32 CPU) both confirm the structural half of the hypothesis, matching yolov8n's own
  head-cut numbers rather than beating them — Add-free RepVGG doesn't measurably help SRAM
  contention here, it's simply not worse. Plain XINT8 (no AdaRound) loses 14.03 points of
  mAP@50-95, more than yolov8n's plain-XINT8 loss on the same convention. AdaRound (same
  recipe used everywhere else in this repo, no yolov6n-specific tuning) recovers +10.65 of
  those points (76%) at zero latency cost, closing to within 3.38 of FP32 — **refuting**
  the "beyond AdaRound's recovery capacity" branch of the falsification criterion below.
  Both halves of the candidate are now closed: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-c-first-candidate-yolov6n-repvgg-backbone).
- **YOLOv11n result:** C2PSA spatial self-attention block is rejected by the DPU compiler
  (4D `MatMul` inside attention loop forces 1294 nodes to CPU; only 6 land on NPU, running at
  33.29 ms eval). However, ablating C2PSA unlocks a single monolithic DPU subgraph of
  1,173 / 1,180 nodes (99.4%) running at **7.08 ms** (141.2 fps) across 5,000 COCO images —
  the fastest YOLO model ever measured on XDNA1 (1.24x faster than YOLOv8n, 1.18x faster than
  Radeon 780M iGPU DML). Plain XINT8 stock loses 12.90 mAP (38.72 → 25.82), while identity
  ablation collapses mAP to 0.19, proving attention cannot be stripped post-hoc without retraining:
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-c-second-candidate-yolov11n-c2psa-attention-block--decoupled-dwconv-head).
- **YOLO-World v2 result:** 5D text cross-attention (`Einsum`, `ReduceMax`) in `C2fAttn` is rejected
  by the DPU compiler, placing only 48 / 1081 nodes on NPU and forcing all 67 Convolutions to CPU
  (178.53 ms latency). Eliminating `Reshape` and cross-attention unlocks a single monolithic DPU subgraph
  of 946 / 953 nodes running at **15.89 ms (62.9 fps)** over 5,000 images — **2.46× faster than
  Radeon 780M iGPU DML** (40.42 ms) and **5.10× faster than Zen 4 CPU** (81.11 ms). Plain XINT8 stock
  collapses to 1.8% mAP due to 5D quantization error, while attention-free ablation collapses to 0.3%
  mAP due to severed CLIP alignment:
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-c-third-candidate-yolo-world-v2-vision-language-decoupled-cross-attention).

### Category D: Monocular Depth Estimation

Dense geometric scene prediction from single monocular camera streams without transformer attention mechanisms.

- **Candidate architectures:**
  1. MiDaS v2.1 Small (EfficientNet-Lite / MobileNet backbone with multiscale feature fusion decoder). **Tested** —
     see below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-d-monocular-depth-estimation-midas-v21-small).
  2. FastDepth (MobileNet encoder with depthwise separable conv decoder). **Tested** —
     see below and [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-d-second-candidate-fastdepth-mobilenet-nnconv5dw).
- **Hypothesis:** Pure convolutional encoder-decoder depth estimation produces dense relative inverse depth maps at 256x256 or 384x384 in 5-8 ms on NPU, providing real-time spatial scene representation for synthetic bokeh and spatial interaction.
- **Target shapes and pipeline:** Static input `(1, 3, 256, 256)` or `(1, 3, 384, 384)` -> static output `(1, 1, H, W)`.
- **Quantization:** Quark XINT8 + AdaRound on NYU-Depth / KITTI image patches.
- **Verification protocol:**
  1. Audit node placement in `vitisai_ep_report.json`.
  2. Check depth boundary sharpness and relative depth metrics (AbsRel, RMSE) against FP32 ground truth.
  3. Confirm whether depthwise decoder layers avoid the scale grid collapse observed in MobileViT.
- **Falsification criteria:** If multiscale residual connections in the decoder cause frequent memory spills or CPU fallback.
- **MiDaS v2.1 Small result:** Stock bilinear upsampling causes CPU fallback on 4 Resize nodes, fragmenting DPU execution into 5 subgraphs and yielding 16.44 ms. Converting decoder Resize layers to nearest-neighbor fuses the model into a single monolithic DPU subgraph (682/684 nodes on NPU, 99.7%), accelerating inference by 34% to **10.81 ms (92.5 fps)** — a **1.53× win over 8-core Zen 4 CPU (16.56 ms)**. Quantization fidelity is strong under plain XINT8 PTQ without requiring AdaRound (Pearson $r = 0.8706$, MAD $26.02 / 255$, RMSE $34.00 / 255$), refuting the depthwise scale grid collapse fear. Full working: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-d-monocular-depth-estimation-midas-v21-small).
- **FastDepth result:** Depthwise separable decoding (`NNConv5dw-skipadd`) compiles natively into a **single monolithic DPU subgraph (255/257 nodes, 99.2%)** with 0 internal CPU round-trips. Executes in **2.87 ms (348.1 fps)** on Phoenix XDNA1 — **1.12× faster than 8-core Zen 4 CPU (3.22 ms)** and **1.05× faster than Radeon 780M iGPU DML FP32 (3.02 ms)**, running 3.77× faster than MiDaS v2.1 Small. Plain XINT8 PTQ preserves depth structure with strong fidelity (Pearson $r = 0.9383$, MAD $16.14 / 255$, RMSE $21.06 / 255$, $\delta < 1.25 = 68.07\%$), outperforming MiDaS accuracy without needing AdaRound. Full working: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#category-d-second-candidate-fastdepth-mobilenet-nnconv5dw).

### Category E: Untested Classification Topologies

Characterizing the boundary conditions of the XDNA1 compiler on alternative convolutional connection topologies: dense concatenation and grouped convolutions.

- **Candidate architectures:**
  1. DenseNet-121 (Dense connectivity via channel `Concat` across blocks; 120 Convs, 62 Concats, 3 AveragePools).
  2. ResNeXt-50 (32x4d) (Grouped convolutions; 53 Convs with 32 groups of 4 channels each).
  3. RegNetX (RegNetX-002 through RegNetX-080; regular linear channel capacity design space).
- **Hypothesis:** DenseNet channel concatenation stresses AIE DMA memory bandwidth as channel width accumulates, revealing whether `Concat` carries higher latency overhead than ResNet elementwise `Add`. ResNeXt tests whether grouped convolutions compile to native AIE micro-kernels or trigger unoptimized scalar loops.
- **Target shapes and pipeline:** Static input `(1, 3, 224, 224)`.
- **Measured findings (DenseNet-121 & ResNeXt-50):**
  1. **Hardware offload is complete**: DenseNet-121 places **1,703 / 1,705 nodes (99.9%)** on NPU at **8.06 ms** (2.69× speedup over Zen 4 CPU FP32 at 21.70 ms); all 58 Concat nodes execute natively on AIE without DMA bottlenecks. ResNeXt-50 places **393 / 395 nodes (99.5%)** on NPU at **9.37 ms** (1.86× speedup over Zen 4 CPU FP32 at 17.40 ms); all 53 `group=32` convs compile natively.
  2. **Coarse per-tensor PTQ fails completely**: Both architectures collapse to **0.10% top-1** under plain XINT8 (vs 78.00% / 81.00% FP32 baselines). Dense block concatenation forces disparate block activations into shared power-of-two scales (causing shift cuts to exceed `[0, 16]` by up to 7 powers of 2), while 4-channel grouped convs produce wide cross-group dynamic range divergence that per-tensor INT8 cannot represent.
  3. **RegNetX-002 proves the collapse is DPU shift-cut scale explosion, not concatenation or narrow groups**: RegNetX-002 (2.68M parameters, regular linear channel capacity) places **324 / 326 nodes (99.4%)** on NPU at **2.45 ms (407.9 FPS)**, but also collapses to **0.50% top-1** under plain XINT8 and **0.50% under AdaRound** (vs 68.50% FP32 baseline). Static inspection via `tools/audit_quant_grid.py` traced this to Quark's DPU shift-cut clamp ($131 \to -108$) forcing $\Delta = 2^{108} \approx 3.25 \times 10^{32}$ on depthwise weights, spanning an $8.5 \times 10^{35}\times$ activation dynamic range. AdaRound cannot modify the scale grid $\Delta$ and thus cannot recover the network.
- **Falsification verdict:** Refuted for latency (neither `Concat` memory copies nor grouped conv micro-kernels stall the NPU), but confirmed as a severe failure mode for plain per-tensor INT8 PTQ and unrecoverable by AdaRound when shift cuts cause floating scale explosion.

## Roadmap

Where the open measurements stand. Everything below that closed has its numbers, method
and caveats in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); this list keeps one line each so the trail from
question to answer stays visible, and the full reasoning behind each question is in the
sections above.

**Closed.**

- **`pipelines/yolov8n-pose` end to end on the NPU.** 1015/1025 nodes, 9.8 ms/frame,
  XINT8 costs 17.2 points of OKS mAP@50-95 on the full 5000-image set — AdaRound for pose
  is measured separately below.
  [Working](docs/BENCHMARKS.md#yolov8n-pose-end-to-end-on-the-npu).
- **AdaRound for YOLOv8s and YOLOv8m at 640×640.** Not RAM-blocked after all, and it
  barely helps either one: +2.58 mAP at s, +1.83 at m, against classification's ~90%
  recovery. Why detection recovers so much less is still unexplained.
  [Working](docs/BENCHMARKS.md#adaround-on-detection-how-much-does-it-actually-recover).
- **Concurrent streams past 2, and on a wider model.** Both saturate — yolov8n at 3
  streams (~167 fps, 2.1×), yolov8m a stream earlier at 1.29×.
  [Working](docs/BENCHMARKS.md#two-cameras-does-independent-concurrency-work-where-batching-doesnt).
- **Is stream saturation compute or memory?** Compute. NPU memory scales linearly with
  stream count (~29 MB/stream yolov8n, ~100 MB/stream yolov8m) with no ceiling through 8,
  decoupled from the throughput plateau. `results/nstream_memory_yolov8{n,m}.log`. **Now
  confirmed on a classifier too, at conservative stream counts (1/8/16/32) after the
  128-stream WDDM abort found below**: `wide_resnet101_2` memory scales exactly linearly
  at ~140.6 MB/stream (steeper than either detection model, reaching 4.5 GB by 32
  streams), GOPS is exactly `46 × streams` with no ceiling, and measured completions/s
  stays flat at ~62 fps from 1 stream onward — the same decoupling, on a third model
  family. `results/session_hold_wide_resnet101_2.log`.
- **Does classification saturate the same way, and does accuracy survive contention?**
  Yes to both: resnet50 saturates at 1.60× by 8 streams and holds flat through 16, with
  bit-identical argmax on all 960 concurrent classifications tested.
  `results/nstream_resnet50.log`. **Does a wider classifier saturate earlier, like
  yolov8m does against yolov8n — done.** `wide_resnet50_2` reaches essentially the same
  final multiplier (1.59× vs resnet50's 1.60×) but flattens by 2 streams instead of 8,
  and every concurrent classification at every stream count still matched the
  uncontended baseline exactly (0.00% mismatch). `results/nstream_wide_resnet50_2.log`.
  **Both remaining gaps closed together: `wide_resnet101_2` and streams past 16, up to
  32.** Its ceiling is lower still (1.32×, flat by 3 streams — the least idle headroom of
  the three classifiers) and holds flat with zero regression through 32 streams, with
  the same 0.00% mismatch guarantee at every count. `results/nstream_wide_resnet101_2.log`.
  **Pushed to 128 to find where it actually breaks — it does, and not gracefully.**
  32/48/64/96 all confirm the same plateau again; 128 aborts partway through session
  construction (~121/128 built) with a fatal XRT hardware-queue error — WDDM unable to
  page in enough concurrent hw-context allocations into video memory, not a Python-side
  OOM, though host RAM was genuinely tight at the time (~6.4 GB free of 32 GB) with
  another process's build also running on this shared machine, so the exact breaking
  point isn't a clean isolated number. `results/nstream_wide_resnet101_2_128_ceiling.log`.
- **yolov8l and yolov8x.** The width trend breaks: x is 2.36× the latency of l for a net
  mAP loss. Both calibrated smaller (32/24) than m's 64, and l's full eval is flaky.
  [Working](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together).
- **A full utilization-vs-TOPS story from `xrt-smi`'s GOPS column — a dead end, and
  retracted.** GOPS scales exactly linearly with stream count while measured completion
  rate is flat from 1 stream on. The real answer came from `tools/estimate_tops.py`
  instead. [Working](docs/BENCHMARKS.md#achieved-opss-a-real-answer-to--of-16-tops-not-a-gops-estimate).
- **5th AIE column on this Phoenix chip — a dead end.** `1x4.xclbin` caps at 4
  independent partitions regardless of process count, and the driver's `5x4_*.xclbin`
  overlays fall back silently to 100% CPU.
  [Working](docs/BENCHMARKS.md#splitting-the-array-into-independent-partitions).
- **int8 GEMM against the CPU's own int8 kernel.** Loses at `whole_array`'s default tile
  (the CPU is strongest at exactly the NPU's headline dtype), wins 1.10×–1.83× at M ≥ 512,
  N ≥ 2048 with `n=64` — a tile int8's half-size buffers fit in L1 and bf16's miss by the
  stack.
  [Working](docs/BENCHMARKS.md#int8-gemm-the-npus-headline-dtype-needs-a-tile-bf16-cant-fit).
- **bf16 GEMM at `n=64`.** Single-buffering `whole_array.py`'s C output tile (a 13-line
  patch, `--c-single-buffer 1`) frees exactly the 16 KB bf16's default double-buffer was
  missing by. Once it fits: the CPU-bf16 win margin widens from 1.19×–1.35× (default
  tile) to 1.29×–1.89× at M, N ≥ 1024 — 2048³'s 1.89× is the largest bf16 GEMM margin
  measured in this project. 512³ still loses (0.70×), same as every other shape here.
  The tile sweep that followed (`results/aie/gemm_tile_sweep_c_single_buffer_npu.log`) found
  64×64 and 32×128 to be the best tiles the CLI can reach — 2501.71 / 2494.61 GFLOPS at
  2048³, 2700.44 at 2048×4096×4096, 1.89× the same-sitting CPU bf16 — 128×64 / 64×128 out of
  reach by exactly the 19,712 B the L1 arithmetic said, and SILICON.md 3.1's B/MAC model
  short of a B-run-length term and a k term.
  [Working](docs/BENCHMARKS.md#the-bf16-tile-sweep-what-the-freed-16-kb-buys-and-where-the-bmac-model-stops).
  [Working](docs/BENCHMARKS.md#bf16-gemm-at-n64-the-same-fix-int8-used).
- **ResNet50's AdaRound latency cost — there wasn't one, and the old 5.63/6.93 ms pair is
  retracted.** A same-sitting `--fresh` rerun of both models at 1000 images (2026-09-07)
  read 5.26 ms and 5.27 ms, with `vitisai_ep_report.json` node-for-node identical between
  them. No log in the repo reproduces the original numbers; `results/bench_xint8_npu.log`
  and its AdaRound sibling exist today at only 100 images, the likely sign of a later probe
  run reusing those log names. [Working](docs/BENCHMARKS.md#results).
- **AdaRound for pose.** Quantized on Desktop 2 (CPU FastFinetune, 300 calib images) and
  evaluated across the full 5,000-image COCO val2017 set: AdaRound recovers +1.68 points
  of OKS mAP@50-95 (32.64 → 34.32) and +4.95 points of OKS mAP@50 (66.90 → 71.85) at
  9.35 ms on NPU (1015/1025 nodes), with zero latency cost. On the original 500-image
  slice, it reads 33.94 (+2.29) / 71.88 (+5.49) against plain XINT8's 31.65 / 66.39.
  Like detection, it does not close the full gap to float (49.86), but delivers a clean,
  cost-free recovery. [Working](docs/BENCHMARKS.md#yolov8n-pose-end-to-end-on-the-npu).
- **AdaRound on the ResNet50 resolution sweep (160², 192², 256², 288²).** Measured on 1000 eval images:
  AdaRound at 160² recovers +9.50 points of top-1 (67.10% → 76.60%) and +6.00 points of top-5
  (85.70% → 91.70%) at 4.55 ms (393/395 nodes, 99.5%, ~220 img/s). AdaRound at 192² recovers
  +8.00 points of top-1 (71.30% → 79.30%) and +5.80 points of top-5 (86.60% → 92.40%) at 5.15 ms
  (~194 img/s). At 256², AdaRound hits 79.80% top-1 / 93.40% top-5 at 5.85 ms (matching 224² top-1
  while gaining +0.90% top-5). At 288², it reaches 78.10% top-1 / 93.40% top-5 at 8.78 ms. This confirms
  that AdaRound creates a broad 79.3%–79.8% accuracy plateau across 192²–256² within 5.15–5.85 ms, with
  160² marking the physical inflection point where spatial downsampling bounds accuracy rather than rounding noise.
  [Working](docs/BENCHMARKS.md#resnet50-input-resolution-does-the-fixed-cost-story-hold-for-a-classifier).
- **A yolov8m mAP row at calibration 200.** Measured on the full 5000-image val2017 set:
  plain XINT8 calibrated on 200 images scores 43.38 mAP@50-95 and 59.62 mAP@50 at 26.95 ms
  (1216/1223 nodes, 99.4%) on NPU (`results/bench/map_yolov8m_cut_xint8_c200_npu.log`,
  `results/bench/lat_yolov8m_cut_xint8_c200_npu.log`, `results/bench/diag_yolov8m_cut_xint8_c200.log`).
  This matches the original calib-64 row (43.49 / 59.79 at 30.80 ms) within 0.11 points, closing the
  calibration-size caveat and confirming that width, not calibration sample count past 64, dominates
  quantization accuracy. The float CPU baseline on the same 5000 images is 49.54 mAP@50-95 / 66.09 mAP@50
  at 144.12 ms (`results/bench/map_yolov8m_cpu.log`, `results/bench/lat_yolov8m_cpu.log`), showing a 5.35×
  NPU speedup. [Working](docs/BENCHMARKS.md#model-size-n-vs-s-measured-together).
- **Category B: Real-Time Portrait Matting (MODNet at 512x512).** Stock MODNet rejected
  outright by VitisAI EP (0/872 nodes on NPU, 132 ms CPU fallback). Bisection through isolated
  subgraphs traced the failure to `IBNorm`'s intra-layer diamond split (`Slice` -> `BatchNorm` +
  `InstanceNorm` -> `Concat`), where an unsupported CPU op (`InstanceNorm`) inside a sliced/concatenated
  layer forces cross-device synchronization and triggers whole-graph compiler refusal. Resolved via
  `modnet_cut`: calibrating empirical running stats for `InstanceNorm` (MAD = 0.0384 vs uncalibrated),
  merging into a unified `BatchNorm2d`, mathematically folding into preceding `Conv2d` weights/biases
  (eliminating all 17 Slice, 17 IN, 17 Concat nodes), refactoring `SEBlock` from Linear to 1x1 Conv2d,
  and cutting the final Sigmoid tail. Measured across 50 validation portraits: **502 of 507 nodes
  (99.0%) on the physical NPU**, **28.45 ms mean latency (35.1 fps)**, delivering a **9.03x speedup
  over 8-core Zen 4 CPU** (256.89 ms) and beating the Radeon 780M iGPU (39.05 ms).
  Those latencies had no backing log and are superseded by a same-sitting re-measurement
  (26.44 ms, 7.93x CPU, 1.75x iGPU); the accuracy figures reproduced exactly.
  Re-quantizing under byte-identical OpenCV preprocessing (2026-09-08) reduced MAD to
  **0.18629** on Cut (-2.1%) and **0.33187** on Zero-Concat (-5.9%), confirming the calibration
  mismatch was inflating error across both models while the 1.78x quality gap remains structural.
  [Working](docs/BENCHMARKS.md#category-b-real-time-portrait-matting-modnet-on-xdna1-npu).
- **Category E: Alternative classification topologies (DenseNet-121, ResNeXt-50, RegNetX-002).**
  Measured on 1000 eval images (DenseNet, ResNeXt) and 200 eval images (RegNetX): all three models
  achieve 99.4%–99.9% NPU placement with zero op-level refusal. DenseNet-121 (1703/1705 nodes on NPU,
  all 58 Concats and 3 AveragePools accepted) runs at 8.06 ms (2.69× speedup over Zen 4 CPU FP32 at 21.70 ms),
  proving Concat does not bottleneck AIE DMA memory bandwidth. ResNeXt-50 (393/395 nodes on NPU) compiles
  all 53 grouped convs (`groups=32`) natively at 9.37 ms (1.86× speedup over Zen 4 CPU FP32 at 17.40 ms).
  RegNetX-002 (324/326 nodes on NPU, 99.4%) compiles natively at **2.45 ms (407.9 FPS)**.
  However, all three topologies suffer catastrophic PTQ collapse under plain XINT8 (0.10% / 0.10% / 0.50% top-1
  vs 78.00% / 81.00% / 68.50% FP32 baselines). RegNetX-002 proves the collapse is not specific to block
  concatenation or narrow groups, and AdaRound cannot rescue it (0.50% top-1): Quark's DPU shift-cut clamp
  forces weight position shifts by 100+ powers of 2 to satisfy 16-bit shift register limits, creating an
  astronomical floating scale distortion ($\Delta = 2^{108} \approx 3.25 \times 10^{32}$) that integer rounding
  cannot recover.
  [Working](docs/BENCHMARKS.md#alternative-classification-topologies-densenet-121-concat-and-resnext-50-grouped-convs).
- **Category C, first candidate: YOLOv6n (RepVGG backbone).** New pipeline
  (`pipelines/yolov6n/`), Meituan's official 0.4.0 release. The structural half of the
  hypothesis holds: RepVGG's `switch_to_deploy()` collapse (no residual `Add`) plus the
  `use_dfl=False` head (no DFL softmax at all) place a single clean 518/525-node (98.7%)
  NPU subgraph, 3.0x faster than FP32 CPU (20.04 -> 6.62 ms at eval settings) — in the same
  range as yolov8n's own head-cut numbers, not better or worse on placement/speed. Plain
  XINT8 (no AdaRound) costs **-14.03 points of mAP@50-95** (36.95 → 22.92) and **-17.00 of
  mAP@50** (51.98 → 34.98) on the full 5000-image set — a substantially bigger hit than
  yolov8n's plain-XINT8 loss — but **AdaRound recovers +10.65 (76%) / +14.86 (87%) of that
  loss** (→ 33.57 / 49.84) at zero latency cost, closing to within 3.38 / 2.14 points of
  FP32. This refutes the "beyond AdaRound's recovery capacity" branch of the
  falsification criterion: re-parameterized RepVGG weights don't carry AdaRound-resistant
  outliers here. Both halves of the candidate are closed.
  [Working](docs/BENCHMARKS.md#category-c-first-candidate-yolov6n-repvgg-backbone).
- **Category C, second candidate: YOLOv11n (C2PSA Attention Block & Decoupled DWConv Head).**
  New pipeline (`pipelines/yolov11/`), Ultralytics YOLOv11 release. Rejection and speed
  records observed in the same model: the C2PSA spatial self-attention block is rejected
  by the VitisAI EP (4D `MatMul` inside attention loop forces 1294 nodes to CPU, placing
  only 6 ancillary nodes on NPU), causing severe host-NPU ping-pong and degrading latency
  to 33.29 ms eval (slower than Zen 4 CPU FP32 at 22.82 ms). However, ablating C2PSA into
  an identity passthrough proves the C3k2 backbone and decoupled DWConv detection heads
  compile into a **single monolithic DPU subgraph of 1,173 / 1,180 nodes (99.4%)**, executing
  in **7.08 ms on 5,000 val2017 images** (141.2 fps) — **the fastest YOLO model recorded
  on XDNA1** (1.24x faster than YOLOv8n, 1.35x faster than YOLOv6n, and 1.18x faster than
  Radeon 780M iGPU DML FP32 at 8.24–8.58 ms). Plain XINT8 stock loses 12.90 mAP (38.72 → 25.82);
  identity ablation collapses mAP to 0.19, demonstrating that attention cannot be bypassed
  without retraining. Both candidate hypotheses closed.
  [Working](docs/BENCHMARKS.md#category-c-second-candidate-yolov11n-c2psa-attention-block--decoupled-dwconv-head).
- **Category C, third candidate: YOLO-World v2 (Vision-Language Decoupled Cross-Attention).**
  New pipeline (`pipelines/yolow/`). Multi-scale text cross-attention (`MaxSigmoidAttnBlock` in
  `C2fAttn`) is rejected by the DPU compiler due to 5D `Einsum` and `ReduceMax` operations,
  ejecting all 67 Convolutions to CPU (48/1081 nodes on NPU, 178.53 ms latency). Eliminating
  `Reshape` and cross-attention unlocks a **single monolithic DPU subgraph of 946 / 953 nodes (99.3%)**,
  running at **15.89 ms across 5,000 val2017 images (62.9 fps)** — **2.46× faster than
  Radeon 780M iGPU DML FP32** (40.42 ms) and **5.10× faster than Zen 4 CPU** (81.11 ms).
  However, open-vocabulary alignment collapses under both plain XINT8 PTQ (1.8% mAP) and attention
  ablation (0.3% mAP), demonstrating that multi-dimensional attention mechanisms cannot be executed
  natively on XDNA1 or severed zero-shot without retraining. All Category C candidate hypotheses
  are now closed.
  [Working](docs/BENCHMARKS.md#category-c-third-candidate-yolo-world-v2-vision-language-decoupled-cross-attention).
- **Category D: Monocular Depth Estimation (MiDaS v2.1 Small and FastDepth).** New pipelines
  (`pipelines/midas/`, `pipelines/fastdepth/`). Nearest-neighbor upsampling in MiDaS fuses all
  RefineNet decoder layers into a single monolithic DPU subgraph (682/684 nodes, 99.7%), eliminating 4 host
  CPU round-trips and accelerating inference by 34% (16.44 -> 10.81 ms, 92.5 fps) — a 1.53x win over Zen 4 CPU.
  FastDepth with pure depthwise-separable decoding (`NNConv5dw-skipadd`) compiles into a single monolithic
  DPU subgraph (255/257 nodes, 99.2%) executing in **2.87 ms on Phoenix XDNA1 (348.1 fps)** — **1.12× faster
  than 8-core Zen 4 CPU** (3.22 ms) and **1.05× faster than Radeon 780M iGPU DML FP32** (3.02 ms), while delivering
  superior INT8 fidelity (Pearson $r = 0.9383$, MAD $16.14 / 255$, $\delta < 1.25 = 68.07\%$) without requiring AdaRound.
  All Category D candidate hypotheses are closed. [Working](docs/BENCHMARKS.md#category-d-second-candidate-fastdepth-mobilenet-nnconv5dw).
- **Category A: Image Super-Resolution (SESR-M7 and Real-ESRGAN).** New pipelines (`pipelines/sesr/`, `pipelines/realesrgan/`).
  Sub-pixel convolution (`DepthToSpace` / PixelShuffle) compiles natively on AIE into a
  single monolithic DPU subgraph (50/52 nodes, 96.2%, zero internal fallbacks). Achieves
  **1.48 ms per 256x256 tile (674.0 fps)** on Phoenix XDNA1 — **5.43x faster than 8-core
  Zen 4 CPU** (8.07 ms) and **3.02x faster than Radeon 780M iGPU DML** (4.48 ms), marking
  the first visual pipeline where the NPU decisively outperforms the iGPU. Plain XINT8 loses
  1.58 dB on Set5 (34.06 dB vs 35.64 FP32); **AdaRound FastFinetune recovers 69.6% (+1.10 dB)
  to reach 35.16 dB (0.9437 SSIM)** at zero latency cost. For high-capacity 4x restoration,
  Real-ESRGAN Compact at 256x256 fractured into 81 subgraphs (12.6 MB activation spill), but
  sizing static tiles to 64x64 and 128x128 resolves SRAM exhaustion completely: AMD 10-RRDBNet
  compiles into a **single monolithic DPU subgraph (1,773 / 1,775 nodes on NPU, 99.9%)**, running
  at **14.02 ms per tile (71.3 fps)** at 64x64 with AdaRound and **27.27 ms (36.7 fps)** at 128x128 on
  plain XINT8 (9.82x faster than Zen 4 CPU, 1.27x faster than Radeon 780M iGPU DML FP32 at 34.67 ms,
  and 2.06x faster throughput than four 64x64 tiles). Both hypotheses closed. [Working](docs/BENCHMARKS.md#category-a-cont-high-capacity-super-resolution-real-esrgan-on-xdna1-npu).

- **Does MODNet's alpha error move once calibration and inference agree?** It moves, and
  the Zero-Concat penalty survives the fix. Every MODNet model measured before 2026-09-08
  was calibrated through PIL bilinear while inference resized with cv2 bilinear — Pillow
  antialiases on downscale, OpenCV does not, so the two were never byte-identical. Both
  variants were re-quantized through the unified `npu/modnet.py` transform and re-evaluated
  on the same 50 validation images: Cut MAD fell **0.19022 → 0.18629** (-2.1%, SAD 50.75k →
  49.11k) and Zero-Concat **0.35269 → 0.33187** (-5.9%, SAD 93.18k → 90.03k), with node
  placement unchanged on both. Under that identical calibration Zero-Concat still reads
  **1.78×** Cut's error, so its worse matte is structural to replacing the skip connections
  with zero-padded channels, not calibration drift.
  [Working](docs/BENCHMARKS.md#5-opencv-calibration-fix-error-reduction-and-the-structural-zero-concat-gap-2026-09-08-desktop-2).
  What the rerun did not close: the calibration reader is still a per-pipeline copy of the
  transform, so nothing structural stops the two drifting apart again. An owned calibration
- **Category B: Real-Time Portrait Matting and Semantic Segmentation (MODNet and BiSeNetV2).**
  MODNet (`pipelines/modnet/`) demonstrated the zero-concat trade-off and confirmed calibration sensitivity.
  BiSeNetV2 (`pipelines/bisenetv2/`) tests bilateral multi-branch segmentation (wide shallow Detail Branch +
  deep Semantic Branch fused via HardSigmoid-gated Bilateral Guided Aggregation). Nearest-neighbor upsampling
  compiles into a **single monolithic DPU subgraph (402/404 nodes, 99.5%)** executing in **13.12 ms on
  Phoenix XDNA1 (76.2 fps)** — **4.43× faster than 8-core Zen 4 CPU (58.07 ms)** and **1.09× faster than
  Radeon 780M iGPU DML FP32 (14.25 ms)**. Bilinear head upsampling causes CPU fallback on 1 Resize (+0.26 ms).
  Under CPU QDQ simulation, XINT8 maintains good fidelity (59.47% Pixel Accuracy, 25.72% mIoU); on physical
  DPU hardware, fixed-point dynamic range truncation across the elementwise bilateral multiplication
  attenuates minority classes (15.33% Pixel Accuracy, 2.44% mIoU), confirming that multi-branch bilateral
  gating requires fine-tuning or AdaRound for physical systolic hardware. All Category B candidate hypotheses
  are closed. [Working](docs/BENCHMARKS.md#category-b-second-candidate-bisenetv2-bilateral-segmentation-network).

**Still open.**

- **How far does Ignition's XINT8 parity extend beyond folded ResNet?**
  [Alpha scope and usage](quant/README.md) and the [implementation backlog](quant/TODO.md)
  now distinguish the runnable release from the broader design.
  [ResNet re-emission and independent calibration](docs/BENCHMARKS.md#owned-resnet50-no-cle-re-emission-and-independent-calibration)
  now reproduce the fresh no-CLE reference from the float export: graph connections,
  scales, zero points and integer weight/bias data match exactly. The owned calibration
  process blocks Quark and torch imports. Full-set accuracy and paired NPU evidence
  are recorded in the linked section. [Controlled acceptance probes](docs/BENCHMARKS.md#ignition-controlled-resnet-qdq-acceptance)
  now isolate domain-only fallback, exact signed-activation NPU parity on the full set,
  and metadata independence on this graph. They also expose numerical failures despite
  NPU placement and an optimizer-dependent CPU reference discrepancy. The
  [refinement probe](docs/BENCHMARKS.md#ignition-refinement-rules-under-perturbation) shows the transcribed shift and alignment
  rules reach the same final positions as Quark's on 20 directed and 800 random
  perturbations of the oracle, so a CLE parity failure on this graph cannot hide in
  refinement. [CLE parity](docs/BENCHMARKS.md#ignition-cle-parity-and-the-default-xint8-preset) now covers the default preset too:
  byte-identical equalized weights, an exact position/integer match with a fresh oracle
  that is itself identical to the repo's original resnet50 XINT8 artifact, and 72.10%
  top-1 at 5.22 ms paired on the NPU for both producers.
  [AdaRound parity](docs/BENCHMARKS.md#ignition-adaround-parity) transcribes Quark's FastFinetune
  AdaRound: byte-identical to a fresh `XINT8_ADAROUND` oracle on the same machine,
  79.40% CPU top-1 for both, while a Sep 5 Quark artifact from an unrecorded machine
  differs by one LSB in 4.1 M weights, so cross-machine parity stays statistical.
  [YOLO preparation parity](docs/BENCHMARKS.md#ignition-yolov8n-cut-preparation-parity)
  extends the gate to the head-cut YOLOv8n export: the prepared float graph equals
  Quark's pre-calibration graph, a fresh same-listing oracle matches position for
  position and integer for integer, and both files read 27.03 mAP@50-95 paired on the
  NPU with byte-identical detections.
  [YOLO AdaRound parity](docs/BENCHMARKS.md#ignition-yolov8n-cut-adaround-parity) closes
  AdaRound on that export too, once the layers are walked in the vendor's topological
  order of the float model rather than the emitted file's: every integer byte-identical
  to a fresh same-listing `XINT8_ADAROUND` oracle, all 819 per-layer log lines equal,
  32.04 mAP@50-95 paired on the NPU for both, and the first same-listing AdaRound
  toggle on this graph: 5.01 points over the plain-XINT8 c64 pair.
  [MODNet](docs/BENCHMARKS.md#ignition-modnet-independent-calibration-and-paired-matte-evaluation)
  is the third family and the first that is not a plain convolutional stack: 35 Clip
  activations, 17 depthwise convolutions, a 16x16 global pool and fractional Resize. Its
  prepared graph diffs empty against Quark's, an independent calibration on the same
  64-image listing matches a fresh oracle with 140/140 int8 byte-identical, and both files
  read 0.17122 MAD on CPU and 0.19021 on the NPU at 502/507 nodes placed. It also answers
  the preprocessing half structurally: `quant/sources.py` calibrates through
  `npu.modnet.preprocess`, the same function inference calls. Two rules the graph exposed:
  the vendor shares a pooling or resize output's quantization parameters with its input
  only when that input is already marked at its own visit order, and the vendor's
  SimplifyModel step is onnxslim, which Ignition calls rather than transcribes.
  Safe departures from power-of-two scales, product-scale INT32 bias execution and
  per-channel compiler memory growth remain open. See
  [`quant/DESIGN.md`](quant/DESIGN.md) for the ordered gates and remaining source questions.
- **Native Windows driver overhead floor and DPU microcode stream — closed.** Low-level
  driver characterization via `pyxrt.pyd` and `amdxe.sys` (`results/aie/windows_xrt_driver_bench.log`,
  `results/aie/windows_context_switch_bench.log`) determined that the physical userspace
  dispatch preparation floor is **8.76 µs** (1.85 µs run allocation + 6.91 µs across 8 arguments),
  hardware runlist batching overhead is **3.39 µs/run**, and sub-microsecond buffer synchronization
  (0.85 µs at 4 KB) establishes that host-device sync on unified APU memory is purely CPU cache
  flush/invalidation. Context scaling discovered a hard driver ceiling of **5 virtual hardware contexts**
  (matching the 5 physical silicon columns, with context 6 rejecting at NTSTATUS `0xc01e0009`), and
  interleaved execution quantified a **747.75 µs context-switch penalty** (7.22x slowdown) when switching
  contexts on a shared partition. Reverse engineering of compiled `.xmodel` microcode
  (`results/aie/dpu_transaction_disasm.log`) revealed 48-byte transaction packets dominated by
  Opcode 3 (Conv2D / 1x1 dense, 43–49%) and Opcode 6 (Depthwise Conv, 33–37%).
  [Working](docs/BENCHMARKS.md#native-windows-xrt-driver-latency-and-dpu-microcode-disassembly).
- **AIE-ML systolic shift-cut feasibility theorem for Project Ignition — closed.** Mathematical
  formulation of the post-accumulator scaling unit proved that operations are physically feasible
  on XDNA1 without numerical distortion if and only if the arithmetic right-shift register
  sigma in [0, 31] (`results/quant/shift_cut_feasibility.log`). In power-of-two quantization, this yields
  the exact closed-form Systolic Scale Feasibility Window: `pos_y in [pos_x + pos_w - 17, pos_x + pos_w + 14]`.
  If sigma < 0 (pos_y > upper bound), 32-bit accumulator overflow destroys accuracy (as observed in
  RegNetX-002, sigma = -90, collapsing top-1 accuracy to 0.50%). If sigma > 31 (pos_y < lower bound),
  the 5-bit physical shifter clamps (as in FastDepth, sigma = 32, clamping to 31). Automated scale
  repair projection (`quant/shift_cut.py::project_scale_to_feasible_basin`) successfully projects FastDepth
  `Conv_96` pos_y 0 -> 1, eliminating the 1-bit overflow without retraining.
  [Working](docs/BENCHMARKS.md#aie-ml-systolic-shift-cut-feasibility-theorem-for-project-ignition).
- **The webcam path (single `4x4.xclbin` session, `./scripts/yolo-demo.sh`) has not
  been exercised end to end.** The related but distinct round-robin-across-4-columns
  demo *has* — see
  [A live demo](docs/BENCHMARKS.md#a-live-demo-does-the-multi-partition-finding-hold-on-a-real-webcam):
  camera-bound at 30 fps through n/m/l, genuinely NPU-bound (22.0–23.5 fps) at x.
- **Column count for the int8 conv kernels.** Both NPU measurements use 1–3 columns of a
  4×5 array; 4 × 146 ≈ 584 GOPS would still lose, but not by 5.6×. The one lever the
  56×56 result doesn't touch.
- **What the array physically is, and what that permits.** The silicon-level inventory,
  the ceilings derived from it, and the objectives list live in
  [`docs/SILICON.md`](docs/SILICON.md). Its objective S0 is done: the core clock is
  **1.80 GHz** in `default` (0.80 `powersaver`, 1.03 `balanced`), measured through the
  trace unit because Peano cannot read the cycle counter (`results/aie/clock_probe_npu.log`).
  Open from it: S0's concurrent-VitisAI-EP leg; S1's bandwidth constants, now with a
  clock behind them (the 7.0 GB/s shim channel is one 32-bit word per cycle); and S2's
  full trace, whose upstream parser mis-times gaps over 2^18 cycles; and S4, the
  package-power delta that would let a work-per-watt verdict exist at all — nothing here
  has ever measured a watt, and no per-NPU rail is exposed to read one from.
- **The cycles themselves, read off the machine code.** Peano's `llvm-objdump` disassembles
  AIE2, so a kernel's inner-loop cost is readable without hardware: the core is a statically
  scheduled VLIW that covers operand latency with explicit nop bundles, and a hardware loop's
  bundle count is its cycle count. S0's two loops measured 9.000 and 2.000 cycles per
  iteration and disassemble to 9 and 2 bundles (`results/aie/aie2_isa_static.log`,
  `tools/aie_disasm.py`). That closed two questions the docs had carried as unverified —
  six issue slots per bundle, stated nowhere before, and the accumulator file, where five
  live 4×8×8 int8 accumulators fit and six spill, correcting both documents that had guessed
  from one kernel. It also reframes the int8 GEMM: its inner loop issues 88.9% of the
  machine's MAC rate while the whole kernel reaches 31.3% of peak, so the loss is outside the
  loop and a kernel rewrite is the wrong lever. Open from it: what the elapsed time is spent
  on instead, which is the trace unit's stall events rather than the disassembly.
- **Why a core is not computing, measured.** The trace unit carries a stall taxonomy, and
  `kernels/pmu_probe/` routes it, calibrated by reproducing the two loops above at 2.0003 and
  9.0001 cycles per iteration before anything else is believed. `cycles alive = issuing +
  memory + stream + lock + cascade stalls` closes to a constant 190-cycle prologue across a
  16× range of work, so the decomposition is sound (`results/aie/pmu_probe_npu.log`). The
  first answer it gives is uncomfortable and useful: on a one-core design the core waits
  8,500–12,700 cycles per dispatch on its input queue's lock, flat in the work done, and 68%
  of the shortest run's cycles are that wait rather than compute. The kernels in this repo
  have been losing on data arrival, not on arithmetic, and that is now a measurement instead
  of an inference. Streaming buffers separates the fixed and per-buffer terms and gives the
  first cost model this repo has for the inside of a kernel: `cycles = n_buffers × (compute +
  ~205) + ~10,000`, where the per-buffer overhead is a constant 717 cycles at every point over
  a 4,096× range of work and the lock wait is flat. A buffer carrying less than a few hundred
  cycles is mostly handoff; a dispatch carrying less than ~10,000 cycles is mostly waiting.
  Open: three of the four stall categories have never been non-zero here, so they are
  unexercised; and the same instrument has not yet been pointed at the conv or the GEMM, whose
  upstream designs carry no trace hook, which is where it would change a verdict.
- **The int8 GEMM's loss is accumulator spill, and that was settled without running it.** The
  two results above compose: if a loop's bundle count is its cycle count and a buffer costs a
  constant on top of its work, a kernel's issuing time is computable from its object file, and
  the difference from a measured time is the cycles the core spent not issuing.
  `tools/gemm_cost_model.py` does that for the best int8 GEMM. The kernel is a **loop nest** —
  two software loops around the hardware loop, with eight accumulators loaded before it and
  stored after it — and the whole body re-runs once per group of accumulators, 16 times per
  call. So the celebrated 88.9% inner-loop MAC density covers only 54 of the 141 cycles a group
  costs. Over a whole call the kernel issues **107.3 MACs per cycle, 41.9% of nameplate**, while
  the core is issuing for **74.6%** of the dispatch. **The larger loss is the schedule, and it
  is spill:** the kernel holds eight live accumulators where five is this branch's measured
  spill-free ceiling, and the 87 non-loop bundles per group are the resulting load, store and
  stack traffic. What survives from the measurement side is a per-buffer floor — halving the
  work per buffer leaves measured cycles per call at **3,274 against 3,160** — which n=32 fills
  to 41% and n=64 to 75%, explaining the sweep's 1.93× tile result and leaving ~1.3× of
  headroom. The cheapest next test is static: rebuild the kernel blocked to four or five
  accumulators and re-run the tool; if the per-group bundle count does not fall, those 87
  bundles are the output tile's mandatory traffic rather than spill. Open: "not issuing" is a
  residual here, not an observation, and attributing it needs the stream-port events that
  `kernels/pmu_probe/`'s core-stream reader cannot yet decode. **An earlier reading of this,
  superseded the same day, modelled the kernel as one loop and reported the core as starved for
  60% of the dispatch with 2.4× of headroom; both were artefacts of missing the nest.**
- **A same-bank paired load costs a cycle, and the int8 GEMM has that collision where the bf16
  GEMM does not.** A core tile's 64 KB of local memory is four banks of 16 KB, and the core has
  **two load units**, so a bundle can issue two loads at once; when both hit one bank the pair
  costs an extra cycle. That is measured on the `research/windows-lowlevel` branch by holding
  the compiled function bytes identical and moving only the operand addresses — 12.0 cycles per
  iteration in one bank against 11.0 across two, r² 1.0, and 1,024 cycles per 64×64×64 panel in
  a real GEMM. Surveying this branch's own builds finds the production int8 GEMM with **both
  input tiles in bank 2 and bank 3 empty**, and the bf16 GEMM — this repo's one genuine NPU
  win — with its inputs correctly split. The reason is inverted from the obvious one: int8's
  tiles are half the size, so the allocator packs the pair into a single bank. The int8 kernel
  is penalised *because* its data is smaller, and its 9-bundle loop is far more exposed than
  bf16's 32-bundle one. Charging it narrows the GEMM's unexplained residual from 25.4% to
  18–23%. **This also corrects our own issue-width row:** slot `b` is the second load unit, not
  the branch slot, which is the whole reason the conflict exists. Open, and cheap, and the
  reason this matters beyond one kernel: bank 3 is empty in both builds, so moving an input
  there changes the core's issuing time by a known amount and changes *nothing* about data
  movement — the controlled lever needed to test whether the per-buffer floor or the schedule
  is the critical path. **That test was run on 2026-09-09 and the machine could not resolve
  it.** The intervention is clean — raising the per-core stack shifts every buffer up, moving A
  wholly into the empty bank 3, with a byte-identical compiled kernel — but across seven
  alternating series the arms overlap and the sign of the difference changes between them,
  while the colliding arm's own floor drifts 5.0% between repeats of the identical build. A
  large speedup is excluded; the ~3% at stake is not separable from zero. The lesson is about
  the observable, not the hypothesis: wall time on a shared machine cannot see a 3% core-side
  change, and the trace unit can, by reading `ACTIVE` against `LOCK_STALL` in core cycles
  inside the dispatch. That needs a trace hook in `whole_array`, which is the same obstacle
  already noted above. Also tested and refuted here: the two-process handoff floor is **not**
  the NPU context switch, despite 789.8 µs sitting near a measured 747.75 µs penalty; the floor
  fits 78.4 ns per element with a 147.2 µs intercept at r² 0.9997 and stays conversion-bound.
- **The shift-cut hazard predictor is not ready to gate the quantizer, and a forward test is
  what showed it.** `quant/shift_cut.py` formulates a real constraint — the DPU requantizer is
  a 15-bit multiplier and a shift confined to σ ∈ [0, 31], so some scale triples genuinely
  cannot be represented — and its flags coincide with two documented failures, RegNetX-002's
  collapse and BiSeNetV2's fall from 59.47% to 15.33% pixel accuracy on hardware. But every
  one of those was retrodiction: each model already had a known outcome. Predictions for 14
  untested artifacts were therefore committed **before** any of them ran, and then four were
  run. **Both models the audit called infeasible on 9 of 9 operations place 50 of 52 nodes on
  the NPU and track their own CPU reference at correlation 0.999**, which is ordinary
  requantization divergence. A third such artifact was already inside the audit's own
  evidence, measuring 34.06 dB PSNR on the NPU against a 35.64 dB float reference. The rule
  fires on an entire working architecture family. This does not show the bound is wrong as
  physics, only that the classifier built on it has a false-positive mode of the widest kind,
  so it belongs as an advisory report rather than a gate. Open: the clean direction is
  unsettled, because raw-tensor correlation proved too sensitive for a detection head whose
  decoded detections are byte-identical between CPU and NPU — a task-level metric is needed
  to test it; and the audit's positive claims on BiSeNetV2 and RegNetX-002 remain untested
  forward.
- **The dispatch floor fell 17×, and it was never a kernel problem.** The single most
  consequential number here is the per-dispatch floor: `docs/SILICON.md` §3.4 says every
  small-op verdict in this repo is conditional on it, and at 617 µs it closed most of them.
  The fix was named in the same log that measured it and then sat unrun for two days, recorded
  as *"Nothing has been run — this is an API-existence check and the next measurement to make,
  not a result."* Run now: batched `pyxrt.runlist` submission amortises a dispatch to
  **36.3 µs**, reproduced at 35.9/36.3/36.0/36.3 across four runs. That is 17× below the IRON
  floor and a **quarter** of the 169.8 µs that was attributed to hardware — so the hardware
  half was not silicon either. Four of the six ops §3.4 lists as closed now clear the floor:
  MobileNetV2 by 48×, MobileViT stage-2 attention and bf16 attention stage 2 by 6.7×,
  GroupNorm at L ≤ 18816 by 6.5×. **They are not thereby wins** — the floor has stopped being
  the reason they lose, which makes kernel quality the deciding question for the first time,
  and the attention README's "the op has to be ~20× larger" becomes ~1.4× for stage 2. Two
  caveats bound it: this is a **throughput** result (36 µs with 64 dispatches in flight; a
  one-shot call still pays ~140 µs raw), and it is **raw pyxrt** — `@iron.jit` uses no
  runlists, so a real design pays the old floor until that host path is written. That, not the
  measurement, is what the objective now blocks on.
- **That host path was then written, and it says the 17× is real but almost none of it reaches
  the caller.** `kernels/dispatch_floor/iron_batch.py` patches IRON's own transaction submit to
  queue unstarted runs into a `pyxrt.runlist`, with nothing outside the repo modified, so an
  ordinary `@iron.jit` design can batch. From inside IRON the device cost per dispatch does
  fall to **37.5–37.9 µs**, reproducing the raw-pyxrt 36.3. But wall time per call improves
  only **1.26–1.37×**, because IRON's per-call host work — ABI validation, buffer preparation,
  instruction-buffer setup — is a near-constant **~500 µs, flat in batch size**, that batching
  cannot touch. So there are **three thresholds, not two**: ~617 µs unbatched through IRON,
  ~531 µs batched through IRON, ~36 µs batched through raw pyxrt. Against ~531 µs *none* of
  the four reopened verdicts survives. This also closes the other caveat — a real 8-core bf16
  kernel (GroupNorm, L=150528) batches to 823.8–838.3 µs per dispatch, which is its own
  compute, matching the 835.8 µs measured independently, so a real kernel's configuration cost
  does not swamp the passthrough's floor. The residue is the same term the floor log called
  447.3 µs, now confirmed independent of how the submit is done, and at 37.5 µs of device
  against ~500 µs of host **the host path is the larger problem by more than an order of
  magnitude.** The open question has moved off the NPU entirely.
- **The conv op class was closed on a kernel using 5% of its issue slots.** This repo's
  most-lost verdict is int8 conv: the vendor DPU does 1650 GOPS per column, the open kernel
  146.1, an 11.3× gap on the same silicon. Objective K1 said "the first trace will say whether
  it is data movement or issue rate", and no trace was ever run because the build cache had
  not survived — *"this repo's most-lost op class was not surveyed"*. Rebuilt and read
  statically: **it is issue rate, and not marginally.** The int8 GEMM issues 0.889 MACs per
  cycle; the best conv loop manages 0.333, the 3×3's main loop 0.222, and the 1×1's hot loop
  **0.045**, with two of its three loops issuing no MAC at all. The 1×1 keeps its accumulator
  in *memory*, loading four quarters and storing four back around a single `vmac` and idling
  six of 22 bundles, while naming 3 of the 9 accumulator registers available. The 3×3 keeps
  its accumulators live and still burns six of eighteen bundles on `vshift` window alignment —
  the exact cost K1 proposes moving to the mem tile's descriptors. So the op class was closed
  against the CPU by a kernel leaving 95% of its MAC slots empty, which is a statement about
  the kernels and not about the silicon. Open, and stated carefully: the 4×–20× are **ceilings
  on unused issue slots, not predictions**, the densities are unweighted by trip count, and
  reaching K1's 1 TOPS bar still needs 6.8×.
- **Candidate model pipelines (Categories A, B, C, D, E).** Test plans, target shapes, and falsification criteria:
  - **Category A:** Image Super-Resolution — SESR-M7 (placement, 1.48 ms latency, 3.02x iGPU win,
    70% AdaRound recovery) and Real-ESRGAN Compact (activation memory spill) closed above.
  - **Category B:** Real-Time Portrait Matting and Semantic Segmentation — MODNet (zero-concat trade-off, calibration fix) and BiSeNetV2 (13.12 ms, 1.09x iGPU win, monolithic DPU subgraph, fixed-point bilateral gating distortion) closed above.
  - **Category C:** Advanced Detection and RepVGG Backbones — YOLOv6n, YOLOv11n, and
    YOLO-World v2 closed above.
  - **Category D:** Monocular Depth Estimation — MiDaS v2.1 Small (bilinear vs nearest fusion,
    10.81 ms, 1.53x CPU win) and FastDepth (depthwise separable decoder, 2.87 ms, 1.05x iGPU win,
    r = 0.9383) closed above.
  - **Category E:** Untested Classification Topologies — DenseNet-121, ResNeXt-50, and RegNetX-002 (placement, Concat DMA, grouped convs, shift-cut scale explosion) closed above.
- **Longer term:** a detector fine-tuned for fixed camera feeds (licence-plate
  recognition), reusing the head-cut + XINT8 + AdaRound recipe rather than re-deriving it.

## How to read the rest of this repository

If you want *what works and how fast*: `README.md`.
If you want *every decision, rejection, and environment trap that produced it*:
`docs/DECISIONS.md`.
If you want *why any of this is being done*: you're reading it.
