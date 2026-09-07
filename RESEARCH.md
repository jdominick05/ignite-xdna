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

The numbers live in `README.md`. What's worth stating as findings, because they
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
| Resolution sensitivity (classification) | Does ResNet50 have the same fixed-cost floor YOLO does, and does accuracy survive moving off the training resolution? | **Answered, 128–384px measured.** Smaller fixed cost (2.6 ms vs YOLO's 2.4, but a smaller share — 23% vs 27% — of a slower run), more compute-bound, and accuracy peaks above training resolution at 256px rather than at it. The 288px turn is a real ceiling, not noise: everything above 256px is strictly worse on both latency and accuracy at once. Table in `README.md`. |
| Width sensitivity (classification) | Does "width is nearly free" (found on yolov8n→s) generalize past YOLO, and does it hold at a bigger step? | **Answered: yes, at two steps.** `wide_resnet50_2` (isolated width, 2.7× params, 1.73× latency) and `wide_resnet101_2` (deeper+wider, 4.96× params, 3.15× latency, 80.2% top-1 beating resnet50's AdaRound headline with plain XINT8) both hold. Table in `README.md`. |
| Width sensitivity (detection) | Does the yolov8n→s ratio hold at a third, bigger width step? | **Answered: yes, latency and accuracy both.** yolov8m: 9.1× the FLOPs of yolov8n for 3.46× the latency (30.80 ms, 1216/1223 nodes), 43.49 mAP@50-95 past yolov8s's 37.40 (calib 64, not the 200 n/s used — noted as a caveat, not a confound). Table in `README.md`. |
| Batch efficiency | Is a static batch >1 an efficient way to use this NPU? | **Answered: no, and it's unsafe in a specific, isolated way.** Batch 2 on resnet50 drops the EP's partition to 80/395 nodes, costs 12.6× the per-image latency, and only writes the first batch slot — slot 0 scores 72.00% (correct), slot 1 scores ~0% with input-independent output. Do not batch on this backend. |
| Width×resolution interaction | Does a wide model at low resolution beat a narrow model at high resolution on both axes? | **Answered: no, falsified.** resnet50@256² (6.32 ms, 74.00% top-1) beats every `wide_resnet50_2` configuration tested (160²/224²/288²) on both axes at once. Traces back to `wide_resnet50_2` never having a real top-1 edge at matched resolution to begin with. Table in `README.md`. |
| Fixed-cost intercept: hardware or graph property? | Does the ~2.6 ms fixed cost stay fixed at 2.7× the width? | **Inconclusive.** Marginal cost triples with width (65.10 → 180.94 ms/Mpixel, solid). Intercept comes out lower, not the same or higher (1.88 ms vs 2.63 ms) — but on only 3 points, not a trustworthy comparison either way. Table in `README.md`. |
| AdaRound at width | Does AdaRound recover less on a wider model, as the YOLO n→s pattern predicts? | **Answered: no, the prediction was wrong.** resnet50 recovers 90.0% of its quantization loss, `wide_resnet50_2` recovers 90.5% — essentially identical, and the wider model's initial loss was actually larger, not smaller. Also settled: AdaRound's RAM wall is resolution-specific (640×640), not width-specific — both fit fine at 224². `wide_resnet50_2`+AdaRound (80.10% top-1, 9.66 ms) is now this repo's best classification speed/accuracy point. Table in `README.md`. |
| AdaRound for YOLOv8s at 640² | Is this actually RAM-blocked, and does it recover as much as classification's ~90%? | **Answered: not blocked, and recovers far less.** `yolov8s_cut_xint8_adaround.onnx` compiles and runs fine (15.5 ms, 922/929 nodes) — the earlier RAM-wall note didn't hold. Full 5000-image mAP@50-95 is 39.98 vs plain XINT8's 37.40 — 2.6 points, nowhere near classification's ~90% recovery. A 500-image slice first suggested 45.19, another instance of the slice-vs-full trap this repo already flags elsewhere. Table in `README.md`. |
| AdaRound for YOLOv8m at 640² | Does AdaRound's recovery keep shrinking as width increases past s, matching the n→s quantization-penalty trend? | **Answered: yes, recovers even less in absolute terms.** Quantized on Desktop 1 (GPU-accelerated FastFinetune) and run on Desktop 2's XDNA1: 45.32 mAP@50-95 vs plain XINT8's 43.49 — +1.83 points, smaller than s's +2.58 despite m's much higher baseline accuracy, and no latency cost (30.46 ms, same as plain XINT8's 30.80). Caveat: the quantized model synced in with no local log of its calibration count, so it isn't a clean like-for-like comparison against the calib-64 plain-XINT8 row. `results/map_yolov8m_cut_xint8_adaround_npu.log`. |
| Concurrent camera streams | Is a static batch>1 the only way to ask this NPU for more than one image at once, and does it fail the same way? | **Answered: no — two independent sessions on two threads is a different request than batching, and it works.** 1.8-1.9× the combined throughput of round-robin, zero cross-talk between streams (checked directly, the same way the batch-2 bug was found, not assumed away). Consistent with yolov8n only reaching about 6.6% of the array's 16 TOPS solo (`tools/estimate_tops.py`, finding 5 above — retracts this row's earlier "~1.1" figure) — there's headroom for a second stream. `tools/dual_stream_bench.py`, `results/dual_stream_{pose,detect}.log`. |
| Concurrent streams beyond 2, and at width | Does the multiplier keep climbing past 2 streams, and does a wider model with less idle headroom get the same multiplier? | **Answered: no on both counts, and the two answers explain each other.** yolov8n's combined throughput saturates at 3 streams (~167 fps, 2.1×) — headroom runs out, concurrency doesn't stop working. yolov8m, which already uses more of the array per call, saturates a stream earlier at a much smaller 1.29×. Zero cross-talk at up to 8 concurrent streams on either model; no throughput regression past the ceiling. `tools/nstream_bench.py`, `results/nstream_{yolov8n,yolov8m}.log`. Table in `README.md`. |
| Is stream saturation compute-bound or memory-bound? | Concurrent sessions each hold their own runtime buffers — is the throughput ceiling actually a memory ceiling in disguise? | **Answered: memory, not the cause.** `xrt-smi examine -r aie-partitions` (the one tool found this session that can actually see NPU memory — Windows' `GPU Engine`/`GPU Adapter Memory` counters can't, the device is a `ComputeAccelerator`, not a WDDM GPU adapter) shows memory scaling linearly with stream count on both yolov8n (~29 MB/stream) and yolov8m (~100 MB/stream), climbing cleanly through 8 streams with no ceiling — well past the 2-3 stream point where throughput already flattened. Memory and throughput are decoupled, ruling memory out and leaving compute headroom as the standing explanation. `tools/session_hold.py`, `results/nstream_memory_yolov8{n,m}.log`. |
| Does classification show the same concurrency shape; does accuracy survive contention past a binary found/not-found check? | Every prior concurrency check only confirmed a binary ground truth (found a person, or didn't) — does real accuracy hold under N-way contention, and does classification saturate the same way detection does? | **Answered: yes on both, cleanly.** resnet50 saturates at 1.60× by 8 concurrent streams (between yolov8n's 2.13× and yolov8m's 1.29×, its own idle-headroom budget) and holds flat through 16 with zero regression. Every concurrent stream ran the identical labeled slice the solo baseline used, so predictions could be diffed exactly rather than compared as an aggregate top-1 a different sample could move on its own — result: bit-identical argmax on all 960 concurrent classifications tested (16 streams × 60 images), at every stream count. `tools/nstream_cls_bench.py`, `results/nstream_resnet50.log`. Table in `README.md`. |
| Width beyond yolov8m: does the n→s→m trend continue at l/x? | Two width steps (n→s, s→m) both bought a clear mAP gain for extra latency — does l→x keep paying off? | **Answered: no, the trend breaks.** l: 49.67 ms, 45.37 mAP@50-95, 1510/1517 nodes. x: 117.11 ms, 45.09 mAP@50-95, same 1510/1517 nodes (l/x share architecture depth, only channel width differs). x is 2.36× the latency of l for a net *loss* in mAP — width alone stops paying off somewhere around l under this recipe (plain XINT8, calib 24-32). Also found: yolov8l's full 5000-image eval failed with a hardware DPU timeout on 2 of 3 attempts, memory confirmed flat (231 MB) during the runs that succeeded — ruling out a simple leak, root cause still unresolved, not seen on any other size. `results/map_yolov8{l,x}_cut_xint8_npu.log`, `results/yolo_cut_{l,x}_{cpu,npu,diag}.log`. Table in `README.md`. |
| Can `xrt-smi`'s GOPS column build a utilization-vs-16-TOPS story? | GOPS is the one other live NPU-side reading `xrt-smi` exposes besides memory — does it track real compute headroom running out the way the throughput ceiling does? | **Answered: no, it's a dead end.** `tools/session_hold.py` extended to parse GOPS and independently count actual completions/s in the same run. GOPS is exactly `9 × streams` (yolov8n) / `80 × streams` (yolov8m) with zero saturation through 8 streams, while measured completions/s is flat from 1 stream onward in the same run — decoupled from real throughput. Simplest explanation: `xrt-smi` credits each context a notional per-context GOPS figure blind to shared-array contention, not a measurement of delivered compute. `results/gops_yolov8{n,m}.log`. Table in `README.md`. |

## Open questions

These are the threads this project would pull next, roughly in the order they'd
resolve, carried forward from earlier engineering notes and updated for what has since
been closed:

- **Resolution vs. accuracy for classification — done, 128–384px.** 256px is the
  measured peak of the speed/accuracy frontier; everything above it (288/320/384px) is
  strictly dominated — worse latency and worse accuracy at once, confirmed by the
  320/384px outlier check rather than assumed from the 288px point alone. AdaRound
  across the same sweep is still open — the table so far is plain XINT8, and AdaRound's
  recovery could plausibly move where the peak sits.
- **AdaRound for `wide_resnet50_2` — done.** Recovers 90.5% of its quantization loss,
  essentially matching resnet50's 90.0% (see the findings section above); the predicted
  "recovers less at width" effect didn't materialize, and the result is now this repo's
  best classification speed/accuracy point (80.10% top-1, 9.66 ms). Still open:
  AdaRound for `wide_resnet101_2` (would it also land near ~90%, or does the confound
  with depth/recipe change the picture?). AdaRound for YOLOv8s and m are both done now
  (see the status table) — neither was actually RAM-blocked once run on a machine with
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
  them (200 for n/s, 64 for m, 32/24 for l/x), a caveat already flagged in `README.md`.
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

## How to read the rest of this repository

If you want *what works and how fast*: `README.md`.
If you want *every decision, rejection, and environment trap that produced it*:
`docs/DECISIONS.md`.
If you want *why any of this is being done*: you're reading it.
