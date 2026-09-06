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
3. **This NPU is not compute-bound at the model sizes tested.** yolov8n uses roughly
   1.1 of its 16 TOPS. Consequences follow directly from this, and both have been
   measured rather than assumed:
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
4. **The full-graph YOLO model is kept in the repo on purpose, not cleaned up.** It's
   the control that still fails — proof that the head-cut result is the fix, not an
   artifact of measuring differently.
5. **Batching is not just inefficient here, it's dangerous — and the exact mechanism
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
6. **Width and resolution don't trade off cleanly — the obvious combined-lever
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
7. **Whether the fixed per-inference cost is a hardware or graph property is still
   open — the three-point check was inconclusive.** Fit the same
   `ms = a + b × Mpixels` model to `wide_resnet50_2` (2.7× resnet50's width): the
   marginal cost `b` nearly tripled (65.10 → 180.94 ms/Mpixel), a solid result
   consistent with a wider graph doing more arithmetic per pixel. The intercept `a`
   came out lower, not higher or the same (1.88 ms vs 2.63 ms) — but on only three
   points with residuals of 0.7-1.3 ms against a ~2 ms intercept, that comparison
   isn't trustworthy either way. More resolutions per model would be needed to
   settle whether the fixed cost is hardware-invariant.
8. **"Quantization penalty shrinks with width" does not generalize — it was a fact
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
  with depth/recipe change the picture?), and for YOLOv8s and larger — still blocked by
  the development machine's RAM at 640×640 specifically (now confirmed
  resolution-specific, not width-specific, since the ResNet AdaRound runs above hit no
  such wall at 224²).
- **Direct NPU utilization measurement.** Windows' `GPU Engine` performance counter
  (the one Task Manager's NPU graph reads) exists for this device but couldn't resolve
  the yolov8m benchmark at all — `Get-Counter`'s own ~0.5-1s per-call cost is coarser
  than the ~31ms inference bursts it was polling for. Effective NPU throughput has so
  far only been inferred indirectly (measured latency against the model's FLOPs puts
  yolov8n at roughly 1.1 of the 16 TOPS on paper) — a lower-level counter or
  AMD's own profiling tool (untried) might resolve this properly.
- **Width beyond what's measured.** Two width steps confirm the trend for
  classification (resnet50→wide_resnet50_2→wide_resnet101_2) and now two for detection
  too (yolov8n→s→m, 9.1× FLOPs for 3.46× latency, 43.49 mAP@50-95 at m). l/x remain
  untested — disk is the binding constraint there (`calib_mb_per_image` extrapolates
  ~1-1.5 GB/calibration-image at 640×640 for l/x, more than the development machine's
  disk allows at any useful sample count).
- **Full 5000-image mAP for every model size**, not just n and s (m/l/x untested at
  scale).
- **The eventual application.** Nothing here commits to it yet, but the shape of a
  next step is visible: a labeled licence-plate dataset from fixed camera feeds,
  fine-tuning a YOLO variant on it, and reusing this same head-cut + XINT8 + AdaRound
  recipe rather than re-deriving it.

## How to read the rest of this repository

If you want *what works and how fast*: `README.md`.
If you want *every decision, rejection, and environment trap that produced it*:
`docs/DECISIONS.md`.
If you want *why any of this is being done*: you're reading it.
