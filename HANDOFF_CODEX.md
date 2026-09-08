# Handoff: quant/ — an owned XINT8 quantizer

Not part of the doc contract (`README.md`/`RESEARCH.md`/`docs/`) — this file is a scratch
briefing, untracked by git, safe to delete once picked through. You have no memory of any
prior session on this repo; everything you need is either in this file or in the two
documents it points at. Read them in this order before writing anything:

1. **`CLAUDE.md`** (repo root, git-ignored — if it isn't on your machine, ask the user for
   a copy before doing anything; it is the load-bearing invariants file and this repo has
   no CI to catch a violation of it).
2. **`quant/DESIGN.md`** — the actual spec. Read all of §1–3 in full; skim §4–8 for shape,
   then re-read the section for whichever phase you're about to touch. It is long on
   purpose: every contract fact in it carries a provenance tag (`[M]`/`[Q]`/`[P]`/`[V]`/
   `[L]`/`[U]`) so you can tell a measured fact from a guess. **Never treat a `[U]` tag as
   settled** — it means "unknown, open a Phase 0/1 question," not "probably fine."

## What exists right now

`quant/DESIGN.md` only. **Zero code under `quant/`, zero logs under `results/quant/`,
nothing measured against this design.** Anyone who tells you otherwise is wrong; check
`git log -- quant/` and `ls results/quant/` yourself rather than trusting a summary,
including this one.

## What this project is, in one paragraph

The repo quantizes ONNX models to the exact QDQ dialect AMD's Quark 0.11rc1 produces, so
the VitisAI EP's X1 backend (the actual DPU compiler) accepts them. `quant/` exists to
replace Quark as the *producer* of that dialect without changing the *consumer* — same EP,
same xclbin, same DPU — while gaining per-tensor transparency Quark doesn't expose, and a
path to deliberately depart from Quark's choices once the acceptance boundary is mapped
(that's Phase 4, not Phase 0 — see the roadmap in `DESIGN.md` §6).

## Hard constraints (repeated here because violating them silently wastes the most time)

- **Nothing under `quant/` may import Quark.** `resnet_env`'s Quark install triggers a
  custom-op build on import; a stray import in a module that runs on every session build
  would slow every future run of this repo, not just yours.
- **Nothing under `npu/` may import `quant/`**, and vice versa is fine only for the
  preprocessing sources and `build_session` (`quant/` may import `npu/`).
- **No new compile-cache key.** Emitted models reuse the existing cache key of whichever
  pipeline's graph they share. The keys are names, not content hashes — pass `--fresh`
  every time you emit a new model, or you will silently run a stale, wrong-architecture
  compile and not find out until the numbers look off.
- **Two conda envs stay separate.** `quant/`'s core (everything except `adaround.py`) must
  import cleanly in both `resnet_env` (has torch/onnx 1.19) and `resnet_env17` (the only
  env that can build an NPU session) — that's what lets `verify.py` sit next to a live NPU
  session. Don't add a torch import outside `adaround.py`, and don't add anything to
  `resnet_env17` beyond what's already there.
- **Every `quant/*` module you land joins the `PIPELINE CHECKS` import list** in both
  `CLAUDE.md` and `CONTRIBUTING.md`, in the same commit. The gate command is in `CLAUDE.md`
  under "Commands — the completion gate"; it must print `PIPELINE CHECKS PASS` after your
  change, every time.
- **Never report a number you didn't measure on this hardware, and never restate one from
  memory.** If you're about to type a latency, an mAP, a node count, or an LSB delta,
  either just read it out of a log/model you have open, or go generate it. This repo has
  been burned by exactly this mistake more than once (see `CLAUDE.md`'s Invariants).
- **A 500-image slice is not the answer for any accuracy figure that goes in a doc.** Full
  dataset only; label a slice as a slice if you use one for a quick sanity check.

## Where to actually start

Phase 0, full stop — don't jump to Phase 1 code first. `DESIGN.md` §2.6 lists every open
`[U]` item (the NPU-CNN registry's exact quantized-op list, bias initial position, which
direction each `refine.py` rule adjusts, the CLE default knobs, `split_large_kernel_pool`'s
threshold, AdaRound's internal schedule, and a couple more). Phase 0 is entirely reading:
Quark 0.11rc1's source under `resnet_env`'s site-packages (`quark/onnx/...` — file:line
citations for the *already-resolved* facts are all through §2 of `DESIGN.md`, so grep
those files for the neighborhood of each `[U]`), plus re-deriving the §2.1 fingerprint
table by inspection of the three named models in `models/`. No hardware, no NPU session,
no code under `quant/` yet beyond maybe a throwaway inspection script.

Deliverable: `results/quant/notes_xint8_dialect.log`, with every `[U]` in `DESIGN.md`
either resolved (quote the file:line you resolved it from) or explicitly still open. Then
edit `DESIGN.md` in place to update those tags — don't leave the doc saying `[U]` once
you've closed it elsewhere.

Only after that log exists and is folded in (see "Closing out work" below) does Phase 1
(`quant/graph.py`, `pow2.py`, `qdq.py`, `refine.py`, `verify.py`, working only against
`models/resnet50_fp32.onnx`) start. Don't build the calibrator (Phase 2) before Phase 1's
`graph_diff` gate is green — `DESIGN.md` §4.1 explains why re-quantizing an already-int8
model would make that gate vacuous.

## Machine routing

This machine's role is decided by `CLAUDE.md`'s "Machines" table — check which one you're
actually on before picking a task:

- Phase 0 (reading, no hardware): any machine.
- Phase 1–2 (quantize + compare against a fresh Quark run): needs `resnet_env`, so Desktop
  1 or Desktop 2 — and if the same sitting also needs an NPU `diag_*` reading, Desktop 2 is
  the only machine where Quark *and* the NPU are both present.
- NPU gates (Phase 1's `run_*_npu.log`, Phase 2's `map_*_npu.log`, all of Phase 4): laptop
  or Desktop 2, never Desktop 1 (no XDNA1 device at all).
- **Before any NPU number**: `xrt-smi examine -r aie-partitions` must say no hardware
  contexts running, or run `tools/hwinfo_npu_bridge.exe --json --no-hwinfo --interval 1
  --smi-interval 1` as a witness and say so in the log. Three sessions can share this NPU;
  a foreign context makes a low number contention, not a finding.

If you're not on the right machine for the next step, say so and stop rather than
approximating with `--device cpu` or a smaller `--limit` — both have already been tried
here for unrelated OOM issues and are documented as not fixing the underlying limit.

## Closing out work — do this every time, not at the end

Same rule as every other experiment in this repo, from `CLAUDE.md`'s "Maintenance rule"
and "Fold a new results/ log into the docs in the same commit, not after":

1. Write the log to `results/quant/`, named for the model/variant (never a bare name two
   different runs could collide on).
2. Fold it into `docs/BENCHMARKS.md` — a new section, not scattered edits — with full
   method, caveats, and the log path. Never delete a superseded number; mark it superseded
   and keep both.
3. Update `RESEARCH.md`'s open-questions list if the log closes or reopens one.
4. Add the log to `results/README.md`'s index.
5. `python tools/check_links.py` must exit 0 before the commit.
6. Commit via `scripts/commit.sh`, never a raw `git commit` — it checks for UTF-8 logs and
   scrubs the local profile path, and requires the session trailer.
7. **Never `git push` without asking first**, even at the very end. This repo has three
   machines pushing to the same `main` with no branch-per-machine convention; a push is
   shared state, not a local action.

## Out of scope for you right now

Anything in `docs/SILICON.md`'s objectives (K0–K5, A1–A4, S0–S3, D1–D4, C1–C2) and anything
int64-related — both belong to other in-flight work on this repo per the last session's
`HANDOFF_GEMINI.md`. If a `quant/` task turns out to brush against either, stop and flag it
rather than resolving it yourself.

## If you get stuck

`DESIGN.md` §7 ("Risks and open questions") lists the specific ways this design is
expected to be wrong — read it before concluding you've found a new problem; you may have
found the one it already named. If a Phase 1 `graph_diff` shows a real, unresolved
mismatch against Quark (not just a rounding-mode LSB), that's the finding — log it as a
negative result the same as any other measurement here, don't paper over it to make the
gate pass.
