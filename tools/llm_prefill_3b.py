#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM study stage 3b: prefill GEMM at Llama-2-7B's shapes again, with the CPU's threads pinned and
DirectML timed over fresh sessions (the user: "Re-run prefill, pinned").

Designed after stage 3's two INCOMPLETE sittings and the noise study (tools/measure_noise.py).
Everything else is stage 3's, from tools/llm_prefill_bench.py, unchanged: the shapes, M, the
inputs, the arms, the NPU artifacts and the NPU sitting itself (run as `llm_prefill_bench.py npu`),
the smoke, the correctness checks, the keep rule and the 10% repeat threshold. Two changes:
  (a) CPU: ONNX Runtime's intra-op threads pinned, one per physical core at 8 threads and one per
      logical CPU at 16, with the calling thread pinned too. Every session's threads are read
      back; a row whose read-back does not match is void.
  (b) DirectML: each row and pass is K fresh sessions; its value is the median of their medians.

    python tools/llm_prefill_3b.py prereg                 # the pre-registration text; re-verifies the NPU pins
    python tools/llm_prefill_3b.py ort --ep {cpu,dml}     # the CPU or DirectML rows (resnet_env17; the sitting)
    python tools/llm_prefill_3b.py verdict NPU CPU DML    # the mechanical verdict from 3b's three logs
    python tools/llm_prefill_3b.py selftest               # pinning read-back on a tiny model; synthetic verdicts
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import llm_prefill_bench as s3                                     # noqa: E402  stage 3, unchanged
from measure_noise import cores, cpu_session, pin_calling_thread, thread_ids, thread_placement  # noqa: E402

K_SESSIONS = 5                        # DirectML: fresh sessions per row and pass
STAGE3_PREREG = ROOT / "results/llm/llm_prefill_prereg_desktop2_20260923.log"


# ---------------------------------------------------------------- (a) the CPU, pinned

def pinned_cpus(threads: int, cs: list) -> list:
    """The logical CPUs a row's threads are pinned to; the first takes the calling thread."""
    if threads == 8:
        return [c[0] for c in cs]                                 # one per physical core
    if threads == 16:
        return [x for c in cs for x in c]                         # every logical CPU
    raise ValueError(threads)


def witness_ok(pool: list, main: list, cpus: list) -> bool:
    """The read-back matches: len(cpus) - 1 pool threads, each pinned to one CPU, together the row's
    CPUs but the caller's; the caller pinned to its CPU; no CPU sets anywhere."""
    return (sorted(t["cpus"] for t in pool) == sorted([c] for c in cpus[1:])
            and len(main) == 1 and main[0]["cpus"] == [cpus[0]]
            and not any(t.get("cpu_sets") for t in pool + main))


def cpu_row(model: bytes, threads: int, cs: list, feed: dict):
    """One pinned session: warmups, the read-back, then the timed reps (only if it matches)."""
    cpus = pinned_cpus(threads, cs)
    before = thread_ids()
    sess = cpu_session(model, threads, cpus)
    pin_calling_thread(cpus[0])
    try:
        for _ in range(s3.WARMUP):
            sess.run(None, feed)
        pool = thread_placement(thread_ids() - before)
        main = thread_placement({ctypes.windll.kernel32.GetCurrentThreadId()})
        pin = {"cpus": cpus, "pool": sorted(t["cpus"] for t in pool), "main": main[0]["cpus"] if main else None,
               "cpu_sets": sum(1 for t in pool + main if t.get("cpu_sets")), "ok": witness_ok(pool, main, cpus)}
        if not pin["ok"]:
            return None, pin, None, 0, 0
        ts = []
        w0 = int(time.time() * 1000)
        for _ in range(s3.REPS):
            t0 = time.perf_counter()
            y = sess.run(None, feed)[0]
            ts.append(time.perf_counter() - t0)
        w1 = int(time.time() * 1000)
    finally:
        pin_calling_thread(None)
        del sess                                                  # released just after its row
        gc.collect()
    return ts, pin, y, w0, w1


# ---------------------------------------------------------------- (b) DirectML, K fresh sessions

def dml_row(model: bytes, feed: dict, score):
    """K fresh sessions, one after another; each is opened, warmed, timed and closed."""
    meds, reps, errs, shas = [], [], [], []
    for _ in range(K_SESSIONS):
        sess = s3.make_session(model, "dml", 0)
        try:
            for _ in range(s3.WARMUP):
                sess.run(None, feed)
            ts = []
            for _ in range(s3.REPS):
                t0 = time.perf_counter()
                y = sess.run(None, feed)[0]
                ts.append(time.perf_counter() - t0)
        finally:
            del sess
            gc.collect()
        ms = [round(t * 1e3, 4) for t in ts]
        meds.append(round(statistics.median(ms), 4))
        reps.append(ms)
        e, h = score(y)                                           # every session's output is checked
        errs.append(e)
        shas.append(h)
    return meds, reps, errs, shas


# ---------------------------------------------------------------- the rows

def ort3b(ep: str) -> int:
    import onnx
    import onnxruntime as ort
    from concurrent_read_bw import gpu_engines
    print("HEADER " + json.dumps({"stage": "3b", "python": platform.python_version(), "onnxruntime": ort.__version__,
                                  "numpy": np.__version__, "onnx": onnx.__version__, "ep": ep,
                                  "env": os.environ.get("CONDA_DEFAULT_ENV", "?"), "host": platform.node(),
                                  "sessions": ("one pinned session per row" if ep == "cpu"
                                               else f"{K_SESSIONS} fresh sessions per row and pass")},
                                 sort_keys=True), flush=True)
    s3.host_gate()
    cs = None
    if ep == "cpu":
        cs = cores()
        if len(cs) != 8 or any(len(c) != 2 for c in cs):
            raise SystemExit(f"expected 8 cores x 2 SMT siblings, found {cs}")
        print("TOPOLOGY_JSON " + json.dumps({"cores": cs, "pinned": {th: pinned_cpus(th, cs) for th in s3.CPU_THREADS}}),
              flush=True)
    clock = s3.ClockWitness() if ep == "cpu" else None
    if ep == "dml":
        gpu_engines()
    names = ["x4", "x11", "x4_q", "x11_q", "sx_x4", "sx_x11"] + [f"w_{s}{t}" for s in s3.SHAPES for t in ("", "_q")] \
        + [f"sw_{s}" for s in s3.SHAPES]
    data = s3.load(names)
    refs = {s: s3.reference(data[s3.X_OF[s]], data[f"w_{s}"]) for s in s3.SHAPES}
    feeds = {}
    for x in ("x4", "x11"):
        feeds[("fp32", "", x)] = data[x]
        feeds[("fp16", "", x)] = data[x].astype(np.float16)
        feeds[("int8", "s8", x)] = data[f"{x}_q"]
        feeds[("int8", "u8zp", x)] = (data[f"{x}_q"].astype(np.int16) + s3.ZP).astype(np.uint8)
    arms = [(a, th) for a in s3.ARMS["cpu"] for th in s3.CPU_THREADS] if ep == "cpu" else [(a, 0) for a in s3.ARMS["dml"]]
    models, forms_of = {}, {}

    def model(arm, s, form):
        if (arm, s, form) not in models:
            models[(arm, s, form)] = s3.ort_model(arm, s3.ort_weights(arm, s, data, form), form)
        return models[(arm, s, form)]

    # which form each arm places: probed once, as in stage 3 (these sessions also start the threads
    # ORT keeps once per process, before any row's read-back counts a pool)
    for arm, th in arms:
        for s in s3.SHAPES:
            forms = [""] if arm != "int8" else (["u8zp"] if ep == "cpu" else ["s8", "u8zp"])
            for form in forms:
                try:
                    sess = s3.make_session(model(arm, s, form), ep, th)
                except Exception as e:                            # DirectML may refuse an int8 form
                    print("SESSION_REFUSED " + json.dumps({"ep": ep, "arm": arm, "shape": s, "form": form,
                                                           "error": str(e)[:300]}), flush=True)
                    continue
                forms_of[(arm, th, s)] = form
                print("SESSION " + json.dumps({"ep": ep, "arm": arm, "threads": th, "shape": s, "form": form,
                                               "providers": sess.get_providers()}), flush=True)
                del sess
                gc.collect()
                break
            else:
                print("ARM_UNAVAILABLE " + json.dumps({"ep": ep, "arm": arm, "shape": s}), flush=True)
    rows = [(arm, th, s, M) for (arm, th, s) in forms_of for M in s3.MS]
    if clock:
        clock.wait_first()
    try:
        for p, order in ((1, rows), (2, rows[::-1])):
            for arm, th, s, M in order:
                form = forms_of[(arm, th, s)]
                feed = {"x": feeds[(arm, form, s3.X_OF[s])][:M]}
                K, N = s3.SHAPES[s]
                row = {"chip": ep, "arm": arm, "threads": th, "form": form, "config": "", "shape": s, "M": M,
                       "K": K, "N": N, "pass": p}

                def score(y):
                    h = None
                    if arm == "int8":
                        h = hashlib.sha256(np.ascontiguousarray(y, dtype=np.int32).tobytes()).hexdigest()
                        y = s3.dequant(y, float(data[f"sx_{s3.X_OF[s]}"][0]), data[f"sw_{s}"])
                    return s3.errors(y, refs[s][:M]), h

                if ep == "cpu":
                    ts, pin, y, w0, w1 = cpu_row(model(arm, s, form), th, cs, feed)
                    row["pin"] = pin
                    if ts is None:                                # the read-back did not match: void
                        print("ROW_VOID " + json.dumps(row), flush=True)
                        time.sleep(0.5)
                        continue
                    row.update(s3.stats(ts))
                    e, h = score(y)
                    row.update(e)
                    if h:
                        row["int32_sha"] = h
                    row["clock_pct"] = clock.covering(w0, w1)
                else:
                    meds, reps, errs, shas = dml_row(model(arm, s, form), feed, score)
                    row.update({"median_ms": round(statistics.median(meds), 4), "session_medians_ms": meds,
                                "session_range": round(max(meds) / min(meds), 4), "ms": [m for r in reps for m in r],
                                "session_ms": reps, "rel_l2": max(x["rel_l2"] for x in errs),
                                "max_abs_rel": max(x["max_abs_rel"] for x in errs),
                                "finite": all(x["finite"] for x in errs)})
                    if arm == "int8":
                        row["int32_sha"] = shas[-1]
                        row["int32_sessions_agree"] = len(set(shas)) == 1
                print("ROW_JSON " + json.dumps(row), flush=True)
                time.sleep(0.5)
    finally:
        if clock:
            clock.close()
    if ep == "dml":
        gpu_engines()
    return 0


# ---------------------------------------------------------------- the verdict

def verdict3b(paths) -> int:
    print("STAGE 3b: the verdict reads 3b's three logs alone; no stage 3 row enters it.\n")
    texts = [Path(p).read_text(encoding="utf-8", errors="replace") for p in paths]
    grab = lambda text, tag: [json.loads(line[len(tag):]) for line in text.splitlines() if line.startswith(tag)]
    problems = []
    cpu_rows, voids = grab(texts[1], "ROW_JSON "), grab(texts[1], "ROW_VOID ")
    for v in voids:
        problems.append(f"cpu {v['arm']} {v['threads']} threads {v['shape']} M={v['M']} pass {v['pass']}: VOID, "
                        f"the pinning read-back did not match ({v['pin']})")
    for r in cpu_rows:
        if not r.get("pin", {}).get("ok"):
            problems.append(f"cpu {r['arm']} {r['threads']} threads {r['shape']} M={r['M']}: no matching read-back")
    print(f"PINNING: {sum(1 for r in cpu_rows if r.get('pin', {}).get('ok'))} CPU rows read back as pinned, "
          f"{len(voids)} void")
    dml_rows = grab(texts[2], "ROW_JSON ")
    print(f"DIRECTML: {K_SESSIONS} fresh sessions per row and pass; a pass's value is the median of their medians")
    for r in sorted(dml_rows, key=lambda r: (r["arm"], r["shape"], r["M"], r["pass"])):
        meds = r.get("session_medians_ms", [])
        if len(meds) != K_SESSIONS:
            problems.append(f"dml {r['arm']} {r['shape']} M={r['M']} pass {r['pass']}: {len(meds)} sessions")
            continue
        agree = "" if r["arm"] != "int8" else ("  int32 sessions agree" if r.get("int32_sessions_agree")
                                               else "  int32 sessions DIFFER")
        print(f"  {r['arm']:4s} {r['shape']} M={r['M']:<4d} pass {r['pass']}  value {r['median_ms']:9.2f} ms  "
              f"sessions {', '.join(f'{m:.2f}' for m in meds)}  range {r['session_range']:.3f}x{agree}")
    print()
    rc, keep = s3.evaluate(paths)                                 # stage 3's rule, unchanged
    if problems:
        print("\nSTAGE 3b PROBLEMS")
        for p in problems:
            print(" ", p)
        rc, keep = 2, []
    print("\nSTAGE 3b VERDICT " + ("INCOMPLETE: it stands, and there is no third run without the user" if rc == 2
                                   else f"KEEP at M = {', '.join(map(str, keep))}" if keep
                                   else "KILL: at neither M does an NPU arm beat both the CPU and DirectML"))
    return rc


# ---------------------------------------------------------------- the pre-registration

PREREG = f"""\
LLM study stage 3b: prefill GEMM at Llama-2-7B's shapes, with the CPU's threads pinned and DirectML
timed over fresh sessions, pre-registered before any sitting

Why 3b exists
  The user's decision: "Re-run prefill, pinned". Stage 3b was designed after stage 3's two
  INCOMPLETE sittings (pre-registered at d51a427, sittings e9efb97 and 8ad8e1e) and after the noise
  study (67790e8, 535229c). Stage 3 stays INCOMPLETE in the record. Stage 3b is a new experiment:
  its verdict comes from its own three logs alone, and no stage 3 row enters it.

Unchanged from stage 3 (results/llm/llm_prefill_prereg_desktop2_20260923.log, and its sitting 2
structure from llm_prefill_prereg_rerun_desktop2_20260923.log)
  The question and the bar: an NPU arm must be faster than both the CPU (ONNX Runtime) and DirectML
  on the 780M, or more accurate than both.
  The workload: S1 (M, 4096) x (4096, 4096) x 4, S2 (M, 4096) x (4096, 11008) x 2, S3 (M, 11008) x
  (11008, 4096) x 1 per layer, M in {s3.MS}; layer time = 4 S1 + 2 S2 + S3.
  The inputs (the manifest pinned below), the dtypes and the arms: CPU fp32 and int8 (u8 x s8, zero
  point {s3.ZP}) at 8 and 16 threads, the faster counting per arm; DirectML fp16, fp32 and int8; NPU
  bf16 and int8 on the same 28 artifacts (re-verified below against stage 3's pins) and tiles, with
  S2 unsliced and sliced.
  The NPU sitting: stage 3's own code, run unchanged (tools/llm_prefill_bench.py npu), with its
  smoke, its correctness checks (bf16 against its own inputs <= {s3.BF16_ACC_MAX:g}; int8 exact) and its
  concatenation timing.
  Timing: {s3.WARMUP} warmups and {s3.REPS} timed runs per session; two passes, the second in reverse row order; one
  ORT session per CPU row, opened just before it and released after (as in stage 3's sitting 2); the
  CPU clock witness, display only (as in sitting 2). The sitting runs NPU, then CPU, then DirectML.
  Accuracy: rel-L2 against the float64 product of the fp32 inputs; an arm's error is its worst shape.
  The keep rule, per M: an NPU arm beats a chip if it is at least {s3.MARGIN:.2f}x faster by layer time than
  every arm of that chip whose rel-L2 is <= {s3.ACC_TIE:.2f}x its own; KEEP needs both chips beaten, and still
  beaten with S2's concatenation and {s3.SWITCHES} context switches of {s3.SWITCH_MS} ms per layer added. KILL otherwise.
  The repeat threshold: {100 * s3.REPEAT_MAX:.0f}%.

Changed, and nothing else (each designed from the noise study)
  (a) CPU: ONNX Runtime's intra-op threads are pinned, and the calling thread too.
      8 threads: one per physical core; the caller on the first CPU below, the 7 pool threads on
      the rest (session.intra_op_thread_affinities). Logical CPUs {{PIN8}}.
      16 threads: one per logical CPU; the caller on the first, the 15 pool threads on the rest.
      Logical CPUs {{PIN16}}.
      After the warmups, every session's threads are read back (GetThreadGroupAffinity and CPU
      sets). The read-back matches if the pool is exactly threads - 1 threads, each pinned to one
      logical CPU, together the row's CPUs but the caller's; the caller is pinned to its CPU; and
      no thread has CPU sets. A row whose read-back does not match is VOID: printed, not timed.
      The noise study: pinned one per core, int8 S1 at M = 512 read one level (4.11 / 4.13 ms)
      where unpinned 8 threads flipped between about 4.1 and 7.2. ORT 1.23.3's default does not pin
      on this machine.
  (b) DirectML: each row and pass is K = {K_SESSIONS} fresh sessions, one after another. Each is opened, run
      {s3.WARMUP} warmups and {s3.REPS} timed runs, and closed. A session's value is the median of its runs. The row's
      pass value is the median of its {K_SESSIONS} session values, and their range (max / min) is reported.
      The row's rel-L2 is the worst of its sessions' outputs; an int8 row also reports whether its
      sessions' int32 outputs agree.
      The noise study: 20 fresh sessions of fp32 S2 at M = 2048 landed at 112.09-133.42 ms (1.19x),
      while one session held 107.46-108.63.

The repeat rule, for the new structure
  CPU: a row's two pass medians (pinned rows) must agree within {100 * s3.REPEAT_MAX:.0f}% of their mean, as before.
  DirectML: a row's two pass values (each the median across its {K_SESSIONS} sessions) must agree within
  {100 * s3.REPEAT_MAX:.0f}% of their mean.
  NPU: a row's two pass medians, as before.
  A row's value is the mean of its two pass values, as before. The rule itself is stage 3's code
  (llm_prefill_bench.evaluate), run on 3b's logs; the last line, "STAGE 3b VERDICT", decides.

INCOMPLETE (no verdict)
  Everything that made stage 3 INCOMPLETE: a log without EXIT_CODE 0 (an NPU raise or timeout stops
  the sitting), the smoke failing, a missing row or pass, a row breaking the repeat rule, a
  non-finite output, DirectML fp16 or fp32 unavailable. Also a VOID CPU row, and a DirectML row
  without {K_SESSIONS} sessions.
  If 3b is INCOMPLETE, it stands. There is no third run without the user.

Written predictions (stated expectations; the rule decides, these do not)
  R1 Every CPU row reads back as pinned, and every CPU row holds the 10% rule (the noise study's
     pinned rows repeated within 0.5%).
  R2 CPU int8: pinned 8 threads within 5% of pinned 16 at every shape and M (the noise study: 4.11 /
     4.13 against 4.09 / 4.10 unpinned 16). CPU fp32: pinned 8 faster than pinned 16 at every shape
     (the noise study: 17.66 against 23.04 at S1, M = 512), so the CPU fp32 arm runs at 8 threads.
  R3 Every DirectML row holds on its across-session values. The row most likely to break is fp32
     S2 at M = 2048.
  R4 At least one DirectML fp32 row shows a session range of 1.10x or more within a pass.
  R5 The NPU rows land within 10% of both stage 3 sittings' values (all 56 held twice).
  R6 Accuracy as in stage 3, on the same inputs: NPU bf16 2.35e-3, every int8 arm 1.54e-2 with
     identical int32 outputs across chips, DirectML fp16 3.61e-4.
  R7 KILL at both M, as stage 3's rows that held suggest (a prior, not part of 3b's verdict): NPU
     bf16 loses to DirectML fp16, which is more accurate and took 0.71-0.80x the NPU's time in
     stage 3's sitting 1; NPU int8 loses to the CPU's int8 at M = 512 (0.80-0.81x) and stays under
     the {s3.MARGIN:.2f} line at M = 2048 (0.99-1.03x). The least certain call: NPU int8 against the pinned
     CPU int8 at M = 2048.

Unverified by design: as stage 3 (attention, norms, RoPE and the LM head; the activations between
GEMMs; DirectML IO binding; other CPU stacks; tiles outside the menu; the NPU's power modes;
prefill beside decode). Also: a pinned thread's placement is set by its affinity and read back once
per session, not observed per run; K = {K_SESSIONS} is a choice, not derived; what sets a DirectML session's
level is unattributed (the noise study).
"""


def prereg() -> int:
    cs = cores()
    text = PREREG.replace("{PIN8}", str(pinned_cpus(8, cs))).replace("{PIN16}", str(pinned_cpus(16, cs)))
    print(text)
    original = STAGE3_PREREG.read_text(encoding="utf-8")
    pinned = re.findall(r"insts ([0-9a-f]{16})  xclbin ([0-9a-f]{16})", original)
    now = []
    for t in s3.tiles():
        d = s3.tdir(t)
        i, x = s3.sha(d / "insts.bin"), s3.sha(d / "final.xclbin")
        now.append((i[:16], x[:16]))
        print(f"ARTIFACT {d.name} insts {i} xclbin {x}")
    manifest = s3.sha(s3.INPUTS / "manifest.json")
    wa, mm = s3.sha(s3.WA_DIR / "whole_array.py"), s3.sha(s3.MM_CC)
    same = (pinned == now and f"PINS inputs manifest {manifest}" in original
            and f"PINS whole_array.py {wa} mm.cc {mm}" in original)
    print(f"PINS {len(now)} NPU artifacts, the inputs manifest, whole_array.py and mm.cc: "
          + ("identical to stage 3's prereg log" if same else "DIFFER from stage 3's prereg log"))
    print("PINS inputs manifest", manifest)
    print("PINS whole_array.py", wa, "mm.cc", mm)
    print("TOPOLOGY_JSON " + json.dumps({"cores": cs, "pinned": {th: pinned_cpus(th, cs) for th in s3.CPU_THREADS}}))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.split("\n")
    n = sum(1 for d in dirty if d.strip())
    print(f"git HEAD {head}" + (f" (+{n} uncommitted paths, this stage's own files)" if n else ""))
    print("PREREG_JSON " + json.dumps({"stage": "3b", "ms": s3.MS, "shapes": s3.SHAPES, "per_layer": s3.PER_LAYER,
                                       "margin": s3.MARGIN, "acc_tie": s3.ACC_TIE, "repeat_max": s3.REPEAT_MAX,
                                       "bf16_acc_max": s3.BF16_ACC_MAX, "switch_ms": s3.SWITCH_MS,
                                       "switches": s3.SWITCHES, "warmup": s3.WARMUP, "reps": s3.REPS,
                                       "cpu_threads": s3.CPU_THREADS, "k_sessions": K_SESSIONS,
                                       "pinned": {th: pinned_cpus(th, cs) for th in s3.CPU_THREADS},
                                       "seed": s3.SEED, "git_head": head}))
    return 0 if same else 1


# ---------------------------------------------------------------- selftest (no chip, no timing)

def selftest() -> int:
    cs = cores()
    w = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 7) / 7
    model = s3.ort_model("fp32", w)
    feed = {"x": np.ones((64, 64), dtype=np.float32)}
    s3.make_session(model, "cpu", 8).run(None, feed)              # ORT's once-per-process threads first
    for th in s3.CPU_THREADS:
        ts, pin, _, _, _ = cpu_row(model, th, cs, feed)
        print(f"selftest pinned {th} threads: read-back {'matches' if pin['ok'] else 'DOES NOT MATCH'} "
              f"(pool {len(pin['pool'])}, caller {pin['main']}, CPU sets {pin['cpu_sets']})")
        assert pin["ok"] and ts is not None
    fake = [{"cpus": [c], "ideal": c, "cpu_sets": 0} for c in pinned_cpus(8, cs)[1:]]
    assert witness_ok(fake, [{"cpus": [0], "cpu_sets": 0}], pinned_cpus(8, cs))
    assert not witness_ok(fake[:-1], [{"cpus": [0], "cpu_sets": 0}], pinned_cpus(8, cs))       # a thread short
    assert not witness_ok(fake, [{"cpus": list(range(16)), "cpu_sets": 0}], pinned_cpus(8, cs))  # caller free
    print("selftest witness_ok: a match and two mismatches as expected")
    tmp = ROOT / "scratch/llm/prefill3b_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    cases = (("kill", 5.0, None, 0), ("keep", 1.0, None, 0), ("void", 5.0, "void", 2), ("dml-breaks", 5.0, "dml", 2))
    for name, dml16, fault, want in cases:
        tf = {"cpu fp32": .8, "cpu int8": 1.7, "dml fp16": dml16, "dml fp32": 1.0, "dml int8": .9,
              "npu bf16": 2.5, "npu int8": 3.5}
        err = {"fp32": 1e-7, "fp16": 5e-4, "bf16": 3e-3, "int8": 1.5e-2}
        logs = {}
        for chip in ("npu", "cpu", "dml"):
            out = []
            if chip == "npu":
                out += [f"SMOKE_JSON {json.dumps({'dtype': d, 'ok': True})}" for d in ("bf16", "i8")]
            for arm in s3.ARMS[chip]:
                for th in (s3.CPU_THREADS if chip == "cpu" else (0,)):
                    for s, (K, N) in s3.SHAPES.items():
                        for M in s3.MS:
                            for lb in (s3.NPU_ROWS[s] if chip == "npu" else [""]):
                                ms = s3.flops(K, N, M) / (tf[f"{chip} {arm}"] * 1e12) * 1e3 * (1.3 if lb == "S2u" else 1)
                                for p in (1, 2):
                                    v = ms * (1 + .01 * p)
                                    if fault == "dml" and chip == "dml" and arm == "fp32" and s == "S2" and M == 2048:
                                        v *= 1.2 if p == 2 else 1.0
                                    r = {"chip": chip, "arm": arm, "threads": th, "form": "", "config": lb, "shape": s,
                                         "M": M, "K": K, "N": N, "pass": p, "median_ms": v, "ms": [],
                                         "rel_l2": err[arm], "max_abs_rel": 0, "finite": True, "ok": True,
                                         "int32_sha": f"{s}{M}" if arm == "int8" else None,
                                         "concat_ms": .5 if lb.startswith("S2s") else 0.0}
                                    if chip == "cpu":
                                        r["pin"] = {"ok": True}
                                    if chip == "dml":
                                        r["session_medians_ms"] = [v] * K_SESSIONS
                                        r["session_range"] = 1.0
                                    tag = "ROW_JSON "
                                    if fault == "void" and chip == "cpu" and arm == "int8" and th == 8 and s == "S1" \
                                            and M == 512 and p == 2:
                                        tag, r["pin"] = "ROW_VOID ", {"ok": False}
                                    out.append(tag + json.dumps(r))
            out.append("EXIT_CODE: 0")
            logs[chip] = tmp / f"{name}_{chip}.log"
            logs[chip].write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\n---- synthetic 3b verdict: {name}")
        got = verdict3b([logs["npu"], logs["cpu"], logs["dml"]])
        assert got == want, (name, got)
    print("\nselftest: the pinning read-back and all four synthetic 3b verdicts as expected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prereg", "selftest"):
        sub.add_parser(c)
    o = sub.add_parser("ort")
    o.add_argument("--ep", choices=("cpu", "dml"), required=True)
    v = sub.add_parser("verdict")
    v.add_argument("logs", nargs=3, help="3b's NPU, CPU and DirectML sitting logs")
    a = ap.parse_args()
    if a.cmd == "ort":
        return ort3b(a.ep)
    if a.cmd == "verdict":
        return verdict3b(a.logs)
    return {"prereg": prereg, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
