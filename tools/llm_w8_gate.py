#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode, the W8A16 speed gate: does streaming half the weight bytes take about half the time?

The planned int8 layout carries 24 outputs x 192 inputs per 4,608-byte object, the size of today's bf16 object (12 x
192). A W8A16 token therefore moves the same objects as a bf16 token whose every projection has half the rows, padded
where 24 does not divide a core's rows: qkv 960 -> 1152, o and down 576 -> 768, and the head split in 4 calls (its
per-row scales must fit a call's first object). This tool times exactly that DMA pattern with today's bf16 kernel,
before any int8 code exists:

    bf16   step 1's arm C: the token's 121 projections at their real shapes, one dispatch (the same-sitting baseline)
    proxy  30 x {576x576, 384x576, 1536x576, 384x1536} + 4 x 6144x576, one dispatch: the W8A16 token's objects
           (149.96 MB against 265.42 MB)

The core's extra work per object (an int8 -> bf16 conversion and twice the MACs) is not in the proxy; it is counted
from the compiled bundles separately. Weights are random bf16 (timing does not depend on values); one output of each
program is checked bit for bit against kernels/llm_gemv/oracle.py first. Run on the NPU through the wrapper:

    python tools/llm_w8_gate.py compile
    ./scripts/research-lowlevel.sh --log results/aie/<name>.log --npu --seconds 600 -- <iron python> tools/llm_w8_gate.py run
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kernels.llm_gemv import oracle, pack  # noqa: E402

BUILD = ROOT / "build" / "llm_w8_gate"
LAYERS = 30
REAL = [(960, 576), (576, 576), (3072, 576), (576, 1536)]
PROXY = [(576, 576), (384, 576), (1536, 576), (384, 1536)]
PROGRAMS = {"bf16": REAL * LAYERS + [(49152, 576)], "proxy": PROXY * LAYERS + [(6144, 576)] * 4}
SHAPES = sorted({s for calls in PROGRAMS.values() for s in calls})
X_SLOTS = {s: i * pack.X_SLOT_WORDS for i, s in enumerate(SHAPES)}
X_WORDS = len(SHAPES) * pack.X_SLOT_WORDS
OUT, _o = {}, 0
for _s in SHAPES:
    OUT[_s] = _o
    _o += _s[0] // 2
OUT_WORDS = _o
WARMUP, TOKENS, ROUNDS, WAIT_MS = 3, 20, 2, 5000
# Written before the sitting: GO when the proxy's median is at most 0.70 x the bf16 median in both rounds (W8A16
# then saves at least 30% of the projections' time); otherwise STOP and report before any int8 code is written.
GO_RATIO = 0.70


def layout(name):
    """(shape, weight word offset) per call; every program packs its calls back to back from word 0."""
    off, out = 0, []
    for s in PROGRAMS[name]:
        out.append((s, off))
        off += pack.block_words(*s)
    return out, off


ARENA_WORDS = max(layout(n)[1] for n in PROGRAMS)


def programs():
    from kernels.llm_gemv.design import Call
    return {n: [Call(off, X_SLOTS[s], OUT[s], *s) for s, off in layout(n)[0]] for n in PROGRAMS}


def compile_all():
    from kernels.llm_gemv import design
    for name, calls in programs().items():
        d = BUILD / name
        if (d / "build.json").exists():
            print("EXISTS", name, flush=True)
            continue
        meta = design.build(d, calls, ARENA_WORDS, X_WORDS, OUT_WORDS, f"w8_gate_{name}")
        print("BUILT", name, json.dumps(meta.get("tokens")), flush=True)


def weights(seed=7):
    """The arena: random bf16 in [-1/16, 1/16), so no sum overflows; values do not change the timing."""
    from ignite_xdna.compiler import engine_bf16_emulator as em
    rng = np.random.default_rng(seed)
    return em.bf16_bits(em.to_bf16(rng.uniform(-1 / 16, 1 / 16, ARENA_WORDS * 2).astype(np.float32)))


def xs(seed=0):
    from ignite_xdna.compiler import engine_bf16_emulator as em
    rng = np.random.default_rng(seed)
    return {s: em.bf16_bits(em.to_bf16(rng.normal(0, 1, s[1]).astype(np.float32))) for s in SHAPES}


def run():
    sys.path.insert(0, str(ROOT / "tools"))
    from llm_gemv_probe import witness
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    w_bits, x = weights(), xs()
    ok = True
    try:
        with XrtSiliconHarness(0) as h:
            xrt = h.pyxrt
            to, fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            h.load_xclbin(str(BUILD / "bf16" / "final.xclbin"))
            arena = h.create_host_bo(ARENA_WORDS * 4, 3)
            view = np.frombuffer(arena.map(), np.uint32)
            # each program's calls get the arena words they read, packed in the core order (pack.pack)
            packed = {}
            for name in PROGRAMS:
                calls, _ = layout(name)
                for s, off in calls:
                    if off not in packed:
                        m = w_bits[off * 2:off * 2 + s[0] * s[1]].reshape(s)
                        view[off:off + pack.block_words(*s)] = pack.pack(m)
                        packed[off] = (s, m)
            arena.sync(to)
            xbo = h.create_host_bo(X_WORDS * 4, 4)
            xbo.write(np.concatenate([pack.x_slot(x[s], s[0]) for s in SHAPES]).tobytes(), 0)
            xbo.sync(to)
            out = h.create_host_bo(OUT_WORDS * 4, 5)
            runs = {}
            for name in PROGRAMS:
                bo, count = h.create_instruction_bo((BUILD / name / "insts.bin").as_posix())
                r = xrt.run(h.kernel)
                for i, v in enumerate((3, bo, count, arena, xbo, out)):
                    r.set_arg(i, v)
                runs[name] = (r, bo)

            def go(r):
                r.start()
                state = r.wait(WAIT_MS)
                if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"dispatch did not complete: {state}; stopping without retry")

            # exactness: each program's last call of each shape (later calls overwrite earlier ones' outputs)
            for name in PROGRAMS:
                out.write(np.full(OUT_WORDS, 0x7FC9_8765, np.uint32).tobytes(), 0)
                out.sync(to)
                go(runs[name][0])
                out.sync(fr)
                o = np.frombuffer(out.read(OUT_WORDS * 4, 0), np.uint32)
                last = {s: off for s, off in layout(name)[0]}
                bad = {}
                for s, off in last.items():
                    want = oracle.gemv(packed[off][1], x[s])
                    bad[f"{s[0]}x{s[1]}"] = int(np.count_nonzero(pack.unpack_y(o[OUT[s]:], s[0]) != want))
                print("EXACT_JSON " + json.dumps(dict(program=name, checked=bad,
                                                      mismatched_elements=sum(bad.values()))), flush=True)
                ok &= sum(bad.values()) == 0
            if not ok:
                print("TIMING SKIPPED: not exact", flush=True)
                return 1
            nbytes = {n: layout(n)[1] * 4 for n in PROGRAMS}
            medians = {}
            for rnd in range(1, ROUNDS + 1):
                for name in PROGRAMS:
                    t = []
                    for i in range(WARMUP + TOKENS):
                        t0 = time.perf_counter_ns()
                        go(runs[name][0])
                        if i >= WARMUP:
                            t.append((time.perf_counter_ns() - t0) / 1e6)
                    med = statistics.median(t)
                    medians[rnd, name] = med
                    print("RESULT_JSON " + json.dumps(dict(
                        round=rnd, program=name, calls=len(PROGRAMS[name]), tokens=TOKENS, warmup=WARMUP,
                        times_ms=t, median_ms=med, min_ms=min(t), weight_bytes=nbytes[name],
                        gbps=nbytes[name] / med / 1e6)), flush=True)
            ratios = [medians[r, "proxy"] / medians[r, "bf16"] for r in range(1, ROUNDS + 1)]
            print("VERDICT_JSON " + json.dumps(dict(rule=f"GO if proxy/bf16 <= {GO_RATIO} in every round",
                                                    ratios=ratios, verdict="GO" if max(ratios) <= GO_RATIO
                                                    else "STOP")), flush=True)
            del runs
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compile", "run", "layout"))
    a = ap.parse_args()
    if a.mode == "compile":
        compile_all()
    elif a.mode == "layout":
        print(json.dumps({n: dict(calls=len(c), weight_bytes=layout(n)[1] * 4) for n, c in PROGRAMS.items()}
                         | dict(arena_words=ARENA_WORDS, x_words=X_WORDS, out_words=OUT_WORDS)))
    else:
        return run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
