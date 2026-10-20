#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small-LLM decode: where a token's 13.7 ms of device time goes, before merging calls.

Each program is a subset of the decoder's own calls (kernels/llm_gemv/decode.py), on the decoder's own buffers and
layout, timed as one dispatch:

    full    the token's 181 calls (its instruction stream must equal build/llm_decode/token's byte for byte)
    proj    the 121 projections with their flags (qkv+norm+RoPE+append, o+residual, gate/up+norm+SiLU,
            down+residual, the head), no attention
    attn    the 60 attention calls; score and pv: their 30 each
    tiny61, tiny121, tiny181   that many chained 192x192 projections (36 KB of weights each): the fixed cost of a
            call, from the slope
    host61, host181   the same calls with x from a host-written slot: 20 shim BDs a call instead of 44, so the slope
            separates a cost per control operation from a fixed latency per call

The programs read whatever earlier dispatches left in the buffers; nothing here is checked for values except the
full stream's identity with the verified build. Run on the NPU through the wrapper:

    python tools/llm_decode_breakdown.py compile
    ./scripts/research-lowlevel.sh --log results/aie/<name>.log --npu --seconds 600 -- <iron python> \\
        tools/llm_decode_breakdown.py run
"""
import argparse
import dataclasses
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

from kernels.llm_gemv import decode as D  # noqa: E402
from kernels.llm_gemv import pack  # noqa: E402

BUILD = ROOT / "build" / "llm_decode_breakdown"
VERIFIED = ROOT / "build" / "llm_decode" / "token"
P = 40                       # the position the tails carry (the attention calls stream all CAP rows regardless)
WARMUP, TOKENS, ROUNDS, WAIT_MS = 3, 20, 2, 10000
TINY = (192, 192)


def _tiny_tail():
    from kernels.llm_gemv.design import Call
    return Call(0, D.X_WORDS, D.OUT["a"], *TINY, x_out=D.OUT["h"]).header().size


HOST_SLOT = D.X_WORDS + _tiny_tail()
X_WORDS = HOST_SLOT + pack.X_SLOT_WORDS


def programs():
    from kernels.llm_gemv.design import Attn, Call
    calls = D.CALLS
    tiny = lambda n: [Call(D.W_OFF[(i % D.LAYERS, "o")], D.X_WORDS, D.OUT["a" if i % 2 else "h1"], *TINY,  # noqa
                           x_out=D.OUT["h1" if i % 2 else "a"]) for i in range(n)]
    host = lambda n: [Call(D.W_OFF[(i % D.LAYERS, "o")], HOST_SLOT, D.OUT["a" if i % 2 else "h1"], *TINY)  # noqa
                      for i in range(n)]
    return {"full": calls,
            "proj": [c for c in calls if not isinstance(c, Attn)],
            "attn": [c for c in calls if isinstance(c, Attn)],
            "score": [c for c in calls if isinstance(c, Attn) and c.flags == pack.F_SCORE],
            "pv": [c for c in calls if isinstance(c, Attn) and c.flags == pack.F_PV],
            "tiny61": tiny(61), "tiny121": tiny(121), "tiny181": tiny(181), "host61": host(61), "host181": host(181)}


def compile_all():
    from kernels.llm_gemv import design
    for name, calls in programs().items():
        d = BUILD / name
        if (d / "build.json").exists():
            print("EXISTS", name, flush=True)
            continue
        design.build(d, calls, D.W_WORDS, X_WORDS, D.OUT_WORDS, f"breakdown_{name}")
    same = (BUILD / "full" / "insts.bin").read_bytes() == (VERIFIED / "insts.bin").read_bytes()
    print("FULL_STREAM_EQUALS_VERIFIED", same, flush=True)
    sets = set()
    for name in programs():
        prj = BUILD / name / "prj"
        sets.add(tuple(sorted(p.read_bytes() for p in [*prj.glob("cdo_main/*.bin"), *prj.rglob("*core_*.elf")])))
    print("ARRAY_CONFIG_SETS", len(sets), flush=True)
    if not same or len(sets) != 1:
        raise SystemExit("the programs do not share the verified array configuration")


def run():
    sys.path.insert(0, str(ROOT / "tools"))
    from llm_gemv_probe import witness
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    from kernels.llm_gemv import reference as R
    if not witness("pre"):
        raise SystemExit("NPU is busy; did not dispatch")
    if (BUILD / "full" / "insts.bin").read_bytes() != (VERIFIED / "insts.bin").read_bytes():
        raise SystemExit("full is not the verified stream")
    w = R.Weights.load(R.safetensors(R.model_file()))
    try:
        with XrtSiliconHarness(0) as h:
            xrt = h.pyxrt
            to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            h.load_xclbin(str(BUILD / "full" / "final.xclbin"))
            wb = h.create_host_bo(D.W_WORDS * 4, 3)
            np.frombuffer(wb.map(), np.uint32)[:] = D.arena(w)
            wb.sync(to)
            from kernels.llm_gemv.design import Call
            x = np.zeros(X_WORDS, np.uint32)
            x[:D.X_WORDS] = D.x_arg(P)
            t = Call(0, D.X_WORDS, D.OUT["a"], *TINY, x_out=D.OUT["h"]).header()
            x[D.X_WORDS:D.X_WORDS + t.size] = t
            x[HOST_SLOT:] = pack.x_slot(np.zeros(TINY[1], np.uint16), TINY[0])
            xb = h.create_host_bo(X_WORDS * 4, 4)
            xb.write(x.tobytes(), 0)
            xb.sync(to)
            ob = h.create_host_bo(D.OUT_WORDS * 4, 5)
            ob.write(np.zeros(D.OUT_WORDS, np.uint32).tobytes(), 0)
            emb = R.em.bf16_bits(np.asarray(w.embed[1], np.float32)).view(np.uint32)
            ob.write(emb.tobytes(), D.OUT["h"] * 4)
            ob.sync(to)
            runs, keep = {}, []
            for name in programs():
                bo, count = h.create_instruction_bo((BUILD / name / "insts.bin").as_posix())
                r = xrt.run(h.kernel)
                for i, v in enumerate((3, bo, count, wb, xb, ob)):
                    r.set_arg(i, v)
                runs[name] = r
                keep.append(bo)

            def go(r):
                r.start()
                state = r.wait(WAIT_MS)
                if state != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"dispatch did not complete: {state}; stopping without retry")

            med = {}
            for rnd in range(1, ROUNDS + 1):
                for name, r in runs.items():
                    ts = []
                    for i in range(WARMUP + TOKENS):
                        t0 = time.perf_counter_ns()
                        go(r)
                        if i >= WARMUP:
                            ts.append((time.perf_counter_ns() - t0) / 1e6)
                    med[rnd, name] = statistics.median(ts)
                    print("RESULT_JSON " + json.dumps(dict(round=rnd, program=name, calls=len(programs()[name]),
                                                           times_ms=ts, median_ms=med[rnd, name], min_ms=min(ts))),
                          flush=True)
            for rnd in range(1, ROUNDS + 1):
                m = {k[1]: v for k, v in med.items() if k[0] == rnd}
                slope = (m["tiny181"] - m["tiny61"]) / 120
                host = (m["host181"] - m["host61"]) / 120
                print("DERIVED_JSON " + json.dumps(dict(
                    round=rnd, ms_per_tiny_call=slope, ms_per_host_call=host, tiny_intercept_ms=m["tiny61"] - 61 * slope,
                    proj_plus_attn_minus_full=m["proj"] + m["attn"] - m["full"],
                    score_plus_pv_minus_attn=m["score"] + m["pv"] - m["attn"])), flush=True)
            del runs
    finally:
        gc.collect()
        witness("post_in_process", wait_s=5)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("compile", "run"))
    a = ap.parse_args()
    if a.mode == "compile":
        compile_all()
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main())
