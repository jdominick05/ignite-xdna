#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Laya as a resident decision service: load the model once, answer typed questions over localhost.

Laya (Convai Innovations, Apache 2.0, receptron/laya-onnx) is ModernBERT-large plus a decision head: one forward
pass per question, a probability per option, no text generated. Loading it takes seconds, so ``serve`` keeps one
ONNX Runtime session resident and ``ask`` sends it one question at a time.

    python tools/laya_daemon.py serve [--backend cpu|dml|npu] --model-dir D [--model M] [--no-pad] [--port 8765]
    python tools/laya_daemon.py ask --question Q --option A --option B [--type choice] [--context T | --context-file F]
    python tools/laya_daemon.py ask --type score --question Q --option "low" --option "high" ...
    python tools/laya_daemon.py ask --type noul --question "Is it safe to ...?" --context-file F
    python tools/laya_daemon.py golden | status | stop     [--port 8765]
    python tools/laya_daemon.py bench [--rounds 4]          latency through the resident session

Question types are Laya's own: ``choice`` picks one of the options (an option may carry a description as
``label: description``); ``score`` rates on the options as ordered levels 0..k-1; ``noul`` is yes/no (options, if
given, describe false then true). ``--context`` is the state the question is about: free text or JSON.

``ask`` prints one JSON object: each option with its calibrated probability (Laya's shipped per-type, per-option-
count temperature), the chosen option, confidence (1 - normalised entropy, as the reference computes it), the
uncalibrated probabilities at T=1, the expected level for ``score``, tokens, latency_ms of the forward pass and the
backend. It exits 2 if no daemon answers. ``golden`` runs receptron's reference support ticket through the resident
session and compares it to the reference's published output to 4 decimals, as tools/laya_accuracy.py golden does.

Backends: ``dml`` (the Radeon 780M through DirectML; needs the allowzero-cleared copy from
``tools/laya_baseline.py dmlcopy``) and ``cpu`` (8 intra-op threads). The default model is the guard-free copy
(``tools/laya_npu_screen.py noguard``), whose logits are bit-identical to the shipped export on 2,403 items and
which runs faster on both. ``npu`` is tools/laya_hybrid.py: the 120 weight products as bf16 GEMMs on the NPU in
one hardware context, everything else on the CPU. It is built for 512 tokens only, and its products are bf16, so
its logits follow the bf16 emulation, not the fp32 reference: ``golden`` is an fp32 check and is not expected to
pass to 4 decimals there.

Inputs are encoded exactly as tools/laya_accuracy.py encodes them (its tokenizer, build_sequence and feed), then
padded with [PAD] under attention mask 0 to Laya's 512-token limit, so every backend runs the same computation (padded,
the guard-free copy stays bit-identical to the shipped export: tools/laya_npu_screen.py padded). ``--no-pad`` runs each question at its own
length instead, which is faster on cpu and dml for short questions. The server binds 127.0.0.1 only and serves one
request at a time. Laya's answer is a probability over the options it was given, not a judgement about anything else.
Run cpu and dml in the ryzen-ai-1.8.0 env (ONNX Runtime with DirectML), npu in mlir-aie-iron (ONNX Runtime CPU and
pyxrt), each activated with conda.
"""
import argparse
import json
import math
import socket
import socketserver
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_accuracy as la  # noqa: E402
import laya_baseline as lb  # noqa: E402

DEFAULT_PORT = 8765
DEFAULT_MODEL = {"dml": "laya_az0_noguard.onnx", "cpu": "laya_az0_noguard.onnx", "npu": "laya_az0_noguard.onnx"}


class Laya:
    def __init__(self, model_dir: Path, model: Path, backend: str, threads: int, pad: bool = True,
                 gemm_dir: Path | None = None):
        self.cfg = json.loads((model_dir / "laya_config.json").read_text(encoding="utf-8"))
        self.enc = la.tokenizer(model_dir)
        self.backend, self.model, self.pad = backend, model, pad or backend == "npu"
        t0 = time.perf_counter()
        if backend == "npu":
            import laya_hybrid as lh
            self.plan = lh.Plan(model, threads)
            self.prod = lh.NpuProducts(self.plan, gemm_dir)
            self.run = lambda feed: self.plan.run(feed, self.prod)
            self.providers = ["NPU (one-context bf16 GEMM) + CPUExecutionProvider"]
        else:
            sess = lb.session(model, backend, threads)
            self.run = lambda feed: sess.run(None, feed)
            self.providers = sess.get_providers()
        self.load_s = time.perf_counter() - t0
        self.calls = 0

    def logits(self, q: dict, state) -> tuple[np.ndarray, int, float]:
        ids, mk = la.build_sequence(self.enc, state, q, self.cfg["max_len"], self.cfg["head_max_len"])
        qt = la.QTYPES[q["t"]]
        feed = la.feed(ids, mk, qt)
        if self.pad:   # [PAD] under attention mask 0 to the full 512: the same computation on every backend
            n = len(ids)
            feed["input_ids"] = np.array([ids + [la.PAD] * (self.cfg["max_len"] - n)], np.int64)
            feed["attention_mask"] = np.array([[1] * n + [0] * (self.cfg["max_len"] - n)], np.int64)
        t0 = time.perf_counter()
        z = self.run(feed)[0][0, :len(mk)]
        self.calls += 1
        return z, len(ids), (time.perf_counter() - t0) * 1e3

    def ask(self, req: dict) -> dict:
        qtype = req.get("type", "choice")
        if qtype not in la.QTYPES:
            raise ValueError(f"type must be one of {sorted(la.QTYPES)}")
        opts = req.get("options") or []
        if qtype == "choice":
            if len(opts) < 2:
                raise ValueError("choice needs at least two options")
            crit = {}
            for o in opts:
                label, _, desc = o.partition(":")
                crit[label.strip()] = desc.strip() or None
            qdef = {"type": "choice", "instructions": req["question"], "criteria": crit}
            labels = list(crit)
        elif qtype == "score":
            if len(opts) < 2:
                raise ValueError("score needs at least two levels")
            qdef = {"type": "score", "instructions": req["question"], "criteria": list(opts)}
            labels = list(opts)
        else:
            if opts and len(opts) != 2:
                raise ValueError("noul takes no options, or exactly two: false then true")
            crit = {"false": opts[0], "true": opts[1]} if opts else None
            qdef = {"type": "noul", "instructions": req["question"], "criteria": crit}
            labels = ["no", "yes"]
        q = la.to_internal(qdef)
        z, tokens, ms = self.logits(q, req.get("context", ""))
        if len(z) != len(labels):
            raise ValueError(f"{len(labels)} options but {len(z)} markers fit in the input; shorten the options")
        t = la.temperature(self.cfg, la.QTYPES[q["t"]], len(z))
        p, p1 = la.softmax_t(z, t), la.softmax_t(z, 1.0)
        ent = -float((p * np.log(np.clip(p, 1e-12, 1))).sum())
        out = {"type": qtype, "options": [{"option": lab, "p": round(float(x), 4), "p_t1": round(float(y), 4)}
                                          for lab, x, y in zip(labels, p, p1)],
               "chosen": labels[int(np.argmax(p))], "confidence": round(1.0 - ent / math.log(len(p)), 4),
               "temperature": round(t, 4), "tokens": tokens, "latency_ms": round(ms, 2), "backend": self.backend}
        if qtype == "score":
            out["expected_level"] = round(float((np.arange(len(p)) * p).sum()), 4)
        return out

    def golden(self) -> dict:
        got, tokens = {}, 0
        for qid, qdef in la.FIXTURE_QS.items():
            q = la.to_internal(qdef)
            z, n, _ = self.logits(q, la.FIXTURE_STATE)
            tokens += n
            got[qid] = la.softmax_t(z, la.temperature(self.cfg, la.QTYPES[q["t"]], len(z)))
        p = got["department"]
        conf = 1.0 - float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()) / math.log(len(p))
        have = {"input_tokens": tokens, "department": [round(float(x), 4) for x in p],
                "department_confidence": round(conf, 4),
                "urgency_score": round(float((np.arange(4) * got["urgency"]).sum()), 4),
                "urgency": [round(float(x), 4) for x in got["urgency"]],
                "churn_risk": round(float(got["churn_risk"][1]), 4)}
        return {"pass": have == la.FIXTURE_EXPECT, "have": have, "expected": la.FIXTURE_EXPECT,
                "backend": self.backend}


def serve(a) -> int:
    model_dir = Path(a.model_dir)
    model = Path(a.model) if a.model else model_dir / DEFAULT_MODEL[a.backend]
    laya = Laya(model_dir, model, a.backend, a.threads, pad=not a.no_pad, gemm_dir=Path(a.gemm_dir))
    laya.ask({"type": "choice", "question": "warm-up", "options": ["a", "b"]})
    print(f"[serve] {model.name} on {a.backend} ({laya.providers[0]}), loaded in {laya.load_s:.1f} s; "
          f"listening on 127.0.0.1:{a.port}", flush=True)
    stop = {"now": False}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline()
            try:
                req = json.loads(line)
                op = req.get("op", "ask")
                if op == "ask":
                    res = laya.ask(req)
                    print(f"[ask] {res['type']} k={len(res['options'])} tokens={res['tokens']} "
                          f"{res['latency_ms']:.1f} ms", flush=True)
                elif op == "golden":
                    res = laya.golden()
                    print(f"[golden] {'PASS' if res['pass'] else 'FAIL'}", flush=True)
                elif op == "status":
                    res = {"backend": laya.backend, "providers": laya.providers, "model": str(laya.model),
                           "padded_to_512": laya.pad,
                           "calls": laya.calls, "load_s": round(laya.load_s, 2)}
                elif op == "stop":
                    res, stop["now"] = {"stopping": True}, True
                else:
                    raise ValueError(f"unknown op {op!r}")
                body = {"ok": True, **res}
            except Exception as e:      # one bad request must not take the daemon down
                body = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(body) + "\n").encode("utf-8"))

    socketserver.TCPServer.allow_reuse_address = False
    with socketserver.TCPServer(("127.0.0.1", a.port), Handler) as srv:
        while not stop["now"]:
            srv.handle_request()
    if a.backend == "npu":
        laya.prod.close()
    print("[serve] stopped", flush=True)
    return 0


def call(port: int, req: dict, timeout: float = 120.0) -> dict:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except OSError as e:
        print(json.dumps({"ok": False, "error": f"no Laya daemon on 127.0.0.1:{port} ({e}); start one with "
                                                f"'python tools/laya_daemon.py serve'"}))
        raise SystemExit(2)
    return json.loads(buf)


BENCH_QS = [
    ("choice", "Which change should land first?", ["fix: a correctness bug", "perf: a 2% speedup", "docs: a typo"]),
    ("choice", "Which machine should run this job?", ["laptop: 16 GB, NPU", "desktop1: 32 GB, discrete GPU, no NPU",
                                                     "desktop2: 32 GB, NPU"]),
    ("noul", "Is it safe to delete this directory?", []),
    ("score", "How risky is this change?", ["trivial", "low", "moderate", "high"]),
    ("choice", "What is the next step?", ["measure", "build", "document", "ask the user"]),
]
BENCH_CTX = ("The job quantizes a large detector with AdaRound and needs about 30 GB of RAM and a GPU for fine "
             "tuning. The directory holds generated build outputs that a script can regenerate in an hour.")


def bench(a) -> int:
    """Latency through the resident session: real-length questions, then one padded to the 512-token limit."""
    rows = []
    for r in range(a.rounds):
        for qtype, question, opts in BENCH_QS:
            t0 = time.perf_counter()
            res = call(a.port, {"op": "ask", "type": qtype, "question": question, "options": opts,
                                "context": BENCH_CTX})
            rt = (time.perf_counter() - t0) * 1e3
            rows.append(("real", res["tokens"], res["latency_ms"], rt))
        long_ctx = " ".join([BENCH_CTX] * 20)
        t0 = time.perf_counter()
        res = call(a.port, {"op": "ask", "type": "choice", "question": BENCH_QS[0][1], "options": BENCH_QS[0][2],
                            "context": long_ctx})
        rows.append(("max_len", res["tokens"], res["latency_ms"], (time.perf_counter() - t0) * 1e3))
    st = call(a.port, {"op": "status"})
    print(f"[bench] backend {st['backend']} ({st['providers'][0]}), model {Path(st['model']).name}, "
          f"{a.rounds} rounds")
    for kind in ("real", "max_len"):
        sel = [x for x in rows if x[0] == kind]
        toks = sorted({x[1] for x in sel})
        fwd, rt = [x[2] for x in sel], [x[3] for x in sel]
        print(f"[bench] {kind:7s} n={len(sel):3d} tokens {toks[0]}-{toks[-1]}: forward median "
              f"{float(np.median(fwd)):.1f} ms (min {min(fwd):.1f}, max {max(fwd):.1f}); round trip median "
              f"{float(np.median(rt)):.1f} ms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--backend", choices=("cpu", "dml", "npu"), default="cpu",
                   help="cpu (default); dml leaves the CPU free; npu runs the 120 weight products on the NPU "
                        "(mlir-aie-iron env)")
    s.add_argument("--no-pad", action="store_true",
                   help="run each question at its own length instead of padding to 512 (not with --backend npu)")
    s.add_argument("--gemm-dir", default="scratch/gemm_generic/aot6", help="npu: the generic GEMM builds")
    s.add_argument("--model-dir", required=True, help="the laya-onnx directory: laya_config.json and tokenizer/")
    s.add_argument("--model", default=None, help="ONNX file (default: <model-dir>/laya_az0_noguard.onnx)")
    s.add_argument("--threads", type=int, default=8, help="cpu backend intra-op threads")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    q = sub.add_parser("ask")
    q.add_argument("--question", required=True)
    q.add_argument("--option", action="append", default=[], help="repeat; 'label: description' for choice")
    q.add_argument("--type", choices=sorted(la.QTYPES), default="choice")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--context", default="")
    g.add_argument("--context-file", default=None)
    q.add_argument("--port", type=int, default=DEFAULT_PORT)
    for name in ("golden", "status", "stop"):
        sub.add_parser(name).add_argument("--port", type=int, default=DEFAULT_PORT)
    b = sub.add_parser("bench", help="latency through the daemon: 5 real-length questions + 1 at 512 tokens, per round")
    b.add_argument("--rounds", type=int, default=4)
    b.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a)
    if a.cmd == "bench":
        return bench(a)
    if a.cmd == "ask":
        ctx = Path(a.context_file).read_text(encoding="utf-8") if a.context_file else a.context
        res = call(a.port, {"op": "ask", "type": a.type, "question": a.question, "options": a.option,
                            "context": ctx})
    else:
        res = call(a.port, {"op": a.cmd})
    print(json.dumps(res, indent=1))
    return 0 if res.get("ok") and res.get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
