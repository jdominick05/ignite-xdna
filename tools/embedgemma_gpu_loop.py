#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GPU worker of tools/embedgemma_concurrency_energy.py: EmbeddingGemma 2 through llama-server on the 780M
(ROCm0), looped over one document length inside a window the coordinator hands out.

The server command is the llama-server GPU baseline's (gemma4-embedding-baseline/scripts/llama_embed_bench.py: one
slot, -b = -ub = context, --pooling mean, --embd-normalize 2, -fa off unless --fa on), so a loop request costs what
a baseline request costs. Protocol: start the server, send --warmup requests, write GO.ready, wait for GO to hold
'START STOP' (epoch seconds), send one document per request from START until STOP, and write OUT with each
request's wall-clock span. Every response is checked: 768 finite floats of unit norm.

    python tools/embedgemma_gpu_loop.py --docs DOCS.json --go GO --out OUT.json [--length 2048] [--fa off]
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SERVER = Path.home() / "tools/llamacpp-b1343-rocm/llama-server.exe"
MODEL = Path.home() / "gemma4-assistant/models/embeddinggemma-2-BF16.gguf"


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(base, ids):
    item = post(base, "/embedding", dict(content=ids))
    item = item[0] if isinstance(item, list) else item
    e = item["embedding"]
    v = np.asarray(e[0] if isinstance(e[0], list) else e, dtype=np.float64)
    if v.shape != (768,) or not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-3:
        raise ValueError(f"bad embedding: shape {v.shape}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--go", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fa", choices=["on", "off"], default="off")
    ap.add_argument("--model", type=Path, default=MODEL)
    a = ap.parse_args()
    docs = [d["ids"] for d in json.loads(a.docs.read_text(encoding="utf-8"))["docs"] if d["length"] == a.length]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [str(SERVER), "-m", str(a.model), "--host", "127.0.0.1", "--port", str(port), "-c", str(a.length), "-np", "1",
           "-b", str(a.length), "-ub", str(a.length), "-t", str(a.threads), "--poll", "0", "-fa", a.fa,
           "--embedding", "--pooling", "mean", "--embd-normalize", "2", "-ngl", "99", "-dev", "ROCm0"]
    log = Path(str(a.out) + ".server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("server readiness")
            time.sleep(0.5)
        for i in range(a.warmup):
            embed(base, docs[i % len(docs)])
        Path(str(a.go) + ".ready").write_text(f"{time.time()}\n", encoding="utf-8")
        print("LOOP_READY", flush=True)
        while not a.go.exists():
            time.sleep(0.1)
        time.sleep(0.2)
        start, stop = map(float, a.go.read_text(encoding="utf-8").split())
        while time.time() < start:
            time.sleep(0.005)
        spans, i = [], 0
        while time.time() < stop:
            t0 = time.time()
            embed(base, docs[i % len(docs)])
            spans.append([t0, time.time(), 1])
            i += 1
        a.out.write_text(json.dumps(dict(device="gpu", length=a.length, start=start, stop=stop, spans=spans,
                                         server_command=cmd)) + "\n", encoding="utf-8")
        print(f"LOOP_DONE {len(spans)} requests", flush=True)
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
