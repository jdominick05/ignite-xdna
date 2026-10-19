#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingGemma 2 retrieval quality on MTEB SciFact (test split), to judge weight formats against fp32 HF.

The 56-document gate set's nearest-neighbour bar proved too coarse to separate W8A16 from W8A8 (BENCHMARKS,
"W8A8 ... under CPU emulation"). This scores real retrieval instead; the bar is registered in DECISIONS
("Quality bar for W8A8"): nDCG@10 at most 0.5 points below fp32 HF's, and mean top-10 overlap with fp32 HF >= 90%.

    tokenize  (env laya-ref: transformers)  --raw DIR (hf download mteb/scifact) --out docs.json
              the test queries with "task: search result | query: ", every corpus document with
              "title: {title or none} | text: {text}", truncated at --max-len tokens, sorted by length so the
              hybrid batches equal lengths. Output is tools/embedgemma_hybrid.py's --docs format plus kind/key.
    hf        (env laya-ref: torch)  --docs docs.json --out X.npz [--quant none|w8a16|w8a8]   HF embeddings, as
              gemma4-embedding-baseline/scripts/hf_reference.py computes them (mean pool, L2 normalise). --quant
              fake-quantizes every text Linear (the encoder's weight products; embeddings untouched) with
              kernels/embedgemma/w4.py's exact functions:
                w8a16  weights G=64 + 20-point clip, dequantized to bf16; inputs rounded to bf16
                w8a8   weights per channel (G = K) + clip; inputs int8 per token (w4.quantize_rows), dequantized
              Everything else stays fp32 HF, so the device's other substitutions (integer softmax, GeGLU model)
              are not in these numbers: they are gated separately and are common to both formats.
    score     --docs docs.json --raw DIR --emb X.npz [--ref ref.npz]   nDCG@10 (qrels test) and, against --ref,
              mean top-10 overlap

The weight-format embeddings come from embedgemma_hybrid.py --backend cpu --docs docs.json --reference ref.npz
--save-embeddings X.npz with that format's flags. Downloaded files are data: this reads them, nothing else.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SNAPSHOT = Path(r"L:\huggingface\hub\models--google--embeddinggemma-2\snapshots\914f7f89142e33e77833254d9c9b90c3cef7303b")
QUERY, DOC = "task: search result | query: ", "title: {} | text: {}"


def jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def qrels(raw):
    out = {}
    for line in (Path(raw) / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        q, c, s = line.split("\t")
        out.setdefault(q, {})[c] = int(s)
    return out


def cmd_tokenize(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    rel = qrels(a.raw)
    items = [("query", q["_id"], QUERY + q["text"]) for q in jsonl(a.raw / "queries.jsonl") if q["_id"] in rel]
    items += [("doc", d["_id"], DOC.format(d.get("title") or "none", d["text"])) for d in jsonl(a.raw / "corpus.jsonl")]
    docs = []
    for kind, key, text in items:
        ids = tok(text, truncation=True, max_length=a.max_len)["input_ids"]
        docs.append(dict(kind=kind, key=key, length=len(ids), ids=ids))
    docs.sort(key=lambda d: d["length"])
    a.out.write_text(json.dumps(dict(source="mteb/scifact test", max_len=a.max_len, docs=docs)), encoding="utf-8")
    n = sum(d["kind"] == "query" for d in docs)
    print(f"{a.out}: {n} queries, {len(docs) - n} documents, lengths {docs[0]['length']}..{docs[-1]['length']}")


def cmd_hf(a):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(a.threads)
    docs = json.loads(a.docs.read_text(encoding="utf-8"))["docs"]
    model = AutoModel.from_pretrained(SNAPSHOT, dtype=torch.float32, attn_implementation="eager").eval()
    if a.quant != "none":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels" / "embedgemma"))
        import w4
        n = 0
        for name, mod in model.named_modules():
            if not (isinstance(mod, torch.nn.Linear) and name.startswith("language_model.")):
                continue
            w = mod.weight.detach().numpy().T.astype(np.float32)   # [K, N], as w4 expects
            q, sc = w4.quantize(w, 64 if a.quant == "w8a16" else w.shape[0], 8, 20)
            mod.weight.data = torch.from_numpy(np.ascontiguousarray(w4.dequant(q, sc).T))

            def pre(m_, args, mode=a.quant):
                x = args[0]
                flat = x.reshape(-1, x.shape[-1]).numpy().astype(np.float32)
                if mode == "w8a16":
                    y = w4.bf16(flat)
                else:
                    qa, sa = w4.quantize_rows(flat)
                    y = qa.astype(np.float32) * sa[:, None]
                return (torch.from_numpy(y.reshape(x.shape)),) + tuple(args[1:])
            mod.register_forward_pre_hook(pre)
            n += 1
        print(f"{a.quant}: {n} text Linear modules quantized", flush=True)
    out = []
    with torch.inference_mode():
        for i, d in enumerate(docs):
            ids = torch.tensor([d["ids"]])
            h = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0].double()
            v = h.mean(0)
            out.append((v / v.norm()).numpy())
            if i % 500 == 0:
                print(f"hf {i}/{len(docs)}", flush=True)
    np.savez(a.out, embeddings=np.stack(out))
    print(a.out)


def ranking(docs, emb, k=10):
    qi = [i for i, d in enumerate(docs) if d["kind"] == "query"]
    di = [i for i, d in enumerate(docs) if d["kind"] == "doc"]
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    s = e[qi] @ e[di].T
    top = np.argsort(-s, axis=1, kind="stable")[:, :k]
    keys = [docs[i]["key"] for i in di]
    return {docs[q]["key"]: [keys[j] for j in row] for q, row in zip(qi, top)}


def ndcg10(rank, rel):
    vals = []
    for q, got in rank.items():
        r = rel[q]
        dcg = sum(r.get(c, 0) / math.log2(i + 2) for i, c in enumerate(got))
        ideal = sorted(r.values(), reverse=True)[:10]
        vals.append(dcg / sum(v / math.log2(i + 2) for i, v in enumerate(ideal)))
    return 100 * float(np.mean(vals))


def cmd_score(a):
    docs = json.loads(a.docs.read_text(encoding="utf-8"))["docs"]
    rel = qrels(a.raw)
    rank = ranking(docs, np.load(a.emb)["embeddings"])
    res = dict(emb=str(a.emb), queries=len(rank), ndcg10=ndcg10(rank, rel))
    if a.ref:
        ref = ranking(docs, np.load(a.ref)["embeddings"])
        res["ref"] = str(a.ref)
        res["ref_ndcg10"] = ndcg10(ref, rel)
        res["ndcg10_delta"] = res["ndcg10"] - res["ref_ndcg10"]
        res["top10_overlap"] = float(np.mean([len(set(rank[q]) & set(ref[q])) / 10 for q in rank]))
        res["passed"] = bool(res["ndcg10_delta"] >= -0.5 and res["top10_overlap"] >= 0.9)
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("--raw", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--max-len", type=int, default=512)
    h = sub.add_parser("hf")
    h.add_argument("--docs", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--threads", type=int, default=8)
    h.add_argument("--quant", choices=["none", "w8a16", "w8a8"], default="none")
    s = sub.add_parser("score")
    s.add_argument("--docs", type=Path, required=True)
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--emb", type=Path, required=True)
    s.add_argument("--ref", type=Path)
    s.add_argument("--out", type=Path)
    a = ap.parse_args()
    return {"tokenize": cmd_tokenize, "hf": cmd_hf, "score": cmd_score}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
