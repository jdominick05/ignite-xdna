"""HF transformers CPU time of EmbeddingGemma 2 with and without its weight products.

Runs the shared docs twice through the same model and dtype: once as is, once with every ``nn.Linear`` replaced by
a stub that returns zeros of the right shape. The difference is the weight products' CPU time; the stubbed run is
what everything else costs on the CPU (attention, norms, GELU, RoPE, residuals, embedding, pooling). A third pass
also stubs eager attention (scores, mask, softmax, value product) the same way, which isolates attention's share.
These ops' cost does not depend on their values, so zeros time them fairly; stubbed runs' embeddings are discarded.
Same docs, eager attention, ``--threads`` as the reference runs, one untimed warm-up document per length.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel

SNAPSHOT = r"L:\huggingface\hub\models--google--embeddinggemma-2\snapshots\914f7f89142e33e77833254d9c9b90c3cef7303b"


class Zero(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.n, self.dtype = lin.out_features, lin.weight.dtype

    def forward(self, x):
        return x.new_zeros(*x.shape[:-1], self.n)


def time_docs(model, docs):
    times, warmed = {}, set()
    with torch.inference_mode():
        for d in docs:
            ids = torch.tensor([d["ids"]])
            if d["length"] not in warmed:
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
                warmed.add(d["length"])
            t = time.perf_counter()
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
            times.setdefault(d["length"], []).append((time.perf_counter() - t) * 1000)
    return {L: statistics.median(v) for L, v in sorted(times.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(a.threads)
    docs = json.loads(a.docs.read_text(encoding="utf-8"))["docs"]
    model = AutoModel.from_pretrained(SNAPSHOT, dtype=getattr(torch, a.dtype), attn_implementation="eager").eval()
    full = time_docs(model, docs)
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, Zero(child))
                n += 1
    rest = time_docs(model, docs)
    import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as mod

    def no_attention(module, query, key, value, attention_mask, **kw):
        B, H, L, hd = query.shape
        return query.new_zeros(B, L, H, hd), None
    mod.eager_attention_forward = no_attention
    neither = time_docs(model, docs)
    rec = dict(dtype=a.dtype, threads=a.threads, torch=torch.__version__, linears_stubbed=n, docs=str(a.docs),
               per_length={str(L): dict(full_ms=full[L], without_linear_ms=rest[L], linear_ms=full[L] - rest[L],
                                        without_linear_or_attention_ms=neither[L], attention_ms=rest[L] - neither[L])
                           for L in full})
    for L in full:
        print(f"L={L:5d}: full {full[L]:8.1f} ms, Linear {full[L] - rest[L]:8.1f}, attention {rest[L] - neither[L]:8.1f}, "
              f"everything else {neither[L]:8.1f} ms")
    print(f"{n} nn.Linear stubbed")
    a.out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
