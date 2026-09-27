#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Laya's decisions and calibration in the precisions an NPU build would use, against fp32 on the CPU.

Step 2 of the Laya study; step 1 (tools/laya_baseline.py) timed it on the CPU and the 780M. An NPU build would run
Laya's weight products in int8 or in bf16 (the NPU's bf16 GEMM takes bf16 operands and accumulates in fp32,
docs/SILICON.md). This asks, before anything is built for the NPU, whether those precisions keep Laya's answers on
the public tasks its model card reports, with the items built the way the card's runs built them:

    python tools/laya_accuracy.py rne --out-dir D                 ORT's Cast fp32->bf16->fp32 against numpy's
                                                                  round-to-nearest-even, bit for bit
    python tools/laya_accuracy.py bf16copy --model M --out O      the bf16 arm: every constant 2-D weight of a
                                                                  MatMul/Gemm rounded to bf16, and a Cast pair
                                                                  rounding that product's activation input
    python tools/laya_accuracy.py items --model-dir C --out F     the labelled items and the reference ticket
    python tools/laya_accuracy.py golden --model-dir C --model M  receptron's reference ticket, to 4 decimals
    python tools/laya_accuracy.py score --model M --items F --out O    one call per item, batch 1, raw logits
    python tools/laya_accuracy.py report --model-dir C --items F --ref R [--arm A ...] [--prereg P]

Every call runs through tools/laya_baseline.py's ``session`` (CPU EP, 8 intra-op threads, sequential, ORT's default
graph optimisation), so the arms differ only in the model file. Batch 1 throughout: ORT's dynamic int8 takes its
activation scale over the whole input tensor, and padding would change it. ``score`` holds a per-run keep-awake
(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) and releases it at exit.

The int8 arm is step 1's copy: ORT's dynamic scheme, per-channel int8 weights and a per-call activation scale. A
static scheme would need calibration data, so it is a proxy for an NPU build whose host computes the scale, not for
a fully static one. The bf16 arm is an emulation (operands rounded, fp32 accumulate, fp32 out), not the NPU kernel.
"""
import argparse
import ctypes
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_baseline as lb  # noqa: E402

CLS, SEP, PAD, MASK = 50281, 50282, 50283, 50284
MASK_TOK = "[MASK]"
QTYPES = {"choice": 0, "score": 1, "noul": 2}
KMAX = 77            # Banking77 asks all 77 intents at once
PREFIX = "bf16emu_"  # name prefix of the Cast nodes bf16copy inserts

# The card's figures for the English checkpoint, with the run each came from (published results JSONs):
# accuracy, ECE and mean confidence at the shipped temperatures; at T=1 where the run published it.
PUBLISHED = {
    "ag_news":    ("CPU fp32, first 400 test rows", 0.950, 0.0316, 0.9220, None, None),
    "emotion":    ("CPU fp32, first 400 test rows", 0.595, 0.3063, 0.9013, None, None),
    "banking77":  ("CPU fp32, first 400 test rows", 0.425, 0.5397, 0.9616, None, None),
    "massive_en": ("T4 fp16, first 300 test rows", 0.7833, 0.2143, 0.9942, 0.1263, 0.9067),
    "xnli_en":    ("T4 fp16, first 300 test rows", 0.8600, 0.0685, 0.9208, 0.1174, 0.9737),
    "sst5":       ("T4 fp16, first 600 test rows", 0.3717, 0.2478, 0.6194, 0.3006, 0.6723),
}
# receptron/laya test/test_model.ts @ fbe9ed68: the Python reference's output on one support ticket.
FIXTURE_STATE = {"subject": "Refund not received",
                 "body": "I cancelled my subscription two weeks ago and I still have not received my refund. This is "
                         "the third time I am writing. If this is not resolved I will dispute the charge with my bank."}
FIXTURE_QS = {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "payments, refunds, invoices", "support": "product help and bugs",
                                "sales": "new purchases and upgrades"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    "churn_risk": {"type": "noul", "instructions": "Is the customer likely to cancel or dispute?"},
}
FIXTURE_EXPECT = {"input_tokens": 267, "department": [0.9415, 0.031, 0.0275], "department_confidence": 0.7603,
                  "urgency_score": 1.3886, "urgency": [0.1752, 0.2947, 0.4962, 0.0338], "churn_risk": 0.0988}


# ---------------------------------------------------------------- input and calibration, as Laya's reference builds them
def tokenizer(model_dir: Path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model_dir / "tokenizer" / "tokenizer.json"))
    for t, i in (("[CLS]", CLS), ("[SEP]", SEP), ("[PAD]", PAD), ("[MASK]", MASK)):
        if tok.token_to_id(t) != i:
            raise SystemExit(f"tokenizer: {t} is {tok.token_to_id(t)}, not {i}")
    return lambda s: tok.encode(s, add_special_tokens=False).ids


def to_internal(qdef: dict) -> dict:
    """rl_agent_api's question conversion: a choice given as a list becomes {c: None}."""
    crit = qdef.get("criteria")
    if qdef["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": qdef["type"], "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


def render_options(q: dict) -> list[str]:
    """rl_common.py render_options (convaiinnovations/laya, L35-44)."""
    if q["t"] == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in q["crit"].items()]
    if q["t"] == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(q["crit"])]
    crit = q["crit"] or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(enc, state, q: dict, max_len: int, hml: int) -> tuple[list[int], list[int]]:
    """rl_common.py build_sequence (L47-77): [CLS] question [SEP] ([MASK] option)* [SEP] state [SEP], the state
    truncated on the right, each option capped at 48 tokens and cut further when the options overflow the head."""
    ins = str(q["ins"]).replace(MASK_TOK, " ")
    head = enc("%s question: %s" % (q["t"], ins))
    opt_ids = [[MASK] + enc(" " + o.replace(MASK_TOK, " "))[:48] for o in render_options(q)]
    budget = hml - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (hml - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = hml - sum(len(o) for o in opt_ids)
    head = head[:max(8, budget)]
    ids, markers = [CLS] + head + [SEP], []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(SEP)
    room = max(0, max_len - len(ids) - 1)
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    ids = ids + enc(st.replace(MASK_TOK, " "))[:room] + [SEP]
    return ids[:max_len], [m for m in markers if m < max_len]


def temperature(cfg: dict, qt: int, k: int) -> float:
    """rl_agent_api's lookup: the option-count bucket first, the question type's temperature otherwise."""
    name = {0: "choice", 1: "score", 2: "noul"}[qt]
    bucket = "%s:%s" % (name, "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")
    return float(cfg["temperature_by_options"].get(bucket, cfg["temperature"][qt]))


def softmax_t(z, t: float = 1.0) -> np.ndarray:
    """bench_local.py softmax_t: float64, the temperature floored at 1e-3."""
    z = np.asarray(z, np.float64) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def ece(conf, correct, bins: int = 15) -> float:
    """bench_local.py ece_score: 15 equal-width bins, the first closed at 0."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - correct[s].mean())
    return float(e)


def feed(ids: list[int], markers: list[int], qt: int) -> dict:
    return {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64),
            "marker_pos": np.array([markers], np.int64), "marker_mask": np.ones((1, len(markers)), bool),
            "qtype": np.array([qt], np.int64)}


def canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def model_shas(model: Path) -> dict:
    return {p.name: lb.sha256(p) for p in sorted(model.parent.glob(model.name + "*"))}


# ---------------------------------------------------------------- rne
def bf16_rne(x: np.ndarray) -> np.ndarray:
    """fp32 -> nearest bf16, ties to even, back to fp32 (finite inputs; a value past bf16's largest rounds to inf)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def cmd_rne(a) -> int:
    """A Cast pair on an input and the same pair feeding a MatMul by the identity, run through the same session as
    the arms. If ORT truncated, rounded ties away from zero, or removed the pair as a no-op, the bits would differ."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(7)
    parts = [
        (rng.standard_normal(20000) * 10.0 ** rng.uniform(-30, 30, 20000)).astype(np.float32),   # wide range
        rng.standard_normal(20000).astype(np.float16).astype(np.float32),                          # Laya's weights are fp16
        (rng.integers(0x00800000, 0x7F000000, 20000, dtype=np.uint64) & ~np.uint64(0xFFFF) | 0x8000)
        .astype(np.uint32).view(np.float32),                                                       # exact ties, both parities
        np.array([0.0, -0.0, 1.0, -1.0, 1e-40, -1e-40, 3.0e38, -3.0e38], np.float32),               # zeros, subnormals, big
    ]
    x = np.concatenate(parts)
    x = np.concatenate([x, -x])
    n = 256
    rows = len(x) // n
    x = x[:rows * n].reshape(rows, n)
    xm = np.clip(np.abs(x), 1e-30, 1e30) * np.sign(x)   # MatMul part: normal range, no inf products
    g = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["xb"], to=TensorProto.BFLOAT16),
         helper.make_node("Cast", ["xb"], ["y"], to=TensorProto.FLOAT),
         helper.make_node("Cast", ["xm"], ["mb"], to=TensorProto.BFLOAT16),
         helper.make_node("Cast", ["mb"], ["mf"], to=TensorProto.FLOAT),
         helper.make_node("MatMul", ["mf", "eye"], ["ym"])],
        "rne", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, n]),
                helper.make_tensor_value_info("xm", TensorProto.FLOAT, [rows, n])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, n]),
         helper.make_tensor_value_info("ym", TensorProto.FLOAT, [rows, n])],
        [numpy_helper.from_array(np.eye(n, dtype=np.float32), "eye")])
    path = Path(a.out_dir) / "rne_probe.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), str(path))
    try:
        y, ym = lb.session(path, "cpu", a.threads).run(None, {"x": x, "xm": xm})
    finally:
        path.unlink(missing_ok=True)
    ref, refm = bf16_rne(x), bf16_rne(xm)
    bad = int((y.view(np.uint32) != ref.view(np.uint32)).sum())
    badm = int((ym.view(np.uint32) != refm.view(np.uint32)).sum())
    ties = parts[2].size
    trunc = int((y.view(np.uint32) != (x.view(np.uint32) & np.uint32(0xFFFF0000))).sum())
    import onnxruntime as ort
    print(f"RNE onnxruntime {ort.__version__}: Cast pair on {x.size} values ({2 * ties} exact ties, both signs): "
          f"{bad} differ from numpy round-to-nearest-even; the same pair into a MatMul by the identity: {badm} differ; "
          f"{trunc} differ from truncation, so the pair is not a no-op and does not truncate")
    print(f"RNE {'PASS' if bad == 0 and badm == 0 and trunc > 0 else 'FAIL'}")
    return 0 if bad == 0 and badm == 0 and trunc > 0 else 1


# ---------------------------------------------------------------- bf16copy
def weight_products(m) -> list[int]:
    """Indices of the products tools/laya_baseline.py macs_per_call counts as weight MatMuls in the fp32 graph:
    MatMul or Gemm whose B is a constant 2-D initializer."""
    init = {i.name: i for i in m.graph.initializer}
    return [j for j, n in enumerate(m.graph.node)
            if n.op_type in ("MatMul", "Gemm") and n.input[1] in init and len(init[n.input[1]].dims) == 2]


def cmd_bf16copy(a) -> int:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    src, out = Path(a.model), Path(a.out)
    for p in (out, out.with_name(out.name + ".data")):
        if p.exists():
            raise SystemExit(f"{p.name} exists; this tool never writes over a model. Delete it by hand to re-run.")
    t0 = time.perf_counter()
    m = onnx.load(str(src))
    want = set(weight_products(m))
    sel = [m.graph.node[j] for j in sorted(want)]
    init = {i.name: i for i in m.graph.initializer}
    rounded, changed, total = set(), 0, 0
    for n in sel:
        w = n.input[1]
        if w in rounded:
            continue
        arr = numpy_helper.to_array(init[w])
        if arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise SystemExit(f"{w}: {arr.dtype}, finite {bool(np.isfinite(arr).all())}")
        r = bf16_rne(arr)
        changed += int((r != arr).sum())
        total += arr.size
        init[w].CopyFrom(numpy_helper.from_array(r, w))
        rounded.add(w)
    kinds = {m.graph.node[j].op_type for j in want}
    n_gemm = sum(m.graph.node[j].op_type == "Gemm" for j in want)
    nodes, k = [], 0
    for j, n0 in enumerate(m.graph.node):
        n = onnx.NodeProto()
        n.CopyFrom(n0)
        if j in want:
            nodes.append(helper.make_node("Cast", [n.input[0]], [f"{PREFIX}{k}_b"], name=f"{PREFIX}{k}_to_bf16",
                                          to=TensorProto.BFLOAT16))
            nodes.append(helper.make_node("Cast", [f"{PREFIX}{k}_b"], [f"{PREFIX}{k}_f"], name=f"{PREFIX}{k}_to_f32",
                                          to=TensorProto.FLOAT))
            n.input[0] = f"{PREFIX}{k}_f"
            k += 1
        nodes.append(n)
    sel_len = len(sel)
    del sel
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=out.name + ".data", size_threshold=1024)
    del m
    # checks: the same products as step 1 counts, every rounded weight bf16-exact on disk, every Cast pair executed
    mac_src, mac_out = lb.macs_per_call(src, 512, 1)[0], lb.macs_per_call(out, 512, 1)[0]
    chk = onnx.load(str(out), load_external_data=False)
    ci = {i.name: i for i in chk.graph.initializer}
    low = sum(int((numpy_helper.to_array(ci[w], str(out.parent)).view(np.uint32) & 0xFFFF).any()) for w in rounded)
    sess = lb.session(out, "cpu", a.threads, profile=str(out.parent / "bf16emu_profile"))
    enc = tokenizer(Path(a.model_dir))
    ids, mk = build_sequence(enc, FIXTURE_STATE, to_internal(FIXTURE_QS["department"]), 512, 192)
    sess.run(None, feed(ids, mk, 0))
    prof = Path(sess.end_profiling())
    ran = {e["name"].removesuffix("_kernel_time") for e in json.loads(prof.read_text(encoding="utf-8"))
           if e.get("cat") == "Node" and e["name"].startswith(PREFIX) and e["name"].endswith("_kernel_time")}
    prof.unlink(missing_ok=True)
    ok = k == sel_len and mac_src == mac_out and low == 0 and len(ran) == 2 * k
    print(f"BF16COPY {out.name} from {src.name} in {time.perf_counter() - t0:.0f} s: {k} {'/'.join(sorted(kinds))} with "
          f"constant 2-D weights ({n_gemm} Gemm), {len(rounded)} weights, {total} values rounded "
          f"to bf16 (nearest, ties to even), {changed} of them changed; a Cast pair (fp32->bf16->fp32) on each "
          f"product's activation input; attention products, norms, embeddings and the rest stay fp32")
    print(f"BF16COPY checks: weight MACs per 512-token call {mac_out / 1e9:.3f} G, fp32 graph {mac_src / 1e9:.3f} G "
          f"(tools/laya_baseline.py); rounded weights with low bits set on disk {low}; inserted Cast nodes executed "
          f"in one profiled call {len(ran)} of {2 * k}; {'PASS' if ok else 'FAIL'}")
    for name, sha in model_shas(out).items():
        print(f"BF16COPY file {name} sha256 {sha}")
    if not ok:
        raise SystemExit("bf16 copy failed its checks")
    return 0


# ---------------------------------------------------------------- items
def task_rows(name: str, cache: Path):
    """(dataset id, config, revision, [(state, questions, gold index)]) for one task, as the card's runs built it:
    bench_apps.py and build_benchmark_nb.py (NandhaKishorM/laya @ ee760389, the commit that published the results)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    spec = {"ag_news": ("fancyzhx/ag_news", None, 400), "emotion": ("dair-ai/emotion", "split", 400),
            "banking77": ("mteb/banking77", None, 400), "massive_en": ("mteb/amazon_massive_intent", "en", 300),
            "xnli_en": ("facebook/xnli", "en", 300), "sst5": ("SetFit/sst5", None, 600)}[name]
    rid, conf, n = spec
    rev = HfApi().dataset_info(rid).sha
    d = load_dataset(rid, conf, split="test", revision=rev, cache_dir=str(cache))
    rows, first = [], [d[i] for i in range(n)]
    if name == "ag_news":
        crit = {"world": "world news and international politics", "sports": "sports",
                "business": "business and economy", "sci_tech": "science and technology"}
        rows = [({"article": r["text"]}, {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                                                    "criteria": dict(crit)}}, int(r["label"])) for r in first]
    elif name == "emotion":
        names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        rows = [({"text": r["text"]}, {"emotion": {"type": "choice", "instructions":
                                                   "Which emotion is most strongly expressed in `text`?",
                                                   "criteria": {x: None for x in names}}}, int(r["label"]))
                for r in first]
    elif name == "banking77":
        keys = [x.replace("_", " ") for x in sorted(set(d["label_text"]))]
        rows = [({"message": r["text"]}, {"intent": {"type": "choice", "instructions":
                                                     "Which banking intent does `message` express?",
                                                     "criteria": {x: None for x in keys}}},
                 keys.index(r["label_text"].replace("_", " "))) for r in first]
    elif name == "massive_en":
        labels, rng = sorted(set(d["label_text"])), random.Random(13)
        for r in first:
            keys = [r["label_text"]] + rng.sample([x for x in labels if x != r["label_text"]], min(19, len(labels) - 1))
            rng.shuffle(keys)
            rows.append(({"utterance": r["text"]},
                         {"label": {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
                                    "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}}},
                         keys.index(r["label_text"])))
    elif name == "xnli_en":
        crit = {"entailment": "the premise implies the hypothesis is true",
                "neutral": "the premise neither implies nor contradicts the hypothesis",
                "contradiction": "the premise implies the hypothesis is false"}
        rows = [({"premise": r["premise"], "hypothesis": r["hypothesis"]},
                 {"relation": {"type": "choice", "instructions":
                               "What is the relationship between `premise` and `hypothesis`?",
                               "criteria": dict(crit)}}, int(r["label"])) for r in first]
    elif name == "sst5":
        crit = ["very negative", "negative", "neutral", "positive", "very positive"]
        rows = [({"text": r["text"]}, {"sentiment": {"type": "score", "instructions":
                                                     "How positive is the sentiment of `text`?", "criteria": crit}},
                 int(r["label"])) for r in first]
    names = getattr(d.features.get("label"), "names", None)
    if names is None and "label_text" in d.column_names and isinstance(first[0]["label"], int):   # int -> text pairs
        pairs = sorted({(int(r["label"]), r["label_text"]) for r in first})
        names = [t for _, t in pairs] if [i for i, _ in pairs] == list(range(len(pairs))) else pairs
    return rid, conf, rev, rows, names, len(d)


def cmd_items(a) -> int:
    mdir, out = Path(a.model_dir), Path(a.out)
    cfg = json.loads((mdir / "laya_config.json").read_text(encoding="utf-8"))
    enc = tokenizer(mdir)
    ml, hml = cfg["max_len"], cfg["head_max_len"]
    items = []
    for qid, qdef in FIXTURE_QS.items():
        q = to_internal(qdef)
        ids, mk = build_sequence(enc, FIXTURE_STATE, q, ml, hml)
        items.append({"task": "fixture", "i": len(items), "qid": qid, "qtype": QTYPES[q["t"]], "k": len(mk),
                      "gold": -1, "ids": ids, "markers": mk})
    print(f"ITEMS fixture: 3 questions (receptron test_model.ts @ fbe9ed68), {sum(len(x['ids']) for x in items)} "
          f"tokens in all; sha256 {canon([[x['ids'], x['markers'], x['qtype'], x['gold']] for x in items])}")
    cache = out.parent / "hf-datasets"
    for name in PUBLISHED:
        rid, conf, rev, rows, names, n_split = task_rows(name, cache)
        start, dropped = len(items), 0
        for i, (state, qs, gold) in enumerate(rows):
            (qid, qdef), = qs.items()
            q = to_internal(qdef)
            ids, mk = build_sequence(enc, state, q, ml, hml)
            if len(mk) != len(render_options(q)):   # the card's runs drop a question whose options do not all fit
                dropped += 1
                continue
            items.append({"task": name, "i": i, "qid": qid, "qtype": QTYPES[q["t"]], "k": len(mk), "gold": gold,
                          "ids": ids, "markers": mk})
        mine = items[start:]
        opts = render_options(to_internal(next(iter(rows[0][1].values()))))
        print(f"ITEMS {name}: {rid}{'/' + conf if conf else ''} @ {rev[:12]}, test split of {n_split}, first "
              f"{len(rows)} rows; {len(mine)} items, {dropped} dropped; {mine[0]['k']} options; tokens mean "
              f"{np.mean([len(x['ids']) for x in mine]):.0f}, max {max(len(x['ids']) for x in mine)}; gold spread "
              f"{np.bincount([x['gold'] for x in mine]).tolist() if mine[0]['k'] <= 6 else '-'}; sha256 "
              f"{canon([[x['ids'], x['markers'], x['qtype'], x['gold']] for x in mine])}")
        if names and mine[0]['k'] <= 6:
            print(f"ITEMS {name}: dataset label names {names} against options {opts[:6]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for x in items:
            f.write(json.dumps(x, separators=(",", ":")) + "\n")
    print(f"ITEMS {out.name}: {len(items)} items ({sum(x['gold'] >= 0 for x in items)} labelled), "
          f"sha256 {lb.sha256(out)}")
    return 0


def read_items(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- golden
def cmd_golden(a) -> int:
    mdir, model = Path(a.model_dir), Path(a.model)
    cfg = json.loads((mdir / "laya_config.json").read_text(encoding="utf-8"))
    enc = tokenizer(mdir)
    sess = lb.session(model, "cpu", a.threads)
    got, tokens, same = {}, 0, True
    for qid, qdef in FIXTURE_QS.items():
        q = to_internal(qdef)
        ids, mk = build_sequence(enc, FIXTURE_STATE, q, cfg["max_len"], cfg["head_max_len"])
        tokens += len(ids)
        z1 = sess.run(None, feed(ids, mk, QTYPES[q["t"]]))[0][0, :len(mk)]
        z2 = sess.run(None, feed(ids, mk, QTYPES[q["t"]]))[0][0, :len(mk)]
        same &= bool(np.array_equal(z1, z2))
        got[qid] = softmax_t(z1, temperature(cfg, QTYPES[q["t"]], len(mk)))
    p = got["department"]
    conf = 1.0 - float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()) / math.log(len(p))
    have = {"input_tokens": tokens, "department": [round(float(x), 4) for x in p],
            "department_confidence": round(conf, 4),
            "urgency_score": round(float((np.arange(4) * got["urgency"]).sum()), 4),
            "urgency": [round(float(x), 4) for x in got["urgency"]], "churn_risk": round(float(got["churn_risk"][1]), 4)}
    raw = np.concatenate([p, got["urgency"], [got["churn_risk"][1], conf, (np.arange(4) * got["urgency"]).sum()]])
    exp = np.array(FIXTURE_EXPECT["department"] + FIXTURE_EXPECT["urgency"] +
                   [FIXTURE_EXPECT["churn_risk"], FIXTURE_EXPECT["department_confidence"],
                    FIXTURE_EXPECT["urgency_score"]])
    ok = have == FIXTURE_EXPECT
    print(f"GOLDEN {model.name}: {json.dumps(have)}")
    print(f"GOLDEN expected (receptron test_model.ts @ fbe9ed68): {json.dumps(FIXTURE_EXPECT)}")
    print(f"GOLDEN largest unrounded difference {float(np.abs(raw - exp).max()):.2e}; a second call bit-identical "
          f"{same}; {'PASS' if ok else 'FAIL'} (all equal after rounding to 4 decimals, as the reference rounds)")
    return 0 if ok else 1


# ---------------------------------------------------------------- score
def cmd_score(a) -> int:
    import onnxruntime as ort
    model, items_path, out = Path(a.model), Path(a.items), Path(a.out)
    if out.suffix != ".npz" or out.exists():
        raise SystemExit(f"{out.name}: must end in .npz and not exist; this tool never writes over a scored file")
    items = read_items(items_path)
    shas = model_shas(model)
    t0 = time.perf_counter()
    sess = lb.session(model, "cpu", a.threads)
    print(f"SCORE {model.name}: onnxruntime {ort.__version__}, providers {sess.get_providers()}, {a.threads} threads, "
          f"batch 1, session load {time.perf_counter() - t0:.1f} s; items {items_path.name} sha256 "
          f"{lb.sha256(items_path)}; " + "; ".join(f"{k} sha256 {v}" for k, v in shas.items()), flush=True)
    es = 0x80000000 | 0x00000001   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, this run only
    held = bool(ctypes.windll.kernel32.SetThreadExecutionState(es))
    print(f"KEEPAWAKE HOLD {held}", flush=True)
    logits = np.full((len(items), KMAX), np.nan, np.float32)
    ms = []
    try:
        for j, x in enumerate(items):
            t = time.perf_counter()
            z = sess.run(None, feed(x["ids"], x["markers"], x["qtype"]))[0]
            ms.append((time.perf_counter() - t) * 1000.0)
            logits[j, :x["k"]] = z[0, :x["k"]]
            if (j + 1) % 200 == 0 or j + 1 == len(items):
                print(f"[score] {j + 1}/{len(items)} items, {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("KEEPAWAKE RELEASE", flush=True)
    finite = bool(np.isfinite(np.concatenate([logits[j, :x["k"]] for j, x in enumerate(items)])).all())
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=logits, k=np.array([x["k"] for x in items]), items_sha256=lb.sha256(items_path),
             model=model.name, model_shas=json.dumps(shas))
    print(f"SCORE {model.name}: {len(items)} items in {sum(ms) / 1000:.0f} s of calls (median {np.median(ms):.0f} ms "
          f"per item; batch 1 on shared hardware, not a timing measurement); logits finite {finite}; {out.name} "
          f"sha256 {lb.sha256(out)}", flush=True)
    return 0 if finite else 1


# ---------------------------------------------------------------- report
def probs(items, z, cfg, calibrated: bool) -> list[np.ndarray]:
    return [softmax_t(z[j, :x["k"]], temperature(cfg, x["qtype"], x["k"]) if calibrated else 1.0)
            for j, x in enumerate(items)]


def task_metrics(items, ps, idx) -> dict:
    g = np.array([items[j]["gold"] for j in idx])
    pred = np.array([int(np.argmax(ps[j])) for j in idx])
    conf = np.array([float(np.max(ps[j])) for j in idx])
    corr = (pred == g).astype(float)
    return {"n": len(idx), "acc": float(corr.mean()), "ece": ece(conf, corr), "conf": float(conf.mean()),
            "nll": float(np.mean([-math.log(max(float(ps[j][items[j]["gold"]]), 1e-12)) for j in idx]))}


def load_scored(path: Path, items_sha: str):
    d = np.load(path)
    if str(d["items_sha256"]) != items_sha:
        raise SystemExit(f"{path.name} was scored on other items")
    return d["logits"], str(d["model"]), json.loads(str(d["model_shas"])), lb.sha256(path)


def cmd_report(a) -> int:
    mdir, items_path = Path(a.model_dir), Path(a.items)
    cfg = json.loads((mdir / "laya_config.json").read_text(encoding="utf-8"))
    items, isha = read_items(items_path), lb.sha256(items_path)
    zr, rname, rshas, rfile = load_scored(Path(a.ref), isha)
    tasks = [t for t in PUBLISHED if any(x["task"] == t for x in items)]
    by = {t: [j for j, x in enumerate(items) if x["task"] == t] for t in tasks}
    lab = [j for t in tasks for j in by[t]]
    fx = [j for j, x in enumerate(items) if x["task"] == "fixture"]
    r1, rs = probs(items, zr, cfg, False), probs(items, zr, cfg, True)
    print(f"REPORT items {items_path.name} sha256 {isha}; reference {rname} ({Path(a.ref).name} sha256 {rfile})")
    for t in tasks:
        m1, ms = task_metrics(items, r1, by[t]), task_metrics(items, rs, by[t])
        run, acc, e, c, e1, c1 = PUBLISHED[t]
        pub1 = f"; published at T=1: ECE {e1:.4f}, confidence {c1:.4f}" if e1 is not None else ""
        print(f"REF {t}: n {ms['n']}, accuracy {ms['acc']:.4f} (published {acc:.4f}, {run}); shipped temperature: "
              f"ECE {ms['ece']:.4f}, confidence {ms['conf']:.4f} (published {e:.4f}, {c:.4f}); T=1: ECE "
              f"{m1['ece']:.4f}, confidence {m1['conf']:.4f}{pub1}")
    pooled1 = task_metrics(items, r1, lab)
    print(f"REF pooled {len(lab)} labelled items: accuracy {pooled1['acc']:.4f}, T=1 ECE {pooled1['ece']:.4f}")
    gap = [float(np.sort(rs[j])[-1] - np.sort(rs[j])[-2]) for j in lab]
    gap1 = [float(np.sort(r1[j])[-1] - np.sort(r1[j])[-2]) for j in lab]
    print(f"REF near ties: top-two gap under 0.048 (twice the authors' largest bf16 choice shift, 0.024) on "
          f"{sum(x < 0.048 for x in gap)} items at the shipped temperatures and {sum(x < 0.048 for x in gap1)} at T=1, "
          f"of {len(lab)}")
    pre = json.loads(Path(a.prereg).read_text(encoding="utf-8")) if a.prereg else None
    if pre:
        if pre["items"]["file_sha256"] != isha or pre["reference"]["logits_sha256"] != rfile:
            raise SystemExit("the items or the reference logits differ from the pre-registered ones")
        print(f"REPORT pre-registration {Path(a.prereg).name} sha256 {lb.sha256(Path(a.prereg))}")
    for arm in a.arm or []:
        za, aname, ashas, afile = load_scored(Path(arm), isha)
        a1, as_ = probs(items, za, cfg, False), probs(items, za, cfg, True)
        agree = {t: sum(int(np.argmax(a1[j]) == np.argmax(r1[j])) for j in by[t]) for t in tasks}
        rows = {}
        for t in tasks:
            ra, aa = task_metrics(items, r1, by[t]), task_metrics(items, a1, by[t])
            rsm, asm = task_metrics(items, rs, by[t]), task_metrics(items, as_, by[t])
            d1 = np.concatenate([np.abs(a1[j] - r1[j]) for j in by[t]])
            ds = np.concatenate([np.abs(as_[j] - rs[j]) for j in by[t]])
            rows[t] = {"agree": agree[t] / len(by[t]), "dacc": aa["acc"] - ra["acc"], "dece1": aa["ece"] - ra["ece"],
                       "deces": asm["ece"] - rsm["ece"]}
            extra = ""
            if t == "sst5":
                sc = [abs(float((np.arange(5) * (as_[j] - rs[j])).sum())) for j in by[t]]
                extra = f"; score answer shift max {max(sc):.4f}, mean {np.mean(sc):.4f}"
            print(f"ARM {aname} {t}: same decision {agree[t]}/{len(by[t])}; accuracy {aa['acc']:.4f} (fp32 "
                  f"{ra['acc']:.4f}, {100 * rows[t]['dacc']:+.2f} pt); T=1 ECE {aa['ece']:.4f} (fp32 {ra['ece']:.4f}, "
                  f"{rows[t]['dece1']:+.4f}); shipped-temperature ECE {asm['ece']:.4f} ({rows[t]['deces']:+.4f}); "
                  f"|dp| at T=1 max {d1.max():.4f}, p99 {np.quantile(d1, 0.99):.4f}, mean {d1.mean():.5f}; at the "
                  f"shipped temperature max {ds.max():.4f}, p99 {np.quantile(ds, 0.99):.4f}{extra}")
        pa, pr = task_metrics(items, a1, lab), task_metrics(items, r1, lab)
        agree_all = sum(agree.values()) / len(lab)
        print(f"ARM {aname} pooled {len(lab)}: same decision {sum(agree.values())}/{len(lab)} ({100 * agree_all:.2f}%); "
              f"accuracy {pa['acc']:.4f} (fp32 {pr['acc']:.4f}, {100 * (pa['acc'] - pr['acc']):+.2f} pt); T=1 ECE "
              f"{pa['ece']:.4f} (fp32 {pr['ece']:.4f}, {pa['ece'] - pr['ece']:+.4f}); {Path(arm).name} sha256 {afile}")
        for j in fx:
            x = items[j]
            print(f"ARM {aname} fixture {x['qid']}: shipped-temperature probabilities "
                  f"{[round(float(v), 4) for v in as_[j]]} (fp32 {[round(float(v), 4) for v in rs[j]]}), max |dp| "
                  f"{float(np.abs(as_[j] - rs[j]).max()):.4f}")
        if pre:
            arm_pre = next((k for k, v in pre["arms"].items() if v["files"] == ashas), None)
            c, eps = pre["criteria"], 1e-9   # a bound can fall exactly on an item count (6 of 400 is 1.5 pt)
            checks = [("C0 the arm is a pre-registered file", arm_pre is not None),
                      (f"C1 pooled same decision >= {c['agree_pooled_min']}", agree_all >= c["agree_pooled_min"] - eps),
                      (f"C2 pooled accuracy change >= {c['dacc_pooled_min']}",
                       pa["acc"] - pr["acc"] >= c["dacc_pooled_min"] - eps),
                      (f"C3 each task's accuracy change >= {c['dacc_task_min']}",
                       all(rows[t]["dacc"] >= c["dacc_task_min"] - eps for t in tasks)),
                      (f"C4 pooled T=1 ECE change <= {c['dece1_pooled_max']}",
                       pa["ece"] - pr["ece"] <= c["dece1_pooled_max"] + eps),
                      (f"C5 each task's T=1 ECE change <= {c['dece1_task_max']}",
                       all(rows[t]["dece1"] <= c["dece1_task_max"] + eps for t in tasks))]
            for text, ok in checks:
                print(f"CRITERION {aname} {text}: {'PASS' if ok else 'FAIL'}")
            v = all(ok for _, ok in checks)
            print(f"VERDICT {aname} ({arm_pre}): {'PASS' if v else 'FAIL'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("rne", "bf16copy", "items", "golden", "score", "report"))
    ap.add_argument("--model")
    ap.add_argument("--model-dir", help="the laya-onnx directory: laya_config.json and tokenizer/")
    ap.add_argument("--out")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--items")
    ap.add_argument("--ref", help="report: the fp32 reference's scored file")
    ap.add_argument("--arm", action="append", help="report: an arm's scored file")
    ap.add_argument("--prereg", help="report: the pre-registration JSON the verdict is taken against")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    return {"rne": cmd_rne, "bf16copy": cmd_bf16copy, "items": cmd_items, "golden": cmd_golden, "score": cmd_score,
            "report": cmd_report}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
