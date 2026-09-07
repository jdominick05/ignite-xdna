"""Measure the MobileViT heterogeneous "splice": cut CNN on NPU + attention on CPU.

    python tools/splice_wall_clock.py --fresh          # first run, or after a model change
    python tools/splice_wall_clock.py --iters 200

Run in resnet_env17 (Git Bash, not WSL). Needs the NPU, so laptop or Desktop 2.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a COST MODEL, not a functional pipeline, and the distinction is the whole
point of the script.

`mobilevit_cut_backbone_xint8.onnx` is [1,3,256,256] -> [1,1000]: a complete
classifier with the 9 transformer blocks *deleted*, not a backbone that hands
intermediate activations to an attention stage. There is no tensor for the CPU
half to consume, so no real dataflow splice can be built from it. What can be
measured honestly is the cost of both halves co-resident in one process:

    per frame:  sess.run(cut CNN on NPU)  then  9 attention blocks on CPU

The two halves are unconnected. The composite computes nothing valid -- the
quantized backbone scores 0% top-1 on its own (results/mobilevit/) and the
attention runs on random tensors of the right shape. Treat the output as "what
would this pipeline cost if it existed", never as a working model.

That is also exactly what the previously published 4.47 ms figure meant. It was a
hardcoded constant in tools/demo_attention.py with a back-solved handoff
residual; this script replaces it with a perf_counter measurement.

TIMING SPLIT
------------
Backbone-only, attention-only and the combined loop are all measured in the SAME
process and the SAME session, so the in-process residual is

    residual = combined - (backbone_only + attention_only)

rather than a number derived from constants. Baselines (full model FP32 on ORT
CPU and on PyTorch, stock quantized graph on NPU) are measured in the same run
where available, because NPU latency on this machine drifts between sessions.

Attention shapes are read off mobilevit_xxs_fp32.onnx rather than hardcoded:
MobileViT-XXS is B=4 (the unfold's patch dim), H=4 heads, 9 blocks total --
2 blocks at N=256/D=16, 4 at N=64/D=20, 3 at N=16/D=24.

WHICH CPU ATTENTION IMPLEMENTATION COUNTS
-----------------------------------------
It changes the verdict, so both are measured and the torch one is the number
that goes in the table. The same nine blocks cost ~12.8 ms in plain numpy and
~1.3 ms in torch (8 threads) on this machine -- 10x, from multithreaded batched
GEMM and a fused softmax. The thing the splice is being compared against is the
full model under ORT's own optimized CPU kernels, so pitting a naive numpy
attention against it would be rigging the comparison. numpy is kept as a
labelled contrast because "which CPU kernel you use decides whether the NPU
looks like a win" is itself worth having on record.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npu.paths import MODELS, MOBILEVIT_CUT_CACHE_KEY, MOBILEVIT_STOCK_CACHE_KEY
from npu.session import build_session, clear_cache

CUT_MODEL = MODELS / "mobilevit_cut_backbone_xint8.onnx"
STOCK_MODEL = MODELS / "mobilevit_xxs_xint8.onnx"
FP32_MODEL = MODELS / "mobilevit_xxs_fp32.onnx"


def attention_blocks_from_graph(path):
    """(B, H, N, D, count) per attention stage, read from the FP32 graph.

    One Softmax == one attention block. Its output is the score matrix
    [B, H, N, N]; the paired V matmul output [B, H, N, D] gives the head dim.
    """
    m = shape_inference.infer_shapes(onnx.load(str(path)))
    vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
          for v in list(m.graph.value_info) + list(m.graph.output)}
    scores = collections.Counter()
    for n in m.graph.node:
        if n.op_type == "Softmax":
            s = tuple(vi.get(n.output[0], []))
            if len(s) == 4:
                scores[s] += 1
    # head dim: a MatMul whose output is [B, H, N, D] with D != N
    dims = {}
    for n in m.graph.node:
        if n.op_type != "MatMul":
            continue
        s = tuple(vi.get(n.output[0], []))
        if len(s) == 4 and s[2] != s[3]:
            dims[(s[0], s[1], s[2])] = s[3]
    out = []
    for (b, h, n_tok, _), count in sorted(scores.items(), key=lambda x: -x[0][2]):
        d = dims.get((b, h, n_tok))
        if d is None:
            raise SystemExit(f"could not find head dim for B={b} H={h} N={n_tok}")
        out.append((b, h, n_tok, d, count))
    return out


def make_attention_inputs(stages, rng):
    """Random Q/K/V of the real shapes. Values are irrelevant to a cost model;
    shapes are not."""
    work = []
    for b, h, n_tok, d, count in stages:
        bh = b * h
        q = rng.standard_normal((bh, n_tok, d), dtype=np.float32)
        k = rng.standard_normal((bh, n_tok, d), dtype=np.float32)
        v = rng.standard_normal((bh, n_tok, d), dtype=np.float32)
        work.append((q, k, v, float(d) ** -0.5, count))
    return work


def run_attention_numpy(work):
    """The 9 blocks' arithmetic: scaled QK^T -> softmax -> PV, in numpy.

    Kept as the slow contrast, not as the reported number -- see the module
    docstring. numpy's batched matmul is effectively single-threaded here and
    its softmax is unfused, which costs ~10x against torch.
    """
    for q, k, v, scale, count in work:
        for _ in range(count):
            s = (q @ k.transpose(0, 2, 1)) * scale
            s -= s.max(axis=-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(axis=-1, keepdims=True)
            _ = s @ v


def make_torch_attention(work):
    """Same nine blocks under torch's multithreaded BLAS + fused softmax.

    Returns None when torch is absent. CLAUDE.md says not to install torch into
    resnet_env17; if it is not there, the numpy number is all this script can
    report and the log says so explicitly rather than silently downgrading.
    """
    try:
        import torch
    except ImportError:
        return None, None
    tw = [(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), scale, count)
          for q, k, v, scale, count in work]

    def run():
        with torch.no_grad():
            for q, k, v, scale, count in tw:
                for _ in range(count):
                    s = torch.bmm(q, k.transpose(1, 2)) * scale
                    _ = torch.bmm(torch.softmax(s, dim=-1), v)

    return run, f"torch {torch.__version__}, {torch.get_num_threads()} threads"


def timed(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min())}


def fmt(tag, st):
    return (f"  {tag:<44s} mean {st['mean']:7.2f} ms   median {st['median']:7.2f}   "
            f"p95 {st['p95']:7.2f}   min {st['min']:7.2f}")


def ep_report(cache_key):
    p = Path(cache_key) / "vitisai_ep_report.json"
    if not p.exists():
        return "  (no vitisai_ep_report.json)"
    r = json.loads(p.read_text())
    dev = {d["name"]: d["nodeNum"] for d in r.get("deviceStat", [])}
    sub = {s["device"]: s["count"] for s in r.get("subgraphStat", [])}
    return f"  {cache_key}: nodes {dev}   subgraphs {sub}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--fresh", action="store_true",
                    help="clear the compile caches first -- do this whenever the "
                         "model or xclbin changed; the cache is keyed by name only")
    ap.add_argument("--skip-stock", action="store_true",
                    help="skip the 49-subgraph stock-EP NPU baseline (it is slow)")
    ap.add_argument("--xclbin", default=None)
    args = ap.parse_args()

    for p in (CUT_MODEL, STOCK_MODEL, FP32_MODEL):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    stages = attention_blocks_from_graph(FP32_MODEL)
    total_blocks = sum(c for *_, c in stages)
    print("=== MobileViT-XXS heterogeneous splice: measured wall clock ===")
    print("COST MODEL, NOT A PIPELINE: the cut CNN is a complete [1,3,256,256]->[1,1000]")
    print("classifier with the transformer blocks deleted, so the two halves are")
    print("unconnected and the composite computes nothing valid. See the module docstring.\n")
    print(f"attention blocks read from {FP32_MODEL.name}: {total_blocks} total")
    for b, h, n_tok, d, count in stages:
        print(f"    B={b} H={h} N={n_tok:>3} D={d:>2}   x{count} blocks")
    print()

    if args.fresh:
        clear_cache(MOBILEVIT_CUT_CACHE_KEY)
        clear_cache(MOBILEVIT_STOCK_CACHE_KEY)
        print("cleared compile caches\n")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 256, 256), dtype=np.float32)
    work = make_attention_inputs(stages, rng)

    print("building NPU session for the cut CNN backbone ...")
    npu_sess = build_session(str(CUT_MODEL), "npu", MOBILEVIT_CUT_CACHE_KEY, args.xclbin)
    npu_in = npu_sess.get_inputs()[0].name
    cpu_sess = build_session(str(CUT_MODEL), "cpu", MOBILEVIT_CUT_CACHE_KEY, None)
    fp32_cpu = build_session(str(FP32_MODEL), "cpu", MOBILEVIT_CUT_CACHE_KEY, None)
    print()

    backbone_npu = lambda: npu_sess.run(None, {npu_in: x})
    backbone_cpu = lambda: cpu_sess.run(None, {npu_in: x})
    attn_torch, torch_desc = make_torch_attention(work)

    print("--- the two halves, measured separately, same process/session ---")
    st_bb = timed(backbone_npu, args.iters)
    st_np = timed(lambda: run_attention_numpy(work), args.iters)
    print(fmt("cut CNN backbone, NPU", st_bb))
    if attn_torch is not None:
        st_at = timed(attn_torch, args.iters)
        print(fmt(f"attention x{total_blocks} blocks, CPU ({torch_desc})", st_at))
        print(fmt(f"attention x{total_blocks} blocks, CPU (numpy, slow contrast)", st_np))
        print(f"  -> implementation alone is {st_np['mean'] / st_at['mean']:.1f}x. The torch "
              f"row is the one that counts:")
        print("     the splice is being compared against the full model under ORT's own")
        print("     optimized CPU kernels, so a naive CPU attention would rig it.")
        attn_fn, attn_tag = attn_torch, "torch"
    else:
        st_at, attn_fn, attn_tag = st_np, lambda: run_attention_numpy(work), "numpy"
        print(fmt(f"attention x{total_blocks} blocks, CPU (numpy)", st_np))
        print("  !! torch not importable in this env -- numpy is the only available")
        print("     CPU attention, and it is ~10x slower than torch on this shape.")
        print("     Treat the splice total below as an UPPER bound, not the number.")

    print(f"\n--- the spliced loop, one perf_counter around both ({attn_tag} attention) ---")

    def spliced():
        npu_sess.run(None, {npu_in: x})
        attn_fn()

    st_sp = timed(spliced, args.iters)
    print(fmt("SPLICE: NPU backbone + CPU attention", st_sp))
    residual = st_sp["mean"] - (st_bb["mean"] + st_at["mean"])
    print(f"  in-process residual = {st_sp['mean']:.2f} - ({st_bb['mean']:.2f} + "
          f"{st_at['mean']:.2f}) = {residual:+.2f} ms")

    print("\n--- baselines, same run (NPU latency drifts between sessions) ---")
    st_bc = timed(backbone_cpu, args.iters)
    st_f32 = timed(lambda: fp32_cpu.run(None, {npu_in: x}), args.iters)
    print(fmt("cut CNN backbone, CPU  (like-for-like vs NPU)", st_bc))
    print(fmt("FULL model FP32, ORT CPU EP", st_f32))

    st_stock = None
    if not args.skip_stock:
        print("\nbuilding NPU session for the stock 49-subgraph graph ...")
        stock = build_session(str(STOCK_MODEL), "npu", MOBILEVIT_STOCK_CACHE_KEY, args.xclbin)
        s_in = stock.get_inputs()[0].name
        st_stock = timed(lambda: stock.run(None, {s_in: x}), max(10, args.iters // 5))
        print(fmt("FULL model XINT8, stock graph on NPU", st_stock))

    print("\n--- EP placement, read fresh from this run's own reports ---")
    print(ep_report(MOBILEVIT_CUT_CACHE_KEY))
    if not args.skip_stock:
        print(ep_report(MOBILEVIT_STOCK_CACHE_KEY))

    print("\n=== VERDICT ===")
    print(f"  splice (cost model)          {st_sp['mean']:7.2f} ms")
    print(f"  vs FULL model FP32 ORT CPU   {st_f32['mean']:7.2f} ms   "
          f"-> {st_f32['mean'] / st_sp['mean']:.2f}x")
    if st_stock:
        print(f"  vs FULL model XINT8 NPU      {st_stock['mean']:7.2f} ms   "
              f"-> {st_stock['mean'] / st_sp['mean']:.2f}x")
    print(f"  backbone alone: NPU {st_bb['mean']:.2f} vs CPU {st_bc['mean']:.2f} ms  "
          f"-> {st_bc['mean'] / st_bb['mean']:.2f}x (like-for-like, the honest CNN number)")
    print(f"  attention on CPU is {st_at['mean']:.2f} ms of the {st_sp['mean']:.2f} ms total "
          f"({100 * st_at['mean'] / st_sp['mean']:.0f}%), using {attn_tag}")
    if attn_torch is not None:
        alt = st_bb["mean"] + st_np["mean"] + residual
        print(f"  with the naive numpy attention instead the same splice is {alt:.2f} ms "
              f"-> {st_f32['mean'] / alt:.2f}x, i.e. it LOSES to plain CPU.")
        print("  The CPU-side kernel choice, not the NPU, decides the verdict here.")
    print("\n  Reminder: the FP32-CPU comparison is the apples-to-apples one (same ORT,")
    print("  same machine, same run). A PyTorch full-model baseline is a different")
    print("  measurement and must not be swapped in for this one.")


if __name__ == "__main__":
    main()
