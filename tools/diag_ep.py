"""
Answer "did the VitisAI EP actually take anything, and if not, why".

Two independent halves:

  --model   static QDQ audit of a quantized ONNX. Needs onnx; runs in either env.
  --cache-key  reads <cacheKey>/vitisai_ep_report.json, which the EP writes on
            every session build (not only on compile). This is the ground truth
            for what landed where, and it survives a cache hit, unlike the
            compile log.

    conda activate resnet_env17
    python tools/diag_ep.py --model models/yolov8n_cut_xint8.onnx --cache-key yolocutcachekey
    python tools/diag_ep.py --model models/resnet50_xint8_adaround.onnx --cache-key modelcachekey

The ResNet pair is the known-good control: its report has an `NPU` entry in
deviceStat with 393 of 395 nodes. A report whose deviceStat has NO `NPU` entry
means the EP registered, walked the graph and claimed nothing - a different
failure from bad partitioning, and the one yolov8n_xint8.onnx hit.

Report schema (verified against both caches):
  deviceStat  [{name, nodeNum, supportedOpType}]  name is one of
              all / NPU / CPU / VITIS_EP_CPU. `supportedOpType` is the set of
              op types *assigned to* that device, not a capability list - do
              not read it as "ops the NPU supports".
  nodeStat    [{output, input, opType, comment, device}] one per node
  shapeInfo   [{name, shape}]
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from npu.paths import CACHE_DIR

# Ops whose presence in a quantized graph is worth calling out, because the
# ResNet50 graph that compiles cleanly contains none of them.
NOT_IN_KNOWN_GOOD = {"HardSigmoid", "Sigmoid", "Slice", "Split", "Resize",
                     "Softmax", "Transpose", "Div", "Sub", "Reshape", "Concat"}

COMPUTE = {"Conv", "Gemm", "Add", "Mul", "Concat", "MaxPool", "Resize", "Split",
           "Sigmoid", "HardSigmoid", "Softmax", "Slice", "Sub", "Div", "Transpose",
           "Reshape", "AveragePool", "GlobalAveragePool"}


def static(path):
    import onnx

    m = onnx.load(path)
    g = m.graph
    ops = Counter(n.op_type for n in g.node)
    print(f"=== {path} ===")
    print(f"ir_version {m.ir_version}   "
          f"opset {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    print(f"producer   {m.producer_name} {m.producer_version}")
    print(f"nodes      {len(g.node)}")
    for i in g.input:
        print(f"input      {i.name} "
              f"{[d.dim_value for d in i.type.tensor_type.shape.dim]}")
    for o in g.output:
        print(f"output     {o.name} "
              f"{[d.dim_value for d in o.type.tensor_type.shape.dim]}")
    print(f"ops        {ops.most_common()}")

    q, dq = ops["QuantizeLinear"], ops["DequantizeLinear"]
    print(f"\nQ/DQ       QuantizeLinear {q}, DequantizeLinear {dq}")
    if q == 0:
        print("  *** no QuantizeLinear at all - not a QDQ model, the EP will take "
              "nothing ***")

    exotic = sorted(set(ops) & NOT_IN_KNOWN_GOOD)
    if exotic:
        print(f"\nops absent from the known-good ResNet50 graph: "
              f"{[(o, ops[o]) for o in exotic]}")
        print("  Any of these could be what GetCapability refuses. Bisect by "
              "removing them, not by guessing.")

    # Float compute islands. Quark's NPU-CNN pass rewrites SiLU as
    # HardSigmoid -> Mul(scale) -> Q/DQ, so a node can sit two hops from its
    # nearest Q/DQ and still be inside a quantized region; walk back a few hops
    # and skip Constants so that pattern does not false-positive.
    prod = {o: n for n in g.node for o in n.output}

    def touches_qdq(node, depth=3):
        if depth == 0:
            return False
        for i in node.input:
            p = prod.get(i)
            if p is None or p.op_type == "Constant":
                continue
            if p.op_type in ("DequantizeLinear", "QuantizeLinear"):
                return True
            if touches_qdq(p, depth - 1):
                return True
        return False

    island = [(n.op_type, n.name) for n in g.node
              if n.op_type in COMPUTE and not touches_qdq(n)]
    print(f"\nfloat compute nodes (no Q/DQ within 3 hops upstream): {len(island)}")
    for t, nm in island[:30]:
        print(f"   {t:16s} {nm}")
    if len(island) > 30:
        print(f"   ... and {len(island) - 30} more")
    if island:
        print("  The EP cannot take these. At the tail they only cost a partition "
              "boundary; mid-graph they cut it in two.")

    for n in g.node:
        if n.op_type in ("Conv", "Gemm"):
            ins = [prod[i].op_type for i in n.input if i in prod]
            print(f"\nfirst Conv/Gemm: {n.name}\n   input producers: {ins}")
            if "DequantizeLinear" not in ins:
                print("   *** input is not dequantized - the EP may reject the whole "
                      "graph rather than partition it ***")
            break


def report(cache_key, top=25):
    path = os.path.join(str(CACHE_DIR), cache_key, "vitisai_ep_report.json")
    if not os.path.isfile(path):
        print(f"\nno {path}")
        print("  The EP writes it on every session build. If it is missing, the EP "
              "never ran - check the xclbin and RYZEN_AI_INSTALLATION_PATH.")
        return
    d = json.load(open(path))
    print(f"\n=== {path} ===")

    devices = {e["name"]: e for e in d.get("deviceStat", [])}
    for name, e in devices.items():
        print(f"  {name:14s} nodeNum={e['nodeNum']:>5}  "
              f"opTypes={len(e.get('supportedOpType') or [])}")

    nodes = d.get("nodeStat", [])
    by_dev = Counter(n["device"] for n in nodes)
    print(f"\n  per-node device counts: {dict(by_dev)}")

    npu_nodes = by_dev.get("NPU", 0)
    if npu_nodes == 0:
        print("\n  *** ZERO nodes on the NPU. The EP registered, walked the graph "
              "and claimed nothing. ***")
        print("  This is NOT a partitioning problem - there is no partition. "
              "Compare op types against a graph that does compile.")
    else:
        print(f"\n  {npu_nodes} nodes on the NPU, "
              f"{sum(by_dev.values()) - npu_nodes} elsewhere.")

    print("\n  op type x device:")
    per = Counter((n["device"], n["opType"]) for n in nodes)
    for (dev, op), k in sorted(per.items()):
        print(f"    {dev:13s} {op:20s} {k}")

    off = [n for n in nodes if n["device"] != "NPU"]
    if npu_nodes and off:
        print(f"\n  {len(off)} nodes NOT on the NPU (first {top}):")
        for n in off[:top]:
            print(f"    {n['device']:13s} {n['opType']:20s} {n['output']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="quantized .onnx to audit")
    ap.add_argument("--cache-key", default=None,
                    help="cache dir holding vitisai_ep_report.json")
    args = ap.parse_args()
    if not args.model and not args.cache_key:
        raise SystemExit("pass --model, --cache-key, or both")
    if args.model:
        static(args.model)
    if args.cache_key:
        report(args.cache_key)


if __name__ == "__main__":
    main()
