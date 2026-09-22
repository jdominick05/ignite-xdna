"""What does AMD's VitisAI EP actually place on the NPU, and at what element width?

`tools/diag_ep.py` reads a `vitisai_ep_report.json` that already exists; this PRODUCES one for an
arbitrary ONNX and then answers the width question the report only implies. The claim it exists to
test is "AMD's EP places zero float nodes on the NPU at any width", which two documents cited to a
file that was never committed.

The report's `comment` field carries every tensor's element type, so placement can be counted by
output dtype rather than by op name:

    @140 [input_QuantizeLinear_Output:(ty=2,shape=[1,3,512,512])] QuantizeLinear [input:(ty=1,...)]
          ^ output name            ^ output element type

Read it with care in one respect. In a QDQ graph the EP places `DequantizeLinear` on the NPU and
that node's OUTPUT is float32, so "a node with a float output ran on the NPU" is already true of
every quantized model and says nothing about float arithmetic. What the claim is about is float
COMPUTE - a convolution whose inputs and output are both float, with no quantization around it -
so the two are counted separately and both are printed.

    python tools/amd_placement_probe.py --model models/modnet/modnet_cut_fp32.onnx \
        --cache-key modnet_cut_fp32_probe_cachekey --fresh

--cache-key is REQUIRED and has no default on purpose: npu.paths picks a cache by filename marker,
so two different graphs can resolve to one key and the second is served the first's compiled
artifacts without recompiling. A probe that inherited that would report the wrong graph's placement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu.ep_report import read_report
from npu.session import build_session, clear_cache

# onnx.TensorProto.DataType, for the widths a vision graph can carry.
DTYPE = {0: "undefined", 1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
         6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64", 16: "bfloat16"}
FLOAT_TYPES = {1, 10, 11, 16}
# Nodes that convert between widths. Their output width says nothing about the arithmetic they do.
QDQ = {"QuantizeLinear", "DequantizeLinear", "Cast"}

# Non-greedy up to the LAST colon before `(ty=`: tensor names contain colons of their own
# (`onnx::Conv_660_DequantizeLinear_Output`), and stopping at the first one drops every node
# carrying an `onnx::`-prefixed initializer - 104 of MODNet-Cut's 507.
OUTPUT_TY = re.compile(r"^@\d+\s*\[.*?:\(ty=(\d+)")


def emit(tag, **payload) -> None:
    print(tag, json.dumps(payload, sort_keys=True), flush=True)


def output_dtype(node) -> int | None:
    """The element type of this node's own output, from the report's comment field."""
    m = OUTPUT_TY.match(node.get("comment", ""))
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="any ONNX, any width")
    ap.add_argument("--cache-key", required=True,
                    help="compile-cache directory. Required, never defaulted: a shared key serves "
                         "another graph's compiled artifacts and the report would describe that graph")
    ap.add_argument("--fresh", action="store_true", help="clear the cache first")
    args = ap.parse_args()

    model = args.model if args.model.is_absolute() else ROOT / args.model
    if not model.exists():
        ap.error(f"no such model: {model}")

    if args.fresh:
        clear_cache(args.cache_key)
    emit("PROBE", model=str(model.relative_to(ROOT)) if model.is_relative_to(ROOT) else str(model),
         cache_key=args.cache_key, fresh=args.fresh)

    # Building the session is what makes the EP walk the graph and write the report.
    build_session(model, "npu", cache_key=args.cache_key)
    report = read_report(ROOT / args.cache_key / "vitisai_ep_report.json")
    nodes = report.raw["nodeStat"]

    by_device = Counter(n["device"] for n in nodes)
    by_device_dtype = Counter((n["device"], DTYPE.get(output_dtype(n), "unparsed")) for n in nodes)
    unparsed = sum(1 for n in nodes if output_dtype(n) is None)

    emit("PLACEMENT", report_sha256=report.sha256, total_nodes=len(nodes),
         device_counts=dict(by_device),
         device_stat={d["name"]: d["nodeNum"] for d in report.raw["deviceStat"]},
         comments_unparsed=unparsed,
         # A node whose width could not be read is not counted either way, so show what they look
         # like: an unparsed majority would mean the dtype tables below are about a minority.
         unparsed_by_op=dict(Counter(n["opType"] for n in nodes if output_dtype(n) is None)),
         unparsed_examples=[n.get("comment", "")[:160]
                            for n in nodes if output_dtype(n) is None][:3])
    emit("BY_OUTPUT_DTYPE",
         counts={f"{dev}:{ty}": n for (dev, ty), n in sorted(by_device_dtype.items())})

    float_compute = [n for n in nodes
                     if output_dtype(n) in FLOAT_TYPES and n["opType"] not in QDQ]
    npu_float_compute = [n for n in float_compute if n["device"] == "NPU"]
    emit("FLOAT_COMPUTE", total=len(float_compute), on_npu=len(npu_float_compute),
         on_npu_by_op=dict(Counter(n["opType"] for n in npu_float_compute)),
         off_npu_by_op=dict(Counter(n["opType"] for n in float_compute
                                    if n["device"] != "NPU")),
         examples_on_npu=[{k: n.get(k) for k in ("opType", "device", "output")}
                          for n in npu_float_compute[:10]])

    # THE COUNTS ABOVE ARE NOT A VERDICT ON A QDQ GRAPH, and saying so is the whole point of this
    # block. In the QuantizeLinear/DequantizeLinear form, a convolution is WRITTEN as float32 -
    # dequantize, convolve in float, requantize - and the EP fuses that pattern back into integer
    # arithmetic. Its declared output type stays float32 either way. So on a quantized model this
    # tool will count hundreds of "float compute nodes on the NPU" that execute as int8, and a
    # verdict derived from them would be exactly backwards. Only a graph carrying NO QuantizeLinear
    # can answer the question, so the verdict is withheld rather than guessed.
    quantized = any(n["opType"] in ("QuantizeLinear", "DequantizeLinear") for n in nodes)
    if quantized:
        emit("VERDICT", claim="AMD's EP places zero float COMPUTE nodes on the NPU",
             applicable=False,
             reason="this graph is QDQ, so a convolution's declared float32 output is the "
                    "quantized form's own notation and not evidence of float arithmetic; "
                    "re-run on a graph with no QuantizeLinear to answer the claim",
             float_declared_compute_on_npu=len(npu_float_compute))
    else:
        emit("VERDICT", claim="AMD's EP places zero float COMPUTE nodes on the NPU",
             applicable=True, holds=not npu_float_compute,
             float_compute_nodes_in_graph=len(float_compute),
             float_compute_on_npu=len(npu_float_compute),
             nodes_placed_on_npu=by_device.get("NPU", 0), total_nodes=len(nodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
