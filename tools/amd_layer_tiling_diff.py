#!/usr/bin/env python3
"""Per-layer re-read factors of AMD's compiled xmodels, joined to the ONNX convolutions they implement.

    python -I tools/amd_layer_tiling_diff.py --onnx models/yolov8s_cut_xint8.onnx \\
        --layers ctl=ctl_layers.txt --layers cmt=cmt_layers.txt [--engine container.ignite]

`--layers` files are `tools/amd_xmodel_census.py --layers` output (AMD_LAYER lines). For each conv2d-fix
layer the declared byte counts become factors of the tensor they move, from the ONNX shapes:

    w_rr  = const_data_load_size     / (Cout * Cin/groups * k * k)      weights + bias/params, int8
    i_rr  = input_feature_load_size  / (Hin * Win * Cin)                 int8 activations
    o_rr  = output_feature_save_size / (Hout * Wout * Cout)
    util  = workload / workload_on_arch
    wl    = workload / (Hout * Wout * Cout * (2 * Cin/groups * k * k + 1))  the join check: 1.0 means the
            layer computes the whole ONNX conv (the +1 is the bias add, an observed convention); a merged
            Split half reads about 0.5.

All are DERIVED from the compiler's declared sizes, not measured traffic. `const_data_load_size` also
carries bias and requantize parameters, so w_rr sits slightly above the true weight re-read count.
The layer name is AMD's `subgraph_(<tensor>_vaip_N)`; the tensor is a Conv output (or its
QuantizeLinear input), which is how the join finds the ONNX node. A layer whose tensor matches no Conv
is printed as unjoined rather than guessed.

`--engine` adds the engine's own per-layer manifest record for the same Conv node: its activation
packets as an input factor (packets * 6400 B / (Hin * Win * Cin), DERIVED: the 6,400 B packet includes
halo and channel padding) and its weight-fill count, which the manifest does not size per layer, so it is
reported as a count. AMD splits each C2f cv1 into two merged Split halves that share one ONNX Conv;
both rows carry the same engine record, and `engine_counted_here` is true on the first only, so an engine
total sums each node once (node_rows says how many AMD rows share it).

AMD's DPU subgraph declares const/input/output bytes at exactly 2x the sum of its per-layer records (all
four xmodels checked, 2026-10-13), while its op counts agree 1x. Which level is per frame is not
established; the factors here use the per-layer records.

Run so far on YOLOv8s and YOLOv8n, AMD default config vs enable_cost_model_tiling (Desktop 2, 2026-10-13).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import onnx
from onnx import numpy_helper, shape_inference


def load_layers(p: Path) -> list[dict]:
    return [json.loads(ln[10:]) for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("AMD_LAYER ")]


def onnx_convs(path: Path) -> dict:
    m = shape_inference.infer_shapes(onnx.load(str(path)))
    g = m.graph
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        d = vi.type.tensor_type.shape.dim
        shapes[vi.name] = [x.dim_value for x in d]
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    out = {}
    for n in g.node:
        if n.op_type != "Conv":
            continue
        w = n.input[1]
        if w not in inits and w in prod and prod[w].op_type == "DequantizeLinear":
            w = prod[w].input[0]
        wshape = list(numpy_helper.to_array(inits[w]).shape) if w in inits else None
        a = {x.name: x for x in n.attribute}
        out[n.output[0]] = {"node": n.name, "w": wshape, "in": shapes.get(n.input[0]), "out": shapes.get(n.output[0]),
                            "groups": a["group"].i if "group" in a else 1}
    return out


def tensor_of(layer_name: str) -> str | None:
    m = re.match(r"subgraph_\((.*?)_vaip_\d+\)", layer_name)
    if not m:
        return None
    return m.group(1)


def factors(L: dict, c: dict) -> dict:
    cout, cin_g, kh, kw = c["w"]
    _, cin, hin, win = c["in"]
    _, _, hout, wout = c["out"]
    r = {}
    r["w_rr"] = L["const_data_load_size"] / (cout * cin_g * kh * kw)
    r["i_rr"] = L["input_feature_load_size"] / (hin * win * cin)
    r["o_rr"] = L["output_feature_save_size"] / (hout * wout * cout)
    r["util"] = L["workload"] / L["workload_on_arch"]
    r["wl"] = L["workload"] / (hout * wout * cout * (2 * cin_g * kh * kw + 1))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--layers", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--engine", type=Path, default=None)
    a = ap.parse_args()
    arms = [(s.partition("=")[0], load_layers(Path(s.partition("=")[2]))) for s in a.layers]
    convs = onnx_convs(a.onnx)
    eng = {}
    if a.engine:
        from ignite_xdna.compiler.serializer import IgniteModelReader
        ge = IgniteModelReader(str(a.engine)).manifest
        ge = ge.get("graph_engine", ge)
        eng = {L["name"]: L for L in ge["layers"]}
    names = [x["layer"] for x in arms[0][1]]
    for _, rows in arms[1:]:
        assert [x["layer"] for x in rows] == names, "arms list different layers"
    per = []
    for i, nm in enumerate(names):
        if arms[0][1][i].get("type") != "conv2d-fix":
            continue
        t = tensor_of(nm)
        c = convs.get(t) or convs.get(re.sub(r"_QuantizeLinear_Input$", "", t or ""))
        if c is None or not c["w"] or not c["in"] or not c["out"]:
            print(f"UNJOINED {i} {nm}")
            continue
        rec = {"i": i, "layer": nm, "node": c["node"], "w": c["w"], "in": c["in"][1:], "out": c["out"][1:]}
        for an, rows in arms:
            rec[an] = {k: round(v, 3) for k, v in factors(rows[i], c).items()}
        rec["workload_equal"] = len({rows[i]["workload"] for _, rows in arms}) == 1
        e = eng.get(c["node"])
        if e:
            _, cin, hin, win = c["in"]
            rec["engine"] = {"i_rr": round(e["packets"] * 6400 / (hin * win * cin), 3), "w_fills": e["w_fills"],
                             "rounds": e["rounds"]}
        per.append(rec)
    # AMD splits a C2f cv1 into two merged Split halves, each with its own declared bytes; the engine runs that
    # Conv once. Mark the halves so an engine total counts the node once.
    seen = {}
    for rec in per:
        seen[rec["node"]] = seen.get(rec["node"], 0) + 1
    first = set()
    for rec in per:
        rec["node_rows"] = seen[rec["node"]]
        rec["engine_counted_here"] = "engine" in rec and rec["node"] not in first
        first.add(rec["node"])
        print("LAYER " + json.dumps(rec), flush=True)
    a0, *rest = [an for an, _ in arms]
    summ = {"conv_layers_joined": len(per), "workload_equal_all": all(r["workload_equal"] for r in per),
            "wl_check_range": [min(r[a0]["wl"] for r in per), max(r[a0]["wl"] for r in per)]}
    for an in rest:
        moved = [r for r in per if r[an] != r[a0]]
        summ[f"{an}_vs_{a0}"] = {
            "layers_changed": len(moved),
            "w_rr_down": sum(r[an]["w_rr"] < r[a0]["w_rr"] - 1e-9 for r in moved),
            "w_rr_up": sum(r[an]["w_rr"] > r[a0]["w_rr"] + 1e-9 for r in moved),
            "i_rr_down": sum(r[an]["i_rr"] < r[a0]["i_rr"] - 1e-9 for r in moved),
            "i_rr_up": sum(r[an]["i_rr"] > r[a0]["i_rr"] + 1e-9 for r in moved),
            "util_down": sum(r[an]["util"] < r[a0]["util"] - 1e-9 for r in moved),
            "util_up": sum(r[an]["util"] > r[a0]["util"] + 1e-9 for r in moved),
            "median_w_rr": [statistics.median(r[a0]["w_rr"] for r in per), statistics.median(r[an]["w_rr"] for r in per)],
            "median_i_rr": [statistics.median(r[a0]["i_rr"] for r in per), statistics.median(r[an]["i_rr"] for r in per)],
        }
    if eng:
        summ["engine_joined"] = sum("engine" in r for r in per)
        summ["engine_nodes"] = sum(r["engine_counted_here"] for r in per)
        summ["shared_node_rows"] = sum(r["node_rows"] > 1 for r in per)
    print("SUMMARY " + json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
