"""Refinement disambiguator: perturb an oracle's positions, diff Quark's and Ignition's refine.

Both refiners receive byte-identical perturbed copies of the fresh no-CLE oracle.
Quark's adjust_quantize_info sees the file-order proto its own pipeline saved and
refined; Ignition's refine sees the topologically sorted copy its loader produces.
Directed cases violate one rule each by construction; random trials perturb several
scales at once and let both refiners chase the cascade. The gate is identical final
position tables. Only this reference process imports Quark; quant/ stays Quark-free.
Nothing is written except the log.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.graph import Graph
from quant.pow2 import pos2scale, scale2pos
from quant.refine import refine

QDQ = ("QuantizeLinear", "DequantizeLinear")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clone(model: onnx.ModelProto) -> onnx.ModelProto:
    return onnx.ModelProto.FromString(model.SerializeToString())


def scale_names(model: onnx.ModelProto) -> list[str]:
    """Ordered unique scale initializers referenced by standard-domain Q/DQ nodes."""
    names = []
    for node in model.graph.node:
        if node.op_type in QDQ and not node.domain and node.input[1] not in names:
            names.append(node.input[1])
    return names


def storage_forms(model: onnx.ModelProto, names: list[str]) -> dict[str, int]:
    inits = {t.name: t for t in model.graph.initializer}
    forms = Counter("raw_data" if inits[n].raw_data else "float_data" if len(inits[n].float_data) else "other"
                    for n in names)
    return dict(sorted(forms.items()))


def read_table(model: onnx.ModelProto, names: list[str]) -> dict[str, int]:
    inits = {t.name: t for t in model.graph.initializer}
    table = {}
    for name in names:
        value = numpy_helper.to_array(inits[name])
        if value.shape != () or value.dtype != np.float32:
            raise ValueError(f"Expected scalar float32 scale {name}")
        pos = scale2pos(value)
        if pos2scale(pos) != value:
            raise ValueError(f"Not a power-of-two scale: {name}")
        table[name] = pos
    return table


def write_pos(model: onnx.ModelProto, name: str, pos: int) -> None:
    """Store 2^-pos in the initializer's existing storage form."""
    tensor = next(t for t in model.graph.initializer if t.name == name)
    scale = pos2scale(pos)
    if tensor.raw_data:
        tensor.raw_data = np.asarray(scale, dtype=np.float32).tobytes()
    elif len(tensor.float_data):
        tensor.float_data[0] = float(scale)
    else:
        raise ValueError(f"Unsupported scale storage at {name}")


def to_raw_data(model: onnx.ModelProto, names: list[str]) -> None:
    for tensor in model.graph.initializer:
        if tensor.name in names:
            tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor), tensor.name))


def integer_digest(model: onnx.ModelProto) -> str:
    digest = hashlib.sha256()
    for tensor in model.graph.initializer:
        if tensor.name.endswith("_quantized"):
            digest.update(tensor.name.encode())
            digest.update(tensor.SerializeToString())
    return digest.hexdigest()


def is_topological(model: onnx.ModelProto) -> bool:
    seen = {v.name for v in model.graph.input} | {t.name for t in model.graph.initializer}
    for node in model.graph.node:
        if any(name and name not in seen for name in node.input):
            return False
        seen.update(node.output)
    return True


def structure(model: onnx.ModelProto) -> dict:
    """Per-node scale names, resolved through the Relu/Mul bridges both refiners use."""
    producers = {o: n for n in model.graph.node for o in n.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def iparam(node, slot=0):
        parent = producers.get(node.input[slot]) if slot < len(node.input) else None
        return parent.input[1] if parent is not None and parent.op_type == "DequantizeLinear" else None

    def oparam(node):
        followers = consumers.get(node.output[0], [])
        direct = next((n.input[1] for n in followers if n.op_type == "QuantizeLinear"), None)
        if direct is not None:
            return direct, False
        for bridge in followers:
            if bridge.op_type in ("Relu", "Mul"):
                q = next((n for n in consumers.get(bridge.output[0], []) if n.op_type == "QuantizeLinear"), None)
                if q is not None:
                    return q.input[1], True
        return None, False

    convs, adds, pools, roles = [], [], [], {}
    for node in model.graph.node:
        if node.op_type in ("Conv", "Gemm"):
            out, bridged = oparam(node)
            rec = {"node": node.name, "op": node.op_type, "i": iparam(node), "w": iparam(node, 1),
                   "b": iparam(node, 2), "o": out, "bridged": bridged}
            convs.append(rec)
            roles[rec["w"]] = "weight"
            if rec["b"]:
                roles[rec["b"]] = "bias"
        elif node.op_type == "Add":
            out, bridged = oparam(node)
            adds.append({"node": node.name, "inputs": [iparam(node, i) for i in range(len(node.input))],
                         "o": out, "bridged": bridged})
        elif node.op_type in ("MaxPool", "GlobalAveragePool"):
            out, bridged = oparam(node)
            pools.append({"node": node.name, "op": node.op_type, "i": iparam(node), "o": out, "bridged": bridged})
    return {"convs": convs, "adds": adds, "pools": pools, "roles": roles}


def violations(table: dict[str, int], layout: dict) -> dict[str, int]:
    """Count final-table violations of the sourced constraints (DESIGN.md 2.3)."""
    counts = Counter()
    for c in layout["convs"]:
        if None in (c["i"], c["w"], c["o"]):
            continue
        sc = table[c["w"]] + table[c["i"]] - table[c["o"]]
        counts["shift_cut"] += not 0 <= sc <= 16
        if c["b"]:
            sb = table[c["w"]] + table[c["i"]] - table[c["b"]]
            counts["shift_bias"] += not min(0, sc - 16) <= sb <= 15
    for a in layout["adds"]:
        if None in a["inputs"] or a["o"] is None:
            continue
        values = [table[n] for n in a["inputs"]]
        counts["shift_read"] += max(values) - min(values) > 7
        counts["shift_write"] += not -7 <= min(values) - table[a["o"]] <= 25
    for p in layout["pools"]:
        if None in (p["i"], p["o"]):
            continue
        counts["align_pool"] += table[p["i"]] != table[p["o"]]
    return {k: v for k, v in sorted(counts.items()) if v}


class QuarkRefiner:
    """Quark's adjust_quantize_info with its screen logger captured, not printed."""

    def __init__(self, max_loops: int):
        from quark.onnx.postprocess.refinement.refine import adjust_quantize_info
        from quark.onnx.utils.model_utils import ONNXQuantizedModel
        self.adjust, self.wrap, self.max_loops = adjust_quantize_info, ONNXQuantizedModel, max_loops
        self.records = []
        logger = logging.getLogger("quark.onnx.postprocess.refinement.refine_screen")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        handler = logging.Handler()
        handler.emit = lambda record: self.records.append((record.levelname, record.getMessage()))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict, list[str]]:
        self.records.clear()
        start = time.perf_counter()
        out = self.adjust(self.wrap(model), max_loop_num=self.max_loops)
        messages = [m for _, m in self.records]
        stats = {"passes": sum(m.startswith("Adjust the quantize info") for m in messages),
                 "limit_warning": any("reached the limit" in m for m in messages),
                 "modify_messages": sum("Modify" in m for m in messages),
                 "seconds": round(time.perf_counter() - start, 3)}
        return out.model, stats, [m for m in messages if "Modify" in m]


def run_ignition(model: onnx.ModelProto, max_loops: int):
    graph = Graph(clone(model))
    graph.topo_sort()
    start = time.perf_counter()
    report = refine(graph, max_loops)
    return graph.model, report, round(time.perf_counter() - start, 3)


def run_case(name: str, kind: str, perturbation: dict[str, int], base: onnx.ModelProto, names: list[str],
             layout: dict, quark: QuarkRefiner, max_loops: int, base_ints: str, verbose: bool) -> dict:
    model = clone(base)
    for scale, pos in perturbation.items():
        write_pos(model, scale, pos)
    start = read_table(model, names)
    q_model, q_stats, q_messages = quark.run(clone(model))
    i_model, i_report, i_seconds = run_ignition(model, max_loops)
    qt, it = read_table(q_model, names), read_table(i_model, names)
    diff = {n: [qt[n], it[n]] for n in names if qt[n] != it[n]}
    record = {
        "case": name, "kind": kind, "perturbation": perturbation, "equal": not diff, "diff": diff,
        "start_violations": violations(start, layout),
        "quark": {**q_stats, "moved": {n: [start[n], qt[n]] for n in names if start[n] != qt[n]},
                  "violations": violations(qt, layout), "integers_unchanged": integer_digest(q_model) == base_ints},
        "ignition": {"loops": i_report.loops, "converged": i_report.converged, "seconds": i_seconds,
                     "changes_by_rule": i_report.changes_by_rule,
                     "moved": {n: [start[n], it[n]] for n in names if start[n] != it[n]},
                     "violations": violations(it, layout), "integers_unchanged": integer_digest(i_model) == base_ints},
    }
    print("CASE", json.dumps(record, sort_keys=True), flush=True)
    if verbose:
        for message in q_messages:
            print("  QUARK", message)
        for change in i_report.log:
            print("  IGNITION", json.dumps(change, sort_keys=True))
    return record


def directed_cases(table: dict[str, int], layout: dict) -> list[tuple[str, str, dict[str, int]]]:
    """One-rule violations computed from the node's actual positions."""
    cases = []
    picks = {"conv_pruned": next(c for c in layout["convs"] if c["op"] == "Conv" and c["bridged"] and c["b"]),
             "conv_direct": next(c for c in layout["convs"] if c["op"] == "Conv" and not c["bridged"] and c["b"]),
             "gemm": next(c for c in layout["convs"] if c["op"] == "Gemm")}
    for label, c in picks.items():
        ip, wp, op, bp = table[c["i"]], table[c["w"]], table[c["o"]], table[c["b"]]
        sc = wp + ip - op
        cases.append((f"{label}_cut_high", "single_rule", {c["w"]: op - ip + 20}))
        cases.append((f"{label}_cut_low", "single_rule", {c["w"]: op - ip - 3}))
        cases.append((f"{label}_bias_high", "single_rule", {c["b"]: wp + ip - 20}))
        cases.append((f"{label}_bias_low", "single_rule", {c["b"]: wp + ip - (min(0, sc - 16) - 3)}))
    adds = [a for a in layout["adds"] if a["bridged"] and None not in a["inputs"]]
    a = adds[0]
    a0, a1 = table[a["inputs"][0]], table[a["inputs"][1]]
    cases.append(("add_read_high_input0", "cascade", {a["inputs"][0]: a1 + 9}))
    cases.append(("add_read_high_input1", "cascade", {a["inputs"][1]: a0 + 9}))
    cases.append(("add_write_low", "cascade", {a["o"]: min(a0, a1) + 10}))
    cases.append(("add_write_high", "cascade", {a["o"]: min(a0, a1) - 30}))
    gap = next(p for p in layout["pools"] if p["op"] == "GlobalAveragePool")
    gi = table[gap["i"]]
    cases.append(("gap_pool_high", "cascade", {gap["o"]: gi + 3}))
    cases.append(("gap_pool_low", "cascade", {gap["o"]: gi - 3}))
    cases.append(("gap_loop_limit", "loop_limit", {gap["o"]: gi - 40}))
    multi = {}
    for (label, c), rule in zip(picks.items(), ("cut_high", "bias_high", "cut_low")):
        ip, wp, op = table[c["i"]], table[c["w"]], table[c["o"]]
        multi[c["w"] if rule.startswith("cut") else c["b"]] = (
            op - ip + 20 if rule == "cut_high" else op - ip - 3 if rule == "cut_low" else wp + ip - 20)
    b, d = adds[1], adds[min(4, len(adds) - 1)]
    multi[b["inputs"][0]] = table[b["inputs"][1]] + 9
    multi[d["o"]] = min(table[n] for n in d["inputs"]) + 10
    multi[gap["o"]] = gi + 3
    cases.append(("multi_rule", "cascade", multi))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("models/resnet50_quark_nocle_c64.onnx"))
    parser.add_argument("--ignition-model", type=Path, default=Path("models/resnet50_ignition_alpha_nocle_c64.onnx"),
                        help="Only its scale storage form is reported")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scales", type=int, default=6)
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument("--max-loops", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="Print both refiners' change logs for directed cases")
    args = parser.parse_args()
    if args.trials < 0 or args.max_scales < 1 or args.max_shift < 1 or args.max_loops < 1:
        parser.error("trials must be nonnegative; max-scales, max-shift and max-loops positive")
    if any(name.startswith("quark") for name in sys.modules):
        raise RuntimeError("Quark must be imported only after the owned modules")

    reference_hash = file_hash(args.reference)
    sidecar = args.reference.with_suffix(".reference.json")
    oracle = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    base = onnx.load(args.reference)
    names = scale_names(base)
    table = read_table(base, names)
    layout = structure(base)
    base_ints = integer_digest(base)

    from quark.onnx.quantization.config.custom_config import XINT8_CONFIG
    from quark.onnx.quantization.quant_utils import annotate_op_type, remove_qdq_op_type
    from quark.onnx.quantizers.interface import set_parameters_and_domain
    remove_before = list(remove_qdq_op_type)
    # The XINT8 pipeline populates the pruning list before quantizing (interface.py);
    # Quark's Relu bridging in refine reads it. Reproduce that state on a throwaway copy.
    set_parameters_and_domain(clone(base), dict(getattr(XINT8_CONFIG, "extra_options", {}) or {}))
    quark = QuarkRefiner(args.max_loops)

    setup = {
        "reference": args.reference.as_posix(), "reference_sha256": reference_hash,
        "sidecar_hash_matches": oracle.get("model_sha256") == reference_hash,
        "oracle_include_cle": oracle.get("include_cle"),
        "versions": {p: metadata.version(p) for p in ("amd-quark", "onnx", "numpy")},
        "remove_qdq_op_type": {"before": remove_before, "after": list(remove_qdq_op_type)},
        "annotate_op_type": list(annotate_op_type),
        "nodes": len(base.graph.node), "file_order_topological": is_topological(base),
        "scales": len(names), "roles": dict(sorted(Counter(layout["roles"].get(n, "activation") for n in names).items())),
        "convs": len(layout["convs"]), "bridged_convs": sum(c["bridged"] for c in layout["convs"]),
        "adds": len(layout["adds"]), "bridged_adds": sum(a["bridged"] for a in layout["adds"]),
        "pools": [(p["op"], p["i"] == p["o"]) for p in layout["pools"]][:3],
        "reference_scale_storage": storage_forms(base, names),
        "baseline_violations": violations(table, layout),
        "max_loops": args.max_loops, "trials": args.trials, "seed": args.seed,
        "max_scales": args.max_scales, "max_shift": args.max_shift,
    }
    if args.ignition_model.is_file():
        ignition = onnx.load(args.ignition_model)
        setup["ignition_model"] = {"path": args.ignition_model.as_posix(), "sha256": file_hash(args.ignition_model),
                                   "scale_storage": storage_forms(ignition, scale_names(ignition))}
    print("REFINE_PROBE_SETUP", json.dumps(setup, indent=2), flush=True)

    results = [run_case("control_unperturbed", "control", {}, base, names, layout, quark, args.max_loops, base_ints, args.verbose)]
    control_ok = results[0]["equal"] and not results[0]["quark"]["moved"] and not results[0]["ignition"]["moved"]
    for name, kind, perturbation in directed_cases(table, layout):
        results.append(run_case(name, kind, perturbation, base, names, layout, quark, args.max_loops, base_ints, args.verbose))
    directed_ok = all(r["equal"] for r in results[1:])
    print("DIRECTED_SUMMARY", json.dumps({
        "cases": len(results) - 1, "equal": sum(r["equal"] for r in results[1:]),
        "unequal": [r["case"] for r in results[1:] if not r["equal"]],
        "converged_both": sum(r["ignition"]["converged"] and not r["quark"]["limit_warning"] for r in results[1:]),
        "ignition_rule_totals": dict(sum((Counter(r["ignition"]["changes_by_rule"]) for r in results[1:]), Counter())),
    }, sort_keys=True), flush=True)

    rng = np.random.default_rng(args.seed)
    fuzz = []
    for trial in range(args.trials):
        count = int(rng.integers(1, args.max_scales + 1))
        chosen = rng.choice(len(names), size=count, replace=False)
        perturbation = {}
        for index in chosen:
            delta = int(rng.integers(1, args.max_shift + 1)) * (1 if rng.random() < 0.5 else -1)
            perturbation[names[int(index)]] = table[names[int(index)]] + delta
        model = clone(base)
        for scale, pos in perturbation.items():
            write_pos(model, scale, pos)
        start = read_table(model, names)
        q_model, q_stats, _ = quark.run(clone(model))
        i_model, i_report, _ = run_ignition(model, args.max_loops)
        qt, it = read_table(q_model, names), read_table(i_model, names)
        diff = {n: [qt[n], it[n]] for n in names if qt[n] != it[n]}
        record = {"trial": trial, "scales": count,
                  "roles": dict(sorted(Counter(layout["roles"].get(n, "activation") for n in perturbation).items())),
                  "equal": not diff, "diff": diff,
                  "quark_passes": q_stats["passes"], "quark_limit": q_stats["limit_warning"],
                  "quark_moved": sum(start[n] != qt[n] for n in names),
                  "ignition_loops": i_report.loops, "ignition_converged": i_report.converged,
                  "ignition_rules": i_report.changes_by_rule,
                  "final_violations": [violations(qt, layout), violations(it, layout)]}
        if diff:
            record["perturbation"] = perturbation
        fuzz.append(record)
        print("TRIAL", json.dumps(record, sort_keys=True), flush=True)
    fuzz_ok = all(r["equal"] for r in fuzz)
    print("FUZZ_SUMMARY", json.dumps({
        "trials": len(fuzz), "equal": sum(r["equal"] for r in fuzz),
        "unequal_trials": [r["trial"] for r in fuzz if not r["equal"]],
        "trials_with_moves": sum(r["quark_moved"] > 0 for r in fuzz),
        "loop_limit_hits": sum(r["quark_limit"] for r in fuzz),
        "ignition_unconverged": sum(not r["ignition_converged"] for r in fuzz),
        "ignition_rule_totals": dict(sum((Counter(r["ignition_rules"]) for r in fuzz), Counter())),
        "converged_with_remaining_violations": sum(bool(r["final_violations"][1]) and r["ignition_converged"] for r in fuzz),
    }, sort_keys=True), flush=True)

    # Hazard: Quark's raw_data scale write assigns to a temporary list (refine.py:49-57).
    hazard_case = next(c for c in directed_cases(table, layout) if c[0] == "conv_pruned_cut_high")
    model = clone(base)
    for scale, pos in hazard_case[2].items():
        write_pos(model, scale, pos)
    float_q, _, _ = quark.run(clone(model))
    float_i, _, _ = run_ignition(model, args.max_loops)
    to_raw_data(model, names)
    start = read_table(model, names)
    raw_q, raw_stats, _ = quark.run(clone(model))
    raw_i, raw_report, _ = run_ignition(model, args.max_loops)
    print("RAW_DATA_HAZARD", json.dumps({
        "case": hazard_case[0], "scale_storage": storage_forms(model, names),
        "quark": {**raw_stats, "table_changed": read_table(raw_q, names) != start,
                  "equals_float_form_result": read_table(raw_q, names) == read_table(float_q, names)},
        "ignition": {"loops": raw_report.loops, "converged": raw_report.converged,
                     "table_changed": read_table(raw_i, names) != start,
                     "equals_float_form_result": read_table(raw_i, names) == read_table(float_i, names)},
    }, sort_keys=True), flush=True)

    passed = control_ok and directed_ok and fuzz_ok
    print("REFINE_PROBE_PASS", passed, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
