"""Structured verification and parity reporting CLI for Project Ignition.

Ingests run logs, vitisai_ep_report.json, and ONNX models/sidecars to generate:
1. NPU node placement ratios and device placement statistics
2. QDQ parameter parity between reference and candidate models
3. CPU vs NPU RMSE, MAD, and latency comparisons
4. Hardware systolic shift-cut contract verification (sigma in [14, 30])
"""

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_vitisai_report(path: Path) -> dict:
    """Extract placement statistics from vitisai_ep_report.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodeStat", [])
    total = len(nodes)
    if total == 0:
        return {}
    counts = defaultdict(int)
    for n in nodes:
        counts[n.get("device", "Unknown")] += 1
    npu_count = counts.get("NPU", 0)
    ratio = (npu_count / total) * 100.0
    return {
        "source": path.as_posix(),
        "total_nodes": total,
        "npu_nodes": npu_count,
        "cpu_nodes": counts.get("CPU", 0),
        "device_counts": dict(counts),
        "npu_ratio_pct": ratio,
        "placement_verdict": "FULL NPU" if npu_count == total else f"PARTIAL ({ratio:.1f}%)" if npu_count > 0 else "ZERO NPU",
    }


def parse_eval_log(path: Path) -> dict:
    """Extract placement, RMSE, MAD, latency, and accuracy metrics from eval/run logs."""
    content = path.read_text(encoding="utf-8", errors="replace")
    info: Dict[str, Any] = {"file": path.as_posix()}

    # Target EP detection
    if "Execution Target:NPU" in content or "Execution Target: NPU" in content or "Phoenix NPU" in content or "_npu" in path.name.lower():
        info["device"] = "NPU"
    elif "Execution Target:CPU" in content or "_cpu" in path.name.lower():
        info["device"] = "CPU"
    elif "_dml" in path.name.lower():
        info["device"] = "DirectML"
    else:
        info["device"] = "Unknown"

    # Model name detection
    model_match = re.search(r"(?:model=|models/|Test Model:\s*|Source ONNX Model:\s*)([a-zA-Z0-9_\-\.]+?\.onnx)", content)
    if model_match:
        info["model"] = model_match.group(1)
    else:
        info["model"] = path.stem.replace("eval_", "").replace("_npu", "").replace("_cpu", "").replace("_dml", "")

    # Node placement ratios
    placement_match = re.search(r"Nodes assigned to VitisAI EP:\s*(\d+)/(\d+)\s*\(([\d\.]+)%\)", content)
    if placement_match:
        npu = int(placement_match.group(1))
        total = int(placement_match.group(2))
        info["npu_nodes"] = npu
        info["total_nodes"] = total
        info["npu_ratio_pct"] = float(placement_match.group(3))
    elif "All nodes placed on [DmlExecutionProvider]" in content:
        info["dml_placement"] = "100.0%"
    elif "All nodes placed on [VitisAIExecutionProvider]" in content or ("Physical Silicon" in content and "Target Subgraph:" in content):
        info["npu_nodes"] = 1
        info["total_nodes"] = 1
        info["npu_ratio_pct"] = 100.0

    # Latency extraction
    lat_match = re.search(r"(?:Mean [Ii]nference [Ll]atency|Mean Latency|latency)[\s:=]+([\d\.]+)\s*ms", content)
    us_lat_match = re.search(r"Pipelined Effective Latency\s*\|\s*([\d\.]+)\s*us", content)
    if lat_match:
        info["latency_ms"] = float(lat_match.group(1))
    elif us_lat_match:
        info["latency_ms"] = float(us_lat_match.group(1)) / 1000.0
    else:
        # Check last infer: Xms in log
        infers = re.findall(r"infer:\s*([\d\.]+)ms", content)
        if infers:
            info["latency_ms"] = float(infers[-1])

    # RMSE extraction
    rmse_match = re.search(r"(?:Root Mean Squared Error|RMSE|Prob RMSE)[\s:=]+([\d\.]+)(?:\s*/\s*255)?", content)
    silicon_rmse_match = re.search(r"Physical Silicon vs ORT CPU Subgraph\s*\|\s*[\d\.]+%\s*\|\s*[\d\.]+\s*\|\s*([\d\.]+)", content)
    if rmse_match:
        info["rmse"] = float(rmse_match.group(1))
    elif silicon_rmse_match:
        info["rmse"] = float(silicon_rmse_match.group(1))

    # MAD extraction
    mad_match = re.search(r"(?:Mean Absolute Diff|MAD|Prob MAD)[\s:=]+([\d\.]+)(?:\s*/\s*255)?", content)
    silicon_mae_match = re.search(r"Physical Silicon vs ORT CPU Subgraph\s*\|\s*[\d\.]+%\s*\|\s*([\d\.]+)", content)
    if mad_match:
        info["mad"] = float(mad_match.group(1))
    elif silicon_mae_match:
        info["mad"] = float(silicon_mae_match.group(1))

    # Pearson r
    r_match = re.search(r"(?:Pearson Correlation \(r\)|Pearson Correlation|Pearson r)[\s:=]+([\d\.]+)", content)
    if r_match:
        info["pearson_r"] = float(r_match.group(1))

    # Accuracy / mAP
    acc_match = re.search(r"(?:Pixel Accuracy|Mean Acc|Top-1|mAP@0\.5|mAP@0\.5:0\.95)[\s:=]+([\d\.]+)%?", content)
    if acc_match:
        info["accuracy"] = float(acc_match.group(1))

    return info


def load_positions(path: Path) -> dict[str, int]:
    """Load quantization positions from .quant.json sidecar or ONNX model."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "positions" in data:
            return {k: v.get("pos", v.get("minmse_pos", 0)) for k, v in data["positions"].items()}
        elif "emission_positions" in data:
            return {k: v.get("pos", 0) for k, v in data["emission_positions"].items()}
        return {}
    elif path.suffix == ".onnx":
        from .graph import Graph
        from .qdq import read_pos_table
        g = Graph.load(path)
        table = read_pos_table(g)
        return {k: tq.pos for k, tq in table.items()}
    return {}


def compare_qdq_parity(ref_path: Path, cand_path: Path) -> dict:
    """Compare tensor quantization positions between reference and candidate models."""
    ref_pos = load_positions(ref_path)
    cand_pos = load_positions(cand_path)
    if not ref_pos or not cand_pos:
        return {"error": "Could not read positions from one or both models"}

    common = sorted(set(ref_pos.keys()) & set(cand_pos.keys()))
    if not common:
        return {"error": "Zero common tensors found between models"}

    matches = 0
    delta_1 = 0
    mismatches = []

    for name in common:
        rp = ref_pos[name]
        cp = cand_pos[name]
        diff = cp - rp
        if diff == 0:
            matches += 1
        elif abs(diff) == 1:
            delta_1 += 1
            mismatches.append({"tensor": name, "ref_pos": rp, "cand_pos": cp, "delta": diff})
        else:
            mismatches.append({"tensor": name, "ref_pos": rp, "cand_pos": cp, "delta": diff})

    match_pct = (matches / len(common)) * 100.0
    return {
        "ref": ref_path.as_posix(),
        "candidate": cand_path.as_posix(),
        "total_tensors": len(common),
        "exact_matches": matches,
        "delta_1_matches": delta_1,
        "divergent_tensors": len(mismatches),
        "parity_pct": match_pct,
        "divergences": mismatches,
    }


def verify_shift_cuts(model_path: Path) -> dict:
    """Check hardware shift range contract (sigma in [14, 30]) across all conv/gemm ops."""
    from .shift_cut import analyze_model_shift_cut, CONTRACT_SIGMA
    hazards = analyze_model_shift_cut(str(model_path))

    resolved = [h for h in hazards if h.resolved]
    unresolved = [h for h in hazards if not h.resolved]
    bound = [h for h in resolved if h.clamp_bound]

    violations = []
    edges = []
    in_contract = []

    for h in bound:
        if not h.in_contract:
            hazard_type = "UNDERFLOW (sigma < 14)" if h.sigma < CONTRACT_SIGMA[0] else "OVERFLOW (sigma > 30)"
            violations.append({
                "node": h.node_name,
                "op": h.op_type,
                "pos_x": h.pos_x,
                "pos_w": h.pos_w,
                "pos_y": h.pos_y,
                "sigma": h.sigma,
                "shift_cut": h.shift_cut,
                "hazard": hazard_type,
            })
        else:
            edge = h.at_contract_edge
            if edge:
                edges.append((h.node_name, h.sigma, edge))
            in_contract.append(h)

    sigmas = [h.sigma for h in bound]
    min_sigma = min(sigmas) if sigmas else None
    max_sigma = max(sigmas) if sigmas else None

    return {
        "model": model_path.as_posix(),
        "total_ops": len(hazards),
        "resolved_ops": len(resolved),
        "unresolved_ops": len(unresolved),
        "contract_bound_ops": len(bound),
        "violations_count": len(violations),
        "violations": violations,
        "edge_ops_count": len(edges),
        "edge_ops": edges,
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
        "contract_passed": len(violations) == 0,
    }


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a clean ASCII formatted table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep]
    hdr = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    lines.append(hdr)
    lines.append(sep)
    for row in rows:
        line = "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(row, widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def run_report(args) -> int:
    """Main verification CLI report runner."""
    all_paths = list(args.paths) + list(args.logs) + list(args.models)
    if args.ref:
        all_paths.append(args.ref)
    if args.candidate:
        all_paths.append(args.candidate)

    # Expand any glob patterns in paths
    expanded_paths: list[Path] = []
    for p in all_paths:
        p_str = str(p)
        if any(char in p_str for char in ("*", "?", "[")):
            matched = [Path(f) for f in glob.glob(p_str, recursive=True)]
            expanded_paths.extend(matched)
        else:
            expanded_paths.append(p)

    log_files = [p for p in expanded_paths if p.suffix == ".log" or p.name == "vitisai_ep_report.json"]
    model_files = [p for p in expanded_paths if p.suffix == ".onnx" or p.suffix == ".json" and p.name != "vitisai_ep_report.json"]

    output_data: Dict[str, Any] = {}

    print("=" * 80)
    print("Project Ignition: Structured Hardware Verification & Parity Report")
    print("=" * 80)

    # 1. NPU Node Placement Section
    placement_rows = []
    ep_reports = [p for p in expanded_paths if p.name == "vitisai_ep_report.json"]
    for ep_p in ep_reports:
        rep = parse_vitisai_report(ep_p)
        if rep:
            placement_rows.append([
                ep_p.parent.name or ep_p.name,
                "VitisAI (NPU)",
                str(rep["total_nodes"]),
                str(rep["npu_nodes"]),
                f"{rep['npu_ratio_pct']:.1f}%",
                rep["placement_verdict"],
            ])

    # Check text logs for placements
    eval_infos = []
    for log_p in log_files:
        if log_p.name != "vitisai_ep_report.json":
            info = parse_eval_log(log_p)
            eval_infos.append(info)
            if "total_nodes" in info and "npu_nodes" in info:
                placement_rows.append([
                    Path(info["file"]).name,
                    info.get("device", "NPU"),
                    str(info["total_nodes"]),
                    str(info["npu_nodes"]),
                    f"{info['npu_ratio_pct']:.1f}%",
                    "FULL NPU" if info["npu_nodes"] == info["total_nodes"] else "PARTIAL",
                ])

    if placement_rows:
        print("\n[SECTION 1: NPU Node Placement Ratios]")
        headers = ["Target / Log", "Device", "Total Nodes", "Placed Nodes", "Placement Ratio", "Status"]
        print(format_table(headers, placement_rows))
        output_data["placement"] = placement_rows

    # 2. CPU vs NPU RMSE & Accuracy Metrics Section
    # Group eval_infos by model
    by_model = defaultdict(dict)
    for info in eval_infos:
        model = info.get("model", "unknown")
        dev = info.get("device", "unknown")
        by_model[model][dev] = info

    metric_rows = []
    for model, devs in sorted(by_model.items()):
        cpu_info = devs.get("CPU", {})
        npu_info = devs.get("NPU", {})
        if cpu_info or npu_info:
            cpu_lat = f"{cpu_info.get('latency_ms', 0):.2f} ms" if "latency_ms" in cpu_info else "N/A"
            npu_lat = f"{npu_info.get('latency_ms', 0):.2f} ms" if "latency_ms" in npu_info else "N/A"
            cpu_rmse = f"{cpu_info.get('rmse', 0):.4f}" if "rmse" in cpu_info else "N/A"
            npu_rmse = f"{npu_info.get('rmse', 0):.4f}" if "rmse" in npu_info else "N/A"
            npu_r = f"{npu_info.get('pearson_r', 0):.4f}" if "pearson_r" in npu_info else "N/A"
            metric_rows.append([
                model, cpu_lat, npu_lat, cpu_rmse, npu_rmse, npu_r
            ])

    if metric_rows:
        print("\n[SECTION 2: CPU vs NPU Quantitative Metrics]")
        headers = ["Model", "CPU Latency", "NPU Latency", "CPU RMSE", "NPU RMSE", "NPU Pearson r"]
        print(format_table(headers, metric_rows))
        output_data["metrics"] = metric_rows

    # 3. QDQ Parameter Parity Section
    if args.ref and args.candidate:
        print("\n[SECTION 3: QDQ Parameter Parity]")
        parity = compare_qdq_parity(args.ref, args.candidate)
        output_data["qdq_parity"] = parity
        if "error" in parity:
            print(f"Parity comparison failed: {parity['error']}")
        else:
            headers = ["Reference", "Candidate", "Tensors", "Exact Match", "Delta=1", "Divergent", "Parity %"]
            row = [
                Path(parity["ref"]).name,
                Path(parity["candidate"]).name,
                str(parity["total_tensors"]),
                str(parity["exact_matches"]),
                str(parity["delta_1_matches"]),
                str(parity["divergent_tensors"]),
                f"{parity['parity_pct']:.1f}%",
            ]
            print(format_table(headers, [row]))
            if parity["divergences"]:
                print(f"\nDivergent Tensors ({len(parity['divergences'])}):")
                div_headers = ["Tensor Name", "Ref Pos", "Cand Pos", "Delta (Cand - Ref)"]
                div_rows = [
                    [d["tensor"], str(d["ref_pos"]), str(d["cand_pos"]), f"{d['delta']:+d}"]
                    for d in parity["divergences"][:30]
                ]
                print(format_table(div_headers, div_rows))

    # 4. Hardware Shift Range Contract Check (sigma in [14, 30])
    onnx_models = [p for p in model_files if p.suffix == ".onnx"]
    if onnx_models:
        print("\n[SECTION 4: Hardware Systolic Shift Range Contract (sigma in [14, 30])]")
        shift_rows = []
        violations_all = []
        for m in onnx_models:
            res = verify_shift_cuts(m)
            status = "PASS (In-Contract)" if res["contract_passed"] else f"FAIL ({res['violations_count']} violations)"
            shift_rows.append([
                m.name,
                str(res["contract_bound_ops"]),
                f"[{res['min_sigma']}, {res['max_sigma']}]" if res["min_sigma"] is not None else "N/A",
                str(res["violations_count"]),
                str(res["edge_ops_count"]),
                status,
            ])
            if res["violations"]:
                violations_all.append((m.name, res["violations"]))

        headers = ["Model", "Bound Ops (Conv/Gemm)", "Sigma Range", "Violations", "Edge Ops (14/30)", "Verdict"]
        print(format_table(headers, shift_rows))
        output_data["shift_cut"] = shift_rows

        if violations_all:
            print("\n*** OUT-OF-CONTRACT SCALE VIOLATIONS DETECTED ***")
            viol_headers = ["Model", "Node Name", "Op", "pos_x", "pos_w", "pos_y", "sigma", "Hardware Hazard"]
            viol_rows = []
            for model_name, viols in violations_all:
                for v in viols:
                    viol_rows.append([
                        model_name, v["node"], v["op"], str(v["pos_x"]), str(v["pos_w"]),
                        str(v["pos_y"]), str(v["sigma"]), v["hazard"]
                    ])
            print(format_table(viol_headers, viol_rows))

    if args.format == "json":
        print("\n[JSON OUTPUT]")
        print(json.dumps(output_data, indent=2))

    print("\nReport generation completed successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Structured verification and parity report across logs and models")
    parser.add_argument("paths", nargs="*", type=Path, help="Log files (*.log), vitisai_ep_report.json, or models (*.onnx, *.quant.json)")
    parser.add_argument("--logs", "-l", nargs="*", type=Path, default=[], help="Run logs or vitisai_ep_report.json")
    parser.add_argument("--models", "-m", nargs="*", type=Path, default=[], help="Quantized ONNX models or .quant.json sidecars")
    parser.add_argument("--ref", type=Path, default=None, help="Reference model/sidecar for parity comparison")
    parser.add_argument("--candidate", type=Path, default=None, help="Candidate model/sidecar for parity comparison")
    parser.add_argument("--format", choices=["table", "markdown", "json"], default="table", help="Output format")
    args = parser.parse_args()
    sys.exit(run_report(args))


if __name__ == "__main__":
    main()
