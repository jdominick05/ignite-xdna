"""Summarize archived probe results, validating model and EP-report provenance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npu.ep_report import read_report
from quant.quantize import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--cpu-audit", type=Path)
    args = parser.parse_args()
    audits = {}
    if args.cpu_audit:
        content = args.cpu_audit.read_text(encoding="utf-8")
        data = json.loads(content.split("CPU_REFERENCE_AUDIT ", 1)[1])
        audits = {(r["model_sha256"], r["inputs_sha256"]): r for r in data["rows"]}
    rows = []
    for tag in args.tags:
        records = sorted(Path("models").glob(tag + "_*.onnx.probe.json"))
        if not records:
            raise ValueError(f"No records for {tag}")
        cohort = None
        for path in records:
            result = json.loads(path.read_text(encoding="utf-8"))
            model = Path(str(path).removesuffix(".probe.json"))
            if file_hash(model) != result["model_sha256"]:
                raise ValueError(f"Model changed since measurement: {model}")
            signature = (result["base_sha256"], result["inputs_sha256"], result.get("xclbin_sha256"))
            if cohort is None:
                cohort = signature
            if signature != cohort:
                raise ValueError(f"Cohort provenance differs: {model}")
            audit = audits.get((result["model_sha256"], result["inputs_sha256"]), {})
            row = {"tag": tag, "mutation": result["mutation"]["name"], "status": result["status"],
                   "scope": result["scope"], "images": result["images"],
                   "model_sha256": result["model_sha256"], "base_sha256": result["base_sha256"],
                   "inputs_sha256": result["inputs_sha256"], "xclbin_sha256": result.get("xclbin_sha256"),
                   "log": f"results/quant/probe_{tag}_{result['mutation']['name']}.log",
                   "cpu_vs_baseline_cpu": result.get("cpu_vs_baseline_cpu"),
                   "cpu_accuracy": result.get("cpu_accuracy"), "npu_accuracy": result.get("npu_accuracy"),
                   "cpu_latency_ms": result.get("cpu_latency_ms"),
                   "ep_latency_ms": result.get("npu_latency_ms"),
                   "ep_vs_optimized_cpu": result.get("npu_vs_cpu"),
                   "ep_vs_unoptimized_cpu": result.get("npu_vs_unoptimized_cpu"),
                   "ep_vs_baseline_npu": result.get("npu_vs_baseline_npu"),
                   "unoptimized_cpu_audit": audit}
            stop_log = Path(f"results/quant/probe_{tag}_{result['mutation']['name']}_resource_stop.log")
            if stop_log.exists():
                row["status"] = "resource_stop"
                row["external_stop_log"] = stop_log.as_posix()
            if result["status"] == "completed":
                archived = read_report(Path(str(model) + ".ep.json")).summary()
                for key in ("npu", "total", "ops_by_device"):
                    if archived[key] != result["ep"][key]:
                        raise ValueError(f"Archived EP placement differs for {model}")
                row["ep"] = {key: archived[key] for key in ("npu", "total", "device_counts", "ops_by_device")}
            rows.append(row)
    print("PROBE_SUMMARY", json.dumps(rows, indent=2))
    print("\n| Cohort / mutation | Status | NPU / total | EP mean / median / p95 ms | EP vs CPU RMSE |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        ep = row.get("ep")
        latency = row["ep_latency_ms"]
        error = (row["ep_vs_unoptimized_cpu"] or row["unoptimized_cpu_audit"].get("npu_vs_unoptimized_cpu")
                 or row["ep_vs_optimized_cpu"])
        placement = f"{ep['npu']} / {ep['total']}" if ep else "not measured"
        times = " / ".join(f"{latency[k]:.3f}" for k in ("mean", "median", "p95")) if latency else "not measured"
        rmse = f"{error['rmse']:.6f}" if error else "not measured"
        print(f"| {row['tag']} / {row['mutation']} | {row['status']} | {placement} | {times} | {rmse} |")


if __name__ == "__main__":
    main()
